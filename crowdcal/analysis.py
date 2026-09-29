"""Cache rows + splits -> results dict (schema: see crowdcal/report.py docstring)."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from crowdcal import calib
from crowdcal.cache import load_arm
from crowdcal.data import prior_baseline
from crowdcal.metrics import ece, js_distance, noise_ceiling, reliability
from crowdcal.stats import bootstrap_mean, crossover, holm, paired_bootstrap

_DIR = re.compile(r"^(?P<arm>.+)@(?P<size>\d+)-s(?P<seed>\d+)$")


def write_sums(raw_dir: Path) -> int:
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.rglob("*.jsonl"))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(raw_dir)}\n" for p in files]
    (raw_dir / "SHA256SUMS").write_text("".join(lines))
    return len(files)


def verify_sums(raw_dir: Path) -> None:
    """No-op unless SHA256SUMS exists; then every jsonl must match and none may be missing or extra."""
    raw_dir = Path(raw_dir)
    sums = raw_dir / "SHA256SUMS"
    if not sums.exists():
        return
    want = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in sums.read_text().splitlines() if l.strip()}
    have = {str(p.relative_to(raw_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.rglob("*.jsonl")}
    if want != have:
        bad = sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))
        raise RuntimeError(f"raw data does not match SHA256SUMS: {bad}")


def _by_item(rows: list[dict], universe: set[str]) -> tuple[dict, set, set]:
    """-> ({item_id: {wording_id: mean p over repeats}}, failed item ids, all item ids seen in universe).
    Failed = any non-ok row, or wording coverage differs from the arm's wordings; missing items are the caller's job."""
    ps: dict = {}
    bad: set = set()
    for r in rows:
        if r["item_id"] not in universe:
            continue
        if r["status"] != "ok" or r["p_yes"] is None:
            bad.add(r["item_id"])
        else:
            ps.setdefault(r["item_id"], {}).setdefault(r["wording_id"], []).append(r["p_yes"])
    wids = {w for d in ps.values() for w in d}
    bad |= {i for i, d in ps.items() if set(d) != wids}
    return {i: {w: float(np.mean(v)) for w, v in d.items()} for i, d in ps.items()}, bad, wids


def _matrix(by_item: dict, items, wids: list) -> np.ndarray:
    return np.array([[by_item[i.id][w] for w in wids] for i in items], float)


def _score(P: np.ndarray, q: np.ndarray) -> dict:
    """P: (items, wordings) -> per-item loss (mean over wordings) and wording-averaged metrics."""
    Q = q[:, None]
    sq = (P - Q) ** 2
    return {"loss": sq.mean(1), "pw": sq.mean(0), "js": float(js_distance(P, Q).mean()),
            "ece": float(np.mean([ece(P[:, w], q) for w in range(P.shape[1])]))}


def _summ(scores: list[dict], wids: list, noise: np.ndarray, n_boot: int, seed: int) -> dict:
    """Scores from one or more seeds -> M dict. Losses are stacked (seeds, items) for the bootstrap."""
    L = np.stack([s["loss"] for s in scores])
    nf = bootstrap_mean(L - noise, n_boot, seed)  # excess over the annotator-noise ceiling
    m = {"soft_brier": bootstrap_mean(L, n_boot, seed), "js": float(np.mean([s["js"] for s in scores])),
         "ece": float(np.mean([s["ece"] for s in scores])), "noise_floor": {"excess": nf["mean"], "ci": nf["ci"]}}
    if wids != [None]:
        pw = np.mean([s["pw"] for s in scores], axis=0)
        m["per_wording"] = {str(w): float(x) for w, x in zip(wids, pw)}
    return m


def _cmp(name, family, a, b, n_boot, seed) -> dict:
    d = paired_bootstrap(a, b, n_boot, seed)
    return {"name": name, "family": family, "diff": d["diff"], "ci": d["ci"], "p": d["p"], "p_holm": None}


def analyze(raw_dir: Path, splits: dict, prereg: dict, calib_dir: Path, ckpt_dir: Path = Path("data/checkpoints")) -> dict:
    raw_dir, calib_dir, ckpt_dir = Path(raw_dir), Path(calib_dir), Path(ckpt_dir)
    verify_sums(raw_dir)
    n_boot, bseed = prereg["n_boot"], prereg["bootstrap_seed"]
    calib_items, test_items = splits["calib"], splits["test"]
    universe = {i.id for i in calib_items + test_items}

    # arm directories: flat arms + every checkpoint of a curve arm/size/seed named in the prereg
    dirs = list(prereg["hosted_arms"])
    ckpts: dict[str, list[str]] = {}  # "laya-ft@1000" -> [dir, ...]
    for p in sorted(raw_dir.iterdir()):
        m = _DIR.match(p.name)
        if p.is_dir() and m and m["arm"] in prereg["curve_arms"] and int(m["seed"]) in prereg["seeds"].get(m["size"], []):
            ckpts.setdefault(f"{m['arm']}@{int(m['size'])}", []).append(p.name)
            dirs.append(p.name)

    # exclusion (PREREG §6): failed/missing in any arm -> dropped from all arms
    rows, data, wids, failed_by_arm = {}, {}, {}, {}
    for d in dirs:
        rows[d] = load_arm(raw_dir, d)
        if not rows[d]:
            raise RuntimeError(f"no cached rows for arm {d!r} in {raw_dir}")
        # PREREG §4: a hosted model silently updated mid-run invalidates the arm
        served = {r["response_model"] for r in rows[d] if r.get("status") == "ok" and r.get("response_model")}
        if len(served) > 1:
            raise RuntimeError(f"arm {d!r} was served by more than one model version: {sorted(served)}")
        data[d], bad, wids[d] = _by_item(rows[d], universe)
        failed_by_arm[d] = bad | (universe - data[d].keys())
    excluded = set().union(*failed_by_arm.values())
    calib_items = [i for i in calib_items if i.id not in excluded]
    test_items = [i for i in test_items if i.id not in excluded]
    rates = {d: len(f) / len(universe) for d, f in failed_by_arm.items()}
    flagged = sorted(d for d, r in rates.items() if r > prereg["failure_flag_rate"])

    qc, qt = np.array([i.q for i in calib_items]), np.array([i.q for i in test_items])
    nt = np.array([i.n for i in test_items])
    noise = qt * (1 - qt) / (nt - 1)
    calib_dir.mkdir(parents=True, exist_ok=True)

    # per arm dir: fit on calib (wordings pooled), score on test raw / temperature / isotonic
    res = {}  # dir -> {kind: score}
    calib_recal_brier = {}
    for d in dirs:
        w = sorted(wids[d], key=str)
        Pc, Pt = _matrix(data[d], calib_items, w), _matrix(data[d], test_items, w)
        pooled_p, pooled_q = Pc.ravel(), np.repeat(qc, len(w))
        params = {"temperature": calib.fit_temperature(pooled_p, pooled_q), "isotonic": calib.fit_isotonic(pooled_p, pooled_q)}
        (calib_dir / f"{d}.json").write_text(json.dumps(params, indent=1))
        calib_recal_brier[d] = float(np.mean((calib.apply(params["temperature"], Pc) - qc[:, None]) ** 2))
        res[d] = {"raw": _score(Pt, qt), "recal": _score(calib.apply(params["temperature"], Pt), qt),
                  "iso": _score(calib.apply(params["isotonic"], Pt), qt), "w": w, "P": Pt,
                  "recal_P": calib.apply(params["temperature"], Pt)}

    flat, curves, L, rel = {}, {}, {}, {}   # L[spec][kind] -> loss array (seeds, items)
    for a in prereg["hosted_arms"]:
        r = res[a]
        flat[a] = {k: _summ([r[k]], r["w"], noise, n_boot, bseed) for k in ("raw", "recal")}
        L[a] = {k: r[k]["loss"][None] for k in ("raw", "recal", "iso")}
        rel[f"{a} (raw)"] = reliability(r["P"].ravel(), np.repeat(qt, len(r["w"])))
        rel[f"{a} (recal)"] = reliability(r["recal_P"].ravel(), np.repeat(qt, len(r["w"])))
    for spec, ds in ckpts.items():
        arm, size = spec.split("@")
        curves.setdefault(arm, {})[int(size)] = {k: _summ([res[d][k] for d in ds], res[ds[0]]["w"], noise, n_boot, bseed)
                                                 for k in ("raw", "recal")}
        L[spec] = {k: np.stack([res[d][k]["loss"] for d in ds]) for k in ("raw", "recal", "iso")}
    for arm, by_size in curves.items():
        top = max(by_size)
        ds = ckpts[f"{arm}@{top}"]
        rel[f"{arm}@{top} (recal)"] = reliability(np.concatenate([res[d]["recal_P"].ravel() for d in ds]),
                                                  np.concatenate([np.repeat(qt, len(res[d]["w"])) for d in ds]))

    # crossover vs best hosted arm, chosen on calib only
    best = min(prereg["hosted_arms"], key=calib_recal_brier.get)
    diffs = {s: paired_bootstrap(L[f"laya-ft@{s}"]["recal"], L[best]["recal"], n_boot, bseed)
             for s in prereg["sizes"] if f"laya-ft@{s}" in L}

    pa, pb = prereg["primary"]["a"], prereg["primary"]["b"]
    comps = [_cmp(f"{pa} - {pb} (recal)", "primary", L[pa]["recal"], L[pb]["recal"], n_boot, bseed)]
    sec = [_cmp(f"{s['a']} - {s['b']} (recal)", "secondary", L[s["a"]]["recal"], L[s["b"]]["recal"], n_boot, bseed)
           for s in prereg["secondary"]]
    for c, adj in zip(sec, holm([c["p"] for c in sec])):
        c["p_holm"] = adj
    comps += sec
    comps += [_cmp(f"{pa} - {pb} (raw)", "exploratory", L[pa]["raw"], L[pb]["raw"], n_boot, bseed),
              _cmp(f"{pa} - {pb} (isotonic)", "exploratory", L[pa]["iso"], L[pb]["iso"], n_boot, bseed),
              _cmp("deepseek - jev (recal)", "exploratory", L["deepseek"]["recal"], L["jev"]["recal"], n_boot, bseed)]

    prior = prior_baseline(splits["train@30k"])
    prior_loss = (prior - qt) ** 2
    t4 = prereg.get("t4_usd_per_hour")
    cost = []
    for name, ds in [(a, [a]) for a in prereg["hosted_arms"]] + [(s, ds) for s, ds in ckpts.items()]:
        ok = [r for d in ds for r in rows[d] if r["status"] == "ok"]
        api = any((r.get("usage") or {}).get("cost_usd") for r in ok)
        metas = [json.loads(m.read_text()) for d in ds if (m := ckpt_dir / d / "train_meta.json").exists()]
        hrs = float(np.mean([m["gpu_hours"] for m in metas])) if metas else None
        secs = [m["infer_sec_per_1k"] for m in metas if "infer_sec_per_1k" in m]
        per_1k = 1000 * float(np.mean([r["usage"]["cost_usd"] for r in ok])) if api else None
        if per_1k is None and t4 is not None and secs:  # ponytail: local inference cost only from a measured throughput
            per_1k = float(np.mean(secs)) / 3600 * t4
        cost.append({"arm": name, "per_1k_usd": per_1k, "train_gpu_hours": hrs,
                     "train_usd": hrs * t4 if hrs is not None and t4 is not None else None})

    return {
        "meta": {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_items": len(test_items),
                 "config_hashes": {d: rows[d][0]["config_hash"] for d in dirs}, "prereg_tag": prereg["prereg_tag"],
                 "exclusions": len(excluded), "failure_rates": rates, "flagged_arms": flagged},
        "noise_ceiling": noise_ceiling(qt, nt),
        "prior_baseline": bootstrap_mean(prior_loss, n_boot, bseed),
        "flat": flat, "curves": curves, "reliability": rel, "comparisons": comps,
        "crossover": {"best_hosted": best, "size": crossover(diffs)},
        "cost": cost,
    }
