"""crowdcal splits | run | pilot | freeze | report | demo"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from crowdcal import data
from crowdcal.analysis import analyze, write_sums
from crowdcal.arms import WORDINGS_PATH, FakeArm, LocalArm, OpenRouterArm, run_arm
from crowdcal.report import render

ROOT = Path(__file__).resolve().parent.parent
PREREG_TAG = "prereg-v1"


def _load(name: str) -> dict:
    return json.loads((ROOT / "config" / name).read_text())


def _has_prereg_tag() -> bool:
    out = subprocess.run(["git", "tag", "-l", PREREG_TAG], capture_output=True, text=True)
    return out.returncode == 0 and bool(out.stdout.strip())


def _build_arm(name: str):
    """name is a flat arm ("jev") or a checkpoint dir ("laya-ft@1000-s2")."""
    cfg = _load("arms.json")
    spec = cfg[name.split("@")[0]]
    if spec["type"] == "openrouter":
        return OpenRouterArm(name, spec["model"], spec["provider"], spec["mode"], os.environ.get(cfg["api_key_env"], ""))
    return LocalArm(name, spec["hf_id"].format(arm=name), spec["revision"], spec["prompted"])


def cmd_run(a) -> int:
    if not _has_prereg_tag():  # every arm in arms.json is a real one
        print(f"refusing to run {a.arm!r}: git tag {PREREG_TAG} does not exist. Commit PREREG.md and tag it "
              "before any real model call.", file=sys.stderr)
        return 2
    arm = _build_arm(a.arm)
    items = data.load_split(a.split)[: a.limit]
    manifest = json.loads(Path("data/splits/manifest.json").read_text())
    prereg = _load("prereg.json")
    wordings = data.load_wordings(WORDINGS_PATH)
    for ds in sorted({i.dataset for i in items}):
        stats = run_arm(arm, [i for i in items if i.dataset == ds], wordings, Path("data/raw"),
                        manifest["dataset_rev"][ds], a.repeats, prereg["budget_usd"])
        print(ds, stats)
    return 0


def pilot_summary(rows: list[dict], seconds: float, n_full_items: int, n_wordings: int) -> dict:
    """Plumbing only (PREREG Phase 3): never reads soft labels, so no metric can leak into config decisions."""
    ok = [r for r in rows if r["status"] == "ok"]
    by_cell: dict = {}
    for r in ok:
        by_cell.setdefault((r["item_id"], r["wording_id"]), []).append(round(r["p_yes"], 4))
    multi = [v for v in by_cell.values() if len(v) > 1]
    identical = sum(len(set(v)) == 1 for v in multi) / len(multi) if multi else None
    cost = sum((r.get("usage") or {}).get("cost_usd") or 0.0 for r in rows)
    reasoning = sum((((r.get("raw") or {}).get("usage") or {}).get("completion_tokens_details") or {})
                    .get("reasoning_tokens") or 0 for r in rows)
    ps = [r["p_yes"] for r in ok]
    repeats = 1 if identical == 1.0 else 3
    return {"rows": len(rows), "ok": len(ok), "fail_rate": round(1 - len(ok) / len(rows), 4) if rows else None,
            "repeat_identical_frac": identical, "planned_repeats": repeats,
            "p_min": min(ps, default=None), "p_max": max(ps, default=None), "p_distinct": len(set(ps)),
            "served_models": sorted({r["response_model"] for r in ok if r.get("response_model")}),
            "reasoning_tokens": reasoning, "cost_usd": round(cost, 6),
            "projected_full_run_usd": round(cost / len(rows) * n_full_items * n_wordings * repeats, 2) if rows else None,
            "sec_per_call": round(seconds / len(rows), 4) if rows else None}


def cmd_pilot(a) -> int:
    if not _has_prereg_tag():
        print(f"refusing to pilot {a.arm!r}: git tag {PREREG_TAG} does not exist.", file=sys.stderr)
        return 2
    arm, out = _build_arm(a.arm), Path("data/pilot")
    items = data.load_split("calib")[: a.n]  # PREREG: pilot items come from calib, never test
    manifest = json.loads(Path("data/splits/manifest.json").read_text())
    wordings = data.load_wordings(WORDINGS_PATH) if arm.prompted else None
    t = time.time()
    run_arm(arm, items, wordings, out, manifest["dataset_rev"]["mnli-dev"], a.repeats, _load("prereg.json")["budget_usd"])
    rows = [r for f in (out / a.arm).glob("*.jsonl") for r in map(json.loads, f.read_text().splitlines()) if r]
    n_full = manifest["counts"]["calib"] + manifest["counts"]["test"]
    s = {"arm": a.arm, "n_items": len(items), **pilot_summary(rows, time.time() - t, n_full, len(wordings or [None]))}
    (out / f"{a.arm}.summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0


def cmd_report(a) -> int:
    splits = {s: data.load_split(s) for s in ("calib", "test", "train@30k")}
    results = analyze(Path(a.raw_dir), splits, _load("prereg.json"), Path("data/calib"))
    render(results, Path(a.out))
    return 0


# demo: fake arms whose noise/temperature/bias differ; fine-tuned arms improve with size, laya-ft faster.
_HOSTED = {"jev": dict(temperature=1.6, noise=0.12, bias=0.0, seed=11),
           "deepseek": dict(temperature=0.6, noise=0.17, bias=-0.3, seed=12),
           "laya-base": dict(temperature=2.0, noise=0.19, bias=0.05, seed=13)}
_CURVE_NOISE = {"laya-ft": {100: 0.30, 1000: 0.17, 10000: 0.09, 30000: 0.05},
                "mbert-ce": {100: 0.36, 1000: 0.26, 10000: 0.17, 30000: 0.12}}
_CURVE_TEMP = {"laya-ft": 1.0, "mbert-ce": 1.4}


def fake_runs(splits: dict, raw_dir: Path, prereg: dict) -> None:
    wordings = data.load_wordings(WORDINGS_PATH)
    arms = [FakeArm(n, prompted=True, **kw) for n, kw in _HOSTED.items()]
    for arm in prereg["curve_arms"]:
        for size in prereg["sizes"]:
            for seed in prereg["seeds"][str(size)]:
                arms.append(FakeArm(f"{arm}@{size}-s{seed}", temperature=_CURVE_TEMP[arm], noise=_CURVE_NOISE[arm][size],
                                    seed=1000 * size + seed, prompted=arm == "laya-ft"))
    items = splits["calib"] + splits["test"]
    for arm in arms:
        run_arm(arm, items, wordings, raw_dir, "synthetic", budget_usd=prereg["budget_usd"])


def cmd_demo(a) -> int:
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)  # a demo is always a fresh run; stale raw files would trip the mixed-config guard
    prereg = {**_load("prereg.json"), "prereg_tag": "SYNTHETIC DEMO - not real results"}
    splits = data.synthetic_splits(a.seed)
    fake_runs(splits, out / "raw", prereg)
    r = analyze(out / "raw", splits, prereg, out / "calib", ckpt_dir=out / "checkpoints")
    render(r, out / "site")
    p = r["comparisons"][0]
    print(f"{p['name']}: {p['diff']:.4f} CI {p['ci'][0]:.4f}..{p['ci'][1]:.4f}; best_hosted={r['crossover']['best_hosted']} "
          f"crossover={r['crossover']['size']}; site: {out / 'site' / 'index.html'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crowdcal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("splits").set_defaults(fn=lambda a: bool(data.build_splits()) and 0)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True)
    r.add_argument("--split", required=True, choices=["calib", "test"])
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(fn=cmd_run)
    pl = sub.add_parser("pilot")
    pl.add_argument("--arm", required=True)
    pl.add_argument("--n", type=int, default=50)
    pl.add_argument("--repeats", type=int, default=2)
    pl.set_defaults(fn=cmd_pilot)
    sub.add_parser("freeze").set_defaults(fn=lambda a: print(f"wrote SHA256SUMS over {write_sums(Path('data/raw'))} files") or 0)
    p = sub.add_parser("report")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="docs")
    p.set_defaults(fn=cmd_report)
    d = sub.add_parser("demo")
    d.add_argument("--out", default="build/demo")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
