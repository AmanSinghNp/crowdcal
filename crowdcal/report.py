"""Render results -> docs/headline.png + docs/index.html. OWNER: report agent.

`render(results, out_dir)` takes this dict (produced by crowdcal.analysis.analyze):

results = {
  "meta": {"generated": str, "n_items": int, "config_hashes": {arm: hash}, "prereg_tag": str},
  "noise_ceiling": float,
  "prior_baseline": {"mean": float, "ci": [lo, hi]},
  "flat":   {arm: {"raw": M, "recal": M}},                       # jev, deepseek, laya-base
  "curves": {arm: {size_int: {"raw": M, "recal": M}}},           # laya-ft, mbert-ce; sizes 100,1000,10000,30000
  #   M = {"soft_brier": {"mean", "ci": [lo, hi]}, "js": float, "ece": float,
  #        "noise_floor": {"excess", "ci"}, "per_wording": {wid: soft_brier_mean} (prompted arms only)}
  "reliability": {label: {"p": [...], "q": [...], "count": [...]}},   # label like "jev (recal)"
  "comparisons": [{"name": str, "family": "primary"|"secondary"|"exploratory",
                   "diff": float, "ci": [lo, hi], "p": float, "p_holm": float | None}],
  "crossover": {"best_hosted": str, "size": int | None},
  "cost": [{"arm": str, "per_1k_usd": float | None, "train_gpu_hours": float | None, "train_usd": float | None}],
}
"""
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TITLE = "How much labelled data does a small model need to beat hosted models?"
QUESTION = ("Does 70% mean 70% of people agree? Soft Brier score against the full human label "
            "distribution, for fine-tuned small models vs hosted models.")
# Okabe-Ito (colorblind-safe). Fine-tuned arms: thick line + markers; flat arms: thin line, no markers.
CURVE_COLOR = {"laya-ft": "#D55E00", "mbert-ce": "#0072B2"}
FLAT_COLOR = {"jev": "#009E73", "deepseek": "#CC79A7", "laya-base": "#E69F00"}
GRAY = "#6b6b6b"
SIZES_LABEL = {100: "100", 1000: "1k", 10000: "10k", 30000: "30k"}


def _spread(ys, gap):
    """Nudge label y-positions apart (sorted, minimum `gap`). ponytail: one-pass, fine for <=6 labels."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def headline_figure(results: dict, path: Path) -> None:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xscale("log")
    labels = []  # (x_end, y, text, color): leader lines start at the series' OWN endpoint

    all_sizes = sorted({int(s) for c in results["curves"].values() for s in c}) or [100, 1000, 10000, 30000]
    x0, x1 = min(all_sizes), max(all_sizes)
    for arm, by_size in results["curves"].items():
        pts = sorted((int(s), m["recal"]["soft_brier"]) for s, m in by_size.items())
        xs = [s for s, _ in pts]
        c = CURVE_COLOR.get(arm, "#444444")
        ax.fill_between(xs, [b["ci"][0] for _, b in pts], [b["ci"][1] for _, b in pts], color=c, alpha=0.18, lw=0)
        ax.plot(xs, [b["mean"] for _, b in pts], color=c, lw=3, marker="o", ms=8, mec="white", mew=1.5, label=arm)
        labels.append((x1, pts[-1][1]["mean"], arm, c))

    flat = [(a, m["recal"]["soft_brier"], FLAT_COLOR.get(a, "#444444")) for a, m in results["flat"].items()]
    flat.append(("prior baseline", results["prior_baseline"], GRAY))
    xe = x1 * 1.12  # flat lines run a little past the last curve point so their endpoints never coincide with a curve's
    for name, b, c in flat:
        ax.fill_between([x0, xe], b["ci"][0], b["ci"][1], color=c, alpha=0.10, lw=0)
        ax.plot([x0, xe], [b["mean"]] * 2, color=c, lw=1.8, ls="-" if name != "prior baseline" else ":")
        labels.append((xe, b["mean"], name, c))

    nc = results["noise_ceiling"]
    ax.hlines(nc, x0, xe, color="black", lw=1.5, ls="--")
    ax.text(x0, nc, "annotator noise ceiling", va="bottom", ha="left", fontsize=12)

    cross = results.get("crossover") or {}
    if cross.get("size"):
        ax.axvline(cross["size"], color=CURVE_COLOR["laya-ft"], lw=1, ls=":")
        ax.annotate(f"laya-ft beats {cross['best_hosted']}", xy=(cross["size"], 0.98), xycoords=("data", "axes fraction"),
                    xytext=(-6, -4), textcoords="offset points", ha="right", va="top", fontsize=12,
                    color=CURVE_COLOR["laya-ft"], fontweight="bold")

    ymin, ymax = ax.get_ylim()
    ys = _spread([y for _, y, _, _ in labels], (ymax - ymin) * 0.045)
    for (xa, y, t, c), yy in zip(labels, ys):
        ax.annotate(t, xy=(xa, y), xytext=(x1 * 1.3, yy), textcoords="data", va="center", fontsize=12,
                    color=c, fontweight="bold" if t in CURVE_COLOR else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=0) if abs(yy - y) > 1e-9 else None)

    ax.set_xlim(x0 / 1.3, x1 * 5)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([SIZES_LABEL.get(s, str(s)) for s in all_sizes])
    ax.minorticks_off()
    ax.set_xlabel("Training examples (log scale)", fontsize=13)
    ax.set_ylabel("Recalibrated soft Brier (lower is better)", fontsize=13)
    ax.set_title(TITLE, fontsize=16, fontweight="bold", loc="left")
    ax.tick_params(labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def reliability_figure(results: dict, path: Path) -> None:
    rel = results["reliability"]
    n = max(len(rel), 1)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax, (label, d) in zip(axes.flat, rel.items()):
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
        mx = max(d["count"]) if d["count"] else 1
        ax.scatter(d["p"], d["q"], s=[20 + 180 * c / mx for c in d["count"]], color="#0072B2", alpha=0.75, edgecolor="white")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("mean predicted p"); ax.set_ylabel("mean human share q")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(rel):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _e(x) -> str:
    return html.escape(str(x))


def _f(x, d=4) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _ci(ci, d=4) -> str:
    return "—" if not ci else f"[{_f(ci[0], d)}, {_f(ci[1], d)}]"


def _mean_ci(b, d=4) -> str:
    return "—" if not b else f"{_f(b['mean'], d)} {_ci(b['ci'], d)}"


def _table(headers, rows) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c if isinstance(c, _Raw) else _e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


class _Raw(str):
    """Marks already-escaped HTML."""


def _comp_table(comps) -> str:
    return _table(["Comparison", "Family", "Diff", "95% CI", "p", "Holm p"],
                  [[c["name"], c["family"], _f(c["diff"]), _ci(c["ci"]), _f(c["p"], 3), _f(c.get("p_holm"), 3)] for c in comps])


def _primary_sentence(c) -> str:
    lo, hi = c["ci"]
    if hi < 0:
        verdict = "The first arm named has the lower (better) recalibrated soft Brier; the 95% interval excludes zero."
    elif lo > 0:
        verdict = "The second arm named has the lower (better) recalibrated soft Brier; the 95% interval excludes zero."
    else:
        verdict = "The 95% interval includes zero, so we cannot tell the two arms apart."
    return (f"{c['name']}: difference in recalibrated soft Brier (first minus second) is {_f(c['diff'])}, "
            f"95% CI {_ci(c['ci'])}, p = {_f(c['p'], 3)}. {verdict}")


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--accent:#0072B2}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--muted:#a0a0a0;--line:#333;--accent:#56B4E9}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;max-width:60rem;margin:0 auto;padding:1rem 16px 3rem}
h1{font-size:1.6rem;line-height:1.25}h2{margin-top:2.5rem;border-bottom:1px solid var(--line);padding-bottom:.25rem}
.q,.note,footer{color:var(--muted)}img{max-width:100%;height:auto;background:#fff;border-radius:4px}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.primary{border-left:4px solid var(--accent);padding-left:1rem}
.exploratory{border:1px dashed var(--muted);padding:0 1rem 1rem;margin-top:2.5rem;border-radius:6px}
.exploratory h2{border:0;margin-top:1rem}
"""


def render(results: dict, out_dir: Path = Path("docs")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headline_figure(results, out / "headline.png")
    reliability_figure(results, out / "reliability.png")

    # arm table: flat arms, then every curve size; raw and recal rows
    arm_rows = []

    def add(name, m):
        for kind in ("raw", "recal"):
            x = m[kind]
            nf = x.get("noise_floor") or {}
            arm_rows.append([name, kind, _mean_ci(x["soft_brier"]), _f(x.get("js")), _f(x.get("ece")),
                             _mean_ci({"mean": nf["excess"], "ci": nf["ci"]}) if nf else "—"])

    for arm, m in results["flat"].items():
        add(arm, m)
    for arm, by_size in results["curves"].items():
        for s in sorted(by_size, key=int):
            add(f"{arm}@{SIZES_LABEL.get(int(s), s)}", by_size[s])
    pb = results["prior_baseline"]
    arm_rows.append(["prior baseline", "—", _mean_ci(pb), "—", "—", "—"])

    # per-wording (recalibrated, prompted arms only)
    pw = {a: m["recal"].get("per_wording") for a, m in results["flat"].items() if m["recal"].get("per_wording")}
    wids = sorted({w for d in pw.values() for w in d}, key=str)
    pw_rows = [[a] + [_f(d.get(w)) for w in wids] for a, d in pw.items()]

    comps = results["comparisons"]
    primary = [c for c in comps if c["family"] == "primary"]
    main = [c for c in comps if c["family"] != "exploratory"]
    expl = [c for c in comps if c["family"] == "exploratory"]

    cost_rows = [[c["arm"], _f(c.get("per_1k_usd")), _f(c.get("train_gpu_hours"), 2), _f(c.get("train_usd"), 2)]
                 for c in results["cost"]]
    meta = results["meta"]
    hashes = ", ".join(f"{_e(a)}: {_e(h)}" for a, h in meta.get("config_hashes", {}).items())
    cross = results.get("crossover") or {}
    cross_txt = (f"laya-ft first beats {_e(cross['best_hosted'])} at {_e(cross['size'])} training examples."
                 if cross.get("size") else "No training size was found at which laya-ft beats the best hosted model.")

    primary_html = "".join(f"<p>{_e(_primary_sentence(c))}</p>" for c in primary) or "<p>—</p>"
    body = f"""<h1>{_e(TITLE)}</h1>
<p class="q">{_e(QUESTION)}</p>
<h2>Headline figure</h2>
<img src="headline.png" alt="{_e(TITLE)}">
<p class="note">{cross_txt}</p>
<h2>Primary result</h2>
<div class="primary">{primary_html}</div>
<h2>Arms</h2>
<p class="note">Soft Brier, JS, ECE and noise-floor excess; lower is better. 95% CIs in brackets.</p>
{_table(["Arm", "Calibration", "Soft Brier [95% CI]", "JS", "ECE", "Noise-floor excess [95% CI]"], arm_rows)}
<h2>Per-wording soft Brier</h2>
{_table(["Arm"] + [f"wording {w}" for w in wids], pw_rows) if pw_rows else "<p>—</p>"}
<h2>Reliability</h2>
<img src="reliability.png" alt="Reliability plots: mean predicted p vs mean human share q">
<h2>Paired comparisons</h2>
{_comp_table(main)}
<h2>Cost</h2>
{_table(["Arm", "USD per 1k decisions", "Training GPU-hours", "Training USD"], cost_rows)}
<div class="exploratory">
<h2>Exploratory</h2>
<p class="note">Not preregistered as confirmatory; no multiplicity correction.</p>
{_comp_table(expl) if expl else "<p>—</p>"}
</div>
<footer><hr><p>Generated {_e(meta.get("generated"))} · n_items {_e(meta.get("n_items"))} · prereg tag {_e(meta.get("prereg_tag"))}<br>Config hashes: {hashes or "—"}</p></footer>"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Crowdcal</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    (out / "index.html").write_text(page, encoding="utf-8")
