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
# Categorical slots 1-5 of the dataviz reference palette, in validated order (adjacent CVD dE >= 8.4 both modes).
# Color follows the arm, so both figures agree. Text is always ink; the colored mark beside it carries identity.
ARM_SLOT = {"mbert-ce": 0, "laya-ft": 1, "jev": 2, "laya-base": 3, "deepseek": 4}
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]},
    "dark": {"surface": "#0d1117", "ink": "#f0f6fc", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#262a30",
             "axis": "#3d434b", "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]},
}
NAME = {"laya-ft": "Laya, fine-tuned", "mbert-ce": "ModernBERT-large, fine-tuned", "jev": "Jev",
        "deepseek": "DeepSeek V4.1 Flash", "laya-base": "Laya, no fine-tuning", "prior baseline": "Prior baseline"}
SIZES_LABEL = {100: "100", 1000: "1k", 10000: "10k", 30000: "30k"}
FONT = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]}


def _color(arm, t):
    return t["series"][ARM_SLOT[arm]] if arm in ARM_SLOT else t["muted"]


def _style(ax, t):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)


def _spread(ys, gap):
    """Nudge label y-positions apart (sorted, minimum `gap`). ponytail: one-pass, fine for <=6 labels."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def _themed(fn, results, path):
    """Render `fn` once per theme: <name>.png (light, plus .svg) and <name>-dark.png for GitHub's dark mode."""
    path = Path(path)
    for theme, t in THEMES.items():
        with plt.rc_context(FONT):
            fig = fn(results, t)
        out = path if theme == "light" else path.with_name(f"{path.stem}-dark{path.suffix}")
        fig.savefig(out, dpi=200, transparent=True)
        if theme == "light" and fn is _headline:
            fig.savefig(path.with_suffix(".svg"), transparent=True)
        plt.close(fig)


def _headline(results: dict, t: dict):
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xscale("log")
    labels = []  # (x_end, y, name, value, color, is_curve)

    all_sizes = sorted({int(s) for c in results["curves"].values() for s in c}) or [100, 1000, 10000, 30000]
    x0, x1 = min(all_sizes), max(all_sizes)
    xe = x1 * 1.15  # flat lines run a little past the last curve point so endpoints never coincide
    lows = []

    flat = [(a, m["recal"]["soft_brier"]) for a, m in results["flat"].items()]
    flat.append(("prior baseline", results["prior_baseline"]))
    for name, b in flat:
        c = _color(name, t)
        ax.fill_between([x0, xe], b["ci"][0], b["ci"][1], color=c, alpha=0.10, lw=0)
        ax.plot([x0, xe], [b["mean"]] * 2, color=c, lw=1.5, ls=(0, (1, 2.5)) if name == "prior baseline" else "-",
                solid_capstyle="round", dash_capstyle="round")
        labels.append((xe, b["mean"], name, b["mean"], c, False))
        lows.append(b["ci"][0])

    for arm, by_size in results["curves"].items():
        pts = sorted((int(s), m["recal"]["soft_brier"]) for s, m in by_size.items())
        xs = [s for s, _ in pts]
        c = _color(arm, t)
        ax.fill_between(xs, [b["ci"][0] for _, b in pts], [b["ci"][1] for _, b in pts], color=c, alpha=0.14, lw=0)
        ax.plot(xs, [b["mean"] for _, b in pts], color=c, lw=2.5, marker="o", ms=9, mec=t["surface"], mew=2,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
        labels.append((x1, pts[-1][1]["mean"], arm, pts[-1][1]["mean"], c, True))
        lows += [b["ci"][0] for _, b in pts]

    cross = results.get("crossover") or {}
    if cross.get("size"):
        ax.axvline(cross["size"], color=t["axis"], lw=1)
        ax.annotate(f"Laya fine-tuned beats {NAME.get(cross['best_hosted'], cross['best_hosted'])}",
                    xy=(cross["size"], 0.98), xycoords=("data", "axes fraction"), xytext=(-6, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=11, color=t["ink"])

    ax.set_ylim(bottom=max(0.0, min(lows) - 0.01))
    ymin, ymax = ax.get_ylim()
    ys = _spread([y for _, y, *_ in labels], (ymax - ymin) * 0.06)
    lx = x1 * 1.45
    for (xa, y, name, v, c, curve), yy in zip(labels, ys):
        # colored leader from the series end to its label: identity sits beside the ink text
        ax.plot([xa, lx / 1.04], [y, yy], color=c, lw=1.2, solid_capstyle="round")
        ax.text(lx, yy, NAME.get(name, name), va="bottom", fontsize=11.5, color=t["ink"],
                fontweight="bold" if curve else "normal")
        ax.text(lx, yy, f"{v:.3f}", va="top", fontsize=10, color=t["ink2"])

    ax.set_xlim(x0 / 1.3, x1 * 16)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([SIZES_LABEL.get(s, str(s)) for s in all_sizes])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter("{x:.2f}")
    _style(ax, t)
    ax.set_xlabel("Training examples for the fine-tuned models (log scale)", fontsize=11.5, color=t["ink2"], labelpad=8)
    ax.set_title(TITLE, fontsize=16, fontweight="bold", loc="left", color=t["ink"], pad=34)
    ax.text(0, 1.035, f"Recalibrated soft Brier, lower is better. Shaded bands are 95% CIs. "
            f"The annotator-noise floor is {results['noise_ceiling']:.4f}.",
            transform=ax.transAxes, fontsize=11.5, color=t["ink2"])
    fig.tight_layout()
    return fig


def _reliability(results: dict, t: dict):
    groups = {}  # arm label -> {"raw"|"recal": bins}
    for label, d in results["reliability"].items():
        name, _, kind = label.rpartition(" (")
        groups.setdefault(name, {})[kind.rstrip(")")] = d
    n = max(len(groups), 1)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.8 * rows + 0.6), squeeze=False, sharex=True, sharey=True)
    for ax, (name, kinds) in zip(axes.flat, groups.items()):
        arm, _, size = name.partition("@")
        size = SIZES_LABEL.get(int(size), size) if size.isdigit() else size
        c = _color(arm, t)
        mx = max((k for d in kinds.values() for k in d["count"]), default=1)  # dot size is relative within a panel
        ax.plot([0, 1], [0, 1], color=t["axis"], lw=1)
        for kind, color, z in (("raw", t["muted"], 2), ("recal", c, 3)):
            d = kinds.get(kind)
            if not d:
                continue
            pts = sorted(zip(d["p"], d["q"], d["count"]))
            ax.plot([p for p, _, _ in pts], [q for _, q, _ in pts], color=color, lw=1.5, alpha=0.9, zorder=z)
            ax.scatter([p for p, _, _ in pts], [q for _, q, _ in pts], s=[16 + 110 * k / mx for _, _, k in pts],
                       color=color, edgecolor=t["surface"], linewidth=1.5, zorder=z)
        title = NAME.get(arm, arm) + (f", {size}" if size else "")
        ax.set_title(title, fontsize=11, color=t["ink"], loc="left", fontweight="bold")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        _style(ax, t)
        ax.grid(axis="x", color=t["grid"], lw=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Human share choosing entailment", fontsize=10, color=t["ink2"])
    spare = list(axes.flat)[len(groups):]
    for j in range(cols):  # x labels on the lowest filled panel of each column
        ax = [a for a in axes[:, j] if a not in spare][-1]
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel("Model P(entailment)", fontsize=10, color=t["ink2"])
    for ax in spare:
        ax.axis("off")
    # key: in the spare cell when there is one, else under the title
    key = spare[0] if spare else None
    handles = [plt.Line2D([], [], color=t["series"][ARM_SLOT["jev"]], marker="o", mec=t["surface"], lw=1.5, ms=7,
                          label="Recalibrated (arm colour)"),
               plt.Line2D([], [], color=t["muted"], marker="o", mec=t["surface"], lw=1.5, ms=7, label="Raw"),
               plt.Line2D([], [], color=t["axis"], lw=1, label="Perfect calibration")]
    leg = (key or fig).legend(handles=handles, loc="center" if key else "upper right", frameon=False, fontsize=10,
                             labelcolor=t["ink2"], title="Dot size = items in bin", title_fontsize=10)
    leg.get_title().set_color(t["muted"])
    fig.suptitle("Does the model's probability match the crowd?", x=0.02, ha="left", fontsize=14,
                 fontweight="bold", color=t["ink"])
    fig.tight_layout()
    return fig


def headline_figure(results: dict, path: Path) -> None:
    _themed(_headline, results, path)


def reliability_figure(results: dict, path: Path) -> None:
    _themed(_reliability, results, path)


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
