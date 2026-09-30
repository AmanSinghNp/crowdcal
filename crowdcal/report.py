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


REPO = "https://github.com/AmanSinghNp/crowdcal"
SHORT = {"laya-ft": "Laya fine-tuned", "mbert-ce": "ModernBERT-large fine-tuned", "jev": "Jev",
         "deepseek": "DeepSeek V4.1 Flash", "laya-base": "Laya (no fine-tuning)"}


def _label(key) -> str:
    """'laya-ft@30000' -> 'Laya fine-tuned · 30k'."""
    arm, _, rest = str(key).partition("@")
    name = SHORT.get(arm, arm)
    if not rest:
        return name
    size, _, seed = rest.partition("-s")     # checkpoint dirs look like 'laya-ft@1000-s2'
    size = SIZES_LABEL.get(int(size), size) if size.isdigit() else size
    return f"{name} · {size}" + (f" · seed {seed}" if seed else "")


def _split_name(name):
    """'laya-ft@30000 - jev (recal)' -> ('laya-ft@30000', 'jev', 'recal')."""
    pair, _, kind = name.partition(" (")
    a, _, b = pair.partition(" - ")
    return a, b, kind.rstrip(")")


def _verdict(c):
    """Every pre-registered comparison predicts the first arm is lower (better). -> (css class, text)."""
    lo, hi = c["ci"]
    sig = (c["p_holm"] < 0.05) if c.get("p_holm") is not None else (hi < 0 or lo > 0)
    if not sig:
        return "tie", "No clear difference"
    return ("yes", "As predicted") if c["diff"] < 0 else ("no", "Opposite of prediction")


def _p(x) -> str:
    """Bootstrap p-values have 1e-4 resolution; don't print a bare 0.000."""
    return "—" if x is None else ("< 0.001" if x < 0.001 else f"{x:.3f}")


def _comp_table(comps) -> str:
    rows = []
    for c in comps:
        a, b, kind = _split_name(c["name"])
        rows.append([_label(a), _label(b), kind, _f(c["diff"]), _ci(c["ci"]), _p(c["p"]), _p(c.get("p_holm"))])
    return _table(["First", "Second", "Scores", "Diff (first − second)", "95% CI", "p", "Holm p"], rows)


def _ledger(comps) -> str:
    items = []
    for c in comps:
        a, b, kind = _split_name(c["name"])
        cls, text = _verdict(c)
        role = "Primary" if c["family"] == "primary" else "Secondary"
        p = f"Holm p {_p(c['p_holm'])}" if c.get("p_holm") is not None else f"p {_p(c['p'])}"
        items.append(
            f'<li class="h"><div class="h-q"><span class="role">{role}</span>'
            f'Predicted: <b>{_e(_label(a))}</b> beats <b>{_e(_label(b))}</b></div>'
            f'<div class="h-v"><span class="chip {cls}">{text}</span></div>'
            f'<div class="h-n mono">{_f(c["diff"])} <span class="ci">{_ci(c["ci"])}</span> · {p}</div></li>')
    return f'<ol class="ledger">{"".join(items)}</ol>'


def _ruler(results) -> str:
    """Every arm on one soft-Brier scale: filled dot = recalibrated (primary) with 95% CI, ring = raw."""
    rows = [(_label(a), a, m["recal"]["soft_brier"], m["raw"]["soft_brier"]["mean"]) for a, m in results["flat"].items()]
    rows += [(_label(f"{a}@{s}"), a, m["recal"]["soft_brier"], m["raw"]["soft_brier"]["mean"])
             for a, by in results["curves"].items() for s, m in by.items()]
    rows.sort(key=lambda r: r[2]["mean"])
    pb, nc = results["prior_baseline"], results["noise_ceiling"]
    top = max([r[2]["ci"][1] for r in rows] + [r[3] for r in rows] + [pb["ci"][1]])
    step = 0.02 if top > 0.06 else 0.01
    xmax = step * (int(top / step) + 1)
    pos = lambda v: f"{100 * v / xmax:.2f}%"
    ticks = "".join(f'<span style="left:{pos(k * step)}">{k * step:.2f}</span>' for k in range(int(round(xmax / step)) + 1))
    floor = f'<span class="nc" style="left:{pos(nc)}"></span><span class="pb" style="left:{pos(pb["mean"])}"></span>'

    def row(label, arm, b, raw):
        lo, hi = b["ci"]
        return (f'<div class="r-lab"><span class="sw" style="--c:var(--{arm})"></span>{_e(label)}</div>'
                f'<div class="r-track">{floor}<span class="r-ci" style="--c:var(--{arm});left:{pos(lo)};width:{pos(hi - lo)}"></span>'
                f'<span class="r-raw" style="--c:var(--{arm});left:{pos(raw)}" title="raw {raw:.4f}"></span>'
                f'<span class="r-dot" style="--c:var(--{arm});left:{pos(b["mean"])}" title="recalibrated {b["mean"]:.4f} [{lo:.4f}, {hi:.4f}]"></span></div>'
                f'<div class="r-val mono">{b["mean"]:.3f}</div>')

    body = "".join(row(*r) for r in rows)
    return (f'<div class="ruler" role="img" aria-label="Recalibrated soft Brier for every arm, lowest first">{body}'
            f'<div></div><div class="r-axis mono">{ticks}</div><div></div></div>'
            f'<p class="legend"><span><i class="k-dot"></i>recalibrated, with 95% CI</span><span><i class="k-raw"></i>raw</span>'
            f'<span><i class="k-nc"></i>annotator noise ceiling ({nc:.4f})</span><span><i class="k-pb"></i>prior baseline ({pb["mean"]:.3f})</span></p>')


def _picture(name, alt) -> str:
    return (f'<picture><source srcset="{name}-dark.png" media="(prefers-color-scheme: dark)">'
            f'<img src="{name}.png" alt="{_e(alt)}" loading="lazy"></picture>')


def _vars(theme) -> str:
    return "".join(f"--{a}:{theme['series'][i]};" for a, i in ARM_SLOT.items())


CSS = """
:root{--bg:#fbfcfd;--panel:#fff;--ink:#141a21;--ink2:#48525f;--muted:#7a8594;--rule:#dde3e9;--link:#1f5fae;
--yes:#17805a;--no:#b3382c;--tie:#8a6a00;--nc:#141a21;VARS_LIGHT
--display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;--sans:"IBM Plex Sans",system-ui,sans-serif;
--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--panel:#121820;--ink:#e8edf3;--ink2:#b3bdc9;--muted:#8591a0;
--rule:#253040;--link:#79b0f2;--yes:#46c08f;--no:#f27a6b;--tie:#dcae45;--nc:#e8edf3;VARS_DARK}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans);-webkit-text-size-adjust:100%}
main{max-width:64rem;margin:0 auto;padding:3rem 16px 4rem}
a{color:var(--link);text-underline-offset:2px}a:focus-visible,summary:focus-visible{outline:2px solid var(--link);outline-offset:3px;border-radius:2px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{font:500 .8rem/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 1.25rem}
h1{font:600 clamp(2rem,5.2vw,3.4rem)/1.05 var(--display);letter-spacing:-.01em;margin:0 0 1rem;max-width:22ch}
.lead{font-size:1.15rem;color:var(--ink2);max-width:62ch;margin:0 0 1.25rem}
.facts{display:flex;flex-wrap:wrap;gap:.4rem 1.5rem;font:.85rem var(--mono);color:var(--muted);margin:0;padding:0;list-style:none}
.facts b{color:var(--ink);font-weight:500}
section{margin-top:4rem}
h2{font:600 1.5rem/1.2 var(--display);margin:0 0 .35rem}
h2+.sub{color:var(--ink2);margin:0 0 1.5rem;max-width:64ch}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:1.25rem 1.25rem 1rem}
.ruler{display:grid;grid-template-columns:minmax(12rem,17.5rem) 1fr 3.5rem;align-items:center;column-gap:1rem;row-gap:.2rem}
.r-lab{font-size:.92rem;display:flex;align-items:center;gap:.55rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sw{width:.7rem;height:.7rem;border-radius:2px;background:var(--c);flex:none}
.r-track{position:relative;height:1.9rem}
.r-track::before{content:"";position:absolute;left:0;right:0;top:50%;border-top:1px solid var(--rule)}
.r-ci{position:absolute;top:calc(50% - 3px);height:6px;border-radius:3px;background:var(--c);opacity:.35}
.r-dot,.r-raw{position:absolute;top:50%;width:.8rem;height:.8rem;border-radius:50%;transform:translate(-50%,-50%)}
.r-dot{background:var(--c);box-shadow:0 0 0 2px var(--panel)}
.r-raw{border:2px solid var(--c);background:var(--panel);width:.6rem;height:.6rem}
.nc,.pb{position:absolute;top:0;bottom:0;border-left:1.5px dashed var(--nc);opacity:.55}
.pb{border-left-style:dotted;border-color:var(--muted);opacity:.9}
.r-val{text-align:right;font-size:.85rem;color:var(--ink2)}
.r-axis{position:relative;height:1.6rem;border-top:1px solid var(--rule);margin-top:.3rem;font-size:.72rem;color:var(--muted)}
.r-axis span{position:absolute;top:.35rem;transform:translateX(-50%)}
.legend{display:flex;flex-wrap:wrap;gap:.5rem 1.25rem;font-size:.82rem;color:var(--ink2);margin:.9rem 0 0}
.legend span{display:inline-flex;align-items:center;gap:.45rem}.legend i{display:inline-block}
.k-dot{width:.7rem;height:.7rem;border-radius:50%;background:var(--ink2)}
.k-raw{width:.6rem;height:.6rem;border-radius:50%;border:2px solid var(--ink2)}
.k-nc,.k-pb{width:0;height:.9rem;border-left:1.5px dashed var(--nc)}.k-pb{border-left:1.5px dotted var(--muted)}
.verdict{border-left:4px solid var(--no);padding:.25rem 0 .25rem 1rem;margin:0 0 1.5rem;font-size:1.1rem;max-width:64ch}
.verdict.yes{border-color:var(--yes)}.verdict.tie{border-color:var(--tie)}
.ledger{list-style:none;margin:0;padding:0;border-top:1px solid var(--rule)}
.h{display:grid;grid-template-columns:1fr auto;gap:.2rem 1rem;padding:.85rem 0;border-bottom:1px solid var(--rule)}
.h-q{font-size:1rem}.h-q b{font-weight:600}
.role{display:inline-block;font:500 .7rem/1 var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-right:.6rem}
.h-v{text-align:right}.h-n{grid-column:1/-1;font-size:.82rem;color:var(--ink2)}.ci{color:var(--muted)}
.chip{display:inline-block;font:600 .75rem/1 var(--mono);text-transform:uppercase;letter-spacing:.05em;padding:.35rem .55rem;border-radius:4px;border:1.5px solid currentColor}
.chip.yes{color:var(--yes)}.chip.no{color:var(--no)}.chip.tie{color:var(--tie)}
figure{margin:0}figure img{display:block;width:100%;height:auto}
figcaption{font-size:.88rem;color:var(--ink2);margin-top:.75rem;max-width:64ch}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th{font:500 .72rem/1.2 var(--mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);text-align:right;padding:.5rem .6rem;border-bottom:1px solid var(--ink2);white-space:nowrap}
td{padding:.45rem .6rem;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;font-family:var(--mono);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-family:var(--sans)}
details{border-top:1px solid var(--rule);padding:.9rem 0}details:last-of-type{border-bottom:1px solid var(--rule)}
summary{cursor:pointer;font:600 1.05rem var(--display)}summary::marker{color:var(--muted)}
details>*:not(summary){margin-top:1rem}
.note{font-size:.88rem;color:var(--muted)}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:.9rem 1rem;overflow-x:auto;font:.85rem/1.5 var(--mono)}
footer{margin-top:4rem;padding-top:1.25rem;border-top:1px solid var(--rule);font-size:.82rem;color:var(--muted)}
footer .mono{word-break:break-all}
@media (max-width:600px){main{padding-top:2rem}.ruler{grid-template-columns:1fr 3rem;column-gap:.5rem}
.r-lab{grid-column:1/-1;margin-top:.5rem}.ruler>div:nth-last-child(3),.ruler>div:last-child{display:none}
.ruler .r-axis{grid-column:1/2}.h{grid-template-columns:1fr}.h-v{text-align:left}.panel{padding:1rem .75rem}}
"""


def render(results: dict, out_dir: Path = Path("docs")) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    headline_figure(results, out / "headline.png")
    reliability_figure(results, out / "reliability.png")

    arm_rows = []

    def add(key, m):
        for kind in ("raw", "recal"):
            x = m[kind]
            nf = x.get("noise_floor") or {}
            arm_rows.append([_label(key), kind, _mean_ci(x["soft_brier"]), _f(x.get("js")), _f(x.get("ece")),
                             _mean_ci({"mean": nf["excess"], "ci": nf["ci"]}) if nf else "—"])

    for arm, m in results["flat"].items():
        add(arm, m)
    for arm, by_size in results["curves"].items():
        for s in sorted(by_size, key=int):
            add(f"{arm}@{s}", by_size[s])
    pb = results["prior_baseline"]
    arm_rows.append(["Prior baseline", "—", _mean_ci(pb), "—", "—", "—"])

    pw = {a: m["recal"].get("per_wording") for a, m in results["flat"].items() if m["recal"].get("per_wording")}
    wids = sorted({w for d in pw.values() for w in d}, key=str)
    pw_rows = [[_label(a)] + [_f(d.get(w)) for w in wids] for a, d in pw.items()]

    comps = results["comparisons"]
    registered = [c for c in comps if c["family"] in ("primary", "secondary")]
    primary = [c for c in comps if c["family"] == "primary"]
    expl = [c for c in comps if c["family"] == "exploratory"]
    cost_rows = [[_label(c["arm"]), _f(c.get("per_1k_usd")), _f(c.get("train_gpu_hours"), 2), _f(c.get("train_usd"), 2)]
                 for c in results["cost"]]

    meta = results["meta"]
    n = meta.get("n_items")
    cross = results.get("crossover") or {}
    best = SHORT.get(cross.get("best_hosted"), cross.get("best_hosted") or "the best hosted model")
    if cross.get("size"):
        thesis = f"Fine-tuned Laya overtakes {best} at {SIZES_LABEL.get(int(cross['size']), cross['size'])} labelled examples."
    else:
        thesis = f"No amount of fine-tuning caught {best}."
    if primary:
        a, b, _ = _split_name(primary[0]["name"])
        cls, text = _verdict(primary[0])
        pd, (lo, hi) = primary[0]["diff"], primary[0]["ci"]
        better = _label(b) if pd > 0 else _label(a)
        verdict = (f'<p class="verdict {cls}"><b>Primary hypothesis: {text.lower()}.</b> '
                   f'{_e(_label(a))} vs {_e(_label(b))}: difference {_f(pd)} (95% CI {_f(lo)} to {_f(hi)}). '
                   + (f'{_e(better)} tracks the crowd more closely.' if cls != "tie" else "The two cannot be told apart.") + "</p>")
    else:
        verdict = ""
    hashes = "".join(f"<tr><td>{_e(_label(a))}</td><td>{_e(h)}</td></tr>" for a, h in meta.get("config_hashes", {}).items())
    tag = _e(meta.get("prereg_tag"))

    body = f"""<main>
<header>
<p class="eyebrow">Crowdcal · pre-registered calibration benchmark · {tag}</p>
<h1>{_e(thesis)}</h1>
<p class="lead">When a model says 0.7, do about 70% of people agree? We compared each model's probability of entailment
with the share of 100 human annotators who chose entailment, on {_e(f"{n:,}" if isinstance(n, int) else n)} deliberately
ambiguous ChaosNLI items. The score is soft Brier, (p − q)²: lower means the model's confidence tracks the crowd.</p>
<ul class="facts"><li><b>{_e(f"{n:,}" if isinstance(n, int) else n)}</b> test items</li><li><b>100</b> annotators each</li>
<li><b>5</b> models</li><li><b>4</b> training sizes</li><li><a href="{REPO}/blob/main/PREREG.md">Pre-registration</a></li>
<li><a href="{REPO}">Code and raw data</a></li></ul>
</header>

<section aria-labelledby="s-ruler">
<h2 id="s-ruler">Every model on one scale</h2>
<p class="sub">Recalibrated soft Brier on the test set, best first. The dashed line is the best any model could score given
annotator noise; the dotted line is always predicting the training base rate.</p>
<div class="panel">{_ruler(results)}</div>
</section>

<section aria-labelledby="s-primary">
<h2 id="s-primary">Primary result</h2>
<p class="sub">Registered before any model was run. Every comparison predicts the first model scores lower (better).</p>
{verdict}
{_ledger(registered)}
</section>

<section aria-labelledby="s-curve">
<h2 id="s-curve">What labelled data buys</h2>
<p class="sub">Fine-tuned models at 100, 1k, 10k and 30k training examples, against the models that get no training.</p>
<figure class="panel">{_picture("headline", TITLE)}<figcaption>At 100 and 1k examples the bands also include variation across three
training seeds; 10k and 30k have one seed each.</figcaption></figure>
</section>

<section aria-labelledby="s-rel">
<h2 id="s-rel">Reliability: does 0.7 mean 70%?</h2>
<p class="sub">Test items grouped into ten equal-sized bins by predicted probability. Points on the diagonal mean the
model's probability matches the share of annotators who agreed.</p>
<figure class="panel">{_picture("reliability", "Reliability plots: mean predicted probability against mean human share")}</figure>
</section>

<section aria-labelledby="s-cost">
<h2 id="s-cost">Cost</h2>
<p class="sub">API arms at list price through OpenRouter; local arms at $0.35 per T4 GPU-hour. Training cost is shown
separately, not spread across decisions.</p>
{_table(["Model", "USD per 1,000 decisions", "Training GPU-hours", "Training USD"], cost_rows)}
</section>

<section aria-labelledby="s-more">
<h2 id="s-more">All the numbers</h2>
<p class="sub">Everything below rebuilds from the frozen raw responses with <code class="mono">crowdcal report</code>.</p>
<details><summary>Every model, raw and recalibrated</summary>
<p class="note">Soft Brier, Jensen–Shannon distance, expected calibration error, and excess over the annotator noise
ceiling. Lower is better throughout; 95% CIs in brackets.</p>
{_table(["Model", "Scores", "Soft Brier [95% CI]", "JS", "ECE", "Excess over noise [95% CI]"], arm_rows)}</details>
<details><summary>Per question wording</summary>
{_table(["Model"] + [f"Wording {w}" for w in wids], pw_rows) if pw_rows else "<p>—</p>"}</details>
<details><summary>Paired comparisons (registered)</summary>{_comp_table(registered)}</details>
<details><summary>Exploratory comparisons</summary>
<p class="note">Not pre-registered as confirmatory, and not corrected for multiple comparisons.</p>
{_comp_table(expl) if expl else "<p>—</p>"}</details>
<details><summary>Reproduce</summary>
<pre>git clone {REPO}
cd crowdcal &amp;&amp; uv sync
uv run crowdcal report   # rebuilds this page from data/raw</pre>
<div class="scroll"><table><thead><tr><th>Model</th><th>Cache config hash</th></tr></thead><tbody>{hashes}</tbody></table></div>
</details>
</section>

<footer>Generated {_e(meta.get("generated"))} · pre-registration tag <span class="mono">{tag}</span> ·
<a href="{REPO}">{REPO.split("//")[1]}</a> · code MIT, data CC BY 4.0</footer>
</main>"""
    css = CSS.replace("VARS_LIGHT", _vars(THEMES["light"])).replace("VARS_DARK", _vars(THEMES["dark"]))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="color-scheme" content="light dark"><title>Crowdcal — {_e(thesis)}</title>'
            f'<meta name="description" content="{_e(TITLE)} Pre-registered calibration benchmark against ChaosNLI human label distributions.">'
            f'<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600'
            f'&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">'
            f"<style>{css}</style></head><body>{body}</body></html>")
    (out / "index.html").write_text(page, encoding="utf-8")
