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
from pathlib import Path


def headline_figure(results: dict, path: Path) -> None: ...
def render(results: dict, out_dir: Path = Path("docs")) -> None: ...
