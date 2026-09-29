"""Paired bootstrap, Holm, crossover. OWNER: numeric agent.

Per-item loss arrays may be 2-D (seeds, items): each bootstrap resample draws one seed row
per arm (so CIs include training noise) and one shared resample of item indices (paired).
1-D arrays are treated as a single seed.
"""
import numpy as np


def bootstrap_mean(loss: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict: ...  # {"mean", "ci": [lo, hi]}
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict: ...
    # diff = mean(a - b); {"diff", "ci": [lo, hi], "p": two-sided bootstrap p}
def holm(pvals: list[float]) -> list[float]: ...                        # adjusted, same order
def crossover(diffs_by_size: dict[int, dict]) -> int | None: ...        # smallest size whose ci[1] < 0
