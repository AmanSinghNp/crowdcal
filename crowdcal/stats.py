"""Paired bootstrap, Holm, crossover. OWNER: numeric agent.

Per-item loss arrays may be 2-D (seeds, items): each bootstrap resample draws one seed row
per arm (so CIs include training noise) and one shared resample of item indices (paired).
1-D arrays are treated as a single seed.
"""
import numpy as np

_CHUNK = 500  # resamples per block; bounds the (chunk, items) index matrix in memory


def _boot_diffs(arrs: list, signs: list, n_boot: int, seed: int) -> np.ndarray:
    """Per-resample mean of sum(sign * arr[seed_row, items]); one shared item resample."""
    arrs = [np.atleast_2d(np.asarray(a, float)) for a in arrs]
    rng = np.random.default_rng(seed)
    n = arrs[0].shape[1]
    out = np.empty(n_boot)
    for s in range(0, n_boot, _CHUNK):
        k = min(_CHUNK, n_boot - s)
        idx = rng.integers(0, n, size=(k, n))
        tot = 0.0
        for a, sign in zip(arrs, signs):
            rows = rng.integers(0, a.shape[0], size=(k, 1))
            tot = tot + sign * a[rows, idx].mean(axis=1)
        out[s:s + k] = tot
    return out


def _ci(d: np.ndarray) -> list:
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def bootstrap_mean(loss: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    d = _boot_diffs([loss], [1], n_boot, seed)
    return {"mean": float(np.mean(loss)), "ci": _ci(d)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    d = _boot_diffs([a, b], [1, -1], n_boot, seed)
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"diff": float(np.mean(a) - np.mean(b)), "ci": _ci(d), "p": float(p)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])  # running max keeps it monotone
        adj[i] = min(1.0, run)
    return adj.tolist()


def crossover(diffs_by_size: dict[int, dict]) -> int | None:
    hits = [s for s, d in diffs_by_size.items() if d["ci"][1] < 0]
    return min(hits) if hits else None
