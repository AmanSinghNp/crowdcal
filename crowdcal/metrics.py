"""Soft-label metrics. All inputs are 1-D float arrays of equal length. OWNER: numeric agent."""
import numpy as np
from scipy.special import xlogy

from crowdcal.stats import bootstrap_mean


def soft_brier(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(q, float)) ** 2


def _h2(x: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in bits; xlogy makes 0*log(0) = 0."""
    return -(xlogy(x, x) + xlogy(1 - x, 1 - x)) / np.log(2)


def js_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = np.asarray(p, float), np.asarray(q, float)
    div = _h2((p + q) / 2) - (_h2(p) + _h2(q)) / 2
    return np.sqrt(np.clip(div, 0.0, None))


def _noise(q: np.ndarray, n: np.ndarray) -> np.ndarray:
    q, n = np.asarray(q, float), np.asarray(n, float)
    return q * (1 - q) / (n - 1)


def noise_ceiling(q: np.ndarray, n: np.ndarray) -> float:
    return float(np.mean(_noise(q, n)))


def reliability(p: np.ndarray, q: np.ndarray, bins: int = 10) -> dict:
    p, q = np.asarray(p, float), np.asarray(q, float)
    order = np.argsort(p, kind="stable")
    chunks = np.array_split(order, bins)
    return {
        "p": [float(p[c].mean()) for c in chunks],
        "q": [float(q[c].mean()) for c in chunks],
        "count": [int(len(c)) for c in chunks],
    }


def ece(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    r = reliability(p, q, bins)
    w = np.array(r["count"], float)
    return float(np.sum(w * np.abs(np.array(r["p"]) - np.array(r["q"]))) / w.sum())


def noise_floor_test(p, q, n, n_boot: int = 10_000, seed: int = 0) -> dict:
    excess = soft_brier(p, q) - _noise(q, n)  # per item; its mean = Brier - ceiling
    b = bootstrap_mean(excess, n_boot=n_boot, seed=seed)
    return {"excess": b["mean"], "ci": b["ci"]}
