"""Soft-label metrics. All inputs are 1-D float arrays of equal length. OWNER: numeric agent."""
import numpy as np


def soft_brier(p: np.ndarray, q: np.ndarray) -> np.ndarray: ...          # per-item (p-q)^2
def js_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray: ...         # per-item, Bernoulli, base 2
def noise_ceiling(q: np.ndarray, n: np.ndarray) -> float: ...            # mean q(1-q)/(n-1)
def reliability(p: np.ndarray, q: np.ndarray, bins: int = 10) -> dict: ...  # {"p": [...], "q": [...], "count": [...]} equal-mass bins
def ece(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float: ...
def noise_floor_test(p, q, n, n_boot: int = 10_000, seed: int = 0) -> dict: ...  # {"excess", "ci": [lo, hi]}
