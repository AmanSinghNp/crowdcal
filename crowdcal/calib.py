"""Recalibration fitted identically for every arm. OWNER: numeric agent."""
import numpy as np
from scipy.optimize import isotonic_regression, minimize_scalar
from scipy.special import expit, logit

_EPS = 1e-6


def _temp(p: np.ndarray, T: float) -> np.ndarray:
    return expit(logit(np.clip(p, _EPS, 1 - _EPS)) / T)


def fit_temperature(p: np.ndarray, q: np.ndarray) -> dict:
    p, q = np.asarray(p, float), np.asarray(q, float)
    res = minimize_scalar(lambda t: np.mean((_temp(p, np.exp(t)) - q) ** 2),
                          bounds=(-5, 5), method="bounded")
    return {"method": "temperature", "T": float(np.exp(res.x))}


def fit_isotonic(p: np.ndarray, q: np.ndarray) -> dict:
    p, q = np.asarray(p, float), np.asarray(q, float)
    x, inv = np.unique(p, return_inverse=True)  # sorted unique p; average q over ties
    w = np.bincount(inv).astype(float)
    y = isotonic_regression(np.bincount(inv, weights=q) / w, weights=w, increasing=True).x
    return {"method": "isotonic", "x": x.tolist(), "y": y.tolist()}


def apply(params: dict, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    if params["method"] == "temperature":
        return _temp(p, params["T"])
    if params["method"] == "isotonic":
        return np.interp(p, params["x"], params["y"])
    raise ValueError(f"unknown method {params['method']!r}")
