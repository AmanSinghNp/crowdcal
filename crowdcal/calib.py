"""Recalibration fitted identically for every arm. OWNER: numeric agent."""
import numpy as np


def fit_temperature(p: np.ndarray, q: np.ndarray) -> dict: ...   # {"method": "temperature", "T": float}; minimizes soft Brier
def fit_isotonic(p: np.ndarray, q: np.ndarray) -> dict: ...      # {"method": "isotonic", "x": [...], "y": [...]}; JSON-serializable
def apply(params: dict, p: np.ndarray) -> np.ndarray: ...
