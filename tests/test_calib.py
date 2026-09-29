import json

import numpy as np
import pytest
from scipy.special import expit, logit

from crowdcal import calib


def test_temperature_recovers_known_T():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 5000)
    q = expit(logit(p) / 2.0)  # p was overconfident by T=2 relative to q
    fit = calib.fit_temperature(p, q)
    assert fit["method"] == "temperature"
    assert fit["T"] == pytest.approx(2.0, rel=1e-2)
    assert np.allclose(calib.apply(fit, p), q, atol=1e-3)


def test_isotonic_monotone_and_json():
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.4])
    q = np.array([0.5, 0.2, 0.3, 0.6, 0.8])
    fit = calib.fit_isotonic(p, q)
    json.dumps(fit)
    assert fit["x"] == [0.1, 0.2, 0.3, 0.4]
    assert fit["y"] == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.7])  # PAVA by hand
    assert np.all(np.diff(fit["y"]) >= -1e-12)
    out = calib.apply(fit, np.array([0.1, 0.25, 0.4]))
    assert out[0] == pytest.approx(fit["y"][0]) and out[2] == pytest.approx(fit["y"][-1])


def test_apply_temperature_identity_and_bad_method():
    p = np.array([0.2, 0.9])
    assert np.allclose(calib.apply({"method": "temperature", "T": 1.0}, p), p)
    with pytest.raises(ValueError):
        calib.apply({"method": "nope"}, p)
