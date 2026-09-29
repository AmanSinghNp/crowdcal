import numpy as np
import pytest

from crowdcal import metrics as m


def test_soft_brier():
    assert m.soft_brier(np.array([0.7]), np.array([0.5]))[0] == pytest.approx(0.04)


def test_js_distance():
    d = m.js_distance(np.array([0.0, 1.0, 0.3, 0.0]), np.array([1.0, 1.0, 0.3, 0.0]))
    assert d[0] == pytest.approx(1.0)  # disjoint point masses: 1 bit -> distance 1
    assert d[1] == 0 and d[2] == 0 and d[3] == 0
    # p=1, q=.5: JS div = H(.75) - .5 bits
    h = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert m.js_distance(np.array([1.0]), np.array([0.5]))[0] == pytest.approx(np.sqrt(h - 0.5))


def test_noise_ceiling():
    # .5*.5/4 = .0625 ; 0 ; mean = .03125
    assert m.noise_ceiling(np.array([0.5, 1.0]), np.array([5, 5])) == pytest.approx(0.03125)


def test_reliability_and_ece():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    q = np.array([0.0, 0.4, 0.6, 1.0])
    r = m.reliability(p, q, bins=2)
    assert r == {"p": pytest.approx([0.15, 0.85]), "q": pytest.approx([0.2, 0.8]), "count": [2, 2]}
    assert m.ece(p, q, bins=2) == pytest.approx(0.05)


def test_noise_floor_test():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.1, 0.9, 2000)
    n = np.full(2000, 5)
    # perfect model + sampling noise: excess ~ 0, CI covers 0
    p_true = rng.uniform(0.1, 0.9, 2000)
    q_obs = rng.binomial(5, p_true) / 5
    ok = m.noise_floor_test(p_true, q_obs, n, n_boot=1000)
    assert ok["ci"][0] < 0 < ok["ci"][1]
    # badly miscalibrated model: excess clearly > 0
    bad = m.noise_floor_test(np.clip(q_obs + 0.3, 0, 1), q_obs, n, n_boot=1000)
    assert bad["ci"][0] > 0
