import time

import numpy as np
import pytest

from crowdcal import stats


def test_bootstrap_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 500)
    r = stats.bootstrap_mean(x, n_boot=2000)
    assert r["mean"] == pytest.approx(x.mean())
    assert r["ci"][0] < x.mean() < r["ci"][1]


def test_paired_ci_excludes_zero_for_clear_effect():
    rng = np.random.default_rng(1)
    b = rng.normal(0.2, 0.1, 1000)
    a = b - 0.05 + rng.normal(0, 0.02, 1000)
    r = stats.paired_bootstrap(a, b, n_boot=2000)
    assert r["diff"] == pytest.approx(-0.05, abs=0.01)
    assert r["ci"][1] < 0 and r["p"] < 0.01


def test_paired_ci_includes_zero_for_no_effect():
    rng = np.random.default_rng(2)
    b = rng.normal(0.2, 0.1, 1000)
    e = rng.normal(0, 0.02, 1000)
    a = b + e - e.mean()  # true mean difference exactly 0
    r = stats.paired_bootstrap(a, b, n_boot=2000)
    assert r["ci"][0] < 0 < r["ci"][1] and r["p"] > 0.05


def test_seed_rows_add_training_noise():
    n = 200
    base = np.random.default_rng(3).normal(0.2, 0.1, n)
    seeds = base + np.array([[-0.1], [0.0], [0.1]])  # 3 seed rows, shifted by +-0.1
    one = stats.bootstrap_mean(seeds[1], n_boot=2000)
    two = stats.bootstrap_mean(seeds, n_boot=2000)
    assert two["ci"][1] - two["ci"][0] > 3 * (one["ci"][1] - one["ci"][0])
    assert two["mean"] == pytest.approx(seeds.mean())
    # 2-D paired: seed variance in arm a widens the CI vs. its 1-D version
    p2 = stats.paired_bootstrap(seeds, base + 0.3, n_boot=2000)
    p1 = stats.paired_bootstrap(seeds[1], base + 0.3, n_boot=2000)
    assert p2["ci"][1] - p2["ci"][0] > p1["ci"][1] - p1["ci"][0]


def test_speed():
    a = np.random.default_rng(4).random((3, 3000))
    b = np.random.default_rng(5).random((3, 3000))
    t = time.time()
    stats.paired_bootstrap(a, b)
    assert time.time() - t < 5


def test_holm_textbook():
    # Holm 1979 style: p = .01,.04,.03,.005 -> sorted .005,.01,.03,.04 x 4,3,2,1 = .02,.03,.06,.04 -> monotone .02,.03,.06,.06
    assert stats.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def test_crossover():
    d = {100: {"ci": [-0.1, 0.1]}, 1000: {"ci": [-0.1, -0.01]}, 10000: {"ci": [-0.2, -0.1]}}
    assert stats.crossover(d) == 1000
    assert stats.crossover({100: {"ci": [-0.1, 0.0]}}) is None
