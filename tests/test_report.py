from crowdcal.report import render


def M(b, w=None):
    return {"soft_brier": {"mean": b, "ci": [b - 0.01, b + 0.01]}, "js": 0.05, "ece": 0.03,
            "noise_floor": {"excess": b - 0.1, "ci": [b - 0.11, b - 0.09]}, "per_wording": w or {}}


def fake(crossover={"best_hosted": "jev", "size": 10000}):
    flat = {"jev": (0.19, {"0": 0.2, "1": 0.18}), "deepseek": (0.21, {"0": 0.22, "1": 0.2}), "laya-base": (0.24, {})}
    return {
        "meta": {"generated": "2026-01-01T00:00:00", "n_items": 1000, "config_hashes": {"jev": "abc<1>"}, "prereg_tag": "prereg-v1"},
        "noise_ceiling": 0.10,
        "prior_baseline": {"mean": 0.26, "ci": [0.25, 0.27]},
        "flat": {a: {"raw": M(b + 0.02, w), "recal": M(b, w)} for a, (b, w) in flat.items()},
        "curves": {"laya-ft": {s: {"raw": M(b + 0.02), "recal": M(b)} for s, b in zip([100, 1000, 10000, 30000], [0.25, 0.22, 0.185, 0.17])},
                   "mbert-ce": {s: {"raw": M(b + 0.02), "recal": M(b)} for s, b in zip([100, 1000, 10000, 30000], [0.26, 0.24, 0.21, 0.19])}},
        "reliability": {"jev (recal)": {"p": [0.1, 0.5, 0.9], "q": [0.15, 0.45, 0.8], "count": [10, 50, 20]},
                        "laya-ft@30k (recal)": {"p": [0.2, 0.6], "q": [0.2, 0.55], "count": [30, 40]}},
        "comparisons": [
            {"name": "laya-ft@30k - jev", "family": "primary", "diff": -0.02, "ci": [-0.03, -0.01], "p": 0.001, "p_holm": None},
            {"name": "laya-ft@100 - mbert-ce@100", "family": "secondary", "diff": -0.01, "ci": [-0.02, 0.0], "p": 0.04, "p_holm": 0.2},
            {"name": "isotonic vs platt", "family": "exploratory", "diff": 0.0, "ci": [-0.01, 0.01], "p": 0.5, "p_holm": None}],
        "crossover": crossover,
        "cost": [{"arm": "jev", "per_1k_usd": 0.5, "train_gpu_hours": None, "train_usd": None},
                 {"arm": "laya-ft", "per_1k_usd": 0.01, "train_gpu_hours": 2.0, "train_usd": 1.2}],
    }


def test_render(tmp_path):
    render(fake(), tmp_path)
    for f in ["headline.png", "headline.svg", "reliability.png", "index.html"]:
        assert (tmp_path / f).exists()
    h = (tmp_path / "index.html").read_text()
    for s in ["Primary result", "Exploratory", "Paired comparisons", "Cost", "Reliability", "laya-ft@30k - jev", "—", "abc&lt;1&gt;"]:
        assert s in h
    assert h.index("Primary result") < h.index("Reliability") < h.index("Cost") < h.index("Exploratory")


def test_crossover_none(tmp_path):
    render(fake({"best_hosted": "jev", "size": None}), tmp_path)
    assert (tmp_path / "headline.png").exists()
