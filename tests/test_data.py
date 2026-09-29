import dataclasses
import hashlib
from pathlib import Path

import pytest

from crowdcal import data
from crowdcal.data import SPLITS, assert_no_leakage, load_wordings, prior_baseline, synthetic_splits

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_deterministic_and_shape():
    a, b, c = synthetic_splits(0), synthetic_splits(0), synthetic_splits(1)
    assert list(a) == list(SPLITS) and a == b and a != c
    assert all(i.dataset == "synthetic" for s in a.values() for i in s)
    assert all(i.label in (0, 1) and i.q is None for i in a["train@100"])
    assert all(i.n == 5 and 0 <= i.q <= 1 for s in ("dev", "calib") for i in a[s])
    assert len(a["test"]) == 500 and all(i.n == 100 and 0 <= i.q <= 1 for i in a["test"])
    assert 0 < prior_baseline(a["train@30k"]) < 1
    assert_no_leakage(a)


def test_train_nesting_synthetic():
    s = synthetic_splits(3)
    names = ["train@100", "train@1k", "train@10k", "train@30k"]
    for small, big in zip(names, names[1:]):
        assert {i.id for i in s[small]} < {i.id for i in s[big]}


def test_leakage_guard_raises():
    s = synthetic_splits(0)
    t = s["test"][0]
    same_id = dataclasses.replace(s["dev"][0], id=t.id)
    same_text = dataclasses.replace(s["calib"][0], premise=t.premise, hypothesis=t.hypothesis)
    for split, bad in (("dev", same_id), ("calib", same_text), ("train@100", same_id)):
        with pytest.raises(AssertionError):
            assert_no_leakage({**s, split: s[split] + [bad]})
    with pytest.raises(AssertionError):  # dev/calib overlap
        assert_no_leakage({**s, "calib": s["calib"] + [s["dev"][0]]})


def test_wordings():
    ws = load_wordings(ROOT / "config/wordings.json")
    assert [w.id for w in ws] == [1, 2, 3]
    assert all(w.sha == hashlib.sha256(w.text.encode()).hexdigest() for w in ws)


needs_data = pytest.mark.skipif(not (ROOT / "data/cache/multinli_1.0.zip").exists() or not (ROOT / "data/splits/manifest.json").exists(),
                                reason="real data not built")


@needs_data
def test_real_splits(monkeypatch):
    monkeypatch.chdir(ROOT)
    s = {k: data.load_split(k) for k in SPLITS}
    assert_no_leakage(s)
    names = ["train@100", "train@1k", "train@10k", "train@30k"]
    for small, big in zip(names, names[1:]):
        assert {i.id for i in s[small]} < {i.id for i in s[big]}
    assert len(s["test"]) == 3113 and all(i.n == 100 for i in s["test"])
    assert all(i.n and i.n <= 5 for i in s["dev"])
    assert 0.2 < prior_baseline(s["train@30k"]) < 0.5
