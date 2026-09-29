import pytest

from crowdcal.cache import Cache, MixedConfigError, config_hash, load_arm, row_key


def test_hash_stable_and_order_insensitive():
    a = config_hash("jev", "v1", "p", {"a": 1, "b": 2})
    assert a == config_hash("jev", "v1", "p", {"b": 2, "a": 1})
    assert len(a) == 16 and a != config_hash("jev", "v2", "p", {"a": 1, "b": 2})
    k = row_key(a, "d", "r", "i", "sha", 0)
    assert len(k) == 64 and k != row_key(a, "d", "r", "i", "sha", 1)


def test_roundtrip_and_latest_wins(tmp_path):
    c = Cache(tmp_path, "jev", "abc")
    c.append({"key": "k1", "status": "error", "p_yes": None})
    c.append({"key": "k1", "status": "ok", "p_yes": 0.7})
    c.append({"key": "k2", "status": "ok", "p_yes": 0.1})
    assert c.get("k1")["p_yes"] == 0.7 and c.get("nope") is None
    c2 = Cache(tmp_path, "jev", "abc")  # reload from disk
    assert c2.get("k1")["status"] == "ok"
    rows = load_arm(tmp_path, "jev")
    assert sorted(r["key"] for r in rows) == ["k1", "k2"]
    assert next(r for r in rows if r["key"] == "k1")["p_yes"] == 0.7


def test_load_arm_empty_and_mixed(tmp_path):
    assert load_arm(tmp_path, "none") == []
    Cache(tmp_path, "jev", "a").append({"key": "k"})
    Cache(tmp_path, "jev", "b").append({"key": "k"})
    with pytest.raises(MixedConfigError):
        load_arm(tmp_path, "jev")
