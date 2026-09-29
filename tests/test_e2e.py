import json
import shutil
import subprocess

import pytest

from crowdcal import cli, data
from crowdcal.analysis import analyze, verify_sums, write_sums
from crowdcal.cache import MixedConfigError


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    assert cli.main(["demo", "--out", str(out)]) == 0
    return out


def _run(demo):
    prereg = json.loads((cli.ROOT / "config" / "prereg.json").read_text())
    return analyze(demo / "raw", data.synthetic_splits(0), {**prereg, "n_boot": 200}, demo / "calib2")


def test_demo_outputs(demo):
    assert (demo / "site" / "headline.png").exists() and (demo / "site" / "index.html").exists()
    assert "Primary result" in (demo / "site" / "index.html").read_text()
    assert (demo / "calib" / "jev.json").exists()


def test_results_shape_and_recal(demo):
    r = _run(demo)
    assert r["comparisons"][0]["family"] == "primary"
    sec = [c for c in r["comparisons"] if c["family"] == "secondary"]
    assert len(sec) == 5 and all(c["p_holm"] is not None for c in sec)
    jev = r["flat"]["jev"]  # distorted by temperature 1.6
    assert jev["recal"]["soft_brier"]["mean"] < jev["raw"]["soft_brier"]["mean"]
    assert r["crossover"]["best_hosted"] in ("jev", "deepseek", "laya-base")


def test_exclusion_is_global(demo):
    base = _run(demo)
    f = next((demo / "raw" / "jev").glob("*.jsonl"))
    lines = f.read_text().splitlines()
    gone = data.synthetic_splits(0)["test"][0].id
    row = next(json.loads(l) for l in lines if json.loads(l)["item_id"] == gone)
    f.write_text("\n".join(lines) + "\n" + json.dumps({**row, "status": "error", "p_yes": None}) + "\n")
    try:
        r = _run(demo)
        assert r["meta"]["exclusions"] == base["meta"]["exclusions"] + 1
        assert r["meta"]["n_items"] == base["meta"]["n_items"] - 1  # dropped from every arm, not just jev
    finally:
        f.write_text("\n".join(lines) + "\n")


def test_mixed_config_propagates(demo):
    f = next((demo / "raw" / "jev").glob("*.jsonl"))
    extra = f.with_name("other.jsonl")
    extra.write_text(f.read_text())
    try:
        with pytest.raises(MixedConfigError):
            _run(demo)
    finally:
        extra.unlink()


def test_sha256sums(demo, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(demo / "raw", raw)
    assert write_sums(raw) > 0
    verify_sums(raw)
    f = next((raw / "jev").glob("*.jsonl"))
    f.write_text(f.read_text() + "\n")
    with pytest.raises(RuntimeError):
        verify_sums(raw)


def test_prereg_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_has_prereg_tag", lambda: False)
    assert cli.main(["run", "--arm", "jev", "--split", "calib"]) == 2
    assert "prereg-v1" in capsys.readouterr().err
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)  # a fresh git repo has no tag
    subprocess.run(["git", "init", "-q"], check=True)
    assert not cli._has_prereg_tag()
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run(["git", "tag", "prereg-v1"], check=True)
    assert cli._has_prereg_tag()
