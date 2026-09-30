"""Keyed append-only JSONL cache. OWNER: runners agent.

Row schema (SPEC §9): key, config_hash, arm, model_version, provider, dataset, dataset_rev,
item_id, wording_id, wording_sha, repeat, p_yes, raw, usage{in,out,cost_usd}, status, ts.
One file per (arm, config_hash): <raw_dir>/<arm>/<config_hash>.jsonl
"""
import gzip
import hashlib
import json
from pathlib import Path


class MixedConfigError(Exception): ...


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_hash(arm: str, model_version: str, provider: str, elicitation: dict) -> str:
    return _sha([arm, model_version, provider, elicitation])[:16]


def row_key(config_hash: str, dataset: str, dataset_rev: str, item_id: str, wording_sha: str | None, repeat: int) -> str:
    return _sha([config_hash, dataset, dataset_rev, item_id, wording_sha, repeat])


def _read(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    gz = path.with_name(path.name + ".gz")
    if gz.exists():  # frozen (crowdcal freeze gzips every jsonl)
        text = gzip.decompress(gz.read_bytes()).decode()
    elif path.exists():
        text = path.read_text()
    else:
        text = ""
    if text:
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row  # later rows supersede earlier ones
    return rows


class Cache:
    def __init__(self, raw_dir: Path, arm: str, config_hash: str):
        self.path = Path(raw_dir) / arm / f"{config_hash}.jsonl"
        self.rows = _read(self.path)

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def append(self, row: dict) -> None:
        if self.path.with_name(self.path.name + ".gz").exists():
            raise RuntimeError(f"{self.path} is frozen (gzipped by crowdcal freeze); refusing to append")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows[row["key"]] = row


def load_arm(raw_dir: Path, arm: str) -> list[dict]:
    hashes = sorted({f.name.split(".")[0] for f in (Path(raw_dir) / arm).glob("*.jsonl*")})
    if len(hashes) > 1:
        raise MixedConfigError(f"arm {arm!r} has {len(hashes)} config hashes: {hashes}")
    return list(_read(Path(raw_dir) / arm / f"{hashes[0]}.jsonl").values()) if hashes else []
