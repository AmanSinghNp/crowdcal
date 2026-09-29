"""Keyed append-only JSONL cache. OWNER: runners agent.

Row schema (SPEC §9): key, config_hash, arm, model_version, provider, dataset, dataset_rev,
item_id, wording_id, wording_sha, repeat, p_yes, raw, usage{in,out,cost_usd}, status, ts.
One file per (arm, config_hash): <raw_dir>/<arm>/<config_hash>.jsonl
"""
from pathlib import Path


class MixedConfigError(Exception): ...


def config_hash(arm: str, model_version: str, provider: str, elicitation: dict) -> str: ...
def row_key(config_hash: str, dataset: str, dataset_rev: str, item_id: str, wording_sha: str | None, repeat: int) -> str: ...


class Cache:
    def __init__(self, raw_dir: Path, arm: str, config_hash: str): ...
    def get(self, key: str) -> dict | None: ...
    def append(self, row: dict) -> None: ...


def load_arm(raw_dir: Path, arm: str) -> list[dict]: ...  # raises MixedConfigError if >1 config_hash file for arm
