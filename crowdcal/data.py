"""Items, wordings, splits, leakage guard, synthetic data. OWNER: data agent."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    id: str                  # globally unique, e.g. "chaosnli-mnli:12345" / "mnli-train:67890"
    dataset: str             # "chaosnli-snli" | "chaosnli-mnli" | "mnli-train" | "mnli-dev" | "synthetic"
    premise: str
    hypothesis: str
    q: float | None = None   # human share choosing entailment (soft label); None for hard-label train items
    n: int | None = None     # annotator count behind q
    label: int | None = None # hard label: 1 = entailment, 0 = not entailment


@dataclass(frozen=True)
class Wording:
    id: int
    text: str
    sha: str                 # sha256 of text


SPLITS = ("train@100", "train@1k", "train@10k", "train@30k", "dev", "calib", "test")


def load_wordings(path: Path = Path("config/wordings.json")) -> list[Wording]: ...
def build_splits(out_dir: Path = Path("data/splits"), seed: int = 0) -> dict: ...   # returns manifest
def load_split(name: str, splits_dir: Path = Path("data/splits")) -> list[Item]: ...
def assert_no_leakage(splits: dict[str, list[Item]]) -> None: ...  # raises AssertionError
def synthetic_splits(seed: int = 0) -> dict[str, list[Item]]: ...  # same keys as SPLITS, small sizes
def prior_baseline(train: list[Item]) -> float: ...                # mean entailment rate
