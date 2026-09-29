"""Items, wordings, splits, leakage guard, synthetic data. OWNER: data agent."""
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


CACHE = Path("data/cache")
HF = "https://huggingface.co/datasets/earino/chaosnli/resolve/eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b/raw/"
# name -> (url, sha256). The original ChaosNLI Dropbox link is dead (HTML "Invalid Link"), so we use the
# HF community mirror pinned to a commit; its files are byte-identical to the Kaggle copy of chaosNLI_v1.0.
SOURCES = {
    "chaosnli_snli.jsonl": (HF + "chaosNLI_snli.jsonl", "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1"),
    "chaosnli_mnli_m.jsonl": (HF + "chaosNLI_mnli_m.jsonl", "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d"),
    "multinli_1.0.zip": ("https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip", "049f507b9e36b1fcb756cfd5aeb3b7a0cfcb84bf023793652987f7e7e0957822"),
    "snli_1.0.zip": ("https://nlp.stanford.edu/projects/snli/snli_1.0.zip", "afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e"),
}
TRAIN_SIZES = {"train@100": 100, "train@1k": 1000, "train@10k": 10000, "train@30k": 30000}
_MNLI = "multinli_1.0/multinli_1.0_{}.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wordings(path: Path = Path("config/wordings.json")) -> list[Wording]:
    ws = json.loads(Path(path).read_text())["wordings"]
    return [Wording(w["id"], w["text"], hashlib.sha256(w["text"].encode()).hexdigest()) for w in ws]


def _fetch(name: str, cache: Path = CACHE) -> Path:
    path = cache / name
    if not path.exists():
        url, sha = SOURCES[name]
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if sha and _sha256(tmp) != sha:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}")
        tmp.rename(path)
    return path


def _zip_rows(name: str, member: str, cache: Path):
    with zipfile.ZipFile(_fetch(name, cache)) as z, z.open(member) as f:
        for line in f:
            yield json.loads(line)


def _clean(s: str) -> str:
    return s.strip()


def _chaos(name: str, dataset: str, cache: Path) -> list[Item]:
    out = []
    for line in _fetch(name, cache).read_text().splitlines():
        r = json.loads(line)
        c = r["label_counter"]  # keys e/n/c are labels, NOT a count
        n = sum(c.values())
        ex = r["example"]
        out.append(Item(f"{dataset}:{r['uid']}", dataset, _clean(ex["premise"]), _clean(ex["hypothesis"]),
                        q=c.get("e", 0) / n, n=n))
    return out


def _mnli(kind: str, cache: Path) -> list[Item]:
    """kind: 'train' (hard labels) or 'dev_matched' (soft q from the 5 annotator labels). Drops gold '-'."""
    ds = "mnli-train" if kind == "train" else "mnli-dev"
    out = []
    for r in _zip_rows("multinli_1.0.zip", _MNLI.format(kind), cache):
        if r["gold_label"] == "-":
            continue
        p, h = _clean(r["sentence1"]), _clean(r["sentence2"])
        if kind == "train":
            out.append(Item(f"{ds}:{r['pairID']}", ds, p, h, label=int(r["gold_label"] == "entailment")))
        else:
            a = r["annotator_labels"]
            out.append(Item(f"{ds}:{r['pairID']}", ds, p, h, q=a.count("entailment") / len(a), n=len(a)))
    return out


def _pair(it: Item) -> tuple[str, str]:
    return " ".join(it.premise.split()), " ".join(it.hypothesis.split())


def _ids(items) -> list[str]:
    return [i.id for i in items]


def _strat_subsets(pool: list[Item], seed: int) -> dict[str, list[Item]]:
    """Nested stratified (by binary label) subsets: quotas are monotone in N over one fixed shuffle per stratum."""
    rng = random.Random(seed)
    pos = sorted((i for i in pool if i.label == 1), key=lambda i: i.id)
    neg = sorted((i for i in pool if i.label == 0), key=lambda i: i.id)
    rng.shuffle(pos)
    rng.shuffle(neg)
    frac = len(pos) / len(pool)
    out = {}
    for name, n in TRAIN_SIZES.items():
        k = round(n * frac)
        out[name] = pos[:k] + neg[: n - k]
    return out


def build_splits(out_dir: Path = Path("data/splits"), seed: int = 0) -> dict:
    cache = CACHE
    snli, mnli_m = _chaos("chaosnli_snli.jsonl", "chaosnli-snli", cache), _chaos("chaosnli_mnli_m.jsonl", "chaosnli-mnli", cache)
    test = snli + mnli_m
    dev_all, train_all = _mnli("dev_matched", cache), _mnli("train", cache)

    # provenance: ChaosNLI uids vs SNLI dev pairIDs / MNLI dev-matched pairIDs (ids, then text)
    snli_dev = {r["pairID"]: r for r in _zip_rows("snli_1.0.zip", "snli_1.0/snli_1.0_dev.jsonl", cache)}
    mnli_dev = {i.id.split(":", 1)[1]: i for i in dev_all}
    mnli_dev_raw = {r["pairID"] for r in _zip_rows("multinli_1.0.zip", _MNLI.format("dev_matched"), cache)}
    uid = lambda i: i.id.split(":", 1)[1]
    snli_id = [i for i in snli if uid(i) in snli_dev]
    snli_txt = [i for i in snli_id if _pair(i) == (" ".join(snli_dev[uid(i)]["sentence1"].split()), " ".join(snli_dev[uid(i)]["sentence2"].split()))]
    mnli_id = [i for i in mnli_m if uid(i) in mnli_dev_raw]
    mnli_txt = [i for i in mnli_m if uid(i) in mnli_dev and _pair(i) == _pair(mnli_dev[uid(i)])]
    prov = {
        "chaosnli_snli": {"total": len(snli), "uid_in_snli_dev": len(snli_id), "text_matches_snli_dev": len(snli_txt)},
        "chaosnli_mnli_m": {"total": len(mnli_m), "uid_in_mnli_dev_matched": len(mnli_id), "text_matches_mnli_dev_matched": len(mnli_txt)},
    }

    test_ids, test_pairs = set(_ids(test)), {_pair(i) for i in test}
    chaos_mnli_uids = {uid(i) for i in mnli_m}
    # dev/calib pool: MNLI dev-matched minus ChaosNLI-MNLI items (by uid or text), deduped by text pair
    seen, pool = set(test_pairs), []
    for i in sorted(dev_all, key=lambda i: i.id):
        if uid(i) in chaos_mnli_uids or _pair(i) in seen:
            continue
        seen.add(_pair(i))
        pool.append(i)
    random.Random(seed).shuffle(pool)
    half = len(pool) // 2
    splits = {"dev": pool[:half], "calib": pool[half:]}
    # train: drop pairs whose text occurs in test/dev/calib (guard), and duplicates
    blocked = {_pair(i) for i in test + pool}
    tp, train_pool = set(), []
    for i in train_all:
        if _pair(i) not in blocked and _pair(i) not in tp:
            tp.add(_pair(i))
            train_pool.append(i)
    splits = {**_strat_subsets(train_pool, seed), **splits, "test": test}
    assert set(splits) == set(SPLITS)
    assert_no_leakage(splits)
    assert len({i for s in ("dev", "calib", "test") for i in _ids(splits[s])}) == sum(len(splits[s]) for s in ("dev", "calib", "test"))
    assert test_ids.isdisjoint(_ids(train_pool))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (out_dir / f"{name}.ids.txt").write_text("".join(i.id + "\n" for i in items))
    files = {n: _sha256(cache / n) for n in SOURCES}
    manifest = {
        "seed": seed,
        "sources": {n: SOURCES[n][0] for n in SOURCES},
        "notes": "Original ChaosNLI Dropbox link is dead; files come from HF mirror earino/chaosnli @ eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b (byte-identical archive of chaosNLI_v1.0). Only SNLI and MNLI-m subsets are used.",
        "checksums": files,
        "dataset_rev": {"chaosnli-snli": files["chaosnli_snli.jsonl"], "chaosnli-mnli": files["chaosnli_mnli_m.jsonl"],
                        "mnli-train": files["multinli_1.0.zip"], "mnli-dev": files["multinli_1.0.zip"]},
        "counts": {k: len(v) for k, v in splits.items()},
        "counts_by_dataset": {"chaosnli-snli": len(snli), "chaosnli-mnli": len(mnli_m)},
        "train_pool_size": len(train_pool),
        "dev_calib_pool_size": len(pool),
        "chaosnli_provenance": prov,
        "id_file_sha256": {n: _sha256(out_dir / f"{n}.ids.txt") for n in splits},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(name: str, splits_dir: Path = Path("data/splits")) -> list[Item]:
    ids = (Path(splits_dir) / f"{name}.ids.txt").read_text().split()
    need = {i.split(":", 1)[0] for i in ids}
    src: dict[str, Item] = {}
    if "chaosnli-snli" in need:
        src.update({i.id: i for i in _chaos("chaosnli_snli.jsonl", "chaosnli-snli", CACHE)})
    if "chaosnli-mnli" in need:
        src.update({i.id: i for i in _chaos("chaosnli_mnli_m.jsonl", "chaosnli-mnli", CACHE)})
    if "mnli-dev" in need:
        src.update({i.id: i for i in _mnli("dev_matched", CACHE)})
    if "mnli-train" in need:
        src.update({i.id: i for i in _mnli("train", CACHE)})
    return [src[i] for i in ids]


def assert_no_leakage(splits: dict[str, list[Item]]) -> None:
    others = {k: v for k, v in splits.items() if k != "test"}
    ids = {k: set(_ids(v)) for k, v in splits.items()}
    pairs = {k: {_pair(i) for i in v} for k, v in splits.items()}
    for k in others:
        assert not ids.get("test", set()) & ids[k], f"test id leaks into {k}"
        assert not pairs.get("test", set()) & pairs[k], f"test (premise, hypothesis) pair leaks into {k}"
    assert not ids.get("dev", set()) & ids.get("calib", set()), "dev/calib id overlap"
    assert not pairs.get("dev", set()) & pairs.get("calib", set()), "dev/calib text-pair overlap"


def synthetic_splits(seed: int = 0) -> dict[str, list[Item]]:
    rng = np.random.default_rng(seed)
    mk = lambda split, i, **kw: Item(f"syn-{split}:{i}", "synthetic", f"premise {split} {i}", f"hypothesis {split} {i}", **kw)
    out = {}
    train = [mk("train", i, label=int(rng.random() < 0.35)) for i in range(200)]  # nested prefixes
    for name, n in zip(("train@100", "train@1k", "train@10k", "train@30k"), (20, 50, 100, 200)):
        out[name] = train[:n]
    for split in ("dev", "calib"):
        r = rng.beta(0.6, 1.2, 100)
        out[split] = [mk(split, i, q=int(rng.binomial(5, r[i])) / 5, n=5) for i in range(100)]
    r = rng.beta(0.6, 1.2, 500)
    out["test"] = [mk("test", i, q=int(rng.binomial(100, r[i])) / 100, n=100) for i in range(500)]
    return {k: out[k] for k in SPLITS}


def prior_baseline(train: list[Item]) -> float:
    return sum(i.label for i in train) / len(train)
