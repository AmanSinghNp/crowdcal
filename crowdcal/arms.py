"""Model arms + run loop. OWNER: runners agent."""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from crowdcal.cache import Cache, config_hash, row_key
from crowdcal.data import Item, Wording

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
WORDINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "wordings.json"


@dataclass
class Prediction:
    p_yes: float | None
    raw: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)   # {"in": int, "out": int, "cost_usd": float}
    status: str = "ok"                          # "ok" | "error"


class BudgetExceeded(Exception): ...

# Every arm exposes: name, model_version, provider, elicitation (dict), prompted (bool),
# and predict(item, wording) -> Prediction. wording is None for unprompted arms (mbert-ce).


def _context_template() -> str:
    return json.loads(WORDINGS_PATH.read_text())["context_template"]


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class FakeArm:
    """p = sigmoid(logit(q_or_label)/temperature + bias) + gaussian noise, clipped. Deterministic per (item, wording, seed)."""

    provider = "fake"

    def __init__(self, name, bias=0.0, temperature=1.0, noise=0.05, seed=0, prompted=True):
        self.name, self.bias, self.temperature, self.noise, self.seed, self.prompted = name, bias, temperature, noise, seed, prompted
        self.model_version = "fake-1"
        self.elicitation = {"kind": "fake", "bias": bias, "temperature": temperature, "noise": noise, "seed": seed}

    def predict(self, item: Item, wording: Wording | None) -> Prediction:
        base = item.q if item.q is not None else item.label
        if base is None:
            raise ValueError(f"item {item.id} has neither q nor label")
        base = min(max(float(base), 1e-3), 1 - 1e-3)
        digest = hashlib.sha256(f"{self.seed}|{item.id}|{wording.id if wording else None}".encode()).digest()
        eps = np.random.default_rng(int.from_bytes(digest[:8], "big")).normal(0.0, self.noise) if self.noise else 0.0
        p = _sigmoid(math.log(base / (1 - base)) / self.temperature + self.bias) + eps
        return Prediction(p_yes=float(min(max(p, 1e-4), 1 - 1e-4)), usage={"in": 0, "out": 0, "cost_usd": 0.0})


_NO_TOKEN = re.compile(r"[^a-z]")
_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*$")


class OpenRouterArm:
    prompted = True

    def __init__(self, name, model, provider, mode, api_key, client: httpx.Client | None = None, context_template: str | None = None):
        assert mode in ("native", "logprobs", "verbalized"), mode
        self.name, self.model, self.provider, self.mode, self.api_key = name, model, provider, mode, api_key
        self.model_version = model
        self.client = client or httpx.Client(timeout=60)
        self.context_template = context_template or _context_template()
        self.elicitation = {"mode": mode, "temperature": 0, "reasoning": "off", "context_template": self.context_template}
        if mode == "logprobs":
            self.elicitation.update(top_logprobs=20, max_tokens=5)
        if mode == "verbalized":
            self.elicitation.update(max_tokens=16)

    def _post(self, url: str, body: dict) -> tuple[dict | None, dict]:
        r = self.client.post(url, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            data = r.json()
        except ValueError:
            data = None
        if r.status_code >= 400 or not isinstance(data, dict):
            return None, {"http_status": r.status_code, "body": r.text[:500]}
        return data, data

    @staticmethod
    def _usage(data: dict) -> dict:
        u = data.get("usage") or {}
        return {"in": u.get("prompt_tokens", u.get("input_tokens", 0)),
                "out": u.get("completion_tokens", u.get("output_tokens", 0)),
                "cost_usd": float(u.get("cost") or 0.0)}

    def predict(self, item: Item, wording: Wording | None) -> Prediction:
        context = self.context_template.format(premise=item.premise, hypothesis=item.hypothesis)
        if self.mode == "native":
            # TODO(pilot): Jev is served by the Decisions API, not chat completions. Request/response shape is taken from
            # https://openrouter.ai/blog/insights/what-is-jev/ ; the full schema page (404 for us) was not readable, so
            # check (a) that `state` should carry premise+hypothesis and `instructions` the wording, (b) whether the
            # endpoint accepts provider pinning (we send none; Jev's only provider is TypeSafe), (c) usage.cost is present.
            body = {"model": self.model, "state": context,
                    "questions": {"q": {"type": "noul", "instructions": wording.text}}}
            url = DECISIONS_URL
        else:
            if self.mode == "logprobs":
                prompt = f"{context}\n\nQuestion: {wording.text}\nAnswer with a single word: Yes or No."
                extra = {"max_tokens": 5, "logprobs": True, "top_logprobs": 20}
            else:
                prompt = (f"{context}\n\nQuestion: {wording.text}\nGive the probability, from 0 to 100, that the answer "
                          "is Yes. Reply with only the number.")
                extra = {"max_tokens": 16}
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "provider": {"order": [self.provider], "allow_fallbacks": False},
                    "reasoning": {"enabled": False}, "usage": {"include": True}, **extra}
            url = CHAT_URL
        data, raw = self._post(url, body)
        if data is None:
            return Prediction(None, raw, status="error")
        usage = self._usage(data)
        try:
            p = {"native": self._parse_native, "logprobs": self._parse_logprobs, "verbalized": self._parse_verbalized}[self.mode](data)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            p = None
        if p is None or not (0.0 <= p <= 1.0) or math.isnan(p):
            return Prediction(None, raw, usage, "error")
        return Prediction(p, raw, usage)

    @staticmethod
    def _parse_native(data: dict) -> float:
        return float(data["answers"]["q"]["noul"])

    @staticmethod
    def _parse_logprobs(data: dict) -> float | None:
        yes = no = 0.0
        for t in data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
            tok = _NO_TOKEN.sub("", t["token"].lower())
            if tok == "yes":
                yes += math.exp(t["logprob"])
            elif tok == "no":
                no += math.exp(t["logprob"])
        return yes / (yes + no) if yes + no > 0 else None

    @staticmethod
    def _parse_verbalized(data: dict) -> float | None:
        m = _NUMBER.match(data["choices"][0]["message"]["content"])
        return float(m.group(1)) / 100 if m else None


class LocalArm:
    """Local HF arm. name starting with "laya" -> Laya package; otherwise a sequence-classification dir (mbert-ce)."""

    provider = "local"

    def __init__(self, name, hf_id, revision, prompted, context_template: str | None = None):
        self.name, self.hf_id, self.revision, self.prompted = name, hf_id, revision, prompted
        self.model_version = f"{hf_id}@{revision}"
        self.kind = "laya" if name.startswith("laya") else "seqcls"
        self.context_template = context_template or _context_template()
        self.elicitation = {"kind": self.kind}
        if prompted:
            self.elicitation["context_template"] = self.context_template
        self._model = self._tok = None

    def predict(self, item: Item, wording: Wording | None) -> Prediction:
        if self.kind == "laya":
            # TODO(pilot): unverified against a real install. Usage from the laya README: laya.load(hf_id[, subfolder=...])
            # then agent.predict(state, questions)["answers"][key]["noul"] = P(true) in [0,1]. laya.load has no documented
            # `revision` arg, so the pinned SHA is NOT enforced here; pin via HF_HUB cache / snapshot download in the pilot.
            # Also unclear which of english / typed-decisions subfolders the paper's "laya-base" means. Fine-tuned laya-ft
            # checkpoints will need their own loader.
            import laya
            if self._model is None:
                self._model = laya.load(self.hf_id)
            state = self.context_template.format(premise=item.premise, hypothesis=item.hypothesis)
            res = self._model.predict(state, {"q": {"type": "noul", "instructions": wording.text}})
            return Prediction(float(res["answers"]["q"]["noul"]), raw={"answer": res["answers"]["q"]})
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._model is None:
            rev = None if Path(self.hf_id).is_dir() else self.revision
            self._tok = AutoTokenizer.from_pretrained(self.hf_id, revision=rev)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.hf_id, revision=rev).eval()
        enc = self._tok(item.premise, item.hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            p = torch.softmax(self._model(**enc).logits, dim=-1)[0, 1].item()
        return Prediction(p)


def _spent(rows) -> float:
    return sum((r.get("usage") or {}).get("cost_usd", 0.0) or 0.0 for r in rows)


def run_arm(arm, items: list[Item], wordings: list[Wording] | None, raw_dir: Path,
            dataset_rev: str, repeats: int = 1, budget_usd: float = 20.0,
            max_retries: int = 3, sleep=time.sleep, backoff: float = 1.0) -> dict:
    """Returns {"calls", "cached", "failed", "cost_usd"}: calls = rows written this run (ok + error),
    cached = ok rows skipped, failed = error rows written, cost_usd = cumulative spend incl. earlier runs."""
    ch = config_hash(arm.name, arm.model_version, arm.provider, arm.elicitation)
    cache = Cache(raw_dir, arm.name, ch)
    spent = _spent(cache.rows.values())  # ponytail: superseded error rows' cost is not re-counted; negligible
    calls = cached = failed = 0
    for item in items:
        for w in (wordings if arm.prompted else [None]):
            for rep in range(repeats):
                key = row_key(ch, item.dataset, dataset_rev, item.id, w.sha if w else None, rep)
                old = cache.get(key)
                if old and old["status"] == "ok":
                    cached += 1
                    continue
                pred, cost = None, 0.0
                for attempt in range(max_retries + 1):
                    if spent >= budget_usd:
                        raise BudgetExceeded(f"spent ${spent:.4f} >= budget ${budget_usd}")
                    try:
                        pred = arm.predict(item, w)
                    except Exception as e:  # network errors etc.
                        pred = Prediction(None, {"exception": repr(e)}, status="error")
                    c = float(pred.usage.get("cost_usd") or 0.0)
                    cost += c
                    spent += c
                    if pred.status == "ok":
                        break
                    if attempt < max_retries:
                        sleep(backoff * 2 ** attempt)
                usage = {"in": pred.usage.get("in", 0), "out": pred.usage.get("out", 0), "cost_usd": cost}
                cache.append({
                    "key": key, "config_hash": ch, "arm": arm.name, "model_version": arm.model_version,
                    "response_model": (pred.raw or {}).get("model"), "provider": arm.provider,
                    "dataset": item.dataset, "dataset_rev": dataset_rev, "item_id": item.id,
                    "wording_id": w.id if w else None, "wording_sha": w.sha if w else None, "repeat": rep,
                    "p_yes": pred.p_yes, "raw": pred.raw, "usage": usage, "status": pred.status,
                    "ts": datetime.now(timezone.utc).isoformat()})
                calls += 1
                failed += pred.status != "ok"
    return {"calls": calls, "cached": cached, "failed": failed, "cost_usd": spent}
