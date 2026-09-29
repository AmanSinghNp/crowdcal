"""Model arms + run loop. OWNER: runners agent."""
from dataclasses import dataclass, field
from pathlib import Path

from crowdcal.data import Item, Wording


@dataclass
class Prediction:
    p_yes: float | None
    raw: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)   # {"in": int, "out": int, "cost_usd": float}
    status: str = "ok"                          # "ok" | "error"


class BudgetExceeded(Exception): ...

# Every arm exposes: name, model_version, provider, elicitation (dict), prompted (bool),
# and predict(item, wording) -> Prediction. wording is None for unprompted arms (mbert-ce).
#
# FakeArm(name, bias=0.0, temperature=1.0, noise=0.05, seed=0, prompted=True)
#   p = sigmoid(logit(q_or_label)/temperature + bias) + gaussian noise, clipped. Deterministic per (item, wording, seed).
# OpenRouterArm(name, model, provider, mode: "native"|"logprobs"|"verbalized", api_key, client: httpx.Client | None)
# LocalArm(name, hf_id, revision, prompted)  # lazy-imports torch/transformers


def run_arm(arm, items: list[Item], wordings: list[Wording] | None, raw_dir: Path,
            dataset_rev: str, repeats: int = 1, budget_usd: float = 20.0,
            max_retries: int = 3) -> dict: ...  # summary {"calls", "cached", "failed", "cost_usd"}
