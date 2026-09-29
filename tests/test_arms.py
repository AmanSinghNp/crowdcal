import json
import math

import httpx
import pytest

from crowdcal.arms import BudgetExceeded, FakeArm, OpenRouterArm, Prediction, run_arm
from crowdcal.cache import load_arm
from crowdcal.data import Item, Wording

TPL = "Premise: {premise}\nHypothesis: {hypothesis}"
W = Wording(1, "Does the premise entail the hypothesis?", "sha1")
W2 = Wording(2, "Must the hypothesis be true?", "sha2")


def item(i=1, q=0.8):
    return Item(id=f"chaosnli-mnli:{i}", dataset="chaosnli-mnli", premise="A man runs.", hypothesis="A man moves.", q=q)


def arm_with(handler, mode, **kw):
    return OpenRouterArm("x", "deepseek/m", "SomeProv", mode, "key",
                         client=httpx.Client(transport=httpx.MockTransport(handler)), context_template=TPL, **kw)


def chat(content=None, top=None, model="deepseek/m-2026", cost=0.001):
    msg = {"choices": [{"message": {"content": content}}], "model": model,
           "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": cost}}
    if top is not None:
        msg["choices"][0]["logprobs"] = {"content": [{"token": top[0][0], "top_logprobs": [
            {"token": t, "logprob": lp} for t, lp in top]}]}
    return msg


# ---- FakeArm
def test_fake_deterministic_and_in_range():
    a = FakeArm("f", noise=0.2, seed=3)
    ps = [a.predict(item(i, q), W).p_yes for i in range(20) for q in (0.0, 0.5, 1.0)]
    assert all(0 < p < 1 for p in ps)
    assert a.predict(item(), W).p_yes == FakeArm("f", noise=0.2, seed=3).predict(item(), W).p_yes
    assert a.predict(item(), W).p_yes != a.predict(item(), W2).p_yes
    assert a.predict(item(), W).p_yes != FakeArm("f", noise=0.2, seed=4).predict(item(), W).p_yes


def test_fake_uses_label_and_bias():
    it = Item("x:1", "mnli-train", "p", "h", q=None, label=1)
    assert FakeArm("f", noise=0).predict(it, None).p_yes > 0.99
    assert FakeArm("f", noise=0, bias=1).predict(item(q=0.5), W).p_yes == pytest.approx(1 / (1 + math.exp(-1)))
    assert FakeArm("f", noise=0).predict(item(q=0.5), W).usage["cost_usd"] == 0


# ---- OpenRouter modes
def test_native_mode():
    seen = {}

    def h(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"model": "typesafe/jev-1.13-2026", "answers": {"q": {"type": "noul", "noul": 0.42}},
                                         "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.00002}})
    p = arm_with(h, "native").predict(item(), W)
    assert p.status == "ok" and p.p_yes == 0.42 and p.usage == {"in": 100, "out": 20, "cost_usd": 0.00002}
    assert p.raw["model"].startswith("typesafe/jev")
    assert seen["url"].endswith("/api/alpha/decisions")
    assert seen["body"]["state"] == "Premise: A man runs.\nHypothesis: A man moves."
    assert seen["body"]["questions"]["q"] == {"type": "noul", "instructions": W.text}


def test_native_malformed():
    for payload in ({"answers": {}}, {"answers": {"q": {"noul": 1.7}}}, {"answers": {"q": {"noul": "x"}}}):
        assert arm_with(lambda r, p=payload: httpx.Response(200, json=p), "native").predict(item(), W).status == "error"


def test_logprobs_mode_and_request_body():
    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=chat("Yes", top=[("Yes", math.log(0.6)), (" yes", math.log(0.1)),
                                                        ("No", math.log(0.2)), ("Maybe", math.log(0.05))]))
    p = arm_with(h, "logprobs").predict(item(), W)
    assert p.status == "ok" and p.p_yes == pytest.approx(0.7 / 0.9)
    assert p.usage["cost_usd"] == 0.001 and p.raw["model"] == "deepseek/m-2026"
    b = seen["body"]
    assert b["provider"] == {"order": ["SomeProv"], "allow_fallbacks": False}
    assert b["reasoning"] == {"enabled": False} and b["temperature"] == 0
    assert b["usage"] == {"include": True} and b["logprobs"] is True and b["top_logprobs"] >= 2
    assert "Premise: A man runs." in b["messages"][0]["content"] and W.text in b["messages"][0]["content"]


def test_logprobs_errors():
    for r in (chat("Maybe", top=[("Maybe", -0.1), ("Perhaps", -2.0)]), {"choices": [{"message": {"content": "Yes"}}]}):
        assert arm_with(lambda req, r=r: httpx.Response(200, json=r), "logprobs").predict(item(), W).status == "error"


def test_verbalized_mode():
    for content, want in (("72", 0.72), (" 5.5% ", 0.055), ("0", 0.0), ("100", 1.0)):
        p = arm_with(lambda r, c=content: httpx.Response(200, json=chat(c)), "verbalized").predict(item(), W)
        assert p.status == "ok" and p.p_yes == pytest.approx(want)
    for bad in ("101", "-3", "about 70", "", None, "seventy"):
        assert arm_with(lambda r, c=bad: httpx.Response(200, json=chat(c)), "verbalized").predict(item(), W).status == "error"


def test_verbalized_request_body():
    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=chat("50"))
    arm_with(h, "verbalized").predict(item(), W)
    assert seen["body"]["provider"]["allow_fallbacks"] is False and seen["body"]["reasoning"] == {"enabled": False}
    assert "logprobs" not in seen["body"]


def test_http_error_is_error_status():
    p = arm_with(lambda r: httpx.Response(429, text="slow down"), "verbalized").predict(item(), W)
    assert p.status == "error" and p.raw["http_status"] == 429


# ---- run_arm
class Scripted:
    name, model_version, provider, elicitation, prompted = "s", "v", "p", {"k": 1}, True

    def __init__(self, results):
        self.results, self.calls = list(results), 0

    def predict(self, item, wording):
        self.calls += 1
        r = self.results.pop(0) if self.results else Prediction(0.5, usage={"cost_usd": 0.0})
        if isinstance(r, Exception):
            raise r
        return r


def test_run_arm_caches(tmp_path):
    arm = FakeArm("f")
    items = [item(1), item(2)]
    s1 = run_arm(arm, items, [W, W2], tmp_path, "rev", repeats=2)
    assert s1 == {"calls": 8, "cached": 0, "failed": 0, "cost_usd": 0}
    s2 = run_arm(arm, items, [W, W2], tmp_path, "rev", repeats=2)
    assert s2["calls"] == 0 and s2["cached"] == 8
    rows = load_arm(tmp_path, "f")
    assert len(rows) == 8 and {"key", "config_hash", "arm", "item_id", "wording_sha", "usage", "status", "ts"} <= rows[0].keys()


def test_run_arm_unprompted(tmp_path):
    arm = FakeArm("m", prompted=False)
    assert run_arm(arm, [item(1), item(2)], None, tmp_path, "rev")["calls"] == 2
    assert load_arm(tmp_path, "m")[0]["wording_sha"] is None


def test_run_arm_retries_then_succeeds(tmp_path):
    arm = Scripted([Prediction(None, status="error"), RuntimeError("net"), Prediction(0.3, usage={"cost_usd": 0.01})])
    sleeps = []
    s = run_arm(arm, [item()], [W], tmp_path, "rev", sleep=sleeps.append)
    assert arm.calls == 3 and s["failed"] == 0 and len(sleeps) == 2 and sleeps[1] > sleeps[0]
    assert load_arm(tmp_path, "s")[0]["p_yes"] == 0.3


def test_run_arm_records_error_then_later_ok_supersedes(tmp_path):
    arm = Scripted([Prediction(None, status="error")] * 4)
    s = run_arm(arm, [item()], [W], tmp_path, "rev", max_retries=3, sleep=lambda t: None)
    assert arm.calls == 4 and s["failed"] == 1
    assert load_arm(tmp_path, "s")[0]["status"] == "error"
    s = run_arm(arm, [item()], [W], tmp_path, "rev", sleep=lambda t: None)  # error rows are retried
    assert s["calls"] == 1 and s["failed"] == 0
    rows = load_arm(tmp_path, "s")
    assert len(rows) == 1 and rows[0]["status"] == "ok"


def test_run_arm_budget(tmp_path):
    arm = Scripted([Prediction(0.5, usage={"cost_usd": 0.6})] * 5)
    with pytest.raises(BudgetExceeded):
        run_arm(arm, [item(i) for i in range(5)], [W], tmp_path, "rev", budget_usd=1.0)
    assert arm.calls == 2 and len(load_arm(tmp_path, "s")) == 2
    arm2 = Scripted([])  # existing spend of 1.2 counts against the budget on the next run
    with pytest.raises(BudgetExceeded):
        run_arm(arm2, [item(9)], [W], tmp_path, "rev", budget_usd=1.0)
    assert arm2.calls == 0
