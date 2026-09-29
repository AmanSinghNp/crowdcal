from crowdcal.cli import pilot_summary


def _row(item, w, p, cost=0.001, status="ok", reasoning=0):
    return {"item_id": item, "wording_id": w, "p_yes": p, "status": status, "response_model": "m-1",
            "usage": {"cost_usd": cost}, "raw": {"usage": {"completion_tokens_details": {"reasoning_tokens": reasoning}}}}


def test_pilot_summary_repeats_and_cost():
    rows = [_row("a", 1, 0.7), _row("a", 1, 0.7), _row("b", 1, 0.2), _row("b", 1, 0.2)]
    s = pilot_summary(rows, seconds=2.0, n_full_items=10, n_wordings=3)
    assert s["repeat_identical_frac"] == 1.0 and s["planned_repeats"] == 1
    assert s["fail_rate"] == 0 and s["served_models"] == ["m-1"]
    assert s["projected_full_run_usd"] == round(0.001 * 10 * 3 * 1, 2) and s["sec_per_call"] == 0.5


def test_pilot_summary_flags_nondeterminism_failures_and_reasoning():
    rows = [_row("a", 1, 0.7), _row("a", 1, 0.6, reasoning=12), _row("b", 1, None, status="error")]
    s = pilot_summary(rows, seconds=1.0, n_full_items=10, n_wordings=3)
    assert s["repeat_identical_frac"] == 0.0 and s["planned_repeats"] == 3
    assert s["fail_rate"] == round(1 / 3, 4) and s["reasoning_tokens"] == 12
