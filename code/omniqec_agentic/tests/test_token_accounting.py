"""Per-turn token accounting: rebuilt from artifacts, never incremented.

Cost is computed from these columns after the fact, so the two properties that
matter are that every role's calls are counted and that re-running the
aggregation on a resumed turn cannot double-count.
"""

from __future__ import annotations

import json

from omniqec_agentic.token_usage import (
    campaign_tokens,
    turn_usage,
    usage_fields,
    write_turn_usage,
)


def _openai_usage(prompt: int, completion: int, reasoning: int = 0, cached: int = 0) -> dict:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "reasoning_tokens": reasoning,
        "raw": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
            "prompt_cache_hit_tokens": cached,
        },
    }


def _write_turn(run_dir, turn: int, *, orch=(), reflector=(), generator=()):
    turn_dir = run_dir / "turns" / f"turn_{turn:04d}"
    (turn_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
    gen_dir = turn_dir / "cycle" / "fast" / "generations" / "g0001"
    gen_dir.mkdir(parents=True, exist_ok=True)

    with (turn_dir / "orchestrator" / "llm_calls.jsonl").open("w", encoding="utf-8") as h:
        for usage in orch:
            h.write(json.dumps({"role": "orchestrator", "model": {"provider": "d", "usage": usage}}) + "\n")
    if reflector:
        # Top level, mirroring reflector.run_reflection: orchestrator/ is the
        # audited directory and holds planner calls only.
        with (turn_dir / "reflection_call.jsonl").open("w", encoding="utf-8") as h:
            for usage in reflector:
                h.write(json.dumps({"role": "reflector", "usage": usage}) + "\n")
    with (gen_dir / "llm_raw.jsonl").open("w", encoding="utf-8") as h:
        for usage in generator:
            h.write(json.dumps({"engine_info": {"provider": "d", "usage": usage}}) + "\n")
    return turn_dir


def test_every_role_is_counted_and_reasoning_stays_a_subset_of_output(tmp_path) -> None:
    turn_dir = _write_turn(
        tmp_path, 0,
        orch=[_openai_usage(1000, 200), _openai_usage(1200, 300)],
        reflector=[_openai_usage(800, 150)],
        generator=[_openai_usage(5042, 19336, reasoning=16862, cached=1664)],
    )
    payload = turn_usage(turn_dir)
    roles = payload["by_role"]
    assert roles["orchestrator"]["calls"] == 2
    assert roles["reflector"]["calls"] == 1
    assert roles["generator"]["calls"] == 1
    assert roles["generator"]["reasoning"] == 16862
    assert roles["generator"]["cache_hit"] == 1664

    total = payload["total"]
    assert total["input"] == 1000 + 1200 + 800 + 5042
    assert total["output"] == 200 + 300 + 150 + 19336
    # Billable total is input+output; reasoning is already inside output.
    assert total["total_tokens"] == total["input"] + total["output"]
    assert total["reasoning"] < total["output"]


def test_anthropic_shaped_usage_normalizes(tmp_path) -> None:
    fields = usage_fields({
        "input_tokens": 900, "output_tokens": 4000, "total_tokens": 4900,
        "raw": {"input_tokens": 900, "output_tokens": 4000, "cache_read_input_tokens": 512},
    })
    # Anthropic states no reasoning breakdown, so it is unknown rather than 0.
    assert fields == {"input": 900, "output": 4000, "reasoning": None,
                      "cache_hit": 512, "cost_usd": 0.0}


def test_missing_or_broken_usage_counts_the_call_but_adds_no_tokens(tmp_path) -> None:
    turn_dir = _write_turn(tmp_path, 0, orch=[{}, None], generator=[{}])
    payload = turn_usage(turn_dir)
    assert payload["by_role"]["orchestrator"]["calls"] == 2
    assert payload["total"]["total_tokens"] == 0


def test_campaign_total_is_rebuilt_not_incremented(tmp_path) -> None:
    for turn in range(3):
        _write_turn(tmp_path, turn, orch=[_openai_usage(100, 50)])
        write_turn_usage(tmp_path / "turns" / f"turn_{turn:04d}")
    assert campaign_tokens(tmp_path) == 3 * 150

    # A resumed turn re-runs the aggregation; the total must not grow.
    write_turn_usage(tmp_path / "turns" / "turn_0001")
    assert campaign_tokens(tmp_path) == 3 * 150


def test_orchestrator_records_the_usage_of_the_call_it_describes(tmp_path) -> None:
    """Regression: usage was snapshotted once, before any call was made.

    `client.info()` reports the LAST completed call's usage, so taking it once
    outside the loop wrote a usage-free record for every planner call — the
    b144 transcripts have `model` blocks with no `usage` key at all, and the
    planner's share of campaign cost was invisible.
    """

    from omniqec_agentic.orchestrator.loop import AgenticPolicy

    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, system: str, user: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"tool": "budget_status", "arguments": {}})
            return json.dumps({"submit_plan": {
                "observations": [],
                "rationale": {"text": "cold start", "cites": ["S-6"]},
                "family_mix": {"bb": 1.0},
                "program_count": 2,
                "promotions": [],
                "directives": "",
                "finalize": False,
            }})

        def info(self) -> dict:
            # Usage only exists after a call, and changes with each one.
            if not self.calls:
                return {"provider": "fake", "model": "scripted"}
            return {"provider": "fake", "model": "scripted",
                    "usage": _openai_usage(100 * self.calls, 10 * self.calls)}

    policy = AgenticPolicy(provider="fake")
    client = _CountingClient()
    policy._client = lambda: client

    state = {
        "run_id": "t", "turn": 0, "run_dir": str(tmp_path),
        "spec": {"qubit_budget": 288, "families": ["bb"]},
        "protocol": {"max_turns": 2, "tool_budget": 8, "proposal_count": 2,
                     "promote_top": 2, "generations_per_turn": 1, "feedback_mode": "both"},
        "budget": {"turns_used": 0}, "hof_summary": [], "baseline_summary": {},
    }
    policy.plan(state)

    rows = [
        json.loads(line)
        for line in (tmp_path / "turns" / "turn_0000" / "orchestrator" / "llm_calls.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 2
    assert all("usage" in (row.get("model") or {}) for row in rows)
    # Each record carries ITS OWN call's usage, not a shared snapshot.
    assert [row["model"]["usage"]["input_tokens"] for row in rows] == [100, 200]

    payload = turn_usage(tmp_path / "turns" / "turn_0000")
    assert payload["by_role"]["orchestrator"]["input"] == 300


def test_unreported_reasoning_is_unknown_not_zero(tmp_path) -> None:
    """A subscription-served Claude call reports no reasoning breakdown.

    Recording that as 0 puts "no reasoning happened" and "nobody said" in the
    same number. A campaign that runs part on a subscription and part on the
    API would then show a reasoning collapse at the switchover that never
    occurred.
    """

    subscription = {"input_tokens": 145, "output_tokens": 4,
                    "raw": {"prompt_tokens": 145, "completion_tokens": 4,
                            "completion_tokens_details": None}}
    api = _openai_usage(100, 50, reasoning=30)

    assert usage_fields(subscription)["reasoning"] is None
    assert usage_fields(api)["reasoning"] == 30

    turn_dir = _write_turn(tmp_path, 0, orch=[subscription, api])
    payload = turn_usage(turn_dir)
    total = payload["total"]
    assert total["calls"] == 2
    assert total["reasoning"] == 30              # only the call that reported
    assert total["reasoning_reported_calls"] == 1
    assert total["reasoning_complete"] is False  # labelled as partial
    # Tokens and cost stay whole either way — only the breakdown is partial.
    assert total["input"] == 245 and total["output"] == 54


def test_a_fully_reported_turn_is_marked_complete(tmp_path) -> None:
    turn_dir = _write_turn(tmp_path, 0, orch=[_openai_usage(10, 5, reasoning=2)] * 2)
    total = turn_usage(turn_dir)["total"]
    assert total["reasoning_reported_calls"] == total["calls"] == 2
    assert total["reasoning_complete"] is True
