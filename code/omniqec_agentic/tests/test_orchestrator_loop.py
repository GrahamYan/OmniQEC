"""Agentic tool-loop regressions (03a §7): scripted fake client, no API.

Covers: tool call -> result -> valid plan (happy path); invalid plan ->
structured repair -> valid plan; exhausted repair budget -> FallbackPolicy
closes the turn; citation validator rejects unknown ids."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic.orchestrator.loop import AgenticPolicy
from omniqec_agentic.orchestrator.plan import validate_turn_plan


class _FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.responses.pop(0)

    def info(self) -> dict:
        return {"provider": "fake", "model": "scripted"}


def _policy_with(
    responses: list[str], **policy_kwargs
) -> tuple[AgenticPolicy, _FakeClient]:
    policy = AgenticPolicy(provider="fake", **policy_kwargs)
    client = _FakeClient(responses)
    policy._client = lambda: client  # scripted transport
    return policy, client


def _state(tmp_path: Path) -> dict:
    return {
        "run_id": "t", "turn": 1, "run_dir": str(tmp_path),
        "spec": {"qubit_budget": 144,
                 "families": ["bb", "hgp", "lp_abelian", "lp_nonabelian"]},
        "protocol": {"max_turns": 3, "tool_budget": 8, "proposal_count": 4,
                     "promote_top": 4, "generations_per_turn": 1,
                     "feedback_mode": "both"},
        "budget": {"turns_used": 0},
        "hof_summary": [], "baseline_summary": {},
    }


VALID_PLAN = {
    "submit_plan": {
        "observations": [{"text": "no measurements yet", "cites": ["S-4"]}],
        "rationale": {"text": "cold start: spread attention evenly", "cites": ["S-6"]},
        "family_mix": {"bb": 0.25, "hgp": 0.25, "lp_abelian": 0.25, "lp_nonabelian": 0.25},
        "strategy_mix": {"novelty": 1.0},
        "program_count": 4,
        "promotions": [],
        "directives": "explore small nonabelian lifts",
        "finalize": False,
    }
}


def test_happy_path_with_tool_call(tmp_path) -> None:
    policy, client = _policy_with([
        json.dumps({"tool": "budget_status", "arguments": {}}),
        json.dumps(VALID_PLAN),
    ])
    plan = policy.plan(_state(tmp_path))
    assert plan["policy"] == "agentic_v1"
    assert plan["proposal_count"] == 4
    assert plan["tool_calls_used"] == 1
    assert "TOOL RESULT for budget_status" in client.calls[1]["user"]
    turn_dir = tmp_path / "turns" / "turn_0001" / "orchestrator"
    assert (turn_dir / "turn_plan.json").exists()
    assert (turn_dir / "context_blocks.json").exists()
    assert (turn_dir / "llm_calls.jsonl").exists()
    assert (turn_dir / "llm_response_00.txt").exists()  # raw before parse
    assert (tmp_path / "manual" / "static" / "tools.md").exists()


def test_invalid_plan_repaired_once(tmp_path) -> None:
    bad = {"submit_plan": {**VALID_PLAN["submit_plan"],
                           "rationale": {"text": "because I said so", "cites": []}}}
    policy, client = _policy_with([json.dumps(bad), json.dumps(VALID_PLAN)])
    plan = policy.plan(_state(tmp_path))
    assert plan["policy"] == "agentic_v1"
    assert "PLAN REJECTED" in client.calls[1]["user"]
    assert "empty cites" in client.calls[1]["user"]


def test_falls_back_once_repair_budget_is_spent(tmp_path) -> None:
    bad = {"submit_plan": {**VALID_PLAN["submit_plan"],
                           "rationale": {"text": "x", "cites": ["S-99"]}}}
    budget = 2
    policy, _ = _policy_with(
        [json.dumps(bad)] * (budget + 1), format_repair_budget=budget
    )
    plan = policy.plan(_state(tmp_path))
    assert plan["policy"] == "agentic_v1_fallback"
    assert "plan invalid beyond repair budget" in plan["fallback_reason"]


def test_repair_budget_survives_a_slip_the_model_recovers_from(tmp_path) -> None:
    """A single malformed reply must not doom the rest of the turn.

    This is the 2026-08-18 failure: an early formatting slip consumed the only
    repair, so an unrelated slip several calls later killed the campaign.
    """
    bad = {"submit_plan": {**VALID_PLAN["submit_plan"],
                           "rationale": {"text": "x", "cites": ["S-99"]}}}
    policy, client = _policy_with(
        ["not json at all", json.dumps(bad), json.dumps(VALID_PLAN)],
        format_repair_budget=4,
    )
    plan = policy.plan(_state(tmp_path))
    assert plan["policy"] == "agentic_v1"
    assert len(client.calls) == 3


def test_strict_policy_raises_instead_of_counting_fallback(tmp_path) -> None:
    bad = {"submit_plan": {**VALID_PLAN["submit_plan"],
                           "rationale": {"text": "x", "cites": ["S-99"]}}}
    policy, _ = _policy_with(
        [json.dumps(bad)] * 3, allow_fallback=False, format_repair_budget=2
    )

    with pytest.raises(RuntimeError, match="forbids fallback turns"):
        policy.plan(_state(tmp_path))


def test_strict_policy_repairs_nonfixed_program_count(tmp_path) -> None:
    short = {
        "submit_plan": {**VALID_PLAN["submit_plan"], "program_count": 3}
    }
    policy, client = _policy_with(
        [json.dumps(short), json.dumps(VALID_PLAN)],
        fixed_program_count=True,
        allow_fallback=False,
    )

    plan = policy.plan(_state(tmp_path))

    assert plan["proposal_count"] == 4
    assert "requires exactly 4" in client.calls[1]["user"]


def test_unparseable_then_valid(tmp_path) -> None:
    policy, client = _policy_with(["I think we should explore.", json.dumps(VALID_PLAN)])
    plan = policy.plan(_state(tmp_path))
    assert plan["policy"] == "agentic_v1"
    assert "VALIDATION ERROR" in client.calls[1]["user"]


def test_citation_validator_rules() -> None:
    base = dict(VALID_PLAN["submit_plan"])
    kwargs = dict(
        source_ids={"S-1", "S-4", "S-6"},
        known_hashes={"a41f00b2deadbeef"},
        ledger_ids={"H-1", "I-2"},
        families=("bb", "hgp", "lp_abelian", "lp_nonabelian"),
        program_cap=32, promote_cap=4, quarantined_hashes={"ffff0000ffff0000"},
    )
    plan, errors = validate_turn_plan(base, **kwargs)
    assert plan is not None and not errors

    hash_cite = {**base, "rationale": {"text": "candidate looks strong", "cites": ["a41f00b2"]}}
    plan, errors = validate_turn_plan(hash_cite, **kwargs)
    assert plan is not None

    unknown = {**base, "rationale": {"text": "x", "cites": ["arXiv:2308.07915"]}}
    plan, errors = validate_turn_plan(unknown, **kwargs)
    assert plan is None and any("unknown citation" in e for e in errors)

    # The one hard exclusion is schedule_catastrophic (unmeasurable), passed in
    # as quarantined_hashes; degraded_* candidates are no longer excluded.
    unmeasurable = {**base, "promotions": [{"code_hash": "ffff0000", "budget_class": "pilot"}]}
    kwargs2 = {**kwargs, "known_hashes": kwargs["known_hashes"] | {"ffff0000ffff0000"}}
    plan, errors = validate_turn_plan(unmeasurable, **kwargs2)
    assert plan is None and any("schedule_catastrophic" in e for e in errors)

    # S-10 is advisory prose, never sufficient on its own for a load-bearing
    # claim (v3.2 §5.2).
    advisory_only = {**base, "rationale": {"text": "last turn said so", "cites": ["S-10"]}}
    plan, errors = validate_turn_plan(
        advisory_only, **{**kwargs, "source_ids": kwargs["source_ids"] | {"S-10"}}
    )
    assert plan is None and any("advisory prose" in e for e in errors)

    bad_mix = {**base, "family_mix": {"bb": 0.9, "hgp": 0.9}}
    plan, errors = validate_turn_plan(bad_mix, **kwargs)
    assert plan is None and any("sum to 1" in e for e in errors)


def test_reference_codes_reach_the_orchestrator_as_block_s11(tmp_path) -> None:
    """Experiment variant (2026-09-09): protocol.reference_codes is rendered as
    S-11 in the orchestrator context and registered like every other block."""
    from omniqec_agentic.orchestrator import reference_codes as rc

    payload = rc.load_reference_codes(Path(rc.__file__).resolve().parents[1] / "examples" / "reference_codes_deepseek_b98_v36.json")
    state = _state(tmp_path)
    state["protocol"]["reference_codes"] = payload
    policy, client = _policy_with([json.dumps(VALID_PLAN)])
    plan = policy.plan(state)
    assert plan["policy"] == "agentic_v1"
    first_user = client.calls[0]["user"]
    assert 'id="S-11" kind="reference_codes"' in first_user or "S-11" in first_user
    assert "BACKGROUND: REFERENCE CODES." in first_user and "ref-1" in first_user
    registry = json.loads((tmp_path / "turns" / "turn_0001" / "orchestrator" / "context_blocks.json").read_text())
    rows = registry if isinstance(registry, list) else registry.get("blocks") or registry.get("rows") or []
    assert any("S-11" in json.dumps(row) for row in rows)
    # Without the key the block is absent.
    policy2, client2 = _policy_with([json.dumps(VALID_PLAN)])
    policy2.plan(_state(tmp_path / "plain"))
    assert "REFERENCE CODES" not in client2.calls[0]["user"]
