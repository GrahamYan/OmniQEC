"""Regression tests for the reply shapes that killed the 2026-08-18 campaign.

Every fixture below is copied from a real transcript under
runs/dg_p005_b*_v4pro/turns/*/orchestrator/llm_response_*.txt.
"""
from __future__ import annotations

from omniqec_agentic.orchestrator.loop import AgenticPolicy

parse = AgenticPolicy._parse_json


def test_single_object_still_parses():
    assert parse('{"tool": "budget_status", "arguments": {}}') == {
        "tool": "budget_status",
        "arguments": {},
    }


def test_batched_objects_take_the_first():
    """b120 turn 4 call 04 and b288 turn 1 call 05: several objects in a row."""
    reply = (
        '{"tool": "query_evidence", "arguments": {"code_hash": "ed8c3b957e5f"}}\n\n'
        '{"tool": "query_evidence", "arguments": {"code_hash": "e54b0c018a79"}}\n\n'
        '{"tool": "query_evidence", "arguments": {"code_hash": "1f0004007a10"}}\n\n'
        '{"tool": "query_evidence", "arguments": {"code_hash": "bf4cc98ee851"}}'
    )
    payload = parse(reply)
    assert payload is not None
    assert payload["tool"] == "query_evidence"
    assert payload["arguments"]["code_hash"] == "ed8c3b957e5f"


def test_hallucinated_tool_transcript_takes_the_real_call():
    """b220 turn 2 call 06: the model invented its own TOOL RESULT blocks."""
    reply = (
        '{"tool": "fingerprint_lookup", "arguments": {"code_hash": "e43c77d77f10"}}\n\n'
        "TOOL RESULT for fingerprint_lookup (26 calls left):\n"
        '{"code_hash": "e43c77d77f10", "novelty_gate": "known_equivalent_or_near_duplicate"}\n\n'
        "TOOL RESULT for fingerprint_lookup (25 calls left):\n"
        '{"code_hash": "720ae517992e", "novelty_gate": "novel"}'
    )
    payload = parse(reply)
    assert payload is not None
    assert payload["tool"] == "fingerprint_lookup"
    assert payload["arguments"]["code_hash"] == "e43c77d77f10"


def test_xml_wrapper_yields_the_argument_object():
    """b120 turn 4 call 02: XML tool syntax instead of JSON.

    The arguments object is all that is recoverable, so the call must not be
    silently mistaken for a tool call; it goes down the repair path instead.
    """
    reply = (
        "<tool>query_evidence</tool>\n"
        '<arguments>{"code_hash": "ed8c3b957e5f"}</arguments>'
    )
    payload = parse(reply)
    assert payload == {"code_hash": "ed8c3b957e5f"}
    assert "tool" not in payload
    assert "submit_plan" not in payload


def test_markdown_fenced_object_parses():
    reply = '```json\n{"tool": "budget_status", "arguments": {}}\n```'
    assert parse(reply) == {"tool": "budget_status", "arguments": {}}


def test_prose_prefix_before_object_parses():
    reply = 'Let me check the budget first.\n\n{"tool": "budget_status", "arguments": {}}'
    assert parse(reply) == {"tool": "budget_status", "arguments": {}}


def test_truncated_object_is_rejected():
    assert parse('{"tool": "query_evidence", "arguments": {"code_hash": "ed8c') is None


def test_non_object_reply_is_rejected():
    assert parse("I could not complete this turn.") is None
    assert parse('["not", "an", "object"]') is None


def test_plan_fields_recognised_for_envelope_rescue():
    """b98 / b260: the plan body arrived without its submit_plan wrapper."""
    from omniqec_agentic.orchestrator.loop import _PLAN_FIELDS

    body = parse(
        '{"observations": [{"text": "empty", "cites": ["S-3"]}], '
        '"rationale": {"text": "seed", "cites": ["S-3"]}, "program_count": 24}'
    )
    assert body is not None
    assert _PLAN_FIELDS & set(body)


def test_tool_call_is_not_mistaken_for_a_plan():
    from omniqec_agentic.orchestrator.loop import _PLAN_FIELDS

    payload = parse('{"tool": "budget_status", "arguments": {}}')
    assert not (_PLAN_FIELDS & set(payload))


def test_repair_budget_is_configurable_and_positive():
    policy = AgenticPolicy(provider="deepseek")
    assert policy.format_repair_budget >= 1
    assert AgenticPolicy(provider="deepseek", format_repair_budget=7).format_repair_budget == 7
    assert AgenticPolicy(provider="deepseek", format_repair_budget=0).format_repair_budget == 1
