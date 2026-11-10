"""Per-turn token accounting, aggregated from the call records already on disk.

Every LLM call in a campaign writes its provider usage payload next to its
transcript: the planner into `orchestrator/llm_calls.jsonl`, the reflector into
`reflection_call.jsonl`, and the generator into each generation's
`llm_raw.jsonl`. This module only sums them, so the numbers can be rebuilt at
any time from artifacts and never depend on a counter surviving a crash.

Tokens are always recorded; cost only when the provider states it. OpenRouter
returns the charge per call, which is better than a price table because it is
what was actually billed at the price in force. For providers that report no
cost the column stays 0 and the token columns are what you price.

Reasoning tokens are counted separately AND are already part of `output` for
every provider we speak to (DeepSeek/OpenAI report them inside
completion_tokens; Anthropic's thinking is billed as output). Do not add the
two columns together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROLE_SOURCES = ("orchestrator", "reflector", "generator")
_FIELDS = ("input", "output", "reasoning", "cache_hit")
# How many calls actually reported a reasoning count. A subscription-served
# Claude call returns no `completion_tokens_details` at all, so its reasoning
# token count is UNKNOWN — recording that as 0 would put "no reasoning" and
# "not reported" in the same number, and a mixed campaign (part subscription,
# part API) would then read as a reasoning collapse that never happened.
_REASONING_CALLS = "reasoning_reported_calls"
# Money, when the provider states it. OpenRouter returns the charge for each
# call; direct providers do not, and there it stays 0 and the token columns
# are what you price. Kept apart from _FIELDS because it is a float.
_COST_FIELD = "cost_usd"


def _blank() -> dict[str, Any]:
    return {"calls": 0, **{field: 0 for field in _FIELDS},
            _COST_FIELD: 0.0, _REASONING_CALLS: 0}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def usage_fields(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalized counts from one call's usage payload (see llm/usage.py)."""

    if not isinstance(usage, dict):
        return {**{field: 0 for field in _FIELDS}, _COST_FIELD: 0.0}
    raw = usage.get("raw") if isinstance(usage.get("raw"), dict) else {}
    reasoning = usage.get("reasoning_tokens")
    if reasoning is None:
        details = raw.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")
    # Left as None when the provider said nothing about it (see _REASONING_CALLS).
    cache_hit = raw.get("prompt_cache_hit_tokens")
    if cache_hit is None:
        details = raw.get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_hit = details.get("cached_tokens")
    if cache_hit is None:
        cache_hit = raw.get("cache_read_input_tokens")
    return {
        "input": int(usage.get("input_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
        "reasoning": None if reasoning is None else int(reasoning),
        "cache_hit": int(cache_hit or 0),
        _COST_FIELD: float(usage.get("cost_usd") or raw.get("cost") or 0.0),
    }


def _accumulate(bucket: dict[str, Any], usage: dict[str, Any] | None) -> None:
    bucket["calls"] += 1
    for field, value in usage_fields(usage).items():
        if field == "reasoning":
            if value is None:
                continue          # unknown: neither counted nor silently zeroed
            bucket[_REASONING_CALLS] += 1
        bucket[field] += value


def turn_usage(turn_dir: Path) -> dict[str, Any]:
    """Aggregate one turn's token spend by role."""

    turn_dir = Path(turn_dir)
    roles = {role: _blank() for role in ROLE_SOURCES}

    for row in _jsonl(turn_dir / "orchestrator" / "llm_calls.jsonl"):
        _accumulate(roles["orchestrator"], (row.get("model") or {}).get("usage"))
    # Top level, NOT inside orchestrator/: the strict audit walks that
    # directory for planner provenance, and an advisory call has no business
    # appearing there. (These two paths disagreed at first, so the reflection's
    # own cost was silently missing from every aggregate — the exact failure
    # this module exists to make impossible.)
    for row in _jsonl(turn_dir / "reflection_call.jsonl"):
        _accumulate(roles["reflector"], row.get("usage"))
    for path in sorted(turn_dir.glob("cycle/fast/generations/*/llm_raw.jsonl")):
        for row in _jsonl(path):
            _accumulate(roles["generator"], (row.get("engine_info") or {}).get("usage"))

    totals = _blank()
    for bucket in roles.values():
        for key in totals:
            totals[key] += bucket[key]
    # A partial count is worse than no count unless it is labelled as partial.
    totals["reasoning_complete"] = totals[_REASONING_CALLS] == totals["calls"]
    # input + output is the billable total; reasoning is a subset of output.
    totals["total_tokens"] = totals["input"] + totals["output"]
    totals[_COST_FIELD] = round(float(totals[_COST_FIELD]), 6)
    return {"schema": "turn_token_usage", "by_role": roles, "total": totals}


def write_turn_usage(turn_dir: Path) -> dict[str, Any]:
    payload = turn_usage(turn_dir)
    path = Path(turn_dir) / "token_usage.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def campaign_tokens(run_dir: Path) -> int:
    """Billable tokens across every turn that has an aggregate on disk."""

    total = 0
    for path in sorted(Path(run_dir).glob("turns/turn_*/token_usage.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += int((payload.get("total") or {}).get("total_tokens") or 0)
    return total
