"""Two cost reductions that must not cost capability.

A. Anthropic prompt caching — the model sees byte-identical text; only the
   price of resending it changes.
B. Batched tool calls — one round trip may carry several INDEPENDENT reads.
   The single-call envelope is untouched, the budget still counts every tool,
   and nothing forces batching, so a dependent lookup chain still works.

The assertions below are mostly about what did NOT change.
"""

from __future__ import annotations

import json

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.openrouter import OpenRouterClient, _wants_cache_control
from omniqec_agentic.orchestrator.loop import AgenticPolicy, _tool_requests


# ---- A: caching -----------------------------------------------------------

def test_cache_control_is_scoped_to_the_provider_that_needs_it() -> None:
    # Anthropic caches only where a breakpoint is marked; OpenAI and DeepSeek
    # cache automatically and take no such marker.
    assert _wants_cache_control("anthropic/claude-opus-5") is True
    assert _wants_cache_control("openai/gpt-5.6-sol") is False
    assert _wants_cache_control("deepseek/deepseek-v4-pro") is False


def test_cached_request_carries_the_same_text_the_uncached_one_does(monkeypatch) -> None:
    """The point of caching is that the MODEL sees no difference."""

    sent: list[dict] = []

    def _capture(**kwargs):
        sent.append(kwargs)
        return _fake_response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-5")
    client = OpenRouterClient(LLMConfig())
    monkeypatch.setattr(client._client.chat.completions, "create", _capture)
    client.complete(system="SYSTEM TEXT", user="USER TEXT")

    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol")
    plain = OpenRouterClient(LLMConfig())
    monkeypatch.setattr(plain._client.chat.completions, "create", _capture)
    plain.complete(system="SYSTEM TEXT", user="USER TEXT")

    cached, uncached = sent
    assert cached["messages"][0]["content"][0]["text"] == "SYSTEM TEXT"
    assert cached["messages"][1]["content"][0]["text"] == "USER TEXT"
    assert cached["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # ...and the plain path is exactly what it always was.
    assert uncached["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    # Same model-visible payload apart from the framing.
    assert cached["max_tokens"] == uncached["max_tokens"]


def test_caching_can_be_switched_off_without_a_code_change(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_CACHE", "off")
    assert _wants_cache_control("anthropic/claude-opus-5") is False


def _fake_response():
    class _U:
        def model_dump(self):
            return {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}

    class _M:
        content = "OK"

    class _C:
        message = _M()

    class _R:
        usage = _U()
        model = "anthropic/claude-opus-5"
        provider = "Anthropic"
        choices = [_C()]

    return _R()


# ---- B: batched tool calls ------------------------------------------------

def test_single_tool_envelope_is_unchanged() -> None:
    assert _tool_requests({"tool": "budget_status", "arguments": {}}) == [
        ("budget_status", {})
    ]
    assert _tool_requests({"tool": "query_archive", "arguments": {"limit": 8}}) == [
        ("query_archive", {"limit": 8})
    ]


def test_batch_envelope_preserves_order_and_drops_only_malformed_entries() -> None:
    requests = _tool_requests({"tools": [
        {"tool": "query_archive", "arguments": {"limit": 8}},
        {"tool": "bandit_status"},
        {"arguments": {"stray": 1}},          # no name — dropped, not fatal
        {"tool": "query_coverage", "arguments": {"limit": 24}},
    ]})
    assert requests == [
        ("query_archive", {"limit": 8}),
        ("bandit_status", {}),
        ("query_coverage", {"limit": 24}),
    ]


def _policy(responses, tmp_path, **kwargs):
    class _Client:
        def __init__(self):
            self.sent: list[str] = []

        def complete(self, *, system, user):
            self.sent.append(user)
            return responses.pop(0)

        def info(self):
            return {"provider": "fake", "model": "scripted"}

    policy = AgenticPolicy(provider="fake", **kwargs)
    client = _Client()
    policy._client = lambda: client
    state = {
        "run_id": "t", "turn": 0, "run_dir": str(tmp_path),
        "spec": {"qubit_budget": 288, "families": ["bb", "hgp"]},
        "protocol": {"max_turns": 2, "tool_budget": 8, "proposal_count": 4,
                     "promote_top": 4, "generations_per_turn": 1, "feedback_mode": "both"},
        "budget": {"turns_used": 0}, "hof_summary": [], "baseline_summary": {},
    }
    return policy, client, state


PLAN = {"submit_plan": {
    "observations": [], "rationale": {"text": "cold start", "cites": ["S-6"]},
    "family_mix": {"bb": 0.5, "hgp": 0.5}, "program_count": 4,
    "promotions": [], "directives": "", "finalize": False,
}}


def test_a_batch_runs_every_tool_and_returns_every_result(tmp_path) -> None:
    policy, client, state = _policy([
        json.dumps({"tools": [
            {"tool": "budget_status", "arguments": {}},
            {"tool": "bandit_status", "arguments": {}},
        ]}),
        json.dumps(PLAN),
    ], tmp_path)

    plan = policy.plan(state)
    assert plan["policy"] == "agentic_v1"        # no fallback

    transcript = json.loads(
        (tmp_path / "turns" / "turn_0000" / "orchestrator" / "tool_transcript.json")
        .read_text(encoding="utf-8")
    )
    assert [row["tool"] for row in transcript] == ["budget_status", "bandit_status"]

    # Both results arrive in ONE follow-up message — that is the saving.
    followup = client.sent[1]
    assert "TOOL RESULT for budget_status" in followup
    assert "TOOL RESULT for bandit_status" in followup
    assert len(client.sent) == 2                  # 2 round trips, not 3


def test_batching_does_not_buy_extra_tool_budget(tmp_path) -> None:
    """A batch is a cheaper round trip, never a bigger allowance."""

    policy, client, state = _policy([
        json.dumps({"tools": [{"tool": "budget_status", "arguments": {}}] * 6}),
        json.dumps(PLAN),
    ], tmp_path)
    state["protocol"]["tool_budget"] = 3

    policy.plan(state)
    transcript = json.loads(
        (tmp_path / "turns" / "turn_0000" / "orchestrator" / "tool_transcript.json")
        .read_text(encoding="utf-8")
    )
    executed = [r for r in transcript if not r.get("error")]
    assert len(executed) <= 3
    assert "tool budget exhausted" in client.sent[1]


def test_dependent_lookups_can_still_be_issued_one_at_a_time(tmp_path) -> None:
    """Batching is optional; the sequential path must remain fully available."""

    policy, client, state = _policy([
        json.dumps({"tool": "budget_status", "arguments": {}}),
        json.dumps({"tool": "bandit_status", "arguments": {}}),
        json.dumps(PLAN),
    ], tmp_path)

    plan = policy.plan(state)
    assert plan["policy"] == "agentic_v1"
    assert len(client.sent) == 3
    transcript = json.loads(
        (tmp_path / "turns" / "turn_0000" / "orchestrator" / "tool_transcript.json")
        .read_text(encoding="utf-8")
    )
    assert [row["tool"] for row in transcript] == ["budget_status", "bandit_status"]


def test_an_empty_batch_costs_a_repair_not_the_turn(tmp_path) -> None:
    policy, client, state = _policy([
        json.dumps({"tools": []}),
        json.dumps(PLAN),
    ], tmp_path)

    plan = policy.plan(state)
    assert plan["policy"] == "agentic_v1"     # recovered, no fallback
    assert "VALIDATION ERROR" in client.sent[1]


def test_growing_conversation_is_sent_block_wise_not_as_one_lump(monkeypatch) -> None:
    """Regression: caching at a single block boundary cached almost nothing.

    Anthropic matches a cached prefix at CONTENT BLOCK boundaries. Sending the
    whole conversation as one block means block N+1 differs from block N, so
    only the separate system block ever matched — measured on opus-5, 6,980
    tokens read against 97,797 sent. One block per segment makes earlier calls
    a literal block-wise prefix of later ones (verified live: second call read
    14,323 of 14,467 from cache, -90% on cost).
    """

    from omniqec_agentic.llm.openrouter import _incremental_parts

    sent: list[dict] = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-5")
    client = OpenRouterClient(LLMConfig())
    monkeypatch.setattr(
        client._client.chat.completions, "create",
        lambda **kw: (sent.append(kw), _fake_response())[1],
    )

    conversation = ["CONTEXT", "TOOL RESULT a", "TOOL RESULT b"]
    client.complete(system="SYS", user="\n\n".join(conversation),
                    user_segments=list(conversation))

    blocks = sent[0]["messages"][1]["content"]
    assert [b["text"] for b in blocks] == conversation          # nothing lost
    # Exactly one breakpoint, on the newest segment: earlier blocks are then a
    # byte-identical prefix on the next call.
    assert sum("cache_control" in b for b in blocks) == 1
    assert "cache_control" in blocks[-1]

    # And the prefix property that makes it work at all.
    earlier = _incremental_parts(conversation[:2])
    later = _incremental_parts(conversation)
    assert [b["text"] for b in earlier] == [b["text"] for b in later[:2]]


def test_clients_without_the_segmented_parameter_are_untouched() -> None:
    from omniqec_agentic.llm.deepseek import DeepSeekClient
    from omniqec_agentic.orchestrator.loop import _accepts_segments

    assert _accepts_segments(OpenRouterClient.__new__(OpenRouterClient)) is True
    assert _accepts_segments(DeepSeekClient.__new__(DeepSeekClient)) is False

    class _Scripted:
        def complete(self, *, system, user):
            return "{}"

    assert _accepts_segments(_Scripted()) is False
