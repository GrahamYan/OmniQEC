"""OpenRouter as the transport for the GPT and Claude arms.

Both arms run as `--provider openrouter`, told apart by OPENROUTER_MODEL. That
keeps one code path and makes the recorded model id (`openai/gpt-5.2`,
`anthropic/claude-opus-4.8`) the thing that identifies the arm — which is also
what the strict turn audit pins.
"""

from __future__ import annotations

import os

import pytest

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.openrouter import OPENROUTER_MAX_OUTPUT_TOKENS, OpenRouterClient
from omniqec_agentic.llm.usage import openai_usage
from omniqec_agentic.token_usage import usage_fields


class _Usage:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, payload, *, model="openai/gpt-5.2", provider="OpenAI", content="OK"):
        self.usage = _Usage(payload)
        self.model = model
        self.provider = provider
        self.choices = [_Choice(content)]


OPENROUTER_PAYLOAD = {
    "prompt_tokens": 5042,
    "completion_tokens": 19336,
    "total_tokens": 24378,
    "completion_tokens_details": {"reasoning_tokens": 16862},
    "prompt_tokens_details": {"cached_tokens": 1664},
    "cost": 0.00024325,
}


def test_openrouter_cost_survives_into_the_accounting_columns() -> None:
    usage = openai_usage(_Response(OPENROUTER_PAYLOAD))
    assert usage["cost_usd"] == pytest.approx(0.00024325)

    fields = usage_fields(usage)
    assert fields["input"] == 5042
    assert fields["output"] == 19336
    assert fields["reasoning"] == 16862
    assert fields["cache_hit"] == 1664
    assert fields["cost_usd"] == pytest.approx(0.00024325)


def test_direct_providers_report_no_cost_and_that_is_not_an_error() -> None:
    payload = dict(OPENROUTER_PAYLOAD)
    payload.pop("cost")
    usage = openai_usage(_Response(payload))
    assert "cost_usd" not in usage
    assert usage_fields(usage)["cost_usd"] == 0.0


def test_output_cap_is_no_longer_the_deepseek_era_ceiling(monkeypatch) -> None:
    # One b144 generation call spent 19,336 completion tokens; the old 16,000
    # default would have truncated a full batch from any thinking model.
    assert OPENROUTER_MAX_OUTPUT_TOKENS > 19_336

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    client = OpenRouterClient(LLMConfig())
    assert client.max_tokens == OPENROUTER_MAX_OUTPUT_TOKENS

    # An explicit cap still wins, and the config ceiling still binds.
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "8000")
    assert OpenRouterClient(LLMConfig()).max_tokens == 8000
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS")
    assert OpenRouterClient(LLMConfig(max_tokens=4096)).max_tokens == 4096


def test_served_model_is_recorded_because_a_router_may_substitute(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.2")
    client = OpenRouterClient(LLMConfig())

    substituted = _Response(OPENROUTER_PAYLOAD, model="openai/gpt-5.1", provider="Azure")
    monkeypatch.setattr(
        client._client.chat.completions, "create", lambda **kwargs: substituted
    )
    assert client.complete(system="s", user="u") == "OK"

    info = client.info()
    # The audit pins the REQUESTED model; without this the substitution would
    # leave no trace anywhere in the artifacts.
    assert info["model"] == "openai/gpt-5.2"
    assert info["served_model"] == "openai/gpt-5.1"
    assert info["served_upstream"] == "Azure"


def test_required_model_follows_the_provider_not_deepseeks_defaults(monkeypatch) -> None:
    """Regression: --strict-turns failed turn 0 for every non-DeepSeek arm.

    `LLMConfig()` carries DeepSeek's env names, so `config.model` resolved to
    `deepseek-chat` whatever `--provider` said. That value became
    `required_model`, and the per-call audit compared it against the model the
    client had actually recorded — a mismatch by construction.
    """

    from omniqec_agentic.llm.factory import resolve_model_name

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.8")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    # resolve_model_name constructs the client, whose __post_init__ requires an
    # API key; without this the assertion depends on the host's ambient env.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

    assert resolve_model_name("openrouter", LLMConfig()) == "anthropic/claude-opus-4.8"
    assert resolve_model_name("deepseek", LLMConfig()) == "deepseek-v4-pro"
    assert LLMConfig().model == "deepseek-v4-pro"  # the trap this replaced


def test_dry_run_resolves_to_no_model_instead_of_raising(monkeypatch) -> None:
    """Regression: run-campaign crashed under its own default provider.

    `resolve_model_name` was added to make `required_model` follow `--provider`
    (the test above), but `dry_run` is not a provider — it selects
    FallbackPolicy, which never calls a model. The unconditional call in
    `run_campaign_command` therefore hit `create_completion_client` and raised
    before any policy was built, so the documented keyless smoke could not run.
    """

    from omniqec_agentic.llm.factory import create_completion_client, resolve_model_name

    for key in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert resolve_model_name("dry_run", LLMConfig()) == ""

    # dry_run stays a non-provider: only the model-name lookup tolerates it, so
    # nothing can accidentally acquire a client for it.
    with pytest.raises(ValueError, match="provider must be one of"):
        create_completion_client("dry_run", LLMConfig())

    # The empty pin must not become a way to slip past the per-call audit: it is
    # reachable only for dry_run, and --strict-turns refuses dry_run outright
    # (pinned in test_strict_turns_no_longer_refuses_the_openrouter_arm).
    with pytest.raises(ValueError):
        resolve_model_name("not_a_provider", LLMConfig())


def test_strict_turns_no_longer_refuses_the_openrouter_arm(monkeypatch, tmp_path) -> None:
    """Regression: the formal protocol was unreachable on the OpenRouter route.

    `--strict-turns` was gated on `--provider deepseek`, a leftover from before
    `resolve_model_name` taught `required_model` to follow the provider. It made
    the GPT and DeepSeek arms this project now runs through OpenRouter — and
    fig2_campaign.pbs's own PROVIDER=openrouter branch — die before turn 0.
    Only dry_run is genuinely incompatible: it IS the FallbackPolicy.
    """

    import argparse

    from omniqec_agentic.cli import run_campaign_command

    def _args(provider: str) -> argparse.Namespace:
        return argparse.Namespace(
            strict_turns=True, provider=provider, budget=40, out=str(tmp_path),
            weight_limit=None, families=None, p=0.005, turns=1, proposal_count=24,
            generations_per_turn=1, rng_seeds_per_program=20, slow_workers=8,
            slow_shots=None, shot_shards=16, final_shot_shards=128, racing=False,
            racing_pilot=5000, racing_batch=5000,
            racing_confirm=20000, racing_max=50000, decoders=None,
            circuit_timeout_s=None, build_timeout_s=None,
            decode_shard_timeout_s=None, distance_timeout_s=None,
            timeout_retries=None, decode_shot_budget_s=None, transient_retries=None,
            final_shots=None, feedback_mode="both",
            resume=False,
        )

    with pytest.raises(ValueError, match="dry_run"):
        run_campaign_command(_args("dry_run"))

    # openrouter must get PAST the guard. It still fails further in (no key, no
    # real run here) — any error that is not the guard's proves the gate opened.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol")
    try:
        run_campaign_command(_args("openrouter"))
    except Exception as exc:  # noqa: BLE001 - the message is the assertion
        assert "strict-turns" not in str(exc), f"still blocked by the guard: {exc}"


def test_provider_routing_preference_is_forwarded_as_the_provider_block(monkeypatch) -> None:
    """2026-09-11: OPENROUTER_PROVIDER_SORT / _ORDER / ALLOW_FALLBACKS become
    the request's `provider` block (OpenRouter provider routing); unset means
    the router's default and no extra body at all."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro")
    for name in ("OPENROUTER_PROVIDER_SORT", "OPENROUTER_PROVIDER_ORDER", "OPENROUTER_ALLOW_FALLBACKS"):
        monkeypatch.delenv(name, raising=False)
    client = OpenRouterClient(LLMConfig())
    seen: dict = {}

    def _create(**kwargs):
        seen.clear(); seen.update(kwargs)
        return _Response(OPENROUTER_PAYLOAD, model="deepseek/deepseek-v4-pro", provider="Venice")

    monkeypatch.setattr(client._client.chat.completions, "create", _create)
    assert client.complete(system="s", user="u") == "OK"
    assert "extra_body" not in seen and client.info()["provider_preferences"] == {}

    monkeypatch.setenv("OPENROUTER_PROVIDER_SORT", "throughput")
    assert client.complete(system="s", user="u") == "OK"
    assert seen["extra_body"] == {"provider": {"sort": "throughput"}}

    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "DeepSeek, Fireworks")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "0")
    assert client.complete(system="s", user="u") == "OK"
    assert seen["extra_body"] == {"provider": {"sort": "throughput", "order": ["DeepSeek", "Fireworks"], "allow_fallbacks": False}}
    assert client.info()["provider_preferences"]["order"] == ["DeepSeek", "Fireworks"]

    monkeypatch.setenv("OPENROUTER_PROVIDER_SORT", "fastest")
    with pytest.raises(ValueError, match="OPENROUTER_PROVIDER_SORT"):
        client.complete(system="s", user="u")
