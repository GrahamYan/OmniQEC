"""Regressions for the hang that cost the 2026-08-19 b60 campaign two hours.

The process sat in poll() on a half-closed socket while a 900s transport
timeout never fired, and the SDK's own retries multiplied the outer retry
budget on top of that. These tests pin both halves of the fix.
"""

from __future__ import annotations

import time

import pytest

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.retry import (
    LLMCallTimeout,
    build_timeout,
    call_with_deadline,
    call_with_retries,
    per_call_deadline_s,
)


def test_deadline_abandons_a_call_that_never_returns():
    """The exact failure mode: a call that blocks forever."""
    started = time.monotonic()
    with pytest.raises(LLMCallTimeout):
        call_with_deadline(lambda: time.sleep(30), deadline_s=0.2)
    assert time.monotonic() - started < 5, "caller waited on the abandoned call"


def test_deadline_returns_a_prompt_result_untouched():
    assert call_with_deadline(lambda: "ok", deadline_s=10) == "ok"


def test_deadline_forwards_the_original_exception():
    def boom():
        raise ValueError("upstream detail")

    with pytest.raises(ValueError, match="upstream detail"):
        call_with_deadline(boom, deadline_s=10)


def test_no_deadline_runs_inline():
    assert call_with_deadline(lambda: 7, deadline_s=None) == 7


def test_hung_call_is_retried_then_surfaced():
    calls = []

    def hang():
        calls.append(1)
        time.sleep(30)

    with pytest.raises(LLMCallTimeout):
        call_with_retries(
            hang, max_retries=2, backoff_s=0.01, per_call_deadline_s=0.1
        )
    assert len(calls) == 3, "a hung call must retry, not fail on first timeout"


def test_total_budget_caps_the_whole_sequence():
    """Retries must not stack their per-call budgets into hours."""
    calls = []

    def hang():
        calls.append(1)
        time.sleep(30)

    started = time.monotonic()
    with pytest.raises(LLMCallTimeout):
        call_with_retries(
            hang,
            max_retries=20,
            backoff_s=0.01,
            per_call_deadline_s=0.2,
            total_deadline_s=0.6,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"budget overrun: {elapsed:.1f}s"
    assert len(calls) < 21, "the total budget did not stop the retry loop"


def test_transient_failure_still_recovers():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503 service unavailable")
        return "recovered"

    result = call_with_retries(
        flaky, max_retries=3, backoff_s=0.01,
        per_call_deadline_s=5, total_deadline_s=30,
    )
    assert result == "recovered"
    assert len(attempts) == 3


def test_auth_failure_is_not_retried():
    attempts = []

    def bad_key():
        attempts.append(1)
        raise RuntimeError("Invalid api key provided")

    with pytest.raises(RuntimeError):
        call_with_retries(
            bad_key, max_retries=5, backoff_s=0.01, per_call_deadline_s=5
        )
    assert len(attempts) == 1


def test_timeout_is_granular_so_connect_cannot_burn_the_read_budget():
    config = LLMConfig()
    timeout = build_timeout(config)
    httpx = pytest.importorskip("httpx")
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == config.connect_timeout_s
    assert timeout.read == config.timeout_s
    assert config.connect_timeout_s < config.timeout_s


def test_hard_deadline_leaves_room_for_the_transport_to_time_out_first():
    config = LLMConfig()
    assert per_call_deadline_s(config) > config.timeout_s


def test_config_exposes_a_bounded_total_budget():
    config = LLMConfig()
    assert config.total_deadline_s > 0
    # A logical call must not be able to outlast a turn by hours.
    assert config.total_deadline_s <= 3600


@pytest.mark.parametrize(
    "module_name, client_name",
    [
        ("deepseek", "DeepSeekClient"),
        ("openai_compat", "OpenAICompatClient"),
        ("anthropic_claude", "AnthropicClaudeClient"),
        ("openrouter", "OpenRouterClient"),
    ],
)
def test_every_client_disables_sdk_side_retries(module_name, client_name):
    """SDK retries multiply ours: 4 outer attempts became 12 HTTP attempts."""
    import importlib
    import inspect

    module = importlib.import_module(f"omniqec_agentic.llm.{module_name}")
    source = inspect.getsource(module)
    assert "max_retries=0" in source, f"{client_name} lets the SDK retry on its own"
    assert "build_timeout(self.config)" in source, f"{client_name} lacks granular timeouts"
    assert "total_deadline_s=self.config.total_deadline_s" in source, (
        f"{client_name} has no hard wall-clock ceiling"
    )
