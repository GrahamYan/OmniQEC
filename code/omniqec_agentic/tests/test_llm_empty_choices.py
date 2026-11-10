"""A 2xx completion with no `choices` is a transient transport failure.

2026-09-11, ChatGPT arm on NSCC (a canary run, b60 turn 0 gen 3): the
Worker -> CLIProxyAPI gateway answered with a body the SDK parsed into
``choices=None``; ``response.choices[0]`` raised a bare
``TypeError: 'NoneType' object is not subscriptable`` outside the transport
retry, the generation was recorded as generation_error with parsed=0/24 and
the strict turn aborted (one full fast-loop restart per occurrence, FATAL_STOP
after four). The guard now lives INSIDE the retried callable, so the call is
retried like a timeout, and the recorded error names the provider and body.
"""

from __future__ import annotations

import json

import pytest

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.deepseek import DeepSeekClient
from omniqec_agentic.llm.openai_compat import OpenAIClient
from omniqec_agentic.llm.openrouter import OpenRouterClient
from omniqec_agentic.llm import retry as retry_mod
from omniqec_agentic.llm.retry import EmptyCompletion, call_with_retries, require_choices


@pytest.fixture(autouse=True)
def _no_server_backoff(monkeypatch):
    """The client tests count attempts with tiny configs; the long 5xx
    schedule (10/30/60/120 s) is exercised by its own test below."""

    monkeypatch.setenv("OMNIQEC_LLM_5XX_BACKOFF_S", "0")


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, *, choices, model="gpt-5.6-sol", error=None):
        self.choices = choices
        self.model = model
        self.usage = None
        self.provider = "gateway"
        self._error = error

    def model_dump(self):
        return {"choices": self.choices, "model": self.model, "error": self._error}


def _good():
    return _Response(choices=[_Choice("OK")])


def _empty(error=None):
    return _Response(choices=None, error=error)


def _fast_config(max_retries: int) -> LLMConfig:
    return LLMConfig(max_retries=max_retries, retry_backoff_s=0.0)


def _clients(monkeypatch, max_retries: int):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-pro")
    cfg = _fast_config(max_retries)
    return {
        "openai": OpenAIClient(cfg),
        "deepseek": DeepSeekClient(cfg),
        "openrouter": OpenRouterClient(cfg),
    }


@pytest.mark.parametrize("provider", ["openai", "deepseek", "openrouter"])
def test_an_empty_completion_is_retried_inside_the_transport(monkeypatch, provider) -> None:
    client = _clients(monkeypatch, max_retries=2)[provider]
    answers = iter([_empty(), _good()])
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(kwargs)
        return next(answers)

    monkeypatch.setattr(client._client.chat.completions, "create", _create)
    monkeypatch.setattr(
        f"omniqec_agentic.llm.{'openai_compat' if provider == 'openai' else provider}.save_reasoning_response",
        lambda *a, **k: None,
    )
    assert client.complete(system="s", user="u") == "OK"
    # One retry, same request: the strict turn never saw the empty body.
    assert len(calls) == 2
    assert calls[0]["messages"] == calls[1]["messages"]


@pytest.mark.parametrize("provider", ["openai", "deepseek", "openrouter"])
def test_a_persistently_empty_completion_names_the_provider_and_body(monkeypatch, provider) -> None:
    client = _clients(monkeypatch, max_retries=1)[provider]
    calls = 0

    def _create(**kwargs):
        nonlocal calls
        calls += 1
        return _empty(error={"message": "upstream hiccup"})

    monkeypatch.setattr(client._client.chat.completions, "create", _create)
    with pytest.raises(EmptyCompletion) as excinfo:
        client.complete(system="s", user="u")
    text = str(excinfo.value)
    assert text.startswith(f"{provider} returned a completion without choices")
    assert "upstream hiccup" in text  # the body survives into generation_error
    assert calls == 2  # max_retries=1 -> two attempts, then it surfaces


def test_require_choices_passes_a_normal_response_through() -> None:
    response = _good()
    assert require_choices(response, provider="openai") is response
    with pytest.raises(EmptyCompletion):
        require_choices(_Response(choices=[]), provider="openai")


def test_funsearch_records_the_traceback_and_the_true_provider(tmp_path, monkeypatch) -> None:
    """The llm_raw row of a failed call must say which provider failed and
    where; the old fallback (config.llm.to_dict()) labelled a ChatGPT
    failure as provider=deepseek / model=deepseek-chat."""
    from omniqec_agentic import funsearch as fs

    def _boom(**kwargs):
        raise TypeError("'NoneType' object is not subscriptable")

    from omniqec_agentic.config import CodeFamily, HardwareConfig, HardwareTarget, RunConfig

    from omniqec_agentic.construction.groups.catalog import (
        get_group_order_budget,
        set_group_order_budget,
    )

    monkeypatch.setattr(fs, "_complete_funsearch_prompt", _boom)
    # run_funsearch sets the process-global group-order budget from
    # n_budget; restore it so the catalog tests after this one keep their cap.
    previous_budget = get_group_order_budget()
    config = RunConfig(
        run_id="t",
        n_budget=60,
        hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir=str(tmp_path),
        proposal_count=2,
        allowed_families=tuple(CodeFamily),
    )
    try:
        fs.run_funsearch(config, generations=1, provider="openai", use_seeds=False)
    except Exception:
        pass  # strict-mode refusal or a plain empty run: the row is what matters
    finally:
        set_group_order_budget(previous_budget)
    rows = []
    for path in tmp_path.rglob("llm_raw.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    assert rows, "no llm_raw.jsonl written"
    row = rows[0]
    assert row["status"] == "generation_error"
    assert row["engine_info"]["provider"] == "openai"
    assert "info_source" in row["engine_info"]
    assert "TypeError" in row["generation_traceback"]
    assert "_boom" in row["generation_traceback"]


def test_an_empty_completion_typed_invalid_request_is_still_retried(monkeypatch) -> None:
    """2026-09-11: CLIProxyAPI answered 200 with {"error": {"message": "stream
    error: stream disconnected before completion ...", "type":
    "invalid_request_error"}}; the non-retryable marker match must not turn
    that transient into a strict-turn abort."""

    client = _clients(monkeypatch, max_retries=1)["openai"]
    answers = iter([_empty(error={"message": "stream error: stream disconnected", "type": "invalid_request_error"}), _good()])
    calls = 0

    def _create(**kwargs):
        nonlocal calls
        calls += 1
        return next(answers)

    monkeypatch.setattr(client._client.chat.completions, "create", _create)
    monkeypatch.setattr("omniqec_agentic.llm.openai_compat.save_reasoning_response", lambda *a, **k: None)
    assert client.complete(system="s", user="u") == "OK"
    assert calls == 2


class _Status503(RuntimeError):
    status_code = 503


def test_server_episodes_retry_on_the_long_schedule(monkeypatch) -> None:
    """2026-09-12: Anthropic "Overloaded" -> CLIProxyAPI cooldown -> 503
    auth_unavailable for about a minute. 2/4/8 s ended inside the episode and
    the strict turn restarted into a 15-minute quota poll; 5xx now waits
    10/30/60/120 s (env-overridable) and gets one attempt per delay."""

    monkeypatch.delenv("OMNIQEC_LLM_5XX_BACKOFF_S", raising=False)
    slept: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: slept.append(s))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise _Status503("Error code: 503 - auth_unavailable: no auth available")
        return "ok"

    assert call_with_retries(flaky, max_retries=3, backoff_s=2.0) == "ok"
    assert slept == [10.0, 30.0, 60.0] and calls["n"] == 4

    # Exhausted: one attempt per delay, then the last error surfaces.
    slept.clear(); calls["n"] = 0

    def down():
        calls["n"] += 1
        raise _Status503("503 still down")

    with pytest.raises(_Status503):
        call_with_retries(down, max_retries=1, backoff_s=2.0)
    assert slept == [10.0, 30.0, 60.0, 120.0] and calls["n"] == 5

    # A generic failure keeps the short schedule and max_retries.
    slept.clear(); calls["n"] = 0

    def generic():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        call_with_retries(generic, max_retries=3, backoff_s=2.0)
    assert slept == [2.0, 4.0, 8.0] and calls["n"] == 4

    # Env override.
    monkeypatch.setenv("OMNIQEC_LLM_5XX_BACKOFF_S", "1,2")
    slept.clear(); calls["n"] = 0
    with pytest.raises(_Status503):
        call_with_retries(down, max_retries=0, backoff_s=2.0)
    assert slept == [1.0, 2.0] and calls["n"] == 3
