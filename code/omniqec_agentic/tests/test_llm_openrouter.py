"""OpenRouter provider regressions; all transport is local and scripted."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.openrouter import OPENROUTER_MAX_OUTPUT_TOKENS
from omniqec_agentic.llm.factory import create_completion_client


class _FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            id="chatcmpl-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="OPENROUTER_OK",
                        reasoning_content="TRACEABLE_REASONING",
                    )
                )
            ],
            model="deepseek/deepseek-chat",
            provider="DeepSeek",
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )


class _FakeOpenAI:
    latest: "_FakeOpenAI | None" = None

    def __init__(self, **kwargs) -> None:
        self.options = kwargs
        completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=completions)
        _FakeOpenAI.latest = self


def test_openrouter_factory_uses_isolated_environment(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "OmniQEC tests")
    monkeypatch.delenv("OPENROUTER_SITE_URL", raising=False)

    client = create_completion_client(
        "openrouter",
        LLMConfig(max_tokens=20000, max_retries=0),
    )
    text = client.complete(system="Return one token.", user="Test")

    assert text == "OPENROUTER_OK"
    assert _FakeOpenAI.latest is not None
    assert _FakeOpenAI.latest.options["base_url"] == "https://openrouter.ai/api/v1"
    assert _FakeOpenAI.latest.options["default_headers"] == {
        "X-OpenRouter-Title": "OmniQEC tests"
    }
    request = _FakeOpenAI.latest.chat.completions.request
    assert request["model"] == "deepseek/deepseek-chat"
    # The config ceiling binds below the provider cap. (This asserted 16000
    # while the cap was DeepSeek's; that cap truncated batches from thinking
    # models, so the check now pins the relationship rather than the number.)
    assert request["max_tokens"] == 20000
    assert OPENROUTER_MAX_OUTPUT_TOKENS > 20000
    info = client.info()
    assert info["provider"] == "openrouter"
    assert info["api_key_present"] is True
    assert "test-openrouter-key" not in repr(info)
    assert info["usage"]["total_tokens"] == 10


def test_openrouter_reasoning_is_saved_with_provenance(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
    monkeypatch.setenv("OMNIQEC_REASONING_DB", str(tmp_path))

    client = create_completion_client("openrouter", LLMConfig(max_retries=0))
    assert client.complete(system="SYSTEM", user="USER") == "OPENROUTER_OK"

    records = list(tmp_path.rglob("*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text(encoding="utf-8"))
    assert records[0].stem == saved["trace_id"]
    assert saved["provider_response_id"] == "chatcmpl-test"
    assert saved["reasoning_text"] == "TRACEABLE_REASONING"
    assert saved["final_answer"] == "OPENROUTER_OK"
    assert saved["requested_model"] == "deepseek/deepseek-chat"
    assert len(saved["user_sha256"]) == len(saved["reasoning_sha256"]) == 64


def test_openrouter_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "omniqec_agentic.llm.openrouter.load_dotenv",
        lambda *args, **kwargs: None,
    )

    try:
        create_completion_client("openrouter", LLMConfig())
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("missing OpenRouter key was accepted")
