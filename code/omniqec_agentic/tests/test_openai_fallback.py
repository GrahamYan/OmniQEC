"""OpenAIClient fallback route (2026-09-11): when the primary gateway path
fails at the transport (connect/DNS/timeout/5xx/empty body), the same request
is retried through a second base URL; 4xx failures are not retried there."""

from __future__ import annotations

import httpx
import openai
import pytest

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm import openai_compat
from omniqec_agentic.llm.openai_compat import OpenAIClient, is_path_failure, load_fallback_endpoint
from omniqec_agentic.llm.retry import EmptyCompletion

PRIMARY = "https://cliproxy.example.net/v1"


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content="OK"):
        self.choices = [_Choice(content)]
        self.usage = None
        self.model = "gpt-5.6-sol"


def _connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://x/v1/chat/completions"))


def _status_error(code: int):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return openai.APIStatusError("boom", response=httpx.Response(code, request=req), body=None)


@pytest.fixture
def fallback_file(tmp_path, monkeypatch):
    path = tmp_path / "gpt_endpoint_fallback.env"
    path.write_text(
        "FALLBACK_FOR=cliproxy.example.net\n"
        "OPENAI_BASE_URL='https://worker.example.workers.dev/tok/cliproxy/v1'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(openai_compat.FALLBACK_FILE_ENV, str(path))
    monkeypatch.delenv("OPENAI_BASE_URL_FALLBACK", raising=False)
    return path


def _client(monkeypatch, max_retries=0):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", PRIMARY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    monkeypatch.setattr(openai_compat, "save_reasoning_response", lambda *a, **k: None)
    return OpenAIClient(LLMConfig(max_retries=max_retries, retry_backoff_s=0.0))


def test_connection_failure_on_the_primary_is_served_by_the_fallback(monkeypatch, fallback_file) -> None:
    client = _client(monkeypatch)
    assert client._fallback == ("https://worker.example.workers.dev/tok/cliproxy/v1", None)
    calls = {"primary": 0, "fallback": 0}

    def primary(**kwargs):
        calls["primary"] += 1
        raise _connection_error()

    def fallback(**kwargs):
        calls["fallback"] += 1
        return _Response()

    monkeypatch.setattr(client._client.chat.completions, "create", primary)
    monkeypatch.setattr(client._fallback_client.chat.completions, "create", fallback)
    assert client.complete(system="s", user="u") == "OK"
    assert calls == {"primary": 1, "fallback": 1}
    assert client.info()["served_via"] == "fallback"
    assert client.info()["fallback_base_url"].startswith("https://worker.example")
    # A later healthy primary call is reported as primary again.
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: _Response("ok2"))
    assert client.complete(system="s", user="u") == "ok2"
    assert client.info()["served_via"] == "primary"


def test_client_errors_do_not_cross_to_the_fallback(monkeypatch, fallback_file) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: (_ for _ in ()).throw(_status_error(401)))
    monkeypatch.setattr(client._fallback_client.chat.completions, "create", lambda **kw: pytest.fail("fallback must not be used for a 401"))
    with pytest.raises(openai.APIStatusError):
        client.complete(system="s", user="u")


def test_without_a_matching_fallback_the_primary_error_propagates(monkeypatch, tmp_path) -> None:
    path = tmp_path / "other.env"
    path.write_text("FALLBACK_FOR=someone.else.example.net\nOPENAI_BASE_URL=https://w/v1\n", encoding="utf-8")
    monkeypatch.setenv(openai_compat.FALLBACK_FILE_ENV, str(path))
    monkeypatch.delenv("OPENAI_BASE_URL_FALLBACK", raising=False)
    client = _client(monkeypatch)
    assert client._fallback is None and client._fallback_client is None
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: (_ for _ in ()).throw(_connection_error()))
    with pytest.raises(openai.APIConnectionError):
        client.complete(system="s", user="u")


def test_env_override_and_path_failure_classification(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL_FALLBACK", "https://w/v1")
    monkeypatch.setenv("OPENAI_API_KEY_FALLBACK", "sk-other")
    assert load_fallback_endpoint("anything") == ("https://w/v1", "sk-other")
    assert is_path_failure(_connection_error())
    assert is_path_failure(_status_error(503))
    assert not is_path_failure(_status_error(429))
    assert not is_path_failure(_status_error(401))
    assert is_path_failure(EmptyCompletion("openai returned a completion without choices"))
    assert not is_path_failure(RuntimeError("invalid api key"))
