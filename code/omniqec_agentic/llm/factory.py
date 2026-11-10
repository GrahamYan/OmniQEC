"""Provider registry for LLM adapters."""

from __future__ import annotations

from typing import Protocol

from omniqec_agentic.config import LLMConfig


class CompletionClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...

    def info(self) -> dict[str, object]: ...


SUPPORTED_LLM_PROVIDERS = ("deepseek", "openrouter", "claude", "openai")


def resolve_model_name(provider: str, config: LLMConfig) -> str:
    """The model id THIS provider will actually send.

    Each client resolves its own model from its own env var, but LLMConfig's
    defaults are DeepSeek's — so reading `config.model` for any other provider
    returns `deepseek-chat`. That value is what --strict-turns pins as
    `required_model`, and the per-call audit compares it against the model the
    client recorded, so a non-DeepSeek campaign failed its very first turn on a
    mismatch between two names that were never meant to be compared.

    Asking the client keeps one source of truth per provider.

    `dry_run` is not a provider: it selects FallbackPolicy, which never calls a
    model, so there is no model id to pin. It returns "" rather than raising --
    the caller pins `required_model` unconditionally, and --strict-turns (the
    only consumer of that pin) already refuses to run with dry_run.
    """

    if str(provider) == "dry_run":
        return ""
    return str(create_completion_client(provider, config).info().get("model") or "")


def create_completion_client(provider: str, config: LLMConfig) -> CompletionClient:
    provider = str(provider)
    if provider == "deepseek":
        from .deepseek import DeepSeekClient

        return DeepSeekClient(config)
    if provider == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(config)
    if provider == "claude":
        from .anthropic_claude import ClaudeClient

        return ClaudeClient(config)
    if provider == "openai":
        from .openai_compat import OpenAIClient

        return OpenAIClient(config)
    raise ValueError(f"provider must be one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}")
