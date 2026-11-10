"""Anthropic Claude client adapter for the multi-model harness."""

from __future__ import annotations

import os
from dataclasses import dataclass

from omniqec_agentic.config import LLMConfig, load_dotenv

from .reasoning_store import save_reasoning_response
from .retry import build_timeout, call_with_retries, per_call_deadline_s
from .usage import anthropic_usage


# Claude's max OUTPUT tokens (64000 for Opus 4.8) is below DeepSeek's 131072
# input-side cap that LLMConfig.max_tokens carries; clamp so the messages API
# never 400s. The proposal output (a handful of construct programs) fits easily.
CLAUDE_MAX_OUTPUT_TOKENS = 64000


@dataclass
class ClaudeClient:
    config: LLMConfig
    api_key_env: str = "ANTHROPIC_API_KEY"
    model_env: str = "ANTHROPIC_MODEL"
    default_model: str = "claude-opus-4-8"

    def __post_init__(self) -> None:
        load_dotenv()
        self._last_usage: dict[str, object] = {}
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("ClaudeClient requires `pip install anthropic`") from exc

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing Anthropic API key in ${self.api_key_env}")
        # max_retries=0: retries belong to call_with_retries; the SDK default
        # of 2 would silently triple every attempt's timeout budget.
        self._client = Anthropic(
            api_key=api_key,
            timeout=build_timeout(self.config),
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env, self.default_model)

    @property
    def temperature(self) -> float:
        # Opus 4.8 (extended thinking) only accepts temperature=1.0 (anything else
        # 400s); other Claude models take the configured value. The search wants
        # 1.0 anyway, so this also guards a bare run whose config default is 0.45.
        if "opus" in self.model.lower():
            return 1.0
        return float(self.config.temperature)

    @property
    def thinking(self) -> dict[str, str] | None:
        # Extended thinking, so reasoning tokens count like DeepSeek/OpenAI's
        # thinking-by-default. Opus 4.8 ONLY accepts {"type": "adaptive"} — the
        # {"type": "enabled", "budget_tokens": N} form 400s ("not supported for
        # this model. Use thinking.type.adaptive"). Env ANTHROPIC_THINKING=off
        # (or 0/false/disabled/none) turns it off.
        mode = os.environ.get("ANTHROPIC_THINKING", "adaptive").strip().lower()
        if mode in {"", "off", "0", "false", "disabled", "none", "no"}:
            return None
        return {"type": "adaptive"}

    def _invoke(self, *, system: str, user: str):
        kwargs = dict(
            model=self.model,
            system=system,
            max_tokens=min(self.config.max_tokens, CLAUDE_MAX_OUTPUT_TOKENS),
            temperature=self.temperature,
            messages=[{"role": "user", "content": user}],
        )
        thinking = self.thinking
        if thinking is None:
            return self._client.messages.create(**kwargs)
        # Adaptive thinking can push a single request past the SDK's 10-minute
        # non-streaming ceiling, so it MUST be streamed (a bare .create() raises
        # "Streaming is required for operations that may take longer than 10
        # minutes"). get_final_message() reassembles the same Message shape.
        kwargs["thinking"] = thinking
        with self._client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def complete(self, *, system: str, user: str) -> str:
        response = call_with_retries(
            lambda: self._invoke(system=system, user=user),
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            per_call_deadline_s=per_call_deadline_s(self.config),
            total_deadline_s=self.config.total_deadline_s,
        )
        self._last_usage = anthropic_usage(response)
        parts: list[str] = []
        for item in response.content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        answer = "\n".join(parts)
        save_reasoning_response(
            response,
            provider="claude",
            model=self.model,
            system=system,
            user=user,
            final_answer=answer,
            usage=self._last_usage,
            served_model=getattr(response, "model", None),
        )
        return answer

    def info(self) -> dict[str, object]:
        payload = {
            "provider": "claude",
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(os.environ.get(self.api_key_env, "").strip()),
            "temperature": self.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout_s": self.config.timeout_s,
            "thinking": self.thinking,
        }
        if self._last_usage:
            payload["usage"] = self._last_usage
        return payload
