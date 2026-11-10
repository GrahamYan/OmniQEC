"""OpenRouter chat client backed by the OpenAI-compatible API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omniqec_agentic.config import LLMConfig, load_dotenv

from .reasoning_store import save_reasoning_response
from .retry import build_timeout, call_with_retries, per_call_deadline_s, require_choices
from .usage import openai_usage


OPENROUTER_DEFAULT_MODEL = "deepseek/deepseek-chat"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# 16000 was a DeepSeek-era cap that silently applied to every model. One
# generation call in the b144 run spent 19,336 completion tokens (16,862 of
# them reasoning), so that ceiling truncates a full 24-program batch from any
# thinking model. Match the OpenAI/Claude arms instead; OpenRouter clamps to
# whatever the chosen model actually allows.
OPENROUTER_MAX_OUTPUT_TOKENS = 64000

# Anthropic bills a cached prefix at 10% of the input price, but ONLY where the
# request marks a breakpoint — there is no automatic caching, so a client that
# sends plain strings pays full price for every resend. The orchestrator's tool
# loop resends its whole conversation on each call (10 calls/turn measured on
# opus-5, 127k input tokens of which ~85% was repeat), and every one of those
# rows came back with cache_hit=0.
#
# Marking system + user gives two breakpoints inside Anthropic's limit of four.
# The conversation only ever grows by appending, so call N+1's prefix contains
# call N's marked text verbatim and reads it from cache. Measured on
# claude-opus-5 through OpenRouter: second call $0.0537 -> $0.0148 (-72%); the
# first pays a 25% write premium, so this wins from the second call onward.
#
# Gated to Anthropic because OpenAI and DeepSeek cache automatically and take
# no such marker. OPENROUTER_CACHE=off disables it.
_CACHE_CONTROL_PREFIXES = ("anthropic/",)


def _wants_cache_control(model: str) -> bool:
    if os.environ.get("OPENROUTER_CACHE", "").strip().lower() in {"off", "0", "false", "no"}:
        return False
    return any(model.startswith(prefix) for prefix in _CACHE_CONTROL_PREFIXES)


def _cached_part(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _incremental_parts(segments: list[str]) -> list[dict[str, Any]]:
    """One content block per conversation segment, breakpoint on the last.

    Anthropic matches a cached prefix at CONTENT BLOCK boundaries. Sending the
    whole growing conversation as a single block therefore caches almost
    nothing: block N+1 has different bytes from block N, so only the separate
    system block ever matched (measured on opus-5: 6,980 tokens read against
    97,797 sent). Emitting each segment as its own block makes calls 1..N-1 a
    literal block-wise prefix of call N, so the marker on the final block reads
    everything before it from cache and extends the entry by one segment.
    """

    parts: list[dict[str, Any]] = [{"type": "text", "text": seg} for seg in segments if seg]
    if parts:
        parts[-1]["cache_control"] = {"type": "ephemeral"}
    return parts


@dataclass
class OpenRouterClient:
    """Use OpenRouter without sharing credentials with other providers."""

    config: LLMConfig
    api_key_env: str = "OPENROUTER_API_KEY"
    model_env: str = "OPENROUTER_MODEL"
    base_url_env: str = "OPENROUTER_BASE_URL"
    max_tokens_env: str = "OPENROUTER_MAX_TOKENS"
    # 2026-09-11: routing preference forwarded to OpenRouter as the request's
    # `provider` block (https://openrouter.ai/docs/provider-routing). The
    # router otherwise picks any upstream serving the model, and for
    # deepseek-v4-pro those differ several-fold in speed (Venice / GMICloud /
    # StreamLake were all observed). Not a protocol field: the model asked
    # for -- and audited -- is unchanged; which upstream answered is already
    # recorded per call as served_upstream.
    #   OPENROUTER_PROVIDER_SORT   throughput | latency | price
    #   OPENROUTER_PROVIDER_ORDER  comma-separated upstream names, tried in order
    #   OPENROUTER_ALLOW_FALLBACKS 1 (default) | 0
    provider_sort_env: str = "OPENROUTER_PROVIDER_SORT"
    provider_order_env: str = "OPENROUTER_PROVIDER_ORDER"
    allow_fallbacks_env: str = "OPENROUTER_ALLOW_FALLBACKS"
    site_url_env: str = "OPENROUTER_SITE_URL"
    app_name_env: str = "OPENROUTER_APP_NAME"

    def __post_init__(self) -> None:
        load_dotenv()
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env.openrouter")
        self._last_usage: dict[str, object] = {}
        self._served: dict[str, object] = {}
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenRouterClient requires `pip install openai`") from exc

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing OpenRouter API key in ${self.api_key_env}")

        headers: dict[str, str] = {}
        site_url = os.environ.get(self.site_url_env, "").strip()
        app_name = os.environ.get(self.app_name_env, "OmniQEC").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-OpenRouter-Title"] = app_name

        # max_retries=0: retries belong to call_with_retries; the SDK default
        # of 2 would silently triple every attempt's timeout budget.
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=headers,
            timeout=build_timeout(self.config),
            max_retries=0,
        )

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env, OPENROUTER_DEFAULT_MODEL).strip()

    @property
    def base_url(self) -> str:
        return os.environ.get(self.base_url_env, OPENROUTER_DEFAULT_BASE_URL).strip()

    @property
    def max_tokens(self) -> int:
        raw = os.environ.get(
            self.max_tokens_env,
            str(OPENROUTER_MAX_OUTPUT_TOKENS),
        )
        value = int(raw)
        if value < 1:
            raise ValueError(f"${self.max_tokens_env} must be >= 1")
        return min(int(self.config.max_tokens), value)

    @property
    def provider_preferences(self) -> dict[str, Any]:
        """The `provider` routing block from the environment ({} = router default)."""

        prefs: dict[str, Any] = {}
        sort = os.environ.get(self.provider_sort_env, "").strip().lower()
        if sort:
            if sort not in ("throughput", "latency", "price"):
                raise ValueError(f"${self.provider_sort_env} must be throughput, latency or price")
            prefs["sort"] = sort
        order = [
            item.strip() for item in os.environ.get(self.provider_order_env, "").split(",") if item.strip()
        ]
        if order:
            prefs["order"] = order
        fallbacks = os.environ.get(self.allow_fallbacks_env, "").strip()
        if fallbacks and prefs:
            prefs["allow_fallbacks"] = fallbacks not in ("0", "false", "no")
        return prefs

    def complete(
        self, *, system: str, user: str, user_segments: list[str] | None = None
    ) -> str:
        """`user_segments` is the same text the caller would have joined.

        Passing the pieces lets an append-only conversation be cached
        block-by-block; joining them first makes every call a cache miss.
        """

        if _wants_cache_control(self.model):
            content = (
                _incremental_parts(user_segments) if user_segments else _cached_part(user)
            )
            messages = [
                {"role": "system", "content": _cached_part(system)},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        prefs = self.provider_preferences
        extra: dict[str, Any] = {"extra_body": {"provider": prefs}} if prefs else {}
        def _create():
            return require_choices(
                self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.config.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    **extra,
                ),
                provider="openrouter",
            )

        response = call_with_retries(
            _create,
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            per_call_deadline_s=per_call_deadline_s(self.config),
            total_deadline_s=self.config.total_deadline_s,
        )
        self._last_usage = openai_usage(response)
        # OpenRouter is a router: it may serve a request from a different
        # model or a different upstream than the one requested, and the strict
        # turn audit only ever checks the model we ASKED for. Record what
        # actually answered so the artifacts can be checked against it later.
        self._served = {
            "served_model": getattr(response, "model", None),
            "served_upstream": getattr(response, "provider", None),
        }
        answer = response.choices[0].message.content or ""
        save_reasoning_response(
            response,
            provider="openrouter",
            model=self.model,
            system=system,
            user=user,
            final_answer=answer,
            usage=self._last_usage,
            **self._served,
        )
        return answer

    def info(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "provider": "openrouter",
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(os.environ.get(self.api_key_env, "").strip()),
            "base_url": self.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.max_tokens,
            "provider_preferences": self.provider_preferences,
            "timeout_s": self.config.timeout_s,
        }
        if self._last_usage:
            payload["usage"] = self._last_usage
        payload.update(self._served)
        return payload
