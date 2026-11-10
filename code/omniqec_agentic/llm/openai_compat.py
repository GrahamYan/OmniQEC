"""OpenAI-兼容 chat 客户端(OpenAI 官方,或 OpenAI-兼容中转如 bltcy)。

凭据从 `.env` / 环境变量读:OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL。
与 Claude 臂口径一致:输出上限 64000、温度强制 1.0(gpt-5 推理模型只接受 1.0,
搜索本就要 1.0)。走 openai SDK 的 chat.completions(与 deepseek 客户端同一套,
deepseek 也是 openai SDK + base_url)。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from omniqec_agentic.config import LLMConfig, load_dotenv

from .reasoning_store import save_reasoning_response
from .retry import EmptyCompletion, LLMCallTimeout, build_timeout, call_with_retries, per_call_deadline_s, require_choices
from .usage import openai_usage

# 与 Claude(CLAUDE_MAX_OUTPUT_TOKENS)一致的输出上限
OPENAI_MAX_OUTPUT_TOKENS = 64000

# 2026-09-11: a second route to the same gateway. The GPT arm's primary is
# the Lab CLIProxyAPI's Tailscale Funnel name (direct from NSCC); its
# fallback is the Cloudflare Worker route to the same instance. Each path
# fails on its own (the Funnel name's DNS on the cluster, 1016 at the edge,
# a compute node's route), rarely both at once. Configured by
# OPENAI_BASE_URL_FALLBACK, or by a file whose FALLBACK_FOR names the
# primary host so the Claude arm (same client class, other host) never
# picks it up by accident.
FALLBACK_FILE_ENV = "OPENAI_ENDPOINT_FALLBACK_FILE"
DEFAULT_FALLBACK_FILE = "~/.cli-proxy-api/gpt_endpoint_fallback.env"


def load_fallback_endpoint(primary_host: str) -> tuple[str, str | None] | None:
    """(base_url, api_key or None) of the fallback for `primary_host`, else None."""

    url = os.environ.get("OPENAI_BASE_URL_FALLBACK", "").strip()
    if url:
        return url, (os.environ.get("OPENAI_API_KEY_FALLBACK", "").strip() or None)
    path = Path(os.environ.get(FALLBACK_FILE_ENV, "").strip() or DEFAULT_FALLBACK_FILE).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    if values.get("FALLBACK_FOR", "").strip().lower() != (primary_host or "").lower():
        return None
    url = values.get("OPENAI_BASE_URL", "").strip()
    if not url:
        return None
    return url, (values.get("OPENAI_API_KEY", "").strip() or None)


def is_path_failure(exc: BaseException) -> bool:
    """Failures worth retrying on the other route: the request never got a
    usable answer from the gateway (connect/DNS/timeout, 5xx, empty body).
    4xx (auth, bad request) would repeat on the other route and are not."""

    if isinstance(exc, (EmptyCompletion, LLMCallTimeout)):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return int(getattr(exc, "status_code", 0) or 0) >= 500
    return False


@dataclass
class OpenAIClient:
    config: LLMConfig
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    model_env: str = "OPENAI_MODEL"
    default_model: str = "gpt-5.5"
    default_base_url: str = "https://api.bltcy.ai/v1"   # 必须带 /v1:openai SDK 会拼 /chat/completions

    def __post_init__(self) -> None:
        load_dotenv()
        self._last_usage: dict[str, object] = {}
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAIClient requires `pip install openai`") from exc

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing OpenAI API key in ${self.api_key_env}")
        base_url = (os.environ.get(self.base_url_env, "") or self.default_base_url).strip()
        # 2026-09-11: the gateway hostname resolves intermittently on NSCC;
        # remember the last good answer and fall back to OMNIQEC_DNS_PINS.
        from .dns_sticky import host_of, install as install_sticky_dns

        install_sticky_dns([host_of(base_url)])
        # max_retries=0: retries belong to call_with_retries; the SDK default
        # of 2 would silently triple every attempt's timeout budget.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=build_timeout(self.config),
            max_retries=0,
        )
        self._base_url = base_url
        self._served_via = "primary"
        self._fallback = load_fallback_endpoint(host_of(base_url))
        self._fallback_client = None
        if self._fallback is not None:
            install_sticky_dns([host_of(self._fallback[0])])
            self._fallback_client = OpenAI(
                api_key=self._fallback[1] or api_key,
                base_url=self._fallback[0],
                timeout=build_timeout(self.config),
                max_retries=0,
            )

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env, self.default_model)

    def _create_with(self, client, *, system: str, user: str):
        def _create():
            return require_choices(
                client.chat.completions.create(
                    model=self.model,
                    temperature=1.0,  # gpt-5 推理模型只接受 1.0;搜索也要 1.0
                    max_tokens=min(self.config.max_tokens, OPENAI_MAX_OUTPUT_TOKENS),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ),
                provider="openai",
            )

        return _create

    def _call(self, client):
        return lambda **kw: call_with_retries(
            self._create_with(client, **kw),
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            per_call_deadline_s=per_call_deadline_s(self.config),
            total_deadline_s=self.config.total_deadline_s,
        )

    def complete(self, *, system: str, user: str) -> str:
        try:
            response = self._call(self._client)(system=system, user=user)
            self._served_via = "primary"
        except Exception as exc:
            if self._fallback_client is None or not is_path_failure(exc):
                raise
            from .dns_sticky import host_of

            print(
                f"[llm] primary endpoint {host_of(self._base_url)} failed "
                f"({type(exc).__name__}: {str(exc)[:160]}); retrying through the "
                f"fallback endpoint {host_of(self._fallback[0])}",
                file=sys.stderr,
                flush=True,
            )
            response = self._call(self._fallback_client)(system=system, user=user)
            self._served_via = "fallback"
        self._last_usage = openai_usage(response)
        answer = response.choices[0].message.content or ""
        save_reasoning_response(
            response,
            provider="openai",
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
            "provider": "openai",
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_present": bool(os.environ.get(self.api_key_env, "").strip()),
            "base_url": os.environ.get(self.base_url_env, self.default_base_url),
            "fallback_base_url": self._fallback[0] if self._fallback else None,
            "served_via": self._served_via,
            "temperature": 1.0,
            "max_tokens": min(self.config.max_tokens, OPENAI_MAX_OUTPUT_TOKENS),
            "timeout_s": self.config.timeout_s,
        }
        if self._last_usage:
            payload["usage"] = self._last_usage
        return payload
