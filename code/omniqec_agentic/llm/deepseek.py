"""DeepSeek chat client.

Credentials are read from `.env` or environment variables. No fallback API key is
defined here by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from omniqec_agentic.config import LLMConfig, load_dotenv

from .reasoning_store import save_reasoning_response
from .retry import build_timeout, call_with_retries, per_call_deadline_s, require_choices
from .usage import openai_usage


@dataclass
class DeepSeekClient:
    config: LLMConfig

    def __post_init__(self) -> None:
        load_dotenv()
        self._last_usage: dict[str, object] = {}
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("DeepSeekClient requires `pip install openai`") from exc

        # max_retries=0: retries belong to call_with_retries. Left at the SDK
        # default of 2, every attempt there becomes three HTTP attempts, so one
        # logical call could stack 12 timeout windows.
        self._client = OpenAI(
            api_key=self.config.require_api_key(),
            base_url=self.config.base_url,
            timeout=build_timeout(self.config),
            max_retries=0,
        )

    def complete(self, *, system: str, user: str) -> str:
        def _create():
            return require_choices(
                self._client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ),
                provider="deepseek",
            )

        response = call_with_retries(
            _create,
            max_retries=self.config.max_retries,
            backoff_s=self.config.retry_backoff_s,
            per_call_deadline_s=per_call_deadline_s(self.config),
            total_deadline_s=self.config.total_deadline_s,
        )
        self._last_usage = openai_usage(response)
        answer = response.choices[0].message.content or ""
        save_reasoning_response(
            response,
            provider="deepseek",
            model=self.config.model,
            system=system,
            user=user,
            final_answer=answer,
            usage=self._last_usage,
            served_model=getattr(response, "model", None),
        )
        return answer

    def info(self) -> dict[str, object]:
        payload = self.config.to_dict()
        if self._last_usage:
            payload["usage"] = self._last_usage
        return payload
