"""LLM token/usage 归一化。

把 provider 各自的 usage 对象(openai SDK 的 ChatCompletion.usage、anthropic 的
Message.usage)转成纯 dict,塞进 client.info() -> engine_info -> llm_raw.jsonl,
这样每次 LLM 调用的 token 开销都被逐代逐次记录下来(消融要统计 token)。

设计:既保留 provider 原始 `raw`(不丢任何字段,如 cache/reasoning 明细),又给出
跨 provider 统一的 `input_tokens/output_tokens/total_tokens/reasoning_tokens`,
分析脚本读归一化字段即可,口径与历史一致。
"""

from __future__ import annotations

from typing import Any


def _to_plain(obj: Any) -> dict[str, Any] | None:
    """尽力把 pydantic/usage 对象转成纯 dict(JSON 可序列化)。"""
    if obj is None:
        return None
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    if isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            val = getattr(obj, key)
        except Exception:
            continue
        if callable(val):
            continue
        out[key] = val
    return out


def openai_usage(response: Any) -> dict[str, Any]:
    """OpenAI-SDK 形状的响应(OpenAI 官方 / DeepSeek 走 base_url,同一套)。"""
    raw = _to_plain(getattr(response, "usage", None))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {"raw": raw}
    if raw.get("prompt_tokens") is not None:
        out["input_tokens"] = raw.get("prompt_tokens")
    if raw.get("completion_tokens") is not None:
        out["output_tokens"] = raw.get("completion_tokens")
    if raw.get("total_tokens") is not None:
        out["total_tokens"] = raw.get("total_tokens")
    details = raw.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        out["reasoning_tokens"] = details.get("reasoning_tokens")
    # OpenRouter returns the charge for the call in USD. That beats
    # reconstructing cost from a price table afterwards: it is the amount
    # actually billed, by the provider that billed it, at the price in force
    # at call time. Absent for direct providers, which is why it is optional.
    if raw.get("cost") is not None:
        out["cost_usd"] = raw.get("cost")
    return out


def anthropic_usage(response: Any) -> dict[str, Any]:
    """Anthropic messages 响应(input/output + cache 明细在 raw 里)。"""
    raw = _to_plain(getattr(response, "usage", None))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {"raw": raw}
    if raw.get("input_tokens") is not None:
        out["input_tokens"] = raw.get("input_tokens")
    if raw.get("output_tokens") is not None:
        out["output_tokens"] = raw.get("output_tokens")
    out["total_tokens"] = int(raw.get("input_tokens") or 0) + int(raw.get("output_tokens") or 0)
    return out
