"""Optional append-only store for provider-exposed reasoning traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REASONING_DB_ENV = "OMNIQEC_REASONING_DB"


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return _plain(method())
            except Exception:
                pass
    return str(value)


def _extract_reasoning(response: Any) -> tuple[str, str | None, Any]:
    choices = _get(response, "choices") or []
    message = _get(choices[0], "message") if choices else None
    if message is not None:
        details = _plain(_get(message, "reasoning_details"))
        rows = details if isinstance(details, list) else ([details] if details else [])
        raw_text: list[str] = []
        summaries: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("type") or "")
            if kind == "reasoning.text" and row.get("text"):
                raw_text.append(str(row["text"]))
            elif kind == "reasoning.summary" and row.get("summary"):
                summaries.append(str(row["summary"]))
        if raw_text:
            return "raw_text", "\n".join(raw_text), details

        reasoning_content = _get(message, "reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            return "raw_text", reasoning_content, details
        reasoning = _get(message, "reasoning")
        if isinstance(reasoning, str) and reasoning:
            return "provider_text", reasoning, details
        if summaries:
            return "summary", "\n".join(summaries), details
        if details:
            return "encrypted", None, details

    blocks: list[Any] = []
    thinking_text: list[str] = []
    for item in _get(response, "content") or []:
        block_type = str(_get(item, "type") or "")
        if block_type not in {"thinking", "redacted_thinking"}:
            continue
        blocks.append(_plain(item))
        text = _get(item, "thinking") or _get(item, "text")
        if isinstance(text, str) and text:
            thinking_text.append(text)
    if thinking_text:
        return "provider_text", "\n".join(thinking_text), blocks
    if blocks:
        return "encrypted", None, blocks
    return "", None, None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def save_reasoning_response(
    response: Any,
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    final_answer: str,
    usage: dict[str, object] | None = None,
    served_model: object = None,
    served_upstream: object = None,
) -> Path | None:
    """Save one self-contained record when the provider exposes reasoning."""

    root_text = os.environ.get(REASONING_DB_ENV, "").strip()
    if not root_text:
        return None
    reasoning_kind, reasoning_text, reasoning_details = _extract_reasoning(response)
    if not reasoning_kind:
        return None

    created_at = datetime.now(timezone.utc)
    trace_id = uuid.uuid4().hex
    reasoning_payload = {
        "kind": reasoning_kind,
        "text": reasoning_text,
        "details": reasoning_details,
    }
    record = {
        "schema": "omniqec_reasoning_v1",
        "trace_id": trace_id,
        "created_at": created_at.isoformat(),
        "provider_response_id": _plain(_get(response, "id")),
        "provider": str(provider),
        "requested_model": str(model),
        "served_model": _plain(served_model),
        "served_upstream": _plain(served_upstream),
        "reasoning_kind": reasoning_kind,
        "reasoning_text": reasoning_text,
        "reasoning_details": reasoning_details,
        "system": system,
        "user": user,
        "final_answer": final_answer,
        "system_sha256": _digest(system),
        "user_sha256": _digest(user),
        "input_sha256": _digest(system + "\0" + user),
        "output_sha256": _digest(final_answer),
        "reasoning_sha256": _digest(
            json.dumps(reasoning_payload, ensure_ascii=False, sort_keys=True)
        ),
        "usage": _plain(usage or {}),
    }

    directory = (
        Path(root_text).expanduser()
        / _path_component(str(provider))
        / _path_component(str(model))
        / created_at.strftime("%Y-%m-%d")
    )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{trace_id}.json"
    temporary = directory / f".{trace_id}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
