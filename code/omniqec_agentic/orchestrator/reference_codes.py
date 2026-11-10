"""Reference codes: background handed to BOTH LLM roles of a campaign.

An experiment variant (2026-09-09, user decision): show the Claude and GPT
arms the codes another arm (DeepSeek) found at the same budget point under the
identical frozen protocol, then let them search again. The payload is a frozen
protocol key (``protocol.reference_codes``) so that (1) every link of a chain
resumes with the same background, (2) the run is a declared PROTOCOL VARIANT
(the formal validator refuses it), and (3) the exact text every turn saw is
reproducible from protocol_snapshot.json.

Two renderings of one payload:
- the generator (FunSearch) user prompt gets a tagged ``<source>`` block
  (``S-R1``) right after the strategy brief;
- the orchestrator context gets block ``S-11``.

Wording is background, not a ranking instruction: the prompt-neutrality rules
(no global ordering, no explicit k/d goal) apply to this block too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "reference_codes_v1"
SOURCE_ID = "S-R1"
ORCHESTRATOR_SOURCE_ID = "S-11"
_MAX_CODES = 16


def load_reference_codes(path: str | Path) -> dict[str, Any]:
    """Read and validate a reference-codes JSON file (returns a plain dict)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_reference_codes(payload)


def validate_reference_codes(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("reference codes payload must be a JSON object")
    codes = payload.get("codes")
    if not isinstance(codes, list) or not codes:
        raise ValueError("reference codes payload needs a non-empty 'codes' list")
    if len(codes) > _MAX_CODES:
        raise ValueError(f"reference codes payload carries {len(codes)} codes; at most {_MAX_CODES}")
    for index, code in enumerate(codes):
        if not isinstance(code, dict):
            raise ValueError(f"codes[{index}] is not an object")
        if not isinstance(code.get("spec"), dict) or not code["spec"].get("family"):
            raise ValueError(f"codes[{index}] needs a 'spec' object with a 'family'")
        if not str(code.get("label") or "").strip():
            raise ValueError(f"codes[{index}] needs a non-empty 'label'")
    schema = str(payload.get("schema") or SCHEMA)
    if schema != SCHEMA:
        raise ValueError(f"unsupported reference codes schema {schema!r} (expected {SCHEMA})")
    # A JSON round trip: the payload becomes part of the frozen protocol
    # snapshot and must compare equal across resumes.
    return json.loads(json.dumps({**payload, "schema": schema}, ensure_ascii=False, sort_keys=True))


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "?"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _code_line(code: dict[str, Any]) -> str:
    parts = [f"{code.get('label')}: {code.get('code') or ''} family={code.get('spec', {}).get('family')}"]
    if code.get("class"):
        parts.append(f"class={code['class']}")
    if code.get("A") is not None or code.get("B") is not None:
        parts.append(f"A = {code.get('A')}; B = {code.get('B')}")
    if code.get("w") is not None:
        parts.append(f"max check weight w={code['w']}")
    if code.get("plpr") is not None:
        ci = code.get("plpr_ci")
        ci_text = f" (95% CI {_fmt(ci[0])}..{_fmt(ci[1])})" if isinstance(ci, list) and len(ci) == 2 else ""
        parts.append(f"measured per_logical_per_round={_fmt(code['plpr'])}{ci_text}")
    if code.get("code_hash8"):
        parts.append(f"code_hash={code['code_hash8']}")
    parts.append("spec=" + json.dumps(code.get("spec"), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "  - " + " | ".join(parts)


def render_reference_codes(payload: dict[str, Any]) -> str:
    """The background text (shared by both renderings)."""

    lines = ["BACKGROUND: REFERENCE CODES."]
    if payload.get("source"):
        lines.append(f"Source: {payload['source']}")
    if payload.get("protocol_note"):
        lines.append(f"Measurement: {payload['protocol_note']}")
    for code in payload.get("codes") or []:
        lines.append(_code_line(code))
    lines.append(
        "Status: background only. These codes are not parents and are not in this "
        "campaign's archive. Constructions that reproduce them, vary them, or "
        "explore the same structural classes are all in scope, on the same footing "
        "as any other construction; their specs above are exact and can be returned "
        "verbatim by a construct program."
    )
    if payload.get("note"):
        lines.append(str(payload["note"]))
    return "\n".join(lines)


def reference_block(payload: dict[str, Any]) -> str:
    """The generator-prompt rendering: one tagged source block."""

    text = render_reference_codes(payload)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return (
        f'<source id="{SOURCE_ID}" kind="reference_codes" origin="protocol.reference_codes" '
        f'hash="sha256:{digest}">\n{text}\n</source>'
    )
