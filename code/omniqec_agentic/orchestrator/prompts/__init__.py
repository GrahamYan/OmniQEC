"""Prompt-as-code (03a R1): versioned files, content hash = identity.

``load_prompt(name)`` returns (text, hash8). prompt_ids go into the protocol
snapshot — changing a prompt is changing the protocol; runs across prompt
versions are never mixed. VERSIONS.md is the human-readable alias table; the
audit record only ever trusts the hash."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def _canonicalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> tuple[str, str]:
    path = PROMPTS_DIR / f"{name}.md"
    text = _canonicalize(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return text, digest


def prompt_ids() -> dict[str, str]:
    """All prompt hashes for the protocol snapshot."""

    ids: dict[str, str] = {}
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        if path.name == "VERSIONS.md":
            continue
        _, digest = load_prompt(path.stem)
        ids[path.stem] = digest
    return ids
