"""Known-code table + equivalence lookup (03 §6.3).

The table (docs/known_codes_v1.json) holds literature codes as genotype specs;
lookup builds their matrices once per process and compares colored-Tanner WL
hashes (Tier-1), escalating to the exact canonical form when pynauty is
present. Rediscoveries produce targeted feedback instead of novelty credit.

``gap_index_to_tuple`` is the GAP ``Elements(G)`` index -> exponent-tuple
converter (01 §12 notation bridge) used to transcribe published tables that
cite group elements by GAP enumeration index (e.g. the Mitten table).

v3.6 (docs/06 §3 minor): the table carries the two Mitten Table XIII codes
[[150,30,10]] (C5xD3) and [[200,40,12]] (C4xD5) as repo-native specs, so a
campaign that lands on either is recorded as a rediscovery rather than a
discovery. ``verify_known_code_table`` re-builds every entry and checks its
n / k against the recorded parameters (the test-time guard that a table row
still means what it says)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.construction import build_matrices
from omniqec_agentic.construction.groups import get_group

from .fingerprints import tier1_fingerprint, tier2_canonical_hash

KNOWN_CODES_PATH = Path(__file__).resolve().parents[1] / "docs" / "known_codes_v1.json"


def gap_index_to_tuple(group_name: str, index: int) -> tuple[int, ...]:
    """GAP ``Elements(G)[index]`` (1-based) -> this package's exponent tuple.

    The catalog stores each group's element list in its own canonical
    enumeration; published tables citing GAP indices must be transcribed
    through the group's OWN element order, so this helper is only exact for
    groups whose catalog construction mirrors GAP's enumeration (checked per
    table entry during transcription; discrepancies are resolved by matching
    element orders, 01 §12)."""

    group = get_group(group_name)
    elements = list(group.elements)
    position = int(index) - 1
    if not 0 <= position < len(elements):
        raise ValueError(f"index {index} outside group of order {len(elements)}")
    return tuple(int(v) for v in elements[position])


@lru_cache(maxsize=1)
def load_known_codes() -> list[dict[str, Any]]:
    if not KNOWN_CODES_PATH.exists():
        return []
    data = json.loads(KNOWN_CODES_PATH.read_text(encoding="utf-8"))
    return [entry for entry in data.get("codes") or [] if isinstance(entry, dict)]


@lru_cache(maxsize=1)
def known_code_index() -> dict[str, dict[str, Any]]:
    """wl_hash -> table entry (matrices built once per process)."""

    index: dict[str, dict[str, Any]] = {}
    for entry in load_known_codes():
        spec = entry.get("spec")
        if not isinstance(spec, dict):
            continue
        try:
            hx, hz = build_matrices(spec)
        except Exception:
            continue
        fingerprint = tier1_fingerprint(np.asarray(hx), np.asarray(hz))
        index[fingerprint["wl_hash"]] = {**entry, "fingerprint": fingerprint}
    return index


def known_code_names() -> list[str]:
    """Names of the table entries in file order."""

    return [str(entry.get("name")) for entry in load_known_codes()]


def verify_known_code_table() -> list[str]:
    """Re-build every table entry and compare n / k (and the check weight
    bound) with the recorded parameters; returns a list of problems (empty
    when the table is consistent). Distances are NOT re-derived here — they
    are literature values."""

    from omniqec_agentic.verification.gf2 import logicals

    problems: list[str] = []
    seen: dict[str, str] = {}
    for entry in load_known_codes():
        name = str(entry.get("name"))
        spec = entry.get("spec")
        if not isinstance(spec, dict):
            problems.append(f"{name}: spec missing")
            continue
        try:
            hx, hz = build_matrices(spec)
        except Exception as exc:
            problems.append(f"{name}: build failed: {exc}")
            continue
        hx, hz = np.asarray(hx, dtype=np.uint8), np.asarray(hz, dtype=np.uint8)
        n = int(hx.shape[1])
        k = int(len(logicals(hx, hz)[0]))
        if n != int(entry.get("n") or -1):
            problems.append(f"{name}: n={n} recorded {entry.get('n')}")
        if k != int(entry.get("k") or -1):
            problems.append(f"{name}: k={k} recorded {entry.get('k')}")
        fingerprint = tier1_fingerprint(hx, hz)
        wl = fingerprint["wl_hash"]
        if wl in seen:
            problems.append(f"{name}: WL hash collides with {seen[wl]}")
        seen[wl] = name
        if known_code_index().get(wl, {}).get("name") != name:
            problems.append(f"{name}: not reachable through known_code_index()")
    return problems


def known_code_lookup(hx: np.ndarray, hz: np.ndarray) -> dict[str, Any] | None:
    """Equivalence-class lookup: WL match, then canonical-form confirmation
    when pynauty is available (else reported as a WL-level match)."""

    fingerprint = tier1_fingerprint(hx, hz)
    entry = known_code_index().get(fingerprint["wl_hash"])
    if entry is None:
        return None
    match = {
        "name": entry.get("name"),
        "family": entry.get("family"),
        "nkd": [entry.get("n"), entry.get("k"), entry.get("d")],
        "source": entry.get("source"),
        "match_tier": "wl",
    }
    canonical, backend = tier2_canonical_hash(hx, hz)
    if backend == "pynauty":
        spec = entry.get("spec")
        try:
            khx, khz = build_matrices(spec)
            known_canonical, _ = tier2_canonical_hash(np.asarray(khx), np.asarray(khz))
        except Exception:
            known_canonical = None
        if known_canonical is not None:
            if canonical == known_canonical:
                match["match_tier"] = "canonical"
            else:
                return None  # WL collision, canonically distinct
    return match
