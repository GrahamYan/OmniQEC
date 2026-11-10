"""Resident BB baselines: measurement-side anchors for beats-BB comparisons
(03 §11.3: prompt-blind, injected by explicit opt-in only)."""

from __future__ import annotations

from typing import Any

from omniqec_agentic.construction import build_matrices

BASELINE_BB_SPEC = {
    "family": "bb",
    "ell": 6,
    "m": 6,
    "A": [[3, 0], [0, 1], [0, 2]],
    "B": [[0, 3], [1, 0], [2, 0]],
}
BASELINE_BB_PROGRAM_ID = "baseline_bb_72_12_6"

# BB [[144,12,12]] (Bravyi et al.): ell=12, m=6, A = x^3 + y + y^2,
# B = y^3 + x + x^2. Total qubits 4*ell*m = 288. (The bb validator
# canonicalizes to ell<=m by swapping the torus axes — a qubit relabeling.)
BASELINE_BB144_SPEC = {
    "family": "bb",
    "ell": 12,
    "m": 6,
    "A": [[3, 0], [0, 1], [0, 2]],
    "B": [[0, 3], [1, 0], [2, 0]],
}
BASELINE_BB144_PROGRAM_ID = "baseline_bb_144_12_12"


# [[n, k, d]] of the resident references (the code parameters, independent
# of any measurement) for callers that need them before the baseline has
# been measured in a run (v3.8 variant generations at turn 0).
RESIDENT_BASELINE_PARAMETERS: dict[str, tuple[int, int, int]] = {
    BASELINE_BB_PROGRAM_ID: (72, 12, 6),
    BASELINE_BB144_PROGRAM_ID: (144, 12, 12),
}


def resident_baseline_parameters(program_id: str) -> tuple[int, int, int]:
    return RESIDENT_BASELINE_PARAMETERS[str(program_id)]


# 2026-11-10 (paper Appendix D3): each BB reference "at its corresponding physical-qubit budget"
# (n + m_X + m_Z = 4 l m): BB72 at N = 144, BB144 at N = 288. Its final measurement runs up to the Table VII
# cumulative limit of THIS budget, whichever budget point the run itself has.
RESIDENT_BASELINE_BUDGETS: dict[str, int] = {
    BASELINE_BB_PROGRAM_ID: 144,
    BASELINE_BB144_PROGRAM_ID: 288,
}


def select_resident_baseline(n_budget: int) -> tuple[dict[str, Any], str] | None:
    """Pick the resident measurement anchor for a total-qubit campaign budget.

    A BB reference is injected where it fits the hardware budget: BB72 costs
    144 total qubits, BB144 costs 288. v3.6 (docs/06 §3 minor): every budget
    point >= 288 anchors on BB144 (b360 previously ran un-anchored although
    BB144 fits), so the cross-budget figure compares each bin against the
    anchor it challenges (144: BB72; >=288: BB144). Budgets strictly between
    144 and 288 run un-anchored because no BB reference fits."""

    budget = int(n_budget)
    if budget == 144:
        return (dict(BASELINE_BB_SPEC), BASELINE_BB_PROGRAM_ID)
    if budget >= 288:
        return (dict(BASELINE_BB144_SPEC), BASELINE_BB144_PROGRAM_ID)
    return None


def with_resident_baseline(
    records: list[dict[str, Any]],
    *,
    baseline_spec: dict[str, Any] | None = None,
    baseline_program_id: str | None = None,
) -> list[dict[str, Any]]:
    """Append the resident BB reference unless an identical code is queued.

    An empty queue stays empty: re-measuring the baseline with no candidates
    to compare against burns slow budget for nothing."""

    if not records:
        return records
    spec = baseline_spec or BASELINE_BB_SPEC
    program_id = baseline_program_id or BASELINE_BB_PROGRAM_ID
    try:
        from omniqec_agentic.verification.structural import code_hash as _code_hash

        hx, hz = build_matrices(spec)
        baseline_hash = _code_hash(hx, hz)
    except Exception:
        baseline_hash = None
    for record in records:
        if record.get("program_id") == program_id:
            return records
        if baseline_hash and record.get("code_hash") == baseline_hash:
            return records
    baseline_record = {
        "program_id": program_id,
        "spec": dict(spec),
        "code_hash": baseline_hash,
        "deformation_mode": "none",
        "is_baseline": True,
        "selection_reasons": ["resident_baseline"],
        "promotion_rule": "resident_baseline",
    }
    return list(records) + [baseline_record]
