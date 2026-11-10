"""Strict structural checks that do not replace the ver1 fast score."""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.construction import (
    build_compiled_code,
    compiled_code_hash,
    symplectic_stabilizers,
)
from omniqec_agentic.schemas import CompiledCode, StructuralResult

from .gf2 import rank2

# Unified per-check stabilizer weight cap (design doc §6): part of the
# cross-family fairness protocol; overridable via VerificationConfig.
DEFAULT_MAX_CHECK_WEIGHT: int | None = None  # v3.7: no cap (see VerificationConfig)


def structural_check_spec(
    program_id: str,
    spec: dict[str, Any],
    *,
    n_budget: int | None = None,
    total_budget: int | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    require_positive_k: bool = True,
    deformation_mode: str | None = None,
    lp_shape_pin: Any = None,
    max_check_weight: int | None = None,
    exact_check_weight: int | None = None,
) -> StructuralResult:
    try:
        compiled = build_compiled_code(
            spec,
            program_id=program_id,
            n_budget=int(n_budget or 0),
            deformation_mode=deformation_mode,
            lp_shape_pin=lp_shape_pin,
        )
    except Exception as exc:
        return StructuralResult(
            program_id=program_id,
            valid=False,
            status="build_error",
            n_budget=n_budget,
            failure_reasons=[f"build:{exc}"],
        )
    return structural_check_compiled(
        program_id,
        compiled,
        n_budget=n_budget,
        total_budget=total_budget,
        n_min=n_min,
        n_max=n_max,
        require_positive_k=require_positive_k,
        max_check_weight=max_check_weight,
        exact_check_weight=exact_check_weight,
    )


def structural_check_matrices(
    program_id: str,
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    n_budget: int | None = None,
    total_budget: int | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    require_positive_k: bool = True,
) -> StructuralResult:
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    if hx.ndim != 2 or hz.ndim != 2:
        return StructuralResult(
            program_id=program_id,
            valid=False,
            status="malformed_matrix",
            n_budget=n_budget,
            failure_reasons=["matrices_must_be_2d"],
        )
    compiled = CompiledCode(
        program_id=program_id,
        n_budget=int(n_budget or 0),
        hx=hx.astype(int).tolist(),
        hz=hz.astype(int).tolist(),
        family="matrix",
        h_mask=[0] * int(hx.shape[1]),
        code_hash=compiled_code_hash(hx, hz),
    )
    return structural_check_compiled(
        program_id,
        compiled,
        n_budget=n_budget,
        total_budget=total_budget,
        n_min=n_min,
        n_max=n_max,
        require_positive_k=require_positive_k,
    )


def structural_check_compiled(
    program_id: str,
    compiled: CompiledCode,
    *,
    n_budget: int | None = None,
    total_budget: int | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    require_positive_k: bool = True,
    max_check_weight: int | None = DEFAULT_MAX_CHECK_WEIGHT,
    exact_check_weight: int | None = None,
) -> StructuralResult:
    hx = np.asarray(compiled.hx, dtype=np.uint8) % 2
    hz = np.asarray(compiled.hz, dtype=np.uint8) % 2
    failures: list[str] = []

    if hx.ndim != 2 or hz.ndim != 2:
        return StructuralResult(
            program_id=program_id,
            valid=False,
            status="malformed_matrix",
            n_budget=n_budget,
            failure_reasons=["matrices_must_be_2d"],
        )

    n_checks_x, n_x = hx.shape
    n_checks_z, n_z = hz.shape
    if n_x != n_z:
        return StructuralResult(
            program_id=program_id,
            valid=False,
            status="shape_mismatch",
            n_budget=n_budget,
            n_checks_x=int(n_checks_x),
            n_checks_z=int(n_checks_z),
            failure_reasons=["shape_mismatch"],
        )

    n_data = int(n_x)
    rank_hx = rank2(hx)
    rank_hz = rank2(hz)
    sx, sz = symplectic_stabilizers(compiled)
    stabilizer_rank = rank2(np.hstack([sx, sz]))
    k = int(n_data - stabilizer_rank)
    ancestor_k = int(n_data - rank_hx - rank_hz)
    css_violations = int(((hx @ hz.T) % 2).sum())
    commutation_violations = int(((sx @ sz.T + sz @ sx.T) % 2).sum())
    if commutation_violations:
        failures.append(
            "invalid_css_commutation" if compiled.is_css else "invalid_stabilizer_commutation"
        )
    if k != ancestor_k:
        failures.append("local_clifford_rank_mismatch")
    if require_positive_k and k <= 0:
        failures.append("k_nonpositive")
    if n_budget is not None and n_data > int(n_budget):
        failures.append("n_budget_violation")
    # Total physical qubits = data + one measurement ancilla per check ROW
    # (matrix rows, not rank: the syndrome-extraction circuit builds one
    # ancilla per row, so BB72 costs 72 + 36 + 36 = 144 total).
    total_qubits = n_data + int(n_checks_x) + int(n_checks_z)
    if total_budget is not None and total_qubits > int(total_budget):
        failures.append("total_qubit_budget_violation")
    if n_min is not None and n_data < int(n_min):
        failures.append("n_below_min")
    if n_max is not None and n_data > int(n_max):
        failures.append("n_above_max")

    observed_check_weight = _max_row_weight(hx, hz)
    if max_check_weight is not None and observed_check_weight > int(max_check_weight):
        failures.append(
            f"max_check_weight_exceeded:{observed_check_weight}>{int(max_check_weight)}"
        )
    # Fig.4 c (2026-10-23): the exact maximum check-weight contract. The code's
    # maximum row weight over H_X and H_Z must EQUAL the target: a lighter code
    # is rejected like a heavier one, so the archive, the parents, the slow
    # queue and the final only ever hold codes of the requested weight class.
    if exact_check_weight is not None and observed_check_weight != int(exact_check_weight):
        failures.append(
            f"check_weight_mismatch:{observed_check_weight}!={int(exact_check_weight)}"
        )
    max_qubit_degree = _max_side_column_weight(hx, hz)
    connected = is_connected_tanner_graph(hx, hz)
    if not connected:
        failures.append("disconnected_tanner_graph")
    canonical_hash = _canonical_hash_or_none(hx, hz)

    return StructuralResult(
        program_id=program_id,
        valid=not failures,
        status="ok" if not failures else "structural_reject",
        n_budget=n_budget,
        n_data=n_data,
        n_checks_x=int(n_checks_x),
        n_checks_z=int(n_checks_z),
        rank_hx=rank_hx,
        rank_hz=rank_hz,
        k=k,
        css_violations=css_violations,
        commutation_violations=commutation_violations,
        stabilizer_rank=stabilizer_rank,
        is_css=compiled.is_css,
        deformation=compiled.deformation,
        max_check_weight=observed_check_weight,
        max_qubit_degree=max_qubit_degree,
        connected=connected,
        code_hash=compiled.code_hash or code_hash(hx, hz, h_mask=compiled.h_mask),
        canonical_hash=canonical_hash,
        failure_reasons=failures,
        metrics={
            "mean_check_weight_x": _mean_row_weight(hx),
            "mean_check_weight_z": _mean_row_weight(hz),
            "max_total_qubit_degree": _max_total_column_weight(hx, hz),
            "mean_total_qubit_degree": _mean_total_column_weight(hx, hz),
            "total_qubits": int(total_qubits),
            "total_budget": int(total_budget) if total_budget is not None else None,
            "ancestor_k": ancestor_k,
            "sector_sizes": list(compiled.sector_sizes),
            "deformation": compiled.deformation,
            # Permutation-invariant Tier-1 fingerprint (03 §6.3), recorded as
            # INFORMATION only: v3.6 (docs/06 §3 B3) never gates on it, since
            # 1-WL merges distinct same-shape LP/BB codes.
            "wl_hash": _wl_hash_or_none(hx, hz),
            # Exact colored-graph canonical form (pynauty); None without
            # pynauty. Together with code_hash this is the slow-loop twin gate.
            "canonical_hash": canonical_hash,
        },
    )


def code_hash(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    h_mask: list[int] | np.ndarray | None = None,
) -> str:
    return compiled_code_hash(hx, hz, h_mask=h_mask)


def _wl_hash_or_none(hx: np.ndarray, hz: np.ndarray) -> str | None:
    try:
        from omniqec_agentic.memory.fingerprints import tanner_wl_hash

        return tanner_wl_hash(hx, hz)
    except Exception:
        return None  # informational only; never gates


def _canonical_hash_or_none(hx: np.ndarray, hz: np.ndarray) -> str | None:
    try:
        from omniqec_agentic.memory.fingerprints import canonical_code_hash

        return canonical_code_hash(hx, hz)
    except Exception:
        return None  # the twin gate degrades to exact code_hash equality


def is_connected_tanner_graph(hx: np.ndarray, hz: np.ndarray) -> bool:
    n_data = hx.shape[1]
    n_checks_x, n_checks_z = hx.shape[0], hz.shape[0]
    total = n_data + n_checks_x + n_checks_z
    if total == 0:
        return False
    parent = list(range(total))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for check_index in range(n_checks_x):
        for qubit in np.nonzero(hx[check_index])[0]:
            union(n_data + check_index, int(qubit))
    for check_index in range(n_checks_z):
        for qubit in np.nonzero(hz[check_index])[0]:
            union(n_data + n_checks_x + check_index, int(qubit))
    roots = {find(index) for index in range(total)}
    return len(roots) == 1


def _max_row_weight(hx: np.ndarray, hz: np.ndarray) -> int:
    x = int(hx.sum(1).max()) if hx.shape[0] else 0
    z = int(hz.sum(1).max()) if hz.shape[0] else 0
    return max(x, z)


def _max_side_column_weight(hx: np.ndarray, hz: np.ndarray) -> int:
    x = int(hx.sum(0).max()) if hx.shape[1] else 0
    z = int(hz.sum(0).max()) if hz.shape[1] else 0
    return max(x, z)


def _max_total_column_weight(hx: np.ndarray, hz: np.ndarray) -> int:
    combined = np.vstack([hx, hz])
    return int(combined.sum(0).max()) if combined.shape[1] else 0


def _mean_row_weight(matrix: np.ndarray) -> float:
    return float(matrix.sum(1).mean()) if matrix.shape[0] else 0.0


def _mean_total_column_weight(hx: np.ndarray, hz: np.ndarray) -> float:
    combined = np.vstack([hx, hz])
    return float(combined.sum(0).mean()) if combined.shape[1] else 0.0
