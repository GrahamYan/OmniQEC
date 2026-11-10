"""Ranked schedule selection (design doc 02 §3, §5; SBB Sec. IV-V).

Residual errors: for a weight-w check whose data qubits are listed in the
circuit's CNOT order (its cycle order: LZ colors then RZ colors for a
Z-check; LX then RX for an X-check), the residual set is the w-1 CNOT-order
suffixes {sum_{s=l..w} e_{i_s} : l = 2..w} (SBB Sec. IV.A; ancilla idling
errors map onto the same set).

Three-stage ranking (02 §3.2 cost structure):
  1. GF2 exact elimination of Delta(E)=1 candidates (a residual that IS a
     logical: E in ker(H_other) \\ span(H_same)) — zero-cost rank tests;
  2. survivors ordered by ancilla idle count tau_A (mid-cycle idles under the
     compaction rule — matches the builder's noise placement exactly);
  3. the head tie group is separated by the extended-code distance d_ext
     probed with the house ISD (SBB Thm 1: for non-interleaved circuits
     d_circ = d_ext of the FULL residual set, so stage 3 ranks by estimated
     circuit distance directly and doubles as the d_circ_ext evidence of the
     winner). v1 simplification: the W residual-distance profile is deferred;
     remaining ties break by candidate index (deterministic).

All randomness is a seeded ISD stream; identical inputs give identical
rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omniqec_agentic.verification.gf2 import kernel_basis, logicals
from omniqec_agentic.verification.proxy_distance import _proxy_side

from .graph import SectorGraphs
from .schedule import LRCSchedule, build_canonical_schedule

DEFAULT_CANDIDATES = 64
DEFAULT_ISD_ITERS = 100  # SBB Appendix D uses T=100 rapid estimates
STAGE3_GROUP_CAP = 8


@dataclass(frozen=True)
class RankedSchedule:
    schedule: LRCSchedule
    catastrophic: bool
    delta_one_eliminated: int
    tau_a: int | None
    d_circ_ext_x: int | None
    d_circ_ext_z: int | None
    stage3_group: int
    metadata: dict[str, Any] = field(default_factory=dict)


def residual_suffixes(schedule: LRCSchedule) -> dict[str, list[list[np.ndarray]]]:
    """Per-check residual suffix supports as binary vectors over data qubits.

    Returns {"X": [per X-check list of vectors], "Z": [...]}; the CNOT order
    is the cycle order induced by the schedule's color permutation.
    """

    n = schedule.n_data
    order: dict[str, dict[int, list[int]]] = {"X": {}, "Z": {}}
    for kind, slots in (("X", ("LX", "RX")), ("Z", ("LZ", "RZ"))):
        for slot in slots:
            offset = 0 if slot in ("LX", "LZ") else schedule.left
            for group in schedule.layers[slot]:
                for check, local in group:
                    order[kind].setdefault(check, []).append(local + offset)
    out: dict[str, list[list[np.ndarray]]] = {"X": [], "Z": []}
    for kind, m in (("X", schedule.m_x), ("Z", schedule.m_z)):
        for check in range(m):
            qubits = order[kind].get(check, [])
            suffixes = []
            for start in range(1, len(qubits)):  # l = 2..w
                vector = np.zeros(n, dtype=np.uint8)
                vector[qubits[start:]] = 1
                suffixes.append(vector)
            out[kind].append(suffixes)
    return out


def _row_reduce(matrix: np.ndarray) -> np.ndarray:
    """GF2 row echelon basis (nonzero rows)."""

    rows = [row.copy() for row in np.asarray(matrix, dtype=np.uint8) % 2]
    basis: list[np.ndarray] = []
    pivots: list[int] = []
    for row in rows:
        for pivot, base in zip(pivots, basis):
            if row[pivot]:
                row ^= base
        nonzero = np.nonzero(row)[0]
        if len(nonzero):
            basis.append(row)
            pivots.append(int(nonzero[0]))
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, matrix.shape[1]), np.uint8)


def _reduce_against(basis: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector = vector.copy()
    for base in basis:
        pivot = int(np.nonzero(base)[0][0])
        if vector[pivot]:
            vector ^= base
    return vector


def delta_one_residuals(
    hx: np.ndarray, hz: np.ndarray, suffixes: dict[str, list[list[np.ndarray]]]
) -> list[dict[str, Any]]:
    """Exact GF2 scan: residuals that are themselves undetected logicals."""

    hits: list[dict[str, Any]] = []
    checks = {"Z": (hx, hz), "X": (hz, hx)}  # residual kind -> (detector H, stabilizer span)
    for kind, (h_detect, h_span) in checks.items():
        span_basis = _row_reduce(h_span)
        for check, residuals in enumerate(suffixes[kind]):
            for suffix_index, vector in enumerate(residuals):
                if ((h_detect @ vector) % 2).any():
                    continue  # nonzero syndrome: detected by opposite-basis checks
                if not _reduce_against(span_basis, vector).any():
                    continue  # a stabilizer, harmless
                hits.append({"kind": kind, "check": int(check), "suffix": suffix_index + 2})
    return hits


def tau_a(schedule: LRCSchedule) -> int:
    """Mid-cycle ancilla idle count per round (compaction rule, 02 §3.2)."""

    total = 0
    for kind, slots, m in (
        ("X", ("LX", "RX"), schedule.m_x),
        ("Z", ("LZ", "RZ"), schedule.m_z),
    ):
        steps: dict[int, list[int]] = {}
        for slot in slots:
            for color, group in enumerate(schedule.layers[slot]):
                base = schedule.window_step(slot, color)
                # Z-cycle RZ happens one round later than its LZ: linearize.
                step = base + (schedule.t if slot == "RZ" else 0)
                for check, _ in group:
                    steps.setdefault(check, []).append(step)
        for check in range(m):
            fired = sorted(steps.get(check, []))
            if len(fired) > 1:
                total += (fired[-1] - fired[0] + 1) - len(fired)
    return total


def extended_matrices(
    h_detect: np.ndarray, logical_rows: np.ndarray, residuals: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """SBB Def. 2: append one column per residual to BOTH H and L."""

    if not residuals:
        return h_detect, logical_rows
    res = np.stack(residuals, axis=1)
    h_cols = (h_detect @ res) % 2
    l_cols = (logical_rows @ res) % 2
    return np.hstack([h_detect, h_cols]), np.hstack([logical_rows, l_cols])


def d_ext_probe(
    hx: np.ndarray,
    hz: np.ndarray,
    suffixes: dict[str, list[list[np.ndarray]]],
    *,
    iters: int,
    seed: int,
) -> tuple[int | None, int | None]:
    """ISD upper-bound estimate of (d_ext^X, d_ext^Z) — probing converges to
    the exact circuit distance for non-interleaved circuits (SBB Thm 1)."""

    lx, lz = logicals(hx, hz)
    results: list[int | None] = []
    for h_detect, logical_rows, kind in ((hx, np.asarray(lx), "Z"), (hz, np.asarray(lz), "X")):
        flat = [v for residuals in suffixes[kind] for v in residuals]
        h_ext, l_ext = extended_matrices(h_detect % 2, logical_rows % 2, flat)
        kernel = kernel_basis(h_ext)
        if kernel.shape[0] == 0 or l_ext.shape[0] == 0:
            results.append(None)
            continue
        rng = np.random.default_rng(seed + (0 if kind == "Z" else 1))
        distance = _proxy_side(kernel, l_ext, int(iters), rng)
        results.append(int(distance) if distance < 10**9 else None)
    d_z_side, d_x_side = results  # Z residuals bound X-detected distance etc.
    return d_x_side, d_z_side


def rank_schedules(
    sectors: SectorGraphs,
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    candidates: int = DEFAULT_CANDIDATES,
    isd_iters: int = DEFAULT_ISD_ITERS,
    seed: int = 0,
) -> RankedSchedule:
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    survivors: list[tuple[int, int, LRCSchedule, dict]] = []  # (tau, idx, sched, suffixes)
    eliminated = 0
    for index in range(max(1, int(candidates))):
        schedule = build_canonical_schedule(sectors, candidate_index=index)
        suffixes = residual_suffixes(schedule)
        if delta_one_residuals(hx, hz, suffixes):
            eliminated += 1
            continue
        survivors.append((tau_a(schedule), index, schedule, suffixes))
    if not survivors:
        canonical = build_canonical_schedule(sectors, candidate_index=0)
        return RankedSchedule(
            schedule=canonical,
            catastrophic=True,
            delta_one_eliminated=eliminated,
            tau_a=tau_a(canonical),
            d_circ_ext_x=None,
            d_circ_ext_z=None,
            stage3_group=0,
            metadata={"reason": "all_candidates_have_single_fault_logical"},
        )
    survivors.sort(key=lambda item: (item[0], item[1]))
    best_tau = survivors[0][0]
    head = [item for item in survivors if item[0] == best_tau][:STAGE3_GROUP_CAP]
    scored = []
    for tau, index, schedule, suffixes in head:
        d_x, d_z = d_ext_probe(hx, hz, suffixes, iters=isd_iters, seed=seed + index)
        combined = min(v for v in (d_x, d_z) if v is not None) if (d_x or d_z) else None
        scored.append((-(combined or 0), index, schedule, d_x, d_z, tau))
    scored.sort(key=lambda item: (item[0], item[1]))
    _, index, schedule, d_x, d_z, tau = scored[0]
    return RankedSchedule(
        schedule=schedule,
        catastrophic=False,
        delta_one_eliminated=eliminated,
        tau_a=tau,
        d_circ_ext_x=d_x,
        d_circ_ext_z=d_z,
        stage3_group=len(head),
        metadata={
            "candidates": int(candidates),
            "survivors": len(survivors),
            "isd_iters": int(isd_iters),
            "seed": int(seed),
            "chosen_candidate_index": int(index),
        },
    )
