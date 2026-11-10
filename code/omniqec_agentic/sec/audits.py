"""SEC audit suite (design doc 02 §4). Only static validity gates (hard);
noiseless validity and the single-fault DEM scan are soft evidence.

All audits parse the GENERATED stim circuit (the artifact), not the
scheduling intent — a builder bug is exactly what they exist to catch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schedule import LRCSchedule
from .stim_builder import build_lrc_memory_circuit


def audit_static(
    circuit: Any,
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: LRCSchedule,
    *,
    rounds: int,
) -> dict[str, Any]:
    """Hard gate: structural validity of the emitted circuit (02 §4.1)."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n, m_z = schedule.n_data, schedule.m_z
    failures: list[str] = []
    if int((hx @ hz.T % 2).sum()) != 0:
        failures.append("css_orthogonality")

    edge_counts: dict[tuple[str, int, int], int] = {}
    segment_conflicts = 0
    prep_meas_mismatch = 0
    orientation_errors = 0
    segment_qubits: set[int] = set()

    def classify(q: int) -> str:
        if q < n:
            return "data"
        return "zanc" if q < n + m_z else "xanc"

    def walk(circ):
        nonlocal segment_conflicts, prep_meas_mismatch, orientation_errors
        nonlocal segment_qubits
        for inst in circ:
            if inst.name == "TICK":
                segment_qubits = set()
                continue
            if hasattr(inst, "body_copy"):
                for _ in range(inst.repeat_count):
                    walk(inst.body_copy())
                continue
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if inst.name == "CX":
                for control, target in zip(targets[0::2], targets[1::2]):
                    if control in segment_qubits or target in segment_qubits:
                        segment_conflicts += 1
                    segment_qubits.update((control, target))
                    kinds = (classify(control), classify(target))
                    if kinds == ("xanc", "data"):
                        check, qubit = control - n - m_z, target
                        if not hx[check, qubit]:
                            orientation_errors += 1
                        edge_counts[("X", check, qubit)] = edge_counts.get(("X", check, qubit), 0) + 1
                    elif kinds == ("data", "zanc"):
                        check, qubit = target - n, control
                        if not hz[check, qubit]:
                            orientation_errors += 1
                        edge_counts[("Z", check, qubit)] = edge_counts.get(("Z", check, qubit), 0) + 1
                    else:
                        orientation_errors += 1
            elif inst.name in ("R", "RX", "M", "MX"):
                for q in targets:
                    kind = classify(q)
                    zlike = inst.name in ("R", "M")
                    if kind == "zanc" and not zlike:
                        prep_meas_mismatch += 1
                    if kind == "xanc" and zlike:
                        prep_meas_mismatch += 1

    walk(circuit)
    if segment_conflicts:
        failures.append(f"two_qubit_conflicts:{segment_conflicts}")
    if orientation_errors:
        failures.append(f"gate_orientation_errors:{orientation_errors}")
    if prep_meas_mismatch:
        failures.append(f"prep_meas_basis_mismatch:{prep_meas_mismatch}")
    # Every Tanner edge fires exactly once per completed cycle: R cycles each.
    expected_x = {("X", c, q): rounds for c in range(hx.shape[0]) for q in np.nonzero(hx[c])[0]}
    expected_z = {("Z", c, q): rounds for c in range(hz.shape[0]) for q in np.nonzero(hz[c])[0]}
    if edge_counts != {**expected_x, **expected_z}:
        failures.append("tanner_edge_cycle_count_mismatch")
    return {"audit": "static", "valid": not failures, "failures": failures}


def audit_noiseless(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: LRCSchedule,
    *,
    rounds: int,
) -> dict[str, Any]:
    """Soft (de facto blocking): p=0 determinism + count formulas (02 §4.2)."""

    failures: list[str] = []
    for basis, m_basis in (("X", schedule.m_x), ("Z", schedule.m_z)):
        try:
            circuit = build_lrc_memory_circuit(
                hx, hz, schedule, basis=basis, rounds=rounds, p=0
            )
        except Exception as exc:
            failures.append(f"{basis}:build_error:{exc}")
            continue
        expected = (rounds + 1) * m_basis
        if circuit.num_detectors != expected:
            failures.append(f"{basis}:detector_count:{circuit.num_detectors}!={expected}")
        try:
            samples = circuit.compile_detector_sampler().sample(8, append_observables=True)
            if samples.any():
                failures.append(f"{basis}:noiseless_events_fire")
        except Exception as exc:
            failures.append(f"{basis}:sampler_error:{exc}")
        try:
            circuit_noisy = build_lrc_memory_circuit(
                hx, hz, schedule, basis=basis, rounds=rounds, p=1e-3
            )
            circuit_noisy.detector_error_model(decompose_errors=False)
        except Exception as exc:
            failures.append(f"{basis}:dem_error:{exc}")
    return {"audit": "noiseless", "valid": not failures, "failures": failures}


def audit_single_fault(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: LRCSchedule,
    *,
    rounds: int,
) -> dict[str, Any]:
    """Soft: saturating-noise DEM scan for single faults that flip an
    observable without any detector (02 §4.3). Cross-validated against the
    Delta=1 GF2 path by the caller."""

    catastrophic: list[dict[str, Any]] = []
    for basis in ("X", "Z"):
        circuit = build_lrc_memory_circuit(
            hx, hz, schedule, basis=basis, rounds=rounds, p=1e-4, saturate_noise=True
        )
        dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            has_detector = False
            observables = []
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    has_detector = True
                elif target.is_logical_observable_id():
                    observables.append(target.val)
            if observables and not has_detector:
                catastrophic.append({"basis": basis, "observables": observables})
    return {
        "audit": "single_fault",
        "valid": not catastrophic,
        "catastrophic_count": len(catastrophic),
        "examples": catastrophic[:8],
    }


def run_all_audits(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: LRCSchedule,
    *,
    rounds: int,
) -> dict[str, Any]:
    circuit = build_lrc_memory_circuit(hx, hz, schedule, basis="X", rounds=rounds, p=1e-3)
    static = audit_static(circuit, hx, hz, schedule, rounds=rounds)
    noiseless = audit_noiseless(hx, hz, schedule, rounds=rounds)
    single_fault = audit_single_fault(hx, hz, schedule, rounds=rounds)
    return {
        "static": static,
        "noiseless": noiseless,
        "single_fault": single_fault,
        "hard_valid": static["valid"],
        "synth_error": not noiseless["valid"],
    }


def audit_twobasis_static(
    circuit: Any,
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: Any,
    *,
    rounds: int,
) -> dict[str, Any]:
    """Hard gate for the two-basis non-interleaved circuit (02 §9.4): parses
    the EMITTED uniform circuit — matchings per tick, gate orientation, prep/
    measure bases, w CX per check per round, layer count, detector /
    observable counts, and deterministic noiseless detectors (DEM builds)."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    rounds = int(rounds)
    n, m_z, m_x = int(schedule.n_data), int(schedule.m_z), int(schedule.m_x)
    failures: list[str] = []
    if int((hx @ hz.T % 2).sum()) != 0:
        failures.append("css_orthogonality")

    edge_counts: dict[tuple[str, int, int], int] = {}
    segment_conflicts = 0
    prep_meas_mismatch = 0
    orientation_errors = 0
    cx_ticks = 0
    segment_qubits: set[int] = set()
    segment_has_cx = False

    def classify(q: int) -> str:
        if q < n:
            return "data"
        return "zanc" if q < n + m_z else "xanc"

    def walk(circ):
        nonlocal segment_conflicts, prep_meas_mismatch, orientation_errors
        nonlocal segment_qubits, cx_ticks, segment_has_cx
        for inst in circ:
            if inst.name == "TICK":
                segment_qubits = set()
                segment_has_cx = False
                continue
            if hasattr(inst, "body_copy"):
                for _ in range(inst.repeat_count):
                    walk(inst.body_copy())
                continue
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if inst.name == "CX":
                if not segment_has_cx:
                    cx_ticks += 1
                    segment_has_cx = True
                for control, target in zip(targets[0::2], targets[1::2]):
                    if control in segment_qubits or target in segment_qubits:
                        segment_conflicts += 1
                    segment_qubits.update((control, target))
                    kinds = (classify(control), classify(target))
                    if kinds == ("xanc", "data"):
                        check, qubit = control - n - m_z, target
                        if not hx[check, qubit]:
                            orientation_errors += 1
                        edge_counts[("X", check, qubit)] = edge_counts.get(("X", check, qubit), 0) + 1
                    elif kinds == ("data", "zanc"):
                        check, qubit = target - n, control
                        if not hz[check, qubit]:
                            orientation_errors += 1
                        edge_counts[("Z", check, qubit)] = edge_counts.get(("Z", check, qubit), 0) + 1
                    else:  # incl. ANY CX touching an X-ancilla and a Z-ancilla
                        orientation_errors += 1
            elif inst.name in ("R", "RX", "M", "MX"):
                for q in targets:
                    kind = classify(q)
                    zlike = inst.name in ("R", "M")
                    if kind == "zanc" and not zlike:
                        prep_meas_mismatch += 1
                    if kind == "xanc" and zlike:
                        prep_meas_mismatch += 1

    walk(circuit)
    if segment_conflicts:
        failures.append(f"two_qubit_conflicts:{segment_conflicts}")
    if orientation_errors:
        failures.append(f"gate_orientation_errors:{orientation_errors}")
    if prep_meas_mismatch:
        failures.append(f"prep_meas_basis_mismatch:{prep_meas_mismatch}")
    expected_x = {("X", c, int(q)): rounds for c in range(hx.shape[0]) for q in np.nonzero(hx[c])[0]}
    expected_z = {("Z", c, int(q)): rounds for c in range(hz.shape[0]) for q in np.nonzero(hz[c])[0]}
    if edge_counts != {**expected_x, **expected_z}:
        failures.append("tanner_edge_cycle_count_mismatch")
    layers_x, layers_z = schedule.layers["X"], schedule.layers["Z"]
    if cx_ticks != rounds * (len(layers_x) + len(layers_z)):
        failures.append("layer_count_mismatch")

    basis = None
    for inst in circuit:
        if inst.name in ("M", "MX"):
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if targets and all(q < n for q in targets):
                basis = "Z" if inst.name == "M" else "X"
    if basis is None:
        failures.append("no_data_readout")
    else:
        m_w, m_opp = (m_x, m_z) if basis == "X" else (m_z, m_x)
        expected = rounds * m_w + (rounds - 1) * m_opp + m_w
        got = int(circuit.num_detectors)
        if got != expected:
            failures.append(f"detector_count:{got}!={expected}")
        from omniqec_agentic.verification.gf2 import logicals as _logicals

        lx, lz = _logicals(hx, hz)
        k = int(np.asarray(lx if basis == "X" else lz).shape[0])
        if int(circuit.num_observables) != k:
            failures.append(f"observable_count:{int(circuit.num_observables)}!={k}")
    try:
        circuit.detector_error_model(decompose_errors=False)
    except Exception as exc:
        failures.append(f"nondeterministic_detectors:{exc}")
    return {"audit": "twobasis_static", "valid": not failures, "failures": failures}
