"""Fig.3 Phase B (2026-10-12): the slow loop's task circuit.

``VerificationConfig.task == "cnot_dense"`` replaces the Fig.2 memory
experiment of a candidate by the gate-dense two-block circuit of the Fig.3
pilot (sec/multiblock_builder.py): two blocks of the same code, the SAME
T rounds of the SAME certified two-basis schedule (T from the rounds policy:
max(12, d_search) under max_Tc_d, 12 for every code under fixed_Tc), and
one transversal logical CNOT (block 0 -> block 1) after every round but the
last, both blocks decoded jointly. The X-basis and Z-basis circuits keep the
memory experiment's preparation and read-out; the detector rule across a
CNOT is the pilot's (X_c -> X_c X_t, Z_t -> Z_c Z_t).

The pair returned here is a ``MemoryExperimentPair`` whose two experiments
carry the task circuits, their DEMs, the per-detector kinds and
``num_logicals = 2k`` (the circuit's logical qubits), so everything
downstream -- the pickled bundle, the telescoping artifacts, the decode
shards and the A4/A5 combination 1 - (1 - block)^(1/(2k T)) -- runs
unchanged and reports the logical error per logical qubit per round of the
task circuit (Ruiz et al. 2025 / Fig.2 convention, user decision 2026-10-12).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

TASK_EXPERIMENT = "cnot"


def cnot_rounds_for(rounds: int) -> tuple[int, ...]:
    """One transversal CNOT after every round but the last: T - 1 gates."""

    rounds = int(rounds)
    if rounds < 2:
        raise ValueError("the cnot_dense task needs at least 2 QEC rounds")
    return tuple(range(1, rounds))


def task_experiment_pair(pair: Any, hx: np.ndarray, hz: np.ndarray, *, noise_point: Any) -> Any:
    """The gate-dense two-block pair built on ``pair``'s schedule and rounds."""

    from omniqec_agentic.sec.multiblock_builder import cnot_experiment_circuits

    rounds = int(pair.x_memory.rounds)
    if int(pair.z_memory.rounds) != rounds:
        raise ValueError("the memory pair's X and Z rounds differ")
    gates = cnot_rounds_for(rounds)
    before = rounds // 2
    circuits = cnot_experiment_circuits(
        np.asarray(hx, dtype=np.uint8), np.asarray(hz, dtype=np.uint8), pair.x_memory.schedule,
        noise_point=noise_point, rounds_before=before, rounds_after=rounds - before,
        experiments=(TASK_EXPERIMENT,), cnot_rounds=gates,
    )[TASK_EXPERIMENT]
    experiments = {}
    for basis, memory in (("X", pair.x_memory), ("Z", pair.z_memory)):
        circuit, kinds = circuits[basis]
        if len(kinds) != int(circuit.num_detectors):
            raise RuntimeError("task emitter recorded a detector kind count != num_detectors")
        dem = circuit.detector_error_model(decompose_errors=False)
        expected = 2 * int(memory.num_logicals)
        if int(dem.num_observables) != expected:
            raise RuntimeError(
                f"task circuit ({basis}) reports {dem.num_observables} observables, expected {expected}"
            )
        experiments[basis] = dataclasses.replace(
            memory,
            circuit=circuit,
            detector_error_model=dem,
            num_data_qubits=2 * int(memory.num_data_qubits),
            num_checks=2 * int(memory.num_checks),
            num_logicals=int(dem.num_observables),
            detector_kinds=tuple(kinds),
        )
    return dataclasses.replace(pair, x_memory=experiments["X"], z_memory=experiments["Z"])


def task_metadata(task: str, rounds: int) -> dict[str, Any]:
    """What the build result records about the task (audit trail)."""

    if str(task) == "cnot_dense":
        return {
            "task": "cnot_dense",
            "blocks": 2,
            "rounds": int(rounds),
            "cnot_rounds": list(cnot_rounds_for(rounds)),
            "per_logical_convention": "1-(1-block)^(1/(2k*T)) over the circuit's 2k logical qubits",
        }
    return {"task": "memory", "blocks": 1, "rounds": int(rounds)}
