"""Multi-block two-basis circuits for logical-operation experiments (Fig.3 stages 0 and 1).

The memory timeline of ``twobasis_builder`` (02 §9.4) run on several blocks of
the SAME CSS code in lock-step, with an optional transversal logical CNOT
between two rounds. Block ``b`` occupies qubits ``b*(n+mZ+mX) + [data | zanc |
xanc]``, so a single block without a CNOT emits exactly the circuit of
``build_twobasis_memory_circuit`` (tested instruction for instruction).

  S0      global prep of every block's data (R for basis Z, RX for basis X)
  round   A RX xanc | X_i CX(xanc->data) | B MX xanc | C R zanc |
          Z_j CX(data->zanc) | D M zanc                  (all blocks per tick)
  CNOT    after round ``rounds_before``: one tick of physical CX(data_c[i] ->
          data_t[i]) for every data qubit i, the same gate noise as an SEC CX
          layer (uniform_paper: DEPOLARIZE2(p) after the layer)
  final   round ``rounds_before + rounds_after`` D, then every block's data
          readout and reconstruction detectors; observables = every block's
          basis logicals, index ``b*k + row``

``EXPERIMENT_LAYOUTS`` names the product-state experiments: stage 0 ``cnot``
(two blocks, CNOT 0 -> 1) and ``fanout`` (three blocks all in the memory
basis, CNOT 0 -> 1 then 0 -> 2 as two consecutive ticks; NOT the GHZ
preparation of stage 1, which lives in ``logical_program``), each against an
``idle`` baseline with the same blocks and rounds and no CNOT.

For two blocks of one CSS code the transversal CNOT is the logical CNOT on
each logical pair (row j of the control with row j of the target: the same
``css_logicals`` rows on the same supports). Detectors across it follow the
Heisenberg map X_c -> X_c X_t, Z_t -> Z_c Z_t: the first measurement after the
CNOT of a control X check is compared with its own and the target's previous
X values, and that of a target Z check with its own and the control's previous
Z values; the other checks compare with their own previous value. A check
whose reference involves a random (never measured) value stays random.

Only the ``uniform_paper_v1`` lowering (Fig.2's noise, SI C b) is supported:
no idle noise, DEPOLARIZE1 after every reset, DEPOLARIZE2 after every CX
layer, the measurement flip before every measurement including the final data
readout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.verification.gf2 import logicals as css_logicals
from omniqec_agentic.verification.noise import NoisePoint, NoiseProfile

from .twobasis_rank import TwoBasisSchedule


def _xor(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    """Parity of two measurement-record sets; None (a random value) absorbs."""
    if a is None or b is None:
        return None
    return sorted(set(a) ^ set(b))


def build_twobasis_multiblock_circuit(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: TwoBasisSchedule,
    *,
    basis: str,
    rounds_before: int,
    rounds_after: int = 0,
    n_blocks: int = 1,
    cnots: tuple[tuple[int, int], ...] = (),
    cnot_rounds: tuple[int, ...] | None = None,
    cnot_noise_scale: float = 1.0,
    noise_point: NoisePoint,
    detector_kinds_out: list | None = None,
):
    """``n_blocks`` blocks of one code; ``cnots`` = (control, target) block
    pairs applied, in order, as transversal CNOT ticks after round
    ``rounds_before`` (empty: a plain multi-block memory of
    ``rounds_before + rounds_after`` rounds). ``cnot_rounds`` lists the rounds
    after which the CNOT ticks are applied instead (each in 1..rounds-1, e.g.
    every round); ``cnot_noise_scale`` multiplies the DEPOLARIZE2 rate of the
    transversal layers only (the SEC gates keep p)."""

    import stim

    basis = str(basis).upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    rounds_before, rounds_after = int(rounds_before), int(rounds_after)
    rounds = rounds_before + rounds_after
    if rounds_before < 1 or rounds < 1:
        raise ValueError("needs at least one round before the CNOT layer")
    if cnots and rounds_after < 1:
        raise ValueError("a CNOT needs at least one round after it")
    apply_at = set(int(r) for r in cnot_rounds) if cnot_rounds else {rounds_before}
    if cnots and not all(1 <= r <= rounds - 1 for r in apply_at):
        raise ValueError(f"CNOT rounds must lie in 1..{rounds - 1}, got {sorted(apply_at)}")
    cnot_noise_scale = float(cnot_noise_scale)
    if cnot_noise_scale < 0:
        raise ValueError("cnot_noise_scale must be >= 0")
    if noise_point is None or noise_point.profile != NoiseProfile.UNIFORM_PAPER:
        raise NotImplementedError("multi-block circuits support uniform_paper_v1 only")
    p = float(noise_point.value)
    n_blocks = int(n_blocks)
    for c, t in cnots:
        if not (0 <= c < n_blocks and 0 <= t < n_blocks and c != t):
            raise ValueError(f"bad CNOT block pair {(c, t)}")

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n, m_z, m_x = int(schedule.n_data), int(schedule.m_z), int(schedule.m_x)
    if hx.shape != (m_x, n) or hz.shape != (m_z, n):
        raise ValueError("schedule dimensions do not match hx/hz")
    stride = n + m_z + m_x
    data = [tuple(b * stride + q for q in range(n)) for b in range(n_blocks)]
    zanc = [tuple(b * stride + n + j for j in range(m_z)) for b in range(n_blocks)]
    xanc = [tuple(b * stride + n + m_z + j for j in range(m_x)) for b in range(n_blocks)]
    layers_x = schedule.layers["X"]
    layers_z = schedule.layers["Z"]
    lx_logical, lz_logical = css_logicals(hx, hz)
    logical_rows = np.asarray(lx_logical if basis == "X" else lz_logical, dtype=np.uint8)
    k = int(logical_rows.shape[0])
    n_checks = {"X": m_x, "Z": m_z}

    circuit = stim.Circuit()
    measure_count = 0
    # ref[(block, kind, check)]: the measurement records whose parity the
    # check's next outcome must equal; [] = deterministic +1, None = random.
    ref: dict[tuple[int, str, int], list[int] | None] = {}
    for b in range(n_blocks):
        for kind_ in ("X", "Z"):
            for check in range(n_checks[kind_]):
                ref[(b, kind_, check)] = [] if kind_ == basis else None
    obs_ref: dict[tuple[int, int], list[int] | None] = {
        (b, row): [] for b in range(n_blocks) for row in range(k)
    }

    def rec(index: int):
        return stim.target_rec(index - measure_count)

    def flat(groups: list[tuple[int, ...]]) -> list[int]:
        return [q for g in groups for q in g]

    def emit_reset(rtype: str, groups: list[tuple[int, ...]]) -> None:
        targets = flat(groups)
        circuit.append(rtype, targets)
        if p > 0:
            circuit.append("DEPOLARIZE1", targets, p)

    def emit_cx(pairs: list[tuple[int, int]], rate: float | None = None) -> None:
        targets = [q for pair in pairs for q in pair]
        circuit.append("CX", targets)
        rate = p if rate is None else rate
        if rate > 0:
            circuit.append("DEPOLARIZE2", targets, min(rate, 15 / 16))

    def emit_measure(mtype: str, groups: list[tuple[int, ...]]) -> list[list[int]]:
        nonlocal measure_count
        targets = flat(groups)
        if p > 0:
            circuit.append("X_ERROR" if mtype == "M" else "Z_ERROR", targets, p)
        circuit.append(mtype, targets)
        out, cursor = [], measure_count
        for g in groups:
            out.append(list(range(cursor, cursor + len(g))))
            cursor += len(g)
        measure_count += len(targets)
        return out

    def emit_cycle_detectors(kind_: str, per_block: list[list[int]]) -> None:
        for b, indices in enumerate(per_block):
            for check, midx in enumerate(indices):
                prev = ref[(b, kind_, check)]
                if prev is not None:
                    circuit.append("DETECTOR", [rec(midx)] + [rec(i) for i in prev])
                    if detector_kinds_out is not None:
                        detector_kinds_out.append(kind_)
                ref[(b, kind_, check)] = [midx]

    def apply_cnot(c: int, t: int) -> None:
        emit_cx([(data[c][q], data[t][q]) for q in range(n)], rate=p * cnot_noise_scale)
        for check in range(m_x):  # X_c -> X_c X_t
            ref[(c, "X", check)] = _xor(ref[(c, "X", check)], ref[(t, "X", check)])
        for check in range(m_z):  # Z_t -> Z_c Z_t
            ref[(t, "Z", check)] = _xor(ref[(t, "Z", check)], ref[(c, "Z", check)])
        for row in range(k):  # the tracked logicals follow the same map
            if basis == "X":
                obs_ref[(c, row)] = _xor(obs_ref[(c, row)], obs_ref[(t, row)])
            else:
                obs_ref[(t, row)] = _xor(obs_ref[(t, row)], obs_ref[(c, row)])

    emit_reset("RX" if basis == "X" else "R", data)
    circuit.append("TICK")
    for round_index in range(1, rounds + 1):
        last_round = round_index == rounds
        emit_reset("RX", xanc)
        circuit.append("TICK")
        for layer in layers_x:
            pairs = []
            for b in range(n_blocks):
                pairs += sorted((xanc[b][int(c)], data[b][int(d)]) for c, d in layer)
            emit_cx(pairs)
            circuit.append("TICK")
        emit_cycle_detectors("X", emit_measure("MX", xanc))
        circuit.append("TICK")
        emit_reset("R", zanc)
        circuit.append("TICK")
        for layer in layers_z:
            pairs = []
            for b in range(n_blocks):
                pairs += sorted((data[b][int(d)], zanc[b][int(c)]) for c, d in layer)
            emit_cx(pairs)
            circuit.append("TICK")
        emit_cycle_detectors("Z", emit_measure("M", zanc))
        if last_round:
            break
        circuit.append("TICK")
        if cnots and round_index in apply_at:
            for c, t in cnots:
                apply_cnot(c, t)
                circuit.append("TICK")

    data_idx = emit_measure("M" if basis == "Z" else "MX", data)
    checks = hx if basis == "X" else hz
    for b in range(n_blocks):
        for check in range(checks.shape[0]):
            prev = ref[(b, basis, check)]
            if prev is None:
                raise ValueError(f"block {b} {basis} check {check} has a random reference at readout")
            targets = [rec(data_idx[b][q]) for q in np.nonzero(checks[check])[0]]
            targets += [rec(i) for i in prev]
            circuit.append("DETECTOR", targets)
            if detector_kinds_out is not None:
                detector_kinds_out.append(basis)
    for b in range(n_blocks):
        for row in range(k):
            prev = obs_ref[(b, row)]
            if prev is None:
                raise ValueError(f"block {b} logical {row} is random at readout")
            targets = [rec(data_idx[b][q]) for q in np.nonzero(logical_rows[row])[0]]
            targets += [rec(i) for i in prev]
            circuit.append("OBSERVABLE_INCLUDE", targets, b * k + row)
    return circuit


def logical_pair_count(hx: np.ndarray, hz: np.ndarray) -> int:
    """k of the code (= observables per block)."""
    lx, _ = css_logicals(np.asarray(hx, dtype=np.uint8) % 2, np.asarray(hz, dtype=np.uint8) % 2)
    return int(np.asarray(lx).shape[0])


# experiment -> (blocks, transversal CNOT (control, target) ticks, in order)
EXPERIMENT_LAYOUTS: dict[str, tuple[int, tuple[tuple[int, int], ...]]] = {
    "idle": (2, ()),
    "cnot": (2, ((0, 1),)),
    "idle3": (3, ()),
    "fanout": (3, ((0, 1), (0, 2))),
}
# operation experiment -> its idle baseline
EXPERIMENT_BASELINES: dict[str, str] = {"cnot": "idle", "fanout": "idle3"}


def cnot_experiment_circuits(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: TwoBasisSchedule,
    *,
    noise_point: NoisePoint,
    rounds_before: int,
    rounds_after: int,
    experiments: tuple[str, ...] = ("idle", "cnot"),
    cnot_rounds: tuple[int, ...] | None = None,
    cnot_noise_scale: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """{experiment: {"X"|"Z": (circuit, detector_kinds)}} for the named
    ``EXPERIMENT_LAYOUTS``: blocks of the code with their transversal CNOT
    ticks (after round ``rounds_before``, or after every round in
    ``cnot_rounds``), or none for the idle baselines."""

    unknown = [e for e in experiments if e not in EXPERIMENT_LAYOUTS]
    if unknown:
        raise ValueError(f"unknown experiments {unknown}; known: {sorted(EXPERIMENT_LAYOUTS)}")
    out: dict[str, dict[str, Any]] = {}
    for name in experiments:
        blocks, cnots = EXPERIMENT_LAYOUTS[name]
        out[name] = {}
        for basis in ("X", "Z"):
            kinds: list[str] = []
            circuit = build_twobasis_multiblock_circuit(
                hx, hz, schedule, basis=basis, rounds_before=rounds_before,
                rounds_after=rounds_after, n_blocks=blocks, cnots=cnots,
                cnot_rounds=cnot_rounds if cnots else None, cnot_noise_scale=cnot_noise_scale,
                noise_point=noise_point, detector_kinds_out=kinds,
            )
            out[name][basis] = (circuit, tuple(kinds))
    return out
