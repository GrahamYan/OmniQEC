"""Multi-block logical programs for Fig.3 stages 2 and 3.

The lock-step two-basis timeline of ``multiblock_builder`` generalised in
three ways, each needed by one experiment:

  * every block carries its OWN code (``hx``, ``hz``, SEC schedule): stage 3
    connects a control code A to a different target code B. Transversal CNOT
    c -> t (data qubit i to data qubit i, equal n) is a logical CNOT when
    S_X(c) <= S_X(t), S_Z(t) <= S_Z(c) and the logicals map injectively;
    every map below is solved over GF(2) and the build fails otherwise;
  * every block has its own preparation basis, readout basis and last round:
    stage 2 teleports block 0 through a Bell pair (blocks 1, 2) and reads
    blocks 0 and 1 out mid-circuit;
  * observables are parities of logical readouts: a block's output logical
    that is random by itself (teleportation) is combined with the readouts of
    the non-output blocks that cancel its randomness (the Pauli frame), found
    by solving over GF(2).

Values are tracked as sets of tokens (measurement record indices and random
symbols) whose parity equals the noiseless value of a check or logical: a
check of the non-prepared kind starts as a fresh symbol and becomes its own
record at its first measurement; a logical of the non-prepared kind keeps its
symbol until a readout parity cancels it. A detector compares a measurement
with a symbol-free reference; a reference with a symbol gives no detector.

Blocks run in lock-step: each SEC layer index is one tick for every active
block that has that layer (uniform_paper has no idle noise, so a shorter
schedule simply waits). With one code, a common prep/readout basis, all
blocks to the end and every block an output, the emission is the
``build_twobasis_multiblock_circuit`` circuit instruction for instruction
(tested), so stages 0-1 and 2-3 share one measurement stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omniqec_agentic.verification.gf2 import logicals as css_logicals
from omniqec_agentic.verification.noise import NoisePoint, NoiseProfile

from .twobasis_rank import TwoBasisSchedule

Token = Any  # int (a measurement record) or tuple (a random symbol)


@dataclass(frozen=True)
class ProgramBlock:
    """One block: its code, SEC schedule, data prep basis, destructive
    readout basis and the round after whose CNOT ticks it is read out
    (None = after the program's last round)."""

    hx: Any
    hz: Any
    schedule: TwoBasisSchedule
    prep: str = "Z"
    readout: str = "Z"
    last_round: int | None = None


def _symbols(value: frozenset) -> frozenset:
    return frozenset(t for t in value if not isinstance(t, int))


def _records(value: frozenset) -> list[int]:
    return sorted(t for t in value if isinstance(t, int))


def row_combination(basis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Coefficients C (len(targets) x len(basis)) with C @ basis = targets over
    GF(2). A target equal to a basis row uses that row alone (the row of the
    same index first), so identical codes map row i to row i. Raises
    ValueError when a target is outside the row space."""

    basis = np.asarray(basis, dtype=np.uint8) % 2
    targets = np.asarray(targets, dtype=np.uint8) % 2
    m, n = basis.shape
    out = np.zeros((targets.shape[0], m), dtype=np.uint8)
    red, comb, pivots = basis.copy(), np.eye(m, dtype=np.uint8), []
    r = 0
    for col in range(n):
        if r == m:
            break
        rows = np.nonzero(red[r:, col])[0]
        if not len(rows):
            continue
        p = r + int(rows[0])
        red[[r, p]] = red[[p, r]]
        comb[[r, p]] = comb[[p, r]]
        hit = np.nonzero(red[:, col])[0]
        hit = hit[hit != r]
        red[hit] ^= red[r]
        comb[hit] ^= comb[r]
        pivots.append(col)
        r += 1
    for i, vec in enumerate(targets):
        same = np.nonzero((basis == vec).all(axis=1))[0] if m else []
        if len(same):
            out[i, i if i in set(int(s) for s in same) else int(same[0])] = 1
            continue
        resid, coeff = vec.copy(), np.zeros(m, dtype=np.uint8)
        for j, col in enumerate(pivots):
            if resid[col]:
                resid ^= red[j]
                coeff ^= comb[j]
        if resid.any():
            raise ValueError("operator outside the row space")
        out[i] = coeff
    return out


def cnot_maps(hx_c, hz_c, hx_t, hz_t, *, standard: bool = True) -> dict[str, np.ndarray]:
    """GF(2) maps that make transversal CNOT c -> t a logical CNOT, or
    ValueError: X checks of c in X checks of t, Z checks of t in Z checks of
    c, X logicals of c in (X logicals | X checks) of t, Z logicals of t in
    (Z logicals | Z checks) of c, with both logical maps of full rank.
    ``standard`` further requires the induced map to be the identity on the
    logical index (row j of c with row j of t, in the codes' ``css_logicals``
    representatives): the plain CNOT^(x)k, no logical basis change."""

    hx_c, hz_c, hx_t, hz_t = (np.asarray(m, dtype=np.uint8) % 2 for m in (hx_c, hz_c, hx_t, hz_t))
    if hx_c.shape[1] != hx_t.shape[1]:
        raise ValueError("transversal CNOT needs equal block lengths")
    lx_c, lz_c = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx_c, hz_c))
    lx_t, lz_t = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx_t, hz_t))
    try:
        mx = row_combination(hx_t, hx_c)
        nz = row_combination(hz_c, hz_t)
        lx = row_combination(np.vstack([lx_t, hx_t]), lx_c)
        lz = row_combination(np.vstack([lz_c, hz_c]), lz_t)
    except ValueError as exc:
        raise ValueError(f"codes are not CNOT-compatible: {exc}") from None
    from omniqec_agentic.verification.gf2 import rank2

    k_c, k_t = lx_c.shape[0], lx_t.shape[0]
    if rank2(lx[:, :k_t]) != k_c or rank2(lz[:, :k_c]) != k_t:
        raise ValueError("codes are not CNOT-compatible: a logical maps into the stabilizer")
    if standard and not (k_c == k_t and (lx[:, :k_t] == np.eye(k_c, dtype=np.uint8)).all()
                         and (lz[:, :k_c] == np.eye(k_t, dtype=np.uint8)).all()):
        raise ValueError("transversal CNOT is a logical CNOT only up to a logical basis change (not standard)")
    return {"mx": mx, "nz": nz, "lx": lx, "lz": lz}


def build_logical_program_circuit(
    blocks: list[ProgramBlock] | tuple[ProgramBlock, ...],
    *,
    rounds: int,
    cnots: tuple[tuple[int, int, int], ...] = (),
    outputs: tuple[int, ...] | None = None,
    noise_point: NoisePoint,
    cnot_noise_scale: float = 1.0,
    detector_kinds_out: list | None = None,
):
    """``cnots`` = (after_round, control, target) transversal CNOT ticks, in
    order; ``outputs`` = blocks whose readout logicals are the observables
    (default every block), observable index = position * k + row, each made
    deterministic with the other blocks' readouts where needed."""

    import stim

    blocks = list(blocks)
    nb = len(blocks)
    rounds = int(rounds)
    if nb < 1 or rounds < 1:
        raise ValueError("needs at least one block and one round")
    if noise_point is None or noise_point.profile != NoiseProfile.UNIFORM_PAPER:
        raise NotImplementedError("logical programs support uniform_paper_v1 only")
    p = float(noise_point.value)
    cnot_noise_scale = float(cnot_noise_scale)
    if cnot_noise_scale < 0:
        raise ValueError("cnot_noise_scale must be >= 0")
    hx = [np.asarray(b.hx, dtype=np.uint8) % 2 for b in blocks]
    hz = [np.asarray(b.hz, dtype=np.uint8) % 2 for b in blocks]
    prep = [str(b.prep).upper() for b in blocks]
    readout = [str(b.readout).upper() for b in blocks]
    last = [rounds if b.last_round is None else int(b.last_round) for b in blocks]
    sizes = []
    for i, b in enumerate(blocks):
        s = b.schedule
        n, m_z, m_x = int(s.n_data), int(s.m_z), int(s.m_x)
        if hx[i].shape != (m_x, n) or hz[i].shape != (m_z, n):
            raise ValueError(f"block {i}: schedule dimensions do not match hx/hz")
        if prep[i] not in ("X", "Z") or readout[i] not in ("X", "Z"):
            raise ValueError(f"block {i}: prep/readout must be 'X' or 'Z'")
        if not 1 <= last[i] <= rounds:
            raise ValueError(f"block {i}: last_round must lie in 1..{rounds}")
        sizes.append((n, m_z, m_x))
    outputs = tuple(range(nb)) if outputs is None else tuple(int(o) for o in outputs)
    maps: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for r, c, t in cnots:
        if not (0 <= c < nb and 0 <= t < nb and c != t):
            raise ValueError(f"bad CNOT block pair {(c, t)}")
        if not 1 <= int(r) <= rounds - 1:
            raise ValueError(f"CNOT rounds must lie in 1..{rounds - 1}, got {r}")
        if last[c] < r or last[t] < r:
            raise ValueError(f"CNOT {(c, t)} after round {r} touches a block already read out")
        if (c, t) not in maps:
            maps[(c, t)] = cnot_maps(hx[c], hz[c], hx[t], hz[t])

    offsets, cursor = [], 0
    for n, m_z, m_x in sizes:
        offsets.append(cursor)
        cursor += n + m_z + m_x
    data = [tuple(offsets[b] + q for q in range(sizes[b][0])) for b in range(nb)]
    zanc = [tuple(offsets[b] + sizes[b][0] + j for j in range(sizes[b][1])) for b in range(nb)]
    xanc = [tuple(offsets[b] + sizes[b][0] + sizes[b][1] + j for j in range(sizes[b][2])) for b in range(nb)]
    logical = []
    for b in range(nb):
        lx, lz = css_logicals(hx[b], hz[b])
        logical.append({"X": np.asarray(lx, dtype=np.uint8), "Z": np.asarray(lz, dtype=np.uint8)})
    k = [int(logical[b]["X"].shape[0]) for b in range(nb)]
    checks = [{"X": hx[b], "Z": hz[b]} for b in range(nb)]

    circuit = stim.Circuit()
    measure_count = 0
    cref: dict[tuple[int, str, int], frozenset] = {}
    lref: dict[tuple[int, str, int], frozenset] = {}
    for b in range(nb):
        for kind in ("X", "Z"):
            for j in range(checks[b][kind].shape[0]):
                cref[(b, kind, j)] = frozenset() if kind == prep[b] else frozenset({("c", b, kind, j)})
            for row in range(k[b]):
                lref[(b, kind, row)] = frozenset() if kind == prep[b] else frozenset({("l", b, kind, row)})
    readout_value: dict[tuple[int, int], tuple[list[int], frozenset]] = {}

    def rec(index: int):
        return stim.target_rec(index - measure_count)

    def flat(groups):
        return [q for g in groups for q in g]

    def emit_reset(rtype: str, groups) -> None:
        targets = flat(groups)
        circuit.append(rtype, targets)
        if p > 0:
            circuit.append("DEPOLARIZE1", targets, p)

    def emit_cx(pairs, rate: float | None = None) -> None:
        targets = [q for pair in pairs for q in pair]
        circuit.append("CX", targets)
        rate = p if rate is None else rate
        if rate > 0:
            circuit.append("DEPOLARIZE2", targets, min(rate, 15 / 16))

    def emit_measure(mtype: str, groups) -> list[list[int]]:
        nonlocal measure_count
        targets = flat(groups)
        if p > 0:
            circuit.append("X_ERROR" if mtype == "M" else "Z_ERROR", targets, p)
        circuit.append(mtype, targets)
        out, cur = [], measure_count
        for g in groups:
            out.append(list(range(cur, cur + len(g))))
            cur += len(g)
        measure_count += len(targets)
        return out

    def emit_cycle_detectors(kind: str, act: list[int], per_block: list[list[int]]) -> None:
        for b, indices in zip(act, per_block):
            for j, midx in enumerate(indices):
                prev = cref[(b, kind, j)]
                if not _symbols(prev):
                    circuit.append("DETECTOR", [rec(midx)] + [rec(i) for i in _records(prev)])
                    if detector_kinds_out is not None:
                        detector_kinds_out.append(kind)
                cref[(b, kind, j)] = frozenset({midx})

    def xor_rows(coeff_row: np.ndarray, values: list[frozenset]) -> frozenset:
        out = frozenset()
        for j in np.nonzero(coeff_row)[0]:
            out = out ^ values[int(j)]
        return out

    def apply_cnot(c: int, t: int) -> None:
        mp = maps[(c, t)]
        emit_cx([(data[c][q], data[t][q]) for q in range(sizes[c][0])], rate=p * cnot_noise_scale)
        tx = [cref[(t, "X", j)] for j in range(checks[t]["X"].shape[0])]
        cz = [cref[(c, "Z", j)] for j in range(checks[c]["Z"].shape[0])]
        tlx = [lref[(t, "X", r)] for r in range(k[t])] + tx
        clz = [lref[(c, "Z", r)] for r in range(k[c])] + cz
        new_c = {}
        for i in range(checks[c]["X"].shape[0]):  # X_c -> X_c X_t
            new_c[(c, "X", i)] = cref[(c, "X", i)] ^ xor_rows(mp["mx"][i], tx)
        for j in range(checks[t]["Z"].shape[0]):  # Z_t -> Z_c Z_t
            new_c[(t, "Z", j)] = cref[(t, "Z", j)] ^ xor_rows(mp["nz"][j], cz)
        new_l = {}
        for r in range(k[c]):
            new_l[(c, "X", r)] = lref[(c, "X", r)] ^ xor_rows(mp["lx"][r], tlx)
        for r in range(k[t]):
            new_l[(t, "Z", r)] = lref[(t, "Z", r)] ^ xor_rows(mp["lz"][r], clz)
        cref.update(new_c)
        lref.update(new_l)

    def read_out(ending: list[int]) -> None:
        for basis, mtype in (("Z", "M"), ("X", "MX")):
            grp = [b for b in ending if readout[b] == basis]
            if not grp:
                continue
            idx = emit_measure(mtype, [data[b] for b in grp])
            for b, ids in zip(grp, idx):
                for j in range(checks[b][basis].shape[0]):
                    prev = cref[(b, basis, j)]
                    if _symbols(prev):
                        raise ValueError(f"block {b} {basis} check {j} has a random reference at readout")
                    targets = [rec(ids[q]) for q in np.nonzero(checks[b][basis][j])[0]]
                    targets += [rec(i) for i in _records(prev)]
                    circuit.append("DETECTOR", targets)
                    if detector_kinds_out is not None:
                        detector_kinds_out.append(basis)
            for b, ids in zip(grp, idx):
                for row in range(k[b]):
                    support = [ids[q] for q in np.nonzero(logical[b][basis][row])[0]]
                    readout_value[(b, row)] = (support, frozenset(support) ^ lref[(b, basis, row)])

    for rtype, basis in (("R", "Z"), ("RX", "X")):
        grp = [data[b] for b in range(nb) if prep[b] == basis]
        if grp:
            emit_reset(rtype, grp)
    circuit.append("TICK")
    active = list(range(nb))
    for round_index in range(1, rounds + 1):
        last_round = round_index == rounds
        emit_reset("RX", [xanc[b] for b in active])
        circuit.append("TICK")
        for layer_i in range(max(len(blocks[b].schedule.layers["X"]) for b in active)):
            pairs = []
            for b in active:
                layers = blocks[b].schedule.layers["X"]
                if layer_i < len(layers):
                    pairs += sorted((xanc[b][int(c)], data[b][int(d)]) for c, d in layers[layer_i])
            emit_cx(pairs)
            circuit.append("TICK")
        emit_cycle_detectors("X", active, emit_measure("MX", [xanc[b] for b in active]))
        circuit.append("TICK")
        emit_reset("R", [zanc[b] for b in active])
        circuit.append("TICK")
        for layer_i in range(max(len(blocks[b].schedule.layers["Z"]) for b in active)):
            pairs = []
            for b in active:
                layers = blocks[b].schedule.layers["Z"]
                if layer_i < len(layers):
                    pairs += sorted((data[b][int(d)], zanc[b][int(c)]) for c, d in layers[layer_i])
            emit_cx(pairs)
            circuit.append("TICK")
        emit_cycle_detectors("Z", active, emit_measure("M", [zanc[b] for b in active]))
        if last_round:
            break
        circuit.append("TICK")
        for r, c, t in cnots:
            if int(r) == round_index:
                apply_cnot(c, t)
                circuit.append("TICK")
        ending = [b for b in active if last[b] == round_index]
        if ending:
            read_out(ending)
            circuit.append("TICK")
            active = [b for b in active if b not in ending]
        if not active:
            break
    if active:
        read_out(active)

    frames = [key for key in sorted(readout_value) if key[0] not in outputs]
    symbol_ids = sorted({s for key in readout_value for s in _symbols(readout_value[key][1])}, key=repr)
    col = {s: i for i, s in enumerate(symbol_ids)}

    def symbol_vector(value: frozenset) -> np.ndarray:
        v = np.zeros(len(symbol_ids), dtype=np.uint8)
        for s in _symbols(value):
            v[col[s]] = 1
        return v

    index = 0
    for o in outputs:
        for row in range(k[o]):
            if (o, row) not in readout_value:
                raise ValueError(f"output block {o} was never read out")
            support, value = readout_value[(o, row)]
            if _symbols(value):
                if not frames:
                    raise ValueError(f"block {o} logical {row} is random at readout")
                fmat = np.array([symbol_vector(readout_value[f][1]) for f in frames], dtype=np.uint8)
                try:
                    coeff = row_combination(fmat, symbol_vector(value)[None, :])[0]
                except ValueError:
                    raise ValueError(f"block {o} logical {row} is random at readout") from None
                for j in np.nonzero(coeff)[0]:
                    value = value ^ readout_value[frames[int(j)]][1]
                if _symbols(value):
                    raise ValueError(f"block {o} logical {row}: frame left a random symbol")
            rest = [i for i in _records(value) if i not in set(support)]
            targets = [rec(i) for i in support] + [rec(i) for i in rest]
            circuit.append("OBSERVABLE_INCLUDE", targets, index)
            index += 1
    return circuit


def teleport_blocks(hx, hz, schedule, *, basis: str, rounds_before: int) -> list[ProgramBlock]:
    """Stage 2 layout: block 0 holds the input (prepared and finally checked in
    ``basis``); blocks 1 (|+>) and 2 (|0>) become a Bell pair with CNOT 1 -> 2
    after round ``rounds_before``; one QEC round later CNOT 0 -> 1 and the Bell
    measurement (block 0 in X, block 1 in Z) follow at once; block 2 runs on
    and is read out in ``basis`` with the Pauli frame of the Bell outcomes."""

    return [
        ProgramBlock(hx, hz, schedule, prep=basis, readout="X", last_round=rounds_before + 1),
        ProgramBlock(hx, hz, schedule, prep="X", readout="Z", last_round=rounds_before + 1),
        ProgramBlock(hx, hz, schedule, prep="Z", readout=basis),
    ]


def ghz_blocks(hx, hz, schedule, *, basis: str) -> list[ProgramBlock]:
    """Stage 1 layout: |+>_0 |0>_1 |0>_2, every block read out in ``basis``.
    With the fan-out CNOT 0 -> 1, QEC round, CNOT 0 -> 2 the state is the
    logical GHZ state; its stabilizers are the observables: basis Z reads
    Z_0 Z_1 and Z_0 Z_2 (outputs 1 and 2, block 0 as the frame), basis X
    reads X_0 X_1 X_2 (output 0, blocks 1 and 2 as the frame)."""

    return [
        ProgramBlock(hx, hz, schedule, prep="X", readout=basis),
        ProgramBlock(hx, hz, schedule, prep="Z", readout=basis),
        ProgramBlock(hx, hz, schedule, prep="Z", readout=basis),
    ]


def program_experiment_circuits(
    codes: list[tuple[Any, Any, TwoBasisSchedule]],
    *,
    noise_point: NoisePoint,
    rounds_before: int,
    rounds_after: int,
    experiments: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """{experiment: {"X"|"Z": (circuit, detector_kinds)}}. ``rb``/``ra`` =
    rounds before the first / after the last operation; the two-operation
    programs (ghz, teleport) have one QEC round between their operations and
    their baselines the same total length:

      ghzidle   |+>|0>|0> blocks of codes[0], rb + 1 + ra rounds, no gate
      ghz       the same with CNOT 0 -> 1 after round rb, CNOT 0 -> 2 after rb + 1
      idle1     one block of codes[0], rb + 1 + ra rounds
      teleport  three blocks of codes[0] (``teleport_blocks``), output = block 2
      hidle     codes[0] and codes[1] side by side, rb + ra rounds, no gate
      hcnot     the same with the transversal CNOT codes[0] -> codes[1] after rb
    """

    rb, ra = int(rounds_before), int(rounds_after)
    out: dict[str, dict[str, Any]] = {}
    for name in experiments:
        out[name] = {}
        for basis in ("X", "Z"):
            kinds: list[str] = []
            rounds = rb + 1 + ra
            if name in ("ghz", "ghzidle"):
                hx, hz, sch = codes[0]
                blocks = ghz_blocks(hx, hz, sch, basis=basis)
                cnots = ((rb, 0, 1), (rb + 1, 0, 2)) if name == "ghz" else ()
                outputs = (1, 2) if basis == "Z" else (0,)
            elif name == "idle1":
                hx, hz, sch = codes[0]
                blocks, cnots, outputs = [ProgramBlock(hx, hz, sch, prep=basis, readout=basis)], (), None
            elif name == "teleport":
                hx, hz, sch = codes[0]
                blocks = teleport_blocks(hx, hz, sch, basis=basis, rounds_before=rb)
                cnots = ((rb, 1, 2), (rb + 1, 0, 1))
                outputs = (2,)
            elif name in ("hidle", "hcnot"):
                rounds = rb + ra
                if len(codes) < 2:
                    raise ValueError(f"{name} needs a control code and a target code")
                blocks = [ProgramBlock(h_x, h_z, s, prep=basis, readout=basis) for h_x, h_z, s in codes[:2]]
                cnots = ((rb, 0, 1),) if name == "hcnot" else ()
                outputs = None
            else:
                raise ValueError(f"unknown program experiment {name!r}")
            circuit = build_logical_program_circuit(
                blocks, rounds=rounds, cnots=cnots, outputs=outputs,
                noise_point=noise_point, detector_kinds_out=kinds,
            )
            out[name][basis] = (circuit, tuple(kinds))
    return out
