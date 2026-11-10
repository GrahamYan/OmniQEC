"""LRC memory-experiment builder: LRCSchedule x lowering x basis -> stim.Circuit.

Every structural rule below was pinned segment-by-segment against the SBB
Zenodo golden circuits (HGP625_LRC, record 10.5281/zenodo.18853601):

  qubit layout   data 0..n-1 | Z-ancillas n..n+mZ-1 | X-ancillas n+mZ..
  gate direction X-check: CX ancilla->data; Z-check: CX data->ancilla
  circuit shape  [global prep] [|C(LZ)| prologue steps: first Z cycle's LZ]
                 [round 1: steps 2..t] [rounds 2..R: steps 1..t]
  final round    the last Z cycle is never started (no reset, no LZ); the
                 pending Z measurement is deferred into the final segment
  final segment  M(zanc, noisy) -> MX(xanc, noisy) + same-basis round-R
                 detectors -> DEPOLARIZE1(data) -> noiseless data readout ->
                 reconstruction detectors -> observables
  detectors      SAME-BASIS checks only; consecutive-measurement pairs
                 (first event single-term), plus final reconstruction rows
  uniform noise  DEPOLARIZE2(p) after CX; X_ERROR(p) after R / before M;
                 Z_ERROR(p) after RX / before MX; idle DEPOLARIZE1(p) under
                 the compaction rule below
  idle rule      (SBB Fig. 7 realized as selective idle noise)
                 - ancilla: noisy only when idle strictly BETWEEN its first
                   and last CNOT of the current reset->measure cycle
                   (prep-delay / measure-advance compaction);
                 - data: noisy while its sector group is active — from the
                   group's first CNOT step through its last CNOT step
                   (left group activates at the prologue, right group at
                   window 1; both retire in the final round);
                 - the final segment adds exactly one DEPOLARIZE1 on data.

Profile lowerings (02 §6.3). The segment timeline above is profile-free;
non-uniform profiles lower each planned segment to native gates:

  si1000   CZ + window-level H frames. CX(c->t) = H(t)·CZ·H(t); adjacent
           H·H cancel, so frames hoist to window granularity: every ancilla
           gets H right after its (prep-delayed) reset and right before its
           (measure-advanced) measurement; data holds the H frame for
           exactly its X-check window (left sector: window 1, right sector:
           window 2). Concretely: one H layer after global prep (zanc, plus
           data for X-basis |+> prep) and four per round — before step 2
           {left data, xanc}, before step t1+2 {all data, zanc}, after step
           t1+3 {zanc}, before step t {right data, xanc}. Each H layer is a
           full tick that counts toward depth and idle accounting. Rates:
           CZ->DEPOLARIZE2(p), H/1Q->DEPOLARIZE1(p/10), R->X_ERROR(2p),
           M->X_ERROR(measurement_factor*p); ordinary idle DEPOLARIZE1(p/10)
           under the same compaction gate as uniform; resonator idle
           DEPOLARIZE1(2p) on every tick containing an R or M (all
           compaction-active qubits except the R/M targets themselves).
  helios   CX kept as scheduled; each CX layer expands to three native
           moments — SQRT_Y_DAG(targets) | SQRT_ZZ(pairs) + virtual
           S_DAG(controls) | SQRT_X_DAG(targets) + virtual S_DAG(targets).
           RZZ noise is PAULI_CHANNEL_2 with the calibrated Pauli vector in
           (control, target) order, so orientation follows check type + edge
           direction naturally (02 §6.3). X-ancilla |+> prep / X-basis
           readout lower to SQRT_Y_DAG + virtual Z at the same boundary
           ticks as si1000's H layers. 1Q rotations get DEPOLARIZE1(scale*
           p1); R->X_ERROR(scale*p_init); M->X_ERROR(scale*p_meas); virtual
           gates (Z, S_DAG) are noiseless; the calibration model carries no
           idle noise.
  uniform_paper (v3.5, OmniQEC paper SI C b "Noise model") the SAME segment
           timeline, gates, TICKs, detectors and observables as the legacy
           uniform kind above; ONLY the noise instructions differ:
           DEPOLARIZE1(p) right after every reset (global prep and every
           ancilla R/RX), DEPOLARIZE2(p) after every CX layer, X_ERROR(p)
           before M / Z_ERROR(p) before MX on every ancilla measurement, and
           the final data readout preceded by the basis-conjugate flip
           (Z_ERROR before MX for the X-memory, X_ERROR before M for the
           Z-memory). NO idle noise and NO final DEPOLARIZE1(data); the
           legacy idle rule is kept only for uniform_legacy_v1 (golden /
           structural audits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omniqec_agentic.verification.gf2 import logicals as css_logicals
from omniqec_agentic.verification.noise import (
    NoisePoint,
    NoiseProfile,
    helios_rzz_probs,
)

from .schedule import LRCSchedule

_SI1000_PROFILES = (NoiseProfile.SI1000, NoiseProfile.SI1000_MEASUREMENT_X2)
_HELIOS_PROFILES = (
    NoiseProfile.HELIOS,
    NoiseProfile.HELIOS_SYMMETRIZED,
    NoiseProfile.HELIOS_PAULI_SHARPENED,
)
_UNIFORM_KINDS = ("uniform", "uniform_paper", "pauli_label")  # share the LRC timeline emission


@dataclass
class _Segment:
    cx_pairs: list[tuple[int, int]] = field(default_factory=list)
    resets: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)  # (rtype, targets)
    measures: list[tuple[str, tuple[int, ...], str]] = field(default_factory=list)  # (mtype, targets, tag)
    h_targets: tuple[int, ...] = ()  # basis-change layer (si1000 H / helios SQRT_Y_DAG+Z)
    final_data: bool = False  # final segment carries the epilogue emission


@dataclass(frozen=True)
class _Lowering:
    """Resolved profile lowering (02 §6.3): which native-gate emission the
    shared emitter uses and the per-profile rates it needs."""

    kind: str  # "uniform" | "uniform_paper" | "pauli_label" | "si1000" | "helios"
    p: float  # effective p (paper/si1000 override the positional p)
    meas_flip: float = 0.0  # si1000: measurement_factor*p ; helios: scale*p_meas
    reset_flip: float = 0.0  # si1000: 2p ; helios: scale*p_init
    he_p1: float = 0.0  # helios only
    he_probs: tuple[float, ...] = ()  # helios only (helios_rzz_probs)


def _resolve_lowering(
    noise_point: NoisePoint | None, p: float, *, saturate_noise: bool
) -> _Lowering:
    """``None`` (or uniform_legacy_v1) is the legacy uniform emission;
    uniform_paper_v1 swaps only the noise instructions (SI C b); si1000 /
    helios lower to native gates (02 §6.3). ``saturate_noise`` audits run on
    the uniform lowering only."""

    kind = "uniform"  # legacy uniform lowering (noise_point=None keeps it)
    si_cfg = he_cfg = None
    p = float(p)
    if noise_point is not None and noise_point.profile != NoiseProfile.UNIFORM_LEGACY:
        if noise_point.profile == NoiseProfile.UNIFORM_PAPER:
            kind = "uniform_paper"  # v3.5: SI C b noise on the uniform timeline
            p = float(noise_point.value)
        elif noise_point.profile == NoiseProfile.PAULI_LABEL_UNIFORM:
            kind = "pauli_label"  # 2026-09-15: Pauli-label-uniform, no ordinary idle
            p = float(noise_point.value)
        elif noise_point.profile in _SI1000_PROFILES:
            kind, si_cfg = "si1000", noise_point.si1000_config()
            p = float(si_cfg.p)
        elif noise_point.profile in _HELIOS_PROFILES:
            kind, he_cfg = "helios", noise_point.helios_config()
        else:  # pragma: no cover - future atom_* profiles
            raise NotImplementedError(
                f"no LRC lowering for profile '{noise_point.profile.value}' (02 §6.3)"
            )
        if saturate_noise:
            raise ValueError("saturate_noise audits run on the uniform lowering only")
    if kind == "si1000":
        return _Lowering(
            kind=kind,
            p=p,
            meas_flip=float(si_cfg.measurement_factor) * p,
            reset_flip=2 * p,
        )
    if kind == "helios":
        return _Lowering(
            kind=kind,
            p=p,
            meas_flip=float(he_cfg.scale) * float(he_cfg.p_meas),
            reset_flip=float(he_cfg.scale) * float(he_cfg.p_init),
            he_p1=float(he_cfg.scale) * float(he_cfg.p1),
            he_probs=tuple(helios_rzz_probs(he_cfg)),
        )
    return _Lowering(kind=kind, p=p)


@dataclass(frozen=True)
class _Timeline:
    """Everything phases 2+3 of the emitter need that is NOT in the segment
    list. Layout is shared by every timeline: data 0..n-1 | Z-ancillas
    n..n+mZ-1 | X-ancillas n+mZ.. ."""

    n_data: int
    m_z: int
    m_x: int
    basis: str  # memory basis "X" | "Z"; final data readout = MX if "X" else M
    # legacy/si1000 idle windows: LRC = (range(left), range(left, n)); twobasis = (range(n),)
    data_groups: tuple[tuple[int, ...], ...]
    # check kinds that get consecutive-pair detectors: LRC = {basis}; twobasis = {"X","Z"}
    detect_kinds: frozenset[str]
    # kinds whose FIRST measurement event gets a single-term detector: {basis}
    first_event_kinds: frozenset[str]


def _emit_circuit(
    segments: list[_Segment],
    timeline: _Timeline,
    lowering: _Lowering,
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    saturate_noise: bool,
    detector_kinds_out: list | None = None,
):
    """Shared emitter: idle annotation (phase 2) + stim emission (phase 3)
    of a planned segment timeline under one profile lowering.

    ``detector_kinds_out`` (v3.6, docs/06 §1.1): when a list is given, the
    check kind of every DETECTOR appended — "X" for detectors of X-type
    checks, "Z" for Z-type checks; the final data-readout reconstruction
    detectors get the memory basis' check kind — is appended to it in
    emission order (index == detector id). The emitted circuit is unchanged."""

    import stim

    kinds_out = detector_kinds_out

    kind = lowering.kind
    p = float(lowering.p)
    meas_flip = float(lowering.meas_flip)
    reset_flip = float(lowering.reset_flip)
    he_p1 = float(lowering.he_p1)
    he_probs = tuple(lowering.he_probs)
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    basis = timeline.basis
    n, m_z, m_x = timeline.n_data, timeline.m_z, timeline.m_x
    data = tuple(range(n))
    zanc = tuple(n + j for j in range(m_z))
    xanc = tuple(n + m_z + j for j in range(m_x))

    # ---- phase 2: idle-noise annotation (compaction rule) --------------
    # Windows are pinned to CNOT activity (golden-derived); basis-change
    # layers are busy in their own tick but do not widen any window.
    busy_steps: dict[int, list[int]] = {}
    boundaries: dict[int, list[int]] = {}  # reset/measure steps per ancilla
    for index, seg in enumerate(segments):
        for a, b in seg.cx_pairs:
            busy_steps.setdefault(a, []).append(index)
            busy_steps.setdefault(b, []).append(index)
        for _, targets in seg.resets:
            for q in targets:
                boundaries.setdefault(q, []).append(index)
        for _, targets, _ in seg.measures:
            for q in targets:
                boundaries.setdefault(q, []).append(index)

    group_of: dict[int, int] = {}
    group_window: list[tuple[int, int]] = []
    for group_index, members in enumerate(timeline.data_groups):
        for q in members:
            group_of[int(q)] = group_index
        steps = sorted(s for q in members for s in busy_steps.get(q, []))
        group_window.append((steps[0], steps[-1]) if steps else (0, -1))

    def idle_noisy(q: int, index: int) -> bool:
        if q < n:
            group = group_of.get(q)
            if group is None:
                return False
            lo, hi = group_window[group]
            return lo <= index <= hi
        steps = busy_steps.get(q, [])
        marks = boundaries.get(q, [])
        prev_boundary = max((m for m in marks if m < index), default=-1)
        next_boundary = min((m for m in marks if m > index), default=len(segments))
        in_cycle = [s for s in steps if prev_boundary < s < next_boundary]
        return bool(in_cycle) and in_cycle[0] < index < in_cycle[-1]

    # ---- phase 3: emit stim --------------------------------------------
    circuit = stim.Circuit()
    measure_count = 0
    last_rec: dict[tuple[str, int], int] = {}
    lx_logical, lz_logical = css_logicals(hx, hz)
    logical_rows = np.asarray(lx_logical if basis == "X" else lz_logical, dtype=np.uint8)

    def rec(index: int):
        return stim.target_rec(index - measure_count)

    def flip_channels(primary: str) -> tuple[str, ...]:
        return ("X_ERROR", "Z_ERROR") if saturate_noise else (primary,)

    def emit_measure(mtype: str, targets: tuple[int, ...], *, noisy: bool = True) -> list[int]:
        nonlocal measure_count
        if kind in _UNIFORM_KINDS:
            if (noisy or saturate_noise) and p > 0:
                for channel in flip_channels("X_ERROR" if mtype == "M" else "Z_ERROR"):
                    circuit.append(channel, targets, p)
        else:
            mtype = "M"  # X-basis readouts already conjugated by a boundary layer
            if noisy and meas_flip > 0:
                circuit.append("X_ERROR", targets, meas_flip)
        circuit.append(mtype, targets)
        out = list(range(measure_count, measure_count + len(targets)))
        measure_count += len(targets)
        return out

    def emit_cycle_detectors(kind_: str, indices: list[int]) -> None:
        if kind_ not in timeline.detect_kinds:
            for check, midx in enumerate(indices):
                last_rec[(kind_, check)] = midx
            return
        for check, midx in enumerate(indices):
            prev = last_rec.get((kind_, check))
            if prev is None and kind_ not in timeline.first_event_kinds:
                last_rec[(kind_, check)] = midx  # random first event: record only
                continue
            targets = [rec(midx)]
            if prev is not None:
                targets.append(rec(prev))
            circuit.append("DETECTOR", targets)
            if kinds_out is not None:
                kinds_out.append(kind_)
            last_rec[(kind_, check)] = midx

    for index, seg in enumerate(segments):
        busy: set[int] = set()
        for rtype, targets in seg.resets:
            if kind == "uniform_paper":  # SI C b: every reset -> DEPOLARIZE1(p)
                circuit.append(rtype, targets)
                if p > 0:
                    circuit.append("DEPOLARIZE1", targets, p)
            elif kind in ("uniform", "pauli_label"):  # reset error p = a flip after the reset
                circuit.append(rtype, targets)
                if p > 0:
                    for channel in flip_channels("X_ERROR" if rtype == "R" else "Z_ERROR"):
                        circuit.append(channel, targets, p)
            else:  # native reset is |0>; the |+> conjugation is a boundary layer
                circuit.append("R", targets)
                if reset_flip > 0:
                    circuit.append("X_ERROR", targets, reset_flip)
            busy.update(targets)
        if seg.h_targets:
            if kind == "si1000":
                circuit.append("H", seg.h_targets)
                if p > 0:
                    circuit.append("DEPOLARIZE1", seg.h_targets, p / 10)
            else:  # helios: H = Z . SQRT_Y_DAG (Z is virtual, noiseless)
                circuit.append("SQRT_Y_DAG", seg.h_targets)
                if he_p1 > 0:
                    circuit.append("DEPOLARIZE1", seg.h_targets, he_p1)
                circuit.append("Z", seg.h_targets)
            busy.update(seg.h_targets)
        if seg.cx_pairs:
            flat = [q for pair in seg.cx_pairs for q in pair]
            if kind == "pauli_label":  # X, Y, Z each with probability p on control AND target
                circuit.append("CX", flat)
                if p > 0:
                    circuit.append("PAULI_CHANNEL_1", flat, (p, p, p))
            elif kind in _UNIFORM_KINDS:
                circuit.append("CX", flat)
                if p > 0:
                    circuit.append("DEPOLARIZE2", flat, p)
            elif kind == "si1000":
                circuit.append("CZ", flat)
                if p > 0:
                    circuit.append("DEPOLARIZE2", flat, p)
            else:  # helios: three native moments per scheduled CX layer
                controls = [a for a, _ in seg.cx_pairs]
                cx_targets = [b for _, b in seg.cx_pairs]
                circuit.append("SQRT_Y_DAG", cx_targets)
                if he_p1 > 0:
                    circuit.append("DEPOLARIZE1", cx_targets, he_p1)
                circuit.append("TICK")
                circuit.append("SQRT_ZZ", flat)
                if sum(he_probs) > 0:
                    circuit.append("PAULI_CHANNEL_2", flat, he_probs)
                circuit.append("S_DAG", controls)
                circuit.append("TICK")
                circuit.append("SQRT_X_DAG", cx_targets)
                if he_p1 > 0:
                    circuit.append("DEPOLARIZE1", cx_targets, he_p1)
                circuit.append("S_DAG", cx_targets)
            busy.update(flat)
        for mtype, targets, tag in seg.measures:
            indices = emit_measure(mtype, targets)
            busy.update(targets)
            emit_cycle_detectors("Z" if tag == "zcycle" else "X", indices)
        if seg.final_data:
            # The final segment's ancilla measures travelled in seg.measures;
            # only the data-readout epilogue remains.
            if kind in ("uniform_paper", "pauli_label"):
                # SI C b: no DEPOLARIZE1(data); the readout is preceded by the
                # basis-conjugate flip (Z_ERROR before MX / X_ERROR before M).
                # pauli_label: the resonator idle of the final ancilla readout
                # segment lands on the data qubits first.
                if kind == "pauli_label" and p > 0 and seg.measures:
                    circuit.append("DEPOLARIZE1", data, p)
                data_indices = emit_measure("M" if basis == "Z" else "MX", data)
            elif kind == "uniform":
                if p > 0:
                    circuit.append("DEPOLARIZE1", data, p)
                data_indices = emit_measure(
                    "M" if basis == "Z" else "MX", data, noisy=False
                )
            elif kind == "si1000":
                if p > 0:  # ordinary + resonator idle during the ancilla readout
                    circuit.append("DEPOLARIZE1", data, p / 10)
                    circuit.append("DEPOLARIZE1", data, 2 * p)
                if basis == "X":
                    circuit.append("H", data)
                    if p > 0:
                        circuit.append("DEPOLARIZE1", data, p / 10)
                if meas_flip > 0:
                    circuit.append("X_ERROR", data, meas_flip)
                data_indices = emit_measure("M", data, noisy=False)
            else:  # helios
                if basis == "X":
                    circuit.append("SQRT_Y_DAG", data)
                    if he_p1 > 0:
                        circuit.append("DEPOLARIZE1", data, he_p1)
                    circuit.append("Z", data)
                if meas_flip > 0:
                    circuit.append("X_ERROR", data, meas_flip)
                data_indices = emit_measure("M", data, noisy=False)
            checks = hx if basis == "X" else hz
            for check in range(checks.shape[0]):
                targets = [rec(data_indices[q]) for q in np.nonzero(checks[check])[0]]
                targets.append(rec(last_rec[(basis, check)]))
                circuit.append("DETECTOR", targets)
                if kinds_out is not None:
                    kinds_out.append(basis)
            for row_index in range(logical_rows.shape[0]):
                targets = [rec(data_indices[q]) for q in np.nonzero(logical_rows[row_index])[0]]
                circuit.append("OBSERVABLE_INCLUDE", targets, row_index)
            continue  # no idle noise / TICK after the final segment
        if kind == "uniform":  # legacy idle rule only (uniform_paper: none)
            if p > 0 and index > 0:
                idle = sorted(
                    q
                    for q in range(n + m_z + m_x)
                    if q not in busy and (saturate_noise or idle_noisy(q, index))
                )
                if idle:
                    circuit.append("DEPOLARIZE1", idle, p)
        elif kind == "pauli_label" and p > 0:
            # resonator idle p on every qubit not itself reset/measured in a
            # segment that resets or measures something; NO ordinary idle.
            mr = {q for _, targets in seg.resets for q in targets}
            mr.update(q for _, targets, _ in seg.measures for q in targets)
            if mr and index > 0:
                resonator = sorted(q for q in range(n + m_z + m_x) if q not in mr)
                if resonator:
                    circuit.append("DEPOLARIZE1", resonator, p)
        elif kind == "si1000" and p > 0:
            idle = sorted(
                q for q in range(n + m_z + m_x) if q not in busy and idle_noisy(q, index)
            )
            if idle:
                circuit.append("DEPOLARIZE1", idle, p / 10)
            mr = {q for _, targets in seg.resets for q in targets}
            mr.update(q for _, targets, _ in seg.measures for q in targets)
            if mr:
                resonator = sorted((busy | set(idle)) - mr)
                if resonator:
                    circuit.append("DEPOLARIZE1", resonator, 2 * p)
        # uniform_paper (SI C b) and helios carry no idle noise
        circuit.append("TICK")
    return circuit


def build_lrc_memory_circuit(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: LRCSchedule,
    *,
    basis: str,
    rounds: int,
    p: float,
    saturate_noise: bool = False,
    noise_point: NoisePoint | None = None,
    detector_kinds_out: list | None = None,
):
    """LRC timeline -> shared emitter (``_emit_circuit``).

    ``saturate_noise`` builds the structural audit variant (02 §4.3): every
    idle slot gets DEPOLARIZE1 (compaction rule suspended) and every prep/meas
    gets BOTH X and Z flips, so no fault location has a zero-probability
    component that could hide a catastrophic mechanism from the DEM scan.

    ``noise_point`` selects the lowering: ``None`` (or uniform_legacy_v1) is
    the legacy uniform emission byte-for-byte; uniform_paper_v1 keeps that
    timeline and swaps only the noise instructions (SI C b); si1000/helios
    lower to native gates (02 §6.3)."""
    try:
        import stim  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"stim unavailable: {exc}") from exc

    basis = str(basis).upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    rounds = int(rounds)
    if rounds < 2:
        raise ValueError("LRC memory needs rounds >= 2 (staggered Z cycles)")

    low = _resolve_lowering(noise_point, p, saturate_noise=saturate_noise)
    kind = low.kind

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n, m_z, m_x = schedule.n_data, schedule.m_z, schedule.m_x
    data = tuple(range(n))
    zanc = tuple(n + j for j in range(m_z))
    xanc = tuple(n + m_z + j for j in range(m_x))
    t1, t2, t = schedule.t1, schedule.t2, schedule.t
    layers = schedule.layers
    lz_layers = layers["LZ"]

    def pairs_of(slot: str, color: int) -> list[tuple[int, int]]:
        out = []
        for check, local in layers[slot][color]:
            if slot in ("LX", "RX"):
                qubit = local + (0 if slot == "LX" else schedule.left)
                out.append((xanc[check], qubit))
            else:
                qubit = local + (0 if slot == "LZ" else schedule.left)
                out.append((qubit, zanc[check]))
        return sorted(out)

    # Basis-change layer target sets per boundary (empty tuples for uniform).
    left_data, right_data = data[: schedule.left], data[schedule.left :]
    if kind == "si1000":
        layer_prep = (data if basis == "X" else ()) + zanc
        layer_w1 = left_data + xanc
        layer_mid = data + zanc
        layer_zprep = zanc
        layer_end = right_data + xanc
    elif kind == "helios":
        layer_prep = data if basis == "X" else ()
        layer_w1 = xanc
        layer_mid = layer_zprep = ()
        layer_end = xanc
    else:
        layer_prep = layer_w1 = layer_mid = layer_zprep = layer_end = ()

    def h_seg(targets: tuple[int, ...]) -> _Segment:
        seg = _Segment()
        seg.h_targets = tuple(targets)
        return seg

    # ---- phase 1: plan the timeline -----------------------------------
    segments: list[_Segment] = []
    prep = _Segment()
    prep.resets.append(("RX" if basis == "X" else "R", data))
    prep.resets.append(("R", zanc))
    prep.resets.append(("RX", xanc))
    segments.append(prep)
    if layer_prep:
        segments.append(h_seg(layer_prep))

    for color in range(len(lz_layers)):  # prologue: first Z cycle's LZ
        seg = _Segment()
        seg.cx_pairs = pairs_of("LZ", color)
        segments.append(seg)

    for round_index in range(1, rounds + 1):
        last_round = round_index == rounds
        for step in range(2 if round_index == 1 else 1, t + 1):
            if step == 2 and layer_w1:
                segments.append(h_seg(layer_w1))
            if step == t1 + 2 and layer_mid:
                segments.append(h_seg(layer_mid))
            if step == t and layer_end:
                segments.append(h_seg(layer_end))
            seg = _Segment()
            if step == 1:
                seg.resets.append(("RX", xanc))
                if len(lz_layers) == t2:  # wrap color of the previous Z cycle
                    seg.cx_pairs += pairs_of("LZ", t2 - 1)
            if 2 <= step <= t1 + 1:
                color = step - 2
                if color < len(layers["LX"]):
                    seg.cx_pairs += pairs_of("LX", color)
                if color < len(layers["RZ"]):
                    seg.cx_pairs += pairs_of("RZ", color)
            if t1 + 2 <= step <= t1 + t2 + 1:
                color = step - t1 - 2
                if color < len(layers["RX"]):
                    seg.cx_pairs += pairs_of("RX", color)
            if not last_round:
                if step == t1 + 2:
                    seg.measures.append(("M", zanc, "zcycle"))
                if step == t1 + 3:
                    seg.resets.append(("R", zanc))
                if t1 + 4 <= step <= t:
                    color = step - t1 - 4
                    if color < min(len(lz_layers), t2 - 1):
                        seg.cx_pairs += pairs_of("LZ", color)
            if step == t:
                if last_round:
                    # Final segment: deferred final Z measurement (complete
                    # cycle), then round-R MX, then the data-readout epilogue.
                    seg.final_data = True
                    seg.measures.append(("M", zanc, "zcycle"))
                    seg.measures.append(("MX", xanc, "xcycle"))
                else:
                    seg.measures.append(("MX", xanc, "xcycle"))
            seg.cx_pairs.sort()
            segments.append(seg)
            if step == t1 + 3 and not last_round and layer_zprep:
                segments.append(h_seg(layer_zprep))

    timeline = _Timeline(
        n_data=n,
        m_z=m_z,
        m_x=m_x,
        basis=basis,
        data_groups=(tuple(range(schedule.left)), tuple(range(schedule.left, n))),
        detect_kinds=frozenset({basis}),
        first_event_kinds=frozenset({basis}),
    )
    return _emit_circuit(
        segments, timeline, low, hx, hz,
        saturate_noise=saturate_noise, detector_kinds_out=detector_kinds_out,
    )


def build_lrc_memory_pair(
    compiled: Any,
    *,
    rounds: int,
    p: float,
    candidate_index: int = 0,
    noise_point: NoisePoint | None = None,
) -> dict[str, Any]:
    """Both memory bases for a CompiledCode, plus schedule metadata."""

    from .graph import split_sectors
    from .schedule import build_canonical_schedule

    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    sectors = split_sectors(hx, hz, compiled.sector_sizes)
    schedule = build_canonical_schedule(sectors, candidate_index=candidate_index)
    return {
        "schedule": schedule,
        "x_memory": build_lrc_memory_circuit(
            hx, hz, schedule, basis="X", rounds=rounds, p=p, noise_point=noise_point
        ),
        "z_memory": build_lrc_memory_circuit(
            hx, hz, schedule, basis="Z", rounds=rounds, p=p, noise_point=noise_point
        ),
        "metadata": schedule.to_metadata(),
    }
