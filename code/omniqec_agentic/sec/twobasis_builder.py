"""Two-basis NON-interleaved memory circuit (02 §9.4) on the shared emitter.

Timeline (T rounds, basis W in {X, Z}; layout data | zanc | xanc as LRC):

  S0   global prep: RX data (W = X) / R data (W = Z)          [data ONLY]
  per round r = 1..T:
    A    RX xanc
    X_i  CX(xanc -> data) for layer i of the X order        (one tick each)
    B    MX xanc                                             ("xcycle")
    C    R zanc
    Z_j  CX(data -> zanc) for layer j of the Z order
    D    M zanc                                              ("zcycle")
  final segment = round-T D + data-readout epilogue (MX data for W = X /
  M data for W = Z; the profile's readout noise exactly as the LRC emitter).

TICK after every segment; uniform ticks = T·(|LX|+|LZ|+4).
Detectors (BOTH bases): W checks get the single-term detector at round 1
and consecutive pairs afterwards; opposite-basis checks are recorded only at
round 1 (random) and paired from round 2; final reconstruction detectors for
W checks only; observables = W logicals from the data readout.

Profile lowerings come from the emitter: uniform_legacy (idle rule with one
data group), uniform_paper (SI C b: DEPOLARIZE1 after every reset incl. S0,
DEPOLARIZE2 after every CX layer, flips before every measurement incl. the
final data readout, no idle, no final DEPOLARIZE1), si1000 / helios with
correctness-first H frames (X-memory: prep-H followed by the frame-H, both
noisy, not cancelled; no window hoisting) — smoke-tested only, not tuned.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.verification.noise import NoisePoint

from .stim_builder import _Segment, _Timeline, _emit_circuit, _resolve_lowering
from .twobasis_rank import DEFAULT_SEC_POLICY, SecPolicyConfig, TwoBasisSchedule


def build_twobasis_memory_circuit(
    hx: np.ndarray,
    hz: np.ndarray,
    schedule: TwoBasisSchedule,
    *,
    basis: str,
    rounds: int,
    p: float,
    noise_point: NoisePoint | None = None,
    saturate_noise: bool = False,
    detector_kinds_out: list | None = None,
):
    """Two-basis timeline -> shared emitter. ``detector_kinds_out`` (v3.6):
    a list that receives the check kind ("X"/"Z") of every DETECTOR in
    emission order; the emitted circuit text is unchanged."""

    try:
        import stim  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"stim unavailable: {exc}") from exc

    basis = str(basis).upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("twobasis memory needs rounds >= 1")

    low = _resolve_lowering(noise_point, p, saturate_noise=saturate_noise)
    kind = low.kind

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n, m_z, m_x = int(schedule.n_data), int(schedule.m_z), int(schedule.m_x)
    if hx.shape != (m_x, n) or hz.shape != (m_z, n):
        raise ValueError("schedule dimensions do not match hx/hz")
    data = tuple(range(n))
    zanc = tuple(n + j for j in range(m_z))
    xanc = tuple(n + m_z + j for j in range(m_x))
    layers_x = schedule.layers["X"]
    layers_z = schedule.layers["Z"]

    def pairs_x(layer) -> list[tuple[int, int]]:
        return sorted((xanc[int(c)], int(d)) for c, d in layer)

    def pairs_z(layer) -> list[tuple[int, int]]:
        return sorted((int(d), zanc[int(c)]) for c, d in layer)

    def h_seg(targets: tuple[int, ...]) -> _Segment:
        seg = _Segment()
        seg.h_targets = tuple(targets)
        return seg

    # Basis-change frames (empty for the uniform kinds).
    if kind == "si1000":
        frame_prep = data if basis == "X" else ()
        frame_x = data + xanc  # CX(xanc->data) = H(data) CZ H(data); xanc |+> prep / X readout
        frame_z = zanc  # CX(data->zanc) = H(zanc) CZ H(zanc)
    elif kind == "helios":
        frame_prep = data if basis == "X" else ()
        frame_x = xanc  # |+> prep / X readout only; CX stays a scheduled CX
        frame_z = ()
    else:
        frame_prep = frame_x = frame_z = ()

    segments: list[_Segment] = []
    prep = _Segment()
    prep.resets.append(("RX" if basis == "X" else "R", data))
    segments.append(prep)
    if frame_prep:
        segments.append(h_seg(frame_prep))

    for round_index in range(1, rounds + 1):
        last_round = round_index == rounds
        seg = _Segment()  # A
        seg.resets.append(("RX", xanc))
        segments.append(seg)
        if frame_x:
            segments.append(h_seg(frame_x))
        for layer in layers_x:  # X_i
            seg = _Segment()
            seg.cx_pairs = pairs_x(layer)
            segments.append(seg)
        if frame_x:
            segments.append(h_seg(frame_x))
        seg = _Segment()  # B
        seg.measures.append(("MX", xanc, "xcycle"))
        segments.append(seg)
        seg = _Segment()  # C
        seg.resets.append(("R", zanc))
        segments.append(seg)
        if frame_z:
            segments.append(h_seg(frame_z))
        for layer in layers_z:  # Z_j
            seg = _Segment()
            seg.cx_pairs = pairs_z(layer)
            segments.append(seg)
        if frame_z:
            segments.append(h_seg(frame_z))
        seg = _Segment()  # D
        seg.measures.append(("M", zanc, "zcycle"))
        if last_round:
            seg.final_data = True
        segments.append(seg)

    timeline = _Timeline(
        n_data=n,
        m_z=m_z,
        m_x=m_x,
        basis=basis,
        data_groups=(tuple(range(n)),),
        detect_kinds=frozenset({"X", "Z"}),
        first_event_kinds=frozenset({basis}),
    )
    return _emit_circuit(
        segments, timeline, low, hx, hz,
        saturate_noise=saturate_noise, detector_kinds_out=detector_kinds_out,
    )


def build_twobasis_memory_pair(
    compiled: Any,
    *,
    rounds: int,
    p: float,
    noise_point: NoisePoint | None = None,
    policy: SecPolicyConfig = DEFAULT_SEC_POLICY,
) -> dict[str, Any]:
    """Both memory bases for a CompiledCode under the ranked twobasis policy,
    plus schedule metadata (mirrors ``build_lrc_memory_pair``)."""

    from .twobasis_rank import rank_twobasis_schedules

    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    ranked = rank_twobasis_schedules(
        hx, hz, layer_blocks=getattr(compiled, "layer_blocks", None), policy=policy
    )
    schedule = ranked.schedule
    return {
        "schedule": schedule,
        "ranked": ranked,
        "x_memory": build_twobasis_memory_circuit(
            hx, hz, schedule, basis="X", rounds=rounds, p=p, noise_point=noise_point
        ),
        "z_memory": build_twobasis_memory_circuit(
            hx, hz, schedule, basis="Z", rounds=rounds, p=p, noise_point=noise_point
        ),
        "metadata": schedule.to_metadata(),
    }
