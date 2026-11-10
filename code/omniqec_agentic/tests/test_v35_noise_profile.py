"""v3.5 noise profile (contract §1/§2, tests §5 a+b): OmniQEC paper SI C b
lowering fidelity and the NoiseSweepConfig defaults.

SI C b, verbatim: "All circuits use a uniform circuit-level depolarizing model
controlled by a single physical error rate p. Each qubit reset is followed by
single-qubit depolarizing noise of strength p; each controlled-NOT is followed
by two-qubit depolarizing noise of strength p; each ancilla measurement is
preceded by a bit-flip error of probability p; and the final data readout is
preceded by a basis-conjugate flip of probability p."  No idling noise.

Section (a) counts instructions in ``str(circuit)`` for BB72 against that
sentence and checks that the non-noise skeleton (gates, TICKs, detectors,
observables) is identical to the legacy uniform emission of the same
schedule. Section (b) pins the NoiseSweepConfig defaults / factories /
round-trip. Nothing else lives here (§5 c-f belong to their owners' files).
"""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation import BASELINE_BB_SPEC
from omniqec_agentic.sec import build_canonical_schedule, split_sectors
from omniqec_agentic.sec.stim_builder import build_lrc_memory_circuit
from omniqec_agentic.verification.noise import (
    UNIFORM_PROFILES,
    NoiseProfile,
    NoiseSweepConfig,
    noise_sweep_from_legacy_p_values,
)

P = 0.005
ROUNDS = 3
NOISE_NAMES = {"X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}


@pytest.fixture(scope="module")
def bb72():
    compiled = build_compiled_code(BASELINE_BB_SPEC)
    hx = np.array(compiled.hx, dtype=np.uint8)
    hz = np.array(compiled.hz, dtype=np.uint8)
    schedule = build_canonical_schedule(split_sectors(hx, hz, compiled.sector_sizes))
    return hx, hz, schedule


def _paper_point(p=P):
    return NoiseSweepConfig.uniform((p,)).point(0)


def _legacy_point(p=P):
    return NoiseSweepConfig.uniform_legacy((p,)).point(0)


def _instructions(circuit):
    """(name, args, targets) per line of str(circuit); the builder emits no
    REPEAT blocks, so every line is one instruction."""
    out = []
    for raw in str(circuit).splitlines():
        line = raw.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        args = ()
        if "(" in head:
            head, _, argtxt = head.partition("(")
            args = tuple(float(a) for a in argtxt.rstrip(")").split(","))
        out.append((head, args, tuple(rest.split())))
    return out


def _qubits(targets):
    return tuple(int(t) for t in targets)


def _build(hx, hz, schedule, basis, **kw):
    return build_lrc_memory_circuit(
        hx, hz, schedule, basis=basis, rounds=ROUNDS, p=P, **kw
    )


def _layout(schedule):
    n, m_z, m_x = schedule.n_data, schedule.m_z, schedule.m_x
    data = tuple(range(n))
    zanc = tuple(range(n, n + m_z))
    xanc = tuple(range(n + m_z, n + m_z + m_x))
    return data, zanc, xanc


# ---------------------------------------------------------------- (a) lowering


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_paper_depolarize1_only_and_exactly_after_every_reset(bb72, basis):
    """'Each qubit reset is followed by single-qubit depolarizing noise of
    strength p' -- and nothing else emits DEPOLARIZE1 (no idle noise)."""
    hx, hz, schedule = bb72
    data, zanc, xanc = _layout(schedule)
    ins = _instructions(_build(hx, hz, schedule, basis, noise_point=_paper_point()))
    resets = [i for i, (name, _, _) in enumerate(ins) if name in ("R", "RX")]
    dep1 = [i for i, (name, _, _) in enumerate(ins) if name == "DEPOLARIZE1"]
    assert resets and len(dep1) == len(resets)
    for i in resets:  # reset -> DEPOLARIZE1(p) on the same targets
        assert ins[i + 1] == ("DEPOLARIZE1", (P,), ins[i][2])
    for i in dep1:  # every DEPOLARIZE1 sits right after a reset
        assert ins[i - 1][0] in ("R", "RX") and ins[i - 1][2] == ins[i][2]
    assert sum(len(ins[i][2]) for i in dep1) == sum(len(ins[i][2]) for i in resets)
    # global prep: data (basis prep), Z-ancillas |0>, X-ancillas |+>
    assert [(ins[i][0], _qubits(ins[i][2])) for i in resets[:3]] == [
        ("RX" if basis == "X" else "R", data),
        ("R", zanc),
        ("RX", xanc),
    ]
    # ancilla re-preparation: R zanc / RX xanc once per non-final cycle
    later = [(ins[i][0], _qubits(ins[i][2])) for i in resets[3:]]
    assert later.count(("R", zanc)) == ROUNDS - 1
    assert later.count(("RX", xanc)) == ROUNDS - 1
    assert len(later) == 2 * (ROUNDS - 1)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_paper_cx_measurement_and_final_readout_noise(bb72, basis):
    """CX -> DEPOLARIZE2(p); ancilla M/MX preceded by X_ERROR/Z_ERROR(p); the
    final data readout preceded by the basis-conjugate flip; no DEPOLARIZE1
    after the last ancilla measurement layer."""
    hx, hz, schedule = bb72
    data, zanc, xanc = _layout(schedule)
    ins = _instructions(_build(hx, hz, schedule, basis, noise_point=_paper_point()))
    cx = [i for i, (name, _, _) in enumerate(ins) if name == "CX"]
    dep2 = [i for i, (name, _, _) in enumerate(ins) if name == "DEPOLARIZE2"]
    assert cx and len(cx) == len(dep2)
    for i in cx:
        assert ins[i + 1] == ("DEPOLARIZE2", (P,), ins[i][2])
    meas = [i for i, (name, _, _) in enumerate(ins) if name in ("M", "MX")]
    anc_m = [i for i in meas if _qubits(ins[i][2]) == zanc]
    anc_mx = [i for i in meas if _qubits(ins[i][2]) == xanc]
    assert len(anc_m) == ROUNDS and len(anc_mx) == ROUNDS
    for i in anc_m:
        assert ins[i][0] == "M" and ins[i - 1] == ("X_ERROR", (P,), ins[i][2])
    for i in anc_mx:
        assert ins[i][0] == "MX" and ins[i - 1] == ("Z_ERROR", (P,), ins[i][2])
    final = meas[-1]
    assert len(meas) == len(anc_m) + len(anc_mx) + 1
    assert _qubits(ins[final][2]) == data
    if basis == "X":
        assert ins[final][0] == "MX"
        assert ins[final - 1] == ("Z_ERROR", (P,), ins[final][2])
    else:
        assert ins[final][0] == "M"
        assert ins[final - 1] == ("X_ERROR", (P,), ins[final][2])
    last_anc = max(anc_m + anc_mx)
    assert not any(name == "DEPOLARIZE1" for name, _, _ in ins[last_anc:])
    assert all(args == (P,) for name, args, _ in ins if name in NOISE_NAMES)
    # the only X_ERROR/Z_ERROR lines are the ones before a measurement
    for i, (name, _, _) in enumerate(ins):
        if name in ("X_ERROR", "Z_ERROR"):
            assert ins[i + 1][0] == ("M" if name == "X_ERROR" else "MX")


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_paper_skeleton_identical_to_legacy_emission(bb72, basis):
    """Same segments/schedule/detectors/observables as the legacy uniform kind:
    strip the noise lines and the two emissions are byte-identical."""
    hx, hz, schedule = bb72
    legacy = _build(hx, hz, schedule, basis)  # noise_point=None => legacy
    paper = _build(hx, hz, schedule, basis, noise_point=_paper_point())
    for attr in (
        "num_measurements", "num_detectors", "num_observables", "num_ticks", "num_qubits",
    ):
        assert getattr(paper, attr) == getattr(legacy, attr), attr
    assert paper.num_detectors == (ROUNDS + 1) * (
        len(hx) if basis == "X" else len(hz)
    )

    def skeleton(circuit):
        return [ins for ins in _instructions(circuit) if ins[0] not in NOISE_NAMES]

    assert skeleton(paper) == skeleton(legacy)
    assert str(paper) != str(legacy)
    # legacy carries idle DEPOLARIZE1 layers; the paper kind has exactly one
    # DEPOLARIZE1 per reset and nothing else
    count = lambda circuit, name: sum(1 for n, _, _ in _instructions(circuit) if n == name)
    resets = count(paper, "R") + count(paper, "RX")
    assert count(paper, "DEPOLARIZE1") == resets < count(legacy, "DEPOLARIZE1")
    dem = paper.detector_error_model()  # undecomposed, as circuit.py keeps it
    assert dem.num_detectors == paper.num_detectors
    assert dem.num_observables == paper.num_observables


def test_paper_point_is_authoritative_and_edge_cases(bb72):
    hx, hz, schedule = bb72
    # saturate_noise stays a legacy-uniform-only structural audit
    with pytest.raises(ValueError, match="uniform lowering only"):
        _build(hx, hz, schedule, "X", noise_point=_paper_point(), saturate_noise=True)
    # noise_point=None and an explicit legacy point are the same emission
    assert str(_build(hx, hz, schedule, "Z")) == str(
        _build(hx, hz, schedule, "Z", noise_point=_legacy_point())
    )
    # the paper point's value wins over a stale positional p (as si1000 does)
    override = build_lrc_memory_circuit(
        hx, hz, schedule, basis="X", rounds=ROUNDS, p=0.5, noise_point=_paper_point()
    )
    assert all(args == (P,) for name, args, _ in _instructions(override) if name in NOISE_NAMES)
    # p = 0: no noise instruction at all, and the noiseless detectors are quiet
    quiet = build_lrc_memory_circuit(
        hx, hz, schedule, basis="X", rounds=ROUNDS, p=0.0, noise_point=_paper_point(0.0)
    )
    assert not any(name in NOISE_NAMES for name, _, _ in _instructions(quiet))
    assert not quiet.compile_detector_sampler().sample(4).any()


def test_memory_experiment_pair_accepts_paper_point(bb72):
    """circuit.py seam (contract §1: no functional change): the paper point
    flows through build_memory_experiment_pair -- static audit on the legacy
    emission of the same schedule, undecomposed DEM, metadata carrying the
    profile and the p axis -- and the emitted circuit is the paper lowering."""
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair

    hx, hz, _ = bb72
    sector_sizes = list(build_compiled_code(BASELINE_BB_SPEC).sector_sizes)
    pair = build_memory_experiment_pair(
        hx, hz, p=P, rounds=ROUNDS, noise_point=_paper_point(), sector_sizes=sector_sizes
    )
    for exp in (pair.x_memory, pair.z_memory):
        md = exp.to_metadata()
        assert md["noise_point"]["profile"] == "uniform_paper_v1"
        assert md["p"] == P and exp.p == P and exp.noise_point.profile is NoiseProfile.UNIFORM_PAPER
        assert exp.detector_error_model.num_detectors == exp.circuit.num_detectors
        assert exp.detector_error_model.num_observables == exp.circuit.num_observables
        ins = _instructions(exp.circuit)
        dep1 = [i for i, (name, _, _) in enumerate(ins) if name == "DEPOLARIZE1"]
        assert dep1 and all(ins[i - 1][0] in ("R", "RX") for i in dep1)
    legacy = build_memory_experiment_pair(
        hx, hz, p=P, rounds=ROUNDS, noise_point=None, sector_sizes=sector_sizes
    )
    assert legacy.x_memory.to_metadata()["noisy_circuit_sha256"] != (
        pair.x_memory.to_metadata()["noisy_circuit_sha256"]
    )
    assert legacy.x_memory.circuit.num_detectors == pair.x_memory.circuit.num_detectors


# --------------------------------------------------------- (b) NoiseSweepConfig


def test_sweep_default_is_paper():
    cfg = NoiseSweepConfig()
    assert cfg.profile is NoiseProfile.UNIFORM_PAPER
    assert cfg.profile.value == "uniform_paper_v1"
    assert NoiseProfile("uniform_paper_v1") is NoiseProfile.UNIFORM_PAPER
    assert cfg.values == (0.005,) and cfg.axis == "p"
    assert cfg.source_kind == "paper_protocol_parameter"
    assert cfg.source_reference == "OmniQEC paper SI C uniform v1"
    assert NoiseSweepConfig.uniform() == cfg
    point = cfg.point(0)
    assert point.profile is NoiseProfile.UNIFORM_PAPER
    assert (point.value, point.axis) == (0.005, "p")
    assert point.to_dict()["profile"] == "uniform_paper_v1"


def test_uniform_factories_and_p_alias():
    paper = NoiseSweepConfig.uniform((0.001, 0.005))
    assert paper.profile is NoiseProfile.UNIFORM_PAPER
    assert paper.values == (0.001, 0.005)
    assert (paper.source_kind, paper.source_reference) == (
        "paper_protocol_parameter", "OmniQEC paper SI C uniform v1",
    )
    legacy = NoiseSweepConfig.uniform_legacy((0.001, 0.005))
    assert legacy.profile is NoiseProfile.UNIFORM_LEGACY
    assert legacy.profile.value == "uniform_legacy_v1"
    assert legacy.values == (0.001, 0.005)
    assert (legacy.source_kind, legacy.source_reference) == (
        "legacy_protocol_parameter", "OmniQEC uniform legacy v1",
    )
    assert legacy.axis == "p"
    assert paper.point(0).config_hash != legacy.point(0).config_hash
    # --p / --p-values aliases mean "the current uniform protocol"
    assert noise_sweep_from_legacy_p_values((0.003, 0.005)) == NoiseSweepConfig.uniform(
        (0.003, 0.005)
    )
    for factory in (NoiseSweepConfig.uniform, NoiseSweepConfig.uniform_legacy):
        with pytest.raises(ValueError):
            factory((1.5,))


def test_uniform_profiles_frozenset():
    assert isinstance(UNIFORM_PROFILES, frozenset)
    assert UNIFORM_PROFILES == frozenset(
        {NoiseProfile.UNIFORM_LEGACY, NoiseProfile.UNIFORM_PAPER}
    )
    for profile in NoiseProfile:
        assert (profile in UNIFORM_PROFILES) == profile.value.startswith("uniform_")


def test_from_dict_round_trips():
    for cfg in (
        NoiseSweepConfig(),
        NoiseSweepConfig.uniform((0.001, 0.005)),
        NoiseSweepConfig.uniform_legacy((0.002,)),
        NoiseSweepConfig.si1000(),
        NoiseSweepConfig.helios(),
    ):
        assert NoiseSweepConfig.from_dict(cfg.to_dict()) == cfg
    assert NoiseSweepConfig.from_dict(None) == NoiseSweepConfig()
    assert NoiseSweepConfig.from_dict({}) == NoiseSweepConfig()
    assert NoiseSweepConfig.from_dict({"profile": "uniform_paper_v1"}) == NoiseSweepConfig()
    # a stored v3.4 dict stays legacy, source strings and all
    legacy_dict = NoiseSweepConfig.uniform_legacy().to_dict()
    assert legacy_dict["profile"] == "uniform_legacy_v1"
    restored = NoiseSweepConfig.from_dict(legacy_dict)
    assert restored == NoiseSweepConfig.uniform_legacy()
    assert restored != NoiseSweepConfig()
    # a profile-only legacy dict picks the legacy source strings, not paper's
    assert NoiseSweepConfig.from_dict({"profile": "uniform_legacy_v1"}) == (
        NoiseSweepConfig.uniform_legacy()
    )
