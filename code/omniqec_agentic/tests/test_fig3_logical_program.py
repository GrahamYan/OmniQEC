"""sec/logical_program.py (Fig.3 stages 2 and 3). The general engine must emit
the stage 0/1 circuits of multiblock_builder instruction for instruction; the
GHZ program (|+>|0>|0>, CNOT 0->1, QEC, CNOT 0->2) must be deterministic
with the GHZ stabilizers as observables and the right fault sensitivity; the
teleportation program must be deterministic with the Bell-outcome Pauli frame
folded into its observables and carry a logical error on the input to the
output; the heterogeneous CNOT (control A, target = A with one qubit frozen)
must be deterministic and act as the standard logical CNOT, and incompatible
codes or non-standard logical maps must be refused."""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from omniqec_agentic.construction.build import build_compiled_code
from omniqec_agentic.sec.logical_program import (
    ProgramBlock,
    build_logical_program_circuit,
    cnot_maps,
    program_experiment_circuits,
)
from omniqec_agentic.sec.multiblock_builder import EXPERIMENT_LAYOUTS, build_twobasis_multiblock_circuit
from omniqec_agentic.sec.twobasis_rank import SecPolicyConfig, rank_twobasis_schedules
from omniqec_agentic.verification.gf2 import logicals as css_logicals
from omniqec_agentic.verification.gf2 import rank2
from omniqec_agentic.verification.noise import NoisePoint, NoiseProfile

FAST_POLICY = SecPolicyConfig(sec_candidates=3, sec_cert_iters=20, sec_cert_seeds=1, sec_cert_iters_scale=0.0,
                              sec_refine_top=0, sec_refine_iters=1)
SPECS = {
    "bb_code_b": {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 5]], "B": [[0, 0], [2, 1]]},
    "lp_abelian_c24": {"family": "lp_abelian", "group": {"factors": [24]},
                       "A": [[[[0], [15]]]], "B": [[[[0], [5], [16]]]]},
}


def _point(p: float):
    return NoisePoint(profile=NoiseProfile.UNIFORM_PAPER, value=float(p), axis="p",
                      source_kind="test", source_reference="test")


def _schedule(hx, hz, layer_blocks=None):
    return rank_twobasis_schedules(hx, hz, layer_blocks=layer_blocks, policy=FAST_POLICY).schedule


@pytest.fixture(scope="module", params=sorted(SPECS))
def code(request):
    compiled = build_compiled_code(SPECS[request.param])
    hx = np.array(compiled.hx, dtype=np.uint8)
    hz = np.array(compiled.hz, dtype=np.uint8)
    return request.param, hx, hz, _schedule(hx, hz, compiled.layer_blocks)


def frozen_qubit_child(hx, hz):
    """Target code B = A with one data qubit frozen in |+>: drop the Z checks
    on it, add the weight-1 X check; the first qubit that keeps k and whose
    transversal CNOT A -> B is the standard logical CNOT."""

    k = hx.shape[1] - rank2(hx) - rank2(hz)
    for q in range(hx.shape[1]):
        hz_b = hz[hz[:, q] == 0]
        x_q = np.zeros((1, hx.shape[1]), dtype=np.uint8)
        x_q[0, q] = 1
        hx_b = np.vstack([hx, x_q])
        if hx.shape[1] - rank2(hx_b) - rank2(hz_b) != k:
            continue
        try:
            cnot_maps(hx, hz, hx_b, hz_b)
        except ValueError:
            continue
        return q, hx_b, hz_b
    raise AssertionError("no qubit gives a standard-CNOT child")


def _quiet_zero(circuit) -> bool:
    det, obs = circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    return not det.any() and not obs.any()


def _flips(circuit, pairs, gate, qubits):
    """observable flips of a certain ``gate`` error on ``qubits`` placed right
    before the first CX layer whose targets are exactly ``pairs``"""

    out, inserted = stim.Circuit(), False
    for inst in circuit.flattened():
        if not inserted and inst.name == "CX" and [t.value for t in inst.targets_copy()] == pairs:
            out.append(gate, qubits, 1.0)
            inserted = True
        out.append(inst)
    assert inserted
    det, obs = out.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
    assert not det.any()  # a logical operator is undetectable
    return obs[0].astype(np.uint8)


# ------------------------------------------------------------ stages 0-1 equality
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("variant", ["single", "every_round", "loud"])
def test_engine_reproduces_the_stage_0_1_circuits(code, basis, variant):
    _, hx, hz, schedule = code
    point = _point(0.004)
    rb, ra = 2, 3
    every = tuple(range(1, rb + ra)) if variant == "every_round" else None
    scale = 7.0 if variant == "loud" else 1.0
    for name, (n_blocks, cnots) in EXPERIMENT_LAYOUTS.items():
        kinds_old: list = []
        old = build_twobasis_multiblock_circuit(
            hx, hz, schedule, basis=basis, rounds_before=rb, rounds_after=ra, n_blocks=n_blocks, cnots=cnots,
            cnot_rounds=every if cnots else None, cnot_noise_scale=scale, noise_point=point,
            detector_kinds_out=kinds_old,
        )
        at = every if every else (rb,)
        ticks = tuple((r, c, t) for r in at for c, t in cnots)
        kinds_new: list = []
        new = build_logical_program_circuit(
            [ProgramBlock(hx, hz, schedule, prep=basis, readout=basis)] * n_blocks, rounds=rb + ra,
            cnots=ticks, noise_point=point, cnot_noise_scale=scale, detector_kinds_out=kinds_new,
        )
        assert str(new) == str(old), name
        assert kinds_new == kinds_old


# ------------------------------------------------------------ stage 1 GHZ
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_ghz_is_deterministic_with_stabilizer_observables(code, basis):
    _, hx, hz, schedule = code
    k = css_logicals(hx, hz)[0].shape[0]
    quiet = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.0), rounds_before=2,
                                        rounds_after=2, experiments=("ghzidle", "ghz"))
    noisy = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.005), rounds_before=2,
                                        rounds_after=2, experiments=("ghzidle", "ghz"))
    for name in ("ghzidle", "ghz"):
        circuit, kinds = quiet[name][basis]
        assert _quiet_zero(circuit)
        assert circuit.num_observables == (2 * k if basis == "Z" else k)
        assert len(kinds) == circuit.num_detectors
        noisy[name][basis][0].detector_error_model()
    # the same rounds as the baseline (rb + 1 + ra): count the Z-ancilla measurement layers
    n_rounds = lambda c: sum(1 for inst in c.flattened() if inst.name == "M" and len(inst.targets_copy()) == 3 * hz.shape[0])
    assert n_rounds(quiet["ghz"][basis][0]) == n_rounds(quiet["ghzidle"][basis][0]) == 5


def test_ghz_observables_are_the_ghz_stabilizers(code):
    """Z basis: Z0Z1, Z0Z2; X basis: X0X1X2. Checked through their fault
    sensitivity: an X_L on the control before the fan-out is a stabilizer of
    |+> and reaches every block (Z0Z1, Z0Z2 unchanged); an X_L on block 1
    flips Z0Z1 only; a Z_L on block 1 before its CNOT becomes Z0Z1, a GHZ
    stabilizer (X0X1X2 unchanged); a Z_L on the control flips X0X1X2."""
    _, hx, hz, schedule = code
    lx, lz = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx, hz))
    k, n = lx.shape
    stride = n + hz.shape[0] + hx.shape[0]
    quiet = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.0), rounds_before=2,
                                        rounds_after=2, experiments=("ghz",))
    first = [q for i in range(n) for q in (i, stride + i)]  # CNOT 0 -> 1, the first transversal tick
    second = [q for i in range(n) for q in (i, 2 * stride + i)]
    for basis in ("X", "Z"):
        insts = list(quiet["ghz"][basis][0].flattened())
        i1 = next(i for i, inst in enumerate(insts) if inst.name == "CX" and [t.value for t in inst.targets_copy()] == first)
        i2 = next(i for i, inst in enumerate(insts) if inst.name == "CX" and [t.value for t in inst.targets_copy()] == second)
        assert sum(1 for inst in insts[i1:i2] if inst.name == "M") == 1  # one QEC round between the fan-out CNOTs
    pairing = (lx @ lz.T) % 2
    zero = np.zeros(k, dtype=np.uint8)
    for r in range(k):
        sx = [int(q) for q in np.nonzero(lx[r])[0]]
        sz = [int(q) for q in np.nonzero(lz[r])[0]]
        assert np.array_equal(_flips(quiet["ghz"]["Z"][0], first, "X_ERROR", sx), np.concatenate([zero, zero]))
        assert np.array_equal(_flips(quiet["ghz"]["Z"][0], first, "X_ERROR", [stride + q for q in sx]),
                              np.concatenate([pairing[r], zero]))
        assert np.array_equal(_flips(quiet["ghz"]["Z"][0], first, "X_ERROR", [2 * stride + q for q in sx]),
                              np.concatenate([zero, pairing[r]]))
        assert np.array_equal(_flips(quiet["ghz"]["X"][0], first, "Z_ERROR", [stride + q for q in sz]), zero)
        assert np.array_equal(_flips(quiet["ghz"]["X"][0], first, "Z_ERROR", sz), pairing[:, r])


# ------------------------------------------------------------ stage 2 teleportation
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_teleport_is_deterministic_with_k_observables(code, basis):
    _, hx, hz, schedule = code
    k = css_logicals(hx, hz)[0].shape[0]
    quiet = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.0), rounds_before=2,
                                        rounds_after=2, experiments=("idle1", "teleport"))
    noisy = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.005), rounds_before=2,
                                        rounds_after=2, experiments=("idle1", "teleport"))
    for name in ("idle1", "teleport"):
        circuit, kinds = quiet[name][basis]
        assert _quiet_zero(circuit)
        assert circuit.num_observables == k
        assert len(kinds) == circuit.num_detectors
        noisy[name][basis][0].detector_error_model()
    # idle1 is the Fig.2 single-block memory circuit of rb + 1 + ra rounds
    one = build_twobasis_multiblock_circuit(hx, hz, schedule, basis=basis, rounds_before=5, n_blocks=1,
                                            noise_point=_point(0.005))
    assert str(noisy["idle1"][basis][0]) == str(one)


def test_teleport_carries_the_input_logical_to_the_output(code):
    _, hx, hz, schedule = code
    lx, lz = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx, hz))
    k, n = lx.shape
    stride = n + hz.shape[0] + hx.shape[0]
    quiet = program_experiment_circuits([(hx, hz, schedule)], noise_point=_point(0.0), rounds_before=2,
                                        rounds_after=2, experiments=("teleport",))
    bell = [q for i in range(n) for q in (stride + i, 2 * stride + i)]  # CNOT 1 -> 2, the first transversal tick
    tele = [q for i in range(n) for q in (i, stride + i)]
    insts = list(quiet["teleport"]["Z"][0].flattened())
    i1 = next(i for i, inst in enumerate(insts) if inst.name == "CX" and [t.value for t in inst.targets_copy()] == bell)
    i2 = next(i for i, inst in enumerate(insts) if inst.name == "CX" and [t.value for t in inst.targets_copy()] == tele)
    assert sum(1 for inst in insts[i1:i2] if inst.name == "M") == 1  # one QEC round between Bell prep and teleport
    pairing = (lx @ lz.T) % 2
    zero = np.zeros(k, dtype=np.uint8)
    for r in range(k):
        sx = [int(q) for q in np.nonzero(lx[r])[0]]
        sz = [int(q) for q in np.nonzero(lz[r])[0]]
        # an input X_L / Z_L reaches the output
        assert np.array_equal(_flips(quiet["teleport"]["Z"][0], bell, "X_ERROR", sx), pairing[r])
        assert np.array_equal(_flips(quiet["teleport"]["X"][0], bell, "Z_ERROR", sz), pairing[:, r])
        # an error on the Bell pair's output half reaches the output too
        assert np.array_equal(_flips(quiet["teleport"]["Z"][0], bell, "X_ERROR", [2 * stride + q for q in sx]),
                              pairing[r])
        # X_L on block 1 (|+>) before the Bell CNOT is a stabilizer of the input state: no flip
        assert np.array_equal(_flips(quiet["teleport"]["Z"][0], bell, "X_ERROR", [stride + q for q in sx]), zero)
        # Z_L on block 2 (|0>) before it likewise
        assert np.array_equal(_flips(quiet["teleport"]["X"][0], bell, "Z_ERROR", [2 * stride + q for q in sz]),
                              zero)


# ------------------------------------------------------------ stage 3 heterogeneous CNOT
def test_frozen_qubit_child_is_a_cnot_target_and_not_a_control(code):
    _, hx, hz, _ = code
    _, hx_b, hz_b = frozen_qubit_child(hx, hz)
    m = cnot_maps(hx, hz, hx_b, hz_b)  # A -> B is the standard logical CNOT
    k = m["lx"].shape[0]
    assert (m["lx"][:, :k] == np.eye(k, dtype=np.uint8)).all() and (m["lz"][:, :k] == np.eye(k, dtype=np.uint8)).all()
    with pytest.raises(ValueError):
        cnot_maps(hx_b, hz_b, hx, hz)  # B -> A is not (B has an X check A lacks)
    # a child whose logical representatives differ is refused as standard, accepted otherwise
    found = False
    for q in range(hx.shape[1]):
        hz_c = hz[hz[:, q] == 0]
        x_q = np.zeros((1, hx.shape[1]), dtype=np.uint8); x_q[0, q] = 1
        hx_c = np.vstack([hx, x_q])
        if hx.shape[1] - rank2(hx_c) - rank2(hz_c) != k:
            continue
        try:
            cnot_maps(hx, hz, hx_c, hz_c)
        except ValueError:
            cnot_maps(hx, hz, hx_c, hz_c, standard=False)
            found = True
            break
    assert found


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_heterogeneous_cnot_is_deterministic(code, basis):
    _, hx, hz, schedule = code
    _, hx_b, hz_b = frozen_qubit_child(hx, hz)
    codes = [(hx, hz, schedule), (hx_b, hz_b, _schedule(hx_b, hz_b))]
    k = css_logicals(hx, hz)[0].shape[0]
    quiet = program_experiment_circuits(codes, noise_point=_point(0.0), rounds_before=2, rounds_after=2,
                                        experiments=("hidle", "hcnot"))
    noisy = program_experiment_circuits(codes, noise_point=_point(0.005), rounds_before=2, rounds_after=2,
                                        experiments=("hidle", "hcnot"))
    for name in ("hidle", "hcnot"):
        circuit, kinds = quiet[name][basis]
        assert _quiet_zero(circuit)
        assert circuit.num_observables == 2 * k
        assert len(kinds) == circuit.num_detectors
        noisy[name][basis][0].detector_error_model()


def test_heterogeneous_cnot_is_a_logical_cnot(code):
    _, hx, hz, schedule = code
    _, hx_b, hz_b = frozen_qubit_child(hx, hz)
    sch_b = _schedule(hx_b, hz_b)
    lx_a, lz_a = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx, hz))
    lx_b, lz_b = (np.asarray(m, dtype=np.uint8) for m in css_logicals(hx_b, hz_b))
    k, n = lx_a.shape
    stride_a = n + hz.shape[0] + hx.shape[0]
    quiet = program_experiment_circuits([(hx, hz, schedule), (hx_b, hz_b, sch_b)], noise_point=_point(0.0),
                                        rounds_before=2, rounds_after=2, experiments=("hcnot",))
    layer = [q for i in range(n) for q in (i, stride_a + i)]
    for r in range(k):
        sx = [int(q) for q in np.nonzero(lx_a[r])[0]]
        # X_L of A on the control: flips the control's Z_L(A) and spreads to the target's Z_L(B)
        expected = np.concatenate([(lz_a @ lx_a[r]) % 2, (lz_b @ lx_a[r]) % 2])
        assert np.array_equal(_flips(quiet["hcnot"]["Z"][0], layer, "X_ERROR", sx), expected)
        # Z_L of B on the target: kicks back onto the control's X_L(A)
        sz = [stride_a + int(q) for q in np.nonzero(lz_b[r])[0]]
        expected = np.concatenate([(lx_a @ lz_b[r]) % 2, (lx_b @ lz_b[r]) % 2])
        assert np.array_equal(_flips(quiet["hcnot"]["X"][0], layer, "Z_ERROR", sz), expected)
        assert expected[:k].any()  # the kick-back is real, not vacuous


def test_programs_refuse_bad_layouts(code):
    _, hx, hz, schedule = code
    block = ProgramBlock(hx, hz, schedule)
    with pytest.raises(ValueError):
        build_logical_program_circuit([block, ProgramBlock(hx, hz, schedule, last_round=1)], rounds=4,
                                      cnots=((2, 0, 1),), noise_point=_point(0.0))
    with pytest.raises(ValueError):  # a random logical with no frame to fix it
        build_logical_program_circuit([ProgramBlock(hx, hz, schedule, prep="Z", readout="X")], rounds=2,
                                      noise_point=_point(0.0))
