"""sec/multiblock_builder.py: two blocks of one code with a transversal logical
CNOT (Fig.3 stage 0). The single-block, CNOT-free circuit must be the Fig.2
memory circuit instruction for instruction; every detector and observable of
the two-block circuits must be deterministic; and the transversal layer must
act as CNOT on each logical pair (an X_L on the control before it flips the
matching Z observables of both blocks)."""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from omniqec_agentic.construction.build import build_compiled_code
from omniqec_agentic.sec.multiblock_builder import (
    build_twobasis_multiblock_circuit,
    cnot_experiment_circuits,
)
from omniqec_agentic.sec.twobasis_builder import build_twobasis_memory_circuit
from omniqec_agentic.sec.twobasis_rank import SecPolicyConfig, rank_twobasis_schedules
from omniqec_agentic.verification.gf2 import logicals as css_logicals
from omniqec_agentic.verification.noise import NoiseProfile

FAST_CERT = dict(sec_cert_seeds=1, sec_cert_iters_scale=0.0, sec_refine_top=0, sec_refine_iters=1)
FAST_POLICY = SecPolicyConfig(sec_candidates=3, sec_cert_iters=20, **FAST_CERT)
SPECS = {
    "bb_code_b": {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 5]], "B": [[0, 0], [2, 1]]},
    "bb_small": {"family": "bb", "ell": 3, "m": 5, "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]]},
    "lp_abelian_c24": {"family": "lp_abelian", "group": {"factors": [24]},
                       "A": [[[[0], [15]]]], "B": [[[[0], [5], [16]]]]},
    "lp_nonabelian": {"family": "lp_nonabelian", "group": "C3:C8(2)",
                      "A": [[[[0, 0], [0, 1]]]], "B": [[[[0, 0], [1, 0], [2, 5]]]]},
}


def _paper_point(p: float):
    """A uniform_paper_v1 point (p = 0 gives the noiseless circuit; the
    sweep config itself only accepts p > 0)."""
    from omniqec_agentic.verification.noise import NoisePoint

    return NoisePoint(profile=NoiseProfile.UNIFORM_PAPER, value=float(p), axis="p",
                      source_kind="test", source_reference="test")


@pytest.fixture(scope="module", params=sorted(SPECS))
def code(request):
    compiled = build_compiled_code(SPECS[request.param])
    hx = np.array(compiled.hx, dtype=np.uint8)
    hz = np.array(compiled.hz, dtype=np.uint8)
    ranked = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=FAST_POLICY)
    return request.param, hx, hz, ranked.schedule


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_one_block_without_cnot_is_the_fig2_memory_circuit(code, basis):
    _, hx, hz, schedule = code
    point = _paper_point(0.005)
    kinds_ref: list = []
    expected = build_twobasis_memory_circuit(
        hx, hz, schedule, basis=basis, rounds=3, p=0.005, noise_point=point, detector_kinds_out=kinds_ref
    )
    kinds: list = []
    got = build_twobasis_multiblock_circuit(
        hx, hz, schedule, basis=basis, rounds_before=3, n_blocks=1, noise_point=point, detector_kinds_out=kinds
    )
    assert str(got) == str(expected)
    assert kinds == kinds_ref


def _noiseless_all_zero(circuit) -> tuple[bool, bool]:
    det, obs = circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    return (not det.any()), (not obs.any())


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_two_block_circuits_are_deterministic_and_build_a_dem(code, basis):
    _, hx, hz, schedule = code
    k = css_logicals(hx, hz)[0].shape[0]
    quiet = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=2, rounds_after=2)
    noisy = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.005), rounds_before=2, rounds_after=2)
    for name in ("idle", "cnot"):
        circuit, kinds = quiet[name][basis]
        assert _noiseless_all_zero(circuit) == (True, True)
        assert circuit.num_observables == 2 * k
        assert len(kinds) == circuit.num_detectors
        circuit_noisy, kinds_noisy = noisy[name][basis]
        assert kinds_noisy == kinds
        dem = circuit_noisy.detector_error_model()  # raises on a non-deterministic detector
        assert dem.num_detectors == circuit_noisy.num_detectors
    # same detector count with and without the CNOT: every check keeps a reference
    assert quiet["idle"][basis][0].num_detectors == quiet["cnot"][basis][0].num_detectors


def _insert_before_transversal_layer(circuit, n: int, stride: int, gate: str, qubits: list[int]):
    out = stim.Circuit()
    pairs = [q for i in range(n) for q in (i, stride + i)]
    inserted = False
    for inst in circuit.flattened():
        if (not inserted and inst.name == "CX"
                and [t.value for t in inst.targets_copy()] == pairs):
            out.append(gate, qubits, 1.0)  # a certain error, not a gate: stim reports flips vs the noiseless reference
            inserted = True
        out.append(inst)
    assert inserted, "transversal CX layer not found"
    return out


def test_transversal_layer_is_a_logical_cnot_on_each_pair(code):
    _, hx, hz, schedule = code
    lx, lz = css_logicals(hx, hz)
    lx, lz = np.asarray(lx, dtype=np.uint8), np.asarray(lz, dtype=np.uint8)
    k, n = lx.shape
    stride = n + hz.shape[0] + hx.shape[0]
    quiet = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=2, rounds_after=2)
    pairing = (lx @ lz.T) % 2  # anticommutation of X_L row r with Z_L row j
    for r in range(k):
        # Z basis: X_L(row r) on the CONTROL before the CNOT flips Z_L j of the
        # control AND of the target wherever X_L r anticommutes with Z_L j.
        circuit = _insert_before_transversal_layer(
            quiet["cnot"]["Z"][0], n, stride, "X_ERROR", [int(q) for q in np.nonzero(lx[r])[0]]
        )
        det, obs = circuit.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert not det.any()  # a logical operator is undetectable
        expected = np.concatenate([pairing[r], pairing[r]]).astype(bool)
        assert np.array_equal(obs[0], expected)
        # ...and on the TARGET it stays on the target.
        circuit = _insert_before_transversal_layer(
            quiet["cnot"]["Z"][0], n, stride, "X_ERROR", [stride + int(q) for q in np.nonzero(lx[r])[0]]
        )
        det, obs = circuit.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert not det.any()
        assert np.array_equal(obs[0], np.concatenate([np.zeros(k), pairing[r]]).astype(bool))
        # X basis: Z_L(row r) on the TARGET flips X_L of the target and the control.
        circuit = _insert_before_transversal_layer(
            quiet["cnot"]["X"][0], n, stride, "Z_ERROR", [stride + int(q) for q in np.nonzero(lz[r])[0]]
        )
        det, obs = circuit.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert not det.any()
        col = pairing[:, r]
        assert np.array_equal(obs[0], np.concatenate([col, col]).astype(bool))


def test_only_uniform_paper_is_supported(code):
    from omniqec_agentic.verification.noise import NoiseSweepConfig

    _, hx, hz, schedule = code
    si = NoiseSweepConfig(profile=NoiseProfile.SI1000, values=(0.002,)).point(0)
    with pytest.raises(NotImplementedError):
        build_twobasis_multiblock_circuit(
            hx, hz, schedule, basis="Z", rounds_before=2, n_blocks=1, noise_point=si,
        )


def test_cnot_every_round_is_one_logical_cnot_overall(code):
    """11 transversal ticks (odd) act as one logical CNOT; deterministic and DEM-buildable."""
    _, hx, hz, schedule = code
    lx, lz = css_logicals(hx, hz)
    lx, lz = np.asarray(lx, dtype=np.uint8), np.asarray(lz, dtype=np.uint8)
    k, n = lx.shape
    stride = n + hz.shape[0] + hx.shape[0]
    rounds = (1, 2, 3)
    quiet = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=1,
                                     rounds_after=3, cnot_rounds=rounds)
    noisy = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.005), rounds_before=1,
                                     rounds_after=3, cnot_rounds=rounds)
    for basis in ("X", "Z"):
        circuit, _ = quiet["cnot"][basis]
        assert _noiseless_all_zero(circuit) == (True, True)
        noisy["cnot"][basis][0].detector_error_model()
        layers = sum(1 for inst in circuit.flattened() if inst.name == "CX"
                     and [t.value for t in inst.targets_copy()] == [q for i in range(n) for q in (i, stride + i)])
        assert layers == len(rounds)
    pairing = (lx @ lz.T) % 2
    for r in range(k):
        circuit = _insert_before_transversal_layer(
            quiet["cnot"]["Z"][0], n, stride, "X_ERROR", [int(q) for q in np.nonzero(lx[r])[0]]
        )
        det, obs = circuit.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert not det.any()
        assert np.array_equal(obs[0], np.concatenate([pairing[r], pairing[r]]).astype(bool))


def test_cnot_noise_scale_touches_only_the_transversal_layer(code):
    _, hx, hz, schedule = code
    n = hx.shape[1]
    stride = n + hz.shape[0] + hx.shape[0]
    transversal = [q for i in range(n) for q in (i, stride + i)]
    base = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.004), rounds_before=2, rounds_after=2)
    loud = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.004), rounds_before=2,
                                    rounds_after=2, cnot_noise_scale=10.0)
    insts_b = list(base["cnot"]["Z"][0].flattened())
    insts_l = list(loud["cnot"]["Z"][0].flattened())
    assert len(insts_b) == len(insts_l)
    changed = []
    for i, (a, b) in enumerate(zip(insts_b, insts_l)):
        if str(a) != str(b):
            changed.append(i)
            assert a.name == b.name == "DEPOLARIZE2"
            assert [t.value for t in insts_b[i - 1].targets_copy()] == transversal
            assert b.gate_args_copy()[0] == pytest.approx(0.04)
    assert len(changed) == 1
    # idle is untouched by the scale
    assert str(base["idle"]["Z"][0]) == str(loud["idle"]["Z"][0])


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_three_block_fanout_circuits_are_deterministic_and_build_a_dem(code, basis):
    _, hx, hz, schedule = code
    k = css_logicals(hx, hz)[0].shape[0]
    quiet = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=2,
                                     rounds_after=2, experiments=("idle3", "fanout"))
    noisy = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.005), rounds_before=2,
                                     rounds_after=2, experiments=("idle3", "fanout"))
    for name in ("idle3", "fanout"):
        circuit, kinds = quiet[name][basis]
        assert _noiseless_all_zero(circuit) == (True, True)
        assert circuit.num_observables == 3 * k
        assert len(kinds) == circuit.num_detectors
        noisy[name][basis][0].detector_error_model()
    assert quiet["idle3"][basis][0].num_detectors == quiet["fanout"][basis][0].num_detectors
    with pytest.raises(ValueError):
        cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=2,
                                 rounds_after=2, experiments=("teleport",))


def test_fanout_is_two_logical_cnots_from_block_0(code):
    """X_L on block 0 before the fan-out reaches all three blocks; X_L on block 1
    stays put; Z_L on block 2 kicks back onto block 0 only."""
    _, hx, hz, schedule = code
    lx, lz = css_logicals(hx, hz)
    lx, lz = np.asarray(lx, dtype=np.uint8), np.asarray(lz, dtype=np.uint8)
    k, n = lx.shape
    stride = n + hz.shape[0] + hx.shape[0]
    quiet = cnot_experiment_circuits(hx, hz, schedule, noise_point=_paper_point(0.0), rounds_before=2,
                                     rounds_after=2, experiments=("fanout",))
    second = [q for i in range(n) for q in (i, 2 * stride + i)]
    assert any(inst.name == "CX" and [t.value for t in inst.targets_copy()] == second
               for inst in quiet["fanout"]["Z"][0].flattened())
    pairing = (lx @ lz.T) % 2
    zero = np.zeros(k, dtype=np.uint8)

    def flips(basis, gate, qubits):
        circuit = _insert_before_transversal_layer(quiet["fanout"][basis][0], n, stride, gate, qubits)
        det, obs = circuit.compile_detector_sampler(seed=1).sample(4, separate_observables=True)
        assert not det.any()
        return obs[0]

    for r in range(k):
        support_x = [int(q) for q in np.nonzero(lx[r])[0]]
        support_z = [int(q) for q in np.nonzero(lz[r])[0]]
        row, col = pairing[r], pairing[:, r]
        assert np.array_equal(flips("Z", "X_ERROR", support_x), np.concatenate([row, row, row]).astype(bool))
        assert np.array_equal(flips("Z", "X_ERROR", [stride + q for q in support_x]),
                              np.concatenate([zero, row, zero]).astype(bool))
        assert np.array_equal(flips("X", "Z_ERROR", [2 * stride + q for q in support_z]),
                              np.concatenate([col, zero, col]).astype(bool))
        assert np.array_equal(flips("X", "Z_ERROR", [stride + q for q in support_z]),
                              np.concatenate([col, col, zero]).astype(bool))
