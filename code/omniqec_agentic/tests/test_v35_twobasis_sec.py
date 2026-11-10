"""v3.5 SEC policy ``twobasis_nonint_v1`` (02 §9.4): layers, circuit shape /
detectors, noise fidelity, LRC byte parity after the emitter refactor,
certification selection / determinism, policy switch + synth cache, and an
end-to-end BPOSD smoke."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation import BASELINE_BB_SPEC
from omniqec_agentic.sec import build_canonical_schedule, split_sectors
from omniqec_agentic.sec.audits import audit_twobasis_static
from omniqec_agentic.sec.coloring import color_bipartite_multigraph, counter_permutation
from omniqec_agentic.sec.layers import BlockStructure, build_basis_layers, build_layer_sets
from omniqec_agentic.sec.stim_builder import build_lrc_memory_circuit
from omniqec_agentic.sec.twobasis_builder import (
    build_twobasis_memory_circuit,
    build_twobasis_memory_pair,
)
from omniqec_agentic.sec.twobasis_rank import (
    DEFAULT_SEC_POLICY,
    LRC_SEC_POLICY,
    SecPolicyConfig,
    TwoBasisSchedule,
    basis_residual_suffixes,
    certify_basis_order,
    rank_twobasis_schedules,
)
from omniqec_agentic.verification.gf2 import logicals
from omniqec_agentic.verification.noise import NoiseProfile, NoiseSweepConfig

# A copy of sec/stim_builder.py from before the v3.5 SEC refactor (a local snapshot, not in the
# repository); the byte-parity test below skips when OMNIQEC_PRE_V35SEC_STIM_BUILDER does not name one.
PRE_V35SEC = Path(os.environ.get("OMNIQEC_PRE_V35SEC_STIM_BUILDER", "") or "pre_v35sec_snapshot_not_configured/stim_builder.py")
P = 0.003
# test-speed policy: v3.6 staged certification switched down to the v3.5
# single-seed reading (one seed, no kernel scaling, no refine) so circuit /
# noise / cache tests stay fast; the staged selector itself is covered in
# test_v36_sec_catalog_coverage.py and the selector tests below.
FAST_CERT = dict(sec_cert_seeds=1, sec_cert_iters_scale=0.0, sec_refine_top=0, sec_refine_iters=1)
FAST_POLICY = SecPolicyConfig(sec_candidates=3, sec_cert_iters=20, **FAST_CERT)

CIRCULANT = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
D6_SPEC = {
    "family": "lp_nonabelian", "group": "D6",
    "A": [[[[0, 0], [4, 0]]]],
    "B": [[[[0, 0], [2, 1], [3, 0], [4, 0], [4, 1], [1, 0]]]],
}
FAMILY_SPECS = {
    "bb": {"family": "bb", "ell": 3, "m": 5,
           "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]]},
    "hgp": {"family": "hgp", "seeds": {"H1": CIRCULANT, "H2": CIRCULANT}},
    "lp_abelian": {"family": "lp_abelian", "group": {"factors": [3, 3]},
                   "A": [[[[2, 0], [0, 0]], [[1, 0], [0, 1]]]],
                   "B": [[[[0, 2], [1, 2], [0, 0]]]]},
    "lp_nonabelian": D6_SPEC,
}


def _mats(compiled):
    return np.array(compiled.hx, dtype=np.uint8), np.array(compiled.hz, dtype=np.uint8)


def _max_degree(h):
    return max(int(h.sum(axis=1).max()), int(h.sum(axis=0).max()))


def _instructions(circuit):
    out = []
    for raw in str(circuit).splitlines():
        line = raw.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        name, _, args = head.partition("(")
        targets = rest.split() if rest else []
        out.append((name, args.rstrip(")"), targets))
    return out


def _bb_layer_blocks_by_hand(spec):
    """Architect encoding built from the monomial formula (independent of
    construction/bb.py::bb_layer_blocks)."""
    ell, m = int(spec["ell"]), int(spec["m"])
    g = ell * m

    def perm_fwd(a, b):
        return [(((t // m) - a) % ell) * m + (((t % m) - b) % m) for t in range(g)]

    def perm_t(a, b):
        return [(((t // m) + a) % ell) * m + (((t % m) + b) % m) for t in range(g)]

    x_terms = [[0, 0, perm_fwd(a, b)] for a, b in spec["A"]] + [
        [0, 1, perm_fwd(a, b)] for a, b in spec["B"]
    ]
    z_terms = [[0, 0, perm_t(a, b)] for a, b in spec["B"]] + [
        [0, 1, perm_t(a, b)] for a, b in spec["A"]
    ]
    return {
        "schema": "layer_blocks_v1",
        "block_size": g,
        "X": {"n_check_blocks": 1, "n_data_blocks": 2, "terms": x_terms},
        "Z": {"n_check_blocks": 1, "n_data_blocks": 2, "terms": z_terms},
    }


def _reconstruct(layer_blocks, basis, shape):
    blocks = BlockStructure.from_layer_blocks(layer_blocks, basis)
    h = np.zeros(shape, dtype=np.uint8)
    for index in range(len(blocks.terms)):
        for r, c in blocks.term_edges(index):
            assert h[r, c] == 0, "overlapping terms"
            h[r, c] = 1
    return h


@pytest.fixture(scope="module")
def bb72():
    compiled = build_compiled_code(BASELINE_BB_SPEC)
    hx, hz = _mats(compiled)
    return compiled, hx, hz


@pytest.fixture(scope="module")
def bb72_ranked(bb72):
    compiled, hx, hz = bb72
    return rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=FAST_POLICY)


# ------------------------------------------------------------------ layers


def test_hand_built_bb72_layer_blocks_reproduce_matrices_and_colour_to_w(bb72):
    compiled, hx, hz = bb72
    blob = _bb_layer_blocks_by_hand(BASELINE_BB_SPEC)
    assert np.array_equal(_reconstruct(blob, "X", hx.shape), hx)
    assert np.array_equal(_reconstruct(blob, "Z", hz.shape), hz)
    json.dumps(blob)  # JSON-safe
    layers = build_layer_sets(hx, hz, layer_blocks=blob)
    for basis, h in (("X", hx), ("Z", hz)):
        assert len(layers[basis]) == _max_degree(h) == 6
        for layer in layers[basis]:
            assert len({c for c, _ in layer}) == len(layer)
            assert len({q for _, q in layer}) == len(layer)
        flat = sorted(e for layer in layers[basis] for e in layer)
        assert flat == sorted((int(r), int(c)) for r, c in zip(*np.nonzero(h)))
    # the compiled code carries an equivalent blob (construction side)
    assert compiled.layer_blocks["schema"] == "layer_blocks_v1"
    assert np.array_equal(_reconstruct(compiled.layer_blocks, "X", hx.shape), hx)
    assert build_layer_sets(hx, hz, layer_blocks=compiled.layer_blocks) == layers


def test_structured_layers_small_lp_and_hgp_fallback():
    for family in ("lp_nonabelian", "lp_abelian"):
        compiled = build_compiled_code(FAMILY_SPECS[family])
        hx, hz = _mats(compiled)
        assert compiled.layer_blocks is not None
        for basis, h in (("X", hx), ("Z", hz)):
            blocks = BlockStructure.from_layer_blocks(compiled.layer_blocks, basis)
            assert blocks is not None
            layers = build_basis_layers(h, blocks=blocks)
            assert len(layers) == _max_degree(h)  # == check weight w for these shapes
            assert len(layers) == int(h.sum(axis=1).max())
            for layer in layers:
                assert len({c for c, _ in layer}) == len(layer)
                assert len({q for _, q in layer}) == len(layer)
    hgp = build_compiled_code(FAMILY_SPECS["hgp"])
    assert hgp.layer_blocks is None
    hx, hz = _mats(hgp)
    layers = build_layer_sets(hx, hz, layer_blocks=None)
    assert len(layers["X"]) == _max_degree(hx) and len(layers["Z"]) == _max_degree(hz)
    assert BlockStructure.from_layer_blocks(None, "X") is None
    assert BlockStructure.from_layer_blocks({"schema": "other"}, "X") is None


def test_multigraph_colouring_handles_parallel_edges_and_is_deterministic():
    edges = [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1), (0, 1), (1, 1), (1, 0)]
    colors = color_bipartite_multigraph(edges, n_left=2, n_right=2)
    assert len(colors) == len(edges) and max(colors) + 1 == 6  # Δ = 6 (check 0)
    for node_side in (0, 1):
        for node in range(2):
            used = [colors[i] for i, e in enumerate(edges) if e[node_side] == node]
            assert len(set(used)) == len(used)
    assert colors == color_bipartite_multigraph(edges, n_left=2, n_right=2)
    assert color_bipartite_multigraph([], n_left=1, n_right=1) == []


def test_layers_assert_coverage_and_matching():
    h = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    bad = {
        "schema": "layer_blocks_v1", "block_size": 2,
        "X": {"n_check_blocks": 1, "n_data_blocks": 1, "terms": [[0, 0, [0, 1]]]},
    }
    with pytest.raises(AssertionError):
        build_basis_layers(h, blocks=BlockStructure.from_layer_blocks(bad, "X"))
    with pytest.raises(ValueError):
        BlockStructure.from_layer_blocks({**bad, "X": {**bad["X"], "terms": [[0, 0, [0, 0]]]}}, "X")


# ------------------------------------------------------ circuit structure


def _expected_detectors(rounds, m_w, m_opp):
    return rounds * m_w + (rounds - 1) * m_opp + m_w


@pytest.mark.parametrize("family", sorted(FAMILY_SPECS))
def test_four_family_circuit_shape_and_detectors(family):
    compiled = build_compiled_code(FAMILY_SPECS[family])
    hx, hz = _mats(compiled)
    ranked = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=FAST_POLICY)
    schedule = ranked.schedule
    assert isinstance(schedule, TwoBasisSchedule)
    lx, lz = logicals(hx, hz)
    n, m_x, m_z = hx.shape[1], hx.shape[0], hz.shape[0]
    n_layers = len(schedule.layers["X"]) + len(schedule.layers["Z"])
    assert schedule.depth == n_layers + 4
    for rounds in (1, 3):
        for basis in ("X", "Z"):
            m_w, m_opp = (m_x, m_z) if basis == "X" else (m_z, m_x)
            k = np.asarray(lx if basis == "X" else lz).shape[0]
            quiet = build_twobasis_memory_circuit(hx, hz, schedule, basis=basis, rounds=rounds, p=0)
            assert not quiet.compile_detector_sampler().sample(8).any()
            noisy = build_twobasis_memory_circuit(hx, hz, schedule, basis=basis, rounds=rounds, p=P)
            noisy.detector_error_model(decompose_errors=False)
            assert noisy.num_ticks == rounds * (n_layers + 4)
            assert noisy.num_measurements == rounds * (m_x + m_z) + n
            assert noisy.num_detectors == _expected_detectors(rounds, m_w, m_opp)
            assert noisy.num_observables == k
            audit = audit_twobasis_static(noisy, hx, hz, schedule, rounds=rounds)
            assert audit["valid"], audit
            # per-check CX count and no ancilla-ancilla CX, parsed from the circuit
            per_check = {}
            for name, _, targets in _instructions(noisy):
                if name != "CX":
                    continue
                qs = [int(t) for t in targets]
                for a, b in zip(qs[0::2], qs[1::2]):
                    assert not (a >= n and b >= n)
                    key = ("X", a - n - m_z) if a >= n + m_z else ("Z", b - n)
                    per_check[key] = per_check.get(key, 0) + 1
            for c in range(m_x):
                assert per_check[("X", c)] == rounds * int(hx[c].sum())
            for c in range(m_z):
                assert per_check[("Z", c)] == rounds * int(hz[c].sum())


def test_round_structure_x_half_then_z_half(bb72, bb72_ranked):
    _, hx, hz = bb72
    schedule = bb72_ranked.schedule
    n, m_z = hx.shape[1], hz.shape[0]
    # the structure assertions below spell out the PRODUCTION (uniform_paper_v1)
    # lowering -- DEPOLARIZE1 after every reset; omitting the point would fall
    # back to the legacy uniform emission (X_ERROR after R).
    circuit = build_twobasis_memory_circuit(
        hx, hz, schedule, basis="Z", rounds=2, p=P,
        noise_point=NoiseSweepConfig.uniform((P,)).point(0),
    )
    segs = [[]]
    for name, _, targets in _instructions(circuit):
        if name == "TICK":
            segs.append([])
        else:
            segs[-1].append((name, targets))
    lx, lz = len(schedule.layers["X"]), len(schedule.layers["Z"])
    assert [name for name, _ in segs[0]] == ["R", "DEPOLARIZE1"]  # prep data only
    assert all(int(t) < n for t in segs[0][0][1])
    for r in range(2):
        base = 1 + r * (lx + lz + 4)
        assert segs[base][0][0] == "RX"  # A
        for i in range(lx):  # X layers: CX xanc -> data
            names = [name for name, _ in segs[base + 1 + i]]
            assert names == ["CX", "DEPOLARIZE2"]
            qs = [int(t) for t in segs[base + 1 + i][0][1]]
            assert all(a >= n + m_z and b < n for a, b in zip(qs[0::2], qs[1::2]))
        assert [name for name, _ in segs[base + 1 + lx]][:2] == ["Z_ERROR", "MX"]  # B
        assert segs[base + 2 + lx][0][0] == "R"  # C
        for j in range(lz):
            qs = [int(t) for t in segs[base + 3 + lx + j][0][1]]
            assert all(a < n and n <= b < n + m_z for a, b in zip(qs[0::2], qs[1::2]))
        assert [name for name, _ in segs[base + 3 + lx + lz]][:2] == ["X_ERROR", "M"]  # D
    with pytest.raises(ValueError):
        build_twobasis_memory_circuit(hx, hz, schedule, basis="X", rounds=0, p=P)


def test_audit_catches_broken_circuits(bb72, bb72_ranked):
    _, hx, hz = bb72
    schedule = bb72_ranked.schedule
    circuit = build_twobasis_memory_circuit(hx, hz, schedule, basis="X", rounds=2, p=1e-3)
    assert audit_twobasis_static(circuit, hx, hz, schedule, rounds=2)["valid"]
    wrong_rounds = audit_twobasis_static(circuit, hx, hz, schedule, rounds=3)
    assert not wrong_rounds["valid"]
    assert "tanner_edge_cycle_count_mismatch" in wrong_rounds["failures"]
    assert "layer_count_mismatch" in wrong_rounds["failures"]
    assert any(f.startswith("detector_count:") for f in wrong_rounds["failures"])
    # an LRC circuit is not a two-basis circuit (interleaved layers, single-basis detectors)
    compiled = build_compiled_code(BASELINE_BB_SPEC)
    lrc = build_canonical_schedule(split_sectors(hx, hz, compiled.sector_sizes))
    lrc_circuit = build_lrc_memory_circuit(hx, hz, lrc, basis="X", rounds=2, p=1e-3)
    assert not audit_twobasis_static(lrc_circuit, hx, hz, schedule, rounds=2)["valid"]


# --------------------------------------------------------- noise fidelity


def test_uniform_paper_noise_on_twobasis_timeline(bb72, bb72_ranked):
    _, hx, hz = bb72
    schedule = bb72_ranked.schedule
    point = NoiseSweepConfig.uniform((P,)).point(0)
    n = hx.shape[1]
    for basis in ("X", "Z"):
        circuit = build_twobasis_memory_circuit(
            hx, hz, schedule, basis=basis, rounds=3, p=P, noise_point=point
        )
        ins = _instructions(circuit)
        noise = [(i, name) for i, (name, _, _) in enumerate(ins)
                 if name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR")]
        assert all(ins[i][1] == str(P) for i, _ in noise)
        resets = [i for i, (name, _, _) in enumerate(ins) if name in ("R", "RX")]
        dep1 = [i for i, name in noise if name == "DEPOLARIZE1"]
        assert dep1 == [i + 1 for i in resets]
        assert all(ins[i][2] == ins[i - 1][2] for i in dep1)  # same targets as the reset
        cxs = [i for i, (name, _, _) in enumerate(ins) if name == "CX"]
        assert [i for i, name in noise if name == "DEPOLARIZE2"] == [i + 1 for i in cxs]
        assert all(ins[i + 1][2] == ins[i][2] for i in cxs)
        measures = [i for i, (name, _, _) in enumerate(ins) if name in ("M", "MX")]
        flips = {i for i, name in noise if name in ("X_ERROR", "Z_ERROR")}
        assert flips == {i - 1 for i in measures}
        for i in measures:
            assert ins[i - 1][0] == ("X_ERROR" if ins[i][0] == "M" else "Z_ERROR")
            assert ins[i - 1][2] == ins[i][2]
        final = ins[measures[-1]]
        assert final[0] == ("MX" if basis == "X" else "M") and len(final[2]) == n
        assert len(noise) == len(resets) + len(cxs) + len(measures)  # nothing else
        assert circuit.num_detectors == build_twobasis_memory_circuit(
            hx, hz, schedule, basis=basis, rounds=3, p=P
        ).num_detectors


def test_legacy_si1000_helios_smoke_on_twobasis_timeline(bb72, bb72_ranked):
    _, hx, hz = bb72
    schedule = bb72_ranked.schedule
    points = {
        "legacy": (NoiseSweepConfig.uniform_legacy((P,)).point(0), {}),
        "si1000": (NoiseSweepConfig.si1000().point(0), {}),
        "helios": (NoiseSweepConfig.helios().point(0), {"approximate_disjoint_errors": 1e-3}),
    }
    for basis in ("X", "Z"):
        expected = build_twobasis_memory_circuit(hx, hz, schedule, basis=basis, rounds=2, p=P)
        for name, (point, kwargs) in points.items():
            circuit = build_twobasis_memory_circuit(
                hx, hz, schedule, basis=basis, rounds=2, p=P, noise_point=point
            )
            dem = circuit.detector_error_model(decompose_errors=False, **kwargs)
            assert dem.num_detectors == expected.num_detectors, (name, basis)
            assert dem.num_observables == expected.num_observables
            assert circuit.num_measurements == expected.num_measurements
            names = {ins[0] for ins in _instructions(circuit)}
            if name == "si1000":
                assert "CZ" in names and "H" in names and "CX" not in names
            elif name == "helios":
                assert "SQRT_ZZ" in names and "PAULI_CHANNEL_2" in names
            else:
                assert str(circuit) == str(expected)
    with pytest.raises(ValueError):
        build_twobasis_memory_circuit(
            hx, hz, schedule, basis="X", rounds=2, p=P,
            noise_point=points["si1000"][0], saturate_noise=True,
        )
    saturated = build_twobasis_memory_circuit(
        hx, hz, schedule, basis="X", rounds=2, p=1e-4, saturate_noise=True
    )
    saturated.detector_error_model(decompose_errors=False)


# ------------------------------------------------- LRC byte parity (refactor)


def _load_pre_v35sec_builder():
    if not PRE_V35SEC.exists():
        pytest.skip("pre_v35sec snapshot not available")
    name = "omniqec_agentic.sec._pre_v35sec_stim_builder"
    spec = importlib.util.spec_from_file_location(name, str(PRE_V35SEC))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_lrc_memory_circuit


def test_lrc_emitter_refactor_is_byte_identical_to_snapshot():
    old = _load_pre_v35sec_builder()
    points = {
        "none": None,
        "uniform_paper": NoiseSweepConfig.uniform((P,)).point(0),
        "si1000": NoiseSweepConfig.si1000().point(0),
        "helios": NoiseSweepConfig.helios().point(0),
    }
    for spec in (BASELINE_BB_SPEC, D6_SPEC):
        compiled = build_compiled_code(spec)
        hx, hz = _mats(compiled)
        schedule = build_canonical_schedule(split_sectors(hx, hz, compiled.sector_sizes))
        for basis in ("X", "Z"):
            for name, point in points.items():
                new = build_lrc_memory_circuit(
                    hx, hz, schedule, basis=basis, rounds=3, p=P, noise_point=point
                )
                ref = old(hx, hz, schedule, basis=basis, rounds=3, p=P, noise_point=point)
                assert str(new) == str(ref), (spec["family"], basis, name)
            new = build_lrc_memory_circuit(
                hx, hz, schedule, basis=basis, rounds=3, p=P, saturate_noise=True
            )
            ref = old(hx, hz, schedule, basis=basis, rounds=3, p=P, saturate_noise=True)
            assert str(new) == str(ref), (spec["family"], basis, "saturate")


# ------------------------------------------------------------ certification


def test_residual_suffixes_follow_layer_order():
    layers = [[(0, 5)], [(0, 2)], [(0, 7)]]
    suffixes = basis_residual_suffixes(layers, m_w=2, n_data=8)
    assert len(suffixes) == 2 and suffixes[1] == []
    assert [sorted(np.nonzero(v)[0].tolist()) for v in suffixes[0]] == [[2, 7], [7]]


def test_delta_one_order_rejected_and_catastrophic_when_only_canonical():
    """[[4,1]] toy: X check 1111, Z checks 1100/0011. The canonical X order
    (q0,q1,q2,q3) leaves the suffix 0011 = a logical X undetected by hz."""
    hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    hz = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    canonical = build_layer_sets(hx, hz, layer_blocks=None)["X"]
    assert [q for layer in canonical for _, q in layer] == [0, 1, 2, 3]
    suffixes = basis_residual_suffixes(canonical, m_w=1, n_data=4)
    _, hit = certify_basis_order(hx, hz, kind="X", suffixes_w=suffixes, iters=10, seed=0)
    assert hit
    good = [canonical[i] for i in (0, 2, 1, 3)]  # q0,q2,q1,q3: every suffix detected
    _, hit = certify_basis_order(
        hx, hz, kind="X", suffixes_w=basis_residual_suffixes(good, m_w=1, n_data=4),
        iters=10, seed=0,
    )
    assert not hit
    ranked = rank_twobasis_schedules(
        hx, hz, layer_blocks=None, policy=SecPolicyConfig(sec_candidates=16, sec_cert_iters=10)
    )
    assert not ranked.catastrophic
    assert ranked.schedule.candidate_index_x != 0
    assert ranked.metadata["per_candidate"]["X"][0][3] is True  # [index, d1, d_refined, delta_one]
    assert ranked.delta_one_eliminated >= 1
    chosen_x = ranked.schedule.candidate_index_x
    assert ranked.metadata["per_candidate"]["X"][chosen_x][3] is False
    only_canonical = rank_twobasis_schedules(
        hx, hz, layer_blocks=None, policy=SecPolicyConfig(sec_candidates=1, sec_cert_iters=10)
    )
    assert only_canonical.catastrophic
    assert only_canonical.schedule.candidate_index_x == 0
    assert only_canonical.d_circ_ext_x is None and only_canonical.d_circ_ext_z is None
    assert only_canonical.metadata["reason"] == "all_candidates_have_single_fault_logical"


def test_selector_picks_largest_d_ext_and_canonical_on_ties(monkeypatch):
    """v3.6 staged selector on a deterministic fake ISD: the stage-1 value of
    a candidate is the MIN over its seeds, the top ``sec_refine_top`` are
    refined with fresh seeds, the largest refined value wins and ties go to
    the smaller index; delta-one candidates never reach the ISD."""

    import omniqec_agentic.sec.twobasis_rank as tr

    compiled = build_compiled_code(BASELINE_BB_SPEC)
    hx, hz = _mats(compiled)
    seed0, side = {"X": 0, "Z": 1}, 100
    # per basis: {candidate index: (stage-1 per-seed values, refine per-seed values)}
    table = {
        "X": {0: ([3, 3], [3, 3]), 1: ([5, 5], [5, 5]), 2: ([9, 9], [9, 9]), 3: ([5, 5], [5, 5])},
        "Z": {0: ([4, 4], [4, 4]), 1: ([4, 4], [4, 4]), 2: ([2, 2], [2, 2]), 3: ([4, 4], [4, 4])},
    }
    calls: list[tuple[str, int, int, int]] = []

    # Map the rng back to its seed: default_rng(seed).integers(...) is a pure
    # function of the seed, so tabulate every seed the ranker can issue.
    seeds = 2  # sec_cert_seeds
    seed_of_draw: dict[int, tuple[str, int, int]] = {}
    for kind in ("X", "Z"):
        for index in range(4):
            for s in range(2 * seeds):
                seed = side + index * 7 + seed0[kind] + 1000 * s
                seed_of_draw[int(np.random.default_rng(seed).integers(0, 2**31))] = (kind, index, s)

    def fake_proxy_valued(kernel, logical_parity, iters, rng):
        draw = int(rng.integers(0, 2**31))
        kind, index, s = seed_of_draw[draw]
        calls.append((kind, index, s, int(iters)))
        stage1, refine = table[kind][index]
        return (stage1[s] if s < seeds else refine[s - seeds])

    monkeypatch.setattr(tr, "_proxy_side", fake_proxy_valued)
    # Z candidate 0 is made delta-one by monkeypatching the exact scan.
    real_hit = tr._delta_one_hit

    def fake_hit(context, suffixes_w):
        if context.kind == "Z" and fake_hit.z_calls == 0:
            fake_hit.z_calls += 1
            return True
        if context.kind == "Z":
            fake_hit.z_calls += 1
        return False

    fake_hit.z_calls = 0
    monkeypatch.setattr(tr, "_delta_one_hit", fake_hit)
    policy = SecPolicyConfig(
        sec_candidates=4, sec_cert_iters=7, sec_cert_seed=side, sec_cert_seeds=seeds,
        sec_cert_iters_scale=0.0, sec_refine_top=2, sec_refine_iters=11,
    )
    ranked = tr.rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks, policy=policy, d_code_proxy=6
    )
    # X: stage-1 order 2 (9) > 1 (5) = 3 (5) > 0 (3); top-2 = {2, 1}; 9 is
    # clamped to the proxy 6 in the record, so the winner is index 2.
    assert ranked.schedule.candidate_index_x == 2
    assert ranked.d_circ_ext_x == 6 and ranked.metadata["d_ext_x_unclamped"] == 9
    assert ranked.metadata["clamped_x"] is True and ranked.metadata["schedule_degraded_x"] is False
    # Z: canonical was delta-one; 1 and 3 tie at 4 -> 1 (smaller index).
    assert ranked.schedule.candidate_index_z == 1
    assert ranked.d_circ_ext_z == 4 and ranked.metadata["schedule_degraded_z"] is True
    assert ranked.delta_one_eliminated == 1 and not ranked.catastrophic
    assert ranked.tau_a is None and ranked.stage3_group == 1
    md = ranked.metadata
    assert md["sec_policy_id"] == "twobasis_nonint_v1"
    assert md["candidates_evaluated"] == 8 and md["candidates"] == 4 and md["cert_iters"] == 7
    assert md["cert_seeds"] == 2 and md["refine_top"] == 2 and md["refine_iters"] == 11
    assert md["d_code_proxy"] == 6 and md["chosen_index_x"] == 2 and md["chosen_index_z"] == 1
    assert md["per_candidate"]["X"] == [[0, 3, None, False], [1, 5, 5, False], [2, 9, 9, False], [3, 5, None, False]]
    assert md["per_candidate"]["Z"] == [[0, None, None, True], [1, 4, 4, False], [2, 2, None, False], [3, 4, 4, False]]
    assert md["refined_indices"] == {"X": [2, 1], "Z": [1, 3]}
    assert md["converged_x"] is True and md["converged_z"] is True
    # stage 1: 7 iterations, seeds s=0,1; refine: 11 iterations, seeds s=2,3
    stage1_calls = [c for c in calls if c[3] == 7]
    refine_calls = [c for c in calls if c[3] == 11]
    assert sorted({(k, i) for k, i, _, _ in stage1_calls}) == [("X", 0), ("X", 1), ("X", 2), ("X", 3), ("Z", 1), ("Z", 2), ("Z", 3)]
    assert {s for _, _, s, _ in stage1_calls} == {0, 1}
    assert sorted({(k, i) for k, i, _, _ in refine_calls}) == [("X", 1), ("X", 2), ("Z", 1), ("Z", 3)]
    assert {s for _, _, s, _ in refine_calls} == {2, 3}
    assert md["isd_runs"] == len(calls) == 7 * 2 + 4 * 2
    # the chosen orders are the counter permutations of the canonical layers
    base = build_layer_sets(hx, hz, layer_blocks=compiled.layer_blocks)
    assert ranked.schedule.layers["X"] == [base["X"][j] for j in counter_permutation(6, candidate=2, slot=0)]
    assert ranked.schedule.layers["Z"] == [base["Z"][j] for j in counter_permutation(6, candidate=1, slot=1)]
    sched_md = ranked.schedule.to_metadata()
    assert (sched_md["cert_seeds"], sched_md["refine_iters"]) == (2, 11)
    assert sched_md["converged_x"] is True and sched_md["d_ext_x"] == 6


def test_certification_real_code_selection_deterministic_and_metadata(bb72):
    compiled, hx, hz = bb72
    policy = SecPolicyConfig(
        sec_candidates=4, sec_cert_iters=40, sec_cert_seeds=2, sec_cert_iters_scale=0.0,
        sec_refine_top=2, sec_refine_iters=60,
    )
    first = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=policy)
    second = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=policy)
    assert first == second
    for kind, chosen in (("X", first.schedule.candidate_index_x), ("Z", first.schedule.candidate_index_z)):
        rows = first.metadata["per_candidate"][kind]
        refined = [(d_ref, i) for i, d1, d_ref, hit in rows if not hit and d_ref is not None]
        assert len(refined) == 2 and first.metadata["refined_indices"][kind] == [i for _, i in refined]
        best = max(d for d, _ in refined)
        assert rows[chosen][2] == best
        assert chosen == min(i for d, i in refined if d == best)
        # the refined value is the min over both stages (both are upper bounds)
        for i, d1, d_ref, hit in rows:
            if d_ref is not None:
                assert d_ref <= d1
    md = first.schedule.to_metadata()
    assert md["sec_policy_id"] == "twobasis_nonint_v1" and md["depth"] == 16
    assert md["layers_x"] == 6 and md["layers_z"] == 6
    assert md["d_ext_x"] == first.d_circ_ext_x and md["d_ext_z"] == first.d_circ_ext_z
    assert md["candidates"] == 4 and md["cert_iters"] == 40
    assert md["cert_seeds"] == 2 and md["refine_iters"] == 60
    json.dumps(md)
    json.dumps(first.metadata)
    assert pickle.loads(pickle.dumps(first)) == first
    from omniqec_agentic.evaluation.evidence import sec_ext_evidence

    evidence = sec_ext_evidence(first)
    assert evidence["status"] == "probe_upper_bound"
    assert evidence["d_circ_ext"] == min(first.d_circ_ext_x, first.d_circ_ext_z)
    assert evidence["tau_a"] is None and evidence["stage3_group"] == 1


# ------------------------------------------------- policy switch + cache


def test_memory_experiment_pair_policy_switch_and_synth_cache(bb72, tmp_path):
    from omniqec_agentic.verification import circuit as circuit_mod

    compiled, hx, hz = bb72
    fast = SecPolicyConfig(sec_candidates=2, sec_cert_iters=20, **FAST_CERT)
    before = len(circuit_mod._SYNTH_CACHE)
    pair = circuit_mod.build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, sec_policy=fast, layer_blocks=compiled.layer_blocks,
        synth_cache_dir=str(tmp_path), d_hint=6,
    )
    assert pair.sec_policy_id == "twobasis_nonint_v1"
    assert isinstance(pair.x_memory.schedule, TwoBasisSchedule)
    for exp in (pair.x_memory, pair.z_memory):
        md = exp.to_metadata()
        assert md["schedule"]["sec_policy_id"] == "twobasis_nonint_v1"
        assert md["schedule"]["depth"] == 16 and md["schedule"]["layers_x"] == 6
        assert md["num_detectors"] == exp.circuit.num_detectors
    assert pair.ranked.metadata["d_code_proxy"] == 6
    assert len(circuit_mod._SYNTH_CACHE) == before + 1
    assert list(tmp_path.glob("synth_*.pkl"))
    # same policy, no layer_blocks -> different key (fallback colouring path)
    circuit_mod.build_memory_experiment_pair(hx, hz, p=P, rounds=2, sec_policy=fast)
    assert len(circuit_mod._SYNTH_CACHE) == before + 2
    # different K -> different key; identical call -> cache hit
    circuit_mod.build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, sec_policy=SecPolicyConfig(sec_candidates=1, sec_cert_iters=20, **FAST_CERT),
        layer_blocks=compiled.layer_blocks,
    )
    assert len(circuit_mod._SYNTH_CACHE) == before + 3
    circuit_mod.build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, sec_policy=fast, layer_blocks=compiled.layer_blocks,
        synth_cache_dir=str(tmp_path),
    )
    assert len(circuit_mod._SYNTH_CACHE) == before + 3
    # LRC policy: historical path, needs sector_sizes, separate cache entry
    with pytest.raises(ValueError):
        circuit_mod.build_memory_experiment_pair(hx, hz, p=P, rounds=2, sec_policy=LRC_SEC_POLICY)
    lrc = circuit_mod.build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, sec_policy=LRC_SEC_POLICY, sector_sizes=compiled.sector_sizes
    )
    assert lrc.sec_policy_id == "lrc_ranked_v1"
    assert lrc.x_memory.to_metadata()["schedule"]["sec_policy_id"] == "lrc_ranked_v1"
    assert len(circuit_mod._SYNTH_CACHE) == before + 4
    assert circuit_mod._LRC_SYNTH_CACHE is circuit_mod._SYNTH_CACHE
    assert DEFAULT_SEC_POLICY.sec_policy_id == "twobasis_nonint_v1"
    assert (LRC_SEC_POLICY.sec_candidates, LRC_SEC_POLICY.sec_cert_iters) == (
        circuit_mod.LRC_SEAM_CANDIDATES, circuit_mod.LRC_SEAM_ISD_ITERS
    )


def test_memory_experiment_pair_lowered_profiles_audit_on_uniform_emission(bb72):
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair

    compiled, hx, hz = bb72
    fast = SecPolicyConfig(sec_candidates=1, sec_cert_iters=10, **FAST_CERT)
    for point in (
        NoiseSweepConfig.uniform((P,)).point(0),
        NoiseSweepConfig.si1000().point(0),
        NoiseSweepConfig.helios().point(0),
    ):
        pair = build_memory_experiment_pair(
            hx, hz, p=P, rounds=2, noise_point=point, sec_policy=fast,
            layer_blocks=compiled.layer_blocks,
        )
        assert pair.x_memory.noise_point.profile is point.profile
        assert pair.x_memory.detector_error_model.num_detectors == pair.x_memory.circuit.num_detectors
    catastrophic = (np.array([[1, 1, 1, 1]], dtype=np.uint8),
                    np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8))
    from omniqec_agentic.verification.circuit import ScheduleCatastrophicError

    with pytest.raises(ScheduleCatastrophicError):
        build_memory_experiment_pair(*catastrophic, p=P, rounds=2, sec_policy=fast)


def test_twobasis_memory_pair_helper(bb72):
    compiled, hx, hz = bb72
    pair = build_twobasis_memory_pair(compiled, rounds=2, p=P, policy=FAST_POLICY)
    assert set(pair) == {"schedule", "ranked", "x_memory", "z_memory", "metadata"}
    assert pair["metadata"]["sec_policy_id"] == "twobasis_nonint_v1"
    assert pair["x_memory"].num_observables == 12 and pair["z_memory"].num_observables == 12


# ------------------------------------------------------------ end-to-end


def test_bposd_smoke_on_default_policy(bb72):
    from omniqec_agentic.config import VerificationConfig
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair
    from omniqec_agentic.verification.decoders import (
        build_observable_decoder,
        estimate_circuit_logical_error_rate,
    )
    from omniqec_agentic.verification.dem import dem_to_matrices

    compiled, hx, hz = bb72
    pair = build_memory_experiment_pair(
        hx, hz, p=P, rounds=3, noise_point=NoiseSweepConfig.uniform((P,)).point(0),
        layer_blocks=compiled.layer_blocks, d_hint=6,
    )
    assert pair.sec_policy_id == DEFAULT_SEC_POLICY.sec_policy_id == "twobasis_nonint_v1"
    assert pair.ranked.metadata["candidates"] == 16 and pair.ranked.metadata["cert_iters"] == 400
    verification = VerificationConfig()
    for exp in (pair.x_memory, pair.z_memory):
        matrices = dem_to_matrices(exp.detector_error_model)
        decoder = build_observable_decoder("bposd", matrices, verification)
        result = estimate_circuit_logical_error_rate(
            decoder, circuit_text=str(exp.circuit), shots=200, seed=1
        )
        assert result.status == "ok", result
        assert result.shots == 200
