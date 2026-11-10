"""SEC golden regression (design doc 02 §9): SBB Zenodo reference parity.

Golden artifacts: HGP625 LRC memory circuits (X and Z basis, r=8, p=0.001)
from Zenodo record 10.5281/zenodo.18853601 (CC-BY-4.0), gzipped in
tests/golden/. The test extracts HX/HZ and the left/right partition from the
golden circuit itself, rebuilds with our sec/ pipeline, and asserts full
segment-level semantic parity (gates, noise placement, detectors,
observables; golden's empty no-op noise instructions are ignored).
Also pins the depth anchors and four-family noiseless validity.
"""

import gzip
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.sec import build_canonical_schedule, split_sectors
from omniqec_agentic.sec.stim_builder import build_lrc_memory_pair

GOLDEN_DIR = Path(__file__).parent / "golden"
N, M_Z, M_X = 625, 300, 300


def _segments(circuit):
    segs = [[]]

    def walk(circ, reps=1):
        for _ in range(reps):
            for inst in circ:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    walk(inst.body_copy(), inst.repeat_count)
                elif inst.name == "TICK":
                    segs.append([])
                else:
                    segs[-1].append(inst)

    walk(circuit)
    return segs


def _signature(segs):
    out = []
    for seg in segs:
        counts = Counter()
        for inst in seg:
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if inst.name in ("CX", "DEPOLARIZE2"):
                counts[inst.name] += len(targets) // 2
            elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                counts[inst.name] += 1
            elif targets:  # skip golden's empty no-op noise instructions
                counts[inst.name] += len(targets)
        out.append(dict(sorted(counts.items())))
    return out


def _golden(name):
    with gzip.open(GOLDEN_DIR / name, "rt") as handle:
        return stim.Circuit(handle.read())


def _extract_code(gold_segments):
    hx = np.zeros((M_X, N), dtype=np.uint8)
    hz = np.zeros((M_Z, N), dtype=np.uint8)
    for seg in gold_segments:
        for inst in seg:
            if inst.name == "CX":
                ts = [t.value for t in inst.targets_copy()]
                for a, b in zip(ts[0::2], ts[1::2]):
                    if a >= N + M_Z:
                        hx[a - N - M_Z, b] = 1
                    elif N <= b < N + M_Z:
                        hz[b - N, a] = 1
    return hx, hz


def test_hgp625_golden_parity_both_bases():
    gold_x = _golden("HGP625_LRC_r8_X_p0.001.stim.gz")
    gold_segments = _segments(gold_x)
    hx, hz = _extract_code(gold_segments)
    assert int((hx @ hz.T % 2).sum()) == 0
    compiled = SimpleNamespace(hx=hx.tolist(), hz=hz.tolist(), sector_sizes=[400, 225])
    pair = build_lrc_memory_pair(compiled, rounds=8, p=0.001)
    for basis_key, golden_name in (
        ("x_memory", "HGP625_LRC_r8_X_p0.001.stim.gz"),
        ("z_memory", "HGP625_LRC_r8_Z_p0.001.stim.gz"),
    ):
        golden = _golden(golden_name)
        golden_sig = _signature(_segments(golden))
        ours_sig = _signature(_segments(pair[basis_key]))
        assert len(golden_sig) == len(ours_sig) == 84
        assert golden_sig == ours_sig, f"{basis_key} segment signature mismatch"
        assert pair[basis_key].num_detectors == golden.num_detectors == 2700
        assert pair[basis_key].num_measurements == golden.num_measurements == 5425
        assert pair[basis_key].num_observables == golden.num_observables == 25


def _depth(spec):
    compiled = build_compiled_code(spec)
    sectors = split_sectors(
        np.array(compiled.hx), np.array(compiled.hz), compiled.sector_sizes
    )
    return build_canonical_schedule(sectors).depth


def test_depth_anchors():
    from omniqec_agentic.evaluation import BASELINE_BB_SPEC

    assert _depth(BASELINE_BB_SPEC) == 8  # Bravyi anchor, w=6
    spec_w9 = {
        "family": "bb", "ell": 6, "m": 6,
        "A": [[3, 0], [0, 1], [0, 2], [1, 1]],
        "B": [[0, 3], [1, 0], [2, 0], [2, 2], [1, 2]],
    }
    assert _depth(spec_w9) == 11  # w_max = 9 saturates the bound


def test_four_family_noiseless_validity():
    circulant = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
    specs = {
        "bb": {"family": "bb", "ell": 3, "m": 5,
               "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]]},
        "hgp": {"family": "hgp", "seeds": {"H1": circulant, "H2": circulant}},
        "lp_abelian": {"family": "lp_abelian", "group": {"factors": [3, 3]},
                       "A": [[[[2, 0], [0, 0]], [[1, 0], [0, 1]]]],
                       "B": [[[[0, 2], [1, 2], [0, 0]]]]},
        "lp_nonabelian": {"family": "lp_nonabelian", "group": "D6",
                          "A": [[[[0, 0], [4, 0]]]],
                          "B": [[[[0, 0], [2, 1], [3, 0], [4, 0], [4, 1], [1, 0]]]]},
    }
    for family, spec in specs.items():
        compiled = build_compiled_code(spec)
        pair = build_lrc_memory_pair(compiled, rounds=3, p=0)
        for key in ("x_memory", "z_memory"):
            samples = pair[key].compile_detector_sampler().sample(8)
            assert not samples.any(), f"{family} {key}: noiseless detectors fired"
            m_basis = len(compiled.hx) if key == "x_memory" else len(compiled.hz)
            assert pair[key].num_detectors == 4 * m_basis  # (rounds+1) * m_basis
