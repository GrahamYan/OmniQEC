"""telescoping_structural_v3_7 decoder tests (docs/06 §1, user decision
2026-09-03: the Mitten paper's open-source telescoping decoder in its CPU
configuration replaces the self-built cascade): detector kinds / X-Z mask,
adapter construction (circuit and materialized-artifact paths agree),
determinism, NC = declared logical error, worker end-to-end (build_dem
materializes the systems, shard_dem loads them), config round trip, the
watchdog cost model, plus the circuit.py seams the decoder relies on
(synth cache key, M5 synthesis lock, TB-4 degraded recompute).

Kept fast: rounds=3, p=0.005, a 2-candidate / 20-iteration SEC policy."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from omniqec_agentic.config import (
    TELESCOPING_COST_S_PER_COLUMN_ITER,
    TELESCOPING_GARI_COLUMN_FACTOR,
    TELESCOPING_PROTOCOL_FIELDS,
    TELESCOPING_VENDORED_COMMIT,
    DecoderName,
    SecPolicyConfig,
    TelescopingDecoderConfig,
    VerificationConfig,
    budget_policy,
)
from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation.baseline import BASELINE_BB_SPEC
from omniqec_agentic.verification import circuit as circuit_mod
from omniqec_agentic.verification.circuit import build_memory_experiment_pair, detector_kind_counts
from omniqec_agentic.verification.decoders import (
    TELESCOPING_STAGES,
    TelescopingObservableDecoder,
    build_decoding_system,
    build_observable_decoder,
    decoder_dependency_report,
    estimate_circuit_logical_error_rate,
    is_x_detector_mask,
    materialize_telescoping_artifacts,
    resolve_telescoping_artifacts,
)
from omniqec_agentic.verification.dem import dem_to_matrices

stim = pytest.importorskip("stim")
td = pytest.importorskip("telescoping_decoder")

ROUNDS = 3
P = 0.005
FAST_POLICY = SecPolicyConfig(
    sec_candidates=2, sec_cert_iters=20, sec_cert_seeds=1,
    sec_cert_iters_scale=0.0, sec_refine_top=0,
)
D6_SPEC = {  # examples/seed_programs.py SEED_LP_NONABELIAN: Lin-Pryadko D6 [[24,8,3]]
    "family": "lp_nonabelian",
    "group": "D6",
    "A": [[[[0, 0], [4, 0]]]],
    "B": [[[[0, 0], [2, 1], [3, 0], [4, 0], [4, 1], [1, 0]]]],
}


def _pair(spec, d_hint):
    compiled = build_compiled_code(spec)
    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    point = VerificationConfig().noise.point(0)
    pair = build_memory_experiment_pair(
        hx, hz, p=float(point.value), noise_point=point, rounds=ROUNDS, d_hint=d_hint,
        sec_policy=FAST_POLICY, layer_blocks=compiled.layer_blocks,
    )
    return compiled, hx, hz, pair


@pytest.fixture(scope="module")
def bb72():
    return _pair(BASELINE_BB_SPEC, 6)


@pytest.fixture(scope="module")
def d6():
    return _pair(D6_SPEC, 3)


def _kinds_from_circuit(circuit, n_data: int, m_z: int) -> list[str]:
    """Independent reference: the check kind of the ancilla whose record a
    DETECTOR references (layout data | zanc | xanc)."""

    meas_qubit: list[int] = []
    kinds: list[str] = []

    def walk(c):
        for inst in c:
            if isinstance(inst, stim.CircuitRepeatBlock):
                for _ in range(inst.repeat_count):
                    walk(inst.body_copy())
                continue
            if inst.name in ("M", "MX", "MY", "MR", "MRX"):
                for t in inst.targets_copy():
                    meas_qubit.append(int(t.value))
            elif inst.name == "DETECTOR":
                qs = [
                    meas_qubit[len(meas_qubit) + int(t.value)]
                    for t in inst.targets_copy()
                    if t.is_measurement_record_target
                ]
                anc = [q for q in qs if q >= n_data]
                assert anc, "every detector references at least one ancilla record"
                kinds.append("Z" if anc[0] < n_data + m_z else "X")

    walk(circuit)
    return kinds


# --------------------------------------------- (a) detector kinds / mask


@pytest.mark.parametrize("which", ["bb72", "d6"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_detector_kinds_match_the_package_mask(which, basis, request):
    compiled, hx, hz, pair = request.getfixturevalue(which)
    exp = pair.x_memory if basis == "X" else pair.z_memory
    m_x, m_z = int(hx.shape[0]), int(hz.shape[0])
    n = int(hx.shape[1])
    kinds = exp.detector_kinds
    assert len(kinds) == exp.circuit.num_detectors == exp.detector_error_model.num_detectors
    counts = detector_kind_counts(kinds)
    # Rounds policy max_Tc_d (2026-09-10): the pair is built at
    # T = max(ROUNDS, d_hint), so bb72 (d_hint 6) runs 6 rounds here while d6
    # (d_hint 3) stays at ROUNDS=3; the counts follow the executed T.
    t = int(exp.rounds)
    assert t == max(ROUNDS, int(pair.d_hint))
    if basis == "X":
        assert counts == {"X": m_x * (t + 1), "Z": m_z * (t - 1)}
    else:
        assert counts == {"Z": m_z * (t + 1), "X": m_x * (t - 1)}
    assert list(kinds) == _kinds_from_circuit(exp.circuit, n, m_z)
    assert exp.to_metadata()["detector_kind_counts"] == counts
    # the emitter's kinds ARE the package's circuit-derived X/Z mask
    mask = is_x_detector_mask(kinds)
    assert np.array_equal(mask, td.circuit_xz_detector_mask(exp.circuit))
    assert int(mask.sum()) == counts["X"]


def test_mask_helpers_reject_bad_inputs(bb72):
    _, _, _, pair = bb72
    exp = pair.x_memory
    with pytest.raises(ValueError, match="'X' or 'Z'"):
        is_x_detector_mask(["X", "Y"])
    kinds = list(exp.detector_kinds)
    flipped = list(kinds)
    flipped[0] = "Z" if flipped[0] == "X" else "X"
    with pytest.raises(ValueError, match="disagree"):
        build_decoding_system(str(exp.circuit), flipped, "X", workdir=None)
    with pytest.raises(ValueError, match="detectors"):
        build_decoding_system(str(exp.circuit), kinds[:-1], "X", workdir=None)
    with pytest.raises(ValueError, match="detector_kinds"):
        build_decoding_system(str(exp.circuit), None, "X", workdir=None)


# ----------------------------------------------- (b) adapter + determinism


def test_bb72_x_memory_is_deterministic_and_artifacts_agree(bb72, tmp_path):
    _, _, _, pair = bb72
    exp = pair.x_memory
    m = dem_to_matrices(exp.detector_error_model)
    text = str(exp.circuit)
    runs = []
    for _ in range(2):
        dec = build_observable_decoder(
            DecoderName.TELESCOPING, m, VerificationConfig(),
            detector_kinds=exp.detector_kinds, basis="X", stage="search", seed=11,
            circuit_text=text,
        )
        assert isinstance(dec, TelescopingObservableDecoder)
        assert dec.system_source == "circuit"
        assert dec.resolved_systems == {"s3": "gari"}
        with dec:
            res = estimate_circuit_logical_error_rate(dec, circuit_text=text, shots=40, seed=5)
            assert res.status == "ok", res.failure_reason
            sample = stim.Circuit(text).compile_detector_sampler(seed=5).sample(8)
            predicted = dec.decode_observables_batch(np.asarray(sample, dtype=np.uint8)).copy()
            runs.append((res.failures, list(dec.last_stages), predicted, res.metrics))
    assert runs[0][0] == runs[1][0]
    assert runs[0][1] == runs[1][1]
    assert np.array_equal(runs[0][2], runs[1][2])
    metrics = runs[0][3]
    assert sum(metrics["stage_counts"].values()) == 40
    assert set(TELESCOPING_STAGES) <= set(metrics["stage_counts"])
    assert metrics["stage_counts"].get("S1", 0) == 0 and metrics["stage_counts"].get("S4", 0) == 0
    assert metrics["declared_failures"] == metrics["stage_counts"]["NC"]
    assert metrics["stage_failures"]["NC"] == metrics["stage_counts"]["NC"]
    assert sum(metrics["stage_failures"].values()) == runs[0][0]
    assert metrics["decode_stage"] == "search" and metrics["nc_policy"] == "fail"
    assert metrics["systems"] == {"s3": "gari"} and metrics["system_source"] == "circuit"
    assert metrics["s3_driver"] == "inprocess"
    assert metrics["gari"]["shape"][0] > m.H.shape[0] and metrics["gari"]["shape"][1] > m.H.shape[1]
    assert metrics["package"] == TelescopingDecoderConfig().package
    assert sum(metrics["label_counts"].values()) == 40  # snapshot taken after the 40-shot estimate
    assert set(metrics["stage_time_s"]) >= set(TELESCOPING_STAGES)

    # the materialized-artifact path is the SAME decoder (bit-identical predictions)
    bundle_path = str(tmp_path / "dem_test.pkl")
    info = materialize_telescoping_artifacts(
        {"X": text, "Z": str(pair.z_memory.circuit)},
        {"X": list(exp.detector_kinds), "Z": list(pair.z_memory.detector_kinds)},
        bundle_path,
    )
    for basis in ("X", "Z"):
        assert info[basis]["gari"].endswith(".npz") and not os.path.isabs(info[basis]["gari"])
        assert info[basis]["n_detectors"] == (pair.x_memory if basis == "X" else pair.z_memory).circuit.num_detectors
    artifacts = resolve_telescoping_artifacts(info["X"], os.path.dirname(bundle_path))
    assert artifacts and os.path.isfile(artifacts["gari"])
    dec_a = build_observable_decoder(
        DecoderName.TELESCOPING, m, VerificationConfig(),
        detector_kinds=exp.detector_kinds, basis="X", seed=11, telescoping_artifacts=artifacts,
    )
    with dec_a:
        assert dec_a.system_source == "artifacts"
        res_a = estimate_circuit_logical_error_rate(dec_a, circuit_text=text, shots=40, seed=5)
        sample = stim.Circuit(text).compile_detector_sampler(seed=5).sample(8)
        predicted_a = dec_a.decode_observables_batch(np.asarray(sample, dtype=np.uint8)).copy()
    assert res_a.failures == runs[0][0]
    assert list(dec_a.last_stages) == runs[0][1]
    assert np.array_equal(predicted_a, runs[0][2])
    # a relocated bundle directory: relative paths resolve, a missing file -> None
    assert resolve_telescoping_artifacts(info["X"], str(tmp_path / "elsewhere")) is None
    assert resolve_telescoping_artifacts(None, str(tmp_path)) is None
    # an artifact of the other basis is refused (wrong init basis)
    with pytest.raises(ValueError, match="memory"):
        build_observable_decoder(
            DecoderName.TELESCOPING, m, VerificationConfig(), detector_kinds=exp.detector_kinds,
            basis="Z", seed=1, telescoping_artifacts=artifacts,
        )
    # a Z-memory decoder and the final stage build too (stage is recorded only)
    z = pair.z_memory
    dec_z = build_observable_decoder(
        "telescoping_cpu", dem_to_matrices(z.detector_error_model), VerificationConfig(),
        detector_kinds=z.detector_kinds, basis="Z", stage="final", seed=1, circuit_text=str(z.circuit),
    )
    with dec_z:
        assert dec_z.stage == "final" and dec_z.basis == "Z"
        assert dec_z.stage_metrics()["decode_stage"] == "final"


def test_inprocess_driver_matches_the_package_pool(bb72):
    """n_procs=1 runs the package's S3 per-shot functions in this process;
    n_procs>1 hands the batch to the package's own pool. Same shots, same
    seeds -> bit-identical stages, labels and predictions."""

    _, _, _, pair = bb72
    exp = pair.x_memory
    m = dem_to_matrices(exp.detector_error_model)
    text = str(exp.circuit)
    sample = stim.Circuit(text).compile_detector_sampler(seed=21).sample(40)
    outcomes = []
    for n_procs in (1, 2):
        ver = dataclasses.replace(
            VerificationConfig(),
            decoder_build=dataclasses.replace(
                VerificationConfig().decoder_build, telescoping=TelescopingDecoderConfig(n_procs=n_procs)
            ),
        )
        dec = build_observable_decoder(
            DecoderName.TELESCOPING, m, ver, detector_kinds=exp.detector_kinds, basis="X", seed=4,
            circuit_text=text,
        )
        with dec:
            predicted = dec.decode_observables_batch(np.asarray(sample, dtype=np.uint8)).copy()
            metrics = dec.stage_metrics()
        outcomes.append((predicted, list(dec.last_stages), metrics["label_counts"], metrics["s3_driver"]))
    assert outcomes[0][3] == "inprocess" and outcomes[1][3] == "package_pool"
    assert np.array_equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][2] == outcomes[1][2]
    assert sum(outcomes[0][2].values()) == 40


def test_nc_shots_are_declared_logical_errors(bb72):
    _, _, _, pair = bb72
    exp = pair.x_memory
    m = dem_to_matrices(exp.detector_error_model)
    text = str(exp.circuit)
    dec = build_observable_decoder(
        DecoderName.TELESCOPING, m, VerificationConfig(),
        detector_kinds=exp.detector_kinds, basis="X", seed=3, circuit_text=text,
    )
    n_obs = int(m.num_observables)

    def all_nc(syndromes, shot_ids=None, true_obs=None):
        count = int(np.asarray(syndromes).shape[0])
        return SimpleNamespace(
            obs_pred=np.zeros((count, n_obs), dtype=np.uint8),
            stage=np.zeros(count, dtype=np.int8),
            label=np.array(["NC_no_ip"] * count, dtype=object),
            converged=np.zeros(count, dtype=bool),
            le=None,
            diagnostics={"n_shots": count, "s3c": {"n_in": count, "n_accepted": 0, "wall_s": 0.5}},
        )

    with dec:
        dec._decode_inprocess = all_nc  # the in-process driver is the protocol path
        res = estimate_circuit_logical_error_rate(dec, circuit_text=text, shots=12, seed=9)
    assert res.status == "ok"
    # every shot is undecoded -> every shot is a logical error, even when the
    # zero prediction happens to match the true observables
    assert res.failures == 12
    assert res.metrics["declared_failures"] == 12
    assert res.metrics["stage_counts"]["NC"] == 12 and res.metrics["stage_failures"]["NC"] == 12
    assert res.metrics["stage_time_s"]["S3C"] == 0.5
    assert res.metrics["label_counts"] == {"NC_no_ip": 12}


def test_builder_and_config_guards(bb72):
    _, _, _, pair = bb72
    m = dem_to_matrices(pair.x_memory.detector_error_model)
    with pytest.raises(ValueError, match="basis"):
        build_observable_decoder(DecoderName.TELESCOPING, m, VerificationConfig())
    with pytest.raises(ValueError, match="detector_kinds"):
        build_observable_decoder(DecoderName.TELESCOPING, m, VerificationConfig(), basis="X")
    with pytest.raises(ValueError, match="circuit text"):
        build_observable_decoder(
            DecoderName.TELESCOPING, m, VerificationConfig(),
            detector_kinds=pair.x_memory.detector_kinds, basis="X",
        )
    with pytest.raises(ValueError, match="stage"):
        TelescopingObservableDecoder(
            m, detector_kinds=pair.x_memory.detector_kinds, basis="X", stage="warmup",
            circuit_text=str(pair.x_memory.circuit),
        )
    # historical decoders ignore the new kwargs
    bposd = build_observable_decoder(DecoderName.BPOSD, m, VerificationConfig(), detector_kinds=None, basis="X")
    assert bposd.decoder_name is DecoderName.BPOSD
    # the CPU protocol refuses the GPU / Gurobi stages and any other NC policy
    for bad in (dict(s1_enabled=True), dict(s2_enabled=True), dict(s4_enabled=True),
                dict(nc_policy="zero"), dict(s3_system="init_dets"), dict(variant_tables="custom"),
                dict(n_procs=0)):
        with pytest.raises(ValueError):
            TelescopingDecoderConfig(**bad)
    cfg = TelescopingDecoderConfig()
    assert cfg.to_dict() == TELESCOPING_PROTOCOL_FIELDS
    assert cfg.package.endswith(TELESCOPING_VENDORED_COMMIT[:12])
    tcfg = cfg.to_telescope_config(base_seed=77)
    assert tcfg.base_seed == 77 and tcfg.use_gari is True
    assert tcfg.s1.enabled is False and tcfg.s2.enabled is False and tcfg.s4.enabled is False
    assert tcfg.s3.enabled is True and tcfg.s3.system == "gari" and tcfg.s3.n_procs == 1
    assert tcfg.s3.b_quorum == 3 and tcfg.s3.b_alpha_max == 0.90 and tcfg.s3.c_n_seeds_default == 10
    # the package tables are untouched
    assert tcfg.s3.a_variants == td.config.DEFAULT_S3A_VARIANTS
    assert tcfg.s3.c_variants == td.config.DEFAULT_S3C_VARIANTS
    assert cfg.worst_case_iterations() == 104_000


def test_dependency_report_versions():
    report = decoder_dependency_report().to_dict()
    assert report["modules"]["telescoping_decoder"] is True
    assert report["versions"]["telescoping_decoder"]
    assert report["versions"]["telescoping_commit"] == TELESCOPING_VENDORED_COMMIT
    assert "scipy" in report["versions"] and "stim" in report["versions"]


def test_budget_policy_and_watchdog_model():
    from omniqec_agentic.evaluation.pool import decode_budget_model, telescoping_worst_case_shot_s

    assert budget_policy(288) == {"large_budget": False, "search_shots": 25_000, "final_shots": 25_000}
    assert budget_policy(360) == {"large_budget": False, "search_shots": 25_000, "final_shots": 25_000}
    assert budget_policy(400) == {"large_budget": True, "search_shots": 5_000, "final_shots": 50_000}
    assert budget_policy(None)["large_budget"] is False
    ver = VerificationConfig()
    worst = telescoping_worst_case_shot_s(ver, 123_960)
    assert worst == pytest.approx(TELESCOPING_COST_S_PER_COLUMN_ITER * 123_960 * TELESCOPING_GARI_COLUMN_FACTOR * 104_000)
    assert telescoping_worst_case_shot_s(ver, None) is None
    assert telescoping_worst_case_shot_s(dataclasses.replace(ver, decoders=(DecoderName.BPOSD,)), 1000) is None
    model = decode_budget_model(ver, 10, 123_960)
    assert model["telescoping_worst_case_shot_s"] == worst and model["telescoping_term_s"] == pytest.approx(10 * worst)


# --------------------------------------------------- (c) worker end-to-end


def test_worker_build_and_shard_end_to_end(bb72, tmp_path):
    from omniqec_agentic.evaluation import worker as worker_mod
    from omniqec_agentic.evaluation.dem_cache import load_dem_bundle

    _, _, _, pair = bb72
    verification_cfg = dataclasses.replace(VerificationConfig(rounds=ROUNDS), sec_policy=FAST_POLICY)
    verification = json.loads(json.dumps(verification_cfg.to_dict()))
    assert verification["decoders"] == ["telescoping_cpu"]
    assert verification["decoder_build"]["telescoping"]["preset"] == "telescoping_cpu_v1"
    path = str(tmp_path / "dem_bb72.pkl")
    build = worker_mod.run_build_dem_request({
        "mode": "build_dem", "verification": verification, "spec": BASELINE_BB_SPEC,
        "matrices_path": path, "noise_index": 0, "d_hint": 6,
    })
    assert build["status"] == "ok", build.get("failure_reason")
    assert build["telescoping"]["status"] == "ok"
    assert set(build["telescoping"]["bases"]) == {"X", "Z"}
    assert build["num_matrix_columns"]["X"] == build["telescoping"]["bases"]["X"]["original_columns"]
    bundle = load_dem_bundle(path)
    assert bundle["detector_kinds"]["X"] == list(pair.x_memory.detector_kinds)
    assert bundle["telescoping"]["status"] == "ok"
    for basis in ("X", "Z"):
        entry = bundle["telescoping"]["bases"][basis]
        assert os.path.isfile(os.path.join(os.path.dirname(path), entry["gari"]))
        assert entry["source_fingerprint"]

    request = {
        "mode": "shard_dem",
        "verification": verification,
        "matrices_path": path,
        "shots": 20,
        "base_seed": 123,
        "shard": 0,
        "shot_shards": 1,
        "rounds_x": ROUNDS,
        "rounds_z": ROUNDS,
        "noise_index": 0,
        "stage": "search",
        "units": [
            {"decoder": "telescoping_cpu", "basis": "X", "decoder_index": 0, "basis_index": 0},
            {"decoder": "telescoping_cpu", "basis": "Z", "decoder_index": 0, "basis_index": 1},
        ],
    }
    payload = worker_mod.run_shard_dem_request(request)
    assert payload["status"] == "ok"
    for basis in ("X", "Z"):
        res = payload["results"]["telescoping_cpu"][basis]
        assert res["status"] == "ok", res.get("failure_reason")
        assert res["shots"] == 20 and res["decode_stage"] == "search"
        metrics = res["metrics"]
        assert sum(metrics["stage_counts"].values()) == 20
        assert set(TELESCOPING_STAGES) <= set(metrics["stage_failures"])
        assert metrics["system_source"] == "artifacts" and metrics["systems"] == {"s3": "gari"}
        assert metrics["declared_failures"] == metrics["stage_counts"]["NC"]
        assert res["memory_experiment"]["detector_kind_counts"]["X"] > 0
    json.dumps(payload, sort_keys=True)
    # the same shard without the materialized systems (circuit path) is bit-identical
    stripped = dict(bundle)
    stripped["telescoping"] = None
    stripped_path = str(tmp_path / "dem_stripped.pkl")
    with open(stripped_path, "wb") as handle:
        pickle.dump(stripped, handle)
    payload_c = worker_mod.run_shard_dem_request(dict(request, matrices_path=stripped_path))
    for basis in ("X", "Z"):
        a = payload["results"]["telescoping_cpu"][basis]
        c = payload_c["results"]["telescoping_cpu"][basis]
        assert c["status"] == "ok" and c["metrics"]["system_source"] == "circuit"
        assert c["failures"] == a["failures"]
        assert c["metrics"]["stage_counts"] == a["metrics"]["stage_counts"]
    # BPOSD still works through the same path
    request_b = dict(request, units=[{"decoder": "bposd", "basis": "X", "decoder_index": 0, "basis_index": 0}], shots=4)
    payload_b = worker_mod.run_shard_dem_request(request_b)
    assert payload_b["results"]["bposd"]["X"]["status"] == "ok"
    assert "stage_counts" not in payload_b["results"]["bposd"]["X"]["metrics"]
    # a pre-v3.6 bundle (no detector kinds, no systems) is refused clearly
    old = dict(bundle)
    old.pop("detector_kinds")
    old.pop("telescoping")
    old_path = str(tmp_path / "old.pkl")
    with open(old_path, "wb") as handle:
        pickle.dump(old, handle)
    loaded_old = load_dem_bundle(old_path)
    assert loaded_old["detector_kinds"] is None and loaded_old["telescoping"] is None
    payload_old = worker_mod.run_shard_dem_request(dict(request, matrices_path=old_path, shots=2))
    res_old = payload_old["results"]["telescoping_cpu"]["X"]
    assert res_old["status"] == "decoder_error" and "detector_kinds" in res_old["failure_reason"]
    # a build whose decoders exclude the telescoping decoder skips the systems
    verification_b = dict(verification, decoders=["bposd"])
    build_b = worker_mod.run_build_dem_request({
        "mode": "build_dem", "verification": verification_b, "spec": BASELINE_BB_SPEC,
        "matrices_path": str(tmp_path / "dem_b.pkl"), "noise_index": 0, "d_hint": 6,
    })
    assert build_b["status"] == "ok" and build_b["telescoping"]["status"] == "skipped"


def test_shot_range_parts_are_bit_identical_to_the_unsplit_shard(bb72, tmp_path):
    """Shard splitting (2026-09-05): decoding a shard as three shot-range jobs
    gives the SAME failures, stage counts and declared failures as the
    unsplit shard through the real worker path (per-shot seeds follow the
    shot's index within the shard, syndromes follow the sampler stream)."""

    from omniqec_agentic.evaluation import worker as worker_mod
    from omniqec_agentic.evaluation.sampling import merge_subshard_results

    verification_cfg = dataclasses.replace(VerificationConfig(rounds=ROUNDS), sec_policy=FAST_POLICY)
    verification = json.loads(json.dumps(verification_cfg.to_dict()))
    path = str(tmp_path / "dem_bb72_split.pkl")
    build = worker_mod.run_build_dem_request({
        "mode": "build_dem", "verification": verification, "spec": BASELINE_BB_SPEC,
        "matrices_path": path, "noise_index": 0, "d_hint": 6,
    })
    assert build["status"] == "ok", build.get("failure_reason")
    request = {
        "mode": "shard_dem", "verification": verification, "matrices_path": path,
        "shots": 22, "base_seed": 321, "shard": 3, "shot_shards": 16,
        "rounds_x": ROUNDS, "rounds_z": ROUNDS, "noise_index": 0, "stage": "search",
        "batch_size": 8,
        "units": [{"decoder": "telescoping_cpu", "basis": "X", "decoder_index": 0, "basis_index": 0}],
    }
    full = worker_mod.run_shard_dem_request(request)["results"]["telescoping_cpu"]["X"]
    assert full["status"] == "ok" and full["shots"] == 22
    parts = []
    for shot_range in ((0, 5), (5, 13), (13, 22)):   # cuts inside and across the 8-shot batches
        part = worker_mod.run_shard_dem_request(dict(request, shot_range=list(shot_range)))
        cell = part["results"]["telescoping_cpu"]["X"]
        assert cell["status"] == "ok", cell.get("failure_reason")
        assert cell["shots"] == shot_range[1] - shot_range[0] and cell["shot_range"] == list(shot_range)
        assert cell["metrics"]["shot_range"] == list(shot_range) and cell["metrics"]["shard_shots"] == 22
        parts.append(cell)
    merged = merge_subshard_results(parts, decoder="telescoping_cpu")
    assert merged["status"] == "ok" and merged["parts"] == 3
    assert merged["shots"] == full["shots"] and merged["failures"] == full["failures"]
    assert merged["metrics"]["stage_counts"] == full["metrics"]["stage_counts"]
    assert merged["metrics"]["stage_failures"] == full["metrics"]["stage_failures"]
    assert merged["metrics"]["declared_failures"] == full["metrics"]["declared_failures"]
    assert merged["metrics"]["requested_shots"] == 22 and merged["metrics"]["sampling_complete"] is True
    assert merged["decode_stage"] == "search" and merged["rounds"] == ROUNDS


def test_verification_from_dict_round_trips_telescoping():
    from omniqec_agentic.evaluation.common import verification_from_dict

    cfg = VerificationConfig(
        decoder_build=dataclasses.replace(
            VerificationConfig().decoder_build,
            telescoping=TelescopingDecoderConfig(b_runs=7, a_alpha_max=0.5, b_use_memory=False),
        )
    )
    back = verification_from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back.decoder_build.telescoping == cfg.decoder_build.telescoping
    assert back.decoder_build.telescoping.b_runs == 7 and back.decoder_build.telescoping.b_use_memory is False
    # an old artifact without the telescoping block gets the defaults
    raw = cfg.to_dict()
    raw["decoder_build"].pop("telescoping")
    raw["decoders"] = ["bposd"]
    old = verification_from_dict(json.loads(json.dumps(raw)))
    assert old.decoder_build.telescoping == TelescopingDecoderConfig()
    assert old.decoders == (DecoderName.BPOSD,)


# ------------------------------------------ circuit.py seams (key, M5, TB-4)


def test_synth_cache_key_covers_every_policy_field(bb72):
    compiled, hx, hz, _ = bb72
    before = len(circuit_mod._SYNTH_CACHE)
    build_memory_experiment_pair(hx, hz, p=P, rounds=2, sec_policy=FAST_POLICY, layer_blocks=compiled.layer_blocks)
    assert len(circuit_mod._SYNTH_CACHE) == before  # module fixture already synthesized it
    for field_name, value in (("sec_cert_seeds", 2), ("sec_refine_top", 1), ("sec_cert_iters_scale", 0.5)):
        policy = dataclasses.replace(FAST_POLICY, **{field_name: value})
        build_memory_experiment_pair(hx, hz, p=P, rounds=2, sec_policy=policy, layer_blocks=compiled.layer_blocks)
    assert len(circuit_mod._SYNTH_CACHE) == before + 3


def test_tb4_degraded_flags_follow_callers_d_hint(bb72):
    compiled, hx, hz, pair = bb72
    d_x = pair.ranked.metadata["d_ext_x"]
    assert pair.ranked.metadata["d_code_proxy"] == 6
    high = build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, d_hint=99, sec_policy=FAST_POLICY, layer_blocks=compiled.layer_blocks
    )
    assert high.ranked.metadata["d_code_proxy"] == 99
    assert high.ranked.metadata["schedule_degraded_x"] is True
    assert high.ranked.metadata["schedule_degraded_z"] is True
    assert high.ranked.metadata["d_ext_x"] == d_x
    low = build_memory_experiment_pair(
        hx, hz, p=P, rounds=2, d_hint=1, sec_policy=FAST_POLICY, layer_blocks=compiled.layer_blocks
    )
    assert low.ranked.metadata["schedule_degraded_x"] is False
    assert low.ranked.metadata["schedule_degraded_z"] is False
    assert pair.ranked.metadata["d_code_proxy"] == 6


def test_m5_synthesis_lock_shares_one_synthesis(tmp_path, monkeypatch):
    monkeypatch.setattr(circuit_mod, "SYNTH_LOCK_POLL_S", 0.02)
    monkeypatch.setattr(circuit_mod, "SYNTH_LOCK_WAIT_S", 10.0)
    key = "deadbeef"
    calls = []

    def compute():
        calls.append(1)
        time.sleep(0.3)
        return {"ranked": len(calls)}

    results = []

    def run():
        results.append(circuit_mod._synthesize_with_lock(str(tmp_path), key, compute))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results == [{"ranked": 1}] * 4
    assert not circuit_mod._synth_lock_path(str(tmp_path), key).exists()
    assert circuit_mod._synth_disk_load(str(tmp_path), key) == {"ranked": 1}

    stale_key = "cafe"
    lock = circuit_mod._synth_lock_path(str(tmp_path), stale_key)
    lock.write_text("dead")
    old = time.time() - 10_000
    os.utime(lock, (old, old))
    monkeypatch.setattr(circuit_mod, "SYNTH_LOCK_STALE_S", 100.0)
    assert circuit_mod._synthesize_with_lock(str(tmp_path), stale_key, lambda: "fresh") == "fresh"
    assert not lock.exists()

    monkeypatch.setattr(circuit_mod, "SYNTH_LOCK_WAIT_S", 0.2)
    monkeypatch.setattr(circuit_mod, "SYNTH_LOCK_STALE_S", 1000.0)
    live = circuit_mod._synth_lock_path(str(tmp_path), "live")
    live.write_text("busy")
    t0 = time.monotonic()
    assert circuit_mod._synthesize_with_lock(str(tmp_path), "live", lambda: "local") == "local"
    assert time.monotonic() - t0 >= 0.2
    assert live.exists()
    assert circuit_mod._synthesize_with_lock(None, "x", lambda: 42) == 42
