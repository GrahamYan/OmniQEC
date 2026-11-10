"""telescoping_structural_v3_7 (docs/06, user decision 2026-09-03): protocol
identity, budget policy, pool-side budget scaling / stage tagging / memory
cap / M5 ordering.

Covers the pool-protocol slice of the plan:
  * state.build_protocol_snapshot applies config.budget_policy(n_budget):
    b360 -> 5000/50000 shots; b144 -> 20000/200000 (3 keys, no decoder
    limits); snapshot keys decoder / telescoping / n_budget / budget_policy /
    sec_policy;
  * strict refusals (decoder set, telescoping drift, shots off-policy);
  * a v3.5 checkpoint is refused on resume by name;
  * the standalone validator accepts a v3.6 snapshot and rejects a v3.5 one;
  * pool.effective_decode_shard_timeout_s scales with the DEM column count;
  * shard requests carry stage / num_matrix_columns, retry_at_final records
    run with the final budgets, ru_maxrss lands in job_execution;
  * resolve_workers caps by memory; the noise-0 build precedes the distance
    job and every other build (M5);
  * cli run-campaign derives --slow-shots / --final-shots from the policy.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

from omniqec_agentic.config import (
    formal_racing,
    DECODE_BUDGET_SCALE_COLS,
    DEFAULT_SEC_POLICY,
    TELESCOPING_PROTOCOL_FIELDS,
    TELESCOPING_VENDORED_COMMIT,
    DecoderName,
    RacingConfig,
    StructuralSearchConfig,
    TelescopingDecoderConfig,
    VerificationConfig,
    budget_policy,
)
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.sampling import merge_shard_results
from omniqec_agentic.hpc import validate_fig2_campaign as validator
from omniqec_agentic.hpc.validate_fig2_campaign import validate
from omniqec_agentic.workflow import state as state_mod
from omniqec_agentic.workflow.checkpoint import validate_resume_protocol
from omniqec_agentic.workflow.state import assert_resumable_protocol, build_protocol_snapshot

from test_campaign_strict_turns import (
    FINAL_SHOTS,
    SEARCH_SHOTS,
    TELESCOPING,
    TELESCOPING_BUILD,
    _rankings,
    _slow_run_config,
    _strict_report,
    _write_json,
    _write_validated_run,
)

PROFILE = "telescoping_structural_v3_7"


# ---------------------------------------------------------------------------
# Budget policy in the protocol snapshot
# ---------------------------------------------------------------------------


def _snapshot(**overrides) -> dict:
    kwargs = dict(
        verification=VerificationConfig(),
        racing=RacingConfig(),
        max_turns=1,
        proposal_count=8,
        generations_per_turn=1,
        slow_workers=4,
        search_rounds=12,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)


def _strict(**overrides) -> dict:
    return _snapshot(
        strict_turns=True,
        required_provider="deepseek",
        required_model="deepseek-v4-pro",
        **overrides,
    )


def test_budget_policy_table_is_the_plan_table() -> None:
    small = budget_policy(144)
    # Three keys only: the telescoping decoder has no per-point limit (the
    # cascade's s4_time_s / s4_time_final_s are gone with the cascade).
    assert set(small) == {"large_budget", "search_shots", "final_shots"}
    assert (small["search_shots"], small["final_shots"]) == (25_000, 25_000)
    assert small["large_budget"] is False
    assert budget_policy(288) == small
    assert budget_policy(360) == small  # 2026-09-14: all nine Fig.2 points share 25k/25k
    assert budget_policy(None) == small
    large = budget_policy(400)
    assert set(large) == set(small)
    assert (large["search_shots"], large["final_shots"]) == (5_000, 50_000)
    assert large["large_budget"] is True
    assert budget_policy(1000) == large


def test_state_constants_name_the_v36_protocol() -> None:
    assert state_mod.FORMAL_PROFILE == PROFILE == StructuralSearchConfig().profile_id
    assert state_mod.FORMAL_DECODER == DecoderName.TELESCOPING.value == "telescoping_cpu"
    assert VerificationConfig().decoders == (DecoderName.TELESCOPING,)
    # The self-built cascade never launched: its profile is retired by prefix.
    assert "cascade_structural" in state_mod.LEGACY_PROFILE_PREFIXES
    assert validator.PROTOCOL_PROFILE == PROFILE
    assert validator.FORMAL_DECODER == state_mod.FORMAL_DECODER
    assert validator.DEFAULT_DECODERS == ("telescoping_cpu",)
    # The validator imports the telescoping table and the vendored commit
    # from config (no mirror); the table is EVERY decoder field.
    assert validator.TELESCOPING_PROTOCOL_FIELDS is TELESCOPING_PROTOCOL_FIELDS
    assert TELESCOPING_PROTOCOL_FIELDS == TelescopingDecoderConfig().to_dict()
    assert validator.REQUIRED_TELESCOPING_COMMIT == TELESCOPING_VENDORED_COMMIT
    assert validator.REQUIRED_VERSION_KEYS == ("telescoping-decoder",)
    assert validator.STRUCTURAL_EXPECTED == StructuralSearchConfig().to_dict()
    assert validator.STRUCTURAL_EXPECTED["reducer_id"] == "structural_proxy_pareto_v3_weight"


def test_snapshot_applies_the_large_budget_policy_at_b400() -> None:
    verification = VerificationConfig(shots=5_000)
    protocol = _strict(verification=verification, n_budget=400)
    assert protocol["protocol_profile"] == PROFILE
    assert protocol["n_budget"] == 400
    assert protocol["budget_policy"] == budget_policy(400)
    assert protocol["verification"]["shots"] == 5_000
    assert protocol["final_shots"] == 50_000
    telescoping = protocol["telescoping"]
    assert telescoping == TELESCOPING_PROTOCOL_FIELDS
    assert protocol["verification"]["decoder_build"]["telescoping"] == telescoping
    assert protocol["decoder"] == "telescoping_cpu"
    assert "cascade" not in protocol and "cascade" not in protocol["verification"]["decoder_build"]
    # The keys survive the JSON round trip the checkpoint / report take.
    assert json.loads(json.dumps(protocol))["budget_policy"]["final_shots"] == 50_000


def test_snapshot_applies_the_small_budget_policy_at_b144() -> None:
    protocol = _strict(n_budget=144)
    assert protocol["budget_policy"] == budget_policy(144)
    assert protocol["verification"]["shots"] == 25_000
    assert protocol["final_shots"] == 25_000
    # The decoder blob does not depend on the point: b144 and b360 freeze
    # the same telescoping config (no S4 limits to rewrite any more).
    large = _strict(verification=VerificationConfig(shots=5_000), n_budget=400)
    assert protocol["telescoping"] == large["telescoping"] == TELESCOPING_PROTOCOL_FIELDS
    # A legacy caller without n_budget resolves the small-budget table.
    legacy = _snapshot()
    assert legacy["n_budget"] is None
    assert legacy["budget_policy"] == budget_policy(None)
    assert legacy["final_shots"] == 25_000
    # The snapshot records the blob AS CONFIGURED (it never rewrites decoder
    # fields); outside strict a variant config passes through unchanged.
    variant = VerificationConfig(
        decoder_build=dataclasses.replace(
            VerificationConfig().decoder_build,
            telescoping=TelescopingDecoderConfig(b_quorum=2),
        )
    )
    assert _snapshot(verification=variant, n_budget=360)["telescoping"]["b_quorum"] == 2


def test_snapshot_records_the_sec_policy_blob_with_the_v36_fields() -> None:
    protocol = _snapshot(n_budget=144)
    blob = protocol["sec_policy"]
    assert blob == DEFAULT_SEC_POLICY.to_dict() == protocol["verification"]["sec_policy"]
    assert blob["sec_cert_seeds"] == 3
    assert blob["sec_cert_iters_scale"] == 4.0
    assert blob["sec_refine_top"] == 4
    assert blob["sec_refine_iters"] == 2000
    assert (protocol["sec_policy_id"], protocol["sec_candidates"], protocol["sec_cert_iters"]) == (
        "twobasis_nonint_v1", 16, 400,
    )


def test_strict_snapshot_refuses_off_policy_shots_and_honours_them_outside_strict() -> None:
    with pytest.raises(ValueError, match=re.escape("search shots per basis at 5000 for n_budget=400")):
        _strict(verification=VerificationConfig(shots=20_000), n_budget=400)
    with pytest.raises(ValueError, match=re.escape("final shots per basis at 50000 for n_budget=400")):
        _strict(verification=VerificationConfig(shots=5_000), n_budget=400, final_shots=200_000)
    with pytest.raises(ValueError, match="search shots per basis at 25000"):
        _strict(verification=VerificationConfig(shots=5_000), n_budget=144)
    # Outside strict the caller's values are honoured (a variant run).
    variant = _snapshot(verification=VerificationConfig(shots=20_000), n_budget=400, final_shots=200_000)
    assert variant["verification"]["shots"] == 20_000
    assert variant["final_shots"] == 200_000
    assert variant["budget_policy"] == budget_policy(400)


def test_strict_snapshot_refuses_another_decoder_or_a_drifted_telescoping_config() -> None:
    with pytest.raises(ValueError, match="telescoping_cpu-only"):
        _strict(verification=VerificationConfig(decoders=(DecoderName.BPOSD,)), n_budget=144)
    drifted = VerificationConfig(
        decoder_build=dataclasses.replace(
            VerificationConfig().decoder_build, telescoping=TelescopingDecoderConfig(b_quorum=2)
        )
    )
    with pytest.raises(ValueError, match="freezes the telescoping decoder fields.*'b_quorum': 2"):
        _strict(verification=drifted, n_budget=144)
    # Outside strict a different decoder retags the profile, never silently.
    variant = _snapshot(verification=VerificationConfig(decoders=(DecoderName.BPOSD,)))
    assert variant["protocol_profile"] == f"{PROFILE}+decbposd"
    assert variant["decoder"] == "bposd"


def test_a_v35_checkpoint_is_refused_on_resume_by_name() -> None:
    current = _snapshot(n_budget=144)
    v35 = json.loads(json.dumps(current))
    v35["protocol_profile"] = "bposd_structural_v3_5"
    for key in ("decoder", "telescoping", "n_budget", "budget_policy", "sec_policy"):
        v35.pop(key)
    v35["verification"]["decoders"] = ["bposd"]
    v35["verification"]["decoder_build"].pop("telescoping")
    with pytest.raises(ValueError, match="bposd_structural_v3_5.*decoder 'bposd'.*not poolable"):
        assert_resumable_protocol(v35)
    # A checkpoint of the never-launched self-built cascade is refused by
    # NAME (retired prefix) and for its missing `telescoping` key.
    cascade_run = json.loads(json.dumps(current))
    cascade_run["protocol_profile"] = "cascade_structural_v3_6"
    cascade_run["decoder"] = "relay_cascade_gari"
    cascade_run["cascade"] = cascade_run.pop("telescoping")
    with pytest.raises(ValueError, match="cascade_structural_v3_6.*relay_cascade_gari.*no telescoping key"):
        assert_resumable_protocol(cascade_run)
    # The generic diff refuses it too (belt and braces).
    with pytest.raises(ValueError, match="protocol snapshot mismatch") as excinfo:
        validate_resume_protocol(v35, current)
    assert "budget_policy" in str(excinfo.value) and "decoder" in str(excinfo.value)
    # A snapshot that merely lacks the budget policy is refused as well.
    no_policy = json.loads(json.dumps(current))
    no_policy.pop("budget_policy")
    with pytest.raises(ValueError, match="no budget_policy key"):
        assert_resumable_protocol(no_policy)
    with pytest.raises(ValueError, match="no protocol snapshot"):
        assert_resumable_protocol(None)
    # The round trip of the current snapshot resumes.
    assert_resumable_protocol(json.loads(json.dumps(current)))
    validate_resume_protocol(json.loads(json.dumps(current)), current)
    # Another budget point is a different policy: refused by the diff.
    other_point = _snapshot(verification=VerificationConfig(shots=5_000), n_budget=400)
    with pytest.raises(ValueError, match="budget_policy"):
        validate_resume_protocol(other_point, current)


# ---------------------------------------------------------------------------
# Standalone validator
# ---------------------------------------------------------------------------


def _validate(run_dir: Path, **kwargs) -> int:
    params = dict(turns=1, p_value=0.005, model="deepseek-v4-pro")
    params.update(kwargs)
    return validate(run_dir, **params)


def test_validator_accepts_a_v36_report_and_derives_final_shots_from_the_policy(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    assert _validate(tmp_path) == 1
    assert _validate(tmp_path, final_shots=FINAL_SHOTS) == 1
    with pytest.raises(ValueError, match="caller declared final_shots=20000"):
        _validate(tmp_path, final_shots=20_000)


def test_validator_rejects_a_v35_shaped_snapshot(tmp_path) -> None:
    report = _strict_report()
    protocol = report["protocol"]
    protocol["protocol_profile"] = "bposd_structural_v3_5"
    for key in ("decoder", "telescoping", "n_budget", "budget_policy", "sec_policy"):
        protocol.pop(key)
    protocol["verification"]["decoders"] = ["bposd"]
    protocol["verification"]["decoder_build"].pop("telescoping")
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="n_budget=None"):
        _validate(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("protocol_profile", "bposd_structural_v3_5"), "protocol profile is 'bposd_structural_v3_5'"),
        (lambda p: p.__setitem__("decoder", "bposd"), "protocol snapshot: decoder='bposd'"),
        (lambda p: p["telescoping"].__setitem__("nc_policy", "count"), "telescoping nc_policy='count'"),
        (lambda p: p["telescoping"].__setitem__("s4_enabled", True), "telescoping s4_enabled=True"),
        (lambda p: p["telescoping"].__setitem__("s3_system", "original"), "telescoping s3_system='original'"),
        (lambda p: p["verification"]["decoder_build"]["telescoping"].__setitem__("b_runs", 50), "telescoping b_runs=50"),
        # Missing is a failure, and so is a type drift (int as float, bool as int).
        (lambda p: p["telescoping"].pop("a_iters"), "telescoping a_iters=None"),
        (lambda p: p["telescoping"].__setitem__("a_iters", 100.0), "telescoping a_iters=100.0"),
        (lambda p: p["telescoping"].__setitem__("b_use_memory", 1), "telescoping b_use_memory=1"),
        # A cascade-era budget policy (with S4 limits) is another protocol.
        (lambda p: p["budget_policy"].__setitem__("s4_time_s", 60.0), "budget_policy keys="),
        (lambda p: p["budget_policy"].__setitem__("final_shots", 50_000), "budget_policy final_shots=50000"),
        (lambda p: p.__setitem__("budget_policy", None), "budget_policy is missing"),
        (lambda p: p["verification"].__setitem__("shots", 5_000), "verification.shots=5000"),
        (lambda p: p["verification"]["sec_policy"].__setitem__("sec_cert_seeds", 1), "sec_cert_seeds=1"),
        (lambda p: p["verification"]["sec_policy"].pop("sec_refine_iters"), "sec_refine_iters=None"),
        (lambda p: p["structural_retention"].__setitem__("reducer_id", "structural_proxy_pareto_v1"), "reducer_id='structural_proxy_pareto_v1'"),
        (lambda p: p["structural_retention"].__setitem__("coordinate_cap", 1), "coordinate_cap=1"),
    ],
)
def test_validator_rejects_a_tampered_v36_snapshot(tmp_path, mutate, message) -> None:
    report = _strict_report()
    mutate(report["protocol"])
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate(tmp_path)


def test_validator_checks_the_point_against_the_report_spec(tmp_path) -> None:
    report = _strict_report()
    report["spec"]["qubit_budget"] = 360
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="n_budget=144, expected 360"):
        _validate(tmp_path)
    report = _strict_report()
    report.pop("spec")
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="spec.qubit_budget=None"):
        _validate(tmp_path)


def test_validator_accepts_a_large_budget_report_with_its_own_policy(tmp_path) -> None:
    policy = budget_policy(400)
    report = _strict_report()
    report["spec"]["qubit_budget"] = 400
    protocol = report["protocol"]
    protocol["n_budget"] = 400
    protocol["budget_policy"] = dict(policy)
    protocol["final_shots"] = policy["final_shots"]
    protocol["verification"]["shots"] = policy["search_shots"]
    protocol["racing"] = formal_racing(policy["search_shots"]).to_dict()
    # The decoder blob is the same at every point: only the shots change.
    assert protocol["telescoping"] == TELESCOPING_BUILD
    report["final_measurement_audit"]["shots_per_basis"] = policy["final_shots"]
    for row in report["final_frontier"]:
        row.update(shots_x=policy["final_shots"], shots_z=policy["final_shots"], total_shots=2 * policy["final_shots"])
    _write_validated_run(tmp_path, report)
    for stage_dir, shards, shots in (
        (tmp_path / "turns" / "turn_0000" / "cycle" / "slow", 16, policy["search_shots"]),
        (tmp_path / "final_slow" / "T12", 128, policy["final_shots"]),
    ):
        config = _slow_run_config(rounds=12, seed=0 if shards == 16 else validator._expected_final_base_seed("final_confirmation_v1"), shot_shards=shards, shots=shots)
        _write_json(stage_dir / "slow_run_config.json", config)
        _write_json(stage_dir / "rankings.json", _rankings(12, shots=shots))
    assert _validate(tmp_path) == 1
    assert _validate(tmp_path, final_shots=50_000) == 1
    with pytest.raises(ValueError, match="caller declared final_shots=200000"):
        _validate(tmp_path, final_shots=200_000)


def test_validator_checks_run_config_shots_and_row_decoders(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12, shots=5_000))
    with pytest.raises(ValueError, match="slow run config: shots=5000, expected 25000"):
        _validate(tmp_path)
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12, stage="final"))
    with pytest.raises(ValueError, match="slow run config: stage='final'"):
        _validate(tmp_path)
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12))
    rankings = _rankings(12)
    rankings["rankings"][0]["decoder"] = "bposd"
    _write_json(slow_dir / "rankings.json", rankings)
    with pytest.raises(ValueError, match="search rankings: row g0001_p0001 decoder='bposd'"):
        _validate(tmp_path)


def test_validator_requires_the_telescoping_package_at_the_vendored_commit(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    # The package version must be resolved (not "absent")...
    _write_json(
        tmp_path / "resolved_versions.json",
        {"telescoping-decoder": "absent", "telescoping_commit": TELESCOPING_VENDORED_COMMIT},
    )
    with pytest.raises(ValueError, match="telescoping-decoder version not resolved"):
        _validate(tmp_path)
    # ...at the vendored commit config froze (protocol identity)...
    _write_json(
        tmp_path / "resolved_versions.json",
        {"telescoping-decoder": "0.1.0", "telescoping_commit": "deadbeef"},
    )
    with pytest.raises(ValueError, match="telescoping_commit is 'deadbeef'"):
        _validate(tmp_path)
    _write_json(tmp_path / "resolved_versions.json", {"telescoping-decoder": "0.1.0"})
    with pytest.raises(ValueError, match="telescoping_commit is None"):
        _validate(tmp_path)
    # ...and nothing else: ldpc / relay_bp / scipy are dev-only decoder deps.
    _write_json(
        tmp_path / "resolved_versions.json",
        {"telescoping-decoder": "0.1.0", "telescoping_commit": TELESCOPING_VENDORED_COMMIT},
    )
    assert _validate(tmp_path) == 1
    # The S5 / BPOSD fallback concept is gone with the cascade: an undecoded
    # shot is a logical error (nc_policy == "fail"), nothing else to check.
    assert not hasattr(validator, "_validate_bposd_fallback_blob")
    assert not hasattr(validator, "LDPC_VERSION")


# ---------------------------------------------------------------------------
# Pool: effective watchdog scales with the DEM size (B5)
# ---------------------------------------------------------------------------


def test_effective_timeout_scales_with_dem_columns() -> None:
    # The flat v3.4 term (8 s/shot x column factor) on a non-telescoping
    # decoder set; with the telescoping decoder the watchdog is ALSO >=
    # shots x the worst-case shot model (tests/test_v36_review_fixes.py),
    # which dominates once the DEM size is known.
    formal = VerificationConfig(decoders=(DecoderName.BPOSD,))
    base = pool_mod.effective_decode_shard_timeout_s(formal, 1250)
    assert base == pytest.approx(10_000.0)
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, None) == pytest.approx(base)
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, 1_000) == pytest.approx(base)
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, DECODE_BUDGET_SCALE_COLS) == pytest.approx(base)
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, 120_000) == pytest.approx(20_000.0)
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, 90_000) == pytest.approx(15_000.0)
    # The floor still wins for tiny shards even at large DEMs.
    assert pool_mod.effective_decode_shard_timeout_s(formal, 10, 120_000) == pytest.approx(1800.0)
    # A disabled budget ignores the size.
    assert pool_mod.effective_decode_shard_timeout_s(
        VerificationConfig(decode_shot_budget_s=0.0), 1250, 120_000
    ) == pytest.approx(1800.0)
    assert pool_mod.decode_budget_scale(None) == 1.0
    assert pool_mod.decode_budget_scale(0) == 1.0
    assert pool_mod.decode_budget_scale(180_000) == 3.0
    assert pool_mod.worker_timeout_for(
        {"mode": "shard_dem", "shots": 1250, "num_matrix_columns": 120_000}, formal
    ) == pytest.approx(20_000.0)
    assert pool_mod.worker_timeout_for({"mode": "shard_dem", "shots": 1250}, formal) == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Pool: stage tagging, columns, ru_maxrss, M5 ordering, memory cap
# ---------------------------------------------------------------------------


def _seed_bb_spec() -> dict:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    return run_construct_source(SEED_BB, program_id="v36_pool_bb", seed=0).constructed.spec


def _recording_worker(*, columns: int | None = 120_000, rss: int | None = 123_456_789):
    """Fake worker recording every request (mode, stage, columns) in order;
    builds report num_matrix_columns, shards report ru_maxrss."""

    seen: list[dict] = []
    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}

    def fake_worker(request, *, timeout_s):
        seen.append({**copy.deepcopy(request), "_timeout_s": timeout_s})
        mode = request.get("mode")
        if mode == "build_dem":
            outcome = {"status": "ok", "rounds_x": 12, "rounds_z": 12}
            if columns is not None:
                outcome["num_matrix_columns"] = {"X": columns, "Z": columns // 2}
            return outcome, dict(fine)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        unit = request["units"][0]
        shots = int(request["shots"])
        cell = {
            "status": "ok", "decoder": unit["decoder"], "shots": shots, "failures": 1,
            "logical_error_rate": 1.0 / shots, "runtime_s": 0.01,
            "metrics": {
                "requested_shots": shots, "actual_shots": shots, "sampling_complete": True,
                "stopped_by_max_errors": False,
                "stage_counts": {"S3A": shots - 3, "S3B": 2, "NC": 1},
            },
        }
        outcome = {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}
        if rss is not None:
            outcome["ru_maxrss"] = rss
        return outcome, dict(fine)

    return fake_worker, seen


def _run_pool(monkeypatch, fake_worker, *, records=None, stage="search", noise_values=(0.005,)):
    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    verification = VerificationConfig(
        shots=2_000, max_errors=None, rounds=12,
        p_values=tuple(noise_values),
    )
    return pool_mod.evaluate_queue_with_jobs(
        records or [{"program_id": "bb_v36", "spec": _seed_bb_spec()}],
        verification=verification,
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=4,
        shot_shards=2,
        stage=stage,
    )


def verification_of(seen: list[dict]) -> VerificationConfig:
    from omniqec_agentic.evaluation.common import verification_from_dict

    return verification_from_dict(next(r["verification"] for r in seen if r["mode"] == "shard_dem"))


def test_shard_requests_carry_stage_and_columns_and_records_carry_rss(monkeypatch) -> None:
    fake_worker, seen = _recording_worker()
    payload = _run_pool(monkeypatch, fake_worker)[0]
    assert payload["status"] == "ok"
    assert payload["stage"] == "search"
    shard_requests = [r for r in seen if r["mode"] == "shard_dem"]
    assert shard_requests and all(r["stage"] == "search" for r in shard_requests)
    assert all(r["num_matrix_columns"] == 120_000 for r in shard_requests)
    # 1000 shots x max(8 s x (120000 / 60000) = 16 s, telescoping worst-case
    # shot) per shard; the flat term alone is 16,000 s and the review fix
    # raises it to the telescoping decoder's iteration-bounded worst case.
    worst = pool_mod.telescoping_worst_case_shot_s(verification_of(seen), 120_000, "search")
    assert worst > 16.0
    assert all(r["_timeout_s"] == pytest.approx(1000 * worst) for r in shard_requests)
    job_config = payload["job_config"]
    assert job_config["stage"] == "search"
    assert job_config["num_matrix_columns"] == 120_000
    assert job_config["decode_budget_scale"] == pytest.approx(2.0)
    assert job_config["effective_decode_shard_timeout_s"] == pytest.approx(1000 * worst)
    assert job_config["decode_budget_model"]["flat_term_s"] == pytest.approx(16_000.0)
    assert job_config["base_decode_shard_timeout_s"] == pytest.approx(8_000.0)
    basis = payload["results"][0]["basis"]["Z"]
    assert basis["shard_max_rss_bytes"] == [123_456_789, 123_456_789]
    assert basis["metrics"]["stage_counts"] == {"S3A": 1994, "S3B": 4, "NC": 2}


def test_final_stage_and_retry_at_final_records_run_with_the_final_budgets(monkeypatch) -> None:
    fake_worker, seen = _recording_worker(columns=None, rss=None)
    payload = _run_pool(monkeypatch, fake_worker, stage="final")[0]
    assert payload["stage"] == "final"
    assert all(r["stage"] == "final" for r in seen if r["mode"] == "shard_dem")
    assert all(r["num_matrix_columns"] is None for r in seen if r["mode"] == "shard_dem")
    assert payload["job_config"]["num_matrix_columns"] is None
    assert payload["job_config"]["effective_decode_shard_timeout_s"] == pytest.approx(8_000.0)
    assert payload["results"][0]["basis"]["Z"]["shard_max_rss_bytes"] == [None, None]
    assert "ru_maxrss" not in (payload["results"][0]["basis"]["Z"].get("job_execution") or {})

    fake_worker, seen = _recording_worker(columns=None, rss=None)
    payload = _run_pool(
        monkeypatch,
        fake_worker,
        records=[{"program_id": "bb_retry", "spec": _seed_bb_spec(), "retry_at_final": True}],
        stage="search",
    )[0]
    assert payload["stage"] == "final"
    assert all(r["stage"] == "final" for r in seen if r["mode"] == "shard_dem")
    with pytest.raises(ValueError, match="stage must be"):
        _run_pool(monkeypatch, fake_worker, stage="warmup")


def test_noise_zero_build_precedes_distance_and_other_builds(monkeypatch) -> None:
    fake_worker, seen = _recording_worker(columns=None, rss=None)
    payload = _run_pool(monkeypatch, fake_worker, noise_values=(0.005, 0.003))[0]
    assert payload["status"] == "ok"
    order = [(r["mode"], r.get("noise_index")) for r in seen]
    first_build = order.index(("build_dem", 0))
    assert first_build == 0
    assert order.index(("distance_job", 0)) > first_build
    assert order.index(("build_dem", 1)) > first_build
    assert sum(1 for m, _ in order if m == "build_dem") == 2
    assert sum(1 for m, _ in order if m == "distance_job") == 1


def test_ru_maxrss_is_recorded_in_job_execution_only_when_reported(monkeypatch) -> None:
    calls = []

    def fake(payload, *, timeout_s):
        calls.append(payload)
        return {"status": "ok", "results": {}, "ru_maxrss": 2**30}, {"timeout": False, "returncode": 0, "stderr_tail": ""}

    monkeypatch.setattr(pool_mod, "run_worker_request", fake)
    outcome = pool_mod.run_worker_request_with_retries(
        {"mode": "shard_dem", "shots": 10}, verification=VerificationConfig()
    )
    assert outcome["job_execution"]["ru_maxrss"] == 2**30

    def fake_without(payload, *, timeout_s):
        return {"status": "ok", "results": {}}, {"timeout": False, "returncode": 0, "stderr_tail": ""}

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_without)
    outcome = pool_mod.run_worker_request_with_retries(
        {"mode": "shard_dem", "shots": 10}, verification=VerificationConfig()
    )
    assert "ru_maxrss" not in outcome["job_execution"]
    merged = merge_shard_results(
        [{"status": "ok", "shots": 1, "failures": 0, "job_execution": {"attempts": 1, "ru_maxrss": 5}}, None],
        decoder=TELESCOPING,
    )
    assert merged["shard_max_rss_bytes"] == [5, None]


def test_resolve_workers_caps_by_memory(monkeypatch) -> None:
    monkeypatch.setenv("OMNIQEC_TOTAL_MEM_BYTES", str(8 * 2**30))
    monkeypatch.setenv("OMNIQEC_WORKER_RSS_BYTES", str(2 * 2**30))
    assert pool_mod.worker_memory_cap() == 4
    assert pool_mod.resolve_workers(127) == 4
    assert pool_mod.resolve_workers(3) == 3
    assert pool_mod.resolve_workers(1) == 1
    monkeypatch.setenv("OMNIQEC_WORKER_RSS_BYTES", str(768 * 2**20))
    assert pool_mod.resolve_workers(127) == 10
    # A disabled allowance removes the cap; a tiny total never goes below 1.
    monkeypatch.setenv("OMNIQEC_WORKER_RSS_BYTES", "0")
    assert pool_mod.worker_memory_cap() is None
    assert pool_mod.resolve_workers(127) == 127
    monkeypatch.setenv("OMNIQEC_WORKER_RSS_BYTES", str(2 * 2**30))
    monkeypatch.setenv("OMNIQEC_TOTAL_MEM_BYTES", str(2**20))
    assert pool_mod.resolve_workers(8) == 1
    assert pool_mod.worker_memory_cap(total_bytes=100 * 2**30, per_worker_bytes=768 * 2**20) == 133
    assert pool_mod.DEFAULT_WORKER_RSS_BYTES == 2 * 2**30


# ---------------------------------------------------------------------------
# CLI: shots derive from the budget policy
# ---------------------------------------------------------------------------


def _cli_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    args = dict(
        strict_turns=True,
        provider="openai",
        budget=360,
        out=str(tmp_path),
        weight_limit=None,
        families=None,
        p=0.005,
        noise_profile="uniform_paper_v1",
        turns=1,
        proposal_count=24,
        generations_per_turn=1,
        rng_seeds_per_program=20,
        slow_workers=127,
        slow_shots=None,
        shot_shards=16,
        final_shot_shards=128,
        racing=False,
        racing_pilot=5000,
        racing_batch=5000,
        racing_confirm=20_000,
        racing_max=50_000,
        decoders=None,
        circuit_timeout_s=None,
        build_timeout_s=None,
        decode_shard_timeout_s=None,
        distance_timeout_s=None,
        timeout_retries=None,
        decode_shot_budget_s=None,
        transient_retries=None,
        final_shots=None,
        feedback_mode="both",
        resume=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_cli_parser_defaults_shots_to_none_and_lists_the_telescoping_decoder(monkeypatch, tmp_path) -> None:
    import omniqec_agentic.cli as cli

    captured: dict[str, object] = {}

    def fake_command(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "run_campaign_command", fake_command)
    assert cli.main(["run-campaign", "--budget", "360", "--out", str(tmp_path)]) == 0
    assert captured["slow_shots"] is None and captured["final_shots"] is None
    assert cli.main([
        "run-campaign", "--budget", "360", "--out", str(tmp_path),
        "--decoders", "telescoping_cpu",
    ]) == 0
    assert captured["decoders"] == ["telescoping_cpu"]


def test_cli_derives_shots_from_the_policy_and_refuses_overrides_under_strict(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)
    # strict runs pin the provider's model id through the live client; the
    # test only exercises the launch-time policy checks.
    monkeypatch.setattr(cli, "resolve_model_name", lambda provider, config: "claude-test")
    assert cli.run_campaign_command(_cli_args(tmp_path)) == 0
    assert captured["verification"].shots == 25_000  # 2026-09-14: b360 shares the 25k/25k policy
    assert captured["final_shots"] == 25_000
    assert captured["verification"].decoders == (DecoderName.TELESCOPING,)
    captured.clear()
    assert cli.run_campaign_command(_cli_args(tmp_path, budget=144)) == 0
    assert captured["verification"].shots == 25_000
    assert captured["final_shots"] == 25_000
    with pytest.raises(SystemExit, match=r"--slow-shots at 25000 for --budget 360"):
        cli.run_campaign_command(_cli_args(tmp_path, slow_shots=20_000))
    with pytest.raises(SystemExit, match=r"--final-shots at 25000 for --budget 360"):
        cli.run_campaign_command(_cli_args(tmp_path, final_shots=200_000))
    with pytest.raises(SystemExit, match="telescoping_cpu-only"):
        cli.run_campaign_command(_cli_args(tmp_path, decoders=["bposd"]))
    # Outside strict the explicit values reach run_campaign (a variant).
    captured.clear()
    assert cli.run_campaign_command(
        _cli_args(tmp_path, strict_turns=False, provider="dry_run", slow_shots=20_000, final_shots=200_000)
    ) == 0
    assert captured["verification"].shots == 20_000
    assert captured["final_shots"] == 200_000


def test_search_and_final_fixture_shots_are_the_policy_values() -> None:
    assert (SEARCH_SHOTS, FINAL_SHOTS) == (25_000, 25_000)
    assert TELESCOPING_BUILD == TELESCOPING_PROTOCOL_FIELDS
    assert "s4_time_s" not in TELESCOPING_BUILD and "fallback" not in TELESCOPING_BUILD
    assert TELESCOPING == "telescoping_cpu"


# ---------------------------------------------------------------------------
# Cross-agent requests resolved here (tmp/v36_notes): SEC retag suffix with
# the v3.6 certification fields; telescoping stage metrics aggregated per basis;
# decode_infeasible_hashes as a checkpointed state channel.
# ---------------------------------------------------------------------------


def test_sec_variant_suffix_encodes_the_certification_fields_only_when_they_differ() -> None:
    from omniqec_agentic.config import SecPolicyConfig

    v35_style = SecPolicyConfig(sec_candidates=8, sec_cert_iters=50, sec_cert_seed=2)
    assert _snapshot(verification=VerificationConfig(sec_policy=v35_style))["protocol_profile"] == (
        f"{PROFILE}+secK8I50S2"
    )
    seeds_only = SecPolicyConfig(sec_cert_seeds=1)
    assert _snapshot(verification=VerificationConfig(sec_policy=seeds_only))["protocol_profile"] == (
        f"{PROFILE}+secK16I400S0N1X4R4F2000"
    )
    refine_only = SecPolicyConfig(sec_refine_top=0, sec_refine_iters=100)
    assert _snapshot(verification=VerificationConfig(sec_policy=refine_only))["protocol_profile"] == (
        f"{PROFILE}+secK16I400S0N3X4R0F100"
    )
    with pytest.raises(ValueError, match="cert_seeds=3"):
        _strict(verification=VerificationConfig(sec_policy=seeds_only), n_budget=144)


def test_merge_aggregates_the_telescoping_stage_metrics_per_basis() -> None:
    def cell(counts, fails, times, declared):
        return {
            "status": "ok", "shots": 10, "failures": 1, "runtime_s": 1.0,
            "metrics": {
                "requested_shots": 10, "stage_counts": counts, "stage_failures": fails,
                "stage_time_s": times, "declared_failures": declared, "nc_policy": "fail",
                "decode_stage": "final", "num_matrix_columns": 1200,
            },
        }

    merged = merge_shard_results(
        [
            cell({"S3A": 8, "S3B": 1, "NC": 1}, {"NC": 1}, {"S3A": 0.5, "S3C": 2.0}, 1),
            cell({"S3A": 9, "S3B": 1, "NC": 0}, {}, {"S3A": 0.25, "S3C": 0.0}, 0),
        ],
        decoder=TELESCOPING,
    )
    metrics = merged["metrics"]
    assert metrics["stage_counts"] == {"S3A": 17, "S3B": 2, "NC": 1}
    assert metrics["stage_failures"] == {"NC": 1}
    assert metrics["stage_time_s"] == {"S3A": 0.75, "S3C": 2.0}
    assert metrics["declared_failures"] == 1
    assert metrics["decode_stage"] == "final" and metrics["num_matrix_columns"] == 1200
    # The cascade-era scalars never appear (no S4 stage under the telescoping
    # CPU protocol) and there is no alias histogram key either.
    assert not any(key.startswith("s4_") for key in metrics)
    assert "cascade_stages" not in metrics
    # A shard set without the telescoping keys keeps the v3.4 metrics untouched.
    plain = merge_shard_results(
        [{"status": "ok", "shots": 10, "failures": 0, "metrics": {"requested_shots": 10}}],
        decoder="beam",
    )
    assert "stage_counts" not in plain["metrics"] and "declared_failures" not in plain["metrics"]


def test_campaign_state_declares_the_decode_infeasible_channel() -> None:
    from omniqec_agentic.workflow.state import CampaignState

    hints = CampaignState.__annotations__
    assert "decode_infeasible_hashes" in hints and "unmeasurable_hashes" in hints


def test_validator_pins_the_schedule_certification_fields() -> None:
    good = {
        "sec_policy_id": "twobasis_nonint_v1", "candidates": 16, "cert_iters": 400,
        "cert_seeds": 3, "refine_iters": 2000, "depth": 16,
    }
    candidate = {"status": "ok", "results": [{"status": "ok", "basis": {"X": {"memory_experiment": {"schedule": dict(good)}}}}]}
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "candidates"
        _write_json(directory / "c.json", candidate)
        validator._validate_candidate_schedules(directory, "x")
        bad = json.loads(json.dumps(candidate))
        bad["results"][0]["basis"]["X"]["memory_experiment"]["schedule"]["cert_seeds"] = 1
        _write_json(directory / "c.json", bad)
        with pytest.raises(ValueError, match="schedule cert_seeds=1"):
            validator._validate_candidate_schedules(directory, "x")
        bad["results"][0]["basis"]["X"]["memory_experiment"]["schedule"]["cert_seeds"] = 3
        del bad["results"][0]["basis"]["X"]["memory_experiment"]["schedule"]["refine_iters"]
        _write_json(directory / "c.json", bad)
        with pytest.raises(ValueError, match="schedule refine_iters=None"):
            validator._validate_candidate_schedules(directory, "x")
