"""Regressions for formal campaigns that must complete every audited turn.

v3.6 (telescoping_structural_v3_7, docs/06 §1, user decision 2026-09-03):
the fixtures below are the v3.6 artifact shapes -- decoder telescoping_cpu
with the frozen telescoping blob (every TelescopingDecoderConfig field,
budget independent), the budget policy (3 keys) / n_budget / decoder /
telescoping snapshot keys, the SEC policy with the multi-seed certification
fields, the telescoping-decoder version + vendored commit in
resolved_versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omniqec_agentic.config import (
    formal_racing,
    DEFAULT_SEC_POLICY,
    TELESCOPING_VENDORED_COMMIT,
    DecoderName,
    StructuralSearchConfig,
    TelescopingDecoderConfig,
    budget_policy,
)
from omniqec_agentic.hpc import validate_fig2_campaign as validator

# Per-turn generator budget: the fixtures follow the validator's frozen
# protocol (2026-09-10: 24 programs x 2 generations per turn, Q = k*d^2).
PROPOSALS = validator.FORMAL_PROPOSAL_COUNT
GENERATIONS = validator.FORMAL_GENERATIONS_PER_TURN
PROPOSALS_PER_TURN = validator.FORMAL_PROPOSALS_PER_TURN
from omniqec_agentic.hpc.validate_fig2_campaign import (
    FORMAL_FINAL_SHOT_SHARDS,
    FORMAL_JOB_POLICY,
    FORMAL_SHOT_SHARDS,
    _expected_final_base_seed,
    validate,
)
from omniqec_agentic.workflow.graph import (
    _audit_turn_execution,
    _complete_turn_audit,
    _stop_decision,
    _validate_all_turn_audits,
    final_rng_base_seed,
)

FINAL_BASE_SEED = final_rng_base_seed("final_confirmation_v1")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _strict_state(tmp_path: Path) -> dict:
    return {
        "run_dir": str(tmp_path),
        "turn": 0,
        "protocol": {
            "strict_turns": True,
            "required_provider": "deepseek",
            "required_model": "deepseek-v4-pro",
            "proposal_count": PROPOSALS,
            "generations_per_turn": GENERATIONS,
            "rng_seeds_per_program": validator.FORMAL_RNG_SEEDS_PER_PROGRAM,
            "max_turns": 1,
        },
    }


def _strict_plan() -> dict:
    return {
        "policy": "agentic_v1",
        "provider": "deepseek",
        "proposal_count": PROPOSALS,
        "generations": GENERATIONS,
        "rng_seeds_per_program": validator.FORMAL_RNG_SEEDS_PER_PROGRAM,
    }


def _strict_summary(tmp_path: Path) -> SimpleNamespace:
    turn_dir = tmp_path / "turns" / "turn_0000"
    fast_dir = turn_dir / "cycle" / "fast"
    summary_path = fast_dir / "funsearch_summary.json"
    generations_dir = fast_dir / "generations"
    _write_json(
        turn_dir / "cycle" / "agentic_cycle_summary.json",
        {
            "status": "ok",
            "provider": "deepseek",
            "proposal_count": PROPOSALS,
            "generations": GENERATIONS,
        },
    )
    _write_json(
        summary_path,
        {
            "schema": "funsearch_summary",
            "provider": "deepseek",
            "proposal_count_requested_total": PROPOSALS_PER_TURN,
            "proposal_count_missing_total": 0,
            "status_counts": {"ok": GENERATIONS},
            "rng_seeds_per_program": validator.FORMAL_RNG_SEEDS_PER_PROGRAM,
        },
    )
    _write_json(turn_dir / "plan.json", _strict_plan())
    _write_jsonl(
        turn_dir / "orchestrator" / "llm_calls.jsonl",
        [{"model": {"provider": "deepseek", "model": "deepseek-v4-pro"}}],
    )
    for generation in range(1, GENERATIONS + 1):
        _write_jsonl(
            generations_dir / f"g{generation:04d}" / "llm_raw.jsonl",
            [
                {
                    "status": "ok",
                    "engine_info": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-pro",
                    },
                    "proposal_count_accumulated": PROPOSALS,
                }
            ],
        )
    payload = {
        "provider": "deepseek",
        "proposal_count": PROPOSALS,
        "generations": GENERATIONS,
        "status": "ok",
        "funsearch_summary": {
            "provider": "deepseek",
            "proposal_count_requested_total": PROPOSALS_PER_TURN,
            "proposal_count_missing_total": 0,
            "status_counts": {"ok": GENERATIONS},
            "rng_seeds_per_program": validator.FORMAL_RNG_SEEDS_PER_PROGRAM,
            "artifact_paths": {
                "summary": str(summary_path),
                "generations": str(generations_dir),
            },
        },
    }
    return SimpleNamespace(to_dict=lambda: payload)


def test_strict_turn_counts_only_after_execution_and_digest_audits(tmp_path) -> None:
    state = _strict_state(tmp_path)

    audit = _audit_turn_execution(state, _strict_plan(), _strict_summary(tmp_path))
    assert audit["status"] == "execution_complete"

    digest_path = tmp_path / "turns" / "turn_0000" / "digest.json"
    _write_json(digest_path, {"turn": 0})
    _complete_turn_audit(state, digest_path=digest_path)

    summary = _validate_all_turn_audits(state)
    assert summary["completed"] == 1
    assert summary["turns"] == [0]
    assert summary["fallback_turns"] == 0


def test_strict_turn_rejects_raw_model_mismatch(tmp_path) -> None:
    state = _strict_state(tmp_path)
    summary = _strict_summary(tmp_path)
    raw_path = (
        tmp_path
        / "turns"
        / "turn_0000"
        / "cycle"
        / "fast"
        / "generations"
        / "g0001"
        / "llm_raw.jsonl"
    )
    _write_jsonl(
        raw_path,
        [{"status": "ok", "engine_info": {"provider": "deepseek", "model": "wrong"}}],
    )

    with pytest.raises(RuntimeError, match="generator call model mismatch"):
        _audit_turn_execution(state, _strict_plan(), summary)


def test_campaign_stops_only_after_all_sixty_turns() -> None:
    state = {"protocol": {"max_turns": 60}}
    for turns_used in range(60):
        assert _stop_decision(state, [], {}, {"turns_used": turns_used}) == {
            "finalize": False
        }
    assert _stop_decision(state, [], {}, {"turns_used": 60}) == {
        "finalize": True,
        "reason": "max_turns",
    }


# v3.6 fixture protocol: budget point b144 (small-budget policy: 20k / 200k
# shots per basis), decoder telescoping_cpu with the frozen blob (the same
# at every budget point: no field of the telescoping decoder is per-point).
N_BUDGET = 144
POLICY = budget_policy(N_BUDGET)
SEARCH_SHOTS = int(POLICY["search_shots"])
FINAL_SHOTS = int(POLICY["final_shots"])
TELESCOPING = DecoderName.TELESCOPING.value
TELESCOPING_BUILD = TelescopingDecoderConfig().to_dict()
# Historical BPOSD tuple (dev-only decoder since v3.6): recorded in the
# decoder_build blob like every other decoder config, never checked.
BPOSD_BUILD = {
    "preset": "2d",
    "bp_method": "product_sum",
    "max_iter": 50,
    "osd_method": "osd_cs",
    "osd_order": 10,
}
# v3.5: the noise model is a protocol field — the paper's SI C uniform model
# (no idle noise) at p=0.005, recorded in the snapshot, every run config and
# every measured row.
NOISE_SWEEP = {
    "profile": "uniform_paper_v1",
    "axis": "p",
    "values": [0.005],
    "source_kind": "paper_protocol_parameter",
    "source_reference": "OmniQEC paper SI C uniform v1",
}
NOISE_POINT = {"profile": "uniform_paper_v1", "axis": "p", "value": 0.005}
# v3.5 SEC: the circuit policy is a protocol field too (twobasis_nonint_v1,
# K=16 candidate orders per basis, 400 ISD iterations, seed 0), recorded in
# the snapshot (three top-level keys + the blob) and every run config.
# v3.6: the blob carries the multi-seed certification fields too
# (sec_cert_seeds=3, sec_cert_iters_scale=4.0, sec_refine_top=4,
# sec_refine_iters=2000).
SEC_POLICY_BLOCK = DEFAULT_SEC_POLICY.to_dict()
# v3.6: profile telescoping_structural_v3_7, reducer structural_proxy_pareto_v3_weight
# (coordinate_cap=2); the validator checks every serialized key.
STRUCTURAL_BLOCK = StructuralSearchConfig().to_dict()


def _strict_report() -> dict:
    return {
        "schema": "campaign_report",
        "turns_completed": 1,
        "stop": {"reason": "max_turns"},
        "final_slow_status": "ok",
        "final_rounds": 12,
        "final_rng_base_seed": FINAL_BASE_SEED,
        "spec": {"qubit_budget": N_BUDGET, "weight_limit": None},
        "protocol": {
            "protocol_profile": "telescoping_structural_v3_7",
            "decoder": TELESCOPING,
            "telescoping": dict(TELESCOPING_BUILD),
            "n_budget": N_BUDGET,
            "budget_policy": dict(POLICY),
            "max_turns": 1,
            "strict_turns": True,
            "fixed_program_count": True,
            "required_provider": "deepseek",
            "required_model": "deepseek-v4-pro",
            "proposal_count": PROPOSALS,
            "generations_per_turn": GENERATIONS,
            "rng_seeds_per_program": validator.FORMAL_RNG_SEEDS_PER_PROGRAM,
            "quality_metric": validator.FORMAL_QUALITY_METRIC,
            "promote_top": 6,
            "anchor_p": 0.005,
            "final_p_values": [0.005],
            "noise_profile": "uniform_paper_v1",
            "noise_values": [0.005],
            "sec_policy_id": "twobasis_nonint_v1",
            "sec_candidates": 16,
            "sec_cert_iters": 400,
            "sec_policy": dict(SEC_POLICY_BLOCK),
            "final_shots": FINAL_SHOTS,
            # v3.4: search shards stay 16; the final stage runs 128 shards.
            "shot_shards": FORMAL_SHOT_SHARDS,
            "final_shot_shards": FORMAL_FINAL_SHOT_SHARDS,
            "rounds_policy": {
                "search": "max_Tc_d",
                "search_rounds": 12,
                "final": "max_Tc_d",
                "final_rounds": 12,
                "final_rng_namespace": "final_confirmation_v1",
                # 2026-09-10: slow-promotion floor d_search >= 5 (was 3).
                "min_promotion_distance": 5,
            },
            "structural_retention": dict(STRUCTURAL_BLOCK),
            # 2026-09-11: the formal protocol races (early stop pilot10x_v1).
            "racing": formal_racing(SEARCH_SHOTS).to_dict(),
            "verification": {
                "decoders": [TELESCOPING],
                "decoder_build": {"telescoping": dict(TELESCOPING_BUILD), "bposd": dict(BPOSD_BUILD)},
                "shots": SEARCH_SHOTS,
                "noise": dict(NOISE_SWEEP),
                "sec_policy": dict(SEC_POLICY_BLOCK),
                **dict(FORMAL_JOB_POLICY),
            },
        },
        "turn_audit": {
            "enabled": True,
            "schema": "campaign_turn_audit_summary_v1",
            "expected": 1,
            "completed": 1,
            "fallback_turns": 0,
            "turns": [0],
        },
        "final_measurement_audit": {
            "enabled": True,
            "schema": "campaign_final_measurement_audit_v1",
            "status": "complete",
            "expected_p": 0.005,
            "shots_per_basis": FINAL_SHOTS,
            "expected_rounds": 12,
            "measured_rows": 1,
            "errors": [],
        },
        "final_frontier": [
            {
                "program_id": "g0001_p0001",
                "noise_point": dict(NOISE_POINT),
                "noise_value": 0.005,
                "shots_x": FINAL_SHOTS,
                "shots_z": FINAL_SHOTS,
                "total_shots": 2 * FINAL_SHOTS,
                "d_search": 5,
                "rounds": 12,
                "decoder": TELESCOPING,
                "decoder_config": dict(TELESCOPING_BUILD),
            }
        ],
    }


def _slow_run_config(
    rounds: int | None = None,
    *,
    seed: int = 0,
    shot_shards: int = FORMAL_SHOT_SHARDS,
    shots: int | None = None,
    stage: str | None = None,
) -> dict:
    # v3.6: the executed shots per basis and the stage tag travel in the run
    # config; a 128-shard config is the final stage (policy final shots).
    is_final = int(shot_shards) == FORMAL_FINAL_SHOT_SHARDS
    if shots is None:
        shots = FINAL_SHOTS if is_final else SEARCH_SHOTS
    if stage is None:
        stage = "final" if is_final else "search"
    verification = {
        "decoders": [TELESCOPING],
        "decoder_build": {"telescoping": dict(TELESCOPING_BUILD), "bposd": dict(BPOSD_BUILD)},
        "shots": int(shots),
        "noise": dict(NOISE_SWEEP),
        "sec_policy": dict(SEC_POLICY_BLOCK),
        **dict(FORMAL_JOB_POLICY),
        # The executed d_search protocol travels in the run config since the
        # single-source fix; the validator checks it on every layer.
        "structural": dict(STRUCTURAL_BLOCK),
        "proxy_iters": 120,
    }
    if rounds is not None:
        verification["rounds"] = rounds
    config = {
        "schema": "slow_queue_run_config",
        "seed": int(seed),
        "shot_shards": int(shot_shards),
        "stage": stage,
        "verification": verification,
    }
    if stage == "search":
        # The slow stage records the racing block it ran under (2026-09-11).
        config["racing"] = formal_racing(int(shots)).to_dict()
    return config


def _rankings(rounds: int, d_search: int = 5, shots: int = SEARCH_SHOTS, **extra) -> dict:
    return {
        "schema": "slow_rankings",
        "rankings": [
            {
                "program_id": "g0001_p0001",
                "status": "ok",
                "decoder": TELESCOPING,
                "decoder_config": dict(TELESCOPING_BUILD),
                "noise_point": dict(NOISE_POINT),
                "noise_value": 0.005,
                "d_search": d_search,
                "rounds": rounds,
                # 2026-09-11: a confirmed measurement decoded the full search shots.
                "shots_x": int(shots),
                "shots_z": int(shots),
                **extra,
            }
        ],
    }


def _proxy_search_provenance() -> dict:
    return {
        "schema": "proxy_distance_search",
        "algorithm_id": "css_permutation_systematic_v2_independent_xz",
        "seeds": [0, 1, 2, 3],
        "iters_per_seed": 120,
        "sectors": ["X", "Z"],
        "per_seed": [
            {"seed": seed, "d_x": 5, "d_z": 5} for seed in (0, 1, 2, 3)
        ],
        "d_x": 5,
        "d_z": 5,
        "d_search": 5,
        "available": True,
    }


def _structural_elites_artifact() -> dict:
    return {
        "schema": "structural_proxy_elites",
        "reducer_id": STRUCTURAL_BLOCK["reducer_id"],
        "profile_id": "telescoping_structural_v3_7",
        "limit": 16,
        "objectives": ["k", "d_search"],
        "count": 1,
        "elites": [
            {
                "schema": "structural_proxy_elite",
                "code_hash": "aa" * 16,
                "program_id": "g0001_p0001",
                "k": 8,
                "d_search": 5,
                "proxy_search": _proxy_search_provenance(),
            }
        ],
    }


RESOLVED_VERSIONS = {
    "telescoping-decoder": "0.1.0",
    "telescoping_commit": TELESCOPING_VENDORED_COMMIT,
    "ldpc": "2.4.1",
    "bposd_backend": "ldpc.BpOsdDecoder",
}


def _write_validated_run(tmp_path: Path, report: dict) -> None:
    state = _strict_state(tmp_path)
    _audit_turn_execution(state, _strict_plan(), _strict_summary(tmp_path))
    digest_path = tmp_path / "turns" / "turn_0000" / "digest.json"
    _write_json(digest_path, {"turn": 0})
    _complete_turn_audit(state, digest_path=digest_path)
    # Every layer the strict validator now reads: the resolved backend, the
    # turn's own slow run config and rows, and the final stage's.
    # v3.6: the formal decoder is the vendored telescoping-decoder package at
    # the frozen commit; ldpc / relay_bp / scipy are recorded for the dev-only
    # decoders and never required.
    _write_json(tmp_path / "resolved_versions.json", dict(RESOLVED_VERSIONS))
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12))
    _write_json(slow_dir / "rankings.json", _rankings(12))
    fast_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "fast"
    _write_json(fast_dir / "structural_proxy_elites.json", _structural_elites_artifact())
    final_dir = tmp_path / "final_slow" / "T12"
    _write_json(
        final_dir / "slow_run_config.json",
        _slow_run_config(rounds=12, seed=FINAL_BASE_SEED, shot_shards=FORMAL_FINAL_SHOT_SHARDS),
    )
    _write_json(final_dir / "rankings.json", _rankings(12))
    _write_json(tmp_path / "campaign_report.json", report)


def test_validator_execution_policy_mirror_matches_package_defaults() -> None:
    """The standalone login-node validator must track the executed defaults."""

    from omniqec_agentic.config import VerificationConfig
    from omniqec_agentic.workflow.state import (
        FORMAL_FINAL_SHOT_SHARDS as STATE_FINAL_SHARDS,
        FORMAL_SHOT_SHARDS as STATE_SHARDS,
    )

    verification = VerificationConfig()
    assert FORMAL_SHOT_SHARDS == STATE_SHARDS == 16
    assert validator.FORMAL_FINAL_SHOT_SHARDS == STATE_FINAL_SHARDS == 128
    assert validator.PROTOCOL_PROFILE == "telescoping_structural_v3_7"
    # v3.4 policy keys: the validator mirror must carry every one of them.
    assert set(FORMAL_JOB_POLICY) == {
        "build_timeout_s",
        "decode_shard_timeout_s",
        "distance_timeout_s",
        "timeout_retries",
        "decode_shot_budget_s",
        "transient_retries",
    }
    for key, expected in FORMAL_JOB_POLICY.items():
        assert getattr(verification, key) == expected


def test_validator_rejects_non_protocol_snapshot_sharding(tmp_path) -> None:
    report = _strict_report()
    report["protocol"]["shot_shards"] = 8
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match="protocol snapshot: shot_shards=8"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_validator_rejects_non_protocol_snapshot_timeout_policy(tmp_path) -> None:
    report = _strict_report()
    report["protocol"]["verification"]["build_timeout_s"] = 120.0
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match="protocol snapshot: build_timeout_s=120.0"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


@pytest.mark.parametrize("stage", ["search", "final"])
def test_validator_rejects_non_protocol_run_config_sharding(
    tmp_path: Path, stage: str
) -> None:
    _write_validated_run(tmp_path, _strict_report())
    if stage == "search":
        path = (
            tmp_path
            / "turns"
            / "turn_0000"
            / "cycle"
            / "slow"
            / "slow_run_config.json"
        )
        seed = 0
    else:
        path = tmp_path / "final_slow" / "T12" / "slow_run_config.json"
        seed = FINAL_BASE_SEED
    config = _slow_run_config(rounds=12, seed=seed)
    config["shot_shards"] = 8
    _write_json(path, config)

    with pytest.raises(ValueError, match="shot_shards=8"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


@pytest.mark.parametrize("stage", ["search", "final"])
def test_validator_rejects_non_protocol_candidate_job_config(
    tmp_path: Path, stage: str
) -> None:
    _write_validated_run(tmp_path, _strict_report())
    if stage == "search":
        candidate_dir = (
            tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "candidates"
        )
    else:
        candidate_dir = tmp_path / "final_slow" / "T12" / "candidates"
    _write_json(
        candidate_dir / "candidate.json",
        {
            "status": "ok",
            "job_config": {
                "workers": 127,
                "shot_shards": 8,
                **dict(FORMAL_JOB_POLICY),
            },
        },
    )

    with pytest.raises(ValueError, match="candidate.json job_config: shot_shards=8"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_validator_rejects_non_protocol_typed_timeout(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    path = (
        tmp_path
        / "turns"
        / "turn_0000"
        / "cycle"
        / "slow"
        / "slow_run_config.json"
    )
    config = _slow_run_config(rounds=12)
    config["verification"]["decode_shard_timeout_s"] = 17.0
    _write_json(path, config)

    with pytest.raises(ValueError, match="decode_shard_timeout_s=17.0"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_independent_validator_checks_raw_turns_and_final_shots(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())

    assert validate(
        tmp_path,
        turns=1,
        p_value=0.005,
        final_shots=FINAL_SHOTS,
        model="deepseek-v4-pro",
    ) == 1


def test_caller_cannot_authorize_a_non_bposd_protocol(tmp_path) -> None:
    """Under v3.3 the decoder IS the protocol.

    The validator used to accept whatever decoder set the caller declared, so a
    job script could authorize its own protocol variant as valid. A different
    decoder is a different experiment; passing it must fail, not relabel."""

    report = _strict_report()
    report["protocol"]["verification"]["decoders"] = ["lsd"]
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match="caller declared decoders"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
            decoders=["lsd"],
        )


def test_validator_rejects_a_non_protocol_telescoping_configuration(tmp_path) -> None:
    """A recorded telescoping blob that is not the frozen config is drift,
    wherever it is recorded (v3.6: package defaults, b_quorum 3, NC = fail)."""

    report = _strict_report()
    report["protocol"]["verification"]["decoder_build"]["telescoping"]["b_quorum"] = 2
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match="telescoping b_quorum=2"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_validator_rejects_a_per_candidate_final_round_count(tmp_path) -> None:
    """max_Tc_d: a d=5 finalist must sit at the T_c=12 floor; T=d (the retired
    rule, T=max(3, d_proxy)) is rejected."""

    report = _strict_report()
    report["final_frontier"][0]["rounds"] = 5
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match=r"rounds=5, expected 12 \(max\(T_c=12, d_search=5\)\)"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_validator_requires_t_equal_d_above_the_floor(tmp_path) -> None:
    """max_Tc_d (2026-09-10): a d_search=14 code is measured at T=14 in both
    stages. Recording the floor T=12 for it fails; recording 14 passes."""

    report = _strict_report()
    report["final_frontier"][0]["d_search"] = 14
    report["final_frontier"][0]["rounds"] = 12
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match=r"final row 0: rounds=12, expected 14"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    report["final_frontier"][0]["rounds"] = 14
    _write_validated_run(tmp_path, report)
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "rankings.json", _rankings(12, d_search=14))
    with pytest.raises(ValueError, match=r"search rankings: row g0001_p0001 rounds=12, expected 14"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    _write_json(slow_dir / "rankings.json", _rankings(14, d_search=14))
    assert validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_an_undeclared_decoder_switch(tmp_path) -> None:
    """The decoder set decides which numbers the campaign produced, so a silent
    switch is exactly the drift this validator exists to catch."""

    report = _strict_report()
    report["protocol"]["verification"]["decoders"] = ["lsd"]
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match="decoder set mismatch"):
        validate(
            tmp_path,
            turns=1,
            p_value=0.005,
            final_shots=FINAL_SHOTS,
            model="deepseek-v4-pro",
        )


def test_validator_rejects_a_search_stage_that_measured_a_different_T(tmp_path) -> None:
    """Audit fix #1: "the config would have set T=12" is not evidence. A search
    ranking row recording T=5 — or no rounds at all — must fail, not pass."""

    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "rankings.json", _rankings(5))

    with pytest.raises(ValueError, match="rounds=5, expected 12"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    tampered = _rankings(12)
    del tampered["rankings"][0]["rounds"]
    _write_json(slow_dir / "rankings.json", tampered)
    with pytest.raises(ValueError, match="rounds=None, expected 12"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_a_search_run_config_without_t12(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=5))

    with pytest.raises(ValueError, match="slow run config: rounds=5"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_a_run_config_without_the_executed_structural_block(tmp_path) -> None:
    """Audit fix #2: the executors read verification.structural, so a run config
    that does not carry the frozen d_search protocol is a different experiment."""

    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    config = _slow_run_config(rounds=12)
    del config["verification"]["structural"]
    _write_json(slow_dir / "slow_run_config.json", config)

    with pytest.raises(ValueError, match="structural profile_id=None"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_declared_120_but_executed_7_iters(tmp_path) -> None:
    """Audit fix #2: the snapshot can DECLARE 120 iters/seed; the per-elite
    proxy_search table is what the executor actually did, and it wins."""

    _write_validated_run(tmp_path, _strict_report())
    artifact = _structural_elites_artifact()
    artifact["elites"][0]["proxy_search"]["iters_per_seed"] = 7
    fast_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "fast"
    _write_json(fast_dir / "structural_proxy_elites.json", artifact)

    with pytest.raises(ValueError, match="iters_per_seed=7"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_an_elite_with_no_proxy_provenance(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    artifact = _structural_elites_artifact()
    del artifact["elites"][0]["proxy_search"]
    fast_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "fast"
    _write_json(fast_dir / "structural_proxy_elites.json", artifact)

    with pytest.raises(ValueError, match="carries no proxy_search provenance"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_a_non_protocol_proxy_effort_cannot_claim_the_frozen_profile() -> None:
    """Audit fix #2 at the config layer: changing the per-seed effort retags the
    profile id, so 'declared v3.3, executed proxy7' is unrepresentable."""

    from omniqec_agentic.config import VerificationConfig

    v = VerificationConfig(proxy_iters=7)
    assert v.structural.proxy_iters_per_seed == 7
    assert v.structural.profile_id == "telescoping_structural_v3_7+proxy7"
    with pytest.raises(ValueError, match="disagree"):
        VerificationConfig(
            proxy_iters=7,
            structural=type(v.structural)(proxy_iters_per_seed=40),
        )


def test_validator_seed_formula_matches_the_graph() -> None:
    """The validator is standalone; its copy of the derivation must stay
    pinned to the one the campaign actually runs with."""

    assert _expected_final_base_seed("final_confirmation_v1") == FINAL_BASE_SEED
    assert FINAL_BASE_SEED >= 10**12


def test_validator_rejects_a_final_stage_without_the_fresh_seed(tmp_path) -> None:
    """Audit fix: 'fresh shots' must be a recorded, checked fact — a final
    stage rerun at seed 0 (or any search-range seed) reuses search keys."""

    _write_validated_run(tmp_path, _strict_report())
    final_dir = tmp_path / "final_slow" / "T12"
    # Keep the v3.4 final geometry valid so the seed check is what fails.
    _write_json(
        final_dir / "slow_run_config.json",
        _slow_run_config(rounds=12, seed=0, shot_shards=FORMAL_FINAL_SHOT_SHARDS),
    )

    with pytest.raises(ValueError, match="namespace-derived"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    report = _strict_report()
    report["final_rng_base_seed"] = 123
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="final_rng_base_seed=123"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_a_search_stage_running_in_the_final_namespace(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(
        slow_dir / "slow_run_config.json",
        _slow_run_config(rounds=12, seed=FINAL_BASE_SEED),
    )

    with pytest.raises(ValueError, match="not a search-range seed"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_rejects_fabricated_proxy_provenance(tmp_path) -> None:
    """Audit fix: length-only checks passed a fabricated per_seed table. The
    seeds must BE the declared seeds and the aggregates must equal the minima."""

    _write_validated_run(tmp_path, _strict_report())
    fast_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "fast"

    artifact = _structural_elites_artifact()
    for entry in artifact["elites"][0]["proxy_search"]["per_seed"]:
        entry["seed"] = 9
    _write_json(fast_dir / "structural_proxy_elites.json", artifact)
    with pytest.raises(ValueError, match="per_seed seeds"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    artifact = _structural_elites_artifact()
    artifact["elites"][0]["proxy_search"]["d_search"] = 4  # min is 5
    _write_json(fast_dir / "structural_proxy_elites.json", artifact)
    with pytest.raises(ValueError, match="aggregate d_x/d_z|not\\s+min"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")

    artifact = _structural_elites_artifact()
    artifact["elites"][0]["proxy_search"]["per_seed"][2]["d_z"] = -3
    _write_json(fast_dir / "structural_proxy_elites.json", artifact)
    with pytest.raises(ValueError, match="plausible distance"):
        validate(tmp_path, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_strict_snapshot_freezes_the_early_stop_policy_and_refuses_drift() -> None:
    """2026-09-11 (pilot10x_v1): a strict campaign freezes the formal racing
    policy; a racing block whose confirm/max shots drift from the budget
    policy, or that cannot execute (single worker), fails at snapshot time
    -- not after 30 turns in the validator."""

    from omniqec_agentic.config import RacingConfig, VerificationConfig
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    common = dict(
        verification=VerificationConfig(), max_turns=1, proposal_count=24,
        generations_per_turn=1, search_rounds=12, strict_turns=True,
        required_provider="deepseek", required_model="deepseek-v4-pro",
    )
    snapshot = build_protocol_snapshot(racing=formal_racing(25_000), slow_workers=4, **common)
    assert snapshot["racing"] == formal_racing(25_000).to_dict()
    assert snapshot["racing"]["policy_id"] == "pilot10x_maxerr100_v3" and snapshot["racing"]["enabled"] is True
    with pytest.raises(ValueError, match="must equal the budget policy's search shots"):
        build_protocol_snapshot(racing=RacingConfig(enabled=True), slow_workers=4, **common)
    with pytest.raises(ValueError, match="needs the job pool"):
        build_protocol_snapshot(racing=formal_racing(25_000), slow_workers=1, **common)
    # Racing OFF is still a legal (declared) variant at snapshot time; the
    # formal validator is what refuses it.
    off = build_protocol_snapshot(racing=RacingConfig(), slow_workers=4, **common)
    assert off["racing"]["enabled"] is False


def _retired_refusal_shape() -> None:
    """Kept only to document the pre-2026-09-11 behaviour (never called)."""

    from omniqec_agentic.config import RacingConfig, VerificationConfig
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    with pytest.raises(ValueError, match="freezes\\s+racing OFF"):
        build_protocol_snapshot(
            verification=VerificationConfig(),
            racing=RacingConfig(enabled=True),
            max_turns=1,
            proposal_count=24,
            generations_per_turn=1,
            slow_workers=1,
            search_rounds=12,
            strict_turns=True,
            required_provider="deepseek",
            required_model="deepseek-v4-pro",
        )


def _run_validator(run_dir):
    return validate(run_dir, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_requires_the_frozen_early_stop_policy(tmp_path) -> None:
    """2026-09-11: racing OFF (the v3.3-v3.6 fixed-shot measurement) is now
    the declared variant; the formal validator refuses it by name."""

    report = _strict_report()
    report["protocol"]["racing"] = {"enabled": False}
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="is not a formal policy"):
        _run_validator(tmp_path)


def test_validator_accepts_racing_stopped_rows_and_rejects_other_short_rows(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    # An early stop: stopped_by=racing with [pilot, search_shots) shots per basis.
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=2_000, stopped_by="racing"))
    _run_validator(tmp_path)
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=6_250, stopped_by="racing"))
    _run_validator(tmp_path)
    # Below the pilot: not a protocol early stop.
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=1_000, stopped_by="racing"))
    with pytest.raises(ValueError, match=r"expected \[2000, 25000\) per basis"):
        _run_validator(tmp_path)
    # A short row without the racing verdict is an incomplete measurement.
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=24_000))
    with pytest.raises(ValueError, match="expected 25000 per basis"):
        _run_validator(tmp_path)
    # A max_errors stop is a confirmed measurement under the capped policy
    # (pilot10x_maxerr100_v2), anywhere in [pilot, search_shots]...
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=6_250, stopped_by="max_errors"))
    _run_validator(tmp_path)
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=25_000, stopped_by="max_errors"))
    _run_validator(tmp_path)
    # ...under v3 the cap is decided from the first shard, so [250, 25000]
    # is confirmed and 100 shots (below one shard) is not.
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=1_000, stopped_by="max_errors"))
    _run_validator(tmp_path)
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=100, stopped_by="max_errors"))
    with pytest.raises(ValueError, match=r"expected \[250, 25000\] per basis"):
        _run_validator(tmp_path)
    # ...but not under a turn that ran the uncapped pilot10x_v1.
    from omniqec_agentic.config import formal_racing_v1

    v1_config = _slow_run_config(rounds=12)
    v1_config["racing"] = formal_racing_v1(SEARCH_SHOTS).to_dict()
    _write_json(slow_dir / "slow_run_config.json", v1_config)
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=6_250, stopped_by="max_errors"))
    with pytest.raises(ValueError, match="has no max_errors cap"):
        _run_validator(tmp_path)
    # Any other stopped_by is a defect.
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12))
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=6_250, stopped_by="timeout"))
    with pytest.raises(ValueError, match="stopped_by='timeout' is not a protocol outcome"):
        _run_validator(tmp_path)


def test_validator_checks_each_turn_against_the_recorded_policy_switch(tmp_path) -> None:
    """2026-09-11 evening: a campaign launched under pilot10x_v1 may switch to
    pilot10x_maxerr100_v2 at a turn boundary on resume. The snapshot records
    the switch turn; a turn's slow_run_config must carry the policy the
    history says it ran under, and its rows are judged by that policy."""

    from omniqec_agentic.config import formal_racing_v1

    v1 = formal_racing_v1(SEARCH_SHOTS).to_dict()
    v2 = formal_racing(SEARCH_SHOTS).to_dict()
    report = _strict_report()
    report["protocol"]["racing"] = v2
    report["protocol"]["runtime_overrides"] = [
        {"turn": 0, "applies_from_turn": 1, "from": {"racing": v1}, "to": {"racing": v2}}
    ]
    _write_validated_run(tmp_path, report)
    _write_json(tmp_path / "protocol_snapshot.json", report["protocol"])
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    # Turn 0 ran under v1 (switch at turn 1): a full-shot row is fine...
    v1_config = _slow_run_config(rounds=12)
    v1_config["racing"] = v1
    _write_json(slow_dir / "slow_run_config.json", v1_config)
    _write_json(slow_dir / "rankings.json", _rankings(12))
    _run_validator(tmp_path)
    # ...a max_errors row is not (v1 has no cap)...
    _write_json(slow_dir / "rankings.json", _rankings(12, shots=6_250, stopped_by="max_errors"))
    with pytest.raises(ValueError, match="has no max_errors cap"):
        _run_validator(tmp_path)
    # ...and a turn claiming v2 before the recorded switch contradicts the history.
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12))
    with pytest.raises(ValueError, match="switch history says 'pilot10x_v1'"):
        _run_validator(tmp_path)
    # An entry without applies_from_turn (written before the field existed)
    # cannot pin the turn: the order rule applies instead -- v1 then v2 is
    # fine, v2 then v1 is not, and the last turn must match the snapshot.
    report["protocol"]["runtime_overrides"] = [{"turn": 0, "from": {"racing": v1}, "to": {"racing": v2}}]
    _write_json(tmp_path / "campaign_report.json", report)
    _write_json(tmp_path / "protocol_snapshot.json", report["protocol"])
    _write_json(slow_dir / "slow_run_config.json", v1_config)
    _write_json(slow_dir / "rankings.json", _rankings(12))
    with pytest.raises(ValueError, match="the last turn ran under 'pilot10x_v1' but the snapshot's racing block"):
        _run_validator(tmp_path)  # turn 0 (the only turn) is v1 while the snapshot says v2
    _write_json(slow_dir / "slow_run_config.json", _slow_run_config(rounds=12))
    _run_validator(tmp_path)  # turn 0 under v2 = the snapshot: consistent
    # A switch to something that is not a formal policy is refused outright.
    bad = dict(v2); bad["max_errors"] = 7
    report["protocol"]["runtime_overrides"] = [{"turn": 1, "applies_from_turn": 1, "from": {"racing": v1}, "to": {"racing": bad}}]
    _write_json(tmp_path / "campaign_report.json", report)
    _write_json(tmp_path / "protocol_snapshot.json", report["protocol"])
    _write_json(slow_dir / "slow_run_config.json", v1_config)
    _write_json(slow_dir / "rankings.json", _rankings(12))
    with pytest.raises(ValueError, match="is not a formal policy"):
        _run_validator(tmp_path)


def test_resume_switches_between_formal_racing_policies_only() -> None:
    """checkpoint.py: v1 -> v2 is a recorded runtime override; any other
    racing change is still a protocol mismatch."""

    from omniqec_agentic.config import formal_racing_v1
    from omniqec_agentic.workflow.checkpoint import resume_runtime_overrides, validate_resume_protocol

    stored = {"racing": formal_racing_v1(25_000).to_dict(), "slow_workers": 127, "max_turns": 30}
    current = {"racing": formal_racing(25_000).to_dict(), "slow_workers": 127, "max_turns": 30}
    assert resume_runtime_overrides(stored, current) == {"racing": current["racing"]}
    validate_resume_protocol(stored, current)  # accepted
    assert resume_runtime_overrides(current, stored) == {"racing": stored["racing"]}  # and back
    drifted = {**current, "racing": {**current["racing"], "epsilon": 2.0}}
    assert resume_runtime_overrides(stored, drifted) == {}
    with pytest.raises(ValueError, match=r"differing keys: \['racing'\]"):
        validate_resume_protocol(stored, drifted)
    other_point = {**current, "racing": formal_racing(5_000).to_dict()}
    with pytest.raises(ValueError, match=r"differing keys: \['racing'\]"):
        validate_resume_protocol(stored, other_point)
