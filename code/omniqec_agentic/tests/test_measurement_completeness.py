"""v3.3 timeout recovery and fixed-shot completeness regressions.

v3.4 (2026-09-02) changes the frozen values these tests pin: the per-shot
decode budget, timeout_retries=0 and the decode_infeasible candidate terminal
state live in test_v34_decode_budget.py; the pins below track the v3.4
defaults."""

from __future__ import annotations

import copy

import pytest

from omniqec_agentic.config import StructuralSearchConfig, VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.common import verification_from_dict
from omniqec_agentic.evaluation.writeback import ranking_rows, slow_queue_status


def test_typed_timeout_defaults_serialize_and_legacy_artifacts_fallback() -> None:
    current = VerificationConfig()
    encoded = current.to_dict()
    assert encoded["build_timeout_s"] == 3600.0
    assert encoded["decode_shard_timeout_s"] == 1800.0
    assert encoded["distance_timeout_s"] == 1800.0
    # v3.4: no identical replay of a deterministic watchdog timeout; one
    # transient replay of a killed / DEM-less worker; 8 s per shot budget.
    assert encoded["timeout_retries"] == 0
    assert encoded["transient_retries"] == 1
    assert encoded["decode_shot_budget_s"] == 8.0
    restored = verification_from_dict(encoded)
    assert restored.build_timeout_s == current.build_timeout_s
    assert restored.decode_shard_timeout_s == current.decode_shard_timeout_s
    assert restored.distance_timeout_s == current.distance_timeout_s
    assert restored.timeout_retries == current.timeout_retries == 0
    assert restored.transient_retries == current.transient_retries == 1
    assert restored.decode_shot_budget_s == current.decode_shot_budget_s == 8.0

    legacy = verification_from_dict({"circuit_timeout_s": 77.0})
    assert legacy.build_timeout_s == 77.0
    assert legacy.decode_shard_timeout_s == 77.0
    assert legacy.distance_timeout_s == 77.0
    assert StructuralSearchConfig().profile_id == "telescoping_structural_v3_7"
    with pytest.raises(ValueError, match="timeout_retries"):
        VerificationConfig(timeout_retries=-1)


def test_timeout_retries_replay_the_identical_request_and_seed(monkeypatch) -> None:
    # v3.4 freezes timeout_retries=0 for the formal protocol, but the replay
    # semantics are unchanged when a run explicitly asks for retries.
    request = {
        "mode": "shard_dem",
        "base_seed": 12345,
        "shard": 7,
        "shot_shards": 16,
        "shots": 1250,
        "units": [{"decoder": "bposd", "basis": "X"}],
    }
    observed: list[dict] = []

    def fake_run_worker_request(payload, *, timeout_s):
        observed.append(copy.deepcopy(payload))
        if len(observed) < 3:
            return None, {"timeout": True, "timeout_s": timeout_s}
        return {"status": "ok", "results": {}}, {"timeout": False}

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_run_worker_request)
    outcome = pool_mod.run_worker_request_with_retries(
        request,
        verification=VerificationConfig(timeout_retries=2),
    )
    assert outcome["status"] == "ok"
    assert outcome["job_execution"]["attempts"] == 3
    assert observed == [request, request, request]
    assert request["base_seed"] == 12345


def test_distance_timeout_is_uncertified_not_a_decode_failure(monkeypatch) -> None:
    def always_timeout(payload, *, timeout_s):
        return None, {"timeout": True, "timeout_s": timeout_s}

    monkeypatch.setattr(pool_mod, "run_worker_request", always_timeout)
    outcome = pool_mod.run_worker_request_with_retries(
        {"mode": "distance_job", "seed": 9},
        verification=VerificationConfig(timeout_retries=2),
    )
    assert outcome["status"] == "uncertified_timeout"
    assert outcome["job_execution"]["attempts"] == 3


def _complete_result() -> dict:
    basis = {
        name: {
            "status": "ok",
            "shots": 20_000,
            "shard_statuses": ["ok", "ok"],
        }
        for name in ("X", "Z")
    }
    return {
        "status": "started",
        "basis": basis,
        "combined": {"per_logical_per_round": 1e-3},
    }


def test_fixed_shot_requires_every_configured_basis_and_shard() -> None:
    complete = _complete_result()
    assert pool_mod.fixed_shot_completion_errors(
        complete,
        required_bases=("X", "Z"),
        expected_shots=20_000,
        expected_shards=2,
    ) == []

    missing_basis = _complete_result()
    del missing_basis["basis"]["Z"]
    assert "missing_Z_basis" in pool_mod.fixed_shot_completion_errors(
        missing_basis,
        required_bases=("X", "Z"),
        expected_shots=20_000,
        expected_shards=2,
    )

    partial_shard = _complete_result()
    partial_shard["basis"]["X"].update(
        status="partial_shards", shots=10_000, shard_statuses=["ok", "timeout"]
    )
    errors = pool_mod.fixed_shot_completion_errors(
        partial_shard,
        required_bases=("X", "Z"),
        expected_shots=20_000,
        expected_shards=2,
    )
    assert "X_shard_incomplete" in errors
    assert "X_shots=10000!=20000" in errors


def test_blocked_measurement_cannot_publish_plpr_or_make_queue_ok(tmp_path) -> None:
    payload = {
        "program_id": "blocked",
        "status": "measurement_blocked",
        "failure_reason": "Z shard timed out",
        "results": [
            {
                "status": "ok",  # defense in depth against a stale producer
                "decoder": "bposd",
                "combined": {
                    "block_ler": 0.1,
                    "per_logical_per_round": 1e-3,
                },
                "basis": {},
            }
        ],
    }
    row = ranking_rows(payload, tmp_path / "blocked.json")[0]
    assert row["status"] == "measurement_blocked"
    assert row["per_logical_per_round"] is None
    assert slow_queue_status(
        record_count=4,
        timeout_candidate_count=0,
        measurement_blocked_candidate_count=1,
        decoder_attempt_candidate_count=4,
        ok_result_count=3,
    ) == "measurement_blocked"


@pytest.mark.parametrize("candidate_status", ["build_error", "input_error", "subprocess_error"])
def test_any_nonterminal_non_ok_candidate_blocks_queue(candidate_status: str) -> None:
    # Exercise the exact production predicate used to build the queue
    # summary, then protect the verdict from treating a surviving cheap
    # candidate as sufficient evidence for the whole queue.
    payloads = [
        {"program_id": "cheap-survivor", "status": "ok"},
        {"program_id": "censored-hard-code", "status": candidate_status},
        {"program_id": "terminal", "status": "schedule_catastrophic"},
    ]
    incomplete = pool_mod.count_incomplete_candidates(payloads)
    assert incomplete == 1
    assert slow_queue_status(
        record_count=len(payloads),
        timeout_candidate_count=0,
        measurement_blocked_candidate_count=0,
        incomplete_candidate_count=incomplete,
        decoder_attempt_candidate_count=1,
        ok_result_count=1,
    ) == "measurement_blocked"


def _seed_bb_spec() -> dict:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    return run_construct_source(
        SEED_BB, program_id="measurement_completeness_bb", seed=0
    ).constructed.spec


def _successful_shard(request: dict) -> tuple[dict, dict]:
    unit = request["units"][0]
    shots = int(request["shots"])
    cell = {
        "status": "ok",
        "decoder": unit["decoder"],
        "shots": shots,
        "failures": 1,
        "logical_error_rate": 1.0 / shots,
        "runtime_s": 0.01,
        "metrics": {
            "requested_shots": shots,
            "actual_shots": shots,
            "sampling_complete": True,
            "stopped_by_max_errors": False,
        },
    }
    return (
        {
            "status": "ok",
            "results": {unit["decoder"]: {unit["basis"]: cell}},
        },
        {"timeout": False},
    )


def test_pool_marks_a_candidate_decode_infeasible_when_one_required_basis_times_out(
    monkeypatch,
) -> None:
    """v3.4: watchdog timeouts alone are a terminal decode_infeasible verdict
    (the code exceeded the frozen per-shot budget), not measurement_blocked.
    Explicit timeout_retries=2 still replays the identical request."""

    z_attempts: dict[int, int] = {}

    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        unit = request["units"][0]
        if unit["basis"] == "Z":
            shard = int(request["shard"])
            z_attempts[shard] = z_attempts.get(shard, 0) + 1
            return None, {"timeout": True, "timeout_s": timeout_s}
        return _successful_shard(request)

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    payload = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_partial", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(
            shots=2_000,
            max_errors=None,
            rounds=12,
            timeout_retries=2,
        ),
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=4,
        shot_shards=2,
    )[0]
    assert z_attempts == {0: 3, 1: 3}
    assert payload["status"] == "decode_infeasible"
    assert payload["decode_infeasible"] is True
    assert payload["results"][0]["status"] == "decode_infeasible"
    assert payload["results"][0]["measurement_complete"] is False
    assert payload["results"][0]["combined"]["per_logical_per_round"] is None
    assert payload["results"][0]["basis"]["Z"]["shard_statuses"] == [
        "timeout",
        "timeout",
    ]


def test_pool_keeps_complete_plpr_when_only_distance_times_out(monkeypatch) -> None:
    distance_attempts = 0

    def fake_worker(request, *, timeout_s):
        nonlocal distance_attempts
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            distance_attempts += 1
            return None, {"timeout": True, "timeout_s": timeout_s}
        return _successful_shard(request)

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    payload = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_distance_timeout", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(
            shots=2_000,
            max_errors=None,
            rounds=12,
            timeout_retries=2,
        ),
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=4,
        shot_shards=2,
    )[0]
    assert distance_attempts == 3
    assert payload["status"] == "ok"
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["combined"]["per_logical_per_round"] is not None
    assert payload["exact_distance"]["status"] == "uncertified_timeout"
