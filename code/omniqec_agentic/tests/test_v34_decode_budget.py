"""bposd_structural_v3_4 (2026-09-02): per-shot decode budget, 128 final
shards, decode_infeasible as a candidate-level terminal state.

Measured on NSCC (b360 canary): BPOSD product_sum/50/osd_cs/10 costs
0.65-0.94 s/shot on typical bb codes at n_data 144-180 and 5.2-5.5 s/shot on
[[180,2,20]] lp_abelian (BP non-convergent, OSD on nearly every shot). v3.3's
flat 1800 s shard watchdog with 16 shards could therefore never finish the
200k-shot final stage at b220+, one slow code halted a whole point
(FATAL_STOP), and a deterministic timeout was replayed three times for
nothing. Two plumbing bugs rode along: a worker that lost its DEM was
persisted as a shard "timeout" with no job_failures entry and no retry
(BUG-1); a killed worker (OOM SIGKILL) became subprocess_error with its
returncode dropped and was never retried (BUG-2).

Every name used here is the v3.4 contract name:
  pool.effective_decode_shard_timeout_s / pool.worker_timeout_for,
  state.FORMAL_FINAL_SHOT_SHARDS, snapshot key "final_shot_shards",
  slow_queue_status(decode_infeasible_candidate_count=...),
  VerificationConfig.decode_shot_budget_s / transient_retries,
  statuses "decode_infeasible" / "dem_load_error", payload flag
  "decode_infeasible", job_execution keys returncode / transient_retries_used.

The v3.4 names are imported inside the tests (not at module level) so the
module collects while the core halves land; those tests fail, not error,
until they do. Validator fixtures extend test_campaign_strict_turns.py.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
from pathlib import Path

import pytest

from omniqec_agentic.config import RacingConfig, StructuralSearchConfig, VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.common import verification_from_dict
from omniqec_agentic.evaluation.sampling import merge_shard_results
from omniqec_agentic.evaluation.writeback import ranking_rows, slow_queue_status
from omniqec_agentic.hpc import validate_fig2_campaign as validator
from omniqec_agentic.hpc.validate_fig2_campaign import FORMAL_JOB_POLICY, validate
from omniqec_agentic.workflow.state import build_protocol_snapshot

from test_campaign_strict_turns import (
    FINAL_BASE_SEED,
    FINAL_SHOTS,
    TELESCOPING,
    TELESCOPING_BUILD,
    _rankings,
    _slow_run_config,
    _strict_report,
    _write_json,
    _write_validated_run,
)

BUDGET_REASON = "decode exceeded the frozen per-shot budget: Z:2/2 shards (timeout_s=8000.0)"
DEM_LOST_REASON = "No such file: matrices.pkl"


# ---------------------------------------------------------------------------
# Frozen defaults and serialization
# ---------------------------------------------------------------------------


def test_v34_verification_defaults_serialize_and_v33_artifacts_load() -> None:
    assert StructuralSearchConfig().profile_id == "telescoping_structural_v3_7"
    current = VerificationConfig()
    assert current.decode_shot_budget_s == 8.0
    assert current.decode_shard_timeout_s == 1800.0  # now the FLOOR of the watchdog
    assert current.timeout_retries == 0
    assert current.transient_retries == 1

    encoded = current.to_dict()
    assert encoded["decode_shot_budget_s"] == 8.0
    assert encoded["timeout_retries"] == 0
    assert encoded["transient_retries"] == 1
    restored = verification_from_dict(encoded)
    assert restored.decode_shot_budget_s == 8.0
    assert restored.timeout_retries == 0
    assert restored.transient_retries == 1

    # A v3.3 artifact carries neither field: dataclass defaults apply, and its
    # own recorded values are kept where present.
    legacy = verification_from_dict({"decode_shard_timeout_s": 1800.0, "timeout_retries": 2})
    assert legacy.timeout_retries == 2
    assert legacy.decode_shot_budget_s == 8.0
    assert legacy.transient_retries == 1
    custom = verification_from_dict({"decode_shot_budget_s": 2.5, "transient_retries": 3})
    assert custom.decode_shot_budget_s == 2.5
    assert custom.transient_retries == 3


# ---------------------------------------------------------------------------
# Effective shard watchdog: max(floor, budget * shots)
# ---------------------------------------------------------------------------


def test_effective_shard_watchdog_scales_with_the_shot_budget_above_the_floor() -> None:
    from omniqec_agentic.evaluation.pool import effective_decode_shard_timeout_s

    formal = VerificationConfig()
    # search: 20000 shots / 16 shards = 1250 shots -> 10,000 s
    assert effective_decode_shard_timeout_s(formal, 1250) == pytest.approx(10_000.0)
    # final: 200000 shots / 128 shards = 1562 shots -> ~12,500 s
    assert effective_decode_shard_timeout_s(formal, 1562) == pytest.approx(8.0 * 1562)
    # small shards keep the 1800 s floor
    assert effective_decode_shard_timeout_s(formal, 100) == pytest.approx(1800.0)
    assert effective_decode_shard_timeout_s(formal, 225) == pytest.approx(1800.0)
    # a zero floor does not disable the watchdog while the budget is positive
    assert effective_decode_shard_timeout_s(
        VerificationConfig(decode_shard_timeout_s=0.0), 1250
    ) == pytest.approx(10_000.0)
    assert effective_decode_shard_timeout_s(
        VerificationConfig(decode_shot_budget_s=2.0), 1250
    ) == pytest.approx(2500.0)


def test_effective_shard_watchdog_without_a_budget_is_the_floor_or_disabled() -> None:
    from omniqec_agentic.evaluation.pool import effective_decode_shard_timeout_s

    floor_only = VerificationConfig(decode_shot_budget_s=0.0)
    assert effective_decode_shard_timeout_s(floor_only, 1250) == pytest.approx(1800.0)
    assert effective_decode_shard_timeout_s(
        VerificationConfig(decode_shot_budget_s=0.0, decode_shard_timeout_s=600.0),
        1_000_000,
    ) == pytest.approx(600.0)
    disabled = VerificationConfig(decode_shot_budget_s=0.0, decode_shard_timeout_s=0.0)
    assert effective_decode_shard_timeout_s(disabled, 1250) is None
    assert effective_decode_shard_timeout_s(
        VerificationConfig(decode_shot_budget_s=-1.0, decode_shard_timeout_s=0.0), 10
    ) is None


def test_worker_timeout_applies_the_budget_to_shard_jobs_only() -> None:
    from omniqec_agentic.evaluation.pool import worker_timeout_for

    formal = VerificationConfig()
    assert worker_timeout_for({"mode": "shard_dem", "shots": 1250}, formal) == pytest.approx(10_000.0)
    assert worker_timeout_for({"mode": "shard_dem", "shots": 10}, formal) == pytest.approx(1800.0)
    assert worker_timeout_for({"mode": "build_dem"}, formal) == pytest.approx(3600.0)
    assert worker_timeout_for({"mode": "distance_job"}, formal) == pytest.approx(1800.0)
    assert (
        worker_timeout_for(
            {"mode": "shard_dem", "shots": 1250},
            VerificationConfig(decode_shot_budget_s=0.0, decode_shard_timeout_s=0.0),
        )
        is None
    )


# ---------------------------------------------------------------------------
# Retry policy: no timeout replay by default; one transient replay
# ---------------------------------------------------------------------------

SHARD_REQUEST = {
    "mode": "shard_dem",
    "base_seed": 12345,
    "shard": 3,
    "shot_shards": 16,
    "shots": 1250,
    "units": [{"decoder": "bposd", "basis": "Z"}],
}
OK = ({"status": "ok", "results": {}}, {"timeout": False, "returncode": 0, "stderr_tail": ""})
TIMEOUT = (None, {"timeout": True})
KILLED = (None, {"timeout": False, "returncode": -9, "stderr_tail": ""})
DEM_LOST = (
    {"status": "dem_load_error", "failure_reason": DEM_LOST_REASON, "results": {}},
    {"timeout": False, "returncode": 0, "stderr_tail": ""},
)


def _script(monkeypatch, responses: list[tuple]) -> list[tuple[dict, float | None]]:
    """Feed run_worker_request one scripted (parsed, meta) pair per attempt,
    recording the request each attempt received and the watchdog it got."""

    calls: list[tuple[dict, float | None]] = []
    queue = [copy.deepcopy(item) for item in responses]

    def fake_run_worker_request(payload, *, timeout_s):
        calls.append((copy.deepcopy(payload), timeout_s))
        assert queue, "worker called more times than the scenario allows"
        parsed, meta = queue.pop(0)
        if meta.get("timeout"):
            meta = {**meta, "timeout_s": timeout_s}
        return parsed, meta

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_run_worker_request)
    return calls


def test_a_watchdog_timeout_is_not_replayed_by_default(monkeypatch) -> None:
    calls = _script(monkeypatch, [TIMEOUT, OK])  # the second attempt must never run
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 1
    assert calls[0][0] == SHARD_REQUEST
    assert calls[0][1] == pytest.approx(10_000.0)
    assert outcome["status"] == "timeout"
    assert re.fullmatch(
        r"shard_dem worker exceeded timeout_s=10000(\.0)? on all 1 attempts",
        outcome["failure_reason"],
    )
    execution = outcome["job_execution"]
    assert execution["attempts"] == 1
    assert execution["retries_used"] == 0
    assert execution["retry_limit"] == 0
    assert execution["transient_retry_limit"] == 1
    assert execution["transient_retries_used"] == 0
    assert execution["timeout_s"] == pytest.approx(10_000.0)
    assert execution["timed_out"] is True
    assert "returncode" in execution
    assert execution["returncode"] is None

    # Transient replays never apply to a watchdog timeout.
    calls = _script(monkeypatch, [TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig(transient_retries=3)
    )
    assert len(calls) == 1
    assert outcome["status"] == "timeout"
    assert outcome["job_execution"]["transient_retries_used"] == 0

    calls = _script(monkeypatch, [TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        {"mode": "distance_job", "seed": 9}, verification=VerificationConfig()
    )
    assert len(calls) == 1
    assert outcome["status"] == "uncertified_timeout"


def test_explicit_timeout_retries_still_replay_the_identical_request(monkeypatch) -> None:
    calls = _script(monkeypatch, [TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig(timeout_retries=1)
    )
    assert [call[0] for call in calls] == [SHARD_REQUEST, SHARD_REQUEST]
    assert outcome["status"] == "ok"
    execution = outcome["job_execution"]
    assert execution["attempts"] == 2
    assert execution["retries_used"] == 1
    assert execution["retry_limit"] == 1
    assert execution["transient_retries_used"] == 0
    assert not execution.get("timed_out")


def test_a_killed_worker_is_replayed_once_with_its_returncode_recorded(monkeypatch) -> None:
    """BUG-2: an OOM-killed worker used to become subprocess_error with no
    returncode and no retry."""

    calls = _script(monkeypatch, [KILLED, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert [call[0] for call in calls] == [SHARD_REQUEST, SHARD_REQUEST]
    assert outcome["status"] == "ok"
    execution = outcome["job_execution"]
    assert execution["attempts"] == 2
    assert execution["retries_used"] == 0
    assert execution["transient_retry_limit"] == 1
    assert execution["transient_retries_used"] == 1
    assert execution["returncode"] == 0  # from the LAST attempt
    assert not execution.get("timed_out")


def test_a_persistently_killed_worker_is_subprocess_error_with_returncode(monkeypatch) -> None:
    calls = _script(monkeypatch, [KILLED, KILLED, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 2
    assert outcome["status"] == "subprocess_error"
    assert outcome["returncode"] == -9
    assert outcome["failure_reason"] == "worker produced no parseable JSON (returncode=-9)"
    execution = outcome["job_execution"]
    assert execution["attempts"] == 2
    assert execution["transient_retries_used"] == 1
    assert execution["transient_retry_limit"] == 1
    assert execution["returncode"] == -9
    assert not execution.get("timed_out")


def test_a_lost_dem_is_replayed_once_then_surfaces_as_dem_load_error(monkeypatch) -> None:
    """BUG-1: a worker that could not load its DEM was not retried at all."""

    calls = _script(monkeypatch, [DEM_LOST, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 2
    assert outcome["status"] == "ok"
    assert outcome["job_execution"]["transient_retries_used"] == 1

    calls = _script(monkeypatch, [DEM_LOST, DEM_LOST, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 2
    assert outcome["status"] == "dem_load_error"
    assert outcome["failure_reason"] == DEM_LOST_REASON
    execution = outcome["job_execution"]
    assert execution["attempts"] == 2
    assert execution["transient_retries_used"] == 1
    assert execution["transient_retry_limit"] == 1
    assert execution["returncode"] == 0


def test_transient_retries_zero_returns_the_first_process_failure(monkeypatch) -> None:
    calls = _script(monkeypatch, [KILLED, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig(transient_retries=0)
    )
    assert len(calls) == 1
    assert outcome["status"] == "subprocess_error"
    assert outcome["returncode"] == -9
    assert outcome["job_execution"]["transient_retry_limit"] == 0
    assert outcome["job_execution"]["transient_retries_used"] == 0


def test_a_timeout_after_a_transient_replay_gets_a_distinct_status(monkeypatch) -> None:
    """v3.4.1: a watchdog expiry that follows a killed worker / lost DEM on an
    earlier attempt of the same request is not evidence of a deterministic
    budget overrun — it is labelled timeout_after_transient, never timeout."""

    calls = _script(monkeypatch, [KILLED, TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 2  # one transient replay, then the timeout is final
    assert [call[0] for call in calls] == [SHARD_REQUEST, SHARD_REQUEST]
    assert outcome["status"] == "timeout_after_transient"
    assert re.fullmatch(
        r"shard_dem worker exceeded timeout_s=10000(\.0)? on all 2 attempts "
        r"after 1 transient replay\(s\)",
        outcome["failure_reason"],
    )
    execution = outcome["job_execution"]
    assert execution["attempts"] == 2
    assert execution["retries_used"] == 0
    assert execution["transient_retries_used"] == 1
    assert execution["timed_out"] is True
    assert execution["returncode"] is None  # the LAST attempt timed out

    # A lost DEM followed by a timeout: same verdict.
    calls = _script(monkeypatch, [DEM_LOST, TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert len(calls) == 2
    assert outcome["status"] == "timeout_after_transient"
    assert outcome["job_execution"]["transient_retries_used"] == 1

    # Explicit timeout replays after a transient replay still end distinct.
    calls = _script(monkeypatch, [KILLED, TIMEOUT, TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig(timeout_retries=1)
    )
    assert len(calls) == 3
    assert outcome["status"] == "timeout_after_transient"
    assert outcome["failure_reason"].endswith("on all 3 attempts after 1 transient replay(s)")
    assert outcome["job_execution"]["retries_used"] == 1

    # A pure first-attempt expiry is unchanged: status timeout, no suffix.
    calls = _script(monkeypatch, [TIMEOUT, OK])
    outcome = pool_mod.run_worker_request_with_retries(
        dict(SHARD_REQUEST), verification=VerificationConfig()
    )
    assert outcome["status"] == "timeout"
    assert "transient replay" not in outcome["failure_reason"]
    assert outcome["job_execution"]["attempts"] == 1

    # Distance jobs keep uncertified_timeout (auxiliary evidence, never a
    # shard cell) but the reason still records the replay.
    calls = _script(monkeypatch, [KILLED, TIMEOUT])
    outcome = pool_mod.run_worker_request_with_retries(
        {"mode": "distance_job", "seed": 9}, verification=VerificationConfig()
    )
    assert outcome["status"] == "uncertified_timeout"
    assert outcome["failure_reason"].endswith("after 1 transient replay(s)")


# ---------------------------------------------------------------------------
# Shard merge, incomplete count, queue verdict, ranking rows (pure)
# ---------------------------------------------------------------------------


def _basis_cell(statuses: list[str], attempts: list | None = None) -> dict:
    return {
        "status": "ok" if all(s == "ok" for s in statuses) else "partial_shards",
        "shard_statuses": list(statuses),
        "shard_attempts": list(attempts) if attempts is not None else [1] * len(statuses),
    }


def test_timeout_only_gate_accepts_only_returned_pure_timeouts() -> None:
    """v3.4.1: decode_infeasible needs evidence of a PURE first-attempt
    watchdog expiry of a job that actually returned."""

    ok = _basis_cell(["ok", "ok"])

    def gate(z: dict) -> bool:
        return pool_mod.fixed_shot_timeout_only(
            {"basis": {"X": ok, "Z": z}}, required_bases=("X", "Z")
        )

    assert gate(_basis_cell(["ok", "timeout"], [1, 1])) is True
    assert gate(_basis_cell(["timeout", "timeout"], [1, 1])) is True
    # explicit timeout_retries: still a pure (replayed-identically) timeout
    assert gate(_basis_cell(["timeout", "timeout"], [3, 3])) is True
    # a transient replay preceded the expiry -> distinct status -> blocked
    assert gate(_basis_cell(["ok", "timeout_after_transient"], [1, 2])) is False
    assert gate(_basis_cell(["timeout", "timeout_after_transient"], [1, 2])) is False
    # a "timeout" placeholder for a job that never returned is not evidence
    assert gate(_basis_cell(["ok", "timeout"], [1, None])) is False
    # nothing of this basis ever returned (e.g. the build failed)
    assert gate(_basis_cell(["timeout", "timeout"], [None, None])) is False
    assert gate(_basis_cell(["ok", "dem_load_error"], [1, 2])) is False
    assert gate(_basis_cell(["ok", "subprocess_error"], [1, 2])) is False
    assert gate(_basis_cell(["ok", "ok"])) is False  # no defect at all
    # no attempts record -> unknown provenance -> never infeasible
    without_attempts = _basis_cell(["ok", "timeout"])
    del without_attempts["shard_attempts"]
    assert gate(without_attempts) is False
    # mismatched records are not trusted either
    assert gate(_basis_cell(["ok", "timeout"], [1])) is False
    # a missing required basis
    assert (
        pool_mod.fixed_shot_timeout_only({"basis": {"X": ok}}, required_bases=("X", "Z"))
        is False
    )
    # both bases infeasible is still infeasible
    assert pool_mod.fixed_shot_timeout_only(
        {"basis": {"X": _basis_cell(["timeout", "ok"]), "Z": _basis_cell(["ok", "timeout"])}},
        required_bases=("X", "Z"),
    ) is True


def test_merge_shard_results_reports_runtime_and_attempts_per_shard() -> None:
    cells = [
        {
            "status": "ok",
            "shots": 10,
            "failures": 1,
            "runtime_s": 1.5,
            "job_execution": {"attempts": 1},
        },
        {
            "status": "dem_load_error",
            "shots": 0,
            "failures": 0,
            "runtime_s": 0.0,
            "failure_reason": DEM_LOST_REASON,
            "job_execution": {"attempts": 2},
        },
        None,  # the job never returned at all
    ]
    merged = merge_shard_results(cells, decoder="bposd")
    assert merged["shard_statuses"] == ["ok", "dem_load_error", "timeout"]
    assert merged["shard_count"] == 3
    assert merged["shots"] == 10
    assert merged["status"] == "partial_shards"
    runtime = merged["shard_runtime_s"]
    assert len(runtime) == 3
    assert runtime[0] == pytest.approx(1.5)
    assert runtime[1] == 0.0  # a synthesized failure cell carries runtime_s 0.0
    assert runtime[2] is None
    assert merged["shard_attempts"] == [1, 2, None]


def test_decode_infeasible_candidates_are_terminal_for_the_incomplete_count() -> None:
    payloads = [
        {"program_id": "a", "status": "ok"},
        {"program_id": "b", "status": "decode_infeasible", "decode_infeasible": True},
        # the flag wins over a stale status
        {"program_id": "c", "status": "measurement_blocked", "decode_infeasible": True},
        {"program_id": "d", "status": "schedule_catastrophic"},
        {"program_id": "e", "status": "measurement_blocked"},
        {"program_id": "f", "status": "subprocess_error"},
    ]
    assert pool_mod.count_incomplete_candidates(payloads) == 2


def test_queue_verdict_subtracts_decode_infeasible_like_catastrophic() -> None:
    common = dict(record_count=4, timeout_candidate_count=0)
    assert slow_queue_status(
        **common, decoder_attempt_candidate_count=4, ok_result_count=3,
        decode_infeasible_candidate_count=1,
    ) == "ok"
    # an all-infeasible queue measured what there was to measure
    assert slow_queue_status(
        **common, decoder_attempt_candidate_count=4, ok_result_count=0,
        decode_infeasible_candidate_count=4,
    ) == "ok"
    assert slow_queue_status(
        **common, decoder_attempt_candidate_count=2, ok_result_count=0,
        catastrophic_candidate_count=2, decode_infeasible_candidate_count=2,
    ) == "ok"
    # real infrastructure censoring still blocks the queue
    assert slow_queue_status(
        record_count=4, timeout_candidate_count=1, decoder_attempt_candidate_count=4,
        ok_result_count=2, decode_infeasible_candidate_count=1,
    ) == "measurement_blocked"
    assert slow_queue_status(
        **common, measurement_blocked_candidate_count=1, decoder_attempt_candidate_count=4,
        ok_result_count=2, decode_infeasible_candidate_count=1,
    ) == "measurement_blocked"
    assert slow_queue_status(
        **common, incomplete_candidate_count=1, decoder_attempt_candidate_count=4,
        ok_result_count=2, decode_infeasible_candidate_count=1,
    ) == "measurement_blocked"
    # the remaining effective queue still needs attempts and results
    assert slow_queue_status(
        record_count=3, timeout_candidate_count=0, decoder_attempt_candidate_count=0,
        ok_result_count=0, decode_infeasible_candidate_count=1,
    ) == "no_valid_candidates"
    assert slow_queue_status(
        record_count=3, timeout_candidate_count=0, decoder_attempt_candidate_count=3,
        ok_result_count=0, decode_infeasible_candidate_count=1,
    ) == "no_valid_results"
    # the kwarg defaults to 0, so every existing caller keeps its verdict
    assert slow_queue_status(
        record_count=2, timeout_candidate_count=0, decoder_attempt_candidate_count=2,
        ok_result_count=2,
    ) == "ok"


def _decode_infeasible_payload() -> dict:
    return {
        "program_id": "slow_code",
        "status": "decode_infeasible",
        "decode_infeasible": True,
        "failure_reason": BUDGET_REASON,
        "structural": {"code_hash": "cd" * 16, "n_data": 180, "k": 2},
        "fast": {"family": "lp_abelian", "n": 180, "k": 2, "d": 20},
        "results": [
            {
                "status": "decode_infeasible",
                "decoder": "bposd",
                "measurement_complete": False,
                "failure_reason": "Z_shard_incomplete; missing_combined_plpr",
                "combined": {
                    "block_ler": None,
                    "per_logical_per_round": None,
                    "failure_reason": "fixed_shot_measurement_incomplete",
                },
                "partial_combined": {"block_ler": 0.01, "per_logical_per_round": 1e-4},
                "basis": {
                    "X": {"status": "ok", "shots": 2000, "failures": 3, "shard_statuses": ["ok", "ok"]},
                    "Z": {
                        "status": "decoder_error",
                        "shots": 0,
                        "failures": 0,
                        "shard_statuses": ["timeout", "timeout"],
                    },
                },
            }
        ],
    }


def test_ranking_rows_publish_no_plpr_for_a_decode_infeasible_candidate(tmp_path) -> None:
    rows = ranking_rows(_decode_infeasible_payload(), tmp_path / "slow_code.json")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "decode_infeasible"
    assert row["per_logical_per_round"] is None
    assert row["block_ler"] is None
    assert "decode exceeded the frozen per-shot budget" in row["failure_reason"]
    assert row["code_hash"] == "cd" * 16


# ---------------------------------------------------------------------------
# Job pool: shard cells, job_failures, candidate status
# ---------------------------------------------------------------------------


def _seed_bb_spec() -> dict:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    return run_construct_source(
        SEED_BB, program_id="v34_decode_budget_bb", seed=0
    ).constructed.spec


def _ok_shard(request: dict) -> tuple[dict, dict]:
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
        {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}},
        {"timeout": False, "returncode": 0, "stderr_tail": ""},
    )


def _pool_worker(z_behaviour):
    """Fake worker: build/distance succeed, X shards succeed, Z shard outcomes
    follow ``z_behaviour(shard, attempt) -> 'ok'|'timeout'|'dem_load_error'|'killed'``.
    Returns the fake and the per-shard Z attempt counter."""

    attempts: dict[int, int] = {}
    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}

    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, dict(fine)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        unit = request["units"][0]
        if unit["basis"] == "Z":
            shard = int(request["shard"])
            attempts[shard] = attempts.get(shard, 0) + 1
            kind = z_behaviour(shard, attempts[shard])
            if kind == "timeout":
                return None, {"timeout": True, "timeout_s": timeout_s}
            if kind == "dem_load_error":
                return (
                    {"status": "dem_load_error", "failure_reason": DEM_LOST_REASON, "results": {}},
                    dict(fine),
                )
            if kind == "killed":
                return None, {"timeout": False, "returncode": -9, "stderr_tail": ""}
        return _ok_shard(request)

    return fake_worker, attempts


def _search_verification() -> VerificationConfig:
    return VerificationConfig(shots=2_000, max_errors=None, rounds=12)


def _evaluate(monkeypatch, fake_worker) -> dict:
    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_v34", "spec": _seed_bb_spec()}],
        verification=_search_verification(),
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=4,
        shot_shards=2,
    )[0]


def _shard_failures(payload: dict) -> list[dict]:
    return [item for item in payload.get("job_failures") or [] if item.get("kind") == "shard"]


def test_timeout_only_incompleteness_is_a_decode_infeasible_terminal_verdict(monkeypatch) -> None:
    fake_worker, attempts = _pool_worker(lambda shard, attempt: "timeout")
    payload = _evaluate(monkeypatch, fake_worker)

    assert attempts == {0: 1, 1: 1}  # v3.4 default: no identical replay of a timeout
    assert payload["status"] == "decode_infeasible"
    assert payload["decode_infeasible"] is True
    assert payload["failure_reason"].startswith("decode exceeded the frozen per-shot budget:")
    assert "Z:2/2 shards" in payload["failure_reason"]
    # 2000 shots / 2 shards = 1000 shots x 8 s > the 1800 s floor
    assert re.search(r"timeout_s=8000(\.0)?", payload["failure_reason"])

    result = payload["results"][0]
    assert result["status"] == "decode_infeasible"
    assert result["measurement_complete"] is False
    assert result["combined"]["per_logical_per_round"] is None
    assert result["combined"]["failure_reason"] == "fixed_shot_measurement_incomplete"
    assert "partial_combined" in result
    assert result["basis"]["X"]["status"] == "ok"
    assert result["basis"]["Z"]["shard_statuses"] == ["timeout", "timeout"]
    assert result["basis"]["Z"]["shard_attempts"] == [1, 1]

    failures = _shard_failures(payload)
    assert sorted(item["shard"] for item in failures) == [0, 1]
    assert {item["status"] for item in failures} == {"timeout"}
    assert all("returncode" in item["job_execution"] for item in failures)
    assert all(item["job_execution"]["timed_out"] is True for item in failures)

    # Terminal: not incomplete; job_config records the v3.4 policy.
    assert pool_mod.count_incomplete_candidates([payload]) == 0
    job_config = payload["job_config"]
    assert job_config["shot_shards"] == 2
    assert job_config["decode_shard_timeout_s"] == 1800.0
    assert job_config["decode_shot_budget_s"] == 8.0
    assert job_config["timeout_retries"] == 0
    assert job_config["transient_retries"] == 1
    assert job_config["effective_decode_shard_timeout_s"] == pytest.approx(8000.0)


def test_a_dem_load_error_shard_is_labelled_and_recorded_not_a_timeout(monkeypatch) -> None:
    """BUG-1 at the pool level: the cell carries the worker's own status and
    the failure lands in job_failures with its execution record."""

    fake_worker, attempts = _pool_worker(
        lambda shard, attempt: "dem_load_error" if shard == 1 else "ok"
    )
    payload = _evaluate(monkeypatch, fake_worker)

    assert attempts == {0: 1, 1: 2}  # one transient replay of the lost DEM
    result = payload["results"][0]
    basis_z = result["basis"]["Z"]
    assert basis_z["shard_statuses"] == ["ok", "dem_load_error"]
    assert basis_z["shard_attempts"] == [1, 2]
    assert basis_z["shard_runtime_s"][0] == pytest.approx(0.01)
    assert basis_z["shots"] == 1000

    failures = _shard_failures(payload)
    assert len(failures) == 1
    assert failures[0]["shard"] == 1
    assert failures[0]["status"] == "dem_load_error"
    assert failures[0]["failure_reason"] == DEM_LOST_REASON
    assert failures[0]["job_execution"]["transient_retries_used"] == 1
    assert failures[0]["job_execution"]["returncode"] == 0

    # Not a budget verdict: the candidate is blocked, not infeasible.
    assert payload["status"] == "measurement_blocked"
    assert not payload.get("decode_infeasible")
    assert result["status"] == "measurement_blocked"
    assert pool_mod.count_incomplete_candidates([payload]) == 1


def test_a_timeout_mixed_with_a_lost_dem_is_measurement_blocked(monkeypatch) -> None:
    fake_worker, _ = _pool_worker(
        lambda shard, attempt: "timeout" if shard == 0 else "dem_load_error"
    )
    payload = _evaluate(monkeypatch, fake_worker)

    assert payload["results"][0]["basis"]["Z"]["shard_statuses"] == ["timeout", "dem_load_error"]
    assert payload["status"] == "measurement_blocked"
    assert not payload.get("decode_infeasible")
    assert payload["results"][0]["status"] == "measurement_blocked"
    assert sorted((item["shard"], item["status"]) for item in _shard_failures(payload)) == [
        (0, "timeout"),
        (1, "dem_load_error"),
    ]
    assert pool_mod.count_incomplete_candidates([payload]) == 1


def test_a_killed_shard_worker_is_replayed_then_recorded_with_its_returncode(monkeypatch) -> None:
    fake_worker, attempts = _pool_worker(
        lambda shard, attempt: "killed" if (shard == 1 and attempt == 1) else "ok"
    )
    payload = _evaluate(monkeypatch, fake_worker)
    assert attempts == {0: 1, 1: 2}
    assert payload["status"] == "ok"
    assert payload["results"][0]["basis"]["Z"]["shard_statuses"] == ["ok", "ok"]
    assert payload["results"][0]["basis"]["Z"]["shard_attempts"] == [1, 2]
    assert not _shard_failures(payload)

    fake_worker, attempts = _pool_worker(lambda shard, attempt: "killed" if shard == 1 else "ok")
    payload = _evaluate(monkeypatch, fake_worker)
    assert attempts == {0: 1, 1: 2}
    assert payload["status"] == "measurement_blocked"
    assert payload["results"][0]["basis"]["Z"]["shard_statuses"] == ["ok", "subprocess_error"]
    failures = _shard_failures(payload)
    assert len(failures) == 1
    assert failures[0]["status"] == "subprocess_error"
    assert failures[0]["job_execution"]["returncode"] == -9
    assert failures[0]["job_execution"]["transient_retries_used"] == 1
    assert failures[0]["returncode"] == -9  # v3.4.1: mirrored at the top level


def test_a_timeout_after_a_killed_worker_is_measurement_blocked_not_infeasible(
    monkeypatch,
) -> None:
    """v3.4.1: killed-then-timeout on the same shard request is an
    infrastructure suspicion (timeout_after_transient), not a budget verdict."""

    fake_worker, attempts = _pool_worker(
        lambda shard, attempt: "killed" if attempt == 1 else "timeout"
    )
    payload = _evaluate(monkeypatch, fake_worker)

    assert attempts == {0: 2, 1: 2}  # one transient replay each, then the watchdog
    result = payload["results"][0]
    basis_z = result["basis"]["Z"]
    assert basis_z["shard_statuses"] == ["timeout_after_transient", "timeout_after_transient"]
    assert basis_z["shard_attempts"] == [2, 2]
    assert basis_z["shots"] == 0
    assert result["basis"]["X"]["status"] == "ok"

    assert payload["status"] == "measurement_blocked"
    assert not payload.get("decode_infeasible")
    assert payload["failure_reason"] == "one or more required fixed-shot cells are incomplete"
    assert result["status"] == "measurement_blocked"
    assert result["measurement_complete"] is False
    assert result["combined"]["per_logical_per_round"] is None

    failures = _shard_failures(payload)
    assert sorted(item["shard"] for item in failures) == [0, 1]
    assert {item["status"] for item in failures} == {"timeout_after_transient"}
    for item in failures:
        assert item["job_execution"]["attempts"] == 2
        assert item["job_execution"]["transient_retries_used"] == 1
        assert item["job_execution"]["timed_out"] is True
        assert item["returncode"] is None
        assert item["failure_reason"].endswith("on all 2 attempts after 1 transient replay(s)")
    assert pool_mod.count_incomplete_candidates([payload]) == 1

    # Mixed: one pure timeout shard + one killed-then-timeout shard -> still
    # blocked (the pure timeout alone is not the ONLY defect).
    fake_worker, _ = _pool_worker(
        lambda shard, attempt: "timeout"
        if shard == 0
        else ("killed" if attempt == 1 else "timeout")
    )
    payload = _evaluate(monkeypatch, fake_worker)
    assert payload["results"][0]["basis"]["Z"]["shard_statuses"] == [
        "timeout",
        "timeout_after_transient",
    ]
    assert payload["status"] == "measurement_blocked"
    assert not payload.get("decode_infeasible")
    assert payload["results"][0]["status"] == "measurement_blocked"


def test_a_pure_first_attempt_timeout_is_still_decode_infeasible(monkeypatch) -> None:
    """v3.4.1 regression guard: the tightened gate must not change the pure
    watchdog verdict (attempts 1, no transient replay)."""

    fake_worker, attempts = _pool_worker(
        lambda shard, attempt: "timeout" if shard == 1 else "ok"
    )
    payload = _evaluate(monkeypatch, fake_worker)
    assert attempts == {0: 1, 1: 1}
    basis_z = payload["results"][0]["basis"]["Z"]
    assert basis_z["shard_statuses"] == ["ok", "timeout"]
    assert basis_z["shard_attempts"] == [1, 1]
    assert payload["status"] == "decode_infeasible"
    assert payload["decode_infeasible"] is True
    assert "Z:1/2 shards" in payload["failure_reason"]
    assert payload["results"][0]["status"] == "decode_infeasible"
    failures = _shard_failures(payload)
    assert [item["status"] for item in failures] == ["timeout"]
    assert failures[0]["job_execution"]["transient_retries_used"] == 0
    assert failures[0]["returncode"] is None
    assert pool_mod.count_incomplete_candidates([payload]) == 0


def _build_failing_worker(build_outcome):
    """Fake worker whose build_dem stage returns ``build_outcome(timeout_s)``;
    distance succeeds; no decode job can ever be submitted."""

    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}
    decode_calls = 0

    def fake_worker(request, *, timeout_s):
        nonlocal decode_calls
        mode = request.get("mode")
        if mode == "build_dem":
            return build_outcome(timeout_s)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        decode_calls += 1
        return _ok_shard(request)

    return fake_worker, lambda: decode_calls


@pytest.mark.parametrize(
    ("build_outcome", "expected_status", "expected_returncode"),
    [
        (
            lambda timeout_s: (
                {"status": "build_error", "failure_reason": "matrix rank check failed"},
                {"timeout": False, "returncode": 0, "stderr_tail": ""},
            ),
            "build_error",
            0,
        ),
        (lambda timeout_s: (None, {"timeout": True, "timeout_s": timeout_s}), "timeout", None),
        (
            lambda timeout_s: (None, {"timeout": False, "returncode": -9, "stderr_tail": ""}),
            "subprocess_error",
            -9,
        ),
    ],
    ids=["build_error", "build_timeout", "build_killed"],
)
def test_a_failed_build_is_never_decode_infeasible_at_any_level(
    monkeypatch, build_outcome, expected_status, expected_returncode
) -> None:
    """v3.4.1: with no decode job ever submitted every shard slot is a None
    placeholder (merged as "timeout" with attempts None). The candidate keeps
    the build's own status (already so in v3.4) and the RESULT rows must be
    measurement_blocked too, never decode_infeasible."""

    fake_worker, decode_calls = _build_failing_worker(build_outcome)
    payload = _evaluate(monkeypatch, fake_worker)

    assert decode_calls() == 0
    assert payload["status"] == expected_status
    assert not payload.get("decode_infeasible")
    assert payload["results"], "one result row per (noise point, decoder) is still assembled"
    for result in payload["results"]:
        assert result["status"] == "measurement_blocked"
        assert result["measurement_complete"] is False
        assert result["combined"]["per_logical_per_round"] is None
        for basis in ("X", "Z"):
            assert result["basis"][basis]["shard_statuses"] == ["timeout", "timeout"]
            assert result["basis"][basis]["shard_attempts"] == [None, None]
            assert result["basis"][basis]["shard_runtime_s"] == [None, None]
    build_failures = [item for item in payload["job_failures"] if item["kind"] == "build"]
    assert len(build_failures) == 1
    assert build_failures[0]["status"] == expected_status
    assert build_failures[0]["returncode"] == expected_returncode
    assert build_failures[0]["job_execution"]["returncode"] == expected_returncode
    assert not _shard_failures(payload)
    assert pool_mod.count_incomplete_candidates([payload]) == 1


def test_promotion_queue_blocks_on_a_timeout_after_a_transient_replay(
    monkeypatch, tmp_path
) -> None:
    from omniqec_agentic.evaluation import evaluate_promotion_queue

    fake_worker, _ = _pool_worker(lambda shard, attempt: "killed" if attempt == 1 else "timeout")
    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"program_id": "bb_v341", "spec": _seed_bb_spec()}) + "\n",
        encoding="utf-8",
    )
    summary = evaluate_promotion_queue(
        queue_path=queue,
        out_dir=tmp_path / "slow",
        verification=_search_verification(),
        include_baseline=False,
        workers=4,
        shot_shards=2,
        seed=0,
    )
    assert summary["status"] == "measurement_blocked"
    assert summary["decode_infeasible_candidate_count"] == 0
    assert summary["measurement_blocked_candidate_count"] == 1
    assert summary["incomplete_candidate_count"] == 1
    assert summary["ok_candidate_count"] == 0
    on_disk = json.loads(Path(summary["artifact_paths"]["summary"]).read_text(encoding="utf-8"))
    assert on_disk["status"] == "measurement_blocked"
    candidate_files = sorted((tmp_path / "slow" / "candidates").glob("*.json"))
    assert candidate_files
    candidate = json.loads(candidate_files[0].read_text(encoding="utf-8"))
    assert candidate["status"] == "measurement_blocked"
    assert candidate["results"][0]["basis"]["Z"]["shard_statuses"] == [
        "timeout_after_transient",
        "timeout_after_transient",
    ]


# ---------------------------------------------------------------------------
# Diagnose helper (hpc/claude_v33_diagnose.py, version-agnostic)
# ---------------------------------------------------------------------------


def _diagnose_module():
    from omniqec_agentic.hpc import claude_v33_diagnose as diagnose

    return diagnose


def _diagnose_candidate(z_statuses: list[str], *, job_failures: list[dict], status: str) -> dict:
    return {
        "program_id": "g0001_p0001",
        "status": status,
        "failure_reason": "one or more required fixed-shot cells are incomplete",
        "job_failures": job_failures,
        "results": [
            {
                "status": "measurement_blocked" if status != "decode_infeasible" else status,
                "decoder": "bposd",
                "basis": {
                    "X": {"status": "ok", "shard_statuses": ["ok", "ok"], "failure_reason": ""},
                    "Z": {
                        "status": "decoder_error",
                        "shard_statuses": z_statuses,
                        # no "returncode=" text anywhere: the hint must come
                        # from the job_failures records
                        "failure_reason": "; ".join("worker stderr: killed" for _ in z_statuses),
                    },
                },
            }
        ],
    }


def test_diagnose_classifies_timeout_after_transient_as_infrastructure() -> None:
    diagnose = _diagnose_module()
    candidate = _diagnose_candidate(
        ["timeout_after_transient", "timeout_after_transient"],
        job_failures=[
            {
                "kind": "shard",
                "shard": shard,
                "status": "timeout_after_transient",
                "returncode": None,
                "job_execution": {"attempts": 2, "transient_retries_used": 1, "returncode": None},
            }
            for shard in (0, 1)
        ],
        status="measurement_blocked",
    )
    verdict = diagnose._classify(candidate)
    assert "Z: 2/2 shards timeout_after_transient" in verdict
    assert "watchdog expiry after a transient replay" in verdict
    assert "treat as infrastructure, not infeasible" in verdict
    assert "decode_infeasible" not in verdict


def test_diagnose_reads_the_kill_returncode_from_job_failures() -> None:
    diagnose = _diagnose_module()
    candidate = _diagnose_candidate(
        ["ok", "subprocess_error"],
        job_failures=[
            {
                "kind": "shard",
                "shard": 1,
                "status": "subprocess_error",
                "job_execution": {"attempts": 2, "transient_retries_used": 1, "returncode": -9},
            }
        ],
        status="measurement_blocked",
    )
    verdict = diagnose._classify(candidate)
    assert "SIGKILL/137 on 1 shard job(s)" in verdict
    assert "most likely OOM" in verdict
    assert "subprocess_error: worker died without parseable JSON (infrastructure" in verdict

    # top-level returncode (v3.4.1 job_failures entry) is honoured as well
    candidate["job_failures"][0] = {
        "kind": "shard", "shard": 1, "status": "subprocess_error", "returncode": 137,
    }
    assert "SIGKILL/137 on 1 shard job(s)" in diagnose._classify(candidate)
    assert diagnose._job_returncodes(candidate) == [137]
    assert diagnose._job_returncodes(candidate, "build") == []

    # a dem_load_error shard is infrastructure too
    candidate = _diagnose_candidate(
        ["ok", "dem_load_error"],
        job_failures=[{"kind": "shard", "shard": 1, "status": "dem_load_error", "returncode": 0}],
        status="measurement_blocked",
    )
    verdict = diagnose._classify(candidate)
    assert "dem_load_error: worker could not load its DEM (infrastructure" in verdict
    assert "OOM" not in verdict

    # A watchdog expiry SIGKILLs the worker itself, so a timeout entry's -9 is
    # the watchdog's own kill: it must never feed the OOM hint (real runs
    # record -9 on every pure timeout, see the v341_infeasible smoke).
    candidate = _diagnose_candidate(
        ["timeout", "dem_load_error"],
        job_failures=[
            {
                "kind": "shard", "shard": 0, "status": "timeout", "returncode": -9,
                "job_execution": {"attempts": 1, "transient_retries_used": 0, "returncode": -9},
            },
            {"kind": "shard", "shard": 1, "status": "dem_load_error", "returncode": 0},
        ],
        status="measurement_blocked",
    )
    verdict = diagnose._classify(candidate)
    assert "Z: 1/2 shards hit the decode watchdog" in verdict
    assert "dem_load_error" in verdict
    assert "OOM" not in verdict and "SIGKILL" not in verdict
    assert diagnose._job_returncodes(candidate) == [0]
    assert diagnose._job_returncodes(candidate, exclude_timeouts=False) == [-9, 0]

    # a candidate that never reached decode: the build job's returncode
    candidate = {
        "program_id": "g0001_p0002",
        "status": "subprocess_error",
        "failure_reason": "worker produced no parseable JSON",
        "job_failures": [{"kind": "build", "status": "subprocess_error", "returncode": -9}],
        "results": [],
    }
    verdict = diagnose._classify(candidate)
    assert verdict.startswith("status=subprocess_error:")
    assert "build job(s) — SIGKILL/137 on 1 build job(s)" in verdict

    # the terminal verdict is unchanged
    infeasible = _diagnose_candidate(
        ["timeout", "timeout"],
        job_failures=[],
        status="decode_infeasible",
    )
    infeasible["decode_infeasible"] = True
    infeasible["failure_reason"] = BUDGET_REASON
    assert diagnose._classify(infeasible).startswith("decode_infeasible (terminal):")

    # and the per-candidate job_failures line is rendered
    line = diagnose._job_failures_line(candidate)
    assert line.strip() == "job_failures: build/subprocess_error x1 (returncode -9)"
    assert diagnose._job_failures_line({"job_failures": []}) == ""


def test_promotion_queue_summary_counts_decode_infeasible_and_stays_ok(monkeypatch, tmp_path) -> None:
    from omniqec_agentic.evaluation import evaluate_promotion_queue

    fake_worker, _ = _pool_worker(lambda shard, attempt: "timeout")
    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"program_id": "bb_v34", "spec": _seed_bb_spec()}) + "\n",
        encoding="utf-8",
    )
    summary = evaluate_promotion_queue(
        queue_path=queue,
        out_dir=tmp_path / "slow",
        verification=_search_verification(),
        include_baseline=False,
        workers=4,
        shot_shards=2,
        seed=0,
    )
    assert summary["status"] == "ok"
    assert summary["decode_infeasible_candidate_count"] == 1
    assert summary["incomplete_candidate_count"] == 0
    assert summary["measurement_blocked_candidate_count"] == 0
    assert summary["timeout_candidate_count"] == 0
    assert summary["ok_candidate_count"] == 0
    on_disk = json.loads(Path(summary["artifact_paths"]["summary"]).read_text(encoding="utf-8"))
    assert on_disk["decode_infeasible_candidate_count"] == 1
    assert on_disk["status"] == "ok"
    rankings = json.loads(
        Path(summary["artifact_paths"]["rankings"]).read_text(encoding="utf-8")
    )["rankings"]
    assert [row["status"] for row in rankings] == ["decode_infeasible"]
    assert rankings[0]["per_logical_per_round"] is None
    assert "decode exceeded the frozen per-shot budget" in rankings[0]["failure_reason"]


# ---------------------------------------------------------------------------
# Workflow: final shards, final audit, reconcile
# ---------------------------------------------------------------------------


def _final_protocol(**overrides) -> dict:
    protocol = {
        "verification": VerificationConfig().to_dict(),
        "rounds_policy": {
            "final": "max_Tc_d",
            "final_rounds": 12,
            "final_rng_namespace": "final_confirmation_v1",
            "min_promotion_distance": 5,
        },
        "final_p_values": [0.005],
        "final_shots": 200_000,
        "slow_workers": 127,
        "shot_shards": 16,
        "final_shot_shards": 128,
        "strict_turns": False,
    }
    protocol.update(overrides)
    return protocol


def _capture_final_slow(monkeypatch, tmp_path: Path, protocol: dict) -> dict:
    import omniqec_agentic.workflow.graph as graph

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "artifact_paths": {}}

    monkeypatch.setattr(graph, "evaluate_promotion_queue", fake_evaluate)
    monkeypatch.setattr(graph, "_validate_all_turn_audits", lambda state: {})
    monkeypatch.setattr(
        graph, "_verification_from_protocol", lambda *args, **kwargs: VerificationConfig()
    )
    monkeypatch.setattr(
        graph,
        "_final_entries",
        lambda state, structural: (
            [
                {
                    "program_id": "candidate",
                    "spec": {"family": "bb", "ell": 6, "m": 1},
                    "code_hash": "abc",
                    "selection_channels": ["structural_high_k"],
                }
            ],
            {"entries": 1},
        ),
    )
    monkeypatch.setattr(
        graph,
        "_audit_final_measurements",
        lambda state, runs: {"enabled": False, "status": "complete", "errors": []},
    )
    state = {
        "run_dir": str(tmp_path),
        "protocol": protocol,
        "spec": {"qubit_budget": 144, "families": ["bb"]},
        "baseline_summary": {},
    }
    result = graph.final_slow_node(state)
    assert result["final_slow"]["status"] == "ok"
    return captured


def test_final_slow_partitions_into_final_shards_while_search_keeps_sixteen(
    monkeypatch, tmp_path
) -> None:
    import omniqec_agentic.campaign as campaign
    import omniqec_agentic.workflow.graph as graph
    from types import SimpleNamespace

    protocol = _final_protocol(
        proposal_count=1,
        generations_per_turn=1,
        feedback_mode="both",
        racing=RacingConfig().to_dict(),
    )

    search_captured: dict[str, object] = {}

    def fake_cycle(*args, **kwargs):
        search_captured.update(kwargs)
        return SimpleNamespace(status="ok", artifact_paths={})

    monkeypatch.setattr(campaign, "run_agentic_cycle", fake_cycle)
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph, "_verification_from_protocol", lambda *args, **kwargs: VerificationConfig()
    )
    graph.execute_plan_node(
        {
            "run_id": "campaign_b144",
            "run_dir": str(tmp_path),
            "turn": 0,
            "plan": {
                "provider": "dry_run",
                "generations": 1,
                "proposal_count": 1,
                "rng_seeds_per_program": 1,
            },
            "protocol": protocol,
            "spec": {"qubit_budget": 144, "families": ["bb"]},
            "structural_proxy_elites": [],
            "unmeasurable_hashes": [],
            "baseline_summary": {},
            "budget": {},
        }
    )
    assert search_captured["slow_shot_shards"] == 16

    final_captured = _capture_final_slow(monkeypatch, tmp_path / "final", protocol)
    assert final_captured["shot_shards"] == 128
    assert final_captured["workers"] == 127


def test_final_slow_falls_back_to_the_search_shards_without_a_final_key(
    monkeypatch, tmp_path
) -> None:
    protocol = _final_protocol()
    del protocol["final_shot_shards"]
    captured = _capture_final_slow(monkeypatch, tmp_path, protocol)
    assert captured["shot_shards"] == 16


def _final_audit_state(tmp_path: Path) -> dict:
    return {
        "run_dir": str(tmp_path),
        "protocol": {
            "strict_turns": True,
            "anchor_p": 0.005,
            "final_shots": 20000,
            "rounds_policy": {
                "final": "max_Tc_d",
                "final_rounds": 12,
                "final_rng_namespace": "final_confirmation_v1",
                "min_promotion_distance": 5,
            },
        },
    }


def _final_row(program_id: str, status: str = "ok") -> dict:
    row = {
        "program_id": program_id,
        "status": status,
        "noise_value": 0.005,
        "shots_x": 20000,
        "shots_z": 20000,
        "total_shots": 40000,
        "rounds": 12,
        "decoder": TELESCOPING,
        "decoder_config": dict(TELESCOPING_BUILD),
        "per_logical_per_round": 1e-4,
    }
    if status == "decode_infeasible":
        # a budget-infeasible row: Z never completed, nothing publishable
        row.update(
            shots_z=0,
            total_shots=20000,
            per_logical_per_round=None,
            decode_infeasible=True,
            failure_reason=BUDGET_REASON,
        )
    elif status != "ok":
        row.update(per_logical_per_round=None)
    return row


def test_final_audit_skips_decode_infeasible_rows_but_still_needs_an_ok_row(tmp_path) -> None:
    from omniqec_agentic.workflow.graph import _audit_final_measurements

    rankings = tmp_path / "rankings.json"
    runs = [{"rounds": 12, "artifact_paths": {"rankings": str(rankings)}}]

    _write_json(
        rankings,
        {"schema": "slow_rankings", "rankings": [
            _final_row("ok_code"),
            _final_row("slow_code", "decode_infeasible"),
        ]},
    )
    audit = _audit_final_measurements(_final_audit_state(tmp_path), runs)
    assert audit["status"] == "complete"
    assert audit["measured_rows"] == 1
    assert audit["errors"] == []

    _write_json(
        rankings,
        {"schema": "slow_rankings", "rankings": [_final_row("slow_code", "decode_infeasible")]},
    )
    with pytest.raises(RuntimeError, match="no successful final measurement rows"):
        _audit_final_measurements(_final_audit_state(tmp_path), runs)

    _write_json(
        rankings,
        {"schema": "slow_rankings", "rankings": [
            _final_row("ok_code"),
            _final_row("blocked_code", "measurement_blocked"),
        ]},
    )
    with pytest.raises(RuntimeError, match="is not ok"):
        _audit_final_measurements(_final_audit_state(tmp_path), runs)


def test_reconcile_retires_decode_infeasible_codes_from_the_slow_queue(
    monkeypatch, tmp_path
) -> None:
    """A budget-infeasible code keeps its structural ledger row (k/d_search are
    still valid parents) but must never be re-queued or shortlisted."""

    import omniqec_agentic.workflow.graph as graph

    rankings = tmp_path / "rankings.json"
    _write_json(
        rankings,
        {"schema": "slow_rankings", "rankings": [
            {**_final_row("ok_code"), "code_hash": "ab" * 16, "d_proxy": 6},
            {**_final_row("slow_code", "decode_infeasible"), "code_hash": "cd" * 16},
            # flag without the status (stale producer): still retired
            {
                **_final_row("flagged_code", "measurement_blocked"),
                "code_hash": "ef" * 16,
                "decode_infeasible": True,
            },
            # an ordinary blocked measurement stays re-queueable
            {**_final_row("blocked_code", "measurement_blocked"), "code_hash": "12" * 16},
        ]},
    )
    stubs = {
        "_append_structural_store": lambda *args, **kwargs: None,
        "_reduce_elites_from_store": lambda *args, **kwargs: [],
        "_structural_ledger_from_cycle": lambda *args, **kwargs: [],
        "_structural_elites_from_cycle": lambda *args, **kwargs: [],
        "_reconcile_learning": lambda *args, **kwargs: {},
        "_run_turn_reflection": lambda *args, **kwargs: None,
        "_complete_turn_audit": lambda *args, **kwargs: None,
        "_campaign_tokens": lambda *args, **kwargs: 0,
        "_fast_top_k": lambda *args, **kwargs: [],
        "_stop_decision": lambda *args, **kwargs: {"finalize": False},
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(graph, name, stub)
    state = {
        "run_id": "campaign_b144",
        "run_dir": str(tmp_path),
        "turn": 0,
        "protocol": {"max_turns": 1, "structural_retention": {"elites_limit": 24}},
        "last_cycle": {"status": "ok", "artifact_paths": {"slow_rankings": str(rankings)}},
        "hof_summary": [],
        "structural_proxy_elites": [],
        "structural_ledger_hashes": [],
        "unmeasurable_hashes": ["11" * 16],
        "baseline_summary": {},
        "budget": {"turns_used": 0},
    }

    result = graph.reconcile_node(state)

    assert set(result["unmeasurable_hashes"]) == {"11" * 16, "cd" * 16, "ef" * 16}
    assert [row["code_hash"] for row in result["hof_summary"]] == ["ab" * 16]


# ---------------------------------------------------------------------------
# Protocol snapshot and strict freezes
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


def test_snapshot_freezes_search_and_final_shards_separately() -> None:
    from omniqec_agentic.workflow.state import FORMAL_FINAL_SHOT_SHARDS, FORMAL_SHOT_SHARDS

    assert (FORMAL_SHOT_SHARDS, FORMAL_FINAL_SHOT_SHARDS) == (16, 128)
    protocol = _snapshot()
    assert protocol["protocol_profile"] == "telescoping_structural_v3_7"
    assert protocol["shot_shards"] == 16  # SEARCH shards
    assert protocol["final_shot_shards"] == 128
    assert protocol["verification"]["decode_shot_budget_s"] == 8.0
    assert protocol["verification"]["decode_shard_timeout_s"] == 1800.0
    assert protocol["verification"]["timeout_retries"] == 0
    assert protocol["verification"]["transient_retries"] == 1
    strict = _strict()
    assert (strict["shot_shards"], strict["final_shot_shards"]) == (16, 128)
    assert strict["protocol_profile"] == "telescoping_structural_v3_7"


def test_nonformal_final_shards_retag_the_profile() -> None:
    assert _snapshot(final_shot_shards=64)["protocol_profile"] == (
        "telescoping_structural_v3_7+finalshards64"
    )
    assert _snapshot(shot_shards=8)["protocol_profile"] == "telescoping_structural_v3_7+shards8"
    assert _snapshot(shot_shards=8, final_shot_shards=64)["protocol_profile"] == (
        "telescoping_structural_v3_7+shards8+finalshards64"
    )
    assert _snapshot(final_shot_shards=64)["final_shot_shards"] == 64


def test_final_shards_are_bounded_by_final_shots() -> None:
    with pytest.raises(ValueError, match="final_shot_shards"):
        _snapshot(final_shot_shards=0)
    with pytest.raises(ValueError, match="final_shot_shards"):
        _snapshot(final_shot_shards=129, final_shots=128)
    assert _snapshot(final_shot_shards=128, final_shots=128)["final_shot_shards"] == 128


def test_strict_snapshot_freezes_both_shard_counts() -> None:
    with pytest.raises(ValueError, match="final_shot_shards"):
        _strict(final_shot_shards=16)
    with pytest.raises(ValueError, match="final_shot_shards"):
        _strict(final_shot_shards=64)
    with pytest.raises(ValueError, match="shot_shards=16"):
        _strict(shot_shards=8)


def test_a_v33_checkpoint_cannot_resume_into_v34() -> None:
    from omniqec_agentic.workflow.checkpoint import validate_resume_protocol

    current = _snapshot()
    legacy = json.loads(json.dumps(current))
    legacy.pop("final_shot_shards")
    legacy["protocol_profile"] = "bposd_structural_v3_3"
    legacy["verification"]["timeout_retries"] = 2
    legacy["verification"].pop("decode_shot_budget_s")
    legacy["verification"].pop("transient_retries")
    with pytest.raises(ValueError, match="protocol snapshot mismatch"):
        validate_resume_protocol(legacy, current)

    # The final shard count alone is a re-partition of the final RNG streams.
    repartition = json.loads(json.dumps(current))
    repartition["final_shot_shards"] = 16
    with pytest.raises(ValueError, match="final_shot_shards"):
        validate_resume_protocol(repartition, current)


def test_run_campaign_defaults_to_the_frozen_final_shards() -> None:
    from omniqec_agentic.workflow.graph import run_campaign
    from omniqec_agentic.workflow.state import FORMAL_FINAL_SHOT_SHARDS

    params = inspect.signature(run_campaign).parameters
    assert params["final_shot_shards"].default == FORMAL_FINAL_SHOT_SHARDS == 128
    assert params["shot_shards"].default == 16


# ---------------------------------------------------------------------------
# CLI freezes and plumbing
# ---------------------------------------------------------------------------


def _cli_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    args = dict(
        strict_turns=True,
        provider="openai",
        budget=144,
        out=str(tmp_path),
        weight_limit=None,
        families=None,
        p=0.005,
        turns=1,
        proposal_count=24,
        generations_per_turn=1,
        rng_seeds_per_program=20,
        slow_workers=127,
        slow_shots=25_000,
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
        final_shots=25_000,
        feedback_mode="both",
        resume=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_strict_cli_freezes_both_shard_counts_and_names_v34(tmp_path) -> None:
    from omniqec_agentic.cli import run_campaign_command

    with pytest.raises(SystemExit, match="telescoping_structural_v3_7"):
        run_campaign_command(_cli_args(tmp_path, final_shot_shards=16))
    with pytest.raises(SystemExit, match="final-shot-shards"):
        run_campaign_command(_cli_args(tmp_path, final_shot_shards=64))
    with pytest.raises(SystemExit, match="telescoping_structural_v3_7"):
        run_campaign_command(_cli_args(tmp_path, shot_shards=8))


def test_cli_plumbs_v34_defaults_and_flags_into_run_campaign(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)

    assert cli.run_campaign_command(_cli_args(tmp_path, strict_turns=False, provider="dry_run")) == 0
    assert captured["shot_shards"] == 16
    assert captured["final_shot_shards"] == 128
    verification = captured["verification"]
    assert verification.decode_shot_budget_s == pytest.approx(8.0)
    assert verification.decode_shard_timeout_s == pytest.approx(1800.0)
    assert verification.timeout_retries == 0
    assert verification.transient_retries == 1

    captured.clear()
    assert cli.run_campaign_command(
        _cli_args(
            tmp_path,
            strict_turns=False,
            provider="dry_run",
            final_shot_shards=64,
            decode_shot_budget_s=2.0,
            transient_retries=0,
            timeout_retries=1,
        )
    ) == 0
    assert captured["final_shot_shards"] == 64
    verification = captured["verification"]
    assert verification.decode_shot_budget_s == pytest.approx(2.0)
    assert verification.transient_retries == 0
    assert verification.timeout_retries == 1


# ---------------------------------------------------------------------------
# Standalone validator: v3.4 geometry accepted, v3.3 values rejected
# ---------------------------------------------------------------------------


def _validate(run_dir: Path) -> int:
    return validate(run_dir, turns=1, p_value=0.005, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def _infeasible_ranking_row(program_id: str = "g0001_p0002") -> dict:
    return {
        "program_id": program_id,
        "status": "decode_infeasible",
        "decode_infeasible": True,
        "decoder": TELESCOPING,
        "decoder_config": dict(TELESCOPING_BUILD),
        "rounds": 12,
        "per_logical_per_round": None,
        "failure_reason": BUDGET_REASON,
    }


def test_validator_pins_the_v34_geometry_constants() -> None:
    assert validator.PROTOCOL_PROFILE == "telescoping_structural_v3_7"
    assert validator.FORMAL_SHOT_SHARDS == 16
    assert validator.FORMAL_FINAL_SHOT_SHARDS == 128
    assert validator.FORMAL_JOB_POLICY == {
        "build_timeout_s": 3600.0,
        "decode_shard_timeout_s": 1800.0,
        "distance_timeout_s": 1800.0,
        "timeout_retries": 0,
        "decode_shot_budget_s": 8.0,
        "transient_retries": 1,
    }


def test_validator_accepts_v34_geometry_with_decode_infeasible_rows(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    final_dir = tmp_path / "final_slow" / "T12"
    for stage_dir in (slow_dir, final_dir):
        rankings = _rankings(12)
        rankings["rankings"].append(_infeasible_ranking_row())
        _write_json(stage_dir / "rankings.json", rankings)
    # Executor-emitted job configs carry each stage's own shard count.
    _write_json(
        slow_dir / "candidates" / "candidate.json",
        {"status": "ok", "job_config": {"workers": 127, "shot_shards": 16, **dict(FORMAL_JOB_POLICY)}},
    )
    _write_json(
        final_dir / "candidates" / "candidate.json",
        {
            "status": "decode_infeasible",
            "decode_infeasible": True,
            "job_config": {"workers": 127, "shot_shards": 128, **dict(FORMAL_JOB_POLICY)},
        },
    )

    assert _validate(tmp_path) == 1


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("timeout_retries", 2, "protocol snapshot: timeout_retries=2"),
        ("decode_shot_budget_s", None, "protocol snapshot: decode_shot_budget_s=None"),
        ("transient_retries", 0, "protocol snapshot: transient_retries=0"),
    ],
)
def test_validator_rejects_v33_execution_policy_in_the_snapshot(
    tmp_path, key: str, value, message: str
) -> None:
    report = _strict_report()
    if value is None:
        del report["protocol"]["verification"][key]
    else:
        report["protocol"]["verification"][key] = value
    _write_validated_run(tmp_path, report)

    with pytest.raises(ValueError, match=re.escape(message)):
        _validate(tmp_path)


def test_validator_rejects_v33_final_sharding_in_the_snapshot(tmp_path) -> None:
    report = _strict_report()
    report["protocol"]["final_shot_shards"] = 16
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="final_shot_shards=16"):
        _validate(tmp_path)

    report = _strict_report()
    del report["protocol"]["final_shot_shards"]
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="final_shot_shards=None"):
        _validate(tmp_path)


def test_validator_rejects_a_final_run_config_with_search_shards(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    _write_json(
        tmp_path / "final_slow" / "T12" / "slow_run_config.json",
        _slow_run_config(rounds=12, seed=FINAL_BASE_SEED, shot_shards=16),
    )
    with pytest.raises(ValueError, match="final run config: shot_shards=16"):
        _validate(tmp_path)


def test_validator_rejects_a_search_run_config_with_final_shards(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    _write_json(
        tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "slow_run_config.json",
        _slow_run_config(rounds=12, shot_shards=128),
    )
    with pytest.raises(ValueError, match="slow run config: shot_shards=128"):
        _validate(tmp_path)


@pytest.mark.parametrize(
    ("stage", "shot_shards"),
    [("search", 128), ("final", 16)],
)
def test_validator_rejects_a_candidate_job_config_with_the_other_stages_shards(
    tmp_path, stage: str, shot_shards: int
) -> None:
    _write_validated_run(tmp_path, _strict_report())
    candidate_dir = (
        tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "candidates"
        if stage == "search"
        else tmp_path / "final_slow" / "T12" / "candidates"
    )
    _write_json(
        candidate_dir / "candidate.json",
        {
            "status": "ok",
            "job_config": {"workers": 127, "shot_shards": shot_shards, **dict(FORMAL_JOB_POLICY)},
        },
    )
    with pytest.raises(ValueError, match=f"candidate.json job_config: shot_shards={shot_shards}"):
        _validate(tmp_path)


def test_validator_rejects_a_run_config_with_v33_retry_policy(tmp_path) -> None:
    path = tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "slow_run_config.json"

    _write_validated_run(tmp_path, _strict_report())
    config = _slow_run_config(rounds=12)
    del config["verification"]["decode_shot_budget_s"]
    _write_json(path, config)
    with pytest.raises(ValueError, match="decode_shot_budget_s=None"):
        _validate(tmp_path)

    config = _slow_run_config(rounds=12)
    config["verification"]["timeout_retries"] = 2
    _write_json(path, config)
    with pytest.raises(ValueError, match="timeout_retries=2"):
        _validate(tmp_path)

    config = _slow_run_config(rounds=12)
    config["verification"]["transient_retries"] = 0
    _write_json(path, config)
    with pytest.raises(ValueError, match="transient_retries=0"):
        _validate(tmp_path)


def test_validator_still_requires_one_ok_final_row(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    _write_json(
        tmp_path / "final_slow" / "T12" / "rankings.json",
        {"schema": "slow_rankings", "rankings": [_infeasible_ranking_row("g0001_p0001")]},
    )
    with pytest.raises(ValueError, match="final rankings: no ok row"):
        _validate(tmp_path)
