"""v3.6.1: a DEM the telescoping decoder cannot represent is the candidate's
own terminal verdict (``decoder_unsupported``), not a measurement_blocked stop.

Measured 2026-09-05 (GPT arm, b360, turn 0): the resident seed program
seed_0004 ([[24,8,3]], circuit distance 2) reached the slow queue, the
telescoping system build raised "DEM contains error mechanisms with identical
detector support ... but different logical-observable support ...", the
candidate became build_error, the queue measurement_blocked, the strict turn
failed and the launcher wrote a FATAL_STOP -- for a code that is simply
unmeasurable by this decoder. Seeds are re-evaluated every turn, so every arm
was one promotion away from the same stop."""

from __future__ import annotations

import pytest

from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation import worker as worker_mod
from omniqec_agentic.evaluation.writeback import slow_queue_status
from omniqec_agentic.hpc.validate_fig2_campaign import TERMINAL_ROW_STATUSES
from omniqec_agentic.verification.decoders import is_decoder_unsupported_reason

TELESCOPING_MESSAGE = (
    "DEM contains error mechanisms with identical detector support [0, 1, 3, 4, 7, 9, 13] "
    "but different logical-observable support [1] and [5]. XORing them gives a detector-free "
    "logical error."
)


def test_only_the_dem_ambiguity_is_a_candidate_verdict() -> None:
    assert is_decoder_unsupported_reason(TELESCOPING_MESSAGE)
    assert is_decoder_unsupported_reason(ValueError(TELESCOPING_MESSAGE))
    assert not is_decoder_unsupported_reason("No module named 'telescoping_decoder'")
    assert not is_decoder_unsupported_reason("detector-kind mask mismatch: 1728 vs 1730")
    assert not is_decoder_unsupported_reason("identical detector support")  # half a marker
    assert not is_decoder_unsupported_reason(None)


def test_worker_maps_the_ambiguity_to_decoder_unsupported_and_the_rest_to_build_error() -> None:
    verdict = worker_mod.telescoping_build_failure(ValueError(TELESCOPING_MESSAGE))
    assert verdict["status"] == "decoder_unsupported" and verdict["decoder_unsupported"] is True
    assert verdict["failure_reason"].startswith("telescoping system build failed: DEM contains")
    infra = worker_mod.telescoping_build_failure(RuntimeError("no C compiler found"))
    assert infra == {"status": "build_error", "failure_reason": "telescoping system build failed: no C compiler found"}


def test_pool_records_decoder_unsupported_as_terminal_not_blocked(monkeypatch) -> None:
    import test_measurement_completeness as tmc

    calls = {"shards": 0}

    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return worker_mod.telescoping_build_failure(ValueError(TELESCOPING_MESSAGE)), {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        calls["shards"] += 1
        return tmc._successful_shard(request)

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    payload = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "seed_like", "spec": tmc._seed_bb_spec()}],
        verification=VerificationConfig(shots=2_000, max_errors=None, rounds=12),
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=4,
        shot_shards=2,
    )[0]
    assert calls["shards"] == 0  # no decode job is ever submitted for a failed build
    assert payload["status"] == "decoder_unsupported"
    assert payload["decoder_unsupported"] is True
    assert payload["failure_reason"].startswith("telescoping system build failed")
    # Terminal for the queue verdict, exactly like decode_infeasible.
    assert pool_mod.count_incomplete_candidates([payload]) == 0
    assert slow_queue_status(
        record_count=1,
        timeout_candidate_count=0,
        decoder_attempt_candidate_count=0,
        ok_result_count=0,
        decoder_unsupported_candidate_count=1,
    ) == "ok"
    # A plain build_error still blocks (infrastructure, needs a human).
    assert pool_mod.count_incomplete_candidates([{"status": "build_error"}]) == 1
    assert slow_queue_status(
        record_count=2,
        timeout_candidate_count=0,
        decoder_attempt_candidate_count=1,
        ok_result_count=1,
        incomplete_candidate_count=1,
        decoder_unsupported_candidate_count=1,
    ) == "measurement_blocked"


def test_validator_and_final_audit_treat_it_as_terminal() -> None:
    assert "decoder_unsupported" in TERMINAL_ROW_STATUSES
    from pathlib import Path

    graph = (Path(__file__).resolve().parents[1] / "workflow" / "graph.py").read_text(encoding="utf-8")
    assert 'if row.get("status") == "decoder_unsupported":' in graph
    assert "unmeasurable |= catastrophic_now | decode_infeasible_now | decoder_unsupported_now" in graph
    assert "decode_infeasible -= catastrophic_now | decoder_unsupported_now" in graph
