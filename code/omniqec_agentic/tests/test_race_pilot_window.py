"""OMNIQEC_RACE_PILOT_WINDOW (2026-11-07): a smaller pilot window decodes fewer discarded shards and records exactly
the same measurement as the whole-pilot wave (the default)."""
from __future__ import annotations

import json

import pytest

from omniqec_agentic.config import VerificationConfig, formal_racing_v3_maxerr30
from omniqec_agentic.evaluation import pool as pool_mod


def _seed_bb_spec() -> dict:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    return run_construct_source(SEED_BB, program_id="race_window_bb", seed=0).constructed.spec


def _fake_worker(failures_per_shard: int, decoded: list[tuple[str, int]]):
    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}

    def fake(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, dict(fine)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        unit = request["units"][0]
        shots = int(request["shots"])
        decoded.append((unit["basis"], int(request["shard"])))
        failures = min(shots, int(failures_per_shard))
        cell = {
            "status": "ok", "decoder": unit["decoder"], "shots": shots, "failures": failures,
            "logical_error_rate": failures / shots, "runtime_s": 0.001,
            "metrics": {"requested_shots": shots, "actual_shots": shots, "sampling_complete": True,
                        "stopped_by_max_errors": False},
        }
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, dict(fine)

    return fake


def _measure(monkeypatch, *, window: int | None, failures_per_shard: int) -> tuple[dict, int]:
    decoded: list[tuple[str, int]] = []
    monkeypatch.setattr(pool_mod, "run_worker_request", _fake_worker(failures_per_shard, decoded))
    if window is None:
        monkeypatch.delenv("OMNIQEC_RACE_PILOT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("OMNIQEC_RACE_PILOT_WINDOW", str(window))
    payload = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_window", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(shots=25_000, max_errors=None, rounds=12),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=16,
        racing=formal_racing_v3_maxerr30(25_000),
    )[0]
    return payload, len(decoded)


def _recorded(payload: dict) -> str:
    racing = dict(payload.get("racing") or {})
    racing.pop("lookahead", None)
    basis = (payload.get("results") or [{}])[0].get("basis") or {}
    keep = {
        "status": payload.get("status"),
        "stages": racing.get("stages"),
        "stopped_by": racing.get("stopped_by"),
        "shots_per_basis": racing.get("shots_per_basis"),
        "counts": {b: (basis[b].get("failures"), basis[b].get("shots")) for b in sorted(basis)},
        "combined": (payload.get("results") or [{}])[0].get("combined"),
    }
    return json.dumps(keep, sort_keys=True, default=str)


@pytest.mark.parametrize("failures_per_shard,stop_shard", [(40, 1), (5, 6), (0, None)])
def test_pilot_window_records_the_same_measurement(monkeypatch, failures_per_shard, stop_shard) -> None:
    full, full_decoded = _measure(monkeypatch, window=None, failures_per_shard=failures_per_shard)
    for window in (2, 1):
        small, small_decoded = _measure(monkeypatch, window=window, failures_per_shard=failures_per_shard)
        assert _recorded(small) == _recorded(full)
        assert small_decoded <= full_decoded
        if stop_shard is not None:
            # both bases stop at the same boundary; at most `window - 1` shards per basis decode past it
            assert small_decoded <= 2 * (stop_shard + window - 1)
    if stop_shard == 1:
        assert full_decoded == 2 * 8  # the whole pilot of both bases went out at once
