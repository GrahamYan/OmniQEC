"""Persistent decode workers vs one-shot subprocess workers: the same
candidate, the same protocol, REAL workers (SEC synthesis + telescoping
decoder) -> identical shard cells, stages and plpr. Runs the vendored
decoder on a [[30,2,5]] BB code with a few hundred shots (~1-2 min); set
OMNIQEC_SKIP_E2E=1 to skip."""
from __future__ import annotations

import os

import pytest

from omniqec_agentic.config import RacingConfig, VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod

pytestmark = pytest.mark.skipif(os.environ.get("OMNIQEC_SKIP_E2E") == "1", reason="OMNIQEC_SKIP_E2E=1")


def _small_bb_spec() -> dict:
    # [[30,2,5]] BB on C3xC5 (A = xy + x^2y^2, B = x + y): the Claude b60 v3.7
    # winner; small enough for a real synthesis + decode in ~1 min and, unlike
    # the seed BB [[30,4,2]], accepted by the telescoping system builder.
    return {"family": "bb", "ell": 3, "m": 5, "A": [[1, 1], [2, 2]], "B": [[1, 0], [0, 1]]}


def _run(monkeypatch, *, persistent: bool, racing: RacingConfig | None):
    if persistent:
        monkeypatch.setenv("OMNIQEC_PERSISTENT_WORKERS", "1")
        monkeypatch.delenv("OMNIQEC_SHARD_SPLIT", raising=False)   # -> 64 chunks per shard
    else:
        monkeypatch.delenv("OMNIQEC_PERSISTENT_WORKERS", raising=False)
        monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "3")
    verification = VerificationConfig(shots=300, max_errors=None, rounds=4, transient_retries=1)
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "ppool_bb", "spec": _small_bb_spec()}],
        verification=verification, hardware=None, seed=7, batch_size=64, prefer_coloring=True,
        workers=4, shot_shards=2, racing=racing, racing_thresholds=None,
    )[0]


def _cells(payload):
    out = {}
    for basis, cell in payload["results"][0]["basis"].items():
        out[basis] = {k: cell.get(k) for k in ("shots", "failures", "shard_count", "shard_statuses", "status")}
        metrics = cell.get("metrics") or {}
        out[basis]["stage_counts"] = metrics.get("stage_counts")
        out[basis]["declared_failures"] = metrics.get("declared_failures")
    out["plpr"] = payload["results"][0]["combined"]["per_logical_per_round"]
    out["status"] = payload["status"]
    return out


def test_persistent_workers_reproduce_the_subprocess_measurement(monkeypatch) -> None:
    base = _run(monkeypatch, persistent=False, racing=None)
    assert base["status"] == "ok", base.get("failure_reason")
    assert base["job_config"]["worker_mode"] == "subprocess" and base["job_config"]["shard_split"] == 3
    fast = _run(monkeypatch, persistent=True, racing=None)
    assert fast["status"] == "ok", fast.get("failure_reason")
    assert fast["job_config"]["worker_mode"] == "persistent" and fast["job_config"]["shard_split"] == 64
    assert _cells(fast) == _cells(base)
    for basis in ("X", "Z"):
        assert fast["results"][0]["basis"][basis]["shots"] == 300


def test_persistent_workers_reproduce_the_racing_decision(monkeypatch) -> None:
    racing = RacingConfig(enabled=True, shots_pilot=100, batch=100, shots_confirm=300, shots_max=300, max_errors=5)
    base = _run(monkeypatch, persistent=False, racing=racing)
    fast = _run(monkeypatch, persistent=True, racing=racing)
    assert base["status"] == "ok" and fast["status"] == "ok"
    assert _cells(fast) == _cells(base)
    assert fast["racing"] == base["racing"]
