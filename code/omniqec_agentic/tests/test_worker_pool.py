"""Persistent decode workers (2026-09-14): dispatch, watchdog kill+respawn,
crash replay and the outcome schema shared with the one-shot path."""
from __future__ import annotations

import os
import time

import pytest

from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.worker_pool import PersistentWorkerPool, persistent_workers_enabled


def _timeout_for(request, verification):
    if request.get("mode") == "sleep":
        return float(request.get("_watchdog_s") or 1.0)
    return 30.0


@pytest.fixture
def ppool():
    pool = PersistentWorkerPool(
        2, verification=VerificationConfig(timeout_retries=0, transient_retries=1), timeout_for=_timeout_for
    )
    yield pool
    pool.shutdown()


def test_ping_round_trips_and_reuses_the_same_process() -> None:
    # One worker. With two idle workers the second request goes to whichever wakes first: key affinity
    # is a preference the dispatcher drops rather than leave a request waiting (worker_pool._next), so a
    # same-pid assertion on the two-worker pool failed about one run in three (2026-11-06, on main too).
    pool = PersistentWorkerPool(
        1, verification=VerificationConfig(timeout_retries=0, transient_retries=1), timeout_for=_timeout_for
    )
    try:
        first = pool.submit({"mode": "ping", "echo": 1}, key="a").result(timeout=60)
        assert first["status"] == "ok" and first["echo"] == 1
        assert first["job_execution"]["attempts"] == 1 and first["job_execution"]["worker_mode"] == "persistent"
        assert "ru_maxrss" in first["job_execution"]
        second = pool.submit({"mode": "ping", "echo": 2}, key="a").result(timeout=60)
        # persistence: the second request is served by the same, still alive process
        assert second["pid"] == first["pid"]
        assert pool.stats["respawns"] == 0
    finally:
        pool.shutdown()


def test_watchdog_kills_and_respawns_then_keeps_serving(ppool) -> None:
    slow = ppool.submit({"mode": "sleep", "seconds": 30, "_watchdog_s": 0.5}, key="a").result(timeout=60)
    assert slow["status"] == "timeout"
    assert slow["job_execution"]["timed_out"] is True and slow["job_execution"]["attempts"] == 1
    assert "exceeded timeout_s=0.5" in slow["failure_reason"]
    after = ppool.submit({"mode": "ping", "echo": 3}, key="a").result(timeout=60)
    assert after["status"] == "ok" and after["echo"] == 3
    assert ppool.stats["timeouts"] == 1 and ppool.stats["respawns"] >= 1


def test_a_dying_worker_is_replayed_once_then_reported(ppool) -> None:
    crash = ppool.submit({"mode": "crash", "code": 7}, key="b").result(timeout=60)
    assert crash["status"] == "subprocess_error"
    assert crash["job_execution"]["attempts"] == 2  # transient_retries=1 replayed it once
    assert crash["job_execution"]["transient_retries_used"] == 1
    assert crash["returncode"] == 7 or crash["returncode"] in (None, -9)
    assert "crash requested" in crash["failure_reason"] or "no parseable JSON" in crash["failure_reason"]
    again = ppool.submit({"mode": "ping", "echo": 4}, key="b").result(timeout=60)
    assert again["status"] == "ok"


def test_many_requests_are_spread_over_the_workers(ppool) -> None:
    futs = [ppool.submit({"mode": "sleep", "seconds": 0.3}, key=i % 2) for i in range(6)]
    started = time.monotonic()
    pids = {f.result(timeout=60)["pid"] for f in futs}
    assert time.monotonic() - started < 6 * 0.3  # two workers ran them in parallel
    assert len(pids) == 2


def test_knob_and_pool_default_split(monkeypatch) -> None:
    monkeypatch.delenv("OMNIQEC_PERSISTENT_WORKERS", raising=False)
    assert persistent_workers_enabled() is False
    monkeypatch.setenv("OMNIQEC_PERSISTENT_WORKERS", "1")
    assert persistent_workers_enabled() is True
    assert pool_mod.PERSISTENT_DEFAULT_SHARD_SPLIT == 64
