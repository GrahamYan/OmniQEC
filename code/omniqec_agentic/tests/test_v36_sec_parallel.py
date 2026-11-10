"""The SEC certification probe pool is bit-identical to the sequential loop.

2026-09-04: rank_twobasis_schedules gained an opt-in process pool for its
(order, seed) ISD probes (OMNIQEC_SEC_WORKERS / ``workers``). Every seed's rng
is a pure function of (seed0, index, side, seed_offset, s), so the pool must
reproduce the sequential result exactly -- schedule, chosen indices, per-seed
values, isd counters, checkpoint entries -- with ``workers=1`` remaining the
historical code path."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from omniqec_agentic.config import SecPolicyConfig
from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation import BASELINE_BB_SPEC
from omniqec_agentic.sec.twobasis_rank import (
    CertificationProgress,
    rank_twobasis_schedules,
    resolve_sec_workers,
)

# Two seeds and a refine stage so both fan-outs are exercised; small iteration
# counts keep the test in seconds.
POLICY = SecPolicyConfig(
    sec_candidates=4, sec_cert_iters=12, sec_cert_seeds=2, sec_cert_iters_scale=0.0,
    sec_refine_top=2, sec_refine_iters=8,
)


def _json(obj) -> str:
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    return json.dumps(dataclasses.asdict(obj), sort_keys=True, default=default)


@pytest.fixture(scope="module")
def bb72():
    compiled = build_compiled_code(BASELINE_BB_SPEC)
    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    return compiled, hx, hz


def test_probe_pool_reproduces_the_sequential_ranking_bit_for_bit(bb72):
    compiled, hx, hz = bb72
    sequential = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=POLICY, workers=1)
    parallel = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=POLICY, workers=3)
    assert _json(sequential) == _json(parallel)
    meta = parallel.metadata
    assert meta["isd_runs"] == sequential.metadata["isd_runs"] > 0
    assert meta["stage1_seed_values"] == sequential.metadata["stage1_seed_values"]
    assert meta["refine_seed_values"] == sequential.metadata["refine_seed_values"]
    assert meta["isd_iters_executed"] == sequential.metadata["isd_iters_executed"]


def test_probe_pool_writes_the_same_checkpoint_and_resumes_from_it(bb72, tmp_path):
    compiled, hx, hz = bb72
    seq_progress = CertificationProgress(tmp_path / "seq.json")
    par_progress = CertificationProgress(tmp_path / "par.json")
    sequential = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=POLICY, progress=seq_progress, workers=1)
    parallel = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=POLICY, progress=par_progress, workers=2)
    assert seq_progress.entries == par_progress.entries
    assert sequential.metadata["progress_entries_reused"] == parallel.metadata["progress_entries_reused"] == 0
    # A resumed synthesis reuses every entry and probes nothing new, with or without the pool.
    resumed = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=POLICY, progress=CertificationProgress(tmp_path / "par.json"), workers=2)
    assert resumed.metadata["progress_entries_reused"] > 0
    assert resumed.metadata["isd_runs"] == 0
    assert _json(resumed).replace('"progress_entries_reused": %d' % resumed.metadata["progress_entries_reused"], '"progress_entries_reused": 0').replace('"isd_iters_executed": 0', '"isd_iters_executed": %d' % sequential.metadata["isd_iters_executed"]).replace('"isd_runs": 0', '"isd_runs": %d' % sequential.metadata["isd_runs"]) == _json(sequential)


def test_sec_workers_come_from_the_environment_and_are_capped(monkeypatch):
    monkeypatch.delenv("OMNIQEC_SEC_WORKERS", raising=False)
    assert resolve_sec_workers() == 1
    monkeypatch.setenv("OMNIQEC_SEC_WORKERS", "31")
    assert resolve_sec_workers() == min(31, __import__("os").cpu_count() or 1)
    monkeypatch.setenv("OMNIQEC_SEC_WORKERS", "garbage")
    assert resolve_sec_workers() == 1
    assert resolve_sec_workers(0) == 1
    assert resolve_sec_workers(2) == 2
