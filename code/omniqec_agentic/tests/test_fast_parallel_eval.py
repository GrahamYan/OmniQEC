"""ArchiveMemory.evaluate_batch (2026-09-11): the fast loop's (program, seed)
evaluations run in a process pool and must leave the archive EXACTLY as the
sequential loop does -- same attempts, same duplicate owners, same MAP cells,
same structural ledger -- because the protocol freezes the fast loop's
verdicts, not its wall clock."""

from __future__ import annotations

import os

import pytest

from omniqec_agentic.construction.groups.catalog import (
    get_group_order_budget,
    set_group_order_budget,
)
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.memory.archive import fast_workers_from_env
from omniqec_agentic.schemas import CandidateProgram

DETERMINISTIC = (
    "def construct(rng):\n"
    "    return {'family': 'bb', 'ell': 3, 'm': 5,\n"
    "            'A': [[1, 0], [0, 1], [2, 2]], 'B': [[0, 2], [2, 0], [1, 1]]}\n"
)
RANDOMISED = (
    "def construct(rng):\n"
    "    variants = [\n"
    "        ([[1, 0], [0, 1], [2, 2]], [[0, 2], [2, 0], [1, 1]]),\n"
    "        ([[1, 0], [0, 2], [2, 1]], [[0, 1], [2, 0], [1, 3]]),\n"
    "        ([[0, 0], [1, 1], [2, 3]], [[1, 0], [0, 2], [2, 4]]),\n"
    "    ]\n"
    "    A, B = variants[int(rng.integers(0, 3))]\n"
    "    return {'family': 'bb', 'ell': 3, 'm': 5, 'A': A, 'B': B}\n"
)
BROKEN = "def construct(rng):\n    return undefined_name\n"
TOO_BIG_GROUP = (
    "def construct(rng):\n"
    "    return {'family': 'lp_nonabelian', 'group': 'C7:C3(2)',\n"
    "            'A': [['e']], 'B': [['e']]}\n"
)
N_BUDGET = 60


def _batch(seeds: int = 5) -> list[CandidateProgram]:
    out: list[CandidateProgram] = []
    for index, source in enumerate((RANDOMISED, DETERMINISTIC, BROKEN, TOO_BIG_GROUP, RANDOMISED), start=1):
        for seed in range(seeds):
            out.append(
                CandidateProgram(
                    program_id=f"p{index:02d}" + ("" if seed == 0 else f"_s{seed:02d}"),
                    source=source,
                    rng_seed=seed,
                    n_budget=N_BUDGET,
                )
            )
    return out


def _snapshot(memory: ArchiveMemory) -> dict:
    return {
        "attempts": memory.attempts,
        "owners": dict(memory.code_hash_owner),
        "cells": {cell: rec.program.program_id for cell, rec in memory.records_by_cell.items()},
        "q_eff": {cell: memory._record_q_eff(rec) for cell, rec in memory.records_by_cell.items()},
        "ledger": sorted(memory.structural_ledger),
    }


@pytest.fixture
def budget():
    previous = get_group_order_budget()
    set_group_order_budget(N_BUDGET)
    yield
    set_group_order_budget(previous)


def test_parallel_batch_is_bit_identical_to_the_sequential_loop(budget) -> None:
    sequential = ArchiveMemory(turn=0)
    seq_records = sequential.evaluate_batch(_batch(), n_budget=N_BUDGET, proxy_iters=8, workers=1)

    parallel = ArchiveMemory(turn=0)
    seen: list[str] = []
    par_records = parallel.evaluate_batch(
        _batch(), n_budget=N_BUDGET, proxy_iters=8, workers=3, on_record=lambda r: seen.append(r.program.program_id)
    )

    assert [r.program.program_id for r in seq_records] == [r.program.program_id for r in par_records]
    assert seen == [r.program.program_id for r in par_records]  # callback in batch order
    assert _snapshot(parallel) == _snapshot(sequential)
    statuses = {a["archive_status"] for a in sequential.attempts}
    # The batch exercises every branch the sequential loop has.
    assert "duplicate" in statuses and "not_accepted" in statuses
    assert any("undefined_name" in a["error"] for a in sequential.attempts)
    assert any("exceeds the budget cap" in a["error"] for a in sequential.attempts)
    # ...and the proxy actually ran for the non-duplicate constructions.
    assert any(a["proxy_valid"] for a in sequential.attempts), statuses


def test_parallel_batch_respects_hashes_already_owned_by_the_archive(budget) -> None:
    """A code the archive already holds is a duplicate on its first appearance
    in the batch, in both modes."""

    warm_seq = ArchiveMemory(turn=0)
    warm_seq.evaluate_batch(_batch(seeds=1)[:1], n_budget=N_BUDGET, proxy_iters=8, workers=1)
    warm_par = ArchiveMemory(turn=0)
    warm_par.evaluate_batch(_batch(seeds=1)[:1], n_budget=N_BUDGET, proxy_iters=8, workers=1)
    warm_seq.evaluate_batch(_batch(seeds=3), n_budget=N_BUDGET, proxy_iters=8, workers=1)
    warm_par.evaluate_batch(_batch(seeds=3), n_budget=N_BUDGET, proxy_iters=8, workers=2)
    assert _snapshot(warm_par) == _snapshot(warm_seq)
    assert warm_seq.attempts[1]["archive_status"] == "duplicate"


def test_fast_workers_from_env(monkeypatch) -> None:
    monkeypatch.delenv("OMNIQEC_FAST_WORKERS", raising=False)
    assert fast_workers_from_env() == 1
    monkeypatch.setenv("OMNIQEC_FAST_WORKERS", "3")
    assert fast_workers_from_env() == min(3, os.cpu_count() or 1)
    monkeypatch.setenv("OMNIQEC_FAST_WORKERS", "garbage")
    assert fast_workers_from_env() == 1
    monkeypatch.setenv("OMNIQEC_FAST_WORKERS", "0")
    assert fast_workers_from_env() == 1
