"""The in-process protocol (n_procs == 1) must not pre-spawn multiprocessing
helper processes: under an init that never reaps (a training Pod whose PID 1 is
``sleep``) each worker's forkserver + resource_tracker stayed as zombies until
pids.max was hit (b60 fatal, 2026-09-06). The start method is an execution
detail, not a protocol field: the snapshot dict is unchanged."""

from __future__ import annotations

import dataclasses

import pytest

from omniqec_agentic.config import TelescopingDecoderConfig

td = pytest.importorskip("telescoping_decoder")


def test_in_process_protocol_uses_the_fork_context_that_spawns_nothing() -> None:
    cfg = TelescopingDecoderConfig().to_telescope_config(base_seed=7)
    assert cfg.s3.n_procs == 1
    assert cfg.s3.mp_start_method == "fork"


def test_pooled_local_benchmarking_keeps_the_package_default() -> None:
    cfg = dataclasses.replace(TelescopingDecoderConfig(), n_procs=2).to_telescope_config(base_seed=7)
    assert cfg.s3.n_procs == 2
    assert cfg.s3.mp_start_method == td.TelescopeConfig().s3.mp_start_method


def test_start_method_is_not_part_of_the_frozen_protocol_dict() -> None:
    snapshot = TelescopingDecoderConfig().to_dict()
    assert "mp_start_method" not in snapshot
    assert "n_procs" in snapshot  # the protocol field that selects the in-process path
