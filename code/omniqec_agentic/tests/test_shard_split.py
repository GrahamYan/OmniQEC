"""Shard splitting (2026-09-05): a decode shard's shot stream may be decoded
as several contiguous shot-range jobs on different workers, folded back into
the one shard cell the protocol records. Every number must be identical to
the unsplit shard; only the wall clock changes."""

from __future__ import annotations

import json

import numpy as np
import pytest
import stim

from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.sampling import merge_subshard_results, shard_partition
from omniqec_agentic.verification.decoders import (
    estimate_circuit_logical_error_rate,
    normalize_shot_range,
)
from omniqec_agentic.workflow.checkpoint import (
    RESUME_RUNTIME_KEYS,
    resume_runtime_overrides,
    validate_resume_protocol,
)

_REP_CIRCUIT = stim.Circuit.generated(
    "repetition_code:memory", distance=5, rounds=4,
    before_round_data_depolarization=0.05, before_measure_flip_probability=0.02,
)


class _RecordingDecoder:
    """Deterministic per-shot verdicts keyed by (shot index, syndrome): shot
    ``i`` is 'right' iff hash(i, syndrome) is even; records every (index,
    syndrome) it decoded so a split run can be compared with the full run."""

    decoder_name = "recorder"

    def __init__(self, num_observables: int):
        self.matrices = type("M", (), {
            "num_observables": num_observables,
            "to_metadata": staticmethod(lambda: {"kind": "recorder"}),
        })()
        self._shot_offset = 0
        self.seen: list[tuple[int, bytes]] = []
        self.last_stages: list[str] = []
        self.last_declared_failures = np.zeros(0, dtype=bool)
        self.stage_counts: dict[str, int] = {"S3A": 0, "NC": 0}
        self.stage_time_s: dict[str, float] = {}

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        out = np.zeros((syndromes.shape[0], self.matrices.num_observables), dtype=np.uint8)
        stages = []
        declared = np.zeros(syndromes.shape[0], dtype=bool)
        for row, syndrome in enumerate(syndromes):
            index = int(self._shot_offset) + row
            key = (index, syndrome.tobytes())
            self.seen.append(key)
            h = hash(key) & 0xFFFF
            if h % 7 == 0:          # a "declared" NC failure, like the telescoping NC verdict
                declared[row] = True
                stages.append("NC")
            else:
                stages.append("S3A")
                out[row, :] = (h >> 1) & 1   # a wrong observable half the time
        self._shot_offset += int(syndromes.shape[0])
        self.last_stages = stages
        self.last_declared_failures = declared
        return out

    def stage_metrics(self):
        return {}


def _run(circuit: stim.Circuit, shots: int, seed: int, batch: int, shot_range=None):
    dec = _RecordingDecoder(circuit.num_observables)
    res = estimate_circuit_logical_error_rate(
        dec, circuit_text=str(circuit), shots=shots, seed=seed, batch_size=batch, shot_range=shot_range,
    )
    return res, dec


def test_normalize_shot_range_bounds() -> None:
    assert normalize_shot_range(None, 10) == (0, 10)
    assert normalize_shot_range((3, 7), 10) == (3, 7)
    assert normalize_shot_range([0, 10], 10) == (0, 10)
    for bad in ((-1, 3), (4, 3), (3, 11)):
        with pytest.raises(ValueError, match="shot_range"):
            normalize_shot_range(bad, 10)


@pytest.mark.parametrize("batch", [1, 7, 16, 1024])
@pytest.mark.parametrize("cuts", [[0, 41], [0, 5, 41], [0, 16, 32, 41], [0, 1, 2, 40, 41]])
def test_shot_ranges_reproduce_the_unsplit_shard_shot_for_shot(batch, cuts) -> None:
    """The union of the parts sees exactly the (index, syndrome) pairs of the
    unsplit run — whatever the batch size and however the range cuts fall
    relative to the batches — and the failure counts sum to the full count."""

    shots, seed = 41, 12345
    full, dec_full = _run(_REP_CIRCUIT, shots, seed, batch)
    assert full.status == "ok" and full.shots == shots
    assert dec_full.seen[0][0] == 0 and dec_full.seen[-1][0] == shots - 1
    seen_parts: list[tuple[int, bytes]] = []
    failures = 0
    for start, stop in zip(cuts, cuts[1:]):
        part, dec_part = _run(_REP_CIRCUIT, shots, seed, batch, shot_range=(start, stop))
        assert part.status == "ok", part.failure_reason
        assert part.shots == stop - start
        assert part.metrics["shot_range"] == [start, stop] and part.metrics["shard_shots"] == shots
        assert part.metrics["requested_shots"] == stop - start and part.metrics["sampling_complete"]
        assert [index for index, _ in dec_part.seen] == list(range(start, stop))
        seen_parts.extend(dec_part.seen)
        failures += part.failures
    assert seen_parts == dec_full.seen
    assert failures == full.failures
    # a shot's syndrome is the same whichever part decodes it: no re-seeding drift
    assert {i: s for i, s in seen_parts} == {i: s for i, s in dec_full.seen}


def test_unsplit_call_is_unchanged_and_stage_metrics_sum_across_parts() -> None:
    shots, seed = 30, 7
    full, _ = _run(_REP_CIRCUIT, shots, seed, 8)
    assert "shot_range" not in full.metrics and full.metrics["requested_shots"] == shots
    a, _ = _run(_REP_CIRCUIT, shots, seed, 8, shot_range=(0, 11))
    b, _ = _run(_REP_CIRCUIT, shots, seed, 8, shot_range=(11, 30))
    for key in ("stage_counts", "stage_failures"):
        summed = {k: a.metrics[key].get(k, 0) + b.metrics[key].get(k, 0) for k in full.metrics[key]}
        assert summed == full.metrics[key]
    assert a.metrics["declared_failures"] + b.metrics["declared_failures"] == full.metrics["declared_failures"]


def _cell(shots, failures, *, status="ok", runtime=1.0, rng=None, rss=100, attempts=1, nc=0):
    cell = {
        "status": status, "decoder": "telescoping_cpu", "shots": shots, "failures": failures,
        "logical_error_rate": (failures / shots) if shots else None, "runtime_s": runtime,
        "rounds": 12, "decode_stage": "search", "memory_experiment": {"x": 1},
        "job_execution": {"attempts": attempts, "ru_maxrss": rss, "returncode": 0},
        "metrics": {
            "requested_shots": shots, "actual_shots": shots, "sampling_complete": True,
            "stopped_by_max_errors": False, "shard_shots": 100,
            "stage_counts": {"S3A": shots - nc, "NC": nc}, "stage_failures": {"S3A": failures - nc, "NC": nc},
            "stage_time_s": {"S3A": runtime * 0.5, "NC": runtime * 0.5}, "declared_failures": nc,
        },
    }
    if rng is not None:
        cell["shot_range"] = list(rng)
        cell["metrics"]["shot_range"] = list(rng)
    if status != "ok":
        cell["failure_reason"] = f"{status} reason"
    return cell


def test_merge_subshard_results_folds_ok_parts_into_one_shard_cell() -> None:
    parts = [
        _cell(40, 3, rng=(0, 40), runtime=2.0, rss=300, nc=1),
        _cell(30, 2, rng=(40, 70), runtime=1.5, rss=900, attempts=2, nc=0),
        _cell(30, 5, rng=(70, 100), runtime=4.0, rss=500, nc=2),
    ]
    merged = merge_subshard_results(parts, decoder="telescoping_cpu")
    assert merged["status"] == "ok"
    assert merged["shots"] == 100 and merged["failures"] == 10
    assert merged["logical_error_rate"] == pytest.approx(0.1)
    assert merged["runtime_s"] == pytest.approx(7.5)
    assert merged["parts"] == 3 and merged["part_shot_ranges"] == [[0, 40], [40, 70], [70, 100]]
    assert merged["part_runtime_s"] == [2.0, 1.5, 4.0] and merged["part_statuses"] == ["ok"] * 3
    assert "shot_range" not in merged and "shot_range" not in merged["metrics"]
    m = merged["metrics"]
    assert m["requested_shots"] == 100 and m["actual_shots"] == 100 and m["sampling_complete"] is True
    assert "shard_shots" not in m
    assert m["stage_counts"] == {"S3A": 97, "NC": 3} and m["stage_failures"] == {"S3A": 7, "NC": 3}
    assert m["stage_time_s"] == {"S3A": 3.75, "NC": 3.75} and m["declared_failures"] == 3
    assert merged["job_execution"]["ru_maxrss"] == 900 and merged["job_execution"]["attempts"] == 2
    assert merged["job_execution"]["parts"] == 3
    assert merged["decode_stage"] == "search" and merged["rounds"] == 12
    # a single part is the shard cell itself
    assert merge_subshard_results([parts[0]], decoder="telescoping_cpu") is parts[0]
    assert merge_subshard_results([None, None], decoder="telescoping_cpu") is None


def test_merge_subshard_results_keeps_the_shard_all_or_nothing() -> None:
    # a watchdog timeout in ONE part is the shard's status: exactly what an
    # unsplit shard would report, so decode_infeasible is decided the same way
    parts = [_cell(50, 1, rng=(0, 50)), _cell(0, 0, status="timeout", rng=(50, 100), runtime=1800.0)]
    merged = merge_subshard_results(parts, decoder="telescoping_cpu")
    assert merged["status"] == "timeout" and "timeout reason" in merged["failure_reason"]
    assert merged["shots"] == 50 and merged["failures"] == 1
    assert merged["part_statuses"] == ["ok", "timeout"]
    assert merged["metrics"]["sampling_complete"] is False
    # a part that never returned (None) is a timeout too
    merged = merge_subshard_results([_cell(50, 1, rng=(0, 50)), None], decoder="telescoping_cpu")
    assert merged["status"] == "timeout" and "never returned" in merged["failure_reason"]
    assert merged["part_statuses"] == ["ok", "timeout"]
    # a worker crash names its own status
    merged = merge_subshard_results(
        [_cell(0, 0, status="subprocess_error", rng=(0, 50)), _cell(50, 2, rng=(50, 100))], decoder="telescoping_cpu"
    )
    assert merged["status"] == "subprocess_error" and merged["shots"] == 50


def _seed_bb_spec() -> dict:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    return run_construct_source(SEED_BB, program_id="shard_split_bb", seed=0).constructed.spec


def _fake_worker_factory(seen: list[dict], *, timeout_part: tuple[str, int, int] | None = None):
    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        unit = request["units"][0]
        seen.append({
            "basis": unit["basis"], "shard": int(request["shard"]), "shots": int(request["shots"]),
            "shot_range": request.get("shot_range"), "timeout_s": timeout_s,
        })
        rng = request.get("shot_range")
        if timeout_part is not None and (unit["basis"], int(request["shard"]), (rng or [0])[0]) == timeout_part:
            return None, {"timeout": True, "timeout_s": timeout_s}
        n = int(request["shots"]) if rng is None else int(rng[1]) - int(rng[0])
        # one failure per decoded shot index divisible by 97: a pure function of the index
        start = 0 if rng is None else int(rng[0])
        failures = sum(1 for i in range(start, start + n) if i % 97 == 0)
        cell = {
            "status": "ok", "decoder": unit["decoder"], "shots": n, "failures": failures,
            "logical_error_rate": failures / n, "runtime_s": 0.01,
            "metrics": {"requested_shots": n, "actual_shots": n, "sampling_complete": True,
                        "stopped_by_max_errors": False,
                        **({"shot_range": list(rng), "shard_shots": int(request["shots"])} if rng else {})},
        }
        if rng is not None:
            cell["shot_range"] = list(rng)
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, {"timeout": False}
    return fake_worker


def _evaluate(monkeypatch, seen, **kwargs):
    monkeypatch.setattr(pool_mod, "run_worker_request", _fake_worker_factory(seen, **kwargs))
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_split", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(shots=2_000, max_errors=None, rounds=12),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=2,
    )[0]


def test_pool_splits_shards_into_ranges_and_folds_them_back(monkeypatch) -> None:
    monkeypatch.delenv("OMNIQEC_SHARD_SPLIT", raising=False)
    seen_unsplit: list[dict] = []
    unsplit = _evaluate(monkeypatch, seen_unsplit)
    assert unsplit["status"] == "ok" and unsplit["job_config"]["shard_split"] == 1
    assert all(job["shot_range"] is None for job in seen_unsplit) and len(seen_unsplit) == 4

    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "3")
    seen: list[dict] = []
    payload = _evaluate(monkeypatch, seen)
    assert payload["status"] == "ok" and payload["job_config"]["shard_split"] == 3
    # 2 bases x 2 shards x 3 parts, contiguous ranges covering each 1000-shot shard,
    # every part carrying the FULL shard's shots (-> the full shard watchdog)
    assert len(seen) == 12
    for basis in ("X", "Z"):
        for shard in (0, 1):
            ranges = sorted(tuple(j["shot_range"]) for j in seen if j["basis"] == basis and j["shard"] == shard)
            assert ranges == [(0, 334), (334, 667), (667, 1000)]
            assert {j["shots"] for j in seen if j["basis"] == basis and j["shard"] == shard} == {1000}
    assert {j["timeout_s"] for j in seen} == {j["timeout_s"] for j in seen_unsplit}
    for basis in ("X", "Z"):
        split_cell = payload["results"][0]["basis"][basis]
        full_cell = unsplit["results"][0]["basis"][basis]
        assert split_cell["shots"] == full_cell["shots"] == 2000
        assert split_cell["failures"] == full_cell["failures"]
        assert split_cell["shard_count"] == 2 and split_cell["shard_statuses"] == ["ok", "ok"]
        assert split_cell["metrics"]["requested_shots"] == 2000 and split_cell["metrics"]["sampling_complete"]
    assert payload["results"][0]["measurement_complete"] is True
    assert payload["results"][0]["combined"]["per_logical_per_round"] == unsplit["results"][0]["combined"]["per_logical_per_round"]
    json.dumps(payload, sort_keys=True)


def test_pool_split_part_timeout_is_the_shard_timeout(monkeypatch) -> None:
    """A pure watchdog timeout in one part of one Z shard: the shard is a
    timeout, the basis is incomplete and the candidate is decode_infeasible
    exactly as with an unsplit timed-out shard (timeout_retries=0)."""

    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "2")
    seen: list[dict] = []
    payload = _evaluate(monkeypatch, seen, timeout_part=("Z", 1, 500))
    assert payload["status"] == "decode_infeasible" and payload["decode_infeasible"] is True
    z = payload["results"][0]["basis"]["Z"]
    assert z["shard_statuses"] == ["ok", "timeout"]
    # the basis merge counts ok shards only (as before): the timed-out shard's
    # ok half stays in its shard cell for diagnostics, not in the basis total
    assert z["shots"] == 1000
    assert z["shard_runtime_s"][1] is not None
    assert payload["results"][0]["basis"]["X"]["shard_statuses"] == ["ok", "ok"]


def test_shard_split_factor_env() -> None:
    import os

    old = os.environ.pop("OMNIQEC_SHARD_SPLIT", None)
    try:
        assert pool_mod.shard_split_factor() == 1
        os.environ["OMNIQEC_SHARD_SPLIT"] = "8"
        assert pool_mod.shard_split_factor() == 8
        os.environ["OMNIQEC_SHARD_SPLIT"] = " "
        assert pool_mod.shard_split_factor() == 1
        for bad in ("0", "-2", "x"):
            os.environ["OMNIQEC_SHARD_SPLIT"] = bad
            with pytest.raises(ValueError, match="OMNIQEC_SHARD_SPLIT"):
                pool_mod.shard_split_factor()
    finally:
        os.environ.pop("OMNIQEC_SHARD_SPLIT", None)
        if old is not None:
            os.environ["OMNIQEC_SHARD_SPLIT"] = old


def test_shard_partition_parts_cover_the_shard() -> None:
    for shots in (1, 2, 3, 100, 1250):
        for split in (1, 2, 3, 4, 8):
            parts = [n for n in shard_partition(shots, split) if n > 0]
            assert sum(parts) == shots and len(parts) == min(split, shots)


def test_resume_accepts_a_different_worker_count_and_nothing_else() -> None:
    # slow_workers was the only runtime key until 2026-09-24; the shard-cap
    # ceilings (09-24/25) and the promotion weight gate (09-29) joined it.
    assert RESUME_RUNTIME_KEYS == frozenset({
        "slow_workers", "final_decode_shard_timeout_cap_s", "search_decode_shard_timeout_cap_s",
        "promotion_max_check_weight",
    })
    stored = {"protocol_profile": "telescoping_structural_v3_7", "slow_workers": 63, "shot_shards": 16, "max_turns": 60}
    current = dict(stored, slow_workers=127)
    validate_resume_protocol(stored, current)            # was: differing keys: ['slow_workers']
    assert resume_runtime_overrides(stored, current) == {"slow_workers": 127}
    assert resume_runtime_overrides(stored, dict(stored)) == {}
    with pytest.raises(ValueError, match=r"differing keys: \['shot_shards'\]"):
        validate_resume_protocol(stored, dict(current, shot_shards=8))
    # max_turns may be LOWERED (test_turn_lowering.py); raising it is still a mismatch
    with pytest.raises(ValueError, match=r"differing keys: \['max_turns'\]"):
        validate_resume_protocol(stored, dict(stored, max_turns=90))
    # a checkpoint that already carries an override history resumes again
    stored_again = dict(stored, slow_workers=127, runtime_overrides=[{"turn": 9, "from": {"slow_workers": 63}, "to": {"slow_workers": 127}}])
    validate_resume_protocol(stored_again, dict(stored, slow_workers=127))
    validate_resume_protocol(stored_again, dict(stored, slow_workers=63))
    assert resume_runtime_overrides(stored_again, dict(stored, slow_workers=63)) == {"slow_workers": 63}


# ---------------------------------------------------------------------------
# 2026-09-11: shard splitting under racing (early stop). A 250-shot pilot
# shard of a 3 s/shot code kept one worker busy for 13 minutes while the rest
# of the node idled, because racing ran unsplit shards. Split shards are
# folded the moment their last part lands and only then count as landed for
# the race, so every decision, every recorded cell and every stage is the
# same as with unsplit shards.


def _evaluate_racing(monkeypatch, seen, *, racing, thresholds=None, **kwargs):
    monkeypatch.setattr(pool_mod, "run_worker_request", _fake_worker_factory(seen, **kwargs))
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_split", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(shots=2_000, max_errors=None, rounds=12),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=2,
        racing=racing, racing_thresholds=thresholds,
    )[0]


def _racing_cfg(**overrides):
    from omniqec_agentic.config import RacingConfig

    base = dict(enabled=True, shots_pilot=1_000, batch=500, shots_confirm=2_000, shots_max=2_000)
    base.update(overrides)
    return RacingConfig(**base)


def _same_measurement(split: dict, unsplit: dict) -> None:
    for basis in ("X", "Z"):
        a = split["results"][0]["basis"][basis]
        b = unsplit["results"][0]["basis"][basis]
        assert (a["shots"], a["failures"], a["shard_count"], a["shard_statuses"]) == (
            b["shots"], b["failures"], b["shard_count"], b["shard_statuses"]
        ), basis
    assert split["results"][0]["combined"]["per_logical_per_round"] == unsplit["results"][0]["combined"]["per_logical_per_round"]
    race_a, race_b = dict(split["racing"]), dict(unsplit["racing"])
    assert race_a == race_b


def test_racing_pool_splits_shards_and_decides_bit_identically(monkeypatch) -> None:
    racing = _racing_cfg()
    monkeypatch.delenv("OMNIQEC_SHARD_SPLIT", raising=False)
    seen_unsplit: list[dict] = []
    unsplit = _evaluate_racing(monkeypatch, seen_unsplit, racing=racing)
    assert unsplit["job_config"]["shard_split"] == 1
    assert unsplit["racing"]["confirmed"] is True and unsplit["racing"]["stopped_by"] is None
    assert unsplit["racing"]["shots_per_basis"] == 2_000

    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "3")
    seen: list[dict] = []
    split = _evaluate_racing(monkeypatch, seen, racing=racing)
    assert split["job_config"]["shard_split"] == 3
    assert len(seen) == 3 * len(seen_unsplit)  # every racing shard became 3 shot-range jobs
    for basis in ("X", "Z"):
        for shard in range(4):  # 2000 / 500 = 4 pre-registered shards
            ranges = sorted(tuple(j["shot_range"]) for j in seen if j["basis"] == basis and j["shard"] == shard)
            assert ranges == [(0, 167), (167, 334), (334, 500)], (basis, shard)
    _same_measurement(split, unsplit)
    json.dumps(split, sort_keys=True)


def test_racing_pool_split_max_errors_stop_is_identical(monkeypatch) -> None:
    """The 100-error cap (pilot10x_maxerr100_v2) decides on whole-shard
    counts: the split run stops at the same boundary with the same numbers."""

    racing = _racing_cfg(max_errors=5)  # ~5 failures per 500-shot shard in the fake decoder
    monkeypatch.delenv("OMNIQEC_SHARD_SPLIT", raising=False)
    unsplit = _evaluate_racing(monkeypatch, [], racing=racing)
    assert unsplit["racing"]["stopped_by"] == "max_errors" and unsplit["racing"]["confirmed"] is True
    assert unsplit["racing"]["shots_per_basis"] == 1_000  # stopped at the pilot boundary
    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "4")
    split = _evaluate_racing(monkeypatch, [], racing=racing)
    _same_measurement(split, unsplit)


def test_racing_pool_split_part_timeout_matches_a_whole_shard_timeout(monkeypatch) -> None:
    """One timed-out part makes its shard a timeout, exactly like an unsplit
    shard that timed out: same shard statuses, same race outcome, no stall."""

    racing = _racing_cfg()
    monkeypatch.delenv("OMNIQEC_SHARD_SPLIT", raising=False)
    unsplit = _evaluate_racing(monkeypatch, [], racing=racing, timeout_part=("Z", 0, 0))
    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "3")
    split = _evaluate_racing(monkeypatch, [], racing=racing, timeout_part=("Z", 0, 167))
    z = split["results"][0]["basis"]["Z"]
    assert z["shard_statuses"][0] == "timeout"
    _same_measurement(split, unsplit)


# ---------------------------------------------------------------------------
# 2026-09-14 (pilot10x_maxerr100_v3): the error cap is decided at every landed
# shard, the pilot included. A code that fails (nearly) every shot stops after
# its first shard instead of decoding the whole 2000-shot pilot; a code below
# the cap is measured exactly as under v2.


def _all_fail_worker_factory(seen: list[dict], *, fail_rate: float):
    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        unit = request["units"][0]
        shots = int(request["shots"])
        lo, hi = request.get("shot_range") or (0, shots)
        n = int(hi) - int(lo)
        failures = int(round(n * fail_rate))
        seen.append({"basis": unit["basis"], "shard": int(request["shard"]), "shot_range": [int(lo), int(hi)]})
        cell = {
            "status": "ok", "decoder": unit["decoder"], "shots": n, "failures": failures,
            "logical_error_rate": failures / n if n else None, "runtime_s": 0.0,
            "shot_range": [int(lo), int(hi)],
            "metrics": {"requested_shots": n, "shard_shots": shots, "actual_shots": n,
                        "sampling_complete": True, "stopped_by_max_errors": False},
        }
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, {"timeout": False}
    return fake_worker


def _evaluate_v3(monkeypatch, seen, *, racing, fail_rate):
    monkeypatch.setattr(pool_mod, "run_worker_request", _all_fail_worker_factory(seen, fail_rate=fail_rate))
    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "2")
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_v3", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(shots=2_000, max_errors=None, rounds=12),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=2,
        racing=racing, racing_thresholds=None,
    )[0]


def test_v3_stops_a_hopeless_code_at_its_first_shard_and_v2_at_the_pilot(monkeypatch) -> None:
    v2 = _racing_cfg(max_errors=100)                      # pilot 1000 = 2 shards of 500
    v3 = _racing_cfg(max_errors=100, max_errors_in_pilot=True)
    seen_v2: list[dict] = []
    under_v2 = _evaluate_v3(monkeypatch, seen_v2, racing=v2, fail_rate=0.5)   # 250 failures per shard
    assert under_v2["racing"]["stopped_by"] == "max_errors" and under_v2["racing"]["confirmed"] is True
    assert under_v2["racing"]["shots_per_basis"] == 1_000                  # decided at the pilot boundary
    seen_v3: list[dict] = []
    under_v3 = _evaluate_v3(monkeypatch, seen_v3, racing=v3, fail_rate=0.5)
    assert under_v3["racing"]["stopped_by"] == "max_errors" and under_v3["racing"]["confirmed"] is True
    assert under_v3["racing"]["shots_per_basis"] == 500                    # decided after shard 0
    assert under_v3["racing"]["stages"][0]["shards_consumed"] == 1
    for basis in ("X", "Z"):
        cell = under_v3["results"][0]["basis"][basis]
        assert (cell["shots"], cell["failures"]) == (500, 250)
    # nothing was submitted past the pilot before the pilot decision
    assert max(j["shard"] for j in seen_v3) <= 1


def test_v3_measures_a_code_below_the_cap_exactly_like_v2(monkeypatch) -> None:
    v2 = _racing_cfg(max_errors=100)
    v3 = _racing_cfg(max_errors=100, max_errors_in_pilot=True)
    under_v2 = _evaluate_v3(monkeypatch, [], racing=v2, fail_rate=0.01)     # 5 failures per shard: never capped
    under_v3 = _evaluate_v3(monkeypatch, [], racing=v3, fail_rate=0.01)
    for basis in ("X", "Z"):
        a = under_v3["results"][0]["basis"][basis]
        b = under_v2["results"][0]["basis"][basis]
        assert (a["shots"], a["failures"], a["shard_count"], a["shard_statuses"]) == (
            b["shots"], b["failures"], b["shard_count"], b["shard_statuses"]
        )
    assert under_v3["results"][0]["combined"]["per_logical_per_round"] == under_v2["results"][0]["combined"]["per_logical_per_round"]
    # the verdict is the same; v3 merely records a (continue) stage per pilot shard
    race_v3 = {k: v for k, v in under_v3["racing"].items() if k != "stages"}
    race_v2 = {k: v for k, v in under_v2["racing"].items() if k != "stages"}
    assert race_v3 == race_v2
    assert under_v3["racing"]["stopped_by"] is None and under_v3["racing"]["shots_per_basis"] == 2_000
    assert [s["action"] for s in under_v3["racing"]["stages"]][:2] == ["continue", "continue"]
    assert under_v3["racing"]["stages"][-1] == under_v2["racing"]["stages"][-1]


def _costly_worker_factory(seen: list[dict], *, cost_s_per_shot: float, fail_rate: float = 0.01):
    """Like _all_fail_worker_factory but every decoded shot reports a fixed
    CPU cost (runtime_s = shots x cost), so the v4 decoder-cost cap can be
    exercised without sleeping."""

    def fake_worker(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        unit = request["units"][0]
        shots = int(request["shots"])
        lo, hi = request.get("shot_range") or (0, shots)
        n = int(hi) - int(lo)
        failures = int(round(n * fail_rate))
        seen.append({"basis": unit["basis"], "shard": int(request["shard"]), "shot_range": [int(lo), int(hi)]})
        cell = {
            "status": "ok", "decoder": unit["decoder"], "shots": n, "failures": failures,
            "logical_error_rate": failures / n if n else None, "runtime_s": float(n) * float(cost_s_per_shot),
            "shot_range": [int(lo), int(hi)],
            "metrics": {"requested_shots": n, "shard_shots": shots, "actual_shots": n,
                        "sampling_complete": True, "stopped_by_max_errors": False},
        }
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, {"timeout": False}
    return fake_worker


def _evaluate_v4(monkeypatch, seen, *, racing, cost_s_per_shot):
    monkeypatch.setattr(pool_mod, "run_worker_request", _costly_worker_factory(seen, cost_s_per_shot=cost_s_per_shot))
    monkeypatch.setenv("OMNIQEC_SHARD_SPLIT", "2")
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_v4", "spec": _seed_bb_spec()}],
        verification=VerificationConfig(shots=2_000, max_errors=None, rounds=12),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=2,
        racing=racing, racing_thresholds=None,
    )[0]


def test_v4_declares_a_code_above_the_decoder_cost_cap_infeasible_at_its_first_judgeable_shard(monkeypatch) -> None:
    """pilot10x_maxerr100_cost_v4 (2026-09-14): the seed BB code has 30 data
    qubits, so its cap is the 8 s/shot floor. A decoder costing 20 s/shot is
    judged at the first decision with >= 250 shots per basis (the first
    shard under v3's every-shard decisions) and the candidate is a terminal
    decode_infeasible finding: recorded, excluded, never a measurement."""

    from omniqec_agentic.config import formal_racing_v3, formal_racing_v4

    v4 = formal_racing_v4(2000)
    assert v4.cost_cap_s_per_shot_per_qubit == 0.10 and v4.cost_cap_min_shots == 250
    seen: list[dict] = []
    out = _evaluate_v4(monkeypatch, seen, racing=v4, cost_s_per_shot=20.0)
    assert out["status"] == "decode_infeasible" and out["decode_infeasible"] is True
    assert out["racing"]["stopped_by"] == "decoder_cost" and out["racing"]["confirmed"] is False
    assert out["racing"]["shots_per_basis"] == 250
    assert out["decoder_cost"]["cost_s_per_shot"] == 20.0 and out["decoder_cost"]["cost_cap_s_per_shot"] == 8.0
    assert out["decoder_cost"]["n_data"] == 30 and out["decoder_cost"]["shots_per_basis"] == 250
    assert "decoder_cost: 20.0 CPU-s/shot" in out["failure_reason"] and "pilot10x_maxerr100_cost_v4" in out["failure_reason"]
    # no stage carries timing: the recorded decisions stay schedule-independent
    assert all("cost_s_per_shot" not in stage for stage in out["racing"]["stages"])
    assert out["racing"]["stages"][-1]["stopped_by"] == "decoder_cost"
    # v3 measures the same code all the way (no cap)
    under_v3 = _evaluate_v4(monkeypatch, [], racing=formal_racing_v3(2000), cost_s_per_shot=20.0)
    assert under_v3["status"] == "ok" and under_v3["racing"]["stopped_by"] is None
    assert under_v3["racing"]["shots_per_basis"] == 2000


def test_v4_measures_a_code_below_the_cost_cap_exactly_like_v3(monkeypatch) -> None:
    from omniqec_agentic.config import formal_racing_v3, formal_racing_v4

    cheap = 1.0   # 1 s/shot: well under the 8 s/shot floor
    under_v3 = _evaluate_v4(monkeypatch, [], racing=formal_racing_v3(2000), cost_s_per_shot=cheap)
    under_v4 = _evaluate_v4(monkeypatch, [], racing=formal_racing_v4(2000), cost_s_per_shot=cheap)
    assert under_v3["status"] == under_v4["status"] == "ok"
    for key in ("shots_per_basis", "stopped_by", "confirmed", "shards_consumed", "stages"):
        assert under_v3["racing"][key] == under_v4["racing"][key], key
    assert under_v3["results"][0]["combined"] == under_v4["results"][0]["combined"]
    assert "decoder_cost" not in under_v4
