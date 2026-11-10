"""validate_fig2_campaign (2026-10-01): two early-stop switches recorded at
the same turn, and a decode_infeasible candidate whose second basis never ran.
Both made the validator fail finished DeepSeek b288/b320/b360 campaigns."""

from __future__ import annotations

import json

import pytest

from omniqec_agentic.hpc.validate_fig2_campaign import (
    FORMAL_SEC_POLICY,
    _exact_racing_switches,
    _validate_candidate_schedules,
)


def _switch(turn: int, before: str | None, after: str) -> dict:
    return {
        "turn": turn,
        "applies_from_turn": turn,
        "from": {"racing": {"policy_id": before}} if before else {},
        "to": {"racing": {"policy_id": after}},
    }


def test_two_switches_at_one_turn_sort_by_turn_and_keep_record_order() -> None:
    entries = [
        _switch(0, "pilot10x_maxerr100_v2", "pilot10x_maxerr100_cost_v4"),
        _switch(0, "pilot10x_maxerr100_cost_v4", "pilot10x_maxerr100_v3"),
        _switch(9, "pilot10x_maxerr100_v3", "pilot10x_maxerr30_v3"),
    ]
    switches = _exact_racing_switches(entries)
    assert [s[0] for s in switches] == [0, 0, 9]
    assert [s[2]["policy_id"] for s in switches] == [
        "pilot10x_maxerr100_cost_v4", "pilot10x_maxerr100_v3", "pilot10x_maxerr30_v3",
    ]
    # out-of-order recording still sorts by turn
    assert [s[0] for s in _exact_racing_switches(list(reversed(entries)))] == [0, 0, 9]


def _candidate(tmp_path, bases: dict) -> None:
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    (cdir / "c.json").write_text(json.dumps({"results": [{"status": "ok", "basis": bases}]}), encoding="utf-8")
    return cdir


def test_a_basis_that_never_ran_is_not_audited(tmp_path) -> None:
    cdir = _candidate(tmp_path, {"Z": {"status": "decoder_error", "shots": 0}})
    _validate_candidate_schedules(cdir, "turn 23: slow candidates")


def test_a_basis_with_shots_but_no_circuit_record_still_fails(tmp_path) -> None:
    cdir = _candidate(tmp_path, {"X": {"status": "partial_shards", "shots": 500}})
    with pytest.raises(ValueError, match="memory_experiment is missing"):
        _validate_candidate_schedules(cdir, "turn 23: slow candidates")


def test_a_recorded_circuit_is_still_held_to_the_frozen_policy(tmp_path) -> None:
    cdir = _candidate(
        tmp_path,
        {"X": {"shots": 500, "memory_experiment": {"schedule": {"sec_policy_id": "not_" + FORMAL_SEC_POLICY}}}},
    )
    with pytest.raises(ValueError, match="sec_policy_id"):
        _validate_candidate_schedules(cdir, "turn 1: slow candidates")
