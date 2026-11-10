"""max_turns lowered on resume (2026-09-05, user decision: 60 -> 30 turns for
every arm). A running campaign resumes under the smaller bound: the turns it
already ran stay on disk, the strict audit and the validator tolerate them,
and the final stage only considers what the campaign knew by the bound."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from omniqec_agentic.workflow.checkpoint import (
    RESUME_LOWERABLE_KEYS,
    original_max_turns,
    resume_runtime_overrides,
    validate_resume_protocol,
)


def _lowered_protocol(**extra) -> dict:
    return {
        "max_turns": 30,
        "strict_turns": True,
        "runtime_overrides": [{"turn": 41, "from": {"max_turns": 60}, "to": {"max_turns": 30}}],
        **extra,
    }


def test_resume_accepts_a_lower_max_turns_but_not_a_higher_one() -> None:
    assert RESUME_LOWERABLE_KEYS == frozenset({"max_turns"})
    stored = {"protocol_profile": "telescoping_structural_v3_7", "max_turns": 60, "slow_workers": 63}
    lower = dict(stored, max_turns=30)
    validate_resume_protocol(stored, lower)
    assert resume_runtime_overrides(stored, lower) == {"max_turns": 30}
    with pytest.raises(ValueError, match=r"differing keys: \['max_turns'\]"):
        validate_resume_protocol(stored, dict(stored, max_turns=90))
    # both runtime kinds at once, and a later resume that keeps the lowered value
    both = dict(stored, max_turns=30, slow_workers=127)
    validate_resume_protocol(stored, both)
    assert resume_runtime_overrides(stored, both) == {"slow_workers": 127, "max_turns": 30}
    lowered_once = dict(
        stored, max_turns=30, slow_workers=127,
        runtime_overrides=[{"turn": 41, "from": {"max_turns": 60, "slow_workers": 63}, "to": {"max_turns": 30, "slow_workers": 127}}],
    )
    validate_resume_protocol(lowered_once, dict(stored, max_turns=30, slow_workers=127))
    assert resume_runtime_overrides(lowered_once, dict(stored, max_turns=30, slow_workers=127)) == {}
    assert original_max_turns(lowered_once) == 60
    assert original_max_turns(stored) is None
    assert original_max_turns({"runtime_overrides": [{"from": {"slow_workers": 63}, "to": {"slow_workers": 127}}]}) is None


def _audit(turn_dir: Path, turn: int) -> None:
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "turn_audit.json").write_text(json.dumps({
        "schema": "campaign_turn_audit_v1", "turn": turn, "status": "complete",
        "checks": {"plan_policy": "agentic_v1"},
    }), encoding="utf-8")


def test_strict_audit_tolerates_turns_run_before_the_lowering(tmp_path: Path) -> None:
    import omniqec_agentic.workflow.graph as graph

    for turn in range(34):        # ran to turn 33 under max_turns=60, then lowered to 30
        _audit(tmp_path / "turns" / f"turn_{turn:04d}", turn)
    state = {"run_dir": str(tmp_path), "protocol": _lowered_protocol()}
    summary = graph._validate_all_turn_audits(state)
    assert summary["completed"] == 30 and summary["turns"] == list(range(30))
    assert summary["beyond_max_turns"] == [f"turn_{t:04d}" for t in range(30, 34)]
    # without the lowering record the extra directories are still an error
    state["protocol"] = {"max_turns": 30, "strict_turns": True}
    with pytest.raises(RuntimeError, match="unexpected turn directories"):
        graph._validate_all_turn_audits(state)
    # a directory beyond the ORIGINAL bound is never tolerated
    _audit(tmp_path / "turns" / "turn_0060", 60)
    state["protocol"] = _lowered_protocol()
    with pytest.raises(RuntimeError, match="unexpected turn directories"):
        graph._validate_all_turn_audits(state)


def _slow_results(turn_dir: Path, rows: list[tuple[str, str]]) -> None:
    slow = turn_dir / "cycle" / "slow"
    slow.mkdir(parents=True, exist_ok=True)
    with (slow / "slow_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["program_id", "code_hash", "status"])
        writer.writeheader()
        for code_hash, status in rows:
            writer.writerow({"program_id": "p_" + code_hash, "code_hash": code_hash, "status": status})


def test_final_stage_ignores_codes_first_seen_after_the_lowered_bound(tmp_path: Path, monkeypatch) -> None:
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic.config import StructuralSearchConfig

    _slow_results(tmp_path / "turns" / "turn_0002", [("early_measured", "ok"), ("late_ok_only_later", "decode_infeasible")])
    _slow_results(tmp_path / "turns" / "turn_0031", [("late_measured", "ok"), ("late_ok_only_later", "ok")])
    spec = {"family": "bb"}
    hof = [
        {"code_hash": "late_measured", "spec": spec, "per_logical_per_round": 1e-4, "last_measured_turn": 31},
        {"code_hash": "early_measured", "spec": spec, "per_logical_per_round": 2e-4, "last_measured_turn": 2},
        {"code_hash": "late_ok_only_later", "spec": spec, "per_logical_per_round": 3e-4, "last_measured_turn": 31},
    ]
    elites = [
        {"code_hash": "elite_late", "spec": spec, "first_seen_turn": 30, "d_search": 8, "k": 4, "n": 60},
        {"code_hash": "elite_early", "spec": spec, "first_seen_turn": 4, "d_search": 8, "k": 4, "n": 60},
    ]
    from omniqec_agentic import ranking

    monkeypatch.setattr(ranking, "select_structural_slots", lambda rows, slots: list(rows)[:slots])
    state = {
        "run_dir": str(tmp_path), "protocol": _lowered_protocol(),
        "hof_summary": hof, "structural_proxy_elites": elites,
        "unmeasurable_hashes": [], "baseline_summary": {},
    }
    entries, _ = graph._final_entries(state, StructuralSearchConfig())
    chosen = {str(e["code_hash"]) for e in entries}
    assert "late_measured" not in chosen and "late_ok_only_later" not in chosen and "elite_late" not in chosen
    assert "early_measured" in chosen and "elite_early" in chosen
    # never lowered: nothing is filtered
    state["protocol"] = {"max_turns": 60, "strict_turns": True}
    entries, _ = graph._final_entries(state, StructuralSearchConfig())
    assert "late_measured" in {str(e["code_hash"]) for e in entries}


def test_first_measured_turns_reads_ok_rows_below_the_bound(tmp_path: Path) -> None:
    import omniqec_agentic.workflow.graph as graph

    _slow_results(tmp_path / "turns" / "turn_0000", [("a", "ok"), ("b", "decode_infeasible")])
    _slow_results(tmp_path / "turns" / "turn_0001", [("b", "ok"), ("a", "ok")])
    _slow_results(tmp_path / "turns" / "turn_0005", [("c", "ok")])
    assert graph._first_measured_turns(tmp_path, upto=5) == {"a": 0, "b": 1}
    assert graph._first_measured_turns(tmp_path, upto=6) == {"a": 0, "b": 1, "c": 5}
    assert graph._first_measured_turns(tmp_path, upto=0) == {}


def test_validator_accepts_a_point_that_ran_past_the_lowered_bound(tmp_path: Path) -> None:
    from omniqec_agentic.hpc.validate_fig2_campaign import validate
    from omniqec_agentic.tests.test_campaign_strict_turns import _strict_report, _write_validated_run

    report = _strict_report()                      # a 1-turn strict campaign fixture
    report["turns_completed"] = 3                  # ...that ran 3 turns before max_turns was lowered to 1
    report["protocol"]["runtime_overrides"] = [{"turn": 3, "from": {"max_turns": 3}, "to": {"max_turns": 1}}]
    _write_validated_run(tmp_path, report)
    for turn in (1, 2):
        (tmp_path / "turns" / f"turn_{turn:04d}").mkdir(parents=True)
    assert validate(tmp_path, turns=1, p_value=0.005, model="deepseek-v4-pro") == 1
    # more turns than the original bound, or no lowering record: refused
    (tmp_path / "turns" / "turn_0003").mkdir()
    with pytest.raises(ValueError, match="turn directory set is not exactly 0..0"):
        validate(tmp_path, turns=1, p_value=0.005, model="deepseek-v4-pro")
    (tmp_path / "turns" / "turn_0003").rmdir()
    report["protocol"].pop("runtime_overrides")
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="incomplete campaign: turns=3/1"):
        validate(tmp_path, turns=1, p_value=0.005, model="deepseek-v4-pro")
