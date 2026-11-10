"""schedule_catastrophic is a terminal state, not an infrastructure failure.

Audit fix #5: the real circuit path RAISES inside the schedule build, so no
decoder result ever exists for such a candidate. Before this fix the flag was
computed from evidence that path could not produce, the candidate never
reached any ledger, and it re-took a slow slot (and could take a FINAL slot)
forever."""

from __future__ import annotations

from pathlib import Path

from omniqec_agentic.config import StructuralSearchConfig
from omniqec_agentic.evaluation.writeback import ranking_rows, slow_queue_status
from omniqec_agentic.verification.circuit import (
    ScheduleCatastrophicError,
    is_schedule_catastrophic_reason,
)
from omniqec_agentic.workflow.graph import _final_entries

CATASTROPHIC_REASON = (
    "schedule_catastrophic: no schedule among the sampled candidates avoids "
    "a single-fault undetected logical (02 §3.3)"
)


def _catastrophic_payload(**overrides) -> dict:
    payload = {
        "program_id": "g0001_p0001",
        "status": "schedule_catastrophic",
        "schedule_catastrophic": True,
        "failure_reason": CATASTROPHIC_REASON,
        "fast": {"family": "bb", "n": 144, "k": 8, "d": 6, "d_search": 6},
        "structural": {"code_hash": "ab" * 16, "n_data": 144, "k": 8},
        "promotion_record": {"selection_channels": ["structural"]},
        "results": [
            {
                "status": "schedule_catastrophic",
                "failure_reason": CATASTROPHIC_REASON,
                "schedule_catastrophic": True,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_the_reason_helper_recognizes_the_token() -> None:
    assert is_schedule_catastrophic_reason(CATASTROPHIC_REASON)
    assert is_schedule_catastrophic_reason(ScheduleCatastrophicError(CATASTROPHIC_REASON))
    assert not is_schedule_catastrophic_reason("decoder_error: OOM")
    assert not is_schedule_catastrophic_reason(None)


def test_ranking_rows_carry_the_candidate_level_verdict(tmp_path) -> None:
    rows = ranking_rows(_catastrophic_payload(), tmp_path / "candidate.json")
    assert rows, "a catastrophic candidate must still produce a row"
    assert all(row["schedule_catastrophic"] for row in rows)
    assert all(row["code_hash"] == "ab" * 16 for row in rows)


def test_a_payload_with_no_results_gets_a_terminal_row(tmp_path) -> None:
    rows = ranking_rows(_catastrophic_payload(results=[]), tmp_path / "candidate.json")
    assert len(rows) == 1
    assert rows[0]["status"] == "schedule_catastrophic"
    assert rows[0]["schedule_catastrophic"] is True
    assert rows[0]["code_hash"] == "ab" * 16
    assert rows[0]["d_search"] == 6


def test_the_flag_wins_even_without_payload_status(tmp_path) -> None:
    """The pool path can carry the verdict on the build outcome only."""

    payload = _catastrophic_payload(status="no_valid_results")
    rows = ranking_rows(payload, tmp_path / "candidate.json")
    assert all(row["schedule_catastrophic"] for row in rows)


def test_an_all_catastrophic_queue_is_a_successful_turn() -> None:
    """The protocol measured what there was to measure: that these codes admit
    no usable schedule. Before the fix this failed the strict turn outright."""

    assert (
        slow_queue_status(
            record_count=4,
            timeout_candidate_count=0,
            decoder_attempt_candidate_count=0,
            ok_result_count=0,
            catastrophic_candidate_count=4,
        )
        == "ok"
    )


def test_catastrophic_candidates_do_not_mask_real_failures() -> None:
    assert (
        slow_queue_status(
            record_count=4,
            timeout_candidate_count=2,
            decoder_attempt_candidate_count=0,
            ok_result_count=0,
            catastrophic_candidate_count=2,
        )
        == "measurement_blocked"
    )
    assert (
        slow_queue_status(
            record_count=4,
            timeout_candidate_count=0,
            decoder_attempt_candidate_count=2,
            ok_result_count=0,
            catastrophic_candidate_count=2,
        )
        == "no_valid_results"
    )
    assert (
        slow_queue_status(
            record_count=0,
            timeout_candidate_count=0,
            decoder_attempt_candidate_count=0,
            ok_result_count=0,
            catastrophic_candidate_count=0,
        )
        == "empty_queue"
    )


def _finalist(code: str, k: int, d: int) -> dict:
    return {
        "code_hash": code * 16,
        "program_id": f"cand_{code}",
        "k": k,
        "d_search": d,
        "spec": {"family": "bb"},
        "construct_source": "def construct(rng): return {}",
    }


def test_unmeasurable_candidates_never_take_a_final_slot() -> None:
    """Audit fix #5: the final shortlist used to ignore unmeasurable_hashes, so
    a catastrophic structural elite consumed one of the 3+1 frozen finalist
    slots — and its fixed-T measurement cannot exist."""

    catastrophic = _finalist("aa", 20, 4)
    state = {
        "structural_proxy_elites": [
            catastrophic,
            _finalist("bb", 4, 12),
            _finalist("cc", 10, 8),
            _finalist("dd", 8, 9),
        ],
        "hof_summary": [dict(_finalist("ee", 6, 6), per_logical_per_round=1e-5)],
        "unmeasurable_hashes": [catastrophic["code_hash"]],
    }
    entries, composition = _final_entries(state, StructuralSearchConfig())
    hashes = {row["code_hash"] for row in entries}
    assert catastrophic["code_hash"] not in hashes
    assert len(entries) == 4
