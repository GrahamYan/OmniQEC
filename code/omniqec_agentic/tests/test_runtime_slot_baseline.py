"""The runtime-slot baseline is ONE list: what acquisition_score reports is
what validate_turn_plan checks, and a rejection names the head it expects.

Regression for GPT arm b288 turn 4 (2026-09-04): the tool ranked measured
candidates first while the validator excluded them, so the model promoted
the tool's head, was told it departed from the baseline without being told
which candidate the validator wanted, and resubmitted the same plan through
4 repairs x 4 retries until the point was FATAL_STOPped."""

from __future__ import annotations

import json
from pathlib import Path

from omniqec_agentic import ranking
from omniqec_agentic.orchestrator.plan import validate_turn_plan
from omniqec_agentic.tools.acquisition_score import ACQUISITION_VERSION, acquisition_score
from omniqec_agentic.tools.context import ToolContext

H_MEASURED = "58d440d8c403e0527430c05e" + "0" * 40   # best plpr, already measured
H_HEAD = "a1b2c3d4e5f60718293a4b5c" + "1" * 40       # unmeasured archive head
H_SECOND = "0f1e2d3c4b5a69788796a5b4" + "2" * 40
H_CATASTROPHIC = "deadbeefdeadbeefdeadbeef" + "3" * 40

ARCHIVE = [
    {"code_hash": H_MEASURED, "n": 72, "k": 12, "d_proxy": 6, "Q": 1.0},
    {"code_hash": H_CATASTROPHIC, "n": 90, "k": 8, "d_proxy": 8, "Q": 5.0},
    {"code_hash": H_HEAD, "n": 96, "k": 8, "d_proxy": 8, "Q": 4.0},
    {"code_hash": H_SECOND, "n": 80, "k": 4, "d_proxy": 7, "Q": 3.0},
]
SLOW = [
    {"code_hash": H_MEASURED, "per_logical_per_round": 0.002, "n": 72, "k": 12, "d_proxy": 6},
    {"code_hash": H_CATASTROPHIC, "schedule_catastrophic": True, "n": 90, "k": 8, "d_proxy": 8},
]


def _ctx(tmp_path: Path) -> ToolContext:
    queue = tmp_path / "archive_queue.jsonl"
    queue.write_text("".join(json.dumps(r) + "\n" for r in ARCHIVE), encoding="utf-8")
    rankings = tmp_path / "rankings.json"
    rankings.write_text(json.dumps({"rankings": SLOW}), encoding="utf-8")
    state = {"last_cycle": {"artifact_paths": {"all_archive_queue": str(queue), "slow_rankings": str(rankings)}}}
    return ToolContext(run_dir=tmp_path, manual_dir=tmp_path / "manual", state=state, turn=4)


def test_baseline_drops_measured_and_catastrophic_and_keeps_archive_order() -> None:
    assert ranking.runtime_slot_baseline(ARCHIVE, SLOW) == [H_HEAD, H_SECOND]
    # duplicates and rows without a hash are ignored
    assert ranking.runtime_slot_baseline(ARCHIVE + [{"code_hash": H_HEAD}, {}], SLOW) == [H_HEAD, H_SECOND]
    assert ranking.runtime_slot_baseline([], SLOW) == []


def test_tool_reports_the_validators_list_not_a_resorted_subset(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # The model asks about the measured best and the two unmeasured ones.
    out = acquisition_score(ctx, code_hashes=[H_MEASURED[:12], H_SECOND[:12], H_HEAD[:12]])
    assert out["version"] == ACQUISITION_VERSION == "acquisition_v5_runtime_slot_expected_plpr"
    assert out["baseline_order"] == [H_HEAD[:12], H_SECOND[:12]]
    assert out["baseline_head"] == H_HEAD[:12]
    assert out["excluded_from_baseline"]["measured"] == [H_MEASURED[:12]]
    # scores still describe the measured candidate (with its plpr), just not as the head
    measured = next(s for s in out["scores"] if s["code_hash"] == H_MEASURED[:12])
    assert measured["measured"] is True and measured["per_logical_per_round"] == 0.002
    # and the tool's list IS the validator's list
    assert [h[:12] for h in ranking.runtime_slot_baseline(ctx.archive_rows, ctx.slow_rows)] == out["baseline_order"]


def _plan(code_hash: str, rationale=None):
    promotion = {"code_hash": code_hash, "budget_class": "pilot"}
    if rationale is not None:
        promotion["rationale"] = rationale
    return {
        "observations": [],
        "rationale": {"text": "turn-level", "cites": ["S-4"]},
        "family_mix": {"bb": 1.0},
        "strategy_mix": {"novelty": 1.0},
        "program_count": 24,
        "promotions": [promotion],
        "directives": "",
        "finalize": False,
    }


def _validate(plan):
    return validate_turn_plan(
        plan,
        source_ids={"S-4"},
        known_hashes={H_MEASURED, H_HEAD, H_SECOND, H_CATASTROPHIC},
        ledger_ids=set(),
        families=("bb",),
        program_cap=96,
        promote_cap=1,
        quarantined_hashes={H_CATASTROPHIC},
        acquisition_order=ranking.runtime_slot_baseline(ARCHIVE, SLOW),
    )


def test_promoting_the_tools_head_passes_without_a_rationale() -> None:
    plan, errors = _validate(_plan(H_HEAD[:12]))
    assert plan is not None and errors == []


def test_rejection_names_the_head_and_both_ways_out() -> None:
    plan, errors = _validate(_plan(H_MEASURED[:12]))
    assert plan is None and len(errors) == 1
    msg = errors[0]
    assert f"the baseline head is '{H_HEAD[:12]}'" in msg
    assert "measured candidates are never in the baseline" in msg
    assert f'{{"code_hash": "{H_HEAD[:12]}", "budget_class": "pilot"}}' in msg
    assert f'{{"code_hash": "{H_MEASURED[:12]}", "budget_class": "pilot", "rationale"' in msg
    # taking either way out passes
    plan, errors = _validate(_plan(H_MEASURED[:12], {"text": "best measured plpr so far", "cites": ["S-4"]}))
    assert plan is not None and errors == []
