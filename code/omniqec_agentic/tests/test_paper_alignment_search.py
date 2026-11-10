"""Search-flow alignment with the paper (2026-11-10, docs/paper_alignment_search.md).

1. The five hand-written seed programs run in the first iteration only (appendix D1).
2. The fast loop rejects, before distance estimation, every realization that is not a valid CSS code with
   k >= 1, a connected Tanner graph and w_max <= 8 (Methods "Search configurations", appendix B2/D1).
3. Every iteration ends with a review: the reflector retries a failed call with backoff and a review that
   still fails is recorded in the turn audit and the campaign audit summary (appendix B4).
4. Table I / B4 details: remaining evaluation capacity, refuted hypotheses with their status, the strategy
   recommendation weights in the generator brief, the fast-loop screening results for the reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omniqec_agentic.config import (
    SEARCH_MAX_CHECK_WEIGHT,
    SEED_PROGRAMS_EVERY_TURN,
    SEED_PROGRAMS_FIRST_TURN,
    HardwareConfig,
    HardwareTarget,
    RacingConfig,
    RunConfig,
    VerificationConfig,
    formal_racing,
    seed_programs_policy_of,
    seed_programs_run_this_turn,
)
from omniqec_agentic.funsearch import _construction_contract_lines
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.orchestrator import reflector
from omniqec_agentic.schemas import CandidateProgram
from omniqec_agentic.verification import fast as fast_mod
from omniqec_agentic.workflow.state import build_protocol_snapshot

SOURCE = "def construct(rng):\n    return %r\n"
# BB [[72,12,6]]: weight 6, connected -- passes every fast-loop check at N = 144.
BB72 = {"family": "bb", "ell": 6, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]}
# A weight-9 BB code (|A| + |B| = 9), k = 8, n = 36, N = 72, connected: rejected only by w_max <= 8.
BB_W9 = {"family": "bb", "ell": 3, "m": 6, "A": [[2, 1], [1, 5], [0, 0], [1, 2]],
         "B": [[2, 5], [1, 4], [2, 2], [1, 3], [0, 0]]}
# A weight-4 BB code whose Tanner graph is disconnected (even exponents only), k = 8.
BB_DISCONNECTED = {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [2, 0]], "B": [[0, 0], [0, 2]]}
# A weight-4 BB code, connected, [[72,2]].
BB_W4 = {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [0, 1]], "B": [[0, 0], [1, 0]]}


def _program(program_id: str, spec: dict, budget: int = 144) -> CandidateProgram:
    return CandidateProgram(program_id=program_id, source=SOURCE % spec, n_budget=budget)


def _count_proxy_calls(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = fast_mod.proxy_distance_search

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fast_mod, "proxy_distance_search", counting)
    return calls


def _snapshot(**kw) -> dict:
    return build_protocol_snapshot(
        verification=VerificationConfig(), racing=formal_racing(25_000), max_turns=30, proposal_count=24,
        generations_per_turn=4, slow_workers=8, n_budget=144, **kw,
    )


# --------------------------------------------------------------------------- 1. seed programs
def test_seed_policy_runs_the_seeds_in_the_first_turn_only() -> None:
    first = {"seed_programs": SEED_PROGRAMS_FIRST_TURN}
    assert seed_programs_run_this_turn(first, 0) is True
    assert [seed_programs_run_this_turn(first, turn) for turn in (1, 2, 29)] == [False, False, False]
    # a snapshot of the earlier code has no key: it keeps running them every turn on resume
    assert seed_programs_policy_of({}) == SEED_PROGRAMS_EVERY_TURN
    assert all(seed_programs_run_this_turn({}, turn) for turn in (0, 1, 7))
    with pytest.raises(ValueError, match="unknown seed_programs"):
        seed_programs_policy_of({"seed_programs": "never"})


def test_snapshot_freezes_first_turn_seeds_and_the_reflection_policy() -> None:
    snapshot = _snapshot()
    assert snapshot["seed_programs"] == SEED_PROGRAMS_FIRST_TURN
    assert snapshot["reflection_policy"] == reflector.reflection_policy()
    assert snapshot["reflection_policy"]["attempts"] == reflector.REFLECTION_ATTEMPTS >= 2
    assert snapshot["verification"]["max_check_weight"] == SEARCH_MAX_CHECK_WEIGHT == 8
    with pytest.raises(ValueError, match="seed_programs must be one of"):
        _snapshot(seed_programs="sometimes")


def test_resume_of_an_every_turn_checkpoint_is_refused() -> None:
    from omniqec_agentic.workflow.checkpoint import validate_resume_protocol

    current = _snapshot()
    legacy = {key: value for key, value in current.items() if key not in ("seed_programs", "reflection_policy")}
    with pytest.raises(Exception):
        validate_resume_protocol(legacy, current)


def _execute_turn(monkeypatch, tmp_path: Path, turn: int, protocol_extra: dict) -> dict:
    from omniqec_agentic import campaign
    from omniqec_agentic.workflow import graph

    captured: dict = {}

    def fake_cycle(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status="ok", artifact_paths={})

    monkeypatch.setattr(campaign, "run_agentic_cycle", fake_cycle)
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *args, **kwargs: VerificationConfig())
    run_dir = tmp_path / f"run_t{turn}"
    run_dir.mkdir()
    protocol = {"proposal_count": 1, "generations_per_turn": 1, "feedback_mode": "both",
                "racing": RacingConfig().to_dict(), "slow_workers": 1, "shot_shards": 16, "strict_turns": False,
                **protocol_extra}
    state = {
        "run_id": "align", "run_dir": str(run_dir), "turn": turn,
        "plan": {"provider": "dry_run", "generations": 1, "proposal_count": 1, "rng_seeds_per_program": 1},
        "protocol": protocol, "spec": {"qubit_budget": 144, "families": ["bb"]},
        "structural_proxy_elites": [], "hof_summary": [], "unmeasurable_hashes": [],
        "baseline_summary": {}, "budget": {},
    }
    graph.execute_plan_node(state)
    return captured


def test_execute_plan_passes_the_seeds_to_turn_zero_only(monkeypatch, tmp_path: Path) -> None:
    first = {"seed_programs": SEED_PROGRAMS_FIRST_TURN}
    assert _execute_turn(monkeypatch, tmp_path, 0, first)["use_seeds"] is True
    assert _execute_turn(monkeypatch, tmp_path, 1, first)["use_seeds"] is False
    assert _execute_turn(monkeypatch, tmp_path, 5, first)["use_seeds"] is False
    assert _execute_turn(monkeypatch, tmp_path, 3, {})["use_seeds"] is True   # legacy snapshot


def test_a_turn_without_seeds_evaluates_only_the_generated_programs(tmp_path: Path) -> None:
    """run_agentic_cycle(use_seeds=False) writes no seed program and records no seed attempt; the
    generated programs are evaluated exactly as before."""

    from omniqec_agentic.campaign import run_agentic_cycle
    from omniqec_agentic.construction.groups.catalog import group_order_budget

    def config(name: str) -> RunConfig:
        return RunConfig(run_id=name, n_budget=144, hardware=HardwareConfig(target=HardwareTarget.ION),
                         out_dir=str(tmp_path / name), proposal_count=2, feedback_mode="fast")

    with group_order_budget(None):
        seeded = run_agentic_cycle(config("t0"), provider="dry_run", generations=1, slow_shots=50,
                                   slow_workers=1, use_seeds=True, ablation="no_slow")
        unseeded = run_agentic_cycle(config("t1"), provider="dry_run", generations=1, slow_shots=50,
                                     slow_workers=1, use_seeds=False, ablation="no_slow")
    assert seeded.funsearch_summary["seed_count"] == 5
    assert sorted(p.name for p in (tmp_path / "t0" / "fast" / "programs").glob("seed_*.py")) == [
        f"seed_{i:04d}.py" for i in range(1, 6)]
    assert unseeded.funsearch_summary["seed_count"] == 0
    assert not list((tmp_path / "t1" / "fast" / "programs").glob("seed_*.py"))
    transcript = [json.loads(line) for line in
                  (tmp_path / "t1" / "fast" / "transcript.jsonl").read_text().splitlines() if line.strip()]
    assert not [row for row in transcript if row.get("source") == "seed"]
    assert unseeded.funsearch_summary["proposal_count"] == seeded.funsearch_summary["proposal_count"] == 2


# --------------------------------------------------------------------------- 2. fast-loop checks
def test_the_default_search_cap_is_eight_and_the_contract_says_so() -> None:
    assert VerificationConfig().max_check_weight == SEARCH_MAX_CHECK_WEIGHT == 8
    config = RunConfig(run_id="cap", n_budget=144, hardware=HardwareConfig(target=HardwareTarget.ION),
                       out_dir="unused")
    lines = _construction_contract_lines(config)
    assert any(line.startswith("CHECK WEIGHT CAP (run protocol): every X and Z check must have weight <= 8")
               for line in lines)
    # an exact Fig.4 c contract under the default cap is consistent; one above it is not
    assert VerificationConfig(exact_check_weight=5).exact_check_weight == 5
    with pytest.raises(ValueError, match=">= exact_check_weight"):
        VerificationConfig(max_check_weight=4, exact_check_weight=6)
    assert VerificationConfig(max_check_weight=None).max_check_weight is None   # the v3.7 as-run protocol


def test_fast_loop_rejects_a_heavy_code_before_distance_estimation(monkeypatch) -> None:
    calls = _count_proxy_calls(monkeypatch)
    memory = ArchiveMemory(turn=0, max_check_weight=8)
    record = memory.evaluate_and_record(_program("w9", BB_W9, budget=72), n_budget=72, proxy_iters=20)
    attempt = memory.attempts[-1]
    assert calls == [], "no distance estimation for a rejected realization"
    assert attempt["archive_status"] == "structural_reject"
    assert "max_check_weight_exceeded:9>8" in attempt["failure_reasons"]
    assert attempt["max_check_weight"] == 9 and attempt["k"] == 8 and attempt["n"] == 36
    assert attempt["d_search"] is None and attempt["Q"] is None and not attempt["structural_eligible"]
    assert record.fast is None and not record.parent_eligible
    assert not memory.records_by_cell and not memory.structural_proxy_elites
    assert attempt["code_hash"] not in memory.code_hash_owner
    # the same code again is rejected for the same reason, not reported as a duplicate
    memory.evaluate_and_record(_program("w9_again", BB_W9, budget=72), n_budget=72, proxy_iters=20)
    assert memory.attempts[-1]["archive_status"] == "structural_reject"
    assert memory.attempts[-1]["duplicate_of"] is None
    # without the cap (the v3.7 as-run protocol) the code is evaluated
    plain = ArchiveMemory(turn=0, max_check_weight=None)
    plain.evaluate_and_record(_program("w9", BB_W9, budget=72), n_budget=72, proxy_iters=20)
    assert plain.attempts[-1]["archive_status"] != "structural_reject" and calls


def test_fast_loop_rejects_a_disconnected_tanner_graph(monkeypatch) -> None:
    calls = _count_proxy_calls(monkeypatch)
    memory = ArchiveMemory(turn=0, max_check_weight=8)
    memory.evaluate_and_record(_program("disc", BB_DISCONNECTED), n_budget=144, proxy_iters=20)
    attempt = memory.attempts[-1]
    assert attempt["archive_status"] == "structural_reject" and calls == []
    assert "disconnected_tanner_graph" in attempt["failure_reasons"]
    assert not memory.structural_proxy_elites


def test_valid_codes_still_pass_and_a_valid_duplicate_is_still_a_duplicate(monkeypatch) -> None:
    calls = _count_proxy_calls(monkeypatch)
    memory = ArchiveMemory(turn=0, max_check_weight=8)
    memory.evaluate_and_record(_program("bb72", BB72), n_budget=144, proxy_iters=20)
    first = memory.attempts[-1]
    assert first["valid_structural"] and first["d_search"] is not None and len(calls) == 1
    memory.evaluate_and_record(_program("bb72_copy", BB72), n_budget=144, proxy_iters=20)
    assert memory.attempts[-1]["archive_status"] == "duplicate" and memory.attempts[-1]["duplicate_of"] == "bb72"
    assert len(calls) == 1


def test_the_process_pool_path_applies_the_same_checks_and_the_exact_contract() -> None:
    batch = [_program("w9", BB_W9, budget=144), _program("bb72", BB72), _program("w4", BB_W4),
             _program("w9_again", BB_W9, budget=144)]
    pooled = ArchiveMemory(turn=0, max_check_weight=8)
    pooled.evaluate_batch(batch, n_budget=144, proxy_iters=20, workers=2)
    serial = ArchiveMemory(turn=0, max_check_weight=8)
    serial.evaluate_batch(batch, n_budget=144, proxy_iters=20, workers=1)
    keys = ("program_id", "archive_status", "failure_reasons", "duplicate_of", "d_search", "code_hash")
    assert [{k: a[k] for k in keys} for a in pooled.attempts] == [{k: a[k] for k in keys} for a in serial.attempts]
    assert [a["archive_status"] for a in pooled.attempts][0] == "structural_reject"
    # Fig.4 c: the exact-weight contract held in the pool path too (it used to be dropped there)
    exact = ArchiveMemory(turn=0, max_check_weight=4, exact_check_weight=4)
    exact.evaluate_batch([_program("bb72", BB72), _program("w4", BB_W4)], n_budget=144, proxy_iters=20, workers=2)
    by_id = {a["program_id"]: a for a in exact.attempts}
    assert "check_weight_mismatch:6!=4" in by_id["bb72"]["failure_reasons"]
    assert by_id["w4"]["structural_eligible"] is True


def test_cli_cap_default_and_off_switch(monkeypatch, tmp_path: Path) -> None:
    """--max-check-weight unset = 8 in the snapshot; 0 = no fast-loop cap (the v3.7 as-run variant)."""

    from omniqec_agentic import cli as cli_mod

    seen: list = []

    def fake_run_campaign(**kwargs):
        seen.append(kwargs["verification"])
        return {"run_id": "campaign_b144", "stop": {"reason": "test"}, "report": {}}

    monkeypatch.setattr("omniqec_agentic.workflow.run_campaign", fake_run_campaign)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    base = ["run-campaign", "--budget", "144", "--turns", "1", "--provider", "dry_run", "--out", str(tmp_path)]
    for extra in ([], ["--max-check-weight", "0"], ["--max-check-weight", "6"]):
        assert cli_mod.main(base + extra) == 0
    assert [v.max_check_weight for v in seen] == [8, None, 6]
    with pytest.raises(SystemExit, match="0 \\(no cap\\) or >= 3"):
        cli_mod.main(base + ["--max-check-weight", "2"])


# --------------------------------------------------------------------------- 3. reflection retries
class _FakeClient:
    def __init__(self, replies: list):
        self.replies = replies

    def complete(self, *, system: str, user: str) -> str:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def info(self) -> dict:
        return {"usage": {"output_tokens": 10}}


GOOD_REPLY = json.dumps({"what_worked": ["bb at w=6"], "what_failed": [], "surprises": [],
                         "candidate_hypotheses": [], "next_turn_advice": "keep the bb front"})


def _reflect(monkeypatch, tmp_path: Path, replies: list, **kwargs) -> tuple[dict, list, list[dict]]:
    from omniqec_agentic.llm import factory

    replies = list(replies)
    monkeypatch.setattr(factory, "create_completion_client", lambda provider, config: _FakeClient(replies))
    sleeps: list = []
    record = reflector.run_reflection(
        provider="openai", llm_config=None, turn=4, digest={"cycle_status": "ok"}, plan={}, coverage={},
        turn_dir=tmp_path / "turn_0004", manual_dir=tmp_path / "manual", sleep=sleeps.append, **kwargs,
    )
    rows = [json.loads(line) for line in
            (tmp_path / "turn_0004" / "reflection_call.jsonl").read_text().splitlines() if line.strip()]
    return record, sleeps, rows


def test_a_failed_review_is_retried_with_backoff(monkeypatch, tmp_path: Path) -> None:
    record, sleeps, rows = _reflect(
        monkeypatch, tmp_path, [RuntimeError("503 auth_unavailable"), "no json here", GOOD_REPLY])
    assert record["status"] == "ok" and record["attempts"] == 3
    assert record["next_turn_advice"] == "keep the bb front" and "error" not in record
    assert sleeps == list(reflector.REFLECTION_BACKOFF_S)
    assert [(row["attempt"], row["status"]) for row in rows] == [(1, "failed"), (2, "failed"), (3, "ok")]
    assert "UnparseableReflection" in rows[1]["error"]
    stored = reflector.read_reflections(tmp_path / "manual")
    assert [row["turn"] for row in stored] == [4] and stored[0]["status"] == "ok"


def test_a_review_that_keeps_failing_is_recorded_as_failed(monkeypatch, tmp_path: Path) -> None:
    errors = [RuntimeError(f"429 rate limit #{i}") for i in range(reflector.REFLECTION_ATTEMPTS)]
    record, sleeps, rows = _reflect(monkeypatch, tmp_path, errors)
    assert record["status"] == "failed" and record["attempts"] == reflector.REFLECTION_ATTEMPTS
    assert "429 rate limit #2" in record["error"]
    assert len(rows) == reflector.REFLECTION_ATTEMPTS and all(row["status"] == "failed" for row in rows)
    on_disk = json.loads((tmp_path / "turn_0004" / "reflection.json").read_text())
    assert on_disk["status"] == "failed" and on_disk["attempts"] == reflector.REFLECTION_ATTEMPTS


def test_no_new_attempt_after_the_wall_budget(monkeypatch, tmp_path: Path) -> None:
    ticks = iter([0.0, 5000.0, 5000.0, 5000.0])
    record, sleeps, rows = _reflect(monkeypatch, tmp_path, [RuntimeError("timeout"), GOOD_REPLY],
                                    clock=lambda: next(ticks))
    assert record["status"] == "failed" and record["attempts"] == 1 and sleeps == [] and len(rows) == 1


def test_a_first_try_success_is_unchanged(monkeypatch, tmp_path: Path) -> None:
    record, sleeps, rows = _reflect(monkeypatch, tmp_path, [GOOD_REPLY])
    assert record["status"] == "ok" and record["attempts"] == 1 and sleeps == [] and len(rows) == 1


def test_the_review_outcome_reaches_the_turn_and_campaign_audits(tmp_path: Path) -> None:
    from omniqec_agentic.workflow.graph import _record_reflection_audit, _validate_all_turn_audits

    protocol = {"strict_turns": True, "max_turns": 3}
    outcomes = [{"status": "ok", "attempts": 1}, {"status": "failed", "attempts": 3, "error": "429"},
                {"status": "ok", "attempts": 2}]
    for turn, outcome in enumerate(outcomes):
        d = tmp_path / "turns" / f"turn_{turn:04d}"
        d.mkdir(parents=True)
        (d / "turn_audit.json").write_text(json.dumps({"status": "complete", "turn": turn,
                                                       "checks": {"plan_policy": "agentic_v1"}}))
        _record_reflection_audit({"run_dir": str(tmp_path), "protocol": protocol, "turn": turn},
                                 turn=turn, reflection=outcome)
    audit = json.loads((tmp_path / "turns" / "turn_0001" / "turn_audit.json").read_text())
    assert audit["reflection"] == outcomes[1] and audit["status"] == "complete"
    summary = _validate_all_turn_audits({"run_dir": str(tmp_path), "protocol": protocol})
    assert summary["completed"] == 3, "a failed review never fails the audit"
    assert summary["reflection_failed_turns"] == [1]
    assert summary["reflection_turns"] == {"failed": [1], "ok": [0, 2]}


def test_the_reviewer_sees_the_fast_loop_screening(tmp_path: Path) -> None:
    from omniqec_agentic.workflow.graph import _with_fast_screening

    fast = tmp_path / "turns" / "turn_0002" / "cycle" / "fast"
    fast.mkdir(parents=True)
    (fast / "feedback.json").write_text(json.dumps(
        {"failure_counts": {"max_check_weight_exceeded:9>8": 40, "disconnected_tanner_graph": 3}}))
    (fast / "funsearch_summary.json").write_text(json.dumps(
        {"seed_count": 0, "proposal_count": 3840, "accepted_count": 120, "structural_reject_count": 43,
         "fast_reject_count": 600, "duplicate_count": 900, "artifact_paths": {"feedback": str(fast / "feedback.json")}}))
    state = {"last_cycle": {"artifact_paths": {"fast_summary": str(fast / "funsearch_summary.json")}}}
    extra = _with_fast_screening(None, state, tmp_path / "turns" / "turn_0002")
    screening = extra["fast_screening"]
    assert screening["structural_reject_count"] == 43 and screening["seed_count"] == 0
    assert list(screening["rejection_reasons_top"]) == ["max_check_weight_exceeded:9>8", "disconnected_tanner_graph"]
    text = json.loads(reflector.reflection_input(turn=2, digest={}, plan={}, coverage={}, extra=extra))
    assert text["fast_loop_screening"]["accepted_count"] == 120 and "task_evidence" not in text


# --------------------------------------------------------------------------- 4. Table I / B4 details
def test_orchestrator_budget_block_states_the_remaining_capacity() -> None:
    from omniqec_agentic.orchestrator.loop import _remaining_capacity

    state = {"protocol": {"max_turns": 30, "promote_top": 6, "structural_retention": {"runtime_slow_slots": 4}},
             "budget": {"turns_used": 12}}
    assert _remaining_capacity(state) == {"turns_including_this_one": 18, "slow_evaluation_slots_per_turn": 6,
                                          "nominable_runtime_slots_per_turn": 4}


def test_the_hypothesis_list_shows_refuted_entries_with_their_status(tmp_path: Path) -> None:
    from omniqec_agentic.memory.bandit import StrategyBandit
    from omniqec_agentic.memory.ledger import Ledger
    from omniqec_agentic.orchestrator.manual import render_dynamic

    ledger = Ledger(tmp_path / "ledger.json")
    result = ledger.propose(scope={"families": ["bb"]}, predicate=[{"field": "max_check_weight", "op": ">=", "value": 9}],
                            claim={"metric": "per_logical_per_round", "direction": "lower"}, proposer="test",
                            turn=1, propose_gen=1, rationale="test")
    assert result["accepted"], result
    ledger.entries[max(ledger.entries)].status = "refuted"
    render_dynamic(tmp_path / "manual", ledger=ledger, bandit=StrategyBandit(tmp_path / "bandit.json"), digest=None)
    text = (tmp_path / "manual" / "dynamic" / "insights.md").read_text()
    assert "## Refuted" in text and "[refuted" in text


def test_the_generator_brief_carries_the_strategy_recommendation_weights(tmp_path: Path) -> None:
    from omniqec_agentic.memory.bandit import STRATEGY_ARMS, StrategyBandit
    from omniqec_agentic.workflow.graph import _strategy_brief

    state = {"run_dir": str(tmp_path), "spec": {"qubit_budget": 144}}
    brief = _strategy_brief({"strategy_mix": {"novelty": 1.0}}, state)
    line = next(l for l in brief.splitlines() if l.startswith("- strategy recommendation weights"))
    assert all(f"{arm}: 0.20" in line for arm in STRATEGY_ARMS), line   # uniform before any reward
    StrategyBandit(tmp_path / "bandit.json").update({"exploit": 1.0}, gen=0)
    line = next(l for l in _strategy_brief({}, state).splitlines() if l.startswith("- strategy recommendation"))
    assert "exploit: 0.80" in line and "novelty: 0.05" in line
