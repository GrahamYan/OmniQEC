"""hpc/extend_turns.py: raise a finished campaign's frozen max_turns and put the
graph back in front of orchestrator_turn, so ``--resume --turns <new>`` runs
the extra turns and a fresh final stage. The heavy nodes are stubbed; what is
under test is the checkpoint surgery and the resume routing around it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from omniqec_agentic.hpc import extend_turns
from omniqec_agentic.orchestrator import FallbackPolicy
from omniqec_agentic.workflow import graph as graph_mod
from omniqec_agentic.workflow.checkpoint import open_checkpointer
from omniqec_agentic.workflow.state import CampaignSpec


def _stub_nodes(monkeypatch: pytest.MonkeyPatch, ran: list[int]) -> None:
    def make_orchestrator(policy):  # noqa: ANN001 — mirrors the real factory
        def node(state: dict[str, Any]) -> dict[str, Any]:
            return {"plan": {"turn": int(state["turn"])}}

        return node

    def execute_plan(state: dict[str, Any]) -> dict[str, Any]:
        turn = int(state["turn"])
        turn_dir = Path(state["run_dir"]) / "turns" / f"turn_{turn:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "turn_audit.json").write_text(
            json.dumps({"status": "complete", "turn": turn, "checks": {"plan_policy": "agentic_v1"}}),
            encoding="utf-8",
        )
        ran.append(turn)
        return {"last_cycle": {"turn": turn}}

    def reconcile(state: dict[str, Any]) -> dict[str, Any]:
        turn = int(state["turn"])
        budget = dict(state.get("budget") or {})
        budget["turns_used"] = int(budget.get("turns_used") or 0) + 1
        stop = graph_mod._stop_decision(state, [], {}, budget)
        return {"turn": turn + 1, "budget": budget, "stop": stop}

    def final_slow(state: dict[str, Any]) -> dict[str, Any]:
        final_dir = Path(state["run_dir"]) / "final_slow"
        final_dir.mkdir(exist_ok=True)
        (final_dir / "queue.jsonl").write_text("{}\n", encoding="utf-8")
        return {"final_slow": {"status": "ok", "runs": [], "at_turn": int(state["turn"])}}

    def report(state: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "turns_completed": int((state.get("budget") or {}).get("turns_used") or 0),
            "stop": state.get("stop"),
            "protocol": state["protocol"],
        }
        (Path(state["run_dir"]) / "campaign_report.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"report": payload}

    monkeypatch.setattr(graph_mod, "make_orchestrator_node", make_orchestrator)
    monkeypatch.setattr(graph_mod, "execute_plan_node", execute_plan)
    monkeypatch.setattr(graph_mod, "reconcile_node", reconcile)
    monkeypatch.setattr(graph_mod, "final_slow_node", final_slow)
    monkeypatch.setattr(graph_mod, "report_node", report)


def _run(run_dir: Path, *, max_turns: int, resume: bool) -> dict[str, Any]:
    return graph_mod.run_campaign(
        spec=CampaignSpec(qubit_budget=60),
        out_dir=run_dir,
        policy=FallbackPolicy(),
        max_turns=max_turns,
        proposal_count=2,
        slow_workers=1,
        resume=resume,
    )


def _pending(run_dir: Path) -> tuple[str, ...]:
    app = graph_mod.build_campaign_graph(FallbackPolicy()).compile(checkpointer=open_checkpointer(run_dir))
    return tuple(app.get_state({"configurable": {"thread_id": "campaign_b60"}}).next or ())


def test_extending_a_finished_run_reenters_the_loop_and_reruns_the_final_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[int] = []
    _stub_nodes(monkeypatch, ran)
    run_dir = tmp_path / "fig2_gptsub_b60_v36x"
    first = _run(run_dir, max_turns=1, resume=False)
    assert first["stop"] == {"finalize": True, "reason": "max_turns"}
    assert ran == [0]
    assert json.loads((run_dir / "campaign_report.json").read_text())["turns_completed"] == 1
    assert _pending(run_dir) == ()

    # Dry run: prints the plan, moves nothing, leaves the checkpoint at END.
    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "3"]) == 0
    assert (run_dir / "campaign_report.json").exists()
    assert _pending(run_dir) == ()
    assert json.loads((run_dir / "protocol_snapshot.json").read_text())["max_turns"] == 1

    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "3", "--apply"]) == 0
    backups = sorted(run_dir.glob("extend_backup_t1_*"))
    assert len(backups) == 1
    assert (backups[0] / "campaign_report.json").exists()
    assert (backups[0] / "final_slow" / "queue.jsonl").exists()
    assert not (run_dir / "campaign_report.json").exists()
    assert not (run_dir / "final_slow").exists()
    snapshot = json.loads((run_dir / "protocol_snapshot.json").read_text())
    assert snapshot["max_turns"] == 3
    assert snapshot["runtime_overrides"][-1] == {
        "turn": 1, "from": {"max_turns": 1}, "to": {"max_turns": 3}, "kind": "extend_turns",
    }
    assert _pending(run_dir) == ("orchestrator_turn",)

    # A second extension of the same checkpoint is refused: it is no longer a finished run.
    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "5", "--apply"]) == 3

    # Resume exactly as the launcher does (--turns must be the NEW value).
    second = _run(run_dir, max_turns=3, resume=True)
    assert ran == [0, 1, 2]
    assert second["stop"] == {"finalize": True, "reason": "max_turns"}
    report = json.loads((run_dir / "campaign_report.json").read_text())
    assert report["turns_completed"] == 3
    assert report["protocol"]["max_turns"] == 3
    assert (run_dir / "final_slow" / "queue.jsonl").exists()
    assert {p.name for p in (run_dir / "turns").iterdir()} == {"turn_0000", "turn_0001", "turn_0002"}
    assert _pending(run_dir) == ()


def test_refusals_do_not_touch_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[int] = []
    _stub_nodes(monkeypatch, ran)
    run_dir = tmp_path / "fig2_gptsub_b60_v36x"
    _run(run_dir, max_turns=1, resume=False)

    # Not a raise.
    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "1", "--apply"]) == 3
    assert (run_dir / "campaign_report.json").exists()
    assert _pending(run_dir) == ()

    # A FATAL_STOP marker blocks the extension.
    (run_dir / "FATAL_STOP.json").write_text("{}", encoding="utf-8")
    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "3", "--apply"]) == 3
    (run_dir / "FATAL_STOP.json").unlink()

    # A copied-but-not-rebased run dir still points at the original: refused before any write.
    copy_dir = tmp_path / "fig2_gptsub_b60_v36copy"
    shutil.copytree(run_dir, copy_dir)
    assert extend_turns.main(["--run-dir", str(copy_dir), "--turns", "3", "--apply"]) == 3
    assert (copy_dir / "campaign_report.json").exists()
    assert not list(copy_dir.glob("extend_backup_*"))

    # The original is untouched by all of the above.
    assert extend_turns.main(["--run-dir", str(run_dir), "--turns", "3"]) == 0
    assert (run_dir / "campaign_report.json").exists()


def test_extended_protocol_records_the_override_like_a_lowering() -> None:
    from omniqec_agentic.workflow.checkpoint import original_max_turns, validate_resume_protocol

    stored = {"max_turns": 30, "protocol_profile": "telescoping_structural_v3_7"}
    merged = extend_turns.extended_protocol(stored, turns=60, at_turn=30)
    assert merged["max_turns"] == 60
    assert original_max_turns(merged) == 30
    # The launcher's fresh snapshot for --turns 60 carries no history: resume must accept it.
    validate_resume_protocol(merged, {"max_turns": 60, "protocol_profile": "telescoping_structural_v3_7"})
    with pytest.raises(ValueError, match="does not exceed"):
        extend_turns.extended_protocol(stored, turns=30, at_turn=30)
