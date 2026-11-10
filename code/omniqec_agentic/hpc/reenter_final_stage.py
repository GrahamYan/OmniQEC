"""Re-enter the final stage of a FINISHED campaign so --resume measures it again.

WHY. A campaign that stopped on ``max_turns`` and wrote its report sits at END
in the checkpoint: ``--resume`` validates the protocol, applies runtime
overrides and returns at once without executing a node, so neither the
per-candidate final-stage resume (2026-09-24) nor a raised
``final_decode_shard_timeout_cap_s`` can take effect. The 2026-10-14 case: the
capped final stage (14400 s per shard, 300 s per 3-shot chunk) ended 15
runtime/baseline finalists of six Fig.2 runs ``decode_infeasible`` after one
non-converging shot; they have to be measured again without the cap, while
the structural-endpoint finalists keep their records.

WHAT. The graph position is rewritten *as if* ``reconcile`` had just run with
``stop`` left as it is (``finalize`` / ``max_turns``) and the ``final_slow`` /
``report`` state cleared, so the next ``app.invoke(None)`` routes through
``continue_or_finalize`` into ``final_slow`` (same shortlist: it is rebuilt
from the unchanged state) and then ``report``. ``final_slow_node`` runs the
queue with ``resume_candidates=True``: every ``final_slow/T<r>/candidates/
<program_id>.json`` with a finished status is reused, the missing ones are
measured. Nothing on disk is touched here; the caller decides beforehand which
candidate files to move aside and must move ``campaign_report.{json,md}``
aside too (the launcher skips a point whose report exists).

USAGE (dry run by default):
    python -m omniqec_agentic.hpc.reenter_final_stage --run-dir /abs/runs/fig2_claudesub_b288_v37
    python -m omniqec_agentic.hpc.reenter_final_stage --run-dir ... --apply

Refuses when: the checkpoint's ``run_dir`` is not this directory, the campaign
did not finish on ``max_turns``, a node is still pending, FATAL_STOP.json is
present, or campaign_report.json is still in place.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from omniqec_agentic.orchestrator import FallbackPolicy
from omniqec_agentic.workflow.checkpoint import CHECKPOINT_FILENAME, open_checkpointer
from omniqec_agentic.workflow.graph import build_campaign_graph

EXPECTED_NEXT = ("final_slow",)


def _thread_ids(db_path: Path) -> list[str]:
    with sqlite3.connect(str(db_path)) as con:
        return [row[0] for row in con.execute("SELECT DISTINCT thread_id FROM checkpoints")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="absolute path of the finished run dir")
    parser.add_argument("--apply", action="store_true", help="write the change (default: dry run, print only)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    db_path = run_dir / CHECKPOINT_FILENAME
    if not db_path.exists():
        print(f"no {CHECKPOINT_FILENAME} under {run_dir}", file=sys.stderr)
        return 2
    threads = _thread_ids(db_path)
    if len(threads) != 1:
        print(f"expected exactly one thread in {db_path}, found {threads}", file=sys.stderr)
        return 2
    thread_id = threads[0]
    app = build_campaign_graph(FallbackPolicy()).compile(checkpointer=open_checkpointer(run_dir))
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)
    if not snapshot or not snapshot.values:
        print(f"thread {thread_id} has no restorable state", file=sys.stderr)
        return 2
    values = snapshot.values
    stored_dir = str(values.get("run_dir") or "")
    if Path(stored_dir).resolve() != run_dir:
        print(
            f"REFUSING: the checkpoint's run_dir is {stored_dir!r}, not this directory — "
            "run hpc/rebase_run_dir.py --apply first",
            file=sys.stderr,
        )
        return 3
    protocol = dict(values.get("protocol") or {})
    stop = dict(values.get("stop") or {})
    pending = tuple(snapshot.next or ())
    if pending or not stop.get("finalize") or stop.get("reason") != "max_turns":
        print(
            f"REFUSING: not a finished max_turns run (pending nodes {list(pending)}, stop={stop})",
            file=sys.stderr,
        )
        return 3
    if (run_dir / "FATAL_STOP.json").exists():
        print("REFUSING: FATAL_STOP.json is present — resolve it first", file=sys.stderr)
        return 3
    if (run_dir / "campaign_report.json").exists():
        print(
            "REFUSING: campaign_report.json is still in place — move it (and campaign_report.md) "
            "aside first, the launcher skips a point whose report exists",
            file=sys.stderr,
        )
        return 3
    max_turns = int(protocol.get("max_turns") or 0)
    turn = int(values.get("turn") or 0)
    turns_used = int((values.get("budget") or {}).get("turns_used") or 0)
    if turn != max_turns or turns_used != max_turns:
        print(
            f"REFUSING: state turn={turn} turns_used={turns_used} do not match max_turns={max_turns}",
            file=sys.stderr,
        )
        return 3
    final_dir = run_dir / "final_slow"
    candidate_files = sorted(p.name for p in final_dir.glob("T*/candidates/*.json"))
    print(f"run dir       : {run_dir}")
    print(f"thread        : {thread_id}")
    print(f"pending nodes : {list(pending)} (finished run, stop={stop})")
    print(f"final cap     : {protocol.get('final_decode_shard_timeout_cap_s')} (protocol runtime key)")
    print(f"candidates    : {len(candidate_files)} file(s) kept for reuse: {candidate_files}")
    print("state writes  : final_slow={}, report={}  as_node=reconcile  (stop unchanged -> routes to final_slow)")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write this change")
        return 0
    try:
        app.update_state(config, {"final_slow": {}, "report": {}}, as_node="reconcile")
    except Exception as exc:  # noqa: BLE001 — surface langgraph's own message
        print(f"update_state failed: {exc}", file=sys.stderr)
        return 4
    after = app.get_state(config)
    if tuple(after.next or ()) != EXPECTED_NEXT:
        print(
            f"ERROR: pending nodes after the rewrite are {list(after.next)}, expected {list(EXPECTED_NEXT)}; "
            "do not resume this checkpoint — inspect it",
            file=sys.stderr,
        )
        return 4
    print(f"\napplied: pending nodes now {list(after.next)}; --resume will re-run the final stage then the report")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
