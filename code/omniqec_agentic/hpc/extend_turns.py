"""Extend a FINISHED campaign: raise the frozen max_turns and re-enter the loop.

WHY. ``max_turns`` is a frozen protocol key. ``--resume`` accepts LOWERING it
only (``RESUME_LOWERABLE_KEYS``, 2026-09-05) and a finished campaign's
checkpoint sits at END with no pending node, so a plain ``--resume --turns 60``
is refused (protocol mismatch) or, once the snapshot said 60, would return
immediately without running anything. Raising the turn budget therefore needs
two edits a launch cannot make:

1. the checkpointed protocol's ``max_turns`` (mirrored to
   ``protocol_snapshot.json``), recorded as a ``runtime_overrides`` history
   line exactly like a lowering, so the audit trail and
   ``original_max_turns`` keep working;
2. the graph position: the state is re-written *as if* ``reconcile`` had just
   run with ``stop`` cleared, so the next ``app.invoke(None)`` routes through
   ``continue_or_finalize`` to ``orchestrator_turn`` (turn = old max_turns)
   instead of END. ``final_slow`` / ``report`` state is cleared and their
   on-disk products (``campaign_report.json``, ``VALIDATED.json``,
   ``final_slow/``) are moved into ``extend_backup_<...>/`` so the launcher no
   longer sees the point as complete and the new final stage starts clean.

The result is a PROTOCOL VARIANT of the formal run. Do it on a COPY of the
formal run dir under its own RUN_TAG (``cp -a`` + ``hpc/rebase_run_dir.py``),
never in place under ``RUN_TAG=v36``, and launch with ``TURNS=<new>``
``VALIDATE_FORMAL=0`` — the launcher's default ``TURNS=30`` would LOWER the
budget straight back and finalize again on the spot.

USAGE (after copying + rebasing the run dir):
    python -m omniqec_agentic.hpc.extend_turns --run-dir /abs/runs/fig2_gptsub_b98_v36x60 --turns 60
    python -m omniqec_agentic.hpc.extend_turns --run-dir ... --turns 60 --apply

Dry-run by default: prints what would change and touches nothing until
``--apply``. Refuses when: the checkpoint's ``run_dir`` is not this directory
(rebase first), the campaign did not stop on ``max_turns`` (a FATAL_STOP or an
unfinished run is not something to extend), or the requested turns do not
exceed the frozen value.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from omniqec_agentic.orchestrator import FallbackPolicy
from omniqec_agentic.workflow.checkpoint import CHECKPOINT_FILENAME, open_checkpointer
from omniqec_agentic.workflow.graph import build_campaign_graph

# Products of the finished run's final stage. Moved aside (never deleted) so
# the launcher stops treating the point as complete and the re-run final stage
# does not read a stale queue.
FINISH_PRODUCTS = ("campaign_report.json", "VALIDATED.json", "final_slow")
EXPECTED_NEXT = ("orchestrator_turn",)


def _thread_ids(db_path: Path) -> list[str]:
    with sqlite3.connect(str(db_path)) as con:
        return [row[0] for row in con.execute("SELECT DISTINCT thread_id FROM checkpoints")]


def extended_protocol(protocol: dict[str, Any], *, turns: int, at_turn: int) -> dict[str, Any]:
    """The stored protocol with max_turns raised to ``turns`` and the history
    line the resume path itself would have written for an override."""
    old = int(protocol["max_turns"])
    if int(turns) <= old:
        raise ValueError(f"--turns {turns} does not exceed the frozen max_turns {old}")
    merged = dict(protocol)
    history = list(merged.get("runtime_overrides") or [])
    history.append(
        {
            "turn": int(at_turn),
            "from": {"max_turns": old},
            "to": {"max_turns": int(turns)},
            "kind": "extend_turns",
        }
    )
    merged["max_turns"] = int(turns)
    merged["runtime_overrides"] = history
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="absolute path of the (copied + rebased) run dir")
    parser.add_argument("--turns", type=int, required=True, help="new max_turns (must exceed the frozen value)")
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run, print only)")
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
    saver = open_checkpointer(run_dir)
    app = build_campaign_graph(FallbackPolicy()).compile(checkpointer=saver)
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
            "run hpc/rebase_run_dir.py --apply first (a moved/copied run resumes into the OLD "
            "location otherwise, i.e. into the formal run you are trying to preserve)",
            file=sys.stderr,
        )
        return 3
    protocol = dict(values.get("protocol") or {})
    if not protocol:
        print("REFUSING: the checkpoint has no protocol snapshot", file=sys.stderr)
        return 3
    stop = dict(values.get("stop") or {})
    pending = tuple(snapshot.next or ())
    if pending or not stop.get("finalize") or stop.get("reason") != "max_turns":
        print(
            f"REFUSING: this campaign is not a finished max_turns run (pending nodes {list(pending)}, "
            f"stop={stop}); extend only a run that completed its turn budget",
            file=sys.stderr,
        )
        return 3
    if (run_dir / "FATAL_STOP.json").exists():
        print("REFUSING: FATAL_STOP.json is present — resolve it before extending", file=sys.stderr)
        return 3
    old_turns = int(protocol["max_turns"])
    turn = int(values.get("turn") or 0)
    turns_used = int((values.get("budget") or {}).get("turns_used") or 0)
    if turn != old_turns or turns_used != old_turns:
        print(
            f"REFUSING: state turn={turn} turns_used={turns_used} do not match max_turns={old_turns}",
            file=sys.stderr,
        )
        return 3
    try:
        merged = extended_protocol(protocol, turns=int(args.turns), at_turn=turn)
    except ValueError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 3

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = run_dir / f"extend_backup_t{old_turns}_{stamp}"
    to_move = [name for name in FINISH_PRODUCTS if (run_dir / name).exists()]

    print(f"run dir       : {run_dir}")
    print(f"thread        : {thread_id}")
    print(f"pending nodes : {list(pending)} (finished run)")
    print(f"max_turns     : {old_turns} -> {int(args.turns)}  (next turn to run: {turn})")
    print(f"state writes  : protocol.max_turns, runtime_overrides +1 line, stop={{}}, final_slow={{}}, report={{}}  as_node=reconcile")
    print(f"move aside    : {to_move or 'nothing'} -> {backup.name}/")
    print(f"snapshot file : {run_dir / 'protocol_snapshot.json'} rewritten with the merged protocol")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write these changes")
        return 0

    updates = {"protocol": merged, "stop": {}, "final_slow": {}, "report": {}}
    try:
        app.update_state(config, updates, as_node="reconcile")
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
    if to_move:
        backup.mkdir(parents=True, exist_ok=False)
        for name in to_move:
            shutil.move(str(run_dir / name), str(backup / name))
    (run_dir / "protocol_snapshot.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\napplied: pending nodes now {list(after.next)}; turns {turn}..{int(args.turns) - 1} will run on resume")
    print(
        f"LAUNCH WITH: TURNS={int(args.turns)} VALIDATE_FORMAL=0 RUN_TAG=<this dir's tag> — "
        f"the launcher default TURNS=30 would lower max_turns back and finalize again immediately"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
