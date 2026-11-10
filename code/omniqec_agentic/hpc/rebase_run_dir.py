"""Rebase the absolute paths frozen into a campaign run after moving it.

WHY. ``run-campaign --out <ABS>`` freezes that absolute path in TWO places:

1. **The LangGraph checkpoint state** (``run_dir``, ``slow_context_ptr``,
   every entry of ``last_cycle.artifact_paths``, and — for runs cut during the
   report phase — ``final_slow.runs[].artifact_paths``). ``--resume`` restores
   the state verbatim (``app.invoke(None)``), so a moved run resumes pointing
   at the OLD location. The state is msgpack inside sqlite: the rewrite must
   go through the LangGraph API — never through sed.
2. **On-disk artifacts**: real (non-relative-``--out``) runs embed absolute
   paths in plain text artifacts — ``rankings.json`` rows carry
   ``candidate_path`` (evaluation/writeback.py), ``slow_results.csv`` an
   ``artifact_path`` column, ``agentic_cycle_summary.json`` its
   ``artifact_paths``, ``turn_audit.json`` a ``digest_path``. Most are
   historical records, but ``candidate_path`` is LOAD-BEARING on one resume
   window: a checkpoint whose pending node includes ``reconcile`` (or
   ``final_slow``/``report``) re-reads those rows, and
   ``workflow/graph.py::_spec_from_candidate`` returns ``spec=None`` for a
   path that no longer exists — the turn's hall-of-fame entrants then drop out
   of final_slow SILENTLY (cross-audited with the GPT-arm session,
   2026-08-31; an earlier "artifacts are clean" reading came from a run
   started with a RELATIVE --out, where every embedded path is relative).

This tool therefore runs two passes: the checkpoint-state rewrite (LangGraph
API) and a plain-text prefix rewrite over ``*.json/*.jsonl/*.csv/*.md`` under
the run dir (``protocol_snapshot.json`` excluded; it holds no paths, and the
frozen protocol is nothing to touch). ``checkpoint.sqlite3*`` is never text-
edited.

USAGE (after moving/copying the run dir to its new location):

    python -m omniqec_agentic.hpc.rebase_run_dir --run-dir /new/abs/path/runs/fig2_claudesub_b98_nscc
    python -m omniqec_agentic.hpc.rebase_run_dir --run-dir ... --apply

Dry-run by default: prints everything it would change and touches nothing
until ``--apply``. The old prefix defaults to the ``run_dir`` stored in the
checkpoint; if the state was already rebased and only the artifact pass is
missing, pass ``--old-prefix`` explicitly. The tool aborts if the rewrite
would alter the graph's pending-node position (``snapshot.next``), and it
refuses to run at all if the frozen ``protocol`` state key contains the old
prefix — that would mean resume was broken before any move, and silently
editing the protocol is exactly what ``validate_resume_protocol`` exists to
prevent.

After ``--apply``, resume exactly as before, with ``--out`` pointing at the
NEW location and all other flags unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from omniqec_agentic.orchestrator import FallbackPolicy
from omniqec_agentic.workflow.checkpoint import CHECKPOINT_FILENAME, open_checkpointer
from omniqec_agentic.workflow.graph import build_campaign_graph

ARTIFACT_SUFFIXES = {".json", ".jsonl", ".csv", ".md"}
ARTIFACT_EXCLUDE = {"protocol_snapshot.json"}


def _rewrite(obj: Any, old: str, new: str, hits: list[tuple[str, str]], crumb: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new, hits, f"{crumb}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, old, new, hits, f"{crumb}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, str) and count_prefix(obj, old, new):
        hits.append((crumb, obj))
        return replace_prefix(obj, old, new)
    return obj


def _prefix_pattern(old: str, new: str) -> "re.Pattern[str]":
    """``old`` as a path prefix, but never the copy of ``old`` that sits inside
    ``new`` when ``new`` starts with ``old`` (…/fig2_x_v36 -> …/fig2_x_v36x60):
    a plain str.replace would count every rewritten path as a leftover and a
    second pass would turn it into …_v36x60x60."""
    if new.startswith(old) and len(new) > len(old):
        return re.compile(re.escape(old) + "(?!" + re.escape(new[len(old):]) + ")")
    return re.compile(re.escape(old))


def count_prefix(text: str, old: str, new: str) -> int:
    return len(_prefix_pattern(old, new).findall(text))


def replace_prefix(text: str, old: str, new: str) -> str:
    return _prefix_pattern(old, new).sub(lambda _m: new, text)


def _thread_ids(db_path: Path) -> list[str]:
    con = sqlite3.connect(str(db_path))
    try:
        return [row[0] for row in con.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    finally:
        con.close()


def _scan_artifacts(run_dir: Path, old: str, new: str) -> list[tuple[Path, int]]:
    found: list[tuple[Path, int]] = []
    for path in sorted(run_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in ARTIFACT_SUFFIXES
            or path.name in ARTIFACT_EXCLUDE
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        count = count_prefix(text, old, new)
        if count:
            found.append((path, count))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--run-dir",
        required=True,
        help="the run directory at its NEW location (must contain checkpoint.sqlite3)",
    )
    parser.add_argument(
        "--old-prefix",
        default=None,
        help="path prefix to replace (default: the run_dir stored in the checkpoint; "
        "pass explicitly when the state was already rebased and only artifacts remain)",
    )
    parser.add_argument(
        "--new-prefix",
        default=None,
        help="replacement prefix (default: the absolute path of --run-dir)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the rewritten state and artifacts back (default: dry run, print only)",
    )
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

    old_prefix = args.old_prefix or str(values.get("run_dir") or "")
    new_prefix = args.new_prefix or str(run_dir)
    if not old_prefix:
        print("checkpoint has no run_dir and --old-prefix was not given", file=sys.stderr)
        return 2
    if old_prefix == new_prefix:
        print(
            f"stored run_dir already points at {new_prefix} — state pass is a no-op.\n"
            f"If artifacts still carry an older prefix, re-run with --old-prefix <old>."
        )
        return 0

    print(f"thread        : {thread_id}")
    print(f"pending nodes : {list(snapshot.next)}")
    print(f"old prefix    : {old_prefix}")
    print(f"new prefix    : {new_prefix}")
    risky = {"reconcile", "final_slow", "report"} & set(snapshot.next)
    if risky:
        print(
            f"NOTE: pending nodes {sorted(risky)} re-read artifact rows on resume — "
            f"the artifact pass below is load-bearing here, not cosmetic"
        )

    # Pass 1: checkpoint state.
    updates: dict[str, Any] = {}
    state_hits: list[tuple[str, str]] = []
    for key, value in values.items():
        hits: list[tuple[str, str]] = []
        rewritten = _rewrite(value, old_prefix, new_prefix, hits, key)
        if not hits:
            continue
        if key == "protocol":
            print(
                f"REFUSING: the frozen protocol contains the old prefix at "
                f"{[crumb for crumb, _ in hits]} — resume was already broken "
                f"before the move; fix that by hand first",
                file=sys.stderr,
            )
            return 3
        updates[key] = rewritten
        state_hits.extend(hits)

    if updates:
        print(f"[state] strings to rewrite: {len(state_hits)} across keys {sorted(updates)}")
        for crumb, value in state_hits:
            print(f"  {crumb}: {value}")
    else:
        print("[state] no state field contains the old prefix")

    # Pass 2: plain-text artifacts.
    artifact_hits = _scan_artifacts(run_dir, old_prefix, new_prefix)
    if artifact_hits:
        total = sum(count for _, count in artifact_hits)
        print(f"[artifacts] {total} occurrence(s) across {len(artifact_hits)} file(s):")
        for path, count in artifact_hits:
            print(f"  {path.relative_to(run_dir)}: {count}")
    else:
        print("[artifacts] no artifact file contains the old prefix")

    if not updates and not artifact_hits:
        print("nothing to rewrite")
        return 0

    if not args.apply:
        print("\nDRY RUN — re-run with --apply to write these changes")
        return 0

    if updates:
        # Attribute the update to the node that wrote the last checkpoint so
        # the set of pending nodes (what resume executes next) is unchanged.
        as_node = None
        metadata = snapshot.metadata or {}
        writes = metadata.get("writes")
        if isinstance(writes, dict) and len(writes) == 1:
            as_node = next(iter(writes))
        try:
            if as_node is not None:
                app.update_state(config, updates, as_node=as_node)
            else:
                app.update_state(config, updates)
        except Exception as exc:  # noqa: BLE001 — surface langgraph's own message
            print(f"update_state failed: {exc}", file=sys.stderr)
            return 4
        after = app.get_state(config)
        if list(after.next) != list(snapshot.next):
            print(
                f"WARNING: pending nodes changed {list(snapshot.next)} -> {list(after.next)}; "
                f"inspect before resuming",
                file=sys.stderr,
            )
            return 5
        print(f"[state] applied — run_dir now {json.dumps(after.values.get('run_dir'))}, "
              f"pending nodes unchanged {list(after.next)}")

    for path, _ in artifact_hits:
        text = path.read_text(encoding="utf-8")
        path.write_text(replace_prefix(text, old_prefix, new_prefix), encoding="utf-8")
    if artifact_hits:
        print(f"[artifacts] rewrote {len(artifact_hits)} file(s)")

    leftovers = _scan_artifacts(run_dir, old_prefix, new_prefix)
    if leftovers:
        print(f"WARNING: {len(leftovers)} file(s) still carry the old prefix", file=sys.stderr)
        return 5
    print("done — resume with --out pointing at the new location, flags unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
