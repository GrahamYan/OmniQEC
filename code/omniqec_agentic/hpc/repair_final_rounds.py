"""Repair the recorded round count of a finished campaign's final frontier.

Before 2026-09-29 report_node wrote the final stage's floor (T_c = 12) into
every final_frontier row, while the measurement itself ran max(T_c, d_search)
rounds per candidate (rounds_policy max_Tc_d). The artifact validator then
refused every campaign whose final contained a d > 12 code ("final row i:
rounds=12, expected 14"). This tool rewrites those rows from the final stage's
own rankings.json (the measured rounds), regenerates campaign_report.md and
the fixed_T_final_audit tables, and records what it changed in the report::

    python -m omniqec_agentic.hpc.repair_final_rounds <run_dir> [--apply]

Dry run by default; nothing else in the report is touched.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any


def measured_rounds_by_key(run_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    """(by code_hash, by program_id) -> measured rounds from every
    final_slow/T*/rankings.json. code_hash is the identity: program ids
    collide across turns (GPT b360's final carried two different codes named
    g0001_p0013, measured at 16 and 17 rounds), so a program_id is only
    usable when it names exactly one code."""

    by_hash: dict[str, int] = {}
    by_pid: dict[str, list[int]] = {}
    for path in sorted((run_dir / "final_slow").glob("T*/rankings.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else (raw.get("rows") or raw.get("rankings") or [])
        for row in rows:
            if not isinstance(row, dict) or not row.get("rounds"):
                continue
            if row.get("code_hash"):
                by_hash[str(row["code_hash"])] = int(row["rounds"])
            if row.get("program_id"):
                by_pid.setdefault(str(row["program_id"]), []).append(int(row["rounds"]))
    unique_pid = {pid: rounds[0] for pid, rounds in by_pid.items() if len(set(rounds)) == 1}
    return by_hash, unique_pid


def _measured_for(row: dict[str, Any], by_hash: dict[str, int], by_pid: dict[str, int]) -> int | None:
    code_hash = str(row.get("code_hash") or "")
    if code_hash:
        for full, rounds in by_hash.items():
            if full == code_hash or full.startswith(code_hash) or code_hash.startswith(full):
                return rounds
    pid = str(row.get("program_id") or "")
    return by_pid.get(pid) if pid else None


def plan(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads((run_dir / "campaign_report.json").read_text(encoding="utf-8"))
    by_hash, by_pid = measured_rounds_by_key(run_dir)
    changes: list[dict[str, Any]] = []
    for index, row in enumerate(report.get("final_frontier") or []):
        want_or_none = _measured_for(row, by_hash, by_pid)
        if want_or_none is None:
            continue
        have = int(row.get("rounds") or 0)
        want = int(want_or_none)
        if have != want:
            changes.append({"index": index, "program_id": row.get("program_id"), "from": have, "to": want})
    return report, changes


def apply(run_dir: Path, report: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    from omniqec_agentic.workflow.graph import _report_markdown, _write_channel_reports

    for change in changes:
        report["final_frontier"][int(change["index"])]["rounds"] = int(change["to"])
    history = list(report.get("final_rounds_repair") or [])
    history.append({
        "repaired_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "reason": "report_node wrote the final-stage floor instead of the measured max(T_c, d_search) rounds (fixed 2026-09-29)",
        "rows": changes,
    })
    report["final_rounds_repair"] = history
    (run_dir / "campaign_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "campaign_report.md").write_text(_report_markdown(report), encoding="utf-8")
    _write_channel_reports(run_dir, report)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    report, changes = plan(run_dir)
    if not changes:
        print(f"[repair] {run_dir.name}: final_frontier rounds already match the measurements")
        return 0
    for change in changes:
        print(f"[repair] {run_dir.name}: final row {change['index']} {change['program_id']}: rounds {change['from']} -> {change['to']}")
    if not args.apply:
        print("[repair] DRY RUN -- re-run with --apply to rewrite campaign_report.json/.md and fixed_T_final_audit.*")
        return 0
    apply(run_dir, report, changes)
    print(f"[repair] {run_dir.name}: {len(changes)} row(s) rewritten; repair recorded in campaign_report.json final_rounds_repair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
