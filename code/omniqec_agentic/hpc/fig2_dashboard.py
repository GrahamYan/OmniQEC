"""Fig.2 monitor: every campaign of one tree on one screen (English only).

    python -m omniqec_agentic.hpc.fig2_dashboard runs/                  # draw once
    python -m omniqec_agentic.hpc.fig2_dashboard runs/ --watch 120      # redraw every 120 s (tmux pane)
    python -m omniqec_agentic.hpc.fig2_dashboard runs/ --arm claudesub --budget 98 --top 5

Per run (runs/fig2_<arm>_b<budget>_<tag>): one header line -- turns done, the
phase the in-flight turn is in and for how long, the last completed turn's
wall minutes, tokens spent by the last turn and by the whole campaign, flags
-- followed by the TOP codes measured in the latest completed turn side by
side with the cumulative top codes of the run (each code's best measurement,
BB baseline excluded, `*` = campaign best). Q = k * d_search^2.

The phase is read from the milestone files of the in-flight turn directory
(plan.json -> generation responses -> fast summary -> slow summary -> audit),
the same way hpc/turn_timeline.py reconstructs a finished turn; nothing here
touches a checkpoint. Unless --no-merge is given, the tree-level
runs/fig2_all_{summary,progress,best}_<model>.csv are refreshed from the
per-run CSVs the campaign writes after every turn, so the merged files are
always current while the monitor runs -- restricted to --tag when one is
given, so a canary/variant chain in the same tree stays out of them.
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import glob
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from omniqec_agentic.progress_csv import collect_progress_rows
from omniqec_agentic.summary_csv import collect_best_rows

RUN_RE = re.compile(r"^fig2_(?P<arm>[A-Za-z0-9]+)_b(?P<budget>\d+)_(?P<tag>[A-Za-z0-9_.-]+)$")
_TURN_RE = re.compile(r"turn_(\d+)$")
# Phase reached once the named milestone exists; the run is IN the phase that
# follows the latest milestone present.
PHASES = ("orch", "gen", "fast", "slow", "wrap")


def _mtime(path: str | Path) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _turn_dirs(run_dir: Path) -> list[Path]:
    dirs = [p for p in (run_dir / "turns").glob("turn_*") if p.is_dir()]
    return sorted(dirs, key=lambda p: int(_TURN_RE.search(p.name).group(1)) if _TURN_RE.search(p.name) else -1)


def _fmt_tokens(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "-"
    if n >= 10_000_000:
        return f"{n / 1e6:.0f}M"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def _fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _fmt_plpr(value: Any) -> str:
    try:
        return f"{float(value):.1e}"
    except (TypeError, ValueError):
        return "-"


def phase_of(run_dir: Path, now: float | None = None) -> tuple[str, float | None]:
    """(phase name, seconds spent in it) for the run's current state.

    done / final / waiting are states between turns; orch/gen/fast/slow/wrap
    are the phases of an in-flight turn (a turn directory without an audit).
    """

    now = time.time() if now is None else now
    if (run_dir / "campaign_report.json").exists():
        return "done", None
    if (run_dir / "FATAL_STOP.json").exists():
        return "FATAL", None
    turns = _turn_dirs(run_dir)
    if (run_dir / "final_slow").is_dir() and (not turns or (turns[-1] / "turn_audit.json").exists()):
        started = _mtime(run_dir / "final_slow")
        return "final", (now - started) if started else None
    if not turns:
        started = _mtime(run_dir)
        return "starting", (now - started) if started else None
    latest = turns[-1]
    audit = _mtime(latest / "turn_audit.json")
    if audit:
        return "waiting", now - audit
    milestones = [
        ("orch", _mtime(latest)),
        ("gen", _mtime(latest / "plan.json")),
    ]
    gen_files = [
        p for p in glob.glob(str(latest / "cycle" / "fast" / "generations" / "**" / "*"), recursive=True)
        if os.path.isfile(p)
    ]
    milestones.append(("fast", max((_mtime(p) or 0) for p in gen_files) if gen_files else None))
    # funsearch_summary.json is written once, when the whole fast loop is
    # done; status.json is rewritten after EVERY generation and must not be
    # read as "fast done" (it made a 4-generation turn look like slow).
    fast_done = _mtime(latest / "cycle" / "fast" / "funsearch_summary.json")
    milestones.append(("slow", fast_done))
    milestones.append(("wrap", _mtime(latest / "cycle" / "slow" / "slow_summary.json")))
    phase, since = "orch", milestones[0][1]
    for name, stamp in milestones:
        if stamp is not None:
            phase, since = name, stamp
    if phase in ("gen", "fast"):
        # Generation progress: g0001.. directories that already hold a raw
        # LLM response, over the snapshot's generations_per_turn.
        gens = sorted(glob.glob(str(latest / "cycle" / "fast" / "generations" / "g*" / "llm_raw.jsonl")))
        total = _load_json(run_dir / "protocol_snapshot.json").get("generations_per_turn")
        if gens or total:
            phase = f"gen {len(gens)}/{total or '?'}"
    return phase, (now - since) if since else None


def _codes_from_slow_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "ok":
                    continue
                if str(raw.get("program_id") or "").startswith("baseline_"):
                    continue
                try:
                    plpr = float(raw.get("per_logical_per_round") or "")
                except ValueError:
                    continue
                rows.append({
                    "n": raw.get("n_data"), "k": raw.get("k"),
                    "d": raw.get("d_search") or raw.get("d_proxy"),
                    "family": raw.get("family") or "?", "plpr": plpr,
                    "ci_low": raw.get("per_logical_per_round_ci_low"),
                    "ci_high": raw.get("per_logical_per_round_ci_high"),
                    "hash8": str(raw.get("code_hash") or raw.get("program_id") or "")[:8],
                })
    except OSError:
        return []
    return sorted(rows, key=lambda r: r["plpr"])


def turn_codes(turn_dir: Path) -> list[dict[str, Any]]:
    """Codes measured in one turn (best first), baseline excluded."""

    out: list[dict[str, Any]] = []
    for csv_path in sorted(turn_dir.rglob("slow_results.csv")):
        out.extend(_codes_from_slow_csv(csv_path))
    return sorted(out, key=lambda r: r["plpr"])


def cumulative_codes(run_dir: Path) -> list[dict[str, Any]]:
    """Best measurement per code over the whole run (summary_csv order), baseline excluded."""

    out: list[dict[str, Any]] = []
    for row in collect_best_rows(run_dir):
        if str(row.get("program_id") or "").startswith("baseline_"):
            continue
        out.append({
            "n": row.get("n_data"), "k": row.get("k"), "d": row.get("d_search_proxy"),
            "family": row.get("family") or "?", "plpr": float(row["plpr"]),
            "ci_low": row.get("ci_low"), "ci_high": row.get("ci_high"),
            "hash8": row.get("code_hash8") or "", "turn": row.get("turn"),
        })
    return out


def _q(code: dict[str, Any]) -> str:
    try:
        return str(int(code["k"]) * int(code["d"]) ** 2)
    except (TypeError, ValueError, KeyError):
        return "-"


def _code_cell(code: dict[str, Any] | None, *, with_turn: bool, best_hash: str = "") -> str:
    if code is None:
        return ""
    name = f"[[{code.get('n')},{code.get('k')},{code.get('d')}]]"
    ci = f"[{_fmt_plpr(code.get('ci_low'))},{_fmt_plpr(code.get('ci_high'))}]"
    parts = [f"{name:<14}", f"{str(code.get('family'))[:8]:<8}", f"{_fmt_plpr(code['plpr']):>7}", f"{ci:<17}", f"Q{_q(code):<5}"]
    if with_turn:
        parts.append(f"t{code.get('turn')}")
    parts.append(str(code.get("hash8") or ""))
    if best_hash and code.get("hash8") == best_hash:
        parts.append("*")
    return " ".join(parts)


def run_report(run_dir: Path, *, top: int = 5, now: float | None = None) -> dict[str, Any]:
    match = RUN_RE.match(run_dir.name)
    arm, budget, tag = (match.group("arm"), match.group("budget"), match.group("tag")) if match else (run_dir.name, "?", "?")
    snapshot = _load_json(run_dir / "protocol_snapshot.json")
    max_turns = snapshot.get("max_turns")
    turns = _turn_dirs(run_dir)
    completed = [t for t in turns if (t / "turn_audit.json").exists()]
    try:
        progress, _ = collect_progress_rows(run_dir)
    except Exception:
        progress = []
    done_rows = [r for r in progress if r.get("audit")]
    last = done_rows[-1] if done_rows else (progress[-1] if progress else {})
    phase, in_phase = phase_of(run_dir, now)
    flags: list[str] = []
    if (run_dir / "VALIDATED.json").exists():
        flags.append("VALIDATED")
    if (run_dir / "FATAL_STOP.json").exists():
        flags.append("FATAL")
    infeasible = _load_json(run_dir / "decode_infeasible_hashes.json")
    n_inf = len(infeasible.get("hashes") or infeasible.get("decode_infeasible_hashes") or []) if infeasible else 0
    if n_inf:
        flags.append(f"infeasible={n_inf}")
    latest_codes = turn_codes(completed[-1]) if completed else []
    cumulative = cumulative_codes(run_dir)
    return {
        "arm": arm, "budget": int(budget) if budget.isdigit() else budget, "tag": tag,
        "model": snapshot.get("required_model") or "?",
        "turns_done": len(completed), "max_turns": max_turns,
        "phase": phase, "in_phase_s": in_phase,
        "last_minutes": last.get("minutes"), "last_turn": last.get("turn"),
        "tok_last": last.get("tok_total"), "tok_cum": last.get("tok_cum"),
        "flags": flags,
        "turn_top": latest_codes[:top], "cum_top": cumulative[:top],
        "best_hash": cumulative[0]["hash8"] if cumulative else "",
    }


def render_run(report: dict[str, Any], *, top: int = 5) -> list[str]:
    head = (
        f"{report['arm']:<9} b{report['budget']:<5} "
        f"{report['turns_done']}/{report['max_turns'] or '?'} turns  "
        f"phase {report['phase']}"
        + (f" {_fmt_minutes(report['in_phase_s'])}" if report["in_phase_s"] is not None else "")
        + "  "
        f"last t{report['last_turn'] if report['last_turn'] is not None else '-'} "
        f"{report['last_minutes'] if report['last_minutes'] is not None else '-'}min  "
        f"tok last {_fmt_tokens(report['tok_last'])} cum {_fmt_tokens(report['tok_cum'])}  "
        f"model {report['model']}"
        + (f"  [{' '.join(report['flags'])}]" if report["flags"] else "")
    )
    lines = [head]
    left_title = f"  latest turn top-{top}"
    right_title = f"cumulative top-{top}"
    width = 72
    lines.append(f"{left_title:<{width}}| {right_title}")
    for i in range(top):
        left = report["turn_top"][i] if i < len(report["turn_top"]) else None
        right = report["cum_top"][i] if i < len(report["cum_top"]) else None
        if left is None and right is None:
            break
        lcell = "  " + _code_cell(left, with_turn=False) if left else ""
        rcell = _code_cell(right, with_turn=True, best_hash=report["best_hash"]) if right else ""
        lines.append(f"{lcell:<{width}}| {rcell}".rstrip())
    return lines


def discover(runs_dir: Path, *, tag: str | None, arm: str | None, budget: int | None) -> list[Path]:
    out: list[tuple[str, int, Path]] = []
    for path in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        match = RUN_RE.match(path.name)
        if not path.is_dir() or not match:
            continue
        if tag and match.group("tag") != tag:
            continue
        if arm and match.group("arm") != arm:
            continue
        if budget is not None and int(match.group("budget")) != int(budget):
            continue
        out.append((match.group("arm"), int(match.group("budget")), path))
    out.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in out]


def render(runs_dir: Path, *, tag: str | None = None, arm: str | None = None, budget: int | None = None,
           top: int = 5, now: float | None = None) -> str:
    stamp = datetime.fromtimestamp(now or time.time()).strftime("%Y-%m-%d %H:%M")
    lines = [f"OmniQEC Fig.2 monitor  tree={runs_dir.resolve().parent}  tag={tag or 'all'}  {stamp}"]
    runs = discover(runs_dir, tag=tag, arm=arm, budget=budget)
    if not runs:
        lines.append("no runs found (expected runs/fig2_<arm>_b<budget>_<tag>/)")
        return "\n".join(lines) + "\n"
    for run_dir in runs:
        lines.append("")
        lines.extend(render_run(run_report(run_dir, top=top, now=now), top=top))
    return "\n".join(lines) + "\n"


def _merge_tree(runs_dir: Path, *, tag: str | None = None) -> str:
    """Refresh runs/fig2_all_*.csv from the same runs the monitor draws: with
    --tag only that site, so a canary/variant chain in the tree never lands in
    the formal tables (2026-09-12: GPT b60 canary2 rows showed up as extra
    turns of the formal GPT b60 point)."""

    try:
        from omniqec_agentic.hpc.merge_fig2_csv import merge

        merge(runs_dir, refresh=False, site=tag)
        scope = f" (site={tag})" if tag else ""
        return "merged: runs/fig2_all_{summary,progress,best}_<model>.csv refreshed" + scope
    except Exception as exc:  # the monitor must keep drawing even if a merge fails
        return f"merge skipped: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--tag", default=None, help="only runs with this RUN_TAG (default: all)")
    parser.add_argument("--arm", default=None, help="only this arm prefix (claudesub, gptsub, deepseek)")
    parser.add_argument("--budget", type=int, default=None, help="only this budget point")
    parser.add_argument("--top", type=int, default=5, help="codes per column (default 5)")
    parser.add_argument("--watch", type=int, default=0, help="redraw every N seconds (0 = once)")
    parser.add_argument("--no-merge", action="store_true", help="do not refresh runs/fig2_all_*.csv")
    args = parser.parse_args(argv)
    while True:
        text = render(args.runs_dir, tag=args.tag, arm=args.arm, budget=args.budget, top=args.top)
        note = "" if args.no_merge else _merge_tree(args.runs_dir, tag=args.tag)
        if args.watch:
            sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(text)
        if note:
            sys.stdout.write("\n" + note + "\n")
        sys.stdout.flush()
        if not args.watch:
            return 0
        time.sleep(max(5, int(args.watch)))


if __name__ == "__main__":
    raise SystemExit(main())
