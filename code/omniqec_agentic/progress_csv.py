"""Per-turn campaign progress export: one row per turn, oldest first.

`summary_csv` answers "which codes are best". This answers "how is the run
going" — the mid-campaign question, asked while most turns are still ahead:
did each turn pass its strict audit, what did the policy ask for, how many
codes actually got measured, and is the best plpr still moving.

Usage:
    python -m omniqec_agentic.progress_csv <run_dir> [-o out.csv]

Re-run any time (reads finished artifacts only; safe while a campaign runs).
The `audit` column is only populated for runs launched with --strict-turns.
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from omniqec_agentic.memory.bandit import STRATEGY_ARMS

BASE_COLUMNS = (
    "turn", "audit", "cycle", "policy", "finished_at", "minutes",
    "proposals", "promotions", "explore_temp", "tools", "orch_llm_calls",
    # What a turn spent (2026-09-10: the input/output/reasoning/cached split
    # is exported too, so cost can be priced from this file alone; reasoning
    # is a subset of output, tok_total = input + output, tok_cum = running
    # campaign total). The per-role split stays in token_usage.json.
    "tok_in", "tok_out", "tok_reasoning", "tok_cached", "tok_total", "tok_cum",
    "cost_usd",
)
TAIL_COLUMNS = (
    "reflection", "gen_notes",
    "measured", "censored", "families_measured",
    "best_plpr_turn", "best_plpr_cum", "vs_bb_cum",
    "new_hof", "evidence_rows", "insight_precision",
    "backtest_supported", "backtest_refuted",
)

_TURN_RE = re.compile(r"turn_(\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_lines(path: Path) -> int | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _slow_stats(turn_dir: Path) -> dict[str, Any]:
    """Measured/censored counts and the best plpr among this turn's rows.

    The BB baseline is re-measured every turn and would otherwise win `best`
    outright; excluding it is what makes vs_bb read as "have we beaten BB yet".
    """

    measured = censored = 0
    best: float | None = None
    families: dict[str, int] = {}
    for csv_path in sorted(turn_dir.rglob("slow_results.csv")):
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "ok":
                    continue
                if str(raw.get("program_id") or "").startswith("baseline_"):
                    continue
                measured += 1
                if raw.get("stopped_by") == "racing":
                    censored += 1
                family = raw.get("family") or "?"
                families[family] = families.get(family, 0) + 1
                plpr = _float(raw.get("per_logical_per_round"))
                if plpr is not None and (best is None or plpr < best):
                    best = plpr
    return {
        "measured": measured,
        "censored": censored,
        "families_measured": " ".join(
            f"{name}:{count}" for name, count in sorted(families.items())
        ),
        "best_plpr_turn": best,
    }


def collect_progress_rows(run_dir: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    turn_dirs = sorted(
        (p for p in (run_dir / "turns").glob("turn_*") if p.is_dir()),
        key=lambda p: int(_TURN_RE.search(p.name).group(1)) if _TURN_RE.search(p.name) else -1,
    )
    seen_families: list[str] = []
    rows: list[dict[str, Any]] = []
    best_cum: float | None = None
    baseline_plpr: float | None = None
    previous_end: datetime | None = None
    tok_cum = 0

    for turn_dir in turn_dirs:
        match = _TURN_RE.search(turn_dir.name)
        if match is None:
            continue
        plan = _load_json(turn_dir / "plan.json")
        digest = _load_json(turn_dir / "digest.json")
        audit = _load_json(turn_dir / "turn_audit.json")
        tokens = (_load_json(turn_dir / "token_usage.json").get("total") or {})
        reflection = _load_json(turn_dir / "reflection.json")
        notes = _load_json(turn_dir / "cycle" / "fast" / "generation_notes.json")
        learning = digest.get("learning") or {}

        if baseline_plpr is None:
            baseline_plpr = _float((digest.get("baseline") or {}).get("per_logical_per_round"))

        stats = _slow_stats(turn_dir)
        turn_best = stats["best_plpr_turn"]
        tok_cum += int(tokens.get("total_tokens") or 0)
        if turn_best is not None and (best_cum is None or turn_best < best_cum):
            best_cum = turn_best

        # A turn's wall-clock is the gap since the previous turn closed; the
        # digest is written last, so its mtime is that closing timestamp.
        digest_path = turn_dir / "digest.json"
        stamp = digest_path if digest_path.exists() else turn_dir
        end = datetime.fromtimestamp(stamp.stat().st_mtime)
        minutes = round((end - previous_end).total_seconds() / 60.0, 1) if previous_end else None
        previous_end = end

        family_mix = plan.get("family_mix") or {}
        for family in family_mix:
            if family not in seen_families:
                seen_families.append(family)
        strategy_mix = plan.get("strategy_mix") or {}

        row: dict[str, Any] = {
            "turn": int(match.group(1)),
            "audit": audit.get("status", ""),
            "cycle": digest.get("cycle_status", ""),
            "policy": plan.get("policy", ""),
            "finished_at": end.strftime("%m-%d %H:%M"),
            "minutes": minutes,
            "proposals": plan.get("proposal_count"),
            "promotions": len(plan.get("promotions") or []),
            "explore_temp": plan.get("explore_temperature"),
            "tools": plan.get("tool_calls_used"),
            "orch_llm_calls": _count_lines(turn_dir / "orchestrator" / "llm_calls.jsonl"),
            "tok_in": tokens.get("input"),
            "tok_out": tokens.get("output"),
            "tok_reasoning": tokens.get("reasoning"),
            "tok_cached": tokens.get("cache_hit"),
            "tok_total": tokens.get("total_tokens"),
            "tok_cum": tok_cum if tokens else None,
            "cost_usd": tokens.get("cost_usd"),
            "reflection": reflection.get("status", ""),
            "gen_notes": len(notes.get("notes") or []) if notes else None,
            "measured": stats["measured"],
            "censored": stats["censored"],
            "families_measured": stats["families_measured"],
            "best_plpr_turn": turn_best,
            "best_plpr_cum": best_cum,
            "vs_bb_cum": (
                round(best_cum / baseline_plpr, 3)
                if best_cum is not None and baseline_plpr else None
            ),
            "new_hof": len(learning.get("new_hof") or []),
            "evidence_rows": learning.get("evidence_rows_total"),
            "insight_precision": (learning.get("insight_precision") or {}).get("precision"),
            "backtest_supported": (learning.get("backtest") or {}).get("supported"),
            "backtest_refuted": (learning.get("backtest") or {}).get("refuted"),
        }
        for family, weight in family_mix.items():
            row[f"mix_{family}"] = weight
        for arm in STRATEGY_ARMS:
            row[f"strat_{arm}"] = strategy_mix.get(arm)
        rows.append(row)

    columns = (
        BASE_COLUMNS
        + tuple(f"mix_{family}" for family in seen_families)
        + tuple(f"strat_{arm}" for arm in STRATEGY_ARMS)
        + TAIL_COLUMNS
    )
    return rows, columns


def write_progress_csv(run_dir: Path, out: Path | None = None) -> tuple[Path, int]:
    """Export <run_dir>/progress.csv atomically (tmp + rename), so a monitor
    tailing the file never sees a half-written table. Called by the campaign
    after every turn (workflow.graph) and by the CLI below."""

    run_dir = Path(run_dir)
    rows, columns = collect_progress_rows(run_dir)
    out = Path(out) if out is not None else (run_dir / "progress.csv")
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(out)
    return out, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    rows, columns = collect_progress_rows(args.run_dir)
    out = args.out or (args.run_dir / "progress.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"{len(rows)} turns -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
