"""ver1-style run analyzer for omniqec_agentic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# BB72 [[72,12,6]] under Q = k*d^2 (kd2_v1). Was 6.0 under ver1's k*d^2/n.
BB_Q = 432.0
ANALYZE_SCHEMA_VERSION = 1


def analyze_runs(run_dirs: list[str | Path], *, bb_q: float = BB_Q) -> list[dict[str, Any]]:
    return [summarize_run(Path(run_dir), bb_q=float(bb_q)) for run_dir in run_dirs]


def summarize_run(run_dir: Path, *, bb_q: float = BB_Q) -> dict[str, Any]:
    fast_dir = _fast_dir(run_dir)
    archive = _load_archive(fast_dir / "archive.json")
    attempts = _load_attempts(fast_dir / "transcript.jsonl")
    archive_rows = [_archive_metrics(record) for record in archive.values()]
    archive_rows = [row for row in archive_rows if row]
    all_slow_rows = _load_slow_candidate_rows(run_dir)
    baseline_rows = [row for row in all_slow_rows if _is_baseline_row(row)]
    slow_rows = [row for row in all_slow_rows if not row.get("is_baseline")]
    best = max(archive_rows, key=lambda row: float(row.get("Q") or 0.0), default=None)
    best_q = max((float(row.get("Q") or 0.0) for row in archive_rows), default=0.0)
    beat_bb = sum(1 for row in archive_rows if float(row.get("Q") or 0.0) > float(bb_q))
    verified_non_degraded = _verified_non_degraded_count(slow_rows)
    verified_non_degraded_bb = _verified_non_degraded_count(
        [row for row in slow_rows if float(row.get("fast_Q") or 0.0) > float(bb_q)]
    )
    total, valid, accepted = _attempt_yield(attempts)
    return {
        "schema": "omniqec_agentic_run_analysis",
        "schema_version": ANALYZE_SCHEMA_VERSION,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "fast_dir": str(fast_dir),
        "niches": len(archive_rows),
        "bestQ": best_q,
        "best_code": _code_label(best),
        "beatBB": beat_bb,
        "verified_non_degraded": verified_non_degraded,
        "verified_non_degraded_bb": verified_non_degraded_bb,
        "circuit_ler_ok": _slow_ler_ok_count(slow_rows),
        "beats_baseline_ler": _beats_baseline_ler_count(slow_rows, baseline_rows),
        "baseline_per_logical_per_round": _baseline_ler(baseline_rows),
        "total": total,
        "valid": valid,
        "accepted": accepted,
        "slow_candidate_count": len(slow_rows),
    }


def format_leaderboard(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'run':>28} {'niches':>7} {'bestQ':>7} {'best code':>16} "
        f"{'beatBB':>7} {'verif':>6} {'ler_ok':>6} {'beatsBB-LER':>11} {'valid/total':>12} {'accept':>7}"
    )
    lines = [header]
    for row in sorted(rows, key=lambda item: (-int(item["verified_non_degraded"]), -float(item["bestQ"]))):
        beats = row.get("beats_baseline_ler")
        lines.append(
            f"{row['run']:>28} {row['niches']:>7} {row['bestQ']:>7.2f} {row['best_code']:>16} "
            f"{row['beatBB']:>7} {row['verified_non_degraded']:>6} {row['circuit_ler_ok']:>6} "
            f"{str(beats) if beats is not None else '-':>11} "
            f"{str(row['valid']) + '/' + str(row['total']):>12} {row['accepted']:>7}"
        )
    return "\n".join(lines)


def _fast_dir(run_dir: Path) -> Path:
    if (run_dir / "archive.json").exists():
        return run_dir
    if (run_dir / "fast" / "archive.json").exists():
        return run_dir / "fast"
    return run_dir


def _load_archive(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("archive"), dict):
        return data["archive"]
    return data if isinstance(data, dict) else {}


def _load_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict) and row.get("kind") == "attempt":
            rows.append(row)
    return rows


def _archive_metrics(record: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    if "Q" in record:
        return {
            "n": record.get("n"),
            "k": record.get("k"),
            "d": record.get("d"),
            "Q": record.get("Q"),
            "program_id": record.get("program_id"),
        }
    fast = record.get("fast") if isinstance(record.get("fast"), dict) else {}
    structural = record.get("structural") if isinstance(record.get("structural"), dict) else {}
    if not fast:
        return None
    return {
        "n": fast.get("n_data") or fast.get("n") or structural.get("n_data"),
        "k": fast.get("k") or structural.get("k"),
        "d": fast.get("d_proxy") or fast.get("d"),
        "Q": fast.get("q_score") or fast.get("Q") or (fast.get("metrics") or {}).get("Q"),
        "program_id": fast.get("program_id") or (record.get("program") or {}).get("program_id"),
    }


def _attempt_yield(attempts: list[dict[str, Any]]) -> tuple[int, int, int]:
    total = len(attempts)
    valid = 0
    accepted = 0
    invalid_statuses = {"sandbox_error", "structural_reject", "fast_reject"}
    for row in attempts:
        if row.get("accepted"):
            accepted += 1
        if row.get("error"):
            continue
        status = str(row.get("archive_status") or "")
        if status and status in invalid_statuses:
            continue
        if row.get("failure_reasons"):
            continue
        valid += 1
    return total, valid, accepted


def _load_slow_candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in _slow_summary_paths(run_dir):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in summary.get("candidates") or []:
            if isinstance(row, dict):
                rows.append(row)
    return _dedupe_by_program(rows)


def _slow_summary_paths(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    cycle_path = run_dir / "agentic_cycle_summary.json"
    if cycle_path.exists():
        try:
            cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
        except Exception:
            cycle = {}
        summary_path = ((cycle.get("artifact_paths") or {}).get("slow_summary") or "")
        if summary_path:
            paths.append(Path(summary_path))
        for item in cycle.get("internal_promotions") or []:
            if isinstance(item, dict):
                nested = ((item.get("artifact_paths") or {}).get("slow_summary") or "")
                if nested:
                    paths.append(Path(nested))
    direct = run_dir / "slow_summary.json"
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted((run_dir / "slow").glob("**/slow_summary.json")) if (run_dir / "slow").exists() else [])
    existing: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        existing.append(path)
    return existing


def _dedupe_by_program(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_program: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("program_id") or row.get("code_hash") or len(by_program))
        current = by_program.get(key)
        if current is None or _slow_row_quality(row) > _slow_row_quality(current):
            by_program[key] = row
    return list(by_program.values())


def _slow_row_quality(row: dict[str, Any]) -> tuple[int, int, float]:
    status_score = 1 if row.get("status") == "ok" else 0
    estimate_score = 1 if _verified_distance_value(row) is not None else 0
    ler = row.get("per_logical_per_round")
    ler_score = -float(ler) if ler is not None else float("-inf")
    return status_score, estimate_score, ler_score


def _slow_ler_ok_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("status") == "ok")


_BB_BASELINE_HASHES: frozenset[str] | None = None


def _bb_baseline_hashes() -> frozenset[str]:
    """code_hashes of the BB72 and BB144 references (computed once, lazily)."""

    global _BB_BASELINE_HASHES
    if _BB_BASELINE_HASHES is None:
        hashes: set[str] = set()
        try:
            from .construction import build_matrices
            from .evaluation import BASELINE_BB144_SPEC, BASELINE_BB_SPEC
            from .verification.structural import code_hash

            for spec in (BASELINE_BB_SPEC, BASELINE_BB144_SPEC):
                hx, hz = build_matrices(spec)
                hashes.add(code_hash(hx, hz))
        except Exception:
            pass
        _BB_BASELINE_HASHES = frozenset(hashes)
    return _BB_BASELINE_HASHES


def _is_baseline_row(row: dict[str, Any]) -> bool:
    """Baseline = explicitly injected reference OR any candidate that IS a
    BB code (e.g. the BB seed elite, which dedups the injected baseline)."""

    if row.get("is_baseline"):
        return True
    return str(row.get("code_hash") or "") in _bb_baseline_hashes()


def _baseline_ler(baseline_rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["per_logical_per_round"])
        for row in baseline_rows
        if row.get("status") == "ok" and row.get("per_logical_per_round") is not None
    ]
    return min(values) if values else None


def _beats_baseline_ler_count(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> int | None:
    """Codes significantly better than the resident BB baseline.

    A win requires non-overlapping per-logical-block confidence intervals.
    Missing intervals are inconclusive and never fall back to point estimates.
    """

    baseline = None
    for row in baseline_rows:
        if row.get("status") == "ok" and row.get("per_logical_per_round") is not None:
            if baseline is None or float(row["per_logical_per_round"]) < float(baseline["per_logical_per_round"]):
                baseline = row
    if baseline is None:
        return None
    baseline_ci = baseline.get("per_logical_block_ci")
    count = 0
    for row in rows:
        if row.get("status") != "ok" or row.get("per_logical_per_round") is None:
            continue
        ci = row.get("per_logical_block_ci")
        if (
            isinstance(ci, (list, tuple))
            and len(ci) == 2
            and isinstance(baseline_ci, (list, tuple))
            and len(baseline_ci) == 2
            and float(ci[1]) < float(baseline_ci[0])
        ):
            count += 1
    return count


def _verified_non_degraded_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        estimate = _verified_distance_value(row)
        d_proxy = row.get("d_proxy")
        if estimate is None or d_proxy is None:
            continue
        if (
            row.get("degraded_estimate")
            or row.get("degraded_estimate_any_decoder")
            or row.get("degraded_isd")
        ):
            continue
        if int(estimate) >= int(d_proxy):
            count += 1
    return count


def _verified_distance_value(row: dict[str, Any]) -> int | None:
    exact = row.get("exact_d_circ")
    if exact is not None and row.get("exact_d_circ_status") == "ok":
        return int(exact)
    estimate = row.get("d_circ_ler_estimate")
    if estimate is not None and row.get("d_circ_ler_estimate_status") == "ok":
        return int(estimate)
    return None


def _code_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    return f"[[{row.get('n')},{row.get('k')},{row.get('d')}]]"
