"""hpc/fig2_dashboard.py (2026-09-10): the one-pane Fig.2 monitor, plus the
per-turn CSV exports the campaign now writes for it (progress.csv with the
token split, summary.csv with tok_turn/tok_cum, both refreshed by
workflow.graph after every turn)."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from omniqec_agentic.hpc import fig2_dashboard as dash
from omniqec_agentic.progress_csv import collect_progress_rows, write_progress_csv
from omniqec_agentic.summary_csv import SUMMARY_COLUMNS, collect_best_rows, write_summary_csv
from omniqec_agentic.workflow.graph import _export_run_csvs

_SLOW_COLUMNS = (
    "program_id", "code_hash", "family", "status", "n_data", "n_checks_x", "n_checks_z", "k",
    "d_search", "decoder", "per_logical_per_round", "per_logical_per_round_ci_low",
    "per_logical_per_round_ci_high", "block_ler", "total_failures", "total_shots", "p", "rounds",
)
_CJK = re.compile(r"[　-鿿]")


def _usage(total: int, *, inp: int, out: int, reasoning: int = 0, cached: int = 0) -> dict:
    return {"schema": "turn_token_usage", "by_role": {}, "total": {
        "calls": 3, "input": inp, "output": out, "reasoning": reasoning, "cache_hit": cached,
        "cost_usd": 0.0, "reasoning_reported_calls": 3, "reasoning_complete": True,
        "total_tokens": total}}


def _code(pid, h, fam, n, k, d, plpr, status="ok"):
    return {"program_id": pid, "code_hash": h * 16, "family": fam, "status": status, "n_data": n,
            "n_checks_x": n // 2, "n_checks_z": n // 2, "k": k, "d_search": d, "decoder": "telescoping_cpu",
            "per_logical_per_round": plpr, "per_logical_per_round_ci_low": plpr * 0.9,
            "per_logical_per_round_ci_high": plpr * 1.1, "block_ler": 0.05, "total_failures": 300,
            "total_shots": 50000, "p": 0.005, "rounds": 12}


def _turn(run_dir: Path, turn: int, codes, *, tokens: dict, complete: bool = True, stamp: float | None = None):
    tdir = run_dir / "turns" / f"turn_{turn:04d}"
    (tdir / "cycle" / "slow").mkdir(parents=True, exist_ok=True)
    (tdir / "orchestrator").mkdir(parents=True, exist_ok=True)
    (tdir / "plan.json").write_text(json.dumps({"policy": "agentic_v1", "proposal_count": 24, "promotions": []}), encoding="utf-8")
    if codes is not None:
        with (tdir / "cycle" / "slow" / "slow_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SLOW_COLUMNS)
            writer.writeheader()
            for row in codes:
                writer.writerow(row)
    (tdir / "token_usage.json").write_text(json.dumps(tokens), encoding="utf-8")
    if complete:
        (tdir / "digest.json").write_text(json.dumps({"turn": turn, "cycle_status": "ok", "baseline": {"per_logical_per_round": 4e-4}}), encoding="utf-8")
        (tdir / "turn_audit.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    if stamp is not None:
        for path in tdir.rglob("*"):
            import os
            os.utime(path, (stamp, stamp))
        import os
        os.utime(tdir, (stamp, stamp))
    return tdir


def _run(tmp_path: Path, name="fig2_claudesub_b98_v36") -> Path:
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "protocol_snapshot.json").write_text(json.dumps({"max_turns": 30, "required_model": "claude-fable-5-1"}), encoding="utf-8")
    now = time.time()
    _turn(run_dir, 0, [
        _code("baseline_bb72", "00", "bb", 72, 12, 6, 4.0e-4),
        _code("g0001_p0001", "aa", "bb", 48, 4, 6, 9.0e-4),
        _code("g0001_p0002", "bb", "lp_abelian", 48, 4, 6, 7.0e-4),
        _code("g0001_p0003", "cc", "hgp", 40, 2, 4, 5.0e-3),
    ], tokens=_usage(300_000, inp=250_000, out=50_000, reasoning=20_000, cached=100_000), stamp=now - 7200)
    _turn(run_dir, 1, [
        _code("g0001_p0004", "dd", "bb", 48, 4, 7, 3.6e-4),
        _code("g0001_p0005", "ee", "lp_nonabelian", 48, 2, 6, 4.1e-4),
        _code("g0001_p0006", "ff", "bb", 48, 4, 8, 2.0e-3),
        _code("g0001_p0007", "99", "hgp", 40, 2, 4, 5.0e-3, status="decode_infeasible"),
    ], tokens=_usage(412_000, inp=340_000, out=72_000, reasoning=30_000, cached=120_000), stamp=now - 3600)
    # Turn 2 is in flight: plan written, no generation yet -> phase "gen".
    _turn(run_dir, 2, None, tokens=_usage(0, inp=0, out=0), complete=False, stamp=now - 600)
    return run_dir


def test_progress_rows_carry_the_token_split_and_running_total(tmp_path) -> None:
    run_dir = _run(tmp_path)
    rows, columns = collect_progress_rows(run_dir)
    assert ("tok_in", "tok_out", "tok_reasoning", "tok_cached", "tok_total", "tok_cum") == tuple(
        c for c in columns if c.startswith("tok_"))
    assert [r["tok_total"] for r in rows[:2]] == [300_000, 412_000]
    assert [r["tok_cum"] for r in rows[:2]] == [300_000, 712_000]
    assert rows[1]["tok_in"] == 340_000 and rows[1]["tok_out"] == 72_000
    assert rows[1]["tok_reasoning"] == 30_000 and rows[1]["tok_cached"] == 120_000
    out, n = write_progress_csv(run_dir)
    assert out == run_dir / "progress.csv" and n == len(rows)
    with out.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "tok_cum" in header and "cost_usd" in header
    assert not (run_dir / "progress.csv.tmp").exists()


def test_summary_rows_carry_tok_turn_and_tok_cum_before_spec(tmp_path) -> None:
    run_dir = _run(tmp_path)
    rows = collect_best_rows(run_dir)
    assert SUMMARY_COLUMNS[-1] == "spec" and SUMMARY_COLUMNS[-3:-1] == ("tok_turn", "tok_cum")
    by_hash = {r["code_hash8"]: r for r in rows}
    assert by_hash["dd" * 4]["tok_turn"] == 412_000 and by_hash["dd" * 4]["tok_cum"] == 712_000
    assert by_hash["aa" * 4]["tok_turn"] == 300_000 and by_hash["aa" * 4]["tok_cum"] == 300_000
    out, n = write_summary_csv(run_dir)
    assert out == run_dir / "summary.csv" and n == len(rows)


def test_campaign_export_hook_writes_both_csvs_and_never_raises(tmp_path, capsys) -> None:
    run_dir = _run(tmp_path)
    _export_run_csvs(run_dir, where="turn 1")
    assert (run_dir / "progress.csv").exists() and (run_dir / "summary.csv").exists()
    # A directory that is not a run yields a note, not an exception.
    _export_run_csvs(tmp_path / "nowhere", where="turn 9")
    assert "csv export skipped (turn 9)" in capsys.readouterr().out or not (tmp_path / "nowhere").exists()


def test_dashboard_reports_phase_turns_tokens_and_the_two_top_lists(tmp_path) -> None:
    run_dir = _run(tmp_path)
    report = dash.run_report(run_dir, top=5)
    assert report["arm"] == "claudesub" and report["budget"] == 98 and report["tag"] == "v36"
    assert report["turns_done"] == 2 and report["max_turns"] == 30
    assert report["phase"] == "gen" and 500 < report["in_phase_s"] < 900
    assert report["tok_last"] == 412_000 and report["tok_cum"] == 712_000
    # Latest completed turn: three ok codes best first; infeasible row excluded.
    assert [c["hash8"] for c in report["turn_top"]] == ["dd" * 4, "ee" * 4, "ff" * 4]
    # Cumulative: best measurement per code over both turns, baseline excluded.
    assert [c["hash8"] for c in report["cum_top"]] == ["dd" * 4, "ee" * 4, "bb" * 4, "aa" * 4, "ff" * 4]
    assert report["best_hash"] == "dd" * 4
    text = dash.render(tmp_path / "runs", tag="v36", top=5)
    assert "claudesub b98" in text and "2/30 turns" in text and "phase gen" in text
    assert "tok last 412k cum 712k" in text
    assert "[[48,4,7]]" in text and "3.6e-04" in text and "Q196" in text and "t1 " in text
    assert "latest turn top-5" in text and "cumulative top-5" in text
    assert text.count("*") == 1  # the campaign best is starred once, in the cumulative column
    assert not _CJK.search(text)  # English only


def test_dashboard_filters_and_marks_done_runs(tmp_path) -> None:
    run_dir = _run(tmp_path)
    assert "no runs found" in dash.render(tmp_path / "runs", arm="gptsub")
    assert "no runs found" in dash.render(tmp_path / "runs", budget=60)
    (run_dir / "campaign_report.json").write_text("{}", encoding="utf-8")
    (run_dir / "VALIDATED.json").write_text("{}", encoding="utf-8")
    text = dash.render(tmp_path / "runs")
    assert "phase done" in text and "[VALIDATED]" in text


def test_dashboard_main_draws_once_and_merges_the_tree(tmp_path, capsys) -> None:
    run_dir = _run(tmp_path)
    write_progress_csv(run_dir)
    write_summary_csv(run_dir)
    assert dash.main([str(tmp_path / "runs"), "--tag", "v36"]) == 0
    out = capsys.readouterr().out
    assert "OmniQEC Fig.2 monitor" in out and "merged:" in out
    assert (tmp_path / "runs" / "fig2_all_summary_claude-fable-5-1.csv").exists()
    assert dash.main([str(tmp_path / "runs"), "--no-merge"]) == 0
    assert "merged:" not in capsys.readouterr().out


def test_dashboard_merge_is_scoped_to_its_tag(tmp_path, capsys) -> None:
    """A canary chain beside the formal run (2026-09-12: fig2_gptsub_b60_v36_canary2)
    must not be merged into the formal tables the --tag v36 monitor refreshes."""

    formal = _run(tmp_path)
    canary = _run(tmp_path, name="fig2_claudesub_b98_v36_canary2")
    for run_dir in (formal, canary):
        write_progress_csv(run_dir)
        write_summary_csv(run_dir)
    merged = tmp_path / "runs" / "fig2_all_summary_claude-fable-5-1.csv"

    assert dash.main([str(tmp_path / "runs"), "--tag", "v36"]) == 0
    assert "refreshed (site=v36)" in capsys.readouterr().out
    with merged.open(newline="", encoding="utf-8") as handle:
        sites = sorted({row["site"] for row in csv.DictReader(handle)})
    assert sites == ["v36"]

    assert dash.main([str(tmp_path / "runs")]) == 0
    assert "(site=" not in capsys.readouterr().out
    with merged.open(newline="", encoding="utf-8") as handle:
        sites = sorted({row["site"] for row in csv.DictReader(handle)})
    assert sites == ["v36", "v36_canary2"]
