from __future__ import annotations

import csv
import json
from pathlib import Path

from omniqec_agentic.hpc.merge_fig2_csv import PREFIX, main, merge, model_slug
from omniqec_agentic.summary_csv import SUMMARY_COLUMNS

_PROGRESS_COLUMNS = ("turn", "audit", "best_plpr_cum")
# summary.csv's own `model` column folds into the prefix column of the same name.
_SUMMARY_BODY = [c for c in SUMMARY_COLUMNS if c not in PREFIX]
_MODEL = "gpt-5.6-sol"


def _summary_row(program_id, plpr, family="hgp", spec='{"family":"hgp"}'):
    row = {c: "" for c in SUMMARY_COLUMNS}
    row.update(program_id=program_id, family=family, plpr=str(plpr), spec=spec,
               decoder="telescoping", code_hash8=program_id[:8])
    return row


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _write_run(runs, name, summary_rows, *, progress_rows=(), progress_columns=_PROGRESS_COLUMNS,
               complete=False, summary_columns=SUMMARY_COLUMNS, model=_MODEL):
    run_dir = runs / name
    run_dir.mkdir()
    if model is not None:
        (run_dir / "protocol_snapshot.json").write_text(
            json.dumps({"required_model": model}), encoding="utf-8")
    _write_csv(run_dir / "summary.csv", summary_columns, summary_rows)
    if progress_rows:
        _write_csv(run_dir / "progress.csv", progress_columns, progress_rows)
    if complete:
        (run_dir / "campaign_report.json").write_text("{}", encoding="utf-8")
    return run_dir


def _read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def test_merge_orders_points_numerically_and_ranks_the_best_codes_per_point(tmp_path) -> None:
    # Written out of plpr order on purpose: the best view must rank, not trust file order.
    _write_run(tmp_path, "fig2_claudesub_b360_v36", [
        _summary_row("g0001_p0002", 0.010),
        _summary_row("baseline_bb_72_12_6", 0.004, family="bb", spec='{"family":"bb"}'),
        _summary_row("seed_0004", 0.002, family="lp_nonabelian", spec='{"family":"lp_nonabelian","group":"D6"}'),
        _summary_row("g0002_p0007", 0.003),
        _summary_row("g0003_p0001", 0.020),
    ], progress_rows=[{"turn": "0", "audit": "complete", "best_plpr_cum": "0.002"}])
    _write_run(tmp_path, "fig2_claudesub_b60_v36", [
        _summary_row("baseline_bb_72_12_6", 0.004, family="bb"),
    ], complete=True)
    _write_run(tmp_path, "fig2_gptsub_b288_v36", [
        _summary_row("g0001_p0001", 0.009),
        _summary_row("g0001_p0005", 0.008),
    ])
    (tmp_path / "not_a_run").mkdir()

    merge(tmp_path, refresh=False, top=2, suffix="none")

    header, rows = _read(tmp_path / "fig2_all_summary.csv")
    assert header == list(PREFIX) + _SUMMARY_BODY
    # b60 before b360 (numeric), arms grouped; model from the snapshot; complete
    # flag from campaign_report.json.
    assert [(r[0], r[1], r[3], r[4]) for r in rows] == (
        [("claudesub", "60", _MODEL, "yes")] + [("claudesub", "360", _MODEL, "no")] * 5
        + [("gptsub", "288", _MODEL, "no")] * 2
    )

    header, rows = _read(tmp_path / "fig2_all_best.csv")
    assert header == list(PREFIX) + ["rank"] + _SUMMARY_BODY
    pid = header.index("program_id")
    rank = header.index("rank")
    by_point = {}
    for row in rows:
        by_point.setdefault((row[0], row[1]), []).append((row[rank], row[pid]))
    assert by_point[("claudesub", "60")] == [("0", "baseline_bb_72_12_6")]
    assert by_point[("claudesub", "360")] == [
        ("0", "baseline_bb_72_12_6"), ("1", "seed_0004"), ("2", "g0002_p0007"),
    ]
    # No baseline row measured yet: ranks start at 1.
    assert by_point[("gptsub", "288")] == [("1", "g0001_p0005"), ("2", "g0001_p0001")]
    spec = header.index("spec")
    assert rows[[r[pid] for r in rows].index("seed_0004")][spec] == '{"family":"lp_nonabelian","group":"D6"}'

    header, rows = _read(tmp_path / "fig2_all_progress.csv")
    assert header == list(PREFIX) + list(_PROGRESS_COLUMNS)
    assert rows == [["claudesub", "360", "v36", _MODEL, "no", "0", "complete", "0.002"]]


def test_merge_aligns_columns_by_name_instead_of_dropping_points(tmp_path, capsys) -> None:
    # progress.csv grows one mix_<family> column per family a run has proposed,
    # in first-seen order, so two healthy runs routinely have different headers.
    _write_run(tmp_path, "fig2_claudesub_b60_v36", [_summary_row("g0001_p0001", 0.01)],
               progress_rows=[{"turn": "0", "mix_bb": "0.5", "mix_hgp": "0.5", "best_plpr_cum": "0.01"}],
               progress_columns=("turn", "mix_bb", "mix_hgp", "best_plpr_cum"))
    _write_run(tmp_path, "fig2_claudesub_b98_v36", [_summary_row("g0001_p0001", 0.02)],
               progress_rows=[{"turn": "0", "mix_lp_abelian": "1.0", "best_plpr_cum": "0.02"}],
               progress_columns=("turn", "mix_lp_abelian", "best_plpr_cum"),
               # An older exporter without `spec`: the point stays, the cell is empty.
               summary_columns=SUMMARY_COLUMNS[:-1])
    _write_run(tmp_path, "fig2_claudesub_b120_v36", [])  # header only: no codes measured yet

    merge(tmp_path, refresh=False, suffix="none")

    header, rows = _read(tmp_path / "fig2_all_progress.csv")
    assert header == list(PREFIX) + ["turn", "mix_bb", "mix_hgp", "best_plpr_cum", "mix_lp_abelian"]
    assert rows == [
        ["claudesub", "60", "v36", _MODEL, "no", "0", "0.5", "0.5", "0.01", ""],
        ["claudesub", "98", "v36", _MODEL, "no", "0", "", "", "0.02", "1.0"],
    ]
    header, rows = _read(tmp_path / "fig2_all_summary.csv")
    assert header == list(PREFIX) + _SUMMARY_BODY
    spec = header.index("spec")
    assert [(r[1], r[spec]) for r in rows] == [("60", '{"family":"hgp"}'), ("98", "")]
    header, rows = _read(tmp_path / "fig2_all_best.csv")
    rank = header.index("rank")
    assert [(r[1], r[rank]) for r in rows] == [("60", "1"), ("98", "1")]

    # The reason a run is absent goes to stdout: the wrapup job discards stderr.
    out = capsys.readouterr().out
    assert "fig2_claudesub_b120_v36: summary.csv 无数据行" in out
    assert "fig2_claudesub_b120_v36: 缺 progress.csv" in out


def test_merge_with_refresh_re_exports_real_turn_artifacts(tmp_path, monkeypatch) -> None:
    """End to end through the subprocess exporters on turn-shaped artifacts:
    plan.json family mixes differ between runs (different progress headers),
    slow_results.csv + candidates/<id>.json feed the spec column."""

    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    slow_cols = ("program_id", "code_hash", "family", "status", "stopped_by", "decoder",
                 "per_logical_per_round", "n_data", "k")

    def run(name, families, program_id, code_hash, plpr, spec):
        run_dir = tmp_path / name
        turn = run_dir / "turns" / "turn_0000"
        slow = turn / "cycle" / "slow"
        (slow / "candidates").mkdir(parents=True)
        (run_dir / "protocol_snapshot.json").write_text(
            json.dumps({"required_model": "gpt-5.6-sol"}), encoding="utf-8")
        (turn / "orchestrator").mkdir()
        (turn / "plan.json").write_text(json.dumps({
            "policy": "agentic_v1", "proposal_count": 24,
            "family_mix": {f: 1.0 / len(families) for f in families},
            "strategy_mix": {}, "explore_temperature": 0.5, "tool_calls_used": 3,
        }), encoding="utf-8")
        (turn / "digest.json").write_text(json.dumps({
            "turn": 0, "cycle_status": "ok",
            "baseline": {"per_logical_per_round": 0.004, "program_id": "baseline_bb_72_12_6"},
            "learning": {"evidence_rows_total": 10},
        }), encoding="utf-8")
        (turn / "turn_audit.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        _write_csv(slow / "slow_results.csv", slow_cols, [
            {"program_id": "baseline_bb_72_12_6", "code_hash": "3c631986aaaa", "family": "bb",
             "status": "ok", "decoder": "telescoping", "per_logical_per_round": "0.004"},
            {"program_id": program_id, "code_hash": code_hash, "family": spec["family"],
             "status": "ok", "decoder": "telescoping", "per_logical_per_round": str(plpr)},
        ])
        (slow / "candidates" / f"{program_id}.json").write_text(
            json.dumps({"program_id": program_id, "spec": spec,
                        "structural": {"code_hash": code_hash}}), encoding="utf-8")
        return run_dir

    run("fig2_gptsub_b360_v36", ["bb", "hgp"], "g0001_p0003", "aaaa1111", 0.002,
        {"family": "hgp", "seeds": {"H1": [[1, 1]]}})
    run("fig2_gptsub_b60_v36", ["lp_abelian"], "seed_0002", "bbbb2222", 0.006,
        {"family": "lp_abelian", "group": "Z7"})
    (tmp_path / "fig2_gptsub_b98_v36").mkdir()  # queued point, no turns yet

    merge(tmp_path, refresh=True, top=1)

    for name in ("fig2_gptsub_b360_v36", "fig2_gptsub_b60_v36", "fig2_gptsub_b98_v36"):
        assert (tmp_path / name / "summary.csv").exists()
        assert (tmp_path / name / "progress.csv").exists()

    # Default naming: the trio carries the runs' model as its suffix.
    assert not (tmp_path / "fig2_all_progress.csv").exists()
    header, rows = _read(tmp_path / "fig2_all_progress_gpt-5.6-sol.csv")
    assert {"mix_bb", "mix_hgp", "mix_lp_abelian"} <= set(header)
    assert [(r[1], r[header.index("audit")], r[header.index("best_plpr_cum")]) for r in rows] == [
        ("60", "complete", "0.006"), ("360", "complete", "0.002"),
    ]

    header, rows = _read(tmp_path / "fig2_all_best_gpt-5.6-sol.csv")
    pid, spec, vs_bb = header.index("program_id"), header.index("spec"), header.index("vs_bb")
    assert header.index("model") == 3 and {r[3] for r in rows} == {"gpt-5.6-sol"}
    assert [(r[1], r[5], r[pid]) for r in rows] == [
        ("60", "0", "baseline_bb_72_12_6"), ("60", "1", "seed_0002"),
        ("360", "0", "baseline_bb_72_12_6"), ("360", "1", "g0001_p0003"),
    ]
    assert json.loads(rows[1][spec]) == {"family": "lp_abelian", "group": "Z7"}
    assert rows[3][vs_bb] == "0.5"


def test_cli_accepts_top_and_no_refresh(tmp_path) -> None:
    _write_run(tmp_path, "fig2_gptsub_b360_v36", [
        _summary_row(f"g0001_p000{i}", 0.001 * (i + 1)) for i in range(4)
    ])
    main([str(tmp_path), "--top", "1", "--no-refresh"])
    header, rows = _read(tmp_path / f"fig2_all_best_{_MODEL}.csv")
    rank, pid = header.index("rank"), header.index("program_id")
    assert [(r[rank], r[pid]) for r in rows] == [("1", "g0001_p0000")]
    main([str(tmp_path), "--top", "1", "--no-refresh", "--suffix", "none"])
    assert (tmp_path / "fig2_all_best.csv").exists()


def test_model_slug_keeps_the_segment_after_the_provider_slash() -> None:
    assert model_slug("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"
    assert model_slug("gpt-5.6-sol") == "gpt-5.6-sol"
    assert model_slug("claude-opus-4-1-20250805") == "claude-opus-4-1-20250805"
    assert model_slug("anthropic/claude opus:4") == "claude-opus-4"
    assert model_slug("") == "unknown-model" and model_slug("?") == "unknown-model"


def test_merge_writes_one_trio_per_model_so_three_arms_can_share_a_folder(tmp_path, capsys) -> None:
    """Three arms pulled back into one runs/ folder: each model gets its own
    fig2_all_*_<model>.csv trio (no overwriting), the model column says which
    LLM measured each row, and a run without a snapshot lands in unknown-model."""

    _write_run(tmp_path, "fig2_claudesub_b98_v36", [_summary_row("g0001_p0001", 0.010)],
               progress_rows=[{"turn": "0", "audit": "complete", "best_plpr_cum": "0.010"}],
               model="claude-opus-4-1-20250805")
    _write_run(tmp_path, "fig2_claudesub_b60_v36", [_summary_row("g0001_p0002", 0.020)],
               model="claude-opus-4-1-20250805")
    _write_run(tmp_path, "fig2_gptsub_b98_v36", [_summary_row("g0001_p0003", 0.011)],
               model="gpt-5.6-sol")
    _write_run(tmp_path, "fig2_deepseek_b98_v36", [_summary_row("g0001_p0004", 0.012)],
               model="deepseek/deepseek-v4-pro")
    _write_run(tmp_path, "fig2_deepseek_b60_pod", [_summary_row("g0001_p0005", 0.030)], model=None)

    merge(tmp_path, refresh=False, top=1)

    names = sorted(p.name for p in tmp_path.glob("fig2_all_*.csv"))
    assert names == sorted([
        f"fig2_all_{kind}_{slug}.csv"
        for kind in ("summary", "best")
        for slug in ("claude-opus-4-1-20250805", "gpt-5.6-sol", "deepseek-v4-pro", "unknown-model")
    ] + ["fig2_all_progress_claude-opus-4-1-20250805.csv"])
    header, rows = _read(tmp_path / "fig2_all_summary_claude-opus-4-1-20250805.csv")
    assert [(r[0], r[1], r[3]) for r in rows] == [
        ("claudesub", "60", "claude-opus-4-1-20250805"), ("claudesub", "98", "claude-opus-4-1-20250805"),
    ]
    _, rows = _read(tmp_path / "fig2_all_summary_deepseek-v4-pro.csv")
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [("deepseek", "98", "v36", "deepseek/deepseek-v4-pro")]
    _, rows = _read(tmp_path / "fig2_all_summary_unknown-model.csv")
    assert [(r[0], r[2], r[3]) for r in rows] == [("deepseek", "pod", "unknown-model")]
    _, rows = _read(tmp_path / "fig2_all_best_gpt-5.6-sol.csv")
    assert [(r[0], r[1], r[5]) for r in rows] == [("gptsub", "98", "1")]
    out = capsys.readouterr().out
    assert "fig2_all_summary_gpt-5.6-sol.csv" in out and "fig2_all_best_deepseek-v4-pro.csv" in out


def test_site_filter_keeps_a_canary_chain_out_of_the_formal_tables(tmp_path, capsys) -> None:
    """2026-09-12: runs/ held the formal fig2_gptsub_b60_v36 next to the aborted
    fig2_gptsub_b60_v36_canary2; the merged GPT tables then carried the canary's
    turns/codes as if they were extra rows of the formal point. `site` (the
    dashboard's --tag) must keep exactly one site; without it every run merges."""

    _write_run(tmp_path, "fig2_gptsub_b60_v36", [_summary_row("g0001_p0001", 0.010)],
               progress_rows=[{"turn": "0", "audit": "complete", "best_plpr_cum": "0.010"}])
    _write_run(tmp_path, "fig2_gptsub_b60_v36_canary2", [_summary_row("g0009_p0009", 0.001)],
               progress_rows=[{"turn": "0", "audit": "complete", "best_plpr_cum": "0.001"}],
               complete=True)
    _write_run(tmp_path, "fig2_gptsub_b98_v36x60", [_summary_row("g0005_p0005", 0.002)])

    merge(tmp_path, refresh=False, top=5)
    _, rows = _read(tmp_path / "fig2_all_summary_gpt-5.6-sol.csv")
    assert sorted(r[2] for r in rows) == ["v36", "v36_canary2", "v36x60"]

    merge(tmp_path, refresh=False, top=5, site="v36")
    for kind in ("summary", "progress", "best"):
        _, rows = _read(tmp_path / f"fig2_all_{kind}_gpt-5.6-sol.csv")
        assert [(r[0], r[1], r[2]) for r in rows] == [("gptsub", "60", "v36")], kind
    _, rows = _read(tmp_path / "fig2_all_best_gpt-5.6-sol.csv")
    assert (rows[0][5], rows[0][7]) == ("1", "g0001_p0001")  # rank, program_id (turn sits between)

    main([str(tmp_path), "--no-refresh", "--site", "v36x60"])
    _, rows = _read(tmp_path / "fig2_all_summary_gpt-5.6-sol.csv")
    assert [(r[1], r[2]) for r in rows] == [("98", "v36x60")]
    assert "fig2_all_summary_gpt-5.6-sol.csv" in capsys.readouterr().out
