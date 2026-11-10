from __future__ import annotations

import csv
import json

from omniqec_agentic.progress_csv import collect_progress_rows


def _write_turn(run_dir, turn, *, plan=None, digest=None, audit=None, slow_rows=()):
    turn_dir = run_dir / "turns" / f"turn_{turn:04d}"
    (turn_dir / "cycle" / "slow").mkdir(parents=True, exist_ok=True)
    (turn_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
    if plan is not None:
        (turn_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    if digest is not None:
        (turn_dir / "digest.json").write_text(json.dumps(digest), encoding="utf-8")
    if audit is not None:
        (turn_dir / "turn_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    columns = ["program_id", "status", "stopped_by", "family", "per_logical_per_round"]
    with (turn_dir / "cycle" / "slow" / "slow_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in slow_rows:
            writer.writerow(row)
    return turn_dir


def _baseline_digest(turn, **extra):
    digest = {
        "turn": turn,
        "cycle_status": "ok",
        "baseline": {"per_logical_per_round": 0.004, "program_id": "baseline_bb_72_12_6"},
        "learning": {"evidence_rows_total": 10 * (turn + 1)},
    }
    digest.update(extra)
    return digest


def test_progress_rows_track_cumulative_best_against_the_baseline(tmp_path) -> None:
    _write_turn(
        tmp_path, 0,
        plan={"policy": "agentic_v1", "proposal_count": 24, "family_mix": {"hgp": 1.0},
              "strategy_mix": {"novelty": 1.0}, "explore_temperature": 0.5, "tool_calls_used": 5},
        digest=_baseline_digest(0),
        audit={"status": "complete"},
        slow_rows=[
            # The baseline is re-measured every turn and must not win `best`.
            {"program_id": "baseline_bb_72_12_6", "status": "ok", "stopped_by": "",
             "family": "bb", "per_logical_per_round": "0.004"},
            {"program_id": "g0001_p0001", "status": "ok", "stopped_by": "",
             "family": "hgp", "per_logical_per_round": "0.008"},
            {"program_id": "g0001_p0002", "status": "ok", "stopped_by": "racing",
             "family": "hgp", "per_logical_per_round": "0.020"},
            {"program_id": "g0001_p0003", "status": "error", "stopped_by": "",
             "family": "hgp", "per_logical_per_round": ""},
        ],
    )
    _write_turn(
        tmp_path, 1,
        plan={"policy": "agentic_v1_fallback", "proposal_count": 24},
        digest=_baseline_digest(1),
        audit={"status": "complete"},
        slow_rows=[
            {"program_id": "g0002_p0001", "status": "ok", "stopped_by": "",
             "family": "lp_abelian", "per_logical_per_round": "0.012"},
        ],
    )

    rows, columns = collect_progress_rows(tmp_path)
    assert [row["turn"] for row in rows] == [0, 1]

    first, second = rows
    assert first["measured"] == 2 and first["censored"] == 1
    assert first["families_measured"] == "hgp:2"
    assert first["best_plpr_turn"] == 0.008
    assert first["vs_bb_cum"] == 2.0
    assert first["policy"] == "agentic_v1"
    assert first["audit"] == "complete"
    assert first["mix_hgp"] == 1.0 and first["strat_novelty"] == 1.0
    assert first["explore_temp"] == 0.5 and first["tools"] == 5

    # A worse turn must not move the cumulative best.
    assert second["best_plpr_turn"] == 0.012
    assert second["best_plpr_cum"] == 0.008
    assert second["vs_bb_cum"] == 2.0
    # Fallback turns carry no mix; the column still exists so the CSV stays square.
    assert "mix_hgp" in columns and second.get("mix_hgp") is None


def test_progress_export_tolerates_a_turn_still_in_flight(tmp_path) -> None:
    _write_turn(
        tmp_path, 0,
        plan={"policy": "agentic_v1", "proposal_count": 24},
        digest=_baseline_digest(0),
        audit={"status": "complete"},
        slow_rows=[{"program_id": "g0001_p0001", "status": "ok", "stopped_by": "",
                    "family": "hgp", "per_logical_per_round": "0.009"}],
    )
    # Turn 1 has started: a directory and a plan, but no digest and no audit yet.
    _write_turn(tmp_path, 1, plan={"policy": "agentic_v1", "proposal_count": 24})

    rows, _ = collect_progress_rows(tmp_path)
    assert len(rows) == 2
    assert rows[1]["audit"] == "" and rows[1]["cycle"] == ""
    assert rows[1]["measured"] == 0
    assert rows[1]["best_plpr_turn"] is None
    assert rows[1]["best_plpr_cum"] == 0.009
