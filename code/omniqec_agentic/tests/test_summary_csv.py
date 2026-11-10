from __future__ import annotations

import json

from omniqec_agentic.summary_csv import _campaign_model


def _write_run_config(run_dir, model):
    fast = run_dir / "turns" / "turn_0000" / "cycle" / "fast"
    fast.mkdir(parents=True, exist_ok=True)
    (fast / "run_config.json").write_text(
        json.dumps({"llm": {"provider": "deepseek", "model": model}}),
        encoding="utf-8",
    )


def test_campaign_model_prefers_protocol_required_model(tmp_path) -> None:
    # run_config.json serializes the LLMConfig defaults object, which reports
    # deepseek-chat regardless of the provider actually in use — the snapshot's
    # required_model (what the strict audit pins) must win over it.
    (tmp_path / "protocol_snapshot.json").write_text(
        json.dumps({"required_provider": "openai", "required_model": "gpt-5.6-sol"}),
        encoding="utf-8",
    )
    _write_run_config(tmp_path, "deepseek-chat")
    assert _campaign_model(tmp_path) == "gpt-5.6-sol"


def test_campaign_model_falls_back_for_pre_snapshot_runs(tmp_path) -> None:
    (tmp_path / "protocol_snapshot.json").write_text(
        json.dumps({"required_model": None}), encoding="utf-8"
    )
    _write_run_config(tmp_path, "deepseek-chat")
    assert _campaign_model(tmp_path) == "deepseek-chat"


def test_campaign_model_unknown_when_nothing_recorded(tmp_path) -> None:
    assert _campaign_model(tmp_path) == "?"


# --- spec column -----------------------------------------------------------

import csv

from omniqec_agentic.summary_csv import SUMMARY_COLUMNS, collect_best_rows, main

_SLOW_COLUMNS = (
    "program_id", "code_hash", "family", "status", "n_data", "n_checks_x",
    "n_checks_z", "k", "d_proxy", "decoder", "per_logical_per_round",
    "per_logical_per_round_ci_low", "per_logical_per_round_ci_high",
    "block_ler", "total_failures", "total_shots", "p", "rounds",
)


def _write_slow(run_dir, turn, rows, specs=None):
    """A slow-stage directory the way evaluation/pool.py lays it out:
    slow_results.csv next to candidates/<program_id>.json."""

    slow = run_dir / "turns" / f"turn_{turn:04d}" / "cycle" / "slow"
    (slow / "candidates").mkdir(parents=True, exist_ok=True)
    with (slow / "slow_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SLOW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in _SLOW_COLUMNS}, **row})
    for program_id, spec in (specs or {}).items():
        (slow / "candidates" / f"{program_id}.json").write_text(
            json.dumps({"program_id": program_id, "spec": spec, "status": "ok"}),
            encoding="utf-8",
        )
    return slow


_BB_SPEC = {"A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]], "ell": 6, "family": "bb", "m": 6}
_LP_SPEC = {"A": [[[[0, 0], [4, 0]]]], "B": [[[[0, 0], [2, 1]]]], "family": "lp_nonabelian", "group": "D6"}


def _ok(program_id, code_hash, family, plpr, **extra):
    return {"program_id": program_id, "code_hash": code_hash, "family": family,
            "status": "ok", "decoder": "telescoping", "per_logical_per_round": str(plpr), **extra}


def test_best_rows_carry_the_candidate_spec_next_to_each_slow_results(tmp_path) -> None:
    _write_slow(
        tmp_path, 0,
        [
            _ok("baseline_bb_72_12_6", "3c631986aaaa", "bb", 0.004, n_data="72", k="12"),
            _ok("seed_0004", "ed8c3b95bbbb", "lp_nonabelian", 0.002, n_data="120", k="8"),
            # Not measured: never a leaderboard row, spec or not.
            {"program_id": "g0001_p0003", "code_hash": "deadbeef", "family": "hgp",
             "status": "decoder_unsupported"},
        ],
        specs={"baseline_bb_72_12_6": _BB_SPEC, "seed_0004": _LP_SPEC, "g0001_p0003": {"family": "hgp"}},
    )
    # A later turn re-measures seed_0004 worse and finds a code with no artifact.
    _write_slow(
        tmp_path, 1,
        [
            _ok("baseline_bb_72_12_6", "3c631986aaaa", "bb", 0.0041),
            _ok("seed_0004", "ed8c3b95bbbb", "lp_nonabelian", 0.003),
            _ok("g0002_p0001", "0badc0de", "hgp", 0.010),
        ],
        specs={"baseline_bb_72_12_6": _BB_SPEC, "seed_0004": _LP_SPEC},
    )

    rows = collect_best_rows(tmp_path)

    assert [r["program_id"] for r in rows] == ["seed_0004", "baseline_bb_72_12_6", "g0002_p0001"]
    assert rows[0]["turn"] == "0" and rows[0]["plpr"] == 0.002   # best measurement wins
    assert json.loads(rows[0]["spec"]) == _LP_SPEC
    assert json.loads(rows[1]["spec"]) == _BB_SPEC
    assert rows[2]["spec"] == ""                                  # no candidates/<id>.json
    assert rows[0]["vs_bb"] == 0.5
    assert rows[0]["code_hash8"] == "ed8c3b95"
    # Compact JSON: one cell, no whitespace to break on.
    assert " " not in rows[0]["spec"] and "\n" not in rows[0]["spec"]


def test_spec_is_the_last_summary_column_and_main_writes_it(tmp_path, monkeypatch, capsys) -> None:
    _write_slow(
        tmp_path, 0,
        [_ok("baseline_bb_72_12_6", "3c631986", "bb", 0.004)],
        specs={"baseline_bb_72_12_6": _BB_SPEC},
    )
    monkeypatch.setattr("sys.argv", ["summary_csv", str(tmp_path)])
    assert main() == 0
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == SUMMARY_COLUMNS
        assert SUMMARY_COLUMNS[-1] == "spec"
        (row,) = list(reader)
    assert json.loads(row["spec"]) == _BB_SPEC
    assert "1 codes" in capsys.readouterr().out


def test_unreadable_candidate_artifact_yields_an_empty_spec(tmp_path) -> None:
    slow = _write_slow(tmp_path, 0, [_ok("seed_0001", "abc", "hgp", 0.01)])
    (slow / "candidates" / "seed_0001.json").write_text("{not json", encoding="utf-8")
    (rows,) = [collect_best_rows(tmp_path)]
    assert rows[0]["spec"] == ""


def test_final_stage_rows_sharing_a_program_id_get_their_own_spec(tmp_path) -> None:
    """program_ids restart every turn, so two different codes can both be
    g0001_p0019. Per-turn dirs keep them apart, but the final stage pools
    finalists into one final_slow/T<r>/candidates/ dir where the second write
    overwrites the first: both CSV rows then point at one artifact. The
    exporter must notice the artifact's code_hash is not the row's and fall
    back to the losing finalist's own turn-stage artifact."""

    spec_a = {"family": "hgp", "seeds": {"H1": [[1, 1, 0]]}}
    spec_b = {"family": "lp_abelian", "group": "Z5"}
    turn_slow = _write_slow(
        tmp_path, 0,
        [_ok("g0001_p0019", "aaaa0001", "hgp", 0.010)],
    )
    (turn_slow / "candidates" / "g0001_p0019.json").write_text(
        json.dumps({"program_id": "g0001_p0019", "spec": spec_a,
                    "structural": {"code_hash": "aaaa0001"},
                    "promotion_record": {"code_hash": "aaaa0001"}}),
        encoding="utf-8",
    )
    final = tmp_path / "final_slow" / "T12"
    (final / "candidates").mkdir(parents=True)
    with (final / "slow_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SLOW_COLUMNS)
        writer.writeheader()
        for row in (_ok("g0001_p0019", "aaaa0001", "hgp", 0.005),
                    _ok("g0001_p0019", "bbbb0002", "lp_abelian", 0.003),
                    _ok("g0002_p0001", "cccc0003", "hgp", 0.004)):
            writer.writerow({**{c: "" for c in _SLOW_COLUMNS}, **row})
    # Only code B's artifact survived the overwrite.
    (final / "candidates" / "g0001_p0019.json").write_text(
        json.dumps({"program_id": "g0001_p0019", "spec": spec_b,
                    "structural": {"code_hash": "bbbb0002"}}),
        encoding="utf-8",
    )
    # A legacy artifact with no code_hash at all is accepted as-is.
    (final / "candidates" / "g0002_p0001.json").write_text(
        json.dumps({"program_id": "g0002_p0001", "spec": {"family": "hgp", "legacy": True}}),
        encoding="utf-8",
    )

    rows = {r["code_hash8"]: r for r in collect_best_rows(tmp_path)}

    assert set(rows) == {"aaaa0001", "bbbb0002", "cccc0003"}
    assert rows["aaaa0001"]["turn"] == "final" and rows["aaaa0001"]["plpr"] == 0.005
    assert json.loads(rows["aaaa0001"]["spec"]) == spec_a      # from turn_0000, not the overwritten file
    assert json.loads(rows["bbbb0002"]["spec"]) == spec_b
    assert json.loads(rows["cccc0003"]["spec"]) == {"family": "hgp", "legacy": True}


def test_spec_is_empty_when_no_artifact_anywhere_matches_the_row_hash(tmp_path) -> None:
    slow = _write_slow(tmp_path, 0, [_ok("seed_0001", "dddd0004", "hgp", 0.01)])
    (slow / "candidates" / "seed_0001.json").write_text(
        json.dumps({"spec": {"family": "hgp"}, "structural": {"code_hash": "eeee0005"}}),
        encoding="utf-8",
    )
    (rows,) = [collect_best_rows(tmp_path)]
    assert rows[0]["spec"] == ""
