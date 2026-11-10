"""2026-11-10 paper alignment: the measurement and final stage exactly as the paper states them.

* Appendix D3 / Table VII: the five leading distinct codes by search-stage pLER, an initial
  confirmation of 25 000 shots per basis, top-up to the cumulative limit of Table VII, the parts pooled
  per basis, the champion by the pooled pLER (evaluation/final_stage.py, workflow.graph._paper_final_*).
* Appendix D3 (2026-11-10 revision): the two BB reference codes under the same protocol, their reported LER
  based on 50 000 (BB72) and 100 000 (BB144) shots per basis in both tasks -- the initial confirmation plus a
  top-up to that total (config.REFERENCE_CODE_SHOTS_PER_BASIS).
* Fig. 4 c (2026-11-10 revision): an exact check-weight run takes its own final stage -- six finalists, the
  initial confirmation only, the three best reported.
* Appendix D3: the search-stage early stop -- 2 000 shots first, then 250-shot batches; stop when the Wilson
  lower bound exceeds ten times BOTH the best measured code's and the applicable reference code's upper
  bound, or both bases hold 30 failures, or 25 000 shots per basis.
* Section "Circuit-level evaluation": R = 12 for every evaluation (fixed_Tc is run-campaign's default).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from omniqec_agentic import config as config_mod
from omniqec_agentic.config import (
    FINAL_CUMULATIVE_SHOT_CAPS,
    FINAL_STAGE_PAPER_ID,
    RACING_POLICY_PAPER_ID,
    RacingConfig,
    VerificationConfig,
    formal_racing,
    formal_racing_policy_id,
    paper_racing,
    structural_search_for_profile,
)
from omniqec_agentic.evaluation import final_stage as fs
from omniqec_agentic.evaluation.sampling import RacingThresholds, racing_decision
from omniqec_agentic.workflow.state import build_protocol_snapshot

ROOT = Path(__file__).resolve().parents[1]
BB72_HASH = fs.resident_reference(144)["code_hash"]


# ---------------------------------------------------------------- Table VII and the protocol block
def test_table_vii_is_the_paper_table_and_matches_the_caps_file() -> None:
    assert FINAL_CUMULATIVE_SHOT_CAPS == {
        "memory": {60: 175_000, 98: 175_000, 144: 175_000, 180: 175_000, 220: 275_000,
                   260: 525_000, 288: 525_000, 320: 875_000, 360: 1_725_000},
        "cnot_dense": {60: 75_000, 98: 75_000, 144: 75_000, 180: 75_000, 220: 125_000,
                       260: 175_000, 288: 325_000, 320: 525_000, 360: 875_000},
    }
    caps = json.loads((ROOT / "hpc" / "top5_extension_caps.json").read_text(encoding="utf-8"))["caps"]
    assert {task: {int(n): v for n, v in table.items()} for task, table in caps.items()} == FINAL_CUMULATIVE_SHOT_CAPS


@pytest.mark.parametrize(
    ("task", "budget", "cap", "ref_id", "ref_cap"),
    [
        ("memory", 144, 175_000, "baseline_bb_72_12_6", 50_000),
        ("memory", 288, 525_000, "baseline_bb_144_12_12", 100_000),
        ("memory", 360, 1_725_000, "baseline_bb_144_12_12", 100_000),   # the reference keeps ITS paper total
        ("cnot_dense", 144, 75_000, "baseline_bb_72_12_6", 50_000),
        ("cnot_dense", 288, 325_000, "baseline_bb_144_12_12", 100_000),
        ("cnot_dense", 98, 75_000, None, None),
        ("memory", 220, 275_000, None, None),
    ],
)
def test_protocol_block_caps_and_reference(task, budget, cap, ref_id, ref_cap) -> None:
    block = fs.final_stage_protocol(task=task, n_budget=budget, confirm_shots=25_000, final_shot_shards=128)
    assert block["id"] == FINAL_STAGE_PAPER_ID and block["top"] == 5 and block["confirm_shots"] == 25_000
    assert block["cumulative_cap"] == cap and block["cumulative_cap_source"] == "table_vii"
    if ref_id is None:
        assert block["reference"] is None
    else:
        assert block["reference"]["program_id"] == ref_id and block["reference"]["cumulative_cap"] == ref_cap
        assert block["reference"]["cumulative_cap_source"] == "paper_reference_shots"


def test_budget_outside_table_vii_has_no_topup_and_override_is_a_variant() -> None:
    block = fs.final_stage_protocol(task="memory", n_budget=72, confirm_shots=25_000, final_shot_shards=128)
    assert block["cumulative_cap"] == 25_000 and block["cumulative_cap_source"] == "budget_not_in_table_vii"
    assert fs.topup_parts(block, cumulative_cap=25_000, group="finalists", rounds=12) == []
    small = fs.final_stage_protocol(task="memory", n_budget=144, confirm_shots=100, final_shot_shards=4,
                                    cumulative_override=300)
    assert small["cumulative_cap"] == 300 and small["reference"]["cumulative_cap"] == 300
    with pytest.raises(ValueError):
        fs.final_stage_protocol(task="memory", n_budget=144, confirm_shots=25_000, final_shot_shards=128,
                                cumulative_override=1_000)


def _snapshot(**overrides):
    kwargs = dict(
        verification=VerificationConfig(structural=structural_search_for_profile("v3_8"), rounds_policy="fixed_Tc"),
        racing=paper_racing(25_000), max_turns=30, proposal_count=24, generations_per_turn=4, slow_workers=0,
        strict_turns=True, required_provider="openrouter", required_model="deepseek/deepseek-v4-pro",
        rng_seeds_per_program=40, n_budget=144, final_stage_id=FINAL_STAGE_PAPER_ID,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)


def test_snapshot_freezes_the_final_stage_only_when_asked() -> None:
    snapshot = _snapshot()
    assert snapshot["final_stage"] == fs.final_stage_protocol(
        task="memory", n_budget=144, confirm_shots=25_000, final_shot_shards=128
    )
    assert "final_stage" not in _snapshot(final_stage_id=None)       # earlier versions' snapshots, byte for byte
    with pytest.raises(ValueError):
        _snapshot(final_cumulative_shots=50_000)                        # strict: Table VII only
    with pytest.raises(ValueError):
        _snapshot(final_stage_id="top5_extension_v1")
    assert _snapshot(strict_turns=False, final_cumulative_shots=50_000)["final_stage"]["cumulative_cap"] == 50_000


def test_run_campaign_and_cli_default_to_the_paper(monkeypatch, tmp_path) -> None:
    import inspect

    from omniqec_agentic import cli as cli_mod
    from omniqec_agentic.workflow import graph

    assert inspect.signature(graph.run_campaign).parameters["final_stage_id"].default == FINAL_STAGE_PAPER_ID
    captured: dict = {}

    def fake_run_campaign(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"run_id": "campaign_b144", "stop": {"reason": "test"}, "report": {}}

    monkeypatch.setattr("omniqec_agentic.workflow.run_campaign", fake_run_campaign)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    base = ["run-campaign", "--budget", "144", "--turns", "1", "--provider", "dry_run", "--out", str(tmp_path)]
    assert cli_mod.main(base) == 0
    assert captured["verification"].rounds_policy == "fixed_Tc"
    assert captured["racing"].policy_id == RACING_POLICY_PAPER_ID
    assert captured["final_cumulative_shots"] is None
    assert cli_mod.main(base + ["--rounds-policy", "max_Tc_d", "--racing-max-errors", "100",
                                "--final-cumulative-shots", "50000"]) == 0
    assert captured["verification"].rounds_policy == "max_Tc_d"
    assert captured["racing"].policy_id == config_mod.RACING_POLICY_V3_ID
    assert captured["final_cumulative_shots"] == 50_000


# ---------------------------------------------------------------- selection
def _row(code: str, plpr: float, ub: float | None = None, **extra) -> dict:
    return {"code_hash": code, "program_id": extra.pop("program_id", f"p_{code}"), "per_logical_per_round": plpr,
            "per_logical_per_round_ci": [plpr / 2, ub if ub is not None else plpr * 2], **extra}


def test_top5_by_search_estimate_distinct_by_canonical_identity() -> None:
    rows = [
        _row("e", 5e-5), _row("a", 1e-5), _row("c", 3e-5), _row("b", 2e-5),
        _row("a_twin", 1.5e-5, canonical_hash="CANON_A"),  # isomorphic to "a"
        _row("f", 6e-5), _row("d", 4e-5), _row("g", 7e-5),
    ]
    finalists, skipped = fs.select_finalists(rows, top=5, canonical_of={"a": "CANON_A"})
    assert [r["code_hash"] for r in finalists] == ["a", "b", "c", "d", "e"]
    assert skipped == [{"code_hash": "a_twin", "program_id": "p_a_twin", "duplicate_of": "a"}]


def test_ties_are_broken_by_the_smaller_upper_bound_and_short_pools_give_all() -> None:
    rows = [_row("x", 1e-5, ub=4e-5), _row("y", 1e-5, ub=3e-5)]
    finalists, _ = fs.select_finalists(rows, top=5)
    assert [r["code_hash"] for r in finalists] == ["y", "x"]
    # the caller's order is kept for a no-slow ablation (no measurement)
    unranked, _ = fs.select_finalists([{"code_hash": "s1"}, {"code_hash": "s2"}], top=5, ranked=False)
    assert [r["code_hash"] for r in unranked] == ["s1", "s2"]


# ---------------------------------------------------------------- parts, pooling, champion
def test_topup_parts_reach_the_cap_with_their_own_seeds_and_shards() -> None:
    block = fs.final_stage_protocol(task="memory", n_budget=360, confirm_shots=25_000, final_shot_shards=128)
    parts = fs.topup_parts(block, cumulative_cap=block["cumulative_cap"], group="finalists", rounds=12)
    assert [p["shots"] for p in parts] == [850_000, 850_000]
    assert 25_000 + sum(p["shots"] for p in parts) == 1_725_000
    assert [p["rng_namespace"] for p in parts] == ["final_topup_v1/p1", "final_topup_v1/p2"]
    assert all(p["shards"] == 4096 for p in parts)  # ~300 shots per shard, never fewer than 128
    ref = fs.topup_parts(block, cumulative_cap=block["reference"]["cumulative_cap"], group="reference", rounds=12)
    assert [p["shots"] for p in ref] == [75_000] and ref[0]["rng_namespace"] == "final_topup_v1/reference/p1"
    assert ref[0]["dir"] != parts[0]["dir"]
    cnot = fs.final_stage_protocol(task="cnot_dense", n_budget=144, confirm_shots=25_000, final_shot_shards=128)
    assert [p["shots"] for p in fs.topup_parts(cnot, cumulative_cap=75_000, group="finalists", rounds=12)] == [50_000]


def test_pool_adds_shots_and_failures_per_basis_and_the_champion_is_the_pooled_best() -> None:
    from omniqec_agentic.verification.ler import combine_basis_ler_results

    confirm = {"name": "confirm", "shots": 25_000}
    topup = {"name": "topup_p1", "shots": 150_000}

    def ok(sx, fx, fz):
        return {"status": "ok", "shots_x": sx, "shots_z": sx, "failures_x": fx, "failures_z": fz, "rounds": 12, "k": 4}

    a = fs.pool_entry({"program_id": "A", "search_rank": 1}, [(confirm, ok(25_000, 10, 12)), (topup, ok(150_000, 90, 80))],
                      logicals=4, rounds=12, ci_z=1.96, cumulative_cap=175_000)
    b = fs.pool_entry({"program_id": "B", "search_rank": 2}, [(confirm, ok(25_000, 14, 15)), (topup, ok(150_000, 40, 41))],
                      logicals=4, rounds=12, ci_z=1.96, cumulative_cap=175_000)
    assert (a["shots_x"], a["failures_x"], a["shots_z"], a["failures_z"]) == (175_000, 100, 175_000, 92)
    assert a["cumulative_complete"] and b["cumulative_complete"]
    expected = combine_basis_ler_results(
        {"X": {"logical_error_rate": 100 / 175_000, "failures": 100, "shots": 175_000},
         "Z": {"logical_error_rate": 92 / 175_000, "failures": 92, "shots": 175_000}}, rounds=12, k=4)
    assert a["per_logical_per_round"] == pytest.approx(expected["per_logical_per_round"], rel=1e-12)
    # B was second in the search but is better pooled: B is the champion
    ranked = fs.rank_pooled([a, b])
    assert [r["program_id"] for r in ranked] == ["B", "A"] and ranked[0]["champion"] and not ranked[1]["champion"]
    # a top-up part that did not complete is not combined; the row says so
    c = fs.pool_entry({"program_id": "C"}, [(confirm, ok(25_000, 3, 3)), (topup, {"status": "decode_infeasible"})],
                      logicals=4, rounds=12, ci_z=1.96, cumulative_cap=175_000)
    assert c["shots_x"] == 25_000 and not c["cumulative_complete"] and c["status"] == "ok"
    # an unconfirmed finalist is not ranked
    d = fs.pool_entry({"program_id": "D"}, [(confirm, {"status": "decode_infeasible"})],
                      logicals=4, rounds=12, ci_z=1.96, cumulative_cap=175_000)
    ref = fs.pool_entry({"program_id": "R", "is_baseline": True}, [(confirm, ok(25_000, 1, 1))],
                        logicals=12, rounds=12, ci_z=1.96, cumulative_cap=175_000)
    ranked = fs.rank_pooled([a, b, d, ref])
    assert [r["final_rank"] for r in ranked] == [1, 2, None, None]
    assert ref["final_rank"] is None and not ref["champion"]


# ---------------------------------------------------------------- the graph's final stage end to end
RATES = {"a": 4e-4, "b": 2e-4, "c": 6e-4, "d": 8e-4, "e": 1e-3, "f": 1.2e-3, "twin": 5e-4}


def _hof() -> list[dict]:
    spec = lambda i: {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [1, 0], [0, 2 + i]], "B": [[0, 0], [0, 1], [2, 0]]}
    rows = []
    for i, (code, plpr) in enumerate([("a", 1e-5), ("b", 2e-5), ("twin", 2.5e-5), ("c", 3e-5), ("d", 4e-5), ("e", 5e-5), ("f", 6e-5)]):
        rows.append({
            "code_hash": code * 8, "program_id": "g0001_p0001" if code in ("a", "c") else f"g0002_p{i:04d}",
            "canonical_hash": "CANON_A" if code in ("a", "twin") else f"canon_{code}", "spec": spec(i),
            "per_logical_per_round": plpr, "per_logical_per_round_ci": [plpr / 2, plpr * 2], "k": 8, "n": 72,
            "d_search": 6, "family": "bb",
        })
    return rows


def _fake_pool(calls: list[dict], *, fail_part: str | None = None):
    def fake(*, queue_path, out_dir, verification, shot_shards, seed, stage, resume_candidates, include_baseline, **_):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = [json.loads(line) for line in Path(queue_path).read_text().splitlines() if line.strip()]
        calls.append({"dir": out_dir.name, "programs": [r["program_id"] for r in records], "shots": verification.shots,
                      "shards": shot_shards, "seed": seed, "stage": stage, "resume": resume_candidates,
                      "rounds": verification.rounds, "include_baseline": include_baseline})
        (out_dir / "slow_run_config.json").write_text(json.dumps({
            "schema": "slow_queue_run_config", "seed": seed, "shot_shards": shot_shards, "stage": stage,
            "verification": verification.to_dict(), "racing": None}))
        blocked = fail_part is not None and fail_part in str(out_dir)
        rows = []
        for record in records:
            code = str(record.get("code_hash") or "")[:1] if not record.get("is_baseline") else "ref"
            rate = RATES.get(code, 3e-4)
            shots = int(verification.shots)
            rows.append({
                "program_id": record["program_id"], "code_hash": record.get("code_hash"), "status": "ok",
                "is_baseline": bool(record.get("is_baseline")), "decoder": "telescoping_cpu",
                "decoder_config": verification.decoder_build.telescoping.to_dict(),
                "noise_point": {"profile": "uniform_paper_v1", "axis": "p", "value": 0.005}, "noise_value": 0.005,
                "rounds": verification.rounds, "k": 12 if record.get("is_baseline") else 8, "n": 72, "d_search": 6,
                "family": "bb", "shots_x": shots, "shots_z": shots,
                "failures_x": int(round(rate * shots)), "failures_z": int(round(rate * shots * 0.9)),
                "per_logical_per_round": rate / 96, "per_logical_per_round_ci": [rate / 200, rate / 50],
            })
        (out_dir / "rankings.json").write_text(json.dumps({"schema": "slow_rankings", "rankings": rows}))
        return {"status": "measurement_blocked" if blocked else "ok",
                "artifact_paths": {"rankings": str(out_dir / "rankings.json")}}

    return fake


def _state(tmp_path: Path, *, budget: int = 144, hof: list[dict] | None = None, task_kwargs: dict | None = None) -> dict:
    protocol = _snapshot(n_budget=budget, **(task_kwargs or {}))
    return {
        "run_dir": str(tmp_path), "run_id": "campaign_b144", "protocol": protocol,
        "spec": {"qubit_budget": budget, "families": ["bb"], "weight_limit": None},
        "hof_summary": _hof() if hof is None else hof, "baseline_summary": {}, "unmeasurable_hashes": [],
        "structural_proxy_elites": [], "budget": {"turns_used": 30}, "stop": {"reason": "max_turns"},
    }


def _run_final(monkeypatch, tmp_path, calls, **kwargs):
    import omniqec_agentic.workflow.graph as graph

    monkeypatch.setattr(graph, "evaluate_promotion_queue", _fake_pool(calls, fail_part=kwargs.pop("fail_part", None)))
    monkeypatch.setattr(graph, "_validate_all_turn_audits", lambda state: {"enabled": True})
    state = _state(tmp_path, **kwargs)
    return graph, state, graph.final_slow_node(state)


def test_final_stage_confirms_tops_up_pools_and_names_the_pooled_champion(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    graph, state, out = _run_final(monkeypatch, tmp_path, calls)
    final = out["final_slow"]
    assert final["status"] == "ok" and final["final_stage_id"] == FINAL_STAGE_PAPER_ID
    # finalists: a, b, c, d, e (the twin of a is skipped, f is sixth); the reference BB72 is measured too
    shortlist = json.loads((tmp_path / "final_slow" / "final_shortlist.json").read_text())
    assert [row["code_hash"][:1] for row in shortlist["selected"]] == ["a", "b", "c", "d", "e"]
    assert shortlist["skipped_duplicates"][0]["code_hash"].startswith("twin")
    # a and c share a search program id; the final queue gives each its own
    confirm = calls[0]
    assert confirm["dir"] == "T12" and confirm["shots"] == 25_000 and confirm["shards"] == 128
    assert confirm["programs"][:2] == ["g0001_p0001_haaaaaaaa", "g0002_p0001_hbbbbbbbb"]
    assert confirm["programs"][-1] == "baseline_bb_72_12_6" and len(set(confirm["programs"])) == 6
    assert confirm["seed"] == graph.final_rng_base_seed("final_confirmation_v1")
    assert all(call["stage"] == "final" and call["resume"] and call["rounds"] == 12 and not call["include_baseline"] for call in calls)
    # top-up: finalists to 175 000, the reference to ITS paper total (BB72: 50 000), separate seeds
    topups = {call["dir"]: call for call in calls[1:]}
    assert topups["T12_finalists_p1"]["shots"] == 150_000 and len(topups["T12_finalists_p1"]["programs"]) == 5
    assert topups["T12_reference_p1"]["programs"] == ["baseline_bb_72_12_6"] and topups["T12_reference_p1"]["shots"] == 25_000
    assert topups["T12_finalists_p1"]["seed"] == graph.final_rng_base_seed("final_topup_v1/p1")
    assert topups["T12_reference_p1"]["seed"] == graph.final_rng_base_seed("final_topup_v1/reference/p1")
    assert len({call["seed"] for call in calls}) == len(calls)
    ranking = json.loads((tmp_path / "final_slow" / "final_ranking.json").read_text())
    # b was second in the search and is best pooled (rate 2e-4): it is the champion; a (search #1) is second
    assert ranking["champion"]["program_id"] == "g0002_p0001_hbbbbbbbb"
    finalists = [row for row in ranking["rows"] if not row["is_baseline"]]
    assert [row["code_hash"][:1] for row in finalists] == ["b", "a", "c", "d", "e"]
    assert [row["final_rank"] for row in finalists] == [1, 2, 3, 4, 5]
    assert [row["search_rank"] for row in finalists] == [2, 1, 3, 4, 5]
    for row in ranking["rows"]:
        total = 50_000 if row["is_baseline"] else 175_000
        assert row["shots_x"] == row["shots_z"] == total == row["cumulative_cap"] and row["cumulative_complete"]
    assert ranking["reference_rows"] == [{"program_id": "baseline_bb_72_12_6", "code_hash": BB72_HASH,
                                          "shared_with_finalist": False, "measured_as": "baseline_bb_72_12_6",
                                          "shots_per_basis": 50_000}]
    assert "reported_top" not in ranking  # the main stage reports the champion, not a Top-3
    assert final["final_measurement_audit"]["status"] == "complete"
    # the report's frontier is the pooled ranking
    report_out = graph.report_node({**state, "final_slow": final, "run_id": "campaign_b144"})
    report = json.loads(Path(report_out["report"]["path"]).read_text())
    assert report["champion"] == ranking["champion"]
    assert [row["program_id"] for row in report["final_frontier"]] == [
        row["program_id"] for row in ranking["rows"] if row["per_logical_per_round"] is not None]


def test_reference_shared_with_a_finalist_is_measured_once(monkeypatch, tmp_path) -> None:
    """The generator's own BB72 is a finalist (N = 144 memory): one measurement sequence -- the confirmation, the
    reference's top-up to 50 000, then an extension to Table VII's 175 000. The reference row pools the first two
    (50 000), the finalist row all three (175 000)."""

    hof = _hof()
    hof[0] = dict(hof[0], code_hash=BB72_HASH, canonical_hash="canon_ref")
    calls: list[dict] = []
    graph, state, out = _run_final(monkeypatch, tmp_path, calls, hof=hof)
    assert "baseline_bb_72_12_6" not in calls[0]["programs"] and len(calls[0]["programs"]) == 5
    shortlist = json.loads((tmp_path / "final_slow" / "final_shortlist.json").read_text())
    assert shortlist["reference"]["row"] == "shared_with_finalist"
    shared_id = next(row["program_id"] for row in shortlist["selected"] if row["also_reference"])
    by_dir = {call["dir"]: call for call in calls[1:]}
    assert list(by_dir) == ["T12_finalists_p1", "T12_reference_p1", "T12_shared_finalist_p1"]
    assert shared_id not in by_dir["T12_finalists_p1"]["programs"] and len(by_dir["T12_finalists_p1"]["programs"]) == 4
    assert by_dir["T12_reference_p1"]["programs"] == [shared_id] and by_dir["T12_reference_p1"]["shots"] == 25_000
    assert by_dir["T12_reference_p1"]["seed"] == graph.final_rng_base_seed("final_topup_v1/reference/p1")
    assert by_dir["T12_shared_finalist_p1"]["programs"] == [shared_id]
    assert by_dir["T12_shared_finalist_p1"]["shots"] == 125_000
    assert by_dir["T12_shared_finalist_p1"]["seed"] == graph.final_rng_base_seed("final_topup_v1/shared_finalist/p1")
    rows = out["final_slow"]["ranking"]
    finalist = next(row for row in rows if row["also_reference"])
    reference = next(row for row in rows if row["is_baseline"])
    assert sum(1 for row in rows if row["is_baseline"]) == 1 and sum(1 for row in rows if row["also_reference"]) == 1
    assert finalist["program_id"] == shared_id and finalist["final_rank"] is not None
    assert finalist["shots_x"] == finalist["shots_z"] == 175_000 == finalist["cumulative_cap"]
    assert [part["part"] for part in finalist["parts"]] == ["confirm", "topup_reference_p1", "topup_shared_finalist_p1"]
    assert reference["program_id"] == "baseline_bb_72_12_6" and reference["measured_as"] == shared_id
    assert reference["shared_with_finalist"] and reference["final_rank"] is None and not reference["champion"]
    assert reference["shots_x"] == reference["shots_z"] == 50_000 == reference["cumulative_cap"]
    assert [part["part"] for part in reference["parts"]] == ["confirm", "topup_reference_p1"]
    # the reference row is a prefix of the finalist's parts: its counts are the first two parts' counts
    assert reference["failures_x"] == finalist["parts"][0]["failures_x"] + finalist["parts"][1]["failures_x"]
    assert out["final_slow"]["final_measurement_audit"]["status"] == "complete"
    ranking = json.loads((tmp_path / "final_slow" / "final_ranking.json").read_text())
    assert ranking["reference_rows"] == [{"program_id": "baseline_bb_72_12_6", "code_hash": BB72_HASH,
                                          "shared_with_finalist": True, "measured_as": shared_id,
                                          "shots_per_basis": 50_000}]


def test_reference_at_n360_tops_up_to_its_own_budget(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    _, _, out = _run_final(monkeypatch, tmp_path, calls, budget=360)
    by_dir = {call["dir"]: call for call in calls}
    assert [by_dir[f"T12_finalists_p{i}"]["shots"] for i in (1, 2)] == [850_000, 850_000]
    assert by_dir["T12_reference_p1"]["shots"] == 75_000 and by_dir["T12_reference_p1"]["programs"] == ["baseline_bb_144_12_12"]
    ref = next(row for row in out["final_slow"]["ranking"] if row["is_baseline"])
    assert ref["shots_x"] == 100_000 and ref["cumulative_cap"] == 100_000


def test_interrupted_final_stage_resumes_with_the_same_queues_and_seeds(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    with pytest.raises(RuntimeError, match="resumed"):
        _run_final(monkeypatch, tmp_path, calls, fail_part="T12_finalists_p1")
    assert not (tmp_path / "final_slow" / "final_ranking.json").exists()
    first = [(c["dir"], c["programs"], c["seed"], c["shards"], c["shots"]) for c in calls]
    again: list[dict] = []
    _, _, out = _run_final(monkeypatch, tmp_path, again)
    assert out["final_slow"]["status"] == "ok"
    assert [(c["dir"], c["programs"], c["seed"], c["shards"], c["shots"]) for c in again][: len(first)] == first
    assert all(c["resume"] for c in again)  # the pool reuses every candidate an earlier attempt finished


def test_cnot_task_counts_two_k_logicals(monkeypatch, tmp_path) -> None:
    from omniqec_agentic.config import fig3_cnot_structural_config

    structural = fig3_cnot_structural_config(VerificationConfig().structural)
    verification = VerificationConfig(task="cnot_dense", structural=structural, promote_top=int(structural.slow_candidate_cap),
                                      rounds_policy="fixed_Tc")
    calls: list[dict] = []
    _, _, out = _run_final(monkeypatch, tmp_path, calls, task_kwargs={"verification": verification})
    assert out["final_slow"]["final_stage"]["cumulative_cap"] == 75_000
    rows = [row for row in out["final_slow"]["ranking"] if not row["is_baseline"]]
    assert all(row["logicals"] == 16 and row["shots_x"] == 75_000 for row in rows)


def test_baselines_take_the_same_final_rule(monkeypatch, tmp_path) -> None:
    import omniqec_agentic.workflow.graph as graph

    state = _state(tmp_path)
    state["protocol"] = dict(state["protocol"], search_baseline={"method": "random"})
    entries, composition = graph._paper_final_entries(state)
    assert composition["selection_source"] == "search_measured_ranking" and composition["search_method"] == "random"
    assert [e["code_hash"][:1] for e in entries if not e["is_baseline"]] == ["a", "b", "c", "d", "e"]


def test_exact_weight_runs_take_six_finalists(monkeypatch, tmp_path) -> None:
    """Fig. 4 c (2026-11-10 revision): six candidates per condition, by the same search-stage selection."""

    import omniqec_agentic.workflow.graph as graph

    state = _state(tmp_path, hof=_hof() + [_hof_row("g", 7e-5, 7)], task_kwargs=_fixed_weight_kwargs(6))
    assert state["protocol"]["final_stage"]["id"] == config_mod.FINAL_STAGE_FIXED_WEIGHT_ID
    entries, composition = graph._paper_final_entries(state)
    assert [e["code_hash"][:1] for e in entries if not e["is_baseline"]] == ["a", "b", "c", "d", "e", "f"]
    assert composition["top"] == 6 and composition["skipped_duplicates"][0]["code_hash"].startswith("twin")


def test_validator_accepts_the_run_and_rejects_tampering(monkeypatch, tmp_path) -> None:
    from omniqec_agentic.hpc import validate_fig2_campaign as v

    calls: list[dict] = []
    graph, state, out = _run_final(monkeypatch, tmp_path, calls)
    report_out = graph.report_node({**state, "final_slow": out["final_slow"], "run_id": "campaign_b144"})
    report = json.loads(Path(report_out["report"]["path"]).read_text())
    v.configure_profile("fig2_final")
    try:
        v._validate_paper_final(report, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        bad = json.loads(json.dumps(report))
        bad["final_stage_parts"][1]["shots"] = 100_000
        with pytest.raises(ValueError, match="planned|expected 100000"):
            v._validate_paper_final(bad, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        bad = json.loads(json.dumps(report))
        bad["final_stage_parts"] = bad["final_stage_parts"][:-1]          # the reference's top-up dropped
        with pytest.raises(ValueError, match="missing"):
            v._validate_paper_final(bad, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        path = tmp_path / "final_slow" / "final_ranking.json"
        ranking = json.loads(path.read_text())
        ranking["rows"][0]["failures_x"] += 1
        path.write_text(json.dumps(ranking))
        with pytest.raises(ValueError, match="pools"):
            v._validate_paper_final(report, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
    finally:
        v.configure_profile("v3_7")


# ---------------------------------------------------------------- the search-stage early stop
def test_paper_racing_policy_is_appendix_d3() -> None:
    racing = paper_racing(25_000)
    assert (racing.shots_pilot, racing.batch, racing.shots_confirm, racing.max_errors) == (2_000, 250, 25_000, 30)
    assert racing.epsilon == 9.0 and racing.baseline_arm and racing.strict_exceed and racing.best_bound == "best_estimate"
    assert not racing.max_errors_in_pilot and racing.cost_cap_s_per_shot_per_qubit is None
    assert formal_racing_policy_id(racing.to_dict()) == RACING_POLICY_PAPER_ID
    # the earlier policies' blocks are unchanged (no new keys at their defaults)
    assert "strict_exceed" not in formal_racing(25_000).to_dict() and "best_bound" not in formal_racing(25_000).to_dict()
    assert RacingConfig.from_dict(racing.to_dict()) == racing


def _decide(lb: float, shots: int = 2_000, failures=(5, 5), hof=1e-5, ref=2e-5):
    return racing_decision(shots=shots, basis_failures={"X": failures[0], "Z": failures[1]}, plpr_ci_low=lb,
                           config=paper_racing(25_000), thresholds=RacingThresholds(hof_plpr_ub=hof, baseline_plpr_ub=ref))


def test_early_stop_needs_both_bounds_exceeded_tenfold() -> None:
    assert _decide(2.1e-4)["stopped_by"] == "racing"                     # > 10x both
    assert _decide(1.5e-4)["action"] == "continue"                        # > 10x the best, not the reference
    assert _decide(2.0e-4)["action"] == "continue"                        # equality does not exceed
    assert _decide(1.5e-4, ref=None)["stopped_by"] == "racing"           # no applicable reference: the best only
    assert _decide(2.5e-4, hof=None)["stopped_by"] == "racing"           # nothing measured yet: the reference only
    assert _decide(1.0, hof=None, ref=None)["action"] == "continue"      # neither: no early stop


def test_decisions_start_at_the_pilot_and_stop_at_thirty_failures_or_the_limit() -> None:
    assert _decide(1.0, shots=1_750, failures=(200, 200))["action"] == "continue"   # nothing inside the 2 000
    assert _decide(1e-9, shots=2_000, failures=(30, 30))["stopped_by"] == "max_errors"
    assert _decide(1e-9, shots=2_250, failures=(30, 29))["action"] == "continue"     # both bases need 30
    stop = _decide(1e-9, shots=25_000, failures=(3, 4))
    assert stop["action"] == "stop" and stop["confirmed"] and stop["stopped_by"] is None


def test_thresholds_use_the_best_codes_own_bound_and_the_reference(tmp_path) -> None:
    import omniqec_agentic.workflow.graph as graph

    hof = [
        {"code_hash": "x", "per_logical_per_round": 1e-5, "per_logical_per_round_ci": [5e-6, 4e-5]},
        {"code_hash": "y", "per_logical_per_round": 2e-5, "per_logical_per_round_ci": [1.5e-5, 2.5e-5]},
    ]
    state = {"hof_summary": hof, "baseline_summary": {"per_logical_per_round_ci": [1e-4, 3e-4]},
             "spec": {"qubit_budget": 144}, "run_dir": str(tmp_path)}
    paper = graph._hof_thresholds(state, paper_racing(25_000))
    assert paper == RacingThresholds(hof_plpr_ub=4e-5, baseline_plpr_ub=3e-4)
    legacy = graph._hof_thresholds(state, formal_racing(25_000))
    assert legacy == RacingThresholds(hof_plpr_ub=2.5e-5, baseline_plpr_ub=None)
    # the generator's own copy of the reference stands in for a missing resident row
    copy = {"code_hash": BB72_HASH, "per_logical_per_round": 9e-5, "per_logical_per_round_ci": [8e-5, 1.1e-4]}
    state = {**state, "hof_summary": hof + [copy], "baseline_summary": {}}
    assert graph._hof_thresholds(state, paper_racing(25_000)).baseline_plpr_ub == pytest.approx(1.1e-4)
    # a budget without a reference (N = 180): the best code only
    state = {**state, "spec": {"qubit_budget": 180}}
    assert graph._hof_thresholds(state, paper_racing(25_000)).baseline_plpr_ub is None


def _pool_measure(monkeypatch, *, failures_per_shard: int, window: int | None, thresholds=None) -> dict:
    """The real job pool with a fake decode worker (tests/test_race_pilot_window.py harness)."""

    from omniqec_agentic.evaluation import pool as pool_mod
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}

    def fake(request, *, timeout_s):
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, dict(fine)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        unit = request["units"][0]
        shots = int(request["shots"])
        failures = min(shots, int(failures_per_shard))
        cell = {"status": "ok", "decoder": unit["decoder"], "shots": shots, "failures": failures,
                "logical_error_rate": failures / shots, "runtime_s": 0.001,
                "metrics": {"requested_shots": shots, "actual_shots": shots, "sampling_complete": True,
                            "stopped_by_max_errors": False}}
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, dict(fine)

    monkeypatch.setattr(pool_mod, "run_worker_request", fake)
    if window is None:
        monkeypatch.delenv("OMNIQEC_RACE_PILOT_WINDOW", raising=False)
    else:
        monkeypatch.setenv("OMNIQEC_RACE_PILOT_WINDOW", str(window))
    spec = run_construct_source(SEED_BB, program_id="paper_race_bb", seed=0).constructed.spec
    return pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_paper", "spec": spec}],
        verification=VerificationConfig(shots=25_000, max_errors=None, rounds=12, rounds_policy="fixed_Tc"),
        hardware=None, seed=0, batch_size=256, prefer_coloring=True, workers=4, shot_shards=16,
        racing=paper_racing(25_000), racing_thresholds=thresholds,
    )[0]


@pytest.mark.parametrize("window", [None, 2])
def test_pool_decides_first_at_the_2000_shot_pilot_and_never_stalls(monkeypatch, window) -> None:
    # 40 failures per 250-shot shard: 30 failures are reached in the first shard, but the paper decides first
    # after the 2 000-shot initial sample (the earlier pilot10x_maxerr30_v3 stopped at 250).
    early = _pool_measure(monkeypatch, failures_per_shard=40, window=window)
    assert early["racing"]["stopped_by"] == "max_errors" and early["racing"]["shots_per_basis"] == 2_000
    assert [stage["shots_per_basis"] for stage in early["racing"]["stages"]] == [2_000]
    # 1 failure per shard: 30 failures per basis after 30 shards = 7 500 shots, decided batch by batch
    late = _pool_measure(monkeypatch, failures_per_shard=1, window=window)
    assert late["racing"]["stopped_by"] == "max_errors" and late["racing"]["shots_per_basis"] == 7_500
    stages = [stage["shots_per_basis"] for stage in late["racing"]["stages"]]
    assert stages[0] == 2_000 and all(b - a == 250 for a, b in zip(stages, stages[1:]))
    # no failures: the 25 000-shot limit
    none = _pool_measure(monkeypatch, failures_per_shard=0, window=window)
    assert none["racing"]["confirmed"] and none["racing"]["shots_per_basis"] == 25_000


def test_pool_early_stop_needs_both_thresholds(monkeypatch) -> None:
    # 10 failures per shard: the Wilson lower bound sits far above 10x a tiny best-code bound, but not above
    # 10x a reference bound of 1 -> no early stop; with the reference bound tiny as well -> racing at 2 000.
    keep = _pool_measure(monkeypatch, failures_per_shard=10, window=None,
                         thresholds=RacingThresholds(hof_plpr_ub=1e-9, baseline_plpr_ub=1.0))
    assert keep["racing"]["stopped_by"] == "max_errors" and keep["racing"]["shots_per_basis"] == 2_000
    stop = _pool_measure(monkeypatch, failures_per_shard=10, window=None,
                         thresholds=RacingThresholds(hof_plpr_ub=1e-9, baseline_plpr_ub=1e-9))
    assert stop["racing"]["stopped_by"] == "racing" and stop["racing"]["shots_per_basis"] == 2_000
    assert stop["status"] == "ok"


def test_top5_extension_refuses_a_paper_final_run(tmp_path) -> None:
    from omniqec_agentic.hpc import top5_extension

    run = tmp_path / "run"
    run.mkdir()
    (run / "protocol_snapshot.json").write_text(json.dumps(_snapshot()))
    with pytest.raises(SystemExit, match="already confirms"):
        top5_extension.run_info(run)


@pytest.mark.parametrize("ablation", ["no_slow", "no_feedback"])
def test_no_slow_ablation_takes_five_structural_codes_in_slot_order(tmp_path, ablation) -> None:
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic import ranking

    state = _state(tmp_path, budget=180)
    state["protocol"] = dict(state["protocol"], ablation=ablation)
    elites = [{"code_hash": f"{i:02x}" * 12, "program_id": f"e{i}", "family": "bb", "n": 72, "n_total": 144,
               "k": 2 + (i % 4), "d_search": 5 + (i % 3), "max_check_weight": 4 + (i % 3),
               "spec": {"family": "bb"}, "pareto_rank": 0, "first_seen_turn": 0} for i in range(10)]
    state["structural_proxy_elites"] = elites
    entries, composition = graph._paper_final_entries(state)
    assert composition["selection_source"] == "structural_front_slot_order" and composition["reference"] is None
    expected = ranking.select_structural_slots(elites, slots=5, distinct_families=True)  # v3_8 snapshot
    assert [row["code_hash"] for row in entries] == [row["code_hash"] for row in expected][:5]
    assert len(entries) == 5 and all(row["search_rank"] == rank for rank, row in enumerate(entries, start=1))


# ---------------------------------------------------------------- 2026-11-10 paper revision
# Appendix D3: "In both the quantum-memory and transversal-CNOT experiments, the reported LERs for the BB [[72,12,6]]
# and [[144,12,12]] reference codes are based on 50,000 and 100,000 shots per basis, respectively."
# Fig. 4 c: "Six candidates are evaluated in the final stage for each condition. The figure reports the mean LER and
# the minimum-maximum range of the three best codes under each weight constraint."
BB144_HASH = fs.resident_reference(288)["code_hash"]


def _hof_row(code: str, plpr: float, i: int) -> dict:
    spec = {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [1, 0], [0, 2 + i]], "B": [[0, 0], [0, 1], [2, 0]]}
    return {"code_hash": code * 8, "program_id": f"g0003_p{i:04d}", "canonical_hash": f"canon_{code}", "spec": spec,
            "per_logical_per_round": plpr, "per_logical_per_round_ci": [plpr / 2, plpr * 2], "k": 8, "n": 72,
            "d_search": 6, "family": "bb"}


def _cnot_verification(**extra) -> VerificationConfig:
    from omniqec_agentic.config import fig3_cnot_structural_config

    structural = fig3_cnot_structural_config(VerificationConfig().structural)
    return VerificationConfig(task="cnot_dense", structural=structural, promote_top=int(structural.slow_candidate_cap),
                              rounds_policy="fixed_Tc", **extra)


def _fixed_weight_kwargs(weight: int) -> dict:
    return {"verification": _cnot_verification(max_check_weight=weight, exact_check_weight=weight),
            "promotion_max_check_weight": weight}


def test_reference_totals_are_the_paper_numbers_and_match_the_reference_codes() -> None:
    from omniqec_agentic.evaluation.baseline import RESIDENT_BASELINE_BUDGETS

    assert config_mod.REFERENCE_CODE_SHOTS_PER_BASIS == {"baseline_bb_72_12_6": 50_000, "baseline_bb_144_12_12": 100_000}
    assert set(config_mod.REFERENCE_CODE_SHOTS_PER_BASIS) == set(RESIDENT_BASELINE_BUDGETS)
    assert fs.resident_reference(144)["shots_per_basis"] == 50_000 and fs.resident_reference(288)["shots_per_basis"] == 100_000


@pytest.mark.parametrize("task", ["memory", "cnot_dense"])
@pytest.mark.parametrize(("budget", "total"), [(144, 50_000), (288, 100_000), (320, 100_000), (360, 100_000)])
@pytest.mark.parametrize("exact", [None, 6])
def test_reference_total_is_the_same_in_both_tasks_and_both_stages(task, budget, total, exact) -> None:
    block = fs.final_stage_protocol(task=task, n_budget=budget, confirm_shots=25_000, final_shot_shards=128,
                                    exact_check_weight=exact)
    reference = block["reference"]
    assert reference["cumulative_cap"] == total and reference["cumulative_cap_source"] == "paper_reference_shots"
    assert reference["shared_with_finalist"] == fs.SHARED_REFERENCE_RULE_ID
    parts = fs.topup_parts(block, cumulative_cap=reference["cumulative_cap"], group="reference", rounds=12)
    assert 25_000 + sum(part["shots"] for part in parts) == total
    assert all(part["rng_namespace"].startswith("final_topup_v1/reference/") for part in parts)
    # the finalists' rule is untouched: Table VII (main) or none (Fig. 4 c)
    if exact is None:
        assert block["cumulative_cap"] == FINAL_CUMULATIVE_SHOT_CAPS[task][budget] and block["top"] == 5
    else:
        assert block["cumulative_cap"] is None and block["top"] == 6


def test_fixed_weight_protocol_block() -> None:
    block = fs.final_stage_protocol(task="cnot_dense", n_budget=98, confirm_shots=25_000, final_shot_shards=128,
                                    exact_check_weight=5)
    assert block["id"] == config_mod.FINAL_STAGE_FIXED_WEIGHT_ID == "paper_fig4c_top6_confirm_v1"
    assert (block["top"], block["reported_top"], block["exact_check_weight"]) == (6, 3, 5)
    assert block["confirm_shots"] == 25_000 and block["cumulative_cap"] is None
    assert block["cumulative_cap_source"] == "none_confirmation_only" and block["reference"] is None
    assert block["selection"] == fs.SELECTION_RULE_ID and block["reported_statistic"] == fs.REPORTED_TOP_STATISTIC
    assert fs.finalist_total(block) == 25_000
    assert fs.plan_final_stage(block, [{"program_id": f"p{i}"} for i in range(6)], rounds=12,
                               confirmed={f"p{i}" for i in range(6)})["topups"] == []
    assert fs.is_paper_final_stage({"final_stage": block}) and fs.is_fixed_weight_stage(block)
    # a smoke override sets the reference's total only (the finalists have no top-up)
    smoke = fs.final_stage_protocol(task="cnot_dense", n_budget=144, confirm_shots=100, final_shot_shards=4,
                                    cumulative_override=300, exact_check_weight=6)
    assert smoke["cumulative_cap"] is None and smoke["reference"]["cumulative_cap"] == 300


def test_the_exact_contract_selects_the_fixed_weight_stage_in_the_snapshot() -> None:
    fixed = _snapshot(**_fixed_weight_kwargs(6))
    assert fixed["exact_check_weight"] == 6
    assert fixed["final_stage"] == fs.final_stage_protocol(task="cnot_dense", n_budget=144, confirm_shots=25_000,
                                                           final_shot_shards=128, exact_check_weight=6)
    named = _snapshot(final_stage_id=config_mod.FINAL_STAGE_FIXED_WEIGHT_ID, **_fixed_weight_kwargs(6))
    assert named["final_stage"] == fixed["final_stage"]
    with pytest.raises(ValueError, match="exact check-weight contract"):
        _snapshot(final_stage_id=config_mod.FINAL_STAGE_FIXED_WEIGHT_ID)
    # the main searches (both tasks) and the ablations keep the top-5 cumulative stage
    main_cnot = _snapshot(verification=_cnot_verification(), promotion_max_check_weight=8)
    assert main_cnot["final_stage"]["id"] == FINAL_STAGE_PAPER_ID and main_cnot["final_stage"]["top"] == 5
    assert main_cnot["final_stage"]["cumulative_cap"] == 75_000
    for ablation in ("no_slow", "no_feedback"):
        ablated = _snapshot(ablation=ablation, feedback_mode=config_mod.ABLATION_FEEDBACK_MODE[ablation])
        assert ablated["final_stage"] == _snapshot()["final_stage"]
        assert ablated["final_stage"]["id"] == FINAL_STAGE_PAPER_ID and ablated["final_stage"]["cumulative_cap"] == 175_000


def test_cli_exact_weight_run_freezes_the_fixed_weight_stage(monkeypatch, tmp_path) -> None:
    """The CLI path (run-campaign --exact-check-weight, what the PBS PRESET path calls) reaches the real
    run_campaign, whose snapshot carries the fixed-weight stage; without the contract it is the top-5 stage."""

    from omniqec_agentic import cli as cli_mod
    from omniqec_agentic.workflow import graph

    frozen: list[dict] = []
    real = graph.build_protocol_snapshot

    class _Stop(Exception):
        pass

    def spy(**kwargs):
        frozen.append(real(**kwargs))
        return frozen[-1]

    def stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(graph, "build_protocol_snapshot", spy)
    monkeypatch.setattr(graph, "open_checkpointer", stop)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    base = ["run-campaign", "--budget", "144", "--turns", "1", "--provider", "dry_run", "--task", "cnot_dense"]
    with pytest.raises(_Stop):
        cli_mod.main(base + ["--out", str(tmp_path / "w6"), "--exact-check-weight", "6"])
    assert frozen[-1]["final_stage"]["id"] == config_mod.FINAL_STAGE_FIXED_WEIGHT_ID
    assert frozen[-1]["final_stage"]["top"] == 6 and frozen[-1]["final_stage"]["cumulative_cap"] is None
    assert frozen[-1]["final_stage"]["reference"]["cumulative_cap"] == 50_000
    with pytest.raises(_Stop):
        cli_mod.main(base + ["--out", str(tmp_path / "main")])
    assert frozen[-1]["final_stage"]["id"] == FINAL_STAGE_PAPER_ID and frozen[-1]["final_stage"]["cumulative_cap"] == 75_000


def _fixed_weight_run(monkeypatch, tmp_path, *, hof=None, budget=144):
    calls: list[dict] = []
    graph, state, out = _run_final(monkeypatch, tmp_path, calls, budget=budget,
                                   hof=_hof() + [_hof_row("g", 7e-5, 7)] if hof is None else hof,
                                   task_kwargs=_fixed_weight_kwargs(6))
    report_out = graph.report_node({**state, "final_slow": out["final_slow"], "run_id": f"campaign_b{budget}"})
    report = json.loads(Path(report_out["report"]["path"]).read_text())
    return graph, calls, out["final_slow"], report


def test_fixed_weight_final_stage_six_finalists_confirmation_only_and_top3(monkeypatch, tmp_path) -> None:
    graph, calls, final, report = _fixed_weight_run(monkeypatch, tmp_path)
    assert final["status"] == "ok" and final["final_stage_id"] == config_mod.FINAL_STAGE_FIXED_WEIGHT_ID
    # the confirmation: the six leading distinct codes (a..f; twin skipped, g seventh) and the BB72 reference
    confirm = calls[0]
    assert confirm["dir"] == "T12" and confirm["shots"] == 25_000 and confirm["shards"] == 128
    assert confirm["seed"] == graph.final_rng_base_seed("final_confirmation_v1")
    assert len(confirm["programs"]) == 7 and confirm["programs"][-1] == "baseline_bb_72_12_6"
    # no cumulative top-up of the finalists: the only further part is the reference's top-up to 50 000
    assert [(call["dir"], call["programs"], call["shots"]) for call in calls[1:]] == [
        ("T12_reference_p1", ["baseline_bb_72_12_6"], 25_000)]
    finalists = [row for row in final["ranking"] if not row["is_baseline"]]
    assert len(finalists) == 6
    for row in finalists:
        assert row["shots_x"] == row["shots_z"] == 25_000 == row["cumulative_cap"] and row["cumulative_complete"]
        assert [part["part"] for part in row["parts"]] == ["confirm"]
    reference = next(row for row in final["ranking"] if row["is_baseline"])
    assert reference["shots_x"] == reference["shots_z"] == 50_000 and reference["final_rank"] is None
    # ranked by the confirmation measurement: b (2e-4) a c d e f
    assert [row["code_hash"][:1] for row in finalists] == ["b", "a", "c", "d", "e", "f"]
    assert [row["final_rank"] for row in finalists] == [1, 2, 3, 4, 5, 6]
    top = final["reported_top"]
    best = finalists[:3]
    assert top["statistic"] == "arithmetic_mean_min_max_v1" and top["count"] == top["required"] == 3 and top["complete"]
    assert [row["program_id"] for row in top["rows"]] == [row["program_id"] for row in best]
    values = [row["per_logical_per_round"] for row in best]
    assert top["mean_per_logical_per_round"] == pytest.approx(sum(values) / 3, rel=1e-12)
    assert (top["min_per_logical_per_round"], top["max_per_logical_per_round"]) == (min(values), max(values))
    ranking = json.loads((tmp_path / "final_slow" / "final_ranking.json").read_text())
    assert ranking["reported_top"] == json.loads(json.dumps(top)) == report["final_reported_top"]
    assert ranking["champion"]["program_id"] == best[0]["program_id"]
    assert final["final_measurement_audit"]["status"] == "complete"
    assert final["final_measurement_audit"]["cumulative_cap"] is None
    assert "no top-up" in (tmp_path / "campaign_report.md").read_text()


def test_fixed_weight_stage_with_a_short_pool_reports_what_it_has(monkeypatch, tmp_path) -> None:
    hof = [_hof_row("a", 1e-5, 0), _hof_row("b", 2e-5, 1)]
    _, calls, final, _ = _fixed_weight_run(monkeypatch, tmp_path, hof=hof, budget=98)
    assert [len(call["programs"]) for call in calls] == [2]          # N = 98: no reference, no top-up at all
    assert final["reported_top"]["count"] == 2 and not final["reported_top"]["complete"]


def test_fixed_weight_reference_shared_with_a_finalist(monkeypatch, tmp_path) -> None:
    """Fig. 4 c at N = 144, the generator's own BB72 among the six: its finalist row keeps the 25 000-shot
    confirmation (the stage's total), its reference row pools the confirmation and the reference top-up."""

    hof = _hof()
    hof[1] = dict(hof[1], code_hash=BB72_HASH, canonical_hash="canon_ref")
    _, calls, final, _ = _fixed_weight_run(monkeypatch, tmp_path, hof=hof)
    shared_id = next(program for program in calls[0]["programs"] if program.endswith(BB72_HASH[:8]))
    assert "baseline_bb_72_12_6" not in calls[0]["programs"] and len(calls[0]["programs"]) == 6
    assert [(call["dir"], call["programs"], call["shots"]) for call in calls[1:]] == [
        ("T12_reference_p1", [shared_id], 25_000)]
    finalist = next(row for row in final["ranking"] if row["also_reference"])
    reference = next(row for row in final["ranking"] if row["is_baseline"])
    assert finalist["shots_x"] == 25_000 == finalist["cumulative_cap"] and finalist["final_rank"] is not None
    assert [part["part"] for part in finalist["parts"]] == ["confirm"]
    assert reference["shots_x"] == 50_000 == reference["cumulative_cap"] and reference["measured_as"] == shared_id
    assert reference["failures_x"] == finalist["failures_x"] + reference["parts"][1]["failures_x"]
    assert final["final_measurement_audit"]["status"] == "complete"


def test_cnot_n288_reference_shared_with_a_finalist_extends_to_table_vii(monkeypatch, tmp_path) -> None:
    hof = _hof()
    hof[0] = dict(hof[0], code_hash=BB144_HASH, canonical_hash="canon_ref")
    calls: list[dict] = []
    graph, state, out = _run_final(monkeypatch, tmp_path, calls, hof=hof, budget=288,
                                   task_kwargs={"verification": _cnot_verification()})
    by_dir = {call["dir"]: (call["shots"], len(call["programs"])) for call in calls[1:]}
    assert by_dir == {"T12_finalists_p1": (150_000, 4), "T12_finalists_p2": (150_000, 4),
                      "T12_reference_p1": (75_000, 1),
                      "T12_shared_finalist_p1": (112_500, 1), "T12_shared_finalist_p2": (112_500, 1)}
    rows = out["final_slow"]["ranking"]
    finalist = next(row for row in rows if row["also_reference"])
    reference = next(row for row in rows if row["is_baseline"])
    assert finalist["shots_x"] == 325_000 and reference["shots_x"] == 100_000
    plain = [row for row in rows if not row["is_baseline"] and not row["also_reference"]]
    assert all(row["shots_x"] == 325_000 for row in plain)
    report_out = graph.report_node({**state, "final_slow": out["final_slow"], "run_id": "campaign_b288"})
    report = json.loads(Path(report_out["report"]["path"]).read_text())
    from omniqec_agentic.hpc import validate_fig2_campaign as v

    v.configure_profile("fig3_final")
    try:
        v._validate_paper_final(report, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        path = tmp_path / "final_slow" / "final_ranking.json"
        ranking = json.loads(path.read_text())
        bad_reference = next(row for row in ranking["rows"] if row["is_baseline"])
        bad_reference["shots_x"] = bad_reference["shots_z"] = 325_000   # a reference row pooling every shared part
        path.write_text(json.dumps(ranking))
        with pytest.raises(ValueError, match="pools"):
            v._validate_paper_final(report, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
    finally:
        v.configure_profile("v3_7")


def test_validator_checks_the_fixed_weight_stage(monkeypatch, tmp_path) -> None:
    from omniqec_agentic.hpc import validate_fig2_campaign as v

    _, _, final, report = _fixed_weight_run(monkeypatch, tmp_path)
    try:
        v.configure_profile("fig3_final", exact_check_weight=6)
        policy = config_mod.budget_policy(144)
        protocol = report["protocol"]
        v._validate_final_protocol_keys(protocol, policy)
        v._validate_paper_final(report, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        for bad in ({**protocol, "exact_check_weight": 5},
                    {**protocol, "promotion_max_check_weight": 8},
                    {**protocol, "final_stage": fs.final_stage_protocol(task="cnot_dense", n_budget=144,
                                                                        confirm_shots=25_000, final_shot_shards=128)}):
            with pytest.raises(ValueError):
                v._validate_final_protocol_keys(bad, policy)
        bad = json.loads(json.dumps(report))
        bad["final_reported_top"]["mean_per_logical_per_round"] *= 1.5
        with pytest.raises(ValueError, match="reported_top"):
            v._validate_paper_final(bad, p_value=0.005, final_shots=25_000, run_dir=tmp_path)
        # without --exact-check-weight the final profile refuses an exact-weight run
        v.configure_profile("fig3_final")
        with pytest.raises(ValueError, match="exact_check_weight"):
            v._validate_final_protocol_keys(protocol, policy)
        with pytest.raises(ValueError, match="fig3_final"):
            v.configure_profile("fig2_final", exact_check_weight=6)
        # the main stage's validation is unchanged: a top-5 run still validates under fig3_final
        v.configure_profile("fig3_final")
        main = _snapshot(verification=_cnot_verification(), promotion_max_check_weight=8)
        v._validate_final_protocol_keys(main, policy)
    finally:
        v.configure_profile("v3_7")
