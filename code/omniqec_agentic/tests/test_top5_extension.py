"""Top-5 extension (final protocol, 2026-11-06): the plan rule, the part split and the champion re-selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic.hpc import top5_extension as t5


def _finalist(code_hash, plpr, fx, fz, *, rank_ci=None, program_id="g0001_p0001", k=4, rounds=12, baseline=False, status="ok"):
    return {
        "code_hash": code_hash, "program_id": program_id, "family": "bb", "n": 90, "k": k, "d_search": 9,
        "max_check_weight": 5, "rounds": rounds, "status": status, "is_baseline": baseline,
        "per_logical_per_round": plpr, "per_logical_per_round_ci": [plpr * 0.5, rank_ci if rank_ci is not None else plpr * 2],
        "failures_x": fx, "failures_z": fz, "shots_x": 25000, "shots_z": 25000,
    }


def _run(tmp_path: Path, name: str, rows: list[dict], *, task: str = "memory", budget: int = 180,
         rounds_policy: dict | None = None) -> Path:
    run = tmp_path / "runs" / name
    (run / "final_slow" / "T12").mkdir(parents=True)
    snapshot = {"n_budget": budget, "protocol_profile": "telescoping_structural_v3_8", "verification": {"task": task}}
    if rounds_policy is not None:
        snapshot["rounds_policy"] = rounds_policy
    (run / "protocol_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (run / "final_slow" / "T12" / "rankings.json").write_text(json.dumps({"rankings": rows}), encoding="utf-8")
    with open(run / "final_slow" / "queue_T12.jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps({"code_hash": r["code_hash"], "program_id": r["program_id"],
                                     "spec": {"family": "bb", "tag": r["code_hash"]}}) + "\n")
    return run


CAPS = {"memory": {180: 175000, 360: 1725000}, "cnot_dense": {180: 75000}}


def test_target_rule() -> None:
    assert t5.target_total(30, 175000) == 0
    assert t5.target_total(45, 175000) == 0
    assert t5.target_total(0, 175000) == 175000
    # 45 * 25k / 10 = 112.5k -> 150k
    assert t5.target_total(10, 175000) == 150000
    # 45 * 25k / 3 = 375k -> capped
    assert t5.target_total(3, 175000) == 175000
    assert t5.target_total(29, 875000) == 50000


def test_split_parts() -> None:
    assert t5.split_parts(0, 900000) == []
    assert t5.split_parts(150000, 900000) == [150000]
    assert t5.split_parts(1700000, 900000) == [850000, 850000]
    parts = t5.split_parts(1000001, 150000)
    assert sum(parts) == 1000001 and len(parts) == 7 and max(parts) <= 150000 + 6


def test_plan_takes_top5_and_queue_specs(tmp_path: Path) -> None:
    rows = [
        _finalist("a" * 24, 2.0e-5, 20, 14, program_id="g1"),
        _finalist("b" * 24, 3.0e-5, 40, 35, program_id="g2"),     # >= 30 in both bases: no extension
        _finalist("c" * 24, 3.1e-5, 0, 3, program_id="g3"),       # shares a program id with the next one
        _finalist("d" * 24, 3.2e-5, 10, 11, program_id="g3"),
        _finalist("e" * 24, 4.0e-5, 25, 23, program_id="g5"),
        _finalist("f" * 24, 9.0e-5, 1, 1, program_id="g6"),       # rank 6: never extended
        _finalist("r" * 24, 1.0e-6, 0, 0, program_id="base", baseline=True),
        _finalist("x" * 24, 1.0e-7, 0, 0, program_id="bad", status="decode_infeasible"),
    ]
    _run(tmp_path, "fig2_claudesub_b180_final", rows)
    plan, report = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], CAPS, check_specs=False)
    by_code = {}
    for item in plan["items"]:
        by_code.setdefault(item["code_hash"][:1], []).append(item)
    assert set(by_code) == {"a", "c", "d", "e"}
    # a: f = 14 -> 45 * 25k / 14 = 80.4k -> 100k total, 75k to add
    assert [i["shots_per_basis"] for i in by_code["a"]] == [75000]
    # c: f = 0 -> cap 175k, 150k to add; its spec comes from the queue by code hash, not the shared candidate file
    assert [i["shots_per_basis"] for i in by_code["c"]] == [150000]
    assert by_code["c"][0]["spec"]["tag"] == "c" * 24 and by_code["d"][0]["spec"]["tag"] == "d" * 24
    # d: f = 10 -> 150k total; e: f = 23 -> 48.9k -> 50k total
    assert [i["shots_per_basis"] for i in by_code["d"]] == [125000]
    assert [i["shots_per_basis"] for i in by_code["e"]] == [25000]
    item = by_code["a"][0]
    assert item["rounds_policy"] == "max_Tc_d" and item["logicals"] == 4 and item["expected_rounds"] == 12
    assert item["seed_namespace"] == f"top5_extension_v1/{item['key']}" and item["max_check_weight"] is None
    assert item["origin"]["target_total"] == 100000 and item["origin"]["cap"] == 175000
    assert len(report) == 5


def test_plan_cnot_and_prior_shots(tmp_path: Path) -> None:
    rows = [_finalist("a" * 24, 6.0e-5, 98, 12, program_id="g1")]
    _run(tmp_path, "fig2_claudesub_b180_f3cnot_final", rows, task="cnot_dense")
    prior = {("fig2_claudesub_b180_f3cnot_final", "a" * 24): {"shots": 25000, "failures_x": 90, "failures_z": 10, "parts": 1}}
    plan, _ = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_f3cnot_final"], CAPS, prior=prior, check_specs=False)
    # f = 12 -> 93.75k -> 100k, capped at 75k; 25k final + 25k earlier already measured -> 25k to add
    (item,) = plan["items"]
    assert item["shots_per_basis"] == 25000 and item["rounds_policy"] == "fixed_Tc" and item["logicals"] == 8


def test_plan_follows_the_run_rounds_policy(tmp_path: Path) -> None:
    # final protocols: T = 12 for every code, also on the memory task (a d = 14 code is measured at T = 12)
    fixed = {"search": "fixed_Tc", "final": "fixed_Tc", "search_rounds": 12, "final_rounds": 12}
    _run(tmp_path, "fig2_claudesub_b180_f2final", [_finalist("a" * 24, 2.0e-5, 20, 14, program_id="g1")],
         rounds_policy=fixed)
    plan, _ = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_f2final"], CAPS, check_specs=False)
    (item,) = plan["items"]
    assert item["rounds_policy"] == "fixed_Tc" and item["expected_rounds"] == 12 and item["logicals"] == 4
    mixed = {"search": "max_Tc_d", "final": "fixed_Tc"}
    _run(tmp_path, "fig2_claudesub_b180_mixed", [_finalist("a" * 24, 2.0e-5, 20, 14, program_id="g1")],
         rounds_policy=mixed)
    with pytest.raises(SystemExit):
        t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_mixed"], CAPS, check_specs=False)


def test_plan_splits_large_extensions(tmp_path: Path) -> None:
    rows = [_finalist("a" * 24, 5.0e-7, 0, 1, program_id="g1", rounds=14)]
    _run(tmp_path, "fig2_claudesub_b360_final", rows, budget=360)
    plan, _ = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b360_final"], CAPS, check_specs=False)
    shots = [i["shots_per_basis"] for i in plan["items"]]
    assert sum(shots) == 1725000 - 25000 and len(shots) == 2 and max(shots) <= 900000
    assert len({i["seed_namespace"] for i in plan["items"]}) == 2
    assert all(i["expected_rounds"] == 14 for i in plan["items"])


def test_plan_needs_a_cap(tmp_path: Path) -> None:
    _run(tmp_path, "fig2_claudesub_b98_final", [_finalist("a" * 24, 3e-4, 5, 5)], budget=98)
    with pytest.raises(SystemExit):
        t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b98_final"], CAPS, check_specs=False)


def _write_result(out_root: Path, item: dict, fx: int, fz: int, *, rounds: int = 12, status: str = "ok") -> None:
    key = item["key"]
    cand_dir = out_root / key / "T12" / "candidates"
    cand_dir.mkdir(parents=True)
    (out_root / key / "result.json").write_text(json.dumps({
        "queue_status": "ok", "config": {"rounds_floor": 12},
        "row": {"status": status, "stopped_by": None, "rounds": rounds, "program_id": "fixed", "code_hash": item["code_hash"],
                "failures_x": fx, "failures_z": fz, "shots_x": item["shots_per_basis"], "shots_z": item["shots_per_basis"]},
    }), encoding="utf-8")
    (cand_dir / "fixed.json").write_text(json.dumps({"results": [{"basis": {
        "X": {"memory_experiment": {"rounds": rounds}}, "Z": {"memory_experiment": {"rounds": rounds}}}}]}), encoding="utf-8")


def test_select_reselects_the_champion(tmp_path: Path) -> None:
    rows = [
        _finalist("a" * 24, 2.0e-5, 20, 14, program_id="g1"),
        _finalist("b" * 24, 3.0e-5, 40, 35, program_id="g2"),
        _finalist("c" * 24, 3.1e-5, 0, 3, program_id="g3"),
    ]
    _run(tmp_path, "fig2_claudesub_b180_final", rows)
    plan, _ = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], CAPS, check_specs=False)
    out_root = tmp_path / "out"
    for item in plan["items"]:
        if item["code_hash"].startswith("a"):
            _write_result(out_root, item, 70, 80)     # a regresses: 90/94 at 100k
        else:
            _write_result(out_root, item, 15, 16)     # c holds: 15/19 at 175k
    table = t5.select([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], plan, [str(out_root)])
    by = {r["code_hash8"][0]: r for r in table}
    assert by["c"]["champion"] == "YES" and by["c"]["pooled_rank"] == 1
    assert by["a"]["pooled_failures_x"] == 90 and by["a"]["pooled_shots_per_basis"] == 100000
    assert by["b"]["extension_status"] == "not_needed" and by["b"]["pooled_shots_per_basis"] == 25000
    assert by["c"]["ge30_each_basis"] == "NO" and by["c"]["pooled_shots_per_basis"] == 175000
    assert all(r["run_status"] == "complete" for r in table)


def test_select_refuses_an_incomplete_run(tmp_path: Path) -> None:
    rows = [_finalist("a" * 24, 2.0e-5, 20, 14, program_id="g1"), _finalist("b" * 24, 3.0e-5, 5, 6, program_id="g2")]
    _run(tmp_path, "fig2_claudesub_b180_final", rows)
    plan, _ = t5.build_plan([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], CAPS, check_specs=False)
    out_root = tmp_path / "out"
    first = next(i for i in plan["items"] if i["code_hash"].startswith("a"))
    _write_result(out_root, first, 30, 30)
    second = next(i for i in plan["items"] if i["code_hash"].startswith("b"))
    _write_result(out_root, second, 30, 30, rounds=14)      # measured at the wrong T
    table = t5.select([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], plan, [str(out_root)])
    assert not any(r["champion"] for r in table)
    assert all(r["run_status"] == "incomplete" for r in table)
    assert {r["code_hash8"][0]: r["extension_status"] for r in table}["b"] == "incomplete"
    table = t5.select([str(tmp_path / "runs")], ["fig2_claudesub_b180_final"], plan, [str(out_root)], allow_incomplete=True)
    assert sum(1 for r in table if r["champion"]) == 1


def test_caps_file_is_valid() -> None:
    caps = t5.load_caps(t5.DEFAULT_CAPS)
    assert caps["memory"][360] == 1725000 and caps["cnot_dense"][360] == 875000
    assert set(caps["memory"]) == set(caps["cnot_dense"]) == {60, 98, 144, 180, 220, 260, 288, 320, 360}
