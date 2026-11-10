"""Fixed-code measurements (2026-10-23): plan rules, protocol overrides, merge."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from omniqec_agentic import config as config_mod
from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.hpc import fixed_code_measure as fcm
from omniqec_agentic.hpc import fixed_code_plan as fcp
from omniqec_agentic.workflow.state import build_protocol_snapshot

BB30 = {"A": [[1, 1], [2, 2]], "B": [[1, 0], [0, 1]], "ell": 3, "family": "bb", "m": 5}  # [[30,2,5]] w4, Fig.2 Claude b60
BB72 = {"A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]], "ell": 6, "family": "bb", "m": 6}


def _snapshot(task: str, **overrides):
    verification = VerificationConfig(task=task, rounds_policy=("fixed_Tc" if task == "cnot_dense" else "max_Tc_d"),
                                      max_check_weight=(6 if task == "cnot_dense" else None))
    kwargs = dict(verification=verification, racing=config_mod.formal_racing(25_000), max_turns=30, proposal_count=24,
                  generations_per_turn=4, slow_workers=0, strict_turns=True, required_provider="openai",
                  required_model="claude-opus-5", rng_seeds_per_program=40, n_budget=144)
    kwargs.update(overrides)
    snap = build_protocol_snapshot(**kwargs)
    snap["protocol_profile"] = fcm.TASK_PROTOCOL_PROFILE[task]
    return snap


def _item(**overrides):
    item = {"key": "k1", "task": "memory", "protocol_run": "fig2_claudesub_b144_v37", "spec": BB30, "p": 0.005,
            "shots_per_basis": 25000, "rounds_floor": 12, "rounds_policy": "fixed_Tc", "max_check_weight": None,
            "seed_namespace": "fixed_code_v1/k1"}
    item.update(overrides)
    return item


# ---------------------------------------------------------------- verification overrides
def test_memory_protocol_at_fixed_T12_and_a_new_p() -> None:
    v = fcm.build_verification(_snapshot("memory"), {"weight_limit": None}, _item(p=0.002, shots_per_basis=123))
    assert v.task == "memory" and v.rounds_policy == "fixed_Tc" and v.rounds == 12
    assert v.p_values == (0.002,) and v.noise.values == (0.002,) and v.noise.profile == "uniform_paper_v1"
    assert v.shots == 123 and v.max_errors is None and v.max_check_weight is None
    assert v.final_decode_shard_timeout_cap_s is None and v.search_decode_shard_timeout_cap_s is None
    assert v.sec_policy.sec_policy_id == "twobasis_nonint_v1"


def test_cnot_protocol_drops_its_search_weight_cap_but_keeps_the_task() -> None:
    v = fcm.build_verification(_snapshot("cnot_dense"), {"weight_limit": None},
                               _item(task="cnot_dense", protocol_run="fig2_claudesub_b144_f3cnot_v2"))
    assert v.task == "cnot_dense" and v.rounds_policy == "fixed_Tc" and v.rounds == 12
    assert v.max_check_weight is None  # the Fig.3 snapshot carries cap 6; a fixed-code evaluation never applies it
    explicit = fcm.build_verification(_snapshot("cnot_dense"), {"weight_limit": None},
                                      _item(task="cnot_dense", protocol_run="x", max_check_weight=6))
    assert explicit.max_check_weight == 6


def test_native_memory_rule_and_watchdog_budget_override() -> None:
    v = fcm.build_verification(_snapshot("memory"), {"weight_limit": None}, _item(rounds_policy="max_Tc_d"), decode_shot_budget_s=30.0)
    assert v.rounds_policy == "max_Tc_d" and v.decode_shot_budget_s == pytest.approx(30.0)
    assert v.rounds == 12  # the floor; T = max(12, d_search) is decided per candidate by the pool


def test_task_and_protocol_mismatches_are_refused() -> None:
    with pytest.raises(SystemExit):
        fcm.build_verification(_snapshot("memory"), {"weight_limit": None}, _item(task="cnot_dense"))
    with pytest.raises(SystemExit):
        fcm.build_verification(_snapshot("cnot_dense"), {"weight_limit": None}, _item(task="memory"))
    with pytest.raises(SystemExit):
        fcm.build_verification(_snapshot("memory"), {"weight_limit": None}, _item(rounds_policy="per_candidate"))
    with pytest.raises(SystemExit):
        fcm.build_verification(_snapshot("memory"), {"weight_limit": None}, _item(p=0.0))


def test_shot_shards_never_below_the_formal_final_partition() -> None:
    protocol = {"final_shot_shards": 128}
    assert fcm.shot_shards_for(protocol, 25000, None) == 128
    assert fcm.shot_shards_for(protocol, 1_000_000, None) == 4096
    assert fcm.shot_shards_for(protocol, 200, 4) == 4


# ---------------------------------------------------------------- plans
def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_fig4_ab_plan_from_the_manifests(tmp_path: Path) -> None:
    a_rows = [
        {"budget": "60", "code_hash": "57cd30c2aaaaaaaaaaaaaaaa", "target_task": "CNOT", "membership_status": "EXISTING_FINALIST",
         "selection_tasks": "Memory", "selection_models": "Claude", "target_rounds": "12", "noise_p": "0.005",
         "measurement_action": "NEW_T12_CONFIRMATION", "reason": "x", "existing_source_file": "", "initial_proposed_shots_per_basis": "25000",
         "measurement_id": "fig4_v25/A/N60/57cd30c2/CNOT/T12/base", "rng_seed": "1", "spec": json.dumps(BB30), "spec_source": "s",
         "w_max": "4", "search_weight_cap_applies": "False", "racing": "False"},
        {"budget": "144", "code_hash": "3c631986bbbbbbbbbbbbbbbb", "target_task": "Memory", "membership_status": "EXISTING_FINALIST",
         "selection_tasks": "CNOT", "selection_models": "GPT", "target_rounds": "12", "noise_p": "0.005",
         "measurement_action": "NEW_T12_CONFIRMATION", "reason": "x", "existing_source_file": "", "initial_proposed_shots_per_basis": "25000",
         "measurement_id": "fig4_v25/A/N144/3c631986/Memory/T12/base", "rng_seed": "2", "spec": json.dumps(BB72), "spec_source": "s",
         "w_max": "6", "search_weight_cap_applies": "False", "racing": "False"},
    ]
    b_rows = [
        {"budget": "288", "model": "Claude", "task": "Memory", "rank": "5", "code_hash": "00dc3a40cccccccccccccccc", "w_max": "6",
         "native_rounds": "14", "source_run": "fig2_claudesub_b288_v37", "source_file": "f", "search_failures_x": "2", "search_failures_z": "0",
         "search_upper_ler": "7.9e-06", "status": "PROPOSED", "spec": json.dumps(BB72), "selection_rule": "r",
         "confirmation_id": "fig4_v25/B/N288/00dc3a40/Memory/T14/native", "rng_seed": "3", "shots_per_basis": "25000",
         "shared_with_A_T12": "False", "racing": "False"},
    ]
    plan = fcp.build_fig4_ab(a_rows, b_rows, structural=True)
    assert plan["schema"] == fcm.PLAN_SCHEMA and len(plan["items"]) == 3
    cnot, mem, native = plan["items"]
    assert cnot["task"] == "cnot_dense" and cnot["protocol_run"] == "fig2_claudesub_b60_f3cnot_v2"
    assert cnot["rounds_policy"] == "fixed_Tc" and cnot["n"] == 30 and cnot["k"] == 2 and cnot["logicals"] == 4 and cnot["w_max"] == 4
    assert mem["task"] == "memory" and mem["protocol_run"] == "fig2_claudesub_b144_v37" and mem["logicals"] == 12 and mem["n"] == 72
    assert native["block"] == "fig4_b" and native["rounds_policy"] == "max_Tc_d" and native["expected_rounds"] == 14
    assert native["protocol_run"] == "fig2_claudesub_b288_v37"
    assert all(i["max_check_weight"] is None and i["p"] == 0.005 and i["shots_per_basis"] == 25000 for i in plan["items"])
    assert len({i["seed_namespace"] for i in plan["items"]}) == 3
    out = tmp_path / "plan.json"
    fcp.write_plan(out, plan)
    assert fcm.load_plan(out)["items"][0]["key"] == cnot["key"]
    # a manifest row that contradicts the built code is refused
    bad = dict(a_rows[0], w_max="6")
    with pytest.raises(SystemExit):
        fcp.build_fig4_ab([bad], [], structural=True)
    # Fig4 a is the T=12 cross-evaluation: another T is a different experiment
    with pytest.raises(SystemExit):
        fcp.build_fig4_ab([dict(a_rows[0], target_rounds="14")], [], structural=False)


def test_s4_shot_sizing_rules() -> None:
    anchor = {"shots_per_basis": 25000, "failures_x": 40, "failures_z": 20, "plpr": None}
    # the pilot saw failures: the weaker basis sizes the run, rounded up to 50k, capped
    shots, rule = fcp.s4_formal_shots({"shots_x": 20000, "shots_z": 20000, "failures_x": 10, "failures_z": 4}, anchor, 6, 0.004,
                                      target_failures=100, cap=1_000_000, round_to=50000)
    assert shots == 500_000 and rule.startswith("pilot:pilot_z")
    shots, rule = fcp.s4_formal_shots({"shots_x": 20000, "shots_z": 20000, "failures_x": 1, "failures_z": 0}, anchor, 6, 0.002,
                                      target_failures=100, cap=1_000_000, round_to=50000)
    assert shots == 1_000_000 and rule.endswith(";capped")
    # no pilot failure: the anchor rate scaled by (p/0.005)^ceil((d+1)/2)
    shots, rule = fcp.s4_formal_shots({"shots_x": 20000, "shots_z": 20000, "failures_x": 0, "failures_z": 0}, anchor, 5, 0.004,
                                      target_failures=100, cap=1_000_000, round_to=50000)
    q = (20 / 25000) * (0.004 / 0.005) ** 3
    assert rule.startswith("anchor_extrapolation") and shots == min(1_000_000, fcp.round_up(100 / q, 50000))
    shots, _ = fcp.s4_formal_shots(None, anchor, 12, 0.001, target_failures=100, cap=1_000_000, round_to=50000)
    assert shots == 1_000_000


def test_s4_plan_has_20_codes_x_8_new_points_and_reuses_the_anchor(tmp_path: Path) -> None:
    rows = []
    for budget in fcp.S4_BUDGETS:
        for arm in fcp.S4_ARMS:
            rows.append({"budget": str(budget), "arm": arm, "code_hash8": f"{budget:03d}{arm[:3].lower()}xx"[:8], "family": "bb", "n": "30",
                         "k": "2", "d_search": "5", "max_check_weight": "4", "rounds": "12", "plotted_plpr": "1e-4",
                         "final_failures_x": "10", "final_failures_z": "8", "extension_shots_per_basis": "", "extension_failures_x": "",
                         "extension_failures_z": "", "source_run": f"fig2_{arm.lower()}_b{budget}_v37", "spec": json.dumps(BB30)})
    rows.append({"budget": "288", "arm": "GPT", "code_hash8": "deadbeef", "family": "bb", "n": "144", "k": "4", "d_search": "14",
                 "max_check_weight": "6", "rounds": "14", "plotted_plpr": "1e-6", "final_failures_x": "1", "final_failures_z": "1",
                 "extension_shots_per_basis": "", "extension_failures_x": "", "extension_failures_z": "", "source_run": "x", "spec": json.dumps(BB72)})
    refs = {"items": [
        {"role": "reference", "budget": 144, "code_hash": "3c631986aaaa", "spec": BB72, "family": "bb", "n": 72, "k": 12, "d_search": 6,
         "max_check_weight": 6, "formal_rounds": 12, "source_run": "fig2_gptsub_b144_v37", "formal_plpr": 8.3e-4},
        {"role": "reference", "budget": 288, "code_hash": "39b6b714aaaa", "spec": dict(BB72, ell=12), "family": "bb", "n": 144, "k": 12,
         "d_search": 12, "max_check_weight": 6, "formal_rounds": 12, "source_run": "fig2_gptsub_b288_v37", "formal_plpr": 1e-5},
        {"role": "champion", "budget": 288, "code_hash": "x", "spec": BB72},
    ]}
    pilot = fcp.build_s4(rows, refs, stage="pilot", pilot_shots=20000)
    assert len(pilot["items"]) == 20 * 8 and all(i["p"] != 0.005 for i in pilot["items"])
    assert all(i["shots_per_basis"] == 20000 and i["rounds_policy"] == "max_Tc_d" and i["task"] == "memory" for i in pilot["items"])
    assert all(i["seed_namespace"].startswith("s4_pilot_v1/") for i in pilot["items"])
    ref = next(i for i in pilot["items"] if "_ref_" in i["key"])
    assert ref["protocol_run"] == "fig2_claudesub_b144_v37" and ref["logicals"] == 12
    fcp.write_plan(tmp_path / "pilot.json", pilot)
    # formal sizing from a merged pilot table: unmeasured pilots fall back to the anchor extrapolation
    merged = [{"key": i["key"], "status": "ok", "shots_x": "20000", "shots_z": "20000", "failures_x": "5", "failures_z": "7"}
              for i in pilot["items"][:3]]
    formal = fcp.build_s4(rows, refs, stage="formal", pilot_results=merged, cap=1_000_000)
    assert len(formal["items"]) == 160 and all(i["seed_namespace"].startswith("s4_formal_v1/") for i in formal["items"])
    sized = {i["key"]: i for i in formal["items"]}
    first = pilot["items"][0]["key"][: -len("_pilot")]
    assert sized[first]["shots_rule"].startswith("pilot:") and sized[first]["shots_per_basis"] == 400_000
    assert all(i["shots_per_basis"] <= 1_000_000 for i in formal["items"])
    with pytest.raises(SystemExit):  # a failed pilot cannot size a formal run silently
        fcp.build_s4(rows, refs, stage="formal", pilot_results=[dict(merged[0], status="decode_infeasible")])


def test_inset_formula() -> None:
    assert fcp.inset_shots(20, 10, 25000, 25000) == 350_000           # q=1.2e-3 -> 333k -> 350k
    assert fcp.inset_shots(0, 0, 25000, 25000) == 10_025_000           # q = 1/25001 -> 10,000,400 -> next 25k multiple
    assert fcp.inset_shots(300, 300, 25000, 25000) == 25_000           # never below the base


def test_inset_plan_unifies_the_two_codes_per_task() -> None:
    def row(task, h, fx, fz):
        return {"key": f"A_N144_{task}_{h}", "task": task, "budget": "144", "code_hash": h + "0" * 16, "status": "ok",
                "failures_x": str(fx), "failures_z": str(fz), "shots_x": "25000", "shots_z": "25000", "family": "bb", "n": "30", "k": "2",
                "d_search": "5", "w_max": "4", "logicals": "2" if task == "memory" else "4", "origin": json.dumps({"spec": BB30})}
    rows = [row("memory", "aaaaaaaa", 20, 10), row("memory", "bbbbbbbb", 2, 1), row("cnot_dense", "aaaaaaaa", 0, 0), row("cnot_dense", "bbbbbbbb", 50, 60),
            row("memory", "cccccccc", 1, 1)]
    plan = fcp.build_a_inset(rows, budget=144, codes=["aaaaaaaa", "bbbbbbbb"])
    by_task = {}
    for item in plan["items"]:
        by_task.setdefault(item["task"], set()).add(item["shots_per_basis"])
    assert by_task["memory"] == {fcp.inset_shots(2, 1, 25000, 25000)} and by_task["cnot_dense"] == {10_025_000}
    assert len(plan["items"]) == 4 and all(i["rounds_policy"] == "fixed_Tc" for i in plan["items"])


def test_supplement_rule_is_the_fig3_rule() -> None:
    assert fcp.supplement_shots(30) == 0 and fcp.supplement_shots(100) == 0
    assert fcp.supplement_shots(0) == 500_000
    assert fcp.supplement_shots(10) == 150_000      # 45*25k/10 = 112.5k -> 150k
    assert fcp.supplement_shots(1) == 500_000       # 1.125M -> capped
    assert fcp.supplement_shots(29) == 50_000       # 38.8k -> 50k


def test_c_supplement_plan_groups_by_budget_and_takes_top_k(tmp_path: Path) -> None:
    def run(name, task, rows):
        d = tmp_path / name
        (d / "final_slow" / "T12" / "candidates").mkdir(parents=True)
        (d / "protocol_snapshot.json").write_text(json.dumps({"verification": {"task": task}, "n_budget": 98}))
        (d / "campaign_spec.json").write_text("{}")
        (d / "final_slow" / "T12" / "rankings.json").write_text(json.dumps({"rankings": rows}))
        # the final queue keeps every finalist's spec by code hash (2026-11-06: the plan's primary source)
        (d / "final_slow" / "queue_T12.jsonl").write_text(
            "".join(json.dumps({"code_hash": r["code_hash"], "program_id": r["program_id"], "spec": BB30}) + "\n" for r in rows))
        for r in rows:
            (d / "final_slow" / "T12" / "candidates" / f"{r['program_id']}.json").write_text(json.dumps({"spec": BB30, "program_id": r["program_id"]}))
    def finalist(pid, plpr, fx, fz, baseline=False):
        return {"program_id": pid, "status": "ok", "is_baseline": baseline, "per_logical_per_round": plpr, "per_logical_per_round_ci": [plpr / 2, plpr * 2],
                "failures_x": fx, "failures_z": fz, "shots_x": 25000, "shots_z": 25000, "code_hash": pid * 3, "family": "bb", "n": 30, "k": 2,
                "d_search": 5, "max_check_weight": 4, "rounds": 12}
    run("f4_b98_cnot_w4_rep1", "cnot_dense", [finalist("p1", 1e-4, 40, 35), finalist("p2", 2e-4, 10, 50), finalist("base", 5e-4, 100, 100, True)])
    run("f4_b98_cnot_w5_rep1", "cnot_dense", [finalist("q1", 3e-4, 0, 5)])
    run("s3_b260_memory_noslow", "memory", [finalist("m1", 1e-5, 3, 4), finalist("m2", 2e-5, 60, 70), finalist("m3", 3e-5, 1, 0)])
    plan, report = fcp.build_c_supplement([str(tmp_path)], ["f4_b98_cnot_w4_rep1", "f4_b98_cnot_w5_rep1"], top=1)
    keys = {i["key"]: i for i in plan["items"]}
    # champion of w4 has f=35 -> own 0, champion of w5 has f=0 -> 500k; the group (N98) maximum applies to both
    assert len(keys) == 2 and all(i["shots_per_basis"] == 500_000 for i in keys.values())
    w4 = next(i for i in keys.values() if "w4" in i["key"])
    assert w4["task"] == "cnot_dense" and w4["logicals"] == 4 and w4["rounds_policy"] == "fixed_Tc" and w4["formal_failures_z"] == 35
    assert w4["budget"] == 98 and w4["protocol_run"] == "f4_b98_cnot_w4_rep1"
    top3, _ = fcp.build_c_supplement([str(tmp_path)], ["s3_b260_memory_noslow"], top=3)
    ranks = sorted(i["origin"]["final_rank"] for i in top3["items"])
    assert ranks == [1, 2, 3] and all(i["shots_per_basis"] == 500_000 for i in top3["items"])  # m3 has f=0 -> group 500k
    assert all(i["rounds_policy"] == "max_Tc_d" and i["logicals"] == 2 for i in top3["items"])


def test_candidate_spec_is_matched_by_code_hash(tmp_path: Path) -> None:
    """Two finalists sharing a program id overwrite each other's candidate file; the spec comes from the
    final queue by code hash, and a candidate file is used only when its compiled hash is the finalist's."""

    other = {**BB30, "A": [[0, 1], [2, 2]]}
    run = tmp_path / "run"
    (run / "final_slow" / "T12" / "candidates").mkdir(parents=True)
    (run / "final_slow" / "queue_T12.jsonl").write_text(
        json.dumps({"code_hash": "aaa", "program_id": "g1", "spec": BB30}) + "\n"
        + json.dumps({"code_hash": "bbb", "program_id": "g1", "spec": other}) + "\n")
    (run / "final_slow" / "T12" / "candidates" / "g1.json").write_text(
        json.dumps({"spec": other, "compiled_code": {"code_hash": "bbb"}}))
    assert fcp._candidate_spec(run, {"program_id": "g1", "code_hash": "aaa"}) == BB30
    assert fcp._candidate_spec(run, {"program_id": "g1", "code_hash": "bbb"}) == other
    (run / "final_slow" / "queue_T12.jsonl").unlink()
    assert fcp._candidate_spec(run, {"program_id": "g1", "code_hash": "bbb"}) == other
    with pytest.raises(SystemExit):   # the surviving file belongs to the other finalist
        fcp._candidate_spec(run, {"program_id": "g1", "code_hash": "aaa"})


# ---------------------------------------------------------------- merge
def test_merge_reports_pending_ok_and_pooled_rows(tmp_path: Path) -> None:
    items = [
        _item(key="done", block="c_supplement", budget=98, code_hash="abc", formal_shots_per_basis=25000, formal_failures_x=10, formal_failures_z=20,
              logicals=4, task="cnot_dense", protocol_run="run"),
        _item(key="later", seed_namespace="fixed_code_v1/later"),
    ]
    plan = {"schema": fcm.PLAN_SCHEMA, "items": items}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    out_root = tmp_path / "out"
    cand_dir = out_root / "done" / "T12" / "candidates"
    cand_dir.mkdir(parents=True)
    cand_dir.joinpath("fixed_abc.json").write_text(json.dumps({
        "compiled_code": {"symplectic_stabilizer_sha256": "sym", "h_mask_sha256": "mask"},
        "results": [{"basis": {
            "X": {"memory_experiment": {"noisy_circuit_sha256": "cx", "dem_sha256": "dx", "num_logicals": 4, "num_data_qubits": 60, "rounds": 12}},
            "Z": {"memory_experiment": {"noisy_circuit_sha256": "cz", "dem_sha256": "dz", "num_logicals": 4, "num_data_qubits": 60, "rounds": 12}},
        }}],
    }))
    row = {"status": "ok", "program_id": "fixed_abc", "rounds": 12, "shots_x": 50000, "shots_z": 50000, "failures_x": 30, "failures_z": 40,
           "ler_x": 6e-4, "ler_z": 8e-4, "block_ler": 1.4e-3, "per_logical_per_round": 2.9e-5, "per_logical_per_round_ci": [2e-5, 4e-5],
           "n": 30, "k": 2, "d_search": 5, "family": "bb", "decoder": "telescoping_cpu", "noise_profile": "uniform_paper_v1",
           "noise_config_hash": "nh", "wl_hash": "wl", "canonical_hash": "ch", "max_check_weight": 4}
    (out_root / "done").mkdir(parents=True, exist_ok=True)
    (out_root / "done" / "result.json").write_text(json.dumps({
        "row": row, "runtime_s": 12.5, "queue_status": "ok",
        "config": {"seed": 123, "rounds_floor": 12, "protocol_profile": "telescoping_structural_v3_7_cnot_v2",
                   "verification": {"sec_policy": {"sec_policy_id": "twobasis_nonint_v1"}, "decoder_build": {"telescoping": {"preset": "telescoping_cpu_v1"}},
                                    "noise": {"profile": "uniform_paper_v1"}}},
    }))
    assert fcm.is_done(out_root, "done") and not fcm.is_done(out_root, "later")
    assert fcm.select_keys(plan, out_root, pending=True, chunk=None) == ["later"]
    assert fcm.select_keys(plan, out_root, pending=False, chunk="2/2") == ["later"]
    rows = [fcm.merge_row(i, out_root) for i in items]
    done, later = rows
    assert later["status"] == "pending" and later["plpr"] is None
    assert done["status"] == "ok" and done["circuit_sha256_x"] == "cx" and done["dem_sha256_z"] == "dz" and done["symplectic_stabilizer_sha256"] == "sym"
    assert done["logicals_measured"] == 4 and done["circuit_data_qubits"] == 60  # two blocks of a [[30,2,5]] code
    assert done["sec_policy_id"] == "twobasis_nonint_v1" and done["decoder_preset"] == "telescoping_cpu_v1"
    assert done["pooled_shots_per_basis"] == 75000 and done["pooled_failures_x"] == 40 and done["pooled_failures_z"] == 60
    expected = fcm.combined_plpr(40, 60, 75000, 75000, 4, 12)
    assert done["pooled_plpr"] == pytest.approx(expected[0]) and done["pooled_plpr_ci_low"] == pytest.approx(expected[1])
    # the CLI round trip
    out_csv = tmp_path / "merged.csv"
    assert fcm.main(["--plan", str(tmp_path / "plan.json"), "--out-root", str(out_root), "--merge", "--out", str(out_csv)]) == 0
    with out_csv.open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert [r["key"] for r in table] == ["done", "later"] and table[0]["pooled_failures_x"] == "40"
