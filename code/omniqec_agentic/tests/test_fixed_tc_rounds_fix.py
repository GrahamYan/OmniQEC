"""2026-10-27: the rounds policy reaches the circuit builder.

Before the fix ``evaluation/pool.py`` computed the candidate's T with the
verification's rounds policy (``fixed_Tc`` -> T_c = 12) and normalised the
row by it, but the build worker called ``build_memory_experiment_pair``
without the policy, so ``select_memory_rounds`` fell back to ``max_Tc_d``:
a fixed_Tc code with d_hint > 12 was MEASURED at T = d_hint (the candidate's
X/Z ``memory_experiment.rounds``) and REPORTED at T = 12.

- the builder: fixed_Tc + d_hint 16 -> 12 rounds; max_Tc_d (the default)
  unchanged; at d_hint <= T_c every policy gives the identical circuit;
- the build worker (job-pool path) follows ``verification.rounds_policy``;
- the pool turns any circuit/row T disagreement into a loud build_error;
- end to end through ``hpc/fixed_code_measure``: the measured X/Z circuits'
  T equals the row's T (fixed_Tc CNOT at 12, native max_Tc_d memory at d);
- fixed_code_measure never counts, merges as ok, or measures over a
  pre-fix result whose circuits ran at the wrong T.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from omniqec_agentic import config as config_mod
from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.construction.build import build_compiled_code
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.hpc import fixed_code_measure as fcm
from omniqec_agentic.verification.circuit import build_memory_experiment_pair
from omniqec_agentic.verification.noise import NoiseSweepConfig

BB30 = {"A": [[1, 1], [2, 2]], "B": [[1, 0], [0, 1]], "ell": 3, "family": "bb", "m": 5}  # [[30,2,5]] w4


def _pair(compiled, **kwargs):
    noise = NoiseSweepConfig.uniform((0.005,)).point(0)
    return build_memory_experiment_pair(
        np.asarray(compiled.hx, dtype=np.uint8), np.asarray(compiled.hz, dtype=np.uint8), p=0.005,
        noise_point=noise, sector_sizes=compiled.sector_sizes, layer_blocks=compiled.layer_blocks, **kwargs,
    )


# ---------------------------------------------------------------- builder
def test_fixed_tc_builds_the_floor_when_d_hint_exceeds_it_and_max_tc_d_is_unchanged() -> None:
    compiled = build_compiled_code(BB30)
    fixed = _pair(compiled, rounds=12, d_hint=16, rounds_policy="fixed_Tc")
    assert fixed.x_memory.rounds == fixed.z_memory.rounds == 12
    native = _pair(compiled, rounds=12, d_hint=16)  # default = max_Tc_d, the formal Fig.2 rule
    assert native.x_memory.rounds == native.z_memory.rounds == 16
    explicit = _pair(compiled, rounds=12, d_hint=16, rounds_policy="max_Tc_d")
    for basis in ("x_memory", "z_memory"):
        assert str(getattr(explicit, basis).circuit) == str(getattr(native, basis).circuit)
        assert str(getattr(fixed, basis).circuit) != str(getattr(native, basis).circuit)
    # d_hint <= T_c: every policy gives the same T = 12 circuit and DEM, and the fixed_Tc build at
    # d_hint 16 is that very circuit (the SEC synthesis cache is rounds-independent by design)
    low = _pair(compiled, rounds=12, d_hint=5)
    low_fixed = _pair(compiled, rounds=12, d_hint=5, rounds_policy="fixed_Tc")
    for other in (low_fixed, fixed):
        for basis in ("x_memory", "z_memory"):
            assert str(getattr(other, basis).circuit) == str(getattr(low, basis).circuit)
            assert str(getattr(other, basis).detector_error_model) == str(getattr(low, basis).detector_error_model)
    with pytest.raises(ValueError, match="unknown rounds policy"):
        _pair(compiled, rounds=12, d_hint=5, rounds_policy="T=d")


# ---------------------------------------------------------------- build worker (job-pool path)
def test_build_worker_follows_the_verifications_rounds_policy(tmp_path: Path) -> None:
    from omniqec_agentic.evaluation.worker import run_build_dem_request

    def build(verification: VerificationConfig, name: str) -> dict:
        return run_build_dem_request({
            "mode": "build_dem", "spec": BB30, "noise_index": 0, "d_hint": 16,
            "matrices_path": str(tmp_path / f"{name}.pkl"), "synth_cache_dir": str(tmp_path / "synth"),
            "verification": verification.to_dict(),
        })

    # decoders=() skips the telescoping materialisation: only the circuit build is under test here
    cnot = build(VerificationConfig(task="cnot_dense", rounds_policy="fixed_Tc", decoders=()), "cnot")
    assert cnot["status"] == "ok", cnot.get("failure_reason")
    assert cnot["rounds_x"] == cnot["rounds_z"] == 12
    assert cnot["task"]["cnot_rounds"] == list(range(1, 12))  # 11 CNOT layers, not 15
    memory = build(VerificationConfig(task="memory", rounds_policy="max_Tc_d", decoders=()), "memory")
    assert memory["status"] == "ok" and memory["rounds_x"] == memory["rounds_z"] == 16  # native Fig.2 rule unchanged
    # the pool's invariant passes both (each matches its own policy's T)
    assert pool_mod.check_build_rounds(cnot, 12) is cnot
    assert pool_mod.check_build_rounds(memory, 16) is memory


def test_pool_turns_a_circuit_row_rounds_disagreement_into_a_build_error() -> None:
    stale = {"status": "ok", "rounds_x": 16, "rounds_z": 16}
    out = pool_mod.check_build_rounds(stale, 12)
    assert out["status"] == "build_error" and out["rounds_mismatch"] is True
    assert "T_X=16" in out["failure_reason"] and "T=12" in out["failure_reason"]
    failed = {"status": "build_error", "failure_reason": "x"}
    assert pool_mod.check_build_rounds(failed, 12) is failed
    assert pool_mod.check_build_rounds(None, 12) is None


# ---------------------------------------------------------------- fixed_code_measure end to end
def _protocol(root: Path, name: str, task: str) -> None:
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    verification = VerificationConfig(task=task, rounds_policy=("fixed_Tc" if task == "cnot_dense" else "max_Tc_d"),
                                      max_check_weight=(6 if task == "cnot_dense" else None))
    snap = build_protocol_snapshot(
        verification=verification, racing=config_mod.formal_racing(25_000), max_turns=30, proposal_count=24,
        generations_per_turn=4, slow_workers=0, strict_turns=True, required_provider="openai",
        required_model="claude-opus-5", rng_seeds_per_program=40, n_budget=60,
    )
    snap["protocol_profile"] = fcm.TASK_PROTOCOL_PROFILE[task]
    (root / name).mkdir(parents=True)
    (root / name / "protocol_snapshot.json").write_text(json.dumps(snap))
    (root / name / "campaign_spec.json").write_text(json.dumps({"weight_limit": None}))


def _item(key: str, task: str, protocol_run: str, policy: str, expected: int, shots: int = 48) -> dict:
    return {"key": key, "block": "test", "task": task, "protocol_run": protocol_run, "spec": BB30, "p": 0.005,
            "shots_per_basis": shots, "rounds_floor": 12, "rounds_policy": policy, "expected_rounds": expected,
            "max_check_weight": None, "seed_namespace": f"fixed_code_test/{key}", "program_id": f"fixed_{key}",
            "logicals": (4 if task == "cnot_dense" else 2), "n": 30, "k": 2}


def test_fixed_code_measure_end_to_end_circuit_rounds_equal_row_rounds(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("stim")
    pytest.importorskip("telescoping_decoder")
    real_proxy = pool_mod.evaluate_proxy

    def proxy_d16(spec, **kwargs):
        # the collaborator's case: a proxy distance above T_c (BB30 is d=5; the parent's d_hint goes to the
        # build worker in the request, so forcing it here reproduces A_N288_cnot_3786d18d's d_hint 16)
        out = dict(real_proxy(spec, **kwargs))
        out["d"] = 16
        return out

    monkeypatch.setattr(pool_mod, "evaluate_proxy", proxy_d16)
    protocols = tmp_path / "protocols"
    _protocol(protocols, "run_cnot", "cnot_dense")
    _protocol(protocols, "run_mem", "memory")
    items = [_item("cnot16", "cnot_dense", "run_cnot", "fixed_Tc", 12), _item("mem16", "memory", "run_mem", "max_Tc_d", 16)]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema": fcm.PLAN_SCHEMA, "items": items}))
    out_root = tmp_path / "out"
    rc = fcm.main(["--plan", str(plan_path), "--protocol-root", str(protocols), "--out-root", str(out_root),
                   "--key", "cnot16", "--key", "mem16", "--workers", "2", "--shot-shards", "2"])
    assert rc == 0
    for key, expected in (("cnot16", 12), ("mem16", 16)):
        res = json.loads((out_root / key / "result.json").read_text())
        assert res["row"]["status"] == "ok" and int(res["row"]["rounds"]) == expected
        assert res["rounds_check"]["errors"] == [] and res["rounds_check"]["circuit_rounds"] == {"X": expected, "Z": expected}
        cand = json.loads(next((out_root / key / "T12" / "candidates").glob("*.json")).read_text())
        for basis in ("X", "Z"):
            assert cand["results"][0]["basis"][basis]["memory_experiment"]["rounds"] == expected
        assert fcm.rounds_verdict(out_root, key)["verdict"] == "valid" and fcm.is_done(out_root, key)
    out_csv = tmp_path / "merged.csv"
    assert fcm.main(["--plan", str(plan_path), "--out-root", str(out_root), "--merge", "--out", str(out_csv)]) == 0
    with out_csv.open(encoding="utf-8") as fh:
        table = {r["key"]: r for r in csv.DictReader(fh)}
    assert table["cnot16"]["status"] == "ok" and table["cnot16"]["circuit_rounds_x"] == "12" == table["cnot16"]["rounds"]
    assert table["cnot16"]["logicals_measured"] == "4" and table["cnot16"]["circuit_data_qubits"] == "60"
    assert table["mem16"]["status"] == "ok" and table["mem16"]["circuit_rounds_z"] == "16" == table["mem16"]["rounds"]


# ---------------------------------------------------------------- guards against pre-fix results
def _fake_result(out_root: Path, item: dict, *, circuit_rounds: int, row_rounds: int = 12) -> None:
    key = item["key"]
    cand_dir = out_root / key / "T12" / "candidates"
    cand_dir.mkdir(parents=True)
    meta = {"noisy_circuit_sha256": "c", "dem_sha256": "d", "num_logicals": 4, "num_data_qubits": 60, "rounds": circuit_rounds}
    (cand_dir / f"fixed_{key}.json").write_text(json.dumps({"results": [{"basis": {"X": {"memory_experiment": meta},
                                                                                     "Z": {"memory_experiment": meta}}}]}))
    row = {"status": "ok", "program_id": f"fixed_{key}", "rounds": row_rounds, "shots_x": 100, "shots_z": 100,
           "failures_x": 1, "failures_z": 1, "per_logical_per_round": 1e-4, "per_logical_per_round_ci": [1e-5, 1e-3]}
    (out_root / key / "result.json").write_text(json.dumps({"plan_item": item, "row": row, "config": {"rounds_floor": 12}}))


def test_a_pre_fix_result_is_not_done_merges_as_rounds_mismatch_and_is_never_measured_over(tmp_path: Path, monkeypatch) -> None:
    bad = _item("bad", "cnot_dense", "run_cnot", "fixed_Tc", 12)
    good = _item("good", "cnot_dense", "run_cnot", "fixed_Tc", 12)
    native = _item("native", "memory", "run_mem", "max_Tc_d", 16)
    later = _item("later", "cnot_dense", "run_cnot", "fixed_Tc", 12)
    plan = {"schema": fcm.PLAN_SCHEMA, "items": [bad, good, native, later]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out_root = tmp_path / "old_out"
    _fake_result(out_root, bad, circuit_rounds=16)  # A_N288_cnot_3786d18d: circuits T16, row T12
    _fake_result(out_root, good, circuit_rounds=12)
    _fake_result(out_root, native, circuit_rounds=16, row_rounds=16)  # Fig4 b: native T = d, consistent

    verdicts = {i["key"]: fcm.rounds_verdict(out_root, i["key"], i)["verdict"] for i in plan["items"]}
    assert verdicts == {"bad": "rounds_mismatch", "good": "valid", "native": "valid", "later": "pending"}
    assert fcm.select_keys(plan, out_root, pending=True, chunk=None) == ["bad", "later"]
    errors = fcm.rounds_verdict(out_root, "bad", bad)["errors"]
    assert "candidate.X.memory_experiment.rounds=16 != row.rounds=12" in errors
    rows = {i["key"]: fcm.merge_row(i, out_root) for i in plan["items"]}
    assert rows["bad"]["status"] == "rounds_mismatch" and rows["bad"]["circuit_rounds_x"] == 16
    assert rows["good"]["status"] == "ok" and rows["native"]["status"] == "ok"

    # --force or not: the old directory is never measured over (resume would reuse the T16 candidate)
    called: list[str] = []
    monkeypatch.setattr(fcm, "measure_one", lambda plan, item, args: called.append(item["key"]) or 0)
    base = ["--plan", str(plan_path), "--protocol-root", str(tmp_path), "--out-root", str(out_root)]
    assert fcm.main(base + ["--key", "bad"]) == 5
    assert fcm.main(base + ["--key", "bad", "--force"]) == 5
    assert called == []
    assert fcm.main(base + ["--key", "good", "--key", "later"]) == 0 and called == ["later"]  # good is done
    # a NEW out-root measures it
    assert fcm.main(["--plan", str(plan_path), "--protocol-root", str(tmp_path), "--out-root", str(tmp_path / "new_out"),
                     "--key", "bad"]) == 0
    assert called == ["later", "bad"]


def test_audit_rounds_prints_one_verdict_line_per_plan_item(tmp_path: Path, capsys) -> None:
    bad = _item("bad", "cnot_dense", "run_cnot", "fixed_Tc", 12)
    later = _item("later", "cnot_dense", "run_cnot", "fixed_Tc", 12)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema": fcm.PLAN_SCHEMA, "items": [bad, later]}))
    _fake_result(tmp_path / "out", bad, circuit_rounds=16)
    assert fcm.main(["--plan", str(plan_path), "--out-root", str(tmp_path / "out"), "--audit-rounds"]) == 0
    lines = [ln.split("\t") for ln in capsys.readouterr().out.splitlines() if not ln.startswith("#")]
    assert [(ln[0], ln[1], ln[3], ln[4], ln[5], ln[6]) for ln in lines] == [
        ("rounds_mismatch", "bad", "12", "12", "16", "16"), ("pending", "later", "12", "", "", ""),
    ]
