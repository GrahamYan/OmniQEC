"""Fig.4 c (2026-10-23): the exact maximum check-weight contract.

The only scientific intervention of the Fig.4 c campaigns on the frozen Fig.3
formal_v2 flow: max(all H_X row weights, all H_Z row weights) == w. It is
enforced at the structural check (so the archive, the proxy frontier and the
parents never hold another weight class), at the slow queue and at the final
shortlist, and it replaces the cap line of the Generator contract. Off (the
default) the code paths, the contract text and the artifact shapes are those
of formal_v2, byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic.campaign import _build_slow_queue
from omniqec_agentic.config import (
    FIG3_CNOT_PROFILE_ID,
    HardwareConfig,
    HardwareTarget,
    RunConfig,
    StructuralSearchConfig,
    VerificationConfig,
    fig3_cnot_structural_config,
    formal_racing,
)
from omniqec_agentic.construction.build import build_compiled_code
from omniqec_agentic.evaluation import verification_from_dict
from omniqec_agentic.funsearch import _construction_contract_lines
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.schemas import CandidateProgram
from omniqec_agentic.verification import fast as fast_mod
from omniqec_agentic.verification.structural import structural_check_compiled, structural_check_spec
from omniqec_agentic.workflow import checkpoint as checkpoint_mod
from omniqec_agentic.workflow.state import build_protocol_snapshot

# the b144 reference code BB [[72,12,6]]: check weight 6
BB72 = {"A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]], "ell": 6, "family": "bb", "m": 6}
# a weight-4 BB code (|A| + |B| = 4; connected Tanner graph, [[72,2]])
BB_W4 = {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [0, 1]], "B": [[0, 0], [1, 0]]}
SOURCE = "def construct(rng):\n    return %r\n"

FORMAL_V2_CAP_LINE = (
    "CHECK WEIGHT CAP (run protocol): every X and Z check must have weight <= 6 "
    "(max row weight of H_X and H_Z). A heavier code is rejected before any evaluation "
    "(max_check_weight_exceeded), so proposing one spends a program on nothing."
)


# ------------------------------------------------------------ structural check
def test_exact_contract_rejects_lighter_and_heavier_codes_alike() -> None:
    compiled = build_compiled_code(BB72)
    for target in (4, 5, 7, 8):
        res = structural_check_compiled("ref", compiled, max_check_weight=None, exact_check_weight=target)
        assert not res.valid
        assert f"check_weight_mismatch:6!={target}" in res.failure_reasons, res.failure_reasons
    exact = structural_check_compiled("ref", compiled, max_check_weight=None, exact_check_weight=6)
    assert exact.valid and not any("check_weight" in str(f) for f in exact.failure_reasons)
    assert exact.max_check_weight == 6
    # off: the formal_v2 behaviour (cap only)
    capped = structural_check_compiled("ref", compiled, max_check_weight=6)
    assert capped.valid
    assert "exact" not in json.dumps(capped.failure_reasons)


def test_exact_contract_reaches_the_spec_wrapper() -> None:
    res = structural_check_spec("w4", BB_W4, total_budget=144, max_check_weight=4, exact_check_weight=4)
    assert res.valid and res.max_check_weight == 4, res.failure_reasons
    res6 = structural_check_spec("w4", BB_W4, total_budget=144, max_check_weight=6, exact_check_weight=6)
    assert not res6.valid and "check_weight_mismatch:4!=6" in res6.failure_reasons


# ------------------------------------------------------------ archive / parents
def _stub_proxy(monkeypatch, distance: int) -> None:
    def _search(hx, hz, *, seeds, iters, algorithm_id=None):
        return {
            "schema": "proxy_distance_search", "algorithm_id": "stub", "seeds": list(seeds),
            "iters_per_seed": int(iters), "sectors": ["X", "Z"], "per_seed": [],
            "d_x": int(distance), "d_z": int(distance), "d_search": int(distance), "available": True,
        }

    monkeypatch.setattr(fast_mod, "proxy_distance_search", _search)


def test_archive_never_admits_another_weight_class(monkeypatch) -> None:
    """The construction gate is the archive gate: a code of the wrong weight is a
    structural reject, so no archive cell, proxy elite, parent or queue can hold it."""

    _stub_proxy(monkeypatch, 6)
    memory = ArchiveMemory(turn=0, max_check_weight=4, exact_check_weight=4)
    memory.evaluate_and_record(CandidateProgram(program_id="w6", source=SOURCE % BB72, n_budget=144), n_budget=144)
    memory.evaluate_and_record(CandidateProgram(program_id="w4", source=SOURCE % BB_W4, n_budget=144), n_budget=144)
    by_id = {a["program_id"]: a for a in memory.attempts}
    assert "check_weight_mismatch:6!=4" in json.dumps(by_id["w6"])
    assert by_id["w6"].get("structural_eligible") is not True
    assert by_id["w4"].get("structural_eligible") is True, by_id["w4"]
    hashes = {row["code_hash"] for row in memory.structural_proxy_elites}
    assert hashes == {by_id["w4"]["code_hash"]}
    # off (formal_v2 cap 6): BB72 is admitted as before
    plain = ArchiveMemory(turn=0, max_check_weight=6)
    plain.evaluate_and_record(CandidateProgram(program_id="w6", source=SOURCE % BB72, n_budget=144), n_budget=144)
    assert plain.attempts[-1].get("structural_eligible") is True


# ------------------------------------------------------------ prompt contract
def _config(**verification):
    return RunConfig(
        run_id="exactw_probe", n_budget=144, hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused", verification=VerificationConfig(**verification),
    )


def test_contract_line_is_the_only_prompt_clause_that_changes() -> None:
    cap = _construction_contract_lines(_config(max_check_weight=6))
    exact = _construction_contract_lines(_config(max_check_weight=4, exact_check_weight=4))
    assert FORMAL_V2_CAP_LINE in cap, "the formal_v2 cap line must render byte for byte when the contract is off"
    assert not any("CHECK WEIGHT CAP" in line for line in exact)
    contract = [line for line in exact if line.startswith("CHECK WEIGHT CONTRACT")]
    assert len(contract) == 1 and "EXACTLY 4" in contract[0] and "check_weight_mismatch" in contract[0]
    # every other line is identical, in the same order
    assert [l for l in cap if not l.startswith("CHECK WEIGHT")] == [l for l in exact if not l.startswith("CHECK WEIGHT")]
    assert not any("CHECK WEIGHT" in line for line in _construction_contract_lines(_config(max_check_weight=None)))
    # 2026-11-10 paper alignment: the default config carries the search cap 8 and its line
    assert any(line.startswith("CHECK WEIGHT CAP (run protocol): every X and Z check must have weight <= 8")
               for line in _construction_contract_lines(_config()))


# ------------------------------------------------------------ config / snapshot
def test_verification_config_validates_and_round_trips_the_contract() -> None:
    with pytest.raises(ValueError, match=">= exact_check_weight"):
        VerificationConfig(max_check_weight=4, exact_check_weight=6)
    # 2026-11-10: an exact contract under a wider cap (e.g. the default 8) is consistent
    assert VerificationConfig(max_check_weight=6, exact_check_weight=4).exact_check_weight == 4
    with pytest.raises(ValueError, match=">= 3"):
        VerificationConfig(exact_check_weight=2)
    plain = VerificationConfig(max_check_weight=6).to_dict()
    assert "exact_check_weight" not in plain, "off: earlier snapshots keep their keys"
    on = VerificationConfig(max_check_weight=5, exact_check_weight=5).to_dict()
    assert on["exact_check_weight"] == 5 and on["max_check_weight"] == 5
    back = verification_from_dict(on)
    assert back.exact_check_weight == 5 and back.max_check_weight == 5
    assert verification_from_dict(plain).exact_check_weight is None


def _snapshot(verification: VerificationConfig, **kw) -> dict:
    structural = fig3_cnot_structural_config(verification.structural)
    return build_protocol_snapshot(
        verification=VerificationConfig(**{**_as_kwargs(verification), "structural": structural, "task": "cnot_dense",
                                           "rounds_policy": "fixed_Tc", "promote_top": 7}),
        racing=formal_racing(25_000), max_turns=30, proposal_count=24, generations_per_turn=4,
        slow_workers=8, n_budget=144, strict_turns=True, fixed_program_count=True,
        required_provider="openai", required_model="claude-opus-5(medium)", **kw,
    )


def _as_kwargs(verification: VerificationConfig) -> dict:
    return {"max_check_weight": verification.max_check_weight, "exact_check_weight": verification.exact_check_weight}


def test_snapshot_carries_the_contract_and_the_campaign_id_only_when_set() -> None:
    plain = _snapshot(VerificationConfig(max_check_weight=6), promotion_max_check_weight=6)
    assert "exact_check_weight" not in plain and "campaign_id" not in plain
    assert plain["structural_retention"]["profile_id"] == FIG3_CNOT_PROFILE_ID
    on = _snapshot(VerificationConfig(max_check_weight=7, exact_check_weight=7), promotion_max_check_weight=7,
                   campaign_id="f4v26_N144_cnot_w7_claude_rep1")
    assert on["exact_check_weight"] == 7 and on["verification"]["exact_check_weight"] == 7
    assert on["promotion_max_check_weight"] == 7 and on["verification"]["max_check_weight"] == 7
    assert on["campaign_id"] == "f4v26_N144_cnot_w7_claude_rep1"
    # the contract is a protocol key, not a runtime key: a resume under another value is a mismatch
    assert "exact_check_weight" not in checkpoint_mod.RESUME_RUNTIME_KEYS
    with pytest.raises(ValueError, match="must equal exact_check_weight"):
        _snapshot(VerificationConfig(max_check_weight=7, exact_check_weight=7), promotion_max_check_weight=6)
    # everything but the two new keys and the weight values is the formal_v2 snapshot
    ignore = {"exact_check_weight", "campaign_id", "promotion_max_check_weight", "verification"}
    assert {k: v for k, v in plain.items() if k not in ignore} == {k: v for k, v in on.items() if k not in ignore}


# ------------------------------------------------------------ slow queue gate
V37 = StructuralSearchConfig()


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _structural(code: str, k: int, d: int, reason: str, w: int) -> dict:
    return {"program_id": f"struct_{code}", "code_hash": code * 16, "k": k, "d_search": d, "max_check_weight": w,
            "structural_slot_reason": reason, "spec": {"family": "lp_abelian"},
            "construct_source": "def construct(rng): return {}"}


def _runtime(program_id: str, code: str, w: int) -> dict:
    return {"program_id": program_id, "code_hash": code * 16, "k": 4, "d_search": 10, "max_check_weight": w}


def _rows(path: str | None) -> list[dict]:
    if not path:
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_slow_queue_gate_holds_the_exact_weight_and_backfills(tmp_path) -> None:
    structural = [_structural("aa", 8, 8, "high_k_endpoint", 4), _structural("bb", 2, 12, "high_d_search_endpoint", 6),
                  _structural("cc", 4, 9, "lightest", 5)]
    fresh = [_runtime("g1", "11", 5), _runtime("g2", "22", 4), _runtime("g3", "33", 6), _runtime("g4", "44", 5),
             _runtime("g5", "55", 5), _runtime("g6", "66", 5), _runtime("g7", "77", 5)]
    out, comp = _build_slow_queue(
        structural_queue=_write(tmp_path / "s.jsonl", structural), promotion_queue=_write(tmp_path / "q.jsonl", fresh),
        all_archive_queue=_write(tmp_path / "a.jsonl", fresh), priority=[], structural_config=V37, out_dir=tmp_path,
        measured_hashes=None, max_check_weight_gate=5, exact_check_weight_gate=5,
    )
    taken = [r["program_id"] for r in _rows(out)]
    assert "struct_aa" not in taken and "struct_bb" not in taken, "a lighter AND a heavier code are both kept out"
    assert "g2" not in taken and "g3" not in taken
    assert taken == ["struct_cc", "g1", "g4", "g5", "g6", "g7"], taken
    gate = comp["weight_gate"]
    assert gate["max_check_weight"] == 5 and gate["exact_check_weight"] == 5
    rejected = {r["code_hash"][:2]: r["max_check_weight"] for r in gate["rejected"]}
    assert rejected == {"aa": 4, "bb": 6, "22": 4, "33": 6}
    # off: the composition keeps its formal_v2 shape (no exact key)
    out, comp = _build_slow_queue(
        structural_queue=_write(tmp_path / "s2.jsonl", structural), promotion_queue=_write(tmp_path / "q2.jsonl", fresh),
        all_archive_queue=_write(tmp_path / "a2.jsonl", fresh), priority=[], structural_config=V37, out_dir=tmp_path,
        measured_hashes=None, max_check_weight_gate=6,
    )
    assert comp["weight_gate"] == {"max_check_weight": 6, "rejected": []}


# ------------------------------------------------------------ final shortlist gate
def test_final_shortlist_holds_the_exact_weight(tmp_path) -> None:
    import omniqec_agentic.workflow.graph as graph

    (tmp_path / "structural_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"code_hash": "light4", "max_check_weight": 4}, {"code_hash": "exact6a", "max_check_weight": 6},
            {"code_hash": "exact6b", "max_check_weight": 6}, {"code_hash": "heavy8", "max_check_weight": 8},
        ]) + "\n", encoding="utf-8")

    def elite(code_hash: str, k: int, d: int) -> dict:
        return {"code_hash": code_hash, "program_id": "p_" + code_hash, "spec": {"family": "bb"},
                "k": k, "d_search": d, "d_search_proxy": d, "pareto_rank": 1}

    elites = [elite("light4", 21, 5), elite("heavy8", 4, 12), elite("exact6a", 2, 8), elite("exact6b", 4, 8)]
    structural = StructuralSearchConfig(final_structural=2, final_runtime=0)
    state = {"run_dir": str(tmp_path), "protocol": {"promotion_max_check_weight": 6, "exact_check_weight": 6},
             "structural_proxy_elites": elites, "hof_summary": [], "unmeasurable_hashes": []}
    entries, comp = graph._final_entries(state, structural)
    assert {e["code_hash"] for e in entries} == {"exact6a", "exact6b"}
    assert comp["weight_gate"] == 6 and comp["exact_check_weight"] == 6
    assert sorted(comp["elites_over_weight_gate"]) == ["heavy8", "light4"]
    # off: the formal_v2 gate (cap only) and no exact key in the composition
    capped = dict(state, protocol={"promotion_max_check_weight": 6})
    entries, comp = graph._final_entries(capped, structural)
    assert {e["code_hash"] for e in entries} == {"light4", "exact6a"} or {e["code_hash"] for e in entries} <= {"light4", "exact6a", "exact6b"}
    assert "exact_check_weight" not in comp and comp["elites_over_weight_gate"] == ["heavy8"]


# ------------------------------------------------------------ CLI
def test_cli_exact_weight_sets_both_gates_and_refuses_contradictions(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli as cli_mod

    captured: dict = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "campaign_b144", "stop": {"reason": "test"}, "report": {}}

    monkeypatch.setattr("omniqec_agentic.workflow.run_campaign", fake_run_campaign)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    base = ["run-campaign", "--budget", "144", "--turns", "1", "--provider", "dry_run", "--out", str(tmp_path),
            "--task", "cnot_dense", "--rounds-policy", "fixed_Tc"]
    assert cli_mod.main(base + ["--exact-check-weight", "7", "--campaign-id", "f4v26_N144_cnot_w7_claude_rep2"]) == 0
    assert captured["promotion_max_check_weight"] == 7
    assert captured["verification"].exact_check_weight == 7 and captured["verification"].max_check_weight == 7
    assert captured["campaign_id"] == "f4v26_N144_cnot_w7_claude_rep2"
    assert captured["verification"].structural.profile_id == FIG3_CNOT_PROFILE_ID
    with pytest.raises(SystemExit, match="contradicts"):
        cli_mod.main(base + ["--exact-check-weight", "7", "--max-check-weight", "6"])
    with pytest.raises(SystemExit, match="contradicts"):
        cli_mod.main(base + ["--exact-check-weight", "7", "--promotion-max-check-weight", "6"])
    with pytest.raises(SystemExit, match=">= 3"):
        cli_mod.main(base + ["--exact-check-weight", "2"])
    # the formal_v2 launch (cap 6) is untouched
    captured.clear()
    assert cli_mod.main(base + ["--max-check-weight", "6", "--promotion-max-check-weight", "6"]) == 0
    assert captured["verification"].exact_check_weight is None and captured["verification"].max_check_weight == 6
    assert captured["campaign_id"] is None
