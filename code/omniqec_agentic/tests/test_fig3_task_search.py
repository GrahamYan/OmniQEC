"""Fig.3 Phase B (2026-10-12): the task-aware search variant.

- the cnot_dense task builds the two-block gate-dense circuit on the memory
  pair's schedule and rounds, with 2k observables, and the memory task is
  byte-identical to before;
- the A4/A5 combination roots over the measured circuit's logical qubits;
- the structural queue's third slot is the front's lightest cell;
- the not-competitive tables and prompt block;
- the CLI switch produces the variant profile and the snapshot carries it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omniqec_agentic import ranking
from omniqec_agentic.config import (
    FIG3_CNOT_PROFILE_ID,
    StructuralSearchConfig,
    VerificationConfig,
    fig3_cnot_structural_config,
)
from omniqec_agentic.evaluation import verification_from_dict
from omniqec_agentic.evaluation.task_circuits import cnot_rounds_for, task_experiment_pair, task_metadata
from omniqec_agentic.verification.ler import combine_basis_ler_results


def _bb72_48() -> tuple[np.ndarray, np.ndarray]:
    """A small BB code ([[48,2,6]]-class) for the circuit tests."""

    from omniqec_agentic.construction.build import build_compiled_code

    spec = {"family": "bb", "ell": 3, "m": 8, "A": [[2, 7], [2, 4]], "B": [[2, 0], [1, 4]]}
    compiled = build_compiled_code(spec)
    return np.asarray(compiled.hx, dtype=np.uint8), np.asarray(compiled.hz, dtype=np.uint8)


def test_cnot_rounds_cover_every_round_but_the_last() -> None:
    assert cnot_rounds_for(12) == tuple(range(1, 12))
    assert cnot_rounds_for(13) == tuple(range(1, 13))


def test_task_pair_has_two_blocks_and_2k_observables_on_the_memory_schedule() -> None:
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair
    from omniqec_agentic.verification.noise import NoiseSweepConfig

    hx, hz = _bb72_48()
    noise = NoiseSweepConfig.uniform((0.005,)).point(0)
    pair = build_memory_experiment_pair(hx, hz, p=0.005, rounds=12, d_hint=6, noise_point=noise)
    task = task_experiment_pair(pair, hx, hz, noise_point=noise)
    for memory, exp in ((pair.x_memory, task.x_memory), (pair.z_memory, task.z_memory)):
        assert exp.rounds == memory.rounds == 12
        assert exp.num_logicals == 2 * memory.num_logicals
        assert exp.detector_error_model.num_observables == 2 * memory.num_logicals
        assert exp.num_data_qubits == 2 * memory.num_data_qubits
        assert exp.schedule is memory.schedule
        assert len(exp.detector_kinds) == exp.detector_error_model.num_detectors
        # 11 transversal layers of n CNOTs each between the two blocks
        text = str(exp.circuit)
        assert text.count("CX") > str(memory.circuit).count("CX")
    meta = task_metadata("cnot_dense", 12)
    assert meta["cnot_rounds"] == list(range(1, 12)) and meta["blocks"] == 2
    assert task_metadata("memory", 12) == {"task": "memory", "blocks": 1, "rounds": 12}


def test_a4_a5_roots_over_the_measured_circuits_logical_qubits() -> None:
    basis = {"X": {"logical_error_rate": 0.02}, "Z": {"logical_error_rate": 0.03}}
    block = 1 - (1 - 0.02) * (1 - 0.03)
    two_k = combine_basis_ler_results(basis, rounds=12, k=4)
    one_k = combine_basis_ler_results(basis, rounds=12, k=2)
    assert abs(two_k["per_logical_per_round"] - (1 - (1 - block) ** (1 / 48))) < 1e-15
    assert two_k["per_logical_per_round"] < one_k["per_logical_per_round"]


def test_pool_prefers_the_build_workers_num_logicals() -> None:
    from omniqec_agentic.evaluation.pool import _num_logicals

    assert _num_logicals({"num_logicals": {"X": 4, "Z": 4}}) == 4
    assert _num_logicals({"num_logicals": 6}) == 6
    assert _num_logicals({}) is None
    assert _num_logicals({"num_logicals": {"X": 4, "Z": 2}}) is None


def _row(code_hash: str, k: int, d: int, w: int, **extra):
    row = {
        "code_hash": code_hash, "program_id": code_hash, "k": k, "d_search": d,
        "max_check_weight": w, "n": 50, "n_total": 100, "Q_search": k * d * d, "family": "bb",
        "spec": {"family": "bb", "ell": 5, "m": 5}, "construct_source": "def construct(rng):\n    return {}\n",
        "rng_seed": 0, "source_agent": "funsearch",
    }
    row.update(extra)
    return row


def test_lightest_endpoint_is_a_third_slot_only_when_asked() -> None:
    rows = [_row("heavy_k", 8, 6, 8), _row("heavy_d", 2, 10, 8), _row("light", 2, 6, 4), _row("light_hi_k", 4, 6, 4)]
    ends = ranking.structural_endpoints(rows)
    assert ends["high_k"]["code_hash"] == "heavy_k" and ends["high_d"]["code_hash"] == "heavy_d"
    # ties in weight break toward the higher k
    assert ends["low_w"]["code_hash"] == "light_hi_k"
    v37 = ranking.select_structural_slots(rows, slots=2)
    assert [r["code_hash"] for r in v37] == ["heavy_k", "heavy_d"]
    cnot = ranking.select_structural_slots(rows, slots=3, lightest=True)
    assert [r["code_hash"] for r in cnot] == ["heavy_k", "heavy_d", "light_hi_k"]
    assert cnot[2]["structural_slot_reason"] == "lightest_endpoint"
    # without the flag the third slot stays a crowding representative
    plain = ranking.select_structural_slots(rows, slots=3)
    assert plain[2]["structural_slot_reason"] != "lightest_endpoint"


def test_archive_emits_the_lightest_endpoint_under_the_variant() -> None:
    from omniqec_agentic.memory.archive import ArchiveMemory

    rows = [_row("heavy_k", 8, 6, 8), _row("heavy_d", 2, 10, 8), _row("light", 2, 6, 4)]
    memory = ArchiveMemory(lightest_endpoint_slot=True)
    memory.seed_structural_elites([dict(r) for r in rows])
    queue = memory.structural_slow_queue(slots=3, measured_hashes=set())
    assert [r["code_hash"] for r in queue[:3]] == ["heavy_k", "heavy_d", "light"]
    assert queue[2]["structural_slot_reason"] == "lightest_endpoint"
    plain = ArchiveMemory()
    plain.seed_structural_elites([dict(r) for r in rows])
    assert plain.structural_slow_queue(slots=3, measured_hashes=set())[2]["structural_slot_reason"] != "lightest_endpoint"


def test_not_competitive_tables_name_losers_and_classes(tmp_path: Path) -> None:
    from omniqec_agentic.workflow.graph import _evidence_tables

    rows = [
        {"code_hash": "best", "program_id": "p_best", "per_logical_per_round": 1e-4, "ci_low": 0.9e-4, "ci_high": 1.1e-4,
         "k": 2, "d_search": 8, "family": "lp_abelian", "n": 72, "max_check_weight": 4},
        {"code_hash": "close", "program_id": "p_close", "per_logical_per_round": 1.5e-4, "ci_low": 1.3e-4, "ci_high": 1.7e-4,
         "k": 4, "d_search": 8, "family": "bb", "n": 72, "max_check_weight": 5},
        {"code_hash": "loser", "program_id": "p_loser", "per_logical_per_round": 4e-4, "ci_low": 3.5e-4, "ci_high": 4.5e-4,
         "k": 4, "d_search": 10, "family": "lp_abelian", "n": 72, "max_check_weight": 6},
        {"code_hash": "overlap", "program_id": "p_overlap", "per_logical_per_round": 2.2e-4, "ci_low": 1.0e-4, "ci_high": 3.4e-4,
         "k": 2, "d_search": 6, "family": "bb", "n": 72, "max_check_weight": 4},
        {"code_hash": "raced", "program_id": "p_raced", "per_logical_per_round": 3e-4, "ci_low": 0.5e-4, "ci_high": 6e-4,
         "k": 12, "d_search": 6, "family": "bb", "n": 72, "max_check_weight": 6, "stopped_by": "racing"},
    ]
    tables = _evidence_tables(tmp_path, rows, not_competitive_ratio=2.0, not_competitive_max_rows=12)
    losers = [r["program_id"] for r in tables["not_competitive"]]
    assert losers == ["p_loser", "p_raced"]  # worst first; the overlapping interval is not named
    assert tables["not_competitive"][0]["ratio_to_best"] == 4.0
    classes = {(r["max_check_weight"], r["d_search"]) for r in tables["losing_classes"]}
    assert classes == {(6, 10), (6, 6)}
    assert tables["not_competitive_ratio"] == 2.0
    formal = _evidence_tables(tmp_path, rows)
    assert "not_competitive" not in formal


def test_prompt_renders_the_not_competitive_block_and_task_header() -> None:
    from omniqec_agentic.funsearch import _funsearch_system_prompt, _not_competitive_lines, _slow_feedback_prompt_lines

    context = {
        "task": "cnot_dense",
        "evidence_tables": {
            "not_competitive_ratio": 2.0,
            "not_competitive": [
                {"program_id": "p_loser", "family": "lp_abelian", "n": 72, "k": 4, "d_search": 10, "max_check_weight": 6,
                 "per_logical_per_round": 4e-4, "ratio_to_best": 4.0, "stopped_by": None},
                {"program_id": "p_raced", "family": "bb", "n": 72, "k": 12, "d_search": 6, "max_check_weight": 6,
                 "per_logical_per_round": 3e-4, "ratio_to_best": 3.0, "stopped_by": "racing"},
            ],
            "losing_classes": [{"max_check_weight": 6, "d_search": 10, "measured": 1, "best_plpr": 4e-4, "ratio_to_best": 4.0}],
        },
    }
    text = "\n".join(_not_competitive_lines(context))
    assert "NOT COMPETITIVE UNDER THIS TASK" in text and "p_loser" in text and "w=6" in text
    assert "stopped early by racing" in text and "CLASSES NOT COMPETITIVE" in text
    assert _not_competitive_lines({"evidence_tables": {"by_check_weight": []}}) == []
    header = _slow_feedback_prompt_lines({"best_results": []}, context)[0]
    assert "CNOT-dense" in header
    memory_header = _slow_feedback_prompt_lines({"best_results": []}, {"task": "memory"})[0]
    assert "CNOT" not in memory_header
    system = _funsearch_system_prompt(allowed_families=["bb"], task="cnot_dense")
    assert "transversal logical CNOT between the two blocks after every round" in system
    assert "weight" not in system.split("OBJECTIVE")[0].lower()  # the task sentence names no preference
    assert "transversal" not in _funsearch_system_prompt(allowed_families=["bb"])


def test_variant_profile_and_snapshot_round_trip() -> None:
    structural = fig3_cnot_structural_config()
    assert structural.profile_id == FIG3_CNOT_PROFILE_ID
    assert (structural.structural_slow_slots, structural.runtime_slow_slots, structural.slow_candidate_cap) == (3, 4, 7)
    assert structural.lightest_endpoint_slot and structural.not_competitive_ratio == 2.0
    # the v3.7 values are untouched
    base = StructuralSearchConfig()
    assert structural.parent_cap == base.parent_cap and structural.final_runtime == base.final_runtime
    assert structural.proxy_iters_per_seed == base.proxy_iters_per_seed
    verification = VerificationConfig(task="cnot_dense", structural=structural)
    blob = json.loads(json.dumps(verification.to_dict()))
    back = verification_from_dict(blob)
    assert back.task == "cnot_dense" and back.structural.to_dict() == structural.to_dict()
    assert verification_from_dict(VerificationConfig().to_dict()).task == "memory"
    assert VerificationConfig().to_dict()["task"] == "memory"


def test_variant_snapshot_builds_with_the_wider_slow_queue() -> None:
    """The launch failure of 2026-10-13 01:17: promote_top (the archive's
    promotion head) must equal the slow queue's cap, so the variant raises
    it to 7 alongside 3S + 4R."""

    from omniqec_agentic.config import RacingConfig
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    structural = fig3_cnot_structural_config()
    verification = VerificationConfig(task="cnot_dense", structural=structural, promote_top=structural.slow_candidate_cap)
    snapshot = build_protocol_snapshot(
        verification=verification, racing=RacingConfig(), max_turns=30, proposal_count=24,
        generations_per_turn=4, slow_workers=4, n_budget=180,
    )
    assert snapshot["verification"]["task"] == "cnot_dense"
    assert snapshot["verification"]["structural"]["profile_id"] == FIG3_CNOT_PROFILE_ID
    assert snapshot["promote_top"] == 7
    import pytest

    with pytest.raises(ValueError, match="promote_top must equal"):
        build_protocol_snapshot(
            verification=VerificationConfig(task="cnot_dense", structural=structural), racing=RacingConfig(),
            max_turns=30, proposal_count=24, generations_per_turn=4, slow_workers=4, n_budget=180,
        )


def test_construction_time_check_weight_cap_rejects_heavy_codes_and_is_in_the_contract() -> None:
    """2026-10-14 (remaining Phase B points): MAX_CHECK_WEIGHT_HARD rejects a heavy code at the
    structural check and tells the Generator; without the cap nothing changes."""

    from omniqec_agentic.config import HardwareConfig, HardwareTarget, RunConfig, VerificationConfig
    from omniqec_agentic.construction.build import build_compiled_code
    from omniqec_agentic.funsearch import _construction_contract_lines
    from omniqec_agentic.verification.structural import structural_check_compiled

    # the b144 reference code BB [[72,12,6]]: check weight 6
    compiled = build_compiled_code(
        {"A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]], "ell": 6, "family": "bb", "m": 6}
    )
    capped = structural_check_compiled("ref", compiled, max_check_weight=5)
    assert any(str(f).startswith("max_check_weight_exceeded:6>5") for f in capped.failure_reasons), capped.failure_reasons
    at_cap = structural_check_compiled("ref", compiled, max_check_weight=6)
    assert not any("max_check_weight_exceeded" in str(f) for f in at_cap.failure_reasons)
    open_ = structural_check_compiled("ref", compiled, max_check_weight=None)
    assert not any("max_check_weight_exceeded" in str(f) for f in open_.failure_reasons)

    def _config(cap):
        return RunConfig(run_id="cap_probe", n_budget=144, hardware=HardwareConfig(target=HardwareTarget.ION),
                         out_dir="unused", verification=VerificationConfig(max_check_weight=cap))

    capped_lines = _construction_contract_lines(_config(6))
    assert any("CHECK WEIGHT CAP" in line and "<= 6" in line for line in capped_lines)
    assert not any("CHECK WEIGHT CAP" in line for line in _construction_contract_lines(_config(None)))


# ---- Fig.3 v2 (2026-10-15): measured / twin feedback -------------------------


def test_v2_profile_carries_the_measured_twin_feedback_flag() -> None:
    structural = fig3_cnot_structural_config()
    assert structural.measured_twin_feedback is True
    assert StructuralSearchConfig().measured_twin_feedback is False
    assert FIG3_CNOT_PROFILE_ID.endswith("_cnot_v2")
    assert structural.to_dict()["measured_twin_feedback"] is True


def test_program_feedback_names_measured_codes_and_their_twins() -> None:
    from omniqec_agentic.funsearch import _measured_lookup, _program_summary_line

    context = {
        "measured_identities": [
            {"code_hash": "aa" * 12, "canonical_hash": "CANON_A", "program_id": "g0003_p0002",
             "per_logical_per_round": 1.4e-4},
        ]
    }
    lookup = _measured_lookup(context)
    base = {"outcome": "accepted", "family": "bb", "n": 72, "k": 2, "d_search": 7, "max_check_weight": 4,
            "seeds": 1, "distinct_codes": 1}
    same = _program_summary_line(dict(base, code_hash="aa" * 12, canonical_hash="CANON_A"), measured=lookup)
    assert "ALREADY MEASURED" in same and "g0003_p0002" in same and "not measured again" in same
    twin = _program_summary_line(dict(base, code_hash="bb" * 12, canonical_hash="CANON_A"), measured=lookup)
    assert "ISOMORPHIC TWIN" in twin and "g0003_p0002" in twin and "not measured again" in twin
    new = _program_summary_line(dict(base, code_hash="cc" * 12, canonical_hash="CANON_C"), measured=lookup)
    assert new.endswith("recorded in its niche")
    # the Fig.2 profile carries no identities: the v3.7 line is unchanged
    plain = _program_summary_line(dict(base, code_hash="aa" * 12, canonical_hash="CANON_A"))
    assert plain.endswith("recorded in its niche")
    assert _measured_lookup({}) is None


def test_last_turn_queue_block_states_the_facts_and_the_rule() -> None:
    from omniqec_agentic.funsearch import _last_turn_queue_lines, _merge_prompt_context_overlay

    queue = {
        "turn": 7, "cap": 7, "selected": 0, "candidates_seen": 9,
        "twin_rejected": [
            {"family": "bb", "n": 72, "k": 2, "d_search": 7, "max_check_weight": 4, "program_id": "g0002_p0005",
             "twin": {"program_id": "g0003_p0002", "per_logical_per_round": 1.4e-4}},
        ],
        "twin_rejected_count": 6,
        "measured_rejected": [
            {"family": "lp_nonabelian", "n": 42, "k": 4, "d_search": 5, "max_check_weight": 5,
             "program_id": "g0001_p0007", "measured": {"program_id": "g0000_p0011", "per_logical_per_round": 3.0e-4}},
        ],
        "measured_rejected_count": 1,
        "weight_gate_max": 6, "weight_gate_rejected_count": 2, "weight_gate_weights": [8, 8],
    }
    text = "\n".join(_last_turn_queue_lines({"last_turn_queue": queue}))
    assert "0 of 7 slots filled -> nothing was measured" in text
    assert "6 were isomorphic twins" in text
    assert "1 were already-measured codes" in text
    assert "2 exceeded the check-weight gate" in text
    assert "twin of measured g0003_p0002" in text and "already measured as g0000_p0011" in text
    assert "A code is measured once" in text
    assert _last_turn_queue_lines({}) == []
    merged = _merge_prompt_context_overlay(
        {}, {"measured_identities": [{"code_hash": "aa"}], "last_turn_queue": queue}
    )
    assert merged["measured_identities"] == [{"code_hash": "aa"}]
    assert merged["last_turn_queue"]["turn"] == 7


def test_previous_turn_queue_summary_resolves_twins_to_measured_codes(tmp_path) -> None:
    import json as _json

    from omniqec_agentic.orchestrator.reflector import reflection_input
    from omniqec_agentic.workflow.graph import _last_turn_queue_summary, _measured_identities, _slow_queue_digest

    run_dir = tmp_path
    (run_dir / "structural_ledger.jsonl").write_text(
        _json.dumps({"code_hash": "aa" * 12, "canonical_hash": "CANON_A"}) + "\n"
        + _json.dumps({"code_hash": "bb" * 12, "canonical_hash": "CANON_A"}) + "\n",
        encoding="utf-8",
    )
    evidence = [{"code_hash": "aa" * 12, "program_id": "g0003_p0002", "per_logical_per_round": 1.4e-4,
                 "k": 2, "d_search": 7, "family": "bb", "n": 72, "max_check_weight": 4}]
    identities = _measured_identities(run_dir, evidence)
    assert identities[0]["canonical_hash"] == "CANON_A" and identities[0]["program_id"] == "g0003_p0002"
    comp_dir = run_dir / "turns" / "turn_0006" / "cycle"
    comp_dir.mkdir(parents=True)
    composition = {
        "cap": 7, "selected": [], "candidates_seen": 3,
        "twin_rejected_rows": [
            {"code_hash": ("bb" * 12)[:12], "program_id": "g0002_p0005", "family": "bb", "n": 72, "k": 2,
             "d_search": 7, "max_check_weight": 4, "twin_of_code_hash": ("aa" * 12)[:12], "twin_of": "measured"},
        ],
        "measured_rejected_rows": [
            {"code_hash": ("aa" * 12)[:12], "program_id": "g0002_p0009", "family": "bb", "n": 72, "k": 2,
             "d_search": 7, "max_check_weight": 4},
        ],
        "weight_gate": {"max_check_weight": 6, "rejected": [{"code_hash": "cc", "max_check_weight": 8}]},
        "structural_slot_shortfall": 3,
    }
    (comp_dir / "slow_queue_composition.json").write_text(_json.dumps(composition), encoding="utf-8")
    state = {"run_dir": str(run_dir)}
    summary = _last_turn_queue_summary(state, 7, identities)
    assert summary["turn"] == 6 and summary["selected"] == 0 and summary["candidates_seen"] == 3
    assert summary["twin_rejected"][0]["twin"]["program_id"] == "g0003_p0002"
    assert summary["measured_rejected"][0]["measured"]["program_id"] == "g0003_p0002"
    assert summary["weight_gate_rejected_count"] == 1 and summary["weight_gate_weights"] == [8]
    assert _last_turn_queue_summary(state, 0, identities) is None
    digest = _slow_queue_digest(str(comp_dir / "slow_queue_composition.json"))
    assert digest == {
        "cap": 7, "selected": 0, "candidates_seen": 3, "already_measured": 1,
        "isomorphic_twins_of_measured": 1, "over_check_weight_gate": 1, "check_weight_gate": 6,
        "structural_slot_shortfall": 3,
    }
    assert _slow_queue_digest(None) is None
    text = reflection_input(turn=7, digest={"slow_queue": digest}, plan={}, coverage=None)
    assert "isomorphic_twins_of_measured" in text


def test_execute_plan_overlay_carries_identities_and_last_queue_under_v2(monkeypatch, tmp_path) -> None:
    """Wiring: with measured_twin_feedback on, execute_plan_node hands the cycle an
    overlay with the measured identities (canonical hashes from the ledger) and the
    previous turn's queue summary; under the Fig.2 profile neither key exists."""

    import json as _json
    from types import SimpleNamespace

    import omniqec_agentic.campaign as campaign
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic.config import RacingConfig

    run_dir = tmp_path
    (run_dir / "evidence_rows.jsonl").write_text(
        _json.dumps({"code_hash": "aa" * 12, "program_id": "g0003_p0002", "per_logical_per_round": 1.4e-4,
                     "k": 2, "d_search": 7, "family": "bb", "n": 72, "max_check_weight": 4}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "structural_ledger.jsonl").write_text(
        _json.dumps({"code_hash": "aa" * 12, "canonical_hash": "CANON_A", "max_check_weight": 4}) + "\n",
        encoding="utf-8",
    )
    comp_dir = run_dir / "turns" / "turn_0000" / "cycle"
    comp_dir.mkdir(parents=True)
    (comp_dir / "slow_queue_composition.json").write_text(
        _json.dumps({"cap": 7, "selected": [], "candidates_seen": 2,
                     "twin_rejected_rows": [{"code_hash": "bbbbbbbbbbbb", "program_id": "g0001_p0004", "family": "bb",
                                             "n": 72, "k": 2, "d_search": 7, "max_check_weight": 4,
                                             "twin_of_code_hash": ("aa" * 12)[:12], "twin_of": "measured"}],
                     "measured_rejected_rows": [], "weight_gate": {"max_check_weight": 6, "rejected": []}}),
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_cycle(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status="ok", artifact_paths={})

    monkeypatch.setattr(campaign, "run_agentic_cycle", fake_cycle)
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    state = {
        "run_id": "campaign_b144", "run_dir": str(run_dir), "turn": 1,
        "plan": {"provider": "dry_run", "generations": 1, "proposal_count": 1, "rng_seeds_per_program": 1},
        "protocol": {"proposal_count": 1, "generations_per_turn": 1, "feedback_mode": "both",
                     "racing": RacingConfig().to_dict(), "slow_workers": 127, "shot_shards": 16, "strict_turns": False},
        "spec": {"qubit_budget": 144, "families": ["bb"]},
        "structural_proxy_elites": [], "unmeasurable_hashes": [], "baseline_summary": {}, "budget": {},
    }
    v2 = VerificationConfig(task="cnot_dense", structural=fig3_cnot_structural_config(), promote_top=7)
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *args, **kwargs: v2)
    graph.execute_plan_node(state)
    overlay = captured["prompt_context_overlay"]
    assert overlay["measured_identities"] == [
        {"code_hash": "aa" * 12, "canonical_hash": "CANON_A", "program_id": "g0003_p0002",
         "per_logical_per_round": 1.4e-4, "family": "bb", "n": 72, "k": 2, "d_search": 7, "max_check_weight": 4}
    ]
    queue = overlay["last_turn_queue"]
    assert queue["turn"] == 0 and queue["selected"] == 0 and queue["twin_rejected_count"] == 1
    assert queue["twin_rejected"][0]["twin"] == {"program_id": "g0003_p0002", "per_logical_per_round": 1.4e-4}
    assert captured["promotion_measured"] == {"aa" * 12}

    captured.clear()
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *args, **kwargs: VerificationConfig())
    graph.execute_plan_node(state)
    overlay = captured["prompt_context_overlay"]
    assert "measured_identities" not in overlay and "last_turn_queue" not in overlay


def test_fig2_validator_accepts_formal_snapshots_that_predate_the_fig3_switches() -> None:
    """The 27 formal Fig.2 snapshots were frozen before lightest_endpoint_slot,
    not_competitive_* and measured_twin_feedback existed; the validator reads an
    absent switch as its formal value and still rejects a present non-formal one."""

    import pytest

    from omniqec_agentic.hpc.validate_fig2_campaign import (
        FIG3_VARIANT_STRUCTURAL_KEYS,
        STRUCTURAL_EXPECTED,
        _check_structural_retention,
    )

    formal = {k: v for k, v in STRUCTURAL_EXPECTED.items() if k not in FIG3_VARIANT_STRUCTURAL_KEYS}
    _check_structural_retention(formal)  # archived v3.7 snapshot: no Fig.3 keys at all
    _check_structural_retention(dict(STRUCTURAL_EXPECTED))  # a current formal snapshot
    with pytest.raises(ValueError, match="measured_twin_feedback=True"):
        _check_structural_retention(dict(STRUCTURAL_EXPECTED, measured_twin_feedback=True))
    with pytest.raises(ValueError, match="lightest_endpoint_slot=True"):
        _check_structural_retention(dict(formal, lightest_endpoint_slot=True))
    with pytest.raises(ValueError, match="slow_candidate_cap=7"):
        _check_structural_retention(dict(formal, slow_candidate_cap=7))
    missing_quota = dict(formal)
    del missing_quota["slow_candidate_cap"]
    with pytest.raises(ValueError, match="slow_candidate_cap=None"):
        _check_structural_retention(missing_quota)


# ---- Fig.3 formal run (2026-10-15): fixed-depth rounds policy ------------------


def test_fixed_tc_rounds_policy_measures_every_code_at_the_floor() -> None:
    import pytest

    from omniqec_agentic.config import ROUNDS_POLICY_IDS, RacingConfig
    from omniqec_agentic.verification.circuit import select_memory_rounds
    from omniqec_agentic.workflow.graph import _protocol_final_rounds
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    assert ROUNDS_POLICY_IDS == ("max_Tc_d", "fixed_Tc")
    assert select_memory_rounds(rounds=12, d_hint=16) == 16  # formal Fig.2 rule
    assert select_memory_rounds(rounds=12, d_hint=16, policy="fixed_Tc") == 12
    assert select_memory_rounds(rounds=12, d_hint=5, policy="fixed_Tc") == 12
    assert select_memory_rounds(rounds=12, d_hint=None, policy="fixed_Tc") == 12
    with pytest.raises(ValueError, match="unknown rounds policy"):
        select_memory_rounds(rounds=12, d_hint=5, policy="T=d")
    with pytest.raises(ValueError, match="rounds_policy must be one of"):
        VerificationConfig(rounds_policy="T=d")
    # the switch is a verification field, serialized and restored, and it names both stages
    v2 = VerificationConfig(
        task="cnot_dense", structural=fig3_cnot_structural_config(), promote_top=7, rounds_policy="fixed_Tc"
    )
    back = verification_from_dict(json.loads(json.dumps(v2.to_dict())))
    assert back.rounds_policy == "fixed_Tc"
    assert verification_from_dict(VerificationConfig().to_dict()).rounds_policy == "max_Tc_d"
    assert verification_from_dict({"task": "memory"}).rounds_policy == "max_Tc_d"  # a Fig.2 artifact
    snapshot = build_protocol_snapshot(
        verification=v2, racing=RacingConfig(), max_turns=30, proposal_count=24,
        generations_per_turn=4, slow_workers=4, n_budget=180, search_rounds=12,
    )
    assert snapshot["rounds_policy"]["search"] == "fixed_Tc" and snapshot["rounds_policy"]["final"] == "fixed_Tc"
    assert snapshot["rounds_policy"]["final_rounds"] == 12
    assert _protocol_final_rounds(snapshot) == 12
    formal = build_protocol_snapshot(
        verification=VerificationConfig(), racing=RacingConfig(), max_turns=30, proposal_count=24,
        generations_per_turn=4, slow_workers=4, n_budget=180, search_rounds=12,
    )
    assert formal["rounds_policy"]["search"] == "max_Tc_d" and formal["rounds_policy"]["final"] == "max_Tc_d"
    with pytest.raises(ValueError, match="must be one of"):
        _protocol_final_rounds({"rounds_policy": {"final": "T=d", "final_rounds": 12}})


def test_execute_plan_keeps_the_snapshots_construction_cap(monkeypatch, tmp_path) -> None:
    """2026-10-16 defect: _verification_from_protocol replaced the snapshot's
    max_check_weight with the spec's weight_limit (None), so the formal run's
    construction cap (--max-check-weight 6) and its contract line vanished from
    every turn. The cap must reach the cycle's RunConfig; a tighter spec limit wins."""

    from types import SimpleNamespace

    import omniqec_agentic.campaign as campaign
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic.config import RacingConfig
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    captured: dict = {}

    def fake_cycle(config, *args, **kwargs):
        captured["config"] = config
        return SimpleNamespace(status="ok", artifact_paths={})

    monkeypatch.setattr(campaign, "run_agentic_cycle", fake_cycle)
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    capped = VerificationConfig(task="cnot_dense", structural=fig3_cnot_structural_config(), promote_top=7, max_check_weight=6)
    protocol = build_protocol_snapshot(
        verification=capped, racing=RacingConfig(), max_turns=30, proposal_count=1, generations_per_turn=1,
        slow_workers=4, n_budget=144, search_rounds=12,
    )
    assert protocol["verification"]["max_check_weight"] == 6
    protocol["strict_turns"] = False
    state = {
        "run_id": "campaign_b144", "run_dir": str(tmp_path), "turn": 0,
        "plan": {"provider": "dry_run", "generations": 1, "proposal_count": 1, "rng_seeds_per_program": 1, "promote_top": 7},
        "protocol": protocol, "spec": {"qubit_budget": 144, "families": ["bb"]},
        "structural_proxy_elites": [], "unmeasurable_hashes": [], "baseline_summary": {}, "budget": {},
    }
    graph.execute_plan_node(state)
    assert captured["config"].verification.max_check_weight == 6
    assert captured["config"].verification.rounds_policy == "max_Tc_d"
    state["spec"] = {"qubit_budget": 144, "families": ["bb"], "weight_limit": 4}
    graph.execute_plan_node(state)
    assert captured["config"].verification.max_check_weight == 4  # the tighter hardware limit wins
    # 2026-11-10 paper alignment: the default snapshot carries the search cap 8 ...
    default = build_protocol_snapshot(
        verification=VerificationConfig(), racing=RacingConfig(), max_turns=30, proposal_count=1,
        generations_per_turn=1, slow_workers=4, n_budget=144, search_rounds=12,
    )
    default["strict_turns"] = False
    state["protocol"] = default; state["spec"] = {"qubit_budget": 144, "families": ["bb"]}
    state["plan"]["promote_top"] = 6
    graph.execute_plan_node(state)
    assert captured["config"].verification.max_check_weight == 8
    # ... and a snapshot without a cap (the v3.7 as-run protocol, --max-check-weight 0) stays uncapped.
    uncapped = build_protocol_snapshot(
        verification=VerificationConfig(max_check_weight=None), racing=RacingConfig(), max_turns=30,
        proposal_count=1, generations_per_turn=1, slow_workers=4, n_budget=144, search_rounds=12,
    )
    uncapped["strict_turns"] = False
    state["protocol"] = uncapped
    graph.execute_plan_node(state)
    assert captured["config"].verification.max_check_weight is None


def test_last_turn_queue_block_keeps_the_batch_requirement() -> None:
    from omniqec_agentic.funsearch import _last_turn_queue_lines

    text = "\n".join(_last_turn_queue_lines({"last_turn_queue": {"turn": 8, "cap": 7, "selected": 7, "candidates_seen": 46}}))
    assert "return the full batch of programs every generation" in text
    assert "propose a code that is not isomorphic" not in text
