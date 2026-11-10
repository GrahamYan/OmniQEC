"""2026-09-29: the slow-queue stabilizer-weight gate (runtime key
promotion_max_check_weight) and the large-code early-stop variant
pilot10x_maxerr30_v3.

Evidence behind the gate (763 tier-2/3 slow measurements, b180-b360): every
w >= 9 candidate measured plpr >= 2.9e-4 (median 5.4e-3) at 10-100x the decode
cost of the good codes; 0 of 413 codes with plpr < 2e-4 had w >= 9."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic import config as config_mod
from omniqec_agentic.campaign import _build_slow_queue
from omniqec_agentic.config import (
    RACING_MAX_ERRORS,
    RACING_MAX_ERRORS_LARGE,
    RACING_POLICY_V3_ID,
    RACING_POLICY_V3_MAXERR30_ID,
    FORMAL_RACING_POLICIES,
    StructuralSearchConfig,
    VerificationConfig,
    formal_racing,
    formal_racing_for_max_errors,
    formal_racing_policy_id,
    formal_racing_v3_maxerr30,
)
from omniqec_agentic.workflow import checkpoint as checkpoint_mod
from omniqec_agentic.workflow.state import build_protocol_snapshot

V37 = StructuralSearchConfig()


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _structural(code: str, k: int, d: int, reason: str, w: int | None) -> dict:
    row = {
        "program_id": f"struct_{code}", "code_hash": code * 16, "k": k, "d_search": d,
        "structural_slot_reason": reason, "spec": {"family": "lp_abelian"},
        "construct_source": "def construct(rng): return {}",
    }
    if w is not None:
        row["max_check_weight"] = w
    return row


def _runtime(program_id: str, code: str, w: int | None) -> dict:
    row = {"program_id": program_id, "code_hash": code * 16, "k": 4, "d_search": 10}
    if w is not None:
        row["max_check_weight"] = w
    return row


def _rows(path: str | None) -> list[dict]:
    if not path:
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _build(tmp_path: Path, structural: list[dict], fresh: list[dict], gate: int | None):
    return _build_slow_queue(
        structural_queue=_write(tmp_path / "s.jsonl", structural),
        promotion_queue=_write(tmp_path / "q.jsonl", fresh),
        all_archive_queue=_write(tmp_path / "a.jsonl", fresh),
        priority=[],
        structural_config=V37,
        out_dir=tmp_path,
        measured_hashes=None,
        max_check_weight_gate=gate,
    )


# ---------------------------------------------------------------- the gate
def test_gate_keeps_heavy_codes_out_and_backfills_their_slots(tmp_path) -> None:
    structural = [
        _structural("aa", 42, 8, "high_k_endpoint", 12),      # gated
        _structural("bb", 4, 17, "high_d_search_endpoint", 9),  # gated (w == 9 > 8)
        _structural("cc", 4, 12, "crowding_representative", 8),  # kept
        _structural("dd", 6, 10, "crowding_representative", None),  # no weight recorded -> kept
    ]
    fresh = [_runtime("g1", "11", 6), _runtime("g2", "22", 10), _runtime("g3", "33", 8),
             _runtime("g4", "44", 7), _runtime("g5", "55", 5), _runtime("g6", "66", 8)]
    out, composition = _build(tmp_path, structural, fresh, gate=8)
    ids = [r["program_id"] for r in _rows(out)]
    assert ids == ["struct_cc", "struct_dd", "g1", "g3", "g4", "g5"], ids
    assert composition["structural_taken"] == 2 and composition["cap"] == 6
    gate = composition["weight_gate"]
    assert gate["max_check_weight"] == 8
    rejected = {r["code_hash"]: r for r in gate["rejected"]}
    assert set(rejected) == {"aa" * 6, "bb" * 6, "22" * 6}
    assert rejected["aa" * 6]["max_check_weight"] == 12
    assert rejected["aa" * 6]["structural_slot_reason"] == "high_k_endpoint"
    assert rejected["aa" * 6]["channel"] == "structural"
    assert rejected["22" * 6]["channel"] == "runtime"
    # a gated code is never in the queue, under any channel
    assert not {"struct_aa", "struct_bb", "g2"} & set(ids)


def test_gate_off_is_the_v37_queue(tmp_path) -> None:
    structural = [_structural("aa", 42, 8, "high_k_endpoint", 12), _structural("bb", 4, 17, "high_d_search_endpoint", 9)]
    fresh = [_runtime(f"g{i}", str(i) * 2, 10) for i in range(1, 6)]
    out, composition = _build(tmp_path, structural, fresh, gate=None)
    assert [r["program_id"] for r in _rows(out)] == ["struct_aa", "struct_bb", "g1", "g2", "g3", "g4"]
    assert composition["weight_gate"] == {"max_check_weight": None, "rejected": []}


def test_gate_is_a_threshold_not_a_family_rule(tmp_path) -> None:
    # w == gate passes; only w > gate is rejected, whatever the family
    structural = [_structural("aa", 4, 12, "high_d_search_endpoint", 8)]
    fresh = [_runtime("g1", "11", 8), _runtime("g2", "22", 9)]
    out, composition = _build(tmp_path, structural, fresh, gate=8)
    assert [r["program_id"] for r in _rows(out)] == ["struct_aa", "g1"]
    assert [r["code_hash"] for r in composition["weight_gate"]["rejected"]] == ["22" * 6]


# ---------------------------------------------------------------- snapshot / resume
def _snapshot(**kw) -> dict:
    return build_protocol_snapshot(
        verification=VerificationConfig(),
        racing=kw.pop("racing", formal_racing(25_000)),
        # slow_workers > 1: with a serial run the snapshot disables racing and
        # the block no longer names a formal policy (the launch would refuse it)
        max_turns=30, proposal_count=24, generations_per_turn=4, slow_workers=8,
        n_budget=288, **kw,
    )


def test_snapshot_carries_the_gate_only_when_set_and_resume_accepts_it() -> None:
    plain = _snapshot()
    assert "promotion_max_check_weight" not in plain
    gated = _snapshot(promotion_max_check_weight=8)
    assert gated["promotion_max_check_weight"] == 8
    with pytest.raises(ValueError):
        _snapshot(promotion_max_check_weight=0)
    assert "promotion_max_check_weight" in checkpoint_mod.RESUME_RUNTIME_KEYS
    checkpoint_mod.validate_resume_protocol(plain, gated)
    assert checkpoint_mod.resume_runtime_overrides(plain, gated) == {"promotion_max_check_weight": 8}
    # and the gate never leaks into the verification block
    assert "promotion_max_check_weight" not in json.dumps(gated["verification"])


# ---------------------------------------------------------------- pilot10x_maxerr30_v3
def test_maxerr30_is_v3_with_a_30_error_cap_and_is_registered() -> None:
    v3 = formal_racing(25_000)
    v30 = formal_racing_v3_maxerr30(25_000)
    assert v3.policy_id == RACING_POLICY_V3_ID and v3.max_errors == RACING_MAX_ERRORS == 100
    assert v30.policy_id == RACING_POLICY_V3_MAXERR30_ID and v30.max_errors == RACING_MAX_ERRORS_LARGE == 30
    assert v30.max_errors_in_pilot and v30.enabled and v30.shots_confirm == v3.shots_confirm
    assert v30.to_dict() == {**v3.to_dict(), "max_errors": 30, "policy_id": RACING_POLICY_V3_MAXERR30_ID}
    assert FORMAL_RACING_POLICIES[RACING_POLICY_V3_MAXERR30_ID] is formal_racing_v3_maxerr30
    assert formal_racing_policy_id(v30.to_dict()) == RACING_POLICY_V3_MAXERR30_ID
    assert formal_racing_policy_id(v3.to_dict()) == RACING_POLICY_V3_ID
    # the frozen default is untouched
    assert config_mod.RACING_POLICY_ID == RACING_POLICY_V3_ID


@pytest.mark.parametrize("max_errors,expected", [(None, RACING_POLICY_V3_ID), (100, RACING_POLICY_V3_ID), (30, RACING_POLICY_V3_MAXERR30_ID)])
def test_racing_max_errors_selects_a_registered_policy(max_errors, expected) -> None:
    assert formal_racing_for_max_errors(25_000, max_errors).policy_id == expected


def test_racing_max_errors_refuses_unregistered_values() -> None:
    with pytest.raises(ValueError):
        formal_racing_for_max_errors(25_000, 50)


def test_a_resumed_chain_may_switch_to_maxerr30_at_a_turn_boundary() -> None:
    stored = _snapshot(racing=formal_racing(25_000))
    current = _snapshot(racing=formal_racing_v3_maxerr30(25_000))
    checkpoint_mod.validate_resume_protocol(stored, current)
    overrides = checkpoint_mod.resume_runtime_overrides(stored, current)
    assert overrides["racing"]["policy_id"] == RACING_POLICY_V3_MAXERR30_ID
    # both new knobs together (the large-code launch)
    both = _snapshot(racing=formal_racing_v3_maxerr30(25_000), promotion_max_check_weight=8)
    checkpoint_mod.validate_resume_protocol(stored, both)
    assert set(checkpoint_mod.resume_runtime_overrides(stored, both)) == {"racing", "promotion_max_check_weight"}


# ---------------------------------------------------------------- CLI plumbing
def test_cli_flags_reach_run_campaign(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli as cli_mod

    captured: dict = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "campaign_b288", "stop": {"reason": "test"}, "report": {}}

    monkeypatch.setattr("omniqec_agentic.workflow.run_campaign", fake_run_campaign)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    argv = [
        "run-campaign", "--budget", "288", "--turns", "1", "--provider", "dry_run",
        "--out", str(tmp_path), "--promotion-max-check-weight", "8", "--racing-max-errors", "30",
    ]
    assert cli_mod.main(argv) == 0
    assert captured["promotion_max_check_weight"] == 8
    assert captured["racing"].policy_id == RACING_POLICY_V3_MAXERR30_ID
    assert captured["racing"].max_errors == 30


# ---------------------------------------------------------------- 2026-09-29 evening fixes
def test_validator_policy_order_accepts_the_v4_rollback_and_the_large_code_variant() -> None:
    from omniqec_agentic.config import (
        RACING_POLICY_V1_ID, RACING_POLICY_V2_ID, RACING_POLICY_V4_ID, RACING_POLICY_RANK, racing_policy_moves_forward,
    )

    assert set(RACING_POLICY_RANK) == set(FORMAL_RACING_POLICIES)
    assert racing_policy_moves_forward(RACING_POLICY_V1_ID, RACING_POLICY_V2_ID)
    assert racing_policy_moves_forward(RACING_POLICY_V2_ID, RACING_POLICY_V3_ID)
    assert racing_policy_moves_forward(RACING_POLICY_V3_ID, RACING_POLICY_V4_ID)
    # the 2026-09-15 withdrawal: every chain that ran a turn under v4 went back to v3
    assert racing_policy_moves_forward(RACING_POLICY_V4_ID, RACING_POLICY_V3_ID)
    assert racing_policy_moves_forward(RACING_POLICY_V3_ID, RACING_POLICY_V3_MAXERR30_ID)
    assert racing_policy_moves_forward(RACING_POLICY_V4_ID, RACING_POLICY_V3_MAXERR30_ID)
    assert racing_policy_moves_forward(RACING_POLICY_V3_MAXERR30_ID, RACING_POLICY_V3_MAXERR30_ID)
    assert not racing_policy_moves_forward(RACING_POLICY_V3_MAXERR30_ID, RACING_POLICY_V3_ID)
    assert not racing_policy_moves_forward(RACING_POLICY_V3_ID, RACING_POLICY_V2_ID)


def test_failure_reasons_are_folded_and_bounded() -> None:
    from omniqec_agentic.evaluation.sampling import FAILURE_REASON_MAX_CHARS, join_failure_reasons

    assert join_failure_reasons([]) == ""
    assert join_failure_reasons(["a"]) == "a"
    assert join_failure_reasons(["t", "t", "c", "t", "c"]) == "t x3; c x2"
    # a capped final: 128 shards x 64 chunks x 2 bases of two messages stay readable
    reasons = ["shard_dem worker exceeded timeout_s=300.0 on all 1 attempts"] * 129 + ["cancelled: ..."] * 16255
    text = join_failure_reasons(reasons)
    assert text == "shard_dem worker exceeded timeout_s=300.0 on all 1 attempts x129; cancelled: ... x16255"
    long = join_failure_reasons([f"reason {i}" for i in range(5000)])
    assert len(long) <= FAILURE_REASON_MAX_CHARS + 40 and long.endswith("chars)")


def test_csv_readers_accept_wide_fields() -> None:
    import csv

    import omniqec_agentic.summary_csv  # noqa: F401  (raises the limit at import)

    assert csv.field_size_limit() >= 2**31 - 1


def test_repair_final_rounds_uses_the_measured_rounds(tmp_path, monkeypatch) -> None:
    from omniqec_agentic.hpc import repair_final_rounds as repair

    run_dir = tmp_path / "run"
    (run_dir / "final_slow" / "T12").mkdir(parents=True)
    (run_dir / "final_slow" / "T12" / "rankings.json").write_text(json.dumps({"schema": "rankings", "rankings": [
        {"program_id": "a", "code_hash": "aa" * 12, "rounds": 12, "d_search": 9, "status": "ok"},
        {"program_id": "b", "code_hash": "bb" * 12, "rounds": 14, "d_search": 14, "status": "ok"},
    ]}), encoding="utf-8")
    report = {"schema": "campaign_report", "final_frontier": [
        {"program_id": "a", "code_hash": "aa" * 12, "rounds": 12, "per_logical_per_round": 1e-4},
        {"program_id": "b", "code_hash": "bb" * 12, "rounds": 12, "per_logical_per_round": 2e-4},
    ]}
    (run_dir / "campaign_report.json").write_text(json.dumps(report), encoding="utf-8")
    _, changes = repair.plan(run_dir)
    assert changes == [{"index": 1, "program_id": "b", "from": 12, "to": 14}]
    monkeypatch.setattr("omniqec_agentic.workflow.graph._report_markdown", lambda r: "# repaired\n")
    monkeypatch.setattr("omniqec_agentic.workflow.graph._write_channel_reports", lambda run_dir, r: None)
    assert repair.main([str(run_dir), "--apply"]) == 0
    fixed = json.loads((run_dir / "campaign_report.json").read_text(encoding="utf-8"))
    assert [row["rounds"] for row in fixed["final_frontier"]] == [12, 14]
    assert fixed["final_rounds_repair"][0]["rows"] == changes
    assert repair.plan(run_dir)[1] == []  # idempotent


def test_repair_final_rounds_keys_by_code_hash_when_program_ids_collide(tmp_path, monkeypatch) -> None:
    from omniqec_agentic.hpc import repair_final_rounds as repair

    run_dir = tmp_path / "run"
    (run_dir / "final_slow" / "T12").mkdir(parents=True)
    # GPT b360: two different codes both named g0001_p0013, measured at 16 and 17 rounds
    (run_dir / "final_slow" / "T12" / "rankings.json").write_text(json.dumps({"rankings": [
        {"program_id": "g0001_p0013", "code_hash": "aa" * 12, "rounds": 16, "status": "ok"},
        {"program_id": "g0001_p0013", "code_hash": "bb" * 12, "rounds": 17, "status": "ok"},
    ]}), encoding="utf-8")
    report = {"final_frontier": [
        {"program_id": "g0001_p0013", "code_hash": "aa" * 6, "rounds": 12},   # report rows carry a hash prefix
        {"program_id": "g0001_p0013", "code_hash": "bb" * 6, "rounds": 12},
        {"program_id": "g0001_p0013", "rounds": 12},                          # no hash + ambiguous id -> untouched
    ]}
    (run_dir / "campaign_report.json").write_text(json.dumps(report), encoding="utf-8")
    _, changes = repair.plan(run_dir)
    assert changes == [
        {"index": 0, "program_id": "g0001_p0013", "from": 12, "to": 16},
        {"index": 1, "program_id": "g0001_p0013", "from": 12, "to": 17},
    ]
