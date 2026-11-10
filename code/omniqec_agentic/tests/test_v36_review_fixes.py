"""v3.6 review fixes (2026-09-03), carried through the decoder pivot to
telescoping_structural_v3_7 (docs/06 §1, user decision 2026-09-03).

Six verified review findings, each pinned here at its root cause:

  1. the serial / in-process slow path hard-coded stage="search" (a final
     stage on workers=1 recorded decode_stage "search") -> `stage` is
     threaded through evaluate_slow_spec, the worker default mode and
     pool._evaluate_one (retry_at_final included);
  2. the decode watchdog (8 s/shot x column factor) was an order of
     magnitude below a shot that exhausts every stage -> the watchdog is
     also >= shots x the telescoping decoder's worst-case shot model (S3
     iterations x GARI columns x a measured per-column-iteration cost; no
     stage-dependent term under the CPU protocol);
  3. writeback looked the decoder-build blob up by decoder NAME, so every
     formal-decoder row carried decoder_config=None (strict final audit +
     validator failed) -> name -> build-key map;
  4. assert_resumable_protocol refused every non-formal decoder set, so a
     legitimate non-strict "+declsd" variant could not resume its own
     checkpoint -> retired profile prefix / missing v3.6 keys only;
  5. the M1 certification of a Mitten-class code costs ~340k ISD iterations
     (2.6 CPU-hours) inside a 1-hour build watchdog and restarted from zero
     after a kill -> deterministic per-code ISD budget with a
     certification_budget_exhausted flag, checkpointed probes, and a build
     watchdog derived from the certification work model;
  6. the coverage gap list enumerated only LP regions and its 12-item cut
     showed one branch twelve times -> bb / hgp region grids, per-family
     counts, round-robin cut.
"""

from __future__ import annotations

import copy
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest

from omniqec_agentic.config import (
    DEFAULT_SEC_POLICY,
    TELESCOPING_COST_S_PER_COLUMN_ITER,
    TELESCOPING_GARI_COLUMN_FACTOR,
    TELESCOPING_PROTOCOL_FIELDS,
    DecoderName,
    RacingConfig,
    SecPolicyConfig,
    TelescopingDecoderConfig,
    VerificationConfig,
)
from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation import pipeline as pipeline_mod
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation import worker as worker_mod
from omniqec_agentic.evaluation.baseline import BASELINE_BB_SPEC
from omniqec_agentic.evaluation.evidence import sec_ext_evidence
from omniqec_agentic.evaluation.writeback import DECODER_BUILD_KEYS, decoder_build_blob, ranking_rows
from omniqec_agentic.hpc import validate_fig2_campaign as validator
from omniqec_agentic.memory import coverage as coverage_mod
from omniqec_agentic.sec import twobasis_rank as tr
from omniqec_agentic.workflow import graph as graph_mod
from omniqec_agentic.workflow.checkpoint import validate_resume_protocol
from omniqec_agentic.workflow.state import assert_resumable_protocol, build_protocol_snapshot

from test_prompt_neutrality import BANNED_PHRASES, GOAL_RE

TELESCOPING = DecoderName.TELESCOPING.value
NOISE_POINT = {"profile": "uniform_paper_v1", "axis": "p", "value": 0.005}
# Cheap certification for the end-to-end probes (protocol values are not
# under test here; the stage tag and the row plumbing are).
FAST_SEC = SecPolicyConfig(
    sec_candidates=2, sec_cert_iters=5, sec_cert_seeds=1, sec_cert_iters_scale=0.0,
    sec_refine_top=0, sec_refine_iters=1,
)


# ---------------------------------------------------------------------------
# 3. writeback: decoder_config of a telescoping row
# ---------------------------------------------------------------------------


def _telescoping_payload(n_budget: int = 144) -> dict:
    # The formal decoder blob is the same at every budget point (no per-point
    # field); the payload records the point for the structural rows only.
    verification = VerificationConfig()
    cell = {
        "status": "ok", "decoder": TELESCOPING, "shots": 100, "failures": 1,
        "logical_error_rate": 0.01, "runtime_s": 1.0, "metrics": {},
        "memory_experiment": {"schedule": {"sec_policy_id": "twobasis_nonint_v1"}},
    }
    return {
        "schema": "slow_verification_run", "program_id": "telescoping_row", "status": "ok",
        "verification": verification.to_dict(),
        "promotion_record": {"code_hash": "a" * 32, "n_budget": n_budget},
        "structural": {"valid": True, "k": 12, "n_data": 72, "n_checks_x": 36, "n_checks_z": 36, "n_budget": n_budget},
        "fast": {"family": "bb", "n": 72, "k": 12, "d": 6},
        "results": [
            {
                "status": "ok", "decoder": TELESCOPING, "noise_point": dict(NOISE_POINT),
                "noise_axis": "p", "noise_value": 0.005, "rounds": 12,
                "sampling_source": "noisy_circuit",
                "basis": {"X": dict(cell), "Z": dict(cell)},
                "combined": {"block_ler": 0.02, "per_logical_per_round": 1e-4, "total_shots": 200, "total_failures": 2},
            }
        ],
    }


def test_decoder_build_keys_cover_every_decoder_name() -> None:
    build = VerificationConfig().decoder_build.to_dict()
    assert set(DECODER_BUILD_KEYS) == {name.value for name in DecoderName}
    assert set(DECODER_BUILD_KEYS.values()) == set(build)
    assert decoder_build_blob(build, TELESCOPING) == build["telescoping"] == TelescopingDecoderConfig().to_dict()
    assert "cascade" not in build and DECODER_BUILD_KEYS[TELESCOPING] == "telescoping"
    assert decoder_build_blob(build, "bposd") == build["bposd"]
    assert decoder_build_blob(build, "no_such_decoder") is None
    assert decoder_build_blob(None, TELESCOPING) is None and decoder_build_blob(build, None) is None


def test_telescoping_ranking_rows_carry_the_frozen_blob_and_pass_the_audits(tmp_path) -> None:
    payload = _telescoping_payload(144)
    rows = ranking_rows(payload, tmp_path / "telescoping_row.json")
    assert len(rows) == 1 and rows[0]["decoder"] == TELESCOPING and rows[0]["status"] == "ok"
    expected = payload["verification"]["decoder_build"]["telescoping"]
    assert rows[0]["decoder_config"] == expected == TELESCOPING_PROTOCOL_FIELDS
    # The two consumers that failed on decoder_config=None now accept the row.
    assert graph_mod._telescoping_config_ok(rows[0]["decoder_config"])
    assert validator._validate_rows_are_protocol_decoder(rows, "review", rounds=12) == 1
    # A large-budget row carries the SAME blob: no field is per-point, so
    # neither audit takes a budget point any more.
    big = ranking_rows(_telescoping_payload(360), tmp_path / "big.json")[0]
    assert big["decoder_config"] == rows[0]["decoder_config"]
    assert graph_mod._telescoping_config_ok(big["decoder_config"])
    # Drift, a missing field and an extra (cascade-era) field all fail.
    assert not graph_mod._telescoping_config_ok({**big["decoder_config"], "b_quorum": 2})
    assert not graph_mod._telescoping_config_ok(
        {k: v for k, v in big["decoder_config"].items() if k != "a_iters"}
    )
    assert not graph_mod._telescoping_config_ok({**big["decoder_config"], "s4_time_s": 60.0})
    assert not graph_mod._telescoping_config_ok(None)
    assert not hasattr(graph_mod, "_cascade_config_ok")


def test_real_smoke_artifact_rows_carry_the_blob_when_present(tmp_path) -> None:
    """The reviewer's evidence artifact (a real bb72 smoke run) -- skipped
    when it is no longer on disk."""

    root = Path(__file__).resolve().parents[1] / "tmp" / "v36_smoke"
    candidates = sorted(root.glob("*/slow/candidates/*.json")) if root.is_dir() else []
    if not candidates:
        pytest.skip("no v3.6 smoke artifact on disk")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    rows = [row for row in ranking_rows(payload, candidates[0]) if row.get("decoder") == TELESCOPING]
    if not rows:
        pytest.skip("smoke artifact has no telescoping rows")
    assert all(isinstance(row["decoder_config"], dict) for row in rows)
    assert all(row["decoder_config"]["preset"] == "telescoping_cpu_v1" for row in rows)


# ---------------------------------------------------------------------------
# 4. resume: non-strict decoder variants resume their own checkpoint
# ---------------------------------------------------------------------------


def _snapshot(**overrides) -> dict:
    kwargs = dict(
        verification=VerificationConfig(),
        racing=RacingConfig(),
        max_turns=1, tool_budget=1, proposal_count=1, generations_per_turn=1,
        slow_workers=1, feedback_mode="raw", strict_turns=False,
        required_provider="", required_model="", rng_seeds_per_program=1,
        n_budget=144,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)


@pytest.mark.parametrize("decoders, tag", [((DecoderName.LSD,), "+declsd"), ((DecoderName.BPOSD,), "+decbposd")])
def test_non_strict_decoder_variants_resume_their_own_checkpoint(decoders, tag) -> None:
    variant = _snapshot(verification=VerificationConfig(decoders=decoders))
    assert variant["protocol_profile"].endswith(tag)
    assert variant["decoder"] == decoders[0].value
    stored = json.loads(json.dumps(variant))
    assert_resumable_protocol(stored)  # was: "refusing to resume ... decoder 'lsd'"
    validate_resume_protocol(stored, variant)
    # ...and the generic diff still refuses it against the formal protocol.
    with pytest.raises(ValueError, match="protocol snapshot mismatch"):
        validate_resume_protocol(stored, _snapshot())


def test_retired_checkpoints_are_still_refused_by_name_or_missing_keys() -> None:
    current = _snapshot()
    v35 = json.loads(json.dumps(current))
    v35["protocol_profile"] = "bposd_structural_v3_5"
    for key in ("decoder", "telescoping", "n_budget", "budget_policy", "sec_policy"):
        v35.pop(key)
    v35["verification"]["decoders"] = ["bposd"]
    with pytest.raises(ValueError, match="bposd_structural_v3_5.*decoder 'bposd'.*not poolable"):
        assert_resumable_protocol(v35)
    # The never-launched self-built cascade (2026-09-03) is retired by name.
    cascade_run = json.loads(json.dumps(current))
    cascade_run["protocol_profile"] = "cascade_structural_v3_6"
    cascade_run["decoder"] = "relay_cascade_gari"
    with pytest.raises(ValueError, match="cascade_structural_v3_6.*decoder 'relay_cascade_gari'.*not poolable"):
        assert_resumable_protocol(cascade_run)
    no_telescoping = json.loads(json.dumps(current))
    no_telescoping.pop("telescoping")
    with pytest.raises(ValueError, match="no telescoping key"):
        assert_resumable_protocol(no_telescoping)
    no_policy = json.loads(json.dumps(current))
    no_policy.pop("budget_policy")
    with pytest.raises(ValueError, match="no budget_policy key"):
        assert_resumable_protocol(no_policy)
    assert_resumable_protocol(json.loads(json.dumps(current)))


# ---------------------------------------------------------------------------
# 1. stage threading through the serial / in-process path
# ---------------------------------------------------------------------------


def test_serial_promotion_queue_forwards_the_stage_and_retry_at_final(monkeypatch, tmp_path) -> None:
    seen: list[dict] = []

    def fake_evaluate_slow_spec(**kwargs):
        seen.append(kwargs)
        return {
            "schema": "slow_verification_run", "program_id": kwargs["program_id"],
            "status": "no_valid_results", "results": [], "promotion_record": kwargs["promotion_record"],
            "verification": kwargs["verification"].to_dict(),
        }

    monkeypatch.setattr(pool_mod, "evaluate_slow_spec", fake_evaluate_slow_spec)
    monkeypatch.setattr(pool_mod, "resolve_workers", lambda workers: 1)
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"program_id": "plain", "spec": BASELINE_BB_SPEC}) + "\n"
        + json.dumps({"program_id": "retry", "spec": BASELINE_BB_SPEC, "retry_at_final": True}) + "\n",
        encoding="utf-8",
    )
    pool_mod.evaluate_promotion_queue(
        queue_path=queue, out_dir=tmp_path / "final", verification=VerificationConfig(shots=10),
        include_baseline=False, workers=1, stage="final",
    )
    assert [kw["stage"] for kw in seen] == ["final", "final"]
    seen.clear()
    pool_mod.evaluate_promotion_queue(
        queue_path=queue, out_dir=tmp_path / "search", verification=VerificationConfig(shots=10),
        include_baseline=False, workers=1, stage="search",
    )
    assert {kw["program_id"]: kw["stage"] for kw in seen} == {"plain": "search", "retry": "final"}
    with pytest.raises(ValueError, match="stage must be"):
        pipeline_mod.evaluate_slow_spec(program_id="x", spec=BASELINE_BB_SPEC, stage="warmup")


def test_subprocess_request_and_worker_default_mode_carry_the_stage(monkeypatch) -> None:
    captured: dict = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, payload, timeout=None):
            captured["request"] = json.loads(payload)
            return json.dumps({"schema": "slow_verification_run", "status": "ok", "results": []}), ""

    monkeypatch.setattr(pipeline_mod.subprocess, "Popen", lambda *a, **k: FakeProcess())
    out = pipeline_mod.evaluate_slow_spec(
        program_id="x", spec=BASELINE_BB_SPEC, verification=VerificationConfig(circuit_timeout_s=5.0),
        stage="final",
    )
    assert out["isolation"] == "subprocess" and captured["request"]["stage"] == "final"

    forwarded: dict = {}

    def fake_inprocess(**kwargs):
        forwarded.update(kwargs)
        return {"status": "ok", "results": []}

    monkeypatch.setattr(pipeline_mod, "evaluate_slow_spec_inprocess", fake_inprocess)
    monkeypatch.setattr(worker_mod.sys, "stdin", io.StringIO(json.dumps({
        "program_id": "x", "spec": BASELINE_BB_SPEC, "stage": "final",
        "verification": VerificationConfig().to_dict(),
    })))
    assert worker_mod.main() == 0
    assert forwarded["stage"] == "final"
    monkeypatch.setattr(worker_mod.sys, "stdin", io.StringIO(json.dumps({
        "program_id": "x", "spec": BASELINE_BB_SPEC, "verification": VerificationConfig().to_dict(),
    })))
    worker_mod.main()
    assert forwarded["stage"] == "search"


def test_inprocess_final_stage_records_the_stage_tag_and_the_telescoping_metrics() -> None:
    """Review fix 1 under the telescoping decoder: the in-process path records
    the stage it was asked for (no stage-dependent limit exists any more; the
    tag is what the audits read) and the per-basis cell carries the
    telescoping stage bookkeeping. A real 4-shot decode of BB72 (T=3)."""

    verification = VerificationConfig(shots=4, max_errors=None, rounds=3, sec_policy=FAST_SEC)
    payload = pipeline_mod.evaluate_slow_spec(
        program_id="bb72_final", spec=BASELINE_BB_SPEC, verification=verification,
        seed=1, isolated=False, stage="final",
    )
    assert payload["status"] == "ok" and payload["stage"] == "final"
    for basis in ("X", "Z"):
        cell = payload["results"][0]["basis"][basis]
        assert cell["decode_stage"] == "final"
        metrics = cell["metrics"]
        assert metrics["decode_stage"] == "final"
        assert metrics["nc_policy"] == "fail"
        assert metrics["package"] == TELESCOPING_PROTOCOL_FIELDS["package"]
        assert set(metrics["stage_counts"]) <= {"S3A", "S3B", "S3C", "NC"}
        assert sum(metrics["stage_counts"].values()) == 4
        assert metrics["declared_failures"] == metrics["stage_counts"].get("NC", 0)
        assert not any(key.startswith("s4_") for key in metrics)
    search = pipeline_mod.evaluate_slow_spec_inprocess(
        program_id="bb72_search", spec=BASELINE_BB_SPEC, verification=verification, seed=1,
    )
    assert search["stage"] == "search"
    assert search["results"][0]["basis"]["X"]["metrics"]["decode_stage"] == "search"


# ---------------------------------------------------------------------------
# 2. decode watchdog: telescoping worst-case shot model
# ---------------------------------------------------------------------------


def test_telescoping_worst_case_shot_model_covers_the_measured_mitten_cost() -> None:
    formal = VerificationConfig()
    telescoping = formal.decoder_build.telescoping
    # Package tables at the defaults: 5 x 100 (S3-A) + 6 x 40 x 100 (S3-B)
    # + the 12 S3-C configurations (11 x 10 seeds + 1 noise-free) = 104,000.
    iterations = telescoping.worst_case_iterations()
    assert iterations == 104_000
    mitten_cols = 123_960  # Mitten200 two-basis DEM (review probe)
    worst_search = pool_mod.telescoping_worst_case_shot_s(formal, mitten_cols, "search")
    worst_final = pool_mod.telescoping_worst_case_shot_s(formal, mitten_cols, "final")
    assert worst_search == pytest.approx(
        TELESCOPING_COST_S_PER_COLUMN_ITER * mitten_cols * TELESCOPING_GARI_COLUMN_FACTOR * iterations
    )
    # No stage-dependent term under the CPU protocol (S4 is off).
    assert worst_final == pytest.approx(worst_search)
    # Measured (docs/05 §9): Mitten200 averages 11-18 CPU-s/shot; the worst
    # case (every stage exhausted) bounds that with margin.
    assert worst_search > 18.0
    # b360 search shard: 5000 / 16 = 312 shots.
    shots = 312
    timeout = pool_mod.effective_decode_shard_timeout_s(formal, shots, mitten_cols, "search")
    assert timeout >= shots * 18.0
    # ...whereas the flat v3.4 term alone would have killed the shard.
    flat = 8.0 * shots * pool_mod.decode_budget_scale(mitten_cols)
    assert flat == pytest.approx(5_157.0, rel=1e-3) and flat < shots * 18.0
    assert timeout == pytest.approx(max(flat, shots * worst_search))
    # Size unknown -> exactly the v3.4 value; other decoders -> flat term only;
    # a disabled per-shot budget -> the flat floor.
    assert pool_mod.effective_decode_shard_timeout_s(formal, 1250, None) == pytest.approx(10_000.0)
    assert pool_mod.telescoping_worst_case_shot_s(formal, None) is None
    bposd = VerificationConfig(decoders=(DecoderName.BPOSD,))
    assert pool_mod.telescoping_worst_case_shot_s(bposd, mitten_cols) is None
    assert pool_mod.effective_decode_shard_timeout_s(bposd, 1250, 120_000) == pytest.approx(20_000.0)
    assert pool_mod.effective_decode_shard_timeout_s(
        VerificationConfig(decode_shot_budget_s=0.0), 1250, 120_000
    ) == pytest.approx(1800.0)
    # The worker request's stage tag does not change the watchdog.
    req = {"mode": "shard_dem", "shots": shots, "num_matrix_columns": mitten_cols}
    assert pool_mod.worker_timeout_for({**req, "stage": "final"}, formal) == pytest.approx(shots * worst_search)
    assert pool_mod.worker_timeout_for({**req, "stage": "search"}, formal) == pytest.approx(shots * worst_search)
    model = pool_mod.decode_budget_model(formal, shots, mitten_cols, "final")
    assert model["stage"] == "final"
    assert model["telescoping_worst_case_shot_s"] == pytest.approx(worst_search)
    assert model["telescoping_term_s"] == pytest.approx(shots * worst_search)
    assert model["flat_term_s"] == pytest.approx(flat)
    assert model["cost_s_per_column_iter"] == TELESCOPING_COST_S_PER_COLUMN_ITER
    assert model["gari_column_factor"] == TELESCOPING_GARI_COLUMN_FACTOR
    assert not any(key.startswith("cascade") for key in model)
    assert json.loads(json.dumps(model)) == model


# ---------------------------------------------------------------------------
# 5. M1 certification: budget, checkpoint, build watchdog model
# ---------------------------------------------------------------------------


def test_certification_plan_is_nominal_without_a_budget_and_shrinks_deterministically() -> None:
    policy = SecPolicyConfig(sec_candidates=2, sec_cert_iters=5, sec_cert_seeds=3, sec_cert_iters_scale=0.0,
                             sec_refine_top=2, sec_refine_iters=9)
    plan = tr.certification_plan(policy, survivors=2, kernel_dim=100)
    assert (plan["stage1_iters"], plan["refine_iters"], plan["refine_count"]) == (5, 9, 2)
    assert plan["isd_iters_planned"] == plan["isd_iters_nominal"] == 2 * 3 * 5 + 2 * 3 * 9
    assert plan["exhausted"] is False and plan["basis_budget"] is None
    # nominal 84 per basis; budget 100 -> 50 per basis: refine keeps <= 12,
    # stage 1 keeps 5 (38 // 6 = 6 >= 5), refine gets (50 - 30) // 6 = 3.
    capped = tr.certification_plan(
        SecPolicyConfig(**{**policy.to_dict(), "sec_cert_iters_budget": 100}), survivors=2, kernel_dim=100
    )
    assert (capped["stage1_iters"], capped["refine_iters"]) == (5, 3)
    assert capped["isd_iters_planned"] == 30 + 18 <= 50 and capped["exhausted"] is True
    # too small even for stage 1: at least 1 iteration each, refine skipped.
    tiny = tr.certification_plan(
        SecPolicyConfig(**{**policy.to_dict(), "sec_cert_iters_budget": 8}), survivors=2, kernel_dim=100
    )
    assert (tiny["stage1_iters"], tiny["refine_iters"], tiny["exhausted"]) == (1, 0, True)
    # the scaled stage-1 rule and the per-code total honour the budget
    formal = DEFAULT_SEC_POLICY
    assert tr.nominal_stage1_iters(formal, 760) == 3040
    assert tr.certification_isd_iterations(formal, kernel_dim=760) == 2 * (16 * 3 * 3040 + 4 * 3 * 2000)
    budgeted = SecPolicyConfig(sec_cert_iters_budget=60_000)
    assert tr.certification_isd_iterations(budgeted, kernel_dim=760) <= 60_000
    assert SecPolicyConfig.from_dict(budgeted.to_dict()) == budgeted
    with pytest.raises(ValueError, match="sec_cert_iters_budget"):
        SecPolicyConfig(sec_cert_iters_budget=-1)


def _bb72():
    compiled = build_compiled_code(BASELINE_BB_SPEC)
    return compiled, np.asarray(compiled.hx, dtype=np.uint8), np.asarray(compiled.hz, dtype=np.uint8)


def test_budget_exhausted_certification_is_flagged_deterministic_and_never_a_failure() -> None:
    compiled, hx, hz = _bb72()
    base = dict(sec_candidates=3, sec_cert_iters=6, sec_cert_seeds=2, sec_cert_iters_scale=0.0,
                sec_refine_top=2, sec_refine_iters=8)
    unlimited = tr.rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=SecPolicyConfig(**base))
    assert unlimited.metadata["certification_budget_exhausted"] is False
    assert unlimited.metadata["cert_plan"]["X"]["exhausted"] is False
    assert unlimited.schedule.to_metadata()["certification_budget_exhausted"] is False
    assert unlimited.metadata["isd_iters_executed"] == 2 * (3 * 2 * 6 + 2 * 2 * 8)
    capped_policy = SecPolicyConfig(**base, sec_cert_iters_budget=60)  # 30 per basis < 68 nominal
    first = tr.rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=capped_policy)
    second = tr.rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=capped_policy)
    assert first == second
    md = first.metadata
    assert md["certification_budget_exhausted"] is True and md["cert_iters_budget"] == 60
    assert md["isd_iters_executed"] <= 60
    for kind in ("X", "Z"):
        plan = md["cert_plan"][kind]
        assert plan["exhausted"] is True and plan["isd_iters_planned"] <= 30
        assert plan["stage1_iters"] <= 6 and plan["refine_iters"] <= 8
        assert md["stage1_iters"][kind] == plan["stage1_iters"]
    assert not first.catastrophic and first.d_circ_ext_x is not None
    assert first.schedule.to_metadata()["certification_budget_exhausted"] is True
    assert first.schedule.to_metadata()["refine_iters"] == 8  # the policy value the validator pins
    evidence = sec_ext_evidence(first)
    assert evidence["certification_budget_exhausted"] is True
    assert evidence["isd_iters_executed"] == md["isd_iters_executed"]
    assert evidence["status"] == "probe_upper_bound"
    assert json.loads(json.dumps(md)) == md


def test_certification_progress_resumes_with_identical_values(tmp_path) -> None:
    compiled, hx, hz = _bb72()
    policy = SecPolicyConfig(sec_candidates=3, sec_cert_iters=6, sec_cert_seeds=2, sec_cert_iters_scale=0.0,
                             sec_refine_top=2, sec_refine_iters=8)
    path = tmp_path / "progress.json"
    progress = tr.CertificationProgress(path)
    full = tr.rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=policy, progress=progress)
    assert path.is_file() and progress.reused == 0
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2 * (3 + 2)  # per basis: 3 stage-1 + 2 refine probes
    # a "killed" synthesis that finished half its probes: same result, half reused
    half = dict(list(entries.items())[: len(entries) // 2])
    path.write_text(json.dumps({"schema": "sec_certification_progress_v1", "entries": half}), encoding="utf-8")
    resumed_progress = tr.CertificationProgress(path)
    resumed = tr.rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks, policy=policy, progress=resumed_progress
    )
    assert resumed.schedule == full.schedule and resumed.d_circ_ext_x == full.d_circ_ext_x
    assert resumed.metadata["per_candidate"] == full.metadata["per_candidate"]
    assert resumed.metadata["progress_entries_reused"] == len(half)
    assert resumed.metadata["isd_iters_executed"] < full.metadata["isd_iters_executed"]
    # a complete checkpoint executes nothing
    again = tr.rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks, policy=policy, progress=tr.CertificationProgress(path)
    )
    assert again.metadata["isd_iters_executed"] == 0 and again.schedule == full.schedule
    # an entry from another plan (different iteration counts) is never reused
    other = SecPolicyConfig(**{**policy.to_dict(), "sec_cert_iters": 7, "sec_refine_iters": 9})
    fresh = tr.rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks, policy=other, progress=tr.CertificationProgress(path)
    )
    assert fresh.metadata["progress_entries_reused"] == 0
    assert tr.CertificationProgress(None).get("X", "stage1", 0, iters=6, seeds=2) is None


def test_synth_cache_checkpoints_and_discards_the_progress_file(tmp_path) -> None:
    from omniqec_agentic.verification import circuit as circuit_mod

    compiled, hx, hz = _bb72()
    seen: list[Path | None] = []
    real = tr.rank_twobasis_schedules

    def spy(*args, progress=None, **kwargs):
        seen.append(None if progress is None else progress.path)
        return real(*args, progress=progress, **kwargs)

    circuit_mod.build_memory_experiment_pair.__globals__  # noqa: B018 - module import guard
    import omniqec_agentic.sec.twobasis_rank as tr_mod

    original = tr_mod.rank_twobasis_schedules
    tr_mod.rank_twobasis_schedules = spy
    try:
        circuit_mod._SYNTH_CACHE.clear()
        circuit_mod.build_memory_experiment_pair(
            hx, hz, p=0.005, rounds=3, d_hint=6, synth_cache_dir=str(tmp_path), sec_policy=FAST_SEC,
            layer_blocks=compiled.layer_blocks,
        )
    finally:
        tr_mod.rank_twobasis_schedules = original
    assert len(seen) == 1 and seen[0] is not None and seen[0].parent == tmp_path
    assert seen[0].name.endswith(".progress.json") and not seen[0].exists()
    assert list(tmp_path.glob("synth_*.pkl"))
    assert circuit_mod._synth_progress_path(None, "k") is None


def test_build_watchdog_derives_from_the_certification_work_model() -> None:
    formal = VerificationConfig()
    model = tr.certification_work_model(DEFAULT_SEC_POLICY, n_data=200)
    assert model["isd_iterations"] == tr.certification_isd_iterations(DEFAULT_SEC_POLICY, kernel_dim=800)
    # Measured 156-223 min for the full Mitten200 certification on Apple
    # M-series; the model (2x margin) is above the slower reading.
    assert model["worst_case_s"] >= 223 * 60
    assert pool_mod.effective_build_timeout_s(formal, 200) == pytest.approx(model["worst_case_s"])
    # BB72 stays on the flat hour; unknown size / LRC policy / disabled: unchanged.
    assert pool_mod.effective_build_timeout_s(formal, 72) == pytest.approx(3600.0)
    assert pool_mod.effective_build_timeout_s(formal, None) == pytest.approx(3600.0)
    from omniqec_agentic.config import LRC_SEC_POLICY

    assert pool_mod.effective_build_timeout_s(VerificationConfig(sec_policy=LRC_SEC_POLICY), 200) == pytest.approx(3600.0)
    assert pool_mod.effective_build_timeout_s(VerificationConfig(build_timeout_s=0.0), 200) is None
    assert pool_mod.worker_timeout_for({"mode": "build_dem", "n_data": 200}, formal) == pytest.approx(model["worst_case_s"])
    assert pool_mod.worker_timeout_for({"mode": "build_dem"}, formal) == pytest.approx(3600.0)
    # a capped policy plans less work and the watchdog follows
    capped = VerificationConfig(sec_policy=SecPolicyConfig(sec_cert_iters_budget=60_000))
    assert pool_mod.effective_build_timeout_s(capped, 200) < model["worst_case_s"]


def test_pool_build_requests_carry_n_data_and_job_config_records_the_models(monkeypatch) -> None:
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    spec = run_construct_source(SEED_BB, program_id="review_bb", seed=0).constructed.spec
    seen: list[dict] = []
    fine = {"timeout": False, "returncode": 0, "stderr_tail": ""}

    def fake_worker(request, *, timeout_s):
        seen.append({**copy.deepcopy(request), "_timeout_s": timeout_s})
        mode = request.get("mode")
        if mode == "build_dem":
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12, "num_matrix_columns": {"X": 120_000, "Z": 60_000}}, dict(fine)
        if mode == "distance_job":
            return {"status": "ok"}, dict(fine)
        unit = request["units"][0]
        shots = int(request["shots"])
        cell = {"status": "ok", "decoder": unit["decoder"], "shots": shots, "failures": 1,
                "logical_error_rate": 1.0 / shots, "runtime_s": 0.01,
                "metrics": {"requested_shots": shots, "actual_shots": shots, "sampling_complete": True, "stopped_by_max_errors": False}}
        return {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}}, dict(fine)

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_worker)
    verification = VerificationConfig(shots=2_000, max_errors=None, rounds=12)
    payload = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "bb_review", "spec": spec}], verification=verification, hardware=None, seed=0,
        batch_size=256, prefer_coloring=True, workers=4, shot_shards=2, stage="final",
    )[0]
    assert payload["status"] == "ok"
    n_data = int(np.asarray(build_compiled_code(spec).hx).shape[1])
    builds = [r for r in seen if r["mode"] == "build_dem"]
    assert builds and all(r["n_data"] == n_data for r in builds)
    expected_build = pool_mod.effective_build_timeout_s(verification, n_data)
    assert all(r["_timeout_s"] == pytest.approx(expected_build) for r in builds)
    shards = [r for r in seen if r["mode"] == "shard_dem"]
    worst = pool_mod.telescoping_worst_case_shot_s(verification, 120_000, "final")
    assert shards and all(r["_timeout_s"] == pytest.approx(1000 * worst) for r in shards)
    job_config = payload["job_config"]
    assert job_config["n_data"] == n_data
    assert job_config["effective_build_timeout_s"] == pytest.approx(expected_build)
    assert job_config["effective_decode_shard_timeout_s"] == pytest.approx(1000 * worst)
    assert job_config["decode_budget_model"]["stage"] == "final"
    assert job_config["decode_budget_model"]["telescoping_worst_case_shot_s"] == pytest.approx(worst)
    assert job_config["decode_budget_model"]["flat_term_s"] == pytest.approx(16_000.0)


def test_sec_budget_is_protocol_identity() -> None:
    assert DEFAULT_SEC_POLICY.sec_cert_iters_budget == 0
    assert DEFAULT_SEC_POLICY.to_dict()["sec_cert_iters_budget"] == 0
    snap = _snapshot()
    assert snap["sec_policy"]["sec_cert_iters_budget"] == 0
    assert snap["protocol_profile"] == "telescoping_structural_v3_7"
    variant = _snapshot(verification=VerificationConfig(sec_policy=SecPolicyConfig(sec_cert_iters_budget=60_000)))
    assert variant["protocol_profile"] == "telescoping_structural_v3_7+secK16I400S0B60000"
    validator._validate_sec_policy_blob(DEFAULT_SEC_POLICY.to_dict(), "x")
    with pytest.raises(ValueError, match="sec_cert_iters_budget=60000"):
        validator._validate_sec_policy_blob(SecPolicyConfig(sec_cert_iters_budget=60_000).to_dict(), "x")
    with pytest.raises(ValueError, match="sec_cert_iters_budget=None"):
        validator._validate_sec_policy_blob({k: v for k, v in DEFAULT_SEC_POLICY.to_dict().items() if k != "sec_cert_iters_budget"}, "x")
    with pytest.raises(ValueError, match="sec_policy"):
        _snapshot(verification=VerificationConfig(sec_policy=SecPolicyConfig(sec_cert_iters_budget=1)),
                  strict_turns=True, required_provider="p", required_model="m")


# ---------------------------------------------------------------------------
# 6. coverage: every family's holes, round-robin cut
# ---------------------------------------------------------------------------

BB144 = {"family": "bb", "ell": 12, "m": 6, "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]}
HGP_SMALL = {"family": "hgp", "seeds": {"H1": [[1, 1, 0], [0, 1, 1]], "H2": [[1, 1], [1, 1]]}}
LP_D6 = {"family": "lp_nonabelian", "group": "D6", "A": [[[[0, 0], [1, 0]]]], "B": [[[[0, 0], [0, 1]]]]}


def _row(spec, code_hash, **extra):
    return {"spec": spec, "code_hash": code_hash, "family": spec.get("family"), **extra}


def test_gap_grid_places_every_family() -> None:
    assert coverage_mod.gap_region_of(BB144) == ("bb", "bb_rect", "60-100")
    assert coverage_mod.gap_region_of({"family": "bb", "ell": 4, "m": 4}) == ("bb", "bb_square", "12-24")
    assert coverage_mod.gap_region_of(HGP_SMALL) == ("hgp", "hgp", "n2-3|n2-3")
    assert coverage_mod.gap_region_of(LP_D6) == ("lp_nonabelian", "dihedral", "1x1|1x1")
    assert coverage_mod.gap_region_of({"family": "bb"}) is None and coverage_mod.gap_region_of(None) is None
    pairs = coverage_mod.feasible_branch_shape_pairs(288)
    by_family = {}
    for family, _branch, _region in pairs:
        by_family[family] = by_family.get(family, 0) + 1
    assert set(by_family) == {"lp_nonabelian", "lp_abelian", "bb", "hgp"}
    assert by_family["lp_nonabelian"] == 64 and by_family["lp_abelian"] == 32
    # bb: 2 classes x 5 bands fit at 288 (100+ needs 400 qubits); hgp: n16+|n16+ (576) does not.
    assert ("bb", "bb_square", "60-100") in pairs and ("bb", "bb_rect", "100+") not in pairs
    assert ("hgp", "hgp", "n2-3|n2-3") in pairs and ("hgp", "hgp", "n16+|n16+") not in pairs
    assert ("bb", "bb_rect", "100+") in coverage_mod.feasible_branch_shape_pairs(400)
    assert [p[0] for p in pairs] == sorted(
        [p[0] for p in pairs], key=lambda f: coverage_mod.GAP_FAMILY_ORDER.index(f)
    )


def test_every_family_gap_is_reported_and_the_cut_is_round_robin() -> None:
    rows = [_row(BB144, "a" * 32, k=12, d_search=12, n_budget=288)]
    gaps = coverage_mod.catalog_gaps(rows, 288)
    never = {(g["family"], g["branch"], g["shapes"]) for g in gaps["never_proposed_branch_shapes"]}
    assert ("bb", "bb_rect", "60-100") not in never  # BB144 lives there
    assert ("bb", "bb_square", "60-100") in never and ("bb", "bb_rect", "24-40") in never
    assert ("hgp", "hgp", "n4-6|n7-10") in never
    assert ("lp_nonabelian", "direct_product", "1x2|1x2") in never
    counts = gaps["never_proposed_by_family"]
    assert counts["bb"] == {"feasible": 10, "never_proposed": 9}
    assert counts["lp_nonabelian"] == {"feasible": 64, "never_proposed": 64}
    assert counts["hgp"]["feasible"] == counts["hgp"]["never_proposed"] > 0
    lines = coverage_mod.coverage_fact_lines(rows, 288, limit=12)
    first = lines[0]
    assert first.startswith("Regions never proposed at this budget (n_budget=288): ")
    shown = re.search(r"\): (.*?) \(\+\d+ more\)", first).group(1).split(", ")
    assert len(shown) == 12
    families = [item.split("/")[0] for item in shown]
    assert set(families) == {"lp_nonabelian", "lp_abelian", "bb", "hgp"}
    assert all(families.count(f) == 3 for f in set(families))  # 12 / 4 families
    branches = [tuple(item.split("/")[:2]) for item in shown]
    # inside a family the branches alternate: 3 lp_nonabelian slots -> 3
    # distinct branches, 3 lp_abelian / bb slots -> at most 2 per branch.
    assert len({b for b in branches if b[0] == "lp_nonabelian"}) == 3
    assert max(branches.count(b) for b in set(branches) if b[0] in ("lp_abelian", "bb")) <= 2
    assert "never proposed of spellable regions by family: bb 9 of 10; hgp" in first
    assert "lp_nonabelian 64 of 64" in first
    assert GOAL_RE.search("\n".join(lines)) is None
    for phrase in BANNED_PHRASES:
        assert phrase not in "\n".join(lines)
    assert re.search(r"\b(prefer|maximi[sz]e|should|must|try)\b", "\n".join(lines), re.I) is None
    # an lp-only archive reports the bb / hgp holes too (the audit's asymmetry)
    lp_lines = coverage_mod.coverage_fact_lines([_row(LP_D6, "b" * 32, k=8, d_search=4, n_budget=288)], 288, limit=12)
    assert "bb/bb_square/0-12" in lp_lines[0] and "hgp/hgp/n2-3|n2-3" in lp_lines[0]
    assert "bb 10 of 10" in lp_lines[0]
    # round_robin_gaps: exhausted families drop out, the rest keep alternating
    items = [{"family": "bb", "branch": "bb_square", "shapes": "a"},
             {"family": "hgp", "branch": "hgp", "shapes": "b"},
             {"family": "hgp", "branch": "hgp", "shapes": "c"},
             {"family": "hgp", "branch": "hgp", "shapes": "d"}]
    assert [i["shapes"] for i in coverage_mod.round_robin_gaps(items, 3)] == ["a", "b", "c"]
    assert coverage_mod.round_robin_gaps(items, 0) == [] and coverage_mod.round_robin_gaps([], 5) == []
