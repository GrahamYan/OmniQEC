"""Campaign graph (03 §2.1): the ONLY layer that imports langgraph.

intake -> [orchestrator_turn -> execute_plan -> reconcile]* -> final_slow ->
report. Policy decisions come from an injected Policy (FallbackPolicy for
dry-run/CI in Milestone A; the agentic Orchestrator lands in Milestone B
behind the same interface). Measurement stays entirely in evaluation/ — the
report renders numbers from artifacts, never from policy output.

Milestone-A seam: execute_plan drives one run_agentic_cycle per turn;
cross-turn memory flows through the slow-aware prompt-context overlay (the
per-turn fast archive is rebuilt from overlay parents, plus the hand-written
seed programs in turn 0 only -- paper appendix D1). Milestone B
dismantles funsearch into per-node steps with a persistent archive."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from omniqec_agentic.config import (
    ABLATION_PLAN_POLICIES,
    BASELINE_PLAN_POLICY_ID,
    BASELINE_PROVIDER,
    FINAL_STAGE_PAPER_ID,
    ROUNDS_POLICY_ID,
    ROUNDS_POLICY_IDS,
    CodeFamily,
    DecoderName,
    HardwareConfig,
    HardwareTarget,
    RacingConfig,
    RunConfig,
    VerificationConfig,
    ablation_of,
    search_method_of,
    seed_programs_run_this_turn,
)
from omniqec_agentic.evaluation import evaluate_promotion_queue
from omniqec_agentic.evaluation.sampling import RacingThresholds
from omniqec_agentic.orchestrator import FallbackPolicy, Policy
from omniqec_agentic.reasoning_fallback import check_fallback_request

from .checkpoint import (
    open_checkpointer,
    original_max_turns,
    resume_runtime_overrides,
    validate_resume_protocol,
    write_versions_snapshot,
)
from .state import (
    FORMAL_FINAL_SHOT_SHARDS,
    FORMAL_SHOT_SHARDS,
    CampaignSpec,
    CampaignState,
    assert_resumable_protocol,
    build_protocol_snapshot,
)

HOF_CAP = 12
_TURN_DIR_RE = re.compile(r"^turn_(\d{4})$")


def _telescoping_config_ok(config: Any) -> bool:
    """Does a recorded decoder-build blob match the frozen telescoping
    protocol (docs/06 §1.1, user decision 2026-09-03)? Every key of
    TELESCOPING_PROTOCOL_FIELDS must be present and equal and no other key
    may appear; nothing depends on the budget point (the cascade's S4 time
    limits, which used to be checked against the policy here, are gone
    with the cascade)."""

    from omniqec_agentic.config import TELESCOPING_PROTOCOL_FIELDS

    if not isinstance(config, dict):
        return False
    return dict(config) == dict(TELESCOPING_PROTOCOL_FIELDS)


def _bposd_config_ok(config: Any) -> bool:
    """Does a recorded decoder-build blob match the frozen BPOSD protocol
    (v3.5 profile; kept for the historical audit path only)?"""

    from omniqec_agentic.config import BPOSD_PROTOCOL_FIELDS

    if not isinstance(config, dict):
        return False
    return all(config.get(key) == value for key, value in BPOSD_PROTOCOL_FIELDS.items())


def _turn_dir(state: CampaignState, turn: int) -> Path:
    return Path(state["run_dir"]) / "turns" / f"turn_{turn:04d}"


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _audit_turn_execution(
    state: CampaignState,
    plan: dict[str, Any],
    summary: Any,
) -> dict[str, Any]:
    """Validate a strict turn before reconcile is allowed to count it."""

    protocol = state["protocol"]
    if not bool(protocol.get("strict_turns")):
        return {}

    turn = int(state["turn"])
    turn_dir = _turn_dir(state, turn)
    expected_provider = str(protocol.get("required_provider") or "")
    expected_model = str(protocol.get("required_model") or "")
    expected_programs = int(protocol["proposal_count"])
    expected_generations = int(protocol["generations_per_turn"])
    expected_rng_seeds = int(protocol["rng_seeds_per_program"])
    payload = summary.to_dict()
    funsearch = dict(payload.get("funsearch_summary") or {})
    slow_summary = payload.get("slow_summary")
    slow_summary = dict(slow_summary) if isinstance(slow_summary, dict) else {}
    errors: list[str] = []

    checks = {
        "turn": turn,
        "plan_policy": plan.get("policy"),
        "provider": plan.get("provider"),
        "model": expected_model,
        "program_count": plan.get("proposal_count"),
        "generations": plan.get("generations"),
        "rng_seeds_per_program": plan.get("rng_seeds_per_program"),
        "cycle_status": payload.get("status"),
        "proposal_count_requested_total": funsearch.get("proposal_count_requested_total"),
        "proposal_count_missing_total": funsearch.get("proposal_count_missing_total"),
        "generation_status_counts": funsearch.get("status_counts"),
        # v3.4, informational only: candidates whose decode exceeded the
        # frozen per-shot budget are a terminal measurement outcome, not a
        # turn defect, so the count is recorded but never gates the audit.
        "decode_infeasible_count": int(
            slow_summary.get("decode_infeasible_candidate_count") or 0
        ),
        # v3.6.1, informational only, same standing as decode_infeasible.
        "decoder_unsupported_count": int(
            slow_summary.get("decoder_unsupported_candidate_count") or 0
        ),
    }

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    # S3 ablation: no_feedback runs without a planner, so its plan policy is
    # the deterministic ablation policy and no orchestrator call exists.
    ablation = ablation_of(protocol)
    allowed_policies = ABLATION_PLAN_POLICIES[ablation]
    checks["ablation"] = ablation
    # 2026-11-06 search baselines: no planner and no model; the generator record is the sampler's.
    search_method = search_method_of(protocol)
    if search_method != "llm":
        allowed_policies = (BASELINE_PLAN_POLICY_ID,)
        checks["search_method"] = search_method
    require(
        plan.get("policy") in allowed_policies,
        f"plan policy is not {' / '.join(allowed_policies)}",
    )
    require(plan.get("provider") == expected_provider, "plan provider mismatch")
    require(int(plan.get("proposal_count") or 0) == expected_programs, "program count mismatch")
    require(int(plan.get("generations") or 0) == expected_generations, "generation count mismatch")
    require(
        int(plan.get("rng_seeds_per_program") or 0) == expected_rng_seeds,
        "rng seeds per program mismatch",
    )
    require(payload.get("provider") == expected_provider, "cycle provider mismatch")
    require(int(payload.get("proposal_count") or 0) == expected_programs, "cycle program count mismatch")
    require(int(payload.get("generations") or 0) == expected_generations, "cycle generations mismatch")
    require(payload.get("status") == "ok", f"cycle status is {payload.get('status')!r}")
    require(funsearch.get("provider") == expected_provider, "generator provider mismatch")
    require(
        int(funsearch.get("proposal_count_requested_total") or 0)
        == expected_programs * expected_generations,
        "generator requested-count mismatch",
    )
    require(
        int(funsearch.get("proposal_count_missing_total") or 0) == 0,
        "generator proposal batch is incomplete",
    )
    require(
        funsearch.get("status_counts") == {"ok": expected_generations},
        "generator status_counts is not all-ok",
    )
    require(
        int(funsearch.get("rng_seeds_per_program") or 0) == expected_rng_seeds,
        "generator rng seeds mismatch",
    )
    # Compliance of the family allocation. A steering channel nobody checks is
    # indistinguishable from no steering at all, so the produced distribution is
    # recorded next to the requested one every turn. Recorded, not enforced: the
    # generator answers in free-form programs and a family can legitimately come
    # up short, but a persistent gap is now visible instead of silent.
    quota = _family_quota(plan)
    if quota:
        produced = _produced_family_counts(turn_dir)
        checks["family_quota_requested"] = quota
        checks["family_quota_produced"] = produced
        checks["family_quota_shortfall"] = {
            family: count - int(produced.get(family, 0))
            for family, count in quota.items()
            if int(produced.get(family, 0)) < count
        }

    cycle_summary_path = turn_dir / "cycle" / "agentic_cycle_summary.json"
    require(cycle_summary_path.exists(), "missing agentic_cycle_summary.json")
    fast_summary_text = str((funsearch.get("artifact_paths") or {}).get("summary") or "")
    fast_summary_path = Path(fast_summary_text) if fast_summary_text else None
    require(
        fast_summary_path is not None and fast_summary_path.is_file(),
        "missing funsearch_summary.json",
    )

    orchestrator_calls = _jsonl_rows(turn_dir / "orchestrator" / "llm_calls.jsonl")
    if search_method != "llm":
        require(not orchestrator_calls, "a search-baseline turn must not call the orchestrator")
    elif ablation == "no_feedback":
        require(not orchestrator_calls, "no_feedback turn must not call the orchestrator")
    else:
        require(bool(orchestrator_calls), "missing orchestrator LLM call records")
    for row in orchestrator_calls:
        model_info = row.get("model") or {}
        require(model_info.get("provider") == expected_provider, "orchestrator provider mismatch")
        require(model_info.get("model") == expected_model, "orchestrator model mismatch")

    generations_text = str((funsearch.get("artifact_paths") or {}).get("generations") or "")
    generations_dir = Path(generations_text) if generations_text else turn_dir / "missing_generations"
    generator_calls: list[dict[str, Any]] = []
    for generation in range(1, expected_generations + 1):
        generator_calls.extend(_jsonl_rows(generations_dir / f"g{generation:04d}" / "llm_raw.jsonl"))
    require(bool(generator_calls), "missing generator LLM call records")
    for row in generator_calls:
        engine_info = row.get("engine_info") or {}
        require(row.get("status") == "ok", "generator LLM call status is not ok")
        require(engine_info.get("provider") == expected_provider, "generator call provider mismatch")
        require(engine_info.get("model") == expected_model, "generator call model mismatch")

    audit = {
        "schema": "campaign_turn_audit_v1",
        "turn": turn,
        "status": "execution_complete" if not errors else "failed",
        "checks": checks,
        "errors": errors,
    }
    audit_path = turn_dir / "turn_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(
            f"strict turn {turn} audit failed: " + "; ".join(errors[:6])
        )
    return audit


def _complete_turn_audit(state: CampaignState, *, digest_path: Path) -> None:
    protocol = state["protocol"]
    if not bool(protocol.get("strict_turns")):
        return
    turn = int(state["turn"])
    audit_path = _turn_dir(state, turn) / "turn_audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"strict turn {turn} is missing turn_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    # Reconcile writes artifacts before LangGraph commits its next checkpoint.
    # If the process dies in that narrow window, resume re-enters reconcile;
    # accepting an already-complete audit keeps that retry idempotent.
    if audit.get("status") == "complete":
        if not digest_path.exists():
            raise RuntimeError(f"strict turn {turn} is missing digest.json")
        return
    if audit.get("status") != "execution_complete":
        raise RuntimeError(f"strict turn {turn} execution audit is not complete")
    if not digest_path.exists():
        raise RuntimeError(f"strict turn {turn} is missing digest.json")
    audit["status"] = "complete"
    audit["digest_path"] = str(digest_path)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_all_turn_audits(state: CampaignState) -> dict[str, Any]:
    """Require exactly turn_0000..turn_0059 before strict final_slow."""

    protocol = state["protocol"]
    if not bool(protocol.get("strict_turns")):
        return {"enabled": False}
    expected_count = int(protocol["max_turns"])
    turns_dir = Path(state["run_dir"]) / "turns"
    expected_names = {f"turn_{turn:04d}" for turn in range(expected_count)}
    actual_names = {
        path.name
        for path in turns_dir.iterdir()
        if path.is_dir() and _TURN_DIR_RE.match(path.name)
    }
    # max_turns lowered on resume (2026-09-05): the turns this campaign ran
    # under the earlier bound stay on disk; they are neither audited here
    # nor eligible for the final stage (see _final_entries).
    before = original_max_turns(protocol)
    tolerated = (
        {f"turn_{turn:04d}" for turn in range(expected_count, int(before))}
        if before is not None and int(before) > expected_count
        else set()
    )
    errors: list[str] = []
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names - tolerated)
    if missing:
        errors.append(f"missing turn directories: {missing[:6]}")
    if unexpected:
        errors.append(f"unexpected turn directories: {unexpected[:6]}")
    allowed_policies = ABLATION_PLAN_POLICIES[ablation_of(protocol)]
    if search_method_of(protocol) != "llm":
        allowed_policies = (BASELINE_PLAN_POLICY_ID,)
    completed: list[int] = []
    reflection_turns: dict[str, list[int]] = {}
    for turn in range(expected_count):
        audit_path = turns_dir / f"turn_{turn:04d}" / "turn_audit.json"
        if not audit_path.exists():
            errors.append(f"turn {turn}: missing audit")
            continue
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"turn {turn}: unreadable audit ({exc})")
            continue
        # Informational (paper B4 alignment): the review outcome of every turn;
        # a failed review never fails the audit, it is listed.
        reflection_status = str((audit.get("reflection") or {}).get("status") or "unrecorded")
        reflection_turns.setdefault(reflection_status, []).append(turn)
        checks = audit.get("checks") or {}
        if audit.get("status") != "complete" or int(audit.get("turn", -1)) != turn:
            errors.append(f"turn {turn}: audit is not complete")
            continue
        if checks.get("plan_policy") not in allowed_policies:
            errors.append(f"turn {turn}: fallback/non-agentic policy")
            continue
        completed.append(turn)
    if completed != list(range(expected_count)):
        errors.append(f"completed audit count is {len(completed)}/{expected_count}")
    if errors:
        raise RuntimeError("strict campaign audit failed: " + "; ".join(errors[:10]))
    return {
        "enabled": True,
        "schema": "campaign_turn_audit_summary_v1",
        "expected": expected_count,
        "completed": len(completed),
        "turns": completed,
        "fallback_turns": 0,
        "beyond_max_turns": sorted(actual_names & tolerated),
        "reflection_turns": {key: value for key, value in sorted(reflection_turns.items())},
        "reflection_failed_turns": list(reflection_turns.get("failed") or []),
    }


def _row_plpr_ub(row: dict[str, Any]) -> float | None:
    ci = row.get("per_logical_per_round_ci")
    return float(ci[1]) if ci else None


def _row_plpr_lb(row: dict[str, Any]) -> float | None:
    ci = row.get("per_logical_per_round_ci")
    return float(ci[0]) if ci else None


def _hof_thresholds(state: CampaignState, racing: RacingConfig | None = None) -> RacingThresholds:
    """Threshold snapshot at execute_plan entry (03 §8): static per batch,
    on the plpr (paper A5) scale via per_logical_per_round_ci.

    2026-09-11 (pilot10x_v1): with ``racing.baseline_arm`` False the BB
    baseline arm is left unset, so the hall-of-fame best is the only
    reference and a turn with an empty hall of fame eliminates nothing."""

    hof_ub = None
    paper_rule = racing is not None and str(getattr(racing, "best_bound", "min_ub")) == "best_estimate"
    if paper_rule:
        # 2026-11-10 (paper Appendix D3): "the upper bound of the current best measured code" -- the
        # hall-of-fame row with the lowest point estimate (ties: the smaller upper bound) gives its OWN bound.
        measured = [
            row
            for row in state.get("hof_summary") or []
            if row.get("per_logical_per_round") is not None and _row_plpr_ub(row) is not None
        ]
        if measured:
            best = min(measured, key=lambda row: (float(row["per_logical_per_round"]), float(_row_plpr_ub(row))))
            hof_ub = _row_plpr_ub(best)
    else:
        for row in state.get("hof_summary") or []:
            ub = _row_plpr_ub(row)
            if ub is not None:
                hof_ub = ub if hof_ub is None else min(hof_ub, ub)
    baseline_ub = _row_plpr_ub(state.get("baseline_summary") or {})
    if baseline_ub is None and paper_rule:
        # The applicable reference code measured as an ordinary candidate (the generator proposed the
        # identical code before the resident row was injected, so no baseline row exists): its own row.
        reference = _resident_reference(state)
        if reference is not None:
            for row in state.get("hof_summary") or []:
                if str(row.get("code_hash") or "") == reference["code_hash"]:
                    baseline_ub = _row_plpr_ub(row)
                    break
            if baseline_ub is None and state.get("run_dir"):
                for row in _measured_evidence_rows(Path(str(state["run_dir"]))):
                    if str(row.get("code_hash") or "") == reference["code_hash"] and row.get("ci_high") is not None:
                        baseline_ub = float(row["ci_high"])
                        break
    if racing is not None and not bool(racing.baseline_arm):
        baseline_ub = None
    return RacingThresholds(hof_plpr_ub=hof_ub, baseline_plpr_ub=baseline_ub)


def _resident_reference(state: CampaignState) -> dict[str, Any] | None:
    """The run's applicable BB reference code (evaluation.baseline.select_resident_baseline of the budget
    point) with its code hash, its corresponding budget and Table VII cumulative limit; None when the budget
    has none."""

    budget = (state.get("spec") or {}).get("qubit_budget")
    if budget is None:
        budget = (state.get("protocol") or {}).get("n_budget")
    if budget is None:
        return None
    from omniqec_agentic.evaluation.final_stage import resident_reference

    return resident_reference(int(budget))


def intake_node(state: CampaignState) -> dict[str, Any]:
    """Freeze spec + protocol artifacts on disk (idempotent on resume)."""

    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "turns").mkdir(exist_ok=True)
    (run_dir / "campaign_spec.json").write_text(
        json.dumps(state["spec"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "protocol_snapshot.json").write_text(
        json.dumps(state["protocol"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_versions_snapshot(run_dir)
    return {
        "turn": int(state.get("turn") or 0),
        "hof_summary": state.get("hof_summary") or [],
        "structural_proxy_elites": state.get("structural_proxy_elites") or [],
        "structural_ledger_hashes": state.get("structural_ledger_hashes") or [],
        "unmeasurable_hashes": state.get("unmeasurable_hashes") or [],
        "decode_infeasible_hashes": state.get("decode_infeasible_hashes") or [],
        "baseline_summary": state.get("baseline_summary") or {},
        "budget": state.get("budget")
        or {
            "turns_used": 0,
            "wallclock_s": 0.0,
            "tokens_used": 0,
            "slow_shots_used": 0,
        },
        "stop": {},
    }


def make_orchestrator_node(policy: Policy):
    def orchestrator_turn(state: CampaignState) -> dict[str, Any]:
        # 2026-11-06 reasoning-saturation rule: once another budget point of this arm raised the flag
        # (env OMNIQEC_REASONING_FALLBACK_FLAG, final-protocol launches only), a run still at the default
        # level stops before its next turn and the launcher restarts it at the fallback level.
        check_fallback_request(state.get("protocol"))
        plan = policy.plan(dict(state))
        turn_dir = _turn_dir(state, int(state["turn"]))
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"plan": plan}

    return orchestrator_turn


def _with_search_shard_cap(verification: VerificationConfig, protocol: dict[str, Any]) -> VerificationConfig:
    """2026-09-25: apply the SEARCH-stage shard-watchdog ceiling (protocol
    runtime key search_decode_shard_timeout_cap_s) to the turn's verification.
    A capped shard is a v3.4 watchdog timeout -> decode_infeasible for that
    candidate, recorded in decode_infeasible_hashes and never retried."""

    cap = protocol.get("search_decode_shard_timeout_cap_s")
    if cap is None or float(cap) <= 0.0:
        return verification
    return dataclasses.replace(verification, search_decode_shard_timeout_cap_s=float(cap))


def execute_plan_node(state: CampaignState) -> dict[str, Any]:
    from omniqec_agentic.campaign import run_agentic_cycle

    start = time.perf_counter()
    plan = state["plan"] or {}
    protocol = state["protocol"]
    spec = state["spec"]
    turn = int(state["turn"])
    turn_dir = _turn_dir(state, turn)
    verification = _with_search_shard_cap(_verification_from_protocol(protocol, spec), protocol)
    racing = RacingConfig.from_dict(protocol.get("racing"))
    slow_workers = int(protocol.get("slow_workers") or 0)
    shot_shards = int(protocol["shot_shards"])
    # Racing needs the job pool: guard on the RESOLVED worker count (0 = auto
    # can resolve to 1 on small machines) and say so instead of silently
    # falling back to fixed-shot measurement.
    from omniqec_agentic.evaluation.pool import resolve_workers

    racing_enabled = bool(racing.enabled) and resolve_workers(slow_workers) > 1
    if bool(racing.enabled) and not racing_enabled:
        print(
            f"[campaign] racing requested but slow_workers={slow_workers} resolves to a "
            "single worker (no job pool); falling back to fixed-shot measurement",
            flush=True,
        )
    thresholds = _hof_thresholds(state, racing) if racing_enabled else None

    # S3 ablation (2026-10-23): "full" | "no_slow" | "no_feedback". Under an
    # ablation the slow-derived channels below are not merely empty, they are
    # not computed: no overlay from a slow stage, no evidence rows, no fact
    # tables, no measured identities, no queue summary, no promotions.
    ablation = ablation_of(protocol)
    overlay = None
    overlay_ptr = state.get("slow_context_ptr")
    if ablation == "full" and overlay_ptr and Path(overlay_ptr).exists():
        overlay = json.loads(Path(overlay_ptr).read_text(encoding="utf-8"))
    # v3.7: every measurement so far (with check weights) -- the runtime
    # order's evidence and the generator's fact tables come from ONE list.
    evidence_rows = _measured_evidence_rows(Path(state["run_dir"])) if ablation == "full" else []
    evidence_tables = _evidence_tables(
        Path(state["run_dir"]), evidence_rows,
        not_competitive_ratio=getattr(verification.structural, "not_competitive_ratio", None),
        not_competitive_max_rows=int(getattr(verification.structural, "not_competitive_max_rows", 12)),
    ) if ablation == "full" else {}
    if evidence_tables:
        overlay = dict(overlay or {})
        overlay["evidence_tables"] = evidence_tables
    overlay = dict(overlay or {})
    overlay["parent_window"] = int(verification.structural.parent_cap)
    # Fig.3 Phase B: the generator's facts header names the measured circuit.
    overlay["task"] = str(getattr(verification, "task", "memory"))
    # Fig.3 v2 (2026-10-15): the generator is told which of its programs are
    # measured codes or relabellings of them, and what became of the previous
    # turn's slow queue. Identities = every measured code with its canonical
    # (pynauty) hash; the queue summary comes from the previous turn's
    # slow_queue_composition.json.
    if ablation == "full" and bool(getattr(verification.structural, "measured_twin_feedback", False)):
        identities = _measured_identities(Path(state["run_dir"]), evidence_rows)
        overlay["measured_identities"] = identities
        last_queue = _last_turn_queue_summary(state, turn, identities)
        if last_queue:
            overlay["last_turn_queue"] = last_queue
    # no_feedback: the generator receives the task/constraint prompt alone --
    # no planner brief and no family allocation (there is no planner).
    brief = _strategy_brief(plan, state) if ablation != "no_feedback" else ""
    if brief:
        overlay = dict(overlay or {})
        overlay["strategy_brief"] = brief
    # family_mix used to reach the generator as one prose "attention" line and
    # nothing allocated the batch, so the policy could not actually steer where
    # proposals went. Convert it to integer slots the prompt states as a
    # requirement and _audit_turn_execution checks afterwards.
    quota = _family_quota(plan) if ablation != "no_feedback" else {}
    if quota:
        overlay = dict(overlay or {})
        overlay["family_quota"] = quota
    # 2026-11-06 search baselines: the fixed prior's draws are keyed by the frozen seed key and the turn.
    search_method = search_method_of(protocol)
    baseline_context: dict[str, Any] | None = None
    if search_method != "llm":
        baseline = dict(protocol.get("search_baseline") or {})
        overlay = dict(overlay or {})
        overlay["baseline_sampler"] = {"seed_key": str(baseline["seed_key"]), "turn": turn}
        baseline_context = {
            "method": search_method,
            "run_dir": str(state["run_dir"]),
            "turn": turn,
            "seed_key": str(baseline["seed_key"]),
            "init_random_turns": int(baseline.get("init_random_turns") or 0),
            "blocks": 2 if str(getattr(verification, "task", "memory")) == "cnot_dense" else 1,
            "bo": dict(baseline.get("bo") or {}),
        }
    # Experiment variant (2026-09-09): frozen background codes reach the
    # generator through the overlay, verbatim from the protocol snapshot.
    if isinstance(protocol.get("reference_codes"), dict):
        overlay = dict(overlay or {})
        overlay["reference_codes"] = protocol["reference_codes"]
    # v3.2 §5.2: raw generator hand-off notes stay on disk for audit but are
    # not fed back verbatim. They are an unstructured channel through which the
    # previous turn's ranking language re-enters the generator prompt; a
    # structured replacement needs its own schema and protocol version.
    _recent_generation_notes(Path(state["run_dir"]), before_turn=turn)

    # Parent order is the machine's (v3.2 §4.5; v3.7: 6 structural + 6
    # runtime interleaved 1:1, then 4 diversity picks), OVERRIDING whatever
    # runtime-first order the slow-aware overlay carried.
    structural_elites = list(state.get("structural_proxy_elites") or [])
    if ablation == "no_feedback":
        # Every turn starts from the prompt alone: no parent of any channel.
        # The authoritative empty list keeps the archive's own parent list
        # (prompt_context_payload) out of the merged context as well.
        parents = []
        overlay["next_parent_specs"] = []
        overlay["parent_specs_authoritative"] = True
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "parent_selection.json").write_text(
            json.dumps({"schema": "machine_parent_selection", "ablation": ablation,
                        "render_order": [], "note": "no_feedback: no parents of any channel"},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        parents = _machine_parent_specs(
            structural_elites=structural_elites,
            # no_slow: the hall of fame does not exist (nothing is measured);
            # passing it empty is belt-and-braces, the quota moves to the
            # structural pool inside.
            hof_summary=list(state.get("hof_summary") or []) if ablation == "full" else [],
            structural_config=verification.structural,
            turn_dir=turn_dir,
            ablation=ablation,
        )
    if parents:
        overlay = dict(overlay or {})
        overlay["next_parent_specs"] = parents
        overlay["parent_specs_authoritative"] = True

    # v3.8: variant-generation parents and the endpoint demotion verdict are
    # computed here, from checkpointed state and the run's own artifacts, and
    # travel to the fast loop through the overlay; both are written to the
    # turn directory for the audit trail.
    structural_rules = verification.structural
    demoted_families: list[str] = []
    if int(getattr(structural_rules, "exploit_generations", 0) or 0) > 0:
        # no_feedback (Table VIII w/o slow & fast): no parent of any channel, the resident reference included,
        # so its variant generations fall back to exploration (variant_fallback_no_parents).
        exploit_parents = [] if ablation == "no_feedback" else _exploit_parent_specs(
            hof_summary=list(state.get("hof_summary") or []),
            baseline_summary=dict(state.get("baseline_summary") or {}),
            n_budget=int(spec["qubit_budget"]),
            structural_config=structural_rules,
        )
        overlay = dict(overlay or {})
        overlay["exploit_parent_specs"] = exploit_parents
        overlay["exploit_generations"] = int(structural_rules.exploit_generations)
        (turn_dir / "exploit_parents.json").write_text(
            json.dumps(
                {
                    "schema": "exploit_parent_selection",
                    "turn": turn,
                    "exploit_generations": int(structural_rules.exploit_generations),
                    "parents": [
                        {
                            "program_id": p.get("program_id"),
                            "code_hash": str(p.get("code_hash") or "")[:12],
                            "family": p.get("family"),
                            "n": p.get("n"), "k": p.get("k"), "d_search": p.get("d_search"),
                            "is_baseline": bool(p.get("is_baseline")),
                            "slow_per_logical_per_round": p.get("slow_per_logical_per_round"),
                        }
                        for p in exploit_parents
                    ],
                },
                ensure_ascii=False, indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if int(getattr(structural_rules, "endpoint_demotion_strikes", 0) or 0) > 0:
        verdict = _endpoint_demotion(
            Path(state["run_dir"]),
            before_turn=turn,
            strikes=int(structural_rules.endpoint_demotion_strikes),
            factor=float(structural_rules.endpoint_demotion_factor),
            reference_plpr=_best_measured_plpr(
                list(state.get("hof_summary") or []), dict(state.get("baseline_summary") or {})
            ),
        )
        demoted_families = list(verdict["demoted_families"])
        overlay = dict(overlay or {})
        overlay["demoted_families"] = demoted_families
        (turn_dir / "endpoint_demotion.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # The slow-queue cap is frozen in the protocol snapshot (v3.2 §4.3). A plan
    # field, a promotion count or a CLI override may not resize it; drift fails
    # the turn instead of being truncated and carried on with.
    planned_cap = plan.get("promote_top")
    if planned_cap is not None and int(planned_cap) != int(verification.promote_top):
        raise RuntimeError(
            f"turn {turn}: plan promote_top={planned_cap} contradicts the frozen "
            f"slow-queue cap {verification.promote_top}"
        )
    # TurnPlan.promotions is a validated, agent-ordered list of code-hash
    # prefixes — it adds named candidates to this turn's slow queue (03a);
    # dropping it made the promotion-validation machinery decorative, and
    # letting it DISPLACE the fresh Q-head starved the slow loop of new
    # codes (b144 attempt 1). The campaign-level merge appends, never
    # displaces, and skips already-measured hashes via the evidence ledger.
    promotion_priority = [
        str(item.get("code_hash"))
        for item in (plan.get("promotions") or [])
        if isinstance(item, dict) and item.get("code_hash")
    ] if ablation == "full" else []
    measured_hashes: set[str] = set()
    evidence_path = Path(state["run_dir"]) / "evidence_rows.jsonl"
    if ablation == "full" and evidence_path.exists():
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                code_hash = json.loads(line).get("code_hash")
            except Exception:
                continue
            if code_hash:
                measured_hashes.add(str(code_hash))
    # A schedule_catastrophic candidate produces no measurement, so it never
    # reaches the evidence ledger — and a structural elite that cannot be
    # measured would otherwise re-take a slow slot every single turn.
    measured_hashes |= {str(item) for item in (state.get("unmeasurable_hashes") or [])}
    config = RunConfig(
        run_id=f"{state['run_id']}_t{turn:04d}",
        n_budget=int(spec["qubit_budget"]),
        hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir=str(turn_dir / "cycle"),
        verification=verification,
        proposal_count=int(plan.get("proposal_count") or protocol["proposal_count"]),
        allowed_families=tuple(CodeFamily(f) for f in spec["families"]),
        feedback_mode=str(plan.get("feedback_mode") or protocol["feedback_mode"]),
    )
    # Paper alignment (2026-11-10, appendix D1): the five hand-written seed
    # programs initialise the search in the first iteration only; later turns
    # keep what turn 0's seeds produced through the cross-turn structural
    # ledger, the hall of fame and the evidence rows, without re-running them.
    use_seeds = seed_programs_run_this_turn(protocol, turn)
    summary = run_agentic_cycle(
        config,
        provider=str(plan.get("provider") or "dry_run"),
        use_seeds=use_seeds,
        generations=int(plan.get("generations") or protocol["generations_per_turn"]),
        slow_shots=int(verification.shots),
        slow_max_errors=verification.max_errors,
        slow_workers=slow_workers,
        slow_shot_shards=shot_shards,
        rng_seeds_per_program=int(plan.get("rng_seeds_per_program") or 1),
        racing=racing if racing_enabled else None,
        racing_thresholds=thresholds,
        prompt_context_overlay=overlay,
        promotion_priority=promotion_priority or None,
        promotion_measured=measured_hashes or None,
        # Baseline is measured once per campaign; state carries the summary
        # (reconcile only ever replaces it with a better measurement).
        baseline_already_measured=bool(state.get("baseline_summary")),
        require_full_llm_batch=bool(protocol.get("strict_turns")),
        structural_elites=structural_elites,
        structural_registry_path=str(Path(state["run_dir"]) / "structural_ledger.jsonl"),
        # v3.7: every measurement so far, with check weights, for the
        # expected-plpr runtime order and the prompt's evidence tables.
        evidence_rows=evidence_rows,
        turn=turn,
        # 2026-09-29 runtime key: the promotion weight gate (None = off).
        promotion_max_check_weight=(
            None if protocol.get("promotion_max_check_weight") is None
            else int(protocol["promotion_max_check_weight"])
        ),
        # Fig.4 c (2026-10-23): the exact-weight contract at the slow queue.
        exact_check_weight=(
            None if protocol.get("exact_check_weight") is None
            else int(protocol["exact_check_weight"])
        ),
        ablation=ablation,
        # v3.8: recorded in slow_queue_composition.json (the structural queue
        # itself was already built without these families).
        demoted_families=demoted_families,
        search_method=search_method,
        baseline_context=baseline_context,
    )
    _audit_turn_execution(state, plan, summary)
    budget = dict(state.get("budget") or {})
    budget["wallclock_s"] = float(budget.get("wallclock_s") or 0.0) + (
        time.perf_counter() - start
    )
    slow_shot_count = 0
    for row in _slow_rows(summary.artifact_paths.get("slow_rankings")):
        slow_shot_count += int(row.get("total_shots") or 0)
    budget["slow_shots_used"] = int(budget.get("slow_shots_used") or 0) + slow_shot_count
    return {
        "last_cycle": {
            "turn": turn,
            "status": summary.status,
            "artifact_paths": summary.artifact_paths,
            "racing_thresholds": thresholds.to_dict() if thresholds else None,
            "turn_audit": str(turn_dir / "turn_audit.json")
            if bool(protocol.get("strict_turns"))
            else "",
        },
        "slow_context_ptr": summary.artifact_paths.get("slow_prompt_context") or "",
        "budget": budget,
    }


def _produced_family_counts(turn_dir: Path) -> dict[str, int]:
    """Family distribution actually produced this turn, from the archive queue."""

    from collections import Counter

    queue = turn_dir / "cycle" / "fast" / "all_archive_queue.jsonl"
    if not queue.exists():
        return {}
    counts: Counter[str] = Counter()
    for line in queue.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        family = row.get("family") or (row.get("spec") or {}).get("family")
        if family:
            counts[str(family)] += 1
    return dict(counts)


def _family_quota(plan: dict[str, Any]) -> dict[str, int]:
    """Turn the plan's family_mix fractions into integer proposal slots.

    Largest-remainder apportionment: the slots sum to exactly program_count, so
    the allocation the prompt states is the allocation the batch can satisfy.
    """

    mix = {
        str(family): float(weight)
        for family, weight in (plan.get("family_mix") or {}).items()
        if float(weight) > 0
    }
    total = int(plan.get("proposal_count") or 0)
    if not mix or total <= 0:
        return {}
    weight_sum = sum(mix.values())
    if weight_sum <= 0:
        return {}
    exact = {family: total * weight / weight_sum for family, weight in mix.items()}
    quota = {family: int(value) for family, value in exact.items()}
    remainder = total - sum(quota.values())
    for family, _ in sorted(
        exact.items(), key=lambda item: (-(item[1] - int(item[1])), item[0])
    ):
        if remainder <= 0:
            break
        quota[family] += 1
        remainder -= 1
    return {family: count for family, count in quota.items() if count > 0}


NOTES_TURN_WINDOW = 3


def _recent_generation_notes(run_dir: Path, *, before_turn: int) -> list[dict[str, Any]]:
    """The last few turns' generator hand-off notes, newest first.

    Read from finished turn artifacts rather than threaded through state: the
    notes are a prompt-shaping convenience, and rebuilding them from disk keeps
    them out of the checkpoint (and therefore out of the protocol snapshot).
    """

    rows: list[dict[str, Any]] = []
    for candidate in range(int(before_turn) - 1, -1, -1):
        if len(rows) >= NOTES_TURN_WINDOW:
            break
        path = run_dir / "turns" / f"turn_{candidate:04d}" / "cycle" / "fast" / "generation_notes.json"
        if not path.is_file():
            continue
        try:
            notes = list((json.loads(path.read_text(encoding="utf-8")) or {}).get("notes") or [])
        except Exception:
            continue
        if notes:
            rows.append({"turn": candidate, "notes": notes})
    return rows


def _latest_reflection_advice(run_dir: Path) -> str:
    """The newest reflection's advice line.

    v3.2 §5.2: kept for the POLICY path (it reaches the orchestrator as S-10)
    and for inspection. It is deliberately no longer part of the generator's
    strategy brief — that route let a turn's own prose re-enter the generator
    as if it were protocol."""

    from omniqec_agentic.orchestrator.reflector import latest_advice, read_reflections

    try:
        return latest_advice(read_reflections(run_dir / "manual"))
    except Exception:
        return ""


def _strategy_brief(plan: dict[str, Any], state: CampaignState) -> str:
    """The generator's strategy-brief block (03a §4.1-1), v3.2 §5.2.

    STRUCTURED FIELDS ONLY. Three channels used to carry free text from the
    policy layer into the generator prompt verbatim — `plan.directives`, the
    reflector's `next_turn_advice`, and raw generation notes — and each of them
    could restate the retired global ranking rules ("plpr decides", "prefer
    higher Q", "degraded: do not clone") no matter how carefully the generator
    prompt itself was neutralised. The policy still records all three; they
    reach the POLICY (S-8, S-10), not the generator.

    Backtester-SUPPORTED insight ids stay: they are adjudicated statements and
    the generator's `# exploits: I-<id>` tags need them to mean anything.
    Un-adjudicated active hypotheses do not."""

    from omniqec_agentic.memory.ledger import Ledger

    parts = ["STRATEGY CONTEXT (structured allocation from the campaign policy):"]
    family_mix = plan.get("family_mix") or {}
    if family_mix:
        rendered = ", ".join(f"{fam}: {weight:.2f}" for fam, weight in sorted(family_mix.items()))
        parts.append(f"- family attention: {rendered}")
    strategy_mix = plan.get("strategy_mix") or {}
    if strategy_mix:
        rendered = ", ".join(
            f"{arm}: {weight:.2f}" for arm, weight in sorted(strategy_mix.items())
        )
        parts.append(f"- strategy emphasis: {rendered}")
    # Paper B4 (2026-11-10 alignment): the strategy recommendation weights
    # (Eq. B4, memory/bandit.py) "inform the orchestrator's next iteration plan
    # and reach the code-search agent through its prompt" -- read literally,
    # the weights themselves, next to the plan's own strategy emphasis.
    weights_line = _strategy_weights_line(Path(state["run_dir"]))
    if weights_line:
        parts.append(weights_line)
    ledger_path = Path(state["run_dir"]) / "ledger.json"
    if ledger_path.exists():
        supported = Ledger(ledger_path).supported()
        if supported:
            parts.append("- adjudicated insights you may cite in your tags:")
            for entry in supported[:12]:
                parts.append(f"  {entry.render_line()}")
    coverage_line = _coverage_fact_line(state)
    if coverage_line:
        parts.append(coverage_line)
    if len(parts) == 1:
        return ""
    return "\n".join(parts)


def _strategy_weights_line(run_dir: Path) -> str:
    """One line with the bandit's recommendation weights (uniform before any
    measured contribution); empty when the bandit cannot be read."""

    try:
        from omniqec_agentic.memory.bandit import StrategyBandit

        mix = StrategyBandit(run_dir / "bandit.json").suggestion().get("mix") or {}
    except Exception:
        return ""
    if not mix:
        return ""
    rendered = ", ".join(f"{arm}: {float(weight):.2f}" for arm, weight in sorted(mix.items()))
    return (
        "- strategy recommendation weights (from accumulated measured contributions; "
        f"recommendations, not quotas): {rendered}"
    )


def _coverage_fact_line(state: CampaignState) -> str:
    """ONE structured coverage line (docs/06 §4.3): neutral facts about which
    (branch x shape) regions have never been proposed at this budget, from
    `memory.coverage.coverage_fact_lines`. The function is being added by the
    coverage work; until it exists the brief simply omits the line."""

    try:
        from omniqec_agentic.memory import coverage as coverage_mod

        fact_lines = coverage_mod.coverage_fact_lines
    except (ImportError, AttributeError):
        return ""
    try:
        n_budget = int((state.get("spec") or {}).get("qubit_budget") or 0)
    except (TypeError, ValueError):
        return ""
    rows = _structural_store_light_rows(Path(state["run_dir"]) / "structural_ledger.jsonl")
    try:
        lines = fact_lines(rows, n_budget)
    except (ImportError, AttributeError):
        return ""
    except Exception:
        return ""
    lines = [str(line).strip() for line in (lines or []) if str(line).strip()]
    if not lines:
        return ""
    return "- coverage facts: " + " | ".join(lines)


def _structural_store_light_rows(store_path: Path) -> list[dict[str, Any]]:
    """Light rows (spec + coordinates) streamed from the run-level ledger."""

    if not store_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with store_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    key: row.get(key)
                    for key in (
                        "code_hash", "family", "spec", "n", "n_total", "k",
                        "d_search", "budget_utilization",
                    )
                }
            )
    return rows


def _rebuilt_code_hash(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Re-derive a row's canonical code from its inheritance payload.

    A parent is only inheritable if `construct_source` + `rng_seed` still
    produce the code the row claims. Rows that cannot be rebuilt (or rebuild
    to a different hash) stay in the report and are recorded as protocol
    errors — they never occupy a parent slot on the strength of a stale
    candidate file."""

    from omniqec_agentic.construction import build_compiled_code, compiled_code_hash
    from omniqec_agentic.sandbox import run_construct_source

    source = str(row.get("construct_source") or "")
    claimed = str(row.get("code_hash") or "")
    if not source.strip():
        return None, "missing construct_source"
    if not claimed:
        return None, "missing code_hash"
    try:
        constructed = run_construct_source(
            source,
            program_id=str(row.get("program_id") or "runtime_parent"),
            seed=int(row.get("rng_seed") or 0),
        ).constructed
        compiled = build_compiled_code(
            constructed.spec,
            deformation_mode=row.get("deformation_mode") or None,
        )
        rebuilt = compiled_code_hash(compiled.hx, compiled.hz, h_mask=compiled.h_mask)
    except Exception as exc:
        return None, f"rebuild failed: {exc}"
    if rebuilt != claimed:
        return None, f"rebuilt hash {rebuilt[:12]} != recorded {claimed[:12]}"
    return constructed.spec, ""


def _runtime_parent_specs(
    hof_summary: list[dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Runtime parents from the CROSS-turn hall of fame (v3.2 §4.5).

    Cross-turn, not this turn's `slow_parent_specs`: the runtime channel's job
    is to carry what measured well across the whole campaign, and a per-turn
    view forgets a champion the moment a turn fails to reproduce it."""

    parents: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in hof_summary:
        if len(parents) >= max(0, int(limit)):
            break
        spec, error = _rebuilt_code_hash(row)
        if spec is None:
            errors.append(f"{str(row.get('code_hash') or '?')[:12]}: {error}")
            continue
        # v3.7: hall-of-fame rows predate the weight column; derive the check
        # weight from the rebuilt code so the parent line, the per-weight
        # class cap and the diversity key see it.
        weight = row.get("max_check_weight")
        if weight is None:
            weight = _spec_check_weight(spec, deformation_mode=row.get("deformation_mode") or None)
        parents.append(
            {
                "code_hash": row.get("code_hash"),
                "program_id": row.get("program_id"),
                "family": row.get("family"),
                "spec": row.get("spec") if isinstance(row.get("spec"), dict) else spec,
                "construct_source": row.get("construct_source"),
                "rng_seed": row.get("rng_seed", 0),
                "source_agent": row.get("source_agent"),
                "deformation_mode": row.get("deformation_mode") or "none",
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search": row.get("d_search") or row.get("d_proxy"),
                "slow_per_logical_per_round": row.get("per_logical_per_round"),
                "slow_decoder": row.get("decoder"),
                "selection_channels": ["runtime"],
                "first_seen_turn": row.get("first_seen_turn"),
                "last_measured_turn": row.get("last_measured_turn"),
                # v3.7: the check weight travels with the parent (facts line).
                "max_check_weight": weight,
            }
        )
    return parents, errors


def _spec_check_weight(spec: Any, *, deformation_mode: str | None = None) -> int | None:
    """Max stabilizer row weight of a spec's compiled code (None if it does
    not build). Used where a row carries no `max_check_weight` column."""

    if not isinstance(spec, dict):
        return None
    try:
        from omniqec_agentic.construction import build_compiled_code

        import numpy as np

        compiled = build_compiled_code(spec, deformation_mode=deformation_mode)
        hx = np.asarray(compiled.hx, dtype=np.int64)
        hz = np.asarray(compiled.hz, dtype=np.int64)
        return int(max(int(hx.sum(axis=1).max()), int(hz.sum(axis=1).max())))
    except Exception:
        return None


# v3.7: the runtime parent window holds at most this many codes of one check
# weight, so the measured-best list still shows more than one weight class.
RUNTIME_PARENTS_PER_WEIGHT_CLASS = 2


def _parent_region(row: dict[str, Any]) -> tuple:
    """(family, A/B shape, weight-symmetry, check weight) of a parent -- the
    diversity key. Shapes from construction.structure, symmetry from the
    weight profile."""

    from omniqec_agentic.construction.structure import structure_descriptor, weight_profile

    spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
    descriptor = structure_descriptor(spec) if spec else {}
    profile = weight_profile(spec) if spec else {}
    return (
        str(spec.get("family") or row.get("family") or "?"),
        f"{descriptor.get('shape_a') or '?'}|{descriptor.get('shape_b') or '?'}",
        profile.get("symmetric"),
        row.get("max_check_weight"),
    )


def _diversity_parent_specs(
    *,
    structural_elites: list[dict[str, Any]],
    hof_rows: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """v3.7 diversity parents: up to `limit` inheritable codes whose region
    (family, A/B shape, |A|=|B| or not, check weight) is not represented by
    the structural + runtime parents already chosen. Candidates come from the
    structural pool and the hall of fame, in their own (channel-neutral) order;
    a region is taken once. Facts about coverage, not a preference: an
    unrepresented region gets one parent, never a slot."""

    limit = max(0, int(limit))
    if limit <= 0:
        return []
    chosen_keys = {str(row.get("code_hash") or row.get("program_id") or "") for row in chosen}
    covered = {_parent_region(row) for row in chosen}
    out: list[dict[str, Any]] = []
    for row in list(structural_elites) + list(hof_rows):
        if len(out) >= limit:
            break
        key = str(row.get("code_hash") or row.get("program_id") or "")
        if not key or key in chosen_keys:
            continue
        if not isinstance(row.get("spec"), dict) or not str(row.get("construct_source") or "").strip():
            continue
        region = _parent_region(row)
        if region in covered:
            continue
        covered.add(region)
        chosen_keys.add(key)
        item = dict(row)
        item["selection_channels"] = ["diversity"]
        item["selection_reasons"] = ["unrepresented_region"]
        out.append(item)
    return out


def _resident_baseline_parent(
    n_budget: int, baseline_summary: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The resident reference code as a v3.8 variant-generation parent.

    The spec is the protocol's (evaluation.baseline), rendered as a one-line
    program; the measured facts come from the run's baseline summary once
    the reference has been measured (turn 0 onwards), the [[n, k, d]] from
    the parameter table before that. None where no reference fits (N between
    144 and 288)."""

    from omniqec_agentic.construction.structure import weight_profile
    from omniqec_agentic.evaluation.baseline import (
        resident_baseline_parameters,
        select_resident_baseline,
    )

    picked = select_resident_baseline(int(n_budget))
    if picked is None:
        return None
    spec, program_id = picked
    n, k, d = resident_baseline_parameters(program_id)
    summary = dict(baseline_summary or {})
    weight = summary.get("max_check_weight")
    if weight is None:
        weight = _spec_check_weight(spec)
    return {
        "code_hash": summary.get("code_hash"),
        "program_id": program_id,
        "family": str(spec.get("family")),
        "spec": dict(spec),
        "construct_source": "def construct(rng):\n    return " + repr(dict(spec)) + "\n",
        "rng_seed": 0,
        "source_agent": "resident_baseline",
        "deformation_mode": "none",
        "n": summary.get("n") or n,
        "k": summary.get("k") or k,
        "d_search": summary.get("d_search") or summary.get("d_proxy") or d,
        "slow_per_logical_per_round": summary.get("per_logical_per_round"),
        "slow_decoder": summary.get("decoder"),
        "selection_channels": ["reference"],
        "is_baseline": True,
        "max_check_weight": weight,
        "weight_profile": weight_profile(dict(spec)),
    }


def _exploit_parent_specs(
    *,
    hof_summary: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
    n_budget: int,
    structural_config: Any,
) -> list[dict[str, Any]]:
    """Parents of a v3.8 variant generation: the hall of fame by measured
    plpr with the runtime window's per-weight-class cap (at most
    runtime_parents - 1 of them) followed by the resident reference code.
    Rendered in that order; the reference is last, never first."""

    from omniqec_agentic.construction.structure import weight_profile

    limit = max(0, int(getattr(structural_config, "runtime_parents", 6)) - 1)
    candidates, _errors = _runtime_parent_specs(hof_summary, limit=len(hof_summary))
    ordered: list[dict[str, Any]] = []
    per_weight: dict[Any, int] = {}
    for row in candidates:
        if len(ordered) >= limit:
            break
        weight = row.get("max_check_weight")
        if per_weight.get(weight, 0) >= RUNTIME_PARENTS_PER_WEIGHT_CLASS:
            continue
        per_weight[weight] = per_weight.get(weight, 0) + 1
        ordered.append(row)
    for row in candidates:
        if len(ordered) >= limit:
            break
        if row not in ordered:
            ordered.append(row)
    for row in ordered:
        row.setdefault("weight_profile", weight_profile(row.get("spec")))
    reference = _resident_baseline_parent(int(n_budget), baseline_summary)
    return ordered + ([reference] if reference else [])


ENDPOINT_SLOT_REASONS = frozenset(
    {"high_k_endpoint", "high_d_search_endpoint", "high_d_search_endpoint_other_family"}
)


def _endpoint_demotion(
    run_dir: Path,
    *,
    before_turn: int,
    strikes: int,
    factor: float,
    reference_plpr: float | None,
) -> dict[str, Any]:
    """v3.8 endpoint demotion verdict for the turn about to run.

    Reads every earlier turn's slow rankings and keeps, per family, the
    chronological list of its structural ENDPOINT measurements. An endpoint
    measurement is a strike when the candidate did not measure (decode
    infeasible / error), was eliminated by the racing rule, or measured a
    plpr >= `factor` x `reference_plpr` (the best measured code of the
    campaign so far, the resident reference included). A family whose last
    `strikes` endpoint measurements are all strikes is demoted: it takes no
    structural slot in this turn (ranking.select_structural_slots). With no
    reference yet, only non-measurements and racing eliminations count."""

    history: dict[str, list[dict[str, Any]]] = {}
    for turn in range(0, max(0, int(before_turn))):
        path = run_dir / "turns" / f"turn_{turn:04d}" / "cycle" / "slow" / "rankings.json"
        if not path.is_file():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8")).get("rankings") or []
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("is_baseline"):
                continue
            if "structural" not in set(row.get("selection_channels") or []):
                continue
            if str(row.get("structural_slot_reason") or "") not in ENDPOINT_SLOT_REASONS:
                continue
            family = str(row.get("family") or "?")
            plpr = row.get("per_logical_per_round")
            status = str(row.get("status") or "")
            strike_reason = None
            if status != "ok" or plpr is None:
                strike_reason = f"status={status or 'missing'}"
            elif str(row.get("stopped_by") or "") == "racing":
                strike_reason = "racing"
            elif reference_plpr is not None and float(plpr) >= float(factor) * float(reference_plpr):
                strike_reason = f"plpr {float(plpr):.2e} >= {float(factor):g} x {float(reference_plpr):.2e}"
            history.setdefault(family, []).append(
                {
                    "turn": turn,
                    "code_hash": str(row.get("code_hash") or "")[:12],
                    "reason": row.get("structural_slot_reason"),
                    "plpr": plpr,
                    "strike": strike_reason is not None,
                    "strike_reason": strike_reason,
                }
            )
    demoted: list[str] = []
    for family, rows in sorted(history.items()):
        if len(rows) >= int(strikes) and all(item["strike"] for item in rows[-int(strikes):]):
            demoted.append(family)
    return {
        "schema": "endpoint_demotion",
        "turn": int(before_turn),
        "strikes": int(strikes),
        "factor": float(factor),
        "reference_plpr": reference_plpr,
        "demoted_families": demoted,
        "history": history,
    }


def _best_measured_plpr(
    hof_summary: list[dict[str, Any]], baseline_summary: dict[str, Any] | None
) -> float | None:
    values: list[float] = []
    for row in list(hof_summary or [])[:1] + [dict(baseline_summary or {})]:
        value = row.get("per_logical_per_round") if isinstance(row, dict) else None
        if value is not None:
            values.append(float(value))
    return min(values) if values else None


def _machine_parent_specs(
    *,
    structural_elites: list[dict[str, Any]],
    hof_summary: list[dict[str, Any]],
    structural_config: Any,
    turn_dir: Path,
    ablation: str = "full",
) -> list[dict[str, Any]]:
    """The turn's 16 parents (v3.7): 6 structural + 6 runtime interleaved
    S,R, then 4 diversity picks.

    S3 ablation "no_slow" (2026-10-23): there is no runtime channel, so the
    runtime quota (6) is handed to the structural pool -- 12 structural
    parents in unique-cell round-robin order, then the 4 diversity picks --
    and the window stays 16 wide.

    v3.6 (docs/06 §4.1): the structural side iterates round-robin over UNIQUE
    cells (k, d_search, weight) — one row per cell in the reducer's pool
    order, then the second realizations — so the window shows the frontier's
    shape rather than several realizations of one endpoint. v3.7: the runtime
    side is the hall of fame by measured plpr with at most two codes per
    check-weight class; the diversity picks cover regions (family / shape /
    symmetry / weight) neither pool represents."""

    from omniqec_agentic import ranking
    from omniqec_agentic.memory.archive import interleave_parent_specs

    from omniqec_agentic.construction.structure import weight_profile

    inheritable = [
        dict(row)
        for row in structural_elites
        if isinstance(row.get("spec"), dict)
        and str(row.get("construct_source") or "").strip()
    ]
    structural_ordered = ranking.structural_coordinate_round_robin(inheritable)
    structural_quota = int(structural_config.structural_parents)
    runtime_quota = int(structural_config.runtime_parents)
    if str(ablation) == "no_slow":
        structural_quota += runtime_quota
        runtime_quota = 0
        hof_summary = []
    structural_parents = structural_ordered[:structural_quota]
    # v3.7: runtime parents = hall of fame by measured plpr, at most
    # RUNTIME_PARENTS_PER_WEIGHT_CLASS per check-weight class so six
    # [[48,4,6]] weight-6 tori cannot fill the whole runtime window; the
    # class-capped picks lead, the remaining hall of fame follows in plpr
    # order (it backfills the window when the diversity pool is short).
    runtime_candidates, errors = _runtime_parent_specs(hof_summary, limit=len(hof_summary))
    runtime_ordered: list[dict[str, Any]] = []
    per_weight: dict[Any, int] = {}
    for row in runtime_candidates:
        if len(runtime_ordered) >= runtime_quota:
            break
        weight = row.get("max_check_weight")
        if per_weight.get(weight, 0) >= RUNTIME_PARENTS_PER_WEIGHT_CLASS:
            continue
        per_weight[weight] = per_weight.get(weight, 0) + 1
        runtime_ordered.append(row)
    for row in runtime_candidates:
        if row not in runtime_ordered:
            runtime_ordered.append(row)
    runtime_parents = runtime_ordered[:runtime_quota]
    diversity_parents = _diversity_parent_specs(
        structural_elites=inheritable,
        hof_rows=runtime_candidates,
        chosen=structural_parents + runtime_parents,
        limit=int(getattr(structural_config, "diversity_parents", 0) or 0),
    )
    for row in structural_ordered + runtime_ordered + diversity_parents:
        row.setdefault("weight_profile", weight_profile(row.get("spec")))
    parents = interleave_parent_specs(
        structural_ordered,
        runtime_ordered,
        structural_quota=structural_quota,
        runtime_quota=runtime_quota,
        cap=int(structural_config.parent_cap),
        diversity=diversity_parents,
        diversity_quota=int(getattr(structural_config, "diversity_parents", 0) or 0),
    )
    turn_dir.mkdir(parents=True, exist_ok=True)
    (turn_dir / "parent_selection.json").write_text(
        json.dumps(
            {
                "schema": "machine_parent_selection",
                "ablation": str(ablation),
                "structural_quota": structural_quota,
                "runtime_quota": runtime_quota,
                "structural_parent_order": "unique_cell_round_robin_v2_weight",
                "runtime_parent_order": "hof_plpr_max2_per_weight_class_v1",
                "diversity_parent_order": "unrepresented_family_shape_weight_v1",
                "structural_available": len(structural_elites),
                "structural_selected": len(structural_parents),
                "structural_coordinates_selected": len(
                    {
                        ranking.structural_cell(row)
                        for row in structural_parents
                        if ranking.structural_cell(row) is not None
                    }
                ),
                "runtime_available": len(hof_summary),
                "runtime_selected": len(runtime_parents),
                "diversity_selected": len(diversity_parents),
                "render_order": [
                    {
                        "code_hash": str(parent.get("code_hash") or "")[:12],
                        "selection_channels": parent.get("selection_channels"),
                        "max_check_weight": parent.get("max_check_weight"),
                        "weight_profile": (parent.get("weight_profile") or {}).get("label"),
                    }
                    for parent in parents
                ],
                "protocol_errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return parents


def _append_structural_store(
    store_path: Path,
    rows: list[dict[str, Any]],
    known_hashes: set[str],
) -> list[str]:
    """Append rows with unseen hashes to the run-level ledger store.

    ``known_hashes`` is the state cursor and is updated in place. A crash
    between the file append and the checkpoint commit can leave duplicate
    lines; every reader deduplicates by hash (earliest line wins), so the
    append does not need to re-scan the store."""

    appended: list[str] = []
    with store_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                continue
            code_hash = str(row.get("code_hash") or "")
            if not code_hash or code_hash in known_hashes:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            known_hashes.add(code_hash)
            appended.append(code_hash)
    return appended


# v3.7.1 (2026-09-29): `max_check_weight` MUST travel with the light rows.
# `ranking._cell_dominates` compares the third axis only when BOTH cells record
# a weight, so omitting it here silently degraded the cross-turn front to the
# pre-v3.7 (k, d_search) front: every code whose only claim to the front was
# its lighter checks was dominated away on every reconcile, and the crowding
# fill lost its third axis. The reducer is weight-aware; this tuple decides
# whether it ever sees a weight.
_STRUCTURAL_LIGHT_FIELDS = (
    "code_hash", "k", "d_search", "d_proxy", "n_total", "Q_search", "Q",
    "max_check_weight",
)


def _measured_evidence_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Every slow-measured code of this campaign as compact rows for the
    v3.7 expected-plpr model: plpr, k, d_search, family, n and the check
    weight. evidence_rows.jsonl carries the measurement; the weight comes
    from the row itself when recorded, else from the run-level structural
    ledger store (streamed, only the measured hashes are kept)."""

    evidence_path = run_dir / "evidence_rows.jsonl"
    if not evidence_path.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict) or row.get("is_baseline"):
            continue
        code_hash = str(row.get("code_hash") or "")
        if not code_hash or row.get("per_logical_per_round") is None:
            continue
        compact = {
            "code_hash": code_hash,
            "per_logical_per_round": row.get("per_logical_per_round"),
            "k": row.get("k"),
            "d_search": row.get("d_search") if row.get("d_search") is not None else row.get("d_proxy"),
            "family": row.get("family"),
            "n": row.get("n") if row.get("n") is not None else row.get("n_data"),
            "max_check_weight": row.get("max_check_weight"),
            # Fig.3 Phase B: the not-competitive block names the program and
            # tests the confidence intervals; racing stops are flagged.
            "program_id": row.get("program_id"),
            "stopped_by": row.get("stopped_by"),
            "ci_low": row.get("per_logical_per_round_ci_low"),
            "ci_high": row.get("per_logical_per_round_ci_high"),
        }
        held = rows.get(code_hash)
        if held is None or float(compact["per_logical_per_round"]) < float(held["per_logical_per_round"]):
            rows[code_hash] = compact
    missing = {h for h, row in rows.items() if row.get("max_check_weight") is None}
    store_path = run_dir / "structural_ledger.jsonl"
    if missing and store_path.exists():
        with store_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                code_hash = str(row.get("code_hash") or "")
                if code_hash in missing and row.get("max_check_weight") is not None:
                    rows[code_hash]["max_check_weight"] = row.get("max_check_weight")
                    missing.discard(code_hash)
                    if not missing:
                        break
    return list(rows.values())


def _symmetry_class(spec: Any, family: Any) -> str:
    """Prompt-facing class of a code's A/B split: symmetric / asymmetric
    weights for bb and 1x1 LP, non-1x1 LP shapes, hgp."""

    from omniqec_agentic.construction.structure import structure_descriptor, weight_profile

    family = str((spec or {}).get("family") if isinstance(spec, dict) else family or "?")
    if family == "hgp":
        return "hgp"
    if not isinstance(spec, dict):
        return "unknown"
    if family in ("lp_abelian", "lp_nonabelian"):
        descriptor = structure_descriptor(spec)
        shapes = (descriptor.get("shape_a"), descriptor.get("shape_b"))
        if shapes != ("1x1", "1x1"):
            return "lp non-1x1 shapes (A or B larger than 1x1)"
    profile = weight_profile(spec)
    if profile.get("symmetric") is None:
        return "unknown"
    return "|A|=|B| (symmetric weights)" if profile["symmetric"] else "|A|!=|B| (asymmetric weights)"


def _weight_census(run_dir: Path) -> tuple[dict[Any, int], dict[Any, int]]:
    """Every candidate CONSTRUCTED this campaign, by check weight, and how many
    of them never reached the structural ledger because they sat below the
    d_search promotion floor.

    v3.7.1 (2026-09-29): the ledger-only `proposed` count of the prompt's
    weight table hides a whole class -- a weight class whose codes all fall
    below the floor renders as absent, which reads as "never tried" when the
    truth is "tried N times, none passed". Telemetry only: no selection path
    reads this, it exists so the class is visible as a fact.

    Written per cycle by funsearch (`weight_census.json`); summed here over the
    campaign's turns (and over the flat run dir for a standalone funsearch)."""

    totals: dict[Any, int] = {}
    below: dict[Any, int] = {}
    paths = sorted(run_dir.glob("turns/*/cycle/fast/weight_census.json"))
    paths.extend(sorted(run_dir.glob("weight_census.json")))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for row in payload.get("rows") or []:
            if not isinstance(row, dict):
                continue
            weight = row.get("max_check_weight")
            try:
                totals[weight] = totals.get(weight, 0) + int(row.get("constructed") or 0)
                below[weight] = below.get(weight, 0) + int(
                    row.get("below_min_promotion_distance") or 0
                )
            except (TypeError, ValueError):
                continue
    return totals, below


def _not_competitive_tables(
    evidence_rows: list[dict[str, Any]], *, ratio: float, max_rows: int
) -> dict[str, Any]:
    """Fig.3 Phase B (2026-10-12, user decision): the measured codes whose task
    error is at least ``ratio`` x the campaign's best AND whose confidence
    interval does not overlap the best's (a racing stop counts without the
    interval test: it lost to the incumbent at pilot depth), worst first, at
    most ``max_rows``; and the (check weight, d_search) classes whose best
    measurement is that far off. Pure functions of this campaign's rows."""

    import statistics

    measured = []
    for row in evidence_rows:
        try:
            plpr = float(row.get("per_logical_per_round"))
        except (TypeError, ValueError):
            continue
        if plpr <= 0.0:
            continue
        measured.append((plpr, row))
    if not measured:
        return {"not_competitive": [], "losing_classes": []}
    best_plpr, best_row = min(measured, key=lambda item: item[0])
    try:
        best_hi = float(best_row.get("ci_high")) if best_row.get("ci_high") is not None else best_plpr
    except (TypeError, ValueError):
        best_hi = best_plpr
    losers: list[dict[str, Any]] = []
    for plpr, row in measured:
        if plpr < float(ratio) * best_plpr:
            continue
        racing = str(row.get("stopped_by") or "") == "racing"
        try:
            lo = float(row.get("ci_low")) if row.get("ci_low") is not None else plpr
        except (TypeError, ValueError):
            lo = plpr
        if not racing and lo <= best_hi:
            continue
        losers.append(
            {
                "program_id": row.get("program_id"),
                "code_hash": str(row.get("code_hash") or "")[:12],
                "family": row.get("family"),
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search": row.get("d_search"),
                "max_check_weight": row.get("max_check_weight"),
                "per_logical_per_round": plpr,
                "ratio_to_best": plpr / best_plpr,
                "stopped_by": row.get("stopped_by"),
            }
        )
    losers.sort(key=lambda item: (-float(item["ratio_to_best"]), str(item["code_hash"])))
    # A class is "not competitive so far" exactly when its BEST measurement
    # is itself a named loser (same ratio + interval rule), so the class
    # verdict never says more than the code verdicts it is made of.
    loser_hashes = {str(item["code_hash"]) for item in losers}
    by_class: dict[tuple[Any, Any], list[tuple[float, dict[str, Any]]]] = {}
    for plpr, row in measured:
        by_class.setdefault((row.get("max_check_weight"), row.get("d_search")), []).append((plpr, row))
    classes = []
    for (weight, d_search), items in by_class.items():
        class_best, class_row = min(items, key=lambda item: item[0])
        values = [plpr for plpr, _ in items]
        if str(class_row.get("code_hash") or "")[:12] not in loser_hashes:
            continue
        classes.append(
            {
                "max_check_weight": weight,
                "d_search": d_search,
                "measured": len(values),
                "best_plpr": class_best,
                "median_plpr": statistics.median(values),
                "ratio_to_best": class_best / best_plpr,
            }
        )
    classes.sort(key=lambda item: (-float(item["ratio_to_best"]), str(item["max_check_weight"]), str(item["d_search"])))
    return {
        "not_competitive": losers[: max(0, int(max_rows))],
        "losing_classes": classes[: max(0, int(max_rows))],
    }


def _evidence_tables(
    run_dir: Path,
    evidence_rows: list[dict[str, Any]],
    *,
    not_competitive_ratio: float | None = None,
    not_competitive_max_rows: int = 12,
) -> dict[str, Any]:
    """v3.7 fact tables for the generator: per check-weight class and per
    A/B-symmetry class, how many distinct codes this campaign proposed
    (structurally eligible ledger rows), how many it measured, and the best /
    median measured plpr. Proposal counts stream the run-level ledger store;
    measurements are the evidence rows. Empty before the first turn."""

    import statistics

    store_path = run_dir / "structural_ledger.jsonl"
    proposed_w: dict[Any, int] = {}
    proposed_s: dict[str, int] = {}
    spec_by_hash: dict[str, dict[str, Any]] = {}
    measured_hashes = {str(row.get("code_hash") or "") for row in evidence_rows}
    if store_path.exists():
        with store_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                weight = row.get("max_check_weight")
                proposed_w[weight] = proposed_w.get(weight, 0) + 1
                klass = _symmetry_class(row.get("spec"), row.get("family"))
                proposed_s[klass] = proposed_s.get(klass, 0) + 1
                code_hash = str(row.get("code_hash") or "")
                if code_hash in measured_hashes and isinstance(row.get("spec"), dict):
                    spec_by_hash[code_hash] = {"spec": row["spec"], "family": row.get("family")}
    census_all, census_below = _weight_census(run_dir)
    if not proposed_w and not evidence_rows and not census_all:
        return {}
    measured_w: dict[Any, list[float]] = {}
    measured_s: dict[str, list[float]] = {}
    for row in evidence_rows:
        try:
            plpr = float(row.get("per_logical_per_round"))
        except (TypeError, ValueError):
            continue
        measured_w.setdefault(row.get("max_check_weight"), []).append(plpr)
        held = spec_by_hash.get(str(row.get("code_hash") or ""))
        klass = _symmetry_class(held["spec"], held["family"]) if held else "unknown"
        measured_s.setdefault(klass, []).append(plpr)

    def _table(keys: list[Any], proposed: dict[Any, int], measured: dict[Any, list[float]], label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in keys:
            values = sorted(measured.get(key) or [])
            rows.append(
                {
                    label: key,
                    "proposed": int(proposed.get(key, 0)),
                    "measured": len(values),
                    "best_plpr": values[0] if values else None,
                    "median_plpr": statistics.median(values) if values else None,
                }
            )
        return rows

    weight_keys = sorted(
        {k for k in list(proposed_w) + list(measured_w) + list(census_all) if k is not None}
    )
    symmetry_keys = sorted(set(list(proposed_s) + list(measured_s)))
    by_check_weight = _table(weight_keys, proposed_w, measured_w, "max_check_weight")
    for row in by_check_weight:
        key = row.get("max_check_weight")
        row["constructed"] = int(census_all.get(key, 0))
        row["below_distance_floor"] = int(census_below.get(key, 0))
    tables = {
        "schema": "evidence_tables_v1",
        "by_check_weight": by_check_weight,
        "by_symmetry": _table(symmetry_keys, proposed_s, measured_s, "class"),
    }
    if not_competitive_ratio is not None and float(not_competitive_ratio) > 1.0:
        tables.update(
            _not_competitive_tables(
                evidence_rows, ratio=float(not_competitive_ratio), max_rows=int(not_competitive_max_rows)
            )
        )
        tables["not_competitive_ratio"] = float(not_competitive_ratio)
    return tables


def _protocol_coordinate_cap(state: CampaignState) -> int | None:
    """StructuralSearchConfig.coordinate_cap as frozen in the protocol
    snapshot; the reducer's own default when the snapshot predates v3.6."""

    from omniqec_agentic import ranking

    retention = (state.get("protocol") or {}).get("structural_retention") or {}
    value = retention.get("coordinate_cap")
    if value is None:
        return ranking.STRUCTURAL_COORDINATE_CAP
    return int(value)


def _reduce_elites_from_store(
    store_path: Path,
    *,
    limit: int,
    coordinate_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Canonical elites = reduce(the WHOLE store), streamed in two passes.

    Pass 1 keeps only the objective fields per row (a few tens of bytes), so
    the 2-D reduction runs over light tuples; pass 2 collects the winners'
    full inheritance rows. The full store is never held in memory (scale
    audit fix #4). ``coordinate_cap`` (v3.6 B4) defaults to the reducer's
    frozen constant."""

    from omniqec_agentic import ranking

    if coordinate_cap is None:
        coordinate_cap = ranking.STRUCTURAL_COORDINATE_CAP
    if not store_path.exists():
        return []
    light: list[dict[str, Any]] = []
    with store_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                light.append({key: row.get(key) for key in _STRUCTURAL_LIGHT_FIELDS})
    reduced = ranking.reduce_structural_proxy_elites(
        light, limit=int(limit), coordinate_cap=coordinate_cap
    )
    order = {str(row.get("code_hash") or ""): index for index, row in enumerate(reduced)}
    annotations = {str(row.get("code_hash") or ""): row for row in reduced}
    winners: dict[str, dict[str, Any]] = {}
    with store_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            code_hash = str(row.get("code_hash") or "")
            if code_hash in order and code_hash not in winners:
                merged = dict(row)
                light_row = annotations[code_hash]
                for key in (
                    "pareto_rank", "crowding_distance", "crowding_boundary", "reducer_id"
                ):
                    if light_row.get(key) is not None:
                        merged[key] = light_row[key]
                winners[code_hash] = merged
    return [
        winners[code_hash]
        for code_hash, _ in sorted(order.items(), key=lambda item: item[1])
        if code_hash in winners
    ]


def _structural_ledger_from_cycle(artifact_paths: dict[str, Any]) -> list[dict[str, Any]]:
    """This turn's full structural ledger, read back from its artifact."""

    path = artifact_paths.get("structural_ledger")
    if not path or not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _structural_elites_from_cycle(artifact_paths: dict[str, Any]) -> list[dict[str, Any]]:
    """This turn's reduced structural cache, read back from its artifact."""

    path = artifact_paths.get("structural_proxy_elites")
    if not path or not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("elites") if isinstance(payload, dict) else None
    return [row for row in rows or [] if isinstance(row, dict)]


def reconcile_node(state: CampaignState) -> dict[str, Any]:
    turn = int(state["turn"])
    cycle = state.get("last_cycle") or {}
    artifact_paths = cycle.get("artifact_paths") or {}
    slow_rows = _slow_rows(artifact_paths.get("slow_rankings"))

    hof = {str(row.get("code_hash")): dict(row) for row in state.get("hof_summary") or []}
    baseline_summary = dict(state.get("baseline_summary") or {})
    for row in slow_rows:
        if row.get("status") != "ok" or row.get("per_logical_per_round") is None:
            continue
        # Fault-tolerance floor: d<3 (proxy or verified) never enters the
        # hall of fame (belt-and-suspenders behind the promotion gate).
        d_proxy = row.get("d_proxy")
        verified = row.get("verified_circuit_distance")
        if not row.get("is_baseline") and (
            (d_proxy is not None and int(d_proxy) < 3)
            or (verified is not None and int(verified) < 3)
        ):
            continue
        light = _light_slow_row(row)
        light["last_measured_turn"] = turn
        if row.get("is_baseline"):
            held_plpr = baseline_summary.get("per_logical_per_round")
            if held_plpr is None or float(light["per_logical_per_round"]) < float(held_plpr):
                baseline_summary = light
            continue
        key = str(row.get("code_hash") or row.get("program_id"))
        held = hof.get(key)
        if held is None:
            light["first_seen_turn"] = turn
            hof[key] = light
        elif float(light["per_logical_per_round"]) < float(held["per_logical_per_round"]):
            light["first_seen_turn"] = held.get("first_seen_turn", turn)
            hof[key] = light
    hof_summary = sorted(
        hof.values(), key=lambda item: float(item["per_logical_per_round"])
    )[:HOF_CAP]

    # Structural cache: this turn's reduced pool becomes the cross-turn state.
    # It is checkpointed independently of the hall of fame — a bad measurement
    # must not be able to delete a structurally strong code.
    # Scale audit fix #4: the lossless ledger is ONE append-only store at the
    # run root; the checkpoint carries only the hash cursor and the reduced
    # elites. This turn's rows are appended (hash-deduplicated), then the
    # canonical elites are re-reduced over the FULL store via streaming — the
    # invariant elites == reduce(everything ever) survives without hundreds
    # of MiB of rows riding in every checkpoint.
    known_hashes = {str(item) for item in (state.get("structural_ledger_hashes") or [])}
    store_path = Path(state["run_dir"]) / "structural_ledger.jsonl"
    _append_structural_store(
        store_path, _structural_ledger_from_cycle(artifact_paths), known_hashes
    )
    elites_limit = int(
        ((state.get("protocol") or {}).get("structural_retention") or {}).get("elites_limit")
        or 16
    )
    structural_elites = (
        _reduce_elites_from_store(
            store_path, limit=elites_limit, coordinate_cap=_protocol_coordinate_cap(state)
        )
        or _structural_elites_from_cycle(artifact_paths)
        or list(state.get("structural_proxy_elites") or [])
    )
    # Candidates that cannot be measured at all keep their structural standing
    # (the code is legal) but must stop consuming SEARCH slow slots.
    # schedule_catastrophic is permanent. v3.4 added decode_infeasible (decode
    # exceeded the frozen per-shot budget) as equally terminal; v3.6 (docs/06
    # §3 B5) keeps it out of every search queue but lets it re-enter the FINAL
    # stage once, where the 128-shard partition and the final-stage decode
    # watchdog (fewer shots per shard) apply — a large-DEM code that timed
    # out under the search budget is not thereby proven unmeasurable.
    from omniqec_agentic.verification.circuit import is_schedule_catastrophic_reason

    unmeasurable = {str(item) for item in (state.get("unmeasurable_hashes") or [])}
    catastrophic_now = {
        str(row.get("code_hash"))
        for row in slow_rows
        if row.get("code_hash")
        and (
            row.get("schedule_catastrophic")
            or row.get("status") == "schedule_catastrophic"
            or is_schedule_catastrophic_reason(row.get("failure_reason"))
        )
    }
    decode_infeasible_now = {
        str(row.get("code_hash"))
        for row in slow_rows
        if row.get("code_hash")
        and (row.get("decode_infeasible") or row.get("status") == "decode_infeasible")
    } - catastrophic_now
    # v3.6.1: the decoder cannot be built for the code's DEM -- terminal and
    # unmeasurable like catastrophic, and (unlike decode_infeasible) it does
    # NOT get the one final-stage retry: a bigger budget changes nothing.
    decoder_unsupported_now = {
        str(row.get("code_hash"))
        for row in slow_rows
        if row.get("code_hash")
        and (row.get("decoder_unsupported") or row.get("status") == "decoder_unsupported")
    } - catastrophic_now
    unmeasurable |= catastrophic_now | decode_infeasible_now | decoder_unsupported_now
    decode_infeasible = _load_decode_infeasible_hashes(state) | decode_infeasible_now
    # A code that is ALSO catastrophic or unsupported never retries.
    decode_infeasible -= catastrophic_now | decoder_unsupported_now
    _store_decode_infeasible_hashes(state, decode_infeasible)

    learning = _reconcile_learning(
        state,
        turn=turn,
        slow_rows=slow_rows,
        hof_before={str(r.get("code_hash")) for r in state.get("hof_summary") or []},
        hof_after=hof_summary,
    )
    digest = {
        "schema": "campaign_turn_digest",
        "turn": turn,
        "fast_top_k": _fast_top_k(artifact_paths.get("promotion_queue")),
        "slow_top_k": hof_summary[:8],
        # Two ledgers, side by side and neither ranking the other: structural
        # retention (k, d_search) and runtime measurement (plpr).
        "structural_frontier": _structural_frontier_rows(structural_elites),
        "baseline": baseline_summary or None,
        "cycle_status": cycle.get("status"),
        # Fig.3 v2: the slow queue's composition in numbers (slots filled,
        # candidates seen, already-measured / twin / over-weight rejections),
        # so the reflector and the planner read an empty turn as "nothing new
        # was proposed", not as a failed direction.
        "slow_queue": _slow_queue_digest(artifact_paths.get("slow_queue_composition")),
        "learning": learning,
    }
    turn_dir = _turn_dir(state, turn)
    turn_dir.mkdir(parents=True, exist_ok=True)
    digest_path = turn_dir / "digest.json"
    digest_path.write_text(
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _complete_turn_audit(state, digest_path=digest_path)
    # S3 ablation no_feedback: no reflector -- its advice is a feedback path
    # into the planner, and there is no planner. (no_slow keeps it: it then
    # reflects on the structural record only, which is the fast layer.)
    if ablation_of(state.get("protocol")) != "no_feedback":
        reflection = _run_turn_reflection(
            state, turn=turn, digest=digest, turn_dir=turn_dir, slow_rows=slow_rows
        )
    else:
        reflection = {"status": "not_applicable", "reason": "ablation no_feedback has no reflector"}
    _record_reflection_audit(state, turn=turn, reflection=reflection)

    budget = dict(state.get("budget") or {})
    budget["turns_used"] = int(budget.get("turns_used") or 0) + 1
    budget["tokens_used"] = _campaign_tokens(state, turn=turn, turn_dir=turn_dir)
    _export_run_csvs(Path(state["run_dir"]), where=f"turn {turn}")
    stop = _stop_decision(state, hof_summary, baseline_summary, budget)
    return {
        "turn": turn + 1,
        "hof_summary": hof_summary,
        "structural_proxy_elites": structural_elites,
        "structural_ledger_hashes": sorted(known_hashes),
        "unmeasurable_hashes": sorted(unmeasurable),
        # v3.6 B5: subset of unmeasurable_hashes that retries once at FINAL.
        # Also persisted to <run_dir>/decode_infeasible_hashes.json because a
        # CampaignState without this key drops it silently.
        "decode_infeasible_hashes": sorted(decode_infeasible),
        "baseline_summary": baseline_summary,
        "budget": budget,
        "stop": stop,
    }


def _export_run_csvs(run_dir: Path, *, where: str) -> None:
    """Refresh <run_dir>/progress.csv and summary.csv (2026-09-10).

    Written by the campaign itself after every turn and after the final
    report, so a monitor (hpc/fig2_dashboard.py) or a plain `tail` sees one
    new progress row per turn without anyone exporting by hand. Both files
    are rebuilt from the artifacts on disk (idempotent under resume) and are
    never allowed to fail a turn: they are views, not protocol state.
    """

    try:
        from omniqec_agentic.progress_csv import write_progress_csv
        from omniqec_agentic.summary_csv import write_summary_csv

        write_progress_csv(run_dir)
        write_summary_csv(run_dir)
    except Exception as exc:  # pragma: no cover - a view must not fail the run
        print(f"[campaign] csv export skipped ({where}): {exc}", flush=True)


def _campaign_tokens(state: CampaignState, *, turn: int, turn_dir: Path) -> int:
    """Write this turn's token aggregate and return the campaign running total.

    Rebuilt from the call records each time rather than incremented, so a
    resumed campaign cannot double-count a turn it re-entered.
    """

    from omniqec_agentic.token_usage import campaign_tokens, write_turn_usage

    try:
        write_turn_usage(turn_dir)
        return campaign_tokens(Path(state["run_dir"]))
    except Exception as exc:  # pragma: no cover - accounting must not fail a turn
        print(f"[campaign] token accounting skipped for turn {turn}: {exc}", flush=True)
        return int((state.get("budget") or {}).get("tokens_used") or 0)


def _run_turn_reflection(
    state: CampaignState,
    *,
    turn: int,
    digest: dict[str, Any],
    turn_dir: Path,
    slow_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask the model what this turn taught, and file it as advice.

    Skipped without a provider (dry-run/CI, the no-LLM baselines) and never
    allowed to raise: the measurements for this turn are already committed,
    and an advisory channel must not be able to fail a turn that succeeded.
    The call itself retries with backoff (reflector.REFLECTION_ATTEMPTS, paper
    B4: every iteration ends with a review). Returns the audit view of the
    outcome: status ok / failed / not_applicable, attempts, error.
    """

    from omniqec_agentic.orchestrator.reflector import run_reflection

    plan = state.get("plan") or {}
    provider = str(plan.get("provider") or "")
    if not provider or provider in ("dry_run", BASELINE_PROVIDER):
        return {"status": "not_applicable", "reason": f"no LLM provider ({provider or 'none'})"}
    coverage: dict[str, Any] = {}
    try:
        from omniqec_agentic.memory.coverage import coverage_report
        from omniqec_agentic.tools.context import ToolContext

        run_dir = Path(state["run_dir"])
        ctx = ToolContext(
            run_dir=run_dir, manual_dir=run_dir / "manual", state=dict(state), turn=turn
        )
        coverage = coverage_report(ctx.archive_rows, ctx.slow_rows, limit=12)
    except Exception:
        coverage = {}
    extra = _reflection_task_evidence(state)
    extra = _with_fast_screening(extra, state, turn_dir)
    try:
        record = run_reflection(
            provider=provider,
            llm_config=None,
            turn=turn,
            digest=digest,
            plan=plan,
            coverage=coverage,
            turn_dir=turn_dir,
            manual_dir=Path(state["run_dir"]) / "manual",
            slow_rows=slow_rows,
            extra=extra,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[campaign] reflection failed for turn {turn}: {exc}", flush=True)
        return {"status": "failed", "attempts": 0, "error": f"{type(exc).__name__}: {exc}"}
    outcome = {
        "status": str(record.get("status") or "failed"),
        "attempts": int(record.get("attempts") or 0),
    }
    if record.get("error"):
        outcome["error"] = str(record["error"])[:500]
        print(
            f"[campaign] reflection FAILED for turn {turn} after {outcome['attempts']} attempt(s): "
            f"{outcome['error']}",
            flush=True,
        )
    return outcome


def _with_fast_screening(
    extra: dict[str, Any] | None, state: CampaignState, turn_dir: Path
) -> dict[str, Any] | None:
    """Paper B4: the review "receives the iteration plan, the generated
    constructions, the screening results and the new circuit-level
    measurements". The digest carries the plan, the leading constructions and
    the measurements; this adds the fast-loop screening outcome of the turn
    (accepted / rejected / duplicate counts, the commonest rejection reasons
    and the produced family counts) under extra["fast_screening"]."""

    try:
        paths = (state.get("last_cycle") or {}).get("artifact_paths") or {}
        summary_path = paths.get("fast_summary")
        summary = (
            json.loads(Path(summary_path).read_text(encoding="utf-8"))
            if summary_path and Path(summary_path).is_file()
            else {}
        )
        feedback: dict[str, Any] = {}
        feedback_path = (summary.get("artifact_paths") or {}).get("feedback")
        if feedback_path and Path(feedback_path).is_file():
            feedback = json.loads(Path(feedback_path).read_text(encoding="utf-8"))
        failure_counts = dict(feedback.get("failure_counts") or {})
        screening = {
            key: summary.get(key)
            for key in (
                "seed_count", "proposal_count", "accepted_count", "structural_reject_count",
                "fast_reject_count", "duplicate_count", "archive_cell_count",
                "structural_elite_count",
            )
            if summary.get(key) is not None
        }
        if failure_counts:
            screening["rejection_reasons_top"] = dict(
                sorted(failure_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:8]
            )
        produced = _produced_family_counts(turn_dir)
        if produced:
            screening["produced_family_counts"] = produced
    except Exception:
        return extra
    if not screening:
        return extra
    merged = dict(extra or {})
    merged["fast_screening"] = screening
    return merged


def _record_reflection_audit(state: CampaignState, *, turn: int, reflection: dict[str, Any]) -> None:
    """Make the iteration review's outcome visible in the strict turn audit
    (turn_audit.json key ``reflection``); the campaign audit summary lists the
    turns whose review failed after every retry. Never fails the turn."""

    if not bool((state.get("protocol") or {}).get("strict_turns")):
        return
    audit_path = _turn_dir(state, turn) / "turn_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["reflection"] = dict(reflection or {"status": "unrecorded"})
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - an audit annotation must not fail a turn
        print(f"[campaign] reflection audit note skipped for turn {turn}: {exc}", flush=True)


def _reflection_task_evidence(state: CampaignState) -> dict[str, Any] | None:
    """Fig.3 Phase B: the reflector sees the same not-competitive tables as
    the generator (so its notes can cite them); None under the Fig.2 profile."""

    try:
        from omniqec_agentic.evaluation import verification_from_dict

        verification = verification_from_dict((state.get("protocol") or {}).get("verification"))
        ratio = getattr(verification.structural, "not_competitive_ratio", None)
        if not ratio:
            return None
        run_dir = Path(state["run_dir"])
        tables = _evidence_tables(
            run_dir, _measured_evidence_rows(run_dir),
            not_competitive_ratio=float(ratio),
            not_competitive_max_rows=int(getattr(verification.structural, "not_competitive_max_rows", 12)),
        )
        return {
            "task": str(getattr(verification, "task", "memory")),
            "not_competitive_ratio": float(ratio),
            "not_competitive": list(tables.get("not_competitive") or []),
            "losing_classes": list(tables.get("losing_classes") or []),
        }
    except Exception:
        return None


def _structure_by_hash(state: CampaignState) -> dict[str, dict[str, Any]]:
    """code_hash -> structural descriptor, read from this turn's archive queue."""

    from omniqec_agentic.construction.structure import structure_descriptor

    paths = (state.get("last_cycle") or {}).get("artifact_paths") or {}
    queue = paths.get("all_archive_queue") or paths.get("promotion_queue")
    if not queue or not Path(queue).exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in Path(queue).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        code_hash = str(row.get("code_hash") or "")
        if code_hash:
            out[code_hash] = structure_descriptor(row.get("spec"))
    return out


def _reconcile_learning(
    state: CampaignState,
    *,
    turn: int,
    slow_rows: list[dict[str, Any]],
    hof_before: set[str],
    hof_after: list[dict[str, Any]],
) -> dict[str, Any]:
    """Learning layer at reconcile (03 §6): accumulate gen-stamped evidence
    rows, run the prospective backtest, credit the strategy bandit from
    hypothesis tags, and record frontier programs as skills."""

    from omniqec_agentic.memory.backtester import insight_precision, run_backtest
    from omniqec_agentic.memory.bandit import (
        StrategyBandit,
        parse_hypothesis_tags,
        turn_rewards_from_tags,
    )
    from omniqec_agentic.memory.ledger import Ledger
    from omniqec_agentic.orchestrator.manual import record_skill

    run_dir = Path(state["run_dir"])
    evidence_path = run_dir / "evidence_rows.jsonl"
    fields = (
        "code_hash", "family", "n", "k", "d_proxy", "fast_Q",
        "per_logical_per_round", "d_circ_ext", "degraded_ext", "degraded_isd",
        "verified_circuit_distance", "stopped_by", "noise_profile", "is_baseline",
        # v3.7.1: ledger.PREDICATE_FIELDS has listed max_check_weight since
        # v3.7, but the column never reached the rows, so the backtester could
        # never adjudicate a weight claim. It travels with the evidence now.
        "max_check_weight", "max_qubit_degree",
    )
    # Structural coordinates travel WITH the evidence: a hypothesis may only be
    # adjudicated over columns the rows carry, so without these the backtester
    # can never rule on a structure-level regularity. Slow rows carry no spec,
    # so they are joined to the fast archive queue by code_hash.
    structure_by_hash = _structure_by_hash(state)
    all_rows: list[dict[str, Any]] = []
    if evidence_path.exists():
        all_rows = [
            json.loads(line)
            for line in evidence_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    # Crash-resume idempotency: the append lands on disk before the LangGraph
    # checkpoint commits, so a re-run of an interrupted reconcile would write
    # this turn's gen-stamped rows a second time — duplicated rows double-count
    # in the backtester bootstrap. Skip the append when the gen already exists.
    if not any(int(r.get("gen") or -1) == turn for r in all_rows):
        with evidence_path.open("a", encoding="utf-8") as handle:
            for row in slow_rows:
                if row.get("status") != "ok" or row.get("per_logical_per_round") is None:
                    continue
                light = {key: row.get(key) for key in fields}
                light.update(structure_by_hash.get(str(row.get("code_hash") or ""), {}))
                light["gen"] = turn
                handle.write(json.dumps(light, ensure_ascii=False, sort_keys=True) + "\n")
                all_rows.append(light)
    ledger = Ledger(run_dir / "ledger.json")
    backtest = run_backtest(ledger, [r for r in all_rows if not r.get("is_baseline")])
    # Endogenous learning metric (03 D7 §6.2): re-validate supported rules on
    # this turn's rows as the held-out generation (no extra runs needed).
    holdout = [
        r for r in all_rows
        if int(r.get("gen") or -1) == turn and not r.get("is_baseline")
    ]
    precision = insight_precision(ledger, holdout)

    new_hof_hashes = {
        str(r.get("code_hash")) for r in hof_after if str(r.get("code_hash")) not in hof_before
    }
    accepted_programs: list[dict[str, Any]] = []
    for row in hof_after:
        if str(row.get("code_hash")) not in new_hof_hashes:
            continue
        payload = _candidate_payload_from_rows(slow_rows, str(row.get("code_hash")))
        promotion = (payload or {}).get("promotion_record") or {}
        source = promotion.get("construct_source") or ""
        tags = parse_hypothesis_tags(source)
        accepted_programs.append({"code_hash": row.get("code_hash"), **tags})
        if source:
            record_skill(
                run_dir / "manual",
                program_id=str(row.get("program_id")),
                construct_source=source,
                evidence={
                    "code_hash": row.get("code_hash"),
                    "family": row.get("family"),
                    "per_logical_per_round": row.get("per_logical_per_round"),
                    "per_logical_block_ci": row.get("per_logical_block_ci"),
                    "turn": turn,
                },
            )
    bandit = StrategyBandit(run_dir / "bandit.json")
    rewards = turn_rewards_from_tags(accepted_programs, new_hof_hashes)
    # gen= makes the persisted update idempotent under crash-resume.
    bandit.update(rewards, gen=turn)
    return {
        "backtest": backtest,
        "insight_precision": precision,
        "new_hof": sorted(h[:8] for h in new_hof_hashes),
        "bandit_rewards": rewards,
        "evidence_rows_total": len(all_rows),
    }


def _candidate_payload_from_rows(
    slow_rows: list[dict[str, Any]], code_hash: str
) -> dict[str, Any] | None:
    for row in slow_rows:
        if str(row.get("code_hash")) == code_hash:
            path = row.get("candidate_path")
            if path and Path(path).exists():
                try:
                    return json.loads(Path(path).read_text(encoding="utf-8"))
                except Exception:
                    return None
    return None


def _stop_decision(
    state: CampaignState,
    hof_summary: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    protocol = state["protocol"]
    # Early finalization DISABLED (user decision 2026-08-17): the Fig.2
    # frontier needs the full iteration curve, so neither the agent's plan
    # finalize nor a beats-baseline CI separation ends the search early —
    # the campaign always runs to max_turns. (b144 v2 hit a CI-separated
    # BB-beating code at turn 2; under the old rule that closed the search
    # after 3 of 60 turns.) The beating evidence still reaches the report
    # via the hall of fame and final_slow.
    if int(budget.get("turns_used") or 0) >= int(protocol["max_turns"]):
        return {"finalize": True, "reason": "max_turns"}
    return {"finalize": False}


def continue_or_finalize(state: CampaignState) -> str:
    return "finalize" if (state.get("stop") or {}).get("finalize") else "continue"


def final_rng_base_seed(namespace: str) -> int:
    """Deterministic, search-disjoint seed base for the final audit (§4.6).

    Search candidates derive seeds from base 0 (index * 1e6); deriving the
    final stage from a namespace-hashed base >= 1e12 makes "fresh shots" an
    auditable fact rather than a claim, since no final sampling key can
    collide with a search one."""

    import hashlib

    digest = hashlib.sha256(str(namespace).encode("utf-8")).hexdigest()
    return 10**12 + int(digest[:12], 16) % 10**9


DECODE_INFEASIBLE_HASHES_FILE = "decode_infeasible_hashes.json"


def _decode_infeasible_path(state: CampaignState) -> Path | None:
    run_dir = state.get("run_dir") if isinstance(state, dict) else None
    if not run_dir:
        return None
    return Path(str(run_dir)) / DECODE_INFEASIBLE_HASHES_FILE


def _structural_ledger_canonical(run_dir: Path, hashes: set[str]) -> dict[str, str]:
    """code_hash -> canonical (pynauty) hash from structural_ledger.jsonl, for the
    given hashes only (streamed; the store is never loaded whole)."""

    out: dict[str, str] = {}
    path = run_dir / "structural_ledger.jsonl"
    if not path.exists() or not hashes:
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            code_hash = str(row.get("code_hash") or "")
            canonical = row.get("canonical_hash")
            if code_hash in hashes and canonical and code_hash not in out:
                out[code_hash] = str(canonical)
                if len(out) == len(hashes):
                    break
    return out


def _measured_identities(run_dir: Path, evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fig.3 v2: every measured code of the campaign with both identities (code
    hash, canonical hash), its program id and its task error, so the generator
    prompt can say "already measured as X" / "isomorphic twin of X"."""

    wanted = {str(row.get("code_hash") or "") for row in evidence_rows if row.get("code_hash")}
    canonical = _structural_ledger_canonical(run_dir, wanted)
    out: list[dict[str, Any]] = []
    for row in evidence_rows:
        code_hash = str(row.get("code_hash") or "")
        if not code_hash:
            continue
        out.append(
            {
                "code_hash": code_hash,
                "canonical_hash": canonical.get(code_hash),
                "program_id": row.get("program_id"),
                "per_logical_per_round": row.get("per_logical_per_round"),
                "family": row.get("family"),
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search": row.get("d_search"),
                "max_check_weight": row.get("max_check_weight"),
            }
        )
    return out


def _slow_queue_digest(composition_path: str | None) -> dict[str, Any] | None:
    """Fig.3 v2: the counts of a turn's slow_queue_composition.json."""

    if not composition_path or not Path(composition_path).exists():
        return None
    try:
        comp = json.loads(Path(composition_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    gate = comp.get("weight_gate") or {}
    gated = [row for row in list(gate.get("rejected") or []) if isinstance(row, dict)]
    return {
        "cap": comp.get("cap"),
        "selected": len(comp.get("selected") or []),
        "candidates_seen": comp.get("candidates_seen"),
        "already_measured": len(comp.get("measured_rejected_rows") or []),
        "isomorphic_twins_of_measured": len(comp.get("twin_rejected_rows") or []),
        "over_check_weight_gate": len(gated),
        "check_weight_gate": gate.get("max_check_weight"),
        "structural_slot_shortfall": comp.get("structural_slot_shortfall"),
    }


def _last_turn_queue_summary(
    state: CampaignState, turn: int, identities: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Fig.3 v2: what the previous turn's slow queue did, with the rejected
    twins / measured re-proposals resolved to the measured code they duplicate."""

    if int(turn) < 1:
        return None
    path = _turn_dir(state, int(turn) - 1) / "cycle" / "slow_queue_composition.json"
    if not path.exists():
        return None
    try:
        comp = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    by_prefix = {str(item.get("code_hash") or "")[:12]: item for item in identities}

    def _resolve(prefix: str | None) -> dict[str, Any]:
        item = by_prefix.get(str(prefix or "")[:12])
        if not item:
            return {}
        return {
            "program_id": item.get("program_id"),
            "per_logical_per_round": item.get("per_logical_per_round"),
        }

    twins = [
        dict(row, twin=_resolve(row.get("twin_of_code_hash")))
        for row in list(comp.get("twin_rejected_rows") or [])
        if isinstance(row, dict)
    ]
    measured = [
        dict(row, measured=_resolve(row.get("code_hash")))
        for row in list(comp.get("measured_rejected_rows") or [])
        if isinstance(row, dict)
    ]
    gate = comp.get("weight_gate") or {}
    gated = [row for row in list(gate.get("rejected") or []) if isinstance(row, dict)]
    return {
        "turn": int(turn) - 1,
        "cap": comp.get("cap"),
        "selected": len(comp.get("selected") or []),
        "candidates_seen": comp.get("candidates_seen"),
        "twin_rejected": twins[:8],
        "twin_rejected_count": len(twins),
        "measured_rejected": measured[:8],
        "measured_rejected_count": len(measured),
        "weight_gate_max": gate.get("max_check_weight"),
        "weight_gate_rejected_count": len(gated),
        "weight_gate_weights": sorted(
            int(row.get("max_check_weight")) for row in gated if row.get("max_check_weight") is not None
        ),
    }


def _structural_ledger_weights(run_dir: Path) -> dict[str, int]:
    """code_hash -> max check weight from structural_ledger.jsonl (every structural
    row records it; the proxy-elite rows in the state do not)."""

    weights: dict[str, int] = {}
    path = run_dir / "structural_ledger.jsonl"
    if not path.exists():
        return weights
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        code_hash = row.get("code_hash")
        weight = row.get("max_check_weight")
        if code_hash and weight is not None:
            weights.setdefault(str(code_hash), int(weight))
    return weights


def _load_decode_infeasible_hashes(state: CampaignState) -> set[str]:
    """decode_infeasible hashes (v3.6 B5): the state key when the checkpoint
    carries it, else the run-level JSON file reconcile writes alongside."""

    hashes = {str(item) for item in (state.get("decode_infeasible_hashes") or []) if item}
    path = _decode_infeasible_path(state)
    if path is not None and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for item in (payload.get("hashes") if isinstance(payload, dict) else payload) or []:
            if item:
                hashes.add(str(item))
    return hashes


def _store_decode_infeasible_hashes(state: CampaignState, hashes: set[str]) -> None:
    path = _decode_infeasible_path(state)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "decode_infeasible_hashes_v1",
                    "note": (
                        "search-stage decode_infeasible candidates; excluded from "
                        "every search slow queue, eligible ONCE for the final stage "
                        "with final-stage budgets (docs/06 §3 B5)"
                    ),
                    "hashes": sorted(hashes),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _first_measured_turns(run_dir: Path, *, upto: int) -> dict[str, int]:
    """code_hash -> the first turn < ``upto`` whose slow stage measured it
    (an ``ok`` row in that turn's slow_results.csv)."""

    import csv

    csv.field_size_limit(2**31 - 1)
    first: dict[str, int] = {}
    for turn in range(int(upto)):
        path = run_dir / "turns" / f"turn_{turn:04d}" / "cycle" / "slow" / "slow_results.csv"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if str(row.get("status") or "") != "ok":
                        continue
                    code_hash = str(row.get("code_hash") or "")
                    if code_hash and code_hash not in first:
                        first[code_hash] = turn
        except OSError:
            continue
    return first


def _final_entries(
    state: CampaignState, structural_config: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """final_structural + final_runtime finalists (2026-09-10: 4 + 2; was 3 + 1),
    deduplicated with backfill (§4.6)."""

    from omniqec_agentic import ranking

    # v3.2 §4.4/§4.6: an unmeasurable candidate keeps its structural standing
    # but cannot occupy one of the FROZEN finalist slots when its fixed-T
    # measurement cannot exist — schedule_catastrophic, permanently. v3.6
    # (docs/06 §3 B5): a SEARCH-stage decode_infeasible candidate is not
    # proven unmeasurable; it is eligible once more here, flagged
    # `retry_at_final` so the pool applies the final-stage decode budget.
    unmeasurable = {str(item) for item in (state.get("unmeasurable_hashes") or [])}
    retry_at_final = _load_decode_infeasible_hashes(state)
    excluded = unmeasurable - retry_at_final
    # max_turns lowered on resume (2026-09-05): every point finishes on the
    # codes it knew by the new bound. A structural elite first seen at or
    # after max_turns, and a measured code whose first measurement lies at
    # or after it, cannot take a finalist slot (the turns beyond the bound
    # stay on disk for the record).
    protocol = state.get("protocol") or {}
    cutoff = int(protocol.get("max_turns") or 0) if original_max_turns(protocol) is not None else None
    measured_by_cutoff = (
        _first_measured_turns(Path(state["run_dir"]), upto=cutoff) if cutoff is not None else None
    )

    def _late(row: dict[str, Any], *, elite: bool) -> bool:
        if cutoff is None:
            return False
        if elite:
            first = row.get("first_seen_turn")
            return first is not None and int(first) >= cutoff
        return str(row.get("code_hash") or "") not in (measured_by_cutoff or {})

    # 2026-10-14: the promotion weight gate (runtime key
    # promotion_max_check_weight) also bounds the structural finalist slots.
    # The proxy frontier keeps every structurally valid code, so its (k, d)
    # endpoints are often w8-w24 codes the gate kept out of the slow loop; the
    # b144 DeepSeek Phase B final pulled a w12 and a w24 lp_abelian code it had
    # never measured and sat on them for hours at the fixed final shots. A
    # final can only measure what the search was allowed to measure.
    gate = protocol.get("promotion_max_check_weight")
    # Fig.4 c (2026-10-23): the exact-weight contract bounds the finalists too.
    exact = protocol.get("exact_check_weight")
    weights = (
        _structural_ledger_weights(Path(state["run_dir"]))
        if gate is not None or exact is not None
        else {}
    )
    over_gate: list[str] = []

    def _over_gate(row: dict[str, Any]) -> bool:
        if gate is None and exact is None:
            return False
        weight = weights.get(str(row.get("code_hash") or ""))
        if weight is None:
            return False
        if (gate is not None and int(weight) > int(gate)) or (
            exact is not None and int(weight) != int(exact)
        ):
            over_gate.append(str(row.get("code_hash") or "")[:12])
            return True
        return False

    structural_elites = [
        row
        for row in (state.get("structural_proxy_elites") or [])
        if isinstance(row.get("spec"), dict)
        and str(row.get("code_hash") or "") not in excluded
        and not _late(row, elite=True)
        and not _over_gate(row)
    ]
    hof = [
        row
        for row in (state.get("hof_summary") or [])
        if isinstance(row.get("spec"), dict)
        and str(row.get("code_hash") or "") not in excluded
        and not _late(row, elite=False)
    ]
    entries: list[dict[str, Any]] = []
    used: set[str] = set()

    def _take(row: dict[str, Any] | None, channel: str) -> bool:
        if row is None:
            return False
        key = str(row.get("code_hash") or row.get("program_id") or "")
        if not key:
            return False
        if key in used:
            for held in entries:
                if str(held.get("code_hash") or held.get("program_id") or "") == key:
                    channels = list(held.get("selection_channels") or [])
                    if channel not in channels:
                        channels.append(channel)
                    held["selection_channels"] = channels
            return False
        used.add(key)
        item = dict(row)
        item["selection_channels"] = [channel]
        if str(row.get("code_hash") or "") in retry_at_final:
            item["retry_at_final"] = True
        entries.append(item)
        return True

    # S3 ablation (2026-10-23): nothing was measured during the search, so
    # the runtime finalist slots do not exist; the whole frozen shortlist
    # (final_structural + final_runtime = 6) is taken from the structural
    # front in the structural channel's own slot order (high-k endpoint,
    # high-d_search endpoint, the lightest cell when the task profile has that
    # slot, then crowding representatives over unique cells).
    ablation = ablation_of(protocol)
    final_structural_slots = int(structural_config.final_structural)
    final_runtime_slots = int(structural_config.final_runtime)
    # v3.8: the final endpoints follow the same family rules as the turns
    # (distinct families; demoted families stay out), judged on the whole run.
    final_demoted: list[str] = []
    if int(getattr(structural_config, "endpoint_demotion_strikes", 0) or 0) > 0 and state.get("run_dir"):
        final_demoted = list(
            _endpoint_demotion(
                Path(str(state["run_dir"])),
                before_turn=10**6,
                strikes=int(structural_config.endpoint_demotion_strikes),
                factor=float(structural_config.endpoint_demotion_factor),
                reference_plpr=_best_measured_plpr(hof, dict(state.get("baseline_summary") or {})),
            )["demoted_families"]
        )
    # The v3.8 keyword arguments are passed only when a rule is on, so the
    # v3.7 call is literally the v3.7 call (tests stub the function with the
    # two-argument signature).
    endpoint_kwargs: dict[str, Any] = {}
    if bool(getattr(structural_config, "endpoint_family_distinct", False)):
        endpoint_kwargs["distinct_families"] = True
    if final_demoted:
        endpoint_kwargs["demoted_families"] = final_demoted
    search_method = search_method_of(protocol)
    if search_method != "llm":
        # 2026-11-06 search baselines: the shortlist is the run's measured ranking (the hall of fame, the
        # Full search's runtime-finalist order), all six slots; the structural front plays no part.
        final_runtime_slots += final_structural_slots
        final_structural_slots = 0
        structural_elites = []
        structural_picks = []
    elif ablation != "full":
        final_structural_slots += final_runtime_slots
        final_runtime_slots = 0
        hof = []
        structural_picks = ranking.select_structural_slots(
            structural_elites,
            slots=final_structural_slots,
            lightest=bool(getattr(structural_config, "lightest_endpoint_slot", False)),
            **endpoint_kwargs,
        )
    else:
        structural_picks = ranking.select_structural_slots(
            structural_elites, slots=final_structural_slots, **endpoint_kwargs
        )
    for row in structural_picks:
        _take(row, "structural")
    # The runtime slot scans the WHOLE hall of fame until one candidate is
    # actually added (audit fix #3): trying only hof[:1] meant that whenever
    # the runtime champion was already a structural pick, the fourth slot was
    # backfilled from the structural pool — 4S+0R instead of 3S+1R, with no
    # independent runtime finalist at all.
    runtime_taken = 0
    for row in hof:
        if runtime_taken >= final_runtime_slots:
            break
        if _take(row, "runtime"):
            runtime_taken += 1
    # Backfill so the audit still measures the frozen number of finalists when
    # one pool is short or the two channels picked the same code.
    target = final_structural_slots + final_runtime_slots
    for row in structural_elites + hof:
        if len(entries) >= target:
            break
        _take(row, "structural" if row in structural_elites else "runtime")
    composition = {
        "schema": "final_shortlist_composition",
        "ablation": ablation,
        **({"search_method": search_method} if search_method != "llm" else {}),
        "structural_slots": final_structural_slots,
        "runtime_slots": final_runtime_slots,
        "structural_available": len(structural_elites),
        "runtime_available": len(hof),
        "unmeasurable_excluded": len(excluded),
        "weight_gate": None if gate is None else int(gate),
        **({"exact_check_weight": int(exact)} if exact is not None else {}),
        "elites_over_weight_gate": over_gate,
        "decode_infeasible_retry_eligible": len(retry_at_final),
        "retry_at_final": [
            str(row.get("code_hash") or "")[:12] for row in entries if row.get("retry_at_final")
        ],
        "selected": [
            {
                "code_hash": str(row.get("code_hash") or "")[:12],
                "program_id": row.get("program_id"),
                "k": row.get("k"),
                "d_search": row.get("d_search") or row.get("d_proxy"),
                "selection_channels": row.get("selection_channels"),
                "retry_at_final": bool(row.get("retry_at_final")),
            }
            for row in entries
        ],
    }
    return entries, composition


def final_slow_node(state: CampaignState) -> dict[str, Any]:
    """The final stage. A run whose protocol snapshot carries one of the paper's final-stage blocks
    (``final_stage.id`` paper_top5_cumulative_v1, or paper_fig4c_top6_confirm_v1 under an exact check-weight
    contract; every run launched by this code) runs ``_paper_final_slow_node``; a snapshot without the block
    (a run frozen by an earlier version, resumed or re-entered for its record) runs the final stage it was
    frozen with."""

    from omniqec_agentic.evaluation.final_stage import is_paper_final_stage

    if is_paper_final_stage(state.get("protocol")):
        return _paper_final_slow_node(state)
    return _legacy_final_slow_node(state)


def _paper_final_entries(state: CampaignState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The finalists of the paper's final stage (Appendix D3) and the applicable reference row.

    Finalists: the hall of fame (every successfully measured non-reference code of the run, status ok,
    racing early stops included) ranked by search-stage pLER, ties by the smaller upper bound, the first
    ``final_stage.top`` canonically distinct codes (5; 6 in the Fig. 4 c fixed-weight stage). The exclusions
    of the earlier final stage apply unchanged: unmeasurable codes and, after a resume that lowered
    max_turns, codes first measured at or past the new bound. A no-slow ablation measured nothing in the search, so its finalists are the first
    five distinct codes of the structural front in the structural channel's slot order."""

    from omniqec_agentic import ranking
    from omniqec_agentic.evaluation import final_stage as fs
    from omniqec_agentic.evaluation.baseline import resident_baseline_parameters

    protocol = state.get("protocol") or {}
    block = dict(protocol.get("final_stage") or {})
    top = int(block.get("top") or 5)
    run_dir = Path(state["run_dir"])
    unmeasurable = {str(item) for item in (state.get("unmeasurable_hashes") or [])}
    cutoff = int(protocol.get("max_turns") or 0) if original_max_turns(protocol) is not None else None
    measured_by_cutoff = _first_measured_turns(run_dir, upto=cutoff) if cutoff is not None else None

    def _late(row: dict[str, Any], *, elite: bool) -> bool:
        if cutoff is None:
            return False
        if elite:
            first = row.get("first_seen_turn")
            return first is not None and int(first) >= cutoff
        return str(row.get("code_hash") or "") not in (measured_by_cutoff or {})

    ablation = ablation_of(protocol)
    search_method = search_method_of(protocol)
    over_gate: list[str] = []
    if ablation != "full":
        gate = protocol.get("promotion_max_check_weight")
        exact = protocol.get("exact_check_weight")
        weights = _structural_ledger_weights(run_dir) if gate is not None or exact is not None else {}

        def _over_gate(row: dict[str, Any]) -> bool:
            weight = weights.get(str(row.get("code_hash") or ""))
            if weight is None:
                return False
            if (gate is not None and int(weight) > int(gate)) or (exact is not None and int(weight) != int(exact)):
                over_gate.append(str(row.get("code_hash") or "")[:12])
                return True
            return False

        pool = [
            row
            for row in (state.get("structural_proxy_elites") or [])
            if isinstance(row.get("spec"), dict)
            and str(row.get("code_hash") or "") not in unmeasurable
            and not _late(row, elite=True)
            and not _over_gate(row)
        ]
        structural_config = _verification_from_protocol(protocol, state.get("spec") or {}).structural
        # the earlier final stage's slot order for an ablation, unchanged (v3.8: distinct-family endpoints)
        endpoint_kwargs = (
            {"distinct_families": True} if bool(getattr(structural_config, "endpoint_family_distinct", False)) else {}
        )
        picks = ranking.select_structural_slots(
            pool, slots=top, lightest=bool(getattr(structural_config, "lightest_endpoint_slot", False)), **endpoint_kwargs
        )
        ordered = list(picks) + [row for row in pool if row not in picks]
        source = "structural_front_slot_order"
        ranked = False
    else:
        ordered = [
            row
            for row in (state.get("hof_summary") or [])
            if isinstance(row.get("spec"), dict)
            and not row.get("is_baseline")
            and row.get("per_logical_per_round") is not None
            and str(row.get("code_hash") or "") not in unmeasurable
            and not _late(row, elite=False)
        ]
        source = "search_measured_ranking"
        ranked = True
    canonical_of = _structural_ledger_canonical(
        run_dir, {str(row.get("code_hash") or "") for row in ordered if row.get("code_hash")}
    )
    finalists, skipped = fs.select_finalists(ordered, top=top, canonical_of=canonical_of, ranked=ranked)
    entries: list[dict[str, Any]] = []
    for rank, row in enumerate(finalists, start=1):
        code_hash = str(row.get("code_hash") or "")
        search_program = str(row.get("program_id") or f"finalist_{rank}")
        item = dict(row)
        item.update(
            # program ids restart every turn, so two finalists can share one: the final queue gives each its
            # own (candidate files and per-candidate resume are keyed by it).
            program_id=f"{search_program}_h{code_hash[:8]}" if code_hash else search_program,
            search_program_id=search_program,
            search_rank=rank,
            search_per_logical_per_round=row.get("per_logical_per_round"),
            search_per_logical_per_round_ci=row.get("per_logical_per_round_ci"),
            canonical_hash=canonical_of.get(code_hash) or row.get("canonical_hash"),
            selection_channels=["top_search" if ranked else "top_structural"],
            is_baseline=False,
        )
        entries.append(item)
    reference_block = block.get("reference") or None
    reference_row = None
    if reference_block:
        reference = fs.resident_reference(int(block.get("n_budget") or (state.get("spec") or {}).get("qubit_budget")))
        if reference is not None:
            shared = [item for item in entries if str(item.get("code_hash") or "") == reference["code_hash"]]
            if shared:
                # The generator's own copy of the reference is a finalist: it is measured once, and the
                # reference row pools the prefix of its parts that reaches the reference total
                # (evaluation.final_stage.plan_final_stage).
                shared[0]["also_reference"] = True
                reference_row = "shared_with_finalist"
            else:
                n, k, d = resident_baseline_parameters(reference["program_id"])
                entries.append(
                    {
                        "program_id": reference["program_id"],
                        "search_program_id": reference["program_id"],
                        "spec": reference["spec"],
                        "code_hash": reference["code_hash"],
                        "family": "bb",
                        "n": n,
                        "k": k,
                        "d_search": d,
                        "is_baseline": True,
                        "selection_channels": ["baseline"],
                    }
                )
                reference_row = "measured"
    composition = {
        "schema": fs.COMPOSITION_SCHEMA,
        "final_stage_id": block.get("id"),
        "selection": block.get("selection"),
        "selection_source": source,
        "ablation": ablation,
        **({"search_method": search_method} if search_method != "llm" else {}),
        "top": top,
        "available": len(ordered),
        "unmeasurable_excluded": len(unmeasurable),
        "skipped_duplicates": skipped,
        **({"elites_over_weight_gate": over_gate} if over_gate else {}),
        "reference": None if not reference_block else {**dict(reference_block), "row": reference_row},
        "selected": [
            {
                "code_hash": str(row.get("code_hash") or "")[:12],
                "canonical_hash": str(row.get("canonical_hash") or "")[:12] or None,
                "program_id": row.get("program_id"),
                "search_program_id": row.get("search_program_id"),
                "search_rank": row.get("search_rank"),
                "search_per_logical_per_round": row.get("search_per_logical_per_round"),
                "search_per_logical_per_round_ci": row.get("search_per_logical_per_round_ci"),
                "k": row.get("k"),
                "d_search": row.get("d_search") or row.get("d_proxy"),
                "also_reference": bool(row.get("also_reference")),
            }
            for row in entries
            if not row.get("is_baseline")
        ],
    }
    return entries, composition


def _paper_final_queue_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "program_id": row["program_id"],
        "search_program_id": row.get("search_program_id"),
        "spec": row["spec"],
        "code_hash": row.get("code_hash"),
        "canonical_hash": row.get("canonical_hash"),
        "selection_channels": row.get("selection_channels"),
        # writeback derives is_baseline from this queue record (see the earlier final stage).
        "is_baseline": bool(row.get("is_baseline")),
        "search_rank": row.get("search_rank"),
        "also_reference": bool(row.get("also_reference")),
    }


def _paper_final_slow_node(state: CampaignState) -> dict[str, Any]:
    """The paper's final stage (Appendix D3, Table VII): the five leading distinct codes of the search
    are confirmed with 25 000 fresh shots per basis and topped up to the cumulative limit of the run's
    (task, N); the applicable BB reference code gets the same confirmation and a top-up to its paper total
    (50 000 / 100 000 shots per basis). The pooled pLER ranks the finalists and names the champion
    (final_slow/final_ranking.json). Under an exact check-weight contract (Fig. 4 c) the six leading
    distinct codes get the confirmation only, and the ranking output names the three best."""

    from omniqec_agentic.evaluation import final_stage as fs

    protocol = state["protocol"]
    block = dict(protocol["final_stage"])
    turn_audit = _validate_all_turn_audits(state)
    spec = state["spec"]
    run_dir = Path(state["run_dir"])
    final_dir = run_dir / "final_slow"
    final_dir.mkdir(parents=True, exist_ok=True)
    verification_template = _verification_from_protocol(protocol, spec)
    rounds_policy = protocol.get("rounds_policy") or {}
    final_rounds = _protocol_final_rounds(protocol)
    namespace = str(
        rounds_policy.get("final_rng_namespace") or verification_template.structural.final_rng_namespace
    )
    if int(block.get("confirm_shots") or 0) != int(protocol["final_shots"]):
        raise RuntimeError(
            f"final_stage.confirm_shots={block.get('confirm_shots')!r} disagrees with final_shots="
            f"{protocol['final_shots']!r}"
        )
    entries, composition = _paper_final_entries(state)
    (final_dir / "final_shortlist.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not [row for row in entries if not row.get("is_baseline")]:
        if bool(protocol.get("strict_turns")):
            raise RuntimeError(
                "strict campaign cannot finish with an empty finalist shortlist; "
                "resume final_slow after producing a measurable candidate"
            )
        return {"final_slow": {"status": "empty_shortlist", "runs": [], "final_stage_id": block.get("id")}}

    p_values = tuple(float(value) for value in protocol["final_p_values"])
    final_cap = protocol.get("final_decode_shard_timeout_cap_s")
    if final_cap is not None and float(final_cap) > 0.0:
        print(
            f"[campaign] final stage: decode-shard watchdog capped at {float(final_cap):.0f} s "
            f"(final_decode_shard_timeout_cap_s)",
            flush=True,
        )

    def _measure(part: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any]:
        queue_path = final_dir / str(part["queue"])
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(
            "\n".join(json.dumps(_paper_final_queue_record(row)) for row in members) + "\n", encoding="utf-8"
        )
        verification = _verification_from_protocol(
            protocol, spec, p_values=p_values, shots=int(part["shots"]), rounds=final_rounds
        )
        if final_cap is not None and float(final_cap) > 0.0:
            verification = dataclasses.replace(verification, final_decode_shard_timeout_cap_s=float(final_cap))
        print(
            f"[campaign] final stage part {part['name']}: {len(members)} code(s) x {int(part['shots'])} shots "
            f"per basis in {int(part['shards'])} shards (seed namespace {part['rng_namespace']})",
            flush=True,
        )
        return evaluate_promotion_queue(
            queue_path=queue_path,
            out_dir=final_dir / str(part["dir"]),
            verification=verification,
            include_baseline=False,
            resume_candidates=True,
            workers=int(protocol.get("slow_workers") or 0),
            shot_shards=int(part["shards"]),
            seed=final_rng_base_seed(str(part["rng_namespace"])),
            stage="final",
        )

    result = fs.execute_final_stage(
        block=block,
        entries=entries,
        rounds=final_rounds,
        confirm_namespace=namespace,
        confirm_shards=int(protocol.get("final_shot_shards") or protocol["shot_shards"]),
        measure=_measure,
        ci_z=float(verification_template.ler_ci_z),
    )
    status = "ok" if result["status"] == "ok" else "partial"
    rows = list(result.get("rows") or [])
    parts = [
        {key: part.get(key) for key in ("name", "kind", "group", "shots", "shards", "rng_namespace", "dir", "members", "status", "artifact_paths")}
        | {"rng_base_seed": final_rng_base_seed(str(part.get("rng_namespace")))}
        for part in result.get("parts") or []
    ]
    champion = next((row for row in rows if row.get("champion")), None)
    reference_rows = [row for row in rows if row.get("is_baseline")]
    reported = fs.reported_top(block, rows)
    ranking_payload = {
        "schema": fs.RANKING_SCHEMA,
        "final_stage": block,
        "rounds": final_rounds,
        "status": status,
        "champion": None if champion is None else {
            "program_id": champion.get("program_id"),
            "search_program_id": champion.get("search_program_id"),
            "code_hash": champion.get("code_hash"),
            "per_logical_per_round": champion.get("per_logical_per_round"),
            "per_logical_per_round_ci": champion.get("per_logical_per_round_ci"),
            "shots_per_basis": champion.get("shots_x"),
        },
        "rows": rows,
        "reference_rows": [
            {
                "program_id": row.get("program_id"),
                "code_hash": row.get("code_hash"),
                "shared_with_finalist": bool(row.get("shared_with_finalist")),
                "measured_as": row.get("measured_as"),
                "shots_per_basis": row.get("shots_x"),
            }
            for row in reference_rows
        ],
        "parts": parts,
        **({"reported_top": reported} if reported is not None else {}),
    }
    if status == "ok":
        (final_dir / "final_ranking.json").write_text(
            json.dumps(ranking_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        _write_final_ranking_csv(final_dir / "final_ranking.csv", rows)
    final_measurement_audit = (
        _audit_paper_final_measurements(state, block=block, parts=parts, rows=rows, rounds=final_rounds)
        if status == "ok"
        else {
            "enabled": bool(protocol.get("strict_turns")),
            "status": "failed",
            "errors": ["a final-stage part did not finish with status ok"],
        }
    )
    payload = {
        "status": status,
        "final_stage_id": block.get("id"),
        "final_stage": block,
        "p_values": list(p_values),
        "rounds": final_rounds,
        "rounds_policy": str(rounds_policy.get("final") or ROUNDS_POLICY_ID),
        "final_rng_namespace": namespace,
        "final_rng_base_seed": final_rng_base_seed(namespace),
        "shortlist": composition,
        "parts": parts,
        "ranking": rows,
        "champion": ranking_payload["champion"],
        **({"reported_top": reported} if reported is not None else {}),
        # The confirmation part, in the shape the earlier final stage reported its single run.
        "runs": [
            {"rounds": final_rounds, "status": part["status"], "artifact_paths": part["artifact_paths"]}
            for part in parts
            if part.get("kind") == "confirm"
        ],
        "turn_audit": turn_audit,
        "final_measurement_audit": final_measurement_audit,
    }
    (final_dir / "final_slow_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if bool(protocol.get("strict_turns")) and status != "ok":
        raise RuntimeError(
            "strict final_slow is incomplete; checkpoint remains before final_slow "
            "so the same job can be resumed (every finished candidate of every part is reused)"
        )
    return {"final_slow": payload}


FINAL_RANKING_CSV_COLUMNS = (
    "final_rank", "champion", "is_baseline", "also_reference", "shared_with_finalist", "measured_as", "program_id",
    "search_program_id", "code_hash",
    "family", "n", "k", "d_search", "max_check_weight", "logicals", "rounds", "search_rank",
    "search_per_logical_per_round", "shots_x", "failures_x", "shots_z", "failures_z", "cumulative_cap",
    "cumulative_complete", "per_logical_per_round", "per_logical_per_round_ci_low", "per_logical_per_round_ci_high",
    "confirmation_status",
)


def _write_final_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FINAL_RANKING_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            ci = row.get("per_logical_per_round_ci") or [None, None]
            writer.writerow(
                {
                    **{key: row.get(key) for key in FINAL_RANKING_CSV_COLUMNS},
                    "per_logical_per_round_ci_low": ci[0],
                    "per_logical_per_round_ci_high": ci[1],
                }
            )


def _audit_paper_final_measurements(
    state: CampaignState,
    *,
    block: dict[str, Any],
    parts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    """Strict runs: every ok row of every part decoded exactly that part's shots per basis at the frozen p,
    T and decoder; every pooled row is the sum of the ok parts it lists (measured under its ``measured_as``
    program id); at least one finalist is confirmed."""

    protocol = state["protocol"]
    if not bool(protocol.get("strict_turns")):
        return {"enabled": False}
    expected_p = float(protocol["anchor_p"])
    fixed = str((protocol.get("rounds_policy") or {}).get("final") or "") == "fixed_Tc"
    errors: list[str] = []
    terminal = 0
    measured = 0
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for part in parts:
        table = tables.setdefault(str(part.get("name")), {})
        for row in _slow_rows((part.get("artifact_paths") or {}).get("rankings")):
            where = f"{part.get('name')}: row {row.get('program_id')}"
            if row.get("status") in ("decode_infeasible", "decoder_unsupported", "schedule_catastrophic"):
                terminal += 1
                continue
            if row.get("status") != "ok":
                errors.append(f"{where} is not ok")
                continue
            measured += 1
            shots = int(part.get("shots") or 0)
            if int(row.get("shots_x") or 0) != shots or int(row.get("shots_z") or 0) != shots:
                errors.append(f"{where} decoded {row.get('shots_x')}/{row.get('shots_z')} shots, planned {shots}")
            try:
                if abs(float(row.get("noise_value")) - expected_p) > 1e-15:
                    errors.append(f"{where} has wrong p")
            except (TypeError, ValueError):
                errors.append(f"{where} has no p")
            row_expected = int(rounds) if fixed else max(int(rounds), _row_d_search(row))
            if int(row.get("rounds") or 0) != row_expected:
                errors.append(f"{where} rounds={row.get('rounds')!r}, expected {row_expected}")
            if str(row.get("decoder") or "") != DecoderName.TELESCOPING.value:
                errors.append(f"{where} is not {DecoderName.TELESCOPING.value}")
            elif not _telescoping_config_ok(row.get("decoder_config")):
                errors.append(f"{where} has a non-protocol telescoping config")
            table[str(row.get("program_id"))] = row
    confirmed = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        if not row.get("is_baseline"):
            confirmed += 1
        measured_as = str(row.get("measured_as") or row.get("program_id"))
        expected = [0, 0, 0, 0]
        for record in row.get("parts") or []:
            if record.get("status") != "ok":
                continue
            part_row = (tables.get(str(record.get("part"))) or {}).get(measured_as)
            if part_row is None:
                errors.append(f"pooled row {row.get('program_id')} lists part {record.get('part')} without an ok row")
                continue
            for index, key in enumerate(("shots_x", "failures_x", "shots_z", "failures_z")):
                expected[index] += int(part_row.get(key) or 0)
        got = [int(row.get(key) or 0) for key in ("shots_x", "failures_x", "shots_z", "failures_z")]
        if got != expected:
            errors.append(f"pooled row {row.get('program_id')} {got} is not the sum of its parts {expected}")
    if confirmed == 0:
        errors.append("no confirmed finalist")
    payload = {
        "enabled": True,
        "schema": "campaign_final_measurement_audit_paper_v1",
        "status": "complete" if not errors else "failed",
        "final_stage_id": block.get("id"),
        "expected_p": expected_p,
        "expected_rounds": int(rounds),
        "confirm_shots_per_basis": int(block.get("confirm_shots") or 0),
        "cumulative_cap": None if block.get("cumulative_cap") is None else int(block["cumulative_cap"]),
        "parts": len(parts),
        "measured_rows": measured,
        "terminal_rows": terminal,
        "confirmed_finalists": confirmed,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("strict final measurement audit failed: " + "; ".join(errors[:10]))
    return payload


def _legacy_final_slow_node(state: CampaignState) -> dict[str, Any]:
    """The fixed-T final audit (v3.2 §4.6).

    v3.7: 2 structural endpoint finalists + 4 runtime finalists by measured
    plpr (2026-09-10: 4 + 2; was 3 + 1), fresh shots, ONE fixed round
    count for every candidate. The previous stage ran T=max(3, d_proxy) while
    labelling it `T=d`: d_proxy is a heuristic search value, not a certified
    distance, so that both mislabelled the measurement and gave every
    candidate a different circuit depth chosen by the search itself."""

    protocol = state["protocol"]
    turn_audit = _validate_all_turn_audits(state)
    spec = state["spec"]
    run_dir = Path(state["run_dir"])
    final_dir = run_dir / "final_slow"
    final_dir.mkdir(parents=True, exist_ok=True)
    verification_template = _verification_from_protocol(protocol, spec)
    structural_config = verification_template.structural
    rounds_policy = protocol.get("rounds_policy") or {}
    final_rounds = _protocol_final_rounds(protocol)
    namespace = str(
        rounds_policy.get("final_rng_namespace") or structural_config.final_rng_namespace
    )
    entries, composition = _final_entries(state, structural_config)
    baseline = state.get("baseline_summary") or {}
    if isinstance(baseline.get("spec"), dict):
        # The resident reference is measured alongside but does not consume a
        # finalist slot. 2026-10-14: when the LLM's own copy of the reference
        # code is already a finalist (b144 Phase B, turn-0 program), the
        # baseline row is skipped: the finalist row measures the identical
        # circuit, and the duplicate cost a 3 h measurement and (before
        # pool.release_shared_pickle) lost the DEM pickle the two shared.
        baseline_hash = str(baseline.get("code_hash") or "")
        if baseline_hash and any(str(row.get("code_hash") or "") == baseline_hash for row in entries):
            composition["baseline_row"] = "skipped_duplicate_finalist"
            print(
                f"  [final] reference code {baseline_hash[:8]} is already a finalist; baseline row skipped",
                flush=True,
            )
        else:
            entries = entries + [dict(baseline, selection_channels=["baseline"])]
    if not entries:
        if bool(protocol.get("strict_turns")):
            raise RuntimeError(
                "strict campaign cannot finish with an empty finalist shortlist; "
                "resume final_slow after producing a measurable candidate"
            )
        return {"final_slow": {"status": "empty_shortlist", "runs": []}}

    p_values = tuple(float(value) for value in protocol["final_p_values"])
    queue_path = final_dir / f"queue_T{final_rounds}.jsonl"
    queue_path.write_text(
        "\n".join(
            json.dumps({
                "program_id": row["program_id"],
                "spec": row["spec"],
                "code_hash": row.get("code_hash"),
                "selection_channels": row.get("selection_channels"),
                # writeback derives is_baseline from this queue record;
                # include_baseline=False here means no injection path sets
                # it, so dropping it would mislabel the BB reference row
                # as a candidate in the final frontier.
                "is_baseline": bool(row.get("is_baseline"))
                or row.get("selection_channels") == ["baseline"],
                # v3.6 B5: a search-stage decode_infeasible candidate retrying
                # once here; the pool applies the final-stage decode budget.
                "retry_at_final": bool(row.get("retry_at_final")),
                "canonical_hash": row.get("canonical_hash"),
            })
            for row in entries
        )
        + "\n",
        encoding="utf-8",
    )
    verification = _verification_from_protocol(
        protocol,
        spec,
        p_values=p_values,
        shots=int(protocol["final_shots"]),
        rounds=final_rounds,
    )
    # 2026-09-24: runtime ceiling of the final-stage shard watchdog (protocol
    # runtime key final_decode_shard_timeout_cap_s, see checkpoint.RESUME_
    # RUNTIME_KEYS). Applied to this stage's verification only; a capped
    # shard is a v3.4 watchdog timeout -> the candidate ends decode_infeasible.
    final_cap = protocol.get("final_decode_shard_timeout_cap_s")
    if final_cap is not None and float(final_cap) > 0.0:
        verification = dataclasses.replace(
            verification, final_decode_shard_timeout_cap_s=float(final_cap)
        )
        print(
            f"[campaign] final stage: decode-shard watchdog capped at {float(final_cap):.0f} s "
            f"(final_decode_shard_timeout_cap_s)",
            flush=True,
        )
    summary = evaluate_promotion_queue(
        queue_path=queue_path,
        out_dir=final_dir / f"T{final_rounds}",
        verification=verification,
        include_baseline=False,
        # 2026-09-24: a link killed at walltime keeps every candidate it
        # finished (candidates/<id>.json); the next attempt measures the rest.
        resume_candidates=True,
        workers=int(protocol.get("slow_workers") or 0),
        # v3.4: the final stage has its own frozen partition count (128) so
        # 200k shots per basis fit the per-shot watchdog; `shot_shards` is
        # the SEARCH count. The fallback only serves snapshots without the
        # key (validate_resume_protocol already refuses those under v3.4).
        shot_shards=int(protocol.get("final_shot_shards") or protocol["shot_shards"]),
        seed=final_rng_base_seed(namespace),
        # v3.6: the pool tags every shard request and result with its stage
        # ("final"); the telescoping CPU protocol has no stage-dependent
        # limit, the tag is what the audits and the validator read.
        stage="final",
    )
    runs = [
        {
            "rounds": final_rounds,
            "status": summary["status"],
            "artifact_paths": summary["artifact_paths"],
        }
    ]
    (final_dir / "final_shortlist.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = "ok" if all(r["status"] == "ok" for r in runs) else "partial"
    final_measurement_audit = _audit_final_measurements(state, runs) if status == "ok" else {
        "enabled": bool(protocol.get("strict_turns")),
        "status": "failed",
        "errors": ["the fixed-T final evaluation did not finish with status ok"],
    }
    payload = {
        "status": status,
        "p_values": list(p_values),
        "rounds": final_rounds,
        "rounds_policy": str((protocol.get("rounds_policy") or {}).get("final") or ROUNDS_POLICY_ID),
        "final_rng_namespace": namespace,
        "final_rng_base_seed": final_rng_base_seed(namespace),
        "shortlist": composition,
        "runs": runs,
        "turn_audit": turn_audit,
        "final_measurement_audit": final_measurement_audit,
        "pending_extensions": [
            "schedule_sensitivity_audit (02 §4.4)",
            "bposd_reference_pass (03 §9.3)",
            "isomorphism_tier23 (03 §6.3)",
            "slope_offset_recalibration (03 §15)",
        ],
    }
    (final_dir / "final_slow_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if bool(protocol.get("strict_turns")) and payload["status"] != "ok":
        raise RuntimeError(
            "strict final_slow is incomplete; checkpoint remains before final_slow "
            "so the same job can be resumed"
        )
    return {"final_slow": payload}


def _row_d_search(row: dict[str, Any]) -> int:
    """The proxy distance a measured row was compiled with (0 when absent)."""

    for key in ("d_search", "d_proxy", "d"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return 0


def _protocol_final_rounds(protocol: dict[str, Any]) -> int:
    """The ONE place final rounds come from (v3.2 §4.6).

    `protocol.verification.rounds` may not become a second source of truth:
    when both are serialized they must agree, and a missing stage field fails
    closed rather than silently falling back to a per-candidate value."""

    rounds_policy = protocol.get("rounds_policy") or {}
    if str(rounds_policy.get("final") or "") not in ROUNDS_POLICY_IDS:
        raise ValueError(
            f"protocol rounds_policy.final must be one of {ROUNDS_POLICY_IDS!r} "
            f"(got {rounds_policy.get('final')!r})"
        )
    final_rounds = rounds_policy.get("final_rounds")
    if final_rounds is None:
        raise ValueError("protocol rounds_policy.final_rounds is missing")
    final_rounds = int(final_rounds)
    verification_rounds = (protocol.get("verification") or {}).get("rounds")
    if verification_rounds is not None and int(verification_rounds) != final_rounds:
        raise ValueError(
            "protocol rounds conflict: verification.rounds="
            f"{verification_rounds} vs rounds_policy.final_rounds={final_rounds}"
        )
    return final_rounds


def _audit_final_measurements(
    state: CampaignState, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify fixed p, fixed per-basis shots, and the fixed T before report."""

    protocol = state["protocol"]
    if not bool(protocol.get("strict_turns")):
        return {"enabled": False}
    expected_p = float(protocol["anchor_p"])
    expected_shots = int(protocol["final_shots"])
    expected_rounds = _protocol_final_rounds(protocol)
    # v3.6: the decoder blob is budget independent (telescoping decoder,
    # user decision 2026-09-03), so the audit needs no budget point.
    errors: list[str] = []
    measured_rows = 0
    decode_infeasible_rows = 0
    decoder_unsupported_rows = 0
    for run in runs:
        rounds = int(run.get("rounds") or 0)
        rankings_path = (run.get("artifact_paths") or {}).get("rankings")
        rows = _slow_rows(rankings_path)
        if not rows:
            errors.append(f"T={rounds}: final rankings are empty")
            continue
        for row in rows:
            if row.get("status") == "decode_infeasible":
                # v3.4: decode exceeded the frozen per-shot budget. A terminal
                # measurement outcome, not an audit defect — the row never
                # reaches the frontier and does not count as a measurement.
                decode_infeasible_rows += 1
                continue
            if row.get("status") == "decoder_unsupported":
                # v3.6.1: the decoder cannot be built for this DEM. Terminal
                # like the row above.
                decoder_unsupported_rows += 1
                continue
            if row.get("status") != "ok":
                errors.append(f"T={rounds}: row {row.get('program_id')} is not ok")
                continue
            measured_rows += 1
            try:
                noise_matches = abs(float(row.get("noise_value")) - expected_p) <= 1e-15
            except (TypeError, ValueError):
                noise_matches = False
            if not noise_matches:
                errors.append(f"T={rounds}: row {row.get('program_id')} has wrong p")
            if int(row.get("shots_x") or 0) != expected_shots:
                errors.append(f"T={rounds}: row {row.get('program_id')} has wrong X shots")
            if int(row.get("shots_z") or 0) != expected_shots:
                errors.append(f"T={rounds}: row {row.get('program_id')} has wrong Z shots")
            if int(row.get("total_shots") or 0) != 2 * expected_shots:
                errors.append(f"T={rounds}: row {row.get('program_id')} has wrong total shots")
            # max_Tc_d: the run-level T is the floor; each row must record
            # T = max(floor, its own d_search). fixed_Tc (Fig.3 formal run):
            # every row records the floor.
            if str((protocol.get("rounds_policy") or {}).get("final") or "") == "fixed_Tc":
                row_expected = expected_rounds
                rule = f"T_c={expected_rounds} (fixed_Tc)"
            else:
                row_expected = max(expected_rounds, _row_d_search(row))
                rule = f"max(T_c={expected_rounds}, d_search={_row_d_search(row)})"
            if rounds != expected_rounds or int(row.get("rounds") or 0) != row_expected:
                errors.append(
                    f"T={rounds}: row {row.get('program_id')} rounds={row.get('rounds')!r}, "
                    f"expected {rule}={row_expected}"
                )
            decoder_config = row.get("decoder_config")
            if str(row.get("decoder") or "") != DecoderName.TELESCOPING.value:
                errors.append(
                    f"T={rounds}: row {row.get('program_id')} is not "
                    f"{DecoderName.TELESCOPING.value}"
                )
            elif not _telescoping_config_ok(decoder_config):
                errors.append(
                    f"T={rounds}: row {row.get('program_id')} has a non-protocol "
                    f"telescoping config {decoder_config!r}"
                )
    if measured_rows == 0:
        errors.append("no successful final measurement rows")
    payload = {
        "enabled": True,
        "schema": "campaign_final_measurement_audit_v1",
        "status": "complete" if not errors else "failed",
        "expected_p": expected_p,
        "shots_per_basis": expected_shots,
        "expected_rounds": expected_rounds,
        "measured_rows": measured_rows,
        "decode_infeasible_rows": decode_infeasible_rows,
        "decoder_unsupported_rows": decoder_unsupported_rows,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("strict final measurement audit failed: " + "; ".join(errors[:10]))
    return payload


def report_node(state: CampaignState) -> dict[str, Any]:
    """Render the campaign report from evaluation/ artifacts only."""

    run_dir = Path(state["run_dir"])
    final = state.get("final_slow") or {}
    channels_by_hash: dict[str, list[str]] = {
        str(row.get("code_hash") or ""): list(row.get("selection_channels") or [])
        for row in ((final.get("shortlist") or {}).get("selected") or [])
        if row.get("code_hash")
    }
    frontier: list[dict[str, Any]] = []
    paper_final = final.get("final_stage_id") is not None
    if paper_final:
        # The paper's final stage: the frontier IS the pooled ranking (finalists by pooled pLER, then the
        # reference row), each row carrying its cumulative shots and failures per basis.
        frontier = [dict(row) for row in final.get("ranking") or [] if row.get("per_logical_per_round") is not None]
    for run in ([] if paper_final else final.get("runs") or []):
        for row in _slow_rows((run.get("artifact_paths") or {}).get("rankings")):
            if row.get("status") == "ok" and row.get("per_logical_per_round") is not None:
                light = _light_slow_row(row)
                # 2026-09-29: the MEASURED round count (max_Tc_d: max(T_c,
                # d_search) per candidate, recorded by the slow row); the
                # stage floor is only the fallback for rows without one. The
                # floor used to overwrite it, so every final with a d > T_c
                # code reported rounds=T_c and failed the artifact validator.
                measured_rounds = row.get("rounds")
                light["rounds"] = int(measured_rounds) if measured_rounds else run.get("rounds")
                light["is_baseline"] = bool(row.get("is_baseline"))
                full_hash = str(row.get("code_hash") or "")
                light["selection_channels"] = next(
                    (
                        channels
                        for prefix, channels in channels_by_hash.items()
                        if prefix and full_hash.startswith(prefix)
                    ),
                    ["baseline"] if row.get("is_baseline") else [],
                )
                light["decoder_config"] = row.get("decoder_config")
                frontier.append(light)
    if not paper_final:
        frontier.sort(key=lambda item: float(item["per_logical_per_round"]))
    structural_elites = list(state.get("structural_proxy_elites") or [])
    report = {
        "schema": "campaign_report",
        "run_id": state["run_id"],
        "spec": state["spec"],
        "protocol": state["protocol"],
        "turns_completed": int((state.get("budget") or {}).get("turns_used") or 0),
        "stop": state.get("stop"),
        "budget": state.get("budget"),
        # Two ledgers, reported side by side and never merged into one score.
        "structural_proxy_elites": structural_elites,
        "hall_of_fame": state.get("hof_summary"),
        "baseline": state.get("baseline_summary"),
        "final_frontier": frontier[:24],
        "final_shortlist": final.get("shortlist"),
        "final_rounds": final.get("rounds"),
        "final_rng_namespace": final.get("final_rng_namespace"),
        "final_rng_base_seed": final.get("final_rng_base_seed"),
        "final_slow_status": final.get("status"),
        "turn_audit": final.get("turn_audit"),
        "final_measurement_audit": final.get("final_measurement_audit"),
        "pending_extensions": final.get("pending_extensions"),
    }
    if paper_final:
        report["final_stage"] = final.get("final_stage")
        report["final_stage_parts"] = [
            {key: part.get(key) for key in ("name", "kind", "group", "shots", "shards", "rng_namespace", "rng_base_seed", "dir", "members", "status")}
            for part in final.get("parts") or []
        ]
        report["champion"] = final.get("champion")
        if final.get("reported_top") is not None:
            report["final_reported_top"] = final.get("reported_top")
    (run_dir / "campaign_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "campaign_report.md").write_text(_report_markdown(report), encoding="utf-8")
    _export_run_csvs(run_dir, where="final")
    _write_channel_reports(run_dir, report)
    return {"report": {"path": str(run_dir / "campaign_report.json")}}


def _write_channel_reports(run_dir: Path, report: dict[str, Any]) -> None:
    """Separate structural / runtime / fixed-T artifacts (v3.2 §4.6).

    `d_search` is labelled a proxy in every column name that carries it: these
    files may be read as a code list, and a proxy distance printed as `d` is
    how a heuristic becomes a claimed [[n,k,d]]."""

    import csv

    def _dump(name: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
        (run_dir / f"{name}.json").write_text(
            json.dumps(
                {"schema": name, "count": len(rows), "rows": rows},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        with (run_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in columns})

    _dump(
        "structural_proxy_elites",
        [
            {
                "code_hash": row.get("code_hash"),
                "program_id": row.get("program_id"),
                "family": row.get("family"),
                "n": row.get("n"),
                "n_total": row.get("n_total"),
                "k": row.get("k"),
                "d_search_proxy": row.get("d_search"),
                "Q_search_proxy": row.get("Q_search"),
                "pareto_rank": row.get("pareto_rank"),
                "crowding_boundary": row.get("crowding_boundary"),
                "first_seen_turn": row.get("first_seen_turn"),
                "last_seen_turn": row.get("last_seen_turn"),
            }
            for row in report.get("structural_proxy_elites") or []
        ],
        (
            "code_hash", "program_id", "family", "n", "n_total", "k",
            "d_search_proxy", "Q_search_proxy", "pareto_rank",
            "crowding_boundary", "first_seen_turn", "last_seen_turn",
        ),
    )
    _dump(
        "runtime_hof",
        [
            {
                "code_hash": row.get("code_hash"),
                "program_id": row.get("program_id"),
                "family": row.get("family"),
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search_proxy": row.get("d_search") or row.get("d_proxy"),
                "per_logical_per_round": row.get("per_logical_per_round"),
                "decoder": row.get("decoder"),
                "rounds": row.get("rounds"),
                "first_seen_turn": row.get("first_seen_turn"),
                "last_measured_turn": row.get("last_measured_turn"),
            }
            for row in report.get("hall_of_fame") or []
        ],
        (
            "code_hash", "program_id", "family", "n", "k", "d_search_proxy",
            "per_logical_per_round", "decoder", "rounds", "first_seen_turn",
            "last_measured_turn",
        ),
    )
    _dump(
        "fixed_T_final_audit",
        [
            {
                "code_hash": row.get("code_hash"),
                "program_id": row.get("program_id"),
                "family": row.get("family"),
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search_proxy": row.get("d_search") or row.get("d_proxy"),
                "rounds": row.get("rounds"),
                "per_logical_per_round": row.get("per_logical_per_round"),
                "decoder": row.get("decoder"),
                "selection_channels": ",".join(row.get("selection_channels") or []),
                "is_baseline": bool(row.get("is_baseline")),
            }
            for row in report.get("final_frontier") or []
        ],
        (
            "code_hash", "program_id", "family", "n", "k", "d_search_proxy",
            "rounds", "per_logical_per_round", "decoder", "selection_channels",
            "is_baseline",
        ),
    )


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Campaign report — {report['run_id']}",
        "",
        f"- qubit budget: {report['spec'].get('qubit_budget')}; "
        f"families: {', '.join(report['spec'].get('families') or [])}",
        f"- profile: {report['protocol'].get('noise_profile')} @ anchor "
        f"p={report['protocol'].get('anchor_p')}",
        f"- turns: {report['turns_completed']}; stop: "
        f"{(report.get('stop') or {}).get('reason')}",
        "",
        f"## Fixed-T final audit (p={report['protocol'].get('anchor_p')}, "
        f"T={report.get('final_rounds')})",
        "",
        "`d_search` is the heuristic search proxy, not a certified distance: "
        "these rows are `[[n,k]]` codes with a proxy distance, not `[[n,k,d]]` "
        "claims.",
        "",
        "| program | family | n | k | d_search (proxy) | plpr | CI | channels | baseline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    stage = report.get("final_stage") or {}
    if stage:
        champion = report.get("champion") or {}
        topup_text = (
            f"then topped up to {stage.get('cumulative_cap')} ({stage.get('cumulative_cap_source')})"
            if stage.get("cumulative_cap") is not None
            else "no top-up (the initial confirmation only)"
        )
        reference = stage.get("reference") or {}
        reference_text = (
            f" Reference {reference.get('program_id')}: {reference.get('cumulative_cap')} shots per basis."
            if reference
            else ""
        )
        lines[-3:-3] = [
            f"Final stage `{stage.get('id')}`: top {stage.get('top')} distinct codes by search-stage pLER, "
            f"{stage.get('confirm_shots')} shots per basis {topup_text}; rows below are pooled over every part. "
            f"Champion: {champion.get('program_id') or '-'}.{reference_text}",
            "",
        ]
    for row in report.get("final_frontier") or []:
        # A5 plpr CI: the only interval on the same scale as the plpr point
        # and comparable across the table's fixed-T round counts (the
        # block CI is ~T x larger and T differs per row).
        ci = row.get("per_logical_per_round_ci")
        ci_text = (
            f"[{ci[0]:.2e}, {ci[1]:.2e}]"
            if ci and all(isinstance(v, (int, float)) for v in ci)
            else "-"
        )
        plpr = row.get("per_logical_per_round")
        plpr_text = f"{plpr:.3e}" if isinstance(plpr, (int, float)) else "-"
        lines.append(
            f"| {row.get('program_id')} | {row.get('family')} "
            f"| {row.get('n')} | {row.get('k')} "
            f"| {row.get('d_search') or row.get('d_proxy')} "
            f"| {plpr_text} "
            f"| {ci_text} "
            f"| {','.join(row.get('selection_channels') or []) or '-'} "
            f"| {'yes' if row.get('is_baseline') else ''} |"
        )
    elites = report.get("structural_proxy_elites") or []
    if elites:
        lines += [
            "",
            "## Structural proxy frontier ((k, d_search) Pareto elites)",
            "",
            "| code | family | n | n_total | k | d_search (proxy) | rank |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in elites:
            lines.append(
                f"| {str(row.get('code_hash') or '')[:12]} | {row.get('family')} "
                f"| {row.get('n')} | {row.get('n_total')} | {row.get('k')} "
                f"| {row.get('d_search')} | {row.get('pareto_rank')} |"
            )
    pending = report.get("pending_extensions") or []
    if pending:
        lines += ["", "## Pending protocol extensions", ""]
        lines += [f"- {item}" for item in pending]
    lines.append("")
    return "\n".join(lines)


def _verification_from_protocol(
    protocol: dict[str, Any],
    spec: dict[str, Any],
    *,
    p_values: tuple[float, ...] | None = None,
    shots: int | None = None,
    rounds: int | None = None,
) -> VerificationConfig:
    import dataclasses

    from omniqec_agentic.evaluation import verification_from_dict

    verification = verification_from_dict(protocol.get("verification"))
    # v3.7: no cap unless the spec declares a (hardware) tightening. 2026-10-16:
    # the snapshot's own cap (--max-check-weight, Fig.3 formal run) is kept --
    # this line used to overwrite it with the spec's None, so the construction
    # cap and its contract line silently vanished from every turn.
    caps = [int(v) for v in (spec.get("weight_limit"), verification.max_check_weight) if v is not None]
    replacements: dict[str, Any] = {
        "max_check_weight": min(caps) if caps else None,
    }
    # Search phase: campaign-fixed T_c from the protocol snapshot (fairness
    # protocol); final_slow explicitly uses the separately frozen final T_c.
    search_rounds = (protocol.get("rounds_policy") or {}).get("search_rounds")
    if rounds is None and search_rounds:
        rounds = int(search_rounds)
    if p_values is not None:
        noise = dataclasses.replace(verification.noise, values=tuple(p_values))
        replacements["noise"] = noise
        replacements["p_values"] = tuple(p_values)
    if shots is not None:
        replacements["shots"] = int(shots)
    if rounds is not None:
        replacements["rounds"] = int(rounds)
    return dataclasses.replace(verification, **replacements)


def _slow_rows(rankings_path: str | None) -> list[dict[str, Any]]:
    if not rankings_path or not Path(rankings_path).exists():
        return []
    data = json.loads(Path(rankings_path).read_text(encoding="utf-8"))
    rows = data.get("rankings") if isinstance(data, dict) else None
    return [row for row in rows or [] if isinstance(row, dict)]


def _structural_frontier_rows(
    elites: list[dict[str, Any]], *, limit: int = 16
) -> list[dict[str, Any]]:
    """Digest view of the structural cache: coordinates and provenance only.

    No plpr, no Q_eff, no ranking language — the row order is the reducer's
    channel-neutral order, and nothing downstream may re-sort it by a runtime
    quantity."""

    return [
        {
            "code_hash": str(row.get("code_hash") or "")[:12],
            "family": row.get("family"),
            "n": row.get("n"),
            "n_total": row.get("n_total"),
            "k": row.get("k"),
            "d_search": row.get("d_search"),
            # v3.7.1: the third front axis. Without it the reflector was shown a
            # two-coordinate front and told retention used two coordinates.
            "max_check_weight": row.get("max_check_weight"),
            "pareto_rank": row.get("pareto_rank"),
            "crowding_boundary": row.get("crowding_boundary"),
            "first_seen_turn": row.get("first_seen_turn"),
            "last_seen_turn": row.get("last_seen_turn"),
        }
        for row in elites[: max(0, int(limit))]
    ]


def _light_slow_row(row: dict[str, Any]) -> dict[str, Any]:
    light = {
        key: row.get(key)
        for key in (
            "program_id",
            "code_hash",
            "canonical_hash",
            "wl_hash",
            "family",
            "n",
            "k",
            "max_check_weight",
            "d_proxy",
            "per_logical_per_round",
            "per_logical_per_round_ci",
            "per_logical_block_ci",
            "rounds",
            "decoder",
            "noise_value",
            "verified_circuit_distance",
            "degraded_ext",
            "stopped_by",
            "shots_x",
            "shots_z",
            "failures_x",
            "failures_z",
            "total_shots",
            "total_failures",
        )
    }
    light["d_search"] = row.get("d_search") or row.get("d_proxy")
    # Inheritance payload travels WITH the hall-of-fame row (v3.2 §4.5): the
    # candidate file is this turn's ingestion input, not a cross-turn source of
    # truth, and without these fields a HoF row cannot be rendered as a parent
    # at all (the renderer requires spec AND construct_source).
    light.update(_inheritance_from_candidate(row.get("candidate_path")))
    return light


def _inheritance_from_candidate(candidate_path: str | None) -> dict[str, Any]:
    """spec + construct_source + rng_seed + source_agent from a candidate file."""

    empty = {
        "spec": None,
        "construct_source": None,
        "rng_seed": 0,
        "source_agent": None,
        "deformation_mode": "none",
    }
    if not candidate_path or not Path(candidate_path).exists():
        return empty
    try:
        payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    except Exception:
        return empty
    promotion = payload.get("promotion_record")
    promotion = promotion if isinstance(promotion, dict) else {}
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        spec = promotion.get("spec") if isinstance(promotion.get("spec"), dict) else None
    return {
        "spec": spec,
        "construct_source": promotion.get("construct_source"),
        "rng_seed": promotion.get("rng_seed", 0),
        "source_agent": promotion.get("source_agent"),
        "deformation_mode": payload.get("deformation_mode")
        or promotion.get("deformation_mode")
        or "none",
    }


def _spec_from_candidate(candidate_path: str | None) -> dict[str, Any] | None:
    return _inheritance_from_candidate(candidate_path)["spec"]


def _fast_top_k(queue_path: str | None, *, limit: int = 8) -> list[dict[str, Any]]:
    if not queue_path or not Path(queue_path).exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(queue_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    key: record.get(key)
                    for key in ("program_id", "family", "n", "k", "d_proxy", "Q", "code_hash")
                }
            )
            if len(rows) >= limit:
                break
    except Exception:
        return rows
    return rows


def build_campaign_graph(policy: Policy) -> StateGraph:
    graph = StateGraph(CampaignState)
    graph.add_node("intake", intake_node)
    graph.add_node("orchestrator_turn", make_orchestrator_node(policy))
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("final_slow", final_slow_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "orchestrator_turn")
    graph.add_edge("orchestrator_turn", "execute_plan")
    graph.add_edge("execute_plan", "reconcile")
    graph.add_conditional_edges(
        "reconcile",
        continue_or_finalize,
        {"continue": "orchestrator_turn", "finalize": "final_slow"},
    )
    graph.add_edge("final_slow", "report")
    graph.add_edge("report", END)
    return graph


def run_campaign(
    *,
    spec: CampaignSpec,
    out_dir: str | Path,
    policy: Policy | None = None,
    verification: VerificationConfig | None = None,
    racing: RacingConfig | None = None,
    max_turns: int = 3,
    proposal_count: int = 8,
    generations_per_turn: int = 1,
    slow_workers: int = 0,
    shot_shards: int = FORMAL_SHOT_SHARDS,
    final_shot_shards: int = FORMAL_FINAL_SHOT_SHARDS,
    feedback_mode: str = "both",
    resume: bool = False,
    final_shots: int | None = None,
    strict_turns: bool = False,
    required_provider: str | None = None,
    required_model: str | None = None,
    rng_seeds_per_program: int = 1,
    reference_codes: dict[str, Any] | None = None,
    final_decode_shard_timeout_cap_s: float | None = None,
    search_decode_shard_timeout_cap_s: float | None = None,
    fixed_program_count: bool | None = None,
    promotion_max_check_weight: int | None = None,
    campaign_id: str | None = None,
    ablation: str | None = None,
    search_baseline: dict[str, Any] | None = None,
    final_stage_id: str | None = FINAL_STAGE_PAPER_ID,
    final_cumulative_shots: int | None = None,
) -> dict[str, Any]:
    """Drive one campaign (budget point) through the graph.

    The final stage is the paper's (``final_stage_id`` = paper_top5_cumulative_v1, Appendix D3 / Table VII);
    ``final_cumulative_shots`` overrides Table VII's per-basis limit outside strict mode (smoke runs).

    checkpoint: SqliteSaver in run_dir, thread_id=run_id, durability="sync",
    recursion_limit = max_turns*4+64 (03 §2.3). --resume validates the stored
    protocol snapshot against the current one and refuses on mismatch."""

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"campaign_b{spec.qubit_budget}"
    policy = policy or FallbackPolicy()
    # 2026-11-10 (paper alignment): R = 12 for every evaluation (fixed_Tc) when the caller gives no
    # verification config (VerificationConfig's own default stays the as-run max_Tc_d that earlier snapshots
    # and their readers pin). Racing stays off unless the caller passes a policy (run-campaign passes the
    # paper's, config.paper_racing).
    verification = verification or VerificationConfig(rounds_policy="fixed_Tc")
    racing = racing or RacingConfig()
    # Fixed search-phase round count, GLOBAL across budget points (user
    # decision 2026-08-16): the paper's Fig.2 frontier contains d>8 codes at
    # these budgets, so T_c=8 would time-truncate them during racing. 12
    # covers the experiment's verifiable distance regime (BB144 d=12 is the
    # reference ceiling) with identical boundary bias for every candidate.
    search_rounds = 12
    protocol = build_protocol_snapshot(
        verification=verification,
        racing=racing,
        max_turns=max_turns,
        proposal_count=proposal_count,
        generations_per_turn=generations_per_turn,
        slow_workers=slow_workers,
        shot_shards=shot_shards,
        final_shot_shards=final_shot_shards,
        feedback_mode=feedback_mode,
        search_rounds=search_rounds,
        final_shots=final_shots,
        strict_turns=strict_turns,
        required_provider=required_provider,
        required_model=required_model,
        rng_seeds_per_program=rng_seeds_per_program,
        # v3.6 (docs/06 §1.4): the budget point selects the shots per basis.
        n_budget=int(spec.qubit_budget),
        reference_codes=reference_codes,
        final_decode_shard_timeout_cap_s=final_decode_shard_timeout_cap_s,
        search_decode_shard_timeout_cap_s=search_decode_shard_timeout_cap_s,
        fixed_program_count=fixed_program_count,
        promotion_max_check_weight=promotion_max_check_weight,
        campaign_id=campaign_id,
        ablation=ablation,
        search_baseline=search_baseline,
        final_stage_id=final_stage_id,
        final_cumulative_shots=final_cumulative_shots,
    )
    checkpointer = open_checkpointer(run_dir)
    app = build_campaign_graph(policy).compile(checkpointer=checkpointer)
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": int(max_turns) * 4 + 64,
    }
    if resume:
        snapshot = app.get_state(config)
        stored = (snapshot.values or {}).get("protocol") if snapshot else None
        # v3.6: a checkpoint frozen under a retired profile (bposd_structural
        # _v3_5, cascade_structural_v3_6) is refused by NAME with the remedy
        # before the generic diff.
        assert_resumable_protocol(stored)
        validate_resume_protocol(stored, protocol)
        overrides = resume_runtime_overrides(stored or {}, protocol)
        if overrides:
            # Runtime keys (worker count) follow this launch, not the
            # checkpoint: the graph reads them from state["protocol"], so the
            # checkpointed protocol is updated in place (the measurement keys
            # were just validated identical) and the on-disk snapshot mirrors
            # it with a history line for the audit trail.
            merged = dict(stored or {})
            history = list(merged.get("runtime_overrides") or [])
            # state["turn"] is the checkpoint's current turn index, which is
            # the LAST COMPLETED turn after a clean link end and the turn
            # being re-run after a strict abort. The count of audited turn
            # directories is unambiguous in both cases: it is the index of
            # the first turn that will run under the new values.
            audited = len(list((run_dir / "turns").glob("turn_*/turn_audit.json")))
            history.append({
                "turn": int((snapshot.values or {}).get("turn") or 0),
                "applies_from_turn": int(audited),
                "from": {key: (stored or {}).get(key) for key in overrides},
                "to": dict(overrides),
            })
            merged.update(overrides)
            merged["runtime_overrides"] = history
            print(
                f"[campaign] resume: runtime override {overrides} (recorded in "
                "protocol_snapshot.json runtime_overrides)",
                flush=True,
            )
            app.update_state(config, {"protocol": merged})
            (run_dir / "protocol_snapshot.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        final_state = app.invoke(None, config=config, durability="sync")
    else:
        initial: CampaignState = {
            "run_id": run_id,
            "turn": 0,
            "run_dir": str(run_dir),
            "spec": json.loads(spec.model_dump_json()),
            "protocol": protocol,
            "hof_summary": [],
            "structural_proxy_elites": [],
            "unmeasurable_hashes": [],
            # v3.6 B5 (needs the CampaignState field to be checkpointed; the
            # run-level decode_infeasible_hashes.json is the fallback).
            "decode_infeasible_hashes": [],
            "baseline_summary": {},
            "budget": {
                "turns_used": 0,
                "wallclock_s": 0.0,
                "tokens_used": 0,
                "slow_shots_used": 0,
            },
        }
        final_state = app.invoke(initial, config=config, durability="sync")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stop": final_state.get("stop"),
        "report": final_state.get("report"),
        "budget": final_state.get("budget"),
        "hall_of_fame": final_state.get("hof_summary"),
        "structural_proxy_elites": final_state.get("structural_proxy_elites"),
    }
