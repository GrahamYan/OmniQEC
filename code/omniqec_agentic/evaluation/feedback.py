"""Prompt-facing slow feedback: parent selection, guidance, slow-aware prompt
context merge. Mechanically extracted from slow.py; consumed by the fast loop
(baseline rows stay measurement-side only, 03 §11.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import SLOW_SCHEMA_VERSION


def build_slow_feedback(
    *,
    queue_path: str,
    hardware: str | None,
    rankings: list[dict[str, Any]],
    candidate_payloads: list[dict[str, Any]],
    distance_evidence: list[dict[str, Any]] | None = None,
    parent_limit: int = 12,
) -> dict[str, Any]:
    """Build prompt-facing memory from slow verification results.

    The injected resident baseline is measurement-side only: its rows are
    filtered out of everything the proposal LLM can see (best_results,
    guidance, parents). An organic archive elite that happens to BE the BB
    code (hash-dedup case) is legitimate search state and stays visible."""

    prompt_rankings = [row for row in rankings if not row.get("is_baseline")]
    candidate_by_program = {
        str(payload.get("program_id")): payload
        for payload in candidate_payloads
        if payload.get("program_id")
    }
    slow_parent_specs: list[dict[str, Any]] = []
    used: set[str] = set()
    for rank, row in enumerate(prompt_rankings, start=1):
        if len(slow_parent_specs) >= max(0, int(parent_limit)):
            break
        if row.get("status") != "ok" or row.get("per_logical_per_round") is None:
            continue
        program_id = str(row.get("program_id") or "")
        payload = candidate_by_program.get(program_id)
        if not payload:
            continue
        promotion = payload.get("promotion_record") if isinstance(payload.get("promotion_record"), dict) else {}
        identity = str(row.get("code_hash") or promotion.get("code_hash") or program_id)
        if not identity or identity in used:
            continue
        parent = slow_parent_record(
            row=row,
            payload=payload,
            promotion=promotion,
            slow_rank=rank,
        )
        if parent is None:
            continue
        used.add(identity)
        slow_parent_specs.append(parent)

    return {
        "schema": "slow_feedback",
        "schema_version": SLOW_SCHEMA_VERSION,
        "source_queue_path": str(queue_path),
        "hardware": hardware,
        # v3.2 §5.1/§6: this module produces RUNTIME FACTS and runtime
        # candidates. It does not own lineage: parent order, promotion and
        # final eligibility are decided by the machine's retention rules, and
        # a "ranking rule" published here used to be copied into the prompt as
        # a global instruction.
        "evidence_model": {
            "runtime_measurement": "paper A4/A5 per_logical_per_round under the frozen BPOSD protocol at the anchor point",
            "distance_evidence": "exact code/circuit distance when configured, plus sec d_circ_ext probes and optional LER-slope distance estimates",
            "note": "these are measurements of how a code RAN; structural retention is decided separately from (k, d_search) and is not visible here.",
        },
        "runtime_order_rule": "per_logical_per_round_then_fast_Q (runtime channel only; not a lineage rule)",
        # v3.2 §5.1 + audit fix #6: the prompt-facing FACTS list is in stable
        # code-hash order. rank_slow_rows' plpr order is the runtime ledger's
        # own ordering (rankings.json, HoF) — but an ordered list shown to the
        # generator is an implicit ranking instruction even with no verb
        # attached, so the facts are re-keyed by hash before anything
        # prompt-facing consumes them.
        "fact_order": "stable_code_hash (channel-neutral; not a quality order)",
        "result_count": len(prompt_rankings),
        "runtime_candidate_count": len(slow_parent_specs),
        "slow_parent_count": len(slow_parent_specs),
        # v3.4.1: only completed measurements are facts. A decode_infeasible
        # (or any other non-ok) row carries plpr None and rendered in the
        # generator prompt as an unexplained per_logical_per_round=unknown
        # line; its terminal verdict lives in the runtime ledger (rankings,
        # slow_summary), not in the prompt. Hash order and fact_order stay.
        "best_results": sorted(
            (row for row in prompt_rankings if row.get("status") == "ok"),
            key=lambda row: (
                str(row.get("code_hash") or ""),
                str(row.get("program_id") or ""),
            ),
        )[: max(0, int(parent_limit))],
        "distance_evidence": [
            row for row in (distance_evidence or []) if not row.get("is_baseline")
        ],
        "slow_parent_specs": slow_parent_specs,
        "proposal_guidance": slow_guidance(
            prompt_rankings,
            [row for row in (distance_evidence or []) if not row.get("is_baseline")],
        ),
    }


def make_slow_aware_prompt_context(
    *,
    prompt_context_path: str | Path,
    slow_feedback_path: str | Path,
    out_path: str | Path,
    parent_limit: int = 12,
) -> dict[str, Any]:
    """Merge slow feedback into an existing prompt_context artifact."""

    prompt_path = Path(prompt_context_path)
    feedback_path = Path(slow_feedback_path)
    context = json.loads(prompt_path.read_text(encoding="utf-8"))
    slow_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not isinstance(context, dict):
        raise ValueError("prompt context must be a JSON object")
    if not isinstance(slow_feedback, dict):
        raise ValueError("slow feedback must be a JSON object")

    slow_parents = [
        parent
        for parent in list(slow_feedback.get("slow_parent_specs") or [])
        if isinstance(parent, dict)
    ]
    # v3.2 §4.5/§6: the slow stage no longer owns parent order. It publishes
    # this turn's measured codes as RUNTIME CANDIDATES; the cross-turn 8
    # structural + 4 runtime quota is applied one layer up, by the campaign
    # graph, and putting a runtime-first prefix here silently overrode it.
    merged = list(context.get("next_parent_specs") or [])
    context["schema"] = "prompt_context_slow_aware"
    context["schema_version"] = SLOW_SCHEMA_VERSION
    context["runtime_parent_candidates"] = slow_parents
    context["slow_feedback"] = slow_feedback
    history_policy = dict(context.get("history_policy") or {})
    proposal_agent_sees = list(history_policy.get("proposal_agent_sees") or [])
    if "slow_feedback" not in proposal_agent_sees:
        proposal_agent_sees.insert(0, "slow_feedback")
    history_policy["proposal_agent_sees"] = proposal_agent_sees
    full_history_artifacts = list(history_policy.get("full_history_artifacts") or [])
    for artifact in (
        "slow_feedback.json",
        "slow_parent_specs.json",
        "rankings.json",
        "distance_evidence.json",
    ):
        if artifact not in full_history_artifacts:
            full_history_artifacts.append(artifact)
    history_policy["full_history_artifacts"] = full_history_artifacts
    context["history_policy"] = history_policy
    campaign_feedback = dict(context.get("campaign_feedback") or {})
    campaign_feedback["slow_feedback"] = slow_feedback
    campaign_feedback["runtime_parent_candidates"] = slow_parents
    guidance = list(campaign_feedback.get("proposal_guidance") or [])
    for item in slow_feedback.get("proposal_guidance") or []:
        if item not in guidance:
            guidance.append(item)
    campaign_feedback["proposal_guidance"] = guidance
    context["campaign_feedback"] = campaign_feedback

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "slow_aware_prompt_context_summary",
        "schema_version": SLOW_SCHEMA_VERSION,
        "status": "ok",
        "prompt_context": str(prompt_path),
        "slow_feedback": str(feedback_path),
        "out": str(output_path),
        "parent_count": len(merged),
        "slow_parent_count": len(slow_parents),
    }


def slow_parent_record(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    promotion: dict[str, Any],
    slow_rank: int,
) -> dict[str, Any] | None:
    spec = promotion.get("spec")
    source = promotion.get("construct_source")
    if not isinstance(spec, dict) or not isinstance(source, str) or not source.strip():
        return None
    fast = payload.get("fast") if isinstance(payload.get("fast"), dict) else {}
    structural = payload.get("structural") if isinstance(payload.get("structural"), dict) else {}
    selection_reasons = ["slow_ler_ranked_elite"]
    if row.get("d_circ_ler_estimate_status") == "ok":
        selection_reasons.append("ler_slope_estimate_available")
    parent = {
        "program_id": payload.get("program_id"),
        "family": row.get("family") or fast.get("family") or promotion.get("family"),
        "n": row.get("n") or fast.get("n") or promotion.get("n"),
        "k": row.get("k") or fast.get("k") or promotion.get("k"),
        "d_proxy": row.get("d_proxy") or fast.get("d") or promotion.get("d_proxy"),
        "Q": row.get("fast_Q") or fast.get("Q") or promotion.get("Q"),
        "overhead": promotion.get("overhead") or fast.get("overhead"),
        "max_check_weight": promotion.get("max_check_weight"),
        "max_qubit_degree": promotion.get("max_qubit_degree"),
        "archive_cell": row.get("archive_cell") or promotion.get("archive_cell"),
        "fast_rank": promotion.get("fast_rank"),
        "slow_rank": int(slow_rank),
        "slow_decoder": row.get("decoder"),
        "slow_noise_point": row.get("noise_point"),
        "slow_noise_axis": row.get("noise_axis"),
        "slow_noise_value": row.get("noise_value"),
        "slow_p": row.get("p"),
        "slow_scale": row.get("scale"),
        "slow_block_ler": row.get("block_ler"),
        "slow_per_logical_per_round": row.get("per_logical_per_round"),
        "slow_per_logical_block_ci": row.get("per_logical_block_ci"),
        "slow_basis_logical_error_rate": row.get("basis_logical_error_rate"),
        "d_circ_ler_estimate": row.get("d_circ_ler_estimate"),
        "degraded_estimate": row.get("degraded_estimate"),
        "degraded_estimate_any_decoder": row.get("degraded_estimate_any_decoder"),
        "d_circ_ler_estimate_status": row.get("d_circ_ler_estimate_status"),
        "d_circ_ler_estimate_decoder": row.get("d_circ_ler_estimate_decoder"),
        "exact_code_distance": row.get("exact_code_distance"),
        "exact_code_distance_status": row.get("exact_code_distance_status"),
        "exact_code_distance_certified": row.get("exact_code_distance_certified"),
        "exact_d_circ": row.get("exact_d_circ"),
        "exact_d_circ_status": row.get("exact_d_circ_status"),
        "exact_d_circ_certified": row.get("exact_d_circ_certified"),
        "d_circ_isd_upper_bound": row.get("d_circ_isd_upper_bound"),
        "d_circ_isd_upper_bound_status": row.get("d_circ_isd_upper_bound_status"),
        "d_circ_ext": row.get("d_circ_ext"),
        "degraded_ext": row.get("degraded_ext"),
        "verified_circuit_distance": row.get("verified_circuit_distance"),
        "selection_reasons": selection_reasons,
        "code_hash": row.get("code_hash") or structural.get("code_hash") or promotion.get("code_hash"),
        "spec": spec,
        "rng_seed": promotion.get("rng_seed", 0),
        "source_agent": promotion.get("source_agent"),
        "construct_source": source,
    }
    return parent


def merge_parent_specs(
    preferred: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Runtime-first prefix merge.

    No longer used to build the parent list (v3.2 §4.5: the 8 structural + 4
    runtime quota is applied by the campaign graph). Kept as a plain helper for
    callers that need to concatenate two parent lists by identity."""

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for parent in list(preferred) + list(fallback):
        if len(selected) >= max(0, int(limit)):
            break
        identity = str(parent.get("code_hash") or parent.get("program_id") or "")
        if not identity or identity in used:
            continue
        used.add(identity)
        selected.append(parent)
    return selected


def slow_guidance(
    rankings: list[dict[str, Any]],
    distance_evidence: list[dict[str, Any]],
) -> list[str]:
    """Factual notes about this turn's measurements.

    v3.2 §5.1: no lineage instructions here. The old lines ("choose parents by
    Q", "prefer parents with lower measured plpr", "do not clone them",
    "current best measured parent") were global commands travelling straight
    into the generator prompt, and they described a selection rule the machine
    no longer follows."""

    ok = [row for row in rankings if row.get("status") == "ok"]
    if not ok:
        return ["No circuit-level measurement completed this turn."]
    guidance = [
        f"{len(ok)} candidate(s) completed circuit-level measurement under the "
        "frozen decoder this turn.",
    ]
    degraded_estimates = [
        row
        for row in distance_evidence
        if row.get("degraded_estimate")
        or row.get("degraded_estimate_any_decoder")
        or row.get("degraded_isd")
        or row.get("degraded_ext")
    ]
    if degraded_estimates:
        ids = ", ".join(str(row.get("program_id")) for row in degraded_estimates[:4])
        guidance.append(
            "Circuit-level distance evidence below the heuristic value (hook "
            f"degradation) was recorded for: {ids}."
        )
    return guidance
