"""End-to-end campaign orchestration across fast and slow verification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omniqec_agentic.config import HardwareConfig, RunConfig, VerificationConfig
from omniqec_agentic.funsearch import run_funsearch
from omniqec_agentic.evaluation import (
    evaluate_promotion_queue,
    make_slow_aware_prompt_context,
    select_resident_baseline,
)
from omniqec_agentic.memory.fingerprints import TWIN_GATE_ID


@dataclass(frozen=True)
class AgenticCycleSummary:
    run_id: str
    out_dir: str
    n_budget: int
    hardware: str
    provider: str
    generations: int
    proposal_count: int
    promote_every: int | None
    promote_top: int
    slow_all_archive: bool
    slow_limit: int | None
    slow_shots: int
    slow_timeout_s: float
    status: str
    funsearch_summary: dict[str, Any]
    slow_summary: dict[str, Any] | None
    slow_context_summary: dict[str, Any] | None
    internal_promotions: list[dict[str, Any]]
    artifact_paths: dict[str, str]
    schema: str = "agentic_cycle_summary"
    # S3 ablation (2026-10-23): "full" | "no_slow" | "no_feedback". Under the
    # two ablation modes the cycle runs NO slow stage (slow_summary is None
    # and the status is still "ok").
    ablation: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cycle_uses_seed_programs(use_seeds: bool, search_method: str | None) -> bool:
    """Whether a cycle evaluates the hand-written seed programs (examples/seed_programs.py).

    Paper alignment (2026-11-10): the seed programs initialise the discovery memory of an LLM discovery run
    (appendix D1) and are not part of the search baselines -- the BO pool is formed from the randomly generated
    proposals (appendix D4) and the random baseline's attempts are all random samples with no discovery memory
    (appendix D5). A search baseline therefore never runs them."""

    return bool(use_seeds) and str(search_method or "llm") == "llm"


def run_agentic_cycle(
    config: RunConfig,
    *,
    provider: str,
    generations: int,
    promote_every: int | None = None,
    slow_limit: int | None = None,
    slow_p_values: tuple[float, ...] | None = None,
    slow_shots: int = 1000,
    slow_batch_size: int = 200,
    slow_max_errors: int | None = 1000,
    slow_all_archive: bool = False,
    slow_workers: int = 1,
    slow_shot_shards: int = 1,
    rng_seeds_per_program: int = 1,
    use_seeds: bool = True,
    seed_sources: list[str] | None = None,
    save_every: int = 10,
    prefer_coloring: bool = True,
    racing: Any = None,
    racing_thresholds: Any = None,
    prompt_context_overlay: dict[str, Any] | None = None,
    promotion_priority: list[str] | None = None,
    promotion_measured: set[str] | None = None,
    baseline_already_measured: bool = False,
    require_full_llm_batch: bool = False,
    structural_elites: list[dict[str, Any]] | None = None,
    structural_registry_path: str | None = None,
    evidence_rows: list[dict[str, Any]] | None = None,
    turn: int = 0,
    promotion_max_check_weight: int | None = None,
    exact_check_weight: int | None = None,
    ablation: str = "full",
    demoted_families: list[str] | None = None,
    search_method: str = "llm",
    baseline_context: dict[str, Any] | None = None,
) -> AgenticCycleSummary:
    """Run one complete generation-to-slow-feedback cycle.

    ``ablation`` (S3, 2026-10-23): "no_slow" and "no_feedback" run the fast
    loop only -- no slow queue is built and no candidate is measured during
    the search; the frozen final stage measures the structural front after
    the last turn (workflow.graph._final_entries).

    ``search_method`` (2026-11-06): "random" / "bo" are the search baselines
    without an LLM; their slow queue is the baseline's own rule over the run's
    cumulative legal pool (baselines/selection.py, ``baseline_context`` from
    the campaign graph) instead of the structural + runtime channels.

    This is deliberately an orchestration layer. It does not change FunSearch
    scoring, archive niche rules, or slow verifier semantics.
    ``promotion_priority`` (validated code-hash prefixes from the orchestrator
    TurnPlan) reorders the final slow queue: agent-promoted candidates measure
    first, in plan order; remaining slots fill from the Q-ordered queue.
    """

    out_dir = Path(config.out_dir)
    fast_dir = out_dir / "fast"
    slow_dir = out_dir / "slow"
    slow_context_path = slow_dir / "prompt_context_slow_aware.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    promote_every = int(promote_every or 0)
    internal_promotions: list[dict[str, Any]] = []
    # The resident BB baseline is measured once per campaign: after any stage
    # has measured it (organic seed elite or injection), later stages skip the
    # injection — analyze aggregates across stages, so one measurement serves
    # every comparison. Which anchor (BB72/BB144/none) depends on the budget:
    # total-budget campaigns only anchor at 144 and 288 where a BB fits.
    resident_baseline = select_resident_baseline(config.n_budget)
    # Under the frozen protocol the baseline is a campaign constant: measuring
    # it once (turn 0) suffices, and the caller (campaign graph) holds the
    # result in state. Re-measuring every turn cost ~1.6 CPU-hours/turn of
    # beam decoding for identical information (observed on b144 v1/v2).
    baseline_state = {"measured": resident_baseline is None or bool(baseline_already_measured)}
    slow_verification = _slow_verification_config(
        config,
        slow_p_values=slow_p_values,
        slow_shots=slow_shots,
        slow_max_errors=slow_max_errors,
    )

    fast_config = type(config)(
        run_id=config.run_id,
        n_budget=config.n_budget,
        hardware=HardwareConfig(
            target=config.hardware.target,
            connectivity_rule=config.hardware.connectivity_rule,
            notes=config.hardware.notes,
        ),
        out_dir=str(fast_dir),
        llm=config.llm,
        verification=config.verification,
        proposal_count=config.proposal_count,
        allowed_families=config.allowed_families,
        deformation_mode=config.deformation_mode,
        feedback_mode=config.feedback_mode,   # 消融开关要透传,否则重建时丢回 both
    )

    promotion_callback = None
    if promote_every > 0:
        def promotion_callback(request: dict[str, Any]) -> dict[str, Any]:
            gen = int(request.get("gen") or 0)
            result = _run_slow_stage(
                stage_id=f"promote_g{gen:04d}",
                queue_path=_selected_slow_queue(
                    request,
                    slow_all_archive=slow_all_archive,
                    structural_config=config.verification.structural,
                    priority=list(promotion_priority or []),
                    measured_hashes=promotion_measured,
                    registry_path=structural_registry_path,
                    out_dir=out_dir,
                ),
                prompt_context_path=str(request["prompt_context"]),
                out_dir=slow_dir / f"promote_g{gen:04d}",
                slow_context_path=slow_dir / f"promote_g{gen:04d}" / "prompt_context_slow_aware.json",
                verification=slow_verification,
                hardware=config.hardware.target,
                slow_limit=slow_limit,
                slow_batch_size=slow_batch_size,
                slow_workers=slow_workers,
                slow_shot_shards=slow_shot_shards,
                prefer_coloring=prefer_coloring,
                include_baseline=not baseline_state["measured"],
                resident_baseline=resident_baseline,
                deformation_mode=config.deformation_mode.value,
                racing=racing,
                racing_thresholds=racing_thresholds,
            )
            if _stage_measured_baseline(result, resident_baseline):
                baseline_state["measured"] = True
            internal_promotions.append(_public_promotion_result(result))
            return result

    funsearch_summary = run_funsearch(
        fast_config,
        generations=int(generations),
        provider=provider,
        seed_sources=seed_sources,
        use_seeds=cycle_uses_seed_programs(use_seeds, search_method),
        save_every=save_every,
        prompt_context_overlay=prompt_context_overlay,
        promote_every=promote_every if promote_every > 0 else None,
        promotion_callback=promotion_callback,
        rng_seeds_per_program=int(rng_seeds_per_program),
        require_full_llm_batch=bool(require_full_llm_batch),
        structural_elites=structural_elites,
        measured_hashes=promotion_measured,
        evidence_rows=evidence_rows,
        turn=int(turn),
    )
    fast_artifacts = funsearch_summary.artifact_paths
    promotion_queue = fast_artifacts.get("promotion_queue")
    all_archive_queue = fast_artifacts.get("all_archive_queue")
    structural_queue = fast_artifacts.get("structural_slow_queue")
    slow_queue_composition: dict[str, Any] | None = None
    ablation = str(ablation or "full")
    if ablation != "full":
        # S3 ablation: the search measures nothing. No queue is built, so no
        # slow stage, no slow feedback overlay and no evidence row can exist
        # for the next turn -- by construction, not by an empty result.
        slow_queue = None
    elif str(search_method or "llm") != "llm":
        from omniqec_agentic.baselines.selection import build_baseline_slow_queue

        slow_queue, slow_queue_composition = build_baseline_slow_queue(
            context=dict(baseline_context or {}),
            cycle_ledger_path=fast_artifacts.get("structural_ledger"),
            out_dir=out_dir,
            measured_hashes=promotion_measured,
            gate=promotion_max_check_weight,
            cap=int(config.verification.structural.slow_candidate_cap),
            resident_baseline=resident_baseline,
        )
    elif slow_all_archive and all_archive_queue:
        # Explicit "measure everything" mode (debug/offline sweeps); the
        # 4-slot protocol does not apply.
        slow_queue = all_archive_queue
    else:
        slow_queue, slow_queue_composition = _build_slow_queue(
            structural_queue=structural_queue,
            promotion_queue=promotion_queue,
            all_archive_queue=all_archive_queue,
            registry_rows=_jsonl_rows(fast_artifacts.get("structural_ledger")),
            registry_path=structural_registry_path,
            priority=list(promotion_priority or []),
            structural_config=config.verification.structural,
            out_dir=out_dir,
            measured_hashes=promotion_measured,
            max_check_weight_gate=promotion_max_check_weight,
            exact_check_weight_gate=exact_check_weight,
            demoted_families=list(demoted_families or []),
        )
    prompt_context = fast_artifacts.get("prompt_context")

    slow_summary: dict[str, Any] | None = None
    slow_context_summary: dict[str, Any] | None = None
    if slow_queue and prompt_context:
        needs_final_slow = promote_every <= 0 or not internal_promotions
        if promote_every > 0 and int(generations) > 0 and int(generations) % promote_every != 0:
            needs_final_slow = True
        if needs_final_slow:
            final_stage = _run_slow_stage(
                stage_id="final",
                queue_path=slow_queue,
                prompt_context_path=prompt_context,
                out_dir=slow_dir if promote_every <= 0 else slow_dir / "final",
                slow_context_path=slow_context_path if promote_every <= 0 else slow_dir / "final" / "prompt_context_slow_aware.json",
                verification=slow_verification,
                hardware=config.hardware.target,
                slow_limit=slow_limit,
                slow_batch_size=slow_batch_size,
                slow_workers=slow_workers,
                slow_shot_shards=slow_shot_shards,
                prefer_coloring=prefer_coloring,
                include_baseline=not baseline_state["measured"],
                resident_baseline=resident_baseline,
                deformation_mode=config.deformation_mode.value,
                racing=racing,
                racing_thresholds=racing_thresholds,
            )
            if promote_every > 0:
                internal_promotions.append(_public_promotion_result(final_stage))
            slow_summary = final_stage["slow_summary"]
            slow_context_summary = final_stage["slow_context_summary"]
        elif internal_promotions:
            slow_summary = internal_promotions[-1].get("slow_summary")
            slow_context_summary = internal_promotions[-1].get("slow_context_summary")

    status = "ok"
    if slow_summary is None:
        # Under an ablation the absence of a slow stage IS the protocol.
        status = "ok" if ablation != "full" else "missing_slow_inputs"
    elif slow_summary.get("status") not in {"ok", "empty_queue"}:
        status = str(slow_summary.get("status") or "slow_error")
    artifact_paths = {
        "fast_dir": str(fast_dir),
        "slow_dir": str(slow_dir),
        "fast_summary": fast_artifacts.get("summary", ""),
        "promotion_queue": promotion_queue or "",
        "all_archive_queue": all_archive_queue or "",
        "structural_proxy_elites": fast_artifacts.get("structural_proxy_elites", ""),
        "structural_ledger": fast_artifacts.get("structural_ledger", ""),
        "structural_slow_queue": structural_queue or "",
        "slow_queue_composition": (
            str(out_dir / "slow_queue_composition.json") if slow_queue_composition else ""
        ),
        "slow_queue": slow_queue or "",
        "fast_prompt_context": prompt_context or "",
        "slow_summary": (slow_summary or {}).get("artifact_paths", {}).get("summary", "") if slow_summary else "",
        "slow_feedback": (slow_summary or {}).get("artifact_paths", {}).get("slow_feedback", "") if slow_summary else "",
        "slow_rankings": (slow_summary or {}).get("artifact_paths", {}).get("rankings", "") if slow_summary else "",
        "slow_distance_evidence": (slow_summary or {}).get("artifact_paths", {}).get("distance_evidence", "") if slow_summary else "",
        "slow_prompt_context": str(slow_context_summary.get("out")) if slow_context_summary else "",
    }
    summary = AgenticCycleSummary(
        run_id=config.run_id,
        out_dir=str(out_dir),
        n_budget=int(config.n_budget),
        hardware=config.hardware.target.value,
        provider=str(provider),
        generations=int(generations),
        proposal_count=int(config.proposal_count),
        promote_every=promote_every if promote_every > 0 else None,
        promote_top=int(config.verification.promote_top),
        slow_all_archive=bool(slow_all_archive),
        slow_limit=slow_limit,
        slow_shots=int(slow_shots),
        slow_timeout_s=float(slow_verification.circuit_timeout_s),
        status=status,
        funsearch_summary=funsearch_summary.to_dict(),
        slow_summary=slow_summary,
        slow_context_summary=slow_context_summary,
        internal_promotions=internal_promotions,
        artifact_paths=artifact_paths,
        ablation=ablation,
    )
    (out_dir / "agentic_cycle_summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _jsonl_rows(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _build_slow_queue(
    *,
    structural_queue: str | None,
    promotion_queue: str | None,
    all_archive_queue: str | None,
    priority: list[str],
    structural_config: Any,
    out_dir: Path,
    measured_hashes: set[str] | None = None,
    registry_rows: list[dict[str, Any]] | None = None,
    registry_path: str | None = None,
    max_check_weight_gate: int | None = None,
    exact_check_weight_gate: int | None = None,
    demoted_families: list[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """The frozen slow queue (v3.2 §4.3; v3.7 quota 2S+4R = 6, was 4S+2R).

    ``demoted_families`` (v3.8): the campaign graph's endpoint-demotion
    verdict for this turn. The structural queue handed in was already built
    without these families (memory.structural_slow_queue); here the verdict
    and the profile's endpoint rules are RECORDED in the composition so a
    reader of the artifact sees why a family took no structural slot.

    ``max_check_weight_gate`` (2026-09-29, runtime key
    promotion_max_check_weight, user decision for the large-code tier): a
    candidate whose max stabilizer weight exceeds the gate is not sent to the
    slow loop; its slot goes to the next candidate of the same channel and
    the rejection is recorded in the composition (``weight_gate``). Evidence
    (763 tier-2/3 slow measurements, b180-b360, 2026-09-29): every w >= 9
    candidate measured plpr >= 2.9e-4 (median 5.4e-3, 50-10,000x the point's
    best) at 10-100x the decode cost of the good codes; 0 of 413 codes with
    plpr < 2e-4 had w >= 9. Rows without a recorded weight are never gated.

    The first `structural_slow_slots` slots are the machine's structural picks
    (v3.7: the high-k and high-d_search endpoints of the weight-aware front)
    produced by the canonical reducer; they are not TurnPlan promotions and
    are not subject to acquisition ordering, plpr, degraded_* or policy
    quarantine. The remaining `runtime_slow_slots` slots are the
    runtime/acquisition channel: each legal named candidate CONSUMES one (a
    named promotion is never an extra slot on top of the cap, which is how
    the old merge let a plan grow the queue), and the archive head -- v3.7:
    ordered by expected plpr under the measured-evidence model, Q_eff before
    the first measurement -- fills whatever the plan did not name.

    Candidates already slow-measured in this campaign are filtered
    unconditionally — the old code applied that set only when a plan happened
    to carry promotions, so an ordinary candidate could be re-measured on a
    later turn under the same frozen protocol."""

    measured = {str(item) for item in (measured_hashes or set())}
    cap = int(structural_config.slow_candidate_cap)
    structural_slots = int(structural_config.structural_slow_slots)
    runtime_slots = int(structural_config.runtime_slow_slots)
    structural_rows = _jsonl_rows(structural_queue)
    q_rows = _jsonl_rows(promotion_queue)
    archive_rows = _jsonl_rows(all_archive_queue)

    # Named-promotion registry (audit fix #4): a legal TurnPlan promotion may
    # name a candidate the policy saw LAST turn. This turn's queues only carry
    # this turn's rows, so resolution also searches the cross-turn structural
    # ledger — the small in-memory rows first, then a STREAMED scan of the
    # append-only run-level store (the full store is deliberately never loaded
    # into memory; scale audit fix #4).
    by_hash: dict[str, dict[str, Any]] = {}
    for row in q_rows + archive_rows + structural_rows + list(registry_rows or []):
        code_hash = str(row.get("code_hash") or "")
        if code_hash and code_hash not in by_hash:
            by_hash[code_hash] = row
    # Twin identities of already-measured codes (v3.6 B3: code_hash + pynauty
    # canonical_hash, NEVER wl_hash), recovered from every row in scope: the
    # hash filter alone let a permutation-equivalent rediscovery (different
    # code_hash, same graph) be measured again on a later turn, while the
    # 1-WL gate merged twelve distinct C4xD5 codes into one and starved the
    # slow loop after the first of them measured. wl_hash of measured codes is
    # counted for the audit trail only.
    measured_canonical: set[str] = {
        str(row.get("canonical_hash"))
        for row in by_hash.values()
        if row.get("canonical_hash") and str(row.get("code_hash") or "") in measured
    }
    measured_wl: set[str] = {
        str(row.get("wl_hash"))
        for row in by_hash.values()
        if row.get("wl_hash") and str(row.get("code_hash") or "") in measured
    }
    # Fig.3 v2 feedback: which measured code a rejected twin is a relabelling of.
    canonical_to_measured: dict[str, str] = {
        str(row.get("canonical_hash")): str(row.get("code_hash"))
        for row in by_hash.values()
        if row.get("canonical_hash") and str(row.get("code_hash") or "") in measured
    }
    prefixes = [str(item) for item in list(priority)[:runtime_slots] if item]
    if registry_path and Path(registry_path).exists():
        for line in Path(registry_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            code_hash = str(row.get("code_hash") or "")
            if not code_hash:
                continue
            if code_hash in measured:
                canonical = row.get("canonical_hash")
                if canonical:
                    measured_canonical.add(str(canonical))
                    canonical_to_measured.setdefault(str(canonical), code_hash)
                wl = row.get("wl_hash")
                if wl:
                    measured_wl.add(str(wl))
            if code_hash not in by_hash and any(
                code_hash.startswith(prefix) for prefix in prefixes
            ):
                by_hash[code_hash] = row

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_canonical: set[str] = set(measured_canonical)
    twin_rejected: set[str] = set()
    gate = None if max_check_weight_gate is None else int(max_check_weight_gate)
    # Fig.4 c (2026-10-23): exact contract at the queue too (defence in depth;
    # the construction check already keeps other weights out of the archive).
    exact = None if exact_check_weight_gate is None else int(exact_check_weight_gate)
    weight_gated: dict[str, dict[str, Any]] = {}
    # Fig.3 v2 feedback: the candidates the measured / twin gates kept out of
    # the queue, with what they duplicate, so the next turn's generator and
    # reflector are told WHY a turn measured little or nothing.
    measured_rejected: dict[str, dict[str, Any]] = {}
    twin_rejected_rows: dict[str, dict[str, Any]] = {}

    def _rejected_row(row: dict[str, Any], channel: str) -> dict[str, Any]:
        return {
            "code_hash": str(row.get("code_hash") or "")[:12],
            "program_id": row.get("program_id"),
            "family": row.get("family"),
            "n": row.get("n") if row.get("n") is not None else row.get("n_data"),
            "k": row.get("k"),
            "d_search": row.get("d_search") or row.get("d_proxy"),
            "max_check_weight": row.get("max_check_weight"),
            "channel": channel,
        }

    def _take(row: dict[str, Any] | None, channel: str) -> bool:
        if row is None or len(selected) >= cap:
            return False
        code_hash = str(row.get("code_hash") or "")
        if not code_hash or code_hash in seen:
            return False
        if code_hash in measured:
            if code_hash not in measured_rejected:
                measured_rejected[code_hash] = _rejected_row(row, channel)
            return False
        if gate is not None or exact is not None:
            raw_w = row.get("max_check_weight")
            try:
                weight = None if raw_w is None else int(raw_w)
            except (TypeError, ValueError):
                weight = None
            if weight is not None and (
                (gate is not None and weight > gate) or (exact is not None and weight != exact)
            ):
                if code_hash not in weight_gated:
                    weight_gated[code_hash] = {
                        "code_hash": code_hash[:12],
                        "max_check_weight": weight,
                        "k": row.get("k"),
                        "d_search": row.get("d_search") or row.get("d_proxy"),
                        "family": row.get("family"),
                        "channel": channel,
                        "structural_slot_reason": row.get("structural_slot_reason"),
                    }
                return False
        canonical = str(row.get("canonical_hash") or "")
        if canonical and canonical in used_canonical:
            twin_rejected.add(code_hash)
            if code_hash not in twin_rejected_rows:
                item = _rejected_row(row, channel)
                twin_of = canonical_to_measured.get(canonical)
                # a twin of a code selected earlier THIS turn (not measured yet)
                # has no measured target; the feedback then names the queue.
                item["twin_of_code_hash"] = twin_of[:12] if twin_of else None
                item["twin_of"] = "measured" if twin_of else "selected_this_turn"
                twin_rejected_rows[code_hash] = item
            return False
        seen.add(code_hash)
        if canonical:
            used_canonical.add(canonical)
        item = dict(row)
        channels = list(item.get("selection_channels") or [])
        if channel not in channels:
            channels.append(channel)
        item["selection_channels"] = channels
        selected.append(item)
        return True

    for row in structural_rows:
        if len(selected) >= structural_slots:
            break
        _take(row, "structural")
    structural_taken = len(selected)
    structural_shortfall = max(0, structural_slots - structural_taken)

    # Runtime slots: up to `runtime_slots` legal named promotions (plan order),
    # each consuming one slot; the frozen runtime head fills the rest.
    named_taken = 0
    for prefix in list(priority)[:runtime_slots]:
        matches = [h for h in by_hash if h.startswith(str(prefix))]
        if len(matches) == 1 and _take(by_hash[matches[0]], "runtime_named"):
            named_taken += 1
    runtime_head_taken = 0
    for row in q_rows:
        if named_taken + runtime_head_taken >= runtime_slots or len(selected) >= cap:
            break
        if _take(row, "runtime"):
            runtime_head_taken += 1

    # Backfill: keep the S:R quota whenever the pools allow it, and record
    # when they do not instead of silently shipping a differently-shaped queue.
    for row in structural_rows:
        if len(selected) >= cap:
            break
        _take(row, "structural")
    for row in q_rows + archive_rows:
        if len(selected) >= cap:
            break
        _take(row, "runtime")

    composition = {
        "schema": "slow_queue_composition",
        "cap": cap,
        "structural_slots": structural_slots,
        "runtime_slots": int(structural_config.runtime_slow_slots),
        "structural_taken": structural_taken,
        "structural_slot_shortfall": structural_shortfall,
        "named_promotion_used": bool(named_taken),
        "named_promotions_taken": int(named_taken),
        "runtime_head_taken": int(runtime_head_taken),
        "named_promotion_candidates": len(priority),
        "selected": [
            {
                "code_hash": str(row.get("code_hash") or "")[:12],
                "selection_channels": row.get("selection_channels"),
                "structural_slot_reason": row.get("structural_slot_reason"),
                "k": row.get("k"),
                "d_search": row.get("d_search") or row.get("d_proxy"),
            }
            for row in selected
        ],
        "measured_hashes_filtered": len(measured),
        # v3.6 B3 twin gate = code_hash + canonical_hash. `measured_wl_seen`
        # is informational only (WL never gates); the pre-v3.6 key
        # `measured_wl_filtered` is kept for readers of old compositions but
        # now reports the canonical count, i.e. what actually filtered.
        "twin_gate_id": TWIN_GATE_ID,
        "measured_canonical_filtered": len(measured_canonical),
        "measured_wl_filtered": len(measured_canonical),
        "measured_wl_seen": len(measured_wl),
        "twin_rejections": len(twin_rejected),
        # Fig.3 v2 feedback: the distinct candidates the three pools offered
        # this turn, and the rows the measured / twin gates removed.
        "candidates_seen": len(
            {str(row.get("code_hash") or "") for row in structural_rows + q_rows + archive_rows if row.get("code_hash")}
        ),
        "measured_rejected_rows": list(measured_rejected.values()),
        "twin_rejected_rows": list(twin_rejected_rows.values()),
        # 2026-09-29 runtime weight gate: the candidates it kept out of the
        # slow loop this turn (never measured; their slots were backfilled).
        "weight_gate": {
            "max_check_weight": gate,
            # the exact key appears only under the Fig.4 contract: every
            # earlier composition file keeps its shape
            **({"exact_check_weight": exact} if exact is not None else {}),
            "rejected": list(weight_gated.values()),
        },
        # v3.8 endpoint family rules (profile fields + this turn's verdict).
        "endpoint_rules": {
            "family_distinct": bool(getattr(structural_config, "endpoint_family_distinct", False)),
            "demotion_strikes": int(getattr(structural_config, "endpoint_demotion_strikes", 0) or 0),
            "demotion_factor": float(getattr(structural_config, "endpoint_demotion_factor", 10.0) or 10.0),
            "demoted_families": sorted(str(item) for item in (demoted_families or [])),
        },
    }
    out_path = Path(out_dir) / "slow_queue.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected)
        + ("\n" if selected else ""),
        encoding="utf-8",
    )
    (Path(out_dir) / "slow_queue_composition.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The path is returned even when the queue is EMPTY. Falling back to the
    # unfiltered promotion queue here is what let a turn re-measure candidates
    # the campaign had already measured: under the frozen protocol a
    # re-measurement adds no information, so an empty queue is the correct
    # outcome of "everything eligible has been measured", not a failure to
    # produce one.
    return str(out_path), composition


def _selected_slow_queue(
    request: dict[str, Any],
    *,
    slow_all_archive: bool,
    structural_config: Any,
    priority: list[str],
    measured_hashes: set[str] | None,
    out_dir: Path,
    registry_path: str | None = None,
) -> str:
    """Queue for an intra-generation promotion stage.

    It runs the SAME 4-slot rule as the end-of-turn queue: the structural
    slots must not exist only on the path a particular flag happens to take."""

    if slow_all_archive and request.get("all_archive_queue"):
        return str(request["all_archive_queue"])
    stage_dir = Path(out_dir) / "queues" / f"g{int(request.get('gen') or 0):04d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    built, _ = _build_slow_queue(
        structural_queue=request.get("structural_slow_queue"),
        promotion_queue=request.get("promotion_queue"),
        all_archive_queue=request.get("all_archive_queue"),
        registry_rows=_jsonl_rows(request.get("structural_ledger")),
        registry_path=registry_path,
        priority=list(priority),
        structural_config=structural_config,
        out_dir=stage_dir,
        measured_hashes=measured_hashes,
    )
    return built


def _slow_verification_config(
    config: RunConfig,
    *,
    slow_p_values: tuple[float, ...] | None,
    slow_shots: int,
    slow_max_errors: int | None,
) -> VerificationConfig:
    import dataclasses

    values = tuple(slow_p_values or config.verification.noise.values)
    noise = dataclasses.replace(config.verification.noise, values=values)
    # dataclasses.replace so every other field (bases, d_cap, dsqrtn_floor,
    # max_check_weight, ler_ci_z, ...) survives into the slow stage — the old
    # explicit field list silently reverted them to defaults.
    return dataclasses.replace(
        config.verification,
        p_values=values,
        noise=noise,
        shots=int(slow_shots),
        max_errors=slow_max_errors,
    )


def _run_slow_stage(
    *,
    stage_id: str,
    queue_path: str,
    prompt_context_path: str,
    out_dir: Path,
    slow_context_path: Path,
    verification: VerificationConfig,
    hardware: Any,
    slow_limit: int | None,
    slow_batch_size: int,
    slow_workers: int = 1,
    slow_shot_shards: int = 1,
    prefer_coloring: bool,
    include_baseline: bool = True,
    resident_baseline: tuple[dict[str, Any], str] | None = None,
    deformation_mode: str | None = None,
    racing: Any = None,
    racing_thresholds: Any = None,
) -> dict[str, Any]:
    slow_summary = evaluate_promotion_queue(
        queue_path=queue_path,
        out_dir=out_dir,
        verification=verification,
        hardware=hardware,
        limit=slow_limit,
        seed=0,
        batch_size=int(slow_batch_size),
        prefer_coloring=prefer_coloring,
        workers=int(slow_workers),
        shot_shards=int(slow_shot_shards),
        include_baseline=include_baseline and resident_baseline is not None,
        baseline_spec=resident_baseline[0] if resident_baseline else None,
        baseline_program_id=resident_baseline[1] if resident_baseline else None,
        deformation_mode=deformation_mode,
        racing=racing,
        racing_thresholds=racing_thresholds,
    )
    slow_feedback = (slow_summary.get("artifact_paths") or {}).get("slow_feedback")
    slow_context_summary: dict[str, Any] | None = None
    prompt_context_overlay: dict[str, Any] | None = None
    if slow_feedback:
        slow_context_summary = make_slow_aware_prompt_context(
            prompt_context_path=prompt_context_path,
            slow_feedback_path=slow_feedback,
            out_path=slow_context_path,
        )
        prompt_context_overlay = json.loads(slow_context_path.read_text(encoding="utf-8"))

    artifact_paths = {
        "slow_dir": str(out_dir),
        "slow_summary": (slow_summary.get("artifact_paths") or {}).get("summary", ""),
        "slow_results_csv": (slow_summary.get("artifact_paths") or {}).get("results_csv", ""),
        "slow_feedback": (slow_summary.get("artifact_paths") or {}).get("slow_feedback", ""),
        "slow_rankings": (slow_summary.get("artifact_paths") or {}).get("rankings", ""),
        "slow_distance_evidence": (slow_summary.get("artifact_paths") or {}).get("distance_evidence", ""),
        "slow_prompt_context": str(slow_context_path) if slow_context_summary else "",
    }
    return {
        "schema": "agentic_slow_stage",
        "stage_id": str(stage_id),
        "status": str(slow_summary.get("status") or "unknown"),
        "slow_parent_count": int(slow_summary.get("slow_parent_count") or 0),
        "slow_summary": slow_summary,
        "slow_context_summary": slow_context_summary,
        "prompt_context_overlay": prompt_context_overlay,
        "artifact_paths": artifact_paths,
    }


def _stage_measured_baseline(
    result: dict[str, Any],
    resident_baseline: tuple[dict[str, Any], str] | None = None,
) -> bool:
    """True when this slow stage measured the resident reference, either as
    the injected baseline or as an organic candidate with the same code hash."""

    if resident_baseline is None:
        return False
    summary = result.get("slow_summary")
    if not isinstance(summary, dict):
        return False
    try:
        from omniqec_agentic.construction import build_matrices
        from omniqec_agentic.verification.structural import code_hash

        bb_hash = code_hash(*build_matrices(resident_baseline[0]))
    except Exception:
        bb_hash = None
    for candidate in summary.get("candidates") or []:
        if not isinstance(candidate, dict) or candidate.get("status") != "ok":
            continue
        if candidate.get("is_baseline"):
            return True
        if bb_hash and candidate.get("code_hash") == bb_hash:
            return True
    return False


def _public_promotion_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "prompt_context_overlay"
    }
