"""ver1-aligned FunSearch loop.

The main loop mirrors the historical ver1 shape:
archive/history -> prompt -> provider writes construct(rng) programs ->
sandbox/verifier -> MAP-Elites archive -> next generation.

Lower-level proposal/evaluation commands remain useful debugging tools, but
this module is the primary search loop.
"""

from __future__ import annotations

import json
import re
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import BASELINE_PROVIDER, HARDWARE_PROMPT_PROFILES, CodeFamily, RunConfig
from .construction.groups.catalog import set_group_order_budget
from .examples import SEEDS
from .memory import ArchiveMemory
from .memory.archive import fast_workers_from_env
from .reasoning_fallback import saturated_generation, saturation_message
from .schemas import CandidateProgram, normalize_source_text


_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
MAX_COMPLETION_ATTEMPTS = 3


@dataclass(frozen=True)
class FunSearchSummary:
    run_id: str
    out_dir: str
    n_budget: int
    hardware: str
    provider: str
    generations: int
    seed_count: int
    proposal_count_requested: int
    proposal_count_requested_total: int
    proposal_count: int
    proposal_count_missing_total: int
    completion_attempt_count: int
    accepted_count: int
    archive_cell_count: int
    promotion_queue_count: int
    all_archive_queue_count: int
    structural_reject_count: int
    fast_reject_count: int
    duplicate_count: int
    status_counts: dict[str, int]
    artifact_paths: dict[str, str]
    structural_elite_count: int = 0
    structural_slow_queue_count: int = 0
    rng_seeds_per_program: int = 1
    schema: str = "funsearch_summary"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_funsearch(
    config: RunConfig,
    *,
    generations: int,
    provider: str = "deepseek",
    seed_sources: Iterable[str] | None = None,
    use_seeds: bool = True,
    save_every: int = 10,
    prompt_context_overlay: dict[str, Any] | None = None,
    promote_every: int | None = None,
    promotion_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    rng_seeds_per_program: int = 1,
    require_full_llm_batch: bool = False,
    structural_elites: list[dict[str, Any]] | None = None,
    measured_hashes: set[str] | None = None,
    evidence_rows: list[dict[str, Any]] | None = None,
    turn: int = 0,
) -> FunSearchSummary:
    """Run a cumulative archive/history proposal loop.

    `rng_seeds_per_program > 1` evaluates each proposed program under several
    RNG seeds and lets every distinct produced code compete for its own niche
    (ver1's PROXY_SEEDS breadth idea). Deterministic programs cost almost
    nothing extra: identical codes dedup by hash before the proxy runs."""

    generations = int(generations)
    if generations < 0:
        raise ValueError("generations must be >= 0")
    # v3.6 (docs/06 §3 M3): every validate / build / layer_blocks call of this
    # run resolves lp_nonabelian groups under the budget-derived order cap
    # n_budget // 4 (b288 -> 72, b360 -> 90) instead of a constant.
    set_group_order_budget(int(config.n_budget))
    save_every = max(1, int(save_every))
    proposal_count = max(1, int(config.proposal_count))
    rng_seeds_per_program = max(1, int(rng_seeds_per_program))

    out_dir = Path(config.out_dir)
    programs_dir = out_dir / "programs"
    generations_dir = out_dir / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    programs_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = out_dir / "run_config.json"
    archive_path = out_dir / "archive.json"
    feedback_path = out_dir / "feedback.json"
    prompt_context_path = out_dir / "prompt_context.json"
    promotion_queue_json_path = out_dir / "promotion_queue.json"
    promotion_queue_jsonl_path = out_dir / "promotion_queue.jsonl"
    all_archive_queue_json_path = out_dir / "all_archive_queue.json"
    all_archive_queue_jsonl_path = out_dir / "all_archive_queue.jsonl"
    structural_elites_path = out_dir / "structural_proxy_elites.json"
    structural_queue_jsonl_path = out_dir / "structural_slow_queue.jsonl"
    structural_ledger_jsonl_path = out_dir / "structural_ledger.jsonl"
    transcript_path = out_dir / "transcript.jsonl"
    summary_path = out_dir / "funsearch_summary.json"
    generation_notes_path = out_dir / "generation_notes.json"
    all_generation_notes: list[str] = []

    structural_config = config.verification.structural
    memory = ArchiveMemory(
        d_cap_enabled=bool(config.verification.d_cap_enabled),
        d_cap=int(config.verification.d_cap),
        dsqrtn_floor=float(config.verification.dsqrtn_floor),
        allowed_families=tuple(
            family.value if isinstance(family, CodeFamily) else str(family)
            for family in config.allowed_families
        ),
        deformation_mode=config.deformation_mode.value,
        lp_shape_pin=config.lp_shape_pin,
        max_check_weight=config.verification.max_check_weight,
        exact_check_weight=getattr(config.verification, "exact_check_weight", None),
        lightest_endpoint_slot=bool(getattr(structural_config, "lightest_endpoint_slot", False)),
        structural_proxy_elites_limit=int(structural_config.elites_limit),
        # v3.6 B4: per-(k, d_search) realization cap of the reducer.
        structural_coordinate_cap=int(structural_config.coordinate_cap),
        structural_parent_quota=int(structural_config.structural_parents),
        runtime_parent_quota=int(structural_config.runtime_parents),
        proxy_seeds=tuple(structural_config.proxy_seeds),
        proxy_algorithm_id=str(structural_config.proxy_algorithm_id),
        turn=int(turn),
        # v3.7: cross-turn measurements feed the expected-plpr runtime order.
        evidence_rows=[dict(row) for row in (evidence_rows or []) if isinstance(row, dict)],
        diversity_parent_quota=int(getattr(structural_config, "diversity_parents", 0) or 0),
        # v3.8 endpoint family rules: the distinct-family flag is protocol
        # (structural config); the demoted families are this turn's machine
        # verdict from the campaign graph (overlay key demoted_families).
        endpoint_family_distinct=bool(getattr(structural_config, "endpoint_family_distinct", False)),
        demoted_families=tuple(
            str(item)
            for item in list((prompt_context_overlay or {}).get("demoted_families") or [])
        ),
    )
    # v3.8 variant generations: the last `exploit_generations` generations of
    # the turn only vary the parents the graph handed in (overlay key
    # exploit_parent_specs: measured-best codes + the resident reference).
    exploit_generations = int(getattr(structural_config, "exploit_generations", 0) or 0)
    exploit_parents = [
        dict(parent)
        for parent in list((prompt_context_overlay or {}).get("exploit_parent_specs") or [])
        if isinstance(parent, dict) and isinstance(parent.get("spec"), dict)
    ]
    # Cross-turn structural survival (v3.2 §4.2): last turn's elites are live
    # candidates and parents again without being regenerated or re-measured.
    memory.seed_structural_elites(structural_elites)
    campaign_measured = {str(item) for item in (measured_hashes or set())}
    active_prompt_context_overlay = prompt_context_overlay if isinstance(prompt_context_overlay, dict) else None
    _write_json(run_config_path, config.to_dict())
    transcript_path.write_text("", encoding="utf-8")

    seeds = list(seed_sources if seed_sources is not None else SEEDS) if use_seeds else []
    for index, source in enumerate(seeds, start=1):
        program_id = f"seed_{index:04d}"
        (programs_dir / f"{program_id}.py").write_text(source, encoding="utf-8")
        memory.evaluate_and_record(
            CandidateProgram(
                program_id=program_id,
                source=source,
                rng_seed=0,
                n_budget=int(config.n_budget),
                source_agent="seed",
                metadata={"funsearch_seed_index": index},
            ),
            n_budget=int(config.n_budget),
            proxy_iters=int(structural_config.proxy_iters_per_seed),
            proxy_n_min=config.verification.proxy_n_min,
            proxy_n_max=config.verification.proxy_n_max,
        )
        _append_transcript(
            transcript_path,
            _transcript_attempt(memory.attempts[-1], gen=0, source="seed"),
        )

    _save_memory_artifacts(
        memory=memory,
        config=config,
        archive_path=archive_path,
        feedback_path=feedback_path,
        prompt_context_path=prompt_context_path,
        promotion_queue_json_path=promotion_queue_json_path,
        promotion_queue_jsonl_path=promotion_queue_jsonl_path,
        all_archive_queue_json_path=all_archive_queue_json_path,
        all_archive_queue_jsonl_path=all_archive_queue_jsonl_path,
        structural_elites_path=structural_elites_path,
        structural_queue_jsonl_path=structural_queue_jsonl_path,
        structural_ledger_jsonl_path=structural_ledger_jsonl_path,
        measured_hashes=campaign_measured,
    )

    status_counts: dict[str, int] = {}
    proposal_count_missing_total = 0
    completion_attempt_count = 0
    for gen in range(1, generations + 1):
        gen_dir = generations_dir / f"g{gen:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        gen_prompt_path = gen_dir / "prompt.json"
        gen_prompt_context_path = gen_dir / "prompt_context.json"
        raw_path = gen_dir / "llm_raw.jsonl"
        base_prompt_context = memory.prompt_context_payload(
            n_budget=int(config.n_budget),
            hardware=config.hardware.target.value,
        )
        prompt_context = _merge_prompt_context_overlay(
            base_prompt_context,
            active_prompt_context_overlay,
        )
        generation_kind = "exploration"
        variant_families: tuple[str, ...] | None = None
        if exploit_generations > 0 and gen > generations - exploit_generations:
            if exploit_parents:
                generation_kind = "variant"
                prompt_context = _variant_prompt_context(
                    prompt_context, exploit_parents, gen=gen, generations=generations
                )
                variant_families = tuple(sorted({_parent_family(parent) for parent in exploit_parents}))
            else:
                # Turn 0 of a budget point without a resident reference: no
                # parent exists yet, so the generation explores and says so.
                generation_kind = "variant_fallback_no_parents"
                prompt_context = dict(prompt_context)
                prompt_context["exploit_generation_fallback"] = (
                    "no measured code and no resident reference code yet"
                )
        _write_json(gen_prompt_context_path, prompt_context)
        prompt = _build_funsearch_prompt(
            memory=memory,
            config=config,
            gen=gen,
            proposal_count=proposal_count,
            prompt_context=prompt_context,
        )
        prompt["generation_kind"] = generation_kind
        _write_json(gen_prompt_path, prompt)

        raw_rows: list[dict[str, Any]] = []
        programs: list[str] = []
        generation_notes: list[str] = []
        generation_errors: list[str] = []
        call_statuses: list[str] = []
        engine_info: dict[str, Any] = {"provider": provider}
        if provider == BASELINE_PROVIDER:
            # 2026-11-06 search baselines: the fixed sampling prior proposes the whole batch and no model is
            # called; the completion loop below then has nothing left to request.
            programs, raw_rows = _baseline_sampler_batch(
                config=config,
                prompt_context=prompt_context,
                turn=int(turn),
                gen=gen,
                generations=generations,
                proposal_count=proposal_count,
                gen_dir=gen_dir,
            )
            call_statuses.append("ok")
        for completion_attempt in range(1, MAX_COMPLETION_ATTEMPTS + 1):
            remaining = proposal_count - len(programs)
            if remaining <= 0:
                break
            user_prompt = prompt["user_prompt"]
            if completion_attempt > 1:
                user_prompt = _refill_funsearch_prompt(
                    prompt["user_prompt"],
                    requested_total=proposal_count,
                    parsed_count=len(programs),
                    remaining=remaining,
                )
            raw_response = ""
            generation_error = ""
            generation_traceback = ""
            call_status = "ok"
            try:
                raw_response, engine_info = _complete_funsearch_prompt(
                    config=config,
                    provider=provider,
                    system=prompt["system_prompt"],
                    user=user_prompt,
                    prompt_context=prompt_context,
                    gen=gen,
                    proposal_count=remaining,
                )
            except Exception as exc:
                call_status = _generation_status_from_exception(exc)
                generation_error = f"{type(exc).__name__}: {exc}"
                generation_errors.append(generation_error)
                # The traceback is the only thing that says WHERE a bare
                # TypeError came from (2026-09-11: the gateway's choices=None
                # body); keep it in the row. The fallback engine_info used
                # to be config.llm.to_dict() -- the DeepSeek defaults -- which
                # mislabelled the failing provider in llm_raw.jsonl.
                generation_traceback = traceback.format_exc()
                engine_info = {"provider": provider, "info_source": "exception_fallback"}
            new_programs = _extract_programs(raw_response, limit=remaining)
            programs.extend(new_programs)
            generation_notes.extend(_extract_notes(raw_response))
            call_statuses.append(call_status)
            raw_rows.append(
                {
                    "schema": "funsearch_llm_raw",
                    "run_id": config.run_id,
                    "gen": gen,
                    "provider": provider,
                    "completion_attempt": completion_attempt,
                    "status": call_status,
                    "proposal_count_requested": remaining,
                    "proposal_count_parsed": len(new_programs),
                    "proposal_count_accumulated": len(programs),
                    "engine_info": engine_info,
                    "response": raw_response,
                    "generation_error": generation_error,
                    "generation_traceback": generation_traceback,
                }
            )
            if call_status != "ok":
                break

        _write_jsonl(raw_path, raw_rows)
        all_generation_notes.extend(generation_notes[:NOTES_MAX_LINES])
        generation_error = "; ".join(error for error in generation_errors if error)
        if not programs:
            status = next((item for item in reversed(call_statuses) if item != "ok"), "no_program")
        elif len(programs) < proposal_count:
            status = "partial_batch"
        else:
            status = "ok"
        if not programs and status == "ok":
            status = "no_program"
        proposal_count_missing_total += max(0, proposal_count - len(programs))
        completion_attempt_count += len(raw_rows)
        status_counts[status] = status_counts.get(status, 0) + 1
        _append_transcript(
            transcript_path,
            {
                "kind": "generation",
                "gen": gen,
                "generation_kind": generation_kind,
                "provider": provider,
                "status": status,
                "proposal_count_requested": proposal_count,
                "proposal_count_parsed": len(programs),
                "completion_attempts": len(raw_rows),
                "proposal_count_missing": max(0, proposal_count - len(programs)),
                "generation_error": generation_error,
                "artifact_paths": {
                    "prompt": str(gen_prompt_path),
                    "prompt_context": str(gen_prompt_context_path),
                    "llm_raw": str(raw_path),
                },
            },
        )

        if require_full_llm_batch and status != "ok":
            # 2026-11-06 reasoning-saturation rule (reasoning_fallback.py): a turn-0 generation whose every
            # completion attempt spent the whole output budget on reasoning is named in the message; the
            # launcher of a final-protocol run restarts the arm at the fallback level, any other launch
            # retries the turn as before (the message still starts with the strict-abort text).
            saturation = ""
            if int(turn) == 0 and not programs and saturated_generation(raw_rows, attempts=MAX_COMPLETION_ATTEMPTS):
                saturation = "; " + saturation_message(turn=int(turn), gen=gen, rows=raw_rows)
            raise RuntimeError(
                "strict campaign requires a complete LLM proposal batch: "
                f"gen={gen}, status={status}, parsed={len(programs)}/{proposal_count}; "
                "retry this turn from checkpoint" + saturation
            )

        if not programs:
            _append_transcript(
                transcript_path,
                {
                    "kind": "attempt",
                    "gen": gen,
                    "source": provider,
                    "error": generation_error or status or "no-program",
                    "family": "?",
                },
            )
            continue

        # Re-asserted where fast evaluation of the batch begins: a callback or
        # tool run in between may have resolved groups under another scope.
        set_group_order_budget(int(config.n_budget))
        # 2026-09-11: the (program, seed) evaluations of a generation are
        # independent until the archive is consulted, so ArchiveMemory.
        # evaluate_batch runs the pure stages in OMNIQEC_FAST_WORKERS
        # processes and replays the archive updates in this exact order
        # (records/attempts/owners identical to the sequential loop; 1 =
        # the sequential loop itself).
        batch: list[CandidateProgram] = []
        proposal_index_of: dict[str, int] = {}
        for proposal_index, src in enumerate(programs, start=1):
            base_program_id = f"g{gen:04d}_p{proposal_index:04d}"
            # A baseline turn has 3840 programs; their sources are in the generation's proposals.jsonl.
            if provider != BASELINE_PROVIDER:
                (programs_dir / f"{base_program_id}.py").write_text(
                    normalize_source_text(src), encoding="utf-8"
                )
            for rng_seed in range(rng_seeds_per_program):
                program_id = base_program_id if rng_seed == 0 else f"{base_program_id}_s{rng_seed:02d}"
                proposal_index_of[program_id] = proposal_index
                batch.append(
                    CandidateProgram(
                        program_id=program_id,
                        source=src,
                        rng_seed=rng_seed,
                        n_budget=int(config.n_budget),
                        source_agent=f"funsearch_{provider}",
                        metadata={
                            "funsearch_generation": int(gen),
                            "funsearch_generation_kind": generation_kind,
                            "funsearch_proposal_index": int(proposal_index),
                            "proposal_count_requested": int(proposal_count),
                            "rng_seed": int(rng_seed),
                            "rng_seeds_per_program": int(rng_seeds_per_program),
                        },
                    )
                )

        def _on_record(record: Any) -> None:
            _append_transcript(
                transcript_path,
                _transcript_attempt(
                    memory.attempts[-1],
                    gen=gen,
                    source=provider,
                    proposal_index=proposal_index_of[record.program.program_id],
                ),
            )

        # v3.8 variant generation: the fast loop admits only the parents'
        # families (archive_status family_not_allowed for anything else), so
        # the quota is a machine rule, not a request the model may ignore.
        saved_families = memory.allowed_families
        if variant_families:
            campaign_allowed = set(saved_families) if saved_families else set(variant_families)
            restricted = tuple(family for family in variant_families if family in campaign_allowed)
            memory.allowed_families = restricted or saved_families
        try:
            memory.evaluate_batch(
                batch,
                n_budget=int(config.n_budget),
                proxy_iters=int(structural_config.proxy_iters_per_seed),
                proxy_n_min=config.verification.proxy_n_min,
                proxy_n_max=config.verification.proxy_n_max,
                workers=fast_workers_from_env(),
                on_record=_on_record,
            )
        finally:
            memory.allowed_families = saved_families

        _write_status(
            out_dir / "status.json",
            memory=memory,
            config=config,
            gen=gen,
            generations=generations,
            provider=provider,
            status_counts=status_counts,
        )
        if gen % save_every == 0:
            _save_memory_artifacts(
                memory=memory,
                config=config,
                archive_path=archive_path,
                feedback_path=feedback_path,
                prompt_context_path=prompt_context_path,
                promotion_queue_json_path=promotion_queue_json_path,
                promotion_queue_jsonl_path=promotion_queue_jsonl_path,
                all_archive_queue_json_path=all_archive_queue_json_path,
                all_archive_queue_jsonl_path=all_archive_queue_jsonl_path,
                structural_elites_path=structural_elites_path,
                structural_queue_jsonl_path=structural_queue_jsonl_path,
                structural_ledger_jsonl_path=structural_ledger_jsonl_path,
                measured_hashes=campaign_measured,
            )
        if promotion_callback is not None and _should_promote(gen, promote_every):
            _save_memory_artifacts(
                memory=memory,
                config=config,
                archive_path=archive_path,
                feedback_path=feedback_path,
                prompt_context_path=prompt_context_path,
                promotion_queue_json_path=promotion_queue_json_path,
                promotion_queue_jsonl_path=promotion_queue_jsonl_path,
                all_archive_queue_json_path=all_archive_queue_json_path,
                all_archive_queue_jsonl_path=all_archive_queue_jsonl_path,
                structural_elites_path=structural_elites_path,
                structural_queue_jsonl_path=structural_queue_jsonl_path,
                structural_ledger_jsonl_path=structural_ledger_jsonl_path,
                measured_hashes=campaign_measured,
            )
            promotion_result = promotion_callback(
                {
                    "schema": "funsearch_promotion_request",
                    "run_id": config.run_id,
                    "gen": int(gen),
                    "out_dir": str(out_dir),
                    "promotion_queue": str(promotion_queue_jsonl_path),
                    "all_archive_queue": str(all_archive_queue_jsonl_path),
                    "structural_slow_queue": str(structural_queue_jsonl_path),
                    "structural_ledger": str(structural_ledger_jsonl_path),
                    "prompt_context": str(prompt_context_path),
                    "archive": str(archive_path),
                    "feedback": str(feedback_path),
                }
            ) or {}
            overlay = promotion_result.get("prompt_context_overlay")
            if isinstance(overlay, dict):
                active_prompt_context_overlay = overlay
            reconciled = _apply_promotion_evidence(memory, promotion_result)
            if reconciled:
                # Re-save so archive.json and the queues reflect the corrected
                # scores/quarantine before the next generation reads them.
                _save_memory_artifacts(
                    memory=memory,
                    config=config,
                    archive_path=archive_path,
                    feedback_path=feedback_path,
                    prompt_context_path=prompt_context_path,
                    promotion_queue_json_path=promotion_queue_json_path,
                    promotion_queue_jsonl_path=promotion_queue_jsonl_path,
                    all_archive_queue_json_path=all_archive_queue_json_path,
                    all_archive_queue_jsonl_path=all_archive_queue_jsonl_path,
                    structural_elites_path=structural_elites_path,
                    structural_queue_jsonl_path=structural_queue_jsonl_path,
                    structural_ledger_jsonl_path=structural_ledger_jsonl_path,
                    measured_hashes=campaign_measured,
                )
            _append_transcript(
                transcript_path,
                {
                    "kind": "promotion",
                    "gen": int(gen),
                    "status": str(promotion_result.get("status") or "unknown"),
                    "artifact_paths": dict(promotion_result.get("artifact_paths") or {}),
                    "slow_parent_count": int(promotion_result.get("slow_parent_count") or 0),
                    "overlay_applied": isinstance(overlay, dict),
                    "reconciled_records": int(reconciled),
                    "degraded_elites": memory.counts().get("degraded_elites", 0),
                },
            )

    _save_memory_artifacts(
        memory=memory,
        config=config,
        archive_path=archive_path,
        feedback_path=feedback_path,
        prompt_context_path=prompt_context_path,
        promotion_queue_json_path=promotion_queue_json_path,
        promotion_queue_jsonl_path=promotion_queue_jsonl_path,
        all_archive_queue_json_path=all_archive_queue_json_path,
        all_archive_queue_jsonl_path=all_archive_queue_jsonl_path,
        structural_elites_path=structural_elites_path,
        structural_queue_jsonl_path=structural_queue_jsonl_path,
        structural_ledger_jsonl_path=structural_ledger_jsonl_path,
        measured_hashes=campaign_measured,
    )
    counts = memory.counts()
    promotion_queue = memory.promotion_queue(limit=int(config.verification.promote_top), include_source=True)
    all_archive_queue = memory.all_archive_queue(include_source=True)
    structural_slow_queue = memory.structural_slow_queue(
        slots=int(structural_config.structural_slow_slots),
        measured_hashes=campaign_measured,
    )
    summary = FunSearchSummary(
        run_id=config.run_id,
        out_dir=str(out_dir),
        n_budget=int(config.n_budget),
        hardware=config.hardware.target.value,
        provider=str(provider),
        generations=generations,
        seed_count=len(seeds),
        proposal_count_requested=proposal_count,
        proposal_count_requested_total=proposal_count * generations,
        proposal_count=max(0, counts["attempts"] - len(seeds)),
        proposal_count_missing_total=proposal_count_missing_total,
        completion_attempt_count=completion_attempt_count,
        accepted_count=counts["accepted"],
        archive_cell_count=counts["archive_cells"],
        promotion_queue_count=len(promotion_queue),
        all_archive_queue_count=len(all_archive_queue),
        structural_reject_count=counts["structural_rejects"],
        fast_reject_count=counts["fast_rejects"],
        duplicate_count=counts["duplicates"],
        structural_elite_count=len(memory.structural_proxy_elites),
        structural_slow_queue_count=len(structural_slow_queue),
        rng_seeds_per_program=rng_seeds_per_program,
        status_counts=status_counts,
        artifact_paths={
            "run_config": str(run_config_path),
            "archive": str(archive_path),
            "feedback": str(feedback_path),
            "prompt_context": str(prompt_context_path),
            "promotion_queue": str(promotion_queue_jsonl_path),
            "promotion_queue_json": str(promotion_queue_json_path),
            "all_archive_queue": str(all_archive_queue_jsonl_path),
            "all_archive_queue_json": str(all_archive_queue_json_path),
            "structural_proxy_elites": str(structural_elites_path),
            "structural_slow_queue": str(structural_queue_jsonl_path),
            "structural_ledger": str(structural_ledger_jsonl_path),
            "transcript": str(transcript_path),
            "programs": str(programs_dir),
            "generations": str(generations_dir),
            "summary": str(summary_path),
            "generation_notes": str(generation_notes_path),
        },
    )
    _write_json(
        generation_notes_path,
        {"schema": "generation_notes", "notes": all_generation_notes[:NOTES_MAX_LINES]},
    )
    _write_json(summary_path, summary.to_dict())
    return summary


def _save_memory_artifacts(
    *,
    memory: ArchiveMemory,
    config: RunConfig,
    archive_path: Path,
    feedback_path: Path,
    prompt_context_path: Path,
    promotion_queue_json_path: Path,
    promotion_queue_jsonl_path: Path,
    all_archive_queue_json_path: Path,
    all_archive_queue_jsonl_path: Path,
    structural_elites_path: Path,
    structural_queue_jsonl_path: Path,
    structural_ledger_jsonl_path: Path,
    measured_hashes: set[str] | None = None,
) -> None:
    structural_config = config.verification.structural
    payload = memory.to_dict()
    feedback = memory.feedback_payload(parent_limit=int(structural_config.parent_cap))
    prompt_context = memory.prompt_context_payload(
        n_budget=int(config.n_budget),
        hardware=config.hardware.target.value,
        parent_limit=int(structural_config.parent_cap),
    )
    promotion_queue = memory.promotion_queue(
        limit=int(config.verification.promote_top),
        include_source=True,
    )
    all_archive_queue = memory.all_archive_queue(include_source=True)
    structural_queue = memory.structural_slow_queue(
        slots=int(structural_config.structural_slow_slots),
        measured_hashes=measured_hashes,
    )
    _write_json(
        structural_elites_path,
        {
            "schema": "structural_proxy_elites",
            "reducer_id": structural_config.reducer_id,
            "profile_id": structural_config.profile_id,
            "limit": int(structural_config.elites_limit),
            "objectives": list(structural_config.structural_axes),
            "count": len(memory.structural_proxy_elites),
            "elites": memory.structural_proxy_elites,
            "slow_slots": int(structural_config.structural_slow_slots),
            "structural_slow_queue": structural_queue,
        },
    )
    _write_jsonl(structural_queue_jsonl_path, structural_queue)
    # Audit fix #3/#4: the FULL structurally-eligible ledger (hash-keyed, no
    # cap, stable hash order). It is the cross-turn retention state and the
    # registry a named TurnPlan promotion is resolved against.
    _write_jsonl(
        structural_ledger_jsonl_path,
        [memory.structural_ledger[key] for key in sorted(memory.structural_ledger)],
    )
    _write_json(archive_path, payload["archive"])
    # v3.7.1: per-cycle construction census by check weight (telemetry; summed
    # across turns by workflow.graph._weight_census for the prompt's fact table).
    _write_json(
        archive_path.parent / "weight_census.json",
        {"schema": "weight_census_v1", "rows": memory.weight_census()},
    )
    _write_json(feedback_path, feedback)
    _write_json(prompt_context_path, prompt_context)
    _write_json(
        promotion_queue_json_path,
        {
            "schema": "fast_surrogate_promotion_queue",
            "promotion_rule": "ver1_q_fast_archive_elite",
            "promote_top": int(config.verification.promote_top),
            "queue_size": len(promotion_queue),
            "queue": promotion_queue,
        },
    )
    _write_jsonl(promotion_queue_jsonl_path, promotion_queue)
    _write_json(
        all_archive_queue_json_path,
        {
            "schema": "all_archive_queue",
            "promotion_rule": "all_archive_cell_elites",
            "queue_size": len(all_archive_queue),
            "queue": all_archive_queue,
        },
    )
    _write_jsonl(all_archive_queue_jsonl_path, all_archive_queue)


def _build_funsearch_prompt(
    *,
    memory: ArchiveMemory,
    config: RunConfig,
    gen: int,
    proposal_count: int,
    prompt_context: dict[str, Any] | None = None,
    n_recent: int | None = None,
) -> dict[str, Any]:
    # The recent-programs block must cover at least one full generation of
    # proposals, otherwise feedback for the first programs of the previous
    # batch is silently dropped.
    if n_recent is None:
        n_recent = max(8, int(proposal_count))
    allowed = [family.value if isinstance(family, CodeFamily) else str(family) for family in config.allowed_families]
    system = _funsearch_system_prompt(
        allowed_families=allowed, n_budget=int(config.n_budget),
        task=str(getattr(config.verification, "task", "memory")),
    )
    user = _funsearch_user_prompt(
        memory=memory,
        config=config,
        gen=gen,
        proposal_count=proposal_count,
        n_recent=n_recent,
        prompt_context=prompt_context or {},
    )
    return {
        "schema": "funsearch_prompt",
        "run_id": config.run_id,
        "gen": int(gen),
        "n_budget": int(config.n_budget),
        "hardware": config.hardware.target.value,
        "proposal_count": int(proposal_count),
        "allowed_families": allowed,
        "lp_shape_pin": config.lp_shape_pin.to_dict(),
        "format_contract_version": "family_format_contract",
        "slow_feedback_present": bool(_slow_feedback_from_context(prompt_context or {})),
        "system_prompt": system,
        "user_prompt": user,
    }


def _allowed_plugins(allowed_families: list[str]):
    from .construction.registry import FAMILY_REGISTRY

    allowed = {str(name) for name in allowed_families}
    return [plugin for name, plugin in FAMILY_REGISTRY.items() if name in allowed]


def _task_measurement_sentence(task: str) -> str:
    """Fig.3 Phase B: what the verifier measures when the task is not the
    Fig.2 memory experiment. A description of the circuit, not of what makes
    a code good at it -- that stays with the measurements."""

    if str(task) != "cnot_dense":
        return ""
    return (
        "The measurement is a two-block circuit: two copies of the code side "
        "by side run T = max(12, d_search) rounds of syndrome extraction, with "
        "one transversal logical CNOT between the two blocks after every "
        "round; both blocks are decoded jointly, and the reported "
        "per_logical_per_round is the logical error per logical qubit per "
        "round of that circuit.\n"
    )


def _funsearch_system_prompt(
    *, allowed_families: list[str], n_budget: int | None = None, task: str = "memory"
) -> str:
    from .construction.registry import format_contract_lines_for_budget

    plugins = _allowed_plugins(allowed_families)
    genotype_lines = "\n".join(plugin.genotype_line for plugin in plugins)
    # Syntax examples are deliberately mediocre, valid codes that are not
    # benchmarks: examples anchor LLM proposals, so they must teach the shape
    # without offering a strong starting point or leaking a known reference
    # construction. Two of them (the BB example and the metacyclic
    # lp_nonabelian example) are the specs of seed programs 1 and 5
    # (examples/seed_programs.py); the others are not seeds.
    example_lines = "\n".join(plugin.syntax_example_line for plugin in plugins)
    extra_notes = "\n".join(note for plugin in plugins for note in plugin.extra_system_notes)
    format_lines = "\n".join(
        line for plugin in plugins for line in format_contract_lines_for_budget(plugin, n_budget)
    )
    return (
        "You are the evolutionary engine of OmniQEC, a system that discovers "
        "CSS quantum LDPC error-correcting codes.\n"
        "Each turn you write a small batch of independent Python functions. "
        "Each candidate is one fenced code block containing `def construct(rng)` "
        "that returns the specification of a CSS code.\n"
        "A deterministic verifier then measures the code and reports back.\n"
        + _task_measurement_sentence(task) + "\n"
        # v3.2 §5: the search's selection pressure lives in machine retention
        # rules (which candidates keep slow slots, which programs are kept
        # across turns, which become parents, which get final slots) and
        # deliberately NOT in this text. An objective sentence here is a global
        # command: it survives every later block, and it reintroduces exactly
        # the single-axis ranking the retention rules exist to replace.
        "OBJECTIVE. Produce novel, legal CSS LDPC constructions that the "
        "frozen verifier can evaluate. Which constructions are retained, which "
        "become parents, and which are measured is decided by the machine's "
        "retention rules from the verified record.\n\n"
        "GENOTYPE. Return a dict in one allowed family: "
        f"{', '.join(allowed_families)}.\n"
        f"{genotype_lines}\n\n"
        "LEGAL SHAPE EXAMPLES (examples of valid syntax, not fixed templates; rows and columns may differ).\n"
        f"{example_lines}\n"
        + (f"{extra_notes}\n\n" if extra_notes else "\n")
        + "STRICT FORMAT CONTRACT.\n"
        f"{format_lines}\n\n"
        "RULES.\n"
        "- Output ONLY the requested number of ```python code blocks.\n"
        "- Each code block must contain exactly one `def construct(rng):` candidate.\n"
        "- `rng` is a numpy.random.Generator; use it for any randomness so results are seedable.\n"
        "- No imports, file I/O, network access, or external tools.\n"
        "- Return integers, not numpy scalars, in the spec.\n"
        "- Do not return numpy arrays, sets, custom classes, strings for ints, or JSON text.\n"
        "- Propose something NEW relative to the constructions shown below: an "
        "unexplored region of a family, or a variation of one that is shown.\n"
        "- Legality floor: a code with heuristic distance below 3 cannot "
        "correct a single fault, so it is not an error-correcting code. It is "
        "recorded for the audit trail and never measured.\n"
        "- A useful mutation must change the code hash. Do not merely reorder rows, "
        "reorder terms, or swap equivalent entries; those can canonicalize to the same code.\n"
        "- Families are construction mechanisms, not search boundaries.\n"
        "- Optionally tag each program's FIRST line with a comment "
        "`# exploits: I-<id>[, ...]` and/or `# tests: H-<id>[, ...]` referring ONLY "
        "to ids listed in the strategy brief; a `strategy: transfer` or "
        "`strategy: wildcat` clause on the same line marks cross-family motif "
        "transfer or deliberate wild exploration. Tags drive credit assignment; "
        "untagged programs are treated as free exploration. Do not invent ids.\n"
        "- Do not claim distance, logical error rate, decoder result, or circuit validity.\n\n"
        "NOTES FOR THE NEXT GENERATION (optional; at most 10 lines).\n"
        "After the code blocks you may append ONE plain-text block, never fenced:\n"
        "NOTES-FOR-NEXT-GENERATION:\n"
        "- tried: <construction you explored this batch>\n"
        "- ruled out: <what did not work, and why>\n"
        "- next: <a direction you did not get to>\n"
        "Only the most recent generations' notes are shown back to you. Write "
        "what you would want to know if you started fresh; do not restate the "
        "archive table, and do not put code in it."
    )


def _funsearch_user_prompt(
    *,
    memory: ArchiveMemory,
    config: RunConfig,
    gen: int,
    proposal_count: int,
    n_recent: int,
    prompt_context: dict[str, Any],
) -> str:
    # Deliberately no "baseline to beat" line: the BB comparison is a
    # measurement-side concern (resident baseline in the slow stage); naming a
    # well-known benchmark in the prompt invites perturbation around it
    # instead of exploration, and weakens the evaluation narrative.
    # 消融门控:show_fast=给 fast 反馈(归档 Q_eff、近 N 程序、结构失败);show_slow=给 slow 反馈
    # (LER 注解、MEASURED CIRCUIT-LEVEL 块);show_memory=显示父代(none 时全关=每代冷启动)。
    mode = getattr(config, "feedback_mode", "both")
    show_fast = mode in ("both", "fast")
    show_slow = mode in ("both", "slow")
    show_memory = mode != "none"
    lines = []
    # v3.8: a variant generation carries no strategy brief and no family
    # quota (both are exploration instruments; _variant_prompt_context has
    # already removed them) and states its rule right after the batch line.
    variant_lines = _variant_generation_lines(prompt_context, proposal_count=int(proposal_count))
    # Strategy brief (03a §4.1-1): the campaign policy's directives, injected
    # as a tagged source block at the head of the user prompt. S3 no_feedback
    # (feedback_mode none): the planner's brief and its family allocation are
    # feedback channels too and stay out.
    strategy_brief = prompt_context.get("strategy_brief") if mode != "none" else None
    if isinstance(strategy_brief, str) and strategy_brief.strip():
        import hashlib as _hashlib

        brief_hash = _hashlib.sha256(strategy_brief.encode()).hexdigest()[:16]
        lines.append(
            f'<source id="S-B1" kind="strategy_brief" origin="turn plan.directives" '
            f'hash="sha256:{brief_hash}">'
        )
        lines.append(strategy_brief.rstrip())
        lines.append("</source>")
        lines.append("")
    # Experiment variant (2026-09-09): frozen background codes (protocol
    # snapshot key reference_codes), shown in every generation regardless
    # of the feedback ablation -- they are protocol, not feedback.
    reference = prompt_context.get("reference_codes")
    if isinstance(reference, dict) and reference.get("codes"):
        from omniqec_agentic.orchestrator.reference_codes import reference_block
        lines.append(reference_block(reference))
        lines.append("")
    if mode != "none":  # none 消融:抹掉代号计数器,40 次抽样逐字相同=真·独立冷采样(零跨代信息)
        lines.append(f"GENERATION: {int(gen)}")
    lines.append(f"Batch target: return exactly {int(proposal_count)} independent Python code blocks.")
    if variant_lines:
        lines.extend(variant_lines)
    quota = dict(prompt_context.get("family_quota") or {}) if mode != "none" else {}
    if quota:
        # The policy's family allocation is a REQUIREMENT here, not an
        # "attention" hint. Left as prose it was silently ignored, so the
        # campaign could never actually steer where proposals went; the
        # produced distribution is audited against this line after the batch.
        rendered = ", ".join(f"{fam}: {count}" for fam, count in sorted(quota.items()) if count)
        lines.append(
            f"Required family allocation for these {int(proposal_count)} blocks: {rendered}. "
            "Tag each block's family in its spec; the produced distribution is checked "
            "against this allocation."
        )
    lines.append(
        f"Total qubit budget: {int(config.n_budget)}. Each check row costs one "
        f"measurement ancilla qubit; keep N_data + N_checks <= {int(config.n_budget)}."
    )
    # Stated as a cost, not as a target band: a prescribed size range is a
    # recipe, and recipes are what collapse the search onto one shape. The
    # model is told what unspent budget buys (nothing) and left to decide.
    lines.append(
        f"WASTED-BUDGET WARNING: this campaign is judged at N={int(config.n_budget)}. "
        "A candidate that spends far below the budget still competes against "
        "candidates that spend all of it. Treat a small code as a deliberate "
        "choice that needs a reason (testing a structural hypothesis, say), "
        "not as a default."
    )
    hardware_profile = HARDWARE_PROMPT_PROFILES.get(
        config.hardware.target.value,
        (f"Hardware target: {config.hardware.target.value}.",),
    )
    lines.extend(hardware_profile)
    lines.append("")
    lines.append("CONSTRUCTION SIZE FORMULAS:")
    allowed_names = [
        family.value if isinstance(family, CodeFamily) else str(family)
        for family in config.allowed_families
    ]
    for plugin in _allowed_plugins(allowed_names):
        lines.append(plugin.size_formula_line)
    lines.append("")
    lines.append("CONSTRUCTION CONTRACT:")
    for line in _construction_contract_lines(config):
        lines.append(f"- {line}")
    if config.verification.proxy_n_min is not None or config.verification.proxy_n_max is not None:
        filters = []
        if config.verification.proxy_n_min is not None:
            filters.append(f"N_data >= {int(config.verification.proxy_n_min)}")
        if config.verification.proxy_n_max is not None:
            filters.append(f"N_data <= {int(config.verification.proxy_n_max)}")
        lines.append("RUN-SPECIFIC FAST-SCREEN SIZE FILTER: " + " and ".join(filters) + ".")
    lines.append("")
    if show_fast:  # 归档上下文 = fast 视图;slow-only/none 不显示
        # v3.2 §5.1: context, not a leaderboard. The rows are listed in niche
        # order with no score column: ordering 24 rows by a quality key is a
        # ranking instruction whatever the header says, and this table used to
        # be the strongest one in the prompt.
        lines.append(
            "RETAINED CONSTRUCTION CONTEXT (one code per (overhead-bin, "
            "heuristic-distance) niche, listed in niche order):"
        )
        archive_rows = memory.archive_summary(limit=24, order="niche")
        if archive_rows:
            lines.append(
                f"{'niche':>12} | {'n':>5} | {'k':>4} | {'d_search':>8} | {'w':>3} | "
                f"{'total':>9} | family/notes"
            )
            for row in archive_rows:
                note = str(row.get("family") or "")
                if row.get("degraded"):
                    note += (
                        f" HOOK-DEGRADED (circuit-level distance evidence "
                        f"{row.get('d_eff')} below the heuristic value)"
                    )
                elif show_slow and row.get("slow_per_logical_per_round") is not None:
                    note += (
                        f" measured per_logical_per_round="
                        f"{_format_metric(row.get('slow_per_logical_per_round'))}"
                        f" ({row.get('slow_decoder')})"
                    )
                failures = row.get("structural_failures") or []
                if failures:
                    note += " structural_failures=" + ",".join(str(item) for item in failures)
                lines.append(
                    f"{str(row.get('cell')):>12} | {str(row.get('n')):>5} | "
                    f"{str(row.get('k')):>4} | {str(row.get('d_search')):>8} | "
                    f"{str(row.get('max_check_weight')):>3} | "
                    f"{_budget_use(row):>9} | {note}"
                )
        else:
            lines.append("  (empty - no construction has been recorded yet)")
        lines.append("")
    recent = memory.recent_program_summaries(limit=n_recent) if show_fast else []
    if recent:
        measured_lookup = _measured_lookup(prompt_context)
        lines.append(f"YOUR LAST {len(recent)} PROGRAMS (newest last):")
        for summary in recent:
            lines.append(f"  - {_program_summary_line(summary, measured=measured_lookup)}")
        lines.append("")
    queue_lines = _last_turn_queue_lines(prompt_context) if show_slow else []
    if queue_lines:
        lines.extend(queue_lines)
        lines.append("")
    failure_actions = _failure_action_lines(memory=memory, prompt_context=prompt_context) if show_fast else []
    if failure_actions:
        lines.append("VERIFIER FAILURE ACTIONS:")
        for item in failure_actions:
            lines.append(f"- {item}")
        lines.append("")
    slow_feedback = _slow_feedback_from_context(prompt_context)
    if slow_feedback and show_slow:
        lines.extend(_slow_feedback_prompt_lines(slow_feedback, prompt_context))
        lines.append("")
    # v3.7: the WHOLE parent window (16) is rendered, not a six-parent prefix.
    parent_lines = _parent_construction_lines(
        prompt_context, limit=int(_parent_window(prompt_context)), show_fast=show_fast, show_slow=show_slow
    ) if show_memory else []
    if parent_lines:
        lines.extend(parent_lines)
        lines.append("")
    evidence_lines = _evidence_table_lines(prompt_context) if (show_memory and show_slow) else []
    if evidence_lines:
        lines.extend(evidence_lines)
        lines.append("")
    guidance = _dedupe_strings(
        (list(memory.guidance()) + list((prompt_context.get("campaign_feedback") or {}).get("proposal_guidance") or [])
         if show_fast else [])
        + (list((slow_feedback or {}).get("proposal_guidance") or []) if show_slow else [])
    )
    if guidance:
        lines.append("FEEDBACK GUIDANCE:")
        for item in guidance:
            lines.append(f"- {item}")
        lines.append("")
    # v3.2 §5.2: the generator's own hand-off notes are still extracted and
    # written to generation_notes.json for the audit trail, but they are not
    # replayed into the prompt. They are unstructured text written by a turn
    # that may have been running under the retired ranking rules, and replaying
    # them is a channel through which those rules re-enter verbatim. Restoring
    # the loop needs a structured schema and its own protocol version.
    lines.append(f"Propose the next batch of {int(proposal_count)} `construct(rng)` candidates.")
    return "\n".join(lines)


NOTES_MARKER = "NOTES-FOR-NEXT-GENERATION"
NOTES_MAX_LINES = 10
NOTES_MAX_CHARS = 200


def _extract_notes(text: str) -> list[str]:
    """The generator's hand-off block, truncated at the write boundary.

    The raw reasoning behind a batch runs to tens of thousands of tokens and
    can never travel between turns. What CAN travel is the handful of lines
    the model writes while that reasoning is still in front of it — which is
    a better selection than any after-the-fact summary. Caps are enforced
    here so a verbose batch cannot grow the next prompt.
    """

    upper = text.upper()
    at = upper.find(NOTES_MARKER)
    if at < 0:
        return []
    body = text[at + len(NOTES_MARKER):].lstrip(":").splitlines()
    notes: list[str] = []
    for raw in body:
        line = raw.strip().lstrip("-*").strip()
        if not line:
            if notes:  # a blank line after content ends the block
                break
            continue
        if line.startswith("```"):
            break
        notes.append(line[:NOTES_MAX_CHARS])
        if len(notes) >= NOTES_MAX_LINES:
            break
    return notes


def _generation_notes_lines(prompt_context: dict[str, Any]) -> list[str]:
    """Render archived hand-off notes.

    NOT part of the v3.2 generator prompt (see `_funsearch_user_prompt`): kept
    for artifact rendering and for the ablation tooling that reads the notes
    back out."""

    recent = [
        row for row in list(prompt_context.get("recent_generation_notes") or [])
        if isinstance(row, dict) and row.get("notes")
    ]
    if not recent:
        return []
    lines = ["NOTES YOU LEFT IN RECENT GENERATIONS (newest first):"]
    for row in recent:
        lines.append(f"  turn {row.get('turn')}:")
        for note in list(row.get("notes") or [])[:NOTES_MAX_LINES]:
            lines.append(f"    - {note}")
    return lines


def _budget_use(row: dict[str, Any]) -> str:
    """`228/288` — data qubits plus check ancillas over the campaign budget."""

    n_data, n_checks, budget = row.get("n"), row.get("n_checks"), row.get("n_budget")
    if n_data is None or n_checks is None or not budget:
        return "?"
    return f"{int(n_data) + int(n_checks)}/{int(budget)}"


def _measured_lookup(prompt_context: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]] | None:
    """Fig.3 v2: the measured codes by exact hash and by canonical hash, from the
    overlay's `measured_identities`; None when the profile does not carry them."""

    identities = prompt_context.get("measured_identities")
    if not isinstance(identities, list):
        return None
    by_hash: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for item in identities:
        if not isinstance(item, dict):
            continue
        code_hash = str(item.get("code_hash") or "")
        canonical = str(item.get("canonical_hash") or "")
        if code_hash:
            by_hash.setdefault(code_hash, item)
        if canonical:
            by_canonical.setdefault(canonical, item)
    return {"by_hash": by_hash, "by_canonical": by_canonical}


def _measured_match(summary: dict[str, Any], measured: dict[str, dict[str, dict[str, Any]]] | None) -> tuple[str, dict[str, Any]] | None:
    """("measured" | "twin", identity row) when the summary's code is a measured
    code or a pynauty-isomorphic relabelling of one; None otherwise."""

    if not measured:
        return None
    code_hash = str(summary.get("code_hash") or "")
    if code_hash and code_hash in measured["by_hash"]:
        return "measured", measured["by_hash"][code_hash]
    canonical = str(summary.get("canonical_hash") or "")
    if canonical and canonical in measured["by_canonical"]:
        return "twin", measured["by_canonical"][canonical]
    return None


def _program_summary_line(
    summary: dict[str, Any], *, measured: dict[str, dict[str, dict[str, Any]]] | None = None
) -> str:
    """One prompt line per proposed program (multi-seed outcomes aggregated).

    Fig.3 v2: with ``measured`` (the campaign's measured identities), a program
    whose code is already measured, or is a relabelling of a measured code,
    says so and names that code -- the twin gate will not measure it again."""

    outcome = str(summary.get("outcome") or "")
    family = str(summary.get("family") or "?")
    if outcome in {"accepted", "not_better", "retained"}:
        code = (
            f"n={summary.get('n')} k={summary.get('k')} "
            f"d_search={summary.get('d_search') or summary.get('d_proxy')}"
        )
        # Fig.3 Phase B: the check weight is a structural fact of the proposal
        # (the slow-result lines already carry it); shown when recorded.
        if summary.get("max_check_weight") is not None:
            code += f" w={summary.get('max_check_weight')}"
        # No score echo (v3.2 §5.1): repeating a single number next to every
        # proposal teaches the model to optimise that number.
        tag = {
            "accepted": "recorded in its niche",
            "not_better": "recorded; another code holds this niche",
            "retained": "recorded and kept by the machine's retention rules",
        }[outcome]
        match = _measured_match(summary, measured)
        if match is not None:
            kind, item = match
            target = (
                f"{item.get('program_id') or 'a measured code'} "
                f"(per_logical_per_round={_format_metric(item.get('per_logical_per_round'))})"
            )
            tag = (
                f"ALREADY MEASURED in this campaign as {target}; not measured again"
                if kind == "measured"
                else f"ISOMORPHIC TWIN (a relabelling) of measured {target}; not measured again"
            )
        line = f"{code} -> {tag}"
        seeds = int(summary.get("seeds") or 1)
        distinct = int(summary.get("distinct_codes") or 1)
        if seeds > 1 and distinct > 1:
            line += (
                f" ({seeds} seeds -> {distinct} distinct codes,"
                f" {int(summary.get('accepted_count') or 0)} accepted)"
            )
        return line
    if outcome == "duplicate":
        return f"{family}: duplicate of {summary.get('duplicate_of')}"
    reasons = "; ".join(str(item) for item in summary.get("failure_reasons") or []) or "rejected"
    return f"{family}: REJECTED ({reasons})"


def _refill_funsearch_prompt(
    base_user_prompt: str,
    *,
    requested_total: int,
    parsed_count: int,
    remaining: int,
) -> str:
    return (
        base_user_prompt
        + "\n\nREFILL REQUIRED.\n"
        + f"The previous response in this generation yielded only {int(parsed_count)} "
        + f"parseable code blocks out of the requested {int(requested_total)}.\n"
        + f"Return exactly {int(remaining)} additional ```python fenced code blocks now.\n"
        + "Output no prose, no numbering outside code blocks, and no markdown except the code fences.\n"
        + "Each code block must contain exactly one complete `def construct(rng):` function."
    )


def _baseline_sampler_batch(
    *,
    config: RunConfig,
    prompt_context: dict[str, Any],
    turn: int,
    gen: int,
    generations: int,
    proposal_count: int,
    gen_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """One generation of a search baseline: `proposal_count` draws of the fixed prior (baselines/sampler.py),
    each rendered as a construct(rng) program, plus the generator-call record the strict turn audit reads. Every
    draw is written to the generation's proposals.jsonl, failed draws included (one draw = one raw proposal)."""

    from .baselines.sampler import SAMPLER_ID, draw_generation, program_source

    context = dict(prompt_context.get("baseline_sampler") or {})
    seed_key = str(context.get("seed_key") or "")
    if not seed_key:
        raise RuntimeError("a baseline generation needs prompt_context.baseline_sampler.seed_key")
    families = [family.value if isinstance(family, CodeFamily) else str(family) for family in config.allowed_families]
    draws = draw_generation(
        n_budget=int(config.n_budget),
        families=families,
        seed_key=seed_key,
        turn=int(turn),
        gen=int(gen),
        count=int(proposal_count),
        generations_per_turn=int(generations),
    )
    programs = [program_source(draw) for draw in draws]
    proposals = []
    for proposal_index, (draw, source) in enumerate(zip(draws, programs), start=1):
        row = draw.to_dict()
        row["program_id"] = f"g{gen:04d}_p{proposal_index:04d}"
        row["construct_source"] = source
        proposals.append(row)
    _write_jsonl(gen_dir / "proposals.jsonl", proposals)
    failures: dict[str, int] = {}
    for draw in draws:
        if draw.failure:
            key = str(draw.failure).split(":", 1)[0]
            failures[key] = failures.get(key, 0) + 1
    raw_row = {
        "schema": "funsearch_llm_raw",
        "run_id": config.run_id,
        "gen": gen,
        "provider": BASELINE_PROVIDER,
        "completion_attempt": 1,
        "status": "ok",
        "proposal_count_requested": int(proposal_count),
        "proposal_count_parsed": len(programs),
        "proposal_count_accumulated": len(programs),
        "engine_info": {"provider": BASELINE_PROVIDER, "model": SAMPLER_ID, "seed_key": seed_key},
        "response": "",
        "sampler": {
            "families": families,
            "draws": len(draws),
            "failed_draws": sum(failures.values()),
            "failure_counts": failures,
            "proposals": str(gen_dir / "proposals.jsonl"),
        },
        "generation_error": "",
        "generation_traceback": "",
    }
    return programs, [raw_row]


def _complete_funsearch_prompt(
    *,
    config: RunConfig,
    provider: str,
    system: str,
    user: str,
    prompt_context: dict[str, Any],
    gen: int,
    proposal_count: int,
) -> tuple[str, dict[str, Any]]:
    provider = str(provider)
    if provider == "dry_run":
        return (
            _dry_run_code_blocks(
                prompt_context=prompt_context,
                gen=gen,
                proposal_count=proposal_count,
            ),
            {"provider": "dry_run"},
        )
    # 非 dry_run 一律委托工厂(它持有 SUPPORTED_LLM_PROVIDERS 并对未知 provider 报错),
    # 这样新增 provider(如 openai)只改 factory.py 一处。
    from .llm.factory import create_completion_client

    client = create_completion_client(provider, config.llm)
    return client.complete(system=system, user=user), client.info()


def _dry_run_code_blocks(*, prompt_context: dict[str, Any], gen: int, proposal_count: int) -> str:
    parents = list(prompt_context.get("next_parent_specs") or [])
    blocks = []
    for index in range(max(1, int(proposal_count))):
        if parents:
            source = str(parents[index % len(parents)].get("construct_source") or "")
        else:
            source = SEEDS[(int(gen) + index - 1) % len(SEEDS)]
        blocks.append("```python\n" + source.strip() + "\n```")
    return "\n\n".join(blocks)


def _construction_contract_lines(config: RunConfig) -> list[str]:
    allowed = [
        family.value if isinstance(family, CodeFamily) else str(family)
        for family in config.allowed_families
    ]
    lines: list[str] = [
        "Return a Python dict, not JSON text, from construct(rng).",
        "Every returned scalar must be a Python int; wrap rng outputs with int(...).",
    ]
    weight_cap = getattr(config.verification, "max_check_weight", None)
    exact_weight = getattr(config.verification, "exact_check_weight", None)
    from .construction.registry import LP_CAPPED_WEIGHT_LINE, LP_UNCAPPED_WEIGHT_LINE

    capped = weight_cap is not None or exact_weight is not None
    for plugin in _allowed_plugins(allowed):
        lines.extend(
            LP_CAPPED_WEIGHT_LINE if capped and line == LP_UNCAPPED_WEIGHT_LINE else line
            for line in plugin.user_contract_lines
        )
    if exact_weight is not None:
        # Fig.4 c (2026-10-23): the exact-weight contract replaces the cap line;
        # it is the only prompt clause the variant changes.
        lines.append(
            f"CHECK WEIGHT CONTRACT (run protocol): the code's maximum check weight (max row weight "
            f"over H_X and H_Z) must be EXACTLY {int(exact_weight)}: at least one X or Z check of "
            f"weight {int(exact_weight)} and none heavier. A code whose maximum check weight is not "
            f"{int(exact_weight)} is rejected before any evaluation (check_weight_mismatch), so "
            "proposing one spends a program on nothing."
        )
    elif weight_cap is not None:
        lines.append(
            f"CHECK WEIGHT CAP (run protocol): every X and Z check must have weight <= {int(weight_cap)} "
            "(max row weight of H_X and H_Z). A heavier code is rejected before any evaluation "
            "(max_check_weight_exceeded), so proposing one spends a program on nothing."
        )
    pin = getattr(config, "lp_shape_pin", None)
    if pin is not None and not pin.is_trivial:
        for label, shape in (("A", pin.a), ("B", pin.b)):
            if shape is not None:
                lines.append(
                    f"LP SHAPE PIN (run protocol): base matrix {label} of lp_abelian/"
                    f"lp_nonabelian MUST be exactly {shape[0]}x{shape[1]}; other shapes "
                    "are rejected."
                )
    return lines


def _failure_action_lines(
    *,
    memory: ArchiveMemory,
    prompt_context: dict[str, Any],
    limit: int = 8,
) -> list[str]:
    signatures: list[dict[str, Any]] = []
    signatures.extend(memory.failure_signatures(limit=limit))
    campaign_feedback = prompt_context.get("campaign_feedback")
    if isinstance(campaign_feedback, dict):
        signatures.extend(
            item
            for item in list(campaign_feedback.get("failure_signatures") or [])
            if isinstance(item, dict)
        )
    actions: list[str] = []
    for signature in signatures:
        text = str(signature.get("signature") or "")
        count = int(signature.get("count") or 1)
        for action in _failure_actions_for_signature(text):
            prefix = f"{text} seen {count}x: " if text else ""
            actions.append(prefix + action)
    return _dedupe_strings(actions)[: max(0, int(limit))]


def _failure_actions_for_signature(signature: str) -> list[str]:
    text = signature.lower()
    actions: list[str] = []
    if "bad proto/l" in text or "object is not iterable" in text:
        actions.append(
            "For LP, fix the proto shape: proto[row][col] must be a list of ints such as [3], never a bare int such as 3."
        )
        actions.append(
            "For LP, keep all rows the same length and keep L as a plain positive int."
        )
    if "ragged protograph" in text:
        actions.append("For LP, make proto rectangular; every row must have the same number of cells.")
    if "l out of range" in text:
        actions.append("For LP, choose 1 <= L <= 64.")
    if "total_qubit_budget_violation" in text or "n_budget_violation" in text:
        actions.append(
            "Before returning, compute the family total-qubit formula (data plus one ancilla per check row) and keep it within the total qubit budget."
        )
    if "exceeds the budget cap" in text or "exceeds the resource guard" in text:
        # v3.6 M3: the LP group-order cap is n_budget // 4; the two ways to
        # fit are a smaller group or a base-matrix shape with fewer blocks
        # (neutral: both are stated, neither is recommended over the other).
        actions.append(
            "For lp_nonabelian, the group order is capped at n_budget // 4 (stated "
            "in the format contract). A group over the cap can be replaced by one "
            "under it, or the same group can be combined with a shape that has "
            "fewer blocks so (nA+mA)*(nB+mB)*|G| fits the budget."
        )
    if "disconnected_tanner_graph" in text:
        actions.append(
            "Avoid disconnected Tanner graphs; do not use repeated, separated, or too-degenerate GB/LP terms that isolate checks or qubits."
        )
    if "<3" in text and ("d_search" in text or "d_proxy" in text):
        actions.append(
            "A heuristic distance below 3 means the code cannot correct a single "
            "fault; such a construction is recorded but never measured."
        )
    if "family_not_allowed" in text:
        actions.append(
            "Use only the allowed families listed in the GENOTYPE section; other families are rejected unevaluated."
        )
    if "trust_filter" in text:
        # v3.2 §4.1/§5.1: this screen is a runtime/acquisition screen, not a
        # rejection. The old advice ("increase the distance or reduce n")
        # pushed the model out of the large-n region on every such record —
        # the exact region where high-k candidates live.
        actions.append(
            "The distance-to-size ratio is below the runtime screen. The full "
            "construction is recorded and remains available to the machine's "
            "retention rules; no change is required for it to be kept."
        )
    if "k_nonpositive" in text or "k<=0" in text:
        actions.append(
            "Keep K positive: too many independent checks leave no logical qubits."
        )
    if "invalid_css_commutation" in text:
        actions.append("Use only constructions that enforce exact GF(2) CSS commutation.")
    if "duplicate" in text or "not better than niche elite" in text:
        actions.append(
            "Do not resubmit the same construction or a pure reordering; change exponents or terms enough to change the verified code."
        )
    return actions


def _apply_promotion_evidence(memory: ArchiveMemory, promotion_result: dict[str, Any]) -> int:
    """Reconcile: write slow candidate evidence back into the fast archive."""

    slow_summary = promotion_result.get("slow_summary")
    if not isinstance(slow_summary, dict):
        return 0
    candidates = [row for row in slow_summary.get("candidates") or [] if isinstance(row, dict)]
    if not candidates:
        return 0
    return memory.apply_slow_evidence(candidates)


def _should_promote(gen: int, promote_every: int | None) -> bool:
    if promote_every is None:
        return False
    promote_every = int(promote_every)
    return promote_every > 0 and int(gen) % promote_every == 0


# v3.8 variant generations (config.StructuralSearchConfig.exploit_generations).
VARIANT_RULE_ID = "variant_generation_v1"


def _parent_family(parent: dict[str, Any]) -> str:
    spec = parent.get("spec") if isinstance(parent.get("spec"), dict) else {}
    return str(parent.get("family") or spec.get("family") or "?")


def _variant_prompt_context(
    prompt_context: dict[str, Any],
    exploit_parents: list[dict[str, Any]],
    *,
    gen: int,
    generations: int,
) -> dict[str, Any]:
    """The prompt context of a v3.8 VARIANT generation.

    The parents are the only constructions shown (the turn's 16-parent
    window is replaced, not extended), the policy's strategy brief and
    family quota -- exploration instruments -- are dropped, and the block is
    tagged so the prompt renderer states the rule and the validator can
    check that the last `exploit_generations` generations carried it. The
    archive context, the evidence tables and the verifier feedback stay."""

    context = json.loads(json.dumps(prompt_context, ensure_ascii=False, default=str))
    parents = [dict(parent) for parent in exploit_parents]
    context["next_parent_specs"] = parents
    context["parent_specs_authoritative"] = True
    context["parent_window"] = len(parents)
    context.pop("strategy_brief", None)
    context.pop("family_quota", None)
    feedback = dict(context.get("campaign_feedback") or {})
    feedback["next_parent_specs"] = parents
    context["campaign_feedback"] = feedback
    context["exploit_generation"] = {
        "rule": VARIANT_RULE_ID,
        "generation": int(gen),
        "of": int(generations),
        "parents": [
            {
                "index": index,
                "program_id": parent.get("program_id"),
                "code_hash": parent.get("code_hash"),
                "family": _parent_family(parent),
                "is_baseline": bool(parent.get("is_baseline")),
            }
            for index, parent in enumerate(parents, start=1)
        ],
        "families": sorted({_parent_family(parent) for parent in parents}),
    }
    return context


def _variant_generation_lines(prompt_context: dict[str, Any], *, proposal_count: int) -> list[str]:
    """The rule paragraph of a variant generation (family-neutral: it names
    the parents by index, never a family or a target value)."""

    variant = prompt_context.get("exploit_generation")
    if not isinstance(variant, dict):
        return []
    families = ", ".join(str(item) for item in variant.get("families") or []) or "the parents' families"
    return [
        f"VARIANT GENERATION (protocol rule {variant.get('rule')}, generation "
        f"{int(variant.get('generation') or 0)} of {int(variant.get('of') or 0)} this turn): "
        f"every one of the {int(proposal_count)} code blocks must be a parameter variation of ONE "
        "of the PARENT CONSTRUCTIONS listed below. Keep that parent's family and construction "
        "mechanism; change its parameters (sizes, polynomial or base-matrix entries, group, "
        "shifts, or how rng draws them). Start each block with the comment "
        "`# variant_of: P<index>` naming the parent you varied.",
        f"The verifier admits only codes in the parents' families ({families}) in this "
        "generation; anything else is recorded as rejected. A variation that canonicalizes to "
        "the parent's own code is a duplicate.",
    ]


def _merge_prompt_context_overlay(
    base_context: dict[str, Any],
    overlay: dict[str, Any] | None,
    *,
    parent_limit: int = 12,
) -> dict[str, Any]:
    if not isinstance(overlay, dict):
        return base_context
    merged = json.loads(json.dumps(base_context, ensure_ascii=False, default=str))
    overlay_parents = [
        parent for parent in list(overlay.get("next_parent_specs") or []) if isinstance(parent, dict)
    ]
    base_parents = [
        parent for parent in list(base_context.get("next_parent_specs") or []) if isinstance(parent, dict)
    ]
    if bool(overlay.get("parent_specs_authoritative")):
        # The campaign graph already applied the machine quota (8 structural +
        # 4 runtime, interleaved 2:1) across CROSS-turn pools. Merging a
        # runtime-first list in front of it, as the old prefix merge did, hands
        # parent order back to the measurement ranking. An authoritative EMPTY
        # list (S3 no_feedback) likewise replaces the archive's own parents.
        merged["next_parent_specs"] = overlay_parents[: max(0, int(parent_limit))]
    elif overlay_parents:
        merged["next_parent_specs"] = _merge_parent_specs(overlay_parents, base_parents, limit=parent_limit)
    slow_feedback = _slow_feedback_from_context(overlay)
    if slow_feedback:
        merged["slow_feedback"] = slow_feedback
    if isinstance(overlay.get("strategy_brief"), str):
        merged["strategy_brief"] = overlay["strategy_brief"]
    if isinstance(overlay.get("family_quota"), dict):
        merged["family_quota"] = {
            str(family): int(count) for family, count in overlay["family_quota"].items()
        }
    if isinstance(overlay.get("recent_generation_notes"), list):
        merged["recent_generation_notes"] = list(overlay["recent_generation_notes"])
    if isinstance(overlay.get("reference_codes"), dict):
        merged["reference_codes"] = overlay["reference_codes"]
    if isinstance(overlay.get("evidence_tables"), dict):
        merged["evidence_tables"] = overlay["evidence_tables"]
    if overlay.get("parent_window") is not None:
        merged["parent_window"] = overlay.get("parent_window")
    # 2026-11-06 search baselines: the fixed prior's seed key (never shown to a model; there is none).
    if isinstance(overlay.get("baseline_sampler"), dict):
        merged["baseline_sampler"] = dict(overlay["baseline_sampler"])
    # Fig.3 v2: measured-code identities and the previous turn's queue summary.
    if isinstance(overlay.get("measured_identities"), list):
        merged["measured_identities"] = list(overlay["measured_identities"])
    if isinstance(overlay.get("last_turn_queue"), dict):
        merged["last_turn_queue"] = dict(overlay["last_turn_queue"])
    history_policy = dict(merged.get("history_policy") or {})
    proposal_agent_sees = list(history_policy.get("proposal_agent_sees") or [])
    if slow_feedback and "slow_feedback" not in proposal_agent_sees:
        proposal_agent_sees.insert(0, "slow_feedback")
    history_policy["proposal_agent_sees"] = proposal_agent_sees
    merged["history_policy"] = history_policy
    campaign_feedback = dict(merged.get("campaign_feedback") or {})
    overlay_feedback = dict(overlay.get("campaign_feedback") or {})
    if slow_feedback:
        campaign_feedback["slow_feedback"] = slow_feedback
    if "next_parent_specs" in merged:
        campaign_feedback["next_parent_specs"] = merged["next_parent_specs"]
    guidance = _dedupe_strings(
        list(campaign_feedback.get("proposal_guidance") or [])
        + list(overlay_feedback.get("proposal_guidance") or [])
        + list((slow_feedback or {}).get("proposal_guidance") or [])
    )
    if guidance:
        campaign_feedback["proposal_guidance"] = guidance
    merged["campaign_feedback"] = campaign_feedback
    merged["schema"] = "prompt_context_with_overlay"
    return merged


def _merge_parent_specs(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent in primary + fallback:
        key = str(parent.get("code_hash") or parent.get("program_id") or parent.get("archive_cell") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(parent)
        if len(merged) >= max(0, int(limit)):
            break
    return merged


def _slow_feedback_from_context(context: dict[str, Any]) -> dict[str, Any] | None:
    direct = context.get("slow_feedback")
    if isinstance(direct, dict):
        return direct
    campaign_feedback = context.get("campaign_feedback")
    if isinstance(campaign_feedback, dict) and isinstance(campaign_feedback.get("slow_feedback"), dict):
        return campaign_feedback["slow_feedback"]
    return None


def _slow_feedback_prompt_lines(
    slow_feedback: dict[str, Any],
    prompt_context: dict[str, Any],
) -> list[str]:
    # v3.2 §5.1: runtime measurements appear as FACTS. No instruction is
    # attached to them ("prefer these", "best first"): parent, promotion and
    # final eligibility are machine decisions, and a ranked list in the prompt
    # is an instruction to rank by that quantity.
    if str(prompt_context.get("task") or "memory") == "cnot_dense":
        lines = [
            "RUNTIME FACTS (circuit-level measurements of the two-block CNOT-dense "
            "circuit under the frozen decoder; per_logical_per_round = logical error "
            "per logical qubit per QEC round of that circuit):"
        ]
    else:
        lines = ["BPOSD RUNTIME FACTS (circuit-level measurements under the frozen decoder):"]
    best_results = [row for row in list(slow_feedback.get("best_results") or []) if isinstance(row, dict)]
    # Audit fix #6 (defense in depth): render in stable hash order even if an
    # older slow_feedback artifact still carries the plpr-ranked list — the
    # display order is itself an implicit ranking instruction.
    best_results.sort(
        key=lambda row: (
            str(row.get("code_hash") or ""),
            str(row.get("program_id") or ""),
        )
    )
    if best_results:
        for row in best_results[:6]:
            lines.append("  - " + _slow_result_line(row))
    lines.extend(_not_competitive_lines(prompt_context))
    return lines


def _last_turn_queue_lines(prompt_context: dict[str, Any]) -> list[str]:
    """Fig.3 v2: what became of the previous turn's slow queue. Facts, then the
    one rule that explains them: a code is measured once, and a relabelling of
    a measured code (same Tanner graph up to permutation) is rejected before
    measurement."""

    queue = prompt_context.get("last_turn_queue")
    if not isinstance(queue, dict):
        return []
    cap = queue.get("cap")
    selected = int(queue.get("selected") or 0)
    seen = queue.get("candidates_seen")
    twins = int(queue.get("twin_rejected_count") or 0)
    measured = int(queue.get("measured_rejected_count") or 0)
    gated = int(queue.get("weight_gate_rejected_count") or 0)
    head = f"LAST TURN'S SLOW QUEUE (turn {queue.get('turn')}): {selected} of {cap} slots filled"
    head += " -> nothing was measured." if selected == 0 else "."
    parts: list[str] = []
    if twins:
        parts.append(f"{twins} were isomorphic twins (relabellings) of codes this campaign has already measured")
    if measured:
        parts.append(f"{measured} were already-measured codes proposed again")
    if gated:
        weights = ", ".join(str(w) for w in list(queue.get("weight_gate_weights") or [])[:12])
        parts.append(
            f"{gated} exceeded the check-weight gate (w <= {queue.get('weight_gate_max')}; proposed w {weights})"
        )
    if parts:
        head += f" Of the {seen if seen is not None else '?'} candidates: " + "; ".join(parts) + "."
    lines = [head]

    def _row(row: dict[str, Any]) -> str:
        return (
            f"{row.get('family')} n={row.get('n')} k={row.get('k')} d_search={row.get('d_search')} "
            f"w={row.get('max_check_weight')} ({row.get('program_id') or row.get('code_hash')})"
        )

    for row in list(queue.get("twin_rejected") or [])[:8]:
        if not isinstance(row, dict):
            continue
        twin = row.get("twin") or {}
        if twin.get("program_id"):
            lines.append(
                f"  - {_row(row)} = twin of measured {twin.get('program_id')} "
                f"(per_logical_per_round={_format_metric(twin.get('per_logical_per_round'))})"
            )
        elif row.get("twin_of") == "selected_this_turn":
            lines.append(f"  - {_row(row)} = twin of a code already in that turn's queue")
        else:
            lines.append(f"  - {_row(row)} = twin of a measured code")
    for row in list(queue.get("measured_rejected") or [])[:8]:
        if not isinstance(row, dict):
            continue
        item = row.get("measured") or {}
        lines.append(
            f"  - {_row(row)} = already measured"
            + (
                f" as {item.get('program_id')} (per_logical_per_round={_format_metric(item.get('per_logical_per_round'))})"
                if item.get("program_id")
                else ""
            )
        )
    lines.append(
        "A code is measured once. Re-proposing a measured code, or a permutation-equivalent "
        "relabelling of it (different generators, same Tanner graph), is rejected before "
        "measurement and spends the proposal on nothing. This changes nothing about the batch: "
        "return the full batch of programs every generation, and spend it on constructions that "
        "are not relabellings of measured codes (other groups, sizes, shapes, families) rather "
        "than on fewer programs."
    )
    return lines


def _not_competitive_lines(prompt_context: dict[str, Any]) -> list[str]:
    """Fig.3 Phase B (2026-10-12, user decision): the one place the prompt
    carries a judgement. Codes measured in THIS campaign whose task error is
    at least `ratio` x the campaign's best with non-overlapping confidence
    intervals, and the (check weight, d_search) classes whose best is that far
    off. Everything here is this campaign's own measurement; a twin of a listed
    code is rejected before measurement by the canonical-hash gate, which is
    why re-proposing one is stated as a wasted proposal."""

    tables = prompt_context.get("evidence_tables")
    if not isinstance(tables, dict):
        return []
    rows = [row for row in list(tables.get("not_competitive") or []) if isinstance(row, dict)]
    classes = [row for row in list(tables.get("losing_classes") or []) if isinstance(row, dict)]
    if not rows and not classes:
        return []
    ratio = tables.get("not_competitive_ratio")
    lines = [
        "NOT COMPETITIVE UNDER THIS TASK (measured in this campaign; task error >= "
        f"{_format_metric(ratio)}x the current best, confidence intervals not overlapping). "
        "These codes and their isomorphic twins are not candidates: a twin resubmission "
        "is rejected before measurement, so proposing one spends a proposal on nothing."
    ]
    for row in rows:
        note = ""
        if str(row.get("stopped_by") or "") == "racing":
            note = " [stopped early by racing: lost to the incumbent at pilot depth]"
        lines.append(
            f"  - {row.get('program_id')} family={row.get('family')} n={row.get('n')} "
            f"k={row.get('k')} d_search={row.get('d_search')} w={row.get('max_check_weight')} "
            f"per_logical_per_round={_format_metric(row.get('per_logical_per_round'))} "
            f"({_format_metric(row.get('ratio_to_best'))}x best){note}"
        )
    if classes:
        lines.append(
            "CLASSES NOT COMPETITIVE SO FAR (check weight, d_search; the class's best "
            "measurement vs the campaign's best):"
        )
        for row in classes:
            lines.append(
                f"  - w={row.get('max_check_weight')} d_search={row.get('d_search')}: "
                f"measured={row.get('measured')} best={_format_metric(row.get('best_plpr'))} "
                f"({_format_metric(row.get('ratio_to_best'))}x best)"
            )
    return lines


def _parent_window(prompt_context: dict[str, Any]) -> int:
    """How many parents the generator sees: the campaign's parent window
    (v3.7: 16) when the overlay declares it, else every parent carried."""

    declared = (prompt_context.get("campaign_feedback") or {}).get("parent_window")
    if declared is None:
        declared = prompt_context.get("parent_window")
    try:
        if declared is not None and int(declared) > 0:
            return int(declared)
    except (TypeError, ValueError):
        pass
    return max(16, len(list(prompt_context.get("next_parent_specs") or [])))


def _evidence_table_lines(prompt_context: dict[str, Any]) -> list[str]:
    """v3.7 measurement facts by check weight and by A/B symmetry: proposed,
    measured, best and median plpr per class. Numbers only -- no verb, no
    ranking, no suggestion; the reader decides what, if anything, they mean."""

    tables = prompt_context.get("evidence_tables")
    if not isinstance(tables, dict):
        return []
    lines: list[str] = []
    by_weight = [row for row in list(tables.get("by_check_weight") or []) if isinstance(row, dict)]
    if by_weight:
        lines.append(
            "MEASUREMENT FACTS BY CHECK WEIGHT (this campaign so far; a check of "
            "weight w is measured with w CNOTs, hook errors grow with w, and the "
            "measured per_logical_per_round already contains that cost):"
        )
        for row in by_weight:
            # v3.7.1: `constructed` counts every candidate written at this
            # weight, `below_d_floor` how many of them the d_search floor kept
            # out of the eligible pool. Without those two a class whose codes
            # all fall below the floor is simply absent from this table.
            extra = ""
            if row.get("constructed"):
                extra = (
                    f" constructed={row.get('constructed')}"
                    f" below_d_floor={row.get('below_distance_floor')}"
                )
            lines.append(
                f"  w={row.get('max_check_weight')}:{extra} proposed={row.get('proposed')} "
                f"measured={row.get('measured')} best_plpr={_format_metric(row.get('best_plpr'))} "
                f"median_plpr={_format_metric(row.get('median_plpr'))}"
            )
    by_symmetry = [row for row in list(tables.get("by_symmetry") or []) if isinstance(row, dict)]
    if by_symmetry:
        lines.append(
            "MEASUREMENT FACTS BY A/B WEIGHT SYMMETRY (check weight = |A| + |B|; an odd "
            "check weight needs |A| != |B|; LP A/B may be 1x1, 1x2, 2x2, ...):"
        )
        for row in by_symmetry:
            lines.append(
                f"  {row.get('class')}: proposed={row.get('proposed')} measured={row.get('measured')} "
                f"best_plpr={_format_metric(row.get('best_plpr'))} median_plpr={_format_metric(row.get('median_plpr'))}"
            )
    return lines


def _parent_construction_lines(
    prompt_context: dict[str, Any],
    *,
    limit: int = 16,
    show_fast: bool = True,
    show_slow: bool = True,
) -> list[str]:
    """Render the strongest parent constructions with spec AND program source.

    The program (not the spec) is the unit of inheritance here: rng-driven
    sampler programs are the empirically dominant winners, and showing only
    the spec would cut off mutation of the sampling strategy itself. Source
    is indented plain text, never fenced, so it cannot be confused with the
    model's own required ```python output blocks. Parents are model-produced
    search state only; the resident baseline never enters the parent pool."""

    # v3.8 variant generation: the resident reference code IS a parent (the
    # protocol hands it to every arm alike); outside variant generations the
    # v3.2 rule stands and the baseline never enters the parent pool.
    variant = isinstance(prompt_context.get("exploit_generation"), dict)
    parents = [
        parent
        for parent in list(prompt_context.get("next_parent_specs") or [])
        if isinstance(parent, dict)
        and (variant or not parent.get("is_baseline"))
        and isinstance(parent.get("spec"), dict)
        and str(parent.get("construct_source") or "").strip()
    ][: max(0, int(limit))]
    if not parents:
        return []
    if variant:
        lines = [
            "PARENT CONSTRUCTIONS FOR THIS VARIANT GENERATION (the measured-best "
            "codes of this campaign and the resident reference code of this budget "
            "point, rendered in the protocol's order; every block must vary one of "
            "them and name it with `# variant_of: P<index>`):"
        ]
    else:
        lines = [
            "SELECTED PARENT CONSTRUCTIONS (chosen by the protocol's retention "
            "rules and rendered in its order; mutate, recombine, or rescale them - "
            "the program is the unit of inheritance, the spec is its best draw):"
        ]
    # Warm-start seeds (program_id like 'seed_NNNN') are codes carried in
    # from a SMALLER qubit budget. While any of them is still a parent, tag
    # it; once the run's own codes displace them the tag vanishes on its own.
    has_seed_parent = False
    for index, parent in enumerate(parents, start=1):
        channels = list(parent.get("selection_channels") or [])
        code = (
            f"n={parent.get('n')} k={parent.get('k')} "
            f"d_search={parent.get('d_search') or parent.get('d_proxy')}"
        )
        # Structural parents carry NO comparative score (v3.2 §5.1): a number
        # printed on every parent line is a ranking the model can sort by, and
        # sorting parents by it is precisely what the machine order replaces.
        # A runtime parent's measurement is stated as a fact, with no verb.
        plpr = parent.get("slow_per_logical_per_round")
        score = ""
        if show_slow and plpr is not None and (variant or "runtime" in (channels or ["runtime"])):
            score = f" per_logical_per_round={_format_metric(plpr)}"
        is_seed = str(parent.get("program_id") or "").startswith("seed_")
        tag = "  [from a smaller qubit budget]" if is_seed else ""
        if parent.get("is_baseline"):
            tag += "  [resident reference code of this budget point]"
        usage = _budget_use(parent)
        if usage != "?":
            tag += f"  [spends {usage}]"
        has_seed_parent = has_seed_parent or is_seed
        channel_text = f" channel={'+'.join(channels)}" if channels else ""
        # v3.7: the check weight and the A/B split are facts on every parent
        # line (they were only in the JSON artifact before).
        weight_text = ""
        if parent.get("max_check_weight") is not None:
            weight_text = f" check_weight={parent.get('max_check_weight')}"
        profile = parent.get("weight_profile") if isinstance(parent.get("weight_profile"), dict) else None
        if profile is None and isinstance(parent.get("spec"), dict):
            from omniqec_agentic.construction.structure import weight_profile as _wp

            profile = _wp(parent.get("spec"))
        if profile and profile.get("label"):
            weight_text += f" ({profile['label']})"
        label = f"P{index}" if variant else f"#{index}"
        lines.append(
            f"{label} {code} {parent.get('family')}{weight_text}{channel_text}{score} "
            f"(rng_seed={int(parent.get('rng_seed') or 0)}){tag}"
        )
        lines.append(f"   spec: {parent.get('spec')}")
        lines.append("   program:")
        for source_line in str(parent.get("construct_source")).strip("\n").splitlines():
            lines.append(f"       {source_line}")
    if has_seed_parent:
        lines.append(
            "Parents tagged [from a smaller qubit budget] were found under a "
            "TIGHTER budget than this campaign's."
        )
    return lines


def _slow_result_line(row: dict[str, Any]) -> str:
    code = (
        f"n={row.get('n')} k={row.get('k')} "
        f"d_search={row.get('d_search') or row.get('d_proxy')} "
        f"w={row.get('max_check_weight')}"
    )
    line = (
        f"{row.get('program_id')} {code} family={row.get('family')} "
        f"decoder={row.get('decoder')} p={row.get('p')} "
        f"per_logical_per_round={_format_metric(row.get('per_logical_per_round'))}"
    )
    return line + _optional_evidence_suffix(row)


def _optional_evidence_suffix(row: dict[str, Any]) -> str:
    """Render distance/hook evidence only when it exists — absent evidence is
    the norm in single-p runs and printing 'unknown' for every field buries
    the signal."""

    parts: list[str] = []
    # Racing stops a candidate as soon as it cannot beat the incumbent, at pilot
    # depth. Hiding that made an unresolved measurement look like a settled bad
    # one, so a region could be written off on evidence that was never gathered.
    if str(row.get("stopped_by") or "") == "racing":
        parts.append("racing_stopped_early(low-precision, not a settled result)")
    if row.get("exact_d_circ") is not None:
        parts.append(f"exact_d_circ={_format_metric(row.get('exact_d_circ'))}")
    if row.get("d_circ_ler_estimate") is not None:
        parts.append(f"d_circ_estimate={_format_metric(row.get('d_circ_ler_estimate'))}")
    if row.get("d_circ_isd_upper_bound") is not None:
        parts.append(f"d_circ_upper_bound={_format_metric(row.get('d_circ_isd_upper_bound'))}")
    if row.get("degraded_estimate") or row.get("degraded_estimate_any_decoder") or row.get("degraded_isd"):
        parts.append("HOOK-DEGRADED")
    return (" " + " ".join(parts)) if parts else ""


def _format_metric(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _dedupe_strings(items: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _extract_programs(text: str, *, limit: int | None = None) -> list[str]:
    blocks = _CODE_BLOCK.findall(text or "")
    candidates = [block for block in blocks if "def construct" in block]
    if not candidates and text and "def construct" in text:
        candidates = [text]
    if not candidates:
        return []
    programs = [normalize_source_text(candidate) for candidate in candidates]
    if limit is not None:
        return programs[: max(0, int(limit))]
    return programs


def _generation_status_from_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "missing deepseek" in text:
        return "blocked_missing_api_key"
    if isinstance(exc, ImportError):
        return "blocked_missing_dependency"
    return "generation_error"


def _transcript_attempt(
    attempt: dict[str, Any],
    *,
    gen: int,
    source: str,
    proposal_index: int | None = None,
) -> dict[str, Any]:
    row = {
        "kind": "attempt",
        "gen": int(gen),
        "source": str(source),
        "program_id": attempt.get("program_id"),
        "family": attempt.get("family") or "?",
        "accepted": bool(attempt.get("accepted")),
        "archive_status": attempt.get("archive_status"),
        "duplicate_of": attempt.get("duplicate_of"),
        "error": attempt.get("error") or "",
        "failure_reasons": list(attempt.get("failure_reasons") or []),
    }
    if proposal_index is not None:
        row["proposal_index"] = int(proposal_index)
    for src, dst in (
        ("n", "n"),
        ("k", "k"),
        ("d_proxy", "d"),
        ("Q", "Q"),
        ("archive_cell", "cell"),
        ("code_hash", "code_hash"),
    ):
        if attempt.get(src) is not None:
            row[dst] = attempt.get(src)
    return row


def _append_transcript(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_status(
    path: Path,
    *,
    memory: ArchiveMemory,
    config: RunConfig,
    gen: int,
    generations: int,
    provider: str,
    status_counts: dict[str, int],
) -> None:
    """One small machine-readable heartbeat per generation, plus a log line.

    This is the cheap thing a monitoring agent polls instead of parsing
    transcripts: current generation, archive size, best Q_eff elite, best
    slow-verified LER, and quarantine count."""

    import time as _time

    counts = memory.counts()
    rows = memory.archive_summary(limit=1)
    best = rows[0] if rows else {}
    best_ler = memory.best_slow_per_logical_per_round()

    # Monotonic headline champion = LER hall-of-fame top (survives the Q-driven
    # niche churn that makes the per-gen archive min non-monotonic). plpr is the
    # goal metric; we also surface the best k>=4 code as a fair-vs-BB144 lens.
    def _label(rec: Any) -> str:
        f = getattr(rec, "fast", None)
        return f"[[{getattr(f, 'n_data', None)},{getattr(f, 'k', None)},{getattr(f, 'd_proxy', None)}]]"

    hof = list(getattr(memory, "hall_of_fame", []) or [])
    champ = hof[0] if hof else None
    champ_plpr = memory._record_plpr(champ) if champ is not None else None
    champ_k4 = next((r for r in hof if getattr(r.fast, "k", 0) and r.fast.k >= 4), None)
    prev_champ = None
    try:
        if path.exists():
            prev_champ = json.loads(path.read_text()).get("champion_plpr")
    except Exception:
        prev_champ = None
    payload = {
        "schema": "funsearch_status",
        "run_id": config.run_id,
        "provider": str(provider),
        "updated_unix": _time.time(),
        "gen": int(gen),
        "generations": int(generations),
        "archive_cells": counts["archive_cells"],
        "accepted": counts["accepted"],
        "attempts": counts["attempts"],
        "duplicates": counts["duplicates"],
        "degraded_elites": counts.get("degraded_elites", 0),
        "best_cell": best.get("cell"),
        "best_code": f"[[{best.get('n')},{best.get('k')},{best.get('d_proxy')}]]" if best else None,
        "best_q_eff": best.get("q_eff"),
        "best_slow_per_logical_per_round": best_ler,
        "champion_plpr": champ_plpr,
        "champion_code": _label(champ) if champ is not None else None,
        "hall_of_fame_size": len(hof),
        "generation_status_counts": dict(status_counts),
    }
    _write_json(path, payload)
    try:
        hof_rows = []
        for rec in hof:
            f = getattr(rec, "fast", None)
            ev = getattr(rec, "slow_evidence", None) or {}
            hof_rows.append({
                "code": _label(rec),
                "family": getattr(f, "family", None),
                "n": getattr(f, "n_data", None),
                "k": getattr(f, "k", None),
                "d": getattr(f, "d_proxy", None),
                "plpr": memory._record_plpr(rec),
                "decoder": ev.get("decoder"),
                "program_id": getattr(rec.program, "program_id", None),
            })
        _write_json(path.parent / "hall_of_fame.json", {"schema": "hall_of_fame", "hall_of_fame": hof_rows})
    except Exception:
        pass
    new_best = False
    try:
        new_best = champ_plpr is not None and (
            prev_champ is None or float(champ_plpr) < float(prev_champ)
        )
    except (TypeError, ValueError):
        new_best = False
    marker = f"  ★NEW-BEST 冠军(原 {prev_champ})" if new_best else ""
    print(
        f"[gen {gen}/{generations}] cells={counts['archive_cells']} accepted={counts['accepted']} "
        f"best={payload['best_code']} q_eff={best.get('q_eff') if best else None} "
        f"best_plpr={best_ler} degraded={payload['degraded_elites']}{marker}",
        flush=True,
    )
    if champ is not None:
        ev = getattr(champ, "slow_evidence", None) or {}
        fam = getattr(champ.fast, "family", "?") if getattr(champ, "fast", None) else "?"
        line = f"   ★史上最优 {_label(champ)} {fam} plpr={champ_plpr} by={ev.get('decoder')}"
        if champ_k4 is not None and champ_k4 is not champ:
            ev4 = getattr(champ_k4, "slow_evidence", None) or {}
            line += f"  |  k>=4最优 {_label(champ_k4)} plpr={memory._record_plpr(champ_k4)} by={ev4.get('decoder')}"
        print(line, flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
