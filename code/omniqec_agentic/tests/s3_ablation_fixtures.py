"""Shared fixtures for the S3 ablation goldens and tests (2026-10-23).

Everything here renders what the two LLM roles SEE (generator prompt,
orchestrator context blocks) and what the protocol freezes (snapshot) for a
fixed, rich campaign state, so a test can assert that the ablation switch, when
off, changes none of it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omniqec_agentic.config import (
    CodeFamily,
    HardwareConfig,
    HardwareTarget,
    RunConfig,
    VerificationConfig,
    fig3_cnot_structural_config,
    formal_racing_for_max_errors,
)
from omniqec_agentic.funsearch import SEEDS, _build_funsearch_prompt
from omniqec_agentic.memory import ArchiveMemory

MODEL_OF = {"memory": "claude-opus-5", "cnot_dense": "claude-opus-5(medium)"}


def verification_for(task: str) -> VerificationConfig:
    if task == "memory":
        return VerificationConfig()
    structural = fig3_cnot_structural_config()
    return VerificationConfig(
        task="cnot_dense", structural=structural, promote_top=int(structural.slow_candidate_cap),
        max_check_weight=6, rounds_policy="fixed_Tc",
    )


def run_config(task: str, mode: str, *, budget: int = 98) -> RunConfig:
    return RunConfig(
        run_id="s3golden", n_budget=budget, hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused", proposal_count=24, allowed_families=tuple(CodeFamily),
        verification=verification_for(task), feedback_mode=mode,
    )


def seeded_memory(budget: int = 98) -> ArchiveMemory:
    """Two built-in seed programs evaluated (cheap proxy) so the archive,
    the recent-programs and the failure blocks all have rows."""

    from omniqec_agentic.schemas import CandidateProgram

    memory = ArchiveMemory(allowed_families=tuple(f.value for f in CodeFamily))
    for index, source in enumerate(list(SEEDS)[:2], start=1):
        memory.evaluate_and_record(
            CandidateProgram(program_id=f"seed_{index:04d}", source=source, rng_seed=0, n_budget=budget,
                             source_agent="seed", metadata={"funsearch_seed_index": index}),
            n_budget=budget, proxy_iters=20,
        )
    return memory


def rich_context(task: str) -> dict[str, Any]:
    """Every block the generator can be shown, including the Fig.3 ones."""

    parent_src = "def construct(rng):\n    return {'family': 'bb'}"
    return {
        "task": task,
        "strategy_brief": "STRATEGY CONTEXT (structured allocation from the campaign policy):\n- family attention: bb: 0.50, hgp: 0.50",
        "family_quota": {"bb": 12, "hgp": 12},
        "parent_window": 16,
        "next_parent_specs": [
            {"program_id": "struct_aa", "code_hash": "a1" * 12, "family": "bb", "n": 72, "n_checks": 72, "n_budget": 98,
             "k": 12, "d_search": 6, "max_check_weight": 6, "rng_seed": 0, "selection_channels": ["structural"],
             "spec": {"family": "bb", "l": 6, "m": 6, "A": [[1, 0], [0, 1], [0, 2]], "B": [[0, 1], [1, 0], [2, 0]]},
             "construct_source": parent_src},
            {"program_id": "rt_bb", "code_hash": "b2" * 12, "family": "hgp", "n": 45, "n_checks": 40, "n_budget": 98,
             "k": 4, "d_search": 5, "max_check_weight": 5, "rng_seed": 0, "selection_channels": ["runtime"],
             "slow_per_logical_per_round": 1.5e-3, "spec": {"family": "hgp"}, "construct_source": parent_src},
            {"program_id": "div_lp", "code_hash": "c3" * 12, "family": "lp_abelian", "n": 46, "n_checks": 44, "n_budget": 98,
             "k": 2, "d_search": 6, "max_check_weight": 4, "rng_seed": 3, "selection_channels": ["diversity"],
             "spec": {"family": "lp_abelian"}, "construct_source": parent_src},
        ],
        "slow_feedback": {
            "best_results": [
                {"program_id": "rt_bb", "code_hash": "b2" * 12, "n": 45, "k": 4, "d_search": 5, "max_check_weight": 5,
                 "family": "hgp", "decoder": "telescoping_cpu", "p": 0.005, "per_logical_per_round": 1.5e-3,
                 "stopped_by": "racing"},
            ],
            "proposal_guidance": ["1 candidate(s) completed circuit-level measurement."],
        },
        "evidence_tables": {
            "by_check_weight": [{"max_check_weight": 5, "constructed": 30, "below_distance_floor": 4, "proposed": 20,
                                 "measured": 1, "best_plpr": 1.5e-3, "median_plpr": 1.5e-3}],
            "by_symmetry": [{"class": "bb symmetric |A|=|B|", "proposed": 10, "measured": 1, "best_plpr": 1.5e-3,
                             "median_plpr": 1.5e-3}],
            "not_competitive": [{"program_id": "g0002_p0003", "family": "bb", "n": 72, "k": 2, "d_search": 5,
                                 "max_check_weight": 6, "per_logical_per_round": 9e-3, "ratio_to_best": 6.0,
                                 "stopped_by": "racing"}],
            "losing_classes": [{"max_check_weight": 6, "d_search": 5, "measured": 2, "best_plpr": 9e-3, "ratio_to_best": 6.0}],
            "not_competitive_ratio": 2.0,
        },
        "measured_identities": [{"code_hash": "b2" * 12, "canonical_hash": "CANON_B", "program_id": "rt_bb",
                                 "per_logical_per_round": 1.5e-3, "family": "hgp", "n": 45, "k": 4, "d_search": 5,
                                 "max_check_weight": 5}],
        "last_turn_queue": {"turn": 4, "cap": 7, "selected": 5, "candidates_seen": 9, "twin_rejected_count": 1,
                            "measured_rejected_count": 1, "weight_gate_rejected_count": 0,
                            "twin_rejected": [{"family": "hgp", "n": 45, "k": 4, "d_search": 5, "max_check_weight": 5,
                                               "program_id": "g0003_p0001",
                                               "twin": {"program_id": "rt_bb", "per_logical_per_round": 1.5e-3}}],
                            "measured_rejected": [{"family": "hgp", "n": 45, "k": 4, "d_search": 5, "max_check_weight": 5,
                                                   "program_id": "g0003_p0002",
                                                   "measured": {"program_id": "rt_bb", "per_logical_per_round": 1.5e-3}}]},
    }


def prompt_text(task: str, mode: str) -> str:
    prompt = _build_funsearch_prompt(
        memory=seeded_memory(), config=run_config(task, mode), gen=3, proposal_count=24,
        prompt_context=rich_context(task),
    )
    return prompt["system_prompt"] + "\n=====USER=====\n" + prompt["user_prompt"] + "\n"


def campaign_state(run_dir: Path, task: str = "memory") -> dict[str, Any]:
    return {
        "run_id": "campaign_b98", "turn": 5, "run_dir": str(run_dir),
        "spec": {"qubit_budget": 98, "families": [f.value for f in CodeFamily]},
        "protocol": {"max_turns": 30, "tool_budget": 32, "proposal_count": 24, "promote_top": 6,
                     "generations_per_turn": 4, "feedback_mode": "both"},
        "budget": {"turns_used": 5, "tokens_used": 1000, "slow_shots_used": 50000},
        "hof_summary": [{"code_hash": "b2" * 12, "program_id": "rt_bb", "family": "hgp", "n": 45, "k": 4,
                         "d_search": 5, "max_check_weight": 5, "per_logical_per_round": 1.5e-3,
                         "spec": {"family": "hgp"}, "first_seen_turn": 2, "last_measured_turn": 4}],
        "baseline_summary": {},
        "structural_proxy_elites": [{"code_hash": "a1" * 12, "family": "bb", "n": 72, "n_total": 144, "k": 12,
                                     "d_search": 6, "max_check_weight": 6, "pareto_rank": 0, "crowding_boundary": True}],
        "plan": {"family_mix": {"bb": 0.5, "hgp": 0.5}, "strategy_mix": {"novelty": 1.0}},
        "last_cycle": {"status": "ok", "artifact_paths": {}},
    }


def orchestrator_blocks_text(run_dir: Path) -> str:
    from omniqec_agentic.orchestrator.loop import AgenticPolicy
    from omniqec_agentic.orchestrator.manual import render_dynamic, render_static
    from omniqec_agentic.tools.context import ToolContext

    state = campaign_state(run_dir)
    manual_dir = run_dir / "manual"
    ctx = ToolContext(run_dir=run_dir, manual_dir=manual_dir, state=dict(state), turn=5)
    render_static(manual_dir)
    render_dynamic(manual_dir, ledger=ctx.ledger, bandit=ctx.bandit, digest=None)
    text, registry = AgenticPolicy(provider="fake")._assemble_blocks(state, ctx)
    return text + "\n=====REGISTRY=====\n" + json.dumps(registry, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def snapshot_dict(task: str, **overrides: Any) -> dict[str, Any]:
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    kwargs: dict[str, Any] = dict(
        verification=verification_for(task), racing=formal_racing_for_max_errors(25000, None), max_turns=30,
        proposal_count=24, generations_per_turn=4, slow_workers=127, shot_shards=16, final_shot_shards=128,
        feedback_mode="both", final_shots=25000, strict_turns=True, required_provider="openai",
        required_model=MODEL_OF[task], rng_seeds_per_program=40, n_budget=98, fixed_program_count=True,
        promotion_max_check_weight=6 if task == "cnot_dense" else None,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)
