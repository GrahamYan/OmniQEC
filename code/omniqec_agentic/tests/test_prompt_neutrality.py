"""No optimisation target in the generator's text, and no way back in (§5).

The design constraint is precise: the search must find codes with a large
absolute `k` OR a large `d_search`, and that pressure must come from the
machine's retention rules — which candidates get slow slots, which programs
survive across turns, which become parents, which get final slots — and never
from a sentence in the prompt.

That makes two things testable. First, the assembled generator prompt contains
no global ranking instruction and no explicit k/d goal. Second, the three
channels that used to carry policy prose into that prompt verbatim (plan
directives, reflector advice, raw generation notes) cannot put one back.
"""

from __future__ import annotations

import re

import pytest

from omniqec_agentic.config import (
    CodeFamily,
    HardwareConfig,
    HardwareTarget,
    RunConfig,
    StructuralSearchConfig,
)
from omniqec_agentic.funsearch import (
    _build_funsearch_prompt,
    _funsearch_system_prompt,
    _funsearch_user_prompt,
    _parent_construction_lines,
)
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.orchestrator import manual as manual_mod
from omniqec_agentic.orchestrator.prompts import load_prompt

# The exact strings the retired rules were written as (v3.2 §7.4).
BANNED_PHRASES = (
    "PLPR decides",
    "larger k wins",
    "higher is better",
    "best first",
    "strongest verified",
    "do not clone",
    "or reduce n",
    "prefer these over Q",
    "KNOWN DEAD END",
    "beat an elite",
)
# An explicit statement of the goal itself, in any of its usual shapes.
GOAL_RE = re.compile(
    r"(maximi[sz]e|increase|prefer|aim for|target)\s+(the\s+)?(absolute\s+)?"
    r"(larger\s+)?(k|d|distance|logical qubits)\b",
    re.I,
)


def _flat(text: str) -> str:
    """Collapse wrapping so a phrase assertion is about the words, not the
    column at which the source file happened to break the line."""

    return " ".join(text.split())


def _config(budget: int = 288) -> RunConfig:
    return RunConfig(
        run_id="neutrality",
        n_budget=budget,
        hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused",
        proposal_count=4,
        allowed_families=tuple(CodeFamily),
    )


def _assembled(prompt_context: dict | None = None, *, memory: ArchiveMemory | None = None) -> str:
    prompt = _build_funsearch_prompt(
        memory=memory or ArchiveMemory(),
        config=_config(),
        gen=3,
        proposal_count=4,
        prompt_context=prompt_context or {},
    )
    return prompt["system_prompt"] + "\n" + prompt["user_prompt"]


def _rich_context() -> dict:
    """A context carrying every block the generator can be shown."""

    return {
        "strategy_brief": (
            "STRATEGY CONTEXT (structured allocation from the campaign policy):\n"
            "- family attention: bb: 0.50, hgp: 0.50"
        ),
        "next_parent_specs": [
            {
                "program_id": "struct_aa", "family": "bb", "n": 288, "n_checks": 288,
                "n_budget": 288, "k": 20, "d_search": 4, "Q": 1.1, "q_eff": 1.1,
                "rng_seed": 0, "selection_channels": ["structural"],
                "spec": {"family": "bb"}, "construct_source": "def construct(rng):\n    return {}",
            },
            {
                "program_id": "rt_bb", "family": "hgp", "n": 144, "n_checks": 144,
                "n_budget": 288, "k": 4, "d_search": 12, "rng_seed": 0,
                "selection_channels": ["runtime"], "slow_per_logical_per_round": 1.5e-3,
                "spec": {"family": "hgp"}, "construct_source": "def construct(rng):\n    return {}",
            },
        ],
        "slow_feedback": {
            "best_results": [
                {"program_id": "rt_bb", "n": 144, "k": 4, "d_search": 12,
                 "family": "hgp", "decoder": "bposd", "p": 0.005,
                 "per_logical_per_round": 1.5e-3, "degraded_estimate": True},
            ],
            "proposal_guidance": ["1 candidate(s) completed circuit-level measurement."],
        },
    }


# ---------------------------------------------------------------------------
# the assembled generator prompt
# ---------------------------------------------------------------------------
def test_assembled_prompt_contains_no_retired_ranking_instruction() -> None:
    text = _assembled(_rich_context())
    for phrase in BANNED_PHRASES:
        assert phrase not in text, phrase


def test_assembled_prompt_states_no_k_or_d_goal() -> None:
    text = _assembled(_rich_context())
    match = GOAL_RE.search(text)
    assert match is None, f"prompt states an explicit goal: {match.group(0)!r}"


def test_structural_and_runtime_numbers_appear_as_facts() -> None:
    """Hiding the data is not the fix; removing the global command is.

    The generator still sees what exists and how it ran — it just is not told
    to sort by any of it."""

    text = _assembled(_rich_context())
    assert "k=20 d_search=4" in text            # structural parent, no score
    assert "per_logical_per_round=0.0015" in text  # runtime fact
    assert "HOOK-DEGRADED" in text              # label kept...
    assert "do not clone" not in text           # ...its imperative removed
    assert "retention rules" in text            # who actually decides


def test_archive_block_is_context_not_a_leaderboard() -> None:
    text = _assembled(_rich_context())
    assert "RETAINED CONSTRUCTION CONTEXT" in text
    assert "CURRENT ARCHIVE" not in text
    assert "Q_eff" not in text


def test_trust_screen_advice_no_longer_pushes_the_model_out_of_large_n() -> None:
    """The old advice on a trust-filter record was "increase the distance or
    reduce n" — coaching away from exactly the region where high-k lives."""

    from omniqec_agentic.funsearch import _failure_actions_for_signature

    actions = " ".join(_failure_actions_for_signature("trust_filter: d_search/sqrt(n)=0.236<0.25"))
    assert "reduce n" not in actions
    assert "recorded" in actions and "retention rules" in actions


def test_feedback_mode_none_leaks_no_parent_or_archive() -> None:
    text = _funsearch_user_prompt(
        memory=ArchiveMemory(),
        config=RunConfig(
            run_id="ablation", n_budget=288,
            hardware=HardwareConfig(target=HardwareTarget.ION), out_dir="unused",
            proposal_count=4, allowed_families=tuple(CodeFamily), feedback_mode="none",
        ),
        gen=3, proposal_count=4, n_recent=4, prompt_context=_rich_context(),
    )
    assert "SELECTED PARENT CONSTRUCTIONS" not in text
    assert "RETAINED CONSTRUCTION CONTEXT" not in text
    assert "def construct" not in text


# ---------------------------------------------------------------------------
# the three back-feed channels (§5.2)
# ---------------------------------------------------------------------------
def test_injected_retired_rules_cannot_re_enter_through_the_strategy_brief() -> None:
    """A policy that still believed the old rules could write them into
    `directives`, which used to be injected verbatim at the head of the
    generator prompt — outranking every neutral block below it."""

    from omniqec_agentic.workflow.graph import _strategy_brief

    plan = {
        "family_mix": {"bb": 1.0},
        "strategy_mix": {"exploit": 1.0},
        "directives": "PLPR decides; prefer higher Q_eff; degraded: do not clone",
    }
    brief = _strategy_brief(plan, {"run_dir": "/nonexistent"})
    for phrase in ("PLPR decides", "do not clone", "higher Q_eff"):
        assert phrase not in brief
    assert "family attention" in brief and "strategy emphasis" in brief

    # ...and nothing downstream re-adds it.
    text = _assembled({**_rich_context(), "strategy_brief": brief})
    for phrase in BANNED_PHRASES:
        assert phrase not in text, phrase


def test_reflection_advice_and_raw_notes_do_not_reach_the_generator(tmp_path) -> None:
    from omniqec_agentic.orchestrator.reflector import append_reflection
    from omniqec_agentic.workflow.graph import _latest_reflection_advice, _strategy_brief

    manual_dir = tmp_path / "manual"
    append_reflection(manual_dir, {
        "turn": 0, "next_turn_advice": "keep preferring lower plpr; do not clone degraded",
    })
    # Still available to the POLICY (S-10)...
    assert "do not clone" in _latest_reflection_advice(tmp_path)
    # ...but absent from the generator's brief.
    brief = _strategy_brief({"family_mix": {"bb": 1.0}}, {"run_dir": str(tmp_path)})
    assert "do not clone" not in brief


def test_raw_generation_notes_are_archived_but_never_replayed() -> None:
    """A turn's own hand-off notes are unstructured text written while the
    retired rules may still have been in force; replaying them verbatim is a
    channel through which they re-enter the prompt."""

    text = _assembled({
        **_rich_context(),
        "recent_generation_notes": [{"turn": 1, "notes": ["Q_eff higher is better"]}],
    })
    assert "NOTES YOU LEFT" not in text
    assert "higher is better" not in text


# ---------------------------------------------------------------------------
# the policy-side documents that feed S-1 / the system prompt
# ---------------------------------------------------------------------------
def test_policy_manual_describes_two_ledgers_and_one_hard_exclusion() -> None:
    text = _flat(manual_mod.RANKING_SEMANTICS + manual_mod.FEEDBACK_VOCABULARY)
    assert "Structural retention" in text and "Runtime measurement" in text
    assert "never parent-eligible" not in text
    assert "PRIMARY" not in text
    assert "diagnostic label" in text.lower()
    assert "schedule_catastrophic" in text


def test_orchestrator_system_prompt_scopes_promotions_and_quarantine() -> None:
    text = _flat(load_prompt("orchestrator_system")[0])
    assert "at most FOUR named candidates" in text
    assert "CONSUMES" in text
    assert "not yours to lift" not in text
    assert "S-10" in text and "never be the only support" in text


def test_reflector_prompt_forbids_restating_a_lineage_rule() -> None:
    text = _flat(load_prompt("reflector")[0])
    assert "STRUCTURAL RETENTION" in text and "RUNTIME MEASUREMENT" in text
    assert "it never reaches the generator" in text
    assert "Do not restate either record as a lineage rule" in text


def test_acquisition_tool_scopes_itself_to_the_runtime_slot() -> None:
    from omniqec_agentic.tools.acquisition_score import ACQUISITION_VERSION, acquisition_score

    assert ACQUISITION_VERSION == "acquisition_v5_runtime_slot_expected_plpr"
    doc = _flat(acquisition_score.__doc__ or "")
    assert "FOUR runtime slow slots" in doc
    assert "no authority over the two structural endpoint slots" in doc


def test_quotas_are_frozen_together() -> None:
    structural = StructuralSearchConfig()
    assert structural.structural_slow_slots + structural.runtime_slow_slots == 6
    assert (
        structural.structural_parents + structural.runtime_parents + structural.diversity_parents
        == structural.parent_cap == 16
    )
    with pytest.raises(ValueError):
        StructuralSearchConfig(structural_slow_slots=4, runtime_slow_slots=1)
    with pytest.raises(ValueError):
        StructuralSearchConfig(structural_parents=9)


# ---------------------------------------------------------------------------
# runtime facts carry no implicit plpr order (audit fix #6)
# ---------------------------------------------------------------------------
def _measured_row(code: str, program_id: str, plpr: float) -> dict:
    return {
        "program_id": program_id,
        "code_hash": code * 16,
        "status": "ok",
        "per_logical_per_round": plpr,
        "n": 144,
        "k": 8,
        "d_search": 6,
        "family": "bb",
        "decoder": "bposd",
    }


def test_prompt_facing_facts_are_in_stable_hash_order_not_plpr_order() -> None:
    """rank_slow_rows orders the runtime ledger by plpr — its own channel's
    business. But a plpr-sorted list handed to the generator is an implicit
    'best first' instruction with no verb attached, so the FACTS list is
    re-keyed by code hash at the build site and again at the render site."""

    from omniqec_agentic.evaluation.feedback import build_slow_feedback
    from omniqec_agentic.funsearch import _slow_feedback_prompt_lines

    # plpr order: cc (best), aa, bb — hash order: aa, bb, cc.
    rankings = [
        _measured_row("cc", "p_best", 1e-6),
        _measured_row("aa", "p_mid", 1e-5),
        _measured_row("bb", "p_worst", 1e-4),
    ]
    feedback = build_slow_feedback(
        queue_path="queue.jsonl",
        hardware="ion",
        rankings=rankings,
        candidate_payloads=[],
    )
    facts = feedback["best_results"]
    assert [row["code_hash"][:2] for row in facts] == ["aa", "bb", "cc"]
    assert "fact_order" in feedback

    # Render defense: even a legacy artifact still carrying plpr order is
    # re-sorted before the generator sees it.
    legacy = {"best_results": rankings}
    lines = _slow_feedback_prompt_lines(legacy, {})
    rendered = "\n".join(lines[1:])
    assert rendered.index("p_mid") < rendered.index("p_worst") < rendered.index("p_best")


def test_prompt_facing_facts_exclude_unmeasured_rows_and_keep_hash_order() -> None:
    """v3.4.1: a decode_infeasible row (status != ok, plpr None) is a terminal
    runtime verdict, not a measurement. Left in best_results it reached the
    generator prompt as an unexplained per_logical_per_round=unknown line.
    Excluding it must not disturb the stable hash order of the ok rows."""

    from omniqec_agentic.evaluation.feedback import build_slow_feedback
    from omniqec_agentic.funsearch import _slow_feedback_prompt_lines

    infeasible = dict(
        _measured_row("bb", "p_infeasible", None),
        status="decode_infeasible",
        decode_infeasible=True,
        failure_reason="decode exceeded the frozen per-shot budget: X:16/16 shards (timeout_s=10000.0)",
    )
    # plpr order: cc (best), aa, dd — hash order: aa, cc, dd; bb is the
    # infeasible row and would sort between them if it were kept.
    rankings = [
        _measured_row("cc", "p_best", 1e-6),
        infeasible,
        _measured_row("aa", "p_mid", 1e-5),
        _measured_row("dd", "p_worst", 1e-4),
    ]
    feedback = build_slow_feedback(
        queue_path="queue.jsonl",
        hardware="ion",
        rankings=rankings,
        candidate_payloads=[],
    )
    facts = feedback["best_results"]
    assert [row["program_id"] for row in facts] == ["p_mid", "p_best", "p_worst"]
    assert [row["code_hash"][:2] for row in facts] == ["aa", "cc", "dd"]
    assert all(row["status"] == "ok" for row in facts)
    assert feedback["fact_order"].startswith("stable_code_hash")
    # The excluded row is still one of this turn's results...
    assert feedback["result_count"] == 4
    # ...but the rendered FACTS block never mentions it.
    rendered = "\n".join(_slow_feedback_prompt_lines(feedback, {}))
    assert "p_infeasible" not in rendered
    assert "per_logical_per_round=unknown" not in rendered
