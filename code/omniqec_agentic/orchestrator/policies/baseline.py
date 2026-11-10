"""Turn plan of the search baselines without an LLM (2026-11-06; omniqec_agentic/baselines).

There is no planner: every turn has the frozen shape of the protocol -- the fixed sampling prior proposes
`proposal_count` draws in each of the protocol's generations, one rng seed each -- with NO family allocation, NO
promotions and NO directives. Which codes are measured is decided by the baseline's slow-queue rule
(baselines/selection.py), not by the plan. Its plan policy id is what the strict turn audit accepts for a run whose
snapshot carries `search_baseline`."""

from __future__ import annotations

from typing import Any

from omniqec_agentic.config import BASELINE_PLAN_POLICY_ID, BASELINE_PROVIDER


class BaselinePolicy:
    def __init__(self, *, generations: int = 1, rng_seeds_per_program: int = 1) -> None:
        self.provider = BASELINE_PROVIDER
        self.generations = int(generations)
        self.rng_seeds_per_program = int(rng_seeds_per_program)

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        protocol = state.get("protocol") or {}
        turn = int(state.get("turn") or 0)
        return {
            "schema": "turn_plan",
            "policy": BASELINE_PLAN_POLICY_ID,
            "turn": turn,
            "provider": self.provider,
            "generations": self.generations,
            "proposal_count": int(protocol.get("proposal_count") or 0),
            "rng_seeds_per_program": self.rng_seeds_per_program,
            "families": list((state.get("spec") or {}).get("families") or []),
            "budget_class": "pilot",
            # Frozen cap, never resized by a plan (v3.2 §4.3).
            "promote_top": int(protocol.get("promote_top") or 0),
            "promotions": [],
            "feedback_mode": str(protocol.get("feedback_mode") or "both"),
            "finalize": False,
            "rationale": (
                "BaselinePolicy (search baseline without an LLM): the fixed prior proposes the frozen batch; "
                "the baseline's own rule picks the slow queue."
            ),
        }
