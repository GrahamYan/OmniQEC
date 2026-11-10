"""S3 ablation policy for the no-feedback arm (appendix S3, 2026-10-23).

"OmniQEC w/o slow & fast": every turn the generator is called from the
task/constraint prompt alone. There is no planner call (nothing to plan from:
no archive, no measurement, no coverage), so this deterministic policy emits
the frozen turn shape -- the protocol's program count, generations and seeds
-- with NO family allocation, NO strategy mix, NO promotions and NO directives.
Its plan policy id is what the strict turn audit accepts under
ablation=no_feedback (config.ABLATION_PLAN_POLICIES)."""

from __future__ import annotations

from typing import Any

NO_FEEDBACK_POLICY_ID = "ablation_no_feedback_v1"


class NoFeedbackPolicy:
    def __init__(
        self,
        *,
        provider: str,
        generations: int = 1,
        rng_seeds_per_program: int = 1,
    ) -> None:
        self.provider = str(provider)
        self.generations = int(generations)
        self.rng_seeds_per_program = int(rng_seeds_per_program)

    def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        protocol = state.get("protocol") or {}
        turn = int(state.get("turn") or 0)
        return {
            "schema": "turn_plan",
            "policy": NO_FEEDBACK_POLICY_ID,
            "turn": turn,
            "provider": self.provider,
            "generations": self.generations,
            "proposal_count": int(protocol.get("proposal_count") or 8),
            "rng_seeds_per_program": self.rng_seeds_per_program,
            "families": list((state.get("spec") or {}).get("families") or []),
            "budget_class": "pilot",
            # Frozen cap, never resized by a plan (v3.2 §4.3); unused here:
            # the ablation runs no slow stage.
            "promote_top": int(protocol.get("promote_top") or 0),
            "promotions": [],
            "feedback_mode": "none",
            "finalize": False,
            "rationale": (
                "NoFeedbackPolicy (S3 ablation w/o slow & fast): the generator is "
                "called from the task/constraint prompt alone every turn; no "
                "planner, no family allocation, no promotions."
            ),
        }
