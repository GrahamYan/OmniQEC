"""Deterministic fallback policy (03 §3.4) — the engineering degradation path.

Fixed program counts, Q_eff-ranked promotion (funsearch's promotion queue is
already Q-ordered), every-turn slow stage. Used for dry-run, CI, and no-API-key
pipeline validation only: it is NOT a science mode, never appears in the paper
narrative, and is not an ablation arm (03 D7). The agentic Orchestrator
(tool loop, 03 §3.2) replaces it in Milestone B behind the same Protocol."""

from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    """One campaign turn: read the state summary, emit a TurnPlan dict."""

    def plan(self, state: dict[str, Any]) -> dict[str, Any]: ...


class FallbackPolicy:
    def __init__(
        self,
        *,
        provider: str = "dry_run",
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
            "policy": "fallback_deterministic_v1",
            "turn": turn,
            "provider": self.provider,
            "generations": self.generations,
            "proposal_count": int(protocol.get("proposal_count") or 8),
            "rng_seeds_per_program": self.rng_seeds_per_program,
            "families": list((state.get("spec") or {}).get("families") or []),
            "budget_class": "pilot",
            "promote_top": int(protocol.get("promote_top") or 8),
            "feedback_mode": str(protocol.get("feedback_mode") or "both"),
            "finalize": False,
            "rationale": (
                "FallbackPolicy: fixed proposal count, Q_eff-ranked promotion, "
                "every-turn slow stage (engineering degradation path, 03 §3.4)."
            ),
        }
