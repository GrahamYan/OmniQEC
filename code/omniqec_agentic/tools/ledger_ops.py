"""Policy-scope tools (03 §4.2): ledger propose/retract. The deterministic
backtester adjudicates immediately-available evidence at reconcile; the agent
has proposal rights only — no tool can set supported/refuted."""

from __future__ import annotations

from typing import Any

from omniqec_agentic.orchestrator.registry import tool

from .context import ToolContext


@tool(
    "policy",
    example=(
        'ledger_propose(predicate=[{"field": "family", "op": "==", "value": "bb"}], '
        'claim={"metric": "per_logical_per_round", "direction": "lower"}, '
        'rationale="bb rows dominate the slow top-k this turn (S-4)")'
    ),
)
def ledger_propose(
    ctx: ToolContext,
    predicate: list[dict[str, Any]],
    claim: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    rationale: str = "",
) -> dict[str, Any]:
    """Propose an insight hypothesis. Predicate = conjunction of comparisons
    over computed columns only; claim defaults to lower per_logical_per_round.
    Rejected when the active-hypothesis cap is reached, the predicate matches
    a graveyarded entry, or the schema is invalid. Adjudication is prospective:
    only evidence arriving AFTER this proposal can support/refute it."""

    return ctx.ledger.propose(
        scope=scope,
        predicate=predicate,
        claim=claim,
        proposer="orchestrator",
        turn=int(ctx.turn),
        propose_gen=int(ctx.turn),
        rationale=rationale,
    )


@tool("policy", example='ledger_retract(entry_id="H-7")')
def ledger_retract(ctx: ToolContext, entry_id: str) -> dict[str, Any]:
    """Retract an active hypothesis or supported entry (moves it to retired;
    retired predicates cannot be re-proposed)."""

    return ctx.ledger.retract(entry_id, turn=int(ctx.turn))
