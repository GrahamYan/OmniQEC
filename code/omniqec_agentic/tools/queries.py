"""Read-scope tools (03 §4.2): manual, archive/evidence queries, fingerprint
and known-code lookups, bandit/budget status."""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.memory.fingerprints import (
    NOVELTY_EPSILON,
    novelty_distance_to_set,
    tier1_fingerprint,
)
from omniqec_agentic.memory.known_codes import known_code_lookup as _known_lookup
from omniqec_agentic.orchestrator.registry import tool

from .context import ToolContext


@tool("read", example='manual_read(section="static/INDEX.md")')
def manual_read(ctx: ToolContext, section: str = "static/INDEX.md") -> dict[str, Any]:
    """Read one manual section. Sections live under the campaign manual
    directory: static/ (tool catalog, the two-ledger semantics, feedback
    vocabulary), dynamic/ (supported insights, bandit snapshot, latest
    digest)."""

    path = (ctx.manual_dir / section).resolve()
    if ctx.manual_dir.resolve() not in path.parents and path != ctx.manual_dir.resolve():
        return {"error": "section escapes the manual directory"}
    if not path.exists():
        available = sorted(
            str(p.relative_to(ctx.manual_dir)) for p in ctx.manual_dir.rglob("*") if p.is_file()
        )
        return {"error": f"no such section '{section}'", "available": available}
    return {"section": section, "content": path.read_text(encoding="utf-8")}


@tool("read", example='query_archive(family="lp_nonabelian", limit=8)')
def query_archive(
    ctx: ToolContext,
    family: str | None = None,
    max_n: int | None = None,
    min_d_proxy: int | None = None,
    max_check_weight: int | None = None,
    promoted_only: bool = False,
    limit: int = 16,
) -> dict[str, Any]:
    """Structured query over the fast archive queue. Rows carry code_hash /
    family / n,k,d_search / Q / max_check_weight, promotion flags and the
    construction `spec` (group name, base-matrix shapes) so a query can compare
    structures and not just scores; use query_evidence for one candidate's full
    chain. `max_check_weight` filters to codes at or below a check weight.

    This is the RUNTIME/MAP archive, ordered by the runtime acquisition key. It
    is not the structural record: retention across turns is decided by the
    (k, d_search, -max_check_weight) proxy-Pareto elites in S-3, which this
    queue does not see."""

    rows = ctx.archive_rows
    out = []
    for row in rows:
        if family and str(row.get("family")) != family:
            continue
        if max_n is not None and int(row.get("n") or 0) > int(max_n):
            continue
        if min_d_proxy is not None and int(row.get("d_proxy") or 0) < int(min_d_proxy):
            continue
        # v3.7.1: the weight was already on every archive row and simply not
        # queryable, so a weight question had to be asked one code at a time.
        if max_check_weight is not None and row.get("max_check_weight") is not None:
            if int(row.get("max_check_weight")) > int(max_check_weight):
                continue
        if promoted_only and not row.get("promotion_rank"):
            continue
        out.append(
            {
                key: row.get(key)
                for key in (
                    "program_id", "code_hash", "family", "n", "k", "d_proxy", "Q",
                    "max_check_weight", "max_qubit_degree",
                    "promotion_rank", "archive_cell", "spec",
                )
            }
        )
        if len(out) >= max(1, int(limit)):
            break
    return {"count": len(out), "rows": out, "archive_total": len(rows)}


@tool("read", example='query_evidence(code_hash="a41f00b2")')
def query_evidence(ctx: ToolContext, code_hash: str) -> dict[str, Any]:
    """Full evidence chain for one candidate (code_hash prefix >= 8 chars):
    slow rankings row(s) + distance evidence + racing metadata."""

    if len(str(code_hash)) < 8:
        return {"error": "code_hash prefix must be >= 8 chars"}
    rows = [
        row for row in ctx.slow_rows if str(row.get("code_hash") or "").startswith(str(code_hash))
    ]
    payload = ctx.candidate_payload(code_hash)
    result: dict[str, Any] = {
        "code_hash": code_hash,
        "slow_rows": [
            {
                key: row.get(key)
                for key in (
                    "program_id", "decoder", "noise_value", "per_logical_per_round",
                    "per_logical_block_ci", "rounds", "status", "stopped_by",
                    "d_circ_ext", "degraded_ext", "degraded_isd",
                    "verified_circuit_distance", "total_shots", "total_failures",
                )
            }
            for row in rows
        ],
    }
    if payload:
        result["distance_evidence"] = payload.get("distance_evidence")
        result["racing"] = payload.get("racing")
        result["fast"] = payload.get("fast")
    if not rows and not payload:
        result["error"] = "no candidate matches this prefix in the current run"
    return result


@tool("read", example='fingerprint_lookup(code_hash="a41f00b2")')
def fingerprint_lookup(ctx: ToolContext, code_hash: str) -> dict[str, Any]:
    """Novelty distance of a candidate against measured candidates and the
    known-code table. Distance 0 = WL-indistinguishable; < epsilon is gated
    from the slow loop as a near-duplicate."""

    payload = ctx.candidate_payload(code_hash)
    matrices = (payload or {}).get("matrices") or {}
    if not matrices.get("hx"):
        return {"error": "candidate matrices unavailable (need a measured candidate)"}
    hx = np.asarray(matrices["hx"], dtype=np.uint8)
    hz = np.asarray(matrices["hz"], dtype=np.uint8)
    fingerprint = tier1_fingerprint(hx, hz)
    seen = []
    for row in ctx.slow_rows:
        other = ctx.candidate_payload(str(row.get("code_hash") or "")[:12])
        other_m = (other or {}).get("matrices") or {}
        if other_m.get("hx") and str(row.get("code_hash")) != code_hash:
            seen.append(
                tier1_fingerprint(
                    np.asarray(other_m["hx"], dtype=np.uint8),
                    np.asarray(other_m["hz"], dtype=np.uint8),
                )
            )
    distance, _ = novelty_distance_to_set(fingerprint, seen)
    known = _known_lookup(hx, hz)
    return {
        "wl_hash": fingerprint["wl_hash"],
        "novelty_distance_to_measured": distance,
        "near_duplicate_gate": bool(distance < NOVELTY_EPSILON),
        "known_code_match": known,
    }


@tool("read", example='known_code_lookup(code_hash="a41f00b2")')
def known_code_lookup(ctx: ToolContext, code_hash: str) -> dict[str, Any]:
    """Equivalence-class lookup against the literature known-code table
    (docs/known_codes_v1.json). A match means rediscovery, not novelty."""

    payload = ctx.candidate_payload(code_hash)
    matrices = (payload or {}).get("matrices") or {}
    if not matrices.get("hx"):
        return {"error": "candidate matrices unavailable"}
    match = _known_lookup(
        np.asarray(matrices["hx"], dtype=np.uint8),
        np.asarray(matrices["hz"], dtype=np.uint8),
    )
    return {"match": match}


@tool("read", example="bandit_status()")
def bandit_status(ctx: ToolContext) -> dict[str, Any]:
    """Strategy-arm suggestion and statistics (UCB1, 03 §6.4). The mix is an
    informed signal: deviating from it requires a cited rationale in your
    plan; every arm keeps the epsilon_arm floor."""

    return ctx.bandit.suggestion()


@tool("read", example="budget_status()")
def budget_status(ctx: ToolContext) -> dict[str, Any]:
    """The four ledgers (tokens / slow shots / wall clock / turns): used so
    far and configured caps. Plans exceeding remaining budget are rejected."""

    protocol = ctx.state.get("protocol") or {}
    budget = ctx.state.get("budget") or {}
    return {
        "used": budget,
        "caps": {
            "max_turns": protocol.get("max_turns"),
            "tool_budget_per_turn": protocol.get("tool_budget"),
        },
        "turn": ctx.turn,
    }


@tool("read", example="query_coverage()")
def query_coverage(ctx: ToolContext, limit: int = 24) -> dict[str, Any]:
    """Structural coverage of the search so far: per region (family / group
    class / group-order band) how many candidates were proposed, how many
    carry a measurement, and how many racing stopped at pilot depth with their
    performance still unresolved. Also lists lp_nonabelian catalog branches
    that have never been proposed. Use it to find unexplored regions — a
    ranked list can only show what was already tried."""

    from omniqec_agentic.memory.coverage import catalog_gaps, coverage_report

    report = coverage_report(ctx.archive_rows, ctx.slow_rows, limit=int(limit))
    report["lp_nonabelian_catalog"] = catalog_gaps(ctx.archive_rows)
    return report
