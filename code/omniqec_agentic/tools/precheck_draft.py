"""Draft-scope tool (03 §4.2): sandboxed pre-check of a construct program.

Runs the SAME pipeline a submitted program would face — sandbox execution,
structural checks, proxy scoring, fingerprint/novelty and known-code lookup —
so near-duplicates and budget violations never reach the generation batch."""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.orchestrator.registry import tool

from .context import ToolContext


@tool("draft", example='precheck_draft(program="def construct(rng): ...")')
def precheck_draft(ctx: ToolContext, program: str, rng_seed: int = 0) -> dict[str, Any]:
    """Pre-check one construct program draft: sandbox -> structural -> proxy
    -> fingerprint (novelty vs known codes). Returns machine-computed truths
    only; passing precheck does NOT guarantee acceptance (the evaluator is
    the source of truth)."""

    from omniqec_agentic.construction import build_matrices
    from omniqec_agentic.memory.fingerprints import (
        NOVELTY_EPSILON,
        tier1_fingerprint,
    )
    from omniqec_agentic.memory.known_codes import known_code_lookup
    from omniqec_agentic.sandbox import run_construct_source
    from omniqec_agentic.verification.fast import evaluate_proxy
    from omniqec_agentic.verification.structural import structural_check_spec

    try:
        constructed = run_construct_source(
            program, program_id="precheck", seed=int(rng_seed)
        ).constructed
        spec = constructed.spec
    except Exception as exc:
        return {"ok": False, "stage": "sandbox", "error": str(exc)}
    spec_budget = (ctx.state.get("spec") or {}).get("qubit_budget")
    structural = structural_check_spec("precheck", spec, require_positive_k=True)
    result: dict[str, Any] = {
        "ok": bool(structural.valid),
        "stage": "structural",
        "spec_family": spec.get("family"),
        "structural": structural.to_dict(),
    }
    if not structural.valid:
        return result
    fast = evaluate_proxy(spec, iters=60)
    result["fast"] = {k: fast.get(k) for k in ("n", "k", "d", "Q", "family")}
    n_data = int(structural.n_data or 0)
    total = n_data + int(structural.n_checks_x or 0) + int(structural.n_checks_z or 0)
    result["total_qubits"] = total
    if spec_budget is not None and total > int(spec_budget):
        result["ok"] = False
        result["stage"] = "budget"
        result["error"] = f"total qubits {total} exceed campaign budget {spec_budget}"
        return result
    try:
        hx, hz = build_matrices(spec)
        fingerprint = tier1_fingerprint(np.asarray(hx), np.asarray(hz))
        result["wl_hash"] = fingerprint["wl_hash"]
        known = known_code_lookup(np.asarray(hx), np.asarray(hz))
        result["known_code_match"] = known
        if known is not None:
            result["note"] = "rediscovery of a known code — no novelty credit"
        result["novelty_epsilon"] = NOVELTY_EPSILON
    except Exception as exc:
        result["fingerprint_error"] = str(exc)
    result["stage"] = "ok"
    return result
