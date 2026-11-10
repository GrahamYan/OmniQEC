"""Orchestrator layer (03 §3): policy agents that decide what to measure and
how much to spend. Zero framework dependencies (pydantic allowed) — only
workflow/ imports langgraph. AgenticPolicy is the LLM tool-loop Orchestrator
(03a); FallbackPolicy is the deterministic engineering-degradation path;
NoFeedbackPolicy is the S3 ablation arm without planner feedback."""

from .policies import NO_FEEDBACK_POLICY_ID, BaselinePolicy, FallbackPolicy, NoFeedbackPolicy, Policy


def make_policy(provider: str, ablation: str | None = None, **kwargs):
    """baseline_sampler -> BaselinePolicy; dry_run -> FallbackPolicy; ablation no_feedback -> NoFeedbackPolicy
    (no planner call, frozen turn shape); anything else -> the agentic tool
    loop (which itself degrades to FallbackPolicy on client/plan failure)."""

    from omniqec_agentic.config import BASELINE_PROVIDER

    if str(provider) == BASELINE_PROVIDER:
        # 2026-11-06 search baselines: no model anywhere, a frozen turn shape.
        return BaselinePolicy(
            generations=int(kwargs.get("generations") or 1),
            rng_seeds_per_program=int(kwargs.get("rng_seeds_per_program") or 1),
        )
    if str(ablation or "full") == "no_feedback":
        kwargs.pop("allow_fallback", None)
        kwargs.pop("fixed_program_count", None)
        return NoFeedbackPolicy(provider=provider, **kwargs)
    if str(provider) == "dry_run":
        kwargs.pop("allow_fallback", None)
        kwargs.pop("fixed_program_count", None)
        return FallbackPolicy(provider=provider, **kwargs)
    from .loop import AgenticPolicy

    return AgenticPolicy(provider=provider, **kwargs)


__all__ = ["NO_FEEDBACK_POLICY_ID", "BaselinePolicy", "FallbackPolicy", "NoFeedbackPolicy", "Policy", "make_policy"]
