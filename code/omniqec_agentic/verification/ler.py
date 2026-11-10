"""Paper-defined logical-error-rate aggregation (SI C, Eqs. A4-A5).

A4: an X-memory failure and a Z-memory failure are independent logical
events, so the combined block error rate multiplies survivals:
    p_L = 1 - (1 - p_X)(1 - p_Z).
A5: the per-logical, per-round rate (the primary figure of merit):
    eps = 1 - (1 - p_L)^(1/(k*T)),
which recovers the plain per-round rate 1-(1-p_L)^(1/T) at k=1. Wilson
confidence intervals on p_X and p_Z are propagated through the same monotone
transforms, so ``per_logical_per_round_ci`` is directly comparable across
candidates measured at different round counts."""

from __future__ import annotations

from typing import Any

from omniqec_agentic import ranking

LER_DEFINITION = "paper_A4_A5_single_T"


def _a5(block_ler: float, *, k: int, rounds: int) -> float:
    value = float(block_ler)
    if not 0.0 <= value <= 1.0:
        raise ValueError("block_ler must be in [0, 1]")
    return 1.0 - (1.0 - value) ** (1.0 / (int(k) * int(rounds)))


def combine_basis_ler_results(
    basis: dict[str, dict[str, Any]],
    *,
    rounds: int,
    k: int,
    ci_z: float = ranking.DEFAULT_CI_Z,
) -> dict[str, Any]:
    """Combine X/Z block failures and apply the paper's A4/A5 definition.

    Each basis result is a block-level failure probability: a shot fails when
    any logical observable is decoded incorrectly.  The two memory-basis
    experiments are combined by multiplying their success probabilities (A4);
    the block rate then maps to the per-logical per-round rate via the
    compound root (A5)."""

    present: dict[str, tuple[dict[str, Any], float]] = {}
    for name, result in basis.items():
        value = result.get("logical_error_rate", result.get("ler"))
        if value is not None:
            present[name] = (result, float(value))
    if not present:
        return {
            "block_ler": None,
            "per_logical_per_round": None,
            "failure_reason": "missing_basis_result",
        }

    rounds = int(rounds)
    k = int(k)
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    survival = 1.0
    for _, value in present.values():
        if not 0.0 <= value <= 1.0:
            raise ValueError("basis logical error rates must be in [0, 1]")
        survival *= 1.0 - value
    block_ler = 1.0 - survival
    output: dict[str, Any] = {
        "block_ler": float(block_ler),
        "per_logical_per_round": float(_a5(block_ler, k=k, rounds=rounds)),
        "basis_logical_error_rate": {
            name: float(value) for name, (_, value) in present.items()
        },
        "ler_definition": LER_DEFINITION,
    }

    if all(int(result.get("shots") or 0) > 0 for result, _ in present.values()):
        block_survival_from_low = 1.0
        block_survival_from_high = 1.0
        basis_ci: dict[str, list[float]] = {}
        for name, (result, _) in present.items():
            interval = ranking.wilson_interval(
                int(result.get("failures", result.get("errors", 0)) or 0),
                int(result.get("shots") or 0),
                z=ci_z,
            )
            basis_ci[name] = [float(interval[0]), float(interval[1])]
            block_survival_from_low *= 1.0 - interval[0]
            block_survival_from_high *= 1.0 - interval[1]
        block_bounds = (
            1.0 - block_survival_from_low,
            1.0 - block_survival_from_high,
        )
        # Per-logical block bounds (the k-root step alone) are kept for
        # continuity of existing artifacts; the plpr bounds are the A5-scale
        # interval used by racing thresholds and statistical-tie grouping.
        per_logical_block_bounds = sorted(
            1.0 - (1.0 - bound) ** (1.0 / k) for bound in block_bounds
        )
        per_round_bounds = sorted(_a5(bound, k=k, rounds=rounds) for bound in block_bounds)
        output.update(
            per_logical_block_ci=[
                float(per_logical_block_bounds[0]),
                float(per_logical_block_bounds[1]),
            ],
            per_logical_per_round_ci=[
                float(per_round_bounds[0]),
                float(per_round_bounds[1]),
            ],
            basis_ler_ci=basis_ci,
            ci_z=float(ci_z),
            total_failures=sum(
                int(result.get("failures", result.get("errors", 0)) or 0)
                for result, _ in present.values()
            ),
            total_shots=sum(int(result.get("shots") or 0) for result, _ in present.values()),
        )
    return output
