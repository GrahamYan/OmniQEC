"""Fast proxy evaluator.

v3.2 splits three verdicts that used to be one boolean (repair plan §4.1):

  ``proxy_valid``           the build/shape/k/proxy pipeline completed and a
                            full n / k / d_search / Q record exists;
  ``trust_filter_passed``   the ver1 d/sqrt(n) screen — a RUNTIME/acquisition
                            screen and an audit field, nothing more;
  ``runtime_screen_passed`` proxy_valid AND d_search >= 3 AND trust filter.

``FastResult.valid`` maps to ``runtime_screen_passed`` for compatibility with
the MAP/runtime channel. Structural eligibility is decided one layer up, in
the archive, where ``structural.valid`` and the inheritance payload are also
in scope — a screen that fires here must never delete the record, because at
n_data=288 the trust filter rejects exactly the high-k / moderate-d endpoint
the structural channel exists to keep.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from omniqec_agentic.construction import build_compiled_code, compiled_code_hash
from omniqec_agentic.ranking import (
    DEFAULT_DSQRTN_FLOOR,
    MIN_PROMOTION_DISTANCE,
    trust_filter_ok,
)
from omniqec_agentic.schemas import FastResult

from .gf2 import rank2
from .proxy_distance import (
    DEFAULT_PROXY_SEEDS,
    PROXY_UNAVAILABLE,
    proxy_distance_search,
)


def evaluate_proxy(
    spec: dict[str, Any],
    *,
    iters: int = 120,
    seeds: tuple[int, ...] | list[int] = DEFAULT_PROXY_SEEDS,
    n_min: int | None = None,
    n_max: int | None = None,
    dsqrtn_floor: float = DEFAULT_DSQRTN_FLOOR,
    deformation_mode: str | None = None,
    lp_shape_pin: Any = None,
    algorithm_id: str | None = None,
) -> dict[str, Any]:
    """Build + `d_search` protocol. Returns a ver1-compatible metrics dict.

    ``error`` is reserved for HARD failures (the record cannot be completed).
    A finished proxy whose distance floor or trust screen fails keeps its full
    metrics and reports the verdict in the flags instead: those candidates are
    the structural channel's raw material and an ``error`` would erase them.
    """

    try:
        compiled = build_compiled_code(
            spec, deformation_mode=deformation_mode, lp_shape_pin=lp_shape_pin
        )
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
    except Exception as exc:
        return {"error": f"build:{exc}", "proxy_valid": False}
    n = int(hx.shape[1])
    if n_min is not None and n < int(n_min):
        return {"error": f"n={n} below configured minimum {int(n_min)}", "proxy_valid": False}
    if n_max is not None and n > int(n_max):
        return {"error": f"n={n} above configured maximum {int(n_max)}", "proxy_valid": False}
    k = n - rank2(hx) - rank2(hz)
    if k <= 0:
        return {"error": "k<=0", "proxy_valid": False}

    search = proxy_distance_search(
        hx, hz, seeds=seeds, iters=iters, algorithm_id=algorithm_id
    )
    if not search["available"]:
        return {
            "error": "d_search unavailable (no logical operators)",
            "proxy_valid": False,
            "proxy_search": search,
        }
    distance = int(search["d_search"])
    d_x, d_z = int(search["d_x"]), int(search["d_z"])
    w_x = int(hx.sum(1).max())
    w_z = int(hz.sum(1).max())
    w = max(w_x, w_z)
    q = int(max(hx.sum(0).max(), hz.sum(0).max()))
    m_x, m_z = int(hx.shape[0]), int(hz.shape[0])

    trust_passed = bool(trust_filter_ok(distance, n, floor=dsqrtn_floor))
    below_floor = bool(distance < MIN_PROMOTION_DISTANCE)
    screen_reasons: list[str] = []
    if below_floor:
        screen_reasons.append(f"d_search={distance}<{MIN_PROMOTION_DISTANCE}")
    if not trust_passed:
        screen_reasons.append(
            f"trust_filter: d_search/sqrt(n)={distance / float(n) ** 0.5:.3f}"
            f"<{float(dsqrtn_floor):g}"
        )

    metrics = {
        "n": n,
        "k": int(k),
        # `d_search` is the single source of truth for the search phase; `d`
        # and `d_proxy` are read-only aliases written only after all seeds and
        # both sectors completed, and are equal to it by construction.
        "d_search": distance,
        "d": distance,
        "d_proxy": distance,
        "d_x": d_x if compiled.is_css else None,
        "d_z": d_z if compiled.is_css else None,
        "w": w,
        "w_x": w_x if compiled.is_css else None,
        "w_z": w_z if compiled.is_css else None,
        "m_x": m_x,
        "m_z": m_z,
        "n_total": n + m_x + m_z,
        "q": q,
        # kd2_v1 (ranking.QUALITY_METRIC_ID): Q = k * d_search^2, no /n.
        "Q": k * distance * distance,
        "Q_search": k * distance * distance,
        "overhead": n / k,
        "family": compiled.family,
        "deformation": compiled.deformation,
        "code_hash": compiled_code_hash(hx, hz, h_mask=compiled.h_mask),
        "proxy_valid": True,
        "trust_filter_passed": trust_passed,
        "below_min_promotion_distance": below_floor,
        "runtime_screen_passed": bool(trust_passed and not below_floor),
        "screen_reasons": screen_reasons,
        "proxy_search": search,
    }
    if not compiled.is_css:
        metrics.update(
            ancestor_d_x=d_x,
            ancestor_d_z=d_z,
            ancestor_w_x=w_x,
            ancestor_w_z=w_z,
            basis_distance_semantics="css_ancestor_only",
        )
    return metrics


def evaluate_fast_result(
    program_id: str,
    spec: dict[str, Any],
    *,
    n_budget: int,
    iters: int = 120,
    seeds: tuple[int, ...] | list[int] = DEFAULT_PROXY_SEEDS,
    n_min: int | None = None,
    n_max: int | None = None,
    dsqrtn_floor: float = DEFAULT_DSQRTN_FLOOR,
    deformation_mode: str | None = None,
    lp_shape_pin: Any = None,
    algorithm_id: str | None = None,
) -> FastResult:
    metrics = evaluate_proxy(
        spec,
        iters=iters,
        seeds=seeds,
        n_min=n_min,
        n_max=n_max,
        dsqrtn_floor=dsqrtn_floor,
        deformation_mode=deformation_mode,
        lp_shape_pin=lp_shape_pin,
        algorithm_id=algorithm_id,
    )
    if "error" in metrics:
        return FastResult(
            program_id=program_id,
            valid=False,
            status="rejected",
            n_budget=int(n_budget),
            family=str(spec.get("family") or ""),
            failure_reasons=[str(metrics["error"])],
            metrics=metrics,
            proxy_valid=False,
            trust_filter_passed=False,
            runtime_screen_passed=False,
        )
    runtime_ok = bool(metrics["runtime_screen_passed"])
    return FastResult(
        program_id=program_id,
        # `valid` IS the runtime screen (MAP niche + acquisition queue). The
        # structural channel reads proxy_valid/d_search instead.
        valid=runtime_ok,
        status="ok" if runtime_ok else "runtime_screen_rejected",
        n_budget=int(n_budget),
        n_data=int(metrics["n"]),
        k=int(metrics["k"]),
        d_proxy=int(metrics["d_search"]),
        d_search=int(metrics["d_search"]),
        q_score=float(metrics["Q"]),
        family=str(metrics["family"]),
        max_check_weight=int(metrics["w"]),
        max_qubit_degree=int(metrics["q"]),
        code_hash=str(metrics["code_hash"]),
        failure_reasons=list(metrics["screen_reasons"]),
        metrics=metrics,
        proxy_valid=True,
        trust_filter_passed=bool(metrics["trust_filter_passed"]),
        runtime_screen_passed=runtime_ok,
        below_min_promotion_distance=bool(metrics["below_min_promotion_distance"]),
    )


__all__ = ["evaluate_proxy", "evaluate_fast_result", "PROXY_UNAVAILABLE"]
