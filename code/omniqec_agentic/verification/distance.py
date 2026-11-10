"""Exact/certified distance backends for slow verification evidence."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse

from omniqec_agentic.config import ExactDistanceConfig
from omniqec_agentic.verification.dem import dem_to_matrices

from .gf2 import kernel_basis, logicals
from .proxy_distance import _proxy_side


def certify_css_code_distance(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    d_proxy: int | None,
    config: ExactDistanceConfig,
) -> dict[str, Any]:
    """Certify CSS code distance with a bounded MILP parity formulation."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n = int(hx.shape[1])
    result: dict[str, Any] = {
        "schema": "exact_code_distance_evidence",
        "backend": "scipy_milp_parity",
        "status": "not_available",
        "n": n,
        "d_proxy": d_proxy,
        "d_exact": None,
        "certified": False,
        "sides": {},
    }
    if not config.enabled or not config.code_milp:
        result["status"] = "disabled"
        return result
    if n > int(config.max_n):
        result["status"] = "skipped_n_too_large"
        result["failure_reason"] = f"n={n} > max_n={int(config.max_n)}"
        return result
    if d_proxy is not None and int(d_proxy) > int(config.max_d_proxy):
        result["status"] = "skipped_d_proxy_too_large"
        result["failure_reason"] = f"d_proxy={int(d_proxy)} > max_d_proxy={int(config.max_d_proxy)}"
        return result

    start = time.perf_counter()
    try:
        lx, lz = logicals(hx, hz)
        z_side = _milp_side_distance(
            checks=hx,
            logical_parity=lx,
            side="Z",
            timeout_s=float(config.code_timeout_s),
        )
        elapsed = time.perf_counter() - start
        remaining = max(0.25, float(config.code_timeout_s) - elapsed)
        x_side = _milp_side_distance(
            checks=hz,
            logical_parity=lz,
            side="X",
            timeout_s=remaining,
        )
    except Exception as exc:
        result["status"] = "backend_error"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["runtime_s"] = time.perf_counter() - start
        return result

    result["sides"] = {"X": x_side, "Z": z_side}
    ok_sides = [side for side in (x_side, z_side) if side.get("status") == "ok" and side.get("distance") is not None]
    if len(ok_sides) == 2:
        result["status"] = "ok"
        result["d_exact"] = min(int(side["distance"]) for side in ok_sides)
        result["certified"] = True
    elif ok_sides:
        result["status"] = "partial"
        result["d_exact"] = min(int(side["distance"]) for side in ok_sides)
        result["certified"] = False
    else:
        result["status"] = "no_solution"
        result["certified"] = False
    result["runtime_s"] = time.perf_counter() - start
    return result


def exact_circuit_distance_from_pair(
    pair: Any,
    *,
    config: ExactDistanceConfig,
) -> dict[str, Any]:
    """Find exact circuit distance for the built X/Z memory circuits via Stim search."""

    result: dict[str, Any] = {
        "schema": "exact_circuit_distance_evidence",
        "backend": "stim_search_for_undetectable_logical_errors",
        "status": "not_available",
        "d_circ_exact": None,
        "certified": False,
        "cap": int(config.circuit_cap),
        "sides": {},
    }
    if not config.enabled or not config.circuit_stim:
        result["status"] = "disabled"
        return result
    start = time.perf_counter()
    sides: dict[str, dict[str, Any]] = {}
    for basis, experiment in (("X_memory", pair.x_memory), ("Z_memory", pair.z_memory)):
        sides[basis] = _stim_circuit_distance(experiment.circuit, cap=int(config.circuit_cap))
    result["sides"] = sides
    ok_sides = [side for side in sides.values() if side.get("status") == "ok" and side.get("distance") is not None]
    if len(ok_sides) == 2:
        result["status"] = "ok"
        result["d_circ_exact"] = min(int(side["distance"]) for side in ok_sides)
        result["certified"] = True
    elif ok_sides:
        result["status"] = "partial"
        result["d_circ_exact"] = min(int(side["distance"]) for side in ok_sides)
        result["certified"] = False
    else:
        result["status"] = "no_solution"
        result["certified"] = False
    result["runtime_s"] = time.perf_counter() - start
    return result


def isd_circuit_distance_from_pair(
    pair: Any,
    *,
    config: ExactDistanceConfig,
) -> dict[str, Any]:
    """Estimate a circuit-distance upper bound by ISD on each memory DEM."""

    result: dict[str, Any] = {
        "schema": "isd_dem_circuit_distance_evidence",
        "backend": "dem_kernel_isd_upper_bound",
        "status": "not_available",
        "d_circ_isd_upper_bound": None,
        "certified": False,
        "sides": {},
    }
    if not config.enabled or not config.circuit_dem_isd:
        result["status"] = "disabled"
        return result
    start = time.perf_counter()
    sides: dict[str, dict[str, Any]] = {}
    for basis, experiment in (("X_memory", pair.x_memory), ("Z_memory", pair.z_memory)):
        sides[basis] = _isd_dem_distance_upper_bound(
            experiment.detector_error_model,
            max_mechanisms=int(config.dem_isd_max_mechanisms),
            iters=int(config.dem_isd_iters),
            seed=int(config.dem_isd_seed) + len(sides),
        )
    result["sides"] = sides
    ok_sides = [
        side
        for side in sides.values()
        if side.get("status") == "ok" and side.get("distance_upper_bound") is not None
    ]
    if len(ok_sides) == 2:
        result["status"] = "ok"
        result["d_circ_isd_upper_bound"] = min(int(side["distance_upper_bound"]) for side in ok_sides)
    elif ok_sides:
        result["status"] = "partial"
        result["d_circ_isd_upper_bound"] = min(int(side["distance_upper_bound"]) for side in ok_sides)
    else:
        statuses = {str(side.get("status")) for side in sides.values()}
        if len(statuses) == 1:
            result["status"] = statuses.pop()
        else:
            result["status"] = "no_solution"
    result["runtime_s"] = time.perf_counter() - start
    return result


def _milp_side_distance(
    *,
    checks: np.ndarray,
    logical_parity: np.ndarray,
    side: str,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:
        return {
            "side": side,
            "status": "dependency_missing",
            "failure_reason": f"scipy.optimize.milp unavailable: {exc}",
            "distance": None,
        }

    checks = np.asarray(checks, dtype=np.uint8) % 2
    logical_parity = np.asarray(logical_parity, dtype=np.uint8) % 2
    m, n = checks.shape
    k = int(logical_parity.shape[0])
    if k <= 0:
        return {"side": side, "status": "no_logicals", "distance": None}

    check_slack = m
    logical_slack = k
    x_offset = 0
    y_offset = n
    check_slack_offset = y_offset + k
    logical_slack_offset = check_slack_offset + check_slack
    variables = logical_slack_offset + logical_slack

    rows: list[scipy.sparse.csr_matrix] = []
    lb: list[float] = []
    ub: list[float] = []

    if m:
        block = scipy.sparse.hstack(
            [
                scipy.sparse.csr_matrix(checks.astype(float)),
                scipy.sparse.csr_matrix((m, k)),
                -2.0 * scipy.sparse.identity(check_slack, format="csr"),
                scipy.sparse.csr_matrix((m, logical_slack)),
            ],
            format="csr",
        )
        rows.append(block)
        lb.extend([0.0] * m)
        ub.extend([0.0] * m)

    logical_block = scipy.sparse.hstack(
        [
            scipy.sparse.csr_matrix(logical_parity.astype(float)),
            -1.0 * scipy.sparse.identity(k, format="csr"),
            scipy.sparse.csr_matrix((k, check_slack)),
            -2.0 * scipy.sparse.identity(logical_slack, format="csr"),
        ],
        format="csr",
    )
    rows.append(logical_block)
    lb.extend([0.0] * k)
    ub.extend([0.0] * k)

    nonzero = scipy.sparse.lil_matrix((1, variables), dtype=float)
    nonzero[0, y_offset : y_offset + k] = 1.0
    rows.append(nonzero.tocsr())
    lb.append(1.0)
    ub.append(float(k))

    constraints = LinearConstraint(scipy.sparse.vstack(rows, format="csr"), np.asarray(lb), np.asarray(ub))
    c = np.zeros(variables, dtype=float)
    c[x_offset : x_offset + n] = 1.0
    lower = np.zeros(variables, dtype=float)
    upper = np.full(variables, np.inf, dtype=float)
    upper[x_offset : x_offset + n] = 1.0
    upper[y_offset : y_offset + k] = 1.0
    upper[check_slack_offset : check_slack_offset + check_slack] = np.asarray(checks.sum(axis=1)).reshape(-1) // 2
    upper[logical_slack_offset : logical_slack_offset + logical_slack] = (
        np.asarray(logical_parity.sum(axis=1)).reshape(-1) // 2
    )
    integrality = np.ones(variables, dtype=int)

    start = time.perf_counter()
    try:
        res = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"time_limit": max(0.25, float(timeout_s)), "mip_rel_gap": 0.0},
        )
    except Exception as exc:
        return {
            "side": side,
            "status": "solver_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "distance": None,
            "runtime_s": time.perf_counter() - start,
        }
    payload: dict[str, Any] = {
        "side": side,
        "solver_status": int(getattr(res, "status", -1)),
        "solver_message": str(getattr(res, "message", "")),
        "runtime_s": time.perf_counter() - start,
        "distance": None,
    }
    if bool(getattr(res, "success", False)) and getattr(res, "fun", None) is not None:
        payload["status"] = "ok"
        payload["distance"] = int(round(float(res.fun)))
    else:
        payload["status"] = "not_certified"
    return payload


def _isd_dem_distance_upper_bound(
    dem: Any,
    *,
    max_mechanisms: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        matrices = dem_to_matrices(dem)
    except Exception as exc:
        return {
            "status": "dem_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "distance_upper_bound": None,
            "runtime_s": time.perf_counter() - start,
        }
    mechanisms = int(matrices.H.shape[1])
    payload: dict[str, Any] = {
        "status": "started",
        "distance_upper_bound": None,
        "num_detectors": int(matrices.H.shape[0]),
        "num_observables": int(matrices.L.shape[0]),
        "num_error_mechanisms": mechanisms,
        "iters": int(iters),
        "seed": int(seed),
    }
    if mechanisms <= 0 or matrices.L.shape[0] <= 0:
        payload["status"] = "no_logicals"
        payload["runtime_s"] = time.perf_counter() - start
        return payload
    if mechanisms > int(max_mechanisms):
        payload["status"] = "skipped_dem_too_large"
        payload["failure_reason"] = f"num_error_mechanisms={mechanisms} > max_mechanisms={int(max_mechanisms)}"
        payload["runtime_s"] = time.perf_counter() - start
        return payload
    try:
        h_dense = np.asarray(matrices.H.toarray(), dtype=np.uint8) % 2
        l_dense = np.asarray(matrices.L.toarray(), dtype=np.uint8) % 2
        kernel = kernel_basis(h_dense)
        if kernel.shape[0] == 0:
            payload["status"] = "no_kernel"
            payload["runtime_s"] = time.perf_counter() - start
            return payload
        rng = np.random.default_rng(int(seed))
        distance = _proxy_side(kernel, l_dense, int(iters), rng)
    except Exception as exc:
        payload["status"] = "backend_error"
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
        payload["runtime_s"] = time.perf_counter() - start
        return payload
    if int(distance) >= 10**9:
        payload["status"] = "no_solution"
    else:
        payload["status"] = "ok"
        payload["distance_upper_bound"] = int(distance)
    payload["kernel_dimension"] = int(kernel.shape[0])
    payload["runtime_s"] = time.perf_counter() - start
    return payload


def _stim_circuit_distance(circuit: Any, *, cap: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        errors = circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=int(cap),
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    except Exception as exc:
        return {
            "status": "backend_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "distance": None,
            "runtime_s": time.perf_counter() - start,
        }
    return {
        "status": "ok",
        "distance": int(len(errors)),
        "runtime_s": time.perf_counter() - start,
    }
