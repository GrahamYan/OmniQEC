"""Distance/LER evidence merging: d_circ_ext, DEM-ISD, slope estimates,
degraded_* flags (03 §7, §11). Mechanically extracted from slow.py; adds the
d_circ_ext channel from the ranked SEC schedule (02 §5, §9.4) and
verified_circuit_distance = min(exact-stim, DEM-ISD, d_circ_ext)."""

from __future__ import annotations

import math
from typing import Any

from omniqec_agentic.config import VerificationConfig

from .common import SLOW_SCHEMA_VERSION, decoder_name, float_or_none, int_or_none

VER1_LER_SLOPE_OFFSET = 1.622
VER1_D_CIRC_MIN_ERRORS = 20
VER1_D_CIRC_MIN_POINTS = 2


def sec_ext_evidence(ranked: Any) -> dict[str, Any]:
    """d_circ_ext channel from the ranked SEC synthesis (SBB Thm 1:
    non-interleaved circuit distance = extended-code distance; kernel-ISD
    upper-bound probes). ``ranked`` is any RankedSchedule-compatible object
    (LRC ``rank_schedules`` or twobasis ``rank_twobasis_schedules``); the
    policy id is recorded because d_circ_ext values from different policies
    use different residual sets / probe budgets and are NOT comparable."""

    if ranked is None:
        return {
            "schema": "sec_d_circ_ext_evidence",
            "status": "not_available",
            "d_circ_ext": None,
        }
    values = [v for v in (ranked.d_circ_ext_x, ranked.d_circ_ext_z) if v is not None]
    metadata = ranked.metadata if isinstance(getattr(ranked, "metadata", None), dict) else {}
    evidence: dict[str, Any] = {
        "schema": "sec_d_circ_ext_evidence",
        "status": "probe_upper_bound" if values else "not_available",
        "d_circ_ext": (int(min(values)) if values else None),
        "d_circ_ext_x": int_or_none(ranked.d_circ_ext_x),
        "d_circ_ext_z": int_or_none(ranked.d_circ_ext_z),
        "schedule_catastrophic": bool(ranked.catastrophic),
        "delta_one_eliminated": int(ranked.delta_one_eliminated),
        "tau_a": int_or_none(ranked.tau_a),
        "stage3_group": int(ranked.stage3_group),
        # LRC RankedSchedule.metadata carries no policy id; twobasis does.
        "sec_policy_id": str(metadata.get("sec_policy_id", "lrc_ranked_v1")),
    }
    for key in ("schedule_degraded_x", "schedule_degraded_z"):
        if key in metadata:
            evidence[key] = bool(metadata[key])
    # v3.6 M1: multi-seed certification convergence (twobasis only; the LRC
    # ranker has no such notion). ``converged`` = every basis that was probed
    # had all refine seeds agree; None when neither basis reports it.
    converged_values = [
        metadata.get(key)
        for key in ("converged_x", "converged_z")
        if metadata.get(key) is not None
    ]
    if converged_values:
        evidence["converged_x"] = metadata.get("converged_x")
        evidence["converged_z"] = metadata.get("converged_z")
        evidence["converged"] = all(bool(v) for v in converged_values)
    # Review fix (M1 cost): the per-code ISD budget shrank the certification.
    # An audit flag only -- it never degrades a tier, strikes or excludes.
    if "certification_budget_exhausted" in metadata:
        evidence["certification_budget_exhausted"] = bool(
            metadata.get("certification_budget_exhausted")
        )
        evidence["isd_iters_executed"] = int_or_none(metadata.get("isd_iters_executed"))
    return evidence


def annotate_slow_checks(payload: dict[str, Any], *, verification: VerificationConfig) -> None:
    fast = payload.get("fast") if isinstance(payload.get("fast"), dict) else {}
    promotion = payload.get("promotion_record") if isinstance(payload.get("promotion_record"), dict) else {}
    d_proxy = int_or_none(fast.get("d"))
    if d_proxy is None:
        d_proxy = int_or_none(promotion.get("d_proxy") or promotion.get("d"))
    performance = {
        "schema": "slow_performance_check",
        "schema_version": SLOW_SCHEMA_VERSION,
        "rank_metric": "per_logical_per_round",
        "decoders": [decoder_name(decoder).value for decoder in verification.decoders],
        "noise": verification.noise.to_dict(),
        "p_values": (
            [float(p) for p in verification.p_values]
            if verification.noise.axis == "p"
            else None
        ),
        "shots": int(verification.shots),
    }
    distance_evidence = distance_evidence_from_results(
        payload.get("results") or [],
        d_proxy=d_proxy,
        decoder_order=[decoder_name(decoder).value for decoder in verification.decoders],
        exact_distance=payload.get("exact_distance") if isinstance(payload.get("exact_distance"), dict) else None,
        slope_offset=verification.slope_offset,
    )
    payload["slow_checks"] = {
        "schema": "slow_checks",
        "schema_version": SLOW_SCHEMA_VERSION,
        "performance": performance,
        "exact_distance": payload.get("exact_distance"),
        "distance_evidence": distance_evidence,
    }
    payload["distance_evidence"] = distance_evidence
    payload["d_circ_ler_estimate"] = distance_evidence.get("d_circ_ler_estimate")
    payload["degraded_estimate"] = distance_evidence.get("degraded_estimate")
    payload["exact_code_distance"] = distance_evidence.get("exact_code_distance")
    payload["exact_code_distance_status"] = distance_evidence.get("exact_code_distance_status")
    payload["exact_d_circ_status"] = distance_evidence.get("exact_d_circ_status")
    payload["exact_d_circ"] = distance_evidence.get("exact_d_circ")
    payload["d_circ_isd_upper_bound_status"] = distance_evidence.get("d_circ_isd_upper_bound_status")
    payload["d_circ_isd_upper_bound"] = distance_evidence.get("d_circ_isd_upper_bound")
    payload["d_circ_ext"] = distance_evidence.get("d_circ_ext")
    payload["degraded_ext"] = distance_evidence.get("degraded_ext")
    payload["verified_circuit_distance"] = distance_evidence.get("verified_circuit_distance")


def distance_evidence_from_results(
    results: list[dict[str, Any]],
    *,
    d_proxy: int | None,
    decoder_order: list[str],
    exact_distance: dict[str, Any] | None = None,
    slope_offset: float | None = VER1_LER_SLOPE_OFFSET,
) -> dict[str, Any]:
    code_exact = exact_distance.get("code") if isinstance(exact_distance, dict) and isinstance(exact_distance.get("code"), dict) else {}
    circuit_exact = (
        exact_distance.get("circuit")
        if isinstance(exact_distance, dict) and isinstance(exact_distance.get("circuit"), dict)
        else {}
    )
    circuit_isd = (
        exact_distance.get("circuit_isd")
        if isinstance(exact_distance, dict) and isinstance(exact_distance.get("circuit_isd"), dict)
        else {}
    )
    circuit_ext = (
        exact_distance.get("circuit_ext")
        if isinstance(exact_distance, dict) and isinstance(exact_distance.get("circuit_ext"), dict)
        else {}
    )
    _vcd_candidates = [
        value
        for value in (
            circuit_exact.get("d_circ_exact"),
            circuit_isd.get("d_circ_isd_upper_bound"),
            circuit_ext.get("d_circ_ext"),
        )
        if value is not None
    ]
    evidence: dict[str, Any] = {
        "schema": "distance_evidence",
        "schema_version": SLOW_SCHEMA_VERSION,
        "method": "exact_distance_isd_then_ler_slope_estimate",
        "status": "missing_d_proxy" if d_proxy is None else "insufficient_points",
        "d_proxy": d_proxy,
        "exact_code_distance_status": code_exact.get("status", "not_available"),
        "exact_code_distance": code_exact.get("d_exact"),
        "exact_code_distance_certified": bool(code_exact.get("certified")),
        "exact_code_distance_backend": code_exact.get("backend"),
        "exact_d_circ_status": circuit_exact.get("status", "not_available"),
        "exact_d_circ": circuit_exact.get("d_circ_exact"),
        "exact_d_circ_certified": bool(circuit_exact.get("certified")),
        "exact_d_circ_backend": circuit_exact.get("backend"),
        "d_circ_isd_upper_bound_status": circuit_isd.get("status", "not_available"),
        "d_circ_isd_upper_bound": circuit_isd.get("d_circ_isd_upper_bound"),
        "d_circ_isd_upper_bound_backend": circuit_isd.get("backend"),
        # A DEM-ISD bound below d_proxy is a constructive degradation proof
        # (an undetectable logical error of that weight exists in the circuit).
        "degraded_isd": bool(
            circuit_isd.get("d_circ_isd_upper_bound") is not None
            and circuit_isd.get("status") in {"ok", "partial"}
            and d_proxy is not None
            and int(circuit_isd["d_circ_isd_upper_bound"]) < int(d_proxy)
        ),
        # d_circ_ext below d_proxy is the same quarantine semantics: the LRC
        # schedule provably degrades the code distance at circuit level (02 §5).
        "d_circ_ext_status": circuit_ext.get("status", "not_available"),
        "d_circ_ext": circuit_ext.get("d_circ_ext"),
        "d_circ_ext_x": circuit_ext.get("d_circ_ext_x"),
        "d_circ_ext_z": circuit_ext.get("d_circ_ext_z"),
        # v3.6 (M2): degraded_ext is an AUDIT flag only (ranking.is_degraded
        # quarantines on schedule_catastrophic alone); it keeps being computed
        # exactly as before so the ledger predicate `degraded_ext` is stable.
        "degraded_ext": bool(
            circuit_ext.get("d_circ_ext") is not None
            and circuit_ext.get("status") in {"ok", "probe_upper_bound"}
            and d_proxy is not None
            and int(circuit_ext["d_circ_ext"]) < int(d_proxy)
        ),
        # M1 certification (sec/twobasis_rank.py) reports whether the
        # multi-seed kernel-ISD converged; an un-converged d_circ_ext is a
        # weaker statement and the flag lets an audit tell the two apart.
        "degraded_ext_converged": (
            bool(circuit_ext.get("converged")) if "converged" in circuit_ext else None
        ),
        "schedule_catastrophic": bool(circuit_ext.get("schedule_catastrophic")),
        "verified_circuit_distance": (
            min(int(value) for value in _vcd_candidates) if _vcd_candidates else None
        ),
        "exact_distance": exact_distance or {},
        "d_circ_ler_estimate": None,
        "d_circ_ler_estimate_raw": None,
        "degraded_estimate": None,
        "primary_decoder": None,
        "degraded_estimate_any_decoder": None,
        "worst_d_circ_ler_estimate": None,
        "estimates": [],
        "slope_offset": None if slope_offset is None else float(slope_offset),
        "ver1_reference": {
            "slope_offset": VER1_LER_SLOPE_OFFSET,
            "minimum_failures_per_point": VER1_D_CIRC_MIN_ERRORS,
            "minimum_points": VER1_D_CIRC_MIN_POINTS,
        },
        "difference_from_ver1": "ver1 had additional certification variants; this package records bounded code MILP, optional Stim circuit search, optional DEM-ISD upper bounds, sec d_circ_ext probes, and optional LER-slope estimates.",
    }
    if d_proxy is None:
        return evidence
    if slope_offset is None:
        evidence["status"] = "uncalibrated_noise_profile"
        evidence["failure_reason"] = (
            "LER-slope calibration is unavailable for this noise profile; "
            "exact code/Stim/DEM-ISD evidence remains valid"
        )
        return evidence
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        decoder = str(result.get("decoder") or "")
        if not decoder:
            continue
        grouped.setdefault(decoder, []).append(result)
    estimates: list[dict[str, Any]] = []
    for decoder in decoder_order + sorted(set(grouped) - set(decoder_order)):
        decoder_results = sorted(grouped.get(decoder) or [], key=lambda item: float(item.get("p") or 0.0))
        if not decoder_results:
            continue
        estimates.append(
            ler_slope_estimate_for_decoder(
                decoder,
                decoder_results,
                d_proxy=d_proxy,
                slope_offset=slope_offset,
            )
        )
    evidence["estimates"] = estimates
    ok_estimates = [item for item in estimates if item.get("status") == "ok"]
    if not ok_estimates:
        evidence["status"] = summarize_ler_slope_status(estimates)
        evidence["degraded_estimate_any_decoder"] = None
        return evidence
    primary = primary_ler_slope_estimate(ok_estimates, decoder_order)
    evidence["status"] = "ok"
    evidence["primary_decoder"] = primary.get("decoder")
    evidence["d_circ_ler_estimate"] = primary.get("d_circ_ler_estimate")
    evidence["d_circ_ler_estimate_raw"] = primary.get("d_circ_ler_estimate_raw")
    evidence["degraded_estimate"] = primary.get("degraded_estimate")
    evidence["degraded_estimate_any_decoder"] = any(bool(item.get("degraded_estimate")) for item in ok_estimates)
    evidence["worst_d_circ_ler_estimate"] = min(
        int(item["d_circ_ler_estimate"])
        for item in ok_estimates
        if item.get("d_circ_ler_estimate") is not None
    )
    return evidence


def ler_slope_estimate_for_decoder(
    decoder: str,
    results: list[dict[str, Any]],
    *,
    d_proxy: int,
    slope_offset: float = VER1_LER_SLOPE_OFFSET,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for result in results:
        combined = result.get("combined") if isinstance(result.get("combined"), dict) else {}
        per_logical_per_round = combined.get("per_logical_per_round")
        total_failures = total_basis_failures(result.get("basis") if isinstance(result.get("basis"), dict) else {})
        p = result.get("p")
        point = {
            "p": float_or_none(p),
            "per_logical_per_round": float_or_none(per_logical_per_round),
            "total_failures": total_failures,
        }
        if (
            point["p"] is not None
            and point["p"] > 0
            and point["per_logical_per_round"] is not None
            and point["per_logical_per_round"] > 0
            and total_failures >= VER1_D_CIRC_MIN_ERRORS
        ):
            points.append(point)
        else:
            skipped.append(point)
    estimate: dict[str, Any] = {
        "decoder": decoder,
        "status": "insufficient_points",
        "d_proxy": int(d_proxy),
        "d_circ_ler_estimate": None,
        "d_circ_ler_estimate_raw": None,
        "degraded_estimate": None,
        "fit_points": points,
        "skipped_points": skipped,
    }
    if len(points) < VER1_D_CIRC_MIN_POINTS:
        estimate["failure_reason"] = (
            f"need at least {VER1_D_CIRC_MIN_POINTS} p-points with "
            f">={VER1_D_CIRC_MIN_ERRORS} logical failures and positive per-logical-per-round LER"
        )
        return estimate
    slope = weighted_log_slope(points)
    if slope is None:
        estimate["status"] = "fit_error"
        estimate["failure_reason"] = "weighted log-log slope is undefined"
        return estimate
    d_raw = 2.0 * (float(slope) - float(slope_offset))
    d_estimate = int(max(1, min(round(d_raw), int(d_proxy))))
    estimate.update(
        {
            "status": "ok",
            "slope": float(slope),
            "slope_offset": float(slope_offset),
            "d_circ_ler_estimate_raw": float(d_raw),
            "d_circ_ler_estimate": d_estimate,
            "degraded_estimate": bool(d_estimate < int(d_proxy)),
            "failure_reason": None,
        }
    )
    return estimate


def weighted_log_slope(points: list[dict[str, Any]]) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    for point in points:
        p = float_or_none(point.get("p"))
        ler = float_or_none(point.get("per_logical_per_round"))
        failures = int_or_none(point.get("total_failures")) or 0
        if p is None or ler is None or p <= 0 or ler <= 0 or failures <= 0:
            continue
        xs.append(math.log10(p))
        ys.append(math.log10(ler))
        ws.append(math.sqrt(float(failures)))
    if len(xs) < VER1_D_CIRC_MIN_POINTS:
        return None
    weight_sum = sum(ws)
    if weight_sum <= 0:
        return None
    x_mean = sum(w * x for w, x in zip(ws, xs, strict=True)) / weight_sum
    y_mean = sum(w * y for w, y in zip(ws, ys, strict=True)) / weight_sum
    denom = sum(w * (x - x_mean) ** 2 for w, x in zip(ws, xs, strict=True))
    if denom <= 0:
        return None
    return sum(
        w * (x - x_mean) * (y - y_mean)
        for w, x, y in zip(ws, xs, ys, strict=True)
    ) / denom


def primary_ler_slope_estimate(
    estimates: list[dict[str, Any]],
    decoder_order: list[str],
) -> dict[str, Any]:
    by_decoder = {str(item.get("decoder")): item for item in estimates}
    for decoder in decoder_order:
        item = by_decoder.get(decoder)
        if item is not None:
            return item
    return estimates[0]


def summarize_ler_slope_status(estimates: list[dict[str, Any]]) -> str:
    if not estimates:
        return "no_decoder_results"
    statuses = {str(item.get("status") or "unknown") for item in estimates}
    if statuses == {"insufficient_points"}:
        return "insufficient_points"
    return "no_valid_fit"


def total_basis_failures(basis: dict[str, Any]) -> int:
    total = 0
    for item in basis.values():
        if isinstance(item, dict):
            total += int(item.get("failures") or 0)
    return total


def best_combined_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Best (lowest per_logical_per_round) ok decoder result for a candidate."""

    best: dict[str, Any] = {}
    best_value: float | None = None
    for result in payload.get("results") or []:
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        combined = result.get("combined") if isinstance(result.get("combined"), dict) else {}
        value = combined.get("per_logical_per_round")
        if value is None:
            continue
        if best_value is None or float(value) < best_value:
            best_value = float(value)
            best = {
                "per_logical_per_round": float(value),
                "per_logical_per_round_ci": combined.get("per_logical_per_round_ci"),
                "per_logical_block_ci": combined.get("per_logical_block_ci"),
                "block_ler": combined.get("block_ler"),
                "decoder": result.get("decoder"),
                "noise_point": result.get("noise_point"),
                "noise_axis": result.get("noise_axis"),
                "noise_value": result.get("noise_value"),
                "p": result.get("p"),
                "scale": result.get("scale"),
            }
    return best


def distance_evidence_feedback(candidate_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in candidate_payloads:
        if not isinstance(payload, dict):
            continue
        evidence = payload.get("distance_evidence")
        if not isinstance(evidence, dict):
            continue
        fast = payload.get("fast") if isinstance(payload.get("fast"), dict) else {}
        structural = payload.get("structural") if isinstance(payload.get("structural"), dict) else {}
        promotion = payload.get("promotion_record") if isinstance(payload.get("promotion_record"), dict) else {}
        rows.append(
            {
                "program_id": payload.get("program_id"),
                "is_baseline": bool(promotion.get("is_baseline")),
                "family": fast.get("family") or promotion.get("family"),
                "n": fast.get("n") or structural.get("n_data"),
                "k": fast.get("k") or structural.get("k"),
                "d_proxy": evidence.get("d_proxy") or fast.get("d") or promotion.get("d_proxy"),
                "fast_Q": fast.get("Q") or promotion.get("Q"),
                "status": evidence.get("status"),
                "method": evidence.get("method"),
                "primary_decoder": evidence.get("primary_decoder"),
                "d_circ_ler_estimate": evidence.get("d_circ_ler_estimate"),
                "d_circ_ler_estimate_raw": evidence.get("d_circ_ler_estimate_raw"),
                "degraded_estimate": evidence.get("degraded_estimate"),
                "degraded_estimate_any_decoder": evidence.get("degraded_estimate_any_decoder"),
                "worst_d_circ_ler_estimate": evidence.get("worst_d_circ_ler_estimate"),
                "exact_code_distance": evidence.get("exact_code_distance"),
                "exact_code_distance_status": evidence.get("exact_code_distance_status"),
                "exact_code_distance_certified": evidence.get("exact_code_distance_certified"),
                "exact_d_circ": evidence.get("exact_d_circ"),
                "exact_d_circ_status": evidence.get("exact_d_circ_status"),
                "exact_d_circ_certified": evidence.get("exact_d_circ_certified"),
                "d_circ_isd_upper_bound": evidence.get("d_circ_isd_upper_bound"),
                "d_circ_isd_upper_bound_status": evidence.get("d_circ_isd_upper_bound_status"),
                "d_circ_ext": evidence.get("d_circ_ext"),
                "d_circ_ext_status": evidence.get("d_circ_ext_status"),
                "degraded_ext": evidence.get("degraded_ext"),
                "degraded_ext_converged": evidence.get("degraded_ext_converged"),
                "schedule_catastrophic": evidence.get("schedule_catastrophic"),
                "verified_circuit_distance": evidence.get("verified_circuit_distance"),
                "code_hash": structural.get("code_hash") or promotion.get("code_hash"),
                "canonical_hash": structural.get("canonical_hash")
                or promotion.get("canonical_hash"),
            }
        )
    return rows
