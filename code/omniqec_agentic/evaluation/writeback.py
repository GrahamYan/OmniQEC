"""Result writeback: candidate JSON, slow_results.jsonl/csv rows, rankings
rows, queue status (03 §7). Mechanically extracted from slow.py; rows gain the
d_circ_ext / degraded_ext / verified_circuit_distance evidence columns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omniqec_agentic.config import DecoderName
from omniqec_agentic.results_csv import split_interval

from .common import int_or_none
from .evidence import best_combined_result


def _first_not_none(*values: Any) -> Any:
    """v3.7.1: the check weight sits in the structural result, the fast result
    and the promotion record; whichever layer of a payload is present wins.
    `or` is wrong here (a legitimate 0 would fall through)."""

    for value in values:
        if value is not None:
            return value
    return None

# Decoder NAME (the row's `decoder`) -> key of its blob in
# VerificationConfig.decoder_build.to_dict(). Under v3.5 the two coincided
# ("bposd"); the v3.6 decoder is named "telescoping_cpu" but keyed
# "telescoping", and a name-keyed lookup silently wrote decoder_config=None on
# every row (strict final audit + validator then failed the run).
DECODER_BUILD_KEYS: dict[str, str] = {
    DecoderName.BEAM.value: "beam",
    DecoderName.LSD.value: "lsd",
    DecoderName.BPOSD.value: "bposd",
    DecoderName.RELAYBP.value: "relaybp",
    DecoderName.TELESCOPING.value: "telescoping",
}


def decoder_build_blob(decoder_build: dict[str, Any] | None, decoder: Any) -> dict[str, Any] | None:
    """The recorded build blob of ``decoder`` (by name), or None."""

    if not isinstance(decoder_build, dict) or decoder is None:
        return None
    name = str(decoder)
    blob = decoder_build.get(DECODER_BUILD_KEYS.get(name, name))
    return blob if isinstance(blob, dict) else None


def write_slow_result(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def slow_queue_status(
    *,
    record_count: int,
    timeout_candidate_count: int,
    decoder_attempt_candidate_count: int,
    ok_result_count: int,
    catastrophic_candidate_count: int = 0,
    measurement_blocked_candidate_count: int = 0,
    incomplete_candidate_count: int = 0,
    decode_infeasible_candidate_count: int = 0,
    decoder_unsupported_candidate_count: int = 0,
) -> str:
    """Queue-level verdict for one slow stage.

    A schedule_catastrophic candidate is a SUCCESSFULLY IDENTIFIED terminal
    state (v3.2 §4.4): the protocol measured what there was to measure — that
    the code admits no usable schedule. It is subtracted from the effective
    queue before the failure verdicts, so an all-catastrophic queue reports
    ``ok`` (and previously failed the whole strict turn as
    no_valid_candidates/results before reconcile could even record it).

    A decode_infeasible candidate (v3.4) is likewise terminal: its decoder
    exceeded the frozen per-shot budget on a deterministic replay, so the
    protocol has established that the code cannot be measured under the
    frozen budget. It is subtracted the same way and never blocks the queue.

    A decoder_unsupported candidate (v3.6.1) is the third terminal kind: the
    decoder cannot be built for its DEM (identical detector support with
    different logical support, i.e. circuit distance 2), which is a finding
    about the code. Subtracted the same way."""

    if int(record_count) <= 0:
        return "empty_queue"
    effective = (
        int(record_count)
        - int(catastrophic_candidate_count)
        - int(decode_infeasible_candidate_count)
        - int(decoder_unsupported_candidate_count)
    )
    if effective <= 0:
        return "ok"
    # A partial queue is not a successful scientific sample. Previously one
    # cheap candidate could make the turn ``ok`` while one or more expensive
    # candidates timed out and silently disappeared from selection. Any
    # infrastructure-censored candidate now pauses the queue as a whole.  The
    # broad incomplete count is deliberate: build/input/subprocess failures
    # must not silently disappear merely because a cheaper peer decoded.
    if (
        int(timeout_candidate_count) > 0
        or int(measurement_blocked_candidate_count) > 0
        or int(incomplete_candidate_count) > 0
    ):
        return "measurement_blocked"
    if int(decoder_attempt_candidate_count) <= 0:
        return "no_valid_candidates"
    if int(ok_result_count) <= 0:
        return "no_valid_results"
    return "ok"


def summarize_evidence_status(rows: list[dict[str, Any]], key: str) -> str:
    statuses = {str(row.get(key) or "not_available") for row in rows}
    if "ok" in statuses:
        return "ok"
    if "partial" in statuses:
        return "partial"
    if not statuses:
        return "not_available"
    if len(statuses) == 1:
        return next(iter(statuses))
    return "mixed"


def has_decoder_attempt(payload: dict[str, Any]) -> bool:
    for result in payload.get("results") or []:
        if isinstance(result, dict) and result.get("decoder"):
            return True
    return False


def compact_slow_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    fast = payload.get("fast") if isinstance(payload.get("fast"), dict) else {}
    structural = payload.get("structural") if isinstance(payload.get("structural"), dict) else {}
    promotion = payload.get("promotion_record") if isinstance(payload.get("promotion_record"), dict) else {}
    distance_evidence = payload.get("distance_evidence") if isinstance(payload.get("distance_evidence"), dict) else {}
    best = best_combined_result(payload)
    return {
        "program_id": payload.get("program_id"),
        "status": payload.get("status"),
        "isolation": payload.get("isolation"),
        "hardware": payload.get("hardware"),
        "path": str(path),
        "is_baseline": bool(promotion.get("is_baseline")),
        "code_hash": structural.get("code_hash") or promotion.get("code_hash"),
        "per_logical_per_round": best.get("per_logical_per_round"),
        "per_logical_per_round_ci": best.get("per_logical_per_round_ci"),
        "per_logical_block_ci": best.get("per_logical_block_ci"),
        "block_ler": best.get("block_ler"),
        "decoder": best.get("decoder"),
        "noise_point": best.get("noise_point"),
        "noise_axis": best.get("noise_axis"),
        "noise_value": best.get("noise_value"),
        "p": best.get("p"),
        "scale": best.get("scale"),
        "family": fast.get("family") or promotion.get("family"),
        "n": fast.get("n") or structural.get("n_data") or promotion.get("n"),
        "k": fast.get("k") or structural.get("k") or promotion.get("k"),
        "d_proxy": fast.get("d") or promotion.get("d_proxy"),
        # v3.7.1: the check weight is computed at construction and was recorded
        # in the structural result, but no slow row ever carried it -- so the
        # cross-turn hall of fame, S-5 and every CSV read null, and the planner
        # could not adjudicate a weight hypothesis it had itself proposed.
        "max_check_weight": _first_not_none(
            structural.get("max_check_weight"),
            fast.get("max_check_weight"),
            promotion.get("max_check_weight"),
        ),
        "max_qubit_degree": _first_not_none(
            structural.get("max_qubit_degree"),
            fast.get("max_qubit_degree"),
            promotion.get("max_qubit_degree"),
        ),
        "d_circ_ler_estimate": distance_evidence.get("d_circ_ler_estimate"),
        "degraded_estimate": distance_evidence.get("degraded_estimate"),
        "degraded_estimate_any_decoder": distance_evidence.get("degraded_estimate_any_decoder"),
        "degraded_isd": distance_evidence.get("degraded_isd"),
        "degraded_ext": distance_evidence.get("degraded_ext"),
        "d_circ_ler_estimate_status": distance_evidence.get("status"),
        "d_circ_ler_estimate_decoder": distance_evidence.get("primary_decoder"),
        "exact_code_distance": distance_evidence.get("exact_code_distance"),
        "exact_code_distance_status": distance_evidence.get("exact_code_distance_status"),
        "exact_code_distance_certified": distance_evidence.get("exact_code_distance_certified"),
        "exact_d_circ": distance_evidence.get("exact_d_circ"),
        "exact_d_circ_status": distance_evidence.get("exact_d_circ_status"),
        "exact_d_circ_certified": distance_evidence.get("exact_d_circ_certified"),
        "d_circ_isd_upper_bound": distance_evidence.get("d_circ_isd_upper_bound"),
        "d_circ_isd_upper_bound_status": distance_evidence.get("d_circ_isd_upper_bound_status"),
        "d_circ_ext": distance_evidence.get("d_circ_ext"),
        "d_circ_ext_status": distance_evidence.get("d_circ_ext_status"),
        "verified_circuit_distance": distance_evidence.get("verified_circuit_distance"),
        "fast_Q": fast.get("Q") or promotion.get("Q"),
        "result_count": len(payload.get("results") or []),
        "runtime_s": payload.get("runtime_s"),
    }


def ranking_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    fast = payload.get("fast") if isinstance(payload.get("fast"), dict) else {}
    fast_metrics = fast.get("metrics") if isinstance(fast.get("metrics"), dict) else fast
    structural = payload.get("structural") if isinstance(payload.get("structural"), dict) else {}
    promotion = payload.get("promotion_record") if isinstance(payload.get("promotion_record"), dict) else {}
    distance_evidence = payload.get("distance_evidence") if isinstance(payload.get("distance_evidence"), dict) else {}
    compiled = payload.get("compiled_code") if isinstance(payload.get("compiled_code"), dict) else {}
    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    decoder_build = verification.get("decoder_build") if isinstance(verification.get("decoder_build"), dict) else {}
    racing = payload.get("racing") if isinstance(payload.get("racing"), dict) else {}
    # Candidate-level verdict (v3.2 §4.4): the real circuit path RAISES before
    # any decoder result exists, so per-result evidence alone cannot carry the
    # flag reliably; take it from any layer that saw it.
    catastrophic = bool(
        payload.get("schedule_catastrophic")
        or payload.get("status") == "schedule_catastrophic"
        or distance_evidence.get("schedule_catastrophic")
        or any(
            isinstance(result, dict) and result.get("schedule_catastrophic")
            for result in payload.get("results") or []
        )
    )
    rows: list[dict[str, Any]] = []
    for result in payload.get("results") or []:
        combined = result.get("combined") if isinstance(result.get("combined"), dict) else {}
        measurement_eligible = (
            payload.get("status") == "ok" and result.get("status") == "ok"
        )
        published_combined = combined if measurement_eligible else {}
        basis = result.get("basis") if isinstance(result.get("basis"), dict) else {}
        basis_x = basis.get("X") if isinstance(basis.get("X"), dict) else {}
        basis_z = basis.get("Z") if isinstance(basis.get("Z"), dict) else {}
        noise_point = result.get("noise_point") if isinstance(result.get("noise_point"), dict) else {}
        rows.append(
            {
                "program_id": payload.get("program_id"),
                "decoder": result.get("decoder"),
                "noise_point": result.get("noise_point"),
                "noise_axis": result.get("noise_axis"),
                "noise_value": result.get("noise_value"),
                "p": result.get("p"),
                "scale": result.get("scale"),
                "rounds": result.get("rounds"),
                "status": (
                    "ok"
                    if measurement_eligible
                    else payload.get("status") or result.get("status") or "measurement_blocked"
                ),
                "isolation": payload.get("isolation"),
                "block_ler": published_combined.get("block_ler"),
                "per_logical_per_round": published_combined.get("per_logical_per_round"),
                "per_logical_per_round_ci": published_combined.get("per_logical_per_round_ci"),
                "per_logical_block_ci": published_combined.get("per_logical_block_ci"),
                "total_failures": published_combined.get("total_failures"),
                "total_shots": published_combined.get("total_shots"),
                "is_baseline": bool(promotion.get("is_baseline")),
                "basis_logical_error_rate": published_combined.get("basis_logical_error_rate"),
                "basis_ler_ci": published_combined.get("basis_ler_ci"),
                "ci_z": published_combined.get("ci_z"),
                "shots_x": basis_x.get("shots"),
                "failures_x": basis_x.get("failures"),
                "ler_x": basis_x.get("logical_error_rate"),
                "shots_z": basis_z.get("shots"),
                "failures_z": basis_z.get("failures"),
                "ler_z": basis_z.get("logical_error_rate"),
                "sampling_source": result.get("sampling_source"),
                "stopped_by": result.get("stopped_by") or racing.get("stopped_by"),
                "racing_shards": racing.get("shards_consumed"),
                "runtime_s": sum(
                    float(item.get("runtime_s") or 0.0)
                    for item in (basis_x, basis_z)
                    if item
                ),
                "failure_reason": "; ".join(
                    str(reason)
                    for reason in (
                        result.get("failure_reason"),
                        payload.get("failure_reason") if not measurement_eligible else None,
                        *(item.get("failure_reason") for item in (basis_x, basis_z)),
                    )
                    if reason
                ),
                "family": fast.get("family") or promotion.get("family"),
                "canonical_family": compiled.get("canonical_family") or fast.get("family") or promotion.get("family"),
                "deformation_mode": compiled.get("deformation_mode") or compiled.get("deformation") or "none",
                "h_mask_weight": compiled.get("h_mask_weight"),
                "h_mask_sha256": compiled.get("h_mask_sha256"),
                "n": fast.get("n") or structural.get("n_data"),
                "n_checks_x": structural.get("n_checks_x"),
                "n_checks_z": structural.get("n_checks_z"),
                "total_budget": (
                    int(structural.get("n_data") or 0)
                    + int(structural.get("n_checks_x") or 0)
                    + int(structural.get("n_checks_z") or 0)
                ),
                "n_budget": structural.get("n_budget"),
                "k": fast.get("k") or structural.get("k"),
                "max_check_weight": _first_not_none(
                    structural.get("max_check_weight"),
                    fast.get("max_check_weight"),
                    promotion.get("max_check_weight"),
                ),
                "max_qubit_degree": _first_not_none(
                    structural.get("max_qubit_degree"),
                    fast.get("max_qubit_degree"),
                    promotion.get("max_qubit_degree"),
                ),
                "d_x_proxy": fast_metrics.get("d_x"),
                "d_z_proxy": fast_metrics.get("d_z"),
                "d_proxy": fast.get("d") or promotion.get("d_proxy"),
                "d_search": (
                    fast.get("d_search") or fast.get("d") or promotion.get("d_search")
                ),
                # Which channel put this candidate in the queue (v3.2 §4.6): a
                # code can be both a structural elite and a runtime champion,
                # and the two columns must not overwrite each other.
                "selection_channels": promotion.get("selection_channels"),
                "structural_slot_reason": promotion.get("structural_slot_reason"),
                "d_circ_ler_estimate": distance_evidence.get("d_circ_ler_estimate"),
                "degraded_estimate": distance_evidence.get("degraded_estimate"),
                "degraded_estimate_any_decoder": distance_evidence.get("degraded_estimate_any_decoder"),
                "degraded_isd": distance_evidence.get("degraded_isd"),
                "degraded_ext": distance_evidence.get("degraded_ext"),
                "d_circ_ler_estimate_status": distance_evidence.get("status"),
                "d_circ_ler_estimate_decoder": distance_evidence.get("primary_decoder"),
                "d_circ_ler_estimate_method": distance_evidence.get("method"),
                "exact_code_distance": distance_evidence.get("exact_code_distance"),
                "exact_code_distance_status": distance_evidence.get("exact_code_distance_status"),
                "exact_code_distance_certified": distance_evidence.get("exact_code_distance_certified"),
                "exact_d_circ": distance_evidence.get("exact_d_circ"),
                "exact_d_circ_status": distance_evidence.get("exact_d_circ_status"),
                "exact_d_circ_certified": distance_evidence.get("exact_d_circ_certified"),
                "d_circ_isd_upper_bound": distance_evidence.get("d_circ_isd_upper_bound"),
                "d_circ_isd_upper_bound_status": distance_evidence.get("d_circ_isd_upper_bound_status"),
                "d_circ_ext": distance_evidence.get("d_circ_ext"),
                "d_circ_ext_status": distance_evidence.get("d_circ_ext_status"),
                # The ONE hard exclusion (v3.2 §4.4): no sampled schedule
                # avoids a single-fault logical, so the candidate cannot be
                # measured at all. It was computed but never surfaced on the
                # rows, which is the only place downstream filters can see it.
                "schedule_catastrophic": catastrophic,
                "verified_circuit_distance": distance_evidence.get("verified_circuit_distance"),
                "fast_Q": fast.get("Q") or promotion.get("Q"),
                "code_hash": structural.get("code_hash") or promotion.get("code_hash"),
                # v3.6 B3: the twin gate reads the canonical (pynauty) hash;
                # the WL hash is recorded for audit only and never gates.
                "canonical_hash": (
                    structural.get("canonical_hash")
                    or (structural.get("metrics") or {}).get("canonical_hash")
                    or promotion.get("canonical_hash")
                ),
                "wl_hash": (
                    (structural.get("metrics") or {}).get("wl_hash")
                    or promotion.get("wl_hash")
                ),
                "promotion_rank": promotion.get("promotion_rank"),
                "archive_cell": promotion.get("archive_cell"),
                "noise_profile": noise_point.get("profile"),
                "noise_config_hash": noise_point.get("config_hash"),
                "decoder_config": decoder_build_blob(decoder_build, result.get("decoder")),
                "candidate_path": str(path),
            }
        )
    if not rows and payload.get("program_id"):
        # Candidate-level terminal row: a payload that failed before any
        # result existed (or an artifact with no noise points) must still be
        # visible to the reconcile/queue filters, or it re-takes a slow slot
        # every turn.
        rows.append(
            {
                "program_id": payload.get("program_id"),
                "decoder": None,
                "status": str(payload.get("status") or "no_results"),
                "failure_reason": payload.get("failure_reason"),
                "schedule_catastrophic": catastrophic,
                "is_baseline": bool(promotion.get("is_baseline")),
                "family": fast.get("family") or promotion.get("family"),
                "n": fast.get("n") or structural.get("n_data"),
                "k": fast.get("k") or structural.get("k"),
                "d_proxy": fast.get("d") or promotion.get("d_proxy"),
                "d_search": (
                    fast.get("d_search") or fast.get("d") or promotion.get("d_search")
                ),
                "selection_channels": promotion.get("selection_channels"),
                "structural_slot_reason": promotion.get("structural_slot_reason"),
                "max_check_weight": _first_not_none(
                    structural.get("max_check_weight"),
                    fast.get("max_check_weight"),
                    promotion.get("max_check_weight"),
                ),
                "max_qubit_degree": _first_not_none(
                    structural.get("max_qubit_degree"),
                    fast.get("max_qubit_degree"),
                    promotion.get("max_qubit_degree"),
                ),
                "code_hash": structural.get("code_hash") or promotion.get("code_hash"),
                "canonical_hash": (
                    structural.get("canonical_hash")
                    or (structural.get("metrics") or {}).get("canonical_hash")
                    or promotion.get("canonical_hash")
                ),
                "wl_hash": (
                    (structural.get("metrics") or {}).get("wl_hash")
                    or promotion.get("wl_hash")
                ),
                "candidate_path": str(path),
            }
        )
    return rows


def slow_csv_rows(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank_index, row in enumerate(rankings, start=1):
        n_data = int_or_none(row.get("n"))
        k = int_or_none(row.get("k"))
        ci_low, ci_high = split_interval(row.get("per_logical_block_ci"))
        plpr_ci_low, plpr_ci_high = split_interval(row.get("per_logical_per_round_ci"))
        output.append(
            {
                "rank": rank_index,
                "program_id": row.get("program_id"),
                "code_hash": row.get("code_hash"),
                "family": row.get("family"),
                "canonical_family": row.get("canonical_family"),
                "deformation_mode": row.get("deformation_mode"),
                "h_mask_weight": row.get("h_mask_weight"),
                "h_mask_sha256": row.get("h_mask_sha256"),
                "status": row.get("status"),
                "artifact_path": row.get("candidate_path"),
                "sampling_source": row.get("sampling_source"),
                "stopped_by": row.get("stopped_by"),
                "racing_shards": row.get("racing_shards"),
                "n_data": n_data,
                "n_checks_x": row.get("n_checks_x"),
                "n_checks_z": row.get("n_checks_z"),
                "total_budget": row.get("total_budget"),
                "n_budget": row.get("n_budget"),
                "k": k,
                "d_x_proxy": row.get("d_x_proxy"),
                "d_z_proxy": row.get("d_z_proxy"),
                "d_proxy": row.get("d_proxy"),
                "d_search": row.get("d_search"),
                "selection_channels": ",".join(row.get("selection_channels") or []),
                "structural_slot_reason": row.get("structural_slot_reason"),
                "exact_code_distance": row.get("exact_code_distance"),
                "exact_code_distance_status": row.get("exact_code_distance_status"),
                "d_circ_ext": row.get("d_circ_ext"),
                "degraded_ext": row.get("degraded_ext"),
                "verified_circuit_distance": row.get("verified_circuit_distance"),
                "noise_profile": row.get("noise_profile"),
                "noise_config_hash": row.get("noise_config_hash"),
                "noise_axis": row.get("noise_axis"),
                "noise_value": row.get("noise_value"),
                "p": row.get("p"),
                "scale": row.get("scale"),
                "decoder": row.get("decoder"),
                "decoder_config": row.get("decoder_config"),
                "rounds": row.get("rounds"),
                "shots_x": row.get("shots_x"),
                "failures_x": row.get("failures_x"),
                "ler_x": row.get("ler_x"),
                "shots_z": row.get("shots_z"),
                "failures_z": row.get("failures_z"),
                "ler_z": row.get("ler_z"),
                "total_shots": row.get("total_shots"),
                "total_failures": row.get("total_failures"),
                "block_ler": row.get("block_ler"),
                "per_logical_per_round": row.get("per_logical_per_round"),
                "per_logical_per_round_ci_low": plpr_ci_low,
                "per_logical_per_round_ci_high": plpr_ci_high,
                "per_logical_block_ci_low": ci_low,
                "per_logical_block_ci_high": ci_high,
                "ci_z": row.get("ci_z"),
                "statistical_group": row.get("statistical_group"),
                "statistically_tied": row.get("statistically_tied"),
                "runtime_s": row.get("runtime_s"),
                "failure_reason": row.get("failure_reason"),
            }
        )
    return output
