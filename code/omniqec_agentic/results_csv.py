"""Stable, spreadsheet-friendly result export shared by slow and Sinter runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from omniqec_agentic.verification.ler import LER_DEFINITION


# v7 (2026-09-29): + max_check_weight / max_qubit_degree (additive columns).
RESULTS_CSV_SCHEMA_VERSION = 7


RESULTS_CSV_COLUMNS = (
    "results_schema_version",
    "ler_definition",
    "rank",
    "program_id",
    "code_hash",
    "family",
    "canonical_family",
    "deformation_mode",
    "h_mask_weight",
    "h_mask_sha256",
    "status",
    "artifact_path",
    "sampling_source",
    "stopped_by",
    "racing_shards",
    "n_data",
    "n_checks_x",
    "n_checks_z",
    "total_budget",
    "n_budget",
    "k",
    "d_x_proxy",
    "d_z_proxy",
    "d_proxy",
    # v3.2: the 4-seed x {X,Z} minimum, and which channel selected the row.
    # `d_proxy` stays as its read-only alias so existing readers keep working.
    "d_search",
    # v3.7.1: the check weight travels with the measurement. It decided
    # structural retention from v3.7 on, yet no result row carried it.
    "max_check_weight",
    "max_qubit_degree",
    "selection_channels",
    "structural_slot_reason",
    "exact_code_distance",
    "exact_code_distance_status",
    "d_circ_ext",
    "degraded_ext",
    "verified_circuit_distance",
    "noise_profile",
    "noise_config_hash",
    "noise_axis",
    "noise_value",
    "p",
    "scale",
    "decoder",
    "decoder_config",
    "rounds",
    "shots_x",
    "failures_x",
    "ler_x",
    "shots_z",
    "failures_z",
    "ler_z",
    "total_shots",
    "total_failures",
    "block_ler",
    "per_logical_per_round",
    "per_logical_per_round_ci_low",
    "per_logical_per_round_ci_high",
    "per_logical_block_ci_low",
    "per_logical_block_ci_high",
    "ci_z",
    "statistical_group",
    "statistically_tied",
    "runtime_s",
    "failure_reason",
)


def write_results_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write a stable-column CSV; nested values become canonical JSON cells."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            normalized.setdefault("results_schema_version", RESULTS_CSV_SCHEMA_VERSION)
            normalized.setdefault("ler_definition", LER_DEFINITION)
            writer.writerow(
                {column: _csv_cell(normalized.get(column)) for column in RESULTS_CSV_COLUMNS}
            )
    return output


def split_interval(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            pass
    return None, None


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value
