"""Shot-shard partitioning, seed derivation, shard merging, and the anchor-p
double-threshold racing controller (03 §7-§8).

The shard stream is pre-registered as (seed, shards_max) so adaptive
consumption depth never changes the RNG protocol; ``racing_decision`` is a
pure function of cumulative counts and the static threshold snapshot, so
elimination is independent of shard completion order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omniqec_agentic.config import RacingConfig
from omniqec_agentic.verification.noise import NoiseProfile

# Per-profile anchor points (03 §8, from 02 §6.1 profile defaults). Racing
# runs only at the anchor; the full p sweep belongs to the final pipeline.
RACING_ANCHOR_P: dict[NoiseProfile, float] = {
    # v3.5 formal model (paper SI C, no idle noise) and the retired v3.4
    # idle-noise model share the paper's single physical error rate.
    NoiseProfile.UNIFORM_PAPER: 0.005,
    NoiseProfile.UNIFORM_LEGACY: 0.005,
    NoiseProfile.SI1000: 1.0e-3,
    NoiseProfile.SI1000_MEASUREMENT_X2: 1.0e-3,
    NoiseProfile.HELIOS: 1.0,
    NoiseProfile.HELIOS_SYMMETRIZED: 1.0,
    NoiseProfile.HELIOS_PAULI_SHARPENED: 1.0,
    NoiseProfile.PAULI_LABEL_UNIFORM: 1.0e-3,  # brute-force study 2026-09-15
}


# 2026-09-29: a capped final candidate produced 128 shards x 64 chunks x 2 bases
# of identical "worker exceeded timeout" / "cancelled" messages, i.e. a 2 MB
# failure_reason that no CSV reader could open. Identical messages are
# counted instead of repeated, and the text is bounded.
FAILURE_REASON_MAX_CHARS = 4000


def join_failure_reasons(reasons: list[str], *, limit: int = FAILURE_REASON_MAX_CHARS) -> str:
    """"; "-join of ``reasons`` with identical messages folded into ``msg xN``
    (first-seen order), truncated to ``limit`` characters with a marker."""

    counts: dict[str, int] = {}
    for reason in reasons:
        text = str(reason)
        counts[text] = counts.get(text, 0) + 1
    joined = "; ".join(text if n == 1 else f"{text} x{n}" for text, n in counts.items())
    if len(joined) > int(limit):
        joined = joined[: int(limit)] + f" ...(+{len(joined) - int(limit)} chars)"
    return joined


def anchor_p_for_profile(profile: NoiseProfile | str) -> float:
    if not isinstance(profile, NoiseProfile):
        profile = NoiseProfile(profile)
    return RACING_ANCHOR_P[profile]


@dataclass(frozen=True)
class RacingThresholds:
    """Static snapshot taken at plan entry (03 §8): plpr-scale (paper A5) CI
    upper bounds of the hall-of-fame best and the resident baseline at the
    anchor — plpr is the objective, and ``per_logical_per_round_ci`` from the
    A5 transform is directly comparable across candidates. ``None`` arms are
    unavailable (early campaign) and never eliminate."""

    hof_plpr_ub: float | None = None
    baseline_plpr_ub: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hof_plpr_ub": self.hof_plpr_ub,
            "baseline_plpr_ub": self.baseline_plpr_ub,
        }


def racing_shard_stream(config: RacingConfig) -> list[int]:
    """Pre-registered shard sizes: batch-size shards covering shots_max."""

    batch = int(config.batch)
    total = int(config.shots_max)
    return [min(batch, total - start) for start in range(0, total, batch)]


def racing_pilot_shards(config: RacingConfig) -> int:
    return -(-int(config.shots_pilot) // int(config.batch))


def decoder_cost_cap_s_per_shot(config: RacingConfig, n: int) -> float | None:
    """pilot10x_maxerr100_cost_v4: the CPU-seconds-per-shot a candidate with
    ``n`` data qubits may cost, or None when the policy has no cap."""

    per_qubit = getattr(config, "cost_cap_s_per_shot_per_qubit", None)
    if per_qubit is None:
        return None
    return max(float(getattr(config, "cost_cap_floor_s_per_shot", 0.0)), float(per_qubit) * int(n))


def decoder_cost_decision(
    *,
    cost_s_per_shot: float | None,
    shots: int,
    n: int,
    config: RacingConfig,
) -> dict[str, Any] | None:
    """pilot10x_maxerr100_cost_v4: stop a candidate whose measured decoder
    cost (CPU-seconds per decoded shot, both bases pooled) exceeds the cap for
    its size once at least ``cost_cap_min_shots`` shots per basis landed.
    Returns the stop decision (stopped_by="decoder_cost", never confirmed) or
    None when the cap is off, not yet judgeable, or not exceeded. The caller
    turns the stop into a decode_infeasible verdict."""

    cap = decoder_cost_cap_s_per_shot(config, n)
    if cap is None or cost_s_per_shot is None:
        return None
    if int(shots) < int(getattr(config, "cost_cap_min_shots", 1)):
        return None
    if float(cost_s_per_shot) > cap:
        return {
            "action": "stop",
            "stopped_by": "decoder_cost",
            "confirmed": False,
            "cost_s_per_shot": float(cost_s_per_shot),
            "cost_cap_s_per_shot": float(cap),
        }
    return None


def racing_decision(
    *,
    shots: int,
    basis_failures: dict[str, int],
    plpr_ci_low: float | None,
    config: RacingConfig,
    thresholds: RacingThresholds | None,
) -> dict[str, Any]:
    """One round-boundary decision from cumulative counts (order-free).

    Returns {"action": "continue"|"stop", "stopped_by": None|"racing"|
    "max_errors", "confirmed": bool}. Threshold arms that are ``None`` never
    eliminate (OR-semantics: surviving one arm is enough). shots_max only
    sizes the pre-registered stream: shots_confirm <= shots_max is enforced
    by RacingConfig, so the confirm check always stops consumption first."""

    shots = int(shots)
    if shots < int(config.shots_pilot):
        # pilot10x_maxerr100_v3: the error cap is decided at every landed
        # shard, the pilot included; v1/v2 never decide before the pilot.
        if (
            bool(getattr(config, "max_errors_in_pilot", False))
            and config.max_errors is not None
            and basis_failures
            and all(int(count) >= int(config.max_errors) for count in basis_failures.values())
        ):
            return {"action": "stop", "stopped_by": "max_errors", "confirmed": True}
        return {"action": "continue", "stopped_by": None, "confirmed": False}
    if thresholds is not None and plpr_ci_low is not None:
        arms = [
            ub * (1.0 + float(config.epsilon))
            for ub in (thresholds.hof_plpr_ub, thresholds.baseline_plpr_ub)
            if ub is not None
        ]
        # Paper policy (strict_exceed, Appendix D3): the lower bound must EXCEED ten times every available
        # upper bound; the earlier policies stop at equality too.
        strict = bool(getattr(config, "strict_exceed", False))
        if arms and all(
            (float(plpr_ci_low) > arm) if strict else (float(plpr_ci_low) >= arm) for arm in arms
        ):
            return {"action": "stop", "stopped_by": "racing", "confirmed": False}
    if config.max_errors is not None and basis_failures and all(
        int(count) >= int(config.max_errors) for count in basis_failures.values()
    ):
        return {"action": "stop", "stopped_by": "max_errors", "confirmed": True}
    if shots >= int(config.shots_confirm):
        return {"action": "stop", "stopped_by": None, "confirmed": True}
    return {"action": "continue", "stopped_by": None, "confirmed": False}


def merge_shard_results(shards: list[dict[str, Any] | None], *, decoder: str) -> dict[str, Any]:
    """Combine shot-shard decode results into one basis result (failures and
    shots sum; the estimate from partial shards is still unbiased sampling)."""

    rows = [row for row in shards if isinstance(row, dict)]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    shots = sum(int(row.get("shots") or 0) for row in ok_rows)
    failures = sum(int(row.get("failures") or 0) for row in ok_rows)
    bad = [row for row in rows if row.get("status") != "ok"] + [None] * (len(shards) - len(rows))
    merged: dict[str, Any] = {
        "status": "ok" if ok_rows and not bad else ("partial_shards" if ok_rows else "decoder_error"),
        "decoder": decoder,
        "shots": shots,
        "failures": failures,
        "logical_error_rate": (failures / shots) if shots else None,
        "runtime_s": sum(float(row.get("runtime_s") or 0.0) for row in rows),
        "failure_reason": join_failure_reasons(
            [
                str(row.get("failure_reason") or row.get("status"))
                for row in rows
                if row.get("status") != "ok"
            ]
            + (["shard_timeout"] if any(row is None for row in shards) else [])
        ),
        "shard_count": len(shards),
        "shard_statuses": [row.get("status") if isinstance(row, dict) else "timeout" for row in shards],
        # v3.4 diagnostics: per-shard wall time and worker attempts (None for
        # a slot whose job never returned at all).
        "shard_runtime_s": [
            (
                float(row["runtime_s"])
                if isinstance(row, dict) and row.get("runtime_s") is not None
                else None
            )
            for row in shards
        ],
        "shard_attempts": [
            (
                int(row["job_execution"]["attempts"])
                if isinstance(row, dict)
                and isinstance(row.get("job_execution"), dict)
                and row["job_execution"].get("attempts") is not None
                else None
            )
            for row in shards
        ],
        # v3.6 (M4): per-shard peak RSS in bytes when the worker reported it
        # (job_execution.ru_maxrss); None for a shard without the record.
        "shard_max_rss_bytes": [
            (
                int(row["job_execution"]["ru_maxrss"])
                if isinstance(row, dict)
                and isinstance(row.get("job_execution"), dict)
                and row["job_execution"].get("ru_maxrss") is not None
                else None
            )
            for row in shards
        ],
    }
    if ok_rows:
        metrics = dict(ok_rows[0].get("metrics") or {})
        requested_shots = sum(
            int((row.get("metrics") or {}).get("requested_shots") or row.get("shots") or 0)
            for row in ok_rows
        )
        metrics.update(
            shards=len(shards),
            requested_shots=requested_shots,
            actual_shots=shots,
            sampling_complete=bool(not bad and shots == requested_shots),
        )
        metrics["stopped_by_max_errors"] = any(
            (row.get("metrics") or {}).get("stopped_by_max_errors") for row in ok_rows
        )
        # v3.6 telescoping stage bookkeeping (docs/06 §1.1 "记录", user
        # decision 2026-09-03): each shard reports how many shots each stage
        # (S3A / S3B / S3C / NC) accepted / got wrong and the time spent per
        # stage (worker metrics stage_counts / stage_failures / stage_time_s,
        # plus the scalar declared_failures: NC shots counted as logical
        # errors); the basis total is their sum so the ledger can audit how
        # often a code exhausted every stage. Absent keys stay absent.
        for key in ("stage_counts", "stage_failures", "stage_time_s"):
            totals: dict[str, float] = {}
            present = False
            for row in ok_rows:
                stages = (row.get("metrics") or {}).get(key)
                if not isinstance(stages, dict):
                    continue
                present = True
                for name, value in stages.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[str(name)] = totals.get(str(name), 0) + value
            if present:
                metrics[key] = {
                    name: (round(float(v), 4) if key == "stage_time_s" else int(v))
                    for name, v in totals.items()
                }
        for key in ("declared_failures",):
            values = [
                (row.get("metrics") or {}).get(key)
                for row in ok_rows
                if isinstance((row.get("metrics") or {}).get(key), (int, float))
                and not isinstance((row.get("metrics") or {}).get(key), bool)
            ]
            if values:
                metrics[key] = int(sum(values))
        merged["metrics"] = metrics
        merged["memory_experiment"] = ok_rows[0].get("memory_experiment")
        # v3.6: the decode stage ("search" | "final") is the pool's tag on
        # every shard (no stage-dependent limit under the telescoping CPU
        # protocol); every shard of one cell runs the same stage, so the
        # first ok shard's value is the cell's (absent for pre-v3.6 workers).
        if "decode_stage" in ok_rows[0]:
            merged["decode_stage"] = ok_rows[0].get("decode_stage")
    return merged


def merge_subshard_results(parts: list[dict[str, Any] | None], *, decoder: str) -> dict[str, Any] | None:
    """Fold the sub-range cells of ONE shard back into the single shard cell
    the rest of the pipeline expects (shard splitting, 2026-09-05).

    A shard's ``shots``-shot stream may be decoded as ``len(parts)``
    contiguous ``shot_range`` jobs on different workers; every shot's
    syndrome and per-shot seed are the same as in the unsplit shard (see
    ``estimate_circuit_logical_error_rate``), so summing the parts'
    bookkeeping reproduces the unsplit shard cell exactly: shots, failures,
    runtime (CPU-ish, as before), stage counts, declared failures. A shard
    keeps its all-or-nothing semantics: if any part is missing (None: the job
    never returned) or non-ok, the shard cell carries THAT status (a pure
    watchdog timeout in one part is still ``timeout``, so decode_infeasible
    is decided exactly as for an unsplit shard); ``shots``/``failures`` of
    the ok parts are still summed for the diagnostics. ``len(parts) == 1``
    returns the part unchanged (no split).
    """

    if len(parts) == 1:
        return parts[0]
    if all(part is None for part in parts):
        return None
    rows = [part for part in parts if isinstance(part, dict)]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    bad_rows = [row for row in rows if row.get("status") != "ok"]
    missing = len(parts) - len(rows)
    shots = sum(int(row.get("shots") or 0) for row in ok_rows)
    failures = sum(int(row.get("failures") or 0) for row in ok_rows)
    runtime = sum(float(row.get("runtime_s") or 0.0) for row in rows)
    first = ok_rows[0] if ok_rows else rows[0]
    merged: dict[str, Any] = {
        key: value
        for key, value in first.items()
        if key not in ("shots", "failures", "logical_error_rate", "runtime_s", "failure_reason",
                       "status", "metrics", "job_execution", "shot_range")
    }
    merged["decoder"] = first.get("decoder") or decoder
    merged["shots"] = shots
    merged["failures"] = failures
    merged["logical_error_rate"] = (failures / shots) if shots else None
    merged["runtime_s"] = runtime
    merged["parts"] = len(parts)
    merged["part_shot_ranges"] = [
        (list(part.get("shot_range")) if isinstance(part, dict) and part.get("shot_range") is not None else None)
        for part in parts
    ]
    merged["part_runtime_s"] = [
        (float(part["runtime_s"]) if isinstance(part, dict) and part.get("runtime_s") is not None else None)
        for part in parts
    ]
    merged["part_statuses"] = [
        (part.get("status") if isinstance(part, dict) else "timeout") for part in parts
    ]
    if bad_rows or missing:
        # the first failing part names the shard's status, as an unsplit
        # shard's single job would have
        failing = next((part for part in parts if not (isinstance(part, dict) and part.get("status") == "ok")), None)
        merged["status"] = failing.get("status") if isinstance(failing, dict) else "timeout"
        reasons = [
            str(row.get("failure_reason") or row.get("status")) for row in bad_rows
        ] + (["shard part never returned"] * missing)
        merged["failure_reason"] = join_failure_reasons(reasons)
        if isinstance(failing, dict) and failing.get("job_execution") is not None:
            merged["job_execution"] = failing.get("job_execution")
    else:
        merged["status"] = "ok"
    if ok_rows:
        metrics = dict(ok_rows[0].get("metrics") or {})
        metrics.pop("shot_range", None)
        metrics["requested_shots"] = sum(
            int((row.get("metrics") or {}).get("requested_shots") or row.get("shots") or 0) for row in ok_rows
        )
        if not bad_rows and not missing and any(
            (row.get("metrics") or {}).get("shard_shots") is not None for row in ok_rows
        ):
            metrics["requested_shots"] = int(
                next((row.get("metrics") or {}).get("shard_shots") for row in ok_rows
                     if (row.get("metrics") or {}).get("shard_shots") is not None)
            )
        metrics.pop("shard_shots", None)
        metrics["actual_shots"] = shots
        metrics["sampling_complete"] = bool(
            not bad_rows and not missing and all((row.get("metrics") or {}).get("sampling_complete", True) for row in ok_rows)
        )
        metrics["stopped_by_max_errors"] = any(
            (row.get("metrics") or {}).get("stopped_by_max_errors") for row in ok_rows
        )
        for key in ("stage_counts", "stage_failures", "stage_time_s"):
            totals: dict[str, float] = {}
            present = False
            for row in ok_rows:
                stages = (row.get("metrics") or {}).get(key)
                if not isinstance(stages, dict):
                    continue
                present = True
                for name, value in stages.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[str(name)] = totals.get(str(name), 0) + value
            if present:
                metrics[key] = {
                    name: (round(float(v), 4) if key == "stage_time_s" else int(v))
                    for name, v in totals.items()
                }
        declared = [
            (row.get("metrics") or {}).get("declared_failures")
            for row in ok_rows
            if isinstance((row.get("metrics") or {}).get("declared_failures"), (int, float))
            and not isinstance((row.get("metrics") or {}).get("declared_failures"), bool)
        ]
        if declared:
            metrics["declared_failures"] = int(sum(declared))
        merged["metrics"] = metrics
    executions = [row.get("job_execution") for row in rows if isinstance(row.get("job_execution"), dict)]
    if executions and "job_execution" not in merged:
        execution = dict(executions[0])
        rss = [int(e["ru_maxrss"]) for e in executions if e.get("ru_maxrss") is not None]
        attempts = [int(e["attempts"]) for e in executions if e.get("attempts") is not None]
        if rss:
            execution["ru_maxrss"] = max(rss)
        if attempts:
            execution["attempts"] = max(attempts)
        execution["parts"] = len(parts)
        merged["job_execution"] = execution
    return merged


def shard_seed(base_seed: int, shard: int, shard_count: int) -> int:
    """shards=1 reproduces the legacy stream exactly; shards>1 derive
    collision-free per-shard seeds via SeedSequence."""

    if shard_count <= 1:
        return int(base_seed)
    import numpy as np

    return int(np.random.SeedSequence([int(base_seed), int(shard)]).generate_state(1)[0])


def shard_partition(total: int, shards: int) -> list[int]:
    shards = max(1, int(shards))
    base, remainder = divmod(max(0, int(total)), shards)
    return [base + (1 if index < remainder else 0) for index in range(shards)]
