"""Final confirmation of a discovery run, as the paper states it (Appendix D3, Table VII; Fig. 4 c).

    "After the final iteration, the successfully evaluated candidates are ranked using their
    search-stage LER estimates, and the five leading distinct codes from each discovery run are
    selected for final confirmation. These finalists receive additional independent sampling under
    the same circuit and decoder configuration. An initial confirmation measurement uses 25,000
    shots per basis, followed by further shots to reach the budget-dependent cumulative limits in
    Table VII. Existing completed measurements and new shots are combined by adding the shot
    counts and block failures separately for each basis. The resulting LER estimates determine the
    final ranking and the champion of the run."

    "The two BB reference codes are evaluated under the same circuit protocols as the discovered codes
    at their corresponding physical-qubit budgets of N = 144 and N = 288. In both the quantum-memory and
    transversal-CNOT experiments, the reported LERs for the BB [[72,12,6]] and [[144,12,12]] reference
    codes are based on 50,000 and 100,000 shots per basis, respectively."

    Fig. 4 c (Appendix D, additional experiments): "separate searches require the maximum
    stabilizer-check weight of a candidate code to be exactly w_max = 4, 5, 6 or 7. [...] Six candidates
    are evaluated in the final stage for each condition. The figure reports the mean LER and the
    minimum-maximum range of the three best codes under each weight constraint."

Two final stages share this module: ``paper_top5_cumulative_v1`` (every main search, ablation and
baseline) and ``paper_fig4c_top6_confirm_v1`` (a run under an exact check-weight contract: six finalists,
the initial confirmation only, no top-up; the ranking output names the three best). Both measure the
applicable reference code with the initial confirmation and a top-up to its paper total
(config.REFERENCE_CODE_SHOTS_PER_BASIS: BB72 50 000, BB144 100 000 shots per basis).

This module is framework-free: it builds the frozen protocol block (snapshot key ``final_stage``),
selects the finalists, plans the measurement parts, drives them through a caller-supplied
``measure`` callback (the campaign graph wraps ``evaluation.pool.evaluate_promotion_queue``) and
pools / ranks the results. Every part is an ordinary fixed-shot final-stage queue with its own seed
namespace and its own directory, measured with per-candidate resume, so an interrupted final stage
continues where it stopped.

Readings of the text where it leaves room (docs/paper_alignment_measurement.md):

* "successfully evaluated": a search measurement with status ok (racing early stops included -- they
  carry a valid estimate); decode_infeasible / failed candidates have no search-stage LER and cannot
  be ranked.
* "distinct": canonical (pynauty) identity, falling back to the code hash.
* ranking ties: the smaller upper Wilson bound, then the code hash.
* "cumulative": the confirmation parts only (25 000 initial + top-up); the early-stopped search
  measurement is not pooled.
* "further shots to reach the ... limits": every finalist is topped up to the limit (no adaptive
  target); a budget N absent from Table VII has no top-up.
* a finalist that IS the run's reference code (same code hash) is measured once, as one sequence of
  parts: the initial confirmation, the reference's own top-up parts up to the reference total, then
  extension parts up to the finalist total. The reference row pools the confirmation and the reference
  parts (its paper total); the finalist row pools every part (its Table VII limit) -- or only the
  confirmation when its total is the confirmation (Fig. 4 c, a budget outside Table VII).
"""

from __future__ import annotations

import functools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from omniqec_agentic.config import (
    FINAL_STAGE_FIXED_WEIGHT_ID,
    FINAL_STAGE_FIXED_WEIGHT_REPORTED,
    FINAL_STAGE_FIXED_WEIGHT_TOP,
    FINAL_STAGE_IDS,
    FINAL_STAGE_PAPER_ID,
    FINAL_STAGE_TOP,
    FINAL_TOPUP_PART_MAX_SHOTS,
    FINAL_TOPUP_RNG_NAMESPACE,
    FINAL_TOPUP_SHOTS_PER_SHARD,
    REFERENCE_CODE_SHOTS_PER_BASIS,
    final_cumulative_cap,
)

from .baseline import RESIDENT_BASELINE_BUDGETS, select_resident_baseline

SELECTION_RULE_ID = "search_plpr_top_distinct_v1"
POOLING_RULE_ID = "sum_shots_failures_per_basis_v1"
RANKING_RULE_ID = "pooled_plpr_then_ci_high_v1"
COMPOSITION_SCHEMA = "final_top5_composition_v1"
RANKING_SCHEMA = "final_pooled_ranking_v1"
CAP_SOURCES = ("table_vii", "budget_not_in_table_vii", "override", "none_confirmation_only")
REFERENCE_CAP_SOURCES = ("paper_reference_shots", "override")
# The rule for a finalist that is the run's reference code (module docstring, last reading).
SHARED_REFERENCE_RULE_ID = "reference_prefix_of_finalist_parts_v1"
# Fig. 4 c: the statistic of the reported best codes (the S3 Top-3 rule: arithmetic mean, minimum, maximum).
REPORTED_TOP_STATISTIC = "arithmetic_mean_min_max_v1"
# Top-up group of a finalist that is the reference code, beyond the reference total.
SHARED_GROUP = "shared_finalist"


def is_paper_final_stage(protocol: dict[str, Any] | None) -> bool:
    """The snapshot carries one of this module's final stages (main top-5 cumulative or Fig. 4 c)."""

    block = (protocol or {}).get("final_stage")
    return isinstance(block, dict) and block.get("id") in FINAL_STAGE_IDS


def is_fixed_weight_stage(block: dict[str, Any] | None) -> bool:
    return isinstance(block, dict) and block.get("id") == FINAL_STAGE_FIXED_WEIGHT_ID


def finalist_total(block: dict[str, Any]) -> int:
    """Per-basis total of a finalist row: the cumulative limit, or the confirmation when the stage has none."""

    cap = block.get("cumulative_cap")
    return int(block["confirm_shots"]) if cap is None else int(cap)


# --------------------------------------------------------------------------- reference code
@functools.lru_cache(maxsize=8)
def _spec_hash(spec_json: str) -> str:
    from omniqec_agentic.construction import build_matrices
    from omniqec_agentic.verification.structural import code_hash

    return str(code_hash(*build_matrices(json.loads(spec_json))))


def resident_reference(n_budget: int) -> dict[str, Any] | None:
    """The applicable BB reference code of a budget point (``select_resident_baseline``: BB72 at N = 144,
    BB144 at N >= 288, none elsewhere) with its code hash, its corresponding budget and the per-basis total
    of its reported measurement (config.REFERENCE_CODE_SHOTS_PER_BASIS)."""

    selected = select_resident_baseline(int(n_budget))
    if selected is None:
        return None
    spec, program_id = selected
    return {
        "program_id": str(program_id),
        "spec": dict(spec),
        "code_hash": _spec_hash(json.dumps(spec, sort_keys=True)),
        "budget": int(RESIDENT_BASELINE_BUDGETS[str(program_id)]),
        "shots_per_basis": int(REFERENCE_CODE_SHOTS_PER_BASIS[str(program_id)]),
    }


# --------------------------------------------------------------------------- protocol block
def _reference_block(n_budget: int | None, *, confirm: int, override: int | None) -> dict[str, Any] | None:
    reference = resident_reference(int(n_budget)) if n_budget is not None else None
    if reference is None:
        return None
    if override is not None:
        cap, source = int(override), "override"
    else:
        cap, source = int(reference["shots_per_basis"]), "paper_reference_shots"
    if cap < confirm:
        raise ValueError(f"reference cumulative shots {cap} must be >= {confirm}")
    return {
        "program_id": reference["program_id"],
        "code_hash": reference["code_hash"],
        "budget": reference["budget"],
        "cumulative_cap": int(cap),
        "cumulative_cap_source": source,
        "shared_with_finalist": SHARED_REFERENCE_RULE_ID,
    }


def final_stage_protocol(
    *,
    task: str,
    n_budget: int | None,
    confirm_shots: int,
    final_shot_shards: int,
    cumulative_override: int | None = None,
    exact_check_weight: int | None = None,
) -> dict[str, Any]:
    """The frozen block of the paper's final stage for one run (snapshot key ``final_stage``).

    ``exact_check_weight`` set (a Fig. 4 c run): the fixed-weight stage -- FINAL_STAGE_FIXED_WEIGHT_TOP
    finalists, the initial confirmation only (``cumulative_cap`` None), FINAL_STAGE_FIXED_WEIGHT_REPORTED
    reported codes. Otherwise the main top-5 stage with the Table VII limit of (task, N).
    ``cumulative_override`` (a declared smoke variant) replaces every cumulative total: Table VII and the
    reference's total in the main stage, the reference's total in the fixed-weight stage."""

    task = str(task or "memory")
    confirm = int(confirm_shots)
    if confirm < 1:
        raise ValueError("confirm_shots must be >= 1")
    if cumulative_override is not None and int(cumulative_override) < confirm:
        raise ValueError(
            f"cumulative final shots {int(cumulative_override)} must be >= the initial confirmation's {confirm}"
        )
    final_cumulative_cap(task, n_budget)  # validates the task column
    reference_block = _reference_block(n_budget, confirm=confirm, override=cumulative_override)
    tail = {
        "reference": reference_block,
        "topup_rng_namespace": FINAL_TOPUP_RNG_NAMESPACE,
        "topup_part_max_shots": int(FINAL_TOPUP_PART_MAX_SHOTS[task]),
        "topup_min_shards": int(final_shot_shards),
        "topup_shots_per_shard": int(FINAL_TOPUP_SHOTS_PER_SHARD),
        "pooling": POOLING_RULE_ID,
        "ranking": RANKING_RULE_ID,
    }
    if exact_check_weight is not None:
        return {
            "id": FINAL_STAGE_FIXED_WEIGHT_ID,
            "task": task,
            "n_budget": None if n_budget is None else int(n_budget),
            "exact_check_weight": int(exact_check_weight),
            "top": int(FINAL_STAGE_FIXED_WEIGHT_TOP),
            "reported_top": int(FINAL_STAGE_FIXED_WEIGHT_REPORTED),
            "reported_statistic": REPORTED_TOP_STATISTIC,
            "selection": SELECTION_RULE_ID,
            "confirm_shots": confirm,
            "cumulative_cap": None,
            "cumulative_cap_source": "none_confirmation_only",
            **tail,
        }
    table_cap = final_cumulative_cap(task, n_budget)
    if cumulative_override is not None:
        cap, source = int(cumulative_override), "override"
    elif table_cap is not None:
        cap, source = int(table_cap), "table_vii"
    else:
        cap, source = confirm, "budget_not_in_table_vii"
    if cap < confirm:
        raise ValueError(f"cumulative final shots {cap} must be >= the initial confirmation's {confirm}")
    return {
        "id": FINAL_STAGE_PAPER_ID,
        "task": task,
        "n_budget": None if n_budget is None else int(n_budget),
        "top": int(FINAL_STAGE_TOP),
        "selection": SELECTION_RULE_ID,
        "confirm_shots": confirm,
        "cumulative_cap": int(cap),
        "cumulative_cap_source": source,
        **tail,
    }


# --------------------------------------------------------------------------- selection
def _ub(row: dict[str, Any]) -> float:
    ci = row.get("per_logical_per_round_ci")
    try:
        return float(ci[1])
    except (TypeError, ValueError, IndexError):
        return math.inf


def search_rank_key(row: dict[str, Any]) -> tuple[float, float, str]:
    """Search-stage order: point estimate, then the smaller upper bound, then the code hash."""

    return (float(row["per_logical_per_round"]), _ub(row), str(row.get("code_hash") or ""))


def select_finalists(
    rows: list[dict[str, Any]],
    *,
    top: int = FINAL_STAGE_TOP,
    canonical_of: dict[str, str] | None = None,
    ranked: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ``top`` leading distinct codes of ``rows``.

    ``ranked``: order ``rows`` by their search-stage estimate first (the paper's rule); False keeps the
    caller's order (a no-slow ablation, whose rows carry no measurement). Two rows are the same code when
    they share the code hash or the canonical identity (``canonical_of[code_hash]`` or the row's own
    ``canonical_hash``). Returns (finalists, skipped duplicates)."""

    canonical_of = dict(canonical_of or {})
    if ranked:
        ordered = sorted(
            (row for row in rows if row.get("per_logical_per_round") is not None),
            key=search_rank_key,
        )
    else:
        ordered = list(rows)
    finalists: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_hash: dict[str, str] = {}
    seen_identity: dict[str, str] = {}
    for row in ordered:
        if len(finalists) >= int(top):
            break
        code_hash = str(row.get("code_hash") or row.get("program_id") or "")
        if not code_hash:
            continue
        identity = str(canonical_of.get(code_hash) or row.get("canonical_hash") or code_hash)
        twin = seen_hash.get(code_hash) or seen_identity.get(identity)
        if twin is not None:
            skipped.append({"code_hash": code_hash[:12], "program_id": row.get("program_id"), "duplicate_of": twin[:12]})
            continue
        seen_hash[code_hash] = code_hash
        seen_identity[identity] = code_hash
        finalists.append(row)
    return finalists, skipped


# --------------------------------------------------------------------------- measurement parts
def split_parts(total: int, part_max: int) -> list[int]:
    """``total`` shots per basis as near-equal parts of at most ``part_max`` (the first takes the remainder)."""

    total = int(total)
    if total <= 0:
        return []
    count = math.ceil(total / max(1, int(part_max)))
    base = total // count
    parts = [base] * count
    parts[0] += total - base * count
    return parts


def topup_shards(shots: int, *, min_shards: int, shots_per_shard: int) -> int:
    """Shards of a top-up part: about ``shots_per_shard`` shots each, never fewer than ``min_shards``."""

    want = 2 ** math.ceil(math.log2(max(1.0, int(shots) / float(max(1, int(shots_per_shard))))))
    return int(max(int(min_shards), want))


def topup_parts(
    block: dict[str, Any], *, cumulative_cap: int, group: str, rounds: int, start: int | None = None
) -> list[dict[str, Any]]:
    """The top-up parts that take a code from ``start`` (default: the initial confirmation) to
    ``cumulative_cap`` shots per basis.

    ``group`` names the member set ("finalists", "reference" or SHARED_GROUP): each group has its own seed
    namespaces and directories, so the reference's parts never shift the finalists' seeds."""

    begin = int(block["confirm_shots"]) if start is None else int(start)
    remaining = int(cumulative_cap) - begin
    prefix = "topup" if group == "finalists" else f"topup_{group}"
    namespace = str(block["topup_rng_namespace"]) + ("" if group == "finalists" else f"/{group}")
    out = []
    for index, shots in enumerate(split_parts(remaining, int(block["topup_part_max_shots"])), start=1):
        out.append(
            {
                "name": f"{prefix}_p{index}",
                "kind": "topup",
                "group": group,
                "shots": int(shots),
                "shards": topup_shards(
                    shots,
                    min_shards=int(block["topup_min_shards"]),
                    shots_per_shard=int(block["topup_shots_per_shard"]),
                ),
                "rng_namespace": f"{namespace}/p{index}",
                "dir": f"topup/T{int(rounds)}_{group}_p{index}",
                "queue": f"topup/queue_T{int(rounds)}_{group}_p{index}.jsonl",
            }
        )
    return out


def plan_final_stage(
    block: dict[str, Any], entries: list[dict[str, Any]], *, rounds: int, confirmed: set[str] | frozenset[str]
) -> dict[str, Any]:
    """The top-up parts of a final stage and the parts each pooled row combines, given the program ids whose
    initial confirmation finished with status ok.

    ``entries``: the finalists (one may carry ``also_reference``: the run's reference code itself) and at most
    one separate reference row (``is_baseline``). Groups and totals per basis:

    * plain finalists: the confirmation + ``finalists`` parts up to ``finalist_total(block)`` (Table VII, or
      the confirmation alone in the Fig. 4 c stage);
    * the reference (separate row, or the finalist that is the reference): the confirmation + ``reference``
      parts up to the reference's paper total (``block.reference.cumulative_cap``);
    * the finalist that is the reference: also SHARED_GROUP parts from the reference total up to its finalist
      total. Its finalist row pools every part; its reference row pools the confirmation and the reference
      parts. When its finalist total is the confirmation, the finalist row pools the confirmation only.

    Returns ``{"topups": [(part, members)], "rows": [{"entry", "measured_as", "total", "parts"}]}``; a top-up
    part whose members are all unconfirmed is left out (nothing to measure). ``rows[*].parts`` lists the
    planned part names (the confirmation first); a member takes a top-up part only once confirmed."""

    confirm = int(block["confirm_shots"])
    run_total = finalist_total(block)
    reference_block = block.get("reference") or {}
    finalists = [entry for entry in entries if not entry.get("is_baseline")]
    plain = [entry for entry in finalists if not entry.get("also_reference")]
    shared = [entry for entry in finalists if entry.get("also_reference")]
    separate = [entry for entry in entries if entry.get("is_baseline")]
    if len(shared) + len(separate) > 1:
        raise ValueError("a final stage measures at most one reference row")
    if (shared or separate) and not reference_block:
        raise ValueError("a reference row needs the final stage's reference block")
    ref_total = int(reference_block.get("cumulative_cap") or confirm) if reference_block else confirm

    finalist_parts = topup_parts(block, cumulative_cap=run_total, group="finalists", rounds=rounds)
    reference_parts = (
        topup_parts(block, cumulative_cap=ref_total, group="reference", rounds=rounds) if (shared or separate) else []
    )
    extension_parts: list[dict[str, Any]] = []
    if shared:
        if run_total > ref_total:
            extension_parts = topup_parts(
                block, cumulative_cap=run_total, start=ref_total, group=SHARED_GROUP, rounds=rounds
            )
        elif run_total not in (confirm, ref_total):
            # Table VII exceeds both reference totals wherever a reference applies, and an override sets both
            # totals alike, so the finalist total always lies on the reference's sequence.
            raise ValueError(
                f"finalist total {run_total} lies strictly between the confirmation ({confirm}) and the "
                f"reference total ({ref_total}): no prefix of the shared parts reaches it"
            )

    def _ok(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [entry for entry in members if str(entry["program_id"]) in confirmed]

    topups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for parts, members in ((finalist_parts, plain), (reference_parts, separate + shared), (extension_parts, shared)):
        for part in parts:
            ok_members = _ok(members)
            if ok_members:
                topups.append((part, ok_members))

    def _names(parts: list[dict[str, Any]]) -> list[str]:
        return [str(part["name"]) for part in parts]

    rows: list[dict[str, Any]] = []
    for entry in plain:
        rows.append({"entry": entry, "measured_as": str(entry["program_id"]), "total": run_total,
                     "parts": ["confirm"] + _names(finalist_parts)})
    for entry in shared:
        pid = str(entry["program_id"])
        finalist_names = ["confirm"] + (_names(reference_parts) + _names(extension_parts) if run_total >= ref_total else [])
        rows.append({"entry": entry, "measured_as": pid, "total": run_total, "parts": finalist_names})
        reference_entry = {
            "program_id": str(reference_block["program_id"]),
            "search_program_id": str(reference_block["program_id"]),
            "code_hash": entry.get("code_hash"),
            "canonical_hash": entry.get("canonical_hash"),
            "family": entry.get("family"),
            "n": entry.get("n"),
            "k": entry.get("k"),
            "d_search": entry.get("d_search"),
            "is_baseline": True,
            "selection_channels": ["baseline"],
            "shared_with_finalist": True,
            "measured_as": pid,
        }
        rows.append({"entry": reference_entry, "measured_as": pid, "total": ref_total,
                     "parts": ["confirm"] + _names(reference_parts)})
    for entry in separate:
        rows.append({"entry": entry, "measured_as": str(entry["program_id"]), "total": ref_total,
                     "parts": ["confirm"] + _names(reference_parts)})
    return {"topups": topups, "rows": rows}


# --------------------------------------------------------------------------- pooling / ranking
def _rows_by_program(rankings_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not rankings_path or not Path(rankings_path).is_file():
        return {}
    payload = json.loads(Path(rankings_path).read_text(encoding="utf-8"))
    rows = payload.get("rankings") if isinstance(payload, dict) else payload
    out: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("program_id") is not None:
            out.setdefault(str(row["program_id"]), row)
    return out


def pool_entry(
    entry: dict[str, Any],
    part_rows: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    logicals: int,
    rounds: int,
    ci_z: float,
    cumulative_cap: int,
) -> dict[str, Any]:
    """One finalist's (or the reference's) pooled record: the ok parts' shots and failures added per basis
    and pushed through Eqs. (D1)/(D2) (``verification.ler.combine_basis_ler_results``)."""

    from omniqec_agentic.verification.ler import combine_basis_ler_results

    shots_x = shots_z = failures_x = failures_z = 0
    parts: list[dict[str, Any]] = []
    first_ok: dict[str, Any] | None = None
    for part, row in part_rows:
        status = (row or {}).get("status") or "missing"
        record = {"part": part["name"], "planned_shots_per_basis": int(part["shots"]), "status": status}
        if status == "ok":
            first_ok = first_ok or row
            sx, sz = int(row.get("shots_x") or 0), int(row.get("shots_z") or 0)
            fx, fz = int(row.get("failures_x") or 0), int(row.get("failures_z") or 0)
            shots_x += sx
            shots_z += sz
            failures_x += fx
            failures_z += fz
            record.update(shots_x=sx, shots_z=sz, failures_x=fx, failures_z=fz)
        parts.append(record)
    confirm_ok = bool(part_rows) and ((part_rows[0][1] or {}).get("status") == "ok")
    out: dict[str, Any] = {
        "program_id": entry.get("program_id"),
        "search_program_id": entry.get("search_program_id") or entry.get("program_id"),
        "code_hash": entry.get("code_hash"),
        "canonical_hash": entry.get("canonical_hash"),
        "is_baseline": bool(entry.get("is_baseline")),
        "also_reference": bool(entry.get("also_reference")),
        "shared_with_finalist": bool(entry.get("shared_with_finalist")),
        # the program id the parts measured this row under (a reference row shared with a finalist: the finalist's)
        "measured_as": str(entry.get("measured_as") or entry.get("program_id")),
        "selection_channels": list(entry.get("selection_channels") or []),
        "search_rank": entry.get("search_rank"),
        "search_per_logical_per_round": entry.get("search_per_logical_per_round"),
        "search_per_logical_per_round_ci": entry.get("search_per_logical_per_round_ci"),
        "family": (first_ok or {}).get("family") or entry.get("family"),
        "n": (first_ok or {}).get("n") or entry.get("n"),
        "k": (first_ok or {}).get("k") or entry.get("k"),
        "d_search": (first_ok or {}).get("d_search") or entry.get("d_search"),
        "d_proxy": (first_ok or {}).get("d_proxy") or entry.get("d_proxy"),
        "max_check_weight": (first_ok or {}).get("max_check_weight") or entry.get("max_check_weight"),
        "logicals": int(logicals),
        "rounds": int((first_ok or {}).get("rounds") or rounds),
        "decoder": (first_ok or {}).get("decoder"),
        "decoder_config": (first_ok or {}).get("decoder_config"),
        "noise_value": (first_ok or {}).get("noise_value"),
        "parts": parts,
        "confirmation_status": (part_rows[0][1] or {}).get("status") if part_rows else "missing",
        "cumulative_cap": int(cumulative_cap),
        "shots_x": shots_x,
        "shots_z": shots_z,
        "failures_x": failures_x,
        "failures_z": failures_z,
        "total_shots": shots_x + shots_z,
        "total_failures": failures_x + failures_z,
        "status": "ok" if confirm_ok else "unconfirmed",
    }
    out["cumulative_complete"] = bool(confirm_ok and shots_x == shots_z == int(cumulative_cap))
    if confirm_ok and shots_x > 0 and shots_z > 0:
        combined = combine_basis_ler_results(
            {
                "X": {"logical_error_rate": failures_x / shots_x, "failures": failures_x, "shots": shots_x},
                "Z": {"logical_error_rate": failures_z / shots_z, "failures": failures_z, "shots": shots_z},
            },
            rounds=int(out["rounds"]),
            k=int(logicals),
            ci_z=float(ci_z),
        )
        for key in (
            "block_ler",
            "per_logical_per_round",
            "per_logical_per_round_ci",
            "per_logical_block_ci",
            "basis_logical_error_rate",
            "basis_ler_ci",
            "ci_z",
        ):
            out[key] = combined.get(key)
    else:
        out["per_logical_per_round"] = None
        out["per_logical_per_round_ci"] = None
    return out


def rank_pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Final ranking: the confirmed non-reference rows by pooled pLER (ties: the smaller upper bound, then
    the search rank); rank 1 is the champion. Reference rows and unconfirmed rows keep final_rank None."""

    ranked = sorted(
        (r for r in rows if not r.get("is_baseline") and r.get("per_logical_per_round") is not None),
        key=lambda r: (float(r["per_logical_per_round"]), _ub(r), int(r.get("search_rank") or 10**6)),
    )
    order = {id(row): index for index, row in enumerate(ranked, start=1)}
    for row in rows:
        row["final_rank"] = order.get(id(row))
        row["champion"] = order.get(id(row)) == 1
    return sorted(
        rows,
        key=lambda r: (
            r["final_rank"] is None,
            r["final_rank"] or 0,
            r.get("per_logical_per_round") is None,
            float(r.get("per_logical_per_round") or 0.0),
        ),
    )


def reported_top(block: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fig. 4 c: the ``block.reported_top`` best ranked finalists of a fixed-weight stage with the arithmetic
    mean, minimum and maximum of their pLER (the figure's mean and minimum-maximum range); None for a stage
    without the key."""

    count = block.get("reported_top")
    if not count:
        return None
    ranked = sorted((row for row in rows if row.get("final_rank") is not None), key=lambda row: int(row["final_rank"]))
    best = ranked[: int(count)]
    values = [float(row["per_logical_per_round"]) for row in best]
    return {
        "statistic": block.get("reported_statistic") or REPORTED_TOP_STATISTIC,
        "required": int(count),
        "count": len(best),
        "complete": len(best) == int(count),
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "final_rank", "program_id", "search_program_id", "code_hash", "family", "n", "k", "d_search",
                    "max_check_weight", "shots_x", "shots_z", "failures_x", "failures_z", "per_logical_per_round",
                    "per_logical_per_round_ci",
                )
            }
            for row in best
        ],
        "mean_per_logical_per_round": statistics.fmean(values) if values else None,
        "min_per_logical_per_round": min(values) if values else None,
        "max_per_logical_per_round": max(values) if values else None,
    }


# --------------------------------------------------------------------------- driver
Measure = Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def execute_final_stage(
    *,
    block: dict[str, Any],
    entries: list[dict[str, Any]],
    rounds: int,
    confirm_namespace: str,
    confirm_shards: int,
    measure: Measure,
    ci_z: float,
) -> dict[str, Any]:
    """Run the initial confirmation of every entry, then the top-up parts of ``plan_final_stage`` (the
    finalists' parts, the reference's own parts, the shared finalist's extension), then pool and rank.

    ``entries``: queue records of the finalists (search order) followed by the reference row (``is_baseline``)
    when the run has one that is not itself a finalist. ``measure(part, records)`` measures one part and
    returns the ``evaluate_promotion_queue`` summary. A part whose queue status is not ok stops the stage
    (status ``partial``) so the caller can resume it."""

    task = str(block.get("task") or "memory")
    confirm_part = {
        "name": "confirm",
        "kind": "confirm",
        "group": "all",
        "shots": int(block["confirm_shots"]),
        "shards": int(confirm_shards),
        "rng_namespace": str(confirm_namespace),
        "dir": f"T{int(rounds)}",
        "queue": f"queue_T{int(rounds)}.jsonl",
    }
    parts_run: list[dict[str, Any]] = []

    def _run(part: dict[str, Any], members: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not members:
            return None
        summary = measure(part, members)
        record = {
            **part,
            "members": [str(m["program_id"]) for m in members],
            "status": summary.get("status"),
            "artifact_paths": summary.get("artifact_paths") or {},
        }
        parts_run.append(record)
        return record

    def _payload(status: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"status": status, "parts": parts_run, "rows": rows or []}

    confirm = _run(confirm_part, entries)
    if confirm is None:
        return _payload("empty_shortlist")
    if confirm["status"] != "ok":
        return _payload("partial")
    confirm_rows = _rows_by_program(confirm["artifact_paths"].get("rankings"))
    confirmed = {
        str(entry["program_id"])
        for entry in entries
        if (confirm_rows.get(str(entry["program_id"])) or {}).get("status") == "ok"
    }
    plan = plan_final_stage(block, entries, rounds=rounds, confirmed=confirmed)
    records: dict[str, dict[str, Any]] = {"confirm": confirm}
    tables: dict[str, dict[str, dict[str, Any]]] = {"confirm": confirm_rows}
    for part, members in plan["topups"]:
        record = _run(part, members)
        if record is None:
            continue
        if record["status"] != "ok":
            return _payload("partial")
        records[str(part["name"])] = record
        tables[str(part["name"])] = _rows_by_program(record["artifact_paths"].get("rankings"))

    rows: list[dict[str, Any]] = []
    for item in plan["rows"]:
        entry, pid = item["entry"], str(item["measured_as"])
        part_rows = []
        for name in item["parts"]:
            record = records.get(name)
            if record is None or (name != "confirm" and pid not in record.get("members", [])):
                continue
            part_rows.append((record, tables[name].get(pid)))
        k = int(entry.get("k") or (confirm_rows.get(pid) or {}).get("k") or 0)
        logicals = 2 * k if task == "cnot_dense" else k
        rows.append(
            pool_entry(
                entry,
                part_rows,
                logicals=max(1, logicals),
                rounds=rounds,
                ci_z=ci_z,
                cumulative_cap=int(item["total"]),
            )
        )
    return _payload("ok", rank_pooled(rows))
