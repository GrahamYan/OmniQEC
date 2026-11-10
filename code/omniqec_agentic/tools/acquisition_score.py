"""Read-scope tool (03 §4.2): deterministic baseline for the RUNTIME slow slot.

acquisition_v2_runtime_slot: the archive ordering key
(ranking.archive_order_key) reduced to a scalar report — measured candidates
by plpr tier, unmeasured by Q_eff. Versioned: changing the formula, or its
scope, bumps the id.

v3.2 scope change: this orders the runtime slow slots (v3.7: four; 2026-09-10:
two; was one). The other two slots come from the structural (k, d_search,
-weight) reducer's endpoints and are outside this tool's authority entirely. It also no longer reports a degraded candidate as
unpromotable: hook degradation is a diagnostic label, not a quarantine."""

from __future__ import annotations

from typing import Any

from omniqec_agentic import ranking
from omniqec_agentic.orchestrator.registry import tool

from .context import ToolContext

# v4 (2026-09-10): the unmeasured key is Q_eff = k*d_eff^2 (ranking kd2_v1),
# no longer k*d_eff^2/n.
# v5 (2026-09-12, telescoping_structural_v3_7): unmeasured candidates order by
# the campaign's measured-evidence expected plpr (memory.expected_plpr:
# same check weight and d_search, shrunk toward the class and global means),
# Q_eff only as the cold-start tie-break; four runtime slots (was two).
ACQUISITION_VERSION = "acquisition_v5_runtime_slot_expected_plpr"


@tool("read", example='acquisition_score(code_hashes=["a41f00b2", "9c00d1ee"])')
def acquisition_score(ctx: ToolContext, code_hashes: list[str]) -> dict[str, Any]:
    """Deterministic baseline order for the FOUR runtime slow slots (code_hash
    prefixes). baseline_order is the full archive queue minus candidates that
    already carry a measurement (fixed shots: nothing is re-measured) and
    minus schedule_catastrophic ones; it is exactly the list the plan
    validator checks. Promoting baseline_head needs no rationale; naming any
    other candidate needs a per-item cited rationale in the plan. scores
    report the requested hashes only (measured ones are flagged, not
    ranked into the baseline). This tool has no authority over the two
    structural endpoint slots, which the (k, d_search, -weight) reducer fills;
    degraded_* candidates are scored normally."""

    by_hash: dict[str, dict[str, Any]] = {}
    for row in ctx.archive_rows:
        full = str(row.get("code_hash") or "")
        if full:
            by_hash.setdefault(full, row)
    slow_by_hash: dict[str, dict[str, Any]] = {}
    for row in ctx.slow_rows:
        full = str(row.get("code_hash") or "")
        if full and (full not in slow_by_hash or (
            row.get("per_logical_per_round") is not None
            and float(row["per_logical_per_round"])
            < float(slow_by_hash[full].get("per_logical_per_round") or float("inf"))
        )):
            slow_by_hash[full] = row

    scored: list[dict[str, Any]] = []
    for prefix in code_hashes:
        prefix = str(prefix)
        matches = [h for h in set(by_hash) | set(slow_by_hash) if h.startswith(prefix)]
        if len(matches) != 1:
            scored.append({"code_hash": prefix, "error": f"{len(matches)} matches"})
            continue
        full = matches[0]
        fast = by_hash.get(full) or {}
        slow = slow_by_hash.get(full) or {}
        # Only the unmeasurable case is excluded (v3.2 §4.4): degraded_* is a
        # diagnostic label and its cost is already inside the measured plpr.
        unmeasurable = bool(slow.get("schedule_catastrophic"))
        key = ranking.archive_order_key(
            n=fast.get("n") or slow.get("n"),
            k=fast.get("k") or slow.get("k"),
            d_proxy=fast.get("d_proxy") or slow.get("d_proxy"),
            slow_evidence=slow or None,
            # v3.7: the archive row carries the expected-plpr prediction the
            # queue was ordered by (None before the first measurement).
            expected_plpr=fast.get("expected_plpr"),
        )
        scored.append(
            {
                "code_hash": full[:12],
                "unmeasurable": unmeasurable,
                "degraded": bool(ranking.is_degraded(slow)),
                "measured": slow.get("per_logical_per_round") is not None,
                "per_logical_per_round": slow.get("per_logical_per_round"),
                "fast_Q": fast.get("Q") or slow.get("fast_Q"),
                "expected_plpr": fast.get("expected_plpr"),
                "max_check_weight": fast.get("max_check_weight") or slow.get("max_check_weight"),
                "order_key": list(key),
            }
        )
    # The baseline is NOT the requested subset re-sorted: it is the same list
    # the plan validator holds (ranking.runtime_slot_baseline), reported in
    # full so the head the validator expects is always visible.
    baseline = ranking.runtime_slot_baseline(ctx.archive_rows, ctx.slow_rows)
    return {
        "version": ACQUISITION_VERSION,
        "scope": "runtime_slow_slot_only",
        "scores": scored,
        "baseline_order": [h[:12] for h in baseline],
        "baseline_head": baseline[0][:12] if baseline else None,
        "excluded_from_baseline": {
            "measured": sorted(s["code_hash"] for s in scored if "error" not in s and s["measured"]),
            "unmeasurable": sorted(s["code_hash"] for s in scored if "error" not in s and s["unmeasurable"]),
        },
        "rule": "promote baseline_head without a rationale, or any other candidate with a per-item cited rationale",
    }
