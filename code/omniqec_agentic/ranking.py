"""Unified ranking semantics for the fast archive and the slow evidence.

This module is the single authority for how candidates are ordered anywhere in
the package. The design (agreed reconcile route) keeps the multi-evidence
artifacts un-collapsed, but makes ordering decisions through one lexicographic
rule:

  1. slow-verified candidates rank by measured `per_logical_per_round`
     (lower is better). In the slow rankings (`rank_slow_rows`) candidates
     whose Wilson confidence intervals overlap are statistically tied and
     fall through to the next key; the fast-archive key
     (`archive_order_key`) uses the point estimate directly — archive cells
     rarely hold two measured codes, so CI grouping is not implemented there;
  2. unverified candidates (and tied verified ones) rank by `Q_eff`, the
     quality score `Q = k * d_eff^2` (QUALITY_METRIC_ID; 2026-09-10: the
     ver1 `/n` factor was dropped, so a code is no longer credited for
     being small) recomputed with the *effective* distance
     `d_eff = min(d_proxy, verified circuit-distance evidence)`, optionally
     capped at the verifiable regime (`d_cap`, ver1's D_CERT_MAX idea);
  3. (retired 03 D12) the hook-surrogate tie-break; (Δmin, τA) columns take
     remaining ties; then d_eff, k and -n as final tiebreakers. The check
     weight is NOT a tie-break here (2026-09-29 correction: the text used to
     promise `(w, q, n)`, which `archive_order_key` never implemented) -- it
     enters the runtime order only through `expected_plpr_v1`, and the
     structural order through the third front axis.

Hard gates that live next to the ordering rule:
  - trust filter: screens RUNTIME/acquisition candidates whose
    d_search / sqrt(n) is below a floor (ver1 tier1 `passes_trust_filter`).
    It does not decide structural retention;
  - degraded quarantine (v3.6 M2, docs/06 §3): ONLY `schedule_catastrophic`
    quarantines (no schedule avoids a single-fault logical, so no
    measurement exists). The `degraded_*` flags (estimate / isd / ext) are
    audit information — `is_flagged_degraded` — and never demote a measured
    code below an unmeasured one, never evict it from the hall of fame or the
    parent pool. BB144 itself is degraded_ext under the two-basis SEC
    (d_ext 10/9 < 12), which is what the old rule quarantined.

v3.2 adds a SECOND, independent ledger in the lower half of this module: the
(k, d_search) proxy-Pareto reducer. Structural retention answers "is this
code structurally strong" and reads only k and d_search; the ordering rules
above answer "how did it run under the frozen decoder". Neither may overwrite
the other.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

DEFAULT_D_CAP = 12
DEFAULT_DSQRTN_FLOOR = 0.25
# Promotion floor on d_search. 2026-08-16: d < 3 codes cannot CORRECT any
# error (d=1 undetectable, d=2 detect-only). 2026-09-10 (user decision): the
# floor is 5. Across the v3.6 b60/b98/b140 runs of all three arms, d_search
# <= 4 codes took 34-60% of the slow measurements and never came within 1.5x
# of the point's best plpr at p = 0.5%; every point's best code had d_search
# >= 5. Codes below the floor stay in the archive for the record but are
# never promoted to the slow loop and never enter the hall of fame. The value
# travels in the protocol snapshot (rounds_policy.min_promotion_distance).
MIN_PROMOTION_DISTANCE = 5
DEFAULT_CI_Z = 1.96
_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def wilson_interval(failures: int, shots: int, *, z: float = DEFAULT_CI_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial failure rate."""

    failures = max(0, int(failures))
    shots = int(shots)
    if shots <= 0:
        return (0.0, 1.0)
    p_hat = failures / shots
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / shots
    center = (p_hat + z2 / (2.0 * shots)) / denom
    half = (float(z) / denom) * math.sqrt(p_hat * (1.0 - p_hat) / shots + z2 / (4.0 * shots * shots))
    return (max(0.0, center - half), min(1.0, center + half))


def intervals_overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    """True when two intervals overlap; missing intervals never overlap."""

    if not a or not b:
        return False
    return float(a[0]) <= float(b[1]) and float(b[0]) <= float(a[1])


# ---------------------------------------------------------------------------
# Effective distance / quality
# ---------------------------------------------------------------------------
def verified_circuit_distance(evidence: dict[str, Any] | None) -> tuple[int | None, str]:
    """Most trustworthy verified circuit-distance value in a slow-evidence dict.

    Priority follows ver1 `fitness.effective_distance`: exact circuit distance,
    then the LER-slope estimate, then the DEM-ISD upper bound (a constructive
    find of an undetectable logical error, so `d_circ <= bound` is a proof and
    legitimately lowers d_eff). Returns (value, source)."""

    if not isinstance(evidence, dict):
        return None, "none"
    exact = evidence.get("exact_d_circ")
    if exact is not None and evidence.get("exact_d_circ_status") == "ok":
        return int(exact), "exact_d_circ"
    estimate = evidence.get("d_circ_ler_estimate")
    if estimate is not None and evidence.get("d_circ_ler_estimate_status") == "ok":
        return int(estimate), "d_circ_ler_estimate"
    bounds: list[tuple[int, str]] = []
    isd_upper = evidence.get("d_circ_isd_upper_bound")
    if isd_upper is not None and evidence.get("d_circ_isd_upper_bound_status") in {"ok", "partial"}:
        bounds.append((int(isd_upper), "d_circ_isd_upper_bound"))
    # d_circ_ext (02 §5): kernel-ISD probe on the extended matrices; SBB Thm 1
    # makes it a circuit-distance upper bound with the same constructive
    # semantics as the DEM-ISD find.
    ext = evidence.get("d_circ_ext")
    if ext is not None and evidence.get("d_circ_ext_status") in {"ok", "probe_upper_bound"}:
        bounds.append((int(ext), "d_circ_ext"))
    if bounds:
        return min(bounds, key=lambda item: item[0])
    return None, "none"


def effective_distance(
    d_proxy: int | None,
    slow_evidence: dict[str, Any] | None = None,
    *,
    d_cap_enabled: bool = True,
    d_cap: int = DEFAULT_D_CAP,
) -> tuple[int | None, str]:
    """d_eff = min(d_proxy, verified circuit distance), capped at d_cap.

    The cap concedes the unverifiable high-d regime (ver1 D_CERT_MAX): distance
    beyond the cap earns no extra reward because only the proxy can claim it."""

    if d_proxy is None:
        return None, "none"
    d_eff = int(d_proxy)
    source = "proxy"
    verified, verified_source = verified_circuit_distance(slow_evidence)
    if verified is not None and verified < d_eff:
        d_eff = int(verified)
        source = verified_source
    if d_cap_enabled and d_eff > int(d_cap):
        d_eff = int(d_cap)
        source = f"{source}+capped"
    return d_eff, source


# Identity of the fast quality score. Frozen into the protocol snapshot
# (workflow.state.build_protocol_snapshot) and checked by the formal validator:
# a run archived under the ver1 score `k*d^2/n` (no key) does not resume or
# validate under this one.
#   kd2_v1 (2026-09-10): Q = k * d^2. The ver1 `/n` made Q scale-free, so a
#   half-budget code read exactly as well as a full-size one in every ranking
#   (archive niche competition, promotion order, parent selection).
QUALITY_METRIC_ID = "kd2_v1"


def q_eff(n: int | None, k: int | None, d_eff_value: int | None) -> float | None:
    """Q = k * d_eff^2 (QUALITY_METRIC_ID). `n` is kept in the signature for
    the callers' sake and is no longer part of the score."""

    if k is None or d_eff_value is None:
        return None
    return float(int(k) * int(d_eff_value) * int(d_eff_value))


def trust_filter_ok(
    d_proxy: int | None,
    n: int | None,
    *,
    floor: float = DEFAULT_DSQRTN_FLOOR,
) -> bool:
    """ver1 tier1 trust filter: reject suspiciously high-k low-distance codes."""

    if floor is None or float(floor) <= 0:
        return True
    if d_proxy is None or n is None or int(n) <= 0:
        return True
    return (int(d_proxy) / math.sqrt(int(n))) >= float(floor)


DEGRADED_FLAG_KEYS = (
    "degraded_estimate",
    "degraded_estimate_any_decoder",
    "degraded_isd",
    # d_circ_ext channel (design doc 02 §5): extended-code distance below the
    # proxy claim. Audit flag only since v3.6 (probe_upper_bound values are
    # not certificates, and the multi-seed certification of M1 may still
    # under-shoot d_search on large kernels).
    "degraded_ext",
)


def is_degraded(slow_evidence: dict[str, Any] | None) -> bool:
    """HARD quarantine (v3.6 M2): True ONLY for `schedule_catastrophic`.

    schedule_catastrophic (02 §3.3): no schedule among the P candidates
    avoids a single-fault logical — never a parent, never measured. Every
    other degraded_* flag is reported by `is_flagged_degraded` and does not
    change tiers, hall-of-fame membership or parent eligibility."""

    if not isinstance(slow_evidence, dict):
        return False
    return bool(slow_evidence.get("schedule_catastrophic"))


def is_flagged_degraded(slow_evidence: dict[str, Any] | None) -> bool:
    """Any degraded_* audit flag (or the hard quarantine). Reporting only."""

    if not isinstance(slow_evidence, dict):
        return False
    if is_degraded(slow_evidence):
        return True
    return any(bool(slow_evidence.get(key)) for key in DEGRADED_FLAG_KEYS)


def degraded_flags(slow_evidence: dict[str, Any] | None) -> list[str]:
    """Names of the raised degraded_* flags, for audit rows."""

    if not isinstance(slow_evidence, dict):
        return []
    flags = [key for key in DEGRADED_FLAG_KEYS if slow_evidence.get(key)]
    if slow_evidence.get("schedule_catastrophic"):
        flags.append("schedule_catastrophic")
    return flags


# ---------------------------------------------------------------------------
# Lexicographic ordering keys
# ---------------------------------------------------------------------------
def slow_ler_value(slow_evidence: dict[str, Any] | None) -> float | None:
    if not isinstance(slow_evidence, dict):
        return None
    value = slow_evidence.get("per_logical_per_round")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def archive_order_key(
    *,
    n: int | None,
    k: int | None,
    d_proxy: int | None,
    slow_evidence: dict[str, Any] | None = None,
    d_cap_enabled: bool = True,
    d_cap: int = DEFAULT_D_CAP,
    expected_plpr: float | None = None,
) -> tuple:
    """Sort key (descending sort) for archive elites, parents, and queues.

    Tier 1: slow-verified candidates, ordered by measured LER per logical per
    round. Tier 0: everything else, ordered by the EXPECTED plpr under the
    campaign's measured-evidence model (memory.expected_plpr, v3.7) when one
    exists, then by Q_eff. With no measurements yet every expected value is
    None and the order is the historical Q_eff order. Check weight enters
    only through that model (what the LER measured for that class), never
    as a rule."""

    d_eff_value, _ = effective_distance(
        d_proxy,
        slow_evidence,
        d_cap_enabled=d_cap_enabled,
        d_cap=d_cap,
    )
    quality = q_eff(n, k, d_eff_value)
    ler = slow_ler_value(slow_evidence)
    verified_tier = 1 if (ler is not None and not is_degraded(slow_evidence)) else 0
    try:
        expected_term = (
            -math.log10(float(expected_plpr))
            if expected_plpr is not None and float(expected_plpr) > 0.0
            else 0.0
        )
    except (TypeError, ValueError):
        expected_term = 0.0
    return (
        verified_tier,
        -ler if ler is not None else 0.0,
        expected_term,
        quality if quality is not None else _NEG_INF,
        int(d_eff_value or 0),
        int(k or 0),
        -int(n if n is not None else 10**6),
    )


def runtime_slot_baseline(
    archive_rows: list[dict[str, Any]],
    slow_rows: list[dict[str, Any]],
) -> list[str]:
    """The ONE deterministic baseline for the runtime slow slot (full hashes).

    The fast-archive queue in its own order, minus every candidate that
    already carries a measurement (a fixed-shot protocol never re-measures)
    and minus schedule_catastrophic ones (no sampled schedule avoids a
    single-fault logical, so they cannot be measured at all).

    This is the list the plan validator checks a named promotion against AND
    the list the acquisition_score tool reports to the orchestrator. They used
    to be computed separately: the tool ranked measured candidates first
    (archive_order_key tier 1) while the validator excluded them, so a model
    that promoted the tool's head was told it "departed from the baseline",
    could not see which candidate the validator wanted, and resubmitted the
    same plan until the repair budget died (GPT arm b288 turn 4, 2026-09-04).
    """

    measured = {
        str(r.get("code_hash"))
        for r in slow_rows
        if r.get("code_hash") and r.get("per_logical_per_round") is not None
    }
    quarantined = {
        str(r.get("code_hash")) for r in slow_rows if r.get("schedule_catastrophic")
    }
    baseline: list[str] = []
    for row in archive_rows:
        full = str(row.get("code_hash") or "")
        if full and full not in measured and full not in quarantined and full not in baseline:
            baseline.append(full)
    return baseline


# ---------------------------------------------------------------------------
# CI-aware slow ranking
# ---------------------------------------------------------------------------
def rank_slow_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order slow ranking rows with statistical-tie awareness.

    Rows with an ok status and a `per_logical_per_round` value sort first by
    that point estimate. Consecutive rows whose per-logical block confidence
    intervals overlap the current group leader form a statistical group and
    reorder internally by fast_Q (desc). Rows
    without a usable LER keep their relative order after all measured rows,
    sorted by fast_Q.

    Each returned row is annotated in place with `statistical_group` (int) and
    `statistically_tied` (bool)."""

    measured: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("per_logical_per_round")
        if row.get("status") == "ok" and value is not None:
            measured.append(row)
        else:
            unmeasured.append(row)

    measured.sort(key=lambda row: float(row["per_logical_per_round"]))

    groups: list[list[dict[str, Any]]] = []
    leader_ci: tuple[float, float] | None = None
    for row in measured:
        ci = _row_ci(row)
        if groups and intervals_overlap(leader_ci, ci):
            groups[-1].append(row)
        else:
            groups.append([row])
            leader_ci = ci
    ordered: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        tied = len(group) > 1
        group.sort(
            key=lambda row: (
                -_float_or(row.get("fast_Q"), _NEG_INF),
                float(row["per_logical_per_round"]),
            )
            if tied
            else (0.0,)
        )
        for row in group:
            row["statistical_group"] = int(group_index)
            row["statistically_tied"] = bool(tied)
            ordered.append(row)

    unmeasured.sort(key=lambda row: -_float_or(row.get("fast_Q"), _NEG_INF))
    for row in unmeasured:
        row["statistical_group"] = -1
        row["statistically_tied"] = False
    return ordered + unmeasured


# ---------------------------------------------------------------------------
# Structural retention: the (k, d_search) proxy-Pareto channel (v3.2 §1)
# ---------------------------------------------------------------------------
# This section is deliberately independent of everything above it. It never
# reads plpr, d_eff, d_cap, degraded_* or circuit distance: those are RUNTIME
# facts, and letting them touch structural retention is exactly what made a
# structurally strong code disappear the moment it measured badly.
# v2 (docs/06 §3 B4 / §4.1): at most STRUCTURAL_COORDINATE_CAP realizations
# of one coordinate survive in the reduced view. Under v1 twenty
# realizations of each endpoint filled the 16 elites and the interior Pareto
# point (40,12) was evicted by codes it does not dominate and that do not
# dominate it. Must match StructuralSearchConfig.reducer_id / coordinate_cap.
#
# v3.7 (2026-09-12, user decision): the structural front is non-dominated in
# THREE quantities -- k (max), d_search (max) and max_check_weight (min). A
# code is dominated only by a code at least as good on all three; a heavier
# code therefore survives exactly when it buys k or d_search that no lighter
# code reaches, and a lighter code with the same (k, d_search) is no longer
# discarded. The endpoint slots stay high-k / high-d_search (no "lightest"
# endpoint: that would hand a slot to a weight-4 toric code every turn), and
# the crowding fill spreads over the three axes. Rows without a recorded
# weight compare on (k, d_search) alone, so legacy artifacts still reduce.
# Also new: k >= MIN_PROMOTION_K -- k = 1 codes (HGP of two small seeds) are
# never structurally interesting and were consuming slots.
STRUCTURAL_REDUCER_ID = "structural_proxy_pareto_v3_weight"
STRUCTURAL_COORDINATE_CAP = 2
STRUCTURAL_OBJECTIVES = ("k", "d_search")
STRUCTURAL_WEIGHT_AXIS = "max_check_weight"
STRUCTURAL_AXES = ("k", "d_search", "-max_check_weight")
MIN_PROMOTION_K = 2

Cell = tuple[int, int, int | None]


def structural_coordinate(row: dict[str, Any]) -> tuple[int, int] | None:
    """(k, d_search) of a row, None when either objective is unusable."""

    k = _structural_int(row, "k")
    d_search = _structural_int(row, "d_search")
    if d_search is None:
        d_search = _structural_int(row, "d_proxy")
    if k is None or d_search is None:
        return None
    return (k, d_search)


def structural_cell(row: dict[str, Any]) -> Cell | None:
    """(k, d_search, max_check_weight) of a row -- the unit the v3.7 reducer
    groups, caps and crowds on. The weight is None when the row does not
    record one (legacy artifacts); such rows compare on (k, d_search)."""

    coord = structural_coordinate(row)
    if coord is None:
        return None
    return (coord[0], coord[1], _structural_int(row, STRUCTURAL_WEIGHT_AXIS))


def structural_coordinate_round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-order a pool so every UNIQUE cell (k, d_search, weight) appears once
    before any cell appears twice (docs/06 §4.1; v3.7: cells, so a light and a
    heavy realization of one (k, d_search) are two entries, not one).

    Round 0 holds the first row of each cell in pool order, round 1 the
    second realizations in the same cell order, and so on. Rows without a
    coordinate keep their relative order at the end of round 0. The pool
    order itself is not changed otherwise, so the channel-neutral reducer
    order still decides which realization of a cell comes first."""

    groups: dict[Cell, list[dict[str, Any]]] = {}
    order: list[Cell] = []
    orphans: list[dict[str, Any]] = []
    for row in rows:
        cell = structural_cell(row) if isinstance(row, dict) else None
        if cell is None:
            orphans.append(row)
            continue
        if cell not in groups:
            groups[cell] = []
            order.append(cell)
        groups[cell].append(row)
    out: list[dict[str, Any]] = []
    depth = max((len(groups[cell]) for cell in order), default=0)
    for index in range(depth):
        for cell in order:
            if index < len(groups[cell]):
                out.append(groups[cell][index])
        if index == 0:
            out.extend(orphans)
    if depth == 0:
        out.extend(orphans)
    return out


def _structural_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def structural_row_ok(row: dict[str, Any]) -> bool:
    """A row usable by the reducer: both objectives present, k >= 2, d >= 5.

    ``d_proxy`` may travel alongside as a read-only alias; a row where the two
    disagree is a protocol error and fails closed rather than silently picking
    the friendlier value."""

    k = _structural_int(row, "k")
    d_search = _structural_int(row, "d_search")
    if k is None or d_search is None or k < MIN_PROMOTION_K or d_search < MIN_PROMOTION_DISTANCE:
        return False
    alias = _structural_int(row, "d_proxy")
    if alias is not None and alias != d_search:
        return False
    return True


def _cell_dominates(a: Cell, b: Cell) -> bool:
    """Cell A dominates cell B: k and d_search at least as large, weight at
    most as large (when both weights are known), strictly better somewhere."""

    if a == b:
        return False
    if a[0] < b[0] or a[1] < b[1]:
        return False
    weight_known = a[2] is not None and b[2] is not None
    if weight_known and a[2] > b[2]:
        return False
    better = a[0] > b[0] or a[1] > b[1] or (weight_known and a[2] < b[2])
    return better


def structural_dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """A dominates B when it is at least as good on every axis -- k, d_search
    and (v3.7) check weight, lighter being better -- and strictly better on
    one. n, budget utilisation, family and degree are recorded but are NOT
    axes: putting n on an axis rebuilds the 'small codes are easier to keep'
    pressure this channel exists to remove. Weight enters only when both rows
    record one."""

    cell_a, cell_b = structural_cell(a), structural_cell(b)
    if cell_a is None or cell_b is None:
        return False
    return _cell_dominates(cell_a, cell_b)


def _structural_tiebreak_key(row: dict[str, Any]) -> tuple:
    """Only ever applied to rows with IDENTICAL cells (§1.2)."""

    n_total = row.get("n_total")
    try:
        n_total_value = int(n_total)
    except (TypeError, ValueError):
        n_total_value = 10**9
    try:
        q_search = float(row.get("Q_search") if row.get("Q_search") is not None else row.get("Q"))
    except (TypeError, ValueError):
        q_search = float("-inf")
    return (n_total_value, -q_search, str(row.get("code_hash") or ""))


def _structural_cell_groups(
    rows: list[dict[str, Any]],
) -> dict[Cell, list[dict[str, Any]]]:
    """Rows grouped by cell, each group in the §1.2 within-cell order
    (n_total asc, Q_search desc, hash)."""

    groups: dict[Cell, list[dict[str, Any]]] = {}
    for row in rows:
        cell = structural_cell(row)
        if cell is None:
            continue
        groups.setdefault(cell, []).append(row)
    for group in groups.values():
        group.sort(key=_structural_tiebreak_key)
    return groups


def _cell_sort(cell: Cell) -> tuple:
    return (-cell[0], -cell[1], cell[2] if cell[2] is not None else 10**6)


def _cell_pareto_layers(
    cells: list[Cell],
    *,
    max_layers: int | None = None,
) -> list[list[Cell]]:
    """Non-dominated sorting of UNIQUE cells under `_cell_dominates`.

    Runs on the objective space (unique cells, typically < 200), never on the
    ledger rows (thousands), so the all-pairs test stays cheap; distinct codes
    sharing a cell never separate here (audit fix #2/#4)."""

    remaining = sorted(set(cells), key=_cell_sort)
    layers: list[list[Cell]] = []
    while remaining and (max_layers is None or len(layers) < max_layers):
        front = [
            cell
            for cell in remaining
            if not any(_cell_dominates(other, cell) for other in remaining if other != cell)
        ]
        layers.append(front)
        remaining = [cell for cell in remaining if cell not in front]
    return layers


def structural_pareto_layers(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Non-dominated sorting on (k, d_search, -weight). Deterministic given the rows."""

    groups = _structural_cell_groups(list(rows))
    layers: list[list[dict[str, Any]]] = []
    for cell_layer in _cell_pareto_layers(list(groups)):
        layer: list[dict[str, Any]] = []
        for cell in sorted(cell_layer, key=_cell_sort):
            layer.extend(groups[cell])
        layers.append(layer)
    return layers


def _annotate_crowding(layer: list[dict[str, Any]]) -> None:
    """Crowding distance over k, d_search and (when every cell of the layer
    records one) check weight, written as FINITE numbers.

    Computed on the layer's UNIQUE cells and shared by every row of a cell
    (audit fix #2): per-row crowding let two realizations of the SAME point
    on the frontier crowd each other out before the §1.2 within-cell
    tie-break ever ran. `crowding_boundary=True` marks the max-k and the
    max-d_search cells only (v3.7: not the minimum ends -- on a 2-D front
    those coincide with the other maximum, on a 3-D front the min-k cell is
    whatever is lightest and must not get boundary priority every turn).
    Booleans instead of an infinite distance: these records are serialized
    to JSON and to a checkpoint, and `Infinity` is not valid JSON."""

    cells = sorted({cell for cell in (structural_cell(row) for row in layer) if cell is not None})
    distance = {cell: 0.0 for cell in cells}
    boundary = {cell: False for cell in cells}
    if len(cells) <= 2:
        for cell in cells:
            boundary[cell] = True
    else:
        axes = [0, 1] + ([2] if all(cell[2] is not None for cell in cells) else [])
        for axis in axes:
            ordered = sorted(cells, key=lambda c: (c[axis], c))
            if axis in (0, 1):
                boundary[ordered[-1]] = True
            low, high = ordered[0][axis], ordered[-1][axis]
            span = high - low
            if span <= 0:
                continue
            for index in range(1, len(ordered) - 1):
                gap = ordered[index + 1][axis] - ordered[index - 1][axis]
                distance[ordered[index]] += gap / span
    for row in layer:
        cell = structural_cell(row)
        if cell is None:
            continue
        row["crowding_distance"] = distance[cell]
        row["crowding_boundary"] = boundary[cell]


def _endpoint(layer: list[dict[str, Any]], objective: str) -> dict[str, Any] | None:
    """The layer's max-`objective` record.

    Ties on the primary objective break toward the OTHER objective first
    (audit fix #1): with the plain §1.2 tie-break, k=20/d=3/n=100 beat
    k=20/d=10/n=288 for the high-k endpoint — a strictly dominated code
    taking an endpoint slot because it is smaller. v3.7: then toward the
    LIGHTER code. Only after that does the within-cell (n_total, Q_search,
    hash) order apply."""

    if not layer:
        return None
    other = "d_search" if objective == "k" else "k"

    def _weight(row: dict[str, Any]) -> int:
        value = _structural_int(row, STRUCTURAL_WEIGHT_AXIS)
        return value if value is not None else 10**6

    return min(
        layer,
        key=lambda row: (
            -int(row[objective]),
            -int(row[other]),
            _weight(row),
            *_structural_tiebreak_key(row),
        ),
    )


def _lightest(layer: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The layer's lightest record (min max_check_weight). Ties break toward
    the HIGHER k, then the higher d_search, then the structural tie-break, so
    the slot goes to the most capable code of the lightest class, not to the
    smallest one. None when no row of the layer carries a weight."""

    rows = [row for row in layer if _structural_int(row, STRUCTURAL_WEIGHT_AXIS) is not None]
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            int(_structural_int(row, STRUCTURAL_WEIGHT_AXIS)),
            -int(row["k"]),
            -int(row["d_search"]),
            *_structural_tiebreak_key(row),
        ),
    )


def _structural_pool_sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Channel-neutral pool order: rank, then endpoints, then diversity.

    Deliberately not ordered by k, by d or by weight: the pool feeds parent
    slots, and leading with any objective would reintroduce a single-axis
    preference through the back door. Codes with DIFFERENT objectives and
    equal crowding are separated by code_hash alone — never by size or Q,
    which is how a "neutral" tie-break becomes a small-code preference. The
    `n_total` / `Q_search` tie-break applies only WITHIN a group of identical
    cells, where it decides nothing about the frontier's shape.
    """

    # Representative hash per (rank, cell) group: identical-objective rows
    # share it, so they sort adjacently and then among themselves.
    representative: dict[tuple, str] = {}

    def _group(row: dict[str, Any]) -> tuple:
        cell = structural_cell(row) or (0, 0, None)
        return (int(row.get("pareto_rank") or 0), cell[0], cell[1], cell[2] if cell[2] is not None else -1)

    for row in rows:
        key = _group(row)
        code_hash = str(row.get("code_hash") or "")
        if key not in representative or code_hash < representative[key]:
            representative[key] = code_hash

    def _key(row: dict[str, Any]) -> tuple:
        group = _group(row)
        return (
            group[0],
            0 if row.get("crowding_boundary") else 1,
            -float(row.get("crowding_distance") or 0.0),
            representative[group],
            _structural_tiebreak_key(row),
        )

    return sorted(rows, key=_key)


def _cap_layer_per_coordinate(
    layer: list[dict[str, Any]],
    forced: list[dict[str, Any]],
    *,
    coordinate_cap: int | None,
) -> list[dict[str, Any]]:
    """Keep at most `coordinate_cap` rows per cell in a layer.

    Forced endpoints always survive; the remaining realizations of a cell are
    taken in the §1.2 within-cell order (n_total asc, Q_search desc, hash),
    which is the order `structural_pareto_layers` already emits.
    ``coordinate_cap`` None or <= 0 means no cap (v1)."""

    if coordinate_cap is None or int(coordinate_cap) <= 0:
        return list(layer)
    cap = int(coordinate_cap)
    counts: dict[Cell, int] = {}
    for row in forced:
        cell = structural_cell(row)
        if cell is not None:
            counts[cell] = counts.get(cell, 0) + 1
    kept: list[dict[str, Any]] = []
    for row in layer:
        if any(row is item for item in forced):
            kept.append(row)
            continue
        cell = structural_cell(row)
        if cell is None:
            continue
        if counts.get(cell, 0) >= cap:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        kept.append(row)
    return kept


def reduce_structural_proxy_elites(
    rows: list[dict[str, Any]],
    *,
    limit: int = 24,
    coordinate_cap: int | None = STRUCTURAL_COORDINATE_CAP,
) -> list[dict[str, Any]]:
    """The canonical structural reducer (§1.3, v2 coordinate cap, v3.7 weight axis).

    Layers by Pareto rank on (k, d_search, -weight); every layer keeps its
    max-k and max-d_search endpoints; then (v3.6 B4) at most
    ``coordinate_cap`` realizations per cell survive; the rest of a truncated
    layer fills by crowding distance. The cap runs BEFORE the crowding fill
    so that dozens of realizations of an endpoint cell (crowding_boundary=
    True, which the fill prefers) cannot evict an interior Pareto cell. Rows
    are returned annotated with `pareto_rank` / `crowding_boundary` /
    `crowding_distance` and in a deterministic, channel-neutral order."""

    usable = [dict(row) for row in rows if structural_row_ok(row)]
    # One record per code: a rediscovery must not consume two elite slots.
    by_hash: dict[str, dict[str, Any]] = {}
    for row in usable:
        key = str(row.get("code_hash") or row.get("program_id") or "")
        if not key:
            continue
        held = by_hash.get(key)
        if held is None or _structural_tiebreak_key(row) < _structural_tiebreak_key(held):
            by_hash[key] = row
    limit = max(0, int(limit))
    selected: list[dict[str, Any]] = []
    for rank, layer in enumerate(structural_pareto_layers(list(by_hash.values())), start=1):
        _annotate_crowding(layer)
        for row in layer:
            row["pareto_rank"] = rank
            row["reducer_id"] = STRUCTURAL_REDUCER_ID
        room = limit - len(selected)
        if room <= 0:
            break
        forced: list[dict[str, Any]] = []
        for objective in STRUCTURAL_OBJECTIVES:
            endpoint = _endpoint(layer, objective)
            if endpoint is not None and not any(item is endpoint for item in forced):
                forced.append(endpoint)
        forced = forced[:room]
        # v2: the per-cell cap applies to EVERY layer (not only truncated
        # ones) so the elite view has one meaning regardless of how full it is.
        layer = _cap_layer_per_coordinate(layer, forced, coordinate_cap=coordinate_cap)
        if len(layer) <= room:
            selected.extend(layer)
            continue
        representative: dict[Cell, str] = {}
        for row in layer:
            cell = structural_cell(row)
            if cell is None:
                continue
            code_hash = str(row.get("code_hash") or "")
            if cell not in representative or code_hash < representative[cell]:
                representative[cell] = code_hash
        rest = sorted(
            (row for row in layer if not any(row is item for item in forced)),
            key=lambda row: (
                0 if row.get("crowding_boundary") else 1,
                -float(row.get("crowding_distance") or 0.0),
                # Cells separate by a shared representative hash (never by
                # size or Q); rows of ONE cell then follow §1.2.
                representative.get(structural_cell(row) or (0, 0, None), ""),
                _structural_tiebreak_key(row),
            ),
        )
        selected.extend(forced)
        selected.extend(rest[: room - len(forced)])
    return _structural_pool_sorted(selected)


def _structural_front(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pool's lowest Pareto layer (rows without a rank are ranked here)."""

    if not pool:
        return []
    if any(row.get("pareto_rank") is None for row in pool):
        front_cells = set(
            _cell_pareto_layers(
                [cell for cell in (structural_cell(row) for row in pool) if cell is not None],
                max_layers=1,
            )[0]
        )
        return [row for row in pool if structural_cell(row) in front_cells]
    lowest = min(int(row["pareto_rank"]) for row in pool)
    return [row for row in pool if int(row["pareto_rank"]) == lowest]


def _row_family(row: dict[str, Any]) -> str:
    spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
    return str(row.get("family") or spec.get("family") or "?")


def structural_endpoints(
    rows: list[dict[str, Any]],
    *,
    distinct_families: bool = False,
) -> dict[str, dict[str, Any] | None]:
    """Max-k / max-d_search records of an already-reduced pool.

    Audit fix #1: endpoints come from the pool's LOWEST Pareto rank only. A
    reduced pool legitimately retains deeper-layer rows (the cache keeps
    diversity), but a slot endpoint drawn from a dominated layer hands one of
    the frozen measurement slots to a code the frontier already beats.
    Rows without a rank annotation are ranked here first.

    v3.8 ``distinct_families``: when the max-d_search endpoint belongs to the
    same family as the max-k endpoint and the front holds rows of another
    family, the second slot goes to the max-d_search row among those other
    families (``high_d_other_family`` says so). The front itself is the
    weight-aware (k, d_search, -w) front, so this never admits a dominated
    code; it only chooses which frontier point the second slot certifies.

    Fig.3 (2026-10-12): ``low_w`` is the front's lightest cell (min
    max_check_weight, ties toward higher k, then higher d_search). It takes a
    slot only when the caller asks for it (StructuralSearchConfig.
    lightest_endpoint_slot, the Fig.3 CNOT profile)."""

    pool = [row for row in rows if structural_row_ok(row)]
    if not pool:
        return {"high_k": None, "high_d": None, "high_d_other_family": False, "low_w": None}
    front = _structural_front(pool)
    high_k = _endpoint(front, "k")
    high_d = _endpoint(front, "d_search")
    other_family = False
    if distinct_families and high_k is not None and high_d is not None:
        if _row_family(high_d) == _row_family(high_k):
            others = [row for row in front if _row_family(row) != _row_family(high_k)]
            if others:
                high_d = _endpoint(others, "d_search")
                other_family = True
    return {"high_k": high_k, "high_d": high_d, "high_d_other_family": other_family, "low_w": _lightest(front)}


def select_structural_slots(
    rows: list[dict[str, Any]],
    *,
    slots: int,
    lightest: bool = False,
    distinct_families: bool = False,
    demoted_families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Machine slot order for the slow queue (§4.3).

    high-k endpoint, then high-d_search endpoint, then further
    crowding/Pareto representatives. v3.6 (docs/06 §3 B4): slot 3 and every
    further slot SKIP cells already selected — a second realization of an
    endpoint cell is not a third frontier point. Second realizations are
    appended after every unique cell has been emitted (reason
    `crowding_representative_repeat_coordinate`), so a caller that emits the
    whole pool (audit fix #4 backfill) still sees them, last. v3.7: the
    formal quota is the two endpoints (StructuralSearchConfig.
    structural_slow_slots = 2); the crowding representatives remain the
    order in which a short pool is backfilled.

    `rows` must already be filtered to UNMEASURED candidates — selecting first
    and dropping measured hashes afterwards is what leaves empty slots from
    turn 1 on.

    v3.8: rows of a ``demoted_families`` family (endpoint demotion, computed
    by the campaign graph from the run's endpoint measurements) are left out
    of the pool entirely -- they neither take an endpoint nor a crowding
    slot; ``distinct_families`` is passed to ``structural_endpoints``.

    Fig.3 (2026-10-12): ``lightest`` adds the front's lightest cell as a third
    machine slot right after the two endpoints."""

    demoted = {str(item) for item in (demoted_families or ())}
    pool = [
        row
        for row in rows
        if structural_row_ok(row) and (not demoted or _row_family(row) not in demoted)
    ]
    slots = max(0, int(slots))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    used_cells: set[Cell] = set()

    def _take(
        row: dict[str, Any] | None, reason: str, *, skip_used_coordinate: bool = False
    ) -> None:
        if row is None or len(selected) >= slots:
            return
        key = str(row.get("code_hash") or row.get("program_id") or "")
        if not key or key in used:
            return
        cell = structural_cell(row)
        if cell is None:
            return
        if skip_used_coordinate and cell in used_cells:
            return
        used.add(key)
        used_cells.add(cell)
        item = dict(row)
        item["structural_slot_reason"] = reason
        selected.append(item)

    endpoints = structural_endpoints(pool, distinct_families=bool(distinct_families))
    _take(endpoints["high_k"], "high_k_endpoint")
    _take(
        endpoints["high_d"],
        "high_d_search_endpoint_other_family"
        if endpoints.get("high_d_other_family")
        else "high_d_search_endpoint",
    )
    if lightest:
        # Fig.3 Phase B: third machine slot = the front's lightest cell.
        _take(endpoints.get("low_w"), "lightest_endpoint")
    ordered = _structural_pool_sorted(pool)
    for row in ordered:
        if len(selected) >= slots:
            break
        _take(row, "crowding_representative", skip_used_coordinate=True)
    for row in ordered:
        if len(selected) >= slots:
            break
        _take(row, "crowding_representative_repeat_coordinate")
    return selected


def _row_ci(row: dict[str, Any]) -> tuple[float, float] | None:
    # plpr-scale (paper A5) intervals compare cleanly across round counts;
    # older artifacts fall back to the per-logical block interval.
    ci = row.get("per_logical_per_round_ci") or row.get("per_logical_block_ci")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        try:
            return (float(ci[0]), float(ci[1]))
        except Exception:
            return None
    return None


def _float_or(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)
