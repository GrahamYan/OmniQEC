"""Structural coverage of the search — the instrument that makes holes visible.

Ranked candidate lists answer "what is best so far". They cannot answer "what
has never been tried", because a region the generator never proposed leaves no
row anywhere. Every count here is therefore reported per structural region
(construction/structure.py), including regions with zero proposals, and the
censored count is kept separate from the measured one: a region can look empty
because it is bad, because it was never proposed, or because racing stopped
every candidate at pilot precision, and those call for opposite responses.

v3.6 (docs/06 §4.3): regions carry the LP base-matrix shapes and the
budget-utilisation band, ``catalog_gaps`` names the (branch x shape) pairs
that were never proposed AT THIS BUDGET (pairs that cannot fit are not
"gaps"), and ``coverage_fact_lines`` renders the same numbers as neutral fact
lines for the policy's S-9 block and the strategy brief.

Review fix (2026-09-03, family symmetry): the gap list used to enumerate only
the two LP families, so a bb or hgp hole could never be reported while every
turn's 12-item cut showed twelve lp_nonabelian/dihedral entries (dict order).
Every family now has its own region grid -- LP: (branch x A-shape|B-shape);
bb: (bb_square | bb_rect x |G| = ell*m band); hgp: (seed column band of H1 |
of H2) -- feasibility uses the family's smallest spelling of the region, the
shown cut is round-robin over families (then branches), and the first line
carries the never/feasible counts per family.

This module only counts. It never ranks regions, scores them, or suggests
where to go next — that judgment belongs to the policy agent reading it. The
rendered lines therefore carry no goal verb (tests/test_prompt_neutrality.py
BANNED_PHRASES / GOAL_RE are re-checked against them in
tests/test_v36_sec_catalog_coverage.py).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from omniqec_agentic.construction.grammars.ga_seed import DEFAULT_ALLOWED_SHAPES
from omniqec_agentic.construction.structure import (
    _ORDER_BANDS,
    spec_total_qubits,
    structure_descriptor,
    structure_key,
)

# lp_nonabelian grammar branches and the smallest |G| each can spell (the
# budget feasibility of a (branch x shape) pair uses the smallest group):
# D3 = 6; C3:C4(2) = 12 (n=2 with k=m-1 is the dihedral spelling, C4:C4(3) is
# order 16 but C3:C4(2) is smaller); C2xD4 = 16 (C2xD{odd} is D{2n}); A4 = 12.
NONABELIAN_BRANCHES: dict[str, int] = {
    "dihedral": 6,
    "metacyclic": 12,
    "direct_product": 16,
    "alternating_symmetric": 12,
}
# lp_abelian branches (structure.py names): one cyclic factor vs a product.
ABELIAN_BRANCHES: dict[str, int] = {"cyclic": 2, "abelian_product": 4}
# bb regions: (bb_square | bb_rect) x |G| = ell*m band (the same bands as
# structure.order_band, plus the open top band). Feasibility = 4 x the
# smallest ell*m the class can spell inside the band (square: a perfect
# square; rect: any product >= 2) <= n_budget.
BB_CLASSES: tuple[str, ...] = ("bb_square", "bb_rect")
BB_ORDER_BANDS: tuple[tuple[int, int | None], ...] = tuple(_ORDER_BANDS) + ((_ORDER_BANDS[-1][1], None),)
# hgp regions: (column-count band of seed H1 | of seed H2). A seed with n
# columns and row/column degrees in [2, 4] needs m >= max(2, ceil(n/2)) rows,
# so the smallest spelling of a band costs (n + m)(n' + m') qubits.
HGP_SEED_COLS_BANDS: tuple[tuple[int, int | None, str], ...] = (
    (2, 4, "n2-3"),
    (4, 7, "n4-6"),
    (7, 11, "n7-10"),
    (11, 16, "n11-15"),
    (16, None, "n16+"),
)
# Fixed family order of the gap grid (reports are comparable across turns).
GAP_FAMILY_ORDER: tuple[str, ...] = ("lp_nonabelian", "lp_abelian", "bb", "hgp")


def _region_row() -> dict[str, Any]:
    return {"proposed": 0, "measured": 0, "censored": 0, "best_plpr": None}


def _row_budget(row: dict[str, Any], n_budget: int | None) -> int | None:
    if n_budget:
        return int(n_budget)
    value = row.get("n_budget")
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def coverage_report(
    archive_rows: Iterable[dict[str, Any]],
    slow_rows: Iterable[dict[str, Any]] | None = None,
    *,
    limit: int = 24,
    n_budget: int | None = None,
) -> dict[str, Any]:
    """Aggregate proposals and measurements into structural regions.

    `archive_rows` are fast-archive queue rows (they carry `spec`); `slow_rows`
    are slow rankings (they carry `code_hash`, `per_logical_per_round` and
    `stopped_by`). Slow rows are joined onto regions through the archive's
    code_hash so the join works even though slow rows carry no spec.

    Region labels use the row's own ``n_budget`` (or the ``n_budget`` given)
    for the utilisation band, and the LP shapes when the spec carries them.
    """

    regions: dict[str, dict[str, Any]] = defaultdict(_region_row)
    hash_to_region: dict[str, str] = {}

    for row in archive_rows or []:
        if not isinstance(row, dict):
            continue
        spec = row.get("spec")
        key = structure_key(spec, _row_budget(row, n_budget))
        regions[key]["proposed"] += 1
        code_hash = str(row.get("code_hash") or "")
        if code_hash:
            hash_to_region[code_hash] = key

    for row in slow_rows or []:
        if not isinstance(row, dict) or row.get("is_baseline"):
            continue
        key = hash_to_region.get(str(row.get("code_hash") or ""))
        if key is None:
            continue
        region = regions[key]
        # "Censored" means racing stopped the candidate before confirm depth:
        # its performance is unresolved, NOT known to be bad.
        if str(row.get("stopped_by") or "") == "racing":
            region["censored"] += 1
        plpr = row.get("per_logical_per_round")
        if plpr is not None:
            region["measured"] += 1
            current = region["best_plpr"]
            if current is None or float(plpr) < float(current):
                region["best_plpr"] = float(plpr)

    rows = [{"region": key, **value} for key, value in regions.items()]
    rows.sort(key=lambda r: (-r["proposed"], r["region"]))
    total_proposed = sum(r["proposed"] for r in rows)
    return {
        "regions": rows[: max(0, int(limit))],
        "regions_seen": len(rows),
        "total_proposed": total_proposed,
        "unmeasured_regions": sorted(
            r["region"] for r in rows if r["measured"] == 0
        )[: max(0, int(limit))],
        "note": (
            "proposed=constructed this turn; measured=has a per_logical_per_round; "
            "censored=racing stopped it at pilot depth (performance unresolved, not "
            "known bad). Regions absent from this table were never proposed."
        ),
    }


def _shape_label(rows_a: int, cols_a: int, rows_b: int, cols_b: int) -> str:
    return f"{rows_a}x{cols_a}|{rows_b}x{cols_b}"


def _shape_cost(rows_a: int, cols_a: int, rows_b: int, cols_b: int) -> int:
    """Total-qubit multiplier of an LP shape pair: (nA+mA)*(nB+mB)."""

    return (rows_a + cols_a) * (rows_b + cols_b)


def _bb_band_label(low: int, high: int | None) -> str:
    return f"{low}-{high}" if high is not None else f"{low}+"


def _bb_min_order(group_class: str, low: int, high: int | None) -> int | None:
    """Smallest ell*m a bb class can spell inside [low, high)."""

    if group_class == "bb_square":
        root = 1
        while root * root < low:
            root += 1
        order = root * root
    else:
        order = max(int(low), 2)
    if high is not None and order >= int(high):
        return None
    return int(order)


def _hgp_seed_cols_band(cols: int | None) -> str | None:
    if cols is None:
        return None
    value = int(cols)
    for low, high, label in HGP_SEED_COLS_BANDS:
        if value >= low and (high is None or value < high):
            return label
    return None


def _hgp_min_seed_qubits(low: int) -> int:
    """n + m for the smallest degree-[2,4] seed with >= ``low`` columns."""

    n = int(low)
    m = max(2, (n + 1) // 2)
    return n + m


def _shape_cols(shape: str | None) -> int | None:
    """'rows x cols' -> cols (None when unreadable)."""

    if not shape or "x" not in str(shape):
        return None
    try:
        return int(str(shape).split("x", 1)[1])
    except (TypeError, ValueError):
        return None


def gap_region_of(spec: Any) -> tuple[str, str, str] | None:
    """The (family, branch, region) gap-grid key of one spec, or None when
    the spec does not carry enough structure to be placed on the grid."""

    if not isinstance(spec, dict):
        return None
    family = str(spec.get("family") or "")
    descriptor = structure_descriptor(spec)
    branch = str(descriptor.get("group_class") or "other")
    shape_a, shape_b = descriptor.get("shape_a"), descriptor.get("shape_b")
    if family in ("lp_nonabelian", "lp_abelian"):
        if shape_a and shape_b:
            return (family, branch, f"{shape_a}|{shape_b}")
        return None
    if family == "bb":
        band = descriptor.get("group_order_band")
        if branch in BB_CLASSES and band:
            return (family, branch, str(band))
        return None
    if family == "hgp":
        band_a = _hgp_seed_cols_band(_shape_cols(shape_a))
        band_b = _hgp_seed_cols_band(_shape_cols(shape_b))
        if band_a and band_b:
            return (family, "hgp", f"{band_a}|{band_b}")
        return None
    return None


def feasible_branch_shape_pairs(n_budget: int | None) -> list[tuple[str, str, str]]:
    """(family, branch, region) triples every family's grammar can spell
    within ``n_budget`` (all when no budget is given), using the smallest
    spelling of each region; the order is fixed (GAP_FAMILY_ORDER, then the
    branch order, then the region order) so reports are comparable across
    turns. LP regions are 'shapeA|shapeB', bb regions the |G| band, hgp
    regions the seed column bands 'bandA|bandB'."""

    budget = int(n_budget) if n_budget else None

    def fits(cost: int) -> bool:
        return budget is None or cost <= budget

    pairs: list[tuple[str, str, str]] = []
    for family in GAP_FAMILY_ORDER:
        if family in ("lp_nonabelian", "lp_abelian"):
            branches = NONABELIAN_BRANCHES if family == "lp_nonabelian" else ABELIAN_BRANCHES
            for branch, min_order in branches.items():
                for rows_a, cols_a in DEFAULT_ALLOWED_SHAPES:
                    for rows_b, cols_b in DEFAULT_ALLOWED_SHAPES:
                        cost = _shape_cost(rows_a, cols_a, rows_b, cols_b) * min_order
                        if fits(cost):
                            pairs.append((family, branch, _shape_label(rows_a, cols_a, rows_b, cols_b)))
        elif family == "bb":
            for group_class in BB_CLASSES:
                for low, high in BB_ORDER_BANDS:
                    order = _bb_min_order(group_class, low, high)
                    if order is not None and fits(4 * order):
                        pairs.append((family, group_class, _bb_band_label(low, high)))
        elif family == "hgp":
            for low_a, _high_a, label_a in HGP_SEED_COLS_BANDS:
                for low_b, _high_b, label_b in HGP_SEED_COLS_BANDS:
                    cost = _hgp_min_seed_qubits(low_a) * _hgp_min_seed_qubits(low_b)
                    if fits(cost):
                        pairs.append((family, "hgp", f"{label_a}|{label_b}"))
    return pairs


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def catalog_gaps(
    archive_rows: Iterable[dict[str, Any]], n_budget: int | None = None
) -> dict[str, Any]:
    """Which grammar branches / shapes have and have not been proposed.

    The lp_nonabelian catalog admits four branches; a run can spend every
    proposal inside one of them and nothing in a ranked list would say so.
    v3.6 adds, for every family: per-region counts (region = structure_key
    with shapes and the utilisation band at ``n_budget``), the best k and
    best d_search seen per region, and the (family, branch, shape-pair)
    triples that are spellable at this budget but were never proposed.
    """

    branches = {"dihedral": 0, "metacyclic": 0, "direct_product": 0, "alternating_symmetric": 0}
    orders: list[int] = []
    regions: dict[str, dict[str, Any]] = {}
    proposed_pairs: set[tuple[str, str, str]] = set()
    for row in archive_rows or []:
        if not isinstance(row, dict):
            continue
        spec = row.get("spec")
        if not isinstance(spec, dict):
            continue
        descriptor = structure_descriptor(spec)
        family = str(spec.get("family") or "")
        branch = str(descriptor.get("group_class") or "other")
        if family == "lp_nonabelian":
            if branch in branches:
                branches[branch] += 1
            order = descriptor.get("group_order")
            if order is not None:
                orders.append(int(order))
        shape_a, shape_b = descriptor.get("shape_a"), descriptor.get("shape_b")
        gap_key = gap_region_of(spec)
        if gap_key is not None:
            proposed_pairs.add(gap_key)
        budget = _row_budget(row, n_budget)
        key = structure_key(spec, budget)
        region = regions.setdefault(
            key,
            {
                "region": key,
                "family": family or None,
                "branch": branch,
                "shapes": (f"{shape_a}|{shape_b}" if shape_a and shape_b else None),
                "proposals": 0,
                "best_k": None,
                "best_d_search": None,
                "max_total_qubits": None,
            },
        )
        region["proposals"] += 1
        k = _int_or_none(row.get("k"))
        d_search = _int_or_none(row.get("d_search"))
        if k is not None and (region["best_k"] is None or k > region["best_k"]):
            region["best_k"] = k
        if d_search is not None and (
            region["best_d_search"] is None or d_search > region["best_d_search"]
        ):
            region["best_d_search"] = d_search
        total = spec_total_qubits(spec)
        if total is None:
            n_data, n_checks = _int_or_none(row.get("n")), _int_or_none(row.get("n_checks"))
            total = (n_data + n_checks) if (n_data is not None and n_checks is not None) else None
        if total is not None and (
            region["max_total_qubits"] is None or total > region["max_total_qubits"]
        ):
            region["max_total_qubits"] = total
    total = sum(branches.values())
    feasible = feasible_branch_shape_pairs(n_budget)
    never = [
        {"family": family, "branch": branch, "shapes": shapes}
        for family, branch, shapes in feasible
        if (family, branch, shapes) not in proposed_pairs
    ]
    by_family: dict[str, dict[str, int]] = {}
    for family, _branch, _shapes in feasible:
        by_family.setdefault(family, {"feasible": 0, "never_proposed": 0})["feasible"] += 1
    for item in never:
        by_family[item["family"]]["never_proposed"] += 1
    region_rows = sorted(regions.values(), key=lambda r: (-r["proposals"], r["region"]))
    return {
        "n_budget": (None if not n_budget else int(n_budget)),
        "lp_nonabelian_proposals": total,
        "by_branch": branches,
        "never_proposed_branches": sorted(k for k, v in branches.items() if v == 0),
        "group_order_range": [min(orders), max(orders)] if orders else None,
        "regions": region_rows,
        "feasible_branch_shape_pairs": len(feasible),
        "never_proposed_branch_shapes": never,
        # Review fix: every family's grid, counted, so no family can be
        # invisible in the gap report.
        "never_proposed_by_family": by_family,
        "note": (
            "Catalog branches with zero proposals are reachable by the grammar "
            "(see the lp_nonabelian format contract) but unexplored so far. "
            "never_proposed_branch_shapes lists (family, branch, region) triples "
            "that fit this budget with the family's smallest spelling of the "
            "region and have no proposal: LP regions are A-shape|B-shape, bb "
            "regions the ell*m band, hgp regions the seed column bands; "
            "never_proposed_by_family counts them per family; regions carry the "
            "best k / d_search seen there."
        ),
    }


def round_robin_gaps(never: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """The first ``limit`` gap items taken round-robin over families (in
    GAP_FAMILY_ORDER / first-seen order) and, inside a family, round-robin
    over branches -- so a 12-item cut never shows one branch twelve times."""

    cap = max(0, int(limit))
    if cap == 0 or not never:
        return []
    families: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in never:
        families.setdefault(str(item["family"]), {}).setdefault(str(item["branch"]), []).append(item)
    family_order = [f for f in GAP_FAMILY_ORDER if f in families] + [
        f for f in families if f not in GAP_FAMILY_ORDER
    ]
    cursors = {family: 0 for family in family_order}  # next branch index per family
    branch_lists = {family: list(families[family].values()) for family in family_order}
    shown: list[dict[str, Any]] = []
    while len(shown) < cap:
        progressed = False
        for family in family_order:
            lists = branch_lists[family]
            if not any(lists):
                continue
            # next non-empty branch of this family (round-robin over branches)
            for _ in range(len(lists)):
                index = cursors[family] % len(lists)
                cursors[family] += 1
                if lists[index]:
                    shown.append(lists[index].pop(0))
                    progressed = True
                    break
            if len(shown) >= cap:
                break
        if not progressed:
            break
    return shown


def coverage_fact_lines(
    archive_rows: Iterable[dict[str, Any]],
    n_budget: int | None,
    *,
    limit: int = 12,
) -> list[str]:
    """Neutral fact lines about coverage at ``n_budget`` for a prompt block.

    Facts only: which regions were never proposed at this budget, and per
    proposed region the proposal count and the best k / d_search seen. No
    ranking, no goal verb — a reader decides what, if anything, to do."""

    gaps = catalog_gaps(list(archive_rows or []), n_budget)
    budget_text = f"n_budget={int(n_budget)}" if n_budget else "n_budget unknown"
    lines: list[str] = []
    never = gaps["never_proposed_branch_shapes"]
    cap = max(0, int(limit))
    by_family = gaps.get("never_proposed_by_family") or {}
    counts = "; ".join(
        f"{family} {by_family[family]['never_proposed']} of {by_family[family]['feasible']}"
        for family in sorted(by_family)
    )
    if never:
        picked = round_robin_gaps(never, cap)
        shown = ", ".join(
            f"{item['family']}/{item['branch']}/{item['shapes']}" for item in picked
        )
        more = f" (+{len(never) - len(picked)} more)" if len(never) > len(picked) else ""
        lines.append(
            f"Regions never proposed at this budget ({budget_text}): {shown}{more}"
            f"; never proposed of spellable regions by family: {counts}"
        )
    else:
        lines.append(
            f"Regions never proposed at this budget ({budget_text}): none among the "
            f"{gaps['feasible_branch_shape_pairs']} spellable regions of the four families"
        )
    for region in gaps["regions"][:cap]:
        best_k = "n/a" if region["best_k"] is None else region["best_k"]
        best_d = "n/a" if region["best_d_search"] is None else region["best_d_search"]
        lines.append(
            f"Region {region['region']}: proposals {region['proposals']}, "
            f"best k {best_k}, best d_search {best_d}"
        )
    return lines
