"""Structural descriptors: the vocabulary for talking about WHERE in the
search space a candidate sits.

A campaign can rank candidates by performance without ever naming the region
they came from, and then nobody — agent or human — can tell that 85% of the
proposals landed in one corner of the grammar. These descriptors give that
region a name, and the same name is used in three places so they stay
comparable: the coverage report the agent reads, the evidence rows the
backtester adjudicates, and the ledger predicates a hypothesis may reference.

Descriptors are deliberately COARSE. They are axes for noticing holes, not a
second code hash: `group_order_band` buckets rather than counts so a histogram
stays readable, and `group_class` names the grammar branch rather than the
group. Nothing here judges quality — a descriptor says where a code is, never
whether it is good.
"""

from __future__ import annotations

import re
from typing import Any

# Grammar branches of the lp_nonabelian catalog (construction/groups/catalog.py).
_DIHEDRAL_RE = re.compile(r"^D(\d+)$")
_METACYCLIC_RE = re.compile(r"^C(\d+):C(\d+)\((\d+)\)$")
_DIRECT_RE = re.compile(r"^C(\d+)x(D\d+|A4|S4)$")

# Bands are open-ended at the top so a descriptor never fails on a big group.
_ORDER_BANDS = ((0, 12), (12, 24), (24, 40), (40, 60), (60, 100))

STRUCTURE_FIELDS = ("group_class", "group_order", "group_order_band", "shape_a", "shape_b")

# v3.6 (docs/06 §4.3): budget-utilisation bins of total_qubits / n_budget. A
# region is "where in the grammar" AND "how much of the budget it spends":
# the audit's 1x2/1x2 direct-product codes at 90-100% utilisation and the
# 1x1 dihedral codes at 30% are different places even when the branch and
# order band agree. Bins are fixed so labels are comparable across turns.
UTILISATION_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.5, "util<0.5"),
    (0.5, 0.75, "util0.5-0.75"),
    (0.75, 0.9, "util0.75-0.9"),
    (0.9, 1.0, "util0.9-1.0"),
)
UTILISATION_LABELS: tuple[str, ...] = tuple(label for _, _, label in UTILISATION_BINS)


def order_band(order: int | None) -> str | None:
    """Bucket a group order into a readable band label."""

    if order is None:
        return None
    value = int(order)
    if value < 0:
        return None
    for low, high in _ORDER_BANDS:
        if low <= value < high:
            return f"{low}-{high}"
    return f"{_ORDER_BANDS[-1][1]}+"


def _nonabelian_group_class(name: str) -> tuple[str, int | None]:
    """Name the catalog branch a group name belongs to, and its order."""

    text = str(name).strip()
    match = _DIHEDRAL_RE.match(text)
    if match:
        return "dihedral", 2 * int(match.group(1))
    match = _METACYCLIC_RE.match(text)
    if match:
        return "metacyclic", int(match.group(1)) * int(match.group(2))
    match = _DIRECT_RE.match(text)
    if match:
        inner = match.group(2)
        inner_order = {"A4": 12, "S4": 24}.get(inner)
        if inner_order is None:
            inner_match = _DIHEDRAL_RE.match(inner)
            inner_order = 2 * int(inner_match.group(1)) if inner_match else None
        order = int(match.group(1)) * inner_order if inner_order else None
        return "direct_product", order
    if text in ("A4", "S4"):
        return "alternating_symmetric", {"A4": 12, "S4": 24}[text]
    return "other", None


def _matrix_shape(matrix: Any) -> str | None:
    """Render an LP base matrix's shape as 'rowsxcols'."""

    if not isinstance(matrix, list) or not matrix:
        return None
    if not isinstance(matrix[0], list):
        return None
    return f"{len(matrix)}x{len(matrix[0])}"


def structure_descriptor(spec: Any) -> dict[str, Any]:
    """Map a construction spec to its coarse structural coordinates.

    Returns every key in STRUCTURE_FIELDS; unknown coordinates are None so a
    caller can always read the same shape of dict regardless of family.
    """

    out: dict[str, Any] = {field: None for field in STRUCTURE_FIELDS}
    if not isinstance(spec, dict):
        return out
    family = str(spec.get("family") or "")

    if family == "lp_nonabelian":
        group_class, order = _nonabelian_group_class(spec.get("group") or "")
        out["group_class"] = group_class
        out["group_order"] = order
    elif family == "lp_abelian":
        group = spec.get("group")
        factors = group.get("factors") if isinstance(group, dict) else None
        if isinstance(factors, list) and factors:
            order = 1
            for factor in factors:
                order *= int(factor)
            out["group_order"] = order
            # A single cyclic factor behaves very differently from a product.
            out["group_class"] = "cyclic" if len(factors) == 1 else "abelian_product"
        else:
            out["group_class"] = "abelian"
    elif family == "bb":
        ell, m = spec.get("ell"), spec.get("m")
        if ell is not None and m is not None:
            out["group_order"] = int(ell) * int(m)
            out["group_class"] = "bb_square" if int(ell) == int(m) else "bb_rect"
        else:
            out["group_class"] = "bb"
    elif family == "hgp":
        out["group_class"] = "hgp"
        seeds = spec.get("seeds")
        if isinstance(seeds, dict):
            h1 = seeds.get("H1")
            shape = _matrix_shape(h1)
            if shape:
                out["shape_a"] = shape
            h2 = seeds.get("H2")
            shape = _matrix_shape(h2)
            if shape:
                out["shape_b"] = shape
        return out

    if family in ("lp_abelian", "lp_nonabelian"):
        out["shape_a"] = _matrix_shape(spec.get("A"))
        out["shape_b"] = _matrix_shape(spec.get("B"))

    out["group_order_band"] = order_band(out["group_order"])
    return out


def utilisation_band(total_qubits: int | None, n_budget: int | None) -> str | None:
    """Bin total_qubits / n_budget into one of UTILISATION_LABELS (None when
    either side is unknown; ratios above 1 land in the top bin, over-budget
    specs are rejected elsewhere)."""

    if total_qubits is None or not n_budget:
        return None
    ratio = float(total_qubits) / float(n_budget)
    if ratio < 0:
        return None
    for low, high, label in UTILISATION_BINS:
        if low <= ratio < high:
            return label
    return UTILISATION_BINS[-1][2]


def _shape_dims(matrix: Any) -> tuple[int, int] | None:
    if not isinstance(matrix, (list, tuple)) or not matrix:
        return None
    widths = {len(row) if isinstance(row, (list, tuple)) else -1 for row in matrix}
    if len(widths) != 1:
        return None
    cols = next(iter(widths))
    return (len(matrix), cols) if cols >= 1 else None


def spec_total_qubits(spec: Any) -> int | None:
    """Total qubits (data + one ancilla per check) read off the RAW spec with
    no construction — the same arithmetic as the registry size formulas.
    None whenever the spec cannot be read with certainty."""

    if not isinstance(spec, dict):
        return None
    family = str(spec.get("family") or "")
    try:
        if family == "bb":
            return 4 * int(spec["ell"]) * int(spec["m"])
        if family == "hgp":
            seeds = spec.get("seeds") or {}
            h1, h2 = _shape_dims(seeds.get("H1")), _shape_dims(seeds.get("H2"))
            if h1 is None or h2 is None:
                return None
            return (h1[0] + h1[1]) * (h2[0] + h2[1])
        if family in ("lp_abelian", "lp_nonabelian"):
            from .lp import abelian_group_order, lp_total_qubits, nonabelian_group_order

            reader = abelian_group_order if family == "lp_abelian" else nonabelian_group_order
            return lp_total_qubits(spec, group_order=reader)
    except Exception:
        return None
    return None


def structure_key(spec: Any, n_budget: int | None = None) -> str:
    """Compact human-readable region label.

    'lp_nonabelian/dihedral/12-24' for a spec without base matrices (the v3.5
    reading, unchanged); with A/B present the LP shapes are appended as
    'shapeA|shapeB' (hgp: seed shapes), and with ``n_budget`` given the
    budget-utilisation band closes the label, e.g.
    'lp_nonabelian/direct_product/24-40/1x2|1x2/util0.9-1.0'."""

    if not isinstance(spec, dict):
        return "unknown"
    descriptor = structure_descriptor(spec)
    parts = [str(spec.get("family") or "unknown"), str(descriptor.get("group_class") or "?")]
    band = descriptor.get("group_order_band")
    if band:
        parts.append(str(band))
    shape_a, shape_b = descriptor.get("shape_a"), descriptor.get("shape_b")
    if shape_a or shape_b:
        parts.append(f"{shape_a or '?'}|{shape_b or '?'}")
    if n_budget:
        util = utilisation_band(spec_total_qubits(spec), int(n_budget))
        if util:
            parts.append(util)
    return "/".join(parts)


def _max_cell_weight(matrix: Any) -> int | None:
    """Max over the rows of a group-algebra matrix of the summed cell term
    counts (the row weight the LP construction gives that block)."""

    try:
        best = None
        for row in matrix:
            total = 0
            for cell in row:
                total += len(cell)
            best = total if best is None else max(best, total)
        return best
    except TypeError:
        return None


def weight_profile(spec: Any) -> dict[str, Any]:
    """v3.7 A/B weight profile of a spec: the two block weights whose sum is
    the check weight, whether they are equal, and a short label.

    bb: |A| and |B| term counts; lp_*: max row weight of the A and B base
    matrices; hgp: max row weight of the two seed matrices. Facts for the
    prompt's symmetry table and the parent lines -- never a selection rule."""

    out: dict[str, Any] = {"w_a": None, "w_b": None, "symmetric": None, "label": None}
    if not isinstance(spec, dict):
        return out
    family = spec.get("family")
    w_a = w_b = None
    try:
        if family == "bb":
            w_a, w_b = len(spec["A"]), len(spec["B"])
        elif family in ("lp_abelian", "lp_nonabelian"):
            w_a, w_b = _max_cell_weight(spec["A"]), _max_cell_weight(spec["B"])
        elif family == "hgp":
            seeds = spec.get("seeds") or {}
            w_a = max(int(sum(row)) for row in seeds["H1"])
            w_b = max(int(sum(row)) for row in seeds["H2"])
    except (KeyError, TypeError, ValueError):
        w_a = w_b = None
    if w_a is None or w_b is None:
        return out
    out.update(
        {
            "w_a": int(w_a),
            "w_b": int(w_b),
            "symmetric": int(w_a) == int(w_b),
            "label": f"|A|={int(w_a)},|B|={int(w_b)}" if family != "hgp" else f"H1w={int(w_a)},H2w={int(w_b)}",
        }
    )
    return out
