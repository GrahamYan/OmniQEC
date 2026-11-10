"""Group-algebra base-matrix seed grammar (design doc §5) — shared by both LP
families (bb keeps an equivalent specialized polynomial validator).

A seed matrix over F_2[G] is a rectangular cell matrix whose shape comes from
the allowed pool; each cell is a non-empty, duplicate-free list of exponent
tuples (F_2 coefficients: a repeated element would cancel itself). The joint
support of A and B must generate the whole group so the lift never degenerates
to a subgroup (this also gives a clearer failure than the downstream
Tanner-connectivity check).
"""

from __future__ import annotations

from typing import Any, Sequence

from ..groups.base import ExponentTuple, Group

# v1.1 shape pool — A and B choose independently; 2x2 opened by user decision.
DEFAULT_ALLOWED_SHAPES: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2))
# Loose per-cell sanity cap; the real arbiter is the compiled per-check weight
# <= 9 (structural check). 8 = the largest cell weight any valid 1x1 code can
# use (w_a + w_b <= 9 with both cells non-empty), e.g. Lin-Pryadko 2BGA codes
# with wgt(a)=2, wgt(b)=6.
DEFAULT_MAX_CELL_WEIGHT = 8

Cells = list[list[list[ExponentTuple]]]


def validate_ga_matrix(
    value: Any,
    *,
    group: Group,
    label: str,
    allowed_shapes: Sequence[tuple[int, int]] = DEFAULT_ALLOWED_SHAPES,
    max_cell_weight: int = DEFAULT_MAX_CELL_WEIGHT,
) -> Cells:
    """Validate one base matrix; returns cells of normalized exponent tuples."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty cell matrix")
    widths = {len(row) if isinstance(row, (list, tuple)) else -1 for row in value}
    if len(widths) != 1 or next(iter(widths)) < 1:
        raise ValueError(f"{label} must be rectangular and non-empty")
    shape = (len(value), next(iter(widths)))
    if shape not in {tuple(entry) for entry in allowed_shapes}:
        pool = ", ".join(f"{r}x{c}" for r, c in allowed_shapes)
        raise ValueError(f"{label} shape {shape[0]}x{shape[1]} not in allowed pool {{{pool}}}")

    cells: Cells = []
    for row_index, row in enumerate(value):
        clean_row: list[list[ExponentTuple]] = []
        for col_index, cell in enumerate(row):
            where = f"{label}[{row_index}][{col_index}]"
            if not isinstance(cell, (list, tuple)) or not cell:
                raise ValueError(f"{where} must be a non-empty list of exponent tuples")
            if len(cell) > max_cell_weight:
                raise ValueError(f"{where} weight {len(cell)} exceeds cap {max_cell_weight}")
            normalized = [group.normalize_tuple(entry) for entry in cell]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{where} contains duplicate elements after reduction")
            clean_row.append(normalized)
        cells.append(clean_row)
    return cells


def support_indices(group: Group, *matrices: Cells) -> set[int]:
    return {
        group.index[element]
        for cells in matrices
        for row in cells
        for cell in row
        for element in cell
    }


def require_supports_generate(group: Group, *matrices: Cells) -> None:
    support = support_indices(group, *matrices)
    if not group.generates(support):
        subgroup = len(group.subgroup_closure(support))
        raise ValueError(
            f"supports of A and B generate a proper subgroup "
            f"(order {subgroup} of {group.order}) of '{group.name}'"
        )
