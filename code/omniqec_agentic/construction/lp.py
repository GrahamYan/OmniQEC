"""Small-seed lifted product over F_2[G] (design doc §3.3/§3.4).

One grammar and one builder shared by lp_abelian and lp_nonabelian — only the
group source differs, which is exactly the controlled variable of the Family
3 vs 4 experiment. The lift math is ported unchanged from the former glp
family (A-side lifts via the left regular representation, B-side via the
right; L(s) and R(t) commute for any group, so

    HX = [ L(A ⊗ I_nB) | R(I_mA ⊗ B*) ]
    HZ = [ R(I_nA ⊗ B) | L(A* ⊗ I_mB) ]

with * = protograph transpose + elementwise antipode is CSS-orthogonal for
any finite group, any shapes, any entries).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .grammars.ga_seed import require_supports_generate, validate_ga_matrix
from .groups.abelian import abelian_group, invariant_factors
from .groups.base import Group
from .groups.catalog import catalog_order, get_group

IndexCells = list[list[list[int]]]

GroupResolver = Callable[[dict[str, Any]], Group]


def resolve_abelian_group(spec: dict[str, Any]) -> Group:
    group_field = spec.get("group")
    if not isinstance(group_field, dict) or "factors" not in group_field:
        raise ValueError("lp_abelian spec needs 'group': {'factors': [q1, q2, ...]}")
    return abelian_group(tuple(int(q) for q in group_field["factors"]))


def resolve_nonabelian_group(spec: dict[str, Any]) -> Group:
    name = spec.get("group")
    if not isinstance(name, str) or not name:
        raise ValueError("lp_nonabelian spec needs 'group': '<catalog name>'")
    return get_group(name)


def validate_lp_spec(
    spec: dict[str, Any],
    *,
    family: str,
    resolve_group: GroupResolver,
) -> dict[str, Any]:
    group = resolve_group(spec)
    a_cells = validate_ga_matrix(spec.get("A"), group=group, label="A")
    b_cells = validate_ga_matrix(spec.get("B"), group=group, label="B")
    require_supports_generate(group, a_cells, b_cells)
    if family == "lp_nonabelian" and group.is_abelian:
        raise ValueError(f"group '{group.name}' is abelian; use lp_abelian")
    group_field: Any
    if group.kind == "abelian":
        group_field = {"factors": [int(q) for q in group.generator_orders]}
    else:
        group_field = group.name  # canonical name (e.g. normalized metacyclic twist)
    return {
        "family": family,
        "group": group_field,
        "A": [[[list(entry) for entry in cell] for cell in row] for row in a_cells],
        "B": [[[list(entry) for entry in cell] for cell in row] for row in b_cells],
    }


def lp_group_metadata(spec: dict[str, Any], *, resolve_group: GroupResolver) -> dict[str, Any]:
    """Group provenance recorded on the compiled code (design doc §4.2)."""

    group = resolve_group(spec)
    metadata: dict[str, Any] = {"group": spec.get("group"), "group_order": group.order}
    if group.kind == "abelian":
        metadata["invariant_factors"] = list(invariant_factors(group.generator_orders))
    if group.gap_id is not None:
        metadata["gap_smallgroup_id"] = list(group.gap_id)
    return metadata


def _index_cells(cells: list, group: Group) -> IndexCells:
    return [
        [[group.element_index(entry) for entry in cell] for cell in row] for row in cells
    ]


def _star(cells: IndexCells, group: Group) -> IndexCells:
    """Protograph transpose + elementwise antipode (g -> g^-1)."""

    rows, cols = len(cells), len(cells[0])
    return [
        [[int(group.inverse[g]) for g in cells[i][j]] for i in range(rows)]
        for j in range(cols)
    ]


def _kron_right(cells: IndexCells, repeat: int) -> IndexCells:
    """proto ⊗ I_repeat at the protograph level."""

    rows, cols = len(cells), len(cells[0])
    out: IndexCells = [[[] for _ in range(cols * repeat)] for _ in range(rows * repeat)]
    for i in range(rows):
        for j in range(cols):
            for t in range(repeat):
                out[i * repeat + t][j * repeat + t] = list(cells[i][j])
    return out


def _kron_left(repeat: int, cells: IndexCells) -> IndexCells:
    """I_repeat ⊗ proto at the protograph level."""

    rows, cols = len(cells), len(cells[0])
    out: IndexCells = [[[] for _ in range(repeat * cols)] for _ in range(repeat * rows)]
    for t in range(repeat):
        for i in range(rows):
            for j in range(cols):
                out[t * rows + i][t * cols + j] = list(cells[i][j])
    return out


def _lift(cells: IndexCells, group: Group, side: str) -> np.ndarray:
    size = group.order
    rows, cols = len(cells), len(cells[0])
    out = np.zeros((rows * size, cols * size), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if cells[i][j]:
                block = (
                    group.left_regular(cells[i][j])
                    if side == "L"
                    else group.right_regular(cells[i][j])
                )
                out[i * size : (i + 1) * size, j * size : (j + 1) * size] = block
    return out


def lp_code(
    spec: dict[str, Any], *, resolve_group: GroupResolver
) -> tuple[np.ndarray, np.ndarray]:
    group = resolve_group(spec)
    a_cells = _index_cells(spec["A"], group)
    b_cells = _index_cells(spec["B"], group)
    m_a, n_a = len(a_cells), len(a_cells[0])
    m_b, n_b = len(b_cells), len(b_cells[0])
    a_star = _star(a_cells, group)
    b_star = _star(b_cells, group)
    hx = np.hstack(
        [
            _lift(_kron_right(a_cells, n_b), group, "L"),
            _lift(_kron_left(m_a, b_star), group, "R"),
        ]
    ).astype(np.uint8)
    hz = np.hstack(
        [
            _lift(_kron_left(n_a, b_cells), group, "R"),
            _lift(_kron_right(a_star, m_b), group, "L"),
        ]
    ).astype(np.uint8)
    if int((hx @ hz.T % 2).sum()) != 0:
        raise ValueError("LP CSS orthogonality failed")
    return hx, hz


def _block_terms(
    cells: IndexCells, group: Group, side: str, *, data_offset: int
) -> list[list[Any]]:
    """One (check_block, data_block, perm) term per member of every kron'd cell.

    ``left_regular`` sets matrix[cayley[member, :], arange] (row g*h for column
    h), so side "L" has perm = cayley[member, :]; ``right_regular`` sets
    matrix[cayley[:, member], arange] (row h*g), so side "R" has perm =
    cayley[:, member]. Distinct members of one cell are distinct permutations
    with disjoint supports (the XOR in ``_lift`` is a plain sum), i.e. parallel
    edges of the block multigraph that never overlap as matrices."""

    terms: list[list[Any]] = []
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            for member in cell:
                gi = int(member)
                perm = (
                    group.cayley[gi, :].tolist() if side == "L" else group.cayley[:, gi].tolist()
                )
                terms.append([int(i), int(data_offset + j), [int(v) for v in perm]])
    return terms


def lp_layer_blocks(spec: dict[str, Any], *, resolve_group: GroupResolver) -> dict[str, Any]:
    """Block structure of the lifted product for the v3.5 SEC layer colouring
    (schemas.CompiledCode.layer_blocks, schema layer_blocks_v1).

    Reuses exactly the four protograph arguments of ``lp_code``: hx =
    [L(A x I_nB) | R(I_mA x B*)], hz = [R(I_nA x B) | L(A* x I_mB)]. Data
    blocks are global: the left half's blocks come first, the right half's
    start at n_a*n_b for BOTH bases (consistent with ``lp_sector_sizes``).
    Cost O(w*|G|) -- no |G|x|G| matrices are formed."""

    group = resolve_group(spec)
    a_cells = _index_cells(spec["A"], group)
    b_cells = _index_cells(spec["B"], group)
    m_a, n_a = len(a_cells), len(a_cells[0])
    m_b, n_b = len(b_cells), len(b_cells[0])
    a_star = _star(a_cells, group)
    b_star = _star(b_cells, group)
    left_blocks = n_a * n_b
    x_terms = _block_terms(_kron_right(a_cells, n_b), group, "L", data_offset=0)
    x_terms += _block_terms(_kron_left(m_a, b_star), group, "R", data_offset=left_blocks)
    z_terms = _block_terms(_kron_left(n_a, b_cells), group, "R", data_offset=0)
    z_terms += _block_terms(_kron_right(a_star, m_b), group, "L", data_offset=left_blocks)
    n_data_blocks = n_a * n_b + m_a * m_b
    return {
        "schema": "layer_blocks_v1",
        "block_size": int(group.order),
        "X": {"n_check_blocks": m_a * n_b, "n_data_blocks": n_data_blocks, "terms": x_terms},
        "Z": {"n_check_blocks": n_a * m_b, "n_data_blocks": n_data_blocks, "terms": z_terms},
    }


def lp_sector_sizes(spec: dict[str, Any], *, resolve_group: GroupResolver) -> list[int]:
    group = resolve_group(spec)
    m_a, n_a = len(spec["A"]), len(spec["A"][0])
    m_b, n_b = len(spec["B"]), len(spec["B"][0])
    return [n_a * n_b * group.order, m_a * m_b * group.order]


def abelian_group_order(spec: dict[str, Any]) -> int | None:
    """|G| for an lp_abelian spec without building the group."""

    group_field = spec.get("group")
    if not isinstance(group_field, dict):
        return None
    try:
        factors = [int(q) for q in group_field["factors"]]
    except Exception:
        return None
    if not factors or any(q < 2 for q in factors):
        return None
    order = 1
    for q in factors:
        order *= q
    return order


def nonabelian_group_order(spec: dict[str, Any]) -> int | None:
    """|G| for an lp_nonabelian spec from the catalog name alone."""

    name = spec.get("group")
    return catalog_order(name) if isinstance(name, str) and name else None


def lp_total_qubits(
    spec: dict[str, Any], *, group_order: Callable[[dict[str, Any]], int | None]
) -> int | None:
    """``(nA+mA)*(nB+mB)*|G|`` — total qubits, from the spec alone.

    This is the same inequality the generator prompt states and the same total
    ``verification.structural`` measures after building; it is predicted here
    so an over-budget candidate is rejected before the lift (which is
    ``|G|``-by-``|G|`` blocks) is ever formed. ``None`` for anything unreadable
    — a shape the validator would reject, or a group name outside the catalog
    grammar — so the builder still produces the real diagnostic.
    """

    order = group_order(spec)
    if not order:
        return None
    shape = 1
    for key in ("A", "B"):
        matrix = spec.get(key)
        if not isinstance(matrix, (list, tuple)) or not matrix:
            return None
        widths = {len(row) if isinstance(row, (list, tuple)) else -1 for row in matrix}
        if len(widths) != 1:
            return None
        cols = next(iter(widths))
        if cols < 1:
            return None
        shape *= len(matrix) + cols
    return shape * order
