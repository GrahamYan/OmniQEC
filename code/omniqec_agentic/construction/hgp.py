"""hgp — hypergraph product of two independent classical seeds (design doc §3.2).

Classic Tillich-Zemor construction; builder ported unchanged from the former
two-seed HGP family. Each seed independently passes the classical-seed grammar
(degree bounds, no duplicate rows); H1 == H2 is allowed, full row rank is not
required (k > 0 is checked globally after construction).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .grammars.classical_seed import validate_classical_seed


def validate_hgp_spec(spec: dict[str, Any]) -> dict[str, Any]:
    seeds = spec.get("seeds")
    if not isinstance(seeds, dict) or "H1" not in seeds or "H2" not in seeds:
        raise ValueError("hgp spec needs 'seeds': {'H1': <matrix>, 'H2': <matrix>}")
    h1 = validate_classical_seed(seeds["H1"], label="H1")
    h2 = validate_classical_seed(seeds["H2"], label="H2")
    return {"family": "hgp", "seeds": {"H1": h1, "H2": h2}}


def hgp_code(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    h1 = np.asarray(spec["seeds"]["H1"], dtype=np.uint8) % 2
    h2 = np.asarray(spec["seeds"]["H2"], dtype=np.uint8) % 2
    r1, n1 = h1.shape
    r2, n2 = h2.shape
    hx = np.hstack(
        [
            np.kron(h1, np.eye(n2, dtype=np.uint8)),
            np.kron(np.eye(r1, dtype=np.uint8), h2.T),
        ]
    ).astype(np.uint8)
    hz = np.hstack(
        [
            np.kron(np.eye(n1, dtype=np.uint8), h2),
            np.kron(h1.T, np.eye(r2, dtype=np.uint8)),
        ]
    ).astype(np.uint8)
    if int((hx @ hz.T % 2).sum()) != 0:
        raise ValueError("HGP CSS orthogonality failed")
    return hx, hz


def hgp_sector_sizes(spec: dict[str, Any]) -> list[int]:
    r1, n1 = len(spec["seeds"]["H1"]), len(spec["seeds"]["H1"][0])
    r2, n2 = len(spec["seeds"]["H2"]), len(spec["seeds"]["H2"][0])
    return [n1 * n2, r1 * r2]


def _matrix_shape(matrix: Any) -> tuple[int, int] | None:
    """(rows, cols) for a rectangular non-empty list-of-lists, else None."""

    if not isinstance(matrix, (list, tuple)) or not matrix:
        return None
    widths = {len(row) if isinstance(row, (list, tuple)) else -1 for row in matrix}
    if len(widths) != 1:
        return None
    cols = next(iter(widths))
    return (len(matrix), cols) if cols >= 1 else None


def hgp_total_qubits(spec: dict[str, Any]) -> int | None:
    """Total qubits (data + one ancilla per check row) from the spec alone.

    ``None`` for anything this cannot read — including a ragged seed, whose
    real shape only the validator can decide — so the caller falls through to
    the builder instead of guessing.
    """

    try:
        seeds = spec["seeds"]
        shape1 = _matrix_shape(seeds["H1"])
        shape2 = _matrix_shape(seeds["H2"])
    except Exception:
        return None
    if shape1 is None or shape2 is None:
        return None
    (r1, n1), (r2, n2) = shape1, shape2
    return (n1 + r1) * (n2 + r2)
