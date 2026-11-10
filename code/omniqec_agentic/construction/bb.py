"""bb — General BB (bivariate bicycle) over Z_ell x Z_m (design doc §3.1).

Two-block CSS skeleton HX = [A|B], HZ = [B^T|A^T] with A, B arbitrary sparse
bivariate polynomials; builder ported unchanged from the former gb family.
Weight grammar 4 <= w_A + w_B <= 9; the validator auto-canonicalizes ell <= m
by swapping the torus axes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

BB_MIN_TOTAL_WEIGHT = 4
BB_MAX_TOTAL_WEIGHT = 9


def validate_bb_spec(spec: dict[str, Any]) -> dict[str, Any]:
    ell, m = int(spec["ell"]), int(spec["m"])
    if ell < 1 or m < 1:
        raise ValueError("ell and m must be positive")
    a_terms = [(int(i) % ell, int(j) % m) for i, j in spec["A"]]
    b_terms = [(int(i) % ell, int(j) % m) for i, j in spec["B"]]
    if not a_terms or not b_terms:
        raise ValueError("A and B must be non-empty")
    for label, terms in (("A", a_terms), ("B", b_terms)):
        if len(set(terms)) != len(terms):
            raise ValueError(f"{label} contains duplicate terms after modular reduction")
    weight = len(a_terms) + len(b_terms)
    if not BB_MIN_TOTAL_WEIGHT <= weight <= BB_MAX_TOTAL_WEIGHT:
        raise ValueError(
            f"total weight w_A + w_B = {weight} outside "
            f"[{BB_MIN_TOTAL_WEIGHT},{BB_MAX_TOTAL_WEIGHT}]"
        )
    canonicalized = bool(spec.get("canonicalized"))
    if ell > m:
        ell, m = m, ell
        a_terms = [(j, i) for i, j in a_terms]
        b_terms = [(j, i) for i, j in b_terms]
        canonicalized = True
    result = {
        "family": "bb",
        "ell": ell,
        "m": m,
        "A": [list(term) for term in a_terms],
        "B": [list(term) for term in b_terms],
    }
    if canonicalized:
        result["canonicalized"] = True
    return result


def _shift_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.uint8)
    for index in range(size):
        matrix[index, (index + 1) % size] = 1
    return matrix


def _polynomial_matrix(
    terms: Iterable[tuple[int, int]], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    matrix = np.zeros(x.shape, dtype=np.uint8)
    for first, second in terms:
        xa = np.linalg.matrix_power(x, int(first)) % 2
        yb = np.linalg.matrix_power(y, int(second)) % 2
        matrix = (matrix + (xa @ yb) % 2) % 2
    return matrix


def bb_code(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ell, m = int(spec["ell"]), int(spec["m"])
    x = np.kron(_shift_matrix(ell), np.eye(m, dtype=np.uint8)) % 2
    y = np.kron(np.eye(ell, dtype=np.uint8), _shift_matrix(m)) % 2
    a_matrix = _polynomial_matrix([tuple(term) for term in spec["A"]], x, y)
    b_matrix = _polynomial_matrix([tuple(term) for term in spec["B"]], x, y)
    hx = np.hstack([a_matrix, b_matrix]).astype(np.uint8)
    hz = np.hstack([b_matrix.T, a_matrix.T]).astype(np.uint8)
    if int((hx @ hz.T % 2).sum()) != 0:
        raise ValueError("BB CSS orthogonality failed")
    return hx, hz


def bb_sector_sizes(spec: dict[str, Any]) -> list[int]:
    size = int(spec["ell"]) * int(spec["m"])
    return [size, size]


def _bb_perm_forward(a: int, b: int, ell: int, m: int) -> list[int]:
    """perm of the monomial x^a y^b: M[(i,j), ((i+a)%ell, (j+b)%m)] = 1, so the
    data-local column t = i*m + j is hit by check-local row
    (((i - a) % ell) * m + (j - b) % m)."""

    return [
        (((t // m) - a) % ell) * m + (((t % m) - b) % m) for t in range(ell * m)
    ]


def _bb_perm_transpose(a: int, b: int, ell: int, m: int) -> list[int]:
    """perm of (x^a y^b)^T: row t maps from column ((i+a)%ell, (j+b)%m)."""

    return [
        (((t // m) + a) % ell) * m + (((t % m) + b) % m) for t in range(ell * m)
    ]


def bb_layer_blocks(spec: dict[str, Any]) -> dict[str, Any]:
    """Block structure of HX = [A|B], HZ = [B^T|A^T] for the v3.5 SEC layer
    colouring (schemas.CompiledCode.layer_blocks, schema layer_blocks_v1).

    Call on the VALIDATED spec (ell <= m canonicalised, exactly what
    ``bb_code`` builds from). One check block (g = ell*m rows), two data
    blocks (left = A columns, right = B columns); every monomial is one
    permutation term. Local index t = i*m + j, row-major, matching
    ``np.kron(_shift_matrix(ell), I_m)``. Cost O(w*g)."""

    ell, m = int(spec["ell"]), int(spec["m"])
    g = ell * m
    a_terms = [(int(a) % ell, int(b) % m) for a, b in spec["A"]]
    b_terms = [(int(a) % ell, int(b) % m) for a, b in spec["B"]]
    x_terms = [[0, 0, _bb_perm_forward(a, b, ell, m)] for a, b in a_terms]
    x_terms += [[0, 1, _bb_perm_forward(a, b, ell, m)] for a, b in b_terms]
    z_terms = [[0, 0, _bb_perm_transpose(a, b, ell, m)] for a, b in b_terms]
    z_terms += [[0, 1, _bb_perm_transpose(a, b, ell, m)] for a, b in a_terms]
    return {
        "schema": "layer_blocks_v1",
        "block_size": g,
        "X": {"n_check_blocks": 1, "n_data_blocks": 2, "terms": x_terms},
        "Z": {"n_check_blocks": 1, "n_data_blocks": 2, "terms": z_terms},
    }


def bb_total_qubits(spec: dict[str, Any]) -> int | None:
    """Total qubits (data + one ancilla per check row) from the spec alone.

    ``None`` for anything this cannot read; the caller must then fall through
    to the real builder rather than guess. The validator's ell <= m
    canonicalization swaps the axes, which leaves the product untouched.
    """

    try:
        ell, m = int(spec["ell"]), int(spec["m"])
    except Exception:
        return None
    if ell < 1 or m < 1:
        return None
    return 4 * ell * m
