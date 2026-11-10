"""Classical parity-check seed grammar (design doc §5) — used twice by hgp."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MIN_DEGREE = 2
DEFAULT_MAX_DEGREE = 4


def validate_classical_seed(
    value: Any,
    *,
    label: str,
    min_degree: int = DEFAULT_MIN_DEGREE,
    max_degree: int = DEFAULT_MAX_DEGREE,
) -> list[list[int]]:
    """Validate one binary seed: rectangular, degree-bounded, no duplicate rows."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty binary matrix")
    clean = [[int(item) for item in row] for row in value]
    widths = {len(row) for row in clean}
    if len(widths) != 1 or next(iter(widths)) < 1:
        raise ValueError(f"{label} must be rectangular and non-empty")
    if any(item not in (0, 1) for row in clean for item in row):
        raise ValueError(f"{label} entries must be 0 or 1")
    matrix = np.asarray(clean, dtype=np.uint8)
    row_degrees = matrix.sum(axis=1)
    col_degrees = matrix.sum(axis=0)
    if int(row_degrees.min()) < min_degree or int(row_degrees.max()) > max_degree:
        raise ValueError(f"{label} row degrees must lie in [{min_degree},{max_degree}]")
    if int(col_degrees.min()) < min_degree or int(col_degrees.max()) > max_degree:
        raise ValueError(f"{label} column degrees must lie in [{min_degree},{max_degree}]")
    if len({tuple(row) for row in clean}) != len(clean):
        raise ValueError(f"{label} must not contain duplicate rows")
    return clean
