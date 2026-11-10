"""Stim detector-error-model helpers for circuit-level observable decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse


@dataclass(frozen=True)
class DemMatrices:
    """Sparse check/logical matrices plus priors extracted from a Stim DEM."""

    H: scipy.sparse.csr_matrix
    L: scipy.sparse.csr_matrix
    probs: np.ndarray
    num_detectors: int
    num_observables: int
    num_error_mechanisms: int
    skipped_detectorless_count: int = 0
    skipped_detectorless_observable_count: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "num_detectors": int(self.num_detectors),
            "num_observables": int(self.num_observables),
            "num_error_mechanisms": int(self.num_error_mechanisms),
            "num_matrix_columns": int(self.H.shape[1]),
            "num_nonzero_H": int(self.H.nnz),
            "num_nonzero_L": int(self.L.nnz),
            "skipped_detectorless_count": int(self.skipped_detectorless_count),
            "skipped_detectorless_observable_count": int(
                self.skipped_detectorless_observable_count
            ),
        }


@dataclass(frozen=True)
class DemSampleBatch:
    syndromes: np.ndarray
    observables: np.ndarray
    error_mechanisms: np.ndarray


def load_detector_error_model(path: str | Path):
    """Load and flatten a Stim detector error model with optional dependency import."""

    try:
        import stim
    except Exception as exc:  # pragma: no cover - depends on local optional deps
        raise RuntimeError(f"stim unavailable: {exc}") from exc
    return stim.DetectorErrorModel.from_file(str(path)).flattened()


def dem_to_matrices(
    dem,
    *,
    include_detectorless_observables: bool = True,
) -> DemMatrices:
    """Convert a Stim DEM into sparse H, sparse L, and per-mechanism priors.

    The external decoder examples use this same H/L/prob construction pattern.
    We keep detectorless logical-only mechanisms by default because they are real
    circuit-level logical failures with no syndrome information.
    """

    h_rows: list[int] = []
    h_cols: list[int] = []
    l_rows: list[int] = []
    l_cols: list[int] = []
    probs: list[float] = []
    var_idx = 0
    skipped_detectorless = 0
    skipped_detectorless_observable = 0

    for instr in dem:
        if getattr(instr, "type", None) != "error":
            continue

        args = instr.args_copy()
        if not args:
            continue
        error_prob = float(args[0])
        detectors: list[int] = []
        observables: list[int] = []

        for target in instr.targets_copy():
            if target.is_separator():
                raise ValueError(
                    "DEM component separators are unsupported by dem_to_matrices; "
                    "use the canonical undecomposed BP-family DEM"
                )
            if target.is_logical_observable_id():
                observables.append(int(target.val))
            else:
                detectors.append(int(target.val))

        if not detectors and not include_detectorless_observables:
            skipped_detectorless += 1
            if observables:
                skipped_detectorless_observable += 1
            continue
        if not detectors and not observables:
            skipped_detectorless += 1
            continue

        for det in detectors:
            h_rows.append(det)
            h_cols.append(var_idx)
        for obs in observables:
            l_rows.append(obs)
            l_cols.append(var_idx)

        probs.append(error_prob)
        var_idx += 1

    num_detectors = int(dem.num_detectors)
    num_observables = int(dem.num_observables)
    H = scipy.sparse.csr_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(num_detectors, var_idx),
        dtype=np.uint8,
    )
    L = scipy.sparse.csr_matrix(
        (np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
        shape=(num_observables, var_idx),
        dtype=np.uint8,
    )
    return DemMatrices(
        H=H,
        L=L,
        probs=np.asarray(probs, dtype=np.float64),
        num_detectors=num_detectors,
        num_observables=num_observables,
        num_error_mechanisms=int(sum(1 for instr in dem if getattr(instr, "type", None) == "error")),
        skipped_detectorless_count=skipped_detectorless,
        skipped_detectorless_observable_count=skipped_detectorless_observable,
    )


def dem_file_to_matrices(
    path: str | Path,
    *,
    include_detectorless_observables: bool = True,
) -> DemMatrices:
    dem = load_detector_error_model(path)
    return dem_to_matrices(
        dem,
        include_detectorless_observables=include_detectorless_observables,
    )


def syndrome_from_error(matrices: DemMatrices, error: np.ndarray) -> np.ndarray:
    value = matrices.H @ np.asarray(error, dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8).reshape(-1) % 2


def observables_from_error(matrices: DemMatrices, error: np.ndarray) -> np.ndarray:
    value = matrices.L @ np.asarray(error, dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8).reshape(-1) % 2


def sample_error(matrices: DemMatrices, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error = (rng.random(matrices.probs.size) < matrices.probs).astype(np.uint8)
    return syndrome_from_error(matrices, error), observables_from_error(matrices, error), error


def sample_error_batch(
    matrices: DemMatrices,
    rng: np.random.Generator,
    n: int,
) -> DemSampleBatch:
    errors = (rng.random((int(n), matrices.probs.size)) < matrices.probs).astype(np.uint8)
    syndromes = (errors @ matrices.H.T) % 2
    observables = (errors @ matrices.L.T) % 2
    return DemSampleBatch(
        syndromes=np.asarray(syndromes, dtype=np.uint8),
        observables=np.asarray(observables, dtype=np.uint8),
        error_mechanisms=errors,
    )
