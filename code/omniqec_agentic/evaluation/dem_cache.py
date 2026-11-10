"""DEM bundle cache: build once per (candidate, noise point), decode many
(03 §7). The pickle keeps raw float64 probabilities EXACTLY (not the text DEM
format, which rounds at ~1e-17 and would break bit-reproducibility)."""

from __future__ import annotations

import os
import pickle
from typing import Any

from omniqec_agentic.verification.dem import dem_to_matrices


def dem_bundle_path(
    dem_dir: str,
    *,
    family: str,
    candidate_hash: str,
    noise_hash: str,
    noise_index: int,
) -> str:
    return os.path.join(
        dem_dir,
        f"dem_{family}_{candidate_hash}_{noise_hash}_{noise_index}.pkl",
    )


def dump_dem_bundle(
    path: str, pair: Any, noise_point: Any, *, extra: dict[str, Any] | None = None
) -> None:
    """Serialize one MemoryExperimentPair into the decode-job bundle.
    ``extra`` (v3.6): additional top-level keys, e.g. ``telescoping`` -- the
    materialized telescoping-decoder systems of both bases."""

    bundle = {
        "X": dem_to_matrices(pair.x_memory.detector_error_model),
        "Z": dem_to_matrices(pair.z_memory.detector_error_model),
        "meta": {"X": pair.x_memory.to_metadata(), "Z": pair.z_memory.to_metadata()},
        "noise_point": noise_point.to_dict(),
        # Noisy circuits for exact circuit-level sampling in the shard jobs
        # (protocol change 2026-08-15: sampling_source = noisy_circuit).
        "circuit_text": {
            "X": str(pair.x_memory.circuit),
            "Z": str(pair.z_memory.circuit),
        },
        # v3.6 (docs/06 §1.1): per-detector check kind ("X"/"Z") recorded by
        # the SEC emitter; the telescoping decoder's GARI X/Z mask (cross-
        # checked against the package's own circuit-derived mask).
        "detector_kinds": {
            "X": list(getattr(pair.x_memory, "detector_kinds", ()) or ()),
            "Z": list(getattr(pair.z_memory, "detector_kinds", ()) or ()),
        },
    }
    if extra:
        bundle.update(dict(extra))
    with open(path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dem_bundle(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    # Pre-v3.6 bundles carry no detector kinds: expose the key as None so
    # callers can tell "unknown" from "recorded" (the telescoping decoder
    # refuses to guess; BPOSD-family decoders never look). ``telescoping``
    # (materialized systems) is None when the build did not write them --
    # the shard then builds the system from the circuit text itself.
    if isinstance(bundle, dict):
        bundle.setdefault("detector_kinds", None)
        bundle.setdefault("telescoping", None)
    return bundle
