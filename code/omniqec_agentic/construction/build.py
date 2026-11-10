"""Compile construction specs into CSS ancestors plus optional LC frames."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from omniqec_agentic.config import DeformationMode, LpShapePin
from omniqec_agentic.deformation import resolve_h_mask
from omniqec_agentic.schemas import CompiledCode

from .registry import get_family

LP_FAMILIES = ("lp_abelian", "lp_nonabelian")


def build_matrices(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    plugin = get_family(str(spec.get("family")))
    if plugin is None:
        raise ValueError("bad family")
    return plugin.build(plugin.validate(spec))


def build_compiled_code(
    spec: dict[str, Any],
    *,
    program_id: str = "",
    n_budget: int = 0,
    deformation_mode: str | DeformationMode | None = None,
    lp_shape_pin: LpShapePin | dict[str, Any] | None = None,
) -> CompiledCode:
    """Build the common code representation used by all verification stages.

    ``lp_shape_pin`` is the run-level seed-shape pin (design doc §3.5); it is
    a search-time grammar constraint, so only the fast path passes it — slow
    re-verification of existing queues rebuilds without it.
    """

    plugin = get_family(str(spec.get("family")))
    if plugin is None:
        raise ValueError("bad family")
    validated = plugin.validate(spec)
    enforce_lp_shape_pin(validated, lp_shape_pin)
    hx, hz = plugin.build(validated)
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    sector_sizes = [int(value) for value in plugin.sector_sizes(validated)]
    mode = DeformationMode(
        str(
            deformation_mode.value
            if isinstance(deformation_mode, DeformationMode)
            else (deformation_mode or DeformationMode.NONE.value)
        )
    )
    h_mask = resolve_h_mask(mode, sector_sizes=sector_sizes, n_data=int(hx.shape[1]))
    metadata = dict(
        plugin.compile_metadata(validated, hx, hz) if plugin.compile_metadata else {}
    )
    # v3.5 SEC: block structure for the structured layer colouring. Asserted
    # against the built matrices here so a wrong perm convention can never
    # reach the circuit builder; deliberately NOT in code_hash / metadata.
    layer_blocks = plugin.layer_blocks(validated) if plugin.layer_blocks else None
    _assert_layer_blocks_match(layer_blocks, hx, hz)
    return CompiledCode(
        program_id=str(program_id),
        n_budget=int(n_budget),
        hx=hx.astype(int).tolist(),
        hz=hz.astype(int).tolist(),
        family=plugin.name,
        deformation=mode.value,
        h_mask=h_mask,
        sector_sizes=sector_sizes,
        construction_metadata=metadata,
        layer_blocks=layer_blocks,
        code_hash=compiled_code_hash(hx, hz, h_mask=h_mask),
    )


def reconstruct_from_layer_blocks(
    layer_blocks: dict[str, Any], basis: str, *, n_rows: int, n_cols: int
) -> np.ndarray:
    """Rebuild H_W from its layer_blocks_v1 terms (the ONLY index convention:
    term [cb, db, perm] sets H[cb*g + perm[t], db*g + t] for t in range(g)).
    Overlapping terms raise so the caller can distinguish "wrong" from
    "XOR-cancelled"."""

    g = int(layer_blocks["block_size"])
    matrix = np.zeros((int(n_rows), int(n_cols)), dtype=np.uint8)
    for cb, db, perm in layer_blocks[basis]["terms"]:
        if sorted(int(v) for v in perm) != list(range(g)):
            raise ValueError("layer_blocks term perm is not a permutation of range(block_size)")
        rows = int(cb) * g + np.asarray(perm, dtype=np.int64)
        cols = int(db) * g + np.arange(g, dtype=np.int64)
        if np.any(matrix[rows, cols]):
            raise ValueError("layer_blocks terms overlap")
        matrix[rows, cols] = 1
    return matrix


def _assert_layer_blocks_match(
    layer_blocks: dict[str, Any] | None, hx: np.ndarray, hz: np.ndarray
) -> None:
    if layer_blocks is None:
        return
    try:
        if layer_blocks.get("schema") != "layer_blocks_v1":
            raise ValueError("unknown layer_blocks schema")
        g = int(layer_blocks["block_size"])
        for basis, h in (("X", hx), ("Z", hz)):
            block = layer_blocks[basis]
            if int(block["n_check_blocks"]) * g != int(h.shape[0]) or int(
                block["n_data_blocks"]
            ) * g != int(h.shape[1]):
                raise ValueError(f"layer_blocks {basis} block counts disagree with H shape")
            rebuilt = reconstruct_from_layer_blocks(
                layer_blocks, basis, n_rows=int(h.shape[0]), n_cols=int(h.shape[1])
            )
            if not np.array_equal(rebuilt, np.asarray(h, dtype=np.uint8) % 2):
                raise ValueError(f"basis {basis} differs")
    except Exception as exc:
        raise ValueError(f"layer_blocks do not reproduce hx/hz: {exc}") from exc


def enforce_lp_shape_pin(
    spec: dict[str, Any], pin: LpShapePin | dict[str, Any] | None
) -> None:
    if pin is None or str(spec.get("family")) not in LP_FAMILIES:
        return
    if isinstance(pin, dict):
        pin = LpShapePin(
            a=tuple(pin["a"]) if pin.get("a") else None,
            b=tuple(pin["b"]) if pin.get("b") else None,
        )
    for label, pinned in (("A", pin.a), ("B", pin.b)):
        if pinned is None:
            continue
        matrix = spec[label]
        shape = (len(matrix), len(matrix[0]))
        if shape != tuple(int(value) for value in pinned):
            raise ValueError(
                f"lp_shape_pin: base matrix {label} must be "
                f"{pinned[0]}x{pinned[1]}, got {shape[0]}x{shape[1]}"
            )


def compiled_code_hash(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    h_mask: list[int] | np.ndarray | None = None,
) -> str:
    """Historical CSS hash, extended only when a non-trivial H mask exists."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    digest = hashlib.sha256()
    digest.update(str(tuple(hx.shape)).encode("ascii"))
    digest.update(hx.tobytes())
    digest.update(str(tuple(hz.shape)).encode("ascii"))
    digest.update(hz.tobytes())
    if h_mask is not None:
        mask = np.asarray(h_mask, dtype=np.uint8).reshape(-1) % 2
        if np.any(mask):
            digest.update(b"|local_clifford_h_mask|")
            digest.update(str(tuple(mask.shape)).encode("ascii"))
            digest.update(mask.tobytes())
    return digest.hexdigest()[:24]


def symplectic_stabilizers(code: CompiledCode) -> tuple[np.ndarray, np.ndarray]:
    """Return the general stabilizer matrix components ``(S_X, S_Z)``."""

    hx = np.asarray(code.hx, dtype=np.uint8) % 2
    hz = np.asarray(code.hz, dtype=np.uint8) % 2
    mask = np.asarray(code.h_mask, dtype=np.uint8).reshape(1, -1) % 2
    keep = 1 - mask
    sx = np.vstack([hx * keep, hz * mask]).astype(np.uint8)
    sz = np.vstack([hx * mask, hz * keep]).astype(np.uint8)
    return sx, sz


def compiled_code_metadata(code: CompiledCode) -> dict[str, Any]:
    """Compact provenance shared by slow and post-search artifacts."""

    hx = np.asarray(code.hx, dtype=np.uint8)
    hz = np.asarray(code.hz, dtype=np.uint8)
    sx, sz = symplectic_stabilizers(code)
    mask = np.asarray(code.h_mask, dtype=np.uint8)
    return {
        "family": code.family,
        "canonical_family": code.family,
        "deformation": code.deformation,
        "deformation_mode": code.deformation,
        "sector_sizes": list(code.sector_sizes),
        "code_hash": code.code_hash,
        "css_ancestor_hash": compiled_code_hash(hx, hz),
        "h_mask_weight": int(mask.sum()),
        "h_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "symplectic_stabilizer_sha256": _matrix_pair_sha256(sx, sz),
        "construction_metadata": dict(code.construction_metadata),
    }


def _matrix_pair_sha256(left: np.ndarray, right: np.ndarray) -> str:
    digest = hashlib.sha256()
    for matrix in (left, right):
        normalized = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint8) % 2)
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(normalized.tobytes())
    return digest.hexdigest()
