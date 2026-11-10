"""Deformation dispatch: run-level mode -> per-qubit Hadamard mask."""

from __future__ import annotations

from omniqec_agentic.config import DeformationMode

from .sector_h import sector_h_mask


def resolve_h_mask(
    mode: str | DeformationMode | None,
    *,
    sector_sizes: list[int],
    n_data: int,
) -> list[int]:
    resolved = DeformationMode(
        str(mode.value if isinstance(mode, DeformationMode) else (mode or "none"))
    )
    if resolved is DeformationMode.NONE:
        return [0] * int(n_data)
    if resolved is DeformationMode.SECTOR_H:
        return sector_h_mask(sector_sizes, n_data)
    raise ValueError(f"unknown deformation mode '{resolved}'")
