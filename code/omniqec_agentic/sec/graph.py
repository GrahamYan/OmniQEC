"""Sector partition: HX/HZ + natural two-sector metadata -> four subgraphs.

The left/right partition comes from the family plugin's sector_sizes export
(design doc 01 §7 / 02 §2.1): left = first sector, right = second. SBB proved
LRC validity holds for ANY partition of ANY CSS code; the natural product
split is what achieves the low depth of §2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SectorGraphs:
    """The four bipartite subgraphs of the tripartite Tanner graph.

    Matrices keep GLOBAL indices implicitly: rows are check indices within
    their own basis (0..mX-1 / 0..mZ-1); columns of L* are data qubits
    0..left-1, columns of R* are data qubits left..n-1 (offset restored via
    ``left``).
    """

    n_data: int
    left: int
    lx: np.ndarray  # mX x left
    rx: np.ndarray  # mX x right
    lz: np.ndarray  # mZ x left
    rz: np.ndarray  # mZ x right

    @property
    def m_x(self) -> int:
        return int(self.lx.shape[0])

    @property
    def m_z(self) -> int:
        return int(self.lz.shape[0])


def split_sectors(
    hx: np.ndarray, hz: np.ndarray, sector_sizes: list[int]
) -> SectorGraphs:
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n = int(hx.shape[1])
    if len(sector_sizes) != 2 or int(sector_sizes[0]) + int(sector_sizes[1]) != n:
        raise ValueError("LRC needs two natural sectors matching N_data")
    left = int(sector_sizes[0])
    return SectorGraphs(
        n_data=n,
        left=left,
        lx=hx[:, :left],
        rx=hx[:, left:],
        lz=hz[:, :left],
        rz=hz[:, left:],
    )
