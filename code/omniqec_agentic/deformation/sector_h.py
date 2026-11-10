"""sector_h — swap the X/Z axes of every qubit in the second natural sector.

Keeps the CSS ancestor HX/HZ; the mask marks the data qubits whose physical
stabilizer axes are exchanged. All four canonical families export a natural
two-sector split, so this mode is family-agnostic by construction.
"""

from __future__ import annotations


def sector_h_mask(sector_sizes: list[int], n_data: int) -> list[int]:
    sizes = [int(value) for value in sector_sizes]
    if len(sizes) != 2 or sum(sizes) != int(n_data):
        raise ValueError("sector_h requires two natural sectors matching N_data")
    if sizes[1] <= 0:
        raise ValueError("sector_h requires a non-empty second natural sector")
    return [0] * sizes[0] + [1] * sizes[1]
