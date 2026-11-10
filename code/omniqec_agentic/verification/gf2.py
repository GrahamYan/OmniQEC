"""GF(2) linear algebra utilities matching the ver1 fast path."""

from __future__ import annotations

import numpy as np


def rank2(matrix: np.ndarray) -> int:
    mat = matrix.copy() % 2
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if mat[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return int(rank)


def rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mat = matrix.copy() % 2
    rows, cols = mat.shape
    pivots: list[int] = []
    rank = 0
    for col in range(cols):
        selected = None
        for row in range(rank, rows):
            if mat[row, col]:
                selected = row
                break
        if selected is None:
            continue
        mat[[rank, selected]] = mat[[selected, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        pivots.append(col)
        rank += 1
    return mat[:rank], pivots


def kernel_basis(check_matrix: np.ndarray) -> np.ndarray:
    reduced, pivots = rref(check_matrix)
    n = check_matrix.shape[1]
    free = [col for col in range(n) if col not in pivots]
    basis = []
    for free_col in free:
        vector = np.zeros(n, np.uint8)
        vector[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            if reduced[row, free_col]:
                vector[pivot_col] = 1
        basis.append(vector)
    return np.array(basis, np.uint8) if basis else np.zeros((0, n), np.uint8)


def logicals(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = hx.shape[1]
    kernel_x = kernel_basis(hx)
    hz_reduced, _ = rref(hz)

    basis = [row.copy() for row in hz_reduced]
    normalized_basis = []
    for basis_row in basis:
        candidate = basis_row.copy()
        for existing in normalized_basis:
            lead = np.argmax(existing)
            if candidate[lead]:
                candidate ^= existing
        if candidate.any():
            normalized_basis.append(candidate)

    basis = normalized_basis
    lz = []
    for vector in kernel_x:
        candidate = vector.copy()
        for existing in basis:
            lead = np.argmax(existing)
            if candidate[lead]:
                candidate ^= existing
        if candidate.any():
            basis.append(candidate)
            lz.append(vector.copy())

    lz_array = np.array(lz, np.uint8).reshape(-1, n)
    if lz_array.shape[0] == 0:
        empty = np.zeros((0, n), np.uint8)
        return empty, empty

    kernel_z = kernel_basis(hz)
    pairing = (kernel_z.astype(int) @ lz_array.T) % 2
    system = pairing.T % 2
    lx = []
    for logical_index in range(lz_array.shape[0]):
        target = np.zeros(system.shape[0], np.uint8)
        target[logical_index] = 1
        augmented = np.concatenate([system.copy(), target[:, None]], 1) % 2
        reduced, _ = rref(augmented)
        coefficients = np.zeros(system.shape[1], np.uint8)
        for row in range(reduced.shape[0]):
            cols = np.where(reduced[row, :-1])[0]
            if len(cols) == 0:
                continue
            coefficients[cols[0]] = reduced[row, -1]
        lx.append((coefficients @ kernel_z) % 2)
    return np.array(lx, np.uint8), lz_array

