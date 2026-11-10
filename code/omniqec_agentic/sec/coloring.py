"""Deterministic minimal bipartite edge coloring + counter-driven permutations.

König: a bipartite (multi)graph has chromatic index exactly Δ. We use the
classical alternating-path (Kempe chain) algorithm: process edges in a fixed
canonical order (check index, then data index); give each edge the lowest
color free at both endpoints, flipping an alternating a/b path when no common
free color exists. Zero randomness: identical input -> identical coloring
(design doc 02 §2.3).

Candidate schedules permute the color classes of each subgraph. Permutations
are drawn from a splitmix64 counter stream keyed by (candidate index,
subgraph slot) — deterministic and reproducible, but spread across the
permutation space instead of clustering like lexicographic enumeration
(design doc 02 §3.1).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Edge = tuple[int, int]  # (check_index, data_column) — data column LOCAL to the subgraph


def minimal_edge_coloring(matrix: np.ndarray) -> list[list[Edge]]:
    """Color the bipartite graph of a binary matrix into Δ classes."""

    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    rows, cols = matrix.shape
    edges = [(int(r), int(c)) for r in range(rows) for c in np.nonzero(matrix[r])[0]]
    if not edges:
        return []
    degree = max(
        int(matrix.sum(axis=1).max(initial=0)), int(matrix.sum(axis=0).max(initial=0))
    )
    # color_at[side][node][color] -> partner node or -1
    check_color = -np.ones((rows, degree), dtype=np.int64)
    data_color = -np.ones((cols, degree), dtype=np.int64)

    def free_color(table: np.ndarray, node: int) -> int:
        for color in range(degree):
            if table[node, color] < 0:
                return color
        raise AssertionError("no free color — Δ computation broken")

    for check, qubit in edges:
        a = free_color(check_color, check)
        b = free_color(data_color, qubit)
        if a != b:
            # Walk the maximal alternating a/b path starting from `qubit`
            # along color a and flip it (a<->b). This frees color a at
            # `qubit`; the path cannot reach `check` (it would close an odd
            # two-colored cycle, impossible in a bipartite graph). Collect
            # first, flip after — flipping in place corrupts the next-hop
            # pointers.
            path: list[tuple[int, int, int]] = []  # (data_node, check_node, color)
            side_is_data, node, color = True, qubit, a
            for _ in range(2 * (rows + cols) + 2):
                table = data_color if side_is_data else check_color
                partner = int(table[node, color])
                if partner < 0:
                    break
                path.append((node, partner, color) if side_is_data else (partner, node, color))
                side_is_data, node = not side_is_data, partner
                color = b if color == a else a
            else:
                raise AssertionError("alternating path exceeded graph size")
            for data_node, check_node, edge_color in path:
                data_color[data_node, edge_color] = -1
                check_color[check_node, edge_color] = -1
            for data_node, check_node, edge_color in path:
                other = b if edge_color == a else a
                data_color[data_node, other] = check_node
                check_color[check_node, other] = data_node
            if check_color[check, a] >= 0:  # bipartite guarantee violated
                raise AssertionError("Kempe flip reached the anchor check")
        check_color[check, a] = qubit
        data_color[qubit, a] = check

    classes: list[list[Edge]] = [[] for _ in range(degree)]
    for check in range(rows):
        for color in range(degree):
            qubit = int(check_color[check, color])
            if qubit >= 0:
                classes[color].append((check, qubit))
    for group in classes:
        group.sort()
    return classes


def splitmix64(state: int) -> tuple[int, int]:
    """One splitmix64 step: returns (new_state, output). Pure integer math."""

    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def counter_permutation(size: int, *, candidate: int, slot: int) -> list[int]:
    """Deterministic permutation of range(size) for (candidate, subgraph slot).

    Candidate 0 is the canonical identity ordering by construction.
    """

    if candidate == 0 or size <= 1:
        return list(range(size))
    order = list(range(size))
    state = (candidate << 8) ^ slot ^ 0xA5A5A5A5A5A5A5A5
    for i in range(size - 1, 0, -1):  # Fisher-Yates on the splitmix stream
        state, draw = splitmix64(state)
        j = draw % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def permuted_coloring(
    classes: list[list[Edge]], *, candidate: int, slot: int
) -> list[list[Edge]]:
    order = counter_permutation(len(classes), candidate=candidate, slot=slot)
    return [classes[i] for i in order]


def color_bipartite_multigraph(
    edges: "Sequence[tuple[int, int]]", *, n_left: int, n_right: int
) -> list[int]:
    """Minimal edge colouring of a bipartite MULTIgraph (parallel edges allowed).

    Same alternating-path (Kempe chain) algorithm as ``minimal_edge_coloring``
    but the (node, colour) tables store EDGE INDICES instead of partner nodes,
    so parallel edges stay distinguishable. Returns the colour of every edge
    (indexed like ``edges``); colours == max degree (König); deterministic in
    edge order. ``minimal_edge_coloring`` is left untouched so LRC does not
    drift.
    """

    edge_list = [(int(u), int(v)) for u, v in edges]
    n_left, n_right = int(n_left), int(n_right)
    if not edge_list:
        return []
    left_degree = [0] * n_left
    right_degree = [0] * n_right
    for u, v in edge_list:
        if not (0 <= u < n_left and 0 <= v < n_right):
            raise ValueError("edge endpoint out of range")
        left_degree[u] += 1
        right_degree[v] += 1
    degree = max(max(left_degree), max(right_degree))
    left_color = -np.ones((n_left, degree), dtype=np.int64)  # (node, colour) -> edge index
    right_color = -np.ones((n_right, degree), dtype=np.int64)
    colors = [-1] * len(edge_list)

    def free_color(table: np.ndarray, node: int) -> int:
        for color in range(degree):
            if table[node, color] < 0:
                return color
        raise AssertionError("no free color — Δ computation broken")

    for edge_index, (u, v) in enumerate(edge_list):
        a = free_color(left_color, u)
        b = free_color(right_color, v)
        if a != b:
            # Maximal alternating a/b path from `v` along colour a; it cannot
            # reach `u` (odd two-coloured cycle impossible in a bipartite graph).
            path: list[tuple[int, int]] = []  # (edge index, current colour)
            side_is_right, node, color = True, v, a
            for _ in range(2 * (n_left + n_right) + 2):
                table = right_color if side_is_right else left_color
                e = int(table[node, color])
                if e < 0:
                    break
                path.append((e, color))
                eu, ev = edge_list[e]
                node = eu if side_is_right else ev
                side_is_right = not side_is_right
                color = b if color == a else a
            else:
                raise AssertionError("alternating path exceeded graph size")
            for e, edge_color in path:
                eu, ev = edge_list[e]
                left_color[eu, edge_color] = -1
                right_color[ev, edge_color] = -1
            for e, edge_color in path:
                other = b if edge_color == a else a
                eu, ev = edge_list[e]
                left_color[eu, other] = e
                right_color[ev, other] = e
                colors[e] = other
            if left_color[u, a] >= 0:  # bipartite guarantee violated
                raise AssertionError("Kempe flip reached the anchor node")
        left_color[u, a] = edge_index
        right_color[v, a] = edge_index
        colors[edge_index] = a
    return colors
