"""Per-basis CX layers for the two-basis non-interleaved circuit (02 §9.4).

Structured path (bb / lp_abelian / lp_nonabelian): the compiled code carries
``layer_blocks`` — every nonzero of H_W belongs to exactly one "atomic
matching" term (check block ``cb``, data block ``db``, permutation ``perm``)
asserting ``H_W[cb*g + perm[t], db*g + t] == 1`` for every t in range(g).
Build the block-level bipartite MULTIgraph (vertices = check blocks ∪ data
blocks, one edge per term, parallel edges allowed) and edge-colour it
minimally (König: colours = max block degree = check weight w for these
families). One colour class = one layer = union of atomic matchings sharing
no check block and no data block, hence a matching of the Tanner graph.

Fallback path (hgp, raw matrices): ``coloring.minimal_edge_coloring(h)``.

Both paths return layers covering every Tanner edge exactly once, each layer a
matching — asserted here, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .coloring import color_bipartite_multigraph, minimal_edge_coloring

Edge = tuple[int, int]  # (check_row, GLOBAL data column)

LAYER_BLOCKS_SCHEMA = "layer_blocks_v1"


@dataclass(frozen=True)
class BlockStructure:
    block_size: int
    n_check_blocks: int
    n_data_blocks: int
    terms: tuple[tuple[int, int, tuple[int, ...]], ...]  # (cb, db, perm)

    @classmethod
    def from_layer_blocks(
        cls, layer_blocks: dict[str, Any] | None, basis: str
    ) -> "BlockStructure | None":
        """None if absent / wrong schema / basis key missing."""

        if not isinstance(layer_blocks, dict):
            return None
        if layer_blocks.get("schema") != LAYER_BLOCKS_SCHEMA:
            return None
        basis = str(basis).upper()
        blob = layer_blocks.get(basis)
        if not isinstance(blob, dict):
            return None
        g = int(layer_blocks["block_size"])
        terms = tuple(
            (int(cb), int(db), tuple(int(v) for v in perm))
            for cb, db, perm in blob.get("terms", [])
        )
        for cb, db, perm in terms:
            if len(perm) != g or sorted(perm) != list(range(g)):
                raise ValueError("layer_blocks term perm is not a permutation of range(g)")
        return cls(
            block_size=g,
            n_check_blocks=int(blob["n_check_blocks"]),
            n_data_blocks=int(blob["n_data_blocks"]),
            terms=terms,
        )

    def term_edges(self, index: int) -> list[Edge]:
        cb, db, perm = self.terms[index]
        g = self.block_size
        return [(cb * g + perm[t], db * g + t) for t in range(g)]


def _assert_layers_cover(h: np.ndarray, layers: list[list[Edge]]) -> None:
    rows, cols = h.shape
    seen: list[Edge] = []
    for layer in layers:
        checks = [c for c, _ in layer]
        qubits = [q for _, q in layer]
        if len(set(checks)) != len(checks):
            raise AssertionError("layer is not a matching: a check appears twice")
        if len(set(qubits)) != len(qubits):
            raise AssertionError("layer is not a matching: a data column appears twice")
        seen.extend(layer)
    expected = sorted((int(r), int(c)) for r in range(rows) for c in np.nonzero(h[r])[0])
    if sorted(seen) != expected:
        raise AssertionError("layers do not cover the nonzeros of H exactly once")


def build_basis_layers(
    h: np.ndarray, *, blocks: BlockStructure | None
) -> list[list[Edge]]:
    """Layers (matchings) of H_W; structured when ``blocks`` is given, else
    the plain König colouring of the Tanner graph. Post-conditions asserted."""

    h = np.asarray(h, dtype=np.uint8) % 2
    if blocks is None:
        layers = [[(int(c), int(q)) for c, q in layer] for layer in minimal_edge_coloring(h)]
    else:
        block_edges = [(cb, db) for cb, db, _ in blocks.terms]
        colors = color_bipartite_multigraph(
            block_edges, n_left=blocks.n_check_blocks, n_right=blocks.n_data_blocks
        )
        n_colors = (max(colors) + 1) if colors else 0
        layers = [[] for _ in range(n_colors)]
        for index, color in enumerate(colors):
            layers[color].extend(blocks.term_edges(index))
        layers = [sorted(layer) for layer in layers]
    _assert_layers_cover(h, layers)
    return layers


def build_layer_sets(
    hx: np.ndarray, hz: np.ndarray, *, layer_blocks: dict[str, Any] | None
) -> dict[str, list[list[Edge]]]:
    return {
        "X": build_basis_layers(hx, blocks=BlockStructure.from_layer_blocks(layer_blocks, "X")),
        "Z": build_basis_layers(hz, blocks=BlockStructure.from_layer_blocks(layer_blocks, "Z")),
    }
