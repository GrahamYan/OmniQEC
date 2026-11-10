"""Three-view chain-complex graphs of a CSS code for the BO adapted embedding (numpy only).

A code with check matrices HX, HZ and logical operators LX, LZ (``verification.gf2.logicals``) becomes five node
classes -- data qubits DQ, X checks SX, Z checks SZ, X logicals LX, Z logicals LZ -- and the three views of the
authors' entry point (BO_for_QECcodes@dc4593f, ``BOplayground.py`` lines 734-747, built by
``bayesian_optimization/encoder.py`` ``RelationEncoder``): ``decode`` (SZ, DQ, SX), ``xlogic`` (DQ, SX, LX) and
``zlogic`` (SZ, DQ, LZ), each with the four directed relations between its check/logical class and the data qubits.
A relation tag ``A_B`` sends messages from class A to class B; its matrix has shape (|B|, |A|) (the authors'
left-multiply convention), e.g. ``SX_DQ`` = HX^T and ``DQ_SX`` = HX. The authors' ``weight_mode="count"`` /
``log1p`` keys in ``views_info`` are not read by their encoder (0/1 matrices), so they have no counterpart here.

Graphs are cached per code_hash in-process: the logical operators of a pool code are computed once per job."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Sorted union of the views' node classes: the authors' ``all_node_types`` (sorted(set)), also their readout order.
NODE_TYPES: tuple[str, ...] = ("DQ", "LX", "LZ", "SX", "SZ")
VIEWS_INFO: tuple[dict[str, Any], ...] = (
    {"name": "decode", "partite_classes": ("SZ", "DQ", "SX"), "relations": ("SZ_DQ", "DQ_SZ", "DQ_SX", "SX_DQ")},
    {"name": "xlogic", "partite_classes": ("DQ", "SX", "LX"), "relations": ("DQ_SX", "SX_DQ", "LX_DQ", "DQ_LX")},
    {"name": "zlogic", "partite_classes": ("SZ", "DQ", "LZ"), "relations": ("SZ_DQ", "DQ_SZ", "LZ_DQ", "DQ_LZ")},
)
# The class whose rows/columns a primitive matrix carries: HX rows are X checks, LX rows X logicals, ...
_MATRIX_OF_CLASS = {"SX": "hx", "SZ": "hz", "LX": "lx", "LZ": "lz"}


def relation_endpoints(tag: str) -> tuple[str, str]:
    """(SRC, DST) of a relation tag 'A_B'."""

    src, _, dst = str(tag).partition("_")
    if not src or not dst:
        raise ValueError(f"bad relation tag {tag!r}; expected 'A_B'")
    return src.upper(), dst.upper()


@dataclass(frozen=True, eq=False)
class CodeGraph:
    """One code's node counts and, per relation tag of the views, the (dst, src) index arrays of its edges."""

    code_hash: str
    sizes: dict[str, int]
    edges: dict[str, tuple[np.ndarray, np.ndarray]]

    def node_count(self) -> int:
        return int(sum(self.sizes.values()))


_GRAPH_CACHE: dict[tuple[str, tuple[int, ...]], CodeGraph] = {}


def _nonzeros(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(np.asarray(matrix, dtype=np.uint8) % 2)
    return rows.astype(np.int64), cols.astype(np.int64)


def code_graph(code_hash: str, hx: np.ndarray, hz: np.ndarray) -> CodeGraph:
    """The three-view graph of one CSS code (cached by code_hash and matrix shapes)."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    key = (str(code_hash), tuple(hx.shape) + tuple(hz.shape))
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    if hx.ndim != 2 or hz.ndim != 2 or hx.shape[1] != hz.shape[1]:
        raise ValueError(f"code {code_hash}: HX {hx.shape} and HZ {hz.shape} do not share n columns")
    from omniqec_agentic.verification.gf2 import logicals

    lx, lz = logicals(hx, hz)
    n = int(hx.shape[1])
    matrices = {"hx": hx, "hz": hz, "lx": np.asarray(lx, dtype=np.uint8), "lz": np.asarray(lz, dtype=np.uint8)}
    sizes = {"DQ": n, "SX": int(hx.shape[0]), "SZ": int(hz.shape[0]), "LX": int(lx.shape[0]), "LZ": int(lz.shape[0])}
    nonzeros = {name: _nonzeros(matrix) for name, matrix in matrices.items()}
    edges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in VIEWS_INFO:
        for tag in view["relations"]:
            if tag in edges:
                continue
            src, dst = relation_endpoints(tag)
            if dst == "DQ":
                # class -> DQ: matrix M^T, an edge (row r, qubit q) of M becomes dst q, src r
                rows, cols = nonzeros[_MATRIX_OF_CLASS[src]]
                edges[tag] = (cols, rows)
            elif src == "DQ":
                # DQ -> class: matrix M, dst r, src q
                rows, cols = nonzeros[_MATRIX_OF_CLASS[dst]]
                edges[tag] = (rows, cols)
            else:
                raise ValueError(f"relation {tag!r} does not touch the data qubits")
    graph = CodeGraph(code_hash=str(code_hash), sizes=sizes, edges=edges)
    _GRAPH_CACHE[key] = graph
    return graph


def clear_graph_cache() -> None:
    _GRAPH_CACHE.clear()
