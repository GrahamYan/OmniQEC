"""Two-tier structural fingerprints (03 §6.3).

Tier-1 (every candidate): cheap invariants — n / check counts / degree
histograms — plus a colored-Tanner-graph Weisfeiler-Leman hash (three colors:
qubit / X-check / Z-check; WL-fingerprint precedent: SCE's novelty penalty).
Tier-2 (on Tier-1 collision, finalist level): exact colored-graph canonical
form via pynauty when available; degrades to Tier-1 with an explicit backend
marker otherwise (precedent: Cruz-Benito's BLISS-via-igraph). Novelty counts
are therefore reported as an UPPER BOUND under permutation equivalence.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import time
from typing import Any

import numpy as np

FINGERPRINT_SCHEMA_VERSION = 1
WL_ITERATIONS = 4
# Near-duplicate promotion gate (03 §6.3): distance < eps never reaches the
# slow loop. Tier-1 WL-identical pairs have distance 0; structurally distinct
# pairs bottom out at 0.2, so the default gate blocks only WL-collisions.
NOVELTY_EPSILON = 0.05
# v3.6 (docs/06 §3 B3): the slow-loop twin gate reads `code_hash` and the
# exact pynauty canonical form ONLY. 1-WL is too coarse for the lifted-product
# grammar — twelve distinct C4xD5 1x2 weight-9 codes share one wl_hash, and
# BB144 collides with the k=4 BB(12,6) — so a WL match is recorded as
# information but never decides admissibility.
TWIN_GATE_ID = "code_hash+pynauty_canonical_v1"
# 2026-10-26: pynauty.certificate never returned on a highly symmetric BB code
# (8x8 torus, A = 1 + x y^3 + x^2 y^6, B = 1 + x^3 y + x^6 y^2; S3 CNOT b260
# turns 17 and 29) and, being C code, cannot be interrupted: the 40 seed copies
# of that one program blocked pool.map and froze the fast generation for 8-13 h.
# The certificate now runs in a forked child with this deadline (normal codes
# take milliseconds); past it the code gets no canonical hash, which callers
# already treat as "fall back to exact code_hash equality". 0 = in-process.
CANONICAL_TIMEOUT_S = float(os.environ.get("OMNIQEC_CANONICAL_TIMEOUT_S", "60"))


def _degree_histogram(degrees: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(degrees, dtype=int), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def tanner_wl_hash(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    iterations: int = WL_ITERATIONS,
) -> str:
    """WL refinement hash of the colored Tanner graph (deterministic).

    Nodes: qubits (color q), X-checks (color x), Z-checks (color z); edges
    from the supports of hx/hz. Permutation-invariant by construction: the
    final hash digests the sorted multiset of stabilized node labels."""

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n = hx.shape[1]
    m_x, m_z = hx.shape[0], hz.shape[0]
    # adjacency as index lists: qubits 0..n-1, X-checks n..n+m_x-1, Z-checks after
    neighbors: list[list[int]] = [[] for _ in range(n + m_x + m_z)]
    for check, qubit in zip(*np.nonzero(hx)):
        neighbors[n + int(check)].append(int(qubit))
        neighbors[int(qubit)].append(n + int(check))
    for check, qubit in zip(*np.nonzero(hz)):
        neighbors[n + m_x + int(check)].append(int(qubit))
        neighbors[int(qubit)].append(n + m_x + int(check))

    labels = ["q"] * n + ["x"] * m_x + ["z"] * m_z
    for _ in range(max(1, int(iterations))):
        new_labels = []
        for node in range(len(labels)):
            signature = labels[node] + "|" + ",".join(sorted(labels[v] for v in neighbors[node]))
            new_labels.append(hashlib.blake2b(signature.encode(), digest_size=8).hexdigest())
        if new_labels == labels:
            break
        labels = new_labels
    digest = hashlib.sha256()
    for label in sorted(labels):
        digest.update(label.encode())
    return digest.hexdigest()


def tier1_fingerprint(hx: np.ndarray, hz: np.ndarray) -> dict[str, Any]:
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    fingerprint = {
        "schema": "tanner_fingerprint_tier1",
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "n": int(hx.shape[1]),
        "m_x": int(hx.shape[0]),
        "m_z": int(hz.shape[0]),
        "row_deg_x": _degree_histogram(hx.sum(axis=1)),
        "row_deg_z": _degree_histogram(hz.sum(axis=1)),
        "col_deg_x": _degree_histogram(hx.sum(axis=0)),
        "col_deg_z": _degree_histogram(hz.sum(axis=0)),
        "wl_hash": tanner_wl_hash(hx, hz),
    }
    canonical = json.dumps(
        {k: v for k, v in fingerprint.items() if k != "wl_hash"}, sort_keys=True
    )
    fingerprint["invariant_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return fingerprint


def _hist_l1(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    total = sum(a.values()) + sum(b.values())
    if total <= 0:
        return 0.0
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / total


def fingerprint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Monotone novelty distance in [0, 1]: 0 = WL-indistinguishable,
    1 = different shape entirely. Structurally distinct same-shape codes
    bottom out at 0.2 so the default near-duplicate gate (eps=0.05) only
    blocks WL collisions."""

    if a.get("wl_hash") and a.get("wl_hash") == b.get("wl_hash"):
        return 0.0
    if (a.get("n"), a.get("m_x"), a.get("m_z")) != (b.get("n"), b.get("m_x"), b.get("m_z")):
        return 1.0
    hist = (
        _hist_l1(a.get("row_deg_x") or {}, b.get("row_deg_x") or {})
        + _hist_l1(a.get("row_deg_z") or {}, b.get("row_deg_z") or {})
        + _hist_l1(a.get("col_deg_x") or {}, b.get("col_deg_x") or {})
        + _hist_l1(a.get("col_deg_z") or {}, b.get("col_deg_z") or {})
    ) / 4.0
    return 0.2 + 0.8 * min(1.0, hist)


def novelty_distance_to_set(
    fingerprint: dict[str, Any],
    seen: list[dict[str, Any]],
) -> tuple[float, dict[str, Any] | None]:
    """Minimum distance to a fingerprint set; returns (distance, nearest)."""

    best: tuple[float, dict[str, Any] | None] = (1.0, None)
    for other in seen:
        distance = fingerprint_distance(fingerprint, other)
        if distance < best[0]:
            best = (distance, other)
        if best[0] == 0.0:
            break
    return best


def tier2_canonical_hash(hx: np.ndarray, hz: np.ndarray) -> tuple[str | None, str]:
    """Exact colored-graph canonical form. Returns (hash, backend); backend
    "tier1_only" when pynauty is absent (03 §6.3 explicit degradation),
    "pynauty_timeout" (hash None) past CANONICAL_TIMEOUT_S."""

    try:
        import pynauty
    except Exception:
        return None, "tier1_only"

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n = hx.shape[1]
    m_x, m_z = hx.shape[0], hz.shape[0]
    total = n + m_x + m_z
    adjacency: dict[int, list[int]] = {node: [] for node in range(total)}
    for check, qubit in zip(*np.nonzero(hx)):
        adjacency[int(qubit)].append(n + int(check))
        adjacency[n + int(check)].append(int(qubit))
    for check, qubit in zip(*np.nonzero(hz)):
        adjacency[int(qubit)].append(n + m_x + int(check))
        adjacency[n + m_x + int(check)].append(int(qubit))
    graph = pynauty.Graph(
        total,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=[
            set(range(n)),
            set(range(n, n + m_x)),
            set(range(n + m_x, total)),
        ],
    )
    certificate = _certificate_with_deadline(pynauty, graph, CANONICAL_TIMEOUT_S)
    if certificate is None:
        return None, "pynauty_timeout"
    return hashlib.sha256(certificate).hexdigest(), "pynauty"


def _certificate_with_deadline(pynauty: Any, graph: Any, timeout_s: float) -> bytes | None:
    """pynauty.certificate(graph) in a forked child; None if it fails or is
    still running after timeout_s (the child is then killed)."""

    if timeout_s <= 0 or not hasattr(os, "fork"):
        return pynauty.certificate(graph)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            os.close(read_fd)
            view = memoryview(pynauty.certificate(graph))
            while view:
                view = view[os.write(write_fd, view):]
            status = 0
        finally:
            os._exit(status)
    os.close(write_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
                timed_out = True
                break
            chunk = os.read(read_fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
        if timed_out:
            os.kill(pid, signal.SIGKILL)
        try:
            _, status = os.waitpid(pid, 0)
        except ChildProcessError:
            status = 0 if chunks else 1
    if timed_out or status != 0 or not chunks:
        return None
    return b"".join(chunks)


def canonical_code_hash(hx: np.ndarray, hz: np.ndarray) -> str | None:
    """Exact permutation-equivalence identity of a CSS code (v3.6 B3).

    The sha256 of the pynauty certificate of the colored Tanner graph; None
    when pynauty is unavailable (or fails), in which case callers must fall
    back to exact `code_hash` equality and must NOT substitute the WL hash."""

    try:
        value, backend = tier2_canonical_hash(hx, hz)
    except Exception:
        return None
    return value if backend == "pynauty" and value else None


def twin_identities(row: dict[str, Any]) -> tuple[str, str]:
    """(code_hash, canonical_hash) of a queue/ledger row, each "" if absent."""

    if not isinstance(row, dict):
        return "", ""
    return str(row.get("code_hash") or ""), str(row.get("canonical_hash") or "")


def is_twin_of(
    row: dict[str, Any],
    *,
    code_hashes: set[str],
    canonical_hashes: set[str],
) -> bool:
    """The v3.6 twin rule: a row is a twin of an already measured / already
    selected code iff its `code_hash` is in ``code_hashes`` or its
    `canonical_hash` is present and in ``canonical_hashes``. `wl_hash` is
    deliberately not consulted (docs/05 §4 B3)."""

    code_hash, canonical = twin_identities(row)
    if code_hash and code_hash in code_hashes:
        return True
    return bool(canonical and canonical in canonical_hashes)
