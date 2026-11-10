"""Fast CSS distance proxy — the v3.2 `d_search` protocol.

Two properties the structural retention channel depends on (repair plan §3):

  * the X and Z sectors draw from INDEPENDENT streams, so changing the effort
    spent on one sector cannot move the other sector's random draw;
  * the reported value is the minimum over a fixed seed set, so a single
    lucky permutation cannot inflate a candidate onto the max-d endpoint.

Both are protocol, not tuning: every family runs the identical seeds and the
identical per-seed effort.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .gf2 import kernel_basis, logicals

# No logical operators / empty kernel: the side has no finite proxy. Kept as a
# large sentinel (never a real weight) rather than None so the min() folds stay
# plain integer arithmetic; callers treat >= this value as "proxy unavailable".
PROXY_UNAVAILABLE = 10**9
# Bumped when the search itself changes: recorded with every proxy result so a
# d_search produced under a different algorithm can never be compared silently.
PROXY_ALGORITHM_ID = "css_permutation_systematic_v2_independent_xz"
DEFAULT_PROXY_SEEDS: tuple[int, ...] = (0, 1, 2, 3)

_SECTOR_STREAM = {"Z": 0, "X": 1}


def _sector_rng(seed: int, sector: str) -> np.random.Generator:
    """Independent per-(seed, sector) stream.

    The entropy pair makes the X and Z streams disjoint by construction: the
    old shared-generator form advanced one stream while computing the other,
    so a change to Z's iteration count silently moved every X draw.
    """

    return np.random.default_rng([int(seed), _SECTOR_STREAM[sector]])


def proxy_distance(hx: np.ndarray, hz: np.ndarray, *, iters: int = 400, seed: int = 0) -> int:
    d_x, d_z = proxy_distances(hx, hz, iters=iters, seed=seed)
    return int(min(d_x, d_z))


def proxy_distances(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    iters: int = 400,
    seed: int = 0,
) -> tuple[int, int]:
    """Return basis-resolved ``(d_X, d_Z)`` proxies for ONE seed."""

    lx_z, lz = logicals(hx, hz)
    if lz.shape[0] == 0:
        return PROXY_UNAVAILABLE, PROXY_UNAVAILABLE
    kernel_x = kernel_basis(hx)
    kernel_z = kernel_basis(hz)
    d_z = _proxy_side(kernel_x, lx_z, iters, _sector_rng(seed, "Z"))
    d_x = _proxy_side(kernel_z, lz, iters, _sector_rng(seed, "X"))
    return int(d_x), int(d_z)


def proxy_distance_search(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    seeds: tuple[int, ...] | list[int] = DEFAULT_PROXY_SEEDS,
    iters: int = 120,
    algorithm_id: str | None = None,
) -> dict[str, Any]:
    """The frozen `d_search` protocol: min over seeds x {X, Z}.

    Returns the aggregate plus the per-seed/per-sector table, so a later run
    can prove it reproduced the same value from the same seeds rather than
    asserting it. ``algorithm_id`` is the caller's DECLARED protocol; if it
    does not name the algorithm this module implements, the search fails
    closed instead of silently executing something the artifact will not
    describe (audit fix: declaration must equal execution)."""

    if algorithm_id is not None and str(algorithm_id) != PROXY_ALGORITHM_ID:
        raise ValueError(
            f"declared d_search algorithm {algorithm_id!r} is not the implemented "
            f"{PROXY_ALGORITHM_ID!r}"
        )
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("d_search needs at least one proxy seed")
    per_seed: list[dict[str, int]] = []
    d_x = PROXY_UNAVAILABLE
    d_z = PROXY_UNAVAILABLE
    for seed in seed_list:
        seed_d_x, seed_d_z = proxy_distances(hx, hz, iters=int(iters), seed=seed)
        per_seed.append({"seed": seed, "d_x": int(seed_d_x), "d_z": int(seed_d_z)})
        d_x = min(d_x, int(seed_d_x))
        d_z = min(d_z, int(seed_d_z))
    available = d_x < PROXY_UNAVAILABLE and d_z < PROXY_UNAVAILABLE
    return {
        "schema": "proxy_distance_search",
        "algorithm_id": PROXY_ALGORITHM_ID,
        "seeds": seed_list,
        "iters_per_seed": int(iters),
        "sectors": ["X", "Z"],
        "per_seed": per_seed,
        "d_x": int(d_x),
        "d_z": int(d_z),
        "d_search": int(min(d_x, d_z)) if available else PROXY_UNAVAILABLE,
        "available": bool(available),
    }


def _proxy_side(
    kernel: np.ndarray,
    logical_parity: np.ndarray,
    iters: int,
    rng: np.random.Generator,
    t_lightest: int = 24,
) -> int:
    rows, n = kernel.shape
    if rows == 0:
        return 10**9
    best = 10**9
    for _ in range(int(iters)):
        cols = rng.permutation(n)
        systematic = _systematic(kernel, cols)
        weights = systematic.sum(1)
        syndromes = (systematic.astype(np.int8) @ logical_parity.T) & 1
        nontrivial = syndromes.any(1)
        if nontrivial.any():
            candidate = int(weights[nontrivial].min())
            if candidate < best:
                best = candidate

        count = min(t_lightest, rows)
        indices = np.argsort(weights)[:count]
        selected_rows = systematic[indices]
        selected_syndromes = syndromes[indices]
        for index in range(count - 1):
            pairs = selected_rows[index] ^ selected_rows[index + 1 :]
            pair_syndromes = selected_syndromes[index] ^ selected_syndromes[index + 1 :]
            pair_weights = pairs.sum(1)
            pair_nontrivial = pair_syndromes.any(1)
            if pair_nontrivial.any():
                candidate = int(pair_weights[pair_nontrivial].min())
                if candidate < best:
                    best = candidate
    return int(best)


def _systematic(generator: np.ndarray, cols: np.ndarray) -> np.ndarray:
    matrix = generator.copy()
    rows = matrix.shape[0]
    rank = 0
    for col in cols:
        column = matrix[rank:, col]
        nonzero = np.nonzero(column)[0]
        if len(nonzero) == 0:
            continue
        pivot = rank + nonzero[0]
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        mask = matrix[:, col].astype(bool)
        mask[rank] = False
        if mask.any():
            matrix[mask] ^= matrix[rank]
        rank += 1
        if rank == rows:
            break
    return matrix
