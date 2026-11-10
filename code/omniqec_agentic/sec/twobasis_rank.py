"""Two-basis NON-interleaved SEC policy ``twobasis_nonint_v1``: cheap
intra-basis order search + certification (02 §9.4; v3.6 M1 repair, 06 §3).

Search space per basis W: permutations of that basis' layer list (canonical
= colour order). Candidate i = ``counter_permutation(len(L), candidate=i,
slot=0 for X / 1 for Z)``; i = 0 is the identity.

Certification per candidate order and basis: because the two halves of a
round never interleave, SBB Thm 1 makes the circuit distance EXACTLY the
extended-code distance of the residual set — no circuit compilation is
needed. Residual suffixes of each check (edges ordered by layer index) extend
the opposite-basis detector matrix (``extended_matrices``); the distance is
probed with the house ISD (``_proxy_side``) — an UPPER-bound estimate,
recorded as evidence only. The exact GF2 Delta=1 scan rejects any order with
a residual that is itself a logical.

v3.6 (docs/05 §2.3, docs/06 §3 M1): a single 400-iteration ISD run does not
converge on the 800+-column extended kernels of n >= 150 codes, so "argmax of
one noisy estimate" selected the luckiest candidate and recorded d_ext values
above the code distance. The certification is therefore staged:

  stage 1  every candidate, ``max(sec_cert_iters, sec_cert_iters_scale x
           kernel_dim)`` iterations, MIN over ``sec_cert_seeds`` independent
           seeds (seed = sec_cert_seed + index*7 + side + 1000*s);
  refine   the top ``sec_refine_top`` stage-1 survivors per basis (ties ->
           smaller index) are re-probed with ``sec_refine_iters`` iterations x
           ``sec_cert_seeds`` fresh seeds (s continues at sec_cert_seeds); the
           refined value is the min over BOTH stages (every run is an upper
           bound, so the tightest one is the estimate);
  select   largest refined value wins, ties -> smallest index (canonical);
  record   d_ext is clamped to <= d_code_proxy when a proxy is given (an
           extended-code distance cannot exceed the code distance), the
           unclamped value is kept alongside, and ``converged_x/z`` says
           whether every refine seed of the chosen candidate agreed.

Everything candidate-independent (logicals, detector matrix, stabilizer span
basis) is computed once per basis (TB-5). Deterministic: identical inputs give
identical results.

Review fix (2026-09-03, docs/05 §4 M1 cost): the staged certification of a
Mitten-class [[200,40]] code needs ~340k ISD iterations (~2.6 CPU-hours), more
than the build watchdog, and the synth cache was written only on completion,
so a killed build restarted from zero and the code was struck out. Three
guards, all deterministic:

  budget   ``policy.sec_cert_iters_budget`` (0 = unlimited) caps the TOTAL ISD
           iterations per code; when the nominal plan exceeds it the stage-1
           and refine iterations shrink by a fixed rule (``certification_plan``)
           and ``certification_budget_exhausted`` is recorded -- an audit
           flag, never a failure;
  progress every probe result is checkpointed (``CertificationProgress``) so
           a killed synthesis resumes where it stopped with identical values
           (seeds are per candidate / seed slot, so a resumed run IS the
           uninterrupted run);
  model    the pool's build watchdog derives from the certification work
           model (``certification_work_model``) instead of a flat hour.
"""

from __future__ import annotations

import concurrent.futures as _futures
import json
import multiprocessing as _mp
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.verification.gf2 import kernel_basis, logicals
from omniqec_agentic.verification.proxy_distance import _proxy_side

from .coloring import counter_permutation
from .layers import Edge, build_layer_sets
from .schedule_rank import (
    RankedSchedule,
    _reduce_against,
    _row_reduce,
    delta_one_residuals,
    extended_matrices,
)

try:  # config.py gains SecPolicyConfig in the same v3.5 change (Agent 3)
    from omniqec_agentic.config import (
        DEFAULT_SEC_POLICY,
        LRC_SEC_POLICY,
        SEC_POLICY_IDS,
        SecPolicyConfig,
    )
except ImportError:  # pragma: no cover - local fallback until config.py lands
    SEC_POLICY_IDS: tuple[str, ...] = ("twobasis_nonint_v1", "lrc_ranked_v1")

    @dataclass(frozen=True)
    class SecPolicyConfig:  # type: ignore[no-redef]
        sec_policy_id: str = "twobasis_nonint_v1"
        sec_candidates: int = 16
        sec_cert_iters: int = 400
        sec_cert_seed: int = 0
        sec_cert_seeds: int = 3
        sec_cert_iters_scale: float = 4.0
        sec_refine_top: int = 4
        sec_refine_iters: int = 2000
        sec_cert_iters_budget: int = 0

        def __post_init__(self) -> None:
            if self.sec_policy_id not in SEC_POLICY_IDS:
                raise ValueError(f"unknown sec_policy_id {self.sec_policy_id!r}")
            if int(self.sec_candidates) < 1:
                raise ValueError("sec_candidates must be >= 1")
            if int(self.sec_cert_iters) < 1:
                raise ValueError("sec_cert_iters must be >= 1")

        @property
        def is_lrc(self) -> bool:
            return self.sec_policy_id == "lrc_ranked_v1"

        def to_dict(self) -> dict[str, Any]:
            return {
                "sec_policy_id": self.sec_policy_id,
                "sec_candidates": int(self.sec_candidates),
                "sec_cert_iters": int(self.sec_cert_iters),
                "sec_cert_seed": int(self.sec_cert_seed),
                "sec_cert_seeds": int(self.sec_cert_seeds),
                "sec_cert_iters_scale": float(self.sec_cert_iters_scale),
                "sec_refine_top": int(self.sec_refine_top),
                "sec_refine_iters": int(self.sec_refine_iters),
                "sec_cert_iters_budget": int(self.sec_cert_iters_budget),
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any] | None) -> "SecPolicyConfig":
            if not data:
                return cls()
            return cls(
                sec_policy_id=str(data.get("sec_policy_id", "twobasis_nonint_v1")),
                sec_candidates=int(data.get("sec_candidates", 16)),
                sec_cert_iters=int(data.get("sec_cert_iters", 400)),
                sec_cert_seed=int(data.get("sec_cert_seed", 0)),
                sec_cert_seeds=int(data.get("sec_cert_seeds", 3)),
                sec_cert_iters_scale=float(data.get("sec_cert_iters_scale", 4.0)),
                sec_refine_top=int(data.get("sec_refine_top", 4)),
                sec_refine_iters=int(data.get("sec_refine_iters", 2000)),
                sec_cert_iters_budget=int(data.get("sec_cert_iters_budget", 0)),
            )

    DEFAULT_SEC_POLICY = SecPolicyConfig()
    LRC_SEC_POLICY = SecPolicyConfig(
        sec_policy_id="lrc_ranked_v1", sec_candidates=64, sec_cert_iters=100, sec_cert_seed=0
    )

TWOBASIS_SEC_POLICY_ID = "twobasis_nonint_v1"
_SLOT = {"X": 0, "Z": 1}  # permutation-stream slot per basis (== ISD side)
_SEED_STRIDE = 1000  # seed = base + index*7 + side + _SEED_STRIDE*s


@dataclass(frozen=True)
class TwoBasisSchedule:
    """Per-basis CX layer orders of the non-interleaved two-basis circuit.
    Plain ints/tuples only (pickled into the synth cache, JSON metadata)."""

    n_data: int
    m_x: int
    m_z: int
    layers: dict[str, list[list[Edge]]]  # {"X": [...], "Z": [...]}, Edge = (check, GLOBAL column)
    candidate_index_x: int = 0
    candidate_index_z: int = 0
    d_ext_x: int | None = None  # certified estimate of the chosen X order (X-kind residual side)
    d_ext_z: int | None = None
    candidates: int = 16  # copied from the policy for to_metadata
    cert_iters: int = 400
    # v3.6 M1 staged certification (defaults = the v3.5 single-seed reading so
    # a pickled v3.5 schedule still unpickles; the synth cache key carries the
    # policy fields, so a v3.5 entry is never served for a v3.6 policy).
    cert_seeds: int = 1
    refine_iters: int = 0
    converged_x: bool | None = None
    converged_z: bool | None = None
    # Review fix: True when the per-code ISD budget shrank the certification
    # (None for a schedule certified before the field existed).
    budget_exhausted: bool | None = None

    @property
    def depth(self) -> int:
        return len(self.layers["X"]) + len(self.layers["Z"]) + 4

    @property
    def source(self) -> str:
        return TWOBASIS_SEC_POLICY_ID

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sec_policy_id": TWOBASIS_SEC_POLICY_ID,
            "depth": int(self.depth),
            "layers_x": len(self.layers["X"]),
            "layers_z": len(self.layers["Z"]),
            "candidate_index_x": int(self.candidate_index_x),
            "candidate_index_z": int(self.candidate_index_z),
            "d_ext_x": (None if self.d_ext_x is None else int(self.d_ext_x)),
            "d_ext_z": (None if self.d_ext_z is None else int(self.d_ext_z)),
            "candidates": int(self.candidates),
            "cert_iters": int(self.cert_iters),
            "cert_seeds": int(self.cert_seeds),
            "refine_iters": int(self.refine_iters),
            "converged_x": self.converged_x,
            "converged_z": self.converged_z,
            "certification_budget_exhausted": self.budget_exhausted,
        }


def basis_residual_suffixes(
    layers_w: list[list[Edge]], *, m_w: int, n_data: int
) -> list[list[np.ndarray]]:
    """Per-check residual suffix supports (binary vectors over data qubits):
    edges of each check ordered by layer index, then EXACTLY the
    ``schedule_rank.residual_suffixes`` semantics (suffixes l = 2..w)."""

    order: dict[int, list[int]] = {}
    for layer in layers_w:  # each layer holds <= 1 edge per check
        for check, qubit in layer:
            order.setdefault(int(check), []).append(int(qubit))
    out: list[list[np.ndarray]] = []
    for check in range(int(m_w)):
        qubits = order.get(check, [])
        suffixes = []
        for start in range(1, len(qubits)):
            vector = np.zeros(int(n_data), dtype=np.uint8)
            vector[qubits[start:]] = 1
            suffixes.append(vector)
        out.append(suffixes)
    return out


# ----------------------------------------------------------------- contexts


@dataclass(frozen=True)
class BasisCertContext:
    """Everything of one basis' certification that does not depend on the
    candidate order (TB-5 hoist): the detector matrix of the opposite basis,
    the threatened logical rows and the row-reduced same-basis stabilizer
    span used by the exact Delta=1 scan."""

    kind: str
    h_detect: np.ndarray
    logical_rows: np.ndarray
    span_basis: np.ndarray


def basis_cert_context(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    kind: str,
    logical_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> BasisCertContext:
    """kind "X": residuals of X checks propagate onto data, are detected by hz
    and threaten the Z-logicals (lz); kind "Z" symmetric — the same side
    mapping as ``schedule_rank.d_ext_probe``. ``logical_pair`` = (lx, lz)
    lets the caller share one ``logicals`` call between both bases."""

    kind = str(kind).upper()
    if kind not in ("X", "Z"):
        raise ValueError("kind must be 'X' or 'Z'")
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    lx, lz = logicals(hx, hz) if logical_pair is None else logical_pair
    if kind == "X":
        h_detect, logical_rows, h_span = hz, np.asarray(lz, dtype=np.uint8) % 2, hx
    else:
        h_detect, logical_rows, h_span = hx, np.asarray(lx, dtype=np.uint8) % 2, hz
    return BasisCertContext(
        kind=kind,
        h_detect=h_detect,
        logical_rows=logical_rows,
        span_basis=_row_reduce(h_span),
    )


def _delta_one_hit(context: BasisCertContext, suffixes_w: list[list[np.ndarray]]) -> bool:
    """Exact GF2 Delta=1 scan of one basis' residuals against a hoisted span
    basis — the same predicate as ``schedule_rank.delta_one_residuals`` (a
    residual with zero opposite-basis syndrome that is not a stabilizer)."""

    for residuals in suffixes_w:
        for vector in residuals:
            if ((context.h_detect @ vector) % 2).any():
                continue
            if not _reduce_against(context.span_basis, vector).any():
                continue
            return True
    return False


@dataclass(frozen=True)
class CertificationResult:
    """One candidate order under one probing stage."""

    d_ext: int | None  # min over seeds (None: empty kernel / logicals, or nothing found)
    per_seed: tuple[int | None, ...]
    iters: int
    kernel_dim: int
    delta_one: bool

    @property
    def converged(self) -> bool | None:
        """All seeds agree (None when nothing was probed)."""

        if not self.per_seed:
            return None
        return len(set(self.per_seed)) == 1


def _probe_order(
    context: BasisCertContext,
    suffixes_w: list[list[np.ndarray]],
    *,
    iters: int,
    seed: int,
    seeds: int = 1,
    iters_scale: float = 0.0,
    seed_offset: int = 0,
    delta_one: bool | None = None,
) -> CertificationResult:
    if delta_one is None:
        delta_one = _delta_one_hit(context, suffixes_w)
    flat = [v for residuals in suffixes_w for v in residuals]
    h_ext, l_ext = extended_matrices(context.h_detect, context.logical_rows, flat)
    kernel = kernel_basis(h_ext)
    kernel_dim = int(kernel.shape[0])
    if kernel_dim == 0 or l_ext.shape[0] == 0:
        return CertificationResult(None, (), 0, kernel_dim, delta_one)
    effective = max(int(iters), int(float(iters_scale) * kernel_dim))
    values: list[int | None] = []
    for s in range(max(1, int(seeds))):
        rng = np.random.default_rng(int(seed) + _SEED_STRIDE * (int(seed_offset) + s))
        distance = _proxy_side(kernel, l_ext, effective, rng)
        values.append(int(distance) if distance < 10**9 else None)
    found = [v for v in values if v is not None]
    return CertificationResult(
        d_ext=min(found) if found else None,
        per_seed=tuple(values),
        iters=int(effective),
        kernel_dim=kernel_dim,
        delta_one=delta_one,
    )


# ------------------------------------------------------- parallel probes
# 2026-09-04 (NSCC utilisation finding): a candidate's certification is
# 2 bases x <=16 orders x n_seeds stage-1 ISD runs plus 2 x refine_top x
# n_seeds refine runs, and the sequential loop below ran them one after the
# other inside the ONE build process of that candidate -- with the slow queue
# capped at 4 candidates, a 128-core node sat 97 % idle for the ~40-75 min
# build wave of every turn. The (order, seed) runs are independent and each
# one's rng seed is a pure function of (seed0, index, side, seed_offset, s),
# so farming them out to a process pool is bit-identical to the loop: same
# per-seed values, same min, same bookkeeping. The pool is opt-in
# (OMNIQEC_SEC_WORKERS or the ``workers`` argument; default 1 = the original
# sequential path, untouched) and per rank_twobasis_schedules call.

_POOL_CONTEXTS: dict[str, BasisCertContext] = {}


def resolve_sec_workers(workers: int | None = None) -> int:
    """Worker count for the certification probes: the argument, else
    OMNIQEC_SEC_WORKERS, else 1; never above the visible CPU count."""

    if workers is None:
        raw = os.environ.get("OMNIQEC_SEC_WORKERS", "").strip()
        try:
            workers = int(raw) if raw else 1
        except ValueError:
            workers = 1
    return max(1, min(int(workers), int(os.cpu_count() or 1)))


def _pool_init(contexts: dict[str, BasisCertContext]) -> None:
    global _POOL_CONTEXTS
    _POOL_CONTEXTS = contexts


def _probe_seed_task(
    kind: str, flat: list[np.ndarray], iters: int, rng_seed: int
) -> tuple[int, int, int | None]:
    """One (order, seed) ISD run = the seed-level unit of ``_probe_order``
    (iters_scale 0, so effective iterations = ``iters``). Returns
    (kernel_dim, logical rows, distance or None)."""

    context = _POOL_CONTEXTS[kind]
    h_ext, l_ext = extended_matrices(context.h_detect, context.logical_rows, flat)
    kernel = kernel_basis(h_ext)
    kernel_dim = int(kernel.shape[0])
    n_logical = int(l_ext.shape[0])
    if kernel_dim == 0 or n_logical == 0:
        return kernel_dim, n_logical, None
    rng = np.random.default_rng(int(rng_seed))
    distance = _proxy_side(kernel, l_ext, int(iters), rng)
    return kernel_dim, n_logical, (int(distance) if distance < 10**9 else None)


class _ProbePool:
    """The probes of one synthesis: sequential (``workers`` <= 1, exactly the
    historical ``_probe_order`` calls in order) or fanned out per (order,
    seed) over a process pool and reassembled in seed order."""

    def __init__(self, workers: int, contexts: dict[str, BasisCertContext]) -> None:
        self.workers = int(workers)
        self.contexts = contexts
        self.executor: _futures.ProcessPoolExecutor | None = None
        if self.workers > 1:
            # fork wherever it exists (Linux, macOS): the build process is
            # single-threaded and numpy-only, and a spawn pool would re-import
            # the caller's __main__ (the stdin-driven worker module) in every
            # child. Windows has no fork and is not a deployment target.
            mp_context = _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else None
            self.executor = _futures.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp_context,
                initializer=_pool_init,
                initargs=(contexts,),
            )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def probe_many(
        self,
        kind: str,
        side: int,
        indices: list[int],
        suffix_cache: dict[int, list[list[np.ndarray]]],
        *,
        iters: int,
        seed0: int,
        n_seeds: int,
        seed_offset: int,
    ) -> dict[int, CertificationResult]:
        context = self.contexts[kind]
        if self.executor is None or not indices:
            return {
                index: _probe_order(
                    context, suffix_cache[index], iters=iters, seed=seed0 + index * 7 + side,
                    seeds=n_seeds, iters_scale=0.0, seed_offset=seed_offset, delta_one=False,
                )
                for index in indices
            }
        futures: dict[tuple[int, int], _futures.Future] = {}
        for index in indices:
            flat = [v for residuals in suffix_cache[index] for v in residuals]
            seed = seed0 + index * 7 + side
            for s in range(max(1, int(n_seeds))):
                rng_seed = int(seed) + _SEED_STRIDE * (int(seed_offset) + s)
                futures[(index, s)] = self.executor.submit(_probe_seed_task, kind, flat, int(iters), rng_seed)
        results: dict[int, CertificationResult] = {}
        for index in indices:
            values: list[int | None] = []
            kernel_dim = 0
            degenerate = False
            for s in range(max(1, int(n_seeds))):
                kernel_dim, n_logical, value = futures[(index, s)].result()
                if kernel_dim == 0 or n_logical == 0:
                    degenerate = True
                values.append(value)
            if degenerate:
                results[index] = CertificationResult(None, (), 0, kernel_dim, False)
                continue
            found = [v for v in values if v is not None]
            results[index] = CertificationResult(
                d_ext=min(found) if found else None,
                per_seed=tuple(values),
                iters=int(iters),
                kernel_dim=kernel_dim,
                delta_one=False,
            )
        return results


def certify_basis_order(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    kind: str,
    suffixes_w: list[list[np.ndarray]],
    iters: int,
    seed: int,
    seeds: int = 1,
    iters_scale: float = 0.0,
    context: BasisCertContext | None = None,
) -> tuple[int | None, bool]:
    """(d_ext estimate of the ``kind`` residual side, delta-one hit).

    The estimate is the MIN over ``seeds`` independent ISD runs of
    ``max(iters, iters_scale * kernel_dim)`` iterations each (seed + 1000*s).
    d_ext is None when the kernel or the logical set is empty or the probe
    finds nothing. ``context`` (from ``basis_cert_context``) skips the
    candidate-independent work when certifying many orders of one code."""

    if context is None:
        context = basis_cert_context(hx, hz, kind=kind)
    elif context.kind != str(kind).upper():
        raise ValueError("context kind does not match the requested kind")
    delta_one_hit = bool(
        delta_one_residuals(
            np.asarray(hx, dtype=np.uint8) % 2,
            np.asarray(hz, dtype=np.uint8) % 2,
            {"X": suffixes_w if context.kind == "X" else [], "Z": suffixes_w if context.kind == "Z" else []},
        )
    )
    result = _probe_order(
        context, suffixes_w, iters=iters, seed=seed, seeds=seeds, iters_scale=iters_scale,
        delta_one=delta_one_hit,
    )
    return result.d_ext, result.delta_one


def _candidate_layers(
    base: list[list[Edge]], *, kind: str, index: int
) -> list[list[Edge]]:
    order = counter_permutation(len(base), candidate=int(index), slot=_SLOT[kind])
    return [base[j] for j in order]


def _rank_value(d_ext: int | None) -> int:
    return -1 if d_ext is None else int(d_ext)


def _better(candidate: tuple[int, int | None], best: tuple[int, int | None] | None) -> bool:
    """Strictly larger estimate wins; ties keep the smaller index (canonical)."""

    if best is None:
        return True
    return _rank_value(candidate[1]) > _rank_value(best[1])


# ------------------------------------------------- budget / progress / model


def nominal_stage1_iters(policy: SecPolicyConfig, kernel_dim: int) -> int:
    """The plan's stage-1 ISD iterations per seed: max(iters, scale x kernel)."""

    iters = int(policy.sec_cert_iters)
    scale = float(getattr(policy, "sec_cert_iters_scale", 0.0))
    return max(iters, int(scale * int(kernel_dim)))


def certification_plan(
    policy: SecPolicyConfig, *, survivors: int, kernel_dim: int
) -> dict[str, Any]:
    """Deterministic ISD allocation of ONE basis under the per-code budget.

    Nominal work = survivors x seeds x stage1 + refine_count x seeds x
    refine_iters. With ``sec_cert_iters_budget`` = 0 (unlimited) the plan is
    the nominal one. Otherwise each basis owns half the budget; when the
    nominal work exceeds it, refine keeps at most a quarter of the basis
    budget (never more than its nominal need), stage 1 takes the rest
    (spread evenly over survivors x seeds, at least 1 iteration), and refine
    gets what stage 1 left (0 iterations = refine skipped). The rule reads
    only the policy and integer counts, so a resumed or re-run synthesis
    plans identically."""

    survivors = max(0, int(survivors))
    seeds = max(1, int(getattr(policy, "sec_cert_seeds", 1)))
    refine_top = max(0, int(getattr(policy, "sec_refine_top", 0)))
    refine_iters = max(0, int(getattr(policy, "sec_refine_iters", 0)))
    budget = max(0, int(getattr(policy, "sec_cert_iters_budget", 0)))
    stage1 = nominal_stage1_iters(policy, kernel_dim)
    refine_count = min(refine_top, survivors) if (refine_top > 0 and refine_iters > 0) else 0
    nominal_stage1_total = survivors * seeds * stage1
    nominal_refine_total = refine_count * seeds * refine_iters
    plan = {
        "survivors": survivors,
        "seeds": seeds,
        "kernel_dim": int(kernel_dim),
        "stage1_iters_nominal": int(stage1),
        "stage1_iters": int(stage1),
        "refine_count": int(refine_count),
        "refine_iters_nominal": int(refine_iters),
        "refine_iters": int(refine_iters if refine_count else 0),
        "basis_budget": None,
        "isd_iters_nominal": int(nominal_stage1_total + nominal_refine_total),
        "exhausted": False,
    }
    if budget <= 0 or survivors == 0:
        plan["isd_iters_planned"] = plan["isd_iters_nominal"]
        return plan
    basis_budget = budget // 2
    plan["basis_budget"] = int(basis_budget)
    if nominal_stage1_total + nominal_refine_total <= basis_budget:
        plan["isd_iters_planned"] = plan["isd_iters_nominal"]
        return plan
    refine_reserve = min(nominal_refine_total, basis_budget // 4)
    stage1_alloc = max(0, basis_budget - refine_reserve)
    stage1_eff = min(stage1, max(1, stage1_alloc // (survivors * seeds)))
    used = survivors * seeds * stage1_eff
    if refine_count:
        refine_eff = min(refine_iters, max(0, (basis_budget - used) // (refine_count * seeds)))
    else:
        refine_eff = 0
    plan["stage1_iters"] = int(stage1_eff)
    plan["refine_iters"] = int(refine_eff)
    plan["isd_iters_planned"] = int(used + refine_count * seeds * refine_eff)
    plan["exhausted"] = True
    return plan


def certification_isd_iterations(policy: SecPolicyConfig, *, kernel_dim: int) -> int:
    """Worst-case ISD iterations of one code (both bases, every candidate
    surviving Delta=1) under ``policy``; honours the per-code budget."""

    candidates = max(1, int(policy.sec_candidates))
    plan = certification_plan(policy, survivors=candidates, kernel_dim=kernel_dim)
    return 2 * int(plan["isd_iters_planned"])


def certification_work_model(policy: SecPolicyConfig, *, n_data: int) -> dict[str, Any]:
    """Predicted wall cost (s) of the certification for a code with
    ``n_data`` data qubits: ISD iterations from the plan (kernel ~ 4 n) x a
    per-iteration cost ~ (n/72)^2.8 (measured 1.5 ms at n=72 / kernel 222,
    27.5 ms at n=200 / kernel 760, both with a 2x margin in the constant).
    The pool sizes the build watchdog from this; it is a bound on the work
    the protocol itself schedules, not a measurement."""

    try:
        from omniqec_agentic.config import (
            SEC_CERT_BUILD_OVERHEAD_S,
            SEC_CERT_ISD_COST_EXPONENT,
            SEC_CERT_ISD_COST_S_AT_N72,
            SEC_CERT_KERNEL_PER_DATA_QUBIT,
        )
    except ImportError:  # pragma: no cover - fallback mirrors config.py
        SEC_CERT_BUILD_OVERHEAD_S = 600.0
        SEC_CERT_ISD_COST_EXPONENT = 2.8
        SEC_CERT_ISD_COST_S_AT_N72 = 3.0e-3
        SEC_CERT_KERNEL_PER_DATA_QUBIT = 4.0
    n = max(1, int(n_data))
    kernel_dim = int(round(SEC_CERT_KERNEL_PER_DATA_QUBIT * n))
    iterations = certification_isd_iterations(policy, kernel_dim=kernel_dim)
    cost_per_iter = float(SEC_CERT_ISD_COST_S_AT_N72) * (n / 72.0) ** float(SEC_CERT_ISD_COST_EXPONENT)
    return {
        "n_data": n,
        "kernel_dim_estimate": kernel_dim,
        "isd_iterations": int(iterations),
        "isd_cost_s_per_iter": float(cost_per_iter),
        "certification_s": float(iterations * cost_per_iter),
        "overhead_s": float(SEC_CERT_BUILD_OVERHEAD_S),
        "worst_case_s": float(iterations * cost_per_iter + SEC_CERT_BUILD_OVERHEAD_S),
    }


class CertificationProgress:
    """Checkpoint of finished probes so a killed synthesis resumes.

    Keyed by (basis, stage, candidate index); a stored entry is reused only
    when its iteration count and seed count match the current plan, so a
    checkpoint written under another policy can never leak in. JSON on disk
    (``path``); ``None`` path = in-memory only (tests)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.entries: dict[str, list[Any]] = {}
        self.reused = 0
        self.load()

    @staticmethod
    def _key(kind: str, stage: str, index: int) -> str:
        return f"{kind}:{stage}:{int(index)}"

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            self.entries = {str(k): list(v) for k, v in data["entries"].items()}

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps({"schema": "sec_certification_progress_v1", "entries": self.entries}),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except Exception:
            pass  # a lost checkpoint only costs recomputation

    def get(self, kind: str, stage: str, index: int, *, iters: int, seeds: int) -> CertificationResult | None:
        entry = self.entries.get(self._key(kind, stage, index))
        if not entry or len(entry) != 5:
            return None
        d_ext, per_seed, stored_iters, kernel_dim, delta_one = entry
        if int(stored_iters) != int(iters) or len(per_seed) != int(seeds):
            return None
        self.reused += 1
        return CertificationResult(
            d_ext=None if d_ext is None else int(d_ext),
            per_seed=tuple(None if v is None else int(v) for v in per_seed),
            iters=int(stored_iters),
            kernel_dim=int(kernel_dim),
            delta_one=bool(delta_one),
        )

    def put(self, kind: str, stage: str, index: int, result: CertificationResult) -> None:
        self.entries[self._key(kind, stage, index)] = [
            result.d_ext, list(result.per_seed), int(result.iters), int(result.kernel_dim), bool(result.delta_one)
        ]
        self.save()

    def discard(self) -> None:
        if self.path is None:
            return
        try:
            self.path.unlink()
        except OSError:
            pass


def _extended_kernel_dim(context: BasisCertContext, suffixes_w: list[list[np.ndarray]]) -> int:
    flat = [v for residuals in suffixes_w for v in residuals]
    h_ext, _l_ext = extended_matrices(context.h_detect, context.logical_rows, flat)
    return int(kernel_basis(h_ext).shape[0])


# --------------------------------------------------------------------- rank


def rank_twobasis_schedules(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    layer_blocks: dict[str, Any] | None,
    policy: SecPolicyConfig,
    d_code_proxy: int | None = None,
    progress: CertificationProgress | None = None,
    workers: int | None = None,
) -> RankedSchedule:
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    n_data = int(hx.shape[1])
    m_x, m_z = int(hx.shape[0]), int(hz.shape[0])
    base = build_layer_sets(hx, hz, layer_blocks=layer_blocks)
    candidates = max(1, int(policy.sec_candidates))
    iters = int(policy.sec_cert_iters)
    seed0 = int(policy.sec_cert_seed)
    n_seeds = max(1, int(getattr(policy, "sec_cert_seeds", 1)))
    iters_scale = float(getattr(policy, "sec_cert_iters_scale", 0.0))
    refine_top = max(0, int(getattr(policy, "sec_refine_top", 0)))
    refine_iters = max(0, int(getattr(policy, "sec_refine_iters", 0)))
    budget = max(0, int(getattr(policy, "sec_cert_iters_budget", 0)))

    # TB-5: logicals once, per-basis detector / logical rows / span basis once.
    logical_pair = logicals(hx, hz)
    contexts = {
        kind: basis_cert_context(hx, hz, kind=kind, logical_pair=logical_pair)
        for kind in ("X", "Z")
    }

    # Probe execution (sequential or a process pool): see _ProbePool. The
    # results and every metadata field are identical either way.
    pool = _ProbePool(resolve_sec_workers(workers), contexts)

    chosen: dict[str, tuple[int, int | None]] = {}
    per_candidate: dict[str, list[list[Any]]] = {"X": [], "Z": []}
    stage1_iters: dict[str, int | None] = {"X": None, "Z": None}
    kernel_dims: dict[str, int | None] = {"X": None, "Z": None}
    refined_indices: dict[str, list[int]] = {"X": [], "Z": []}
    refine_seed_values: dict[str, list[list[Any]]] = {"X": [], "Z": []}
    stage1_seed_values: dict[str, list[list[Any]]] = {"X": [], "Z": []}
    converged: dict[str, bool | None] = {"X": None, "Z": None}
    plans: dict[str, dict[str, Any] | None] = {"X": None, "Z": None}
    exhausted: dict[str, bool] = {"X": False, "Z": False}
    isd_runs = 0
    isd_iters_executed = 0
    eliminated = 0
    basis_catastrophic: dict[str, bool] = {}
    try:
        for kind, m_w in (("X", m_x), ("Z", m_z)):
            context = contexts[kind]
            side = _SLOT[kind]
            rows: dict[int, list[Any]] = {}
            suffix_cache: dict[int, list[list[np.ndarray]]] = {}
            # Pass 1 (cheap, exact): residual suffixes and the Delta=1 scan of
            # every candidate; the survivors' count sizes the ISD plan.
            survivors: list[int] = []
            for index in range(candidates):
                layers_w = _candidate_layers(base[kind], kind=kind, index=index)
                suffixes = basis_residual_suffixes(layers_w, m_w=m_w, n_data=n_data)
                if _delta_one_hit(context, suffixes):
                    eliminated += 1
                    rows[index] = [int(index), None, None, True]
                    continue
                suffix_cache[index] = suffixes
                survivors.append(index)
            if not survivors:
                per_candidate[kind] = [rows[i] for i in range(candidates)]
                basis_catastrophic[kind] = True
                chosen[kind] = (0, None)
                continue
            basis_catastrophic[kind] = False
            kernel_dim = _extended_kernel_dim(context, suffix_cache[survivors[0]])
            kernel_dims[kind] = kernel_dim
            plan = certification_plan(policy, survivors=len(survivors), kernel_dim=kernel_dim)
            plans[kind] = plan
            exhausted[kind] = bool(plan["exhausted"])
            stage1_eff = int(plan["stage1_iters"])
            refine_eff = int(plan["refine_iters"])
            refine_count = int(plan["refine_count"])
            stage1_iters[kind] = stage1_eff

            # Stage 1: every survivor, stage1_eff iterations x n_seeds seeds.
            # (checkpoint lookups first, then the missing probes -- in one batch
            # when a pool is configured -- then the bookkeeping in survivor order)
            stage1: list[tuple[int, int | None]] = []
            stage1_results: dict[int, CertificationResult] = {}
            pending: list[int] = []
            for index in survivors:
                hit = None
                if progress is not None:
                    hit = progress.get(kind, "stage1", index, iters=stage1_eff, seeds=n_seeds)
                if hit is not None:
                    stage1_results[index] = hit
                else:
                    pending.append(index)
            computed = pool.probe_many(
                kind, side, pending, suffix_cache,
                iters=stage1_eff, seed0=seed0, n_seeds=n_seeds, seed_offset=0,
            )
            for index in pending:
                result = computed[index]
                isd_runs += len(result.per_seed)
                isd_iters_executed += len(result.per_seed) * int(result.iters)
                if progress is not None:
                    progress.put(kind, "stage1", index, result)
                stage1_results[index] = result
            for index in survivors:
                result = stage1_results[index]
                if result.kernel_dim and result.kernel_dim != kernel_dim and index == survivors[0]:
                    kernel_dims[kind] = int(result.kernel_dim)
                rows[index] = [int(index), result.d_ext, None, False]
                stage1_seed_values[kind].append([int(index), list(result.per_seed)])
                stage1.append((index, result.d_ext))
            per_candidate[kind] = [rows[i] for i in range(candidates)]

            # Refine: top-K stage-1 survivors (largest estimate, ties -> smaller
            # index), fresh seeds continuing the same stream (s = n_seeds..2n-1).
            ranked = sorted(stage1, key=lambda item: (-_rank_value(item[1]), item[0]))
            head = ranked[:refine_count] if (refine_count > 0 and refine_eff > 0) else []
            best: tuple[int, int | None] | None = None
            best_converged: bool | None = None
            if head:
                refine_results: dict[int, CertificationResult] = {}
                pending = []
                for index, _d_stage1 in head:
                    hit = None
                    if progress is not None:
                        hit = progress.get(kind, "refine", index, iters=refine_eff, seeds=n_seeds)
                    if hit is not None:
                        refine_results[index] = hit
                    else:
                        pending.append(index)
                computed = pool.probe_many(
                    kind, side, pending, suffix_cache,
                    iters=refine_eff, seed0=seed0, n_seeds=n_seeds, seed_offset=n_seeds,
                )
                for index in pending:
                    result = computed[index]
                    isd_runs += len(result.per_seed)
                    isd_iters_executed += len(result.per_seed) * int(result.iters)
                    if progress is not None:
                        progress.put(kind, "refine", index, result)
                    refine_results[index] = result
                for index, d_stage1 in head:
                    result = refine_results[index]
                    found = [v for v in (d_stage1, result.d_ext) if v is not None]
                    refined = min(found) if found else None
                    rows[index][2] = refined
                    refined_indices[kind].append(int(index))
                    refine_seed_values[kind].append([int(index), list(result.per_seed)])
                    if _better((index, refined), best):
                        best = (index, refined)
                        best_converged = result.converged
            else:  # refine disabled or budget-skipped: stage-1 order decides
                for index, d_stage1 in stage1:
                    if _better((index, d_stage1), best):
                        best = (index, d_stage1)
                best_converged = None
            assert best is not None
            chosen[kind] = best
            converged[kind] = best_converged

    finally:
        pool.close()

    catastrophic = any(basis_catastrophic.values())
    if catastrophic:  # canonical orders, as rank_schedules does
        chosen = {"X": (0, None), "Z": (0, None)}
        converged = {"X": None, "Z": None}
    idx_x, raw_x = chosen["X"]
    idx_z, raw_z = chosen["Z"]
    proxy = None if d_code_proxy is None else int(d_code_proxy)

    def _clamp(value: int | None) -> int | None:
        if value is None or proxy is None:
            return value
        return min(int(value), proxy)

    d_x, d_z = _clamp(raw_x), _clamp(raw_z)
    budget_exhausted = bool(exhausted["X"] or exhausted["Z"])
    schedule = TwoBasisSchedule(
        n_data=n_data,
        m_x=m_x,
        m_z=m_z,
        layers={
            "X": _candidate_layers(base["X"], kind="X", index=idx_x),
            "Z": _candidate_layers(base["Z"], kind="Z", index=idx_z),
        },
        candidate_index_x=int(idx_x),
        candidate_index_z=int(idx_z),
        d_ext_x=d_x,
        d_ext_z=d_z,
        candidates=candidates,
        cert_iters=iters,
        cert_seeds=n_seeds,
        refine_iters=refine_iters,
        converged_x=converged["X"],
        converged_z=converged["Z"],
        budget_exhausted=budget_exhausted,
    )
    metadata = {
        "sec_policy_id": TWOBASIS_SEC_POLICY_ID,
        "candidates": candidates,
        "cert_iters": iters,
        "cert_seed": seed0,
        "cert_seeds": n_seeds,
        "cert_iters_scale": iters_scale,
        "refine_top": refine_top,
        "refine_iters": refine_iters,
        "stage1_iters": stage1_iters,
        "kernel_dim": kernel_dims,
        "chosen_index_x": int(idx_x),
        "chosen_index_z": int(idx_z),
        "candidates_evaluated": 2 * candidates,
        "isd_runs": int(isd_runs),
        "d_ext_x": d_x,
        "d_ext_z": d_z,
        "d_ext_x_unclamped": raw_x,
        "d_ext_z_unclamped": raw_z,
        "d_code_proxy": proxy,
        "clamped_x": bool(raw_x is not None and d_x != raw_x),
        "clamped_z": bool(raw_z is not None and d_z != raw_z),
        "converged_x": converged["X"],
        "converged_z": converged["Z"],
        "schedule_degraded_x": bool(raw_x is not None and proxy is not None and raw_x < proxy),
        "schedule_degraded_z": bool(raw_z is not None and proxy is not None and raw_z < proxy),
        "catastrophic": bool(catastrophic),
        # rows: [index, d_stage1, d_refined | None, delta_one]
        "per_candidate": per_candidate,
        "refined_indices": refined_indices,
        "stage1_seed_values": stage1_seed_values,
        "refine_seed_values": refine_seed_values,
        # Review fix: the per-code ISD budget, the per-basis plan it produced,
        # the iterations actually executed and whether the plan was shrunk.
        "cert_iters_budget": int(budget),
        "cert_plan": plans,
        "isd_iters_executed": int(isd_iters_executed),
        "certification_budget_exhausted": budget_exhausted,
        "certification_budget_exhausted_x": bool(exhausted["X"]),
        "certification_budget_exhausted_z": bool(exhausted["Z"]),
        "progress_entries_reused": int(progress.reused) if progress is not None else 0,
    }
    if catastrophic:
        metadata["reason"] = "all_candidates_have_single_fault_logical"
    return RankedSchedule(
        schedule=schedule,
        catastrophic=bool(catastrophic),
        delta_one_eliminated=int(eliminated),
        tau_a=None,
        d_circ_ext_x=d_x,
        d_circ_ext_z=d_z,
        stage3_group=1,
        metadata=metadata,
    )
