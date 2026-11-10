"""Archive and memory.

The fast metric is Q = k*d^2 (ranking.QUALITY_METRIC_ID kd2_v1; ver1 used
k*d^2/n until 2026-09-10) and
`cell=(overhead-bin, d)`. The agentic archive adds explicit hard eligibility
checks, such as `N_data <= N_budget`, through the structural result. This keeps
the ver1 score auditable while preventing budget-invalid codes from becoming
parents or elites.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from omniqec_agentic import ranking
from omniqec_agentic.memory.expected_plpr import ExpectedPlprModel, evidence_points
from omniqec_agentic.construction.groups.catalog import set_group_order_budget
from omniqec_agentic.sandbox import ProgramError, run_construct_source
from omniqec_agentic.schemas import ArchiveRecord, CandidateProgram, FastResult, StructuralResult
from omniqec_agentic.verification.fast import evaluate_fast_result
from omniqec_agentic.verification.structural import structural_check_spec

# A queue candidate whose slow verification fails this many consecutive times
# without yielding a measurement stops being re-queued (it would otherwise
# retry on every promotion and burn slow budget indefinitely).
MAX_SLOW_FAILURES = 2


# ---- staged fast evaluation (2026-09-11) ----------------------------------
# evaluate_and_record used to run construct -> structural check -> duplicate
# lookup -> proxy distance -> archive insertion as ONE sequential call per
# (program, rng seed). With 24 proposals x 40 seeds and a 10 s construct
# timeout on every seed of a pathological program, one generation of the
# DeepSeek arm took 40+ minutes on a node with 127 idle cores. The stages
# below split the pure work (construct, structural check, proxy) from the
# order-dependent work (duplicate ownership, MAP-cell competition, ledger,
# attempt log), so evaluate_batch can run the pure part in a process pool and
# replay the order-dependent part in the original sequence -- same records,
# same attempts, same owners, bit for bit.


@dataclass(frozen=True)
class ConstructStage:
    """Everything evaluate_and_record derived from a program before it looked
    at the archive: a sandbox error, a family rejection, or the constructed
    spec with its structural verdict."""

    program_id: str
    error: str = ""
    constructed: Any = None
    structural: StructuralResult | None = None
    family_rejected: bool = False


def construct_stage(
    *,
    source: str,
    program_id: str,
    seed: int,
    n_budget: int,
    allowed_families: tuple[str, ...] | None,
    deformation_mode: str | None,
    lp_shape_pin: Any,
    max_check_weight: int | None,
    exact_check_weight: int | None = None,
) -> ConstructStage:
    try:
        sandbox_result = run_construct_source(source, program_id=program_id, seed=seed)
    except ProgramError as exc:
        return ConstructStage(program_id=program_id, error=str(exc))
    constructed = sandbox_result.constructed
    if allowed_families is not None and constructed.family not in allowed_families:
        return ConstructStage(program_id=program_id, constructed=constructed, family_rejected=True)
    # n_budget is the TOTAL qubit budget (data + one ancilla per check
    # row); the historical data-only gate is no longer used here.
    structural = structural_check_spec(
        program_id,
        constructed.spec,
        total_budget=n_budget,
        deformation_mode=deformation_mode,
        lp_shape_pin=lp_shape_pin,
        max_check_weight=max_check_weight,
        exact_check_weight=exact_check_weight,
    )
    return ConstructStage(program_id=program_id, constructed=constructed, structural=structural)


def proxy_stage(
    *,
    program_id: str,
    spec: dict[str, Any],
    n_budget: int,
    iters: int,
    seeds: tuple[int, ...],
    algorithm_id: str | None,
    n_min: int | None,
    n_max: int | None,
    dsqrtn_floor: float,
    deformation_mode: str | None,
    lp_shape_pin: Any,
) -> FastResult:
    return evaluate_fast_result(
        program_id,
        spec,
        n_budget=n_budget,
        iters=iters,
        seeds=seeds,
        algorithm_id=algorithm_id,
        n_min=n_min,
        n_max=n_max,
        dsqrtn_floor=dsqrtn_floor,
        deformation_mode=deformation_mode,
        lp_shape_pin=lp_shape_pin,
    )


def _construct_stage_worker(kwargs: dict[str, Any]) -> ConstructStage:
    # Pool workers resolve lp_nonabelian groups under the same order cap the
    # main process set for this batch (funsearch re-asserts it per gen).
    set_group_order_budget(int(kwargs["n_budget"]))
    return construct_stage(**kwargs)


def _proxy_stage_worker(kwargs: dict[str, Any]) -> FastResult:
    set_group_order_budget(int(kwargs["n_budget"]))
    return proxy_stage(**kwargs)


def fast_workers_from_env(default: int = 1) -> int:
    """OMNIQEC_FAST_WORKERS: process count for evaluate_batch (1 = the
    original sequential loop). Clamped to the visible CPU count."""

    import os

    raw = os.environ.get("OMNIQEC_FAST_WORKERS", "").strip()
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    cpus = os.cpu_count() or 1
    return max(1, min(value, cpus))


@dataclass
class ArchiveMemory:
    records_by_cell: dict[str, ArchiveRecord] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    code_hash_owner: dict[str, str] = field(default_factory=dict)
    # Structural retention channel (v3.2 §4.2). Independent of records_by_cell
    # and of the LER hall of fame: a code evicted from its MAP niche by a
    # higher-Q_eff neighbour, or measured badly, keeps its full inheritance
    # payload here as long as it holds a (k, d_search) Pareto position.
    #
    # Audit fix #3: the ledger below holds EVERY structurally eligible row ever
    # seen (hash-keyed, no cap). The elite view is a batch reduce over the full
    # ledger, computed lazily at read time — reducing an already-capped pool
    # against each arriving candidate (the old streaming shape) is not
    # associative, so the surviving elites depended on LLM output order.
    structural_ledger: dict[str, dict[str, Any]] = field(default_factory=dict)
    _structural_elites_cache: list[dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )
    structural_proxy_elites_limit: int = 24
    # v3.6 (docs/06 §3 B4): at most this many realizations of one
    # (k, d_search) coordinate in the reduced view (StructuralSearchConfig.
    # coordinate_cap; None/0 = the v1 uncapped reducer).
    structural_coordinate_cap: int | None = ranking.STRUCTURAL_COORDINATE_CAP
    structural_parent_quota: int = 6
    runtime_parent_quota: int = 6
    diversity_parent_quota: int = 4
    # Frozen d_search protocol (v3.2 §3): identical for every family. The
    # algorithm id is the DECLARED protocol; the proxy layer fails closed if
    # it does not match the implementation (declaration = execution).
    proxy_seeds: tuple[int, ...] = (0, 1, 2, 3)
    proxy_algorithm_id: str | None = None
    # Turn stamp for first_seen/last_seen bookkeeping in the elite cache.
    turn: int = 0
    # LER hall-of-fame: refs to the best-measured (lowest plpr) NON-degraded
    # records ever, kept alive even after their MAP-Elites niche is overwritten
    # by a higher-Q_eff code (the niche competition is Q-driven, fast_Q does NOT
    # predict LER, so a low-Q/great-LER code can be evicted from records_by_cell;
    # this hall-of-fame preserves it for the parent pool + the headline champion).
    hall_of_fame: list[ArchiveRecord] = field(default_factory=list)
    hall_of_fame_limit: int = 16
    # Unified ranking knobs (see ranking.py); funsearch wires these from
    # VerificationConfig so fast/slow ordering share one semantics.
    d_cap_enabled: bool = True
    d_cap: int = ranking.DEFAULT_D_CAP
    dsqrtn_floor: float = ranking.DEFAULT_DSQRTN_FLOOR
    # Per-family campaign gate: when set, proposals outside these families are
    # rejected before the proxy runs (keeps single-family arms uncontaminated).
    allowed_families: tuple[str, ...] | None = None
    deformation_mode: str | None = None
    # Run-level LP seed-shape pin (design doc §3.5); enforced at build time.
    lp_shape_pin: Any = None
    # Per-check weight cap; v3.7: None (no cap) unless a hardware spec tightens it.
    max_check_weight: int | None = None
    # Fig.4 c (2026-10-23): exact maximum check weight; a code whose max row
    # weight differs is a structural reject (check_weight_mismatch).
    exact_check_weight: int | None = None
    # Fig.3 Phase B: the structural queue's third slot is the front's lightest
    # cell (StructuralSearchConfig.lightest_endpoint_slot).
    lightest_endpoint_slot: bool = False
    # v3.7: cross-turn measured evidence (code_hash, plpr, max_check_weight,
    # d_search, k, family) handed in by the campaign graph. Together with the
    # records measured in THIS cycle it feeds the expected-plpr model that
    # orders the runtime channel (ranking.archive_order_key expected_plpr).
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    # v3.8 endpoint family rules (config.StructuralSearchConfig): the second
    # endpoint slot prefers another family than the first, and families the
    # campaign graph demoted (endpoint_demotion_strikes) leave the structural
    # channel. Both off = the v3.7 slot order.
    endpoint_family_distinct: bool = False
    demoted_families: tuple[str, ...] = ()
    _plpr_model_cache: Any = field(default=None, repr=False, compare=False)

    def evaluate_and_record(
        self,
        program: CandidateProgram,
        *,
        n_budget: int,
        proxy_iters: int = 120,
        proxy_n_min: int | None = None,
        proxy_n_max: int | None = None,
    ) -> ArchiveRecord:
        stage = construct_stage(
            source=program.source,
            program_id=program.program_id,
            seed=program.rng_seed,
            n_budget=n_budget,
            allowed_families=self.allowed_families,
            deformation_mode=self.deformation_mode,
            lp_shape_pin=self.lp_shape_pin,
            max_check_weight=self.max_check_weight,
            exact_check_weight=self.exact_check_weight,
        )
        fast: FastResult | None = None
        if (
            stage.structural is not None
            and stage.structural.valid
            and self._duplicate_owner(stage.structural) is None
        ):
            fast = self._proxy_for(
                stage,
                n_budget=n_budget,
                proxy_iters=proxy_iters,
                proxy_n_min=proxy_n_min,
                proxy_n_max=proxy_n_max,
            )
        return self._finalize(
            program,
            stage,
            fast,
            n_budget=n_budget,
            proxy_iters=proxy_iters,
            proxy_n_min=proxy_n_min,
            proxy_n_max=proxy_n_max,
        )

    def _proxy_kwargs(
        self,
        stage: ConstructStage,
        *,
        n_budget: int,
        proxy_iters: int,
        proxy_n_min: int | None,
        proxy_n_max: int | None,
    ) -> dict[str, Any]:
        return {
            "program_id": stage.program_id,
            "spec": stage.constructed.spec,
            "n_budget": n_budget,
            "iters": proxy_iters,
            "seeds": self.proxy_seeds,
            "algorithm_id": self.proxy_algorithm_id,
            "n_min": proxy_n_min,
            "n_max": proxy_n_max,
            "dsqrtn_floor": self.dsqrtn_floor,
            "deformation_mode": self.deformation_mode,
            "lp_shape_pin": self.lp_shape_pin,
        }

    def _proxy_for(self, stage: ConstructStage, **kwargs: Any) -> FastResult:
        return proxy_stage(**self._proxy_kwargs(stage, **kwargs))

    def evaluate_batch(
        self,
        programs: list[CandidateProgram],
        *,
        n_budget: int,
        proxy_iters: int = 120,
        proxy_n_min: int | None = None,
        proxy_n_max: int | None = None,
        workers: int = 1,
        on_record: Any = None,
    ) -> list[ArchiveRecord]:
        """evaluate_and_record over a whole batch, in the batch's order.

        workers > 1: the pure stages run in a process pool (fork), then the
        order-dependent finalisation replays sequentially. Duplicate ownership
        is simulated first so the proxy is computed exactly for the entries
        the sequential loop would have computed it for; should the simulation
        ever disagree with _finalize, _finalize computes the proxy itself
        (slower, never wrong). `on_record(record)` runs after each record is
        appended, i.e. where the sequential caller used to look at
        attempts[-1]."""

        records: list[ArchiveRecord] = []
        workers = max(1, int(workers))
        if workers == 1 or len(programs) <= 1:
            for program in programs:
                record = self.evaluate_and_record(
                    program,
                    n_budget=n_budget,
                    proxy_iters=proxy_iters,
                    proxy_n_min=proxy_n_min,
                    proxy_n_max=proxy_n_max,
                )
                records.append(record)
                if on_record is not None:
                    on_record(record)
            return records

        import multiprocessing as _mp

        ctx = _mp.get_context("fork") if "fork" in _mp.get_all_start_methods() else _mp.get_context()
        construct_jobs = [
            {
                "source": program.source,
                "program_id": program.program_id,
                "seed": program.rng_seed,
                "n_budget": n_budget,
                "allowed_families": self.allowed_families,
                "deformation_mode": self.deformation_mode,
                "lp_shape_pin": self.lp_shape_pin,
                "max_check_weight": self.max_check_weight,
                # 2026-11-10: forwarded here too -- the pool path used to drop it,
                # so the Fig.4 c exact contract held only with OMNIQEC_FAST_WORKERS=1.
                "exact_check_weight": self.exact_check_weight,
            }
            for program in programs
        ]
        fasts: dict[int, FastResult] = {}
        with ctx.Pool(processes=min(workers, len(programs))) as pool:
            stages = pool.map(_construct_stage_worker, construct_jobs, chunksize=1)
            # Replay of the sequential duplicate rule: the FIRST structurally
            # valid entry of a code hash (archive-wide, then batch order) owns
            # it and gets the proxy; later entries are duplicates and skip it.
            # A structural reject never owns a hash and never gets the proxy
            # (paper B2: the checks run before distance estimation).
            owners = set(self.code_hash_owner)
            need: list[int] = []
            for index, stage in enumerate(stages):
                if stage.structural is None or not stage.structural.valid:
                    continue
                code_hash = stage.structural.code_hash
                if code_hash and code_hash in owners:
                    continue
                need.append(index)
                if code_hash:
                    owners.add(code_hash)
            if need:
                proxy_jobs = [
                    self._proxy_kwargs(
                        stages[index],
                        n_budget=n_budget,
                        proxy_iters=proxy_iters,
                        proxy_n_min=proxy_n_min,
                        proxy_n_max=proxy_n_max,
                    )
                    for index in need
                ]
                for index, fast in zip(need, pool.map(_proxy_stage_worker, proxy_jobs, chunksize=1)):
                    fasts[index] = fast
        for index, (program, stage) in enumerate(zip(programs, stages)):
            record = self._finalize(
                program,
                stage,
                fasts.get(index),
                n_budget=n_budget,
                proxy_iters=proxy_iters,
                proxy_n_min=proxy_n_min,
                proxy_n_max=proxy_n_max,
            )
            records.append(record)
            if on_record is not None:
                on_record(record)
        return records

    def _finalize(
        self,
        program: CandidateProgram,
        stage: ConstructStage,
        fast: FastResult | None,
        *,
        n_budget: int,
        proxy_iters: int,
        proxy_n_min: int | None,
        proxy_n_max: int | None,
    ) -> ArchiveRecord:
        if stage.error:
            record = ArchiveRecord(program=program, notes=[stage.error])
            self._append_attempt(
                program=program,
                constructed=None,
                structural=None,
                fast=None,
                archive_status="not_accepted",
                accepted=False,
                duplicate_of=None,
                error=stage.error,
            )
            return record

        constructed = stage.constructed
        if stage.family_rejected:
            record = ArchiveRecord(program=program, constructed=constructed)
            self._append_attempt(
                program=program,
                constructed=constructed,
                structural=None,
                fast=None,
                archive_status="family_not_allowed",
                accepted=False,
                duplicate_of=None,
                error=f"family_not_allowed:{constructed.family}",
            )
            return record
        structural = stage.structural
        assert structural is not None
        if not structural.valid:
            # Paper alignment (2026-11-10, appendix B2): every realization
            # passes the validity checks -- CSS commutation, k > 0, total
            # qubits <= N, w_max <= cap (or the exact Fig.4 c contract), a
            # connected Tanner graph -- BEFORE the duplicate check and BEFORE
            # distance estimation. A failing realization is rejected with its
            # reasons recorded; it gets no proxy, owns no code hash, and never
            # reaches the MAP archive, the structural pool, the parents or a
            # slow queue.
            record = ArchiveRecord(
                program=program,
                constructed=constructed,
                structural=structural,
                archive_cell=None,
                parent_eligible=False,
            )
            self._append_attempt(
                program=program,
                constructed=constructed,
                structural=structural,
                fast=None,
                archive_status="structural_reject",
                accepted=False,
                duplicate_of=None,
                error="",
            )
            return record
        duplicate_of = self._duplicate_owner(structural)
        if duplicate_of:
            # Duplicate code (identical HX/HZ): skip the expensive proxy
            # distance entirely — the verdict cannot change. This makes
            # multi-rng-seed evaluation of deterministic programs nearly free.
            record = ArchiveRecord(
                program=program,
                constructed=constructed,
                structural=structural,
                archive_cell=None,
                parent_eligible=False,
            )
            self._append_attempt(
                program=program,
                constructed=constructed,
                structural=structural,
                fast=None,
                archive_status="duplicate",
                accepted=False,
                duplicate_of=duplicate_of,
                error=f"duplicate of {duplicate_of}",
            )
            return record
        if fast is None:
            fast = self._proxy_for(
                stage,
                n_budget=n_budget,
                proxy_iters=proxy_iters,
                proxy_n_min=proxy_n_min,
                proxy_n_max=proxy_n_max,
            )
        archive_status = "not_accepted"
        accepted = False
        cell = cell_of_fast(fast)
        structural_eligible = bool(
            fast.proxy_valid
            and not fast.below_min_promotion_distance
            and structural.valid
            and (fast.k or 0) > 0
            and not duplicate_of
            and constructed is not None
            and str(program.source or "").strip()
        )
        # MAP/runtime eligibility keeps the historical meaning (it still runs
        # through the d/sqrt(n) trust screen); structural eligibility does not.
        map_runtime_eligible = bool(
            fast.runtime_screen_passed and structural.valid and (fast.k or 0) > 0 and not duplicate_of
        )
        record = ArchiveRecord(
            program=program,
            constructed=constructed,
            structural=structural,
            fast=fast,
            archive_cell=cell,
            parent_eligible=map_runtime_eligible,
        )
        # THE ORDER MATTERS (v3.2 §4.1): the structural pool is updated with
        # the complete record BEFORE the MAP cell competition runs, because
        # that competition keeps only the per-cell Q_eff incumbent and the
        # loser's spec/construct_source stops existing anywhere else.
        if structural_eligible:
            self.record_structural_candidate(record)
        if fast.proxy_valid and not structural.valid:
            archive_status = "structural_reject"
        elif map_runtime_eligible and cell is not None:
            current = self.records_by_cell.get(cell)
            # Niche competition compares Q_eff: an incumbent downgraded by slow
            # evidence (d_circ < d_search) defends with its corrected score.
            if current is None or self._record_q_eff(record) > self._record_q_eff(current):
                self.records_by_cell[cell] = record
                archive_status = "accepted"
                accepted = True
        if not accepted and archive_status == "not_accepted" and structural_eligible:
            # Retained by the structural channel: it is NOT a rejection, and
            # reporting it as one to the generator teaches the search to avoid
            # exactly the region this channel was built to keep.
            archive_status = "structural_only" if not map_runtime_eligible else "not_better"

        if structural.code_hash and structural.code_hash not in self.code_hash_owner:
            self.code_hash_owner[structural.code_hash] = program.program_id

        self._append_attempt(
            program=program,
            constructed=constructed,
            structural=structural,
            fast=fast,
            archive_status=archive_status,
            accepted=accepted,
            duplicate_of=duplicate_of,
            error="",
            structural_eligible=structural_eligible,
            map_runtime_eligible=map_runtime_eligible,
        )
        return record

    # ---- structural retention channel (v3.2 §4.1-4.3) ---------------------
    def structural_candidate_row(self, record: ArchiveRecord) -> dict[str, Any] | None:
        """The persisted structural artifact for one candidate (§4.2)."""

        fast = record.fast
        structural = record.structural
        constructed = record.constructed
        if fast is None or structural is None or constructed is None:
            return None
        metrics = dict(fast.metrics or {})
        code_hash = structural.code_hash or metrics.get("code_hash")
        if not code_hash:
            return None
        m_x, m_z = metrics.get("m_x"), metrics.get("m_z")
        n_total = metrics.get("n_total")
        budget = fast.n_budget or None
        return {
            "schema": "structural_proxy_elite",
            "reducer_id": ranking.STRUCTURAL_REDUCER_ID,
            "code_hash": str(code_hash),
            "program_id": record.program.program_id,
            "family": fast.family,
            "n": fast.n_data,
            "m_x": m_x,
            "m_z": m_z,
            "n_total": n_total,
            "budget_utilization": (
                float(n_total) / float(budget) if n_total is not None and budget else None
            ),
            "k": fast.k,
            "d_search": fast.d_search,
            "d_proxy": fast.d_search,
            "Q_search": metrics.get("Q_search", fast.q_score),
            "proxy_search": metrics.get("proxy_search"),
            "proxy_valid": bool(fast.proxy_valid),
            "runtime_screen_passed": bool(fast.runtime_screen_passed),
            "trust_filter_passed": bool(fast.trust_filter_passed),
            "archive_cell": record.archive_cell,
            "max_check_weight": fast.max_check_weight,
            "max_qubit_degree": fast.max_qubit_degree,
            "spec": constructed.spec,
            "construct_source": record.program.source,
            "rng_seed": record.program.rng_seed,
            "source_agent": record.program.source_agent,
            "deformation_mode": structural.deformation,
            # wl_hash is INFORMATIONAL (v3.6 B3); canonical_hash + code_hash
            # are the twin gate. Both persist in the structural ledger rows.
            "wl_hash": (structural.metrics or {}).get("wl_hash"),
            "canonical_hash": structural.canonical_hash
            or (structural.metrics or {}).get("canonical_hash"),
            "first_seen_turn": int(self.turn),
            "last_seen_turn": int(self.turn),
            "selection_channels": ["structural"],
        }

    @property
    def structural_proxy_elites(self) -> list[dict[str, Any]]:
        """The reduced (k, d_search) Pareto view over the FULL ledger (§4.2).

        Read-only and order-independent: whatever order candidates arrived in,
        the elites are ``reduce(all structurally eligible rows ever seen)``."""

        if self._structural_elites_cache is None:
            self._structural_elites_cache = ranking.reduce_structural_proxy_elites(
                [dict(row) for row in self.structural_ledger.values()],
                limit=int(self.structural_proxy_elites_limit),
                coordinate_cap=self.structural_coordinate_cap,
            )
        return self._structural_elites_cache

    def record_structural_candidate(self, record: ArchiveRecord) -> None:
        row = self.structural_candidate_row(record)
        if row is None:
            return
        self._merge_structural_row(row)

    def _merge_structural_row(self, row: dict[str, Any]) -> None:
        """Merge ONE structural artifact into the lossless ledger.

        A hash seen again keeps its earliest source/spec and first_seen_turn;
        an identity conflict on the same hash is recorded as a protocol error
        instead of silently taking the friendlier value."""

        if not isinstance(row, dict):
            return
        key = str(row.get("code_hash") or "")
        if not key:
            return
        self._structural_elites_cache = None
        held = self.structural_ledger.get(key)
        if held is None:
            self.structural_ledger[key] = dict(row)
            return
        conflicts = [
            field_name
            for field_name in ("n", "k")
            if held.get(field_name) is not None
            and row.get(field_name) is not None
            and int(held[field_name]) != int(row[field_name])
        ]
        if conflicts:
            errors = list(held.get("protocol_errors") or [])
            errors.append(
                f"canonical identity conflict on {','.join(conflicts)} for {key[:12]}"
            )
            held["protocol_errors"] = errors
            return
        # Earliest provenance wins; proxy evidence and last_seen refresh.
        held["last_seen_turn"] = max(
            int(held.get("last_seen_turn") or 0), int(row.get("last_seen_turn") or 0)
        )
        held["first_seen_turn"] = min(
            int(held.get("first_seen_turn") or 0), int(row.get("first_seen_turn") or 0)
        )
        for field_name in (
            "proxy_search",
            "d_search",
            "d_proxy",
            "Q_search",
            "proxy_valid",
            "runtime_screen_passed",
            "trust_filter_passed",
            "archive_cell",
        ):
            if row.get(field_name) is not None:
                held[field_name] = row[field_name]
        # Identity fields are filled in when a pre-v3.6 row lacked them (the
        # canonical hash of a code never changes, so no conflict is possible).
        for field_name in ("canonical_hash", "wl_hash"):
            if held.get(field_name) is None and row.get(field_name) is not None:
                held[field_name] = row[field_name]

    def reconcile_structural_elites(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge new structural artifacts into the ledger; return the elites."""

        for row in rows:
            self._merge_structural_row(row)
        return self.structural_proxy_elites

    def seed_structural_elites(self, rows: list[dict[str, Any]] | None) -> None:
        """Load the previous turns' LEDGER (cross-turn survival, §4.2).

        A candidate stays a live structural citizen without being regenerated
        or re-measured this turn; seeding the full ledger (not the capped
        elite view) keeps cross-turn retention order-independent too."""

        for row in rows or []:
            self._merge_structural_row(row)

    def structural_slow_queue(
        self,
        *,
        slots: int,
        measured_hashes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """The machine's structural candidates for this turn, in slot order
        (§4.3 slots 1-3).

        Audit fix #4: the WHOLE unmeasured elite pool is emitted (slot order:
        high-k endpoint, high-d_search endpoint, then crowding
        representatives), not a pre-truncated top-``slots``. The queue builder
        applies hash/WL dedup and takes the first ``slots`` survivors —
        truncating here first meant one WL collision silently reshaped the
        queue to 2S+2R. ``slots`` is recorded on the rows for the audit trail;
        it no longer truncates the emission."""

        measured = {str(item) for item in (measured_hashes or set())}
        unmeasured = [
            row
            for row in self.structural_proxy_elites
            if str(row.get("code_hash") or "") not in measured
        ]
        selected = ranking.select_structural_slots(
            unmeasured,
            slots=len(unmeasured),
            lightest=bool(self.lightest_endpoint_slot),
            distinct_families=bool(self.endpoint_family_distinct),
            demoted_families=tuple(self.demoted_families or ()),
        )
        for index, row in enumerate(selected, start=1):
            row["promotion_rank"] = index
            row["promotion_status"] = "selected_for_slow_pending"
            row["promotion_rule"] = "structural_proxy_pareto_slot"
            row["structural_slots"] = int(slots)
            row["selection_channels"] = ["structural"]
        return selected

    def structural_parent_specs(self, *, limit: int) -> list[dict[str, Any]]:
        """Structural parents in the cache's channel-neutral order, iterated
        round-robin over UNIQUE (k, d_search) coordinates (§4.5, v3.6 §4.1):
        one row per coordinate in pool order first, then the second
        realizations. Same k with different d therefore appear side by side
        in the parent window instead of two realizations of one endpoint."""

        inheritable = [
            row
            for row in self.structural_proxy_elites
            if isinstance(row.get("spec"), dict)
            and str(row.get("construct_source") or "").strip()
        ]
        parents: list[dict[str, Any]] = []
        for row in ranking.structural_coordinate_round_robin(inheritable)[: max(0, int(limit))]:
            parent = dict(row)
            parent["selection_channels"] = ["structural"]
            parent["selection_reasons"] = ["structural_proxy_pareto"]
            parents.append(parent)
        return parents

    def archive_summary(self, *, limit: int = 12, order: str = "runtime") -> list[dict[str, Any]]:
        """Occupied MAP niches.

        ``order="runtime"`` keeps the historical quality ordering for internal
        callers (status heartbeat, diagnostics). ``order="niche"`` is the
        prompt-facing view: cells in their own order, and when there are more
        cells than fit, an evenly spaced sample across them rather than the
        quality head — showing the top of a ranking IS a ranking instruction.
        """

        if order == "niche":
            items = sorted(self.records_by_cell.items(), key=lambda item: _cell_sort_key(item[0]))
            if len(items) > int(limit) > 0:
                step = len(items) / float(int(limit))
                items = [items[min(len(items) - 1, int(index * step))] for index in range(int(limit))]
        else:
            items = sorted(
                self.records_by_cell.items(),
                key=lambda item: self._record_sort_key(item[1]),
                reverse=True,
            )[: int(limit)]
        rows = []
        for cell, record in items:
            fast = record.fast
            structural = record.structural
            evidence = record.slow_evidence
            d_eff_value, d_eff_source = ranking.effective_distance(
                fast.d_proxy if fast else None,
                evidence,
                d_cap_enabled=self.d_cap_enabled,
                d_cap=self.d_cap,
            )
            metrics = fast.metrics if fast is not None else {}
            rows.append(
                {
                    "cell": cell,
                    "program_id": record.program.program_id,
                    "family": fast.family if fast else None,
                    "n": fast.n_data if fast else None,
                    "n_checks": _check_total(metrics),
                    "n_budget": fast.n_budget if fast else None,
                    "k": fast.k if fast else None,
                    "d_search": fast.d_search if fast else None,
                    "d_proxy": fast.d_proxy if fast else None,
                    "Q": fast.q_score if fast else None,
                    "d_eff": d_eff_value,
                    "d_eff_source": d_eff_source,
                    "q_eff": self._record_q_eff(record),
                    "degraded": self._record_degraded(record),
                    "flagged_degraded": ranking.is_flagged_degraded(evidence),
                    "slow_per_logical_per_round": evidence.get("per_logical_per_round"),
                    "slow_decoder": evidence.get("decoder"),
                    "max_check_weight": fast.max_check_weight if fast else None,
                    "max_qubit_degree": fast.max_qubit_degree if fast else None,
                    "structural_valid": structural.valid if structural else None,
                    "structural_failures": structural.failure_reasons if structural else [],
                }
            )
        return rows

    def recent_attempts(self, *, limit: int = 12) -> list[dict[str, Any]]:
        return self.attempts[-int(limit) :]

    def recent_program_summaries(self, *, limit: int = 8) -> list[dict[str, Any]]:
        """Aggregate recent attempts per proposed program (multi-rng-seed aware).

        With `rng_seeds_per_program > 1`, a deterministic program leaves one
        real evaluation plus K-1 'duplicate of itself' attempt rows; rendering
        raw attempts would tell the model its proposal was rejected when seed 0
        was in fact accepted. One summary per base program reports the actual
        outcome; same-program seed duplicates are internal bookkeeping and are
        not surfaced."""

        groups: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for attempt in self.attempts:
            base = _base_program_id(str(attempt.get("program_id") or ""))
            if base not in groups:
                groups[base] = []
                order.append(base)
            groups[base].append(attempt)

        summaries: list[dict[str, Any]] = []
        for base in order[-max(0, int(limit)) :]:
            rows = groups[base]
            evaluated = [row for row in rows if not row.get("duplicate_of")]
            cross_dups = [
                row
                for row in rows
                if row.get("duplicate_of")
                and _base_program_id(str(row.get("duplicate_of"))) != base
            ]
            accepted_rows = [row for row in evaluated if row.get("accepted")]
            runtime_rows = [row for row in evaluated if row.get("map_runtime_eligible")]
            structural_rows = [row for row in evaluated if row.get("structural_eligible")]
            distinct_codes = {
                row.get("code_hash") for row in rows if row.get("code_hash")
            }
            best = max(
                accepted_rows or runtime_rows or structural_rows,
                key=lambda row: float(row.get("Q") or 0.0),
                default=None,
            )
            if best is not None:
                if accepted_rows:
                    outcome = "accepted"
                elif runtime_rows:
                    outcome = "not_better"
                else:
                    # Kept by the structural channel (it holds a (k, d_search)
                    # Pareto position); reporting this as a rejection would
                    # steer the search away from the region on purpose.
                    outcome = "retained"
            elif evaluated:
                outcome = "rejected"
            elif cross_dups:
                outcome = "duplicate"
            else:
                # Every seed duplicated this program's own first code and that
                # code never entered evaluation here (cannot normally happen).
                outcome = "rejected"
            failure_reasons: list[str] = []
            for row in evaluated:
                for reason in row.get("failure_reasons") or []:
                    if reason not in failure_reasons:
                        failure_reasons.append(str(reason))
            summaries.append(
                {
                    "program_id": base,
                    "family": next(
                        (row.get("family") for row in rows if row.get("family")), None
                    ),
                    "outcome": outcome,
                    "n": best.get("n") if best else None,
                    "k": best.get("k") if best else None,
                    "d_search": best.get("d_search") if best else None,
                    "d_proxy": best.get("d_proxy") if best else None,
                    "Q": best.get("Q") if best else None,
                    "seeds": len(rows),
                    "distinct_codes": len(distinct_codes),
                    # Fig.3 v2: identities of the reported code, for the
                    # measured / isomorphic-twin annotation of the feedback.
                    "code_hash": best.get("code_hash") if best else None,
                    "canonical_hash": best.get("canonical_hash") if best else None,
                    "accepted_count": len(accepted_rows),
                    "duplicate_of": _base_program_id(str(cross_dups[0]["duplicate_of"]))
                    if cross_dups and not best and not evaluated
                    else None,
                    "failure_reasons": failure_reasons[:4],
                }
            )
        return summaries

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "archive_memory",
            "archive": {
                cell: record.to_dict()
                for cell, record in sorted(self.records_by_cell.items())
            },
            "attempts": list(self.attempts),
            "code_hash_owner": dict(self.code_hash_owner),
            "structural_proxy_elites": list(self.structural_proxy_elites),
            "structural_ledger_size": len(self.structural_ledger),
        }

    def _append_attempt(
        self,
        *,
        program: CandidateProgram,
        constructed,
        structural: StructuralResult | None,
        fast: FastResult | None,
        archive_status: str,
        accepted: bool,
        duplicate_of: str | None,
        error: str,
        structural_eligible: bool = False,
        map_runtime_eligible: bool = False,
    ) -> None:
        failure_reasons: list[str] = []
        if structural is not None:
            failure_reasons.extend(structural.failure_reasons)
        if fast is not None:
            failure_reasons.extend(fast.failure_reasons)
        if error:
            failure_reasons.append(error)
        self.attempts.append(
            {
                "program_id": program.program_id,
                "family": constructed.family if constructed is not None else None,
                "valid_fast": fast.valid if fast is not None else False,
                "valid_structural": structural.valid if structural is not None else False,
                # The four states of v3.2 §4.1, recorded on every attempt so a
                # trust-only failure can be told apart from a real rejection.
                "proxy_valid": bool(fast.proxy_valid) if fast is not None else False,
                "runtime_screen_passed": (
                    bool(fast.runtime_screen_passed) if fast is not None else False
                ),
                "trust_filter_passed": (
                    bool(fast.trust_filter_passed) if fast is not None else False
                ),
                "structural_eligible": bool(structural_eligible),
                "map_runtime_eligible": bool(map_runtime_eligible),
                # A structural reject has no proxy result (B2: rejected before
                # distance estimation); its n / k come from the structural check.
                "n": fast.n_data if fast is not None else (
                    structural.n_data if structural is not None and archive_status == "structural_reject" else None
                ),
                "k": fast.k if fast is not None else (
                    structural.k if structural is not None and archive_status == "structural_reject" else None
                ),
                # v3.7.1: the weight of what was CONSTRUCTED, and whether the
                # d_search floor held it out of the structural ledger. Without
                # both, a weight class every one of whose codes is below the
                # floor is indistinguishable from one nobody ever tried.
                "max_check_weight": (
                    structural.max_check_weight if structural is not None else None
                ),
                "below_min_promotion_distance": (
                    bool(fast.below_min_promotion_distance) if fast is not None else False
                ),
                "d_search": fast.d_search if fast is not None else None,
                "d_proxy": fast.d_proxy if fast is not None else None,
                "Q": fast.q_score if fast is not None else None,
                "archive_cell": cell_of_fast(fast),
                "archive_status": archive_status,
                "accepted": accepted,
                "parent_eligible": bool(structural_eligible or map_runtime_eligible),
                "code_hash": structural.code_hash if structural is not None else None,
                # Fig.3 v2: the exact-isomorphism identity, so the program
                # feedback can name a relabelled copy of a measured code.
                "canonical_hash": (
                    (structural.canonical_hash or (structural.metrics or {}).get("canonical_hash"))
                    if structural is not None
                    else None
                ),
                "duplicate_of": duplicate_of,
                "failure_reasons": failure_reasons,
                "error": error,
            }
        )

    def _duplicate_owner(self, structural: StructuralResult) -> str | None:
        if not structural.code_hash:
            return None
        return self.code_hash_owner.get(structural.code_hash)

    # ---- reconcile: slow evidence write-back ------------------------------
    def _record_q_eff(self, record: ArchiveRecord | None) -> float:
        if record is None or record.fast is None or not record.fast.valid:
            return float("-inf")
        fast = record.fast
        d_eff_value, _ = ranking.effective_distance(
            fast.d_proxy,
            record.slow_evidence,
            d_cap_enabled=self.d_cap_enabled,
            d_cap=self.d_cap,
        )
        value = ranking.q_eff(fast.n_data, fast.k, d_eff_value)
        return float(value) if value is not None else float("-inf")

    def expected_plpr_model(self) -> ExpectedPlprModel | None:
        """The v3.7 measured-evidence model, or None before any measurement.

        Built from the campaign's cross-turn evidence rows plus every record
        of this cycle that already carries slow evidence; rebuilt lazily after
        `apply_slow_evidence` (the cache is dropped there)."""

        if self._plpr_model_cache is not None:
            return self._plpr_model_cache or None
        rows: list[dict[str, Any]] = list(self.evidence_rows or [])
        for record in self.records_by_cell.values():
            evidence = record.slow_evidence or {}
            if evidence.get("per_logical_per_round") is None:
                continue
            fast = record.fast
            structural = record.structural
            rows.append(
                {
                    "code_hash": structural.code_hash if structural else record.program.program_id,
                    "per_logical_per_round": evidence.get("per_logical_per_round"),
                    "max_check_weight": fast.max_check_weight if fast else None,
                    "d_search": fast.d_search if fast else None,
                    "k": fast.k if fast else None,
                    "family": fast.family if fast else None,
                }
            )
        points = evidence_points(rows)
        model = ExpectedPlprModel(points) if points else None
        # `False` marks "computed, empty" so the empty case is cached too.
        self._plpr_model_cache = model if model is not None else False
        return model

    def expected_plpr_of(self, record: ArchiveRecord) -> float | None:
        """Model prediction for an UNMEASURED record; None when measured or
        when the campaign has no measurements yet."""

        fast = record.fast
        if fast is None or (record.slow_evidence or {}).get("per_logical_per_round") is not None:
            return None
        model = self.expected_plpr_model()
        if model is None:
            return None
        return model.predict(
            max_check_weight=fast.max_check_weight,
            d_search=fast.d_search,
            k=fast.k,
            family=fast.family,
        )

    def _record_sort_key(self, record: ArchiveRecord) -> tuple:
        fast = record.fast
        return ranking.archive_order_key(
            n=fast.n_data if fast else None,
            k=fast.k if fast else None,
            d_proxy=fast.d_proxy if fast else None,
            slow_evidence=record.slow_evidence,
            d_cap_enabled=self.d_cap_enabled,
            d_cap=self.d_cap,
            expected_plpr=self.expected_plpr_of(record),
        )

    def _record_degraded(self, record: ArchiveRecord) -> bool:
        # v3.6 (M2): HARD quarantine only (schedule_catastrophic). degraded_ext
        # / degraded_isd / degraded_estimate are audit flags and no longer
        # demote a measured record below unmeasured ones or evict it from the
        # hall of fame / parent pool.
        return ranking.is_degraded(record.slow_evidence)

    def apply_slow_evidence(
        self,
        candidates: list[dict[str, Any]],
    ) -> int:
        """Write slow verification results back into matching archive records.

        `candidates` are compact slow payload rows (must carry `code_hash`).
        Returns the number of archive records updated. Degraded records stay
        archived (audit trail) but lose parent eligibility
        through `ranking.is_degraded` at selection time."""

        by_hash: dict[str, ArchiveRecord] = {}
        for record in self.records_by_cell.values():
            if record.structural is not None and record.structural.code_hash:
                by_hash[record.structural.code_hash] = record
        updated = 0
        self._plpr_model_cache = None  # v3.7: new measurements re-fit the model
        for row in candidates:
            if not isinstance(row, dict):
                continue
            code_hash = str(row.get("code_hash") or "")
            record = by_hash.get(code_hash)
            if record is None:
                continue
            evidence = record.slow_evidence
            for key in (
                "per_logical_per_round",
                "per_logical_block_ci",
                "block_ler",
                "decoder",
                "p",
                "d_circ_ler_estimate",
                "d_circ_ler_estimate_status",
                "degraded_estimate",
                "degraded_estimate_any_decoder",
                "degraded_isd",
                "exact_d_circ",
                "exact_d_circ_status",
                "exact_code_distance",
                "exact_code_distance_status",
                "d_circ_isd_upper_bound",
                "d_circ_isd_upper_bound_status",
                "d_circ_ext",
                "d_circ_ext_status",
                "degraded_ext",
                "degraded_ext_converged",
                "schedule_catastrophic",
                "verified_circuit_distance",
            ):
                if row.get(key) is not None:
                    evidence[key] = row.get(key)
            evidence["slow_program_id"] = row.get("program_id")
            evidence["slow_status"] = row.get("status")
            # Consecutive-failure counter: a candidate whose slow run keeps
            # failing without producing a measurement (timeout, build error,
            # no_valid_results) is excluded from future queues after
            # MAX_SLOW_FAILURES strikes instead of retrying every promotion.
            if str(row.get("status") or "") == "ok":
                evidence["slow_failure_count"] = 0
            elif row.get("per_logical_per_round") is None:
                evidence["slow_failure_count"] = int(evidence.get("slow_failure_count") or 0) + 1
            d_eff_value, d_eff_source = ranking.effective_distance(
                record.fast.d_proxy if record.fast else None,
                evidence,
                d_cap_enabled=self.d_cap_enabled,
                d_cap=self.d_cap,
            )
            evidence["d_eff"] = d_eff_value
            evidence["d_eff_source"] = d_eff_source
            evidence["q_eff"] = self._record_q_eff(record)
            updated += 1
        self.update_hall_of_fame()
        return updated

    @staticmethod
    def _record_plpr(record: ArchiveRecord) -> float | None:
        value = (record.slow_evidence or {}).get("per_logical_per_round")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def update_hall_of_fame(self) -> None:
        """Refresh the LER hall-of-fame from the freshly-reconciled cell records.

        Runs right after slow evidence is written, while this gen's measured
        codes are still in their niches (before a later gen's higher-Q_eff code
        can evict them). Keeps one ref per code_hash, the lowest plpr, top-N;
        excludes hard-quarantined (schedule_catastrophic, v3.6 M2) and
        resident-baseline records (blinding). degraded_ext / degraded_isd
        records STAY: their measurement is real."""

        pool: dict[str, ArchiveRecord] = {}
        for record in list(self.hall_of_fame) + list(self.records_by_cell.values()):
            if self._record_degraded(record):
                continue
            structural = record.structural
            code_hash = structural.code_hash if structural else None
            plpr = self._record_plpr(record)
            if plpr is None or not code_hash:
                continue
            if str(getattr(record.program, "program_id", "") or "").startswith("baseline"):
                continue
            incumbent = pool.get(code_hash)
            if incumbent is None or plpr < (self._record_plpr(incumbent) or float("inf")):
                pool[code_hash] = record
        self.hall_of_fame = sorted(
            pool.values(), key=lambda r: self._record_plpr(r) or float("inf")
        )[: max(0, int(self.hall_of_fame_limit))]

    def best_slow_per_logical_per_round(self) -> float | None:
        """Best measured LER among non-degraded archive elites.

        Degraded (quarantined) elites are excluded: their measurement exists
        for audit, but reporting one as the run's headline LER would be
        misleading."""

        best: float | None = None
        for record in self.records_by_cell.values():
            if self._record_degraded(record):
                continue
            value = (record.slow_evidence or {}).get("per_logical_per_round")
            if value is not None and (best is None or float(value) < best):
                best = float(value)
        return best

    def counts(self) -> dict[str, int]:
        sandbox_errors = sum(
            1
            for attempt in self.attempts
            if attempt.get("error") and attempt.get("family") is None
        )
        return {
            "attempts": len(self.attempts),
            "accepted": sum(1 for attempt in self.attempts if attempt.get("accepted")),
            "archive_cells": len(self.records_by_cell),
            "sandbox_errors": sandbox_errors,
            "structural_rejects": sum(
                1
                for attempt in self.attempts
                if attempt.get("family") is not None
                and not attempt.get("valid_structural")
                and not attempt.get("duplicate_of")
            ),
            "fast_rejects": sum(
                1
                for attempt in self.attempts
                if attempt.get("family") is not None
                and not attempt.get("valid_fast")
                and not attempt.get("duplicate_of")
            ),
            "duplicates": sum(1 for attempt in self.attempts if attempt.get("duplicate_of")),
            "degraded_elites": sum(
                1 for record in self.records_by_cell.values() if self._record_degraded(record)
            ),
            # Audit-only count of records carrying any degraded_* flag (M2).
            "flagged_degraded_elites": sum(
                1
                for record in self.records_by_cell.values()
                if ranking.is_flagged_degraded(record.slow_evidence)
            ),
        }

    def parent_records(
        self,
        *,
        limit: int = 12,
        per_family: int = 4,
        include_source: bool = True,
        temperature: float = 0.0,
        rng: "random.Random | None" = None,
    ) -> list[dict[str, Any]]:
        """Return a compact quality-diverse parent pool for future proposals.

        Hard-quarantined records (schedule_catastrophic, v3.6 M2) are kept
        out of the pool here; they stay in the archive for audit and labels.
        Records that merely carry degraded_* audit flags remain parents.

        `temperature` softens the elite tier from an argmax to a rank-Boltzmann
        draw (0.0 = deterministic, the historical behaviour). The campaign
        policy sets it per turn, so how hard the search exploits its current
        best is the agent's call rather than a constant baked in here."""

        ranked = [
            (cell, record)
            for cell, record in sorted(
                self.records_by_cell.items(),
                key=lambda item: self._record_sort_key(item[1]),
                reverse=True,
            )
            if not self._record_degraded(record)
        ]
        ranked = _boltzmann_order(ranked, temperature=float(temperature), rng=rng)
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        # Reserve a few slots for the LER hall-of-fame (lowest measured plpr, even
        # if Q-low or already evicted from its niche): they EXPLOIT proven good
        # decoders so the search keeps building on them; the Q-diverse selection
        # below EXPLORES. plpr is the goal metric, so the hall-of-fame ranks by it.
        hof = sorted(
            (r for r in self.hall_of_fame if not self._record_degraded(r)),
            key=lambda r: self._record_plpr(r) or float("inf"),
        )
        for rank, record in enumerate(hof[: min(3, limit)], start=1):
            _try_add_parent(
                selected,
                used,
                record.archive_cell,
                record,
                reason="hall_of_fame",
                rank=rank,
                limit=limit,
                include_source=include_source,
            )

        for rank, (cell, record) in enumerate(ranked[: min(3, limit)], start=1):
            _try_add_parent(
                selected,
                used,
                cell,
                record,
                reason="global_elite",
                rank=rank,
                limit=limit,
                include_source=include_source,
            )

        by_family: dict[str, list[tuple[int, str, ArchiveRecord]]] = defaultdict(list)
        for rank, (cell, record) in enumerate(ranked, start=1):
            family = record.fast.family if record.fast and record.fast.family else "unknown"
            by_family[str(family)].append((rank, cell, record))

        # Family slots are interleaved rather than appended. The prompt renders
        # only a PREFIX of this pool, and with the diverse tier appended last a
        # 6-parent window showed hall-of-fame + global elites and nothing else —
        # every family-diversity slot was truncated away, leaving pure
        # exploitation. Round-robin across families keeps any prefix diverse.
        family_queues: list[list[tuple[int, str, ArchiveRecord]]] = [
            by_family[family][: max(1, int(per_family))] for family in sorted(by_family)
        ]
        depth = max((len(queue) for queue in family_queues), default=0)
        for index in range(depth):
            for queue in family_queues:
                if len(selected) >= limit:
                    break
                if index >= len(queue):
                    continue
                rank, cell, record = queue[index]
                family = record.fast.family if record.fast and record.fast.family else "unknown"
                _try_add_parent(
                    selected,
                    used,
                    cell,
                    record,
                    reason=f"family_elite:{family}",
                    rank=rank,
                    limit=limit,
                    include_source=include_source,
                )

        for rank, (cell, record) in enumerate(ranked, start=1):
            if len(selected) >= limit:
                break
            _try_add_parent(
                selected,
                used,
                cell,
                record,
                reason="quality_fill",
                rank=rank,
                limit=limit,
                include_source=include_source,
                )
        return selected

    def promotion_queue(
        self,
        *,
        limit: int = 3,
        include_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Return fast-surrogate elites selected for later slow verification.

        Records that already carry slow evidence are skipped: re-verifying a
        measured code wastes slow budget, and degraded codes need repair, not
        re-measurement."""

        ranked = sorted(
            self.records_by_cell.items(),
            key=lambda item: self._record_sort_key(item[1]),
            reverse=True,
        )
        selected = []
        used: set[str] = set()
        used_canonical = self._measured_canonical_hashes()
        for rank, (cell, record) in enumerate(ranked, start=1):
            if len(selected) >= max(0, int(limit)):
                break
            fast = record.fast
            structural = record.structural
            if fast is None or structural is None or not fast.valid or not structural.valid:
                continue
            # The floor reads d_search (the 4-seed minimum), never a single
            # seed's value — the two are the same number by construction, and
            # naming d_search here keeps it that way.
            if fast.d_search is None or int(fast.d_search) < ranking.MIN_PROMOTION_DISTANCE:
                continue  # below the promotion floor (fairness protocol)
            if record.slow_evidence.get("per_logical_per_round") is not None:
                continue
            if int(record.slow_evidence.get("slow_failure_count") or 0) >= MAX_SLOW_FAILURES:
                continue
            identity = structural.code_hash or record.program.program_id
            if identity in used:
                continue
            # Twin gate (v3.6 B3): an exact permutation-equivalent twin
            # (pynauty canonical form) of a measured or already-queued code
            # never reaches the slow loop. WL is never consulted.
            canonical = _record_canonical_hash(record)
            if canonical and canonical in used_canonical:
                continue
            used.add(identity)
            if canonical:
                used_canonical.add(canonical)
            selected.append(
                _promotion_record(
                    cell,
                    record,
                    promotion_rank=len(selected) + 1,
                    fast_rank=rank,
                    include_source=include_source,
                    promotion_rule=self.runtime_promotion_rule(),
                    expected_plpr=self.expected_plpr_of(record),
                )
            )
        return selected

    def runtime_promotion_rule(self) -> str:
        """Name of the order the runtime queue was built in (frozen id): the
        v3.7 expected-plpr order once the model has evidence, the historical
        Q_eff order before the first measurement (turn 0)."""

        return (
            "expected_plpr_v1_archive_elite"
            if self.expected_plpr_model() is not None
            else "ver1_q_fast_archive_elite"
        )

    def _measured_canonical_hashes(self) -> set[str]:
        """Canonical hashes of every slow-measured record (twin gate seed
        set, v3.6 B3). Records without a canonical hash contribute nothing —
        the gate then degrades to exact code_hash equality, never to WL."""

        hashes: set[str] = set()
        for record in self.records_by_cell.values():
            if record.structural is None:
                continue
            if record.slow_evidence.get("per_logical_per_round") is None:
                continue
            canonical = _record_canonical_hash(record)
            if canonical:
                hashes.add(canonical)
        return hashes

    def all_archive_queue(
        self,
        *,
        include_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Return every valid, not-yet-measured archive cell elite for slow
        verification."""

        ranked = sorted(
            self.records_by_cell.items(),
            key=lambda item: self._record_sort_key(item[1]),
            reverse=True,
        )
        selected = []
        used: set[str] = set()
        used_canonical = self._measured_canonical_hashes()
        for rank, (cell, record) in enumerate(ranked, start=1):
            fast = record.fast
            structural = record.structural
            if fast is None or structural is None or not fast.valid or not structural.valid:
                continue
            # The floor reads d_search (the 4-seed minimum), never a single
            # seed's value — the two are the same number by construction, and
            # naming d_search here keeps it that way.
            if fast.d_search is None or int(fast.d_search) < ranking.MIN_PROMOTION_DISTANCE:
                continue  # below the promotion floor (fairness protocol)
            if record.slow_evidence.get("per_logical_per_round") is not None:
                continue
            if int(record.slow_evidence.get("slow_failure_count") or 0) >= MAX_SLOW_FAILURES:
                continue
            identity = structural.code_hash or record.program.program_id
            if identity in used:
                continue
            # Twin gate (v3.6 B3): exact canonical-form twins of measured or
            # already-queued codes stay out of the slow loop; WL never gates.
            canonical = _record_canonical_hash(record)
            if canonical and canonical in used_canonical:
                continue
            used.add(identity)
            if canonical:
                used_canonical.add(canonical)
            selected.append(
                _promotion_record(
                    cell,
                    record,
                    promotion_rank=len(selected) + 1,
                    fast_rank=rank,
                    include_source=include_source,
                    reason="archive_cell_elite",
                    promotion_rule="all_archive_cell_elites",
                    expected_plpr=self.expected_plpr_of(record),
                )
            )
        return selected

    def weight_census(self) -> list[dict[str, Any]]:
        """Per check-weight class of this cycle: how many candidates were
        constructed, how many were structurally eligible, and how many were
        held below the d_search promotion floor.

        v3.7.1 telemetry for the prompt's weight table. No selection path reads
        it: the census reports what was tried, it never decides anything."""

        census: dict[Any, dict[str, int]] = {}
        for attempt in self.attempts:
            row = census.setdefault(
                attempt.get("max_check_weight"),
                {
                    "constructed": 0,
                    "structural_eligible": 0,
                    "below_min_promotion_distance": 0,
                },
            )
            row["constructed"] += 1
            if attempt.get("structural_eligible"):
                row["structural_eligible"] += 1
            if attempt.get("below_min_promotion_distance"):
                row["below_min_promotion_distance"] += 1
        return [
            {"max_check_weight": weight, **counts}
            for weight, counts in sorted(
                census.items(), key=lambda item: (item[0] is None, item[0])
            )
        ]

    def failure_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for attempt in self.attempts:
            if attempt.get("duplicate_of"):
                continue
            reasons = attempt.get("failure_reasons") or []
            if not reasons and attempt.get("archive_status") not in {"accepted"}:
                reasons = [str(attempt.get("archive_status") or "not_accepted")]
            for reason in reasons:
                counts[str(reason)] += 1
        return dict(counts)

    def failure_signatures(self, *, limit: int = 12) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for attempt in self.attempts:
            if attempt.get("duplicate_of"):
                continue
            if attempt.get("archive_status") == "accepted" and not attempt.get("failure_reasons"):
                continue
            reasons = [str(reason) for reason in attempt.get("failure_reasons") or []]
            if not reasons:
                reasons = [str(attempt.get("archive_status") or "not_accepted")]
            signature = "|".join(sorted(set(reasons)))
            item = grouped.setdefault(
                signature,
                {
                    "signature": signature,
                    "count": 0,
                    "families": Counter(),
                    "statuses": Counter(),
                    "examples": [],
                },
            )
            item["count"] += 1
            item["families"][str(attempt.get("family") or "unknown")] += 1
            item["statuses"][str(attempt.get("archive_status") or "unknown")] += 1
            if len(item["examples"]) < 3:
                item["examples"].append(_compact_attempt(attempt))

        ranked = sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item["signature"])))
        return [
            {
                "signature": item["signature"],
                "count": int(item["count"]),
                "families": dict(item["families"]),
                "statuses": dict(item["statuses"]),
                "examples": item["examples"],
            }
            for item in ranked[: max(0, int(limit))]
        ]

    def duplicate_signatures(self, *, limit: int = 12) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for attempt in self.attempts:
            duplicate_of = attempt.get("duplicate_of")
            if not duplicate_of:
                continue
            code_hash = str(attempt.get("code_hash") or "unknown")
            signature = f"code_hash:{code_hash}"
            item = grouped.setdefault(
                signature,
                {
                    "signature": signature,
                    "code_hash": code_hash,
                    "duplicate_of": duplicate_of,
                    "count": 0,
                    "families": Counter(),
                    "examples": [],
                },
            )
            item["count"] += 1
            item["families"][str(attempt.get("family") or "unknown")] += 1
            if len(item["examples"]) < 3:
                item["examples"].append(_compact_attempt(attempt))

        ranked = sorted(grouped.values(), key=lambda item: (-int(item["count"]), str(item["signature"])))
        return [
            {
                "signature": item["signature"],
                "code_hash": item["code_hash"],
                "duplicate_of": item["duplicate_of"],
                "count": int(item["count"]),
                "families": dict(item["families"]),
                "examples": item["examples"],
            }
            for item in ranked[: max(0, int(limit))]
        ]

    def guidance(self) -> list[str]:
        failures = self.failure_counts()
        # Only cross-program duplicates count: a deterministic program's own
        # seeds 1..K-1 deduplicating against its seed 0 is our multi-seed
        # evaluation mechanism, not the model resubmitting a construction.
        duplicates = sum(
            1
            for attempt in self.attempts
            if attempt.get("duplicate_of")
            and _base_program_id(str(attempt.get("duplicate_of")))
            != _base_program_id(str(attempt.get("program_id") or ""))
        )
        guidance: list[str] = []
        if any("configured minimum" in reason or "configured maximum" in reason for reason in failures):
            guidance.append("Keep N_data within the size bounds stated in the rejection messages.")
        if failures.get("n_budget_violation"):
            guidance.append(
                "Compute N_data from the size formula before returning; proposals over the N budget are rejected without evaluation."
            )
        if failures.get("invalid_css_commutation"):
            guidance.append("Use constructions that enforce exact GF(2) CSS commutation.")
        if failures.get("k_nonpositive"):
            guidance.append("Keep K positive: too many independent checks leave no logical qubits.")
        if failures.get("disconnected_tanner_graph"):
            guidance.append("Keep the Tanner graph connected; isolated checks or qubits make the code invalid.")
        if duplicates:
            guidance.append(
                "Reordering rows or terms produces the same code and is rejected as a duplicate; "
                "change exponents, terms, or structure enough to produce a genuinely different code."
            )
        if not guidance:
            guidance.append("Prior attempts are valid so far; explore variations that keep N_data <= N_budget and K positive.")
        return guidance

    def feedback_payload(
        self,
        *,
        parent_limit: int = 12,
        recent_limit: int = 12,
    ) -> dict[str, Any]:
        duplicates = [
            {
                "program_id": attempt.get("program_id"),
                "duplicate_of": attempt.get("duplicate_of"),
                "code_hash": attempt.get("code_hash"),
            }
            for attempt in self.attempts
            if attempt.get("duplicate_of")
        ]
        # Parent order is the machine's, not the runtime ranking's: 8
        # structural + 4 runtime interleaved 2:1 (v3.2 §4.5). In a campaign the
        # graph overrides this with the CROSS-turn pools; this is the same rule
        # applied to what a standalone funsearch run can see.
        structural_parents = self.structural_parent_specs(limit=self.structural_parent_quota)
        runtime_parents = self.parent_records(
            limit=max(0, int(parent_limit) - len(structural_parents)) or int(parent_limit),
            include_source=True,
        )
        next_parent_specs = interleave_parent_specs(
            structural_parents,
            runtime_parents,
            structural_quota=self.structural_parent_quota,
            runtime_quota=self.runtime_parent_quota,
            cap=int(parent_limit),
            diversity_quota=int(self.diversity_parent_quota),
        )
        return {
            "schema": "archive_memory_feedback",
            "summary": self.counts(),
            "archive_summary": self.archive_summary(),
            "structural_proxy_elites": list(self.structural_proxy_elites),
            "next_parent_specs": next_parent_specs,
            "recent_attempts": self.recent_attempts(limit=recent_limit),
            "failure_counts": self.failure_counts(),
            "failure_signatures": self.failure_signatures(),
            "duplicate_signatures": self.duplicate_signatures(),
            "duplicates": duplicates[-12:],
            "proposal_guidance": self.guidance(),
        }

    def prompt_context_payload(
        self,
        *,
        n_budget: int,
        hardware: str,
        parent_limit: int = 12,
        recent_limit: int = 12,
    ) -> dict[str, Any]:
        feedback = self.feedback_payload(parent_limit=parent_limit, recent_limit=recent_limit)
        return {
            "schema": "prompt_context",
            "n_budget": int(n_budget),
            "hardware": str(hardware),
            "history_policy": {
                "proposal_agent_sees": [
                    "archive_summary",
                    "next_parent_specs",
                    "recent_attempts",
                    "failure_signatures",
                    "duplicate_signatures",
                    "proposal_guidance",
                ],
                "machine_only": [
                    # Retention state the generator is never shown: the
                    # selection pressure is the slot/parent/final quota, not
                    # prompt text.
                    "structural_proxy_elites",
                ],
                "full_history_artifacts": [
                    "attempts.jsonl",
                    "archive.json",
                    "feedback.json",
                    "prompt_context.json",
                    "next_parent_specs.json",
                ],
            },
            "archive_summary": feedback["archive_summary"],
            "next_parent_specs": feedback["next_parent_specs"],
            "recent_attempts": feedback["recent_attempts"],
            "structural_proxy_elites": feedback["structural_proxy_elites"],
            "campaign_feedback": feedback,
        }


def interleave_parent_specs(
    structural: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    *,
    structural_quota: int = 6,
    runtime_quota: int = 6,
    cap: int = 16,
    diversity: list[dict[str, Any]] | None = None,
    diversity_quota: int = 4,
) -> list[dict[str, Any]]:
    """Merge the parent pools in the frozen S,R,S,R,... order, then the
    diversity picks (v3.7 §4.5; was S,S,R with 8 + 4).

    The generator renders the whole window (16), so the alternation keeps
    structural and runtime parents visible in equal measure at every prefix.
    When one pool is short the other backfills, and hash duplicates collapse
    into a single entry carrying both selection channels. `diversity` rows
    (regions of family / shape / weight not represented by the first two
    pools) are appended last, up to `diversity_quota`, never displacing a
    structural or runtime parent."""

    def _key(parent: dict[str, Any]) -> str:
        return str(parent.get("code_hash") or parent.get("program_id") or "")

    structural_queue = [dict(row) for row in structural if _key(row)]
    runtime_queue = [dict(row) for row in runtime if _key(row)]
    merged: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    def _push(parent: dict[str, Any], channel: str) -> bool:
        key = _key(parent)
        held = by_key.get(key)
        if held is not None:
            channels = list(held.get("selection_channels") or [])
            if channel not in channels:
                channels.append(channel)
            held["selection_channels"] = channels
            return False
        parent["selection_channels"] = [channel]
        by_key[key] = parent
        merged.append(parent)
        return True

    structural_left = max(0, int(structural_quota))
    runtime_left = max(0, int(runtime_quota))
    diversity_left = max(0, int(diversity_quota))
    cap = max(0, int(cap))
    main_cap = max(0, cap - min(diversity_left, len([row for row in (diversity or []) if _key(row)])))
    # Pattern S,R repeated; a pool that runs dry yields its turns to the
    # other one so the window is still filled.
    pattern_index = 0
    while len(merged) < main_cap and (structural_queue or runtime_queue):
        want_structural = (pattern_index % 2) == 0
        pattern_index += 1
        pools = (
            [("structural", structural_queue), ("runtime", runtime_queue)]
            if want_structural
            else [("runtime", runtime_queue), ("structural", structural_queue)]
        )
        for channel, queue in pools:
            budget = structural_left if channel == "structural" else runtime_left
            if not queue or budget <= 0:
                continue
            parent = queue.pop(0)
            if _push(parent, channel):
                if channel == "structural":
                    structural_left -= 1
                else:
                    runtime_left -= 1
            break
        else:
            # Both quotas exhausted while candidates remain: let whichever
            # pool still has rows fill the cap rather than returning short.
            if structural_queue:
                structural_left = cap
            elif runtime_queue:
                runtime_left = cap
            else:
                break
    for row in [dict(row) for row in (diversity or []) if _key(row)]:
        if len(merged) >= cap or diversity_left <= 0:
            break
        if _push(row, "diversity"):
            diversity_left -= 1
    # Backfill from whatever is left so a short diversity pool never shrinks
    # the window below the cap.
    for channel, queue in (("structural", structural_queue), ("runtime", runtime_queue)):
        while queue and len(merged) < cap:
            _push(queue.pop(0), channel)
    return merged[:cap]


def _base_program_id(program_id: str) -> str:
    """Strip the multi-rng-seed suffix: g0009_p0005_s04 -> g0009_p0005."""

    base, sep, tail = program_id.rpartition("_s")
    if sep and base and tail.isdigit():
        return base
    return program_id


def _check_total(metrics: dict[str, Any]) -> int | None:
    """Measurement ancillas a code costs: one per check row of HX and HZ.

    Total qubits = n_data + this. Rendering it lets the prompt show what a
    candidate actually spends against the budget it is judged at.
    """

    m_x, m_z = metrics.get("m_x"), metrics.get("m_z")
    if m_x is None or m_z is None:
        return None
    return int(m_x) + int(m_z)


def _cell_sort_key(cell: str) -> tuple:
    """Numeric order for `"<overhead>,<distance>"` cell ids."""

    parts = str(cell).split(",")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (10**9, 10**9)


def cell_of_fast(fast: FastResult | None) -> str | None:
    if fast is None or not fast.valid:
        return None
    overhead = fast.metrics.get("overhead")
    distance = fast.metrics.get("d")
    if overhead is None or distance is None:
        return None
    # Cap 40 (was 20): at n~144 every k<=7 code landed in one overhead bin,
    # collapsing the high-overhead niche dimension exactly where large-budget
    # campaigns need diversity.
    return f"{min(int(overhead), 40)},{int(distance)}"



def _boltzmann_order(
    ranked: list[tuple[str, "ArchiveRecord"]],
    *,
    temperature: float,
    rng: "random.Random | None" = None,
) -> list[tuple[str, "ArchiveRecord"]]:
    """Reorder a quality ranking by a rank-Boltzmann draw without replacement.

    Weights come from RANK, not raw score, so the draw behaves the same whether
    the metric is Q_eff or an error rate many orders of magnitude apart. At
    temperature 0 the input order is returned untouched, so the deterministic
    path stays bit-identical.
    """

    if temperature <= 0 or len(ranked) < 2:
        return ranked
    generator = rng or random.Random(0)
    pool = list(ranked)
    weights = [math.exp(-index / temperature) for index in range(len(pool))]
    out: list[tuple[str, "ArchiveRecord"]] = []
    while pool:
        total = sum(weights)
        if total <= 0:
            out.extend(pool)
            break
        draw = generator.random() * total
        cumulative = 0.0
        pick = len(pool) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw <= cumulative:
                pick = index
                break
        out.append(pool.pop(pick))
        weights.pop(pick)
    return out


def _try_add_parent(
    selected: list[dict[str, Any]],
    used: set[str],
    cell: str,
    record: ArchiveRecord,
    *,
    reason: str,
    rank: int,
    limit: int,
    include_source: bool,
) -> None:
    if len(selected) >= int(limit):
        return
    fast = record.fast
    structural = record.structural
    if fast is None or structural is None or not fast.valid or not structural.valid:
        return
    identity = structural.code_hash or record.program.program_id
    if identity in used:
        for parent in selected:
            if parent.get("code_hash") == identity or parent.get("program_id") == identity:
                reasons = list(parent.get("selection_reasons") or [])
                if reason not in reasons:
                    reasons.append(reason)
                parent["selection_reasons"] = reasons
        return
    used.add(identity)
    selected.append(
        _parent_record(
            cell,
            record,
            reason=reason,
            rank=rank,
            include_source=include_source,
        )
    )


def _parent_record(
    cell: str,
    record: ArchiveRecord,
    *,
    reason: str,
    rank: int,
    include_source: bool,
) -> dict[str, Any]:
    fast = record.fast
    structural = record.structural
    constructed = record.constructed
    metrics = fast.metrics if fast is not None else {}
    parent = {
        "program_id": record.program.program_id,
        "family": fast.family if fast else None,
        "n": fast.n_data if fast else None,
        "n_checks": _check_total(metrics),
        "n_budget": fast.n_budget if fast else None,
        "k": fast.k if fast else None,
        "d_search": fast.d_search if fast else None,
        "d_proxy": fast.d_proxy if fast else None,
        "n_total": metrics.get("n_total"),
        "Q": fast.q_score if fast else None,
        "Q_search": metrics.get("Q_search"),
        "overhead": metrics.get("overhead"),
        "selection_channels": ["runtime"],
        "max_check_weight": fast.max_check_weight if fast else None,
        "max_qubit_degree": fast.max_qubit_degree if fast else None,
        "archive_cell": cell,
        "fast_rank": int(rank),
        "selection_reasons": [reason],
        "code_hash": structural.code_hash if structural else None,
        # v3.6 B3: identity fields travel with promotion / slow rows so the
        # cross-turn queue builder can apply the canonical twin gate.
        "canonical_hash": _record_canonical_hash(record) or None,
        "wl_hash": (structural.metrics or {}).get("wl_hash") if structural else None,
        "deformation_mode": structural.deformation if structural else "none",
        "spec": constructed.spec if constructed else None,
        "rng_seed": record.program.rng_seed,
        "source_agent": record.program.source_agent,
    }
    if record.slow_evidence:
        parent["slow_evidence"] = dict(record.slow_evidence)
        parent["slow_per_logical_per_round"] = record.slow_evidence.get("per_logical_per_round")
        parent["d_eff"] = record.slow_evidence.get("d_eff")
        parent["q_eff"] = record.slow_evidence.get("q_eff")
        # M2: `degraded` = hard quarantine; the audit flags stay visible.
        parent["degraded"] = ranking.is_degraded(record.slow_evidence)
        parent["flagged_degraded"] = ranking.is_flagged_degraded(record.slow_evidence)
        parent["degraded_flags"] = ranking.degraded_flags(record.slow_evidence)
    if include_source:
        parent["construct_source"] = record.program.source
    return parent


def _record_canonical_hash(record: ArchiveRecord) -> str:
    """The record's pynauty canonical hash, "" when unavailable."""

    structural = record.structural
    if structural is None:
        return ""
    value = structural.canonical_hash or (structural.metrics or {}).get("canonical_hash")
    return str(value or "")


def _promotion_record(
    cell: str,
    record: ArchiveRecord,
    *,
    promotion_rank: int,
    fast_rank: int,
    include_source: bool,
    reason: str = "fast_surrogate_elite",
    promotion_rule: str = "ver1_q_fast_archive_elite",
    expected_plpr: float | None = None,
) -> dict[str, Any]:
    item = _parent_record(
        cell,
        record,
        reason=reason,
        rank=fast_rank,
        include_source=include_source,
    )
    item.update(
        {
            "promotion_rank": int(promotion_rank),
            "promotion_status": "selected_for_slow_pending",
            "promotion_rule": promotion_rule,
            # v3.7: the measured-evidence prediction the runtime order used
            # (None before the first measurement / for measured records).
            "expected_plpr": expected_plpr,
        }
    )
    return item


def _compact_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "program_id",
        "family",
        "n",
        "k",
        "d_proxy",
        "Q",
        "archive_status",
        "accepted",
        "valid_fast",
        "valid_structural",
        "duplicate_of",
        "failure_reasons",
    ]
    return {key: attempt.get(key) for key in keys}
