"""v3.6 retention fixes (docs/06 §3 rows B3, B4, M2, B5-graph-side, minor
rows; §4.1), each with the audit scenario from docs/05 §4 that motivated it.

  B3  near-duplicate gate = code_hash + pynauty canonical_hash (never 1-WL);
  B4  reducer caps realizations per (k, d_search); slot 3 skips selected
      coordinates; parents iterate round-robin over unique coordinates;
  M2  `degraded_ext` / `degraded_isd` are audit flags, schedule_catastrophic
      is the only hard quarantine;
  B5  decode_infeasible stays out of search queues, retries once at FINAL;
  minor: BB144 anchor for every budget >= 288; ledger scope reads n_total;
      strategy brief carries one guarded coverage fact line.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omniqec_agentic import ranking
from omniqec_agentic.campaign import _build_slow_queue
from omniqec_agentic.config import StructuralSearchConfig
from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.evaluation.baseline import (
    BASELINE_BB144_PROGRAM_ID,
    BASELINE_BB_PROGRAM_ID,
    select_resident_baseline,
)
from omniqec_agentic.evaluation.evidence import distance_evidence_from_results
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.memory import archive as archive_mod
from omniqec_agentic.memory.fingerprints import canonical_code_hash, is_twin_of, tanner_wl_hash
from omniqec_agentic.memory.ledger import Scope
from omniqec_agentic.schemas import (
    ArchiveRecord,
    CandidateProgram,
    FastResult,
    StructuralResult,
)
from omniqec_agentic.verification import fast as fast_mod
from omniqec_agentic.verification.structural import structural_check_spec
from omniqec_agentic.workflow.graph import (
    _final_entries,
    _load_decode_infeasible_hashes,
    _machine_parent_specs,
    _store_decode_infeasible_hashes,
    _strategy_brief,
)

STRUCTURAL = StructuralSearchConfig()
pynauty = pytest.importorskip("pynauty")

# Two DISTINCT [[200, 40]] C4xD5 1x2/1x2 weight-9 lifted-product codes (both
# cells of every base matrix have weight 3, so every check has weight 9 and
# the total is 40 * 9 = 360 qubits). They differ in the A cells, share one
# 1-WL hash (docs/05 §4 B3: twelve such codes collapsed onto one wl_hash) and
# have different pynauty canonical forms.
C4XD5_TWIN_A = {
    "family": "lp_nonabelian", "group": "C4xD5",
    "A": [[[[1, 2, 0], [0, 1, 0], [1, 0, 0]], [[0, 0, 1], [3, 1, 1], [3, 2, 0]]]],
    "B": [[[[2, 2, 1], [1, 2, 1], [0, 1, 1]], [[3, 2, 0], [1, 4, 0], [2, 2, 1]]]],
}
C4XD5_TWIN_B = {
    "family": "lp_nonabelian", "group": "C4xD5",
    "A": [[[[1, 3, 0], [0, 3, 0], [0, 1, 1]], [[0, 2, 0], [0, 3, 0], [1, 3, 0]]]],
    "B": [[[[3, 0, 1], [0, 4, 1], [2, 3, 0]], [[1, 0, 0], [0, 3, 0], [2, 0, 1]]]],
}
SOURCE = "def construct(rng):\n    return %r\n"


def _stub_proxy(monkeypatch, distance: int) -> None:
    def _search(hx, hz, *, seeds, iters, algorithm_id=None):
        return {
            "schema": "proxy_distance_search", "algorithm_id": "stub",
            "seeds": list(seeds), "iters_per_seed": int(iters),
            "sectors": ["X", "Z"], "per_seed": [],
            "d_x": int(distance), "d_z": int(distance),
            "d_search": int(distance), "available": True,
        }

    monkeypatch.setattr(fast_mod, "proxy_distance_search", _search)


def _matrices(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    compiled = build_compiled_code(spec)
    return (
        np.asarray(compiled.hx, dtype=np.uint8),
        np.asarray(compiled.hz, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# B3: the twin gate
# ---------------------------------------------------------------------------
def test_canonical_hash_separates_wl_twins_and_equals_on_true_duplicates() -> None:
    hx_a, hz_a = _matrices(C4XD5_TWIN_A)
    hx_b, hz_b = _matrices(C4XD5_TWIN_B)
    assert hx_a.shape == hx_b.shape == (80, 200)
    assert int(hx_a.sum(1).max()) == int(hz_a.sum(1).max()) == 9
    assert tanner_wl_hash(hx_a, hz_a) == tanner_wl_hash(hx_b, hz_b)  # WL cannot tell
    canonical_a = canonical_code_hash(hx_a, hz_a)
    canonical_b = canonical_code_hash(hx_b, hz_b)
    assert canonical_a and canonical_b and canonical_a != canonical_b
    # A true duplicate (same matrices, permuted qubits) has the same identity.
    perm = np.random.default_rng(1).permutation(hx_a.shape[1])
    assert canonical_code_hash(hx_a[:, perm], hz_a[:, perm]) == canonical_a
    assert canonical_code_hash(hx_a, hz_a) == canonical_a  # deterministic


def test_canonical_hash_forked_child_matches_in_process(monkeypatch) -> None:
    from omniqec_agentic.memory import fingerprints

    hx_a, hz_a = _matrices(C4XD5_TWIN_A)
    forked = canonical_code_hash(hx_a, hz_a)
    monkeypatch.setattr(fingerprints, "CANONICAL_TIMEOUT_S", 0.0)
    assert forked and canonical_code_hash(hx_a, hz_a) == forked


def test_canonical_hash_past_the_deadline_is_none(monkeypatch) -> None:
    import time

    import pynauty

    from omniqec_agentic.memory import fingerprints

    hx_a, hz_a = _matrices(C4XD5_TWIN_A)
    monkeypatch.setattr(pynauty, "certificate", lambda graph: time.sleep(30) or b"x")
    monkeypatch.setattr(fingerprints, "CANONICAL_TIMEOUT_S", 0.5)
    start = time.monotonic()
    assert fingerprints.tier2_canonical_hash(hx_a, hz_a) == (None, "pynauty_timeout")
    assert canonical_code_hash(hx_a, hz_a) is None
    assert time.monotonic() - start < 5.0


def test_canonical_hash_of_the_s3_b260_hang_code_is_bounded(monkeypatch) -> None:
    # The code whose pynauty certificate never returned on NSCC (S3 CNOT b260,
    # turns 17 and 29): the structural check must now finish within the deadline.
    import time

    from omniqec_agentic.memory import fingerprints

    spec = {"family": "bb", "ell": 8, "m": 8,
            "A": [[0, 0], [1, 3], [2, 6]], "B": [[0, 0], [3, 1], [6, 2]]}
    monkeypatch.setattr(fingerprints, "CANONICAL_TIMEOUT_S", 2.0)
    start = time.monotonic()
    result = structural_check_spec("s3_b260_hang", spec, total_budget=260)
    assert result.n_data == 128 and time.monotonic() - start < 15.0


def test_structural_result_persists_canonical_hash_next_to_wl_hash() -> None:
    result = structural_check_spec("twin_a", C4XD5_TWIN_A, total_budget=360)
    assert result.valid and result.k == 40 and result.max_check_weight == 9
    assert result.canonical_hash and result.metrics["canonical_hash"] == result.canonical_hash
    assert result.metrics["wl_hash"]
    assert "canonical_hash" in result.to_dict()


def test_twin_rule_reads_code_hash_and_canonical_never_wl() -> None:
    measured = {"code_hashes": {"h1"}, "canonical_hashes": {"c1"}}
    assert is_twin_of({"code_hash": "h1", "wl_hash": "w9"}, **measured)
    assert is_twin_of({"code_hash": "h2", "canonical_hash": "c1"}, **measured)
    assert not is_twin_of({"code_hash": "h2", "wl_hash": "w1"}, **measured)
    assert not is_twin_of({"code_hash": "h2", "canonical_hash": "c2", "wl_hash": "w1"}, **measured)
    assert not is_twin_of({"code_hash": "h2"}, **measured)  # no canonical: hash only


def test_two_distinct_c4xd5_codes_share_one_queue_and_survive_each_other(
    monkeypatch, tmp_path
) -> None:
    """docs/05 §4 B3 regression: two real C4xD5 1x2 weight-9 codes are both
    admissible in one queue, the second stays admissible after the first is
    measured, and a true duplicate (same matrices) is still filtered."""

    _stub_proxy(monkeypatch, 6)
    # One MAP cell per program so the runtime queues see both codes.
    monkeypatch.setattr(
        archive_mod, "cell_of_fast",
        lambda fast: f"cell_{fast.program_id}" if fast is not None else None,
    )
    memory = ArchiveMemory(turn=0)
    for program_id, spec in (("twin_a", C4XD5_TWIN_A), ("twin_b", C4XD5_TWIN_B)):
        memory.evaluate_and_record(
            CandidateProgram(program_id=program_id, source=SOURCE % spec, n_budget=360),
            n_budget=360,
        )
    hashes = {a["program_id"]: a["code_hash"] for a in memory.attempts}
    assert len(set(hashes.values())) == 2
    # Both in the runtime queues, both in the structural pool and queue.
    assert {r["program_id"] for r in memory.promotion_queue(limit=4)} == {"twin_a", "twin_b"}
    assert {r["program_id"] for r in memory.all_archive_queue()} == {"twin_a", "twin_b"}
    assert all(r["canonical_hash"] for r in memory.promotion_queue(limit=4))
    structural_queue = memory.structural_slow_queue(slots=3)
    assert {r["program_id"] for r in structural_queue} == {"twin_a", "twin_b"}
    assert all(r["canonical_hash"] and r["wl_hash"] for r in memory.structural_proxy_elites)

    # Measure twin_a: twin_b must remain admissible everywhere.
    record_a = memory.records_by_cell["cell_twin_a"]
    memory.apply_slow_evidence([
        {"code_hash": hashes["twin_a"], "per_logical_per_round": 1e-3, "status": "ok"}
    ])
    assert record_a.slow_evidence["per_logical_per_round"] == 1e-3
    assert [r["program_id"] for r in memory.promotion_queue(limit=4)] == ["twin_b"]
    assert [r["program_id"] for r in memory.all_archive_queue()] == ["twin_b"]
    queue_path = tmp_path / "s.jsonl"
    queue_path.write_text(
        "\n".join(json.dumps(r) for r in structural_queue) + "\n", encoding="utf-8"
    )
    out, composition = _build_slow_queue(
        structural_queue=str(queue_path), promotion_queue=None, all_archive_queue=None,
        priority=[], structural_config=STRUCTURAL, out_dir=tmp_path,
        measured_hashes={hashes["twin_a"]},
    )
    rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
    assert [r["program_id"] for r in rows] == ["twin_b"]
    assert composition["twin_rejections"] == 0

    # A true duplicate of twin_a (identical matrices) is still filtered.
    memory.evaluate_and_record(
        CandidateProgram(program_id="twin_a_again", source=SOURCE % C4XD5_TWIN_A, n_budget=360),
        n_budget=360,
    )
    assert memory.attempts[-1]["archive_status"] == "duplicate"
    assert [r["program_id"] for r in memory.promotion_queue(limit=4)] == ["twin_b"]


# ---------------------------------------------------------------------------
# B4: coordinate cap, slot 3, parent round-robin (docs/05 §4 B4 scenario)
# ---------------------------------------------------------------------------
def _row(code: str, k: int, d: int, n_total: int = 360, **extra) -> dict:
    row = {
        "code_hash": code.ljust(32, "0"), "program_id": f"p_{code}", "family": "bb",
        "n": 200, "n_total": n_total, "k": k, "d_search": d, "d_proxy": d,
        "Q_search": float(k * d * d),
        "spec": {"family": "bb", "ell": 12, "m": 6},
        "construct_source": "def construct(rng):\n    return {}\n",
        "rng_seed": 0, "source_agent": "funsearch",
    }
    row.update(extra)
    return row


def _audit_pool() -> list[dict]:
    pool = [_row(f"k{i:02d}", 45, 5) for i in range(20)]
    pool += [_row(f"d{i:02d}", 4, 15) for i in range(20)]
    pool += [_row("mid", 40, 12), _row("low", 30, 8), _row("upp", 36, 10)]
    return pool


def test_reducer_keeps_the_interior_pareto_point_under_an_endpoint_flood() -> None:
    """20x(45,5) + 20x(4,15) + (40,12) + (30,8) + (36,10), limit 16: under
    v1 the 16 elites were endpoint realizations only and (40,12) — dominated
    by nothing — was evicted (docs/05 §4 B4)."""

    elites = ranking.reduce_structural_proxy_elites(_audit_pool(), limit=16)
    coords = [(r["k"], r["d_search"]) for r in elites]
    assert (40, 12) in coords
    assert coords.count((45, 5)) == 2 and coords.count((4, 15)) == 2
    assert (36, 10) in coords and (30, 8) in coords
    assert len(elites) == 7 <= 16
    assert all(r["reducer_id"] == "structural_proxy_pareto_v3_weight" for r in elites)
    assert ranking.STRUCTURAL_REDUCER_ID == STRUCTURAL.reducer_id
    assert ranking.STRUCTURAL_COORDINATE_CAP == STRUCTURAL.coordinate_cap == 2
    # Within a coordinate the §1.2 tie-break still decides which two survive.
    kept_k = sorted(r["code_hash"] for r in elites if r["k"] == 45)
    assert kept_k == sorted(r["code_hash"] for r in _audit_pool() if r["k"] == 45)[:2]
    # The uncapped v1 rule is still reproducible for audits of old runs.
    legacy = ranking.reduce_structural_proxy_elites(_audit_pool(), limit=16, coordinate_cap=0)
    assert (40, 12) not in {(r["k"], r["d_search"]) for r in legacy}


def test_slot_three_is_the_interior_point_not_a_second_endpoint_realization() -> None:
    elites = ranking.reduce_structural_proxy_elites(_audit_pool(), limit=16)
    slots = ranking.select_structural_slots(elites, slots=3)
    assert [(r["k"], r["d_search"]) for r in slots] == [(45, 5), (4, 15), (40, 12)]
    assert [r["structural_slot_reason"] for r in slots] == [
        "high_k_endpoint", "high_d_search_endpoint", "crowding_representative",
    ]
    # Emitting the whole pool: unique coordinates first, repeats last.
    whole = ranking.select_structural_slots(elites, slots=len(elites))
    reasons = [r["structural_slot_reason"] for r in whole]
    first_repeat = reasons.index("crowding_representative_repeat_coordinate")
    assert len({(r["k"], r["d_search"]) for r in whole[:first_repeat]}) == first_repeat == 5
    assert reasons[first_repeat:] == ["crowding_representative_repeat_coordinate"] * 2


def test_repeat_coordinates_backfill_only_after_every_unique_one() -> None:
    pool = [_row(f"a{i}", 10, 8) for i in range(3)] + [_row(f"b{i}", 8, 10) for i in range(3)]
    elites = ranking.reduce_structural_proxy_elites(pool, limit=16)
    assert len(elites) == 4  # 2 per coordinate
    slots = ranking.select_structural_slots(elites, slots=3)
    assert [(r["k"], r["d_search"]) for r in slots][:2] == [(10, 8), (8, 10)]
    assert slots[2]["structural_slot_reason"] == "crowding_representative_repeat_coordinate"


def test_structural_parents_iterate_round_robin_over_unique_coordinates(tmp_path) -> None:
    elites = ranking.reduce_structural_proxy_elites(_audit_pool(), limit=16)
    parents = _machine_parent_specs(
        structural_elites=elites, hof_summary=[], structural_config=STRUCTURAL,
        turn_dir=tmp_path,
    )
    structural = [p for p in parents if "structural" in p["selection_channels"]]
    coords = [(p["k"], p["d_search"]) for p in structural[:6]]
    assert (40, 12) in coords[:5]
    assert len(set(coords[:5])) == 5  # every unique coordinate before any repeat
    # v3.7: six structural parents -- the sixth is a second realization of an endpoint.
    assert coords[5] in {(45, 5), (4, 15)}
    selection = json.loads((tmp_path / "parent_selection.json").read_text())
    assert selection["structural_parent_order"] == "unique_cell_round_robin_v2_weight"
    assert selection["structural_coordinates_selected"] == 5

    # Same rule inside the standalone archive.
    memory = ArchiveMemory(turn=0)
    memory.seed_structural_elites(_audit_pool())
    coords = [(p["k"], p["d_search"]) for p in memory.structural_parent_specs(limit=8)]
    assert (40, 12) in coords[:5] and len(set(coords[:5])) == 5


def test_parent_window_keeps_the_s_r_interleave_with_round_robin(tmp_path) -> None:
    from omniqec_agentic.memory.archive import interleave_parent_specs

    elites = ranking.reduce_structural_proxy_elites(_audit_pool(), limit=16)
    structural = ranking.structural_coordinate_round_robin(elites)[:6]
    runtime = [{"code_hash": f"r{i}" * 16, "program_id": f"r{i}"} for i in range(6)]
    parents = interleave_parent_specs(structural, runtime)
    channels = [p["selection_channels"][0] for p in parents[:6]]
    assert channels == ["structural", "runtime"] * 3  # v3.7: 1:1 alternation
    # (40,12) sits inside the first 10 rendered parents (5 structural cells).
    assert (40, 12) in [(p.get("k"), p.get("d_search")) for p in parents[:10]]


def test_round_robin_helper_is_order_preserving() -> None:
    rows = [_row("a", 10, 8), _row("b", 10, 8), _row("c", 8, 10), {"code_hash": "x"}]
    ordered = ranking.structural_coordinate_round_robin(rows)
    assert [r["code_hash"][:1] for r in ordered] == ["a", "c", "x", "b"]
    assert ranking.structural_coordinate_round_robin([]) == []


# ---------------------------------------------------------------------------
# M2: degraded quarantine
# ---------------------------------------------------------------------------
def _measured_record(code: str, *, cell: str) -> ArchiveRecord:
    fast = FastResult(
        program_id=f"p_{code}", valid=True, status="ok", n_budget=288, n_data=144,
        k=12, d_proxy=12, d_search=12, q_score=1728.0, family="bb",
        code_hash=code * 16, proxy_valid=True, trust_filter_passed=True,
        runtime_screen_passed=True, metrics={"overhead": 12, "d": 12},
    )
    structural = StructuralResult(
        program_id=f"p_{code}", valid=True, status="ok", n_data=144, k=12,
        code_hash=code * 16, canonical_hash=f"canon_{code}",
        metrics={"wl_hash": f"wl_{code}", "canonical_hash": f"canon_{code}"},
    )
    return ArchiveRecord(
        program=CandidateProgram(program_id=f"p_{code}", source="def construct(rng): return {}"),
        structural=structural, fast=fast, archive_cell=cell, parent_eligible=True,
    )


def test_is_degraded_is_schedule_catastrophic_only() -> None:
    flags = {"degraded_ext": True, "degraded_isd": True, "degraded_estimate": True,
             "degraded_estimate_any_decoder": True}
    assert ranking.is_degraded(flags) is False
    assert ranking.is_flagged_degraded(flags) is True
    assert ranking.degraded_flags(flags) == list(ranking.DEGRADED_FLAG_KEYS)
    assert ranking.is_degraded({"schedule_catastrophic": True}) is True
    assert ranking.is_flagged_degraded({"schedule_catastrophic": True}) is True
    assert ranking.is_degraded(None) is False and ranking.is_flagged_degraded({}) is False


def test_measured_degraded_ext_row_sorts_above_unmeasured_and_stays_in_hof() -> None:
    measured = ranking.archive_order_key(
        n=144, k=12, d_proxy=12,
        slow_evidence={"per_logical_per_round": 5e-4, "degraded_ext": True, "d_circ_ext": 9},
    )
    unmeasured = ranking.archive_order_key(n=144, k=12, d_proxy=12, slow_evidence={})
    assert measured > unmeasured  # tier 1 vs tier 0 (descending sort)
    catastrophic = ranking.archive_order_key(
        n=144, k=12, d_proxy=12,
        slow_evidence={"per_logical_per_round": 5e-4, "schedule_catastrophic": True},
    )
    assert catastrophic < unmeasured

    memory = ArchiveMemory(turn=0)
    memory.records_by_cell["c_ext"] = _measured_record("aa", cell="c_ext")
    memory.records_by_cell["c_cat"] = _measured_record("bb", cell="c_cat")
    memory.records_by_cell["c_new"] = _measured_record("cc", cell="c_new")
    memory.apply_slow_evidence([
        {"code_hash": "aa" * 16, "per_logical_per_round": 5e-4, "status": "ok",
         "degraded_ext": True, "d_circ_ext": 9, "degraded_isd": True},
        {"code_hash": "bb" * 16, "per_logical_per_round": 1e-6, "status": "ok",
         "schedule_catastrophic": True},
    ])
    hof = [r.program.program_id for r in memory.hall_of_fame]
    assert hof == ["p_aa"]  # degraded_ext kept; catastrophic excluded
    parents = memory.parent_records(limit=12)
    ids = [p["program_id"] for p in parents]
    assert ids[0] == "p_aa" and "p_bb" not in ids and "p_cc" in ids
    assert parents[0]["degraded"] is False and parents[0]["flagged_degraded"] is True
    assert parents[0]["degraded_flags"] == ["degraded_estimate_any_decoder"] or set(
        parents[0]["degraded_flags"]
    ) == {"degraded_isd", "degraded_ext"}
    assert memory.best_slow_per_logical_per_round() == 5e-4
    counts = memory.counts()
    assert counts["degraded_elites"] == 1 and counts["flagged_degraded_elites"] == 2
    serialized = memory.records_by_cell["c_ext"].to_dict()
    assert serialized["degraded"] is False and serialized["flagged_degraded"] is True
    assert serialized["parent_eligible"] is True
    summary = {row["program_id"]: row for row in memory.archive_summary()}
    assert summary["p_aa"]["degraded"] is False and summary["p_aa"]["flagged_degraded"] is True


def test_evidence_keeps_the_flags_and_adds_the_convergence_marker() -> None:
    base = {"circuit_ext": {"d_circ_ext": 9, "status": "probe_upper_bound"}}
    evidence = distance_evidence_from_results([], d_proxy=12, decoder_order=["x"], exact_distance=base)
    assert evidence["degraded_ext"] is True and evidence["degraded_ext_converged"] is None
    base["circuit_ext"]["converged"] = False
    evidence = distance_evidence_from_results([], d_proxy=12, decoder_order=["x"], exact_distance=base)
    assert evidence["degraded_ext"] is True and evidence["degraded_ext_converged"] is False
    assert ranking.is_degraded(evidence) is False and ranking.is_flagged_degraded(evidence)


# ---------------------------------------------------------------------------
# B5 (graph side): decode_infeasible retries once at FINAL
# ---------------------------------------------------------------------------
def test_decode_infeasible_is_a_finalist_with_retry_flag_but_catastrophic_is_not(tmp_path) -> None:
    infeasible = _row("inf", 40, 12)
    catastrophic = _row("cat", 45, 5)
    state = {
        "run_dir": str(tmp_path),
        "structural_proxy_elites": [catastrophic, infeasible, _row("d", 4, 15), _row("m", 30, 8)],
        "hof_summary": [dict(_row("h", 6, 6), per_logical_per_round=1e-5)],
        "unmeasurable_hashes": [catastrophic["code_hash"], infeasible["code_hash"]],
        "decode_infeasible_hashes": [infeasible["code_hash"]],
    }
    entries, composition = _final_entries(state, STRUCTURAL)
    by_id = {row["program_id"]: row for row in entries}
    assert "p_cat" not in by_id
    assert by_id["p_inf"]["retry_at_final"] is True
    assert all(not row.get("retry_at_final") for row in entries if row["program_id"] != "p_inf")
    assert composition["decode_infeasible_retry_eligible"] == 1
    assert composition["unmeasurable_excluded"] == 1
    assert composition["retry_at_final"] == [infeasible["code_hash"][:12]]
    assert len(entries) == 4

    # Search queues keep excluding it (the graph unions it into measured).
    queue = tmp_path / "s.jsonl"
    queue.write_text(json.dumps(infeasible) + "\n", encoding="utf-8")
    _, search = _build_slow_queue(
        structural_queue=str(queue), promotion_queue=None, all_archive_queue=None,
        priority=[], structural_config=STRUCTURAL, out_dir=tmp_path,
        measured_hashes={infeasible["code_hash"]},
    )
    assert search["selected"] == []


def test_decode_infeasible_hashes_survive_a_state_without_the_key(tmp_path) -> None:
    state = {"run_dir": str(tmp_path)}
    _store_decode_infeasible_hashes(state, {"h1", "h2"})
    payload = json.loads((tmp_path / "decode_infeasible_hashes.json").read_text())
    assert payload["hashes"] == ["h1", "h2"]
    assert _load_decode_infeasible_hashes(state) == {"h1", "h2"}
    assert _load_decode_infeasible_hashes({"run_dir": str(tmp_path), "decode_infeasible_hashes": ["h3"]}) == {"h1", "h2", "h3"}
    assert _load_decode_infeasible_hashes({}) == set()
    # A stateless final stage still retries it.
    infeasible = _row("inf", 40, 12, code_hash="h1")
    entries, _ = _final_entries(
        {"run_dir": str(tmp_path), "structural_proxy_elites": [infeasible],
         "hof_summary": [], "unmeasurable_hashes": ["h1"]},
        STRUCTURAL,
    )
    assert [row.get("retry_at_final") for row in entries] == [True]


# ---------------------------------------------------------------------------
# minor rows
# ---------------------------------------------------------------------------
def test_every_budget_from_288_up_gets_the_bb144_anchor() -> None:
    assert select_resident_baseline(144)[1] == BASELINE_BB_PROGRAM_ID
    assert select_resident_baseline(288)[1] == BASELINE_BB144_PROGRAM_ID
    assert select_resident_baseline(360)[1] == BASELINE_BB144_PROGRAM_ID
    assert select_resident_baseline(320)[1] == BASELINE_BB144_PROGRAM_ID
    assert select_resident_baseline(200) is None and select_resident_baseline(98) is None


def test_ledger_scope_reads_total_qubits() -> None:
    mitten = {"family": "lp_nonabelian", "n": 200, "n_total": 360}
    assert Scope(budget_min=288).contains(mitten) is True
    assert Scope(budget_max=300).contains(mitten) is False
    assert Scope(budget_min=288).contains({"family": "lp_nonabelian", "n": 200}) is False
    assert Scope(budget_min=288).contains({"family": "bb", "n": 144, "n_total": None}) is False


def test_strategy_brief_carries_one_guarded_coverage_line(tmp_path, monkeypatch) -> None:
    from omniqec_agentic.memory import coverage as coverage_mod

    (tmp_path / "structural_ledger.jsonl").write_text(
        json.dumps(_row("x", 40, 12)) + "\n", encoding="utf-8"
    )
    state = {"run_dir": str(tmp_path), "spec": {"qubit_budget": 360}}
    plan = {"family_mix": {"bb": 0.5, "lp_nonabelian": 0.5}}

    seen: dict = {}

    def _facts(rows, n_budget, **kwargs):
        seen["rows"] = list(rows)
        seen["n_budget"] = n_budget
        return ["never proposed at b360: lp_nonabelian/direct_product 1x2|1x2",
                "region bb: 1 proposed, best k=40, best d_search=12"]

    monkeypatch.setattr(coverage_mod, "coverage_fact_lines", _facts, raising=False)
    brief = _strategy_brief(plan, state)
    coverage_lines = [line for line in brief.splitlines() if line.startswith("- coverage facts:")]
    assert len(coverage_lines) == 1
    assert "never proposed" in coverage_lines[0] and "best d_search=12" in coverage_lines[0]
    assert seen["n_budget"] == 360 and seen["rows"][0]["k"] == 40

    # Missing function -> the brief simply omits the line (no exception).
    monkeypatch.delattr(coverage_mod, "coverage_fact_lines", raising=False)
    brief = _strategy_brief(plan, state)
    assert "coverage facts" not in brief and "family attention" in brief

    def _broken(rows, n_budget, **kwargs):
        raise AttributeError("not ready")

    monkeypatch.setattr(coverage_mod, "coverage_fact_lines", _broken, raising=False)
    assert "coverage facts" not in _strategy_brief(plan, state)
