"""The d_search protocol and the structural (k, d_search) reducer (v3.2 §3, §1).

Two properties carry the whole repair:

  * `d_search` is the minimum over a FROZEN seed set and both sectors, drawn
    from independent streams, so one lucky permutation cannot push a candidate
    onto the max-d endpoint and one sector's effort cannot move the other's;
  * the reducer keeps the max-k and max-d_search endpoints and reads NOTHING
    from the runtime channel, so a structurally strong code cannot be deleted
    by a bad measurement, a trust screen, or a Q-driven niche competition.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from omniqec_agentic import ranking
from omniqec_agentic.construction import build_matrices
from omniqec_agentic.memory import archive as archive_mod
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.schemas import CandidateProgram
from omniqec_agentic.verification import fast as fast_mod
from omniqec_agentic.verification.gf2 import kernel_basis, logicals
from omniqec_agentic.verification.proxy_distance import (
    DEFAULT_PROXY_SEEDS,
    _proxy_side,
    _sector_rng,
    proxy_distance_search,
    proxy_distances,
)

# n_data = 2 * ell * m. 288 is where the d/sqrt(n) floor of 0.25 rejects
# d_search = 4 (0.236); 144 is where it accepts d_search = 3 (0.250).
SPEC_288 = {"family": "bb", "ell": 12, "m": 12,
            "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]}
# 432 (k=12) is where d_search = 5 clears the 2026-09-10 promotion floor but
# still fails the trust screen (5/sqrt(432) = 0.241).
SPEC_432 = {"family": "bb", "ell": 18, "m": 12,
            "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]}
SPEC_144 = {"family": "bb", "ell": 12, "m": 6,
            "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]]}
SMALL = {"family": "bb", "ell": 3, "m": 5,
         "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]]}

SOURCE = "def construct(rng):\n    return %r\n"


def _stub_proxy(monkeypatch, distance: int) -> None:
    """Pin d_search without paying for the search (the search itself is
    covered by the tests above this helper)."""

    def _search(hx, hz, *, seeds, iters, algorithm_id=None):
        return {
            "schema": "proxy_distance_search", "algorithm_id": "stub",
            "seeds": list(seeds), "iters_per_seed": int(iters),
            "sectors": ["X", "Z"], "per_seed": [],
            "d_x": int(distance), "d_z": int(distance),
            "d_search": int(distance), "available": True,
        }

    monkeypatch.setattr(fast_mod, "proxy_distance_search", _search)


# ---------------------------------------------------------------------------
# d_search protocol
# ---------------------------------------------------------------------------
def test_x_and_z_sectors_draw_from_independent_streams() -> None:
    """Computing the Z sector must not move the X sector's random draw.

    The old implementation shared one generator across both sectors, so the X
    result depended on how many permutations Z had already consumed — changing
    the effort on one side silently changed the other side's value."""

    hx, hz = (np.asarray(m, dtype=np.uint8) for m in build_matrices(SMALL))
    lx_z, lz = logicals(hx, hz)
    standalone_x = _proxy_side(kernel_basis(hz), lz, 60, _sector_rng(0, "X"))
    d_x, _ = proxy_distances(hx, hz, iters=60, seed=0)
    assert d_x == standalone_x
    assert _sector_rng(0, "X").random() != _sector_rng(0, "Z").random()


def test_d_search_is_the_minimum_over_seeds_and_sectors() -> None:
    hx, hz = (np.asarray(m, dtype=np.uint8) for m in build_matrices(SMALL))
    result = proxy_distance_search(hx, hz, seeds=DEFAULT_PROXY_SEEDS, iters=40)
    assert result["seeds"] == list(DEFAULT_PROXY_SEEDS)
    assert len(result["per_seed"]) == len(DEFAULT_PROXY_SEEDS)
    expected = min(
        min(row["d_x"], row["d_z"]) for row in result["per_seed"]
    )
    assert result["d_search"] == expected
    assert result["d_x"] == min(row["d_x"] for row in result["per_seed"])
    assert result["d_z"] == min(row["d_z"] for row in result["per_seed"])
    # Reproducible: the same seeds and effort give the same number.
    again = proxy_distance_search(hx, hz, seeds=DEFAULT_PROXY_SEEDS, iters=40)
    assert again["per_seed"] == result["per_seed"]


def test_proxy_provenance_travels_with_the_result() -> None:
    metrics = fast_mod.evaluate_proxy(SMALL, iters=20)
    search = metrics["proxy_search"]
    assert search["algorithm_id"] and search["iters_per_seed"] == 20
    assert search["seeds"] == list(DEFAULT_PROXY_SEEDS)
    assert metrics["d"] == metrics["d_proxy"] == metrics["d_search"]
    assert metrics["code_hash"]


# ---------------------------------------------------------------------------
# proxy_valid / trust screen / structural eligibility (§4.1)
# ---------------------------------------------------------------------------
def test_trust_screen_failure_keeps_the_full_proxy_record(monkeypatch) -> None:
    """A trust-screen failure is not a rejection.

    The candidate at n_data=288 with d_search=4 fails the d/sqrt(n) screen
    (and, since 2026-09-10, the d_search >= 5 promotion floor); erasing its
    metrics (as the old `error` path did) would delete the audit record
    entirely, so the full proxy record must survive both gates."""

    _stub_proxy(monkeypatch, 4)
    result = fast_mod.evaluate_fast_result("big", SPEC_288, n_budget=600)
    assert (result.n_data, result.k, result.d_search) == (288, 16, 4)
    assert result.proxy_valid is True
    assert result.trust_filter_passed is False
    assert result.runtime_screen_passed is False
    assert result.valid is False  # runtime alias
    assert result.metrics["Q_search"] == pytest.approx(16 * 16)  # kd2_v1: k*d^2, no /n
    assert "error" not in result.metrics


def test_b144_distance_five_passes_both_screens_and_three_only_the_trust_screen(monkeypatch) -> None:
    """d/sqrt(n) accepts d_search = 3 at n = 144 (0.250), but the 2026-09-10
    promotion floor (ranking.MIN_PROMOTION_DISTANCE = 5) is a separate gate:
    the record stays complete, the runtime screen fails on the floor alone."""

    _stub_proxy(monkeypatch, 5)
    result = fast_mod.evaluate_fast_result("mid", SPEC_144, n_budget=600)
    assert (result.n_data, result.d_search) == (144, 5)
    assert result.trust_filter_passed is True
    assert result.runtime_screen_passed is True
    assert result.below_min_promotion_distance is False

    _stub_proxy(monkeypatch, 3)
    result = fast_mod.evaluate_fast_result("mid", SPEC_144, n_budget=600)
    assert (result.n_data, result.d_search) == (144, 3)
    assert result.trust_filter_passed is True
    assert result.below_min_promotion_distance is True
    assert result.runtime_screen_passed is False
    assert "error" not in result.metrics


def test_floor_and_trust_screen_run_after_the_multi_seed_minimum(monkeypatch) -> None:
    """Both gates read the 4-seed minimum, never a single seed's value."""

    seen: dict[str, object] = {}

    def _search(hx, hz, *, seeds, iters, algorithm_id=None):
        seen["seeds"] = list(seeds)
        # Seed 0 alone would clear both gates; the minimum does not.
        return {
            "schema": "proxy_distance_search", "algorithm_id": "stub",
            "seeds": list(seeds), "iters_per_seed": int(iters),
            "sectors": ["X", "Z"],
            "per_seed": [{"seed": 0, "d_x": 8, "d_z": 8},
                         {"seed": 1, "d_x": 2, "d_z": 8}],
            "d_x": 2, "d_z": 8, "d_search": 2, "available": True,
        }

    monkeypatch.setattr(fast_mod, "proxy_distance_search", _search)
    result = fast_mod.evaluate_fast_result("floor", SPEC_144, n_budget=600)
    assert seen["seeds"] == list(DEFAULT_PROXY_SEEDS)
    assert result.d_search == 2
    assert result.below_min_promotion_distance is True
    assert result.runtime_screen_passed is False
    # ...and the record is still complete enough to audit.
    assert result.metrics["n"] == 144 and result.metrics["k"] == 12


def test_distance_two_is_auditable_but_never_structural(monkeypatch) -> None:
    _stub_proxy(monkeypatch, 2)
    memory = ArchiveMemory(turn=0)
    memory.evaluate_and_record(
        CandidateProgram(program_id="d2", source=SOURCE % SPEC_144, n_budget=600),
        n_budget=600,
    )
    attempt = memory.attempts[-1]
    assert attempt["proxy_valid"] is True
    assert attempt["structural_eligible"] is False
    assert attempt["n"] == 144 and attempt["d_search"] == 2  # auditable
    assert memory.structural_proxy_elites == []
    assert memory.structural_slow_queue(slots=3) == []


def test_trust_only_failure_is_structural_but_not_runtime(monkeypatch) -> None:
    _stub_proxy(monkeypatch, 5)
    memory = ArchiveMemory(turn=2)
    memory.evaluate_and_record(
        CandidateProgram(program_id="big", source=SOURCE % SPEC_432, n_budget=900),
        n_budget=900,
    )
    attempt = memory.attempts[-1]
    assert attempt["structural_eligible"] is True
    assert attempt["map_runtime_eligible"] is False
    assert attempt["archive_status"] == "structural_only"
    # It is in the structural pool, with its inheritance payload...
    elites = memory.structural_proxy_elites
    assert [row["k"] for row in elites] == [12]
    assert elites[0]["spec"]["ell"] == 12 and elites[0]["construct_source"]
    assert elites[0]["first_seen_turn"] == 2
    # ...it can take a structural slow slot...
    assert [row["code_hash"] for row in memory.structural_slow_queue(slots=3)] == [
        elites[0]["code_hash"]
    ]
    # ...and it is absent from the runtime queues, which still screen on trust.
    assert memory.promotion_queue(limit=4) == []
    assert memory.all_archive_queue() == []


def test_structural_endpoint_survives_losing_its_map_cell(monkeypatch) -> None:
    """The MAP niche keeps one Q_eff incumbent per cell and the loser's spec
    used to stop existing anywhere; the structural pool is updated BEFORE that
    competition runs precisely so the loser survives."""

    _stub_proxy(monkeypatch, 5)
    monkeypatch.setattr(
        archive_mod, "cell_of_fast", lambda fast: "18,5" if fast is not None else None
    )
    memory = ArchiveMemory(turn=0)
    for program_id, spec in (("big", SPEC_288), ("small", SMALL)):
        memory.evaluate_and_record(
            CandidateProgram(program_id=program_id, source=SOURCE % spec, n_budget=600),
            n_budget=600,
        )
    assert list(memory.records_by_cell) == ["18,5"]  # one incumbent, as before
    ks = sorted(row["k"] for row in memory.structural_proxy_elites)
    assert 16 in ks and len(ks) == 2  # both survive structurally


# ---------------------------------------------------------------------------
# the reducer itself (§1.3)
# ---------------------------------------------------------------------------
def _row(code: str, k: int, d: int, **extra) -> dict:
    row = {"code_hash": code * 8, "k": k, "d_search": d,
           "n_total": 288, "Q_search": float(k * d * d)}
    row.update(extra)
    return row


def test_both_endpoints_survive_cap_truncation() -> None:
    rows = [
        _row("aa", 20, 5),    # max-k endpoint
        _row("bb", 4, 12),    # max-d endpoint
        _row("cc", 10, 8),
        _row("dd", 12, 6),
        _row("ee", 8, 9),
    ]
    kept = ranking.reduce_structural_proxy_elites(rows, limit=3)
    assert len(kept) == 3
    assert {(row["k"], row["d_search"]) for row in kept} >= {(20, 5), (4, 12)}


def test_a_high_q_compromise_cannot_delete_both_endpoints() -> None:
    """Q = k*d^2 prefers the middle: [[.,10,8]] scores 640 against the
    endpoints' 500 and 576, and under a single-score reducer it plus one
    neighbour would fill a cap of 2."""

    rows = [_row("aa", 20, 5), _row("bb", 4, 12), _row("cc", 10, 8)]
    kept = ranking.reduce_structural_proxy_elites(rows, limit=2)
    assert {(row["k"], row["d_search"]) for row in kept} == {(20, 5), (4, 12)}


def test_reducer_ignores_every_runtime_quantity() -> None:
    rows = [_row("aa", 20, 5), _row("bb", 4, 12), _row("cc", 10, 8), _row("dd", 6, 10)]
    baseline = ranking.reduce_structural_proxy_elites(rows, limit=3)
    polluted = [
        dict(row,
             per_logical_per_round=1e-2 if row["k"] == 20 else 1e-9,
             d_eff=3, d_cap=12, degraded_ext=True, degraded_isd=True,
             q_eff=99.0 if row["k"] == 10 else 0.0)
        for row in rows
    ]
    assert ranking.reduce_structural_proxy_elites(polluted, limit=3) == [
        dict(row, per_logical_per_round=1e-2 if row["k"] == 20 else 1e-9,
             d_eff=3, d_cap=12, degraded_ext=True, degraded_isd=True,
             q_eff=99.0 if row["k"] == 10 else 0.0)
        for row in baseline
    ]


def test_ties_break_only_on_identical_objectives() -> None:
    """Smaller n_total wins ONLY between codes with the same (k, d_search);
    making it a general preference would rebuild the small-code pressure."""

    twins = [
        _row("aa", 10, 8, n_total=288),
        _row("bb", 10, 8, n_total=200),
        _row("cc", 20, 5, n_total=576),  # bigger, but an endpoint
    ]
    kept = ranking.reduce_structural_proxy_elites(twins, limit=3)
    assert any(row["k"] == 20 for row in kept)
    ten_eight = [row for row in kept if (row["k"], row["d_search"]) == (10, 8)]
    assert [row["n_total"] for row in ten_eight][0] == 200


def test_dominated_and_illegal_rows_never_enter() -> None:
    rows = [
        _row("aa", 10, 8),
        _row("bb", 8, 8),      # dominated by aa
        _row("cc", 2, 2),      # below the fault-tolerance floor
        {"code_hash": "dd" * 8, "k": 5},                      # no d_search
        _row("ee", 6, 6, d_proxy=9),                          # alias conflict
    ]
    kept = ranking.reduce_structural_proxy_elites(rows, limit=16)
    codes = {row["code_hash"][:2] for row in kept}
    assert codes == {"aa", "bb"}  # bb is rank 2, still legal
    assert all(row["d_search"] >= ranking.MIN_PROMOTION_DISTANCE for row in kept)


def test_output_is_json_safe_and_deterministic() -> None:
    rows = [_row("aa", 20, 5), _row("bb", 4, 12), _row("cc", 10, 8)]
    kept = ranking.reduce_structural_proxy_elites(rows, limit=16)
    text = json.dumps(kept)
    assert "Infinity" not in text and "NaN" not in text
    assert all(isinstance(row["crowding_distance"], float) for row in kept)
    assert kept == ranking.reduce_structural_proxy_elites(list(reversed(rows)), limit=16)


def test_reducer_caps_realizations_per_coordinate_at_two() -> None:
    """v3.6 B4 (docs/06 §3): a third realization of one (k, d_search) never
    takes an elite slot, so endpoint floods cannot evict interior points."""

    rows = [_row(f"a{i}", 20, 5, n_total=100 + i) for i in range(5)] + [_row("bb", 4, 12)]
    kept = ranking.reduce_structural_proxy_elites(rows, limit=16)
    twenty_four = [row for row in kept if (row["k"], row["d_search"]) == (20, 5)]
    assert len(twenty_four) == 2
    assert sorted(row["n_total"] for row in twenty_four) == [100, 101]  # §1.2 order
    assert all(row["reducer_id"] == ranking.STRUCTURAL_REDUCER_ID for row in kept)


def test_slot_order_is_high_k_then_high_d_then_crowding() -> None:
    rows = ranking.reduce_structural_proxy_elites(
        [_row("aa", 20, 5), _row("bb", 4, 12), _row("cc", 10, 8)], limit=16
    )
    slots = ranking.select_structural_slots(rows, slots=3)
    assert [row["structural_slot_reason"] for row in slots] == [
        "high_k_endpoint", "high_d_search_endpoint", "crowding_representative"
    ]
    assert [row["k"] for row in slots] == [20, 4, 10]
