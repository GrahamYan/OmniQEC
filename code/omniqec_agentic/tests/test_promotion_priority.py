"""The frozen slow queue (v3.2 §4.3; 2026-09-10 quota 4S+2R = 6; v3.7 2S+4R).

Replaces the pre-v3.2 contract, where the Q-ordered head kept `promote_top`
slots and the plan's named candidates were APPENDED as extra ones. Under v3.2
the queue was exactly four slots (3S+1R); since 2026-09-10 it is six: four
filled mechanically from the structural (k, d_search) reducer, two runtime
slots that named candidates CONSUME (the frozen Q head fills unnamed ones).
"""

from __future__ import annotations

import json
from pathlib import Path

from omniqec_agentic.campaign import _build_slow_queue
from omniqec_agentic.config import StructuralSearchConfig

# The queue builder's MECHANICS (named promotions consume runtime slots, twins,
# measured filtering, backfill) are quota-independent; these tests pin the
# 2026-09-10 4S+2R shape they were written against. The v3.7 formal quota
# (2 structural endpoints + 4 runtime) is asserted separately below.
STRUCTURAL = StructuralSearchConfig(
    structural_slow_slots=4, runtime_slow_slots=2, final_structural=4, final_runtime=2
)
V37 = StructuralSearchConfig()


def test_default_quota_is_two_structural_endpoints_plus_four_runtime(tmp_path) -> None:
    assert (V37.structural_slow_slots, V37.runtime_slow_slots, V37.slow_candidate_cap) == (2, 4, 6)
    structural = [_structural("aa", 20, 5, "high_k_endpoint"), _structural("bb", 4, 12, "high_d_search_endpoint"),
                  _structural("cc", 10, 8, "crowding_representative")]
    fresh = [_runtime(f"g0008_p000{i}", str(i) * 2) for i in range(1, 6)]
    out, composition = _build_slow_queue(
        **_queues(tmp_path, structural=structural, fresh=fresh), priority=[],
        structural_config=V37, out_dir=tmp_path, measured_hashes=None,
    )
    rows = _rows(out)
    assert [r["program_id"] for r in rows] == [
        "struct_aa", "struct_bb", "g0008_p0001", "g0008_p0002", "g0008_p0003", "g0008_p0004",
    ]
    assert (composition["structural_slots"], composition["runtime_slots"], composition["cap"]) == (2, 4, 6)
    assert composition["structural_taken"] == 2 and composition["runtime_head_taken"] == 4


def _write_queue(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _structural(code: str, k: int, d: int, reason: str) -> dict:
    return {
        "program_id": f"struct_{code}", "code_hash": code * 16,
        "k": k, "d_search": d, "structural_slot_reason": reason,
        "spec": {"family": "bb"}, "construct_source": "def construct(rng): return {}",
    }


def _runtime(program_id: str, code: str) -> dict:
    return {"program_id": program_id, "code_hash": code * 16}


def _queues(tmp_path: Path, *, structural: list[dict], fresh: list[dict],
            archive: list[dict] | None = None) -> dict:
    return {
        "structural_queue": _write_queue(tmp_path / "s.jsonl", structural) if structural else None,
        "promotion_queue": _write_queue(tmp_path / "q.jsonl", fresh) if fresh else None,
        "all_archive_queue": _write_queue(tmp_path / "a.jsonl", archive or fresh)
        if (archive or fresh) else None,
    }


def _build(tmp_path: Path, *, structural, fresh, archive=None, priority=(), measured=None):
    return _build_slow_queue(
        **_queues(tmp_path, structural=structural, fresh=fresh, archive=archive),
        priority=list(priority),
        structural_config=STRUCTURAL,
        out_dir=tmp_path,
        measured_hashes=measured,
    )


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_queue_is_four_structural_slots_plus_two_runtime(tmp_path) -> None:
    structural = [
        _structural("aa", 20, 5, "high_k_endpoint"),
        _structural("bb", 4, 12, "high_d_search_endpoint"),
        _structural("cc", 10, 8, "crowding_representative"),
        _structural("dd", 8, 9, "crowding_representative"),
    ]
    fresh = [_runtime("g0008_p0001", "ee"), _runtime("g0008_p0002", "ff"), _runtime("g0008_p0003", "99")]
    out, composition = _build(tmp_path, structural=structural, fresh=fresh)
    rows = _rows(out)
    assert len(rows) == STRUCTURAL.slow_candidate_cap == 6
    assert [r["program_id"] for r in rows[:4]] == ["struct_aa", "struct_bb", "struct_cc", "struct_dd"]
    assert [r["program_id"] for r in rows[4:]] == ["g0008_p0001", "g0008_p0002"]
    assert composition["structural_taken"] == 4
    assert composition["structural_slot_shortfall"] == 0
    assert composition["runtime_head_taken"] == 2 and composition["named_promotions_taken"] == 0
    assert [r["selection_channels"] for r in rows] == [
        ["structural"], ["structural"], ["structural"], ["structural"], ["runtime"], ["runtime"]
    ]


def test_named_promotion_consumes_the_runtime_slot_and_never_adds_one(tmp_path) -> None:
    """The old merge appended agent picks as EXTRA slots, so a plan could grow
    the measured queue beyond the frozen cap."""

    structural = [_structural(c, 10, 8, "crowding_representative") for c in ("aa", "bb", "cc", "dd")]
    fresh = [_runtime("fresh_head", "ee"), _runtime("fresh_second", "ff")]
    archive = fresh + [_runtime("seed_0002", "99")]
    out, composition = _build(
        tmp_path, structural=structural, fresh=fresh, archive=archive,
        priority=["99" * 8],
    )
    rows = _rows(out)
    assert len(rows) == 6
    assert composition["named_promotion_used"] is True
    assert composition["named_promotions_taken"] == 1 and composition["runtime_head_taken"] == 1
    # The name consumed one of the two runtime slots; the head fills the other.
    assert rows[4]["program_id"] == "seed_0002" and rows[5]["program_id"] == "fresh_head"
    assert "fresh_second" not in [r["program_id"] for r in rows]


def test_only_the_first_runtime_slots_many_named_candidates_are_honoured(tmp_path) -> None:
    structural = [_structural(c, 10, 8, "crowding_representative") for c in ("aa", "bb", "cc", "dd")]
    archive = [_runtime("seed_0002", "ee"), _runtime("seed_0003", "ff"), _runtime("seed_0004", "99")]
    out, composition = _build(
        tmp_path, structural=structural, fresh=[_runtime("head", "88")],
        archive=archive + [_runtime("head", "88")],
        priority=["ee" * 8, "ff" * 8, "99" * 8],
    )
    ids = [r["program_id"] for r in _rows(out)]
    assert len(ids) == 6
    assert "seed_0002" in ids and "seed_0003" in ids
    assert "seed_0004" not in ids and "head" not in ids
    assert composition["named_promotions_taken"] == 2


def test_illegal_named_candidate_falls_back_to_the_runtime_head(tmp_path) -> None:
    structural = [_structural(c, 10, 8, "crowding_representative") for c in ("aa", "bb", "cc")]
    fresh = [_runtime("fresh_head", "dd")]
    out, composition = _build(
        tmp_path, structural=structural, fresh=fresh, priority=["9999" * 4]
    )
    assert composition["named_promotion_used"] is False
    assert [r["program_id"] for r in _rows(out)][3] == "fresh_head"
    assert composition["runtime_head_taken"] == 1


def test_measured_hashes_are_filtered_unconditionally(tmp_path) -> None:
    """The campaign-wide measured set used to be applied only when a plan
    carried promotions, so an ordinary candidate could be re-measured on a
    later turn under the same frozen protocol."""

    structural = [
        _structural("aa", 20, 4, "high_k_endpoint"),
        _structural("bb", 4, 12, "high_d_search_endpoint"),
    ]
    fresh = [_runtime("already_measured", "cc"), _runtime("fresh", "dd")]
    out, composition = _build(
        tmp_path, structural=structural, fresh=fresh,
        measured={"aa" * 16, "cc" * 16},   # no plan promotions at all
    )
    ids = [r["program_id"] for r in _rows(out)]
    assert "struct_aa" not in ids and "already_measured" not in ids
    assert ids == ["struct_bb", "fresh"]
    assert composition["structural_slot_shortfall"] == 3


def test_overlapping_slots_backfill_from_the_frozen_order(tmp_path) -> None:
    """high-k and high-d can be the same code; the slot then refills rather
    than leaving a hole."""

    same = _structural("aa", 20, 12, "high_k_endpoint")
    structural = [same, _structural("bb", 10, 8, "crowding_representative")]
    fresh = [_runtime("head", "cc")]
    out, _ = _build(tmp_path, structural=structural + [same], fresh=fresh)
    ids = [r["program_id"] for r in _rows(out)]
    assert ids == ["struct_aa", "struct_bb", "head"]  # no duplicate, no hole


def test_runtime_backfills_when_structural_is_short_and_records_it(tmp_path) -> None:
    out, composition = _build(
        tmp_path,
        structural=[_structural("aa", 20, 4, "high_k_endpoint")],
        fresh=[_runtime("r1", "bb"), _runtime("r2", "cc"), _runtime("r3", "dd")],
    )
    rows = _rows(out)
    assert len(rows) == 4  # cap is 6, but only four candidates exist
    assert composition["structural_taken"] == 1
    assert composition["structural_slot_shortfall"] == 3
    # Two runtime slots from the head, then r3 as backfill for the empty structural slots.
    assert [r["program_id"] for r in rows[1:]] == ["r1", "r2", "r3"]
    assert composition["runtime_head_taken"] == 2


def test_canonical_twins_do_not_take_two_slots(tmp_path) -> None:
    """v3.6 B3: the twin gate is code_hash + pynauty canonical_hash."""

    twin_a = dict(_structural("aa", 20, 4, "high_k_endpoint"), canonical_hash="c1")
    twin_b = dict(_structural("bb", 4, 12, "high_d_search_endpoint"), canonical_hash="c1")
    out, composition = _build(
        tmp_path, structural=[twin_a, twin_b], fresh=[_runtime("head", "cc")]
    )
    ids = [r["program_id"] for r in _rows(out)]
    assert ids == ["struct_aa", "head"]
    assert composition["twin_rejections"] == 1
    assert composition["twin_gate_id"] == "code_hash+pynauty_canonical_v1"


def test_wl_collision_alone_never_gates(tmp_path) -> None:
    """v3.6 B3 (docs/05 §4): twelve distinct C4xD5 1x2 weight-9 codes share
    ONE wl_hash, and BB144 collides with the k=4 BB(12,6). A WL match with
    different code_hash and no canonical evidence is therefore admissible;
    wl_hash stays on the rows as information."""

    twin_a = dict(_structural("aa", 20, 4, "high_k_endpoint"), wl_hash="w1")
    twin_b = dict(_structural("bb", 4, 12, "high_d_search_endpoint"), wl_hash="w1")
    out, composition = _build(
        tmp_path, structural=[twin_a, twin_b], fresh=[_runtime("head", "cc")]
    )
    rows = _rows(out)
    assert [r["program_id"] for r in rows] == ["struct_aa", "struct_bb", "head"]
    assert all(r.get("wl_hash") == "w1" for r in rows[:2])  # recorded, not gating
    assert composition["twin_rejections"] == 0
    # Same WL, DIFFERENT canonical hashes: still two codes.
    twin_a["canonical_hash"], twin_b["canonical_hash"] = "c1", "c2"
    out, _ = _build(tmp_path, structural=[twin_a, twin_b], fresh=[_runtime("head", "cc")])
    assert [r["program_id"] for r in _rows(out)] == ["struct_aa", "struct_bb", "head"]


def test_everything_measured_yields_an_empty_queue_not_a_fallback(tmp_path) -> None:
    """An all-measured turn must produce an EMPTY queue, not silently fall back
    to the unfiltered promotion queue — that fallback re-measured candidates the
    campaign had already measured under the same frozen protocol."""

    out, composition = _build(
        tmp_path, structural=[], fresh=[_runtime("only", "aa")], measured={"aa" * 16}
    )
    assert out is not None and Path(out).exists()
    assert _rows(out) == []
    assert composition["selected"] == []


def test_twin_collision_does_not_reshape_the_queue_to_2s2r(tmp_path) -> None:
    """Audit fix #4: the structural file used to carry exactly 3 rows, so a
    twin among them silently turned the queue into 2S+2R. The emission now
    carries the whole ordered pool and the builder scans until 3 survive
    (v3.6: twins are canonical-hash twins, not WL twins)."""

    structural = [
        dict(_structural("aa", 20, 5, "high_k_endpoint"), canonical_hash="C1"),
        dict(_structural("bb", 4, 12, "high_d_search_endpoint"), canonical_hash="C1"),
        dict(_structural("cc", 10, 8, "crowding_representative"), canonical_hash="C2"),
        dict(_structural("dd", 8, 9, "crowding_representative"), canonical_hash="C3"),
        dict(_structural("ee", 6, 10, "crowding_representative"), canonical_hash="C4"),
    ]
    fresh = [_runtime("g0008_p0001", "ff")]
    out, composition = _build(tmp_path, structural=structural, fresh=fresh)
    rows = _rows(out)
    assert [r["program_id"] for r in rows[:4]] == ["struct_aa", "struct_cc", "struct_dd", "struct_ee"]
    assert composition["structural_taken"] == 4
    assert composition["structural_slot_shortfall"] == 0
    assert rows[4]["program_id"] == "g0008_p0001"


def test_named_promotion_resolves_from_the_cross_turn_registry(tmp_path) -> None:
    """Audit fix #4: a legal plan may name LAST turn's candidate. This turn's
    queues do not carry it, so without the registry the named promotion
    silently degraded to the runtime head every time."""

    structural = [
        _structural("aa", 20, 4, "high_k_endpoint"),
        _structural("bb", 4, 12, "high_d_search_endpoint"),
        _structural("cc", 10, 8, "crowding_representative"),
    ]
    fresh = [_runtime("g0008_p0001", "dd")]
    last_turn_row = {
        "program_id": "old_ff", "code_hash": "ff" * 16, "k": 12, "d_search": 6,
        "spec": {"family": "bb"}, "construct_source": "def construct(rng): return {}",
    }
    out, composition = _build_slow_queue(
        **_queues(tmp_path, structural=structural, fresh=fresh),
        registry_rows=[last_turn_row],
        priority=["ff" * 4],
        structural_config=STRUCTURAL,
        out_dir=tmp_path,
        measured_hashes=None,
    )
    rows = _rows(out)
    assert composition["named_promotion_used"] is True
    assert rows[3]["program_id"] == "old_ff"
    assert "runtime_named" in rows[3]["selection_channels"]


def test_canonical_rediscovery_of_a_measured_code_is_not_requeued(tmp_path) -> None:
    """Audit (round 3, secondary): the measured filter was hash-only, so a
    permutation-equivalent code (same Tanner graph, different code_hash)
    generated on a later turn was measured again. The canonical hash of
    measured codes is recovered by streaming the run-level ledger store
    (v3.6 B3: canonical_hash, never wl_hash)."""

    measured_hash = "ee" * 16
    store = tmp_path / "structural_ledger.jsonl"
    store.write_text(
        json.dumps({
            "code_hash": measured_hash, "canonical_hash": "C_MEASURED",
            "wl_hash": "W_MEASURED", "k": 8, "d_search": 5,
        })
        + "\n",
        encoding="utf-8",
    )
    rediscovery = dict(
        _structural("aa", 8, 5, "high_k_endpoint"),
        canonical_hash="C_MEASURED", wl_hash="W_MEASURED",
    )
    fresh_structural = [
        rediscovery,
        dict(_structural("bb", 4, 12, "high_d_search_endpoint"), canonical_hash="C_OTHER"),
    ]
    out, composition = _build_slow_queue(
        **_queues(tmp_path, structural=fresh_structural, fresh=[_runtime("g0009_p0001", "cc")]),
        registry_path=str(store),
        priority=[],
        structural_config=STRUCTURAL,
        out_dir=tmp_path,
        measured_hashes={measured_hash},
    )
    rows = _rows(out)
    assert all(r["code_hash"] != rediscovery["code_hash"] for r in rows), (
        "a canonical rediscovery of a measured code must not re-enter the queue"
    )
    assert composition["measured_canonical_filtered"] == 1
    assert composition["measured_wl_seen"] == 1  # informational only


def test_wl_only_rediscovery_of_a_measured_code_is_admissible(tmp_path) -> None:
    """The counterpart of the test above: a later-turn code sharing only the
    WL hash of a measured code (different code_hash, canonical hash unknown
    or different) is a NEW code under v3.6 and enters the queue."""

    measured_hash = "ee" * 16
    store = tmp_path / "structural_ledger.jsonl"
    store.write_text(
        json.dumps({
            "code_hash": measured_hash, "canonical_hash": "C_MEASURED",
            "wl_hash": "W_MEASURED", "k": 8, "d_search": 5,
        })
        + "\n",
        encoding="utf-8",
    )
    for canonical in (None, "C_DIFFERENT"):
        look_alike = dict(_structural("aa", 8, 5, "high_k_endpoint"), wl_hash="W_MEASURED")
        if canonical:
            look_alike["canonical_hash"] = canonical
        out, composition = _build_slow_queue(
            **_queues(tmp_path, structural=[look_alike], fresh=[_runtime("g0009_p0001", "cc")]),
            registry_path=str(store),
            priority=[],
            structural_config=STRUCTURAL,
            out_dir=tmp_path,
            measured_hashes={measured_hash},
        )
        rows = _rows(out)
        assert rows and rows[0]["code_hash"] == look_alike["code_hash"]
        assert composition["twin_rejections"] == 0


def test_composition_records_what_the_measured_and_twin_gates_removed(tmp_path) -> None:
    """Fig.3 v2 feedback (2026-10-15): a re-proposed measured code and a relabelled
    twin of one are listed in the composition with the measured code they
    duplicate, so the next turn's generator can be told why nothing was measured."""

    measured_hash = "ee" * 16
    store = tmp_path / "structural_ledger.jsonl"
    store.write_text(
        json.dumps({"code_hash": measured_hash, "canonical_hash": "C_MEASURED", "k": 8, "d_search": 5}) + "\n",
        encoding="utf-8",
    )
    twin = dict(_structural("aa", 8, 5, "high_k_endpoint"), canonical_hash="C_MEASURED", program_id="g0007_p0001")
    again = dict(_structural("bb", 4, 12, "high_d_search_endpoint"), code_hash=measured_hash, program_id="g0007_p0002")
    out, composition = _build_slow_queue(
        **_queues(tmp_path, structural=[twin, again], fresh=[_runtime("g0009_p0001", "cc")]),
        registry_path=str(store),
        priority=[],
        structural_config=STRUCTURAL,
        out_dir=tmp_path,
        measured_hashes={measured_hash},
    )
    rows = _rows(out)
    assert len(rows) == 1 and rows[0]["code_hash"] not in {twin["code_hash"], measured_hash}
    assert composition["candidates_seen"] == 3
    (twin_row,) = composition["twin_rejected_rows"]
    assert twin_row["twin_of_code_hash"] == measured_hash[:12] and twin_row["twin_of"] == "measured"
    assert twin_row["program_id"] == "g0007_p0001" and twin_row["k"] == 8 and twin_row["code_hash"] == twin["code_hash"][:12]
    (measured_row,) = composition["measured_rejected_rows"]
    assert measured_row["code_hash"] == measured_hash[:12] and measured_row["program_id"] == "g0007_p0002"
    assert composition["twin_rejections"] == 1
