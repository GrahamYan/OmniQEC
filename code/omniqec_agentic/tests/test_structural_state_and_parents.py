"""Cross-turn structural state, the 8S:4R parent quota, and the fixed-T final
audit (v3.2 §4.2, §4.5, §4.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic.config import StructuralSearchConfig
from omniqec_agentic.construction import build_compiled_code, compiled_code_hash
from omniqec_agentic.examples.seed_programs import SEED_BB
from omniqec_agentic.funsearch import _merge_prompt_context_overlay, _parent_construction_lines
from omniqec_agentic.memory.archive import interleave_parent_specs
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.sandbox import run_construct_source
from omniqec_agentic.workflow.graph import (
    _final_entries,
    _inheritance_from_candidate,
    _light_slow_row,
    _machine_parent_specs,
    _protocol_final_rounds,
    _rebuilt_code_hash,
    _runtime_parent_specs,
    final_rng_base_seed,
)

STRUCTURAL = StructuralSearchConfig()


def _bb_identity() -> tuple[dict, str]:
    spec = run_construct_source(SEED_BB, program_id="bb", seed=0).constructed.spec
    compiled = build_compiled_code(spec)
    return spec, compiled_code_hash(compiled.hx, compiled.hz, h_mask=compiled.h_mask)


def _elite(code: str, k: int, d: int, **extra) -> dict:
    row = {
        "code_hash": code * 16, "program_id": f"struct_{code}", "family": "bb",
        "n": 144, "n_total": 288, "k": k, "d_search": d, "Q_search": k * d * d,
        "spec": {"family": "bb", "ell": 12, "m": 6},
        "construct_source": "def construct(rng):\n    return {}\n",
        "rng_seed": 0, "source_agent": "funsearch",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# cross-turn survival (§4.2)
# ---------------------------------------------------------------------------
def test_last_turns_elite_survives_without_being_regenerated() -> None:
    """The archive is rebuilt from seeds every turn, so without this cache a
    candidate that is not re-proposed simply disappears."""

    previous = [_elite("aa", 20, 5), _elite("bb", 4, 12)]
    memory = ArchiveMemory(turn=1)
    memory.seed_structural_elites(previous)
    kept = {row["code_hash"]: row for row in memory.structural_proxy_elites}
    assert set(kept) == {"aa" * 16, "bb" * 16}
    assert all(row["spec"] and row["construct_source"] for row in kept.values())
    # ...and it is immediately eligible for a slow slot again.
    assert len(memory.structural_slow_queue(slots=3)) == 2


def test_rediscovery_keeps_the_earliest_provenance() -> None:
    memory = ArchiveMemory(turn=5)
    memory.seed_structural_elites([_elite("aa", 20, 5, first_seen_turn=1, last_seen_turn=1)])
    memory.reconcile_structural_elites(
        [_elite("aa", 20, 5, first_seen_turn=5, last_seen_turn=5, program_id="later")]
    )
    row = memory.structural_proxy_elites[0]
    assert row["first_seen_turn"] == 1 and row["last_seen_turn"] == 5
    assert row["program_id"] == "struct_aa"  # earliest source kept


def test_identity_conflict_on_one_hash_is_a_protocol_error() -> None:
    memory = ArchiveMemory(turn=0)
    memory.seed_structural_elites([_elite("aa", 20, 5)])
    conflicting = dict(_elite("aa", 20, 5), k=21)  # same hash, different k
    memory.reconcile_structural_elites([conflicting])
    row = memory.structural_proxy_elites[0]
    assert row["k"] == 20  # no silent upgrade
    assert any("identity conflict" in e for e in row["protocol_errors"])


# ---------------------------------------------------------------------------
# runtime hall-of-fame inheritance (§4.5)
# ---------------------------------------------------------------------------
def test_hof_row_carries_the_inheritance_payload(tmp_path) -> None:
    """`hof_summary` used to keep only `spec`, so a HoF row could never be
    rendered as a parent (the renderer needs construct_source too)."""

    spec, code_hash = _bb_identity()
    candidate = tmp_path / "cand.json"
    candidate.write_text(json.dumps({
        "program_id": "g0001_p0001", "spec": spec,
        "promotion_record": {
            "construct_source": SEED_BB, "rng_seed": 0, "source_agent": "funsearch_x",
        },
    }), encoding="utf-8")
    light = _light_slow_row({
        "program_id": "g0001_p0001", "code_hash": code_hash, "family": "bb",
        "n": 30, "k": 4, "d_proxy": 2, "per_logical_per_round": 1e-3,
        "candidate_path": str(candidate),
    })
    assert light["construct_source"] == SEED_BB
    assert light["rng_seed"] == 0 and light["source_agent"] == "funsearch_x"
    assert light["spec"] == spec


def test_runtime_parent_must_rebuild_to_its_recorded_hash(tmp_path) -> None:
    spec, code_hash = _bb_identity()
    good = {"code_hash": code_hash, "program_id": "ok", "construct_source": SEED_BB,
            "rng_seed": 0, "spec": spec, "per_logical_per_round": 1e-3}
    rebuilt, error = _rebuilt_code_hash(good)
    assert rebuilt is not None and not error

    stale = dict(good, code_hash="dead" * 16)
    assert _rebuilt_code_hash(stale)[0] is None

    parents, errors = _runtime_parent_specs([stale, good], limit=4)
    assert [p["program_id"] for p in parents] == ["ok"]
    assert errors and "!=" in errors[0]

    missing = {"code_hash": code_hash, "program_id": "no_source"}
    assert _runtime_parent_specs([missing], limit=4) == ([], [f"{code_hash[:12]}: missing construct_source"])


# ---------------------------------------------------------------------------
# the 12-parent quota (§4.5)
# ---------------------------------------------------------------------------
def test_parents_alternate_structural_and_runtime_then_diversity() -> None:
    """v3.7 window: 6 structural + 6 runtime interleaved S,R, then up to 4
    diversity picks; 16 in total."""

    structural = [_elite(f"s{i}", 10 + i, 6) for i in range(8)]
    runtime = [{"code_hash": f"r{i}" * 16, "program_id": f"r{i}"} for i in range(8)]
    diversity = [{"code_hash": f"d{i}" * 16, "program_id": f"d{i}"} for i in range(6)]
    parents = interleave_parent_specs(structural, runtime, diversity=diversity)
    assert len(parents) == 16
    channels = [p["selection_channels"][0] for p in parents]
    assert channels[:12] == ["structural", "runtime"] * 6
    assert channels[12:] == ["diversity"] * 4
    # Without diversity rows the two main pools fill the window.
    parents = interleave_parent_specs(structural, runtime)
    assert len(parents) == 16 and [p["selection_channels"][0] for p in parents[:6]] == ["structural", "runtime"] * 3


def test_a_code_in_both_pools_keeps_both_channel_labels() -> None:
    shared = _elite("aa", 20, 5)
    parents = interleave_parent_specs([shared], [dict(shared)])
    assert len(parents) == 1
    assert parents[0]["selection_channels"] == ["structural", "runtime"]


def test_short_pool_backfills_to_the_cap() -> None:
    structural = [_elite(f"s{i}", 10 + i, 6) for i in range(16)]
    parents = interleave_parent_specs(structural, [])
    assert len(parents) == 16  # runtime quota yields rather than leaving holes


def test_structural_parent_survives_bad_measurements(tmp_path) -> None:
    """A structurally strong code that measured badly, or carries a hook
    degradation flag, is still a structural parent (v3.2 §4.4)."""

    structural = [_elite("aa", 20, 5)]
    hof = [{"code_hash": "aa" * 16, "program_id": "aa",
            "per_logical_per_round": 0.5, "degraded_ext": True}]
    parents = _machine_parent_specs(
        structural_elites=structural, hof_summary=hof,
        structural_config=STRUCTURAL, turn_dir=tmp_path,
    )
    assert [p["program_id"] for p in parents] == ["struct_aa"]
    assert parents[0]["selection_channels"] == ["structural"]


def test_structural_parents_visit_every_coordinate_before_repeating_one(tmp_path) -> None:
    """v3.6 §4.1: two realizations of the high-k endpoint no longer occupy two
    of the first structural parent slots ahead of a different coordinate."""

    structural = [
        _elite("aa", 20, 5), _elite("ab", 20, 5, n_total=300),
        _elite("bb", 4, 12), _elite("bc", 4, 12, n_total=300),
        _elite("cc", 10, 8),
    ]
    parents = _machine_parent_specs(
        structural_elites=structural, hof_summary=[],
        structural_config=STRUCTURAL, turn_dir=tmp_path,
    )
    coords = [(p["k"], p["d_search"]) for p in parents]
    assert len(set(coords[:3])) == 3
    assert (10, 8) in coords[:3]


def test_machine_parents_override_the_runtime_first_overlay() -> None:
    machine = [_elite("aa", 20, 5), _elite("bb", 4, 12)]
    base = {"next_parent_specs": [{"code_hash": "cc" * 16, "program_id": "runtime_first"}]}
    overlay = {"next_parent_specs": machine, "parent_specs_authoritative": True}
    merged = _merge_prompt_context_overlay(base, overlay, parent_limit=12)
    assert [p["program_id"] for p in merged["next_parent_specs"]] == ["struct_aa", "struct_bb"]
    # Without the flag the historical prefix merge still applies.
    legacy = _merge_prompt_context_overlay(base, {"next_parent_specs": machine}, parent_limit=12)
    assert len(legacy["next_parent_specs"]) == 3


def test_structural_parents_render_without_a_comparative_score() -> None:
    lines = "\n".join(_parent_construction_lines(
        {"next_parent_specs": [
            dict(_elite("aa", 20, 5), selection_channels=["structural"], q_eff=9.9, Q=9.9),
            dict(_elite("bb", 4, 12), selection_channels=["runtime"],
                 slow_per_logical_per_round=1.5e-3),
        ]},
        limit=6,
    ))
    assert "Q_eff" not in lines
    assert "channel=structural" in lines and "channel=runtime" in lines
    assert "d_search=20" not in lines and "k=20 d_search=5" in lines
    # A runtime measurement may appear as a fact.
    assert "per_logical_per_round=0.0015" in lines


# ---------------------------------------------------------------------------
# fixed-T final audit (§4.6)
# ---------------------------------------------------------------------------
def _final_state(structural: list[dict], hof: list[dict]) -> dict:
    return {"structural_proxy_elites": structural, "hof_summary": hof}


def test_final_shortlist_is_two_structural_endpoints_plus_four_by_plpr() -> None:
    """v3.7: the 25k/25k confirmation goes to the two structural endpoints
    and the FOUR best measured codes (was 4 + 2)."""

    structural = [_elite("aa", 20, 5), _elite("bb", 4, 12), _elite("cc", 10, 8),
                  _elite("dd", 8, 9), _elite("ff", 6, 10)]
    hof = [{"code_hash": f"{h}" * 16, "program_id": pid, "spec": {"family": "bb"},
            "per_logical_per_round": plpr}
           for h, pid, plpr in (("ee", "runtime_champion", 1e-4), ("99", "runtime_second", 2e-4),
                                ("77", "runtime_third", 3e-4), ("55", "runtime_fourth", 4e-4),
                                ("33", "runtime_fifth", 5e-4))]
    entries, composition = _final_entries(_final_state(structural, hof), STRUCTURAL)
    assert len(entries) == 6
    channels = [e["selection_channels"][0] for e in entries]
    assert channels.count("structural") == 2 and channels.count("runtime") == 4
    assert [e["program_id"] for e in entries[2:]] == [
        "runtime_champion", "runtime_second", "runtime_third", "runtime_fourth"
    ]
    assert composition["selected"][0]["k"] == 20      # high-k endpoint
    assert composition["selected"][1]["d_search"] == 12  # high-d endpoint
    assert (composition["structural_slots"], composition["runtime_slots"]) == (2, 4)


def test_final_shortlist_backfills_and_keeps_dual_channel_labels() -> None:
    shared = _elite("aa", 20, 5)
    structural = [shared, _elite("bb", 4, 12)]
    hof = [dict(shared, per_logical_per_round=1e-4)]
    entries, _ = _final_entries(_final_state(structural, hof), STRUCTURAL)
    labels = {e["program_id"]: e["selection_channels"] for e in entries}
    assert labels["struct_aa"] == ["structural", "runtime"]
    assert len(entries) == 2  # nothing left to backfill with, and no duplicates


def test_final_rounds_have_exactly_one_source_of_truth() -> None:
    protocol = {
        "rounds_policy": {"final": "max_Tc_d", "final_rounds": 12},
        "verification": {"rounds": 12},
    }
    assert _protocol_final_rounds(protocol) == 12

    with pytest.raises(ValueError, match="rounds conflict"):
        _protocol_final_rounds({
            "rounds_policy": {"final": "max_Tc_d", "final_rounds": 12},
            "verification": {"rounds": 5},
        })
    # 2026-10-15: fixed_Tc (the Fig.3 formal run) is a second legal id; the
    # retired "T=d" label and anything else is still refused.
    with pytest.raises(ValueError, match="must be one of"):
        _protocol_final_rounds({"rounds_policy": {"final": "T=d"}, "verification": {}})
    assert _protocol_final_rounds({"rounds_policy": {"final": "fixed_Tc", "final_rounds": 12}, "verification": {}}) == 12
    with pytest.raises(ValueError, match="final_rounds is missing"):
        _protocol_final_rounds({"rounds_policy": {"final": "max_Tc_d"}, "verification": {}})


def test_final_sampling_keys_cannot_collide_with_search_ones() -> None:
    """"Fresh shots" has to be a fact about the seed derivation, not a claim:
    search candidates derive from base 0 + index * 1e6."""

    base = final_rng_base_seed("final_confirmation_v1")
    assert base >= 10**12
    assert base != final_rng_base_seed("search_v1")
    assert final_rng_base_seed("final_confirmation_v1") == base  # deterministic
    search_keys = {index * 1_000_000 for index in range(1000)}
    final_keys = {base + index * 1_000_000 for index in range(1000)}
    assert not (search_keys & final_keys)


def test_unmeasurable_candidate_stops_consuming_a_slow_slot() -> None:
    """schedule_catastrophic produces no measurement, so it never reaches the
    evidence ledger — and a structural elite that cannot be measured would
    otherwise be re-selected for a structural slot every single turn."""

    from omniqec_agentic.campaign import _build_slow_queue
    from omniqec_agentic.evaluation.writeback import ranking_rows

    # The verdict is now carried on the ranking row, which is the only place a
    # downstream filter can see it.
    rows = ranking_rows(
        {
            "program_id": "cat",
            "structural": {"code_hash": "aa" * 16},
            "distance_evidence": {"schedule_catastrophic": True},
            "results": [{"decoder": "bposd", "status": "ok"}],
        },
        Path("/tmp/cand.json"),
    )
    assert rows[0]["schedule_catastrophic"] is True

    import json as _json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        queue = tmp_path / "s.jsonl"
        queue.write_text(_json.dumps(_elite("aa", 20, 5)) + "\n", encoding="utf-8")
        out, composition = _build_slow_queue(
            structural_queue=str(queue), promotion_queue=None,
            all_archive_queue=None, priority=[],
            structural_config=STRUCTURAL, out_dir=tmp_path,
            measured_hashes={"aa" * 16},  # graph unions unmeasurable into this
        )
        assert composition["selected"] == []


def test_missing_candidate_file_degrades_instead_of_crashing() -> None:
    assert _inheritance_from_candidate(None)["spec"] is None
    assert _inheritance_from_candidate("/nonexistent/x.json")["construct_source"] is None


# ---------------------------------------------------------------------------
# arrival-order independence (audit fix #3)
# ---------------------------------------------------------------------------
def _crowded_pool() -> list[dict]:
    """More Pareto-eligible rows than the cap, across several fronts, so the
    reducer must actually drop candidates."""

    rows = []
    codes = "abcdefghijklmnopqrstuvwxyz"
    for index in range(24):
        k = 2 + index
        d = 3 + (23 - index) // 2
        rows.append(_elite(codes[index] * 2, k, d, program_id=f"p_{index:02d}"))
    return rows


def test_elites_do_not_depend_on_candidate_arrival_order() -> None:
    """Streaming one-by-one, streaming reversed, and one batch must all yield
    the same elite set: reduce(reduced-pool + next) is not associative, so the
    old per-candidate cap made retention depend on LLM output order."""

    pool = _crowded_pool()

    def _elite_hashes(rows: list[dict]) -> set[str]:
        memory = ArchiveMemory(turn=0)
        for row in rows:
            memory.reconcile_structural_elites([row])
        return {r["code_hash"] for r in memory.structural_proxy_elites}

    batch_memory = ArchiveMemory(turn=0)
    batch_memory.reconcile_structural_elites(list(pool))
    batch = {r["code_hash"] for r in batch_memory.structural_proxy_elites}

    assert len(batch_memory.structural_proxy_elites) <= batch_memory.structural_proxy_elites_limit
    assert _elite_hashes(pool) == batch
    assert _elite_hashes(list(reversed(pool))) == batch
    import random as _random

    shuffled = list(pool)
    _random.Random(7).shuffle(shuffled)
    assert _elite_hashes(shuffled) == batch


def test_the_ledger_is_lossless_beyond_the_elite_cap() -> None:
    """A row dropped from the 16-elite view is still in the ledger, so a later
    turn's reduce sees the full history (and named promotions can find it)."""

    pool = _crowded_pool()
    memory = ArchiveMemory(turn=0)
    for row in pool:
        memory.reconcile_structural_elites([row])
    assert len(memory.structural_ledger) == len(pool)
    assert len(memory.structural_proxy_elites) < len(pool)


def test_structural_queue_emits_the_whole_pool_in_slot_order() -> None:
    """Audit fix #4: the builder needs backfill material — a pre-truncated
    top-3 emission is what let one WL collision reshape the queue."""

    memory = ArchiveMemory(turn=0)
    memory.reconcile_structural_elites(
        [_elite("aa", 20, 5), _elite("bb", 4, 12), _elite("cc", 10, 8), _elite("dd", 8, 9)]
    )
    queue = memory.structural_slow_queue(slots=3)
    assert len(queue) == 4  # whole unmeasured pool, not 3
    assert queue[0]["structural_slot_reason"] == "high_k_endpoint"
    assert queue[1]["structural_slot_reason"] == "high_d_search_endpoint"
    assert all(row["structural_slots"] == 3 for row in queue)


# ---------------------------------------------------------------------------
# endpoint domination + coordinate crowding (audit round 3, fixes #1/#2)
# ---------------------------------------------------------------------------
def test_high_k_endpoint_never_selects_a_dominated_code() -> None:
    """A=(20,5,n100) is strictly dominated by B=(20,10,n288); the old §1.2
    tie-break (n_total first) still handed A the high-k endpoint slot because
    it is smaller, and D=(10,11) fell off the 3-slot queue entirely."""

    from omniqec_agentic import ranking

    pool = [
        _elite("aa", 20, 5, n_total=100),
        _elite("bb", 20, 10, n_total=288),
        _elite("cc", 4, 12, n_total=288),
        _elite("dd", 10, 11, n_total=288),
    ]
    reduced = ranking.reduce_structural_proxy_elites(pool, limit=16)
    slots = ranking.select_structural_slots(reduced, slots=3)
    picked = {row["program_id"]: row["structural_slot_reason"] for row in slots}
    assert set(picked) == {"struct_bb", "struct_cc", "struct_dd"}
    assert picked["struct_bb"] == "high_k_endpoint"
    assert picked["struct_cc"] == "high_d_search_endpoint"


def test_same_coordinate_realizations_share_crowding_and_break_by_size() -> None:
    """Per-row crowding split the two (10,8) realizations, so n=200 could beat
    the smaller n=100 before the within-coordinate tie-break ever ran."""

    from omniqec_agentic import ranking

    pool = [
        _elite("aa", 10, 8, n_total=200),
        _elite("bb", 10, 8, n_total=100),
        _elite("cc", 20, 5, n_total=144),
        _elite("dd", 3, 12, n_total=144),
        _elite("ee", 15, 6, n_total=144),
        _elite("ff", 6, 10, n_total=144),
    ]
    full = ranking.reduce_structural_proxy_elites(pool, limit=16)
    crowding = {
        row["program_id"]: (row["crowding_distance"], row["crowding_boundary"])
        for row in full
        if row["program_id"] in ("struct_aa", "struct_bb")
    }
    assert crowding["struct_aa"] == crowding["struct_bb"]

    # limit=4: endpoints (20,5)/(3,12) forced, room for 2 of the 4 interior
    # rows — the coordinate pair must split by n_total, keeping n=100.
    cut = ranking.reduce_structural_proxy_elites(pool, limit=4)
    names = {row["program_id"] for row in cut}
    assert "struct_bb" in names or "struct_aa" not in names
    if "struct_aa" in names:
        assert "struct_bb" in names  # n=200 may never survive INSTEAD of n=100
    assert "struct_bb" in names


def test_final_runtime_slot_scans_past_a_structural_duplicate() -> None:
    """Audit fix #3: trying only hof[:1] meant a runtime champion that was
    already a structural pick silently produced a 4S+0R final shortlist."""

    from omniqec_agentic.config import StructuralSearchConfig
    from omniqec_agentic.workflow.graph import _final_entries

    def _cand(code: str, k: int, d: int, plpr: float | None = None) -> dict:
        row = {
            "code_hash": code * 16,
            "program_id": f"cand_{code}",
            "k": k,
            "d_search": d,
            "spec": {"family": "bb"},
            "construct_source": "def construct(rng): return {}",
        }
        if plpr is not None:
            row["per_logical_per_round"] = plpr
        return row

    structural = [_cand("aa", 20, 5), _cand("bb", 4, 12), _cand("cc", 10, 8), _cand("dd", 8, 9)]
    state = {
        "structural_proxy_elites": structural,
        # hof[0] is ALSO the structural high-k pick; hof[1] is the genuine
        # runtime finalist the old code never reached.
        "hof_summary": [dict(_cand("aa", 20, 5), per_logical_per_round=1e-6),
                        dict(_cand("ee", 6, 6), per_logical_per_round=2e-6)],
    }
    entries, _ = _final_entries(state, StructuralSearchConfig())
    by_id = {row["program_id"]: row["selection_channels"] for row in entries}
    # 4 structural + the one genuine runtime finalist (2026-09-10 quota 4S+2R:
    # the second runtime slot has no material left in this hall of fame).
    assert len(entries) == 5
    assert "cand_ee" in by_id, "the runtime slot must scan past the duplicate"
    assert by_id["cand_ee"] == ["runtime"]


def test_run_level_store_reduce_is_streamed_and_order_independent(tmp_path) -> None:
    """Audit fix #4 (scale): the canonical elites are reduce(the whole
    append-only store); appending in any order yields the same elite set, and
    duplicate lines from a crash-resume are deduplicated on read."""

    from omniqec_agentic.workflow.graph import (
        _append_structural_store,
        _reduce_elites_from_store,
    )

    pool = _crowded_pool()

    def _elites_for(order: list[dict]) -> list[tuple[str, int]]:
        store = tmp_path / f"store_{id(order)}.jsonl"
        known: set[str] = set()
        _append_structural_store(store, order, known)
        # a crash between append and checkpoint can duplicate rows
        _append_structural_store(store, order[:3], set())
        reduced = _reduce_elites_from_store(store, limit=16)
        return sorted((row["code_hash"], row["pareto_rank"]) for row in reduced)

    forward = _elites_for(list(pool))
    backward = _elites_for(list(reversed(pool)))
    assert forward == backward
    assert 0 < len(forward) <= 16
    # winners keep their full inheritance payload from the store
    store = tmp_path / "full.jsonl"
    _append_structural_store(store, pool, set())
    for row in _reduce_elites_from_store(store, limit=16):
        assert row["spec"] and row["construct_source"]
