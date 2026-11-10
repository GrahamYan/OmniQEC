"""2026-09-24: the fixed-T final stage resumes per candidate and may cap its
decode-shard watchdog.

Background: the DeepSeek b144/b220/b260 finals each ran a full 24 h link and
died at walltime with nothing written -- the pool only wrote candidates after
ALL of them finished, and one non-converging d>=14 shard was allowed to run for
a day by the telescoping worst-case term. Now every assembled candidate is
persisted at once, an attempt reuses what an earlier attempt of the same queue
measured (same seed / shots / shards / verification), and the final stage can
carry a hard ceiling on the shard watchdog (a protocol RUNTIME key).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from omniqec_agentic import config as config_mod
from omniqec_agentic.config import VerificationConfig
from omniqec_agentic.evaluation import pool as pool_mod
from omniqec_agentic.evaluation.common import verification_from_dict
from omniqec_agentic.workflow import checkpoint as checkpoint_mod
from omniqec_agentic.workflow.state import build_protocol_snapshot


# ---------------------------------------------------------------- watchdog cap
def _formal() -> VerificationConfig:
    return VerificationConfig()


def test_final_cap_only_lowers_the_final_stage_watchdog() -> None:
    formal = _formal()
    shots, cols = 195, 120_000  # a 25k/128 final shard on a large two-basis DEM
    uncapped_search = pool_mod.effective_decode_shard_timeout_s(formal, shots, cols, "search")
    uncapped_final = pool_mod.effective_decode_shard_timeout_s(formal, shots, cols, "final")
    assert uncapped_final is not None and uncapped_final > 14_400.0  # the problem: > 4 h per shard

    capped = dataclasses.replace(formal, final_decode_shard_timeout_cap_s=14_400.0)
    assert pool_mod.effective_decode_shard_timeout_s(capped, shots, cols, "final") == pytest.approx(14_400.0)
    # the search stage never sees the ceiling
    assert pool_mod.effective_decode_shard_timeout_s(capped, shots, cols, "search") == uncapped_search
    # a ceiling above the model changes nothing
    loose = dataclasses.replace(formal, final_decode_shard_timeout_cap_s=uncapped_final * 10)
    assert pool_mod.effective_decode_shard_timeout_s(loose, shots, cols, "final") == uncapped_final
    # the job-config record names the ceiling for the final stage only
    model_final = pool_mod.decode_budget_model(capped, shots, cols, "final")
    model_search = pool_mod.decode_budget_model(capped, shots, cols, "search")
    assert model_final["final_decode_shard_timeout_cap_s"] == pytest.approx(14_400.0)
    assert model_search["final_decode_shard_timeout_cap_s"] is None


def test_final_cap_is_a_runtime_key_that_never_enters_the_verification_dict() -> None:
    formal = _formal()
    capped = dataclasses.replace(formal, final_decode_shard_timeout_cap_s=3600.0)
    # to_dict is the frozen protocol's verification block: unchanged by the cap
    assert capped.to_dict() == formal.to_dict()
    assert "final_decode_shard_timeout_cap_s" not in capped.to_dict()
    # round trip keeps the default; a hand-written dict may still set it
    assert verification_from_dict(capped.to_dict()).final_decode_shard_timeout_cap_s is None
    raw = dict(formal.to_dict(), final_decode_shard_timeout_cap_s=120.0)
    assert verification_from_dict(raw).final_decode_shard_timeout_cap_s == pytest.approx(120.0)
    # resume treats the key like slow_workers
    assert "final_decode_shard_timeout_cap_s" in checkpoint_mod.RESUME_RUNTIME_KEYS
    assert "final_decode_shard_timeout_cap_s" in checkpoint_mod.RESUME_IGNORED_KEYS


def _snapshot(**overrides):
    kwargs = dict(
        verification=_formal(),
        racing=config_mod.formal_racing(25_000),
        max_turns=30,
        proposal_count=24,
        generations_per_turn=4,
        slow_workers=0,
        strict_turns=True,
        required_provider="openrouter",
        required_model="deepseek/deepseek-v4-pro",
        rng_seeds_per_program=40,
        n_budget=288,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)


def test_snapshot_carries_the_cap_only_when_set_and_resume_accepts_it() -> None:
    plain = _snapshot()
    assert "final_decode_shard_timeout_cap_s" not in plain
    capped = _snapshot(final_decode_shard_timeout_cap_s=14_400)
    assert capped["final_decode_shard_timeout_cap_s"] == pytest.approx(14_400.0)
    with pytest.raises(ValueError):
        _snapshot(final_decode_shard_timeout_cap_s=0)
    # an old checkpoint (no key) resumes under the capped launch: the diff
    # ignores the runtime key and reports it as an override
    checkpoint_mod.validate_resume_protocol(plain, capped)
    assert checkpoint_mod.resume_runtime_overrides(plain, capped) == {
        "final_decode_shard_timeout_cap_s": pytest.approx(14_400.0)
    }


# ---------------------------------------------------------------- per-candidate resume
def _queue(tmp_path: Path, n: int = 3) -> Path:
    queue = tmp_path / "queue.jsonl"
    rows = []
    for i in range(n):
        rows.append({
            "program_id": f"cand_{i}",
            "spec": {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [1, 0], [0, 2 + i]], "B": [[0, 0], [0, 1], [2, 0]]},
            "selection_channels": ["structural"],
            "is_baseline": False,
            "retry_at_final": False,
        })
    queue.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return queue


def _fake_evaluate_slow_spec(calls: list[str]):
    def fake(*, program_id, spec, verification, seed, promotion_record, **_kw):
        calls.append(program_id)
        return {
            "schema": "slow_verification_run",
            "schema_version": pool_mod.SLOW_SCHEMA_VERSION,
            "program_id": program_id,
            "status": "ok",
            "seed": int(seed),
            "spec": spec,
            "hardware": "ion",
            "promotion_record": promotion_record,
            "results": [],
            "runtime_s": 0.0,
        }

    return fake


def _run(tmp_path: Path, queue: Path, calls: list[str], monkeypatch, *, seed: int = 7, resume: bool = True, shots: int = 100):
    monkeypatch.setattr(pool_mod, "evaluate_slow_spec", _fake_evaluate_slow_spec(calls))
    # the serial path (workers=1) is enough to exercise the reuse logic; the
    # job pool takes the same `preloaded` mapping and the same persist hook
    ver = dataclasses.replace(_formal(), shots=shots)
    return pool_mod.evaluate_promotion_queue(
        queue_path=queue,
        out_dir=tmp_path / "T12",
        verification=ver,
        include_baseline=False,
        workers=1,
        shot_shards=1,
        seed=seed,
        stage="final",
        resume_candidates=resume,
    )


def test_second_attempt_reuses_finished_candidates_and_keeps_their_seeds(tmp_path, monkeypatch) -> None:
    queue = _queue(tmp_path)
    calls: list[str] = []
    _run(tmp_path, queue, calls, monkeypatch)
    assert calls == ["cand_0", "cand_1", "cand_2"]
    out = tmp_path / "T12"
    files = sorted(p.name for p in (out / "candidates").glob("*.json"))
    assert files == ["cand_0.json", "cand_1.json", "cand_2.json"]

    # simulate a link killed after two candidates: the third file is gone
    (out / "candidates" / "cand_2.json").unlink()
    (out / "slow_results.jsonl").write_text("", encoding="utf-8")
    calls.clear()
    summary = _run(tmp_path, queue, calls, monkeypatch)
    assert calls == ["cand_2"], "only the unfinished candidate is measured again"
    rows = [json.loads(l) for l in (out / "slow_results.jsonl").read_text().splitlines() if l.strip()]
    assert [r["program_id"] for r in rows] == ["cand_0", "cand_1", "cand_2"]
    # seeds are index-derived from the frozen base: the reused rows carry the
    # protocol's seeds and the re-measured one got its own index seed
    seeds = {json.loads((out / "candidates" / f).read_text())["program_id"]:
             json.loads((out / "candidates" / f).read_text())["seed"] for f in files}
    assert seeds == {"cand_0": 7, "cand_1": 7 + 1_000_000, "cand_2": 7 + 2_000_000}
    # the fake payloads carry no results, so the queue status is the
    # "no valid candidates" verdict; what matters here is the reuse above
    assert isinstance(summary["status"], str) and summary["artifact_paths"]


def test_a_changed_measurement_config_measures_everything_again(tmp_path, monkeypatch) -> None:
    queue = _queue(tmp_path)
    calls: list[str] = []
    _run(tmp_path, queue, calls, monkeypatch, shots=100)
    calls.clear()
    _run(tmp_path, queue, calls, monkeypatch, shots=200)  # different shots -> nothing reusable
    assert calls == ["cand_0", "cand_1", "cand_2"]
    calls.clear()
    _run(tmp_path, queue, calls, monkeypatch, shots=200, seed=8)  # different seed -> nothing reusable
    assert calls == ["cand_0", "cand_1", "cand_2"]


def _run_with_final_cap(tmp_path: Path, queue: Path, calls: list[str], monkeypatch, *, cap: float | None):
    monkeypatch.setattr(pool_mod, "evaluate_slow_spec", _fake_evaluate_slow_spec(calls))
    ver = dataclasses.replace(_formal(), shots=100, final_decode_shard_timeout_cap_s=cap)
    return pool_mod.evaluate_promotion_queue(
        queue_path=queue, out_dir=tmp_path / "T12", verification=ver, include_baseline=False,
        workers=1, shot_shards=1, seed=7, stage="final", resume_candidates=True,
    )


def test_a_raised_shard_cap_still_reuses_finished_candidates(tmp_path, monkeypatch) -> None:
    """2026-10-14: the final cap is a protocol RUNTIME key (a later link may
    raise it); the slow_run_config comparison must not void the reuse, or the
    uncapped attempt measures every finalist again (the capped structural
    endpoints included) instead of only the candidates it was asked for."""
    queue = _queue(tmp_path)
    calls: list[str] = []
    _run_with_final_cap(tmp_path, queue, calls, monkeypatch, cap=14400.0)
    assert calls == ["cand_0", "cand_1", "cand_2"]
    out = tmp_path / "T12"
    assert json.loads((out / "slow_run_config.json").read_text())["final_decode_shard_timeout_cap_s"] == 14400.0
    (out / "candidates" / "cand_1.json").unlink()  # the one candidate to measure again
    calls.clear()
    _run_with_final_cap(tmp_path, queue, calls, monkeypatch, cap=2592000.0)
    assert calls == ["cand_1"], "a different ceiling keeps the finished candidates"
    assert json.loads((out / "slow_run_config.json").read_text())["final_decode_shard_timeout_cap_s"] == 2592000.0
    calls.clear()
    _run_with_final_cap(tmp_path, queue, calls, monkeypatch, cap=None)
    assert calls == [], "no ceiling at all reuses every finished candidate as well"


def test_a_candidate_file_with_a_different_spec_or_id_is_not_reused(tmp_path, monkeypatch) -> None:
    queue = _queue(tmp_path)
    calls: list[str] = []
    _run(tmp_path, queue, calls, monkeypatch)
    out = tmp_path / "T12"
    # the queue changed under the same program_id: cand_1 now names another code
    rows = [json.loads(l) for l in queue.read_text().splitlines()]
    rows[1]["spec"]["A"][2] = [0, 5]
    queue.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    calls.clear()
    _run(tmp_path, queue, calls, monkeypatch)
    assert calls == ["cand_1"]
    # a candidate file that never finished ("started") is measured again
    p = out / "candidates" / "cand_0.json"
    d = json.loads(p.read_text()); d["status"] = "started"; p.write_text(json.dumps(d))
    calls.clear()
    _run(tmp_path, queue, calls, monkeypatch)
    assert calls == ["cand_0"]


def test_resume_off_keeps_the_v37_behaviour(tmp_path, monkeypatch) -> None:
    queue = _queue(tmp_path)
    calls: list[str] = []
    _run(tmp_path, queue, calls, monkeypatch, resume=False)
    calls.clear()
    _run(tmp_path, queue, calls, monkeypatch, resume=False)
    assert calls == ["cand_0", "cand_1", "cand_2"]


# ---------------------------------------------------------------- final_slow_node plumbing
def _final_node_state(tmp_path: Path, cap: float | None) -> dict:
    protocol = {
        "verification": VerificationConfig().to_dict(),
        "rounds_policy": {"final": "max_Tc_d", "final_rounds": 12, "final_rng_namespace": "final_confirmation_v1"},
        "final_p_values": [0.005],
        "final_shots": 25_000,
        "slow_workers": 127,
        "shot_shards": 16,
        "final_shot_shards": 128,
        "strict_turns": False,
    }
    if cap is not None:
        protocol["final_decode_shard_timeout_cap_s"] = cap
    return {
        "run_dir": str(tmp_path),
        "protocol": protocol,
        "spec": {"qubit_budget": 144, "families": ["bb"]},
        "baseline_summary": {},
    }


@pytest.mark.parametrize("cap", [None, 14_400.0])
def test_final_slow_node_applies_the_cap_and_asks_for_candidate_resume(monkeypatch, tmp_path, cap) -> None:
    """The 2026-09-24 NameError regression: with a cap set final_slow_node
    replaces the verification (dataclasses) -- the module must import it."""

    import omniqec_agentic.workflow.graph as graph

    captured: dict = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "artifact_paths": {}}

    monkeypatch.setattr(graph, "evaluate_promotion_queue", fake_evaluate)
    monkeypatch.setattr(graph, "_validate_all_turn_audits", lambda state: {})
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *a, **k: VerificationConfig())
    monkeypatch.setattr(
        graph, "_final_entries",
        lambda state, structural: ([{"program_id": "candidate", "spec": {"family": "bb", "ell": 6, "m": 1},
                                     "code_hash": "abc", "selection_channels": ["structural_high_k"]}], {"entries": 1}),
    )
    monkeypatch.setattr(graph, "_audit_final_measurements", lambda state, runs: {"enabled": False, "status": "complete", "errors": []})
    graph.final_slow_node(_final_node_state(tmp_path, cap))
    assert captured["stage"] == "final" and captured["resume_candidates"] is True
    assert captured["verification"].final_decode_shard_timeout_cap_s == (None if cap is None else pytest.approx(cap))
    assert captured["shot_shards"] == 128


# ---------------------------------------------------------------- search-stage ceiling (2026-09-25)
def test_search_cap_only_lowers_the_search_stage_watchdog() -> None:
    formal = _formal()
    shots, cols = 1563, 120_000  # a 25k/16 search shard on a large two-basis DEM
    uncapped_search = pool_mod.effective_decode_shard_timeout_s(formal, shots, cols, "search")
    uncapped_final = pool_mod.effective_decode_shard_timeout_s(formal, shots, cols, "final")
    assert uncapped_search is not None and uncapped_search > 28_800.0
    capped = dataclasses.replace(formal, search_decode_shard_timeout_cap_s=28_800.0)
    assert pool_mod.effective_decode_shard_timeout_s(capped, shots, cols, "search") == pytest.approx(28_800.0)
    assert pool_mod.effective_decode_shard_timeout_s(capped, shots, cols, "final") == uncapped_final
    both = dataclasses.replace(capped, final_decode_shard_timeout_cap_s=14_400.0)
    assert pool_mod.decode_shard_timeout_cap_s(both, "search") == pytest.approx(28_800.0)
    assert pool_mod.decode_shard_timeout_cap_s(both, "final") == pytest.approx(14_400.0)
    assert pool_mod.decode_shard_timeout_cap_s(both, "warmup") is None
    model = pool_mod.decode_budget_model(both, shots, cols, "search")
    assert model["decode_shard_timeout_cap_s"] == pytest.approx(28_800.0)
    assert model["final_decode_shard_timeout_cap_s"] is None
    # never part of the frozen verification block
    assert both.to_dict() == formal.to_dict()
    assert "search_decode_shard_timeout_cap_s" in checkpoint_mod.RESUME_RUNTIME_KEYS


def test_search_cap_reaches_the_turn_verification_and_the_snapshot() -> None:
    import omniqec_agentic.workflow.graph as graph

    base = _formal()
    assert graph._with_search_shard_cap(base, {}) is base
    assert graph._with_search_shard_cap(base, {"search_decode_shard_timeout_cap_s": 0}) is base
    capped = graph._with_search_shard_cap(base, {"search_decode_shard_timeout_cap_s": 28_800})
    assert capped.search_decode_shard_timeout_cap_s == pytest.approx(28_800.0)
    assert capped.final_decode_shard_timeout_cap_s is None
    plain = _snapshot()
    assert "search_decode_shard_timeout_cap_s" not in plain
    snap = _snapshot(search_decode_shard_timeout_cap_s=28_800, final_decode_shard_timeout_cap_s=14_400)
    assert snap["search_decode_shard_timeout_cap_s"] == pytest.approx(28_800.0)
    checkpoint_mod.validate_resume_protocol(plain, snap)
    assert set(checkpoint_mod.resume_runtime_overrides(plain, snap)) == {
        "search_decode_shard_timeout_cap_s", "final_decode_shard_timeout_cap_s"
    }
    with pytest.raises(ValueError):
        _snapshot(search_decode_shard_timeout_cap_s=-1)


# ---------------------------------------------------------------- candidate-level abort (2026-09-26)
def test_persistent_pool_cancel_where_resolves_only_queued_matching_items() -> None:
    from collections import deque
    import threading
    from omniqec_agentic.evaluation.worker_pool import PersistentWorkerPool

    # build the object without starting worker threads: the queue logic is pure
    p = PersistentWorkerPool.__new__(PersistentWorkerPool)
    p._lock = threading.Condition(); p._order = deque(); p._by_key = {}; p._closing = False
    p.stats = {"submitted": 0, "completed": 0, "respawns": 0, "timeouts": 0, "affinity_hits": 0}
    futs = {}
    for ctx, shard in [(7, 0), (7, 1), (3, 0), (7, 2)]:
        futs[(ctx, shard)] = p.submit({"mode": "shard_dem", "shard": shard}, key=(ctx, 0, (("telescoping_cpu", "X"),)))
    # one of candidate 7's chunks was already taken by a worker
    p._order[1]["taken"] = True
    n = p.cancel_where(lambda key: key[0] == 7, lambda item: {"status": "timeout", "job_execution": {"attempts": 1, "cancelled": True}})
    assert n == 2
    assert futs[(7, 0)].done() and futs[(7, 0)].result()["status"] == "timeout"
    assert futs[(7, 2)].done()
    assert not futs[(7, 1)].done(), "a taken request is left to its own watchdog"
    assert not futs[(3, 0)].done(), "another candidate is untouched"
    assert p.stats["cancelled"] == 2
    # the by-key queue no longer offers the cancelled items
    remaining = [i for dq in p._by_key.values() for i in dq if not i["taken"]]
    assert [i["request"]["shard"] for i in remaining] == [0] and remaining[0]["key"][0] == 3


def test_a_cancelled_chunk_counts_as_a_returned_timeout_for_the_infeasible_gate() -> None:
    from omniqec_agentic.evaluation.sampling import merge_shard_results

    real = {"status": "timeout", "decoder": "telescoping_cpu", "shots": 0, "failures": 0,
            "logical_error_rate": None, "runtime_s": 14400.0,
            "job_execution": {"attempts": 1, "timeout": True}}
    cancelled = {"status": "timeout", "decoder": "telescoping_cpu", "shots": 0, "failures": 0,
                 "logical_error_rate": None, "runtime_s": 0.0,
                 "job_execution": {"attempts": 1, "timeout": True, "cancelled": True}}
    merged = {b: merge_shard_results([real, cancelled, cancelled], decoder="telescoping_cpu") for b in ("X", "Z")}
    assert merged["X"]["shard_statuses"] == ["timeout"] * 3
    assert merged["X"]["shard_attempts"] == [1, 1, 1]
    result = {"basis": merged}
    assert pool_mod.fixed_shot_timeout_only(result, required_bases=("X", "Z")) is True
    # a chunk that never returned at all keeps the queue measurement_blocked
    merged_missing = {b: merge_shard_results([real, None, cancelled], decoder="telescoping_cpu") for b in ("X", "Z")}
    assert pool_mod.fixed_shot_timeout_only({"basis": merged_missing}, required_bases=("X", "Z")) is False


# ---------------------------------------------------------------- fixed program count for variants (2026-09-26)
def test_fixed_program_count_can_be_frozen_without_strict_turns() -> None:
    assert _snapshot(strict_turns=False)["fixed_program_count"] is False
    assert _snapshot(strict_turns=False, fixed_program_count=True)["fixed_program_count"] is True
    assert _snapshot(strict_turns=True, fixed_program_count=False)["fixed_program_count"] is True
    assert _snapshot(strict_turns=True)["fixed_program_count"] is True


# ---------------------------------------------------------------- per-chunk share of the ceiling (2026-09-29)
def test_a_split_chunk_gets_its_proportional_share_of_the_ceiling() -> None:
    capped = dataclasses.replace(_formal(), final_decode_shard_timeout_cap_s=14_400.0,
                                 search_decode_shard_timeout_cap_s=28_800.0)
    # final: a 195-shot shard split 64 ways -> ~3-shot chunks -> the 300 s floor
    assert pool_mod.effective_decode_shard_timeout_s(capped, 195, 120_000, "final", chunk_shots=3) == pytest.approx(300.0)
    # search: a 1563-shot shard split 64 ways -> 24-shot chunks -> 28800 x 24/1563
    assert pool_mod.effective_decode_shard_timeout_s(capped, 1563, 120_000, "search", chunk_shots=24) == pytest.approx(28_800.0 * 24 / 1563)
    # an unsplit shard keeps the whole ceiling; no ceiling -> the chunk changes nothing
    assert pool_mod.effective_decode_shard_timeout_s(capped, 1563, 120_000, "search") == pytest.approx(28_800.0)
    plain = _formal()
    assert pool_mod.effective_decode_shard_timeout_s(plain, 1563, 120_000, "search", chunk_shots=24) == \
        pool_mod.effective_decode_shard_timeout_s(plain, 1563, 120_000, "search")
    # the worker-side resolver derives the chunk from the request's shot_range
    req = {"mode": "shard_dem", "shots": 195, "num_matrix_columns": 120_000, "stage": "final", "shot_range": [6, 9]}
    assert pool_mod.worker_timeout_for(req, capped) == pytest.approx(300.0)
    req_whole = dict(req); req_whole.pop("shot_range")
    assert pool_mod.worker_timeout_for(req_whole, capped) == pytest.approx(14_400.0)


def _final_node_with_baseline(monkeypatch, tmp_path, baseline_hash: str) -> list[dict]:
    """Run final_slow_node with one finalist of hash 'abc' and a baseline of ``baseline_hash``;
    return the rows of the queue file the node wrote."""

    import json
    import omniqec_agentic.workflow.graph as graph

    monkeypatch.setattr(graph, "evaluate_promotion_queue", lambda **kwargs: {"status": "ok", "artifact_paths": {}})
    monkeypatch.setattr(graph, "_validate_all_turn_audits", lambda state: {})
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *a, **k: VerificationConfig())
    monkeypatch.setattr(
        graph, "_final_entries",
        lambda state, structural: ([{"program_id": "g0001_p0001", "spec": {"family": "bb", "ell": 6, "m": 6},
                                     "code_hash": "abc", "selection_channels": ["structural"]}], {"entries": 1}),
    )
    monkeypatch.setattr(graph, "_audit_final_measurements", lambda state, runs: {"enabled": False, "status": "complete", "errors": []})
    state = _final_node_state(tmp_path, None)
    state["baseline_summary"] = {"program_id": "baseline_bb_72_12_6", "spec": {"family": "bb", "ell": 6, "m": 6},
                                 "code_hash": baseline_hash, "is_baseline": True}
    graph.final_slow_node(state)
    queue = (tmp_path / "final_slow" / "queue_T12.jsonl").read_text().splitlines()
    return [json.loads(line) for line in queue if line.strip()]


def test_baseline_row_is_skipped_when_the_reference_code_is_already_a_finalist(monkeypatch, tmp_path) -> None:
    """2026-10-14 b144 Phase B: the LLM's turn-0 copy of BB [[72,12,6]] was a finalist and the
    baseline row measured the identical circuit next to it (and lost the shared DEM pickle)."""

    import json

    rows = _final_node_with_baseline(monkeypatch, tmp_path, "abc")
    assert [row["program_id"] for row in rows] == ["g0001_p0001"]
    assert not any(row["is_baseline"] for row in rows)
    shortlist = json.loads((tmp_path / "final_slow" / "final_shortlist.json").read_text())
    assert shortlist["baseline_row"] == "skipped_duplicate_finalist"


def test_baseline_row_stays_when_no_finalist_shares_its_code(monkeypatch, tmp_path) -> None:
    rows = _final_node_with_baseline(monkeypatch, tmp_path, "zzz")
    assert [row["program_id"] for row in rows] == ["g0001_p0001", "baseline_bb_72_12_6"]
    assert [row["is_baseline"] for row in rows] == [False, True]


def test_final_structural_slots_respect_the_promotion_weight_gate(tmp_path) -> None:
    """2026-10-14 b144 DeepSeek final: the proxy frontier's (k, d) endpoints were w12 / w24
    codes the gate had kept out of the slow loop; the final must not measure them."""

    import json
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic.config import StructuralSearchConfig

    ledger = tmp_path / "structural_ledger.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(r) for r in [
            {"code_hash": "heavyk", "max_check_weight": 24},
            {"code_hash": "heavyd", "max_check_weight": 12},
            {"code_hash": "light1", "max_check_weight": 4},
            {"code_hash": "light2", "max_check_weight": 6},
        ]) + "\n", encoding="utf-8",
    )

    def elite(code_hash: str, k: int, d: int) -> dict:
        return {"code_hash": code_hash, "program_id": "p_" + code_hash, "spec": {"family": "bb"},
                "k": k, "d_search": d, "d_search_proxy": d, "pareto_rank": 1}

    elites = [elite("heavyk", 21, 5), elite("heavyd", 4, 12), elite("light1", 2, 8), elite("light2", 4, 8)]
    structural = StructuralSearchConfig(final_structural=2, final_runtime=0)

    gated = {"run_dir": str(tmp_path), "protocol": {"promotion_max_check_weight": 6},
             "structural_proxy_elites": elites, "hof_summary": [], "unmeasurable_hashes": []}
    entries, composition = graph._final_entries(gated, structural)
    assert {e["code_hash"] for e in entries} == {"light1", "light2"}
    assert composition["weight_gate"] == 6
    assert sorted(composition["elites_over_weight_gate"]) == ["heavyd", "heavyk"]

    open_state = dict(gated, protocol={})
    entries, composition = graph._final_entries(open_state, structural)
    assert {e["code_hash"] for e in entries} == {"heavyk", "heavyd"}, "no gate: the frontier endpoints as before"
    assert composition["weight_gate"] is None and composition["elites_over_weight_gate"] == []
