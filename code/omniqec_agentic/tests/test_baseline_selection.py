"""Search baselines without an LLM (2026-11-06): the slow-queue rule (baselines/selection.py), the generator branch,
the plan policy, the snapshot key, the final shortlist and the launcher."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from omniqec_agentic.baselines import selection
from omniqec_agentic.config import (
    BASELINE_PLAN_POLICY_ID,
    BASELINE_PROVIDER,
    StructuralSearchConfig,
    VerificationConfig,
    formal_racing,
    search_method_of,
)

PACKAGE = Path(__file__).resolve().parents[1]


def _row(code_hash: str, *, canonical: str = "", k: int = 2, d: int = 6, w: int = 6, spec: bool = True) -> dict:
    return {
        "code_hash": code_hash,
        "canonical_hash": canonical or f"canon_{code_hash}",
        "family": "bb",
        "n": 30,
        "k": k,
        "d_search": d,
        "max_check_weight": w,
        "spec": {"family": "bb", "ell": 3, "m": 5, "A": [[0, 0]], "B": [[0, 1]]} if spec else None,
        "construct_source": "def construct(rng):\n    return {}\n" if spec else "",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _queue(run_dir: Path, turn: int, rows: list[dict]) -> None:
    _write_jsonl(run_dir / "turns" / f"turn_{turn:04d}" / "cycle" / "slow_queue.jsonl", rows)


def test_pool_keeps_legal_distinct_unnominated_codes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_jsonl(
        run / "structural_ledger.jsonl",
        [
            _row("a1"),
            _row("a1", d=9),  # a later line of the same hash: the earliest wins
            _row("b2", d=4),  # below the d_search floor
            _row("c3", w=9),  # over the weight gate
            _row("d4"),  # nominated on turn 0
            _row("e5", canonical="canon_d4"),  # twin of the nominated code
            _row("f6", canonical="canon_a1"),  # twin of a pool code seen earlier
            _row("g7", spec=False),  # no spec
            _row("h8"),  # measured (evidence hash)
            _row("ref", canonical="canon_ref"),  # the resident reference code
            _row("i9", k=1),  # k = 1 is legal for the baselines (runtime channel rule)
        ],
    )
    cycle = tmp_path / "cycle_ledger.jsonl"
    _write_jsonl(cycle, [_row("j0"), _row("a1")])
    pool, counts = selection.candidate_pool(
        run_dir=run,
        cycle_ledger_path=str(cycle),
        nominated=[_row("d4")],
        measured_hashes={"h8"},
        gate=8,
        reference=("ref", "canon_ref"),
    )
    assert [row["code_hash"] for row in pool] == ["a1", "i9", "j0"]
    assert pool[0]["d_search"] == 6
    assert counts == {
        "rows_seen": 11,
        "distinct_codes": 3,
        "no_spec": 1,
        "below_floor": 1,
        "over_weight_gate": 1,
        "nominated_or_measured": 2,
        "twin_of_nominated": 1,
        "reference_code": 1,
        "twin_in_pool": 1,
    }


def _context(run: Path, method: str, turn: int, **extra) -> dict:
    return {"method": method, "run_dir": str(run), "turn": turn, "seed_key": "test|memory|N60|bb|r01",
            "init_random_turns": 4, "blocks": 1, **extra}


def test_random_queue_is_seeded_bounded_and_excludes_earlier_nominations(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_jsonl(run / "structural_ledger.jsonl", [_row(f"h{i:02d}") for i in range(20)])
    _queue(run, 0, [_row("h03"), _row("h07")])
    picks = []
    for attempt in range(2):
        out = tmp_path / f"cycle{attempt}"
        path, composition = selection.build_baseline_slow_queue(
            context=_context(run, "random", 1), cycle_ledger_path=None, out_dir=out,
            measured_hashes=set(), gate=8, cap=6, resident_baseline=None,
        )
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        picks.append([row["code_hash"] for row in rows])
        assert composition["rule"] == "uniform" and composition["selected_count"] == 6
        assert composition["pool"]["distinct_codes"] == 18 and composition["previously_nominated"] == 2
        assert all(row["selection_channels"] == ["baseline_random"] for row in rows)
        assert json.loads((out / "slow_queue_composition.json").read_text())["selection_id"] == "uniform_pool_v1"
    assert picks[0] == picks[1]  # a retried turn draws the same batch
    assert not {"h03", "h07"} & set(picks[0]) and len(set(picks[0])) == 6


def test_bo_initial_turns_nominate_like_random_of_the_same_seed_key(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_jsonl(run / "structural_ledger.jsonl", [_row(f"h{i:02d}") for i in range(20)])
    chosen = {}
    for method in ("random", "bo"):
        path, composition = selection.build_baseline_slow_queue(
            context=_context(run, method, 2), cycle_ledger_path=None, out_dir=tmp_path / method,
            measured_hashes=set(), gate=8, cap=6, resident_baseline=None,
        )
        chosen[method] = [json.loads(line)["code_hash"] for line in Path(path).read_text().splitlines()]
        assert composition["rule"] == "uniform"
    assert chosen["random"] == chosen["bo"]


def test_short_pool_reports_unused_slots(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_jsonl(run / "structural_ledger.jsonl", [_row("only")])
    _, composition = selection.build_baseline_slow_queue(
        context=_context(run, "random", 0), cycle_ledger_path=None, out_dir=tmp_path / "c",
        measured_hashes=None, gate=8, cap=7, resident_baseline=None,
    )
    assert composition["selected_count"] == 1 and composition["unused_slots"] == 6


def test_measured_observations_read_per_basis_counts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    slow = run / "turns" / "turn_0000" / "cycle" / "slow"
    candidate = slow / "candidates" / "g0001_p0003_abcd1234.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps({
        "spec": {"family": "bb", "ell": 3, "m": 5, "A": [[0, 0]], "B": [[0, 1]]},
        "results": [{"rounds": 12, "basis": {"X": {"failures": 3, "shots": 2000}, "Z": {"failures": 0, "shots": 2000}}}],
    }))
    _write_jsonl(slow / "slow_results.jsonl", [
        # the recorded path no longer resolves (moved run directory): looked up by file name
        {"code_hash": "c1", "status": "ok", "per_logical_per_round": 1e-4, "k": 2, "is_baseline": False,
         "path": "/elsewhere/candidates/g0001_p0003_abcd1234.json"},
        {"code_hash": "c2", "status": "decode_infeasible", "per_logical_per_round": None, "k": 2, "path": ""},
    ])
    rows = selection.measured_observations(run, before_turn=1)
    assert len(rows) == 1
    row = rows[0]
    assert (row["code_hash"], row["k"], row["rounds"], row["fail_x"], row["shots_x"], row["fail_z"], row["shots_z"]) == (
        "c1", 2, 12, 3, 2000, 0, 2000
    )
    assert selection.measured_observations(run, before_turn=0) == []


def test_reference_identity_matches_the_resident_bb72() -> None:
    from omniqec_agentic.evaluation.baseline import select_resident_baseline

    assert selection.reference_identity(None) is None
    code_hash, canonical = selection.reference_identity(select_resident_baseline(144))
    assert len(code_hash) >= 12 and canonical


def test_search_method_of_and_policy_shape() -> None:
    from omniqec_agentic.orchestrator import make_policy

    assert search_method_of(None) == "llm" and search_method_of({"search_baseline": {"method": "random"}}) == "random"
    with pytest.raises(ValueError):
        search_method_of({"search_baseline": {"method": "llm"}})
    policy = make_policy(BASELINE_PROVIDER, generations=4, rng_seeds_per_program=1)
    plan = policy.plan({"protocol": {"proposal_count": 960, "promote_top": 6}, "turn": 5, "spec": {"families": ["bb"]}})
    assert plan["policy"] == BASELINE_PLAN_POLICY_ID and plan["provider"] == BASELINE_PROVIDER
    assert (plan["generations"], plan["proposal_count"], plan["rng_seeds_per_program"]) == (4, 960, 1)
    assert plan["promotions"] == []


def _snapshot(**kwargs):
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    return build_protocol_snapshot(
        verification=VerificationConfig(), racing=formal_racing(25_000), max_turns=30, proposal_count=960,
        generations_per_turn=4, slow_workers=8, n_budget=144, **kwargs,
    )


def test_snapshot_records_the_baseline_and_leaves_the_llm_snapshot_alone() -> None:
    baseline = {"method": "random", "families": ["bb"], "seed_key": "k"}
    llm = _snapshot(required_provider="openai", required_model="m")
    assert "search_baseline" not in llm
    snap = _snapshot(required_provider=BASELINE_PROVIDER, required_model="literal_uniform_v2", search_baseline=baseline)
    assert snap["search_baseline"] == baseline and search_method_of(snap) == "random"
    with pytest.raises(ValueError, match="requires provider"):
        _snapshot(required_provider="openai", search_baseline=baseline)
    with pytest.raises(ValueError, match="cannot carry an ablation"):
        _snapshot(required_provider=BASELINE_PROVIDER, search_baseline=baseline, ablation="no_slow", feedback_mode="fast")


def test_generator_branch_records_every_draw(tmp_path: Path) -> None:
    from omniqec_agentic.config import CodeFamily, default_run_config
    from omniqec_agentic.funsearch import _baseline_sampler_batch
    import dataclasses

    config = dataclasses.replace(
        default_run_config(run_id="t", n_budget=98, hardware="ion", out_dir=str(tmp_path)),
        allowed_families=(CodeFamily("bb"), CodeFamily("hgp")),
    )
    programs, raw = _baseline_sampler_batch(
        config=config, prompt_context={"baseline_sampler": {"seed_key": "k|r01", "turn": 3}}, turn=3, gen=2,
        generations=4, proposal_count=12, gen_dir=tmp_path,
    )
    proposals = [json.loads(line) for line in (tmp_path / "proposals.jsonl").read_text().splitlines()]
    assert len(programs) == len(proposals) == 12
    assert [row["program_id"] for row in proposals] == [f"g0002_p{i:04d}" for i in range(1, 13)]
    assert all(row["construct_source"] == source for row, source in zip(proposals, programs))
    assert raw[0]["engine_info"] == {"provider": BASELINE_PROVIDER, "model": "literal_uniform_v2", "seed_key": "k|r01"}
    assert raw[0]["status"] == "ok" and raw[0]["sampler"]["draws"] == 12
    with pytest.raises(RuntimeError, match="seed_key"):
        _baseline_sampler_batch(config=config, prompt_context={}, turn=0, gen=1, generations=4, proposal_count=2,
                                gen_dir=tmp_path)


def test_final_shortlist_of_a_baseline_is_the_measured_ranking(tmp_path: Path) -> None:
    from omniqec_agentic.workflow.graph import _final_entries

    hof = [dict(_row(f"m{i}"), per_logical_per_round=1e-4 * (i + 1)) for i in range(8)]
    elites = [_row(f"s{i}") for i in range(4)]
    state = {
        "protocol": {"search_baseline": {"method": "bo"}, "promotion_max_check_weight": 8},
        "run_dir": str(tmp_path),
        "structural_proxy_elites": elites,
        "hof_summary": hof,
        "unmeasurable_hashes": [],
    }
    entries, composition = _final_entries(state, StructuralSearchConfig())
    assert [row["code_hash"] for row in entries] == [f"m{i}" for i in range(6)]
    assert composition["search_method"] == "bo"
    assert (composition["structural_slots"], composition["runtime_slots"]) == (0, 6)


def test_launcher_is_valid_bash_and_freezes_the_final_protocol() -> None:
    script = PACKAGE / "hpc" / "nscc_baseline_grid.pbs"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for flag in (
        "--provider baseline_sampler", '--search-method "$METHOD"', "--rounds-policy fixed_Tc",
        "--promotion-max-check-weight 8", "--proposal-count 960",
        "--generations-per-turn 4", "--rng-seeds-per-program 1", "--strict-turns",
        'readonly NOISE_PROFILE="uniform_paper_v1"', 'readonly P_VALUE="0.005"', "readonly SHOT_SHARDS=16",
        "readonly FINAL_SHOT_SHARDS=128", "readonly DECODE_SHOT_BUDGET_S=8",
    ):
        assert flag in text, flag
    # the paper's early stop is run-campaign's default policy; --racing-max-errors selects an older one
    command = text[text.index("python -m omniqec_agentic.cli run-campaign"):]
    assert "--racing-max-errors" not in command and "OMNIQEC_RACE_PILOT_WINDOW=" not in text
    full = (PACKAGE / "hpc" / "nscc_claude_grid_v36.pbs").read_text()
    for value in ("readonly SHOT_SHARDS=16", "readonly FINAL_SHOT_SHARDS=128", "readonly DECODE_SHOT_BUDGET_S=8",
                  "readonly BUILD_TIMEOUT_S=3600", "readonly DECODE_SHARD_TIMEOUT_S=1800",
                  "readonly DISTANCE_TIMEOUT_S=1800", "readonly TIMEOUT_RETRIES=0", "readonly TRANSIENT_RETRIES=1"):
        assert value in full and value in text, value
