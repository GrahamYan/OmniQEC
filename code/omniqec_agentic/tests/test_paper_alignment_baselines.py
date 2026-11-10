"""Paper alignment of the comparison methods (2026-11-10; docs/paper_alignment_baselines.md).

The BO baseline (appendix D4), the Random configuration and the two feedback ablations (Table VIII, appendix D5,
Fig. 10 / 11) and the fixed-weight CNOT searches of Fig. 4 c (appendix D5), checked against the paper's text."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from omniqec_agentic.baselines import sampler as S
from omniqec_agentic.config import (
    ABLATION_FEEDBACK_MODE,
    ABLATION_IDS,
    BASELINE_GENERATIONS_PER_TURN,
    BASELINE_PROPOSALS_PER_GENERATION,
    BASELINE_PROVIDER,
    BASELINE_TURNS,
    CodeFamily,
    StructuralSearchConfig,
    default_run_config,
    fig3_cnot_structural_config,
)
from omniqec_agentic.construction import bb as bb_constructor
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.schemas import CandidateProgram
from omniqec_agentic.verification import fast as fast_mod

HPC = Path(__file__).resolve().parents[1] / "hpc"
BB72 = {"A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]], "ell": 6, "family": "bb", "m": 6}
BB_W4 = {"family": "bb", "ell": 6, "m": 6, "A": [[0, 0], [0, 1]], "B": [[0, 0], [1, 0]]}
SOURCE = "def construct(rng):\n    return %r\n"


@pytest.fixture(autouse=True)
def _keep_the_group_order_budget():
    """run_funsearch sets the catalogue's process-wide order cap; restore it for the tests that follow."""

    from omniqec_agentic.construction.groups.catalog import group_order_budget

    with group_order_budget(None):
        yield


def _stub_proxy(monkeypatch, distance: int = 6) -> None:
    def _search(hx, hz, *, seeds, iters, algorithm_id=None):
        return {
            "schema": "proxy_distance_search", "algorithm_id": "stub", "seeds": list(seeds),
            "iters_per_seed": int(iters), "sectors": ["X", "Z"], "per_seed": [],
            "d_x": int(distance), "d_z": int(distance), "d_search": int(distance), "available": True,
        }

    monkeypatch.setattr(fast_mod, "proxy_distance_search", _search)


# ------------------------------------------------------------ D4: the BO proposals
def test_bo_proposals_are_3840_bb_codes_with_4_to_8_terms_and_4lm_within_budget() -> None:
    """Appendix D4: 3,840 BB proposals per iteration; 4 l m <= N; |A| + |B| between four and eight, the terms from
    C_l x C_m. The LLM's BB constructor keeps its own 4..9 bound."""

    assert BASELINE_PROPOSALS_PER_GENERATION * BASELINE_GENERATIONS_PER_TURN == 3840
    assert S.PRIOR_CONSTANTS["bb_total_weight"] == [4, 8] and S.SAMPLER_ID == "literal_uniform_v2"
    assert (bb_constructor.BB_MIN_TOTAL_WEIGHT, bb_constructor.BB_MAX_TOTAL_WEIGHT) == (4, 9)
    for n_budget in (60, 144, 360):
        weights = Counter()
        for gen in range(1, 5):
            for draw in S.draw_generation(n_budget=n_budget, families=("bb",), seed_key="align", turn=0, gen=gen,
                                          count=240, generations_per_turn=4):
                spec = draw.spec
                assert spec is not None and spec["family"] == "bb"
                ell, m = spec["ell"], spec["m"]
                assert 4 * ell * m <= n_budget
                total = len(spec["A"]) + len(spec["B"])
                assert 4 <= total <= 8 and draw.skeleton["W"] == total
                assert len(spec["A"]) >= 1 and len(spec["B"]) >= 1
                for term in spec["A"] + spec["B"]:
                    assert 0 <= term[0] < ell and 0 <= term[1] < m
                weights[total] += 1
        assert set(weights) == {4, 5, 6, 7, 8}, (n_budget, weights)


def test_the_proposal_prior_reads_no_feedback() -> None:
    """Random / BO generation uses no LLM, no discovery memory and no measurement (D4, D5): the sampler's only
    inputs are the budget, the families, the frozen seed key and the position of the draw."""

    params = set(inspect.signature(S.draw_generation).parameters)
    assert params == {"n_budget", "families", "seed_key", "turn", "gen", "count", "generations_per_turn"}


# ------------------------------------------------------------ the shared fast loop
def test_baseline_proposals_go_through_the_shared_fast_loop_checks(monkeypatch, tmp_path) -> None:
    """Every drawn spec is checked by the structural check the LLM programs use (memory.archive.construct_stage ->
    structural_check_spec, called from ArchiveMemory.evaluate_batch); the slow-loop pool reads the structural
    ledger, which holds only the codes that passed it."""

    from omniqec_agentic.funsearch import run_funsearch
    from omniqec_agentic.memory import archive as archive_mod

    _stub_proxy(monkeypatch)
    checked: list[str] = []
    real = archive_mod.structural_check_spec

    def counting(program_id, spec, **kwargs):
        checked.append(str(program_id))
        return real(program_id, spec, **kwargs)

    monkeypatch.setattr(archive_mod, "structural_check_spec", counting)
    monkeypatch.setenv("OMNIQEC_FAST_WORKERS", "1")
    import dataclasses

    config = dataclasses.replace(
        default_run_config(run_id="align", n_budget=98, hardware="ion", out_dir=str(tmp_path / "fast")),
        allowed_families=(CodeFamily("bb"),),
        proposal_count=16,
    )
    run_funsearch(
        config, generations=1, provider=BASELINE_PROVIDER, use_seeds=False, rng_seeds_per_program=1,
        prompt_context_overlay={"baseline_sampler": {"seed_key": "align|r01", "turn": 0}}, turn=0,
    )
    proposals = [json.loads(line) for line in (tmp_path / "fast" / "generations" / "g0001" / "proposals.jsonl")
                 .read_text().splitlines()]
    assert len(proposals) == 16
    assert sorted(checked) == sorted(row["program_id"] for row in proposals if row["spec"] is not None)
    assert not any(item.startswith("seed_") for item in checked)
    ledger = tmp_path / "fast" / "structural_ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    assert {row["program_id"] for row in rows} <= set(checked)
    assert all(int(row["max_check_weight"]) <= 8 for row in rows)


def test_search_baselines_run_no_hand_written_seed_programs(monkeypatch, tmp_path) -> None:
    """The five seed programs initialise an LLM discovery run (D1); the baseline pools are formed from the random
    proposals only (D4, D5)."""

    from omniqec_agentic import campaign

    assert campaign.cycle_uses_seed_programs(True, "llm") is True
    assert campaign.cycle_uses_seed_programs(False, "llm") is False
    assert campaign.cycle_uses_seed_programs(True, "random") is False
    assert campaign.cycle_uses_seed_programs(True, "bo") is False

    seen: dict[str, bool] = {}

    class _Stop(Exception):
        pass

    def fake_run_funsearch(config, **kwargs):
        seen["use_seeds"] = bool(kwargs["use_seeds"])
        raise _Stop

    monkeypatch.setattr(campaign, "run_funsearch", fake_run_funsearch)
    config = default_run_config(run_id="align", n_budget=98, hardware="ion", out_dir=str(tmp_path))
    for method, expected in (("llm", True), ("random", False), ("bo", False)):
        with pytest.raises(_Stop):
            campaign.run_agentic_cycle(config, provider="dry_run", generations=1, search_method=method)
        assert seen["use_seeds"] is expected, method


# ------------------------------------------------------------ CLI of the baselines
def _cli_capture(monkeypatch):
    from omniqec_agentic import cli as cli_mod

    captured: dict = {}

    def fake_run_campaign(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"run_id": "campaign", "stop": {"reason": "test"}, "report": {}}

    monkeypatch.setattr("omniqec_agentic.workflow.run_campaign", fake_run_campaign)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **k: None, raising=False)
    return cli_mod, captured


def _baseline_argv(tmp_path, method="random", task="memory", families=("bb", "hgp", "lp_abelian", "lp_nonabelian")):
    return ["run-campaign", "--budget", "98", "--out", str(tmp_path), "--provider", BASELINE_PROVIDER,
            "--search-method", method, "--strict-turns", "--proposal-count", "960", "--generations-per-turn", "4",
            "--rng-seeds-per-program", "1", "--task", task, "--families", *families]


def test_baseline_cli_defaults_to_30_iterations_and_6_or_7_slots(monkeypatch, tmp_path) -> None:
    cli_mod, captured = _cli_capture(monkeypatch)
    assert BASELINE_TURNS == 30
    assert cli_mod.main(_baseline_argv(tmp_path)) == 0
    assert captured["max_turns"] == 30
    assert captured["verification"].structural.slow_candidate_cap == 6  # Memory: six per iteration
    assert captured["search_baseline"]["sampler_id"] == "literal_uniform_v2"
    assert captured["search_baseline"]["final_selection"] == "common_final_stage_top5"
    assert cli_mod.main(_baseline_argv(tmp_path, method="bo", task="cnot_dense", families=("bb",))) == 0
    assert captured["max_turns"] == 30
    assert captured["verification"].structural.slow_candidate_cap == 7  # CNOT: seven per iteration
    assert captured["search_baseline"]["init_random_turns"] == 4  # uniform in iterations 1-4, qLogNEI from 5
    assert captured["search_baseline"]["bo"]["bo_id"] == "bo_adapted_v2"
    assert cli_mod.main(_baseline_argv(tmp_path) + ["--turns", "12"]) == 0
    assert captured["max_turns"] == 12
    # the LLM search keeps its own default
    assert cli_mod.main(["run-campaign", "--budget", "98", "--out", str(tmp_path / "llm"), "--provider", "dry_run"]) == 0
    assert captured["max_turns"] == 3


def test_baseline_cli_refuses_the_v38_variant_generations(monkeypatch, tmp_path) -> None:
    cli_mod, _ = _cli_capture(monkeypatch)
    with pytest.raises(SystemExit, match="v3.7 structural profile"):
        cli_mod.main(_baseline_argv(tmp_path) + ["--protocol-profile", "v3_8"])


# ------------------------------------------------------------ BO model settings (D4)
def test_bo_settings_are_the_papers() -> None:
    from omniqec_agentic.baselines import bo

    cfg = bo.BO_CONFIG
    assert bo.BO_ID == "bo_adapted_v2" and cfg["init_turns"] == 4
    assert cfg["objective"]["posterior_samples"] == 4096 and "Beta(f + 1/2, s - f + 1/2)" in cfg["objective"]["posterior"]
    assert cfg["objective"]["target"] == "-log10(pLER)"
    assert cfg["embedding"]["d_model"] == 128 and cfg["embedding"]["num_layers"] == 4
    assert [view["name"] for view in cfg["embedding"]["views"]] == ["decode", "xlogic", "zlogic"]
    assert cfg["embedding"]["readout"].startswith("per-type mean pool")
    assert "nu=1.5" in cfg["gp"]["kernel"] and "ard_num_dims" in cfg["gp"]["kernel"] and cfg["gp"]["mean"].startswith("LinearMean")
    assert "learnable residual" in cfg["gp"]["observation_noise"]
    training = cfg["training"]
    assert training["optimizer"] == "adamw" and training["max_steps"] == 80 and training["warm_start"] is True
    assert training["early_stop"] == {"patience": 10, "min_delta": 1e-6}
    acquisition = cfg["acquisition"]
    assert acquisition["name"] == "qLogNoisyExpectedImprovement" and acquisition["mc_samples"] == 256
    assert "Sobol" in acquisition["sampler"] and "X_pending" in acquisition["batch"]
    # the trainer stops on the patience count alone (no relative-tolerance condition)
    source = inspect.getsource(bo._train)
    assert "if no_improve >= patience:" in source and "rel_improvement" not in source


# ------------------------------------------------------------ Table VIII
def test_table_viii_configurations() -> None:
    """Full / w/o slow / w/o slow & fast / Random: LLM generation, code-level feedback, circuit-level feedback."""

    from omniqec_agentic.orchestrator import make_policy

    def feedback(mode: str) -> tuple[bool, bool]:
        # funsearch._funsearch_user_prompt: show_fast / show_slow of the generator's feedback_mode
        return mode in ("both", "fast"), mode in ("both", "slow")

    table = {}
    for ablation in ABLATION_IDS:
        code_level, circuit_level = feedback(ABLATION_FEEDBACK_MODE[ablation])
        # under an ablation the campaign builds no slow queue (campaign.run_agentic_cycle), so nothing circuit-level
        # exists during the search
        circuit_level = circuit_level and ablation == "full"
        table[ablation] = (True, code_level, circuit_level)
    plan = make_policy(BASELINE_PROVIDER, generations=4, rng_seeds_per_program=1).plan(
        {"protocol": {"proposal_count": 960, "promote_top": 6}, "turn": 7, "spec": {"families": list(S.FAMILIES)}}
    )
    table["random"] = (plan["provider"] != BASELINE_PROVIDER, False, False)
    assert table == {
        "full": (True, True, True),
        "no_slow": (True, True, False),
        "no_feedback": (True, False, False),
        "random": (False, False, False),
    }
    # Random samples the four families in equal shares (D5)
    labels = S.turn_family_plan(families=S.FAMILIES, per_turn=3840, seed_key="k", turn=0)
    assert Counter(labels) == {family: 960 for family in S.FAMILIES}
    # and evaluates six (Memory) / seven (CNOT) codes per iteration, like the LLM searches
    assert StructuralSearchConfig().slow_candidate_cap == 6
    assert fig3_cnot_structural_config().slow_candidate_cap == 7


# ------------------------------------------------------------ Fig. 4 c (D5)
def test_exact_weight_holds_in_the_parallel_fast_loop(monkeypatch) -> None:
    """A code whose maximum check weight is not exactly w is rejected in the fast loop, with one worker or many
    (the worker jobs used to drop exact_check_weight)."""

    _stub_proxy(monkeypatch)
    programs = [
        CandidateProgram(program_id="w6", source=SOURCE % BB72, n_budget=144),
        CandidateProgram(program_id="w4", source=SOURCE % BB_W4, n_budget=144),
    ]
    for workers in (1, 2):
        memory = ArchiveMemory(turn=0, max_check_weight=6, exact_check_weight=6)
        memory.evaluate_batch(list(programs), n_budget=144, workers=workers)
        by_id = {attempt["program_id"]: attempt for attempt in memory.attempts}
        assert "check_weight_mismatch:4!=6" in json.dumps(by_id["w4"]), workers
        assert by_id["w4"].get("structural_eligible") is not True, workers
        assert by_id["w6"].get("structural_eligible") is True, workers
        assert {row["code_hash"] for row in memory.structural_proxy_elites} == {by_id["w6"]["code_hash"]}


def _preset(env: dict[str, str], echo: str) -> subprocess.CompletedProcess:
    text = (HPC / "nscc_claude_grid_v36.pbs").read_text(encoding="utf-8")
    start = text.index('PRESET="${PRESET:-}"')
    end = text.index('CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-5}"', start)
    script = "set -euo pipefail\n" + text[start:end] + f'\necho "{echo}"\n'
    return subprocess.run(["bash", "-c", script], env={"PATH": os.environ["PATH"], **env}, capture_output=True, text=True)


ALL = "TASK=$TASK RP=$ROUNDS_POLICY PROF=$PROTOCOL_PROFILE GATE=$PROMOTION_MAX_CHECK_WEIGHT ERR=$RACING_MAX_ERRORS TAG=$RUN_TAG"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_ablations_and_fixed_weight_searches_run_on_the_final_presets() -> None:
    """The w/o slow / w/o slow & fast configurations and the Fig. 4 c searches use the main searches' settings
    (R = 12, w <= 8 gate, the paper's early stop -- no --racing-max-errors --, same models) with one change each,
    in their own run directories."""

    common = ("RP=fixed_Tc", "ERR= ")
    out = _preset({"PRESET": "fig2_final", "ABLATION": "no_slow"}, ALL)
    assert out.returncode == 0, out.stderr
    assert all(item in out.stdout for item in common + ("TASK=memory", "PROF=v3_8", "GATE=8", "TAG=f2final_no_slow"))
    out = _preset({"PRESET": "fig3_final", "ABLATION": "no_feedback"}, ALL)
    assert out.returncode == 0, out.stderr
    assert all(item in out.stdout for item in common + ("TASK=cnot_dense", "GATE=8", "TAG=f3final_no_feedback"))
    for w in (4, 5, 6, 7):
        out = _preset({"PRESET": "fig3_final", "EXACT_CHECK_WEIGHT": str(w)}, ALL)
        assert out.returncode == 0, out.stderr
        assert all(item in out.stdout for item in common + ("TASK=cnot_dense", "PROF=v3_7", f"GATE={w}", f"TAG=f3final_w{w}"))
    assert "TAG=mine" in _preset({"PRESET": "fig3_final", "EXACT_CHECK_WEIGHT": "5", "RUN_TAG": "mine"}, ALL).stdout
    for env in ({"PRESET": "fig3_final", "EXACT_CHECK_WEIGHT": "9"},
                {"PRESET": "fig3_final", "EXACT_CHECK_WEIGHT": "5", "PROMOTION_MAX_CHECK_WEIGHT": "8"},
                {"PRESET": "fig3_final", "EXACT_CHECK_WEIGHT": "5", "ABLATION": "no_slow"},
                {"PRESET": "fig2_final", "EXACT_CHECK_WEIGHT": "5"},
                {"PRESET": "fig2_final", "ABLATION": "nope"},
                {"PRESET": "fig3_final", "MAX_CHECK_WEIGHT_HARD": "6"},
                {"PRESET": "fig2_final", "ABLATION": "no_slow", "ROUNDS_POLICY": "max_Tc_d"}):
        assert _preset(env, ALL).returncode == 2, env
    submitter = (HPC / "submit_claude_v36_grid.sh").read_text(encoding="utf-8")
    assert 'fig3_final) RUN_TAG="${RUN_TAG:-f3final$PRESET_VARIANT}" ;;' in submitter
    assert " ABLATION EXACT_CHECK_WEIGHT CAMPAIGN_ID " in submitter
    for name in ("nscc_claude_grid_v36.pbs", "submit_claude_v36_grid.sh"):
        subprocess.run(["bash", "-n", str(HPC / name)], check=True)


# Site-account shapes no launch entry may carry: a numeric PBS project id, a personal project, a cluster user
# directory, a tailnet host name. The project is always a parameter (PROJECT / PROJ, placeholder <PROJECT>).
_ACCOUNT_SHAPES = (
    r"-P\s+['\"]?[0-9]{5,}",
    r"\bpersonal[-_][A-Za-z$<]",
    r"/(?:scratch|home)/users?/[A-Za-z]",
    r"\.ts\.net\b",
)


def test_baseline_scripts_carry_no_site_account_and_the_papers_numbers() -> None:
    import re

    pbs = (HPC / "nscc_baseline_grid.pbs").read_text(encoding="utf-8")
    submit = (HPC / "submit_baseline_grid.sh").read_text(encoding="utf-8")
    for text in (pbs, submit):
        for shape in _ACCOUNT_SHAPES + (r"\bomniqec2\b",):
            assert not re.search(shape, text), shape
    assert 'TURNS="${TURNS:-30}"' in pbs and "4-8 terms" in pbs and "torch_geometric" not in pbs
    assert '-q "$QUEUE"' in submit and 'QUEUE="${QUEUE:-normal}"' in submit
    for name in ("nscc_baseline_grid.pbs", "submit_baseline_grid.sh"):
        subprocess.run(["bash", "-n", str(HPC / name)], check=True)


def test_llm_launch_entries_carry_no_site_account() -> None:
    import re

    for name in ("nscc_claude_grid_v36.pbs", "submit_claude_v36_grid.sh", "submit_gpt_v36_serial.sh"):
        text = (HPC / name).read_text(encoding="utf-8")
        for shape in _ACCOUNT_SHAPES:
            assert not re.search(shape, text), (name, shape)
    assert "#PBS -P <PROJECT>" in (HPC / "nscc_claude_grid_v36.pbs").read_text(encoding="utf-8")


@pytest.mark.parametrize("ablation, has_reference", [("no_slow", True), ("no_feedback", False)])
def test_no_feedback_variant_generations_have_no_parent_on_the_v38_profile(monkeypatch, tmp_path, ablation,
                                                                           has_reference) -> None:
    """PRESET=fig2_final runs the v3.8 profile; under w/o slow & fast its variant generations get no parent (not even
    the resident reference), as every other channel of that configuration."""

    from types import SimpleNamespace

    import omniqec_agentic.campaign as campaign
    import omniqec_agentic.workflow.graph as graph
    from omniqec_agentic.config import RacingConfig, VerificationConfig, structural_search_for_profile

    captured: dict = {}
    monkeypatch.setattr(campaign, "run_agentic_cycle",
                        lambda *a, **k: captured.update(k) or SimpleNamespace(status="ok", artifact_paths={}))
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    verification = VerificationConfig(structural=structural_search_for_profile("v3_8"))
    assert verification.structural.exploit_generations > 0
    monkeypatch.setattr(graph, "_verification_from_protocol", lambda *args, **kwargs: verification)
    state = {
        "run_id": "campaign_b144", "run_dir": str(tmp_path), "turn": 1,
        "plan": {"provider": "dry_run", "generations": 4, "proposal_count": 1, "rng_seeds_per_program": 1},
        "protocol": {"proposal_count": 1, "generations_per_turn": 4, "feedback_mode": ABLATION_FEEDBACK_MODE[ablation],
                     "racing": RacingConfig().to_dict(), "slow_workers": 1, "shot_shards": 1, "strict_turns": False,
                     "ablation": ablation},
        "spec": {"qubit_budget": 144, "families": ["bb"]},
        "structural_proxy_elites": [], "hof_summary": [], "unmeasurable_hashes": [],
        "baseline_summary": {}, "budget": {},
    }
    graph.execute_plan_node(state)
    parents = captured["prompt_context_overlay"]["exploit_parent_specs"]
    assert bool(parents) is has_reference
    assert all(row.get("is_baseline") for row in parents)  # never a measured code: nothing is measured
