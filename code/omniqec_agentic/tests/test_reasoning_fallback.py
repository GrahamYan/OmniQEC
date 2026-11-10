"""Final protocols (2026-11-06): the reasoning-saturation rule and the fig2_final / fig3_final presets and profiles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from omniqec_agentic import reasoning_fallback as rf

PBS = Path(__file__).resolve().parents[1] / "hpc" / "nscc_claude_grid_v36.pbs"


def _row(*, out: int = 64000, cap: int = 64000, parsed: int = 0, response: str = "", status: str = "ok",
         model: str = "claude-opus-5") -> dict:
    return {"status": status, "proposal_count_parsed": parsed, "response": response,
            "engine_info": {"provider": "openai", "model": model, "max_tokens": cap, "usage": {"output_tokens": out}}}


def _run_dir(tmp_path: Path, name: str = "fig2_claudesub_b180_f2final", *, model: str = "claude-opus-5",
             saturated: bool = True) -> Path:
    run = tmp_path / "runs" / name
    gen = run / "turns" / "turn_0000" / "cycle" / "fast" / "generations" / "g0001"
    gen.mkdir(parents=True)
    rows = [_row(model=model)] * 3 if saturated else [_row(out=41000, parsed=24, response="```python ...```")]
    (gen / "llm_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / "protocol_snapshot.json").write_text(json.dumps({"required_model": model}), encoding="utf-8")
    return run


# ---------------------------------------------------------------- detection
def test_model_names() -> None:
    assert rf.effort_suffix("claude-opus-5") is None
    assert rf.effort_suffix("claude-opus-5(medium)") == "medium"
    assert rf.base_model("claude-opus-5(medium)") == "claude-opus-5"
    assert rf.fallback_model("claude-opus-5") == "claude-opus-5(medium)"
    assert rf.effort_suffix("deepseek/deepseek-v4-pro") is None


def test_a_saturated_attempt_is_an_empty_answer_at_the_output_cap() -> None:
    assert rf.saturated_attempt(_row())
    assert not rf.saturated_attempt(_row(out=63999))
    assert not rf.saturated_attempt(_row(parsed=3))
    assert not rf.saturated_attempt(_row(response="```python\ndef construct(rng): ..."))
    assert not rf.saturated_attempt(_row(status="generation_error"))
    assert rf.saturated_generation([_row()] * 3, attempts=3)
    assert not rf.saturated_generation([_row()] * 2, attempts=3)
    assert not rf.saturated_generation([_row(), _row(), _row(out=41000)], attempts=3)
    message = rf.saturation_message(turn=0, gen=1, rows=[_row()] * 3)
    assert message.startswith(rf.SATURATION_MARKER) and "3 of 3" in message and "64000" in message


def _funsearch_once(tmp_path: Path, monkeypatch, *, turn: int, output_tokens: int) -> str:
    from omniqec_agentic import funsearch as fs
    from omniqec_agentic.config import CodeFamily, HardwareConfig, HardwareTarget, RunConfig
    from omniqec_agentic.construction.groups.catalog import get_group_order_budget, set_group_order_budget

    def _empty(**kwargs):
        return "", {"provider": "openai", "model": "claude-opus-5", "max_tokens": 64000,
                    "usage": {"output_tokens": output_tokens}}

    monkeypatch.setattr(fs, "_complete_funsearch_prompt", _empty)
    previous = get_group_order_budget()
    config = RunConfig(run_id="t", n_budget=60, hardware=HardwareConfig(target=HardwareTarget.ION),
                       out_dir=str(tmp_path / f"t{turn}_{output_tokens}"), proposal_count=2,
                       allowed_families=tuple(CodeFamily))
    try:
        with pytest.raises(RuntimeError) as info:
            fs.run_funsearch(config, generations=1, provider="openai", use_seeds=False,
                             require_full_llm_batch=True, turn=turn)
    finally:
        set_group_order_budget(previous)
    return str(info.value)


def test_the_generator_names_a_saturated_turn_zero(tmp_path, monkeypatch) -> None:
    message = _funsearch_once(tmp_path, monkeypatch, turn=0, output_tokens=64000)
    assert message.startswith("strict campaign requires a complete LLM proposal batch")
    assert rf.SATURATION_MARKER in message and "3 of 3" in message
    # a later turn, or an empty answer that did not use the whole budget, is an ordinary strict abort
    assert rf.SATURATION_MARKER not in _funsearch_once(tmp_path, monkeypatch, turn=1, output_tokens=64000)
    assert rf.SATURATION_MARKER not in _funsearch_once(tmp_path, monkeypatch, turn=0, output_tokens=1200)


# ---------------------------------------------------------------- flag, record, retire
def test_raise_flag_needs_evidence_and_the_claude_arm(tmp_path) -> None:
    flag = tmp_path / "runs" / "reasoning_fallback_claudesub_f2final.json"
    run = _run_dir(tmp_path)
    with pytest.raises(SystemExit) as info:
        rf.raise_flag(flag, arm="gpt", run_tag="f2final", model="gpt-5.6-sol", point="b180", run_dir=run, job="1")
    assert info.value.code == 3
    with pytest.raises(SystemExit) as info:
        rf.raise_flag(flag, arm="claude", run_tag="f2final", model="claude-opus-5(medium)", point="b180",
                      run_dir=run, job="1")
    assert info.value.code == 3
    calm = _run_dir(tmp_path, "fig2_claudesub_b98_f2final", saturated=False)
    with pytest.raises(SystemExit) as info:
        rf.raise_flag(flag, arm="claude", run_tag="f2final", model="claude-opus-5", point="b98", run_dir=calm, job="1")
    assert info.value.code == 4
    assert not flag.exists()
    payload = rf.raise_flag(flag, arm="claude", run_tag="f2final", model="claude-opus-5", point="b180", run_dir=run,
                            job="1", log_line=rf.SATURATION_MARKER + ": turn 0 generation 1")
    assert payload["from_model"] == "claude-opus-5" and payload["to_model"] == "claude-opus-5(medium)"
    assert payload["evidence"][0]["output_tokens"] == [64000] * 3
    # a second trigger keeps the first flag
    other = _run_dir(tmp_path, "fig2_claudesub_b60_f2final")
    again = rf.raise_flag(flag, arm="claude", run_tag="f2final", model="claude-opus-5", point="b60", run_dir=other, job="2")
    assert again["trigger_point"] == "b180" == rf.read_flag(flag)["trigger_point"]


def test_switch_retire_and_record(tmp_path) -> None:
    flag = tmp_path / "runs" / "reasoning_fallback_claudesub_f2final.json"
    run = _run_dir(tmp_path)
    assert rf.current_model(flag, "claude-opus-5") == "claude-opus-5"
    assert not rf.is_stale(run, flag)
    rf.raise_flag(flag, arm="claude", run_tag="f2final", model="claude-opus-5", point="b180", run_dir=run, job="1")
    assert rf.current_model(flag, "claude-opus-5") == "claude-opus-5(medium)"
    assert rf.is_stale(run, flag)
    dest = rf.retire(run, tmp_path / "runs")
    assert dest.parent.name == rf.RETIRED_DIRNAME and (dest / "protocol_snapshot.json").is_file() and not run.exists()
    run.mkdir()
    assert rf.write_record(run, flag_path=flag, model="claude-opus-5", retired=None) is None
    record = rf.write_record(run, flag_path=flag, model="claude-opus-5(medium)", retired=str(dest))
    assert record["flag"] == rf.read_flag(flag) and record["replaces"] == str(dest)
    assert json.loads((run / rf.RECORD_NAME).read_text())["model"] == "claude-opus-5(medium)"
    (run / "protocol_snapshot.json").write_text(json.dumps({"required_model": "claude-opus-5(medium)"}))
    assert not rf.is_stale(run, flag)
    with pytest.raises(ValueError):
        rf.write_record(tmp_path / "x", flag_path=tmp_path / "none.json", model="claude-opus-5(medium)")


def test_a_default_run_stops_at_its_next_turn_once_the_arm_switched(tmp_path, monkeypatch) -> None:
    flag = tmp_path / "runs" / "reasoning_fallback_claudesub_f3final.json"
    monkeypatch.setenv(rf.FLAG_ENV, str(flag))
    rf.check_fallback_request({"required_model": "claude-opus-5"})          # no flag yet
    rf.raise_flag(flag, arm="claude", run_tag="f3final", model="claude-opus-5", point="b144",
                  run_dir=_run_dir(tmp_path, "fig2_claudesub_b144_f3final"), job="1")
    with pytest.raises(RuntimeError, match=rf.FALLBACK_REQUEST_MARKER):
        rf.check_fallback_request({"required_model": "claude-opus-5"})
    rf.check_fallback_request({"required_model": "claude-opus-5(medium)"})  # already switched
    monkeypatch.delenv(rf.FLAG_ENV)
    rf.check_fallback_request({"required_model": "claude-opus-5"})          # not a final-protocol launch


def test_the_cli_drives_the_launcher_steps(tmp_path) -> None:
    flag = tmp_path / "runs" / "reasoning_fallback_claudesub_f2final.json"
    run = _run_dir(tmp_path)
    assert rf.main(["stale", "--flag", str(flag), "--run-dir", str(run)]) == 1
    assert rf.main(["raise", "--flag", str(flag), "--arm", "claude", "--run-tag", "f2final", "--model", "claude-opus-5",
                    "--point", "b180", "--run-dir", str(run), "--job", "7"]) == 0
    assert rf.main(["stale", "--flag", str(flag), "--run-dir", str(run)]) == 0
    with pytest.raises(SystemExit) as info:
        rf.main(["raise", "--flag", str(flag), "--arm", "deepseek", "--run-tag", "f2final",
                 "--model", "deepseek/deepseek-v4-pro", "--point", "b180", "--run-dir", str(run), "--job", "7"])
    assert info.value.code == 3


# ---------------------------------------------------------------- validator profiles
@pytest.fixture()
def validator():
    from omniqec_agentic.hpc import validate_fig2_campaign as v

    yield v
    v.configure_profile("v3_7")


def test_final_profiles(validator) -> None:
    v = validator
    v.configure_profile("fig2_final")
    assert (v.FORMAL_TASK, v.FORMAL_ROUNDS_POLICY, v.FINAL_PROTOCOL) == ("memory", "fixed_Tc", True)
    assert v.PROTOCOL_PROFILE == "telescoping_structural_v3_8"
    assert v._expected_rounds(12, {"d_search": 16}) == 12
    v.configure_profile("fig3_final")
    assert (v.FORMAL_TASK, v.FORMAL_ROUNDS_POLICY, v.FINAL_PROTOCOL) == ("cnot_dense", "fixed_Tc", True)
    assert v.PROTOCOL_PROFILE == "telescoping_structural_v3_7_cnot_v2" and "lightest_endpoint" in v.FORMAL_STRUCTURAL_SLOT_REASONS
    assert v.STRUCTURAL_EXPECTED["structural_slow_slots"] == 3
    v.configure_profile("v3_8")
    assert (v.FORMAL_ROUNDS_POLICY, v.FINAL_PROTOCOL) == ("max_Tc_d", False)
    assert v._expected_rounds(12, {"d_search": 16}) == 16


def test_final_protocol_keys(validator) -> None:
    from omniqec_agentic.config import budget_policy, formal_racing, formal_racing_for_max_errors, paper_racing
    from omniqec_agentic.evaluation.final_stage import final_stage_protocol

    v = validator
    v.configure_profile("fig2_final")
    policy = budget_policy(180)
    # 2026-11-10 (paper alignment): the final profiles require the paper's early stop and final stage.
    stage = final_stage_protocol(task="memory", n_budget=180, confirm_shots=25000, final_shot_shards=128)
    good = {"promotion_max_check_weight": 8, "racing": paper_racing(25000).to_dict(), "n_budget": 180, "final_stage": stage}
    v._validate_final_protocol_keys(good, policy)
    for bad in ({**good, "promotion_max_check_weight": 6},
                {**good, "racing": formal_racing(25000).to_dict()},
                {**good, "racing": formal_racing_for_max_errors(25000, 30).to_dict()},
                {**good, "final_stage": {**stage, "cumulative_cap": 25000}},
                {k: val for k, val in good.items() if k != "final_stage"},
                {**good, "ablation": "no_slow"},
                {**good, "runtime_overrides": [{"turn": 3, "to": {"promotion_max_check_weight": 8}}]}):
        with pytest.raises(ValueError):
            v._validate_final_protocol_keys(bad, policy)


def test_fixed_tc_checks_the_circuit_rounds(validator, tmp_path) -> None:
    v = validator
    v.configure_profile("fig2_final")

    def cand(t: int) -> str:
        return json.dumps({"results": [{"basis": {b: {"memory_experiment": {"rounds": t}} for b in ("X", "Z")}}]})

    run = tmp_path / "run"
    slow = run / "turns" / "turn_0003" / "cycle" / "slow" / "candidates"
    slow.mkdir(parents=True)
    (slow / "g0001_p0001.json").write_text(cand(12))
    assert v._validate_circuit_rounds(run) == 2
    final = run / "final_slow" / "T12" / "candidates"
    final.mkdir(parents=True)
    (final / "g0002_p0003.json").write_text(cand(16))   # a d = 16 code built at T = d (the 2026-10-27 bug)
    with pytest.raises(ValueError, match="T=16"):
        v._validate_circuit_rounds(run)


def test_reasoning_rule_in_the_validator(validator, tmp_path) -> None:
    v = validator
    v.configure_profile("fig3_final")
    flag = tmp_path / "runs" / "reasoning_fallback_claudesub_f3final.json"
    default_run = tmp_path / "runs" / "fig2_claudesub_b98_f3final"
    default_run.mkdir(parents=True)
    v._validate_reasoning_rule(default_run, model="claude-opus-5", provider="openai", flag_path=flag)
    v._validate_reasoning_rule(default_run, model="deepseek/deepseek-v4-pro", provider="openrouter", flag_path=flag)
    with pytest.raises(ValueError):
        v._validate_reasoning_rule(default_run, model="claude-opus-5-5", provider="openai", flag_path=flag)
    trigger = _run_dir(tmp_path, "fig2_claudesub_b144_f3final")
    rf.raise_flag(flag, arm="claude", run_tag="f3final", model="claude-opus-5", point="b144", run_dir=trigger, job="1")
    with pytest.raises(ValueError, match="default level"):
        v._validate_reasoning_rule(default_run, model="claude-opus-5", provider="openai", flag_path=flag)
    medium_run = tmp_path / "runs" / "fig2_claudesub_b60_f3final"
    medium_run.mkdir()
    with pytest.raises(ValueError, match=rf.RECORD_NAME):
        v._validate_reasoning_rule(medium_run, model="claude-opus-5(medium)", provider="openai", flag_path=flag)
    rf.write_record(medium_run, flag_path=flag, model="claude-opus-5(medium)")
    v._validate_reasoning_rule(medium_run, model="claude-opus-5(medium)", provider="openai", flag_path=flag)
    with pytest.raises(ValueError):
        v._validate_reasoning_rule(medium_run, model="claude-opus-5(high)", provider="openai", flag_path=flag)


# ---------------------------------------------------------------- launcher presets
def _preset(env: dict[str, str], echo: str) -> subprocess.CompletedProcess:
    text = PBS.read_text(encoding="utf-8")
    start = text.index('PRESET="${PRESET:-}"')
    end = text.index('CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-5}"', start)
    script = "set -euo pipefail\n" + text[start:end] + f'\necho "{echo}"\n'
    return subprocess.run(["bash", "-c", script], env={"PATH": os.environ["PATH"], **env}, capture_output=True, text=True)


ALL = ("TASK=$TASK RP=$ROUNDS_POLICY PROF=$PROTOCOL_PROFILE GATE=$PROMOTION_MAX_CHECK_WEIGHT ERR=$RACING_MAX_ERRORS "
       "CLAUDE=$CLAUDE_MODEL GPT=$GPT_MODEL DS=$DEEPSEEK_MODEL TAG=$RUN_TAG VP=$VALIDATOR_PROFILE")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_launcher_presets() -> None:
    # 2026-11-10: the presets run the paper's early stop (run-campaign's default): no RACING_MAX_ERRORS.
    common = ("RP=fixed_Tc", "GATE=8", "ERR= ", "CLAUDE=claude-opus-5 ", "GPT=gpt-5.6-sol", "DS=deepseek/deepseek-v4-pro")
    out = _preset({"PRESET": "fig2_final"}, ALL)
    assert out.returncode == 0, out.stderr
    assert all(item in out.stdout for item in common + ("TASK=memory", "PROF=v3_8", "TAG=f2final", "VP=fig2_final"))
    out = _preset({"PRESET": "fig3_final"}, ALL)
    assert out.returncode == 0, out.stderr
    assert all(item in out.stdout for item in common + ("TASK=cnot_dense", "PROF=v3_7", "TAG=f3final", "VP=fig3_final"))
    assert _preset({}, "VP=$VALIDATOR_PROFILE").stdout.strip() == "VP=v3_7"
    for env in ({"PRESET": "fig3_final", "CLAUDE_MODEL": "claude-opus-5(medium)"},   # the rule sets the level
                {"PRESET": "fig2_final", "ROUNDS_POLICY": "max_Tc_d"},
                {"PRESET": "fig2_final", "PROMOTION_MAX_CHECK_WEIGHT": "6"},
                {"PRESET": "fig2_final", "RACING_MAX_ERRORS": "30"},
                {"PRESET": "fig2_final", "MAX_CHECK_WEIGHT_HARD": "6"},
                {"PRESET": "fig2_final", "EXACT_CHECK_WEIGHT": "5"},   # the fixed-weight search is CNOT only
                {"PRESET": "fig2_final", "RUN_TAG": "v37"},
                {"PRESET": "fig4"}):
        assert _preset(env, ALL).returncode == 2, env


def test_submitters_forward_the_preset() -> None:
    for name in ("submit_claude_v36_grid.sh", "submit_gpt_v36_serial.sh"):
        text = (PBS.parent / name).read_text(encoding="utf-8")
        # the Claude submitter appends the ablation / exact-weight suffix (2026-11-10 paper alignment)
        assert 'fig2_final) RUN_TAG="${RUN_TAG:-f2final' in text and 'fig3_final) RUN_TAG="${RUN_TAG:-f3final' in text
        assert "for optional_name in PRESET OUT_ROOT " in text, name
        assert 'if (( VALIDATE_FORMAL )) && [[ -n "$PRESET" ]]' in text, name


def test_launcher_wires_the_rule() -> None:
    text = PBS.read_text(encoding="utf-8")
    rule = text.index('if [[ -n "$REASONING_FLAG" ]] && grep -qE "reasoning saturated at the start of the run')
    assert rule < text.index('if looks_like_quota "$ATTEMPT_LOG"', rule) < text.index('if looks_like_strict_abort "$ATTEMPT_LOG"', rule)
    assert text.index("python -m omniqec_agentic.cli check-slow-deps") < text.index('REASONING_FLAG="$OUT_ROOT/reasoning_fallback_${ARM_PREFIX}_${RUN_TAG}.json"')
    assert 'export OMNIQEC_REASONING_FALLBACK_FLAG="$REASONING_FLAG"' in text
    assert '--profile "$VALIDATOR_PROFILE" "${REASONING_VALIDATOR_ARGS[@]}"' in text
    assert 'python -m omniqec_agentic.reasoning_fallback stale --flag "$REASONING_FLAG" --run-dir "$OUT_DIR" && RETIRE_PENDING=1' in text
    assert rf.SATURATION_MARKER in text and rf.FALLBACK_REQUEST_MARKER in text
