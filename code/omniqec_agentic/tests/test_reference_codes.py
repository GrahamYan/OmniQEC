"""Reference codes (experiment variant, 2026-09-09): background codes from
another arm, frozen into the protocol and shown to both LLM roles.

Checks: the shipped DeepSeek b98 payload loads and renders; the protocol
snapshot carries the key only when asked; the generator prompt shows the block
through the overlay and still satisfies the prompt-neutrality rules; the CLI
wires the file into run_campaign; the launcher and both submitters pass it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from omniqec_agentic.config import (
    CodeFamily,
    HardwareConfig,
    HardwareTarget,
    RacingConfig,
    RunConfig,
    VerificationConfig,
)
from omniqec_agentic.funsearch import _build_funsearch_prompt, _merge_prompt_context_overlay
from omniqec_agentic.memory import ArchiveMemory
from omniqec_agentic.orchestrator import reference_codes as rc
from omniqec_agentic.workflow.state import build_protocol_snapshot

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_PATH = ROOT / "examples" / "reference_codes_deepseek_b98_v36.json"
HPC = ROOT / "hpc"

# The same rules test_prompt_neutrality.py applies to the generator prompt.
GOAL_RE = re.compile(
    r"(maximi[sz]e|increase|prefer|aim for|target)\s+(the\s+)?(absolute\s+)?"
    r"(larger\s+)?(k|d|distance|logical qubits)\b",
    re.I,
)
BANNED = ("PLPR decides", "larger k wins", "higher is better", "best first", "strongest verified",
          "do not clone", "or reduce n", "prefer these over Q", "KNOWN DEAD END", "beat an elite")


def _snapshot(**extra):
    return build_protocol_snapshot(
        verification=VerificationConfig(), racing=RacingConfig(), max_turns=30,
        proposal_count=24, generations_per_turn=1, slow_workers=1, n_budget=98, **extra,
    )


def test_shipped_deepseek_b98_payload_loads_and_rebuilds_the_named_codes() -> None:
    from omniqec_agentic.construction import build_matrices
    from omniqec_agentic.verification.structural import code_hash

    payload = rc.load_reference_codes(PAYLOAD_PATH)
    assert payload["schema"] == rc.SCHEMA
    assert [c["label"] for c in payload["codes"]] == ["ref-1", "ref-2", "ref-3", "ref-4", "ref-5"]
    for code in payload["codes"]:
        hx, hz = build_matrices(code["spec"])
        # The spec builds the very code the payload names (hash8 from the campaign CSV).
        assert code_hash(hx, hz)[:8] == code["code_hash8"], code["label"]
        n = hx.shape[1]
        assert code["code"].startswith(f"[[{n},"), code["label"]


def test_render_is_background_and_passes_the_neutrality_rules() -> None:
    payload = rc.load_reference_codes(PAYLOAD_PATH)
    text = rc.render_reference_codes(payload)
    assert text.startswith("BACKGROUND: REFERENCE CODES.")
    assert "ref-1" in text and '"family":"bb"' in text and "code_hash=2bddcc45" in text
    assert "background only" in text
    assert GOAL_RE.search(text) is None
    for phrase in BANNED:
        assert phrase not in text
    block = rc.reference_block(payload)
    assert block.startswith(f'<source id="{rc.SOURCE_ID}" kind="reference_codes" origin="protocol.reference_codes" hash="sha256:')
    assert block.endswith("</source>")


def test_validate_refuses_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="non-empty 'codes'"):
        rc.validate_reference_codes({"codes": []})
    with pytest.raises(ValueError, match="needs a 'spec'"):
        rc.validate_reference_codes({"codes": [{"label": "x"}]})
    with pytest.raises(ValueError, match="label"):
        rc.validate_reference_codes({"codes": [{"spec": {"family": "bb"}}]})
    with pytest.raises(ValueError, match="schema"):
        rc.validate_reference_codes({"schema": "v0", "codes": [{"label": "x", "spec": {"family": "bb"}}]})


def test_protocol_snapshot_carries_the_key_only_when_given() -> None:
    assert "reference_codes" not in _snapshot()
    payload = rc.load_reference_codes(PAYLOAD_PATH)
    with_ref = _snapshot(reference_codes=payload)
    assert with_ref["reference_codes"] == payload
    # Everything else is unchanged: the variant differs from a formal snapshot by this key only.
    base = _snapshot()
    assert {k: v for k, v in with_ref.items() if k != "reference_codes"} == base
    # A second build from the same file compares equal (resume-safe).
    assert _snapshot(reference_codes=rc.load_reference_codes(PAYLOAD_PATH)) == with_ref


def test_generator_prompt_shows_the_block_from_the_overlay_and_stays_neutral() -> None:
    payload = rc.load_reference_codes(PAYLOAD_PATH)
    merged = _merge_prompt_context_overlay({}, {"reference_codes": payload, "strategy_brief": "STRATEGY CONTEXT: bb 0.5"})
    assert merged["reference_codes"] == payload
    config = RunConfig(
        run_id="ref", n_budget=98, hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused", proposal_count=4, allowed_families=tuple(CodeFamily),
    )
    prompt = _build_funsearch_prompt(memory=ArchiveMemory(), config=config, gen=2, proposal_count=4, prompt_context=merged)
    user = prompt["user_prompt"]
    assert f'<source id="{rc.SOURCE_ID}" kind="reference_codes"' in user
    assert user.index('kind="strategy_brief"') < user.index('kind="reference_codes"')
    assert "ref-5" in user and '"group":"D12"' in user
    text = prompt["system_prompt"] + "\n" + user
    assert GOAL_RE.search(text) is None
    for phrase in BANNED:
        assert phrase not in text
    # Without the key nothing is rendered.
    plain = _build_funsearch_prompt(memory=ArchiveMemory(), config=config, gen=2, proposal_count=4,
                                    prompt_context=_merge_prompt_context_overlay({}, {}))
    assert "reference_codes" not in plain["user_prompt"]


def test_cli_flag_loads_the_file_into_run_campaign(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)
    base = dict(
        strict_turns=False, provider="dry_run", budget=98, out=str(tmp_path), weight_limit=None, families=None,
        p=0.005, turns=1, proposal_count=8, generations_per_turn=1, rng_seeds_per_program=20, slow_workers=1,
        slow_shots=20_000, shot_shards=16, final_shot_shards=128, racing=False, racing_pilot=5000,
        racing_batch=5000, racing_confirm=20_000, racing_max=50_000, decoders=None, circuit_timeout_s=900.0,
        build_timeout_s=1200.0, decode_shard_timeout_s=0.0, distance_timeout_s=1800.0, timeout_retries=0,
        decode_shot_budget_s=8.0, transient_retries=1, final_shots=200_000, feedback_mode="both", resume=False,
    )
    assert cli.run_campaign_command(argparse.Namespace(**base, reference_codes_json=None)) == 0
    assert captured["reference_codes"] is None
    assert cli.run_campaign_command(argparse.Namespace(**base, reference_codes_json=str(PAYLOAD_PATH))) == 0
    assert captured["reference_codes"]["codes"][0]["code_hash8"] == "2bddcc45"
    # The parser knows the flag.
    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_campaign_command", lambda args: seen.update(vars(args)) or 0)
    assert cli.main(["run-campaign", "--budget", "98", "--out", str(tmp_path),
                     "--reference-codes-json", str(PAYLOAD_PATH)]) == 0
    assert seen["reference_codes_json"] == str(PAYLOAD_PATH)


def test_launcher_and_submitters_pass_the_file_as_a_declared_variant() -> None:
    pbs = (HPC / "nscc_claude_grid_v36.pbs").read_text(encoding="utf-8")
    assert 'REFERENCE_CODES_JSON="${REFERENCE_CODES_JSON:-}"' in pbs
    assert 'REFERENCE_ARGS=(--reference-codes-json "$REFERENCE_CODES_JSON")' in pbs
    # 2026-09-29: the variant flags (FINAL_CAP_ARGS) sit between the reference file and --resume.
    assert '"${REFERENCE_ARGS[@]}" \\\n      "${FINAL_CAP_ARGS[@]}" \\\n      "${RESUME_ARGS[@]}"' in pbs
    assert 'if (( VALIDATE_FORMAL )) || [[ "$RUN_TAG" == "v37" ]]; then' in pbs
    for name in ("submit_claude_v36_grid.sh", "submit_gpt_v36_serial.sh"):
        text = (HPC / name).read_text(encoding="utf-8")
        assert "OMNIQEC_SHARD_SPLIT REFERENCE_CODES_JSON; do" in text, name
    validator = (HPC / "validate_fig2_campaign.py").read_text(encoding="utf-8")
    assert '"reference_codes" not in protocol' in validator
