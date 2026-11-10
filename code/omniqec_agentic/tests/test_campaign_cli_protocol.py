from __future__ import annotations

import argparse

import pytest


def test_run_campaign_parser_defaults_to_sixteen_shards(monkeypatch, tmp_path) -> None:
    import omniqec_agentic.cli as cli

    captured: dict[str, object] = {}

    def fake_command(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "run_campaign_command", fake_command)

    assert cli.main(["run-campaign", "--budget", "144", "--out", str(tmp_path)]) == 0
    assert captured["shot_shards"] == 16
    # v3.4 parser defaults: 128 final shards, 8 s/shot decode budget, one
    # transient replay; --timeout-retries follows the config default (0).
    assert captured["final_shot_shards"] == 128
    assert captured["decode_shot_budget_s"] == pytest.approx(8.0)
    assert captured["transient_retries"] == 1
    assert captured["timeout_retries"] in (None, 0)


def test_run_campaign_cli_wires_shards_and_stage_timeouts(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)
    args = argparse.Namespace(
        strict_turns=False,
        provider="dry_run",
        budget=144,
        out=str(tmp_path),
        weight_limit=None,
        families=None,
        p=0.005,
        turns=1,
        proposal_count=8,
        generations_per_turn=1,
        rng_seeds_per_program=20,
        slow_workers=127,
        slow_shots=20_000,
        shot_shards=16,
        final_shot_shards=128,
        racing=False,
        racing_pilot=5000,
        racing_batch=5000,
        racing_confirm=20_000,
        racing_max=50_000,
        decoders=None,
        circuit_timeout_s=900.0,
        build_timeout_s=1200.0,
        decode_shard_timeout_s=0.0,
        distance_timeout_s=1800.0,
        timeout_retries=3,
        decode_shot_budget_s=4.0,
        transient_retries=3,
        final_shots=40_000,
        feedback_mode="both",
        resume=False,
    )

    assert cli.run_campaign_command(args) == 0
    assert captured["shot_shards"] == 16
    assert captured["final_shot_shards"] == 128
    verification = captured["verification"]
    assert verification.circuit_timeout_s == pytest.approx(900.0)
    # The legacy value initializes all three stages; explicit values win.
    assert verification.build_timeout_s == pytest.approx(1200.0)
    assert verification.decode_shard_timeout_s == pytest.approx(0.0)
    assert verification.distance_timeout_s == pytest.approx(1800.0)
    assert verification.timeout_retries == 3
    # v3.4 flags reach the verification config unchanged (non-strict run).
    assert verification.decode_shot_budget_s == pytest.approx(4.0)
    assert verification.transient_retries == 3


def test_strict_run_campaign_cli_rejects_nonformal_shards(tmp_path) -> None:
    from omniqec_agentic.cli import run_campaign_command

    args = argparse.Namespace(
        strict_turns=True,
        provider="openai",
        budget=144,
        out=str(tmp_path),
        weight_limit=None,
        families=None,
        p=0.005,
        turns=1,
        proposal_count=24,
        generations_per_turn=1,
        rng_seeds_per_program=20,
        slow_workers=127,
        slow_shots=25_000,
        shot_shards=8,
        final_shot_shards=128,
        racing=False,
        racing_pilot=5000,
        racing_batch=5000,
        racing_confirm=20_000,
        racing_max=50_000,
        decoders=None,
        circuit_timeout_s=None,
        build_timeout_s=None,
        decode_shard_timeout_s=None,
        distance_timeout_s=None,
        timeout_retries=None,
        decode_shot_budget_s=None,
        transient_retries=None,
        final_shots=40_000,
        feedback_mode="both",
        resume=False,
    )

    with pytest.raises(SystemExit, match="shot-shards at 16"):
        run_campaign_command(args)
