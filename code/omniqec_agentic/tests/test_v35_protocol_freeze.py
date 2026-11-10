"""The noise-model protocol field (introduced by bposd_structural_v3_5,
2026-09-02; carried unchanged into telescoping_structural_v3_7, 2026-09-03 --
the profile / decoder assertions below are the v3.6 ones).

v3.5 returns the noise model to the OmniQEC paper SI C uniform circuit-level
depolarizing model (uniform_paper_v1: depolarizing after every reset and CX, a
flip before every measurement, NO idle noise) at p=0.005; the SEC stays LRC.
The v3.4 idle-noise model (uniform_legacy_v1) stays selectable for history but
is a different measurement: its snapshot is retagged, refused under
strict_turns, refused on --resume and refused by the standalone validator.

Contract §5 items covered here: (c) CLI default --noise-profile, (d) snapshot
freeze / strict refusal / resume mismatch, (e) validator constants and checks,
(f) RACING_ANCHOR_P. The lowering itself is test_v35_noise_profile.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from omniqec_agentic.config import RacingConfig, StructuralSearchConfig, VerificationConfig
from omniqec_agentic.evaluation.common import noise_result_fields
from omniqec_agentic.evaluation.sampling import RACING_ANCHOR_P, anchor_p_for_profile
from omniqec_agentic.hpc import validate_fig2_campaign as validator
from omniqec_agentic.hpc.validate_fig2_campaign import validate
from omniqec_agentic.verification.noise import (
    UNIFORM_PROFILES,
    NoiseProfile,
    NoiseSweepConfig,
)
from omniqec_agentic.workflow import state as state_mod
from omniqec_agentic.workflow.checkpoint import validate_resume_protocol
from omniqec_agentic.workflow.state import build_protocol_snapshot

from test_campaign_strict_turns import (
    FINAL_BASE_SEED,
    FINAL_SHOTS,
    NOISE_POINT,
    NOISE_SWEEP,
    TELESCOPING,
    TELESCOPING_BUILD,
    _rankings,
    _slow_run_config,
    _strict_report,
    _write_json,
    _write_validated_run,
)

PAPER = NoiseProfile.UNIFORM_PAPER.value
LEGACY = NoiseProfile.UNIFORM_LEGACY.value
LEGACY_POINT = {"profile": LEGACY, "axis": "p", "value": 0.005}
LEGACY_SWEEP = {**NOISE_SWEEP, "profile": LEGACY}


# ---------------------------------------------------------------------------
# (c) CLI default --noise-profile
# ---------------------------------------------------------------------------


def _capture_command(monkeypatch, name: str) -> dict:
    import omniqec_agentic.cli as cli

    captured: dict[str, object] = {}

    def fake_command(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        captured["_args"] = args
        return 0

    monkeypatch.setattr(cli, name, fake_command)
    return captured


def test_run_campaign_parser_defaults_to_the_paper_noise_profile(monkeypatch, tmp_path) -> None:
    import omniqec_agentic.cli as cli

    captured = _capture_command(monkeypatch, "run_campaign_command")
    assert cli.main(["run-campaign", "--budget", "144", "--out", str(tmp_path)]) == 0
    assert captured["noise_profile"] == PAPER
    assert captured["p"] == pytest.approx(0.005)

    # The retired v3.4 model stays selectable (history / A-B), SI/Helios do not:
    # run-campaign is a fixed-p uniform campaign.
    captured.clear()
    assert cli.main([
        "run-campaign", "--budget", "144", "--out", str(tmp_path),
        "--noise-profile", LEGACY, "--p", "0.003",
    ]) == 0
    assert captured["noise_profile"] == LEGACY
    assert captured["p"] == pytest.approx(0.003)
    with pytest.raises(SystemExit):
        cli.main([
            "run-campaign", "--budget", "144", "--out", str(tmp_path),
            "--noise-profile", NoiseProfile.SI1000.value,
        ])


@pytest.mark.parametrize(
    ("command", "handler", "multi"),
    [
        ("slow-verify-seed", "slow_verify_seed_command", True),
        ("build-dem-smoke", "build_dem_smoke_command", False),
    ],
)
def test_shared_noise_parser_defaults_to_the_paper_profile(
    monkeypatch, command: str, handler: str, multi: bool
) -> None:
    import omniqec_agentic.cli as cli

    captured = _capture_command(monkeypatch, handler)
    assert cli.main([command]) == 0
    assert captured["noise_profile"] == PAPER
    noise = cli._noise_from_args(captured["_args"], multi=multi)
    assert noise == NoiseSweepConfig.uniform((0.005,))
    assert noise.profile == NoiseProfile.UNIFORM_PAPER

    # --p / --p-values are aliases of EITHER uniform profile ...
    p_flag = ["--p-values", "0.003"] if multi else ["--p", "0.003"]
    captured.clear()
    assert cli.main([command, "--noise-profile", LEGACY, *p_flag]) == 0
    legacy = cli._noise_from_args(captured["_args"], multi=multi)
    assert legacy == NoiseSweepConfig.uniform_legacy((0.003,))
    captured.clear()
    assert cli.main([command, *p_flag]) == 0
    assert cli._noise_from_args(captured["_args"], multi=multi) == NoiseSweepConfig.uniform(
        (0.003,)
    )
    # ... and of nothing else: the SI/Helios mixing error stays.
    captured.clear()
    assert cli.main([command, "--noise-profile", NoiseProfile.SI1000.value, *p_flag]) == 0
    with pytest.raises(ValueError, match="cannot be mixed with SI/Helios"):
        cli._noise_from_args(captured["_args"], multi=multi)


def _campaign_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    args = dict(
        strict_turns=True,
        provider="openai",
        budget=144,
        out=str(tmp_path),
        weight_limit=None,
        families=None,
        p=0.005,
        noise_profile=PAPER,
        turns=1,
        proposal_count=24,
        generations_per_turn=1,
        rng_seeds_per_program=20,
        slow_workers=127,
        slow_shots=25_000,
        shot_shards=16,
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
        final_shots=25_000,
        feedback_mode="both",
        resume=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_run_campaign_command_plumbs_the_noise_profile_into_verification(
    monkeypatch, tmp_path
) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)

    assert cli.run_campaign_command(_campaign_args(tmp_path, strict_turns=False, provider="dry_run")) == 0
    verification = captured["verification"]
    assert verification.noise == NoiseSweepConfig.uniform((0.005,))
    assert verification.p_values == (0.005,)

    # A v3.4-era caller that never learned the flag gets the current protocol.
    captured.clear()
    args = _campaign_args(tmp_path, strict_turns=False, provider="dry_run")
    del args.noise_profile
    assert cli.run_campaign_command(args) == 0
    assert captured["verification"].noise.profile == NoiseProfile.UNIFORM_PAPER

    captured.clear()
    assert cli.run_campaign_command(
        _campaign_args(tmp_path, strict_turns=False, provider="dry_run", noise_profile=LEGACY, p=0.003)
    ) == 0
    verification = captured["verification"]
    assert verification.noise == NoiseSweepConfig.uniform_legacy((0.003,))
    assert verification.p_values == (0.003,)
    # The uniform family keeps its fast-loop proxy calibration under both ids.
    assert verification.slope_offset in (None, 0.958)


def test_strict_cli_freezes_the_paper_noise_model_at_launch(tmp_path) -> None:
    from omniqec_agentic.cli import run_campaign_command

    with pytest.raises(SystemExit, match="telescoping_structural_v3_7.*uniform_paper_v1"):
        run_campaign_command(_campaign_args(tmp_path, noise_profile=LEGACY))
    with pytest.raises(SystemExit, match=r"freezes --p at 0\.005"):
        run_campaign_command(_campaign_args(tmp_path, p=0.003))
    with pytest.raises(SystemExit, match="fixed-p uniform campaign"):
        run_campaign_command(
            _campaign_args(tmp_path, strict_turns=False, noise_profile=NoiseProfile.SI1000.value)
        )


# ---------------------------------------------------------------------------
# (d) snapshot freeze, strict refusal, resume mismatch
# ---------------------------------------------------------------------------


def _snapshot(**overrides) -> dict:
    kwargs = dict(
        verification=VerificationConfig(),
        racing=RacingConfig(),
        max_turns=1,
        proposal_count=8,
        generations_per_turn=1,
        slow_workers=4,
        search_rounds=12,
    )
    kwargs.update(overrides)
    return build_protocol_snapshot(**kwargs)


def _strict(**overrides) -> dict:
    return _snapshot(
        strict_turns=True,
        required_provider="deepseek",
        required_model="deepseek-v4-pro",
        **overrides,
    )


def test_state_constants_pin_the_paper_model() -> None:
    assert state_mod.FORMAL_NOISE_PROFILE == PAPER == "uniform_paper_v1"
    assert state_mod.FORMAL_P == 0.005
    assert NoiseSweepConfig().profile.value == state_mod.FORMAL_NOISE_PROFILE
    assert NoiseSweepConfig().values == (state_mod.FORMAL_P,)
    assert StructuralSearchConfig().profile_id == "telescoping_structural_v3_7"
    assert UNIFORM_PROFILES == frozenset({NoiseProfile.UNIFORM_PAPER, NoiseProfile.UNIFORM_LEGACY})


def test_snapshot_records_the_noise_model_as_protocol_keys() -> None:
    protocol = _snapshot()
    assert protocol["protocol_profile"] == "telescoping_structural_v3_7"
    assert protocol["noise_profile"] == PAPER
    assert protocol["noise_values"] == [0.005]
    assert protocol["anchor_p"] == pytest.approx(0.005)
    assert protocol["verification"]["noise"]["profile"] == PAPER
    assert protocol["verification"]["noise"]["values"] == [0.005]
    strict = _strict()
    assert strict["protocol_profile"] == "telescoping_structural_v3_7"
    assert (strict["noise_profile"], strict["noise_values"]) == (PAPER, [0.005])
    # The keys survive the JSON round trip the checkpoint / report take.
    assert json.loads(json.dumps(protocol))["noise_values"] == [0.005]


def test_a_non_paper_noise_model_retags_the_profile_outside_strict() -> None:
    legacy = _snapshot(verification=VerificationConfig(noise=NoiseSweepConfig.uniform_legacy()))
    assert legacy["protocol_profile"] == "telescoping_structural_v3_7+noiseuniform_legacy_v1"
    assert legacy["noise_profile"] == LEGACY
    assert legacy["noise_values"] == [0.005]
    assert legacy["verification"]["noise"]["profile"] == LEGACY
    # Retags compose in a fixed order: shards, final shards, then noise.
    both = _snapshot(
        verification=VerificationConfig(noise=NoiseSweepConfig.uniform_legacy()),
        shot_shards=8,
        final_shot_shards=64,
    )
    assert both["protocol_profile"] == (
        "telescoping_structural_v3_7+shards8+finalshards64+noiseuniform_legacy_v1"
    )
    # A different p on the paper model is recorded (anchor_p / noise_values),
    # not retagged: p was never part of the profile id.
    other_p = _snapshot(verification=VerificationConfig(p_values=(0.003,)))
    assert other_p["protocol_profile"] == "telescoping_structural_v3_7"
    assert other_p["noise_values"] == [0.003]


def test_strict_snapshot_refuses_the_legacy_model_and_any_other_p() -> None:
    with pytest.raises(ValueError, match="telescoping_structural_v3_7.*noise_profile=uniform_paper_v1"):
        _strict(verification=VerificationConfig(noise=NoiseSweepConfig.uniform_legacy()))
    with pytest.raises(ValueError, match=re.escape("noise_values=[0.005]")):
        _strict(verification=VerificationConfig(p_values=(0.003,)))
    with pytest.raises(ValueError, match=re.escape("noise_values=[0.005]")):
        _strict(verification=VerificationConfig(noise=NoiseSweepConfig.uniform((0.004,))))
    # The paper model at p=0.005 is exactly what strict accepts.
    assert _strict(verification=VerificationConfig(noise=NoiseSweepConfig.uniform((0.005,))))[
        "noise_profile"
    ] == PAPER


def test_a_v34_checkpoint_cannot_resume_into_v35() -> None:
    current = _snapshot()

    # A v3.4 snapshot: legacy idle-noise profile, no noise_values key, v3.4 id.
    v34 = json.loads(json.dumps(current))
    v34.pop("noise_values")
    v34["noise_profile"] = LEGACY
    v34["protocol_profile"] = "bposd_structural_v3_4"
    v34["verification"]["noise"]["profile"] = LEGACY
    with pytest.raises(ValueError, match="protocol snapshot mismatch") as excinfo:
        validate_resume_protocol(v34, current)
    message = str(excinfo.value)
    assert "noise_values" in message and "noise_profile" in message

    # The profile alone (same p, same everything else) is a different model.
    legacy_only = json.loads(json.dumps(current))
    legacy_only["noise_profile"] = LEGACY
    with pytest.raises(ValueError, match="noise_profile"):
        validate_resume_protocol(legacy_only, current)

    # The p alone is a different point.
    other_p = json.loads(json.dumps(current))
    other_p["noise_values"] = [0.003]
    with pytest.raises(ValueError, match="noise_values"):
        validate_resume_protocol(other_p, current)

    # And the round trip of the current snapshot resumes.
    validate_resume_protocol(json.loads(json.dumps(current)), current)


# ---------------------------------------------------------------------------
# (e) validator constants and checks
# ---------------------------------------------------------------------------


def _validate(run_dir: Path, p_value: float = 0.005) -> int:
    return validate(run_dir, turns=1, p_value=p_value, final_shots=FINAL_SHOTS, model="deepseek-v4-pro")


def test_validator_constants_mirror_the_package() -> None:
    assert validator.PROTOCOL_PROFILE == "telescoping_structural_v3_7"
    assert validator.PROTOCOL_PROFILE == StructuralSearchConfig().profile_id
    assert validator.STRUCTURAL_EXPECTED["profile_id"] == "telescoping_structural_v3_7"
    assert validator.FORMAL_NOISE_PROFILE == state_mod.FORMAL_NOISE_PROFILE == PAPER
    assert validator.FORMAL_P == state_mod.FORMAL_P == 0.005
    assert validator.FORMAL_P == RACING_ANCHOR_P[NoiseProfile.UNIFORM_PAPER]
    # The fixtures' noise blobs are the shapes the package actually serializes.
    real_sweep = VerificationConfig().to_dict()["noise"]
    assert (real_sweep["profile"], real_sweep["values"]) == (
        NOISE_SWEEP["profile"],
        NOISE_SWEEP["values"],
    )
    validator._validate_noise_sweep(real_sweep, "package default")
    snapshot = _snapshot()
    validator._validate_noise_sweep(snapshot["verification"]["noise"], "snapshot")
    real_row = noise_result_fields(NoiseSweepConfig().point(0))
    assert real_row["noise_point"]["profile"] == NOISE_POINT["profile"]
    assert real_row["noise_point"]["value"] == NOISE_POINT["value"]
    validator._validate_noise_point(real_row["noise_point"], real_row["noise_value"], "pipeline row")
    legacy_row = noise_result_fields(NoiseSweepConfig.uniform_legacy().point(0))
    with pytest.raises(ValueError, match="noise_point profile='uniform_legacy_v1'"):
        validator._validate_noise_point(legacy_row["noise_point"], legacy_row["noise_value"], "x")


def test_validator_accepts_a_paper_model_run(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    # Candidate artifacts: one with results at the frozen point, one that never
    # reached the pool (job_config only) — both accepted.
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(
        slow_dir / "candidates" / "measured.json",
        {
            "status": "ok",
            "job_config": {"workers": 127, "shot_shards": 16, **dict(validator.FORMAL_JOB_POLICY)},
            "results": [
                {"status": "ok", "noise_point": dict(NOISE_POINT), "noise_value": 0.005},
                {"status": "circuit_error", "noise_point": dict(NOISE_POINT), "noise_value": 0.005},
            ],
        },
    )
    _write_json(slow_dir / "candidates" / "rejected.json", {"status": "structural_reject"})
    assert _validate(tmp_path) == 1


def test_validator_refuses_a_caller_declared_other_p(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    with pytest.raises(ValueError, match=r"caller declared p=0\.003, but telescoping_structural_v3_7"):
        _validate(tmp_path, p_value=0.003)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda protocol: protocol.__setitem__("noise_profile", LEGACY),
            "protocol snapshot: noise_profile='uniform_legacy_v1'",
        ),
        (
            lambda protocol: protocol.pop("noise_profile"),
            "protocol snapshot: noise_profile=None",
        ),
        (
            lambda protocol: protocol.pop("noise_values"),
            "protocol snapshot: noise_values=None",
        ),
        (
            lambda protocol: protocol.__setitem__("noise_values", [0.003]),
            "protocol snapshot: noise_values=[0.003]",
        ),
        (
            lambda protocol: protocol.__setitem__("noise_values", [0.005, 0.003]),
            "protocol snapshot: noise_values=[0.005, 0.003]",
        ),
        (
            lambda protocol: protocol["verification"].__setitem__("noise", dict(LEGACY_SWEEP)),
            "protocol snapshot: noise profile='uniform_legacy_v1'",
        ),
        (
            lambda protocol: protocol["verification"].pop("noise"),
            "protocol snapshot: noise sweep is missing",
        ),
    ],
)
def test_validator_rejects_a_v34_shaped_or_tampered_snapshot(tmp_path, mutate, message) -> None:
    report = _strict_report()
    mutate(report["protocol"])
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate(tmp_path)


def test_validator_rejects_a_v34_profile_id_in_the_snapshot(tmp_path) -> None:
    report = _strict_report()
    report["protocol"]["protocol_profile"] = "bposd_structural_v3_4"
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match="protocol profile is 'bposd_structural_v3_4'"):
        _validate(tmp_path)


@pytest.mark.parametrize("stage", ["search", "final"])
def test_validator_rejects_a_run_config_with_another_noise_model(tmp_path, stage: str) -> None:
    _write_validated_run(tmp_path, _strict_report())
    if stage == "search":
        path = tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "slow_run_config.json"
        config = _slow_run_config(rounds=12)
        where = "slow run config"
    else:
        path = tmp_path / "final_slow" / "T12" / "slow_run_config.json"
        config = _slow_run_config(
            rounds=12, seed=FINAL_BASE_SEED, shot_shards=validator.FORMAL_FINAL_SHOT_SHARDS
        )
        where = "final run config"
    config["verification"]["noise"] = dict(LEGACY_SWEEP)
    _write_json(path, config)
    with pytest.raises(ValueError, match=f"{where}: noise profile='uniform_legacy_v1'"):
        _validate(tmp_path)

    config["verification"]["noise"] = {**NOISE_SWEEP, "values": [0.003]}
    _write_json(path, config)
    with pytest.raises(ValueError, match=re.escape(f"{where}: noise values=[0.003]")):
        _validate(tmp_path)

    del config["verification"]["noise"]
    _write_json(path, config)
    with pytest.raises(ValueError, match=f"{where}: noise sweep is missing"):
        _validate(tmp_path)


@pytest.mark.parametrize("stage", ["search", "final"])
def test_validator_rejects_a_measured_row_at_another_noise_point(tmp_path, stage: str) -> None:
    _write_validated_run(tmp_path, _strict_report())
    if stage == "search":
        path = tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "rankings.json"
        where = "search rankings"
    else:
        path = tmp_path / "final_slow" / "T12" / "rankings.json"
        where = "final rankings"

    rankings = _rankings(12)
    rankings["rankings"][0]["noise_point"] = dict(LEGACY_POINT)
    _write_json(path, rankings)
    with pytest.raises(
        ValueError, match=f"{where}: row g0001_p0001: noise_point profile='uniform_legacy_v1'"
    ):
        _validate(tmp_path)

    rankings = _rankings(12)
    rankings["rankings"][0]["noise_point"] = {**NOISE_POINT, "value": 0.003}
    rankings["rankings"][0]["noise_value"] = 0.003
    _write_json(path, rankings)
    with pytest.raises(ValueError, match=re.escape(f"{where}: row g0001_p0001: noise_point value=0.003")):
        _validate(tmp_path)

    rankings = _rankings(12)
    rankings["rankings"][0]["noise_value"] = 0.003
    _write_json(path, rankings)
    with pytest.raises(ValueError, match=re.escape(f"{where}: row g0001_p0001: noise_value=0.003")):
        _validate(tmp_path)

    rankings = _rankings(12)
    del rankings["rankings"][0]["noise_point"]
    _write_json(path, rankings)
    with pytest.raises(ValueError, match=f"{where}: row g0001_p0001: noise_point is missing"):
        _validate(tmp_path)


def test_validator_skips_terminal_rows_but_reads_every_candidate_result(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    # Terminal rows carry no measurement and no noise point: accepted.
    rankings = _rankings(12)
    rankings["rankings"].append(
        {
            "program_id": "g0001_p0002",
            "status": "decode_infeasible",
            "decoder": TELESCOPING,
            "decoder_config": dict(TELESCOPING_BUILD),
            "rounds": 12,
        }
    )
    _write_json(slow_dir / "rankings.json", rankings)
    assert _validate(tmp_path) == 1

    # A candidate result at the retired model fails, whatever its status.
    _write_json(
        slow_dir / "candidates" / "candidate.json",
        {
            "status": "ok",
            "results": [
                {"status": "ok", "noise_point": dict(NOISE_POINT), "noise_value": 0.005},
                {"status": "circuit_error", "noise_point": dict(LEGACY_POINT), "noise_value": 0.005},
            ],
        },
    )
    with pytest.raises(
        ValueError,
        match="slow candidates: candidate.json results\\[1\\]: noise_point profile='uniform_legacy_v1'",
    ):
        _validate(tmp_path)

    # An ok result that records no noise point at all is not a measurement.
    _write_json(
        slow_dir / "candidates" / "candidate.json",
        {"status": "ok", "results": [{"status": "ok", "noise_value": 0.005}]},
    )
    with pytest.raises(ValueError, match="candidate.json results\\[0\\]: noise_point is missing"):
        _validate(tmp_path)

    final_dir = tmp_path / "final_slow" / "T12"
    _write_json(slow_dir / "candidates" / "candidate.json", {"status": "structural_reject"})
    _write_json(
        final_dir / "candidates" / "candidate.json",
        {"status": "ok", "results": [{"status": "ok", "noise_point": {**NOISE_POINT, "value": 0.003}, "noise_value": 0.003}]},
    )
    with pytest.raises(ValueError, match=re.escape("final candidates: candidate.json results[0]: noise_point value=0.003")):
        _validate(tmp_path)


# ---------------------------------------------------------------------------
# (f) RACING_ANCHOR_P
# ---------------------------------------------------------------------------


def test_racing_anchor_covers_the_paper_profile() -> None:
    assert RACING_ANCHOR_P[NoiseProfile.UNIFORM_PAPER] == 0.005
    assert RACING_ANCHOR_P[NoiseProfile.UNIFORM_LEGACY] == 0.005
    assert anchor_p_for_profile("uniform_paper_v1") == 0.005
    assert anchor_p_for_profile(NoiseProfile.UNIFORM_PAPER) == 0.005
    # Every profile the CLI can select has an anchor.
    assert set(RACING_ANCHOR_P) == set(NoiseProfile)


# ---------------------------------------------------------------------------
# VerificationConfig: both uniform profiles are "uniform" for the alias sync
# ---------------------------------------------------------------------------


def test_p_values_alias_follows_the_uniform_family_and_keeps_its_profile() -> None:
    default = VerificationConfig()
    assert default.noise == NoiseSweepConfig.uniform()
    assert default.p_values == (0.005,)

    via_alias = VerificationConfig(p_values=(0.003,))
    assert via_alias.noise == NoiseSweepConfig.uniform((0.003,))
    assert via_alias.p_values == (0.003,)

    legacy_alias = VerificationConfig(p_values=(0.003,), noise=NoiseSweepConfig.uniform_legacy())
    assert legacy_alias.noise == NoiseSweepConfig.uniform_legacy((0.003,))
    assert legacy_alias.p_values == (0.003,)

    from_sweep = VerificationConfig(noise=NoiseSweepConfig.uniform_legacy((0.004,)))
    assert from_sweep.p_values == (0.004,)

    # slope_offset=0.958 is the uniform family's proxy calibration under BOTH
    # ids; every other profile is uncalibrated.
    assert VerificationConfig(slope_offset=0.958).slope_offset == 0.958
    assert (
        VerificationConfig(noise=NoiseSweepConfig.uniform_legacy(), slope_offset=0.958).slope_offset
        == 0.958
    )
    assert VerificationConfig(noise=NoiseSweepConfig.si1000(), slope_offset=0.958).slope_offset is None


# ---------------------------------------------------------------------------
# Artifact round-trip: the noise dict is authoritative; its absence is history
# ---------------------------------------------------------------------------


def test_verification_from_dict_keeps_artifact_noise_and_reads_pre_profile_artifacts_as_legacy() -> None:
    from omniqec_agentic.evaluation.common import verification_from_dict

    # A serialized v3.5 config carries the paper profile and comes back as it.
    encoded = VerificationConfig().to_dict()
    assert encoded["noise"]["profile"] == PAPER
    assert verification_from_dict(encoded).noise == NoiseSweepConfig.uniform()

    # A v3.4 artifact names uniform_legacy_v1 explicitly and is NOT upgraded.
    legacy_encoded = VerificationConfig(noise=NoiseSweepConfig.uniform_legacy()).to_dict()
    assert legacy_encoded["noise"]["profile"] == LEGACY
    assert verification_from_dict(legacy_encoded).noise == NoiseSweepConfig.uniform_legacy()

    # An artifact that predates the noise-profile architecture (no ``noise``
    # dict, only p_values) was lowered with the idle-noise rule: it reads back
    # as uniform_legacy_v1 with its own p values, never as the paper profile.
    ancient = verification_from_dict({"p_values": [0.003]})
    assert ancient.noise == NoiseSweepConfig.uniform_legacy((0.003,))
    assert ancient.p_values == (0.003,)
    assert verification_from_dict(None).noise.profile is NoiseProfile.UNIFORM_LEGACY
