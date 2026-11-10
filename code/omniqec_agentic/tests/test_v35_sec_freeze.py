"""v3.5 SEC = twobasis_nonint_v1 (2026-09-02): the circuit policy is a protocol
field. Freeze / plumbing regressions (contract §2, §3 "Freeze"):

  - SecPolicyConfig defaults, validation, to_dict/from_dict (absent -> default,
    never a silent re-measure: the snapshot/resume check refuses v3.4).
  - VerificationConfig carries sec_policy through to_dict / verification_from_dict.
  - CompiledCode.layer_blocks (bb / lp_abelian / lp_nonabelian; None for hgp)
    reproduces hx/hz exactly, is JSON-safe and is NOT part of code_hash.
  - Snapshot keys, strict refusal of lrc (and of any K/iters/seed variant),
    non-strict retag suffixes, resume refusal of a v3.4-shaped snapshot.
  - Validator constants pinned to the package, snapshot / run-config /
    candidate-schedule checks.
  - CLI: --sec-policy default on every circuit-building subcommand, strict
    refusal of lrc at launch, plumbing into VerificationConfig.
  - Worker requests pass sec_policy + layer_blocks into
    build_memory_experiment_pair (build_dem and distance_job); every package
    call site passes layer_blocks= (grep-based).
  - sec_ext_evidence accepts both RankedSchedule shapes and records the id.
  - Launcher self-check names the SEC default.

The circuit / layers / certification themselves live in
test_v35_twobasis_sec.py (other agents); nothing here needs stim.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omniqec_agentic import sec as sec_pkg
from omniqec_agentic.config import (
    DEFAULT_SEC_POLICY,
    budget_policy,
    LRC_SEC_POLICY,
    SEC_POLICY_IDS,
    RacingConfig,
    SecPolicyConfig,
    VerificationConfig,
    sec_policy_for_id,
)
from omniqec_agentic.construction import build_compiled_code
from omniqec_agentic.construction.build import reconstruct_from_layer_blocks
from omniqec_agentic.evaluation.common import verification_from_dict
from omniqec_agentic.evaluation.evidence import sec_ext_evidence
from omniqec_agentic.hpc import validate_fig2_campaign as validator
from omniqec_agentic.hpc.validate_fig2_campaign import validate
from omniqec_agentic.workflow import state as state_mod
from omniqec_agentic.workflow.checkpoint import validate_resume_protocol
from omniqec_agentic.workflow.state import build_protocol_snapshot

from test_campaign_strict_turns import (
    FINAL_BASE_SEED,
    NOISE_POINT,
    _rankings,
    _slow_run_config,
    _strict_report,
    _write_json,
    _write_validated_run,
)

TWOBASIS = "twobasis_nonint_v1"
LRC = "lrc_ranked_v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# v3.6: the profile id and the full SEC policy block come from the package
# (docs/06 §3 M1 added sec_cert_seeds / sec_cert_iters_scale / sec_refine_top
# / sec_refine_iters to the frozen policy; the v3.5 four-key block is a
# strict subset of it).
PROFILE = state_mod.FORMAL_PROFILE
SEC_POLICY_BLOCK = DEFAULT_SEC_POLICY.to_dict()
# The end-to-end validator tests below write a whole strict run from the
# shared fixtures in test_campaign_strict_turns.py. Until those fixtures are
# re-cut for the v3.6 protocol (decoder, profile id, spec.qubit_budget, budget
# policy) the validator refuses the fixture run before it reaches the SEC
# checks under test, so they are skipped -- visibly -- rather than failing on
# a fixture that another change owns. They resume automatically once the
# fixture's profile id matches the validator's.
_FIXTURE_IS_CURRENT = _strict_report()["protocol"].get("protocol_profile") == validator.PROTOCOL_PROFILE
needs_v36_fixtures = pytest.mark.skipif(
    not _FIXTURE_IS_CURRENT,
    reason="test_campaign_strict_turns fixtures are not on the validator's protocol profile "
    f"({_strict_report()['protocol'].get('protocol_profile')!r} != {validator.PROTOCOL_PROFILE!r})",
)

BB72_SPEC = {
    "family": "bb", "ell": 6, "m": 6,
    "A": [[3, 0], [0, 1], [0, 2]], "B": [[0, 3], [1, 0], [2, 0]],
}
BB_SMALL_SPEC = {
    "family": "bb", "ell": 3, "m": 3,
    "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]],
}
LP_NONABELIAN_D6_SPEC = {
    "family": "lp_nonabelian", "group": "D6",
    "A": [[[[0, 0], [1, 0], [0, 1]]]], "B": [[[[0, 0], [2, 1], [3, 0]]]],
}
LP_ABELIAN_SPEC = {
    "family": "lp_abelian", "group": {"factors": [3, 3]},
    "A": [[[[0, 0], [1, 0]], [[0, 1], [1, 1]]]], "B": [[[[0, 0], [2, 1]]]],
}
HGP_SPEC = {
    "family": "hgp",
    "seeds": {
        "H1": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        "H2": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
    },
}


# ---------------------------------------------------------------------------
# SecPolicyConfig
# ---------------------------------------------------------------------------


def test_sec_policy_defaults_are_the_frozen_twobasis_protocol() -> None:
    assert SEC_POLICY_IDS == (TWOBASIS, LRC)
    assert DEFAULT_SEC_POLICY == SecPolicyConfig()
    assert DEFAULT_SEC_POLICY.to_dict() == {
        "sec_policy_id": TWOBASIS,
        "sec_candidates": 16,
        "sec_cert_iters": 400,
        "sec_cert_seed": 0,
        # v3.6 M1 staged certification (docs/06 §3): >= 3 seeds, iterations
        # scaled by the extended-kernel dimension, top-4 refined at 2000.
        "sec_cert_seeds": 3,
        "sec_cert_iters_scale": 4.0,
        "sec_refine_top": 4,
        "sec_refine_iters": 2000,
        # Review fix (M1 cost): per-code ISD iteration budget, 0 = unlimited.
        "sec_cert_iters_budget": 0,
    }
    assert PROFILE == "telescoping_structural_v3_7"
    assert not DEFAULT_SEC_POLICY.is_lrc
    assert LRC_SEC_POLICY.is_lrc
    assert (LRC_SEC_POLICY.sec_candidates, LRC_SEC_POLICY.sec_cert_iters) == (64, 100)
    assert sec_policy_for_id(TWOBASIS) is DEFAULT_SEC_POLICY
    assert sec_policy_for_id(LRC) is LRC_SEC_POLICY
    with pytest.raises(ValueError, match="unknown sec_policy_id"):
        sec_policy_for_id("interleaved_v9")
    with pytest.raises(ValueError, match="unknown sec_policy_id"):
        SecPolicyConfig(sec_policy_id="interleaved_v9")
    with pytest.raises(ValueError, match="sec_candidates"):
        SecPolicyConfig(sec_candidates=0)
    with pytest.raises(ValueError, match="sec_cert_iters"):
        SecPolicyConfig(sec_cert_iters=0)
    # The four copies of the policy constants are pinned together.
    assert sec_pkg.SEC_POLICY_ID == DEFAULT_SEC_POLICY.sec_policy_id == TWOBASIS
    assert sec_pkg.LRC_SEC_POLICY_ID == LRC_SEC_POLICY.sec_policy_id == LRC
    assert state_mod.FORMAL_SEC_POLICY == validator.FORMAL_SEC_POLICY == TWOBASIS
    assert validator.FORMAL_SEC_CANDIDATES == DEFAULT_SEC_POLICY.sec_candidates == 16
    assert validator.FORMAL_SEC_CERT_ITERS == DEFAULT_SEC_POLICY.sec_cert_iters == 400
    assert validator.FORMAL_SEC_CERT_SEED == DEFAULT_SEC_POLICY.sec_cert_seed == 0
    # The LRC seam constants of the historical path equal the LRC policy.
    from omniqec_agentic.verification import circuit as circuit_mod

    assert circuit_mod.LRC_SEAM_CANDIDATES == LRC_SEC_POLICY.sec_candidates
    assert circuit_mod.LRC_SEAM_ISD_ITERS == LRC_SEC_POLICY.sec_cert_iters


def test_sec_policy_round_trips_and_absent_means_default() -> None:
    assert SecPolicyConfig.from_dict(None) == DEFAULT_SEC_POLICY
    assert SecPolicyConfig.from_dict({}) == DEFAULT_SEC_POLICY
    assert SecPolicyConfig.from_dict(LRC_SEC_POLICY.to_dict()) == LRC_SEC_POLICY
    variant = SecPolicyConfig(sec_candidates=8, sec_cert_iters=50, sec_cert_seed=3)
    assert SecPolicyConfig.from_dict(json.loads(json.dumps(variant.to_dict()))) == variant
    assert variant != DEFAULT_SEC_POLICY


def test_verification_config_carries_the_sec_policy() -> None:
    default = VerificationConfig()
    assert default.sec_policy == DEFAULT_SEC_POLICY
    encoded = default.to_dict()
    assert encoded["sec_policy"] == DEFAULT_SEC_POLICY.to_dict()
    assert verification_from_dict(encoded).sec_policy == DEFAULT_SEC_POLICY

    lrc = VerificationConfig(sec_policy=LRC_SEC_POLICY)
    lrc_encoded = json.loads(json.dumps(lrc.to_dict()))
    assert lrc_encoded["sec_policy"]["sec_policy_id"] == LRC
    assert verification_from_dict(lrc_encoded).sec_policy == LRC_SEC_POLICY

    # A v3.4 artifact has no sec_policy block: it reads back as the default
    # (the snapshot/resume check refuses it; nothing is re-measured silently).
    v34 = dict(encoded)
    del v34["sec_policy"]
    assert verification_from_dict(v34).sec_policy == DEFAULT_SEC_POLICY
    assert verification_from_dict(None).sec_policy == DEFAULT_SEC_POLICY
    assert verification_from_dict({"sec_policy": "garbage"}).sec_policy == DEFAULT_SEC_POLICY


# ---------------------------------------------------------------------------
# CompiledCode.layer_blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec", [BB72_SPEC, BB_SMALL_SPEC, LP_NONABELIAN_D6_SPEC, LP_ABELIAN_SPEC],
    ids=["bb72", "bb_small", "lp_nonabelian_d6", "lp_abelian_c3xc3"],
)
def test_layer_blocks_reproduce_hx_hz_and_are_json_safe(spec: dict) -> None:
    compiled = build_compiled_code(spec)
    blocks = compiled.layer_blocks
    assert isinstance(blocks, dict) and blocks["schema"] == "layer_blocks_v1"
    assert json.loads(json.dumps(blocks)) == blocks  # plain ints / lists only
    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    g = int(blocks["block_size"])
    for basis, h in (("X", hx), ("Z", hz)):
        block = blocks[basis]
        assert block["n_check_blocks"] * g == h.shape[0]
        assert block["n_data_blocks"] * g == h.shape[1]
        rebuilt = reconstruct_from_layer_blocks(
            blocks, basis, n_rows=h.shape[0], n_cols=h.shape[1]
        )
        assert np.array_equal(rebuilt, h)
        # One term per (check block, data block, permutation); every term has
        # exactly g edges and the terms tile the nonzeros exactly once.
        assert sum(g for _ in block["terms"]) == int(h.sum())
        for cb, db, perm in block["terms"]:
            assert sorted(perm) == list(range(g))
            assert 0 <= cb < block["n_check_blocks"] and 0 <= db < block["n_data_blocks"]
    # Data blocks follow the sector split: left-sector blocks first.
    left_blocks = compiled.sector_sizes[0] // g
    assert any(db < left_blocks for _, db, _ in blocks["X"]["terms"])
    assert any(db >= left_blocks for _, db, _ in blocks["X"]["terms"])


def test_bb_layer_blocks_follow_the_canonicalised_spec() -> None:
    # ell > m is canonicalised by the validator (axes swapped); the block
    # records must describe the matrices actually built, not the raw spec.
    raw = {**BB72_SPEC, "ell": 9, "m": 6}
    compiled = build_compiled_code(raw)
    assert compiled.layer_blocks["block_size"] == 54
    assert np.asarray(compiled.hx).shape == (54, 108)


def test_hgp_has_no_layer_blocks_and_code_hash_ignores_them() -> None:
    hgp = build_compiled_code(HGP_SPEC)
    assert hgp.layer_blocks is None
    bb = build_compiled_code(BB72_SPEC)
    assert bb.layer_blocks is not None
    # code_hash hashes hx/hz/h_mask only; layer_blocks add nothing to the
    # identity and nothing to the compact metadata artifact.
    from omniqec_agentic.construction import compiled_code_metadata
    from omniqec_agentic.construction.build import compiled_code_hash

    assert bb.code_hash == compiled_code_hash(np.asarray(bb.hx), np.asarray(bb.hz))
    assert "layer_blocks" not in compiled_code_metadata(bb)
    assert bb.to_dict()["layer_blocks"] == bb.layer_blocks


def test_layer_blocks_that_do_not_reproduce_the_matrices_are_refused(monkeypatch) -> None:
    from omniqec_agentic.construction import registry as registry_mod

    plugin = registry_mod.get_family("bb")

    def wrong(spec):
        blocks = plugin.layer_blocks(spec)
        first = blocks["X"]["terms"][0]
        perm = list(first[2])
        perm[0], perm[1] = perm[1], perm[0]
        blocks["X"]["terms"][0] = [first[0], first[1], perm]
        return blocks

    monkeypatch.setitem(
        registry_mod.FAMILY_REGISTRY, "bb", dataclasses.replace(plugin, layer_blocks=wrong)
    )
    with pytest.raises(ValueError, match="layer_blocks do not reproduce hx/hz"):
        build_compiled_code(BB_SMALL_SPEC)


# ---------------------------------------------------------------------------
# Snapshot freeze / strict refusal / resume
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


def test_snapshot_records_the_sec_policy_as_protocol_keys() -> None:
    protocol = _snapshot()
    assert protocol["protocol_profile"] == PROFILE
    assert protocol["sec_policy_id"] == TWOBASIS
    assert protocol["sec_candidates"] == 16 and type(protocol["sec_candidates"]) is int
    assert protocol["sec_cert_iters"] == 400 and type(protocol["sec_cert_iters"]) is int
    assert protocol["sec_synthesis"] == {"candidates": 16, "isd_iters": 400}
    assert protocol["verification"]["sec_policy"] == SEC_POLICY_BLOCK
    strict = _strict()
    assert strict["sec_policy_id"] == TWOBASIS
    assert json.loads(json.dumps(protocol))["sec_policy_id"] == TWOBASIS


def test_a_non_default_sec_policy_retags_the_profile_outside_strict() -> None:
    lrc = _snapshot(verification=VerificationConfig(sec_policy=LRC_SEC_POLICY))
    assert lrc["protocol_profile"] == f"{PROFILE}+seclrc_ranked_v1"
    assert lrc["sec_policy_id"] == LRC
    assert (lrc["sec_candidates"], lrc["sec_cert_iters"]) == (64, 100)
    assert lrc["sec_synthesis"] == {"candidates": 64, "isd_iters": 100}
    variant = _snapshot(
        verification=VerificationConfig(
            sec_policy=SecPolicyConfig(sec_candidates=8, sec_cert_iters=50, sec_cert_seed=2)
        )
    )
    assert variant["protocol_profile"] == f"{PROFILE}+secK8I50S2"
    # Retags compose in a fixed order: shards, final shards, noise, sec.
    from omniqec_agentic.verification.noise import NoiseSweepConfig

    both = _snapshot(
        verification=VerificationConfig(
            noise=NoiseSweepConfig.uniform_legacy(), sec_policy=LRC_SEC_POLICY
        ),
        shot_shards=8,
    )
    assert both["protocol_profile"] == (
        f"{PROFILE}+shards8+noiseuniform_legacy_v1+seclrc_ranked_v1"
    )


def test_strict_snapshot_refuses_lrc_and_any_sec_variant() -> None:
    with pytest.raises(ValueError, match=f"{PROFILE}.*sec_policy=twobasis_nonint_v1"):
        _strict(verification=VerificationConfig(sec_policy=LRC_SEC_POLICY))
    with pytest.raises(ValueError, match="sec_policy=twobasis_nonint_v1"):
        _strict(verification=VerificationConfig(sec_policy=SecPolicyConfig(sec_candidates=8)))
    with pytest.raises(ValueError, match="sec_policy=twobasis_nonint_v1"):
        _strict(verification=VerificationConfig(sec_policy=SecPolicyConfig(sec_cert_seed=1)))
    assert _strict(verification=VerificationConfig(sec_policy=DEFAULT_SEC_POLICY))[
        "sec_policy_id"
    ] == TWOBASIS


def test_a_v34_lrc_checkpoint_cannot_resume_into_v35() -> None:
    current = _snapshot()
    # A v3.4 snapshot: lrc id, LRC sec_synthesis, no sec_candidates /
    # sec_cert_iters keys, no verification.sec_policy blob.
    v34 = json.loads(json.dumps(current))
    v34["sec_policy_id"] = LRC
    v34["sec_synthesis"] = {"candidates": 64, "isd_iters": 100}
    v34.pop("sec_candidates")
    v34.pop("sec_cert_iters")
    v34["verification"].pop("sec_policy")
    with pytest.raises(ValueError, match="protocol snapshot mismatch") as excinfo:
        validate_resume_protocol(v34, current)
    message = str(excinfo.value)
    for key in ("sec_policy_id", "sec_candidates", "sec_cert_iters", "sec_synthesis", "verification"):
        assert key in message
    # Same id, different certification budget: refused too.
    other = json.loads(json.dumps(current))
    other["sec_cert_iters"] = 100
    with pytest.raises(ValueError, match="sec_cert_iters"):
        validate_resume_protocol(other, current)
    # The seed lives only in the blob and still refuses.
    seeded = json.loads(json.dumps(current))
    seeded["verification"]["sec_policy"]["sec_cert_seed"] = 7
    with pytest.raises(ValueError, match="verification"):
        validate_resume_protocol(seeded, current)
    validate_resume_protocol(json.loads(json.dumps(current)), current)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate(run_dir: Path) -> int:
    # v3.6: final shots are a budget-policy value (docs/06 §1.4); the strict
    # fixture is a b144 run, so the caller-declared value must equal the
    # small-budget policy (200k) or the validator refuses it.
    return validate(
        run_dir, turns=1, p_value=0.005,
        final_shots=int(budget_policy(144)["final_shots"]), model="deepseek-v4-pro",
    )


def _schedule_blob(**overrides) -> dict:
    blob = {
        "sec_policy_id": TWOBASIS, "depth": 16, "layers_x": 6, "layers_z": 6,
        "candidate_index_x": 0, "candidate_index_z": 3, "d_ext_x": 6, "d_ext_z": 6,
        "candidates": 16, "cert_iters": 400,
        # v3.6 M1 staged certification pins (docs/06 §3).
        "cert_seeds": int(DEFAULT_SEC_POLICY.sec_cert_seeds),
        "refine_iters": int(DEFAULT_SEC_POLICY.sec_refine_iters),
    }
    blob.update(overrides)
    return blob


def _candidate_with_schedules(schedule_x: dict, schedule_z: dict | None = None) -> dict:
    schedule_z = schedule_x if schedule_z is None else schedule_z
    return {
        "status": "ok",
        "results": [
            {
                "status": "ok",
                "noise_point": dict(NOISE_POINT),
                "noise_value": 0.005,
                "basis": {
                    "X": {"status": "ok", "memory_experiment": {"schedule": schedule_x}},
                    "Z": {"status": "ok", "memory_experiment": {"schedule": schedule_z}},
                },
            }
        ],
    }


def test_validator_sec_blob_check_matches_the_package_serialization() -> None:
    validator._validate_sec_policy_blob(VerificationConfig().to_dict()["sec_policy"], "package")
    validator._validate_sec_policy_blob(_snapshot()["verification"]["sec_policy"], "snapshot")
    with pytest.raises(ValueError, match="package: sec_policy is missing"):
        validator._validate_sec_policy_blob(None, "package")
    with pytest.raises(ValueError, match="sec_policy_id='lrc_ranked_v1'"):
        validator._validate_sec_policy_blob(LRC_SEC_POLICY.to_dict(), "x")
    with pytest.raises(ValueError, match="sec_candidates=8"):
        validator._validate_sec_policy_blob({**SEC_POLICY_BLOCK, "sec_candidates": 8}, "x")
    # Ints by type: a float 400.0 is not the frozen int.
    with pytest.raises(ValueError, match="sec_cert_iters=400.0"):
        validator._validate_sec_policy_blob({**SEC_POLICY_BLOCK, "sec_cert_iters": 400.0}, "x")
    with pytest.raises(ValueError, match="sec_cert_seed=1"):
        validator._validate_sec_policy_blob({**SEC_POLICY_BLOCK, "sec_cert_seed": 1}, "x")


@needs_v36_fixtures
def test_validator_accepts_a_twobasis_run_with_candidate_schedules(tmp_path) -> None:
    _write_validated_run(tmp_path, _strict_report())
    slow_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow"
    _write_json(slow_dir / "candidates" / "measured.json", _candidate_with_schedules(_schedule_blob()))
    final_dir = tmp_path / "final_slow" / "T12"
    _write_json(
        final_dir / "candidates" / "measured.json",
        _candidate_with_schedules(_schedule_blob(candidate_index_x=5, d_ext_x=None)),
    )
    # A candidate that never reached the pool, and a failed result, are skipped.
    _write_json(slow_dir / "candidates" / "rejected.json", {"status": "structural_reject"})
    _write_json(
        slow_dir / "candidates" / "failed.json",
        {"status": "circuit_error", "results": [{"status": "circuit_error", "noise_point": dict(NOISE_POINT), "noise_value": 0.005}]},
    )
    assert _validate(tmp_path) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("sec_policy_id", LRC), "protocol snapshot: sec_policy_id='lrc_ranked_v1'"),
        (lambda p: p.pop("sec_policy_id"), "protocol snapshot: sec_policy_id=None"),
        (lambda p: p.pop("sec_candidates"), "protocol snapshot: sec_candidates=None"),
        (lambda p: p.__setitem__("sec_candidates", 64), "protocol snapshot: sec_candidates=64"),
        (lambda p: p.__setitem__("sec_cert_iters", 100), "protocol snapshot: sec_cert_iters=100"),
        (lambda p: p["verification"].pop("sec_policy"), "protocol snapshot: sec_policy is missing"),
        (
            lambda p: p["verification"].__setitem__("sec_policy", LRC_SEC_POLICY.to_dict()),
            "protocol snapshot: sec_policy_id='lrc_ranked_v1'",
        ),
    ],
)
@needs_v36_fixtures
def test_validator_rejects_a_v34_shaped_or_tampered_sec_snapshot(tmp_path, mutate, message) -> None:
    report = _strict_report()
    mutate(report["protocol"])
    _write_validated_run(tmp_path, report)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate(tmp_path)


@pytest.mark.parametrize("stage", ["search", "final"])
@needs_v36_fixtures
def test_validator_rejects_a_run_config_with_another_sec_policy(tmp_path, stage: str) -> None:
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
    config["verification"]["sec_policy"] = LRC_SEC_POLICY.to_dict()
    _write_json(path, config)
    with pytest.raises(ValueError, match=f"{where}: sec_policy_id='lrc_ranked_v1'"):
        _validate(tmp_path)
    del config["verification"]["sec_policy"]
    _write_json(path, config)
    with pytest.raises(ValueError, match=f"{where}: sec_policy is missing"):
        _validate(tmp_path)


@pytest.mark.parametrize("stage", ["search", "final"])
@needs_v36_fixtures
def test_validator_rejects_a_candidate_measured_under_another_schedule(tmp_path, stage: str) -> None:
    _write_validated_run(tmp_path, _strict_report())
    if stage == "search":
        candidate_dir = tmp_path / "turns" / "turn_0000" / "cycle" / "slow" / "candidates"
        where = "slow candidates"
    else:
        candidate_dir = tmp_path / "final_slow" / "T12" / "candidates"
        where = "final candidates"
    prefix = f"{where}: candidate.json results\\[0\\] basis Z"

    _write_json(
        candidate_dir / "candidate.json",
        _candidate_with_schedules(_schedule_blob(), _schedule_blob(sec_policy_id=LRC)),
    )
    with pytest.raises(ValueError, match=f"{prefix}: schedule sec_policy_id='lrc_ranked_v1'"):
        _validate(tmp_path)

    _write_json(
        candidate_dir / "candidate.json",
        _candidate_with_schedules(_schedule_blob(), _schedule_blob(candidates=8)),
    )
    with pytest.raises(ValueError, match=f"{prefix}: schedule candidates=8"):
        _validate(tmp_path)

    _write_json(
        candidate_dir / "candidate.json",
        _candidate_with_schedules(_schedule_blob(), _schedule_blob(cert_iters=100)),
    )
    with pytest.raises(ValueError, match=f"{prefix}: schedule cert_iters=100"):
        _validate(tmp_path)

    _write_json(
        candidate_dir / "candidate.json",
        _candidate_with_schedules(_schedule_blob(), _schedule_blob(depth=None)),
    )
    with pytest.raises(ValueError, match=f"{prefix}: schedule depth=None"):
        _validate(tmp_path)

    # An ok basis that recorded shots without any schedule metadata is not a
    # measurement (since aaa1ad7 a basis with NO shots is one that never ran
    # and is not audited, so the fixture must say it ran).
    payload = _candidate_with_schedules(_schedule_blob())
    del payload["results"][0]["basis"]["Z"]["memory_experiment"]
    payload["results"][0]["basis"]["Z"]["shots"] = 1000
    _write_json(candidate_dir / "candidate.json", payload)
    with pytest.raises(ValueError, match=f"{prefix}: memory_experiment is missing"):
        _validate(tmp_path)


# ---------------------------------------------------------------------------
# CLI
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


@pytest.mark.parametrize(
    ("command", "handler", "extra"),
    [
        ("build-dem-smoke", "build_dem_smoke_command", []),
        ("slow-verify-seed", "slow_verify_seed_command", []),
        ("slow-verify-queue", "slow_verify_queue_command", ["--promotion-queue", "q.jsonl", "--hardware", "atom", "--out", "o"]),
        ("export-stim-seed", "export_stim_seed_command", ["--out", "o"]),
        ("export-stim-queue", "export_stim_queue_command", ["--queue", "q.jsonl", "--out", "o"]),
        ("run-campaign", "run_campaign_command", ["--budget", "144", "--out", "o"]),
    ],
)
def test_every_circuit_building_subcommand_defaults_to_twobasis(
    monkeypatch, command: str, handler: str, extra: list[str]
) -> None:
    import omniqec_agentic.cli as cli

    captured = _capture_command(monkeypatch, handler)
    assert cli.main([command, *extra]) == 0
    assert captured["sec_policy"] == TWOBASIS
    assert cli._sec_policy_from_args(captured["_args"]) is DEFAULT_SEC_POLICY
    captured.clear()
    assert cli.main([command, *extra, "--sec-policy", LRC]) == 0
    assert captured["sec_policy"] == LRC
    assert cli._sec_policy_from_args(captured["_args"]) is LRC_SEC_POLICY
    with pytest.raises(SystemExit):
        cli.main([command, *extra, "--sec-policy", "interleaved_v9"])
    # A caller that never learned the flag gets the current protocol.
    assert cli._sec_policy_from_args(argparse.Namespace()) is DEFAULT_SEC_POLICY


def _campaign_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    args = dict(
        strict_turns=True, provider="openai", budget=144, out=str(tmp_path), weight_limit=None,
        families=None, p=0.005, noise_profile="uniform_paper_v1", sec_policy=TWOBASIS,
        turns=1, proposal_count=24, generations_per_turn=1, rng_seeds_per_program=20,
        slow_workers=127, slow_shots=25_000, shot_shards=16, final_shot_shards=128,
        racing=False, racing_pilot=5000, racing_batch=5000, racing_confirm=20_000,
        racing_max=50_000, decoders=None, circuit_timeout_s=None, build_timeout_s=None,
        decode_shard_timeout_s=None, distance_timeout_s=None, timeout_retries=None,
        decode_shot_budget_s=None, transient_retries=None, final_shots=25_000,
        feedback_mode="both", resume=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_run_campaign_command_plumbs_the_sec_policy_and_strict_refuses_lrc(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.workflow as workflow

    captured: dict[str, object] = {}

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)
        return {"run_id": "test", "stop": {}, "report": {"path": "report.json"}}

    monkeypatch.setattr(workflow, "run_campaign", fake_run_campaign)

    assert cli.run_campaign_command(_campaign_args(tmp_path, strict_turns=False, provider="dry_run")) == 0
    assert captured["verification"].sec_policy == DEFAULT_SEC_POLICY

    captured.clear()
    assert cli.run_campaign_command(
        _campaign_args(tmp_path, strict_turns=False, provider="dry_run", sec_policy=LRC)
    ) == 0
    assert captured["verification"].sec_policy == LRC_SEC_POLICY

    # A v3.4-era caller that never learned the flag gets the current protocol.
    captured.clear()
    args = _campaign_args(tmp_path, strict_turns=False, provider="dry_run")
    del args.sec_policy
    assert cli.run_campaign_command(args) == 0
    assert captured["verification"].sec_policy == DEFAULT_SEC_POLICY

    with pytest.raises(SystemExit, match="freezes --sec-policy at twobasis_nonint_v1"):
        cli.run_campaign_command(_campaign_args(tmp_path, sec_policy=LRC))


def test_slow_queue_and_seed_commands_plumb_the_sec_policy(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.evaluation as evaluation

    captured: dict[str, object] = {}

    def fake_queue(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    def fake_seed(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(evaluation, "evaluate_promotion_queue", fake_queue)
    monkeypatch.setattr(evaluation, "evaluate_slow_spec", fake_seed)
    queue = tmp_path / "q.jsonl"
    queue.write_text("", encoding="utf-8")
    assert cli.main([
        "slow-verify-queue", "--promotion-queue", str(queue), "--hardware", "atom",
        "--out", str(tmp_path / "o"), "--sec-policy", LRC,
    ]) == 0
    assert captured["verification"].sec_policy == LRC_SEC_POLICY
    captured.clear()
    assert cli.main(["slow-verify-queue", "--promotion-queue", str(queue), "--hardware", "atom", "--out", str(tmp_path / "o")]) == 0
    assert captured["verification"].sec_policy == DEFAULT_SEC_POLICY
    captured.clear()
    assert cli.main(["slow-verify-seed", "--sec-policy", LRC]) == 0
    assert captured["verification"].sec_policy == LRC_SEC_POLICY


def test_export_stim_commands_thread_the_sec_policy(monkeypatch, tmp_path) -> None:
    from omniqec_agentic import cli
    import omniqec_agentic.export_stim as export_stim

    captured: dict[str, object] = {}

    def fake_spec(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    def fake_queue(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(export_stim, "export_stim_circuits_for_spec", fake_spec)
    monkeypatch.setattr(export_stim, "export_stim_circuits_for_queue", fake_queue)
    assert cli.main(["export-stim-seed", "--out", str(tmp_path), "--sec-policy", LRC]) == 0
    assert captured["sec_policy"] == LRC_SEC_POLICY
    captured.clear()
    assert cli.main(["export-stim-queue", "--queue", "q.jsonl", "--out", str(tmp_path)]) == 0
    assert captured["sec_policy"] == DEFAULT_SEC_POLICY


# ---------------------------------------------------------------------------
# Worker / call-site plumbing
# ---------------------------------------------------------------------------


def _verification_request(sec_policy: SecPolicyConfig) -> dict:
    verification = VerificationConfig(
        sec_policy=sec_policy,
        exact_distance=dataclasses.replace(VerificationConfig().exact_distance, enabled=False, code_milp=False),
    )
    return json.loads(json.dumps(verification.to_dict()))


@pytest.mark.parametrize("policy", [DEFAULT_SEC_POLICY, LRC_SEC_POLICY], ids=[TWOBASIS, LRC])
def test_build_dem_request_passes_sec_policy_and_layer_blocks(monkeypatch, tmp_path, policy) -> None:
    from omniqec_agentic.evaluation import worker as worker_mod

    captured: dict[str, object] = {}

    def fake_pair(hx, hz, **kwargs):
        captured.update(kwargs)
        raise ValueError("captured")

    monkeypatch.setattr(worker_mod, "build_memory_experiment_pair", fake_pair)
    payload = worker_mod.run_build_dem_request({
        "mode": "build_dem",
        "spec": BB_SMALL_SPEC,
        "verification": _verification_request(policy),
        "matrices_path": str(tmp_path / "m.pkl"),
    })
    assert payload["status"] == "build_error" and "captured" in payload["failure_reason"]
    assert captured["sec_policy"] == policy
    expected = build_compiled_code(BB_SMALL_SPEC).layer_blocks
    assert captured["layer_blocks"] == expected
    assert captured["sector_sizes"] == [9, 9]


def test_distance_job_request_passes_sec_policy_and_layer_blocks(monkeypatch) -> None:
    from omniqec_agentic.evaluation import worker as worker_mod

    captured: dict[str, object] = {}

    def fake_pair(hx, hz, **kwargs):
        captured.update(kwargs)
        raise ValueError("captured")

    monkeypatch.setattr(worker_mod, "build_memory_experiment_pair", fake_pair)
    evidence = worker_mod.run_distance_job_request({
        "mode": "distance_job",
        "spec": HGP_SPEC,
        "verification": _verification_request(DEFAULT_SEC_POLICY),
    })
    assert evidence["status"] == "circuit_error"
    assert captured["sec_policy"] == DEFAULT_SEC_POLICY
    assert captured["layer_blocks"] is None  # hgp: fallback colouring

    # A request without the sec_policy block (a v3.4 worker request) runs the
    # default twobasis policy -- the snapshot check is what refuses v3.4 runs.
    captured.clear()
    request = _verification_request(DEFAULT_SEC_POLICY)
    del request["sec_policy"]
    worker_mod.run_distance_job_request({"mode": "distance_job", "spec": HGP_SPEC, "verification": request})
    assert captured["sec_policy"] == DEFAULT_SEC_POLICY


def test_every_package_call_site_passes_layer_blocks_and_sec_policy() -> None:
    """Any call site left on the old signature would silently run the default
    policy with the FALLBACK colouring (a different, unverified layer set for
    bb/lp). Every real call in the package must name both kwargs."""

    # parse, do not regex: a nested call in an argument (``p=float(x.value)``)
    # truncates a non-greedy ``\(.*?\)`` match and hides the real kwargs.
    sites: dict[str, int] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts or "tmp" in path.parts:
            continue  # tmp/ holds audit prototypes, not package call sites
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(PACKAGE_ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "build_memory_experiment_pair":
                continue
            sites[rel] = sites.get(rel, 0) + 1
            kwargs = {kw.arg for kw in node.keywords}
            where = f"{rel}:{node.lineno}"
            assert "layer_blocks" in kwargs, f"{where}: call without layer_blocks="
            assert "sec_policy" in kwargs, f"{where}: call without sec_policy="
    assert {"cli.py", "qualification.py", "export_stim.py", "evaluation/worker.py", "evaluation/pipeline.py"} <= set(sites)
    assert sites["evaluation/worker.py"] == 2 and sites["export_stim.py"] == 2


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_sec_ext_evidence_accepts_both_ranked_shapes_and_records_the_policy() -> None:
    lrc_like = SimpleNamespace(
        d_circ_ext_x=6, d_circ_ext_z=7, catastrophic=False, delta_one_eliminated=3, tau_a=4,
        stage3_group=8, metadata={"candidates": 64, "isd_iters": 100},
    )
    evidence = sec_ext_evidence(lrc_like)
    assert evidence["sec_policy_id"] == LRC
    assert (evidence["d_circ_ext"], evidence["tau_a"], evidence["stage3_group"]) == (6, 4, 8)
    assert "schedule_degraded_x" not in evidence

    twobasis_like = SimpleNamespace(
        d_circ_ext_x=9, d_circ_ext_z=None, catastrophic=False, delta_one_eliminated=2, tau_a=None,
        stage3_group=1,
        metadata={
            "sec_policy_id": TWOBASIS, "candidates": 16, "cert_iters": 400,
            "schedule_degraded_x": True, "schedule_degraded_z": False,
        },
    )
    evidence = sec_ext_evidence(twobasis_like)
    assert evidence["sec_policy_id"] == TWOBASIS
    assert evidence["status"] == "probe_upper_bound"
    assert (evidence["d_circ_ext"], evidence["d_circ_ext_z"], evidence["tau_a"]) == (9, None, None)
    assert (evidence["schedule_degraded_x"], evidence["schedule_degraded_z"]) == (True, False)
    assert json.loads(json.dumps(evidence)) == evidence
    assert sec_ext_evidence(None)["status"] == "not_available"
