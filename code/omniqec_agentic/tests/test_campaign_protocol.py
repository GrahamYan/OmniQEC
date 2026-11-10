from __future__ import annotations

import json

import pytest

from omniqec_agentic.config import (
    TelescopingDecoderConfig,
    DecoderName,
    RacingConfig,
    VerificationConfig,
    budget_policy,
)
from omniqec_agentic.verification.noise import NoiseSweepConfig
from omniqec_agentic.workflow.state import build_protocol_snapshot


def _snapshot(verification: VerificationConfig) -> dict:
    return build_protocol_snapshot(
        verification=verification,
        racing=RacingConfig(),
        max_turns=60,
        proposal_count=24,
        generations_per_turn=1,
        slow_workers=40,
        search_rounds=12,
    )


def test_campaign_freezes_one_p_for_search_and_final() -> None:
    verification = VerificationConfig(
        p_values=(0.005,),
        noise=NoiseSweepConfig.uniform((0.005,)),
    )

    protocol = _snapshot(verification)

    assert protocol["anchor_p"] == pytest.approx(0.005)
    assert protocol["final_p_values"] == [pytest.approx(0.005)]
    assert protocol["max_turns"] == 60
    # v3.6: an absent final_shots resolves to the budget policy (small-budget
    # table without n_budget), never to the racing confirm count.
    assert protocol["final_shots"] == budget_policy(None)["final_shots"] == 25_000
    assert protocol["shot_shards"] == 16
    assert protocol["final_shot_shards"] == 128
    assert protocol["strict_turns"] is False


def test_campaign_rejects_multi_p_sweep() -> None:
    verification = VerificationConfig(
        p_values=(0.003, 0.005),
        noise=NoiseSweepConfig.uniform((0.003, 0.005)),
    )

    with pytest.raises(ValueError, match="exactly one fixed noise point"):
        _snapshot(verification)


def test_formal_protocol_freezes_final_shots_and_strict_llm_contract() -> None:
    verification = VerificationConfig(
        p_values=(0.005,),
        noise=NoiseSweepConfig.uniform((0.005,)),
    )

    # v3.3 freezes racing OFF; requesting it under the strict profile is now
    # refused at snapshot time (see test_strict_snapshot_refuses_racing_at_launch).
    protocol = build_protocol_snapshot(
        verification=verification,
        racing=RacingConfig(),
        max_turns=60,
        proposal_count=24,
        generations_per_turn=1,
        slow_workers=38,
        search_rounds=12,
        final_shots=25000,
        strict_turns=True,
        required_provider="deepseek",
        required_model="deepseek-v4-pro",
        rng_seeds_per_program=20,
    )

    assert protocol["final_shots"] == 25000
    assert protocol["racing"]["shots_confirm"] == 20000
    assert protocol["strict_turns"] is True
    assert protocol["fixed_program_count"] is True
    assert protocol["required_provider"] == "deepseek"
    assert protocol["required_model"] == "deepseek-v4-pro"
    assert protocol["rng_seeds_per_program"] == 20


def test_strict_protocol_requires_provider_and_model() -> None:
    verification = VerificationConfig(
        p_values=(0.005,),
        noise=NoiseSweepConfig.uniform((0.005,)),
    )

    with pytest.raises(ValueError, match="requires required_provider and required_model"):
        build_protocol_snapshot(
            verification=verification,
            racing=RacingConfig(),
            max_turns=60,
            proposal_count=24,
            generations_per_turn=1,
            slow_workers=38,
            strict_turns=True,
        )


def test_snapshot_freezes_the_v36_profile_shards_and_its_quotas() -> None:
    protocol = _snapshot(
        VerificationConfig(p_values=(0.005,), noise=NoiseSweepConfig.uniform((0.005,)))
    )

    assert protocol["protocol_profile"] == "telescoping_structural_v3_7"
    assert protocol["shot_shards"] == 16
    assert protocol["final_shot_shards"] == 128
    # v3.4 execution policy: per-shot budget, no timeout replay, one transient
    # replay of a killed / DEM-less worker.
    assert protocol["verification"]["decode_shot_budget_s"] == 8.0
    assert protocol["verification"]["decode_shard_timeout_s"] == 1800.0
    assert protocol["verification"]["timeout_retries"] == 0
    assert protocol["verification"]["transient_retries"] == 1
    # v3.6: ONE decoder, the telescoping decoder, with its full frozen blob
    # (budget independent) recorded twice: in the verification config and as
    # the top-level `telescoping` key next to `decoder`.
    assert protocol["verification"]["decoders"] == [DecoderName.TELESCOPING.value]
    assert protocol["decoder"] == "telescoping_cpu"
    assert protocol["verification"]["decoder_build"]["telescoping"] == TelescopingDecoderConfig().to_dict()
    assert protocol["telescoping"] == protocol["verification"]["decoder_build"]["telescoping"]
    assert protocol["telescoping"]["nc_policy"] == "fail" and protocol["telescoping"]["s4_enabled"] is False
    assert "cascade" not in protocol
    assert protocol["budget_policy"] == budget_policy(None)
    rounds = protocol["rounds_policy"]
    assert (rounds["search"], rounds["search_rounds"]) == ("max_Tc_d", 12)
    # The final stage is fixed-T too: the retired "T=d" label ran
    # T=max(3, d_proxy), i.e. a depth chosen from the search's own heuristic.
    assert (rounds["final"], rounds["final_rounds"]) == ("max_Tc_d", 12)
    assert rounds["final_rng_namespace"] == "final_confirmation_v1"
    retention = protocol["structural_retention"]
    assert retention["reducer_id"] == "structural_proxy_pareto_v3_weight"
    assert retention["coordinate_cap"] == 2
    assert retention["proxy_seeds"] == [0, 1, 2, 3]
    # v3.7: 2 structural endpoints + 4 runtime (expected-plpr order); 16 parents.
    assert (retention["structural_slow_slots"], retention["runtime_slow_slots"]) == (2, 4)
    assert (retention["structural_parents"], retention["runtime_parents"], retention["diversity_parents"]) == (6, 6, 4)
    assert retention["parent_cap"] == 16 and retention["elites_limit"] == 24
    assert retention["runtime_order"] == "expected_plpr_v1"
    assert (retention["final_structural"], retention["final_runtime"]) == (2, 4)
    assert retention["min_promotion_k"] == 2
    assert list(retention["structural_axes"]) == ["k", "d_search", "-max_check_weight"]
    # 2026-11-10 paper alignment: the fast-loop cap w_max <= 8 (v3.7 itself ran without one).
    assert protocol["verification"]["max_check_weight"] == 8
    assert protocol["promote_top"] == retention["slow_candidate_cap"] == 6


def test_snapshot_rejects_a_queue_cap_that_drifts_from_the_quota() -> None:
    import dataclasses

    verification = VerificationConfig(
        p_values=(0.005,), noise=NoiseSweepConfig.uniform((0.005,))
    )
    with pytest.raises(ValueError, match="promote_top must equal"):
        _snapshot(dataclasses.replace(verification, promote_top=8))


def test_a_pre_v33_checkpoint_cannot_be_resumed() -> None:
    """The protocol changed in ways a resumed run cannot reconcile: a different
    decoder, a different d_search, different retention rules, a different final
    stage. Resume must refuse rather than continue with mixed evidence."""

    from omniqec_agentic.workflow.checkpoint import validate_resume_protocol

    current = _snapshot(
        VerificationConfig(p_values=(0.005,), noise=NoiseSweepConfig.uniform((0.005,)))
    )
    legacy = json.loads(json.dumps(current))
    legacy.pop("protocol_profile")
    legacy.pop("structural_retention")
    legacy["rounds_policy"] = {"search": "fixed_Tc", "search_rounds": 12, "final": "T=d"}
    legacy["verification"]["decoders"] = ["beam", "lsd"]

    with pytest.raises(ValueError, match="protocol snapshot mismatch"):
        validate_resume_protocol(legacy, current)
    with pytest.raises(ValueError, match="no protocol snapshot"):
        validate_resume_protocol(None, current)


def test_shard_count_is_frozen_and_resume_rejects_rng_repartition() -> None:
    from omniqec_agentic.workflow.checkpoint import validate_resume_protocol

    current = _snapshot(
        VerificationConfig(p_values=(0.005,), noise=NoiseSweepConfig.uniform((0.005,)))
    )
    old_partition = json.loads(json.dumps(current))
    old_partition["shot_shards"] = 1

    with pytest.raises(ValueError, match="shot_shards"):
        validate_resume_protocol(old_partition, current)


@pytest.mark.parametrize("shot_shards", [0, -1])
def test_protocol_rejects_non_positive_shard_count(shot_shards: int) -> None:
    with pytest.raises(ValueError, match="shot_shards must be >= 1"):
        build_protocol_snapshot(
            verification=VerificationConfig(),
            racing=RacingConfig(),
            max_turns=1,
            proposal_count=8,
            generations_per_turn=1,
            slow_workers=4,
            shot_shards=shot_shards,
        )


def test_protocol_rejects_more_shards_than_search_or_final_shots() -> None:
    with pytest.raises(ValueError, match="search shots per basis"):
        build_protocol_snapshot(
            verification=VerificationConfig(shots=8),
            racing=RacingConfig(),
            max_turns=1,
            proposal_count=8,
            generations_per_turn=1,
            slow_workers=4,
            shot_shards=16,
        )

    with pytest.raises(ValueError, match="final shots per basis"):
        build_protocol_snapshot(
            verification=VerificationConfig(shots=20_000),
            racing=RacingConfig(),
            max_turns=1,
            proposal_count=8,
            generations_per_turn=1,
            slow_workers=4,
            shot_shards=16,
            final_shots=8,
        )


def test_nonformal_shard_count_retags_the_protocol() -> None:
    protocol = build_protocol_snapshot(
        verification=VerificationConfig(),
        racing=RacingConfig(),
        max_turns=1,
        proposal_count=8,
        generations_per_turn=1,
        slow_workers=4,
        shot_shards=8,
    )

    assert protocol["shot_shards"] == 8
    assert protocol["protocol_profile"].endswith("+shards8")


def test_strict_protocol_rejects_nonformal_shard_count() -> None:
    with pytest.raises(ValueError, match="freezes shot_shards=16"):
        build_protocol_snapshot(
            verification=VerificationConfig(),
            racing=RacingConfig(),
            max_turns=1,
            proposal_count=8,
            generations_per_turn=1,
            slow_workers=4,
            shot_shards=8,
            strict_turns=True,
            required_provider="deepseek",
            required_model="deepseek-v4-pro",
        )


def test_protocol_rejects_zero_final_shots_instead_of_using_racing_default() -> None:
    verification = VerificationConfig(
        p_values=(0.005,),
        noise=NoiseSweepConfig.uniform((0.005,)),
    )

    with pytest.raises(ValueError, match="final_shots must be >= 1"):
        build_protocol_snapshot(
            verification=verification,
            racing=RacingConfig(),
            max_turns=60,
            proposal_count=24,
            generations_per_turn=1,
            slow_workers=38,
            final_shots=0,
        )
