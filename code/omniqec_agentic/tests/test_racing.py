"""Racing controller regressions (03 §14.3): elimination round + stopped_by,
thresholds-off degenerates to fixed shots, order-independence of decisions."""

from __future__ import annotations

import dataclasses

import pytest

from omniqec_agentic.config import RacingConfig
from omniqec_agentic.evaluation.sampling import (
    RacingThresholds,
    merge_shard_results,
    racing_decision,
    racing_pilot_shards,
    racing_shard_stream,
)

CFG = RacingConfig(
    enabled=True, shots_pilot=1000, batch=500, shots_confirm=3000, shots_max=5000
)
BOTH = RacingThresholds(hof_plpr_ub=0.01, baseline_plpr_ub=0.02)


def test_pilot_never_eliminates() -> None:
    decision = racing_decision(
        shots=500, basis_failures={"X": 400, "Z": 400}, plpr_ci_low=0.9,
        config=CFG, thresholds=BOTH,
    )
    assert decision == {"action": "continue", "stopped_by": None, "confirmed": False}


def test_elimination_when_both_arms_crossed() -> None:
    decision = racing_decision(
        shots=1000, basis_failures={"X": 300, "Z": 280}, plpr_ci_low=0.05,
        config=CFG, thresholds=BOTH,
    )
    assert decision["action"] == "stop"
    assert decision["stopped_by"] == "racing"
    assert decision["confirmed"] is False


def test_single_arm_survival_or_semantics() -> None:
    # LB above the HoF arm but below the baseline arm: survives.
    decision = racing_decision(
        shots=1000, basis_failures={"X": 30, "Z": 28}, plpr_ci_low=0.015,
        config=CFG, thresholds=BOTH,
    )
    assert decision["action"] == "continue"


def test_unavailable_arms_never_eliminate_and_confirm_at_fixed_shots() -> None:
    # Early campaign (no thresholds): degenerates to fixed shots_confirm.
    for shots in (1000, 1500, 2500):
        decision = racing_decision(
            shots=shots, basis_failures={"X": 500, "Z": 500}, plpr_ci_low=0.9,
            config=CFG, thresholds=RacingThresholds(),
        )
        assert decision["action"] == "continue"
    decision = racing_decision(
        shots=3000, basis_failures={"X": 500, "Z": 500}, plpr_ci_low=0.9,
        config=CFG, thresholds=RacingThresholds(),
    )
    assert decision == {"action": "stop", "stopped_by": None, "confirmed": True}


def test_max_errors_needs_both_bases() -> None:
    cfg = RacingConfig(
        enabled=True, shots_pilot=1000, batch=500, shots_confirm=3000,
        shots_max=5000, max_errors=100,
    )
    one_sided = racing_decision(
        shots=1500, basis_failures={"X": 150, "Z": 10}, plpr_ci_low=0.001,
        config=cfg, thresholds=BOTH,
    )
    assert one_sided["action"] == "continue"
    both = racing_decision(
        shots=1500, basis_failures={"X": 150, "Z": 120}, plpr_ci_low=0.001,
        config=cfg, thresholds=BOTH,
    )
    assert both == {"action": "stop", "stopped_by": "max_errors", "confirmed": True}


def test_shard_stream_preregistration() -> None:
    assert racing_shard_stream(CFG) == [500] * 10
    assert racing_pilot_shards(CFG) == 2
    ragged = RacingConfig(
        enabled=True, shots_pilot=700, batch=500, shots_confirm=1200, shots_max=1200
    )
    assert racing_shard_stream(ragged) == [500, 500, 200]
    assert racing_pilot_shards(ragged) == 2


def test_decision_inputs_are_shard_order_independent() -> None:
    shards = [
        {"status": "ok", "shots": 500, "failures": 3, "runtime_s": 1.0, "metrics": {}},
        {"status": "ok", "shots": 500, "failures": 7, "runtime_s": 2.0, "metrics": {}},
        {"status": "ok", "shots": 500, "failures": 5, "runtime_s": 3.0, "metrics": {}},
    ]
    forward = merge_shard_results(shards, decoder="beam")
    backward = merge_shard_results(list(reversed(shards)), decoder="beam")
    for key in ("shots", "failures", "logical_error_rate", "status"):
        assert forward[key] == backward[key]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RacingConfig(shots_pilot=2000, shots_max=1000)
    with pytest.raises(ValueError):
        RacingConfig(shots_confirm=9000, shots_max=5000)


def test_stream_exhaustion_stops_instead_of_crashing(monkeypatch) -> None:
    """Systematically failing primary-decoder shards must end with
    stopped_by="stream_exhausted", not an IndexError past the pre-registered
    stream (regression: NSCC smoke 2026-08-17, beam unavailable)."""

    from omniqec_agentic.config import VerificationConfig
    from omniqec_agentic.evaluation import pool as pool_mod
    from omniqec_agentic.examples.seed_programs import SEED_BB
    from omniqec_agentic.sandbox import run_construct_source

    # The primary decoder is whichever one leads the configured set; naming it
    # here rather than hard-coding "beam" keeps the regression meaningful after
    # a protocol change to the decoder set.
    primary = VerificationConfig().decoders[0].value

    def fake_run_worker_request(request, timeout_s=None):
        mode = request.get("mode")
        if mode == "build_dem":
            # the T the pool derives for this candidate (max(12, d)); a build reporting another T is
            # a rounds mismatch and becomes a build_error since 2026-10-27 (pool.check_build_rounds)
            return {"status": "ok", "rounds_x": 12, "rounds_z": 12}, {"timeout": False}
        if mode == "distance_job":
            return {"status": "ok"}, {"timeout": False}
        assert mode == "shard_dem"
        unit = request["units"][0]
        shots = int(request["shots"])
        if unit["decoder"] == primary:  # primary decoder always fails
            cell = {
                "status": "decoder_error", "shots": 0, "failures": 0,
                "logical_error_rate": None, "runtime_s": 0.0,
                "failure_reason": "beam unavailable (test)",
            }
        else:
            cell = {
                "status": "ok", "shots": shots, "failures": 1,
                "logical_error_rate": 1.0 / shots, "runtime_s": 0.0,
                "metrics": {
                    "requested_shots": shots, "actual_shots": shots,
                    "sampling_complete": True, "stopped_by_max_errors": False,
                },
            }
        return (
            {"status": "ok", "results": {unit["decoder"]: {unit["basis"]: cell}}},
            {"timeout": False},
        )

    monkeypatch.setattr(pool_mod, "run_worker_request", fake_run_worker_request)
    spec = run_construct_source(SEED_BB, program_id="seed_bb", seed=0).constructed.spec
    racing = RacingConfig(
        enabled=True, shots_pilot=1000, batch=1000, shots_confirm=2000, shots_max=4000
    )
    payloads = pool_mod.evaluate_queue_with_jobs(
        [{"program_id": "seed_bb", "spec": spec}],
        verification=VerificationConfig(shots=2000),
        hardware=None,
        seed=0,
        batch_size=256,
        prefer_coloring=True,
        workers=2,
        shot_shards=1,
        racing=racing,
        racing_thresholds=None,
    )

    race = payloads[0]["racing"]
    assert race["stopped_by"] == "stream_exhausted"
    assert race["confirmed"] is False
    assert race["shards_consumed"] == len(racing_shard_stream(racing))
    assert race["shots_per_basis"] == 0  # actual decoded shots, not requested


def test_formal_early_stop_policy_follows_the_search_shots() -> None:
    """2026-09-11 (pilot10x_v1): pilot 2000 in 250-shot shards, confirm = max
    = the point's search shots, 10x threshold (epsilon 9) against the
    hall-of-fame best only, no max_errors cap."""

    from omniqec_agentic.config import (
        RACING_POLICY_ID,
        RACING_POLICY_V1_ID,
        RACING_POLICY_V2_ID,
        RACING_POLICY_V3_ID,
        RACING_POLICY_V4_ID,
        formal_racing,
        formal_racing_policy_id,
        formal_racing_v1,
        formal_racing_v2,
        formal_racing_v3,
        formal_racing_v4,
    )

    small = formal_racing(25_000)
    # 2026-09-11 evening: v2 = v1 + a 100-error cap per basis (user decision:
    # "a few dozen errors screen a candidate; only the final stage needs
    # 25k/25k"). 2026-09-14: v3 = v2 with the cap decided at every shard,
    # the pilot included (a code failing every shot stops at shard 1).
    # 2026-09-15 morning: the formal policy is v3 again. v4 (v3 + a
    # decoder-cost cap) ran overnight, mis-measured warm-up as decode cost and
    # cut legitimate codes; it stays registered so those turns validate.
    assert small.enabled and small.policy_id == RACING_POLICY_ID == RACING_POLICY_V3_ID == "pilot10x_maxerr100_v3"
    assert (small.shots_pilot, small.batch, small.shots_confirm, small.shots_max) == (2000, 250, 25_000, 25_000)
    assert small.epsilon == 9.0 and small.baseline_arm is False and small.max_errors == 100
    assert small.max_errors_in_pilot is True and small.cost_cap_s_per_shot_per_qubit is None
    v3 = formal_racing_v3(25_000)
    assert v3 == small
    v4 = formal_racing_v4(25_000)
    assert v4.policy_id == RACING_POLICY_V4_ID == "pilot10x_maxerr100_cost_v4"
    assert (v4.cost_cap_s_per_shot_per_qubit, v4.cost_cap_floor_s_per_shot, v4.cost_cap_min_shots) == (0.10, 8.0, 250)
    assert dataclasses.replace(v3, cost_cap_s_per_shot_per_qubit=0.10, cost_cap_floor_s_per_shot=8.0, cost_cap_min_shots=250, policy_id=RACING_POLICY_V4_ID) == v4
    assert formal_racing_policy_id(v3.to_dict()) == RACING_POLICY_V3_ID
    assert formal_racing_policy_id(v4.to_dict()) == RACING_POLICY_V4_ID
    # a v3 block frozen before the v4 fields existed still resolves to v3
    legacy_v3 = {k: v for k, v in v3.to_dict().items() if not k.startswith("cost_cap_")}
    assert formal_racing_policy_id(legacy_v3) == RACING_POLICY_V3_ID
    v1 = formal_racing_v1(25_000)
    assert v1.policy_id == RACING_POLICY_V1_ID == "pilot10x_v1" and v1.max_errors is None
    v2 = formal_racing_v2(25_000)
    assert v2.policy_id == RACING_POLICY_V2_ID == "pilot10x_maxerr100_v2" and v2.max_errors == 100
    assert v2.max_errors_in_pilot is False
    assert dataclasses.replace(v1, max_errors=100, policy_id=RACING_POLICY_V2_ID) == v2
    assert dataclasses.replace(v2, max_errors_in_pilot=True, policy_id=RACING_POLICY_V3_ID) == v3
    assert formal_racing_policy_id(small.to_dict()) == RACING_POLICY_ID
    assert formal_racing_policy_id(v2.to_dict()) == RACING_POLICY_V2_ID
    assert formal_racing_policy_id(v1.to_dict()) == RACING_POLICY_V1_ID
    # a v2 block frozen before the v3 field existed still resolves to v2
    legacy_v2 = {k: v for k, v in v2.to_dict().items() if k != "max_errors_in_pilot"}
    assert formal_racing_policy_id(legacy_v2) == RACING_POLICY_V2_ID
    # v4 decoder-cost cap: 0.10 s/shot/qubit with an 8 s/shot floor, judged
    # only once 250 shots/basis landed; the stop is never a confirmation
    from omniqec_agentic.evaluation.sampling import decoder_cost_cap_s_per_shot, decoder_cost_decision
    assert decoder_cost_cap_s_per_shot(v4, 144) == 14.4 and decoder_cost_cap_s_per_shot(v4, 30) == 8.0
    assert decoder_cost_cap_s_per_shot(v3, 144) is None and decoder_cost_cap_s_per_shot(small, 144) is None
    assert decoder_cost_decision(cost_s_per_shot=20.0, shots=250, n=144, config=v4) == {
        "action": "stop", "stopped_by": "decoder_cost", "confirmed": False,
        "cost_s_per_shot": 20.0, "cost_cap_s_per_shot": 14.4,
    }
    assert decoder_cost_decision(cost_s_per_shot=20.0, shots=249, n=144, config=v4) is None   # too early to judge
    assert decoder_cost_decision(cost_s_per_shot=14.4, shots=2000, n=144, config=v4) is None  # at the cap: allowed
    assert decoder_cost_decision(cost_s_per_shot=20.0, shots=2000, n=144, config=v3) is None     # v3 has no cap
    assert decoder_cost_decision(cost_s_per_shot=None, shots=2000, n=144, config=v4) is None
    # inside the pilot: v3 stops a code that already has the cap in EVERY
    # basis, v2/v1 keep decoding; the 10x arm still waits for the pilot
    early = dict(shots=250, basis_failures={"X": 120, "Z": 101}, plpr_ci_low=0.5, thresholds=RacingThresholds(hof_plpr_ub=1e-4))
    assert racing_decision(config=small, **early) == {"action": "stop", "stopped_by": "max_errors", "confirmed": True}
    assert racing_decision(config=v2, **early) == {"action": "continue", "stopped_by": None, "confirmed": False}
    assert racing_decision(config=v1, **early) == {"action": "continue", "stopped_by": None, "confirmed": False}
    half = dict(early, basis_failures={"X": 120, "Z": 99})
    assert racing_decision(config=small, **half) == {"action": "continue", "stopped_by": None, "confirmed": False}
    assert formal_racing_policy_id({**small.to_dict(), "epsilon": 2.0}) is None
    assert formal_racing_policy_id({"enabled": False}) is None
    assert racing_pilot_shards(small) == 8 and len(racing_shard_stream(small)) == 100
    large = formal_racing(5_000)
    assert (large.shots_pilot, large.shots_confirm, large.shots_max) == (2000, 5_000, 5_000)
    assert RacingConfig.from_dict(small.to_dict()) == small
    # 10x rule: a candidate at exactly 10x the best's upper bound stops; 9x survives.
    thresholds = RacingThresholds(hof_plpr_ub=1e-4, baseline_plpr_ub=None)
    stop = racing_decision(shots=2000, basis_failures={"X": 50, "Z": 50}, plpr_ci_low=1e-3,
                           config=small, thresholds=thresholds)
    assert stop == {"action": "stop", "stopped_by": "racing", "confirmed": False}
    go = racing_decision(shots=2000, basis_failures={"X": 50, "Z": 50}, plpr_ci_low=9e-4,
                         config=small, thresholds=thresholds)
    assert go == {"action": "continue", "stopped_by": None, "confirmed": False}
    # Nothing to compare against (empty hall of fame): never eliminates; a
    # candidate with fewer than 100 failures per basis confirms at the full shots...
    none = racing_decision(shots=25_000, basis_failures={"X": 60, "Z": 60}, plpr_ci_low=0.5,
                           config=small, thresholds=RacingThresholds())
    assert none == {"action": "stop", "stopped_by": None, "confirmed": True}
    # ...and one that already has 100 failures in EVERY basis stops there,
    # confirmed (v2's cap); v1 would have marched on to the full shots.
    capped = racing_decision(shots=2_250, basis_failures={"X": 100, "Z": 130}, plpr_ci_low=0.5,
                             config=small, thresholds=RacingThresholds())
    assert capped == {"action": "stop", "stopped_by": "max_errors", "confirmed": True}
    one_basis = racing_decision(shots=2_250, basis_failures={"X": 100, "Z": 40}, plpr_ci_low=0.5,
                                config=small, thresholds=RacingThresholds())
    assert one_basis == {"action": "continue", "stopped_by": None, "confirmed": False}
    uncapped = racing_decision(shots=2_250, basis_failures={"X": 100, "Z": 130}, plpr_ci_low=0.5,
                               config=v1, thresholds=RacingThresholds())
    assert uncapped == {"action": "continue", "stopped_by": None, "confirmed": False}
