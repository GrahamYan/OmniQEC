"""Campaign state schema and protocol snapshot (03 §2.2, §3.1).

No langgraph import here: the state is a plain TypedDict (light pointers and
metrics only — matrices/DEMs/shots live on the filesystem), and CampaignSpec
is a strictly validated pydantic model whose field whitelist is the paper's
Fig.1 intake surface."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omniqec_agentic import ranking
from omniqec_agentic.config import (
    ROUNDS_POLICY_ID,
    SEED_PROGRAMS_FIRST_TURN,
    SEED_PROGRAMS_POLICIES,
    DEFAULT_SEC_POLICY,
    TELESCOPING_PROTOCOL_FIELDS,
    CodeFamily,
    DecoderName,
    RacingConfig,
    VerificationConfig,
    budget_policy,
)

ALL_FAMILIES = tuple(family.value for family in CodeFamily)
DEFAULT_WEIGHT_LIMIT: int | None = None  # v3.7: no check-weight cap by default
# Frozen protocol identity (telescoping_structural_v3_6, docs/06 §1, user
# decision 2026-09-03): ONE decoder -- the Mitten paper's open-source
# telescoping decoder (vendored package, CPU configuration) on the GARI
# transform of the two-basis DEM -- everywhere (search, runtime hall of fame,
# fixed-T final). Both are snapshot keys (`protocol_profile`, `decoder`); a
# v3.5 checkpoint (bposd_structural_v3_5 / decoder bposd, no `budget_policy`
# key) or a checkpoint of the never-launched self-built cascade
# (cascade_structural_v3_6 / decoder relay_cascade_gari, `cascade` key) is
# refused on resume by `assert_resumable_protocol` with a message naming the
# cause.
FORMAL_PROFILE = "telescoping_structural_v3_7"
FORMAL_DECODER = DecoderName.TELESCOPING.value
# Profile prefixes of retired protocols whose checkpoints must never be
# resumed into a v3.6 campaign (different decoder => different numbers).
# cascade_structural (the self-built Relay-BP cascade) was replaced by the
# telescoping decoder before any NSCC launch (2026-09-03).
LEGACY_PROFILE_PREFIXES = ("bposd_structural", "cascade_structural")
# Frozen partition counts (unchanged since bposd_structural_v3_4). Search shards drive the
# per-turn slow stage; final shards drive fixed-T confirmation (200k shots
# per basis at full budget needs 128 shards to fit the per-shot watchdog).
# Both select the index-derived RNG streams, so both are protocol fields.
FORMAL_SHOT_SHARDS = 16
FORMAL_FINAL_SHOT_SHARDS = 128
# Frozen noise model (since bposd_structural_v3_5, 2026-09-02): the OmniQEC paper
# SI C uniform circuit-level depolarizing model — depolarizing after every
# reset and CX, a flip before every measurement, NO idle noise — at ONE
# physical error rate. Both are snapshot keys (`noise_profile`,
# `noise_values`), so a v3.4 checkpoint (uniform_legacy_v1 idle noise, no
# noise_values key) refuses to resume into a v3.5 campaign. Mirrored by
# hpc/validate_fig2_campaign.py; the tests pin the copies together.
FORMAL_NOISE_PROFILE = "uniform_paper_v1"
FORMAL_P = 0.005
# Frozen SEC policy (twobasis_nonint_v1 since v3.5; v3.6 adds the multi-seed
# certification fields sec_cert_seeds / sec_cert_iters_scale / sec_refine_top
# / sec_refine_iters, docs/06 §3 M1): both bases measured every round, NO
# interleaving, K=16 candidate layer orders per basis (config.DEFAULT_SEC_POLICY).
# Snapshot keys `sec_policy_id` / `sec_candidates` / `sec_cert_iters` plus the
# full blob (`sec_policy`, also inside `verification.sec_policy`) make a v3.4
# LRC checkpoint refuse to resume; the standalone validator mirrors the constant.
FORMAL_SEC_POLICY = "twobasis_nonint_v1"


class CampaignSpec(BaseModel):
    """Fig.1 intake whitelist (03 §3.1). Anything outside this schema is an
    explicit rejection at intake, never a silent ignore."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qubit_budget: int = Field(gt=0, description="TOTAL qubit budget point (data + ancilla)")
    # v3.7: None = no ADDITIONAL check-weight cap (the formal default; the
    # fast loop's own cap is VerificationConfig.max_check_weight, 8 since the
    # 2026-11-10 paper alignment); an int is a hardware-motivated tightening
    # (the tighter of the two applies) and makes the run a declared variant.
    weight_limit: int | None = Field(default=DEFAULT_WEIGHT_LIMIT)
    connectivity: Literal["all-to-all"] = "all-to-all"
    families: tuple[str, ...] = ALL_FAMILIES
    objective: Literal["plpr_at_anchor"] = "plpr_at_anchor"
    # Recorded but explicitly NOT a search target (03 §3.1).
    secondary_metrics: tuple[str, ...] = ()

    @field_validator("weight_limit")
    @classmethod
    def _tighten_only(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) < 3:
            raise ValueError("weight_limit must be >= 3 (or None for no cap)")
        return int(value)

    @field_validator("families")
    @classmethod
    def _known_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - set(ALL_FAMILIES)
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}; valid: {ALL_FAMILIES}")
        if not value:
            raise ValueError("at least one family required")
        return tuple(dict.fromkeys(value))


class CampaignState(TypedDict, total=False):
    run_id: str
    turn: int
    run_dir: str
    spec: dict[str, Any]            # frozen CampaignSpec dump
    protocol: dict[str, Any]        # frozen protocol snapshot (see below)
    archive_path: str
    queue_path: str
    manual_dir: str                 # Milestone B (manual/); reserved
    ledger_path: str                # Milestone B (learning layer); reserved
    fingerprint_index_path: str     # Milestone B; reserved
    hof_summary: list[dict[str, Any]]
    # Structural retention cache (v3.2 §4.2): the cross-turn (k, d_search)
    # proxy-Pareto elites. Checkpointed like hof_summary; a candidate survives
    # here without being regenerated or re-measured.
    structural_proxy_elites: list[dict[str, Any]]
    # Cursor over the lossless structural ledger (audit fixes #3/#4). The
    # ledger itself — EVERY structurally eligible row ever seen — is an
    # append-only JSONL store at <run_dir>/structural_ledger.jsonl, never
    # loaded whole into memory and never checkpointed: at 60x24x20 candidates
    # the full rows are hundreds of MiB. The state keeps only the hash set
    # (dedup for appends) plus the reduced elites above, which reconcile
    # recomputes each turn by STREAMING the store — still reduce(everything
    # ever), so retention stays independent of candidate arrival order.
    structural_ledger_hashes: list[str]
    # Candidates the measurement protocol cannot evaluate at all
    # (schedule_catastrophic, or v3.4 decode_infeasible: decode exceeded the
    # frozen per-shot budget). They keep their structural standing but stop
    # consuming slow slots; they never reach the evidence ledger, so without
    # this set they would be re-queued every turn.
    unmeasurable_hashes: list[str]
    # v3.6 B5: subset of unmeasurable_hashes eligible ONCE for the final
    # stage (retry_at_final) -- search-stage decode_infeasible candidates
    # re-measured with the final budgets (128 shards, final-stage watchdog).
    decode_infeasible_hashes: list[str]
    baseline_summary: dict[str, Any]
    bandit_stats: dict[str, Any]    # Milestone B; reserved
    budget: dict[str, Any]          # turns / wallclock_s / tokens / slow_shots ledgers
    plan: dict[str, Any] | None
    transcript_ptr: str
    rng_cursor: dict[str, Any]
    stop: dict[str, Any]
    last_cycle: dict[str, Any]      # light pointers into the last turn's artifacts
    slow_context_ptr: str           # slow-aware prompt overlay for the next turn
    final_slow: dict[str, Any]
    report: dict[str, Any]


def build_protocol_snapshot(
    *,
    verification: VerificationConfig,
    racing: RacingConfig,
    max_turns: int,
    proposal_count: int,
    generations_per_turn: int,
    slow_workers: int,
    shot_shards: int = FORMAL_SHOT_SHARDS,
    final_shot_shards: int = FORMAL_FINAL_SHOT_SHARDS,
    tool_budget: int = 32,
    feedback_mode: str = "both",
    final_shots: int | None = None,
    search_rounds: int | None = None,
    strict_turns: bool = False,
    required_provider: str | None = None,
    required_model: str | None = None,
    rng_seeds_per_program: int = 1,
    n_budget: int | None = None,
    reference_codes: dict[str, Any] | None = None,
    final_decode_shard_timeout_cap_s: float | None = None,
    search_decode_shard_timeout_cap_s: float | None = None,
    fixed_program_count: bool | None = None,
    promotion_max_check_weight: int | None = None,
    campaign_id: str | None = None,
    ablation: str | None = None,
    search_baseline: dict[str, Any] | None = None,
    final_stage_id: str | None = None,
    final_cumulative_shots: int | None = None,
    seed_programs: str = SEED_PROGRAMS_FIRST_TURN,
) -> dict[str, Any]:
    """The frozen protocol (03 §2.2): everything the policy agent may NOT
    touch. --resume refuses to continue when this snapshot changed.

    ``final_stage_id`` (2026-11-10, paper alignment): ``paper_top5_cumulative_v1`` freezes the paper's
    final stage (Appendix D3, Table VII; evaluation/final_stage.py) under key ``final_stage`` --
    run_campaign passes it for every new run. None leaves the snapshot exactly as earlier versions wrote
    it (no key; the final stage of those versions). ``final_cumulative_shots`` overrides Table VII's
    cumulative per-basis limit (a declared variant, refused under strict_turns). A run under an exact
    check-weight contract (``verification.exact_check_weight``, Fig. 4 c) takes the paper's fixed-weight
    final stage ``paper_fig4c_top6_confirm_v1`` instead (2026-11-10 paper revision: six finalists, the
    initial confirmation only): the contract selects it, whichever paper id the caller names.

    ``seed_programs`` (paper alignment 2026-11-10, appendix D1): "first_turn"
    runs the five hand-written seed programs in turn 0 only; the key is always
    written, so a checkpoint of the earlier every-turn code (no key) refuses
    to resume under this one instead of silently changing behaviour mid-run.

    ``search_baseline`` (2026-11-06): the no-LLM search baselines
    (omniqec_agentic/baselines) freeze their method, families, sampler and
    model settings under this key; None leaves the snapshot exactly as the
    LLM search writes it (no key).

    ``ablation`` (S3, 2026-10-23): "no_slow" / "no_feedback" freeze the
    search-feedback ablation into the snapshot (key ``ablation``); None or
    "full" leaves the snapshot exactly as the formal flow writes it (no key).

    v3.6 (docs/06 §1.4, user decision 2026-09-03): ``n_budget`` (the TOTAL
    qubit budget point) selects the budget policy -- the search/final shots
    per basis, and nothing else: the telescoping decoder is iteration
    bounded and carries no per-point limit, so its blob is the same at every
    point. Every code at one budget point gets the same shots (fairness);
    points >= 300 relax them (5k/50k). The policy is applied HERE:
    ``verification.shots`` and ``final_shots`` follow it unless the caller
    explicitly overrides them outside strict mode (an override under
    strict_turns is refused), and the resolved policy is a snapshot key.
    ``n_budget=None`` (legacy callers) resolves the small-budget policy.
    """

    import dataclasses

    from omniqec_agentic.evaluation.pool import resolve_workers

    policy = budget_policy(n_budget)
    # v3.3 (kept since) freezes racing OFF. Checked on the REQUESTED flag,
    # before the worker-count adjustment below could silently disable it: a
    # strict campaign must fail at snapshot time, not at post-hoc validation
    # after 60 turns of budget.
    # 2026-09-11 (pilot10x_v1): racing (early stop) is ON under the formal
    # profile. A strict campaign must fail HERE, at snapshot time, when the
    # racing it would freeze is inconsistent with the budget policy or
    # cannot execute -- not after 30 turns of budget in the validator.
    if strict_turns and racing.enabled:
        if int(racing.shots_confirm) != int(policy["search_shots"]) or int(
            racing.shots_max
        ) != int(policy["search_shots"]):
            raise ValueError(
                "strict_turns: racing shots_confirm/shots_max "
                f"({racing.shots_confirm}/{racing.shots_max}) must equal the budget "
                f"policy's search shots ({policy['search_shots']}) -- use "
                "config.formal_racing(search_shots)"
            )
        if resolve_workers(int(slow_workers)) <= 1:
            raise ValueError(
                "strict_turns with racing needs the job pool (slow_workers > 1); "
                "a single worker would silently fall back to fixed shots"
            )
    from omniqec_agentic.orchestrator.prompts import prompt_ids
    from omniqec_agentic.orchestrator.reflector import reflection_policy

    # Freeze the EFFECTIVE racing protocol: racing needs the job pool, so a
    # worker count that resolves to 1 disables it — the snapshot must record
    # what will actually be measured, not the requested flag.
    if racing.enabled and resolve_workers(int(slow_workers)) <= 1:
        racing = dataclasses.replace(racing, enabled=False)

    # v3.6 decoder blob: recorded AS CONFIGURED, never rewritten here -- no
    # field of the telescoping decoder depends on the budget point (the
    # cascade's S4 time limits, which this function used to rewrite from the
    # policy, are gone with the cascade). Strict mode checks it below.
    telescoping_blob = verification.decoder_build.telescoping.to_dict()
    # Search shots per basis: the policy value, unless explicitly overridden
    # outside strict mode (VerificationConfig carries the requested value; a
    # strict run may not deviate from the point's policy).
    policy_search_shots = int(policy["search_shots"])
    if strict_turns and int(verification.shots) != policy_search_shots:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, whose budget policy "
            f"freezes search shots per basis at {policy_search_shots} for "
            f"n_budget={n_budget} (got {int(verification.shots)})"
        )
    # Final shots per basis: policy when not given; an explicit value is
    # honoured outside strict mode and refused under strict when it differs.
    policy_final_shots = int(policy["final_shots"])
    if final_shots is None:
        resolved_final_shots = policy_final_shots
    else:
        resolved_final_shots = int(final_shots)
        if strict_turns and resolved_final_shots != policy_final_shots:
            raise ValueError(
                f"strict_turns certifies a {FORMAL_PROFILE} run, whose budget "
                f"policy freezes final shots per basis at {policy_final_shots} for "
                f"n_budget={n_budget} (got {resolved_final_shots})"
            )
    if resolved_final_shots < 1:
        raise ValueError("final_shots must be >= 1")

    profile = verification.noise.profile
    noise_values = tuple(float(value) for value in verification.noise.values)
    if len(noise_values) != 1:
        raise ValueError(
            "campaign protocol requires exactly one fixed noise point; "
            f"received {noise_values}"
        )
    fixed_p = noise_values[0]
    # v3.5: the noise model is a protocol field. The formal run measures the
    # paper's SI C uniform model at p=0.005; the v3.4 idle-noise model
    # (uniform_legacy_v1) or any other p is a different measurement.
    if strict_turns and profile.value != FORMAL_NOISE_PROFILE:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes "
            f"noise_profile={FORMAL_NOISE_PROFILE} (got {profile.value})"
        )
    if strict_turns and abs(fixed_p - FORMAL_P) > 1e-15:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes "
            f"noise_values=[{FORMAL_P}] (got {list(noise_values)})"
        )
    # v3.5 SEC: the circuit policy is a protocol field. Any deviation (the
    # historical lrc_ranked_v1, or a different K / cert budget / seed of the
    # twobasis search) is a different circuit and a different measurement.
    sec_policy = verification.sec_policy
    if strict_turns and sec_policy != DEFAULT_SEC_POLICY:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes "
            f"sec_policy={FORMAL_SEC_POLICY} (candidates="
            f"{DEFAULT_SEC_POLICY.sec_candidates}, cert_iters="
            f"{DEFAULT_SEC_POLICY.sec_cert_iters}, seed="
            f"{DEFAULT_SEC_POLICY.sec_cert_seed}, cert_seeds="
            f"{DEFAULT_SEC_POLICY.sec_cert_seeds}, refine_top="
            f"{DEFAULT_SEC_POLICY.sec_refine_top}); got {sec_policy.to_dict()}"
        )
    # v3.6 decoder: ONE decoder at ONE configuration. The telescoping blob is
    # checked field by field against TELESCOPING_PROTOCOL_FIELDS (every field
    # of TelescopingDecoderConfig; none depends on the budget point).
    decoder_names = tuple(decoder.value for decoder in verification.decoders)
    if strict_turns and decoder_names != (FORMAL_DECODER,):
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which is "
            f"{FORMAL_DECODER}-only (got decoders={list(decoder_names)})"
        )
    if strict_turns:
        drift = {
            key: telescoping_blob.get(key)
            for key in sorted(set(TELESCOPING_PROTOCOL_FIELDS) | set(telescoping_blob))
            if telescoping_blob.get(key) != TELESCOPING_PROTOCOL_FIELDS.get(key)
        }
        if drift:
            raise ValueError(
                f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes the "
                f"telescoping decoder fields {TELESCOPING_PROTOCOL_FIELDS}; got {drift}"
            )
    if int(rng_seeds_per_program) < 1:
        raise ValueError("rng_seeds_per_program must be >= 1")
    if str(seed_programs) not in SEED_PROGRAMS_POLICIES:
        raise ValueError(f"seed_programs must be one of {SEED_PROGRAMS_POLICIES} (got {seed_programs!r})")
    resolved_shot_shards = int(shot_shards)
    if resolved_shot_shards < 1:
        raise ValueError("shot_shards must be >= 1")
    if resolved_shot_shards > int(verification.shots):
        raise ValueError(
            "shot_shards must not exceed search shots per basis "
            f"({resolved_shot_shards} > {int(verification.shots)})"
        )
    if resolved_shot_shards > resolved_final_shots:
        raise ValueError(
            "shot_shards must not exceed final shots per basis "
            f"({resolved_shot_shards} > {resolved_final_shots})"
        )
    if strict_turns and resolved_shot_shards != FORMAL_SHOT_SHARDS:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes "
            f"shot_shards={FORMAL_SHOT_SHARDS}"
        )
    # v3.4: the final stage has its own partition count. 200k shots per basis
    # over 16 shards is 12,500 shots per shard — 8,000-68,000 s for the slow
    # codes measured on NSCC (2026-09-02) — so the formal final could never
    # finish at b220+; 128 shards keeps every shard inside the per-shot budget.
    resolved_final_shot_shards = int(final_shot_shards)
    if resolved_final_shot_shards < 1:
        raise ValueError("final_shot_shards must be >= 1")
    if resolved_final_shot_shards > resolved_final_shots:
        raise ValueError(
            "final_shot_shards must not exceed final shots per basis "
            f"({resolved_final_shot_shards} > {resolved_final_shots})"
        )
    if strict_turns and resolved_final_shot_shards != FORMAL_FINAL_SHOT_SHARDS:
        raise ValueError(
            f"strict_turns certifies a {FORMAL_PROFILE} run, which freezes "
            f"final_shot_shards={FORMAL_FINAL_SHOT_SHARDS}"
        )
    # promote_top is the slow-queue cap and the structural config is where the
    # cap's S:R split lives (4S+2R since 2026-09-10); letting them drift means the queue shape stops
    # matching the protocol it claims to follow.
    if int(verification.promote_top) != int(verification.structural.slow_candidate_cap):
        raise ValueError(
            "promote_top must equal structural.slow_candidate_cap "
            f"({verification.promote_top} != {verification.structural.slow_candidate_cap})"
        )
    if strict_turns and (not required_provider or not required_model):
        raise ValueError("strict_turns requires required_provider and required_model")
    structural = verification.structural
    # v3.8: the variant generations are a suffix of the turn's generations;
    # at least one exploration generation must remain.
    if int(getattr(structural, "exploit_generations", 0) or 0) >= int(generations_per_turn):
        raise ValueError(
            f"structural.exploit_generations={structural.exploit_generations} must be smaller "
            f"than generations_per_turn={generations_per_turn}"
        )
    protocol_profile = str(structural.profile_id)
    if resolved_shot_shards != FORMAL_SHOT_SHARDS:
        protocol_profile = f"{protocol_profile}+shards{resolved_shot_shards}"
    if resolved_final_shot_shards != FORMAL_FINAL_SHOT_SHARDS:
        protocol_profile = f"{protocol_profile}+finalshards{resolved_final_shot_shards}"
    if profile.value != FORMAL_NOISE_PROFILE:
        # A non-paper noise model (e.g. the v3.4 idle-noise uniform_legacy_v1)
        # is a different measurement and may not carry the plain profile id.
        protocol_profile = f"{protocol_profile}+noise{profile.value}"
    if sec_policy.sec_policy_id != DEFAULT_SEC_POLICY.sec_policy_id:
        # The historical LRC circuit (or any other policy id) is a different
        # circuit family; its numbers never pool with twobasis ones.
        protocol_profile = f"{protocol_profile}+sec{sec_policy.sec_policy_id}"
    elif sec_policy != DEFAULT_SEC_POLICY:
        # Same policy, different search/certification budget: still a variant.
        # v3.6: the multi-seed certification fields are appended only when
        # they differ from the default, so the historical K/I/S suffix of a
        # v3.5-style variant is unchanged.
        protocol_profile = (
            f"{protocol_profile}+secK{int(sec_policy.sec_candidates)}"
            f"I{int(sec_policy.sec_cert_iters)}S{int(sec_policy.sec_cert_seed)}"
        )
        cert_fields = (
            ("N", int(sec_policy.sec_cert_seeds), int(DEFAULT_SEC_POLICY.sec_cert_seeds)),
            ("X", float(sec_policy.sec_cert_iters_scale), float(DEFAULT_SEC_POLICY.sec_cert_iters_scale)),
            ("R", int(sec_policy.sec_refine_top), int(DEFAULT_SEC_POLICY.sec_refine_top)),
            ("F", int(sec_policy.sec_refine_iters), int(DEFAULT_SEC_POLICY.sec_refine_iters)),
        )
        if any(value != default for _, value, default in cert_fields):
            protocol_profile += "".join(
                f"{tag}{value:g}" if isinstance(value, float) else f"{tag}{value}"
                for tag, value, _default in cert_fields
            )
        # Review fix: the per-code ISD budget (0 = unlimited) is appended
        # only when it differs, so every historical suffix is unchanged.
        if int(sec_policy.sec_cert_iters_budget) != int(DEFAULT_SEC_POLICY.sec_cert_iters_budget):
            protocol_profile += f"B{int(sec_policy.sec_cert_iters_budget)}"
    if decoder_names != (FORMAL_DECODER,):
        # v3.6: the decoder IS the protocol. Any other set (the historical
        # BPOSD, LSD/beam variants, several decoders) is a different
        # measurement and may not carry the plain profile id.
        protocol_profile = f"{protocol_profile}+dec{'-'.join(decoder_names)}"
    snapshot: dict[str, Any] = {
        "schema": "campaign_protocol_snapshot",
        # The profile id IS the protocol identity: changing the decoder, the
        # d_search protocol, the reducer or any quota below makes a run
        # incomparable with a telescoping_structural_v3_6 run and must bump this.
        "protocol_profile": protocol_profile,
        # v3.6 decoder identity: the single decoder name and its full frozen
        # blob (every TelescopingDecoderConfig field, budget independent). A
        # v3.5 snapshot has neither key and refuses to resume; a cascade
        # snapshot carries `cascade` instead of `telescoping` and is refused too.
        "decoder": FORMAL_DECODER if decoder_names == (FORMAL_DECODER,) else "+".join(decoder_names),
        "telescoping": telescoping_blob,
        # v3.6 budget policy (docs/06 §1.4): the budget point and the shots
        # per basis it selects (3 keys: large_budget, search_shots,
        # final_shots). Resume refuses a snapshot frozen at another point or
        # under another policy table.
        "n_budget": None if n_budget is None else int(n_budget),
        "budget_policy": dict(policy),
        # v3.5 SEC policy (twobasis_nonint_v1 by default). Three top-level
        # keys plus the full blob (v3.6: with the multi-seed certification
        # fields) here and in verification.sec_policy: a v3.4 snapshot carries
        # sec_policy_id=lrc_ranked_v1 and no sec_candidates / sec_cert_iters
        # keys, so it refuses to resume.
        "sec_policy_id": str(sec_policy.sec_policy_id),
        "sec_candidates": int(sec_policy.sec_candidates),
        "sec_cert_iters": int(sec_policy.sec_cert_iters),
        "sec_policy": sec_policy.to_dict(),
        # Prompt-as-code (03a R1): changing a prompt changes the protocol.
        "prompt_ids": prompt_ids(),
        # 2026-09-10: the fast quality score is a protocol field. kd2_v1 =
        # Q = k*d^2; a snapshot without this key was ranked under ver1's
        # k*d^2/n and refuses to resume here.
        "quality_metric": ranking.QUALITY_METRIC_ID,
        # Kept for readers of the historical key; mirrors the policy above.
        "sec_synthesis": {
            "candidates": int(sec_policy.sec_candidates),
            "isd_iters": int(sec_policy.sec_cert_iters),
        },
        "noise_profile": profile.value,
        # v3.5: the noise model is a protocol field (paper SI C uniform, no
        # idle noise, p=0.005). A v3.4 snapshot has no noise_values key and
        # therefore cannot resume into a v3.5 campaign.
        "noise_values": [float(value) for value in noise_values],
        # Fig.2 is a single-point comparison. Search and fixed-T final
        # confirmation use this same frozen value; recording it separately
        # makes resume reject checkpoints created under the former p sweep.
        "anchor_p": fixed_p,
        "final_p_values": [fixed_p],
        # Fairness protocol (2026-08-16): search runs at ONE fixed round count
        # per campaign (uniform boundary bias, no time truncation).
        # v3.3: the final audit is ALSO fixed-T. The old "T=d" label ran
        # T=max(3, d_proxy) — a proxy value, not a certified distance — which
        # both mislabels the measurement and prices candidates by a number the
        # search itself produced. One fixed T compares every finalist on the
        # same circuit depth. THIS block is the single source of truth for
        # both stages: no other layer may recompute rounds.
        # 2026-09-10: max_Tc_d — search_rounds / final_rounds are the T_c
        # FLOOR (12); a candidate with d_search > T_c is measured at T =
        # d_search in both stages (verification.circuit.select_memory_rounds).
        # 2026-10-15: the policy id comes from the verification config
        # (max_Tc_d formal; fixed_Tc = the Fig.3 formal run's fixed depth).
        "rounds_policy": {
            "search": str(getattr(verification, "rounds_policy", ROUNDS_POLICY_ID)),
            "search_rounds": int(search_rounds) if search_rounds else None,
            "final": str(getattr(verification, "rounds_policy", ROUNDS_POLICY_ID)),
            "final_rounds": int(structural.final_rounds),
            "final_rng_namespace": str(structural.final_rng_namespace),
            # 2026-09-10: 5 (was 3); the single source is ranking.
            "min_promotion_distance": int(ranking.MIN_PROMOTION_DISTANCE),
        },
        # Structural retention channel: reducer identity, caps and quotas are
        # frozen here so a plan or a CLI override cannot resize them.
        "structural_retention": structural.to_dict(),
        # The verification blob carries the policy-resolved shots and the
        # telescoping decoder blob: it is what every executor reads back.
        "verification": verification.to_dict(),
        "racing": racing.to_dict(),
        # Search/racing confirmation and final reporting are intentionally
        # separate. Larger budget bins can pre-register more final shots
        # without multiplying the cost of every search turn.
        "final_shots": resolved_final_shots,
        # Two frozen partition counts: `shot_shards` drives the per-turn slow
        # (search) stage, `final_shot_shards` drives fixed-T confirmation
        # (v3.4). sampling.shard_seed ignores the shard count: shard i gets
        # the same seed under any count > 1 (shards=1 is the legacy single
        # stream). What a different count changes is the shots-per-shard
        # partition and therefore the sampled stream, so two counts are not
        # the same measurement and cannot be mixed within one campaign;
        # resume still rejects a changed value because both are snapshot
        # keys (a v3.3 checkpoint has no final_shot_shards key and refuses
        # to resume).
        "shot_shards": resolved_shot_shards,
        "final_shot_shards": resolved_final_shot_shards,
        "sampling_source": "noisy_circuit",
        "max_turns": int(max_turns),
        "tool_budget": int(tool_budget),
        "proposal_count": int(proposal_count),
        "generations_per_turn": int(generations_per_turn),
        "promote_top": int(verification.promote_top),
        "slow_workers": int(slow_workers),
        "feedback_mode": str(feedback_mode),
        "strict_turns": bool(strict_turns),
        # 2026-09-26: --fixed-program-count freezes the count for a protocol
        # VARIANT too (Fig.3 cannot run strict); strict still implies it.
        "fixed_program_count": bool(strict_turns) or bool(fixed_program_count),
        "required_provider": str(required_provider or ""),
        "required_model": str(required_model or ""),
        "rng_seeds_per_program": int(rng_seeds_per_program),
        # Paper alignment (2026-11-10, appendix D1): the five hand-written seed
        # programs run in the first iteration only (config.seed_programs_run_this_turn).
        "seed_programs": str(seed_programs),
        # Paper alignment (2026-11-10, appendix B4): every iteration ends with a
        # review; its bounded retry is frozen here (orchestrator/reflector.py).
        "reflection_policy": reflection_policy(),
    }
    if reference_codes is not None:
        # Experiment variant (2026-09-09): background codes shown to both LLM
        # roles. Frozen here so every resume/link sees the same text; absent
        # from formal snapshots (the validator refuses the key).
        from omniqec_agentic.orchestrator.reference_codes import validate_reference_codes
        snapshot["reference_codes"] = validate_reference_codes(reference_codes)
    if final_decode_shard_timeout_cap_s is not None:
        # 2026-09-24: runtime key (checkpoint.RESUME_RUNTIME_KEYS) -- only
        # present when set, so an unchanged launch keeps its snapshot.
        cap = float(final_decode_shard_timeout_cap_s)
        if cap <= 0.0:
            raise ValueError("final_decode_shard_timeout_cap_s must be positive when set")
        snapshot["final_decode_shard_timeout_cap_s"] = cap
    if search_decode_shard_timeout_cap_s is not None:
        cap = float(search_decode_shard_timeout_cap_s)
        if cap <= 0.0:
            raise ValueError("search_decode_shard_timeout_cap_s must be positive when set")
        snapshot["search_decode_shard_timeout_cap_s"] = cap
    if promotion_max_check_weight is not None:
        # 2026-09-29: runtime key (checkpoint.RESUME_RUNTIME_KEYS): the slow
        # queue's max stabilizer weight gate; only present when set.
        gate = int(promotion_max_check_weight)
        if gate < 1:
            raise ValueError("promotion_max_check_weight must be >= 1 when set")
        snapshot["promotion_max_check_weight"] = gate
    # Fig.4 c (2026-10-23): the exact-weight contract is a protocol key (not a
    # runtime key): it decides which codes exist in the run. Mirrored from the
    # verification config so the auditor and the gates read one top-level value.
    exact = getattr(verification, "exact_check_weight", None)
    if exact is not None:
        exact = int(exact)
        gate_value = snapshot.get("promotion_max_check_weight")
        if gate_value is not None and int(gate_value) != exact:
            raise ValueError(
                f"promotion_max_check_weight ({gate_value}) must equal exact_check_weight ({exact})"
            )
        snapshot["exact_check_weight"] = exact
    if campaign_id:
        snapshot["campaign_id"] = str(campaign_id)
    if ablation is not None and str(ablation) != "full":
        # S3 ablation (2026-10-23): a protocol VARIANT. Only present when set,
        # so a formal launch keeps its snapshot byte for byte. The generator's
        # feedback_mode must agree with the mode (the CLI derives it).
        from omniqec_agentic.config import ABLATION_FEEDBACK_MODE, ABLATION_IDS

        mode = str(ablation)
        if mode not in ABLATION_IDS:
            raise ValueError(f"unknown ablation mode {mode!r} (expected one of {ABLATION_IDS})")
        expected_feedback = ABLATION_FEEDBACK_MODE[mode]
        if str(feedback_mode) != expected_feedback:
            raise ValueError(
                f"ablation={mode} requires feedback_mode={expected_feedback!r} (got {feedback_mode!r})"
            )
        snapshot["ablation"] = mode
    if search_baseline is not None:
        from omniqec_agentic.config import BASELINE_PROVIDER, SEARCH_METHODS

        method = str(search_baseline.get("method") or "")
        if method not in SEARCH_METHODS or method == "llm":
            raise ValueError(f"unknown search baseline {method!r} (expected one of {SEARCH_METHODS[1:]})")
        if "ablation" in snapshot:
            raise ValueError("a search baseline runs the full feedback flow; it cannot carry an ablation")
        if str(required_provider or "") != BASELINE_PROVIDER:
            raise ValueError(f"a search baseline requires provider {BASELINE_PROVIDER!r} (got {required_provider!r})")
        snapshot["search_baseline"] = json.loads(json.dumps(search_baseline, sort_keys=True))
    if final_stage_id is not None:
        from omniqec_agentic.config import FINAL_STAGE_FIXED_WEIGHT_ID, FINAL_STAGE_IDS
        from omniqec_agentic.evaluation.final_stage import final_stage_protocol

        if str(final_stage_id) not in FINAL_STAGE_IDS:
            raise ValueError(f"unknown final_stage_id {final_stage_id!r} (expected one of {FINAL_STAGE_IDS!r})")
        if str(final_stage_id) == FINAL_STAGE_FIXED_WEIGHT_ID and exact is None:
            raise ValueError(f"final_stage_id {FINAL_STAGE_FIXED_WEIGHT_ID!r} needs an exact check-weight contract")
        if strict_turns and final_cumulative_shots is not None:
            raise ValueError(
                "strict_turns freezes the final stage's cumulative shots at Table VII; "
                "--final-cumulative-shots is a declared variant"
            )
        snapshot["final_stage"] = final_stage_protocol(
            task=str(getattr(verification, "task", "memory") or "memory"),
            n_budget=n_budget,
            confirm_shots=resolved_final_shots,
            final_shot_shards=resolved_final_shot_shards,
            cumulative_override=final_cumulative_shots,
            # Fig. 4 c: the exact-weight contract selects the fixed-weight final stage.
            exact_check_weight=exact,
        )
    elif final_cumulative_shots is not None:
        raise ValueError("final_cumulative_shots needs the paper final stage (final_stage_id)")
    return snapshot


def assert_resumable_protocol(stored_protocol: dict[str, Any] | None) -> None:
    """Refuse to resume a checkpoint frozen under a retired protocol.

    ``validate_resume_protocol`` (workflow/checkpoint.py) already rejects any
    key-level difference, but its message only lists differing keys. A v3.5
    checkpoint (``bposd_structural_v3_5`` / decoder ``bposd``, no
    ``budget_policy`` key) or a self-built-cascade checkpoint
    (``cascade_structural_v3_6`` / decoder ``relay_cascade_gari``, a
    ``cascade`` key instead of ``telescoping``) measured every candidate with
    a different decoder, so it must be refused by NAME, before the generic
    diff, with the remedy (a fresh run directory). Called by run_campaign
    before the diff.
    """

    if not stored_protocol:
        raise ValueError("resume requested but the checkpoint has no protocol snapshot")
    stored_profile = str(stored_protocol.get("protocol_profile") or "")
    stored_decoder = stored_protocol.get("decoder")
    if stored_decoder is None:
        decoders = (stored_protocol.get("verification") or {}).get("decoders") or []
        stored_decoder = "+".join(str(item) for item in decoders) if decoders else None
    # Retired = a retired profile name, or a snapshot without the v3.6 keys
    # (`budget_policy`, `telescoping`). The decoder NAME alone does not retire a
    # checkpoint: a non-strict variant (`+declsd`, `+decbposd`, allowed by
    # the CLI outside --strict-turns) carries the v3.6 keys and must resume
    # its own checkpoint; validate_resume_protocol decides sameness for it.
    missing = [key for key in ("budget_policy", "telescoping") if key not in stored_protocol]
    legacy = (
        any(stored_profile.startswith(prefix) for prefix in LEGACY_PROFILE_PREFIXES)
        or bool(missing)
    )
    if legacy:
        raise ValueError(
            "refusing to resume: the checkpoint was frozen under protocol "
            f"{stored_profile or '<unknown>'!r} with decoder {stored_decoder!r}"
            f"{' and no ' + ' / '.join(missing) + ' key' if missing else ''}"
            f"; the current protocol is {FORMAL_PROFILE} ({FORMAL_DECODER}). Its "
            "numbers are not poolable with v3.6 ones — start a fresh run "
            "directory (the _v36 suffix) instead of resuming this checkpoint"
        )
