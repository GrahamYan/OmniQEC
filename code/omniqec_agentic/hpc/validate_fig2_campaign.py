#!/usr/bin/env python3
"""Validate one formal fixed-p campaign before accepting its output.

Profiles (--profile): v3_7 (the module default) and v3_8 check the as-run Fig.2 rules
(T = max(12, d_search)); fig2_final and fig3_final check the final protocols of
docs/final_protocol_20261106.md (T = 12 for every code, slow-loop check-weight gate w <= 8 and
the 30-failure early stop from turn 0, the final models, and the reasoning-saturation rule).
fig3_final with --exact-check-weight w checks a Fig. 4 c fixed-weight CNOT search: the exact contract
w at construction and at the slow-loop gate, and the paper's fixed-weight final stage (six finalists,
the initial confirmation only, the three best reported; docs/paper_alignment_measurement.md).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# v3.6 (docs/06 §1.2): the decoder identity, the SEC policy, the structural
# retention config and the budget-point policy are imported from the package
# rather than mirrored, so the validator can never drift from what the
# campaign froze. The launcher runs it from the deployed repo (PYTHONPATH set),
# never as a detached copy.
from omniqec_agentic import ranking
from omniqec_agentic.config import (
    ROUNDS_POLICY_ID,
    DEFAULT_SEC_POLICY,
    PROTOCOL_PROFILE_V38,
    TELESCOPING_PROTOCOL_FIELDS,
    TELESCOPING_VENDORED_COMMIT,
    V38_STRUCTURAL_RULES,
    DecoderName,
    RACING_POLICY_ID,
    StructuralSearchConfig,
    structural_search_for_profile,
    budget_policy,
    fig3_cnot_structural_config,
    formal_racing_for_max_errors,
    formal_racing,
    formal_racing_policy_id,
    FORMAL_RACING_POLICIES,
    RACING_POLICY_PAPER_ID,
    paper_racing,
    racing_policy_moves_forward,
)
from omniqec_agentic.evaluation import final_stage as paper_final


from omniqec_agentic import reasoning_fallback


TURN_RE = re.compile(r"^turn_(\d{4})$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _original_max_turns(protocol: dict[str, Any] | None) -> int | None:
    """max_turns before any lowering, from the snapshot's runtime_overrides
    history (workflow.checkpoint.original_max_turns mirror); None otherwise."""

    for entry in (protocol or {}).get("runtime_overrides") or []:
        before = ((entry or {}).get("from") or {}).get("max_turns")
        if before is not None:
            return int(before)
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL artifact: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"empty JSONL artifact: {path}")
    return rows


def _same_float(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


PROTOCOL_PROFILE = "telescoping_structural_v3_7"
# 2026-09-10 generator budget: 24 programs per generator call, FOUR generator
# calls (generations) per turn = 96 programs per turn, each program run under
# 40 rng seeds (a deterministic program yields one code; a randomised one up
# to 40). The turn budget was halved from 60 to 30 on 2026-09-05; the user
# then chose to widen the fast loop instead (30 x 96 = 2880 programs, twice
# the 60-turn total). Every per-turn artifact is checked against the product,
# the plan/cycle records against the factors.
FORMAL_PROPOSAL_COUNT = 24
FORMAL_GENERATIONS_PER_TURN = 4
FORMAL_PROPOSALS_PER_TURN = FORMAL_PROPOSAL_COUNT * FORMAL_GENERATIONS_PER_TURN
FORMAL_RNG_SEEDS_PER_PROGRAM = 40
# 2026-09-10 fast quality score: Q = k*d^2 (ranking.QUALITY_METRIC_ID kd2_v1).
# A snapshot without the key ranked its archive under ver1's k*d^2/n.
FORMAL_QUALITY_METRIC = "kd2_v1"
# v3.6 (docs/06 §1, user decision 2026-09-03): ONE decoder -- the Mitten
# paper's open-source telescoping decoder (vendored package, CPU
# configuration: S1/S2/S4 off, S3-A/B/C on the GARI transform of the
# two-basis DEM, an undecoded shot counts as a logical error) -- at ONE
# configuration (config.TELESCOPING_PROTOCOL_FIELDS: every field of the
# decoder config, none budget dependent). The search/final shots per basis
# come from the budget-point policy (config.budget_policy: <= 288 ->
# 25k/25k since 2026-09-10; >= 300 -> 5k/50k). There is no fallback decoder: BPOSD is a
# dev-only decoder whose blob is recorded but never checked.
FORMAL_DECODER = DecoderName.TELESCOPING.value
# v3.5 (2026-09-02): the noise model is a protocol field. The formal run
# measures the OmniQEC paper SI C uniform circuit-level depolarizing model
# (depolarizing after every reset and CX, a flip before every measurement,
# NO idle noise) at ONE physical error rate. Mirrored from
# omniqec_agentic.workflow.state so the validator stays standalone; the
# tests pin the copies together. A v3.4 run (uniform_legacy_v1 idle noise,
# no noise_values key) fails the snapshot checks by construction.
FORMAL_NOISE_PROFILE = "uniform_paper_v1"
FORMAL_P = 0.005
# v3.5 SEC (2026-09-02): the syndrome-extraction circuit policy is a protocol
# field. twobasis_nonint_v1 = both bases measured every round, NO interleaving,
# K=16 candidate layer orders per basis certified with 400 ISD iterations at
# seed 0 (omniqec_agentic.config.DEFAULT_SEC_POLICY; mirrored here so the
# validator stays standalone, tests pin the copies). Checked in the snapshot,
# every run config and every candidate's per-basis schedule metadata; a v3.4
# run (lrc_ranked_v1, no sec_policy blob) fails by construction.
FORMAL_SEC_POLICY = "twobasis_nonint_v1"
FORMAL_SEC_CANDIDATES = int(DEFAULT_SEC_POLICY.sec_candidates)
FORMAL_SEC_CERT_ITERS = int(DEFAULT_SEC_POLICY.sec_cert_iters)
FORMAL_SEC_CERT_SEED = int(DEFAULT_SEC_POLICY.sec_cert_seed)
# v3.6 SEC certification (docs/06 §3 M1): >= 3 seeds (min), iterations scaled
# by the extended-kernel dimension, top-4 re-certified with 2000 iterations.
FORMAL_SEC_CERT_SEEDS = int(DEFAULT_SEC_POLICY.sec_cert_seeds)
FORMAL_SEC_CERT_ITERS_SCALE = float(DEFAULT_SEC_POLICY.sec_cert_iters_scale)
FORMAL_SEC_REFINE_TOP = int(DEFAULT_SEC_POLICY.sec_refine_top)
FORMAL_SEC_REFINE_ITERS = int(DEFAULT_SEC_POLICY.sec_refine_iters)
# Review fix (M1 cost): per-code ISD iteration budget, 0 = unlimited.
FORMAL_SEC_CERT_ITERS_BUDGET = int(DEFAULT_SEC_POLICY.sec_cert_iters_budget)
# Frozen execution geometry and stage-specific watchdog policy. Mirrored here
# so the validator remains standalone on an NSCC login node; tests pin these
# copies to the package defaults.
# v3.4: search and final stages have DIFFERENT partition counts (16 / 128),
# the decode-shard watchdog is max(decode_shard_timeout_s,
# decode_shot_budget_s * shard shots), timeouts are never replayed
# (timeout_retries 0) and dead/DEM-less workers are replayed once
# (transient_retries 1).
FORMAL_SHOT_SHARDS = 16
FORMAL_FINAL_SHOT_SHARDS = 128
FORMAL_JOB_POLICY = {
    "build_timeout_s": 3600.0,
    "decode_shard_timeout_s": 1800.0,
    "distance_timeout_s": 1800.0,
    "timeout_retries": 0,
    "decode_shot_budget_s": 8.0,
    "transient_retries": 1,
}
# Integer-typed policy keys (compared exactly, not as floats).
_INT_POLICY_KEYS = frozenset({"timeout_retries", "transient_retries"})
# Row statuses that are TERMINAL measurement outcomes rather than defects:
# schedule_catastrophic (no circuit can exist) and, since v3.4,
# decode_infeasible (decode exceeded the frozen per-shot budget). Such rows
# carry no measurement, are skipped by the row checks below, and never
# count towards the >= 1 ok final row the report must still contain.
# v3.6.1 adds decoder_unsupported (the telescoping decoder cannot be built for
# the candidate's DEM: identical detector support, different logical support).
TERMINAL_ROW_STATUSES = frozenset({"schedule_catastrophic", "decode_infeasible", "decoder_unsupported"})
# The formal decoder is the vendored telescoping-decoder package: its
# distribution version must be resolved and its vendored commit (protocol
# identity: the S3 variant tables, the seed derivation and the C kernels
# live there) must be the one config froze. ldpc / relay_bp / scipy serve
# the dev-only decoders and are no longer required.
REQUIRED_VERSION_KEYS = ("telescoping-decoder",)
REQUIRED_TELESCOPING_COMMIT = TELESCOPING_VENDORED_COMMIT
# Search-stage sampling keys derive from base 0 (candidate index * 1e6); the
# final audit derives from a namespace-hashed base >= 1e12. Mirrored from
# omniqec_agentic.workflow.graph.final_rng_base_seed so this script stays
# standalone; the test suite pins the two together.
SEARCH_SEED_CEILING = 10**12


def _expected_final_base_seed(namespace: str) -> int:
    import hashlib

    digest = hashlib.sha256(str(namespace).encode("utf-8")).hexdigest()
    return 10**12 + int(digest[:12], 16) % 10**9

# The T_c FLOOR of the rounds policy max_Tc_d (2026-09-10): the protocol
# snapshot and every per-turn run config carry 12; every measured row must
# record T = max(12, its own d_search) — a d>12 code is measured at T=d.
SEARCH_ROUNDS = 12
FORMAL_ROUNDS_POLICY = ROUNDS_POLICY_ID
# The task the profile measures (verification.task; absent = memory).
FORMAL_TASK = "memory"
# Final profiles only (configure_profile): the slow-loop check-weight gate and the early-stop
# policy, both required from turn 0 (no runtime override), and the base model per provider.
FINAL_PROTOCOL = False
FINAL_PROMOTION_GATE = 8
# 2026-11-10 (paper alignment): the final profiles require the paper's early stop (Appendix D3) from turn 0
# and the paper's final stage (paper_top5_cumulative_v1, Table VII); see docs/paper_alignment_measurement.md.
FINAL_RACING_POLICY = RACING_POLICY_PAPER_ID
# Paper alignment (2026-11-10, os/search): the final protocols cap the check weight in the FAST loop too
# (config.SEARCH_MAX_CHECK_WEIGHT, paper Methods / appendix B2, D1: w_max <= 8 for every main search). The
# as-run profiles v3_7 / v3_8 had no construction-time cap (None).
FINAL_FAST_LOOP_MAX_CHECK_WEIGHT = 8
# 2026-11-10 (paper revision, Fig. 4 c): fig3_final --exact-check-weight w validates a fixed-weight search -- the
# construction cap and the slow-loop gate are w, the snapshot carries exact_check_weight = w and the final stage is
# the fixed-weight one (evaluation/final_stage.py, paper_fig4c_top6_confirm_v1). None: no exact contract allowed.
FINAL_EXACT_CHECK_WEIGHT: int | None = None
FINAL_MODELS = {"openai": ("claude-opus-5", "gpt-5.6-sol"), "openrouter": ("deepseek/deepseek-v4-pro",)}


def _row_d_search(row: dict[str, Any]) -> int:
    for key in ("d_search", "d_proxy", "d"):
        value = row.get(key)
        if value is not None:
            return int(value)
    return 0


def _expected_rounds(t_c: int, row: dict[str, Any]) -> int:
    if FORMAL_ROUNDS_POLICY == "fixed_Tc":
        return int(t_c)
    return max(int(t_c), _row_d_search(row))


def _rounds_rule(t_c: int, row: dict[str, Any]) -> str:
    if FORMAL_ROUNDS_POLICY == "fixed_Tc":
        return f"(fixed_Tc: T = T_c = {int(t_c)})"
    return f"(max(T_c={int(t_c)}, d_search={_row_d_search(row)}))"
# The frozen structural retention config IS the package default (v3.6:
# reducer structural_proxy_pareto_v2_coordcap with coordinate_cap=2); every
# key of its serialized form is checked, so a drifted field cannot pass.
STRUCTURAL_EXPECTED = StructuralSearchConfig().to_dict()
assert STRUCTURAL_EXPECTED["profile_id"] == PROTOCOL_PROFILE, STRUCTURAL_EXPECTED
# Fig.3 variant switches (2026-10-12 / 10-15) and the v3.8 rules (2026-10-17) that
# did not exist when the 27 formal Fig.2 snapshots were frozen: absent means the
# package default (the v3.7 value). Present with another value, or absent where the
# profile expects a non-default value, is a different protocol and is rejected.
FIG3_VARIANT_STRUCTURAL_KEYS = frozenset(
    {"lightest_endpoint_slot", "not_competitive_ratio", "not_competitive_max_rows", "measured_twin_feedback"}
)
LATER_STRUCTURAL_KEYS = FIG3_VARIANT_STRUCTURAL_KEYS | frozenset(V38_STRUCTURAL_RULES)
STRUCTURAL_DEFAULTS = StructuralSearchConfig().to_dict()
for _key in LATER_STRUCTURAL_KEYS:
    assert _key in STRUCTURAL_EXPECTED, _key


def _check_structural_retention(structural: dict[str, Any], *, where: str = "structural_retention") -> None:
    """Every key of the frozen structural config must match (quotas, caps,
    reducer identity); a key added after the formal Fig.2 snapshots (Fig.3
    switches, v3.8 rules) may be absent, in which case it is at its default."""

    for key, expected in STRUCTURAL_EXPECTED.items():
        if key in LATER_STRUCTURAL_KEYS:
            value = structural.get(key, STRUCTURAL_DEFAULTS[key])
        else:
            value = structural.get(key)
        _require(
            value == expected,
            f"{where} {key}={value!r}, expected {expected!r}",
        )
# v3.7: the two structural slots are the endpoints of the weight-aware front;
# a short pool backfills from the runtime channel, never from crowding picks
# taking a structural slot under another name. (Backfilled structural rows
# keep these reasons too: the queue builder re-walks the same endpoint list.)
FORMAL_STRUCTURAL_SLOT_REASONS_V37 = frozenset(
    {"high_k_endpoint", "high_d_search_endpoint", "crowding_representative",
     "crowding_representative_repeat_coordinate"}
)
FORMAL_STRUCTURAL_SLOT_REASONS = FORMAL_STRUCTURAL_SLOT_REASONS_V37
# v3.7: the runtime archive order (memory.expected_plpr) -- turn 0 has no
# measurements and legitimately runs the historical Q_eff order.
FORMAL_RUNTIME_PROMOTION_RULES = frozenset(
    {"expected_plpr_v1_archive_elite", "ver1_q_fast_archive_elite", "all_archive_cell_elites"}
)


def _validate_decoder_blob(blob: Any, where: str) -> None:
    """Every recorded telescoping decoder blob must BE the frozen protocol.

    Missing is a failure, not a pass: an absent field is how a differently
    configured decoder slips through a check that only compares what it finds.
    Every TELESCOPING_PROTOCOL_FIELDS key is compared type-exactly (an int
    field recorded as a float, or a bool as an int, is drift too). No field
    depends on the budget point, so there is no per-point branch and no
    fallback decoder to check (an undecoded shot is a logical error)."""

    _require(isinstance(blob, dict), f"{where}: decoder config is missing")
    for key, expected in TELESCOPING_PROTOCOL_FIELDS.items():
        actual = blob.get(key)
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, float):
            matches = type(actual) is float and _same_float(actual, float(expected))
        elif isinstance(expected, int):
            matches = type(actual) is int and actual == expected
        else:
            matches = type(actual) is type(expected) and actual == expected
        _require(
            matches,
            f"{where}: telescoping {key}={actual!r}, expected {expected!r}",
        )


def _validate_budget_policy(blob: Any, n_budget: int | None, where: str) -> None:
    """A recorded budget_policy blob must equal budget_policy(n_budget):
    exactly its three keys (large_budget, search_shots, final_shots), each
    compared by type. An extra key (the retired cascade's s4_time_s /
    s4_time_final_s) marks a run of another protocol."""

    expected = budget_policy(n_budget)
    _require(isinstance(blob, dict), f"{where}: budget_policy is missing")
    _require(
        set(blob) == set(expected),
        f"{where}: budget_policy keys={sorted(blob)!r}, expected {sorted(expected)!r}",
    )
    for key, value in expected.items():
        actual = blob.get(key)
        if isinstance(value, bool):
            matches = actual is value
        elif isinstance(value, float):
            matches = _same_float(actual, value)
        else:
            matches = type(actual) is int and actual == value
        _require(
            matches,
            f"{where}: budget_policy {key}={actual!r}, expected {value!r} "
            f"(n_budget={n_budget})",
        )


def _validate_noise_sweep(blob: Any, where: str) -> None:
    """The recorded noise sweep must BE the frozen paper model at ONE p.

    Missing is a failure, not a pass: a run config or snapshot without a
    noise block is how a differently modelled run would slip through."""

    _require(isinstance(blob, dict), f"{where}: noise sweep is missing")
    _require(
        blob.get("profile") == FORMAL_NOISE_PROFILE,
        f"{where}: noise profile={blob.get('profile')!r}, "
        f"expected {FORMAL_NOISE_PROFILE!r}",
    )
    values = blob.get("values")
    _require(
        isinstance(values, list)
        and len(values) == 1
        and _same_float(values[0], FORMAL_P),
        f"{where}: noise values={values!r}, expected [{FORMAL_P}]",
    )


def _validate_sec_policy_blob(blob: Any, where: str) -> None:
    """The recorded SEC policy must BE the frozen twobasis policy.

    Missing is a failure, not a pass: a v3.4 artifact has no sec_policy block
    and measured a different circuit. All four fields exact; ints by type."""

    _require(isinstance(blob, dict), f"{where}: sec_policy is missing")
    _require(
        blob.get("sec_policy_id") == FORMAL_SEC_POLICY,
        f"{where}: sec_policy_id={blob.get('sec_policy_id')!r}, "
        f"expected {FORMAL_SEC_POLICY!r}",
    )
    for key, expected in (
        ("sec_candidates", FORMAL_SEC_CANDIDATES),
        ("sec_cert_iters", FORMAL_SEC_CERT_ITERS),
        ("sec_cert_seed", FORMAL_SEC_CERT_SEED),
        # v3.6 multi-seed certification fields (docs/06 §3 M1).
        ("sec_cert_seeds", FORMAL_SEC_CERT_SEEDS),
        ("sec_refine_top", FORMAL_SEC_REFINE_TOP),
        ("sec_refine_iters", FORMAL_SEC_REFINE_ITERS),
        ("sec_cert_iters_budget", FORMAL_SEC_CERT_ITERS_BUDGET),
    ):
        actual = blob.get(key)
        _require(
            type(actual) is int and actual == expected,
            f"{where}: sec_policy {key}={actual!r}, expected {expected!r}",
        )
    _require(
        _same_float(blob.get("sec_cert_iters_scale"), FORMAL_SEC_CERT_ITERS_SCALE),
        f"{where}: sec_policy sec_cert_iters_scale={blob.get('sec_cert_iters_scale')!r}, "
        f"expected {FORMAL_SEC_CERT_ITERS_SCALE!r}",
    )


def _validate_noise_point(point: Any, value: Any, where: str) -> None:
    """A measured row/result must record the frozen noise point it sampled."""

    _require(isinstance(point, dict), f"{where}: noise_point is missing")
    _require(
        point.get("profile") == FORMAL_NOISE_PROFILE,
        f"{where}: noise_point profile={point.get('profile')!r}, "
        f"expected {FORMAL_NOISE_PROFILE!r}",
    )
    _require(
        _same_float(point.get("value"), FORMAL_P),
        f"{where}: noise_point value={point.get('value')!r}, expected {FORMAL_P}",
    )
    _require(
        _same_float(value, FORMAL_P),
        f"{where}: noise_value={value!r}, expected {FORMAL_P}",
    )


def _validate_execution_geometry(
    *,
    shot_shards: Any,
    policy: Any,
    where: str,
    expected_shards: int = FORMAL_SHOT_SHARDS,
) -> None:
    """Require the frozen 128-core sharding and typed watchdog policy.

    These values affect which RNG streams are sampled and whether a hard cell
    can disappear on a timeout, so they are protocol fields rather than mere
    launcher tuning. v3.4: the expected partition count is stage-specific
    (search 16 / final 128), so the caller names it; a policy blob missing
    any v3.4 key (decode_shot_budget_s, transient_retries) or carrying the
    v3.3 timeout_retries=2 fails here.
    """

    _require(
        type(shot_shards) is int and shot_shards == int(expected_shards),
        f"{where}: shot_shards={shot_shards!r}, expected {int(expected_shards)}",
    )
    _require(isinstance(policy, dict), f"{where}: execution policy is missing")
    for key, expected in FORMAL_JOB_POLICY.items():
        actual = policy.get(key)
        if key in _INT_POLICY_KEYS:
            matches = type(actual) is int and actual == expected
        else:
            matches = _same_float(actual, float(expected))
        _require(
            matches,
            f"{where}: {key}={actual!r}, expected {expected!r}",
        )


def _validate_rows_are_protocol_decoder(
    rows: Sequence[dict[str, Any]],
    where: str,
    *,
    rounds: int | None = None,
    shots: int | None = None,
    racing: dict[str, Any] | None = None,
) -> int:
    """Check every measured (ok) row; return how many were checked.

    Rows in TERMINAL_ROW_STATUSES (schedule_catastrophic, v3.4
    decode_infeasible) are accepted as terminal outcomes and skipped; any
    other non-ok row carries no decoder evidence to check either.

    With ``shots`` given (2026-09-11, pilot10x_v1) every row must either have
    decoded exactly ``shots`` per basis (a confirmed measurement) or carry
    stopped_by="racing" with at least the pilot and fewer than ``shots``
    per basis (an early stop); any other stopped_by value is a defect.
    """

    checked = 0
    for row in rows:
        if row.get("status") in TERMINAL_ROW_STATUSES:
            continue
        if row.get("status") != "ok":
            continue
        checked += 1
        _require(
            str(row.get("decoder") or "") == FORMAL_DECODER,
            f"{where}: row {row.get('program_id')} decoder={row.get('decoder')!r}",
        )
        _validate_decoder_blob(
            row.get("decoder_config"),
            f"{where}: row {row.get('program_id')}",
        )
        # v3.5: every measured row records the noise point it sampled.
        _validate_noise_point(
            row.get("noise_point"),
            row.get("noise_value"),
            f"{where}: row {row.get('program_id')}",
        )
        if rounds is not None:
            # Audit fix #1: "the config would have set T" is not evidence that
            # T was measured — every accepted row must RECORD the frozen round
            # count, and a missing field fails instead of passing.
            # max_Tc_d: the frozen count is max(T_c, the row's d_search).
            expected = _expected_rounds(rounds, row)
            _require(
                int(row.get("rounds") or 0) == expected,
                f"{where}: row {row.get('program_id')} rounds={row.get('rounds')!r}, "
                f"expected {expected} {_rounds_rule(int(rounds), row)}",
            )
        if shots is not None:
            _validate_row_shots(row, where, shots=int(shots), racing=racing or {})
    return checked


def _validate_row_shots(row: dict[str, Any], where: str, *, shots: int, racing: dict[str, Any]) -> None:
    """Confirmed rows decoded exactly ``shots`` per basis; racing-stopped
    rows decoded [pilot, shots) per basis and say so in stopped_by;
    max_errors stops (capped policies) decoded [pilot, shots]."""

    prefix = f"{where}: row {row.get('program_id')}"
    shots_x = int(row.get("shots_x") or 0)
    shots_z = int(row.get("shots_z") or 0)
    stopped_by = row.get("stopped_by")
    if stopped_by in (None, ""):
        _require(
            shots_x == shots and shots_z == shots,
            f"{prefix}: shots_x/shots_z={shots_x}/{shots_z}, expected {shots} per basis "
            "(confirmed measurement)",
        )
        return
    pilot = int(racing.get("shots_pilot") or 0)
    if stopped_by == "max_errors":
        # pilot10x_maxerr100_v2: every basis reached the error cap; the row is
        # confirmed at [pilot, shots] per basis (the cap can coincide with
        # the confirm boundary). pilot10x_maxerr100_v3 decides the cap at
        # every shard, so the floor is one shard (batch) instead of the pilot.
        _require(
            bool(racing.get("enabled")) and racing.get("max_errors") is not None,
            f"{prefix}: stopped_by='max_errors' but this turn's racing policy "
            f"{racing.get('policy_id')!r} has no max_errors cap",
        )
        floor = int(racing.get("batch") or 0) if bool(racing.get("max_errors_in_pilot")) else pilot
        _require(
            floor <= min(shots_x, shots_z) and max(shots_x, shots_z) <= shots,
            f"{prefix}: max_errors stop decoded shots_x/shots_z={shots_x}/{shots_z}, expected "
            f"[{floor}, {shots}] per basis",
        )
        return
    _require(
        stopped_by == "racing" and bool(racing.get("enabled")),
        f"{prefix}: stopped_by={stopped_by!r} is not a protocol outcome "
        "(only a racing early stop, a max_errors stop under a capped policy, or a confirmed measurement is)",
    )
    _require(
        pilot <= min(shots_x, shots_z) and max(shots_x, shots_z) < shots,
        f"{prefix}: early stop decoded shots_x/shots_z={shots_x}/{shots_z}, expected "
        f"[{pilot}, {shots}) per basis",
    )


def _validate_run_config(
    path: Path,
    where: str,
    *,
    rounds: int | None = None,
    seed_exact: int | None = None,
    seed_below: int | None = None,
    expected_shards: int = FORMAL_SHOT_SHARDS,
    n_budget: int | None = None,
    expected_shots: int | None = None,
    expected_stage: str | None = None,
) -> None:
    config = _read_json(path)
    verification = config.get("verification") or {}
    _validate_execution_geometry(
        shot_shards=config.get("shot_shards"),
        policy=verification,
        where=where,
        expected_shards=expected_shards,
    )
    # Fresh-seed audit: "fresh shots" must be a recorded fact. The final stage
    # must run at EXACTLY the namespace-derived base (>= 1e12); search stages
    # must stay strictly below that ceiling so no final key can collide with
    # a search key.
    if seed_exact is not None:
        _require(
            int(config.get("seed") if config.get("seed") is not None else -1)
            == int(seed_exact),
            f"{where}: seed={config.get('seed')!r}, expected the namespace-derived "
            f"final base {int(seed_exact)}",
        )
    if seed_below is not None:
        _require(
            config.get("seed") is not None
            and 0 <= int(config.get("seed")) < int(seed_below),
            f"{where}: seed={config.get('seed')!r} is not a search-range seed "
            f"(< {int(seed_below)})",
        )
    _require(
        list(verification.get("decoders") or []) == [FORMAL_DECODER],
        f"{where}: decoders={verification.get('decoders')!r}",
    )
    decoder_build = verification.get("decoder_build") or {}
    _validate_decoder_blob(decoder_build.get("telescoping"), where)
    # v3.6 budget policy: the executed shots per basis are the point's.
    if expected_shots is not None:
        shots = verification.get("shots")
        _require(
            type(shots) is int and shots == int(expected_shots),
            f"{where}: shots={shots!r}, expected {int(expected_shots)} "
            f"(budget policy for n_budget={n_budget})",
        )
    # v3.6: the pool records the stage it tagged every shard request with.
    if expected_stage is not None and config.get("stage") is not None:
        _require(
            config.get("stage") == expected_stage,
            f"{where}: stage={config.get('stage')!r}, expected {expected_stage!r}",
        )
    # v3.5: the executed noise sweep travels in the run config too.
    _validate_noise_sweep(verification.get("noise"), where)
    # v3.5 SEC: so does the executed circuit policy.
    _validate_sec_policy_blob(verification.get("sec_policy"), where)
    if rounds is not None:
        _require(
            int(verification.get("rounds") or 0) == rounds,
            f"{where}: rounds={verification.get('rounds')!r}, expected {rounds}",
        )
    # Audit fix #2: the run config is what the EXECUTORS read (fast/slow proxy
    # both source the structural block since the single-source fix), so the
    # executed d_search protocol is validated here — not just the snapshot's
    # declaration of it.
    structural = verification.get("structural") or {}
    _check_structural_retention(structural, where=f"{where}: structural")
    _require(
        int(verification.get("proxy_iters") or 0)
        == STRUCTURAL_EXPECTED["proxy_iters_per_seed"],
        f"{where}: proxy_iters={verification.get('proxy_iters')!r} disagrees with "
        "the frozen per-seed effort",
    )


def _validate_candidate_job_configs(
    candidate_dir: Path, where: str, *, expected_shards: int = FORMAL_SHOT_SHARDS
) -> None:
    """Cross-check executor-emitted job geometry when candidate artifacts exist.

    Some structurally rejected candidates legitimately never reach the pool and
    therefore carry no ``job_config``. Every config the pool did emit must,
    however, match the frozen partition and watchdog policy.
    """

    if not candidate_dir.is_dir():
        return
    for path in sorted(candidate_dir.glob("*.json")):
        payload = _read_json(path)
        job_config = payload.get("job_config")
        if job_config is None:
            continue
        _require(
            isinstance(job_config, dict),
            f"{where}: {path.name} job_config is not an object",
        )
        _validate_execution_geometry(
            shot_shards=job_config.get("shot_shards"),
            policy=job_config,
            where=f"{where}: {path.name} job_config",
            expected_shards=expected_shards,
        )


def _validate_candidate_noise_points(candidate_dir: Path, where: str) -> None:
    """v3.5: every slow result a candidate recorded sampled the frozen point.

    A candidate that never reached the pool carries no results; every
    result that does exist names its noise point (the pipeline stamps it on
    ok and failed results alike), and an ok result without one fails."""

    if not candidate_dir.is_dir():
        return
    for path in sorted(candidate_dir.glob("*.json")):
        payload = _read_json(path)
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            if result.get("status") != "ok" and result.get("noise_point") is None:
                continue
            _validate_noise_point(
                result.get("noise_point"),
                result.get("noise_value"),
                f"{where}: {path.name} results[{index}]",
            )


def _exact_racing_switches(racing_entries: list[dict[str, Any]]) -> list[tuple[int, Any, Any]]:
    """(applies_from_turn, from-racing, to-racing) for every switch that
    records its turn, ordered by turn and, within one turn, in the order the
    switches were recorded.

    2026-10-01: two switches can apply from the same turn (a resume that
    walked v2 -> v4 -> v3 at turn 0); the old plain ``sorted`` then compared
    the racing dicts and raised TypeError on a finished campaign. The sort is
    by turn only and stable, so the later switch at a turn wins, exactly as
    the run applied them."""

    return sorted(
        (
            (
                int(entry["applies_from_turn"]),
                (entry.get("from") or {}).get("racing"),
                (entry.get("to") or {}).get("racing"),
            )
            for entry in racing_entries
            if entry.get("applies_from_turn") is not None
        ),
        key=lambda item: item[0],
    )


def _validate_candidate_schedules(candidate_dir: Path, where: str) -> None:
    """v3.5 SEC: every measured basis records the frozen circuit policy.

    The per-basis memory_experiment.schedule blob is what the circuit builder
    actually emitted (policy id, K, cert budget, depth), so it is checked on
    every ok result rather than trusting the config's declaration."""

    if not candidate_dir.is_dir():
        return
    for path in sorted(candidate_dir.glob("*.json")):
        payload = _read_json(path)
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for index, result in enumerate(results):
            if not isinstance(result, dict) or result.get("status") != "ok":
                continue
            bases = result.get("basis")
            if not isinstance(bases, dict):
                continue
            for basis, blob in bases.items():
                prefix = f"{where}: {path.name} results[{index}] basis {basis}"
                _require(isinstance(blob, dict), f"{prefix}: basis result is not an object")
                experiment = blob.get("memory_experiment")
                # 2026-10-01: a basis that never ran has nothing to audit. A
                # candidate aborted by the shard watchdog (decode_infeasible)
                # can stop before its second basis builds a circuit: zero
                # shots, no memory_experiment. Every basis that recorded a
                # shot is still held to the frozen circuit policy below.
                if experiment is None and not int(blob.get("shots") or 0):
                    continue
                _require(
                    isinstance(experiment, dict),
                    f"{prefix}: memory_experiment is missing",
                )
                sched = experiment.get("schedule")
                _require(isinstance(sched, dict), f"{prefix}: schedule metadata is missing")
                _require(
                    sched.get("sec_policy_id") == FORMAL_SEC_POLICY,
                    f"{prefix}: schedule sec_policy_id={sched.get('sec_policy_id')!r}, "
                    f"expected {FORMAL_SEC_POLICY!r}",
                )
                _require(
                    sched.get("candidates") == FORMAL_SEC_CANDIDATES,
                    f"{prefix}: schedule candidates={sched.get('candidates')!r}, "
                    f"expected {FORMAL_SEC_CANDIDATES}",
                )
                _require(
                    sched.get("cert_iters") == FORMAL_SEC_CERT_ITERS,
                    f"{prefix}: schedule cert_iters={sched.get('cert_iters')!r}, "
                    f"expected {FORMAL_SEC_CERT_ITERS}",
                )
                # v3.6 multi-seed certification (docs/06 §3 M1): the emitted
                # schedule metadata names the seeds and the refine budget.
                _require(
                    sched.get("cert_seeds") == FORMAL_SEC_CERT_SEEDS,
                    f"{prefix}: schedule cert_seeds={sched.get('cert_seeds')!r}, "
                    f"expected {FORMAL_SEC_CERT_SEEDS}",
                )
                _require(
                    sched.get("refine_iters") == FORMAL_SEC_REFINE_ITERS,
                    f"{prefix}: schedule refine_iters={sched.get('refine_iters')!r}, "
                    f"expected {FORMAL_SEC_REFINE_ITERS}",
                )
                _require(
                    isinstance(sched.get("depth"), int),
                    f"{prefix}: schedule depth={sched.get('depth')!r} is not an int",
                )


def _validate_structural_proxy_provenance(path: Path, where: str) -> None:
    """Every structural elite must PROVE its d_search ran the frozen protocol.

    Audit fix #2: a snapshot can declare 120 iters/seed while the executor ran
    something else; the per-seed provenance table each elite carries is actual
    execution evidence, so it is what gets checked."""

    payload = _read_json(path)
    _require(
        payload.get("profile_id") == PROTOCOL_PROFILE,
        f"{where}: profile_id={payload.get('profile_id')!r}",
    )
    for row in payload.get("elites") or []:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("code_hash") or row.get("program_id") or "?")[:12]
        search = row.get("proxy_search")
        _require(
            isinstance(search, dict),
            f"{where}: elite {row_id} carries no proxy_search provenance",
        )
        _require(
            search.get("algorithm_id") == STRUCTURAL_EXPECTED["proxy_algorithm_id"],
            f"{where}: elite {row_id} algorithm_id={search.get('algorithm_id')!r}",
        )
        _require(
            list(search.get("seeds") or []) == STRUCTURAL_EXPECTED["proxy_seeds"],
            f"{where}: elite {row_id} seeds={search.get('seeds')!r}",
        )
        _require(
            int(search.get("iters_per_seed") or 0)
            == STRUCTURAL_EXPECTED["proxy_iters_per_seed"],
            f"{where}: elite {row_id} iters_per_seed={search.get('iters_per_seed')!r}",
        )
        per_seed = search.get("per_seed") or []
        _require(
            len(per_seed) == len(STRUCTURAL_EXPECTED["proxy_seeds"]),
            f"{where}: elite {row_id} per_seed table has {len(per_seed)} entries",
        )
        # Deep consistency (audit): a fabricated table with the right LENGTH
        # used to pass. The per-seed seeds must BE the declared seeds, every
        # distance must be a plausible positive integer, and the aggregates
        # must equal the minima the protocol defines them as.
        _require(
            [entry.get("seed") for entry in per_seed]
            == STRUCTURAL_EXPECTED["proxy_seeds"],
            f"{where}: elite {row_id} per_seed seeds="
            f"{[entry.get('seed') for entry in per_seed]!r}",
        )
        for entry in per_seed:
            for axis in ("d_x", "d_z"):
                value = entry.get(axis)
                _require(
                    isinstance(value, int) and 1 <= value <= 10**6,
                    f"{where}: elite {row_id} per_seed {axis}={value!r} is not a "
                    "plausible distance",
                )
        min_d_x = min(int(entry["d_x"]) for entry in per_seed)
        min_d_z = min(int(entry["d_z"]) for entry in per_seed)
        _require(
            int(search.get("d_x") or 0) == min_d_x
            and int(search.get("d_z") or 0) == min_d_z,
            f"{where}: elite {row_id} aggregate d_x/d_z disagree with the "
            "per-seed minima",
        )
        _require(
            int(search.get("d_search") or 0) == min(min_d_x, min_d_z),
            f"{where}: elite {row_id} d_search={search.get('d_search')!r} is not "
            "min(d_x, d_z)",
        )
        _require(
            search.get("available") is True,
            f"{where}: elite {row_id} proxy_search is not marked available",
        )
        _require(
            int(row.get("d_search") or 0) == int(search.get("d_search") or -1),
            f"{where}: elite {row_id} row d_search={row.get('d_search')!r} disagrees "
            "with its own provenance",
        )


def _validate_versions(run_dir: Path) -> None:
    """The vendored telescoping-decoder package must be resolved (a version
    string, not "absent") at the vendored commit the protocol froze
    (config.TELESCOPING_VENDORED_COMMIT, recorded by
    workflow.checkpoint.resolved_versions as "telescoping_commit"); ldpc /
    relay_bp / scipy serve the dev-only decoders only."""

    versions = _read_json(run_dir / "resolved_versions.json")
    for key in REQUIRED_VERSION_KEYS:
        value = versions.get(key)
        _require(
            isinstance(value, str) and value not in ("", "absent"),
            f"{key} version not resolved: {value!r}",
        )
    _require(
        versions.get("telescoping_commit") == REQUIRED_TELESCOPING_COMMIT,
        f"telescoping_commit is {versions.get('telescoping_commit')!r}, expected "
        f"{REQUIRED_TELESCOPING_COMMIT} (config.TELESCOPING_VENDORED_COMMIT)",
    )


def _report_n_budget(report: dict[str, Any]) -> int:
    """The TOTAL qubit budget point the campaign ran at (spec.qubit_budget);
    the snapshot's own n_budget key must agree."""

    spec = report.get("spec") or {}
    n_budget = spec.get("qubit_budget")
    _require(
        type(n_budget) is int and n_budget > 0,
        f"report spec.qubit_budget={n_budget!r} is not a positive int",
    )
    protocol = report.get("protocol") or {}
    _require(
        protocol.get("n_budget") == n_budget,
        f"protocol snapshot: n_budget={protocol.get('n_budget')!r}, expected "
        f"{n_budget} (spec.qubit_budget)",
    )
    return int(n_budget)


def _validate_protocol(
    report: dict[str, Any],
    *,
    turns: int,
    p_value: float,
    final_shots: int,
    model: str,
    provider: str,
    racing_pilot: int,
    racing_batch: int,
    racing_confirm: int,
    racing_max: int,
    decoders: Sequence[str],
    n_budget: int,
) -> None:
    protocol = report.get("protocol") or {}
    racing = protocol.get("racing") or {}
    policy = budget_policy(n_budget)
    rounds = protocol.get("rounds_policy") or {}
    verification = protocol.get("verification") or {}
    _require(report.get("schema") == "campaign_report", "wrong report schema")
    # v3.7: no check-weight cap in the formal protocol (a hardware-tightened
    # run is a declared variant, not a formal point). The final profiles
    # (2026-11-10 paper alignment) cap the fast loop at w <= 8.
    expected_cap = (
        (FINAL_EXACT_CHECK_WEIGHT if FINAL_EXACT_CHECK_WEIGHT is not None else FINAL_FAST_LOOP_MAX_CHECK_WEIGHT)
        if FINAL_PROTOCOL
        else None
    )
    _require(
        verification.get("max_check_weight") == expected_cap,
        f"protocol snapshot: verification.max_check_weight={verification.get('max_check_weight')!r}, "
        f"expected {expected_cap!r} (construction-time check-weight cap of this profile)",
    )
    _require(
        str(verification.get("task") or "memory") == FORMAL_TASK,
        f"protocol snapshot: verification.task={verification.get('task')!r}, expected {FORMAL_TASK!r}",
    )
    if FINAL_PROTOCOL:
        _validate_final_protocol_keys(protocol, policy)
    # v3.7: the final shortlist is 2 structural endpoints + 4 by measured plpr.
    shortlist = report.get("final_shortlist") or {}
    if FINAL_PROTOCOL:
        # The paper's final stage: the top 5 (Fig. 4 c: 6) distinct codes by search-stage pLER.
        top = int(paper_final.FINAL_STAGE_FIXED_WEIGHT_TOP if FINAL_EXACT_CHECK_WEIGHT is not None else paper_final.FINAL_STAGE_TOP)
        _require(
            isinstance(shortlist, dict) and shortlist.get("schema") == paper_final.COMPOSITION_SCHEMA,
            f"final shortlist schema {shortlist.get('schema') if isinstance(shortlist, dict) else shortlist!r}, "
            f"expected {paper_final.COMPOSITION_SCHEMA!r}",
        )
        selected = list(shortlist.get("selected") or [])
        distinct_available = int(shortlist.get("available") or 0) - len(shortlist.get("skipped_duplicates") or [])
        _require(
            shortlist.get("selection_source") == "search_measured_ranking"
            and int(shortlist.get("top") or 0) == top
            and 1 <= len(selected) <= top
            and (len(selected) == top or len(selected) == distinct_available)
            and [row.get("search_rank") for row in selected] == list(range(1, len(selected) + 1)),
            f"final shortlist {shortlist.get('selection_source')!r} / top {shortlist.get('top')!r} / "
            f"{len(selected)} finalists is not the top {top} distinct codes by search-stage pLER",
        )
        plprs = [row.get("search_per_logical_per_round") for row in selected]
        _require(
            all(value is not None for value in plprs) and plprs == sorted(plprs),
            f"final shortlist: finalists are not in search-stage pLER order ({plprs!r})",
        )
    elif isinstance(shortlist, dict) and shortlist.get("schema") == "final_shortlist_composition":
        _require(
            (int(shortlist.get("structural_slots") or 0), int(shortlist.get("runtime_slots") or 0))
            == (STRUCTURAL_EXPECTED["final_structural"], STRUCTURAL_EXPECTED["final_runtime"]),
            f"final shortlist composition {shortlist.get('structural_slots')!r}S+"
            f"{shortlist.get('runtime_slots')!r}R is not the frozen "
            f"{STRUCTURAL_EXPECTED['final_structural']}S+{STRUCTURAL_EXPECTED['final_runtime']}R",
        )
    # max_turns lowered on resume (2026-09-05): a point that had already run
    # past the new bound reports every turn it ran; the bound itself is what
    # the snapshot must carry (checked below) and what the turn audit covers.
    lowered_from = _original_max_turns(protocol)
    if lowered_from is not None and int(lowered_from) > turns:
        _require(
            turns <= int(report.get("turns_completed") or 0) <= int(lowered_from),
            f"incomplete campaign: turns={report.get('turns_completed')!r}/{turns} "
            f"(max_turns lowered from {lowered_from})",
        )
    else:
        _require(
            int(report.get("turns_completed") or 0) == turns,
            f"incomplete campaign: turns={report.get('turns_completed')!r}/{turns}",
        )
    _require((report.get("stop") or {}).get("reason") == "max_turns", "wrong stop reason")
    _require(report.get("final_slow_status") == "ok", "final_slow_status is not ok")
    _require(int(protocol.get("max_turns") or 0) == turns, "protocol max_turns mismatch")
    _require(protocol.get("strict_turns") is True, "strict_turns is not enabled")
    _require("reference_codes" not in protocol,
             "protocol carries reference_codes: a declared protocol variant, not a formal run")
    _require(protocol.get("fixed_program_count") is True, "program count is not fixed")
    _require(
        protocol.get("required_provider") == provider,
        f"provider is not {provider}",
    )
    _require(protocol.get("required_model") == model, "required model mismatch")
    _require(
        int(protocol.get("proposal_count") or 0) == FORMAL_PROPOSAL_COUNT,
        f"proposal_count is not {FORMAL_PROPOSAL_COUNT}",
    )
    _require(
        int(protocol.get("generations_per_turn") or 0) == FORMAL_GENERATIONS_PER_TURN,
        f"generations_per_turn is not {FORMAL_GENERATIONS_PER_TURN}",
    )
    _require(
        int(protocol.get("rng_seeds_per_program") or 0) == FORMAL_RNG_SEEDS_PER_PROGRAM,
        f"rng_seeds_per_program is not {FORMAL_RNG_SEEDS_PER_PROGRAM}",
    )
    _require(
        protocol.get("quality_metric") == FORMAL_QUALITY_METRIC,
        f"protocol snapshot: quality_metric={protocol.get('quality_metric')!r}, "
        f"expected {FORMAL_QUALITY_METRIC!r} (Q = k*d^2; a missing key means the "
        "archive was ranked under ver1's k*d^2/n)",
    )
    # v3.5: the caller cannot AUTHORIZE another p. Checked before the anchor
    # comparison so a v3.4-style `--p 0.003` is refused as a protocol
    # variant, not reported as a bookkeeping mismatch.
    _require(
        _same_float(p_value, FORMAL_P),
        f"caller declared p={p_value!r}, but {PROTOCOL_PROFILE} freezes "
        f"p={FORMAL_P}; a different p is a different protocol and must not "
        "be validated as this one",
    )
    _require(_same_float(protocol.get("anchor_p"), p_value), "anchor p mismatch")
    final_p_values = protocol.get("final_p_values") or []
    _require(
        len(final_p_values) == 1 and _same_float(final_p_values[0], p_value),
        "final p values mismatch",
    )
    _require(int(protocol.get("final_shots") or 0) == final_shots, "final shots mismatch")
    # v3.6 budget policy: the point's shots are a protocol table, not a
    # launcher knob. The caller's final_shots was already resolved against the
    # policy (validate()), the snapshot's search shots are checked here.
    _validate_budget_policy(protocol.get("budget_policy"), n_budget, "protocol snapshot")
    search_shots = verification.get("shots")
    _require(
        type(search_shots) is int and search_shots == int(policy["search_shots"]),
        f"protocol snapshot: verification.shots={search_shots!r}, expected "
        f"{policy['search_shots']} (budget policy for n_budget={n_budget})",
    )
    _require(
        int(protocol.get("final_shots") or 0) == int(policy["final_shots"]),
        f"protocol snapshot: final_shots={protocol.get('final_shots')!r}, expected "
        f"{policy['final_shots']} (budget policy for n_budget={n_budget})",
    )
    # v3.5: the noise model is a protocol field. The paper's SI C uniform
    # model at p=0.005 IS the protocol (the caller's p was already checked
    # against it above), and a v3.4 snapshot (uniform_legacy_v1 idle noise,
    # no noise_values key) fails here by construction.
    _require(
        protocol.get("noise_profile") == FORMAL_NOISE_PROFILE,
        f"protocol snapshot: noise_profile={protocol.get('noise_profile')!r}, "
        f"expected {FORMAL_NOISE_PROFILE!r}",
    )
    noise_values = protocol.get("noise_values")
    _require(
        isinstance(noise_values, list)
        and len(noise_values) == 1
        and _same_float(noise_values[0], FORMAL_P),
        f"protocol snapshot: noise_values={noise_values!r}, expected [{FORMAL_P}]",
    )
    _validate_noise_sweep(verification.get("noise"), "protocol snapshot")
    # v3.5 SEC: the circuit policy is a protocol field. A v3.4 snapshot says
    # lrc_ranked_v1 and has no sec_candidates / sec_cert_iters keys.
    _require(
        protocol.get("sec_policy_id") == FORMAL_SEC_POLICY,
        f"protocol snapshot: sec_policy_id={protocol.get('sec_policy_id')!r}, "
        f"expected {FORMAL_SEC_POLICY!r}",
    )
    sec_candidates = protocol.get("sec_candidates")
    _require(
        type(sec_candidates) is int and sec_candidates == FORMAL_SEC_CANDIDATES,
        f"protocol snapshot: sec_candidates={sec_candidates!r}, "
        f"expected {FORMAL_SEC_CANDIDATES}",
    )
    sec_cert_iters = protocol.get("sec_cert_iters")
    _require(
        type(sec_cert_iters) is int and sec_cert_iters == FORMAL_SEC_CERT_ITERS,
        f"protocol snapshot: sec_cert_iters={sec_cert_iters!r}, "
        f"expected {FORMAL_SEC_CERT_ITERS}",
    )
    _validate_sec_policy_blob(verification.get("sec_policy"), "protocol snapshot")
    _validate_execution_geometry(
        shot_shards=protocol.get("shot_shards"),
        policy=verification,
        where="protocol snapshot",
        expected_shards=FORMAL_SHOT_SHARDS,
    )
    # v3.4: the final stage's own partition count is a protocol field too. A
    # v3.3 snapshot has no such key and fails here by construction.
    final_shot_shards = protocol.get("final_shot_shards")
    _require(
        type(final_shot_shards) is int and final_shot_shards == FORMAL_FINAL_SHOT_SHARDS,
        f"protocol snapshot: final_shot_shards={final_shot_shards!r}, "
        f"expected {FORMAL_FINAL_SHOT_SHARDS}",
    )
    _require(
        protocol.get("protocol_profile") == PROTOCOL_PROFILE,
        f"protocol profile is {protocol.get('protocol_profile')!r}, "
        f"expected {PROTOCOL_PROFILE!r}",
    )
    _require(
        rounds.get("search") == FORMAL_ROUNDS_POLICY,
        f"search rounds policy is {rounds.get('search')!r}, expected {FORMAL_ROUNDS_POLICY!r}",
    )
    _require(
        int(rounds.get("search_rounds") or 0) == SEARCH_ROUNDS,
        f"search rounds floor is not T_c={SEARCH_ROUNDS}",
    )
    # 2026-09-10: the final stage applies the same max(T_c, d_search) rule
    # with the same floor (the retired "T=d" label meant T=max(3, d_proxy)).
    _require(
        rounds.get("final") == FORMAL_ROUNDS_POLICY,
        f"final rounds policy is {rounds.get('final')!r}, expected {FORMAL_ROUNDS_POLICY!r}",
    )
    _require(int(rounds.get("final_rounds") or 0) == SEARCH_ROUNDS, f"final rounds floor is not T_c={SEARCH_ROUNDS}")
    _require(
        rounds.get("final_rng_namespace") == STRUCTURAL_EXPECTED["final_rng_namespace"],
        f"final RNG namespace is {rounds.get('final_rng_namespace')!r}",
    )
    # 2026-09-10: the slow-promotion floor is d_search >= 5 (was 3). A run
    # promoted under a lower floor spent its slow slots differently.
    _require(
        int(rounds.get("min_promotion_distance") or 0) == int(ranking.MIN_PROMOTION_DISTANCE),
        f"min_promotion_distance={rounds.get('min_promotion_distance')!r}, "
        f"expected {int(ranking.MIN_PROMOTION_DISTANCE)}",
    )
    # Structural retention: reducer identity, caps and quotas. A run whose
    # quotas differ produced its codes under different selection pressure and
    # is not comparable, however similar the numbers look.
    structural = protocol.get("structural_retention") or {}
    _check_structural_retention(structural)
    _require(
        int(protocol.get("promote_top") or 0) == STRUCTURAL_EXPECTED["slow_candidate_cap"],
        f"promote_top={protocol.get('promote_top')!r} is not the frozen slow-queue cap",
    )
    # 2026-09-11: the formal protocol races (early stop, pilot10x_v1). The
    # snapshot's racing block must BE the frozen policy for this point's
    # search shots: pilot 2000 in 250-shot shards, confirm = max = the
    # policy's search shots, epsilon 9.0 (10x the hall-of-fame best),
    # baseline arm off, no max_errors. Racing OFF (the v3.3-v3.6 fixed-shot
    # measurement) is now the declared variant, not the formal run.
    # 2026-09-11 evening: two formal policies exist (pilot10x_v1 and
    # pilot10x_maxerr100_v2); a campaign may switch from one to the other at a
    # turn boundary on resume, recorded in runtime_overrides. The snapshot's
    # block is the policy in force at the end; every turn is checked below
    # against the block it actually ran under.
    _require(
        formal_racing_policy_id(racing) is not None
        and int(racing.get("shots_confirm") or 0) == int(policy["search_shots"]),
        f"racing block {racing!r} is not a formal policy for search_shots="
        f"{policy['search_shots']} (expected one of "
        f"{[factory(int(policy['search_shots'])).to_dict() for factory in FORMAL_RACING_POLICIES.values()]!r})",
    )
    for entry in protocol.get("runtime_overrides") or []:
        entry = entry or {}
        for side in ("from", "to"):
            block = (entry.get(side) or {}).get("racing")
            if block is None:
                continue
            _require(
                formal_racing_policy_id(block) is not None
                and int(block.get("shots_confirm") or 0) == int(policy["search_shots"]),
                f"runtime_overrides entry at turn {entry.get('turn')!r}: racing {side!r} "
                f"block {block!r} is not a formal policy",
            )
    # ONE decoder, at ONE configuration, in the snapshot itself. The caller
    # cannot AUTHORIZE a different one: since v3.3 the decoder is the
    # protocol, so `--decoders lsd` describes a different experiment, not a
    # declared variant of this one.
    _require(
        list(decoders) == list(DEFAULT_DECODERS),
        f"caller declared decoders={list(decoders)!r}, but {PROTOCOL_PROFILE} "
        f"is {list(DEFAULT_DECODERS)!r}-only; a different decoder set is a "
        "different protocol and must not be validated as this one",
    )
    _require(
        list(verification.get("decoders") or []) == list(DEFAULT_DECODERS),
        f"decoder set mismatch: report={verification.get('decoders')!r} "
        f"expected={list(DEFAULT_DECODERS)!r}",
    )
    # v3.6: the snapshot names the decoder and carries the full telescoping
    # blob at the top level too (a v3.5 snapshot has neither key; a snapshot
    # of the retired self-built cascade carries `cascade` instead).
    _require(
        protocol.get("decoder") == FORMAL_DECODER,
        f"protocol snapshot: decoder={protocol.get('decoder')!r}, expected {FORMAL_DECODER!r}",
    )
    _validate_decoder_blob(protocol.get("telescoping"), "protocol snapshot: telescoping")
    decoder_build = verification.get("decoder_build") or {}
    _validate_decoder_blob(decoder_build.get("telescoping"), "protocol snapshot")
    _require(
        (protocol.get("telescoping") or {}) == (decoder_build.get("telescoping") or {}),
        "protocol snapshot: top-level telescoping blob disagrees with "
        "verification.decoder_build.telescoping",
    )


def _validate_final_protocol_keys(protocol: dict[str, Any], policy: dict[str, Any]) -> None:
    """Final profiles: the slow-loop gate and the early-stop policy held from turn 0, and none of the
    other declared variants is present. With FINAL_EXACT_CHECK_WEIGHT (fig3_final --exact-check-weight) the
    run carries exactly that contract, its gate is the contract's weight and its final stage is the
    fixed-weight one."""

    exact = FINAL_EXACT_CHECK_WEIGHT
    refused = ("ablation", "reference_codes") + (() if exact is not None else ("exact_check_weight",))
    for key in refused:
        _require(key not in protocol, f"protocol snapshot carries {key!r}: a declared variant, not a final-protocol run")
    if exact is not None:
        _require(
            protocol.get("exact_check_weight") == exact
            and (protocol.get("verification") or {}).get("exact_check_weight", exact) == exact,
            f"protocol snapshot: exact_check_weight={protocol.get('exact_check_weight')!r}, expected {exact} "
            "(the fixed-weight search this validation was asked for)",
        )
    gate = exact if exact is not None else FINAL_PROMOTION_GATE
    _require(
        protocol.get("promotion_max_check_weight") == gate,
        f"promotion_max_check_weight={protocol.get('promotion_max_check_weight')!r}, expected {gate}",
    )
    expected_racing = paper_racing(int(policy["search_shots"])).to_dict()
    _require(
        protocol.get("racing") == expected_racing,
        f"racing policy {formal_racing_policy_id(protocol.get('racing'))!r}, expected "
        f"{FINAL_RACING_POLICY!r} ({expected_racing!r})",
    )
    expected_stage = paper_final.final_stage_protocol(
        task=FORMAL_TASK,
        n_budget=protocol.get("n_budget"),
        confirm_shots=int(policy["final_shots"]),
        final_shot_shards=FORMAL_FINAL_SHOT_SHARDS,
        exact_check_weight=exact,
    )
    _require(
        protocol.get("final_stage") == expected_stage,
        f"protocol snapshot: final_stage={protocol.get('final_stage')!r}, expected the paper's final stage "
        f"{expected_stage!r}",
    )
    for entry in protocol.get("runtime_overrides") or []:
        changed = sorted(set((entry or {}).get("to") or {}) & {"promotion_max_check_weight", "racing"})
        _require(not changed, f"runtime_overrides at turn {(entry or {}).get('turn')!r} changed {changed}: "
                              "the final protocols hold them from turn 0")


def _validate_circuit_rounds(run_dir: Path) -> int:
    """fixed_Tc: every measured circuit -- each basis of each slow-loop and final-stage candidate -- ran at
    T = T_c. A row records the T its rate was normalised by; the candidate file records the T of the circuit
    actually sampled, which is where the 2026-10-27 rounds bug (d > 12 codes built at T = d) showed."""

    checked = 0
    paths = (
        sorted(run_dir.glob("turns/turn_*/cycle/slow/candidates/*.json"))
        + sorted(run_dir.glob("final_slow/T*/candidates/*.json"))
        + sorted(run_dir.glob("final_slow/topup/*/candidates/*.json"))
    )
    for path in paths:
        cand = _read_json(path)
        for result in cand.get("results") or []:
            for basis in ("X", "Z"):
                meta = (((result or {}).get("basis") or {}).get(basis) or {}).get("memory_experiment") or {}
                if meta.get("rounds") is None:
                    continue
                checked += 1
                _require(
                    int(meta["rounds"]) == SEARCH_ROUNDS,
                    f"{path.relative_to(run_dir)}: {basis} circuit ran at T={meta['rounds']}, expected "
                    f"T = {SEARCH_ROUNDS} (fixed_Tc)",
                )
    return checked


def _validate_reasoning_rule(run_dir: Path, *, model: str, provider: str, flag_path: Path | None) -> None:
    """Final profiles: the model is the provider's final model at its default level, or the fallback level of
    the reasoning-saturation rule with the run's reasoning_fallback.json and the arm's flag behind it."""

    _require(
        reasoning_fallback.base_model(model) in FINAL_MODELS.get(provider, ()),
        f"model {model!r} is not a final-protocol model for provider {provider!r} ({FINAL_MODELS.get(provider)})",
    )
    flag = reasoning_fallback.read_flag(flag_path) if flag_path is not None else None
    record_path = run_dir / reasoning_fallback.RECORD_NAME
    if reasoning_fallback.effort_suffix(model) is None:
        _require(not record_path.exists(), f"{record_path.name} present but the run is at the default level")
        _require(flag is None or str(flag["from_model"]) != model,
                 f"the arm moved to {flag and flag['to_model']!r} under the reasoning-saturation rule, "
                 f"but this run is at the default level {model!r}")
        return
    _require(model == reasoning_fallback.fallback_model(model),
             f"model {model!r}: the reasoning-saturation rule only lowers to ({reasoning_fallback.FALLBACK_LEVEL})")
    _require(record_path.is_file(), f"a run at {model!r} must carry {record_path.name} (the rule's record)")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    recorded_flag = record.get("flag") or {}
    _require(
        record.get("schema") == reasoning_fallback.RECORD_SCHEMA and record.get("model") == model
        and recorded_flag.get("to_model") == model
        and recorded_flag.get("from_model") == reasoning_fallback.base_model(model)
        and bool(recorded_flag.get("evidence")),
        f"{record_path.name} does not record the switch {reasoning_fallback.base_model(model)!r} -> {model!r} with its evidence",
    )
    if flag_path is not None:
        _require(flag == recorded_flag, f"{record_path.name} disagrees with the arm's flag {flag_path}")


def _validate_turns(
    run_dir: Path, *, turns: int, model: str, provider: str, n_budget: int,
    lowered_from: int | None = None,
) -> None:
    policy = budget_policy(n_budget)
    turns_dir = run_dir / "turns"
    _require(turns_dir.is_dir(), f"missing turns directory: {turns_dir}")
    expected_names = {f"turn_{turn:04d}" for turn in range(turns)}
    actual_names = {
        path.name
        for path in turns_dir.iterdir()
        if path.is_dir() and TURN_RE.match(path.name)
    }
    # max_turns lowered on resume: turn dirs in [turns, original) may exist
    # (they were run under the earlier bound; the final stage ignored them).
    if lowered_from is None:
        snapshot_path = run_dir / "protocol_snapshot.json"
        lowered_from = _original_max_turns(_read_json(snapshot_path) if snapshot_path.is_file() else {})
    tolerated = (
        {f"turn_{turn:04d}" for turn in range(turns, int(lowered_from))}
        if lowered_from is not None and int(lowered_from) > turns
        else set()
    )
    snapshot_for_racing = _read_json(run_dir / "protocol_snapshot.json") if (run_dir / "protocol_snapshot.json").is_file() else {}
    racing_entries = [
        entry
        for entry in (snapshot_for_racing.get("runtime_overrides") or [])
        if isinstance(entry, dict) and (entry.get("to") or {}).get("racing") is not None
    ]
    # A switch entry says from which turn the new policy applies
    # ("applies_from_turn" = the audited-turn count at resume, unambiguous);
    # entries written before that field existed only carry the checkpoint's
    # turn index, which is ambiguous by one, so those runs are checked with
    # the order rule below instead of an exact turn.
    exact_switches = _exact_racing_switches(racing_entries)
    exact_history = bool(racing_entries) and len(exact_switches) == len(racing_entries)
    # 2026-09-29: the forward-only rule is config.RACING_POLICY_RANK (v4 ranks
    # with v3: the 09-15 withdrawal v3 -> v4 -> v3 is a recorded rollback).
    seen_turn_policies: list[tuple[int, str]] = []

    def racing_expected_for_turn(turn: int) -> dict[str, Any] | None:
        """The racing block the switch history says `turn` ran under (exact
        histories only); None when the history cannot say."""

        if not racing_entries:
            expected = snapshot_for_racing.get("racing")
            return expected if isinstance(expected, dict) else None
        if not exact_history:
            return None
        expected = exact_switches[0][1]  # the policy frozen at launch
        for switch_turn, _before, after in exact_switches:
            if turn >= switch_turn:
                expected = after
        return expected if isinstance(expected, dict) else None

    def note_turn_policy(turn: int, policy_id: str) -> None:
        """Policies may only move forward (pilot10x_v1 -> pilot10x_maxerr100_v2
        -> pilot10x_maxerr100_v3) and never back; the last turn must match the
        snapshot's block."""

        if seen_turn_policies:
            _prev_turn, prev = seen_turn_policies[-1]
            _require(
                racing_policy_moves_forward(prev, policy_id),
                f"turn {turn}: racing policy {policy_id!r} after {prev!r} -- policies may only move forward",
            )
        seen_turn_policies.append((turn, policy_id))
    _require(
        expected_names <= actual_names and not (actual_names - expected_names - tolerated),
        f"turn directory set is not exactly 0..{turns - 1}"
        + (f" (plus the turns run before max_turns was lowered from {lowered_from})" if tolerated else ""),
    )

    # Final profiles: the per-turn racing thresholds are the paper's (Appendix D3) -- the upper bound of the
    # best measured code (lowest search-stage pLER, ties: smaller upper bound) of the earlier turns and the
    # upper bound of the applicable reference code once it has been measured -- recomputed from the rows.
    threshold_rows: list[tuple[float, float]] = []
    reference_ub: list[float] = []
    reference = paper_final.resident_reference(int(n_budget)) if FINAL_PROTOCOL else None
    for turn in range(turns):
        turn_dir = turns_dir / f"turn_{turn:04d}"
        audit = _read_json(turn_dir / "turn_audit.json")
        checks = audit.get("checks") or {}
        _require(audit.get("schema") == "campaign_turn_audit_v1", f"turn {turn}: schema")
        _require(int(audit.get("turn", -1)) == turn, f"turn {turn}: audit number")
        _require(audit.get("status") == "complete", f"turn {turn}: audit incomplete")
        _require(not (audit.get("errors") or []), f"turn {turn}: audit errors present")
        _require(checks.get("plan_policy") == "agentic_v1", f"turn {turn}: fallback policy")
        _require(checks.get("provider") == provider, f"turn {turn}: audit provider")
        _require(checks.get("model") == model, f"turn {turn}: audit model")
        _require(
            int(checks.get("program_count") or 0) == FORMAL_PROPOSAL_COUNT,
            f"turn {turn}: program count",
        )
        _require(
            int(checks.get("generations") or 0) == FORMAL_GENERATIONS_PER_TURN,
            f"turn {turn}: generations",
        )
        _require(
            int(checks.get("rng_seeds_per_program") or 0) == FORMAL_RNG_SEEDS_PER_PROGRAM,
            f"turn {turn}: RNG seeds",
        )
        _require(checks.get("cycle_status") == "ok", f"turn {turn}: cycle status")
        _require(
            int(checks.get("proposal_count_requested_total") or 0) == FORMAL_PROPOSALS_PER_TURN,
            f"turn {turn}: requested proposals",
        )
        _require(
            int(checks.get("proposal_count_missing_total", -1)) == 0,
            f"turn {turn}: missing proposals",
        )
        _require(
            checks.get("generation_status_counts") == {"ok": FORMAL_GENERATIONS_PER_TURN},
            f"turn {turn}: generation status",
        )
        _require((turn_dir / "digest.json").is_file(), f"turn {turn}: missing digest")

        plan = _read_json(turn_dir / "plan.json")
        _require(plan.get("policy") == "agentic_v1", f"turn {turn}: plan fallback")
        _require(plan.get("provider") == provider, f"turn {turn}: plan provider")
        _require(
            int(plan.get("proposal_count") or 0) == FORMAL_PROPOSAL_COUNT,
            f"turn {turn}: plan proposals",
        )
        _require(
            int(plan.get("generations") or 0) == FORMAL_GENERATIONS_PER_TURN,
            f"turn {turn}: plan generations",
        )
        _require(
            int(plan.get("rng_seeds_per_program") or 0) == FORMAL_RNG_SEEDS_PER_PROGRAM,
            f"turn {turn}: plan RNG seeds",
        )

        orchestrator_calls = _read_jsonl(turn_dir / "orchestrator" / "llm_calls.jsonl")
        for row in orchestrator_calls:
            info = row.get("model") or {}
            _require(info.get("provider") == provider, f"turn {turn}: orchestrator provider")
            _require(info.get("model") == model, f"turn {turn}: orchestrator model")

        cycle = _read_json(turn_dir / "cycle" / "agentic_cycle_summary.json")
        _require(cycle.get("status") == "ok", f"turn {turn}: raw cycle status")
        # v3.7: every turn's slow queue was composed under the frozen 2S+4R
        # quota and only by the named channels (structural endpoints, the
        # expected-plpr / cold-start archive order, named promotions).
        # The cycle summary names the composition artifact it wrote; a summary
        # without the reference (pre-v3.7 shape) is not silently accepted as a
        # formal turn -- it is checked below like any other frozen artifact.
        composition_ref = (cycle.get("artifact_paths") or {}).get("slow_queue_composition")
        composition_path = turn_dir / "cycle" / "slow_queue_composition.json"
        composition = _read_json(composition_path) if composition_path.is_file() else None
        if composition is None and not composition_ref:
            # Synthetic / pre-v3.7 tree: nothing was written, nothing is claimed;
            # the quota is then checked against the snapshot alone (above).
            composition = {
                "cap": STRUCTURAL_EXPECTED["slow_candidate_cap"],
                "structural_slots": STRUCTURAL_EXPECTED["structural_slow_slots"],
                "runtime_slots": STRUCTURAL_EXPECTED["runtime_slow_slots"],
                "selected": [],
                "_absent": True,
            }
        _require(
            not composition_ref or composition_path.is_file(),
            f"turn {turn}: cycle summary names slow_queue_composition but the file is missing",
        )
        _require(
            (
                int(composition.get("cap") or 0),
                int(composition.get("structural_slots") or 0),
                int(composition.get("runtime_slots") or 0),
            )
            == (
                STRUCTURAL_EXPECTED["slow_candidate_cap"],
                STRUCTURAL_EXPECTED["structural_slow_slots"],
                STRUCTURAL_EXPECTED["runtime_slow_slots"],
            ),
            f"turn {turn}: slow queue composition {composition.get('structural_slots')!r}S+"
            f"{composition.get('runtime_slots')!r}R (cap {composition.get('cap')!r}) is not the frozen quota",
        )
        for item in composition.get("selected") or []:
            channels = set(item.get("selection_channels") or [])
            _require(
                bool(channels & {"structural", "runtime", "runtime_named"}),
                f"turn {turn}: slow queue row {item.get('code_hash')!r} has no recognised channel {sorted(channels)!r}",
            )
            if "structural" in channels and "runtime" not in channels and "runtime_named" not in channels:
                _require(
                    str(item.get("structural_slot_reason") or "") in FORMAL_STRUCTURAL_SLOT_REASONS,
                    f"turn {turn}: structural slot reason {item.get('structural_slot_reason')!r} "
                    f"not in {sorted(FORMAL_STRUCTURAL_SLOT_REASONS)!r}",
                )
        _require(cycle.get("provider") == provider, f"turn {turn}: raw cycle provider")
        _require(
            int(cycle.get("proposal_count") or 0) == FORMAL_PROPOSAL_COUNT,
            f"turn {turn}: raw cycle count",
        )
        _require(
            int(cycle.get("generations") or 0) == FORMAL_GENERATIONS_PER_TURN,
            f"turn {turn}: raw cycle generations",
        )

        funsearch = _read_json(turn_dir / "cycle" / "fast" / "funsearch_summary.json")
        _require(funsearch.get("provider") == provider, f"turn {turn}: generator provider")
        _require(
            int(funsearch.get("proposal_count_requested_total") or 0) == FORMAL_PROPOSALS_PER_TURN,
            f"turn {turn}: generator requested count",
        )
        _require(
            int(funsearch.get("proposal_count_missing_total", -1)) == 0,
            f"turn {turn}: generator missing count",
        )
        _require(
            funsearch.get("status_counts") == {"ok": FORMAL_GENERATIONS_PER_TURN},
            f"turn {turn}: generator status",
        )
        _require(
            int(funsearch.get("rng_seeds_per_program") or 0) == FORMAL_RNG_SEEDS_PER_PROGRAM,
            f"turn {turn}: generator RNG seeds",
        )

        # Every layer that could carry a different decoder: the turn's slow run
        # config and every measured search row. Search stage = 16 shards.
        _validate_run_config(
            turn_dir / "cycle" / "slow" / "slow_run_config.json",
            f"turn {turn}: slow run config",
            rounds=SEARCH_ROUNDS,
            seed_below=SEARCH_SEED_CEILING,
            expected_shards=FORMAL_SHOT_SHARDS,
            n_budget=n_budget,
            expected_shots=int(policy["search_shots"]),
            expected_stage="search",
        )
        _validate_candidate_job_configs(
            turn_dir / "cycle" / "slow" / "candidates",
            f"turn {turn}: slow candidates",
            expected_shards=FORMAL_SHOT_SHARDS,
        )
        _validate_candidate_noise_points(
            turn_dir / "cycle" / "slow" / "candidates",
            f"turn {turn}: slow candidates",
        )
        _validate_candidate_schedules(
            turn_dir / "cycle" / "slow" / "candidates",
            f"turn {turn}: slow candidates",
        )
        # The racing block this turn RAN under (recorded by the slow stage);
        # it must be a formal policy for the point and must agree with the
        # switch history in the snapshot (initial policy until the recorded
        # switch turn, the new policy from there on).
        turn_racing = (_read_json(turn_dir / "cycle" / "slow" / "slow_run_config.json") or {}).get("racing") or {}
        _require(
            formal_racing_policy_id(turn_racing) is not None
            and int(turn_racing.get("shots_confirm") or 0) == int(policy["search_shots"]),
            f"turn {turn}: slow run config racing block {turn_racing!r} is not a formal policy",
        )
        expected_turn_racing = racing_expected_for_turn(turn)
        _require(
            expected_turn_racing is None or turn_racing == expected_turn_racing,
            f"turn {turn}: ran under racing policy {formal_racing_policy_id(turn_racing)!r} "
            f"but the snapshot's switch history says {formal_racing_policy_id(expected_turn_racing)!r}",
        )
        note_turn_policy(turn, str(formal_racing_policy_id(turn_racing)))
        # Legacy (turn-only) histories cannot pin the switch turn; the last
        # turn must then at least agree with the snapshot's final block. An
        # exact history is already checked turn by turn above (a switch
        # recorded for a turn the campaign never reached is legitimate).
        if turn == turns - 1 and racing_entries and not exact_history and isinstance(snapshot_for_racing.get("racing"), dict):
            _require(
                turn_racing == snapshot_for_racing["racing"],
                f"turn {turn}: the last turn ran under {formal_racing_policy_id(turn_racing)!r} but the "
                f"snapshot's racing block is {formal_racing_policy_id(snapshot_for_racing['racing'])!r}",
            )
        search_rankings = _read_json(turn_dir / "cycle" / "slow" / "rankings.json")
        _validate_rows_are_protocol_decoder(
            [row for row in search_rankings.get("rankings") or [] if isinstance(row, dict)],
            f"turn {turn}: search rankings",
            rounds=SEARCH_ROUNDS,
            shots=int(policy["search_shots"]),
            racing=turn_racing,
        )
        if FINAL_PROTOCOL:
            recorded = (_read_json(turn_dir / "cycle" / "slow" / "slow_run_config.json") or {}).get("racing_thresholds") or {}
            expected_hof = min(threshold_rows)[1] if threshold_rows else None
            expected_ref = reference_ub[0] if reference_ub else None
            for key, expected in (("hof_plpr_ub", expected_hof), ("baseline_plpr_ub", expected_ref)):
                actual = recorded.get(key)
                _require(
                    (actual is None and expected is None)
                    or (actual is not None and expected is not None and math.isclose(float(actual), float(expected), rel_tol=1e-12)),
                    f"turn {turn}: racing threshold {key}={actual!r}, expected {expected!r} (paper Appendix D3)",
                )
            for row in search_rankings.get("rankings") or []:
                if not isinstance(row, dict) or row.get("status") != "ok" or row.get("per_logical_per_round") is None:
                    continue
                ci = row.get("per_logical_per_round_ci") or [None, None]
                if ci[1] is None:
                    continue
                if row.get("is_baseline"):
                    if not reference_ub:
                        reference_ub.append(float(ci[1]))
                    continue
                d_proxy, verified = row.get("d_proxy"), row.get("verified_circuit_distance")
                if (d_proxy is not None and int(d_proxy) < 3) or (verified is not None and int(verified) < 3):
                    continue
                threshold_rows.append((float(row["per_logical_per_round"]), float(ci[1])))
                if reference is not None and str(row.get("code_hash") or "") == reference["code_hash"] and not reference_ub:
                    reference_ub.append(float(ci[1]))
        _validate_structural_proxy_provenance(
            turn_dir / "cycle" / "fast" / "structural_proxy_elites.json",
            f"turn {turn}: structural proxy provenance",
        )

        # Every generation of the turn must have accumulated a full batch from
        # a successful raw LLM call (not only g0001).
        generator_calls: list[dict[str, Any]] = []
        for generation in range(1, FORMAL_GENERATIONS_PER_TURN + 1):
            calls = _read_jsonl(
                turn_dir / "cycle" / "fast" / "generations" / f"g{generation:04d}" / "llm_raw.jsonl"
            )
            _require(bool(calls), f"turn {turn}: generation {generation} has no raw LLM call")
            _require(
                int(calls[-1].get("proposal_count_accumulated") or 0) == FORMAL_PROPOSAL_COUNT,
                f"turn {turn}: generation {generation} raw LLM response did not "
                f"accumulate {FORMAL_PROPOSAL_COUNT} programs",
            )
            generator_calls.extend(calls)
        for row in generator_calls:
            info = row.get("engine_info") or {}
            _require(row.get("status") == "ok", f"turn {turn}: generator call failed")
            _require(info.get("provider") == provider, f"turn {turn}: call provider")
            _require(info.get("model") == model, f"turn {turn}: call model")

        # v3.8 (--profile v3_8): the last `exploit_generations` generations
        # carried the variant rule (or the recorded no-parent fallback), and
        # the slow queue composition records the endpoint family rules.
        if PROTOCOL_PROFILE == PROTOCOL_PROFILE_V38:
            exploit = int(STRUCTURAL_EXPECTED.get("exploit_generations") or 0)
            for generation in range(FORMAL_GENERATIONS_PER_TURN - exploit + 1, FORMAL_GENERATIONS_PER_TURN + 1):
                context = _read_json(
                    turn_dir / "cycle" / "fast" / "generations" / f"g{generation:04d}" / "prompt_context.json"
                ) or {}
                variant = context.get("exploit_generation")
                _require(
                    isinstance(variant, dict) or bool(context.get("exploit_generation_fallback")),
                    f"turn {turn}: generation {generation} is not a v3.8 variant generation",
                )
                if isinstance(variant, dict):
                    _require(
                        int(variant.get("generation") or 0) == generation
                        and int(variant.get("of") or 0) == FORMAL_GENERATIONS_PER_TURN
                        and bool(variant.get("parents")),
                        f"turn {turn}: generation {generation} variant block is malformed",
                    )
            for generation in range(1, FORMAL_GENERATIONS_PER_TURN - exploit + 1):
                context = _read_json(
                    turn_dir / "cycle" / "fast" / "generations" / f"g{generation:04d}" / "prompt_context.json"
                ) or {}
                _require(
                    not isinstance(context.get("exploit_generation"), dict),
                    f"turn {turn}: exploration generation {generation} carries the variant rule",
                )
            if not composition.get("_absent"):
                rules = composition.get("endpoint_rules") or {}
                _require(
                    bool(rules.get("family_distinct")) == bool(STRUCTURAL_EXPECTED["endpoint_family_distinct"])
                    and int(rules.get("demotion_strikes") or 0) == int(STRUCTURAL_EXPECTED["endpoint_demotion_strikes"])
                    and float(rules.get("demotion_factor") or 0.0) == float(STRUCTURAL_EXPECTED["endpoint_demotion_factor"]),
                    f"turn {turn}: slow queue composition endpoint_rules {rules!r} are not the v3.8 rules",
                )


def _validate_final(
    report: dict[str, Any],
    *,
    p_value: float,
    final_shots: int,
    turns: int,
    run_dir: Path | None = None,
    n_budget: int | None = None,
) -> None:
    audit = report.get("turn_audit") or {}
    _require(audit.get("enabled") is True, "turn audit summary is disabled")
    _require(audit.get("schema") == "campaign_turn_audit_summary_v1", "audit summary schema")
    _require(int(audit.get("expected") or 0) == turns, "audit expected count mismatch")
    _require(int(audit.get("completed") or 0) == turns, "audit completed count mismatch")
    _require(int(audit.get("fallback_turns", -1)) == 0, "fallback turns are present")
    _require(audit.get("turns") == list(range(turns)), "audit turn sequence mismatch")

    if FINAL_PROTOCOL:
        _validate_paper_final(report, p_value=p_value, final_shots=final_shots, run_dir=run_dir)
        return

    measurement_audit = report.get("final_measurement_audit") or {}
    _require(measurement_audit.get("enabled") is True, "final measurement audit disabled")
    _require(
        measurement_audit.get("schema") == "campaign_final_measurement_audit_v1",
        "final measurement audit schema",
    )
    _require(measurement_audit.get("status") == "complete", "final measurement audit failed")
    _require(not (measurement_audit.get("errors") or []), "final measurement errors present")
    _require(
        int(measurement_audit.get("shots_per_basis") or 0) == final_shots,
        "final measurement audit shots mismatch",
    )
    _require(
        _same_float(measurement_audit.get("expected_p"), p_value),
        "final measurement audit p mismatch",
    )

    final_rounds = int(
        ((report.get("protocol") or {}).get("rounds_policy") or {}).get("final_rounds") or 0
    )
    _require(
        int(report.get("final_rounds") or 0) == final_rounds,
        "final_slow rounds disagree with the protocol snapshot",
    )
    _require(
        int(measurement_audit.get("expected_rounds") or 0) == final_rounds,
        "final measurement audit rounds disagree with the protocol snapshot",
    )
    expected_final_seed = _expected_final_base_seed(
        str(STRUCTURAL_EXPECTED["final_rng_namespace"])
    )
    _require(
        int(report.get("final_rng_base_seed") or 0) == expected_final_seed,
        f"report final_rng_base_seed={report.get('final_rng_base_seed')!r}, "
        f"expected {expected_final_seed}",
    )
    if run_dir is not None:
        # v3.4: the final stage runs on its own frozen partition count (128).
        _validate_run_config(
            run_dir / "final_slow" / f"T{final_rounds}" / "slow_run_config.json",
            "final run config",
            rounds=final_rounds,
            seed_exact=expected_final_seed,
            expected_shards=FORMAL_FINAL_SHOT_SHARDS,
            n_budget=n_budget,
            expected_shots=int(final_shots),
            expected_stage="final",
        )
        _validate_candidate_job_configs(
            run_dir / "final_slow" / f"T{final_rounds}" / "candidates",
            "final candidates",
            expected_shards=FORMAL_FINAL_SHOT_SHARDS,
        )
        _validate_candidate_noise_points(
            run_dir / "final_slow" / f"T{final_rounds}" / "candidates",
            "final candidates",
        )
        _validate_candidate_schedules(
            run_dir / "final_slow" / f"T{final_rounds}" / "candidates",
            "final candidates",
        )
        final_rankings = _read_json(
            run_dir / "final_slow" / f"T{final_rounds}" / "rankings.json"
        )
        final_ok_rows = _validate_rows_are_protocol_decoder(
            [row for row in final_rankings.get("rankings") or [] if isinstance(row, dict)],
            "final rankings",
            rounds=final_rounds,
        )
        # decode_infeasible rows are terminal, not measurements: a final stage
        # made only of them has nothing to report.
        _require(
            final_ok_rows >= 1,
            "final rankings: no ok row (terminal decode_infeasible/"
            "schedule_catastrophic rows do not count as measurements)",
        )

    frontier = report.get("final_frontier") or []
    _require(bool(frontier), "final frontier is empty")
    _require(
        int(measurement_audit.get("measured_rows") or 0) == len(frontier),
        "final measurement audit row count mismatch",
    )
    for index, row in enumerate(frontier):
        prefix = f"final row {index}"
        _require(_same_float(row.get("noise_value"), p_value), f"{prefix}: p mismatch")
        _require(int(row.get("shots_x") or 0) == final_shots, f"{prefix}: X shots mismatch")
        _require(int(row.get("shots_z") or 0) == final_shots, f"{prefix}: Z shots mismatch")
        _require(
            int(row.get("total_shots") or 0) == 2 * final_shots,
            f"{prefix}: total shots mismatch",
        )
        expected_rounds = _expected_rounds(final_rounds, row)
        _require(
            int(row.get("rounds") or 0) == expected_rounds,
            f"{prefix}: rounds={row.get('rounds')!r}, expected {expected_rounds} "
            + _rounds_rule(final_rounds, row),
        )
        _require(
            str(row.get("decoder") or "") == FORMAL_DECODER, f"{prefix}: decoder mismatch"
        )
        _validate_decoder_blob(row.get("decoder_config"), prefix)


def _rows_by_program(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("rankings") or []:
        if isinstance(row, dict) and row.get("program_id") is not None:
            out.setdefault(str(row["program_id"]), row)
    return out


def _validate_paper_final(
    report: dict[str, Any], *, p_value: float, final_shots: int, run_dir: Path | None
) -> None:
    """The paper's final stage (Appendix D3, Table VII; Fig. 4 c; evaluation/final_stage.py), checked from the
    artifacts: the parts are the planned ones (initial confirmation of every finalist and the reference at the
    final shots, then the top-up parts of ``plan_final_stage`` -- the finalists' to the Table VII limit (none in
    the fixed-weight stage), the reference's to its paper total, a shared finalist's extension -- each at its
    namespace-derived seed), every ok row of every part decoded exactly its part's shots at T = 12, every pooled
    row is the sum of its planned parts with the pLER recomputed from the sums, the ranking is by pooled pLER,
    rank 1 is the champion, and a fixed-weight stage reports its three best ranked codes."""

    from omniqec_agentic.verification.ler import combine_basis_ler_results

    protocol = report.get("protocol") or {}
    block = protocol.get("final_stage") or {}
    rounds = int((protocol.get("rounds_policy") or {}).get("final_rounds") or 0)
    _require(int(report.get("final_rounds") or 0) == rounds, "final_slow rounds disagree with the protocol snapshot")
    _require(report.get("final_stage") == block, "report final_stage disagrees with the protocol snapshot")
    _require(int(block.get("confirm_shots") or 0) == int(final_shots), "final_stage confirm_shots mismatch")
    audit = report.get("final_measurement_audit") or {}
    _require(audit.get("enabled") is True, "final measurement audit disabled")
    _require(audit.get("schema") == "campaign_final_measurement_audit_paper_v1", "final measurement audit schema")
    _require(audit.get("status") == "complete" and not (audit.get("errors") or []), "final measurement audit failed")
    _require(_same_float(audit.get("expected_p"), p_value), "final measurement audit p mismatch")
    _require(int(audit.get("expected_rounds") or 0) == rounds, "final measurement audit rounds mismatch")
    confirm_namespace = str(STRUCTURAL_EXPECTED["final_rng_namespace"])
    _require(
        int(report.get("final_rng_base_seed") or 0) == _expected_final_base_seed(confirm_namespace),
        f"report final_rng_base_seed={report.get('final_rng_base_seed')!r}, expected "
        f"{_expected_final_base_seed(confirm_namespace)}",
    )
    parts = list(report.get("final_stage_parts") or [])
    _require(bool(parts) and parts[0].get("kind") == "confirm", "final stage: the first part is not the confirmation")
    confirm = parts[0]
    _require(
        int(confirm.get("shots") or 0) == int(final_shots)
        and int(confirm.get("shards") or 0) == FORMAL_FINAL_SHOT_SHARDS
        and confirm.get("rng_namespace") == confirm_namespace
        and confirm.get("dir") == f"T{rounds}",
        f"final stage confirmation part {confirm!r} is not {final_shots} shots in {FORMAL_FINAL_SHOT_SHARDS} "
        f"shards at namespace {confirm_namespace!r}",
    )
    shortlist = report.get("final_shortlist") or {}
    finalist_ids = [str(row.get("program_id")) for row in shortlist.get("selected") or []]
    reference = block.get("reference") or None
    reference_row = ((shortlist.get("reference") or {}).get("row")) if reference else None
    expected_members = finalist_ids + ([str(reference["program_id"])] if reference_row == "measured" else [])
    _require(
        list(confirm.get("members") or []) == expected_members,
        f"final stage confirmation members {confirm.get('members')!r}, expected {expected_members!r}",
    )
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    if run_dir is not None:
        for part in parts:
            part_dir = run_dir / "final_slow" / str(part.get("dir"))
            where = f"final part {part.get('name')}"
            _validate_run_config(
                part_dir / "slow_run_config.json",
                f"{where} run config",
                rounds=rounds,
                seed_exact=_expected_final_base_seed(str(part.get("rng_namespace"))),
                expected_shards=int(part.get("shards") or 0),
                expected_shots=int(part.get("shots") or 0),
                expected_stage="final",
            )
            _validate_candidate_job_configs(part_dir / "candidates", f"{where} candidates", expected_shards=int(part.get("shards") or 0))
            _validate_candidate_noise_points(part_dir / "candidates", f"{where} candidates")
            _validate_candidate_schedules(part_dir / "candidates", f"{where} candidates")
            rankings_path = part_dir / "rankings.json"
            rows = [row for row in (_read_json(rankings_path).get("rankings") or []) if isinstance(row, dict)]
            _validate_rows_are_protocol_decoder(rows, f"{where} rankings", rounds=rounds)
            for row in rows:
                if row.get("status") == "ok":
                    _require(
                        int(row.get("shots_x") or 0) == int(part["shots"]) == int(row.get("shots_z") or 0),
                        f"{where}: row {row.get('program_id')} decoded {row.get('shots_x')}/{row.get('shots_z')} "
                        f"shots per basis, planned {part['shots']}",
                    )
            tables[str(part.get("name"))] = _rows_by_program(rankings_path)
        # The top-up parts are exactly the planned ones for the confirmed members of each group.
        confirm_rows = tables.get("confirm") or {}
        plan_entries = [
            {"program_id": pid, "also_reference": bool(row.get("also_reference"))}
            for pid, row in zip(finalist_ids, shortlist.get("selected") or [])
        ] + ([{"program_id": str(reference["program_id"]), "is_baseline": True}] if reference_row == "measured" else [])
        confirmed = {
            str(entry["program_id"])
            for entry in plan_entries
            if (confirm_rows.get(str(entry["program_id"])) or {}).get("status") == "ok"
        }
        plan = paper_final.plan_final_stage(block, plan_entries, rounds=rounds, confirmed=confirmed)
        expected_parts = [confirm.get("name")]
        for planned, members in plan["topups"]:
            expected_parts.append(planned["name"])
            actual = next((part for part in parts if part.get("name") == planned["name"]), None)
            _require(actual is not None, f"final stage: planned part {planned['name']} is missing")
            for key in ("shots", "shards", "rng_namespace", "dir"):
                _require(
                    actual.get(key) == planned[key],
                    f"final part {planned['name']}: {key}={actual.get(key)!r}, planned {planned[key]!r}",
                )
            ok_members = [str(member["program_id"]) for member in members]
            _require(
                list(actual.get("members") or []) == ok_members,
                f"final part {planned['name']}: members {actual.get('members')!r}, expected the confirmed {ok_members!r}",
            )
        _require(
            [part.get("name") for part in parts] == expected_parts,
            f"final stage parts {[part.get('name') for part in parts]!r}, planned {expected_parts!r}",
        )
        ranking = _read_json(run_dir / "final_slow" / "final_ranking.json")
        _require(ranking.get("schema") == paper_final.RANKING_SCHEMA, "final ranking schema")
        pooled_rows = [row for row in ranking.get("rows") or [] if isinstance(row, dict)]
        planned_rows = {str(item["entry"]["program_id"]): item for item in plan["rows"]}
        _require(
            sorted(str(row.get("program_id")) for row in pooled_rows) == sorted(planned_rows),
            f"final ranking rows {sorted(str(row.get('program_id')) for row in pooled_rows)!r}, planned {sorted(planned_rows)!r}",
        )
        parts_by_name = {str(part.get("name")): part for part in parts}
        for row in pooled_rows:
            pid = str(row.get("program_id"))
            item = planned_rows[pid]
            measured_as = str(item["measured_as"])
            _require(
                str(row.get("measured_as") or pid) == measured_as,
                f"final ranking: {pid} measured as {row.get('measured_as')!r}, planned {measured_as!r}",
            )
            sums = [0, 0, 0, 0]
            for name in item["parts"]:
                part = parts_by_name.get(name)
                if part is None or (name != "confirm" and measured_as not in (part.get("members") or [])):
                    continue
                part_row = (tables.get(name) or {}).get(measured_as)
                if part_row is not None and part_row.get("status") == "ok":
                    for index, key in enumerate(("shots_x", "failures_x", "shots_z", "failures_z")):
                        sums[index] += int(part_row.get(key) or 0)
            got = [int(row.get(key) or 0) for key in ("shots_x", "failures_x", "shots_z", "failures_z")]
            _require(got == sums, f"final ranking: {pid} pools {got}, its ok planned parts sum to {sums}")
            cap = int(item["total"])
            _require(
                int(row.get("cumulative_cap") or 0) == cap,
                f"final ranking: {pid} cumulative_cap={row.get('cumulative_cap')!r}, planned {cap}",
            )
            if row.get("status") != "ok":
                continue
            _require(row["shots_x"] == row["shots_z"] <= cap, f"final ranking: {pid} shots exceed its limit {cap}")
            combined = combine_basis_ler_results(
                {
                    "X": {"logical_error_rate": sums[1] / sums[0], "failures": sums[1], "shots": sums[0]},
                    "Z": {"logical_error_rate": sums[3] / sums[2], "failures": sums[3], "shots": sums[2]},
                },
                rounds=int(row.get("rounds") or rounds),
                k=int(row.get("logicals") or 0),
            )
            _require(
                math.isclose(float(row["per_logical_per_round"]), float(combined["per_logical_per_round"]), rel_tol=1e-12),
                f"final ranking: {pid} pooled pLER {row['per_logical_per_round']!r} is not the pLER of its sums",
            )
        ranked = sorted(
            (row for row in pooled_rows if row.get("final_rank") is not None), key=lambda row: int(row["final_rank"])
        )
        keys = [
            (float(row["per_logical_per_round"]), float((row.get("per_logical_per_round_ci") or [0, math.inf])[1]))
            for row in ranked
        ]
        _require(keys == sorted(keys), "final ranking: final_rank is not the pooled-pLER order")
        _require(
            [int(row["final_rank"]) for row in ranked] == list(range(1, len(ranked) + 1)) and bool(ranked),
            "final ranking: no ranked finalist",
        )
        _require(
            all(not row.get("is_baseline") for row in ranked), "final ranking: a reference row carries a final rank"
        )
        champion = ranking.get("champion") or {}
        _require(
            champion.get("program_id") == ranked[0].get("program_id")
            and (report.get("champion") or {}).get("program_id") == ranked[0].get("program_id"),
            "final ranking: the champion is not the rank-1 finalist",
        )
        if reference:
            references = [row for row in pooled_rows if row.get("is_baseline")]
            _require(len(references) == 1, f"final ranking: expected one reference row, found {len(references)}")
        expected_reported = paper_final.reported_top(block, pooled_rows)
        _require(
            ranking.get("reported_top") == expected_reported and report.get("final_reported_top") == expected_reported,
            f"final ranking: reported_top {ranking.get('reported_top')!r} is not the "
            f"{block.get('reported_top')} best ranked codes {expected_reported!r}",
        )
    frontier = report.get("final_frontier") or []
    _require(bool(frontier), "final frontier is empty")
    for index, row in enumerate(frontier):
        prefix = f"final row {index}"
        _require(_same_float(row.get("noise_value"), p_value), f"{prefix}: p mismatch")
        _require(int(row.get("shots_x") or 0) == int(row.get("shots_z") or 0) >= int(final_shots), f"{prefix}: shots mismatch")
        _require(int(row.get("rounds") or 0) == rounds, f"{prefix}: rounds={row.get('rounds')!r}, expected {rounds}")
        _require(str(row.get("decoder") or "") == FORMAL_DECODER, f"{prefix}: decoder mismatch")
        _validate_decoder_blob(row.get("decoder_config"), prefix)


# Defaults ARE the formal Fig.2 protocol (03 section 8 / 02 D6). A job script
# that overrides either one must pass the override here too, so the recorded
# protocol and the accepted protocol can never diverge silently.
DEFAULT_RACING_PILOT = 5000
DEFAULT_RACING_BATCH = 5000
DEFAULT_RACING_CONFIRM = 20000
DEFAULT_RACING_MAX = 50000
# telescoping_structural_v3_6 measures with exactly one decoder.
DEFAULT_DECODERS: tuple[str, ...] = (FORMAL_DECODER,)


def validate(
    run_dir: Path,
    *,
    turns: int,
    p_value: float,
    final_shots: int | None = None,
    model: str,
    provider: str = "deepseek",
    racing_pilot: int = DEFAULT_RACING_PILOT,
    racing_batch: int = DEFAULT_RACING_BATCH,
    racing_confirm: int = DEFAULT_RACING_CONFIRM,
    racing_max: int = DEFAULT_RACING_MAX,
    decoders: Sequence[str] = DEFAULT_DECODERS,
    reasoning_fallback_flag: Path | None = None,
) -> int:
    report = _read_json(run_dir / "campaign_report.json")
    if FINAL_PROTOCOL:
        _validate_reasoning_rule(run_dir, model=model, provider=provider, flag_path=reasoning_fallback_flag)
    # v3.6: the budget point selects the shots per basis. The caller may
    # state final_shots (the launcher's log line) but cannot AUTHORIZE another
    # value: a different final shot count is a different protocol.
    n_budget = _report_n_budget(report)
    policy = budget_policy(n_budget)
    if final_shots is None:
        final_shots = int(policy["final_shots"])
    _require(
        int(final_shots) == int(policy["final_shots"]),
        f"caller declared final_shots={final_shots!r}, but {PROTOCOL_PROFILE} freezes "
        f"final_shots={policy['final_shots']} at n_budget={n_budget}; a different "
        "shot count is a different protocol and must not be validated as this one",
    )
    _validate_protocol(
        report,
        turns=turns,
        p_value=p_value,
        final_shots=int(final_shots),
        model=model,
        provider=provider,
        racing_pilot=racing_pilot,
        racing_batch=racing_batch,
        racing_confirm=racing_confirm,
        racing_max=racing_max,
        decoders=decoders,
        n_budget=n_budget,
    )
    _validate_versions(run_dir)
    if FORMAL_ROUNDS_POLICY == "fixed_Tc":
        _require(_validate_circuit_rounds(run_dir) > 0, "fixed_Tc: no candidate file records its circuit rounds")
    _validate_turns(
        run_dir, turns=turns, model=model, provider=provider, n_budget=n_budget,
        lowered_from=_original_max_turns(report.get("protocol") or {}),
    )
    _validate_final(
        report,
        p_value=p_value,
        final_shots=int(final_shots),
        turns=turns,
        run_dir=run_dir,
        n_budget=n_budget,
    )
    rows = len(report.get("final_frontier") or [])
    print(
        f"validated {PROTOCOL_PROFILE} campaign: n_budget={n_budget}, turns={turns}, "
        f"p={p_value}, search_shots_per_basis={policy['search_shots']}, "
        f"final_shots_per_basis={final_shots}, final_rows={rows}, "
        f"shot_shards={FORMAL_SHOT_SHARDS}, "
        f"final_shot_shards={FORMAL_FINAL_SHOT_SHARDS}, "
        f"decode_shot_budget_s={FORMAL_JOB_POLICY['decode_shot_budget_s']}, "
        f"timeout_retries={FORMAL_JOB_POLICY['timeout_retries']}, "
        f"transient_retries={FORMAL_JOB_POLICY['transient_retries']}, "
        f"noise={FORMAL_NOISE_PROFILE}@p={FORMAL_P}, "
        f"sec={FORMAL_SEC_POLICY}(K={FORMAL_SEC_CANDIDATES},I={FORMAL_SEC_CERT_ITERS},"
        f"seeds={FORMAL_SEC_CERT_SEEDS},refine={FORMAL_SEC_REFINE_TOP}x{FORMAL_SEC_REFINE_ITERS}), "
        f"provider={provider}, model={model}, "
        f"decoder={FORMAL_DECODER}({TELESCOPING_PROTOCOL_FIELDS['preset']}, "
        f"package {TELESCOPING_PROTOCOL_FIELDS['package']}, "
        f"s3_system {TELESCOPING_PROTOCOL_FIELDS['s3_system']}, "
        f"NC={TELESCOPING_PROTOCOL_FIELDS['nc_policy']}, "
        f"n_procs {TELESCOPING_PROTOCOL_FIELDS['n_procs']}), racing={RACING_POLICY_ID}"
    )
    return rows


def configure_profile(profile: str, *, exact_check_weight: int | None = None) -> None:
    """Switch the frozen structural expectations to the named profile.

    v3_7 is the module default (the formal Fig.2 protocol). v3_8 swaps the
    expected retention config, the profile id and the admissible structural
    slot reasons (the distinct-family second endpoint) and turns on the
    per-turn variant-generation / endpoint-rule checks. ``exact_check_weight``
    (fig3_final only): the Fig. 4 c fixed-weight CNOT search with that weight."""

    name = str(profile).strip()
    if exact_check_weight is not None and name != "fig3_final":
        raise ValueError(f"--exact-check-weight validates the fixed-weight CNOT searches: profile fig3_final (got {name})")
    if exact_check_weight is not None and not 3 <= int(exact_check_weight) <= FINAL_FAST_LOOP_MAX_CHECK_WEIGHT:
        raise ValueError(f"--exact-check-weight must be 3..{FINAL_FAST_LOOP_MAX_CHECK_WEIGHT} (got {exact_check_weight})")
    globals()["FINAL_EXACT_CHECK_WEIGHT"] = None if exact_check_weight is None else int(exact_check_weight)
    if name == "fig3_final":
        # 2026-11-06: the final Fig.3 protocol -- the CNOT task under the Fig.3 structural profile
        # (3S + 4R with the lightest-cell slot).
        expected = fig3_cnot_structural_config(StructuralSearchConfig())
        task, rounds_policy, final = "cnot_dense", "fixed_Tc", True
    elif name == "fig2_final":
        expected = structural_search_for_profile("v3_8")
        task, rounds_policy, final = "memory", "fixed_Tc", True
    else:
        expected = structural_search_for_profile(name)
        task, rounds_policy, final = "memory", ROUNDS_POLICY_ID, False
    globals()["PROTOCOL_PROFILE"] = str(expected.profile_id)
    globals()["STRUCTURAL_EXPECTED"] = expected.to_dict()
    globals()["FORMAL_TASK"] = task
    globals()["FORMAL_ROUNDS_POLICY"] = rounds_policy
    globals()["FINAL_PROTOCOL"] = final
    reasons = set(FORMAL_STRUCTURAL_SLOT_REASONS_V37)
    if expected.profile_id == PROTOCOL_PROFILE_V38:
        reasons.add("high_d_search_endpoint_other_family")
    if expected.lightest_endpoint_slot:
        reasons.add("lightest_endpoint")
    globals()["FORMAL_STRUCTURAL_SLOT_REASONS"] = frozenset(reasons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--turns", type=int, required=True)
    parser.add_argument("--p", type=float, required=True)
    # v3.6: optional -- the budget policy of the report's qubit_budget is the
    # truth; a stated value is checked against it, never used instead of it.
    parser.add_argument("--final-shots", type=int, default=None)
    # The campaign pins ONE provider in its protocol snapshot and audits every
    # recorded call against it. Defaulting to deepseek keeps existing invocations
    # (and the DianGuang scripts) unchanged, while a non-DeepSeek arm can now be
    # validated instead of failing seven assertions it was never meant to fail.
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", required=True)
    # Protocol variants. The job script knows what it actually launched with;
    # pass the same values through so an intentional override validates and an
    # accidental one still fails.
    parser.add_argument("--racing-pilot", type=int, default=DEFAULT_RACING_PILOT)
    parser.add_argument("--racing-batch", type=int, default=DEFAULT_RACING_BATCH)
    parser.add_argument("--racing-confirm", type=int, default=DEFAULT_RACING_CONFIRM)
    parser.add_argument("--racing-max", type=int, default=DEFAULT_RACING_MAX)
    parser.add_argument(
        "--decoders", nargs="+", default=list(DEFAULT_DECODERS)
    )
    # 2026-10-17: --profile v3_8 validates a telescoping_structural_v3_8 run
    # (the declared variant: variant generations + endpoint family rules).
    # 2026-11-06: fig2_final / fig3_final validate the final protocols; the launcher of a PRESET run
    # passes its arm's reasoning-saturation flag (present or not) as --reasoning-fallback-flag.
    parser.add_argument("--profile", choices=("v3_7", "v3_8", "fig2_final", "fig3_final"), default="v3_7")
    parser.add_argument("--reasoning-fallback-flag", type=Path, default=None)
    # 2026-11-10 (paper revision, Fig. 4 c): fig3_final only -- the run is the fixed-weight search with this
    # exact maximum check weight (and its fixed-weight final stage).
    parser.add_argument("--exact-check-weight", type=int, default=None)
    args = parser.parse_args()
    configure_profile(args.profile, exact_check_weight=args.exact_check_weight)
    validate(
        args.run_dir,
        turns=args.turns,
        p_value=args.p,
        final_shots=args.final_shots,
        model=args.model,
        provider=args.provider,
        racing_pilot=args.racing_pilot,
        racing_batch=args.racing_batch,
        racing_confirm=args.racing_confirm,
        racing_max=args.racing_max,
        decoders=args.decoders,
        reasoning_fallback_flag=args.reasoning_fallback_flag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
