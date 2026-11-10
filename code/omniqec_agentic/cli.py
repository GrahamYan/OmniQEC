"""Command-line entrypoint for omniqec_agentic."""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any
from pathlib import Path

from .config import (
    ABLATION_FEEDBACK_MODE,
    ABLATION_IDS,
    BASELINE_GENERATIONS_PER_TURN,
    BASELINE_INIT_RANDOM_TURNS,
    BASELINE_TURNS,
    BASELINE_PROPOSALS_PER_GENERATION,
    BASELINE_PROVIDER,
    SEARCH_METHODS,
    DEFAULT_SEC_POLICY,
    SEC_POLICY_IDS,
    BposdDecoderConfig,
    CodeFamily,
    DecoderBuildConfig,
    DecoderName,
    DeformationMode,
    ExactDistanceConfig,
    HardwareTarget,
    LpShapePin,
    SecPolicyConfig,
    VerificationConfig,
    budget_policy,
    formal_racing,
    canonical_code_family,
    default_run_config,
    load_dotenv,
    sec_policy_for_id,
)
from .funsearch import run_funsearch
from .llm.factory import SUPPORTED_LLM_PROVIDERS, resolve_model_name
from .verification.noise import (
    HELIOS_PRIMARY_SCALE,
    UNIFORM_PROFILES,
    NoiseProfile,
    NoiseSweepConfig,
    RzzOperandOrientation,
    SI1000_REFERENCE_P,
)

PROVIDER_CHOICES = [*SUPPORTED_LLM_PROVIDERS, "dry_run"]


def _add_noise_arguments(parser: argparse.ArgumentParser, *, multi: bool) -> None:
    parser.add_argument(
        "--noise-profile",
        choices=[profile.value for profile in NoiseProfile],
        # v3.5: the paper's SI C uniform model (no idle noise) is the default;
        # uniform_legacy_v1 (idle noise, v3.4) stays selectable for history.
        default=NoiseProfile.UNIFORM_PAPER.value,
    )
    if multi:
        parser.add_argument("--si-p-values", type=float, nargs="+")
        parser.add_argument("--helios-scales", type=float, nargs="+")
    else:
        parser.add_argument("--si-p", type=float)
        parser.add_argument("--helios-scale", type=float)
    parser.add_argument(
        "--helios-orientation",
        choices=[item.value for item in RzzOperandOrientation],
        default=RzzOperandOrientation.CONTROL_FIRST.value,
    )


def _add_sec_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sec-policy",
        choices=list(SEC_POLICY_IDS),
        default=DEFAULT_SEC_POLICY.sec_policy_id,
        help=(
            "Syndrome-extraction-circuit policy. The formal telescoping_structural_v3_7 "
            "protocol freezes twobasis_nonint_v1 (both bases every round, no "
            "interleaving, K=16 candidate orders per basis certified with 400 ISD "
            "iterations); lrc_ranked_v1 is the historical left-right circuit, a "
            "different protocol whose numbers never pool with twobasis ones and "
            "which is refused under --strict-turns."
        ),
    )


def _sec_policy_from_args(args: argparse.Namespace) -> SecPolicyConfig:
    return sec_policy_for_id(
        str(getattr(args, "sec_policy", None) or DEFAULT_SEC_POLICY.sec_policy_id)
    )


def _add_deformation_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deformation-mode",
        choices=[mode.value for mode in DeformationMode],
        default=DeformationMode.NONE.value,
        help="Run-level deterministic local-Clifford deformation.",
    )


def _add_lp_shape_pin_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lp-shape-pin-a",
        metavar="RxC",
        help="Pin the LP base matrix A shape (e.g. 1x2); omit to let the LLM choose.",
    )
    parser.add_argument(
        "--lp-shape-pin-b",
        metavar="RxC",
        help="Pin the LP base matrix B shape (e.g. 1x2); omit to let the LLM choose.",
    )


def _lp_shape_pin_from_args(args: argparse.Namespace) -> LpShapePin:
    return LpShapePin(
        a=LpShapePin.parse_shape(getattr(args, "lp_shape_pin_a", None)),
        b=LpShapePin.parse_shape(getattr(args, "lp_shape_pin_b", None)),
    )


def _noise_from_args(
    args: argparse.Namespace,
    *,
    multi: bool,
    legacy_attr: str | None = None,
) -> NoiseSweepConfig:
    profile = NoiseProfile(str(args.noise_profile))
    legacy = getattr(args, legacy_attr or ("p_values" if multi else "p"), None)
    # Single-point commands (--p) carry a scalar; the multi ones a list.
    if legacy is not None and not isinstance(legacy, (list, tuple)):
        legacy = [legacy]
    si_values = getattr(args, "si_p_values", None) if multi else (
        [args.si_p] if getattr(args, "si_p", None) is not None else None
    )
    helios_values = getattr(args, "helios_scales", None) if multi else (
        [args.helios_scale] if getattr(args, "helios_scale", None) is not None else None
    )
    if profile in UNIFORM_PROFILES:
        if si_values or helios_values:
            raise ValueError("SI/Helios values require the matching --noise-profile")
        values = tuple(float(value) for value in (legacy or [0.005]))
        if profile == NoiseProfile.UNIFORM_LEGACY:
            return NoiseSweepConfig.uniform_legacy(values)
        return NoiseSweepConfig.uniform(values)
    if legacy:
        raise ValueError(
            "--p/--p-values are uniform-profile aliases (uniform_paper_v1 / "
            "uniform_legacy_v1) and cannot be mixed with SI/Helios"
        )
    if profile in {NoiseProfile.SI1000, NoiseProfile.SI1000_MEASUREMENT_X2}:
        values = tuple(float(value) for value in (si_values or (SI1000_REFERENCE_P,)))
        if profile == NoiseProfile.SI1000_MEASUREMENT_X2:
            return NoiseSweepConfig.si1000_measurement_enhanced(values)
        return NoiseSweepConfig.si1000(
            values,
            source_kind="user_supplied_model_parameter" if si_values else "published_reference_point",
        )
    values = tuple(float(value) for value in (helios_values or (HELIOS_PRIMARY_SCALE,)))
    source_kind = (
        "published_calibration"
        if values == (HELIOS_PRIMARY_SCALE,)
        else "synthetic_global_stress_test"
    )
    if profile == NoiseProfile.HELIOS_PAULI_SHARPENED:
        return NoiseSweepConfig.helios_bias_enhanced(
            values,
            orientation=RzzOperandOrientation(str(args.helios_orientation)),
        )
    return NoiseSweepConfig.helios(
        values,
        symmetrized=profile == NoiseProfile.HELIOS_SYMMETRIZED,
        orientation=RzzOperandOrientation(str(args.helios_orientation)),
        source_kind=source_kind,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic QEC discovery utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    funsearch = subparsers.add_parser(
        "run-funsearch",
        help="Run the ver1-aligned cumulative archive/history FunSearch loop.",
    )
    funsearch.add_argument("--run-id", default="funsearch")
    funsearch.add_argument("--n-budget", type=int, required=True)
    funsearch.add_argument("--hardware", choices=[item.value for item in HardwareTarget], required=True)
    funsearch.add_argument("--out", required=True)
    funsearch.add_argument("--generations", type=int, default=1)
    funsearch.add_argument("--provider", choices=PROVIDER_CHOICES, default="deepseek")
    funsearch.add_argument("--proposal-count", type=int, default=8)
    funsearch.add_argument("--promote-top", type=int)
    funsearch.add_argument("--proxy-iters", type=int, default=120)
    funsearch.add_argument("--proxy-n-min", type=int)
    funsearch.add_argument("--proxy-n-max", type=int)
    funsearch.add_argument("--no-seeds", action="store_true")
    funsearch.add_argument("--save-every", type=int, default=10)
    funsearch.add_argument(
        "--rng-seeds-per-program",
        type=int,
        default=20,
        help="RNG seeds evaluated per proposed program; every distinct code competes for its own niche (deterministic programs dedup almost for free).",
    )
    funsearch.add_argument(
        "--families",
        nargs="+",
        choices=[item.value for item in CodeFamily],
        help="Construction families to search (default: all four canonical grammars).",
    )
    funsearch.add_argument("--llm-temperature", type=float, help="Provider sampling temperature (default 0.45).")
    _add_deformation_argument(funsearch)
    _add_lp_shape_pin_arguments(funsearch)
    funsearch.add_argument(
        "--feedback", choices=["both", "fast", "slow", "none"], default="both",
        help="消融开关:both=fast+slow 反馈(默认);fast=只给 fast ranking;none=不给反馈(每代冷启动)。fast-only 消融用 fast/none。",
    )
    funsearch.set_defaults(func=run_funsearch_command)

    cycle = subparsers.add_parser(
        "run-agentic-cycle",
        help="Run FunSearch, slow verification, and slow-aware prompt memory in one command.",
    )
    cycle.add_argument("--run-id", default="agentic_cycle")
    cycle.add_argument(
        "--n-budget",
        type=int,
        required=True,
        help="TOTAL qubit budget: N_data + one measurement ancilla per check row (BB72 costs 144 total).",
    )
    cycle.add_argument("--hardware", choices=[item.value for item in HardwareTarget], required=True)
    cycle.add_argument("--out", required=True)
    cycle.add_argument("--generations", type=int, default=1)
    cycle.add_argument(
        "--promote-every",
        type=int,
        default=0,
        help="Run slow verification every N generations and feed slow feedback into the next prompt; 0 keeps the old final-only slow stage.",
    )
    cycle.add_argument("--provider", choices=PROVIDER_CHOICES, default="deepseek")
    cycle.add_argument("--proposal-count", type=int, default=2)
    cycle.add_argument(
        "--promote-top",
        type=int,
        help="Fast archive elites to place in the promotion queue. Defaults to at least --slow-limit.",
    )
    cycle.add_argument("--proxy-iters", type=int, default=120)
    cycle.add_argument("--proxy-n-min", type=int)
    cycle.add_argument("--proxy-n-max", type=int)
    cycle.add_argument("--no-seeds", action="store_true")
    cycle.add_argument(
        "--feedback", choices=["both", "fast", "slow", "none"], default="both",
        help="消融开关:both=fast+slow 反馈(默认);fast/slow=prompt 只给一种;none=不给反馈(每代冷启动)。",
    )
    cycle.add_argument(
        "--seed-sources-json",
        help="JSON file with a list of construct(rng) source strings to seed the gen-0 archive (cross-N warm-start). Overrides built-in example seeds.",
    )
    cycle.add_argument("--save-every", type=int, default=10)
    cycle.add_argument(
        "--slow-all-archive",
        action="store_true",
        help="Slow-verify every valid archive cell elite instead of only the fast top-N promotion queue.",
    )
    cycle.add_argument(
        "--slow-limit",
        type=int,
        help="Cap slow candidates. Defaults to 2 for top-N mode and unlimited for --slow-all-archive.",
    )
    cycle.add_argument("--slow-p-values", type=float, nargs="+")
    _add_noise_arguments(cycle, multi=True)
    _add_deformation_argument(cycle)
    cycle.add_argument(
        "--slow-rounds",
        type=int,
        help="Fixed syndrome-extraction rounds for every slow candidate; omit for the legacy d_hint-based selection.",
    )
    cycle.add_argument("--slow-shots", type=int, default=1000)
    cycle.add_argument("--slow-batch-size", type=int, default=200)
    cycle.add_argument("--slow-max-errors", type=int, default=1000)
    cycle.add_argument(
        "--slow-timeout-s",
        type=float,
        help="Timeout per slow candidate subprocess in seconds; 0 disables the application-level timeout.",
    )
    cycle.add_argument(
        "--slow-workers",
        type=int,
        default=1,
        help="Global slow job pool size; 0 = auto (cpu_count - 1). Jobs are (candidate, p, decoder, basis, shard) subprocesses with dynamic scheduling; results are deterministic regardless of worker count.",
    )
    cycle.add_argument(
        "--slow-shot-shards",
        type=int,
        default=1,
        help="Split each decode job's shots into K parallel shards (statistically equivalent; the shard count is part of the measurement protocol — 1 reproduces the serial path exactly).",
    )
    cycle.add_argument("--exact-circuit-distance", action="store_true", help="Enable Stim exact circuit-distance search inside slow verification.")
    cycle.add_argument(
        "--dem-isd-distance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DEM-ISD circuit-distance upper-bound evidence inside slow verification (default on; the only automatic degradation signal in single-p runs). Disable with --no-dem-isd-distance.",
    )
    cycle.add_argument("--dem-isd-max-mechanisms", type=int, default=8000)
    cycle.add_argument("--dem-isd-iters", type=int, default=120)
    cycle.add_argument(
        "--decoders",
        choices=[item.value for item in DecoderName],
        nargs="+",
        default=None,
        help="Slow-verification decoders (default: the config's "
        "telescoping_cpu-only profile). Anything else is a protocol variant "
        "and is refused by the strict telescoping_structural_v3_7 validator.",
    )
    cycle.add_argument(
        "--bases",
        choices=["X", "Z"],
        nargs="+",
        default=None,
        help="Memory bases to decode (default: keep config's X Z = combined LER). "
        "Pass 'Z' for Z-memory only (focus on correcting X errors), ~2x faster; "
        "the resident baseline is measured with the same bases. Confirm winners on both.",
    )
    cycle.add_argument(
        "--rng-seeds-per-program",
        type=int,
        default=20,
        help="RNG seeds evaluated per proposed program; every distinct code competes for its own niche (deterministic programs dedup almost for free).",
    )
    cycle.add_argument(
        "--d-cap",
        type=int,
        help="Distance cap for Q_eff ranking (off by default since 2026-09-10: rounds follow max(12, d_search), so no verifiable-regime cap applies; pass a value to enable, 0 disables).",
    )
    cycle.add_argument(
        "--families",
        nargs="+",
        choices=[item.value for item in CodeFamily],
        help="Construction families to search (default: all four canonical grammars). Single-family campaigns keep arms uncontaminated; off-family proposals are rejected unevaluated.",
    )
    _add_lp_shape_pin_arguments(cycle)
    cycle.add_argument(
        "--llm-temperature",
        type=float,
        help="Provider sampling temperature (default 0.45). Vary across replicas for exploration diversity.",
    )
    cycle.add_argument("--no-coloring", action="store_true")
    cycle.set_defaults(func=run_agentic_cycle_command)

    slow_deps = subparsers.add_parser(
        "check-slow-deps",
        help="Report optional circuit-level decoder dependencies and default decoder build config.",
    )
    slow_deps.set_defaults(func=check_slow_deps_command)

    dem_smoke = subparsers.add_parser(
        "build-dem-smoke",
        help="Build a ver1-style seed CSS memory circuit/DEM smoke artifact.",
    )
    dem_smoke.add_argument("--seed-index", type=int, default=1, help="1-based built-in seed index.")
    dem_smoke.add_argument("--p", type=float)
    _add_noise_arguments(dem_smoke, multi=False)
    dem_smoke.add_argument("--rounds", type=int)
    dem_smoke.add_argument("--proxy-iters", type=int, default=120)
    _add_deformation_argument(dem_smoke)
    _add_sec_policy_argument(dem_smoke)
    dem_smoke.add_argument("--no-coloring", action="store_true")
    dem_smoke.set_defaults(func=build_dem_smoke_command)

    slow_seed = subparsers.add_parser(
        "slow-verify-seed",
        help="Run circuit-level slow verification on one built-in seed candidate.",
    )
    slow_seed.add_argument("--seed-index", type=int, default=1, help="1-based built-in seed index.")
    slow_seed.add_argument("--p-values", type=float, nargs="+")
    _add_noise_arguments(slow_seed, multi=True)
    slow_seed.add_argument("--decoders", choices=[item.value for item in DecoderName], nargs="+", default=["bposd"], help="Historical bposd default for this dev command; the formal telescoping_structural_v3_7 profile is telescoping_cpu-only (pass it explicitly); other sets are protocol variants.")
    slow_seed.add_argument("--shots", type=int, default=64)
    slow_seed.add_argument("--rounds", type=int)
    slow_seed.add_argument("--max-errors", type=int, default=80)
    slow_seed.add_argument("--batch-size", type=int, default=64)
    slow_seed.add_argument("--timeout-s", type=float, default=600.0, help="Slow verifier timeout in seconds; 0 disables it.")
    slow_seed.add_argument("--exact-circuit-distance", action="store_true", help="Enable Stim exact circuit-distance search.")
    slow_seed.add_argument("--dem-isd-distance", action="store_true", help="Enable DEM-ISD circuit-distance upper-bound evidence.")
    slow_seed.add_argument("--dem-isd-max-mechanisms", type=int, default=8000)
    slow_seed.add_argument("--dem-isd-iters", type=int, default=120)
    slow_seed.add_argument("--proxy-iters", type=int, default=120)
    slow_seed.add_argument("--no-coloring", action="store_true")
    _add_sec_policy_argument(slow_seed)
    slow_seed.add_argument("--out")
    slow_seed.set_defaults(func=slow_verify_seed_command)

    slow_queue = subparsers.add_parser(
        "slow-verify-queue",
        help="Run circuit-level slow verification on a promotion_queue.json/jsonl file.",
    )
    slow_queue.add_argument("--promotion-queue", required=True)
    slow_queue.add_argument("--hardware", choices=[item.value for item in HardwareTarget], required=True)
    slow_queue.add_argument("--out", required=True)
    slow_queue.add_argument("--limit", type=int)
    slow_queue.add_argument("--p-values", type=float, nargs="+")
    _add_noise_arguments(slow_queue, multi=True)
    slow_queue.add_argument("--decoders", choices=[item.value for item in DecoderName], nargs="+", default=["bposd"], help="Historical bposd default for this dev command; the formal telescoping_structural_v3_7 profile is telescoping_cpu-only (pass it explicitly); other sets are protocol variants.")
    slow_queue.add_argument("--shots", type=int, default=64)
    slow_queue.add_argument("--rounds", type=int)
    slow_queue.add_argument("--max-errors", type=int, default=80)
    slow_queue.add_argument("--batch-size", type=int, default=64)
    slow_queue.add_argument("--timeout-s", type=float, default=600.0, help="Per-candidate timeout in seconds; 0 disables it.")
    slow_queue.add_argument("--exact-circuit-distance", action="store_true", help="Enable Stim exact circuit-distance search.")
    slow_queue.add_argument("--dem-isd-distance", action="store_true", help="Enable DEM-ISD circuit-distance upper-bound evidence.")
    slow_queue.add_argument("--dem-isd-max-mechanisms", type=int, default=8000)
    slow_queue.add_argument("--dem-isd-iters", type=int, default=120)
    slow_queue.add_argument("--proxy-iters", type=int, default=120)
    slow_queue.add_argument("--no-coloring", action="store_true")
    slow_queue.add_argument("--workers", type=int, default=1, help="Global slow job pool size; 0 = auto (cpu_count - 1).")
    slow_queue.add_argument("--shot-shards", type=int, default=1, help="Parallel shot shards per decode job (protocol parameter; 1 = serial-exact).")
    slow_queue.add_argument("--osd-order", type=int, default=BposdDecoderConfig().osd_order, help="BPOSD OSD order (default: the frozen profile value, 10). Any other value is a protocol variant and is refused by the strict validator.")
    slow_queue.add_argument(
        "--with-bb72-baseline",
        action="store_true",
        default=False,
        help="Inject the BB72 resident baseline into the queue (explicit opt-in; was unconditional before 2026-08-15).",
    )
    _add_deformation_argument(slow_queue)
    _add_sec_policy_argument(slow_queue)
    slow_queue.set_defaults(func=slow_verify_queue_command)

    slow_context = subparsers.add_parser(
        "make-slow-aware-context",
        help="Merge slow feedback into an existing prompt_context.json artifact.",
    )
    slow_context.add_argument("--prompt-context", required=True)
    slow_context.add_argument("--slow-feedback", required=True)
    slow_context.add_argument("--out", required=True)
    slow_context.add_argument("--parent-limit", type=int, default=12)
    slow_context.set_defaults(func=make_slow_aware_context_command)

    stim_seed = subparsers.add_parser(
        "export-stim-seed",
        help="Export concrete X/Z memory Stim circuits and DEMs for one built-in seed.",
    )
    stim_seed.add_argument("--seed-index", type=int, default=1, help="1-based built-in seed index.")
    stim_seed.add_argument("--out", required=True)
    stim_seed.add_argument("--p", type=float)
    _add_noise_arguments(stim_seed, multi=False)
    stim_seed.add_argument("--rounds", type=int)
    stim_seed.add_argument("--proxy-iters", type=int, default=120)
    _add_sec_policy_argument(stim_seed)
    stim_seed.set_defaults(func=export_stim_seed_command)

    stim_queue = subparsers.add_parser(
        "export-stim-queue",
        help="Export concrete X/Z memory Stim circuits and DEMs for queue candidates.",
    )
    stim_queue.add_argument("--queue", required=True)
    stim_queue.add_argument("--out", required=True)
    stim_queue.add_argument("--limit", type=int)
    stim_queue.add_argument("--p", type=float)
    _add_noise_arguments(stim_queue, multi=False)
    stim_queue.add_argument("--rounds", type=int)
    _add_sec_policy_argument(stim_queue)
    stim_queue.set_defaults(func=export_stim_queue_command)

    analyze = subparsers.add_parser(
        "analyze-runs",
        help="Print a ver1-style leaderboard for fast archive and slow verification artifacts.",
    )
    analyze.add_argument("runs", nargs="+")
    analyze.add_argument("--bb-q", type=float, default=432.0, help="Reference Q = k*d^2 (default BB72: 12*6^2).")
    analyze.add_argument("--json", action="store_true", help="Emit JSON rows instead of the text leaderboard.")
    analyze.set_defaults(func=analyze_runs_command)

    calibrate = subparsers.add_parser(
        "calibrate-slope-offset",
        help=(
            "Measure the BB [[72,12,6]] LER curve under this package's slow "
            "protocol and report the slope offset that maps the fitted slope "
            "back to d_circ=6. Pass the suggested offset via VerificationConfig."
        ),
    )
    calibrate.add_argument("--p-values", type=float, nargs="+", default=[0.003, 0.0045, 0.007])
    calibrate.add_argument("--decoders", choices=[item.value for item in DecoderName], nargs="+", default=["lsd"])
    calibrate.add_argument("--shots", type=int, default=4000)
    calibrate.add_argument("--max-errors", type=int, default=200)
    calibrate.add_argument("--batch-size", type=int, default=500)
    calibrate.add_argument("--timeout-s", type=float, default=1800.0)
    calibrate.add_argument("--proxy-iters", type=int, default=120)
    calibrate.add_argument("--d-true", type=int, default=6, help="Known circuit distance of the reference code.")
    calibrate.add_argument("--out")
    calibrate.set_defaults(func=calibrate_slope_offset_command)

    qualify = subparsers.add_parser(
        "qualify-decoders",
        help=(
            "Decoder qualification harness (03 §10): run the versioned "
            "qualification set through the evaluation pipeline per profile, "
            "score candidates against references (pairwise agreement >= 0.9 "
            "on reference-separated pairs, >= 2 p-levels/profile), measure "
            "throughput, and run the beam determinism + num_results=32 checks."
        ),
    )
    qualify.add_argument("--set", dest="set_path",
                         help="Qualification set JSON (default: docs/qualification/qualification_set_v1.json)")
    qualify.add_argument("--out", required=True)
    # v3.6 (docs/06 §1.3): the reference decoder is the frozen telescoping
    # decoder (user decision 2026-09-03); the
    # default run is the acceptance harness (uniform p=0.005, BB 1000 shots /
    # basis, Mitten rows 100) through the same worker path as a campaign.
    qualify.add_argument("--candidates", nargs="+", default=[DecoderName.TELESCOPING.value],
                         choices=[item.value for item in DecoderName])
    qualify.add_argument("--references", nargs="+", default=[DecoderName.TELESCOPING.value],
                         choices=[item.value for item in DecoderName])
    qualify.add_argument("--profiles", nargs="+", default=["uniform"],
                         choices=["uniform", "si1000", "helios"])
    qualify.add_argument("--shots", type=int, default=1000)
    qualify.add_argument("--seed", type=int, default=0)
    qualify.add_argument("--workers", type=int, default=0)
    qualify.add_argument("--limit", type=int)
    qualify.add_argument("--skip-determinism", action="store_true")
    qualify.add_argument("--skip-beam32", action="store_true")
    qualify.set_defaults(func=qualify_decoders_command)

    campaign_cmd = subparsers.add_parser(
        "run-campaign",
        help=(
            "Run one campaign (budget point) through the LangGraph workflow "
            "(03 §2): intake -> turn loop -> final_slow -> report, with "
            "SqliteSaver checkpointing and --resume."
        ),
    )
    campaign_cmd.add_argument("--budget", type=int, required=True,
                              help="TOTAL qubit budget point (data + ancilla).")
    campaign_cmd.add_argument("--out", required=True)
    campaign_cmd.add_argument("--families", nargs="+",
                              choices=[item.value for item in CodeFamily])
    campaign_cmd.add_argument(
        "--weight-limit", type=int, default=None,
        help="per-check stabilizer weight cap; default None = no cap (telescoping_structural_v3_7)",
    )
    campaign_cmd.add_argument(
        "--turns", type=int, default=None,
        help=(
            "Search turns (iterations). Default 3; a search baseline (--search-method random / bo) defaults to "
            f"{BASELINE_TURNS}, the 30 iterations of every discovery run (paper appendix D4/D5)."
        ),
    )
    campaign_cmd.add_argument("--proposal-count", type=int, default=8)
    campaign_cmd.add_argument("--generations-per-turn", type=int, default=1)
    campaign_cmd.add_argument(
        "--rng-seeds-per-program", type=int, default=40,
        help="RNG seeds evaluated per proposed program (formal protocol since 2026-09-10: 40; was 20).",
    )
    campaign_cmd.add_argument("--provider", default="dry_run",
                              help="LLM provider for generation; dry_run degrades to FallbackPolicy pipeline validation.")
    campaign_cmd.add_argument("--slow-workers", type=int, default=0)
    campaign_cmd.add_argument(
        "--slow-shots",
        type=int,
        default=None,
        help=(
            "Search-stage shots per basis. Default: the telescoping_structural_v3_7 "
            "budget policy of --budget (20,000 below 300 total qubits, 5,000 at "
            ">= 300); any other value is a protocol variant and is refused under "
            "--strict-turns."
        ),
    )
    campaign_cmd.add_argument(
        "--shot-shards",
        type=int,
        default=16,
        help=(
            "Split each candidate/basis shot stream into this many parallel "
            "decode jobs in the per-turn slow (search) stage. The formal "
            "telescoping_structural_v3_7 protocol freezes this at 16; the final "
            "confirmation stage has its own count (--final-shot-shards)."
        ),
    )
    campaign_cmd.add_argument(
        "--final-shot-shards",
        type=int,
        default=128,
        help=(
            "Split each finalist/basis shot stream of the fixed-T final "
            "confirmation into this many parallel decode jobs (200k shots per "
            "basis at full budget need 128 shards to fit the per-shot decode "
            "watchdog). The formal telescoping_structural_v3_7 protocol freezes this "
            "at 128."
        ),
    )
    campaign_cmd.add_argument(
        "--p",
        type=float,
        default=0.005,
        help=(
            "Single fixed uniform physical-error probability used by both "
            "search and fixed-T=12 final confirmation (Fig.2 protocol: 0.005)."
        ),
    )
    campaign_cmd.add_argument(
        "--noise-profile",
        choices=[NoiseProfile.UNIFORM_PAPER.value, NoiseProfile.UNIFORM_LEGACY.value],
        default=NoiseProfile.UNIFORM_PAPER.value,
        help=(
            "Uniform circuit-level noise model that --p parameterizes. The "
            "formal telescoping_structural_v3_7 protocol freezes this at "
            "uniform_paper_v1 (OmniQEC paper SI C: depolarizing after every "
            "reset and CX, flips before every measurement, NO idle noise); "
            "uniform_legacy_v1 is the retired v3.4 idle-noise model, kept for "
            "history and refused under --strict-turns."
        ),
    )
    _add_sec_policy_argument(campaign_cmd)
    campaign_cmd.add_argument("--racing", action="store_true",
                              help="Enable anchor-p double-threshold racing (03 §8).")
    campaign_cmd.add_argument("--racing-pilot", type=int, default=5000)
    campaign_cmd.add_argument("--racing-batch", type=int, default=5000)
    campaign_cmd.add_argument("--racing-confirm", type=int, default=20000)
    campaign_cmd.add_argument("--racing-max", type=int, default=50000)
    campaign_cmd.add_argument(
        "--no-early-stop", action="store_true",
        help=("PROTOCOL VARIANT: measure every promoted candidate at the full search "
              "shots (fixed-shot, no racing). The formal protocol since 2026-09-11 "
              "is config.formal_racing (pilot10x_v1) and needs no flag."),
    )
    campaign_cmd.add_argument(
        "--decoders",
        nargs="+",
        choices=[d.value for d in DecoderName],
        help=(
            "Decoders to measure each candidate with. The formal protocol "
            "(telescoping_structural_v3_7) is telescoping_cpu-only and is the "
            "default; any other set is a different protocol, produces numbers "
            "that are not comparable with it, and is refused under --strict-turns."
        ),
    )
    campaign_cmd.add_argument(
        "--circuit-timeout-s",
        type=float,
        help=(
            "Legacy compatibility ceiling. When supplied, initializes build, "
            "decode-shard and distance timeouts to the same value; an explicit "
            "stage-specific flag overrides it for that stage. Zero disables "
            "all three application watchdogs."
        ),
    )
    campaign_cmd.add_argument(
        "--build-timeout-s",
        type=float,
        help="Wall-clock watchdog for one DEM-build subprocess; 0 disables it.",
    )
    campaign_cmd.add_argument(
        "--decode-shard-timeout-s",
        type=float,
        help=(
            "FLOOR of the wall-clock watchdog for one decode shot-shard "
            "subprocess: the effective watchdog is "
            "max(this, --decode-shot-budget-s x shots in the shard x "
            "max(1, DEM columns / 60,000)). 0 together with a zero budget "
            "disables it."
        ),
    )
    campaign_cmd.add_argument(
        "--decode-shot-budget-s",
        type=float,
        default=8.0,
        help=(
            "Per-shot wall budget for one decode shard subprocess; the shard "
            "watchdog becomes max(--decode-shard-timeout-s, budget x shard "
            "shots), so a slow code gets proportionally more time instead of "
            "a flat ceiling (v3.6: also scaled by the DEM size, "
            "max(1, columns / 60,000)). 0 disables the budget (flat floor "
            "only). The formal telescoping_structural_v3_7 protocol freezes this "
            "at 8.0."
        ),
    )
    campaign_cmd.add_argument(
        "--final-shard-timeout-cap-s",
        type=float,
        default=None,
        help=(
            "2026-09-24: hard CEILING (seconds) on the FINAL-stage decode-shard "
            "watchdog; the search stage is untouched. A capped shard expires as a "
            "v3.4 watchdog timeout and the candidate ends decode_infeasible. A "
            "protocol runtime key (accepted on --resume, like --slow-workers). "
            "Unset = no ceiling."
        ),
    )
    campaign_cmd.add_argument(
        "--search-shard-timeout-cap-s",
        type=float,
        default=None,
        help=(
            "2026-09-25: the same hard ceiling for the SEARCH-stage slow queue "
            "(per-turn shards). Runtime key; a capped shard -> decode_infeasible "
            "for that candidate. Keep it above the honest budget "
            "(--decode-shot-budget-s x shard shots x DEM scale). Unset = no ceiling."
        ),
    )
    campaign_cmd.add_argument(
        "--promotion-max-check-weight",
        type=int,
        default=None,
        help=(
            "2026-09-29: runtime key. A candidate whose max stabilizer weight "
            "exceeds this is never sent to the slow loop (its slot is backfilled, "
            "the rejection is recorded in slow_queue_composition.json). Evidence: "
            "0 of 413 tier-2/3 codes with plpr < 2e-4 had w >= 9; every w >= 9 "
            "candidate was >= 50x off the point's best at 10-100x the decode cost. "
            "Unset = the fast-loop cap (8, the paper's w_max <= 8; no gate with --max-check-weight 0, "
            "the v3.7 behaviour)."
        ),
    )
    campaign_cmd.add_argument(
        "--max-check-weight",
        type=int,
        default=None,
        help=(
            "Construction-time (fast-loop) check-weight cap. A code whose max row weight of "
            "H_X / H_Z exceeds this is rejected by the structural check "
            "(max_check_weight_exceeded) before distance estimation, never enters the archive, "
            "the proxy frontier or the parents, and the Generator contract says so. Default "
            "(unset): 8, the paper's w_max <= 8 for every main search (config."
            "SEARCH_MAX_CHECK_WEIGHT, 2026-11-10 paper alignment). 0 = no cap (the v3.7 "
            "as-run protocol); any value other than 8 is a protocol VARIANT (recorded in the "
            "snapshot as verification.max_check_weight)."
        ),
    )
    campaign_cmd.add_argument(
        "--exact-check-weight",
        type=int,
        default=None,
        help=(
            "2026-10-23 (Fig.4 c): exact maximum check-weight contract. The code's max row "
            "weight over H_X / H_Z must EQUAL this value: lighter and heavier codes are both "
            "rejected at construction (check_weight_mismatch), never enter the archive, the "
            "parents, the slow queue or the final, and the Generator contract says so. Sets "
            "verification.max_check_weight and the promotion gate to the same value (an "
            "explicit different --max-check-weight / --promotion-max-check-weight is refused). "
            "The run takes the paper's fixed-weight final stage (Fig. 4 c, paper_fig4c_top6_confirm_v1: "
            "the six leading distinct codes by search-stage pLER, 25 000 shots per basis each, no "
            "cumulative top-up, the three best reported). "
            "A protocol VARIANT (snapshot keys verification.exact_check_weight, exact_check_weight)."
        ),
    )
    campaign_cmd.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help=(
            "2026-10-23 (Fig.4 c): planning identity of this campaign (e.g. f4v26_N98_cnot_w6_claude_rep1), "
            "recorded in the protocol snapshot as campaign_id; no effect on the search."
        ),
    )
    campaign_cmd.add_argument(
        "--task",
        choices=["memory", "cnot_dense"],
        default="memory",
        help=(
            "2026-10-12 (Fig.3 Phase B): what the slow loop measures. memory = the "
            "Fig.2 two-basis memory experiment (formal); cnot_dense = two blocks of "
            "the code with a transversal logical CNOT after every QEC round of the "
            "same T rounds (see --rounds-policy), decoded jointly and reported per "
            "logical qubit per round. A protocol VARIANT: it also switches the "
            "structural profile to telescoping_structural_v3_7_cnot (3S + 4R slow "
            "queue with the lightest-endpoint slot; not-competitive prompt block)."
        ),
    )
    campaign_cmd.add_argument(
        "--rounds-policy",
        choices=["max_Tc_d", "fixed_Tc"],
        default="fixed_Tc",
        help=(
            "How many QEC rounds a candidate is measured for, in the search and the final stage. "
            "fixed_Tc (default, the paper: R = 12 for every evaluation) = 12 for every code, so the "
            "gate-dense task is the same fixed-depth circuit (11 transversal CNOTs) for all codes; "
            "max_Tc_d = max(12, d_search), the rule of the as-run Fig.2 v3.7/v3.8 data (a protocol "
            "VARIANT, recorded in the snapshot as rounds_policy.search/final)."
        ),
    )
    campaign_cmd.add_argument(
        "--protocol-profile",
        choices=("v3_7", "v3_8"),
        default="v3_7",
        help=(
            "2026-10-17: structural retention profile. v3_7 (default) = the formal "
            "Fig.2 protocol; v3_8 = v3.7 + variant generations (the last 2 of each "
            "turn's 4 generations only vary the measured-best codes and the resident "
            "reference) + endpoint family rules (distinct-family endpoints, a family "
            "demoted after 3 consecutive >= 10x endpoint strikes). A declared variant: "
            "the protocol snapshot carries telescoping_structural_v3_8."
        ),
    )
    campaign_cmd.add_argument(
        "--racing-max-errors",
        type=int,
        default=None,
        help=(
            "Without this flag the search uses the paper's early stop (Appendix D3, policy "
            "paper_d3_wilson10x_ref_maxerr30_v1: 2000-shot pilot, 250-shot batches, Wilson 10x "
            "against the best measured code and the applicable reference code, 30 failures per "
            "basis, 25000 shots per basis). 100 / 30 select the earlier as-run policies "
            "pilot10x_maxerr100_v3 / pilot10x_maxerr30_v3 (protocol VARIANTS); other values are "
            "refused; a resumed chain switches policy at the turn boundary."
        ),
    )
    campaign_cmd.add_argument(
        "--final-cumulative-shots",
        type=int,
        default=None,
        help=(
            "PROTOCOL VARIANT (refused under --strict-turns): the per-basis cumulative shot limit of "
            "the final stage's top-up instead of the paper's Table VII value for (--task, --budget) "
            "and of the reference code's paper total (50 000 / 100 000); for smoke runs. A budget "
            "absent from Table VII has no top-up unless this is given. Under --exact-check-weight "
            "(no finalist top-up) it sets the reference's total only."
        ),
    )
    campaign_cmd.add_argument(
        "--distance-timeout-s",
        type=float,
        help="Wall-clock watchdog for one distance-evidence subprocess; 0 disables it.",
    )
    campaign_cmd.add_argument(
        "--timeout-retries",
        type=int,
        help=(
            "Identical replays of a timed-out worker request before it is "
            "incomplete (default follows the config: 0 under "
            "telescoping_structural_v3_7 — a watchdog timeout is deterministic, "
            "replaying it only burns the watchdog again)."
        ),
    )
    campaign_cmd.add_argument(
        "--transient-retries",
        type=int,
        default=1,
        help=(
            "Identical replays of a worker request whose subprocess died "
            "(no parseable JSON, killed/OOM) or lost its DEM (dem_load_error); "
            "never applies to timeouts. The formal telescoping_structural_v3_7 "
            "protocol freezes this at 1."
        ),
    )
    campaign_cmd.add_argument(
        "--final-shots",
        type=int,
        help=(
            "Fixed confirmation shots per X/Z basis after all turns. Default: "
            "the telescoping_structural_v3_7 budget policy of --budget (200,000 "
            "below 300 total qubits, 50,000 at >= 300); kept separate from "
            "--racing-confirm so final precision can increase with budget "
            "without multiplying every search turn. Any other value is a "
            "protocol variant and is refused under --strict-turns."
        ),
    )
    campaign_cmd.add_argument(
        "--strict-turns",
        action="store_true",
        help=(
            "Formal-run mode: require an agentic DeepSeek plan, exactly "
            "--proposal-count programs, a complete generator batch, and an "
            "all-ok turn audit before incrementing turns_used."
        ),
    )
    campaign_cmd.add_argument(
        "--fixed-program-count",
        action="store_true",
        help=(
            "2026-09-26: freeze the per-generation program count at "
            "--proposal-count WITHOUT the rest of --strict-turns (a protocol "
            "VARIANT such as Fig.3 cannot run strict). The orchestrator's plan "
            "may not lower it; the snapshot records fixed_program_count=true."
        ),
    )
    campaign_cmd.add_argument("--feedback-mode", default="both",
                              choices=["both", "fast", "slow", "none"])
    campaign_cmd.add_argument(
        "--ablation", default="full", choices=list(ABLATION_IDS),
        help=(
            "2026-10-23 (appendix S3): search-feedback ablation. no_slow = fast-loop-only "
            "(no slow-loop measurement during the search; the generator sees the structural "
            "layer only; feedback_mode fast); no_feedback = every turn from the task/constraint "
            "prompt alone (no archive, parents, planner, reflector or measurement; feedback_mode "
            "none; 4 independent generator calls per turn). The frozen final stage measures the "
            "structural front's slot order (final_structural + final_runtime codes). A protocol "
            "VARIANT recorded in the snapshot (key ablation); --feedback-mode must agree or be "
            "left at its default."
        ),
    )
    campaign_cmd.add_argument(
        "--search-method", default="llm", choices=list(SEARCH_METHODS),
        help=(
            "2026-11-06: llm = the LLM search; random / bo = the search baselines without an LLM "
            "(omniqec_agentic/baselines): a fixed sampling prior proposes 4 x 960 draws per turn over "
            "--families, and the slow queue is a uniform (random) or Gaussian-process (bo) choice from "
            "the run's cumulative legal pool; the final shortlist is the measured ranking. Requires "
            "--provider baseline_sampler and --strict-turns with --proposal-count 960 "
            "--generations-per-turn 4 --rng-seeds-per-program 1. A protocol VARIANT recorded in the "
            "snapshot (key search_baseline)."
        ),
    )
    campaign_cmd.add_argument(
        "--search-seed-id", default="r01",
        help="search baselines: the pre-specified search seed (r01 = the main figure's run; r00 = pilot).",
    )
    campaign_cmd.add_argument("--resume", action="store_true")
    campaign_cmd.add_argument(
        "--reference-codes-json", default=None,
        help=("PROTOCOL VARIANT: JSON file (schema reference_codes_v1) of background codes "
              "shown to the generator and the orchestrator every turn; frozen into the "
              "protocol snapshot, refused by the formal validator."),
    )
    campaign_cmd.set_defaults(func=run_campaign_command)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def run_funsearch_command(args: argparse.Namespace) -> int:
    load_dotenv()
    config = default_run_config(
        run_id=args.run_id,
        n_budget=args.n_budget,
        hardware=args.hardware,
        out_dir=args.out,
    )
    config = _replace_proposal_count(config, int(args.proposal_count))
    config = dataclasses.replace(
        config,
        deformation_mode=DeformationMode(str(args.deformation_mode)),
    )
    config = _replace_verification_proxy_iters(config, int(args.proxy_iters))
    config = _replace_fast_proxy_bounds(config, args.proxy_n_min, args.proxy_n_max)
    config = dataclasses.replace(config, lp_shape_pin=_lp_shape_pin_from_args(args))
    if args.promote_top is not None:
        config = _replace_promote_top(config, int(args.promote_top))
    if args.families:
        config = dataclasses.replace(
            config, allowed_families=tuple(canonical_code_family(name) for name in args.families)
        )
    if args.llm_temperature is not None:
        config = dataclasses.replace(
            config, llm=dataclasses.replace(config.llm, temperature=float(args.llm_temperature))
        )
    if getattr(args, "feedback", None):
        config = dataclasses.replace(config, feedback_mode=str(args.feedback))
    summary = run_funsearch(
        config,
        generations=int(args.generations),
        provider=args.provider,
        use_seeds=not bool(args.no_seeds),
        save_every=int(args.save_every),
        rng_seeds_per_program=int(args.rng_seeds_per_program),
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def run_agentic_cycle_command(args: argparse.Namespace) -> int:
    from .campaign import run_agentic_cycle

    load_dotenv()
    config = default_run_config(
        run_id=args.run_id,
        n_budget=args.n_budget,
        hardware=args.hardware,
        out_dir=args.out,
    )
    config = _replace_proposal_count(config, int(args.proposal_count))
    config = dataclasses.replace(
        config,
        deformation_mode=DeformationMode(str(args.deformation_mode)),
    )
    config = _replace_verification_proxy_iters(config, int(args.proxy_iters))
    config = _replace_fast_proxy_bounds(config, args.proxy_n_min, args.proxy_n_max)
    noise = _noise_from_args(args, multi=True, legacy_attr="slow_p_values")
    config = _replace_verification(
        config,
        p_values=noise.values,
        noise=noise,
        slope_offset=0.958 if noise.profile in UNIFORM_PROFILES else None,
    )
    if args.slow_rounds is not None:
        if int(args.slow_rounds) < 1:
            raise ValueError("--slow-rounds must be >= 1")
        config = _replace_verification(config, rounds=int(args.slow_rounds))
    if args.slow_timeout_s is not None:
        config = _replace_circuit_timeout(config, float(args.slow_timeout_s))
    config = _replace_exact_circuit_distance(
        config,
        enabled=bool(args.exact_circuit_distance),
        dem_isd_enabled=bool(args.dem_isd_distance),
        dem_isd_max_mechanisms=int(args.dem_isd_max_mechanisms),
        dem_isd_iters=int(args.dem_isd_iters),
    )
    if args.d_cap is not None:
        if int(args.d_cap) <= 0:
            config = _replace_verification(config, d_cap_enabled=False)
        else:
            config = _replace_verification(config, d_cap=int(args.d_cap))
    if getattr(args, "decoders", None):
        config = _replace_verification(
            config, decoders=tuple(DecoderName(name) for name in args.decoders)
        )
    if getattr(args, "bases", None):
        config = _replace_verification(config, bases=tuple(args.bases))
    config = dataclasses.replace(config, lp_shape_pin=_lp_shape_pin_from_args(args))
    if args.families:
        config = dataclasses.replace(
            config, allowed_families=tuple(canonical_code_family(name) for name in args.families)
        )
    if args.llm_temperature is not None:
        config = dataclasses.replace(
            config, llm=dataclasses.replace(config.llm, temperature=float(args.llm_temperature))
        )
    if getattr(args, "feedback", None):
        config = dataclasses.replace(config, feedback_mode=str(args.feedback))
    slow_limit = args.slow_limit
    if slow_limit is None and not bool(args.slow_all_archive):
        slow_limit = 2
    promote_top = args.promote_top
    if promote_top is None and slow_limit is not None:
        promote_top = max(config.verification.promote_top, int(slow_limit))
    if promote_top is not None:
        config = _replace_promote_top(config, int(promote_top))
    summary = run_agentic_cycle(
        config,
        provider=args.provider,
        generations=int(args.generations),
        promote_every=int(args.promote_every),
        slow_limit=slow_limit,
        slow_p_values=tuple(float(value) for value in noise.values),
        slow_shots=int(args.slow_shots),
        slow_batch_size=int(args.slow_batch_size),
        slow_max_errors=args.slow_max_errors,
        slow_all_archive=bool(args.slow_all_archive),
        slow_workers=int(args.slow_workers),
        slow_shot_shards=int(args.slow_shot_shards),
        rng_seeds_per_program=int(args.rng_seeds_per_program),
        use_seeds=not bool(args.no_seeds),
        seed_sources=_load_seed_sources(getattr(args, "seed_sources_json", None)),
        save_every=int(args.save_every),
        prefer_coloring=not bool(args.no_coloring),
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def _load_seed_sources(path: str | None) -> list[str] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("seed-sources-json must be a JSON list of construct(rng) source strings")
    return [str(s) for s in data if str(s).strip()]


def check_slow_deps_command(args: argparse.Namespace) -> int:
    from .verification.decoders import decoder_dependency_report

    load_dotenv()
    config = VerificationConfig()
    report = decoder_dependency_report()
    payload = {
        "status": report.status,
        "dependencies": report.to_dict(),
        "decoder_build": config.decoder_build.to_dict(),
        "verification_defaults": config.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_dem_smoke_command(args: argparse.Namespace) -> int:
    import numpy as np

    from .construction import build_compiled_code
    from .examples.seed_programs import SEEDS
    from .sandbox import run_construct_source
    from .verification.circuit import build_memory_experiment_pair
    from .verification.fast import evaluate_proxy

    load_dotenv()
    seed_index, program_id, seed_source = _resolve_seed_source(SEEDS, int(args.seed_index))
    constructed = run_construct_source(seed_source, program_id=program_id, seed=0).constructed
    compiled = build_compiled_code(
        constructed.spec,
        program_id=program_id,
        deformation_mode=args.deformation_mode,
    )
    hx = np.asarray(compiled.hx, dtype=np.uint8)
    hz = np.asarray(compiled.hz, dtype=np.uint8)
    fast = evaluate_proxy(
        constructed.spec,
        iters=int(args.proxy_iters),
        deformation_mode=args.deformation_mode,
    )
    d_hint = fast.get("d") if "error" not in fast else None
    noise = _noise_from_args(args, multi=False)
    noise_point = noise.point(0)
    payload = {
        "seed_index": seed_index,
        "program_id": program_id,
        "family": constructed.family,
        "n_data": int(hx.shape[1]),
        "n_checks_x": int(hx.shape[0]),
        "n_checks_z": int(hz.shape[0]),
        "noise": noise.to_dict(),
        "noise_point": noise_point.to_dict(),
        noise_point.axis: float(noise_point.value),
        "rounds_requested": args.rounds,
        "d_hint": d_hint,
        "fast": fast,
        "sec_policy": _sec_policy_from_args(args).to_dict(),
    }
    try:
        pair = build_memory_experiment_pair(
            hx,
            hz,
            p=float(noise_point.value),
            rounds=args.rounds,
            d_hint=int(d_hint) if d_hint is not None else None,
            prefer_coloring=not bool(args.no_coloring),
            noise_point=noise_point,
            h_mask=compiled.h_mask,
            deformation=compiled.deformation,
            sector_sizes=compiled.sector_sizes,
            sec_policy=_sec_policy_from_args(args),
            layer_blocks=compiled.layer_blocks,
        )
    except Exception as exc:
        payload["status"] = "tool_unavailable" if "stim unavailable" in str(exc) else "error"
        payload["failure_reason"] = str(exc)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    payload["status"] = "ok"
    payload["memory_experiments"] = pair.to_metadata()
    payload["css_commutation_violations"] = int(((np.asarray(hx) @ np.asarray(hz).T) % 2).sum())
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def slow_verify_seed_command(args: argparse.Namespace) -> int:
    from .examples.seed_programs import SEEDS
    from .sandbox import run_construct_source
    from .evaluation import evaluate_slow_spec, write_slow_result

    load_dotenv()
    seed_index, program_id, seed_source = _resolve_seed_source(SEEDS, int(args.seed_index))
    constructed = run_construct_source(seed_source, program_id=program_id, seed=0).constructed
    noise = _noise_from_args(args, multi=True)
    verification = VerificationConfig(
        proxy_iters=int(args.proxy_iters),
        decoders=tuple(DecoderName(name) for name in args.decoders),
        p_values=noise.values,
        noise=noise,
        shots=int(args.shots),
        rounds=args.rounds,
        max_errors=args.max_errors,
        circuit_timeout_s=float(args.timeout_s),
        exact_distance=ExactDistanceConfig(
            circuit_stim=bool(args.exact_circuit_distance),
            circuit_dem_isd=bool(args.dem_isd_distance),
            dem_isd_max_mechanisms=int(args.dem_isd_max_mechanisms),
            dem_isd_iters=int(args.dem_isd_iters),
        ),
        sec_policy=_sec_policy_from_args(args),
    )
    payload = evaluate_slow_spec(
        program_id=program_id,
        spec=constructed.spec,
        verification=verification,
        seed=0,
        batch_size=int(args.batch_size),
        prefer_coloring=not bool(args.no_coloring),
    )
    payload["seed_index"] = seed_index
    if args.out:
        output_path = write_slow_result(payload, args.out)
        payload["artifact_path"] = str(output_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_seed_source(seeds, seed_index: int):
    """Resolve a 1-based CLI seed index to a built-in seed source."""

    seed_index = int(seed_index)
    if seed_index < 1 or seed_index > len(seeds):
        raise ValueError(f"seed-index must be in [1,{len(seeds)}]")
    return seed_index, f"seed_{seed_index:04d}", seeds[seed_index - 1]


def slow_verify_queue_command(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_promotion_queue

    load_dotenv()
    noise = _noise_from_args(args, multi=True)
    verification = VerificationConfig(
        proxy_iters=int(args.proxy_iters),
        decoders=tuple(DecoderName(name) for name in args.decoders),
        p_values=noise.values,
        noise=noise,
        shots=int(args.shots),
        rounds=args.rounds,
        max_errors=args.max_errors,
        circuit_timeout_s=float(args.timeout_s),
        decoder_build=dataclasses.replace(
            DecoderBuildConfig(),
            bposd=dataclasses.replace(BposdDecoderConfig(), osd_order=int(args.osd_order)),
        ),
        exact_distance=ExactDistanceConfig(
            circuit_stim=bool(args.exact_circuit_distance),
            circuit_dem_isd=bool(args.dem_isd_distance),
            dem_isd_max_mechanisms=int(args.dem_isd_max_mechanisms),
            dem_isd_iters=int(args.dem_isd_iters),
        ),
        sec_policy=_sec_policy_from_args(args),
    )
    summary = evaluate_promotion_queue(
        queue_path=args.promotion_queue,
        out_dir=args.out,
        verification=verification,
        hardware=HardwareTarget(args.hardware),
        limit=args.limit,
        seed=0,
        batch_size=int(args.batch_size),
        prefer_coloring=not bool(args.no_coloring),
        include_baseline=bool(args.with_bb72_baseline),
        workers=int(args.workers),
        shot_shards=int(args.shot_shards),
        deformation_mode=args.deformation_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def calibrate_slope_offset_command(args: argparse.Namespace) -> int:
    """Recalibrate the LER-slope -> d_circ offset on the BB reference.

    offset = fitted_slope - d_true/2 (from slope ~ d_circ/2 + offset). The
    payload reports one suggestion per decoder plus the primary suggestion."""

    from .evaluation import BASELINE_BB_PROGRAM_ID, BASELINE_BB_SPEC, evaluate_slow_spec, write_slow_result

    load_dotenv()
    verification = VerificationConfig(
        proxy_iters=int(args.proxy_iters),
        decoders=tuple(DecoderName(name) for name in args.decoders),
        p_values=tuple(float(p) for p in args.p_values),
        shots=int(args.shots),
        max_errors=args.max_errors,
        circuit_timeout_s=float(args.timeout_s),
    )
    payload = evaluate_slow_spec(
        program_id=BASELINE_BB_PROGRAM_ID,
        spec=dict(BASELINE_BB_SPEC),
        verification=verification,
        seed=0,
        batch_size=int(args.batch_size),
    )
    d_true = int(args.d_true)
    suggestions: list[dict[str, object]] = []
    evidence = payload.get("distance_evidence") if isinstance(payload.get("distance_evidence"), dict) else {}
    for estimate in evidence.get("estimates") or []:
        if not isinstance(estimate, dict) or estimate.get("slope") is None:
            continue
        slope = float(estimate["slope"])
        suggestions.append(
            {
                "decoder": estimate.get("decoder"),
                "slope": slope,
                "suggested_slope_offset": slope - d_true / 2.0,
                "fit_points": len(estimate.get("fit_points") or []),
            }
        )
    result = {
        "schema": "slope_offset_calibration",
        "reference": f"BB [[72,12,6]] (d_true={d_true})",
        "status": "ok" if suggestions else "no_slope_fit",
        "p_values": [float(p) for p in args.p_values],
        "shots": int(args.shots),
        "suggestions": suggestions,
        "primary_suggested_slope_offset": suggestions[0]["suggested_slope_offset"] if suggestions else None,
        "current_default_offset": float(VerificationConfig.slope_offset),
        "slow_payload_status": payload.get("status"),
    }
    if not suggestions:
        result["hint"] = (
            "Need >=2 p-values each reaching >=20 logical failures; raise --shots "
            "or widen --p-values."
        )
    if args.out:
        write_slow_result({"calibration": result, "slow_payload": payload}, args.out)
        result["artifact_path"] = str(args.out)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def make_slow_aware_context_command(args: argparse.Namespace) -> int:
    from .evaluation import make_slow_aware_prompt_context

    summary = make_slow_aware_prompt_context(
        prompt_context_path=args.prompt_context,
        slow_feedback_path=args.slow_feedback,
        out_path=args.out,
        parent_limit=int(args.parent_limit),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def export_stim_seed_command(args: argparse.Namespace) -> int:
    from .examples.seed_programs import SEEDS
    from .export_stim import export_stim_circuits_for_spec
    from .sandbox import run_construct_source

    seed_index, program_id, seed_source = _resolve_seed_source(SEEDS, int(args.seed_index))
    constructed = run_construct_source(seed_source, program_id=program_id, seed=0).constructed
    noise = _noise_from_args(args, multi=False)
    payload = export_stim_circuits_for_spec(
        program_id=program_id,
        spec=constructed.spec,
        out_dir=args.out,
        p=float(noise.values[0]),
        noise=noise,
        rounds=args.rounds,
        proxy_iters=int(args.proxy_iters),
        sec_policy=_sec_policy_from_args(args),
    )
    payload["seed_index"] = seed_index
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def export_stim_queue_command(args: argparse.Namespace) -> int:
    from .export_stim import export_stim_circuits_for_queue

    noise = _noise_from_args(args, multi=False)
    summary = export_stim_circuits_for_queue(
        queue_path=args.queue,
        out_dir=args.out,
        limit=args.limit,
        p=float(noise.values[0]),
        noise=noise,
        rounds=args.rounds,
        sec_policy=_sec_policy_from_args(args),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def analyze_runs_command(args: argparse.Namespace) -> int:
    from .analyze import analyze_runs, format_leaderboard

    rows = analyze_runs(list(args.runs), bb_q=float(args.bb_q))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    else:
        print(format_leaderboard(rows))
    return 0


def _replace_verification(config, **overrides):
    """Rebuild RunConfig with updated VerificationConfig fields.

    dataclasses.replace keeps every field not named here, so newly added
    VerificationConfig knobs (d_cap, slope_offset, ...) survive CLI overrides.
    """

    return dataclasses.replace(
        config,
        verification=dataclasses.replace(config.verification, **overrides),
    )


def _replace_verification_proxy_iters(config, proxy_iters: int):
    return _replace_verification(config, proxy_iters=int(proxy_iters))


def _replace_fast_proxy_bounds(config, proxy_n_min, proxy_n_max):
    return _replace_verification(config, proxy_n_min=proxy_n_min, proxy_n_max=proxy_n_max)


def _replace_circuit_timeout(config, timeout_s: float):
    return _replace_verification(config, circuit_timeout_s=float(timeout_s))


def qualify_decoders_command(args: argparse.Namespace) -> int:
    from .qualification import run_decoder_qualification

    report = run_decoder_qualification(
        set_path=args.set_path,
        out_dir=args.out,
        candidates=tuple(args.candidates),
        references=tuple(args.references),
        profiles=tuple(args.profiles),
        shots=int(args.shots),
        seed=int(args.seed),
        workers=int(args.workers),
        limit=args.limit,
        determinism_check=not args.skip_determinism,
        beam32_throughput=not args.skip_beam32,
    )
    acceptance = report.get("acceptance") or {}
    if acceptance:
        print(
            f"[qualify] acceptance ({acceptance.get('decoder')} @ p={acceptance.get('p')}): "
            f"{'PASS' if acceptance.get('passed') else 'FAIL'}"
        )
        for entry in acceptance.get("codes") or []:
            print(
                f"[qualify]   {entry['program_id']}: ler_x={entry.get('ler_x')} "
                f"ler_z={entry.get('ler_z')} plpr={entry.get('per_logical_per_round')} "
                f"ceiling={entry.get('max_basis_ler')} pass={entry.get('passed')}"
            )
    determinism = report.get("determinism") or {}
    if determinism:
        print(
            f"[qualify] same-seed determinism: "
            f"{'PASS' if determinism.get('deterministic') else 'FAIL'}"
        )
    for candidate, cand in (report.get("candidates_report") or {}).items():
        print(
            f"[qualify] {candidate} vs {cand['reference']}: "
            f"{'QUALIFIED' if cand['qualified'] else 'NOT QUALIFIED'}"
        )
    extras = report.get("beam_extras") or {}
    if extras:
        print(f"[qualify] beam determinism: {'PASS' if extras.get('deterministic') else 'FAIL'}")
    print(f"[qualify] report: {report['artifact_paths']['report_md']}")
    return 0


def run_campaign_command(args: argparse.Namespace) -> int:
    from .config import LLMConfig, RacingConfig, VerificationConfig
    from .orchestrator import make_policy
    from .workflow import CampaignSpec, run_campaign

    load_dotenv()
    # strict_turns forbids fallback turns and pins ONE model id, so the only
    # incompatible provider is dry_run -- make_policy returns FallbackPolicy for
    # it by construction. The guard used to name deepseek explicitly, which
    # predates resolve_model_name() teaching required_model to follow the
    # provider (2026-08-23): it silently made the openrouter arm -- the GPT and
    # DeepSeek route this project now runs on -- unlaunchable under the formal
    # protocol, including via fig2_campaign.pbs's own PROVIDER=openrouter branch.
    # v3.3 (kept by v3.4) freezes racing OFF; the external validator already
    # rejects a run with racing enabled, but only AFTER the whole campaign has
    # burned its budget. Refuse the combination at launch instead (audit fix).
    if bool(args.strict_turns) and bool(getattr(args, "racing", False)):
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            "freezes racing OFF and measures fixed shots; --racing is a "
            "different protocol and must be launched without --strict-turns"
        )
    if bool(args.strict_turns) and str(args.provider) == "dry_run":
        raise ValueError(
            "--strict-turns cannot be used with --provider dry_run "
            "(dry_run is the FallbackPolicy, and strict turns forbid fallback)"
        )
    shot_shards = int(getattr(args, "shot_shards", 16))
    if shot_shards < 1:
        raise SystemExit("--shot-shards must be >= 1")
    if bool(args.strict_turns) and shot_shards != 16:
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            "freezes --shot-shards at 16"
        )
    # v3.4: the final stage has its own frozen partition count so 200k shots
    # per basis fit the per-shot decode watchdog (16 shards could never
    # finish the formal final at b220+ on NSCC).
    final_shot_shards = int(getattr(args, "final_shot_shards", 128))
    if final_shot_shards < 1:
        raise SystemExit("--final-shot-shards must be >= 1")
    if bool(args.strict_turns) and final_shot_shards != 128:
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            "freezes --final-shot-shards at 128"
        )
    # v3.6 budget policy (docs/06 §1.4, user decision 2026-09-03): the budget
    # point selects the search and final shots per basis and nothing else
    # (the telescoping decoder carries no per-point limit). Absent flags
    # derive from it; an explicit value is a protocol variant, honoured
    # outside strict mode and refused under it.
    policy = budget_policy(int(args.budget))
    slow_shots_arg = getattr(args, "slow_shots", None)
    slow_shots = int(policy["search_shots"]) if slow_shots_arg is None else int(slow_shots_arg)
    if slow_shots < 1:
        raise SystemExit("--slow-shots must be >= 1")
    if bool(args.strict_turns) and slow_shots != int(policy["search_shots"]):
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, whose budget "
            f"policy freezes --slow-shots at {policy['search_shots']} for "
            f"--budget {int(args.budget)} (got {slow_shots})"
        )
    final_shots_arg = getattr(args, "final_shots", None)
    final_shots = int(policy["final_shots"]) if final_shots_arg is None else int(final_shots_arg)
    if final_shots < 1:
        raise SystemExit("--final-shots must be >= 1")
    if bool(args.strict_turns) and final_shots != int(policy["final_shots"]):
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, whose budget "
            f"policy freezes --final-shots at {policy['final_shots']} for "
            f"--budget {int(args.budget)} (got {final_shots})"
        )
    llm_config = LLMConfig()
    spec = CampaignSpec(
        qubit_budget=int(args.budget),
        weight_limit=int(args.weight_limit) if args.weight_limit is not None else None,
        families=tuple(args.families) if args.families else CampaignSpec.model_fields["families"].default,
    )
    p_value = float(args.p)
    # v3.5: the noise model is a protocol field. run-campaign is a fixed-p
    # UNIFORM campaign (SI/Helios have no LRC formal protocol); the formal
    # profile measures the paper's SI C model at p=0.005, and the v3.4
    # idle-noise model or any other p is a different protocol.
    noise_profile = NoiseProfile(
        str(getattr(args, "noise_profile", None) or NoiseProfile.UNIFORM_PAPER.value)
    )
    if noise_profile not in UNIFORM_PROFILES:
        raise SystemExit(
            "run-campaign is a fixed-p uniform campaign; --noise-profile must be "
            f"one of {sorted(profile.value for profile in UNIFORM_PROFILES)} "
            f"(got {noise_profile.value})"
        )
    if bool(args.strict_turns) and noise_profile != NoiseProfile.UNIFORM_PAPER:
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            f"freezes --noise-profile at {NoiseProfile.UNIFORM_PAPER.value}; "
            f"{noise_profile.value} is a different protocol and must be "
            "launched without --strict-turns"
        )
    if bool(args.strict_turns) and abs(p_value - 0.005) > 1e-15:
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            f"freezes --p at 0.005 (got {p_value!r})"
        )
    noise = (
        NoiseSweepConfig.uniform_legacy((p_value,))
        if noise_profile == NoiseProfile.UNIFORM_LEGACY
        else NoiseSweepConfig.uniform((p_value,))
    )
    # v3.5 SEC: the circuit policy is a protocol field. The formal profile
    # freezes twobasis_nonint_v1 (K=16, cert_iters=400, seed 0); the
    # historical lrc_ranked_v1 is a different circuit and a different protocol.
    sec_policy = _sec_policy_from_args(args)
    if bool(args.strict_turns) and sec_policy != DEFAULT_SEC_POLICY:
        raise SystemExit(
            "--strict-turns runs the telescoping_structural_v3_7 protocol, which "
            "freezes --sec-policy at twobasis_nonint_v1 (K=16, cert_iters=400); "
            "lrc_ranked_v1 is a different protocol and must be launched without "
            "--strict-turns"
        )
    verification_kwargs: dict[str, Any] = {}
    if getattr(args, "decoders", None):
        decoders = tuple(DecoderName(name) for name in args.decoders)
        # telescoping_structural_v3_7 measures with exactly one decoder. A
        # different set is a different protocol, so it may not be launched
        # under the strict profile that certifies runs AS this protocol.
        if bool(args.strict_turns) and decoders != (DecoderName.TELESCOPING,):
            raise SystemExit(
                "--strict-turns runs the telescoping_structural_v3_7 protocol, which is "
                f"{DecoderName.TELESCOPING.value}-only; --decoders "
                f"{[d.value for d in decoders]} is a different protocol and must "
                "be launched without --strict-turns"
            )
        verification_kwargs["decoders"] = decoders
    legacy_timeout = getattr(args, "circuit_timeout_s", None)
    if legacy_timeout is not None:
        if float(legacy_timeout) < 0:
            raise SystemExit("--circuit-timeout-s must be >= 0 (0 disables it)")
        verification_kwargs["circuit_timeout_s"] = float(legacy_timeout)
        for field in (
            "build_timeout_s",
            "decode_shard_timeout_s",
            "distance_timeout_s",
        ):
            verification_kwargs[field] = float(legacy_timeout)
    for field, flag in (
        ("build_timeout_s", "--build-timeout-s"),
        ("decode_shard_timeout_s", "--decode-shard-timeout-s"),
        ("distance_timeout_s", "--distance-timeout-s"),
    ):
        value = getattr(args, field, None)
        if value is None:
            continue
        if float(value) < 0:
            raise SystemExit(f"{flag} must be >= 0 (0 disables it)")
        verification_kwargs[field] = float(value)
    timeout_retries = getattr(args, "timeout_retries", None)
    if timeout_retries is not None:
        if int(timeout_retries) < 0:
            raise SystemExit("--timeout-retries must be >= 0")
        verification_kwargs["timeout_retries"] = int(timeout_retries)
    # v3.4: per-shot decode budget (the shard watchdog scales with shard
    # shots above the flat floor) and transient replays of dead workers.
    decode_shot_budget = getattr(args, "decode_shot_budget_s", None)
    if decode_shot_budget is not None:
        if float(decode_shot_budget) < 0:
            raise SystemExit(
                "--decode-shot-budget-s must be >= 0 (0 disables the per-shot budget)"
            )
        verification_kwargs["decode_shot_budget_s"] = float(decode_shot_budget)
    transient_retries = getattr(args, "transient_retries", None)
    if transient_retries is not None:
        if int(transient_retries) < 0:
            raise SystemExit("--transient-retries must be >= 0")
        verification_kwargs["transient_retries"] = int(transient_retries)
    # 2026-10-17: --protocol-profile v3_8 selects the structural retention
    # profile telescoping_structural_v3_8 (variant generations + endpoint
    # family rules); v3_7 (default) is the formal Fig.2 protocol.
    from .config import StructuralSearchConfig as _StructuralSearchConfig
    from .config import structural_search_for_profile

    structural_profile = structural_search_for_profile(getattr(args, "protocol_profile", None) or "v3_7")
    if structural_profile.profile_id != _StructuralSearchConfig().profile_id:
        print(
            f"[campaign] PROTOCOL VARIANT: structural profile {structural_profile.profile_id} "
            f"(exploit_generations={structural_profile.exploit_generations}, "
            f"endpoint_family_distinct={structural_profile.endpoint_family_distinct}, "
            f"endpoint_demotion={structural_profile.endpoint_demotion_strikes}x"
            f"{structural_profile.endpoint_demotion_factor:g})",
            flush=True,
        )
    verification = VerificationConfig(
        p_values=(p_value,),
        noise=noise,
        shots=int(slow_shots),
        max_errors=None,
        sec_policy=sec_policy,
        structural=structural_profile,
        **verification_kwargs,
    )
    weight_cap = getattr(args, "max_check_weight", None)
    if weight_cap is not None:
        import dataclasses as _dc_cap

        from .config import SEARCH_MAX_CHECK_WEIGHT as _SEARCH_CAP

        if int(weight_cap) == 0:
            # The v3.7 as-run protocol: no fast-loop cap (the slow-loop gate
            # --promotion-max-check-weight, if any, still applies).
            verification = _dc_cap.replace(verification, max_check_weight=None)
            print(
                "[campaign] PROTOCOL VARIANT: max_check_weight=None (no fast-loop check-weight cap; "
                f"the paper's searches use {_SEARCH_CAP})",
                flush=True,
            )
            weight_cap = None
        elif int(weight_cap) < 3:
            raise SystemExit(f"--max-check-weight must be 0 (no cap) or >= 3 (got {weight_cap})")
        else:
            verification = _dc_cap.replace(verification, max_check_weight=int(weight_cap))
            if int(weight_cap) != int(_SEARCH_CAP):
                print(
                    f"[campaign] PROTOCOL VARIANT: max_check_weight={int(weight_cap)} (codes with a heavier "
                    "X or Z check are rejected at construction and never evaluated)",
                    flush=True,
                )
    exact_weight = getattr(args, "exact_check_weight", None)
    if exact_weight is not None:
        import dataclasses as _dc_exact

        exact_weight = int(exact_weight)
        if exact_weight < 3:
            raise SystemExit(f"--exact-check-weight must be >= 3 (got {exact_weight})")
        if weight_cap is not None and int(weight_cap) != exact_weight:
            raise SystemExit(
                f"--max-check-weight {int(weight_cap)} contradicts --exact-check-weight {exact_weight}"
            )
        gate = getattr(args, "promotion_max_check_weight", None)
        if gate is not None and int(gate) != exact_weight:
            raise SystemExit(
                f"--promotion-max-check-weight {int(gate)} contradicts --exact-check-weight {exact_weight}"
            )
        args.promotion_max_check_weight = exact_weight
        verification = _dc_exact.replace(
            verification, max_check_weight=exact_weight, exact_check_weight=exact_weight
        )
        print(
            f"[campaign] PROTOCOL VARIANT: exact_check_weight={exact_weight} (a code whose maximum "
            "X/Z check weight differs is rejected at construction and never evaluated; the slow-queue "
            "and final gates hold the same value; final stage paper_fig4c_top6_confirm_v1: six finalists, "
            "the initial confirmation only)",
            flush=True,
        )
    if getattr(args, "promotion_max_check_weight", None) is None and verification.max_check_weight is not None:
        # 2026-11-10 (paper: w_max <= 8 for every main search): the slow-loop gate defaults to the fast-loop cap,
        # so the snapshot states the same bound at both stages (8 unless --max-check-weight sets a variant;
        # no gate with --max-check-weight 0).
        args.promotion_max_check_weight = int(verification.max_check_weight)
    task = str(getattr(args, "task", "memory") or "memory")
    if task != "memory":
        import dataclasses as _dc

        from .config import fig3_cnot_structural_config

        if structural_profile.profile_id != _StructuralSearchConfig().profile_id:
            # the CNOT task has one structural profile (telescoping_structural_v3_7_cnot_v2, built on
            # v3.7); a v3.8 base would carry the variant-generation / endpoint rules under the CNOT id
            raise SystemExit(
                f"--task {task} runs the Fig.3 CNOT profile; --protocol-profile "
                f"{getattr(args, 'protocol_profile', None)} applies to the memory task only"
            )
        structural = fig3_cnot_structural_config(verification.structural)
        # promote_top is the archive's promotion head = the slow queue's cap
        # (build_protocol_snapshot refuses a mismatch); 3S + 4R -> 7.
        verification = _dc.replace(
            verification, task=task, structural=structural, promote_top=int(structural.slow_candidate_cap)
        )
        print(
            f"[campaign] PROTOCOL VARIANT: task={task} (two-block transversal-CNOT circuit, one "
            "CNOT per QEC round, reported per logical qubit per round); structural profile "
            f"{verification.structural.profile_id}: {verification.structural.structural_slow_slots}S + "
            f"{verification.structural.runtime_slow_slots}R slow queue with the lightest-endpoint slot, "
            f"not-competitive prompt block at {verification.structural.not_competitive_ratio}x",
            flush=True,
        )
    # 2026-11-10 (paper alignment): fixed_Tc (R = 12 for every evaluation) is the default; max_Tc_d is the
    # rule of the as-run Fig.2 data and a declared variant.
    rounds_policy = str(getattr(args, "rounds_policy", "fixed_Tc") or "fixed_Tc")
    if rounds_policy != str(verification.rounds_policy):
        import dataclasses as _dc_rounds

        verification = _dc_rounds.replace(verification, rounds_policy=rounds_policy)
    if rounds_policy != "fixed_Tc":
        print(
            f"[campaign] PROTOCOL VARIANT: rounds_policy={rounds_policy} (a candidate with d_search > T_c is "
            "measured at T = d_search in the search and the final stage; the paper uses R = 12 throughout)",
            flush=True,
        )
    if getattr(args, "no_early_stop", False):
        racing = RacingConfig()  # PROTOCOL VARIANT: fixed shots, no early stop
        print("[campaign] PROTOCOL VARIANT: early stop disabled (--no-early-stop)", flush=True)
    elif bool(args.racing):
        # Legacy custom racing (03 §8 double threshold): a declared variant.
        racing = RacingConfig(
            enabled=True,
            shots_pilot=int(args.racing_pilot),
            batch=int(args.racing_batch),
            shots_confirm=int(args.racing_confirm),
            shots_max=int(args.racing_max),
        )
    elif getattr(args, "racing_max_errors", None) is None:
        # 2026-11-10 (paper alignment): the paper's early stop (Appendix D3) follows the budget point's
        # search shots: 2000-shot pilot, 250-shot batches, Wilson 10x against the best measured code AND the
        # applicable reference code, 30 failures per basis.
        from .config import paper_racing

        racing = paper_racing(int(budget_policy(int(args.budget))["search_shots"]))
    else:
        # --racing-max-errors 100 / 30: the earlier as-run policies (pilot10x_maxerr100_v3 /
        # pilot10x_maxerr30_v3), declared variants of the paper's rule.
        from .config import formal_racing_for_max_errors

        racing = formal_racing_for_max_errors(
            int(budget_policy(int(args.budget))["search_shots"]),
            getattr(args, "racing_max_errors", None),
        )
        print(f"[campaign] PROTOCOL VARIANT: early-stop policy {racing.policy_id} "
              f"(max_errors={racing.max_errors}; the paper's rule is the default)", flush=True)
    reference_codes = None
    if getattr(args, "reference_codes_json", None):
        from .orchestrator.reference_codes import load_reference_codes

        reference_codes = load_reference_codes(str(args.reference_codes_json))
        print(f"[campaign] PROTOCOL VARIANT: {len(reference_codes['codes'])} reference codes from "
              f"{args.reference_codes_json} frozen into the protocol snapshot", flush=True)
    # S3 ablation (2026-10-23): the generator's feedback_mode follows the mode.
    ablation = str(getattr(args, "ablation", "full") or "full")
    feedback_mode = str(args.feedback_mode)
    if ablation != "full":
        expected_feedback = ABLATION_FEEDBACK_MODE[ablation]
        if feedback_mode not in ("both", expected_feedback):
            raise SystemExit(
                f"--ablation {ablation} runs the generator with --feedback-mode "
                f"{expected_feedback}; got --feedback-mode {feedback_mode}"
            )
        feedback_mode = expected_feedback
        print(
            f"[campaign] PROTOCOL VARIANT: ablation={ablation} (no slow-loop measurement during "
            f"the search; generator feedback_mode={feedback_mode}; "
            + ("no planner, no reflector, no parents; " if ablation == "no_feedback" else "")
            + "final stage = structural front slot order)",
            flush=True,
        )
    # 2026-11-06 search baselines without an LLM: the method, families, sampler catalogue, seed key and (bo)
    # model settings are frozen into the snapshot; the generator record names the sampler instead of a model.
    search_method = str(getattr(args, "search_method", "llm") or "llm")
    search_baseline = None
    required_model = None
    if search_method != "llm":
        import re as _re

        from .baselines.sampler import SAMPLER_ID, catalog_digest
        from .baselines.selection import SELECTION_IDS

        expected_shape = (BASELINE_PROPOSALS_PER_GENERATION, BASELINE_GENERATIONS_PER_TURN, 1)
        shape = (int(args.proposal_count), int(args.generations_per_turn), int(args.rng_seeds_per_program))
        if str(args.provider) != BASELINE_PROVIDER:
            raise SystemExit(f"--search-method {search_method} requires --provider {BASELINE_PROVIDER}")
        if not bool(args.strict_turns):
            raise SystemExit(f"--search-method {search_method} runs strict turns (--strict-turns)")
        if ablation != "full":
            raise SystemExit(f"--search-method {search_method} cannot be combined with --ablation {ablation}")
        if structural_profile.exploit_generations or structural_profile.endpoint_demotion_strikes:
            # The v3.8 variant generations admit only the parents' families in the fast loop, which would discard
            # the fixed prior's draws of the other families; the baselines take only the slot counts from the
            # structural profile (v3.7 memory: 6 per turn; the CNOT profile: 7 per turn).
            raise SystemExit(
                f"--search-method {search_method} runs the v3.7 structural profile (got {structural_profile.profile_id})"
            )
        if shape != expected_shape:
            raise SystemExit(
                f"--search-method {search_method} runs --proposal-count {expected_shape[0]} --generations-per-turn "
                f"{expected_shape[1]} --rng-seeds-per-program 1 (got {shape})"
            )
        seed_id = str(getattr(args, "search_seed_id", "r01") or "r01")
        if not _re.fullmatch(r"r[0-9]{2}", seed_id):
            raise SystemExit(f"--search-seed-id must look like r01 (got {seed_id!r})")
        families = [canonical_code_family(item).value for item in spec.families]
        search_baseline = {
            "schema": "search_baseline_v1",
            "method": search_method,
            "families": families,
            "sampler_id": SAMPLER_ID,
            "sampler_catalog_sha256": catalog_digest(families, int(args.budget)),
            "proposals_per_generation": BASELINE_PROPOSALS_PER_GENERATION,
            "generations_per_turn": BASELINE_GENERATIONS_PER_TURN,
            "seed_id": seed_id,
            "seed_key": f"{SAMPLER_ID}|{task}|N{int(args.budget)}|{'+'.join(families)}|{seed_id}",
            "selection_id": SELECTION_IDS[search_method],
            # Descriptive: no baseline-specific shortlist; the campaign's common final stage (snapshot key
            # final_stage) takes the five leading distinct codes by search-stage pLER, as for the LLM searches.
            "final_selection": "common_final_stage_top5",
        }
        if search_method == "bo":
            from .baselines.bo import BO_CONFIG, BO_ID

            search_baseline["init_random_turns"] = BASELINE_INIT_RANDOM_TURNS
            search_baseline["bo"] = {"bo_id": BO_ID, **dict(BO_CONFIG)}
        required_model = SAMPLER_ID
        print(
            f"[campaign] PROTOCOL VARIANT: search baseline {search_method} (no LLM; families "
            f"{'+'.join(families)}; seed {seed_id}; slow queue {SELECTION_IDS[search_method]}; final stage = "
            "the common one, top 5 by search-stage pLER)",
            flush=True,
        )
    turns = getattr(args, "turns", None)
    if turns is None:
        turns = BASELINE_TURNS if search_method != "llm" else 3
    summary = run_campaign(
        spec=spec,
        out_dir=args.out,
        policy=make_policy(
            str(args.provider),
            ablation=ablation,
            generations=int(args.generations_per_turn),
            rng_seeds_per_program=int(args.rng_seeds_per_program),
            allow_fallback=not bool(args.strict_turns),
            fixed_program_count=bool(args.strict_turns) or bool(getattr(args, "fixed_program_count", False)),
        ),
        verification=verification,
        racing=racing,
        fixed_program_count=bool(args.strict_turns) or bool(getattr(args, "fixed_program_count", False)),
        max_turns=int(turns),
        proposal_count=int(args.proposal_count),
        generations_per_turn=int(args.generations_per_turn),
        slow_workers=int(args.slow_workers),
        shot_shards=shot_shards,
        final_shot_shards=final_shot_shards,
        feedback_mode=feedback_mode,
        resume=bool(args.resume),
        final_shots=int(final_shots),
        strict_turns=bool(args.strict_turns),
        required_provider=str(args.provider),
        required_model=required_model or resolve_model_name(str(args.provider), llm_config),
        rng_seeds_per_program=int(args.rng_seeds_per_program),
        reference_codes=reference_codes,
        final_decode_shard_timeout_cap_s=getattr(args, "final_shard_timeout_cap_s", None),
        search_decode_shard_timeout_cap_s=getattr(args, "search_shard_timeout_cap_s", None),
        promotion_max_check_weight=getattr(args, "promotion_max_check_weight", None),
        campaign_id=getattr(args, "campaign_id", None),
        ablation=ablation,
        search_baseline=search_baseline,
        final_cumulative_shots=getattr(args, "final_cumulative_shots", None),
    )
    stop = summary.get("stop") or {}
    print(f"[campaign] {summary['run_id']} stopped: {stop.get('reason')}")
    print(f"[campaign] report: {(summary.get('report') or {}).get('path')}")
    return 0


def _replace_promote_top(config, promote_top: int):
    return _replace_verification(config, promote_top=max(0, int(promote_top)))


def _replace_exact_circuit_distance(
    config,
    *,
    enabled: bool,
    dem_isd_enabled: bool | None = None,
    dem_isd_max_mechanisms: int | None = None,
    dem_isd_iters: int | None = None,
):
    exact = config.verification.exact_distance
    overrides = {"circuit_stim": bool(enabled)}
    if dem_isd_enabled is not None:
        overrides["circuit_dem_isd"] = bool(dem_isd_enabled)
    if dem_isd_max_mechanisms is not None:
        overrides["dem_isd_max_mechanisms"] = int(dem_isd_max_mechanisms)
    if dem_isd_iters is not None:
        overrides["dem_isd_iters"] = int(dem_isd_iters)
    return _replace_verification(
        config,
        exact_distance=dataclasses.replace(exact, **overrides),
    )


def _replace_proposal_count(config, proposal_count: int):
    return dataclasses.replace(config, proposal_count=int(proposal_count))


if __name__ == "__main__":
    raise SystemExit(main())
