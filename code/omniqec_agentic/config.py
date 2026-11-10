"""Configuration contracts and environment loading for omniqec_agentic."""

from __future__ import annotations

import dataclasses

import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .verification.noise import NoiseSweepConfig


def _default_noise_sweep() -> "NoiseSweepConfig":
    # Deferred import avoids config -> verification.__init__ -> schemas ->
    # config during initial package import.
    from .verification.noise import NoiseSweepConfig

    return NoiseSweepConfig.uniform()


class HardwareTarget(str, Enum):
    ATOM = "atom"
    ION = "ion"


# Prompt-facing hardware introduction. Scope is deliberately narrow: platform
# connectivity statements ONLY. Check weight is intentionally never mentioned
# (project decision): the platforms impose no weight limit, the model chooses
# weights freely, and the circuit-level LER measurement naturally prices
# heavy checks (deeper extraction, more fault locations) — the feedback loop
# is the regulator, not prompt text. No scoring/feedback content here either;
# the verification protocol is unified across platforms.
HARDWARE_PROMPT_PROFILES: dict[str, tuple[str, ...]] = {
    "atom": (
        "Hardware target: atom (neutral-atom array). All-to-all connectivity: "
        "any check may couple any subset of qubits, at any distance.",
    ),
    "ion": (
        "Hardware target: ion (trapped ion). All-to-all connectivity via "
        "shared motional modes: any check may couple any subset of qubits.",
    ),
}


class DecoderName(str, Enum):
    BEAM = "beam"
    LSD = "lsd"
    BPOSD = "bposd"
    RELAYBP = "relaybp"
    # telescoping_structural_v3_6 (2026-09-03, docs/06 §1): the Mitten paper's
    # open-source telescoping decoder (arXiv:2607.28795, github.com/a7b/yarn,
    # vendored at third_party/telescoping-decoder) in its CPU configuration
    # on the GARI transform of the two-basis DEM.
    TELESCOPING = "telescoping_cpu"


class CodeFamily(str, Enum):
    """The four canonical Fig.2 search grammars (docs/01_code_family_design.md)."""

    BB = "bb"
    HGP = "hgp"
    LP_ABELIAN = "lp_abelian"
    LP_NONABELIAN = "lp_nonabelian"


class DeformationMode(str, Enum):
    """Run-level, deterministic local-Clifford deformation (off by default)."""

    NONE = "none"
    SECTOR_H = "sector_h"


DEFAULT_FAMILIES: tuple[CodeFamily, ...] = tuple(CodeFamily)

# Catalog cap for LP group orders.
#
# v3.6 (docs/06 §3 M3): the binding bound is BUDGET-DERIVED —
# ``group_order_cap_for_budget(n_budget)`` = n_budget // 4, the largest |G| a
# 1x1/1x1 lifted product can spend on that total-qubit budget (b288 -> 72,
# b360 -> 90). The constant below is only the absolute resource guard that
# stops an absurd name from costing seconds of Cayley-table construction; it
# must never be the reason a legal-on-budget group is rejected.
MAX_GROUP_ORDER = 200


def group_order_cap_for_budget(n_budget: int | None) -> int:
    """Largest LP group order admissible on a total-qubit budget (1x1 shapes
    cost 4|G| qubits). ``None`` (no budget in scope) -> the resource guard."""

    if n_budget is None:
        return int(MAX_GROUP_ORDER)
    return max(1, min(int(MAX_GROUP_ORDER), int(n_budget) // 4))


def canonical_code_family(value: str | CodeFamily) -> CodeFamily:
    return value if isinstance(value, CodeFamily) else CodeFamily(str(value))


@dataclass(frozen=True)
class LpShapePin:
    """Run-level seed-shape pin for both LP families (design doc §3.5).

    ``None`` leaves that base matrix's shape to the LLM (within the allowed
    pool); a pair like ``(1, 2)`` pins it. Pinning both matrices locks the
    exact structure grid (e.g. the Mitten 1x2/1x2 form). Part of the
    measurement protocol: recorded in run config and artifacts.
    """

    a: tuple[int, int] | None = None
    b: tuple[int, int] | None = None

    @property
    def is_trivial(self) -> bool:
        return self.a is None and self.b is None

    @staticmethod
    def parse_shape(text: str | None) -> tuple[int, int] | None:
        if not text:
            return None
        rows, _, cols = str(text).lower().partition("x")
        return (int(rows), int(cols))

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": list(self.a) if self.a else None,
            "b": list(self.b) if self.b else None,
        }


@dataclass(frozen=True)
class BeamDecoderConfig:
    """Beam search decoder — the default (design doc 02 §11.1, D6).

    Paper-faithful beam64 hyperparameters (Ye-Wecker-Delfosse,
    arXiv:2512.07057, Table 1): the repo constructor defaults are the beam8
    tuning and MUST NOT be used with beam_width=64. num_results=32 is the 17x
    list-decoding config; its runtime is unreported — kept a switch, default 1
    until qualification measures throughput (02 §11.1).
    Implementation vendored at third_party/BeamSearchDecoder (CC BY-NC-SA
    4.0, IonQ; commit pinned in VENDORED_COMMIT).
    """

    beam_width: int = 64
    max_rounds: int = 20
    initial_iters: int = 40
    iters_per_round: int = 30
    num_results: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LsdDecoderConfig:
    """BP+LSD — the decoder ver1's d_circ_ler protocol was calibrated on.

    Paper settings (Hillmann et al., Nat Commun 16, 8214 (2025), Methods 4.3):
    min-sum, ~30 iterations, alpha=0.625, parallel schedule; LSD-0 is the
    paper's headline config and higher order gives small extra accuracy.
    NOTE (ldpc 2.4.1 footgun): without an explicit ``lsd_method``,
    BpLsdDecoder silently resets lsd_order to 0 — runs labeled "order 8"
    before 2026-08-15 were actually BP+LSD-0 with ms_scaling_factor=1.0.
    """

    max_iter: int = 50
    lsd_order: int = 8
    bp_method: str = "ms"
    # Mandatory for lsd_order > 0 to take effect (see docstring).
    lsd_method: str = "lsd_cs"
    # Paper's alpha; BpLsdDecoder's constructor default 1.0 is NOT the paper setting.
    ms_scaling_factor: float = 0.625

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BposdDecoderConfig:
    """The single frozen decoder of the ``bposd_structural_v3_5`` profile.

    ``preset`` is a compatibility label only: ldpc 2.4.1's BpOsdDecoder takes
    no such argument, so the adapter passes the four effective fields below
    and nothing else. Changing any of them changes the measurement protocol
    and therefore needs a new profile id, not a flag.
    """

    preset: str = "2d"
    max_iter: int = 50
    osd_order: int = 10
    bp_method: str = "product_sum"
    osd_method: str = "osd_cs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelayBpDecoderConfig:
    preset: str = "2d_reduced"
    alpha: float = 1.0
    gamma0: float = 0.15
    gamma_dist_interval: tuple[float, float] = (-0.226, 0.622)
    num_sets: int = 60
    pre_iter: int = 12
    set_max_iter: int = 12
    stop_nconv: int = 1
    # Sequential by definition of the protocol: relay-bp's internal Rust
    # threads consume RNG in scheduling-dependent order, so parallel=True and
    # parallel=False are different (each reproducible) random streams. The
    # slow job pool owns the parallelism; keeping this False makes results
    # identical across worker counts and machines.
    parallel: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gamma_dist_interval"] = list(self.gamma_dist_interval)
        return data


# The vendored telescoping-decoder package (third_party/telescoping-decoder;
# github.com/a7b/yarn, MIT, VENDORED_COMMIT). The commit is protocol identity:
# the S3 variant tables, the seed derivation and the C kernels live there.
TELESCOPING_PACKAGE = "telescoping-decoder"
TELESCOPING_VENDORED_COMMIT = "c97099b759dcb53520d0ce210c6893dd9e0e0ea3"
# Package-default S3 variant tables (DEFAULT_S3A/B/C_VARIANTS): 5 S3-A
# variants, 6 S3-B prior variants, 12 S3-C configurations of which 11 carry
# prior noise (c_n_seeds_default seeds each) and one is noise-free (1 seed).
# Their worst-case BP iteration count at the package defaults, used by the
# decode watchdog model when the package cannot be imported.
TELESCOPING_S3_WORST_CASE_ITERATIONS_DEFAULT = 104_000


@dataclass(frozen=True)
class TelescopingDecoderConfig:
    """The single frozen decoder of the ``telescoping_structural_v3_6``
    profile (docs/06 §1, user decision 2026-09-03): the Mitten paper's
    open-source telescoping decoder (arXiv:2607.28795 App. I.1; package
    ``telescoping_decoder``, vendored at third_party/telescoping-decoder) in
    its CPU configuration on the GARI transform of the TWO-BASIS DEM:

      S1 / S2   GPU stages (flooding BP, GPU relay-BP)     OFF (no CUDA)
      S3-A/B/C  CPU relay stages on the GARI system         ON, package
                defaults: variant tables DEFAULT_S3A/B/C_VARIANTS, per-shot
                seeds derived from (base_seed, shot id) -- pool-order
                independent, so a decode is a pure function of the
                (matrices, shard seed, shot index)
      S4        Gurobi integer-programming certification    OFF (license;
                CPU-only protocol)
      NC        a shot no stage accepts COUNTS AS A LOGICAL ERROR (the
                paper's rule for undecoded shots; identical for every code,
                an upper bound on the LER)

    Every field is protocol identity: changing one needs a new profile id.
    ``base_seed`` is not a field -- it is the shard seed. ``n_procs=1``: the
    slow job pool owns the parallelism (one shard per core) and the adapter
    runs the package's S3 per-shot functions in the shard process itself (no
    child process); ``n_procs > 1`` hands the batch to the package's own
    process pool (local benchmarking only; identical results).
    """

    preset: str = "telescoping_cpu_v1"
    package: str = f"{TELESCOPING_PACKAGE}@{TELESCOPING_VENDORED_COMMIT[:12]}"
    s1_enabled: bool = False
    s2_enabled: bool = False
    s3_system: str = "gari"
    s4_enabled: bool = False
    nc_policy: str = "fail"
    # S3-A: standalone variant ensemble (package table, 5 variants).
    a_iters: int = 100
    a_alpha_max: float = 0.99
    a_tau: float = 5.0
    # S3-B: relay-BP with memory + coset quorum (package table, 6 priors).
    b_runs: int = 40
    b_iters: int = 100
    b_alpha_max: float = 0.90
    b_tau: float = 5.0
    b_min_cost_select: bool = False
    b_use_memory: bool = True
    b_gamma0: float = 0.1
    b_gamma0_iters: int = 80
    b_gamma_center: float = 0.21
    b_gamma_width: float = 0.90
    b_stop_nconv: int = 5
    b_quorum: int = 3
    # S3-C: sequential parameter sweep (package table, 12 configurations).
    c_tau: float = 5.0
    c_n_seeds_default: int = 10
    # The S3 variant tables are the package's shipped ones (never edited).
    variant_tables: str = "package_default"
    n_procs: int = 1

    def __post_init__(self) -> None:
        if bool(self.s1_enabled) or bool(self.s2_enabled):
            raise ValueError("telescoping_cpu: the GPU stages S1/S2 are not part of the CPU protocol")
        if bool(self.s4_enabled):
            raise ValueError("telescoping_cpu: the Gurobi S4 stage is not part of the CPU protocol")
        if str(self.s3_system) not in ("gari", "original"):
            raise ValueError("telescoping_cpu: s3_system must be 'gari' or 'original'")
        if str(self.nc_policy) != "fail":
            raise ValueError("telescoping_cpu: nc_policy must be 'fail' (undecoded shot = logical error)")
        if str(self.variant_tables) != "package_default":
            raise ValueError("telescoping_cpu: variant_tables must be 'package_default'")
        if int(self.n_procs) < 1:
            raise ValueError("telescoping_cpu: n_procs must be >= 1")
        if int(self.b_quorum) < 1 or int(self.b_stop_nconv) < 1:
            raise ValueError("telescoping_cpu: b_quorum and b_stop_nconv must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TelescopingDecoderConfig":
        raw = dict(data or {})
        if not raw:
            return cls()
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            if f.type == "bool":
                kwargs[f.name] = bool(value)
            elif f.type == "int":
                kwargs[f.name] = int(value)
            elif f.type == "float":
                kwargs[f.name] = float(value)
            else:
                kwargs[f.name] = str(value)
        return cls(**kwargs)

    def to_telescope_config(self, *, base_seed: int):
        """The package's ``TelescopeConfig`` for this protocol at one shard
        seed (deferred import: the package is an optional dependency)."""

        from telescoping_decoder import TelescopeConfig

        cfg = TelescopeConfig(base_seed=int(base_seed), use_gari=str(self.s3_system) == "gari")
        cfg.s1.enabled = bool(self.s1_enabled)
        cfg.s2.enabled = bool(self.s2_enabled)
        s3 = cfg.s3
        s3.enabled = True
        s3.system = str(self.s3_system)
        s3.n_procs = int(self.n_procs)
        # n_procs == 1 runs S3 in the shard process and never creates a pool,
        # but TelescopingDecoder.prepare() still pre-spawns the package's
        # default "forkserver" context: one forkserver + one resource_tracker
        # helper process per worker. Under an init that never reaps (a training
        # Pod whose PID 1 is `sleep`) both stayed behind as zombies for every
        # worker job until the container's pids.max (30000) was hit and the
        # next fork failed with EAGAIN (b60, 2026-09-06). "fork" spawns
        # nothing up front. Not a protocol field: to_dict() / the snapshot
        # are unchanged and the in-process S3 per-shot functions never see it.
        if int(self.n_procs) == 1:
            s3.mp_start_method = "fork"
        s3.a_iters = int(self.a_iters)
        s3.a_alpha_max = float(self.a_alpha_max)
        s3.a_tau = float(self.a_tau)
        s3.b_runs = int(self.b_runs)
        s3.b_iters = int(self.b_iters)
        s3.b_alpha_max = float(self.b_alpha_max)
        s3.b_tau = float(self.b_tau)
        s3.b_min_cost_select = bool(self.b_min_cost_select)
        s3.b_use_memory = bool(self.b_use_memory)
        s3.b_gamma0 = float(self.b_gamma0)
        s3.b_gamma0_iters = int(self.b_gamma0_iters)
        s3.b_gamma_center = float(self.b_gamma_center)
        s3.b_gamma_width = float(self.b_gamma_width)
        s3.b_stop_nconv = int(self.b_stop_nconv)
        s3.b_quorum = int(self.b_quorum)
        s3.c_tau = float(self.c_tau)
        s3.c_n_seeds_default = int(self.c_n_seeds_default)
        cfg.s4.enabled = bool(self.s4_enabled)
        return cfg

    def worst_case_iterations(self) -> int:
        """BP iterations of a shot that exhausts every enabled S3 stage
        (the decode watchdog's cost model; docs/06 §1.4 / B5)."""

        try:
            from telescoping_decoder.config import (
                DEFAULT_S3A_VARIANTS,
                DEFAULT_S3B_VARIANTS,
                DEFAULT_S3C_VARIANTS,
            )
        except Exception:
            return int(TELESCOPING_S3_WORST_CASE_ITERATIONS_DEFAULT)
        total = len(DEFAULT_S3A_VARIANTS) * int(self.a_iters)
        total += len(DEFAULT_S3B_VARIANTS) * int(self.b_runs) * int(self.b_iters)
        for _label, _scale, iters, noise_std, _alpha_max, n_seeds in DEFAULT_S3C_VARIANTS:
            seeds = (
                int(n_seeds)
                if n_seeds is not None
                else (int(self.c_n_seeds_default) if float(noise_std) > 0.0 else 1)
            )
            total += int(iters) * seeds
        return int(total)


@dataclass(frozen=True)
class DecoderBuildConfig:
    beam: BeamDecoderConfig = field(default_factory=BeamDecoderConfig)
    lsd: LsdDecoderConfig = field(default_factory=LsdDecoderConfig)
    bposd: BposdDecoderConfig = field(default_factory=BposdDecoderConfig)
    relaybp: RelayBpDecoderConfig = field(default_factory=RelayBpDecoderConfig)
    telescoping: TelescopingDecoderConfig = field(default_factory=TelescopingDecoderConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam": self.beam.to_dict(),
            "lsd": self.lsd.to_dict(),
            "bposd": self.bposd.to_dict(),
            "relaybp": self.relaybp.to_dict(),
            "telescoping": self.telescoping.to_dict(),
        }


@dataclass(frozen=True)
class ExactDistanceConfig:
    enabled: bool = True
    code_milp: bool = True
    circuit_stim: bool = False
    circuit_dem_isd: bool = False
    max_n: int = 100
    max_d_proxy: int = 8
    code_timeout_s: float = 5.0
    circuit_cap: int = 8
    # 8000 keeps the ISD degradation proof weight-neutral at n<=72: measured
    # DEM sizes are 1584 mechanisms for BB72 (w=6, ~25s/candidate at 120
    # iters) and 2664 for a w=8 d=9 code; w=10-12 high-distance combinations
    # approach 4000, so the old 4000 cap would have silently skipped exactly
    # the heavy codes the model is now free to propose. Heavier DEMs cost
    # proportionally more ISD time, bounded by the per-job timeout.
    dem_isd_max_mechanisms: int = 8000
    dem_isd_iters: int = 120
    dem_isd_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The four effective BPOSD fields of the HISTORICAL bposd_structural_v3_5
# profile (kept for the validator's history mode and the dev-only decoders).
BPOSD_PROTOCOL_FIELDS: dict[str, Any] = {
    "bp_method": "product_sum",
    "max_iter": 50,
    "osd_method": "osd_cs",
    "osd_order": 10,
}

# The effective decoder fields of the formal telescoping_structural_v3_6
# profile: EVERY field of TelescopingDecoderConfig at its default (no field
# depends on the budget point; the shots do, see budget_policy). The strict
# validator checks every recorded decoder-build blob against this table.
TELESCOPING_PROTOCOL_FIELDS: dict[str, Any] = TelescopingDecoderConfig().to_dict()

# Budget-point policy (docs/06 §1.4, user decision 2026-09-03): every code at
# one budget point gets the same shots (fairness), but large points relax
# them because a b360 Mitten-class DEM costs ~10-20 CPU-s/shot under the
# telescoping decoder (measured 2026-09-03, docs/05 §9).
# 2026-09-14 (user decision, tier 3 launch): every Fig.2 point up to 360 uses the
# same 25k/25k shots -- at n >= 300 a good code has ~1e-3 block LER, so 5k
# search shots would leave the search-stage ranking with a handful of errors.
# The large-budget relaxation is kept for points above 400 only.
LARGE_BUDGET_THRESHOLD = 400
LARGE_BUDGET_SEARCH_SHOTS = 5_000
LARGE_BUDGET_FINAL_SHOTS = 50_000
# The decode-shard watchdog scales with DEM size: budget_s/shot x
# max(1, num_matrix_columns / DECODE_BUDGET_SCALE_COLS).
DECODE_BUDGET_SCALE_COLS = 60_000
# Telescoping worst-case shot model (docs/06 §1.4 / B5): the S3 stages are
# ITERATION bounded (deterministic), so the worst case of one shot is a cost
# model on the GARI system size, not a flat number only small codes meet:
#   t(shot) = TELESCOPING_COST_S_PER_COLUMN_ITER x (num_matrix_columns x
#             TELESCOPING_GARI_COLUMN_FACTOR) x worst_case_iterations()
# num_matrix_columns is the two-basis DEM column count reported by the build
# job; the GARI matrix is ~1.25x wider (BB144 83,880 / 67,104; Mitten200
# 153,800 / 123,960). Measured on an unsatisfiable syndrome (every stage
# exhausted, n_procs=1, Apple M-series, 2026-09-03): BB144 51 s/shot ->
# 5.9e-9 s per GARI-column-iteration, Mitten200 106 s/shot -> 6.6e-9; the
# constant carries a ~3x margin for slower cluster cores. The watchdog is
# max(v3.4 term, shots x worst-case shot).
TELESCOPING_COST_S_PER_COLUMN_ITER = 2.0e-8
TELESCOPING_GARI_COLUMN_FACTOR = 1.3
# SEC certification work model for the build watchdog (review fix, docs/06
# §3 M1): the staged certification costs ~1.5 ms per ISD iteration at n=72
# (kernel 222) and ~27.5 ms at n=200 (kernel 760), i.e. ~ (n/72)^2.8, over
# 2 x (K x seeds x max(iters, scale x kernel) + refine_top x seeds x
# refine_iters) iterations with kernel ~ 4 n. SEC_CERT_ISD_COST_S_AT_N72
# carries a 2x margin; the build watchdog is max(build_timeout_s, model).
SEC_CERT_ISD_COST_S_AT_N72 = 3.0e-3
SEC_CERT_ISD_COST_EXPONENT = 2.8
SEC_CERT_KERNEL_PER_DATA_QUBIT = 4.0
SEC_CERT_BUILD_OVERHEAD_S = 600.0


# 2026-09-10 (user decision): the small-budget tier measures 25,000 shots per
# basis = 50,000 per candidate in BOTH stages. At p=0.5% every promoted code
# has hundreds of failures per basis at this depth (CI half-width ~10%), so
# the former 10x final stage (200k) bought nothing the ranking could use; the
# final stage keeps its own RNG namespace and re-measures the survivors at the
# same 50k. The large-budget tier (>= 300) is unchanged.
SMALL_BUDGET_SEARCH_SHOTS = 25_000
SMALL_BUDGET_FINAL_SHOTS = 25_000
# 2026-09-10 rounds policy: T = max(T_c, d_search) per candidate, identical in
# the search and final stages (config id checked by the validator; the T_c
# floor stays 12 = DEFAULT_SEARCH_ROUNDS / StructuralSearchConfig.final_rounds).
ROUNDS_POLICY_ID = "max_Tc_d"
# 2026-10-15 (Fig.3 formal run, user decision): ``fixed_Tc`` measures EVERY
# candidate at the floor T_c regardless of d_search -- the gate-dense task is a
# fixed-depth circuit (one transversal CNOT per round, no logical measurement, so
# the d-round requirement of surgery-type instructions does not apply; Cain et
# al. 2024 / Zhou et al. 2024 use fixed depth). The formal Fig.2 policy stays
# max_Tc_d; the validator accepts only that one for formal runs.
ROUNDS_POLICY_IDS: tuple[str, ...] = ("max_Tc_d", "fixed_Tc")
# 2026-09-11 (user decision): EARLY STOP in the slow loop. Every promoted
# candidate decodes a pilot of RACING_PILOT_SHOTS per basis in RACING_BATCH
# shot shards; a candidate whose plpr CI lower bound is already >= 10x the
# hall-of-fame best's CI upper bound (epsilon = 9.0 -> ub * 10) stops there
# with stopped_by="racing" (never enters the frontier); everything else
# marches to the point's full search shots (shots_confirm = shots_max =
# budget_policy search_shots). The BB baseline arm is OFF: the reference is
# the current best code only. Baseline candidates are never raced.
# formal_racing() below is the frozen parameter set; the validator requires
# the snapshot's racing block to equal it exactly.
# 2026-09-11 evening (user decision): the search stage stops a candidate once
# EVERY basis has RACING_MAX_ERRORS failures ("a few dozen errors is enough to
# screen; only the final stage needs 25k/25k") -- policy pilot10x_maxerr100_v2.
# pilot10x_v1 (no max_errors cap) stays a formal policy: the tier-1 runs
# started under it and switch at a turn boundary on resume (the switch is
# recorded in protocol_snapshot.json runtime_overrides; the validator checks
# every turn against the policy that turn actually ran).
# 2026-09-14 (pilot10x_maxerr100_v3): the max_errors cap is checked at EVERY
# landed shard boundary, including inside the pilot. Under v2 the first
# decision waited for the whole 2000-shot pilot, so a code that fails every
# shot still decoded 2000 x 2 shots (a w>=12 code: ~2 h of a 128-core node
# for a plpr it had proven by shot 250). The 10x arm is unchanged (pilot
# first); rows that do not reach the cap inside the pilot are measured
# exactly as under v2, so every surviving code's numbers are identical.
RACING_POLICY_V1_ID = "pilot10x_v1"
RACING_POLICY_V2_ID = "pilot10x_maxerr100_v2"
RACING_POLICY_V3_ID = "pilot10x_maxerr100_v3"
RACING_POLICY_V4_ID = "pilot10x_maxerr100_cost_v4"
# 2026-09-15 morning: the formal policy is v3 again. v4's decoder-cost cap
# (below) was judged from a candidate's first 500 shots, which under the
# persistent worker pool are dominated by every worker's one-time decoder
# warm-up, and its 0.10 s/shot/qubit calibration was 2x too tight for tier 3
# (BB n~176, d~20 codes cost ~14 CPU-s/shot in steady state): it cut 16
# candidates overnight, several of them legitimate high-distance BB codes.
# v4 stays registered so the turns that ran under it validate; no new turn
# runs under it.
RACING_POLICY_ID = RACING_POLICY_V3_ID
# v4 decoder-cost cap (CPU-seconds per shot per data qubit; floor per shot; shots per basis before judging)
RACING_COST_CAP_S_PER_SHOT_PER_QUBIT = 0.10
RACING_COST_CAP_FLOOR_S_PER_SHOT = 8.0
RACING_COST_CAP_MIN_SHOTS = 250
RACING_PILOT_SHOTS = 2_000
RACING_BATCH_SHOTS = 250
RACING_EPSILON = 9.0
RACING_MAX_ERRORS = 100
# 2026-09-29 (user decision, the >= 288 points only): pilot10x_maxerr30_v3 is
# v3 with a 30-error cap per basis. It is a declared VARIANT of the frozen
# policy for the large-code tier (recorded by policy_id in every turn's
# slow_run_config.json and in the snapshot's racing block); the <= 260 points
# stay on pilot10x_maxerr100_v3. Registered as a formal policy so a resumed
# chain may switch to it at a turn boundary (checkpoint._formal_racing_switch)
# and the validator accepts the max_errors stops it produces.
RACING_POLICY_V3_MAXERR30_ID = "pilot10x_maxerr30_v3"
RACING_MAX_ERRORS_LARGE = 30
# 2026-11-10 (paper alignment): the search-stage early stop exactly as the paper's Appendix D3 states it,
# and the open-source default (run-campaign without --racing-max-errors; the fig2_final / fig3_final
# presets). 2 000 shots per basis first, then 250-shot batches; after the 2 000 and after every batch
# (and never inside the first 2 000) the candidate stops when its Wilson lower pLER bound EXCEEDS ten
# times the upper bound of the current best measured code AND ten times the upper bound of the run's
# applicable reference code (each comparison only where that code has been measured), or when both bases
# hold 30 failures, or at 25 000 shots per basis. See docs/paper_alignment_measurement.md.
RACING_POLICY_PAPER_ID = "paper_d3_wilson10x_ref_maxerr30_v1"
RACING_MAX_ERRORS_PAPER = 30
RACING_BEST_BOUND_RULES: tuple[str, ...] = ("min_ub", "best_estimate")


# 2026-11-10 (paper alignment): the final confirmation of every discovery run (paper Appendix D3, Table VII;
# evaluation/final_stage.py builds and executes it, the protocol snapshot freezes it under key `final_stage`).
# After the last turn the successfully measured non-reference candidates are ranked by their search-stage
# pLER and the FINAL_STAGE_TOP leading distinct codes are re-measured with fresh shots: an initial
# confirmation of the budget policy's final shots (25 000 per basis), then further shots up to the
# cumulative per-basis limit of Table VII for the run's task and budget N; the parts are pooled per basis
# (shots and failures added) and the pooled pLER ranks the finalists and names the champion. The applicable
# BB reference code gets the same initial confirmation, then a top-up to REFERENCE_CODE_SHOTS_PER_BASIS.
FINAL_STAGE_PAPER_ID = "paper_top5_cumulative_v1"
FINAL_STAGE_TOP = 5
# 2026-11-10 (paper revision, Appendix D "Additional experiments", Fig. 4 c): "Six candidates are evaluated in the
# final stage for each condition. The figure reports the mean LER and the minimum-maximum range of the three best
# codes under each weight constraint." A run under an exact check-weight contract (exact_check_weight) has its own
# final stage (evaluation/final_stage.py, selected by the contract in workflow.state.build_protocol_snapshot): the
# FINAL_STAGE_FIXED_WEIGHT_TOP leading distinct codes by search-stage pLER, each measured once with the initial
# confirmation (25 000 shots per basis) and no cumulative top-up, ranked by that measurement; the run's ranking
# output carries the FINAL_STAGE_FIXED_WEIGHT_REPORTED best (their pLER, arithmetic mean, minimum and maximum).
FINAL_STAGE_FIXED_WEIGHT_ID = "paper_fig4c_top6_confirm_v1"
FINAL_STAGE_FIXED_WEIGHT_TOP = 6
FINAL_STAGE_FIXED_WEIGHT_REPORTED = 3
FINAL_STAGE_IDS: tuple[str, ...] = (FINAL_STAGE_PAPER_ID, FINAL_STAGE_FIXED_WEIGHT_ID)
# 2026-11-10 (paper revision, Appendix D3): "In both the quantum-memory and transversal-CNOT experiments, the
# reported LERs for the BB [[72,12,6]] and [[144,12,12]] reference codes are based on 50,000 and 100,000 shots per
# basis, respectively." Per-basis total of the reported reference measurement (the 25 000-shot initial
# confirmation included), keyed by the reference's program id (evaluation.baseline.RESIDENT_BASELINE_BUDGETS; the
# tests pin the two key sets together). Replaces the Table VII limits of the references' budgets (175 000 /
# 525 000 memory, 75 000 / 325 000 CNOT) that the earlier paper text gave.
REFERENCE_CODE_SHOTS_PER_BASIS: dict[str, int] = {
    "baseline_bb_72_12_6": 50_000,
    "baseline_bb_144_12_12": 100_000,
}
# Table VII (per measurement basis, the 25 000-shot initial confirmation included). Mirrored by
# hpc/top5_extension_caps.json; the tests pin the two together.
FINAL_CUMULATIVE_SHOT_CAPS: dict[str, dict[int, int]] = {
    "memory": {
        60: 175_000, 98: 175_000, 144: 175_000, 180: 175_000, 220: 275_000,
        260: 525_000, 288: 525_000, 320: 875_000, 360: 1_725_000,
    },
    "cnot_dense": {
        60: 75_000, 98: 75_000, 144: 75_000, 180: 75_000, 220: 125_000,
        260: 175_000, 288: 325_000, 320: 525_000, 360: 875_000,
    },
}
# Fresh-seed namespace of the top-up parts (the initial confirmation keeps rounds_policy.final_rng_namespace).
FINAL_TOPUP_RNG_NAMESPACE = "final_topup_v1"
# Execution only (not in the paper): a top-up larger than this many shots per basis is measured in several
# parts, each with its own seed namespace, so a walltime-limited link can resume part by part; shards per
# part follow the extended-statistics rule max(final_shot_shards, 2^ceil(log2(shots / 300))).
FINAL_TOPUP_PART_MAX_SHOTS: dict[str, int] = {"memory": 900_000, "cnot_dense": 150_000}
FINAL_TOPUP_SHOTS_PER_SHARD = 300


def final_cumulative_cap(task: str, n_budget: int | None) -> int | None:
    """Table VII's cumulative per-basis shot limit for (task, N), or None for a budget the table does not list."""

    table = FINAL_CUMULATIVE_SHOT_CAPS.get(str(task or "memory"))
    if table is None:
        raise ValueError(f"no Table VII column for task {task!r} (known: {sorted(FINAL_CUMULATIVE_SHOT_CAPS)})")
    if n_budget is None:
        return None
    return table.get(int(n_budget))


def budget_policy(n_budget: int | None) -> dict[str, Any]:
    """Search / final shots per basis for a total-qubit budget point."""

    large = n_budget is not None and int(n_budget) >= LARGE_BUDGET_THRESHOLD
    return {
        "large_budget": bool(large),
        "search_shots": LARGE_BUDGET_SEARCH_SHOTS if large else SMALL_BUDGET_SEARCH_SHOTS,
        "final_shots": LARGE_BUDGET_FINAL_SHOTS if large else SMALL_BUDGET_FINAL_SHOTS,
    }


# Structural-retention profile ids. v3.7 is the formal Fig.2 protocol (the
# package default below); v3.8 = v3.7 + the two rules documented on the
# fields `exploit_generations` / `endpoint_*` (2026-10-17).
PROTOCOL_PROFILE_V37 = "telescoping_structural_v3_7"
PROTOCOL_PROFILE_V38 = "telescoping_structural_v3_8"
# The v3.8 rule set (every value is checked against the profile id).
V38_STRUCTURAL_RULES: dict[str, Any] = {
    "exploit_generations": 2,
    "endpoint_family_distinct": True,
    "endpoint_demotion_strikes": 3,
    "endpoint_demotion_factor": 10.0,
}
STRUCTURAL_PROFILE_IDS: tuple[str, ...] = (PROTOCOL_PROFILE_V37, PROTOCOL_PROFILE_V38)


@dataclass(frozen=True)
class StructuralSearchConfig:
    """Frozen structural-retention channel (bposd_structural_v3_5 §1-§4).

    The search's selection pressure toward absolutely large `k` OR absolutely
    large `d_search` lives HERE, in machine rules, and deliberately nowhere in
    the generator prompt: which candidates get slow slots, which programs are
    kept across turns, which become parents, which get final slots.
    """

    # v3.6 (2026-09-03, docs/06): decoder = telescoping_cpu_v1 (Mitten paper
    # telescoping decoder, CPU configuration, GARI system) on the
    # two-basis DEM, budget-point shot policy, per-coordinate reducer cap,
    # WL gate replaced by canonical hashing, multi-seed SEC certification.
    # v3.5 artifacts (BPOSD) are NOT poolable with v3.6 ones, hence the bump.
    # telescoping_structural_v3_7 (2026-09-12, user decision after the tier-1
    # audit): the structural front is non-dominated in (k, d_search,
    # -max_check_weight); the check-weight cap is gone; the per-turn slow
    # quota is 2 structural endpoints + 4 runtime slots ordered by the
    # measured-evidence plpr model (ranking.archive_order_key expected_plpr);
    # the final shortlist is 2 structural + 4 by measured plpr; the parent
    # window is 16 = 6 structural + 6 runtime + 4 diversity. v3.6 artifacts
    # were produced under a weight-blind (k, d_search) front and a Q_eff
    # runtime order and are NOT poolable with v3.7 ones, hence the bump.
    profile_id: str = "telescoping_structural_v3_7"
    # v2 (docs/06 §3 B4): at most `coordinate_cap` realizations per cell
    # (k, d_search, weight) in the reduced view; slot 3 and the parent window
    # iterate over UNIQUE cells. v3.7: reducer with the weight axis.
    reducer_id: str = "structural_proxy_pareto_v3_weight"
    coordinate_cap: int = 2
    # v3.7: dominance axes of the structural front (weight minimised) and the
    # k floor of the structural channel (k = 1 codes never take a slot).
    structural_axes: tuple[str, ...] = ("k", "d_search", "-max_check_weight")
    min_promotion_k: int = 2
    # d_search protocol (§3): identical seeds and effort for every family.
    proxy_seeds: tuple[int, ...] = (0, 1, 2, 3)
    proxy_iters_per_seed: int = 120
    proxy_algorithm_id: str = "css_permutation_systematic_v2_independent_xz"
    # Cross-turn (k, d_search, weight) proxy-Pareto cache (§1.3 / §4.2).
    # v3.7: 24 (was 16) -- a three-axis front holds more non-dominated cells.
    elites_limit: int = 24
    # Slow promotion quota (§4.3). v3.2-v3.6: 3 machine structural slots + 1
    # runtime slot; 2026-09-10: 4 structural + 2 runtime. v3.7 (2026-09-12,
    # user decision): 2 structural (high-k endpoint, high-d_search endpoint)
    # + 4 runtime slots (named promotions consume them; the rest fill from
    # the archive queue ordered by expected plpr under the measured-evidence
    # model, Q_eff only as the cold-start tie-break), i.e. 6 slow
    # measurements per turn, with the d_search >= 5 promotion floor
    # (ranking.MIN_PROMOTION_DISTANCE) and the k >= 2 structural floor.
    slow_candidate_cap: int = 6
    structural_slow_slots: int = 2
    runtime_slow_slots: int = 4
    # v3.7: the runtime channel's ordering rule (frozen id; the validator and
    # the acquisition tool name it).
    runtime_order: str = "expected_plpr_v1"
    # Next-turn parent quota (§4.5). v3.7: 16 = 6 structural (unique-cell
    # round-robin over the front) + 6 runtime (hall of fame by measured plpr,
    # at most 2 per check-weight class) + 4 diversity (regions of
    # family/shape/weight not yet represented), rendered S,R,S,R,... then D.
    parent_cap: int = 16
    structural_parents: int = 6
    runtime_parents: int = 6
    diversity_parents: int = 4
    # Fixed-T final audit (§4.6).
    # Final shortlist (§4.6): v3.7 2 structural endpoints + 4 by measured
    # plpr (was 4 + 2): the expensive 25k/25k confirmation goes to the codes
    # that measured best, the two endpoints keep the high-Q points certified.
    final_structural: int = 2
    final_runtime: int = 4
    final_rounds: int = 12
    final_rng_namespace: str = "final_confirmation_v1"
    # Fig.3 Phase B task-aware variant (2026-10-12, user decision; see
    # VerificationConfig.task and fig3_cnot_structural_config): a THIRD
    # structural slow slot takes the front's lightest cell (min
    # max_check_weight, ties toward higher k, then higher d_search), so one
    # light code is measured every turn even when both v3.7 endpoints are
    # heavy; the generator prompt names measured codes whose task error is at
    # least not_competitive_ratio x the campaign's best (with non-overlapping
    # confidence intervals) as not competitive. Both are off under the formal
    # Fig.2 profile (False / None).
    lightest_endpoint_slot: bool = False
    not_competitive_ratio: float | None = None
    not_competitive_max_rows: int = 12
    # Fig.3 formal run (2026-10-15, user decision): the generator is TOLD when a
    # proposal is a code this campaign already measured, or a relabelling
    # (pynauty-isomorphic twin) of one -- which measured code, at what task
    # error -- and sees what happened to the previous turn's slow queue (how
    # many niche winners were measured, twins, already measured, over the
    # check-weight gate). Under the Phase B v1 profile such a program was
    # reported as "recorded in its niche" and the twin gate rejected it
    # silently, so GPT b144 re-proposed the same relabelled codes in 8 of 30
    # turns and measured nothing in them. Off under the formal Fig.2 profile.
    measured_twin_feedback: bool = False
    # telescoping_structural_v3_8 (2026-10-17, user decision after the tier-3
    # replicate of the GPT arm): two machine rules the v3.7 search left to the
    # model. (1) VARIANT GENERATIONS: the last `exploit_generations` of a
    # turn's generations are restricted to parameter variations of the
    # measured-best codes and the resident reference code (the parents are
    # the only constructions shown; programs outside the parents' families
    # are rejected by the fast loop). (2) ENDPOINT FAMILY RULES: the two
    # structural endpoint slots come from two different families when the
    # front offers them, and a family whose last `endpoint_demotion_strikes`
    # endpoint measurements were all >= `endpoint_demotion_factor` x the
    # best measured code at this budget loses its endpoint slots for the
    # rest of the run. Evidence (formal tier 3, 2026-10-16): the GPT arm
    # proposed 7-12 % BB codes (Claude/DeepSeek 37-59 %), reported 6 %
    # exploit strategies against 78 % novelty every turn, and spent 13-24
    # of its ~75 endpoint slots per run on HGP codes that measured >= 1e-4
    # (its b360 replicate: 21 of 32). The defaults below are the v3.7
    # behaviour; `structural_search_v38()` returns the v3.8 profile.
    exploit_generations: int = 0
    endpoint_family_distinct: bool = False
    endpoint_demotion_strikes: int = 0
    endpoint_demotion_factor: float = 10.0

    def __post_init__(self) -> None:
        if self.structural_slow_slots + self.runtime_slow_slots != self.slow_candidate_cap:
            raise ValueError("structural + runtime slow slots must equal slow_candidate_cap")
        if self.structural_parents + self.runtime_parents + self.diversity_parents != self.parent_cap:
            raise ValueError("structural + runtime + diversity parents must equal parent_cap")
        if not self.proxy_seeds:
            raise ValueError("proxy_seeds must not be empty")
        if int(self.exploit_generations) < 0 or int(self.endpoint_demotion_strikes) < 0:
            raise ValueError("exploit_generations and endpoint_demotion_strikes must be >= 0")
        if float(self.endpoint_demotion_factor) <= 1.0:
            raise ValueError("endpoint_demotion_factor must be > 1")
        # v3.8 declaration IS execution: the v3.8 id requires exactly the v3.8
        # rule set, and the v3.7 id may not carry any of the v3.8 rules under
        # the plain name (it is retagged, and the strict validator rejects it).
        v38_fields = {key: getattr(self, key) for key in V38_STRUCTURAL_RULES}
        if self.profile_id == PROTOCOL_PROFILE_V38 and v38_fields != V38_STRUCTURAL_RULES:
            raise ValueError(
                f"{PROTOCOL_PROFILE_V38} requires {V38_STRUCTURAL_RULES}, got {v38_fields}"
            )
        # Declaration IS execution (audit fix #2): the d_search protocol fields
        # are part of the profile identity. A config that keeps the frozen
        # profile_id while changing seeds/effort/algorithm would let a run
        # DECLARE bposd_structural_v3_5 and EXECUTE something else, so any
        # drift retags the id — and the strict validator rejects the retag.
        frozen = StructuralSearchConfig.__dataclass_fields__
        drift: list[str] = []
        if int(self.proxy_iters_per_seed) != int(frozen["proxy_iters_per_seed"].default):
            drift.append(f"proxy{int(self.proxy_iters_per_seed)}")
        if tuple(self.proxy_seeds) != frozen["proxy_seeds"].default:
            drift.append("seeds" + "-".join(str(s) for s in self.proxy_seeds))
        if str(self.proxy_algorithm_id) != frozen["proxy_algorithm_id"].default:
            drift.append("alg")
        if self.profile_id == frozen["profile_id"].default:
            # A v3.7-labelled config running any v3.8 rule is a variant too.
            if int(self.exploit_generations) != 0:
                drift.append(f"exploit{int(self.exploit_generations)}")
            if bool(self.endpoint_family_distinct):
                drift.append("epfam")
            if int(self.endpoint_demotion_strikes) != 0:
                drift.append(f"epdemote{int(self.endpoint_demotion_strikes)}x{self.endpoint_demotion_factor:g}")
        if drift and self.profile_id == frozen["profile_id"].default:
            object.__setattr__(self, "profile_id", "+".join([self.profile_id, *drift]))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["proxy_seeds"] = list(self.proxy_seeds)
        data["structural_axes"] = list(self.structural_axes)
        return data


DEFAULT_STRUCTURAL_SEARCH = StructuralSearchConfig()


def structural_search_v38() -> StructuralSearchConfig:
    """The telescoping_structural_v3_8 retention profile: v3.7 plus the
    variant generations and the endpoint family rules (V38_STRUCTURAL_RULES)."""

    return StructuralSearchConfig(profile_id=PROTOCOL_PROFILE_V38, **V38_STRUCTURAL_RULES)


def structural_search_for_profile(profile: str) -> StructuralSearchConfig:
    """Retention profile by short name (``v3_7`` / ``v3_8``) or full id."""

    name = str(profile).strip()
    if name in ("v3_7", "v3.7", PROTOCOL_PROFILE_V37):
        return StructuralSearchConfig()
    if name in ("v3_8", "v3.8", PROTOCOL_PROFILE_V38):
        return structural_search_v38()
    raise ValueError(f"unknown structural profile {profile!r}; expected one of v3_7, v3_8")


# Syndrome-extraction-circuit (SEC) policies. The policy id versions the whole
# circuit family (timeline, detector rule, order search, certification); LER
# and d_circ_ext values across different ids are NOT comparable (02 §8).
#   twobasis_nonint_v1 (v3.5 default, 2026-09-02): both bases measured every
#     round, NO interleaving (X half then Z half), per-basis layer-order
#     search over K candidates certified by an ISD d_ext probe.
#   lrc_ranked_v1: the historical left-right circuit (kept on disk, selectable,
#     never the default).
SEC_POLICY_IDS: tuple[str, ...] = ("twobasis_nonint_v1", "lrc_ranked_v1")


@dataclass(frozen=True)
class SecPolicyConfig:
    """Frozen SEC policy parameters (bposd_structural_v3_5: twobasis_nonint_v1,
    K=16 candidate orders per basis, 400 ISD iterations per certification,
    seed 0). Part of the measurement protocol: recorded in the run config,
    the protocol snapshot and every memory-experiment schedule blob."""

    sec_policy_id: str = "twobasis_nonint_v1"
    sec_candidates: int = 16
    # v3.6 certification (docs/06 §3 M1): the per-candidate ISD budget is
    # max(sec_cert_iters, sec_cert_iters_scale x extended-kernel dimension),
    # the estimate is the MIN over sec_cert_seeds independent seeds, the top
    # sec_refine_top candidates per basis are re-certified with
    # sec_refine_iters iterations, and d_ext is clamped to <= d_code_proxy
    # (an ISD upper bound above the code distance is a non-converged probe).
    sec_cert_iters: int = 400
    sec_cert_seed: int = 0
    sec_cert_seeds: int = 3
    sec_cert_iters_scale: float = 4.0
    sec_refine_top: int = 4
    sec_refine_iters: int = 2000
    # Review fix (2026-09-03): a deterministic per-code cap on the TOTAL ISD
    # iterations of the certification (both bases, every candidate, every
    # seed, both stages). 0 = unlimited (the plan's M1 as specified). When
    # the nominal work exceeds the cap the ranker shrinks the stage-1 and
    # refine iterations by a fixed, seed-independent rule and records
    # ``certification_budget_exhausted`` (audit flag, never a strike or a
    # tier change) instead of running into the build watchdog.
    sec_cert_iters_budget: int = 0

    def __post_init__(self) -> None:
        if str(self.sec_policy_id) not in SEC_POLICY_IDS:
            raise ValueError(
                f"unknown sec_policy_id {self.sec_policy_id!r}; valid: {list(SEC_POLICY_IDS)}"
            )
        if int(self.sec_candidates) < 1:
            raise ValueError("sec_candidates must be >= 1")
        if int(self.sec_cert_iters) < 1:
            raise ValueError("sec_cert_iters must be >= 1")
        if int(self.sec_cert_seeds) < 1 or int(self.sec_refine_top) < 0:
            raise ValueError("sec_cert_seeds must be >= 1 and sec_refine_top >= 0")
        if int(self.sec_cert_iters_budget) < 0:
            raise ValueError("sec_cert_iters_budget must be >= 0 (0 = unlimited)")

    @property
    def is_lrc(self) -> bool:
        return self.sec_policy_id == "lrc_ranked_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sec_policy_id": str(self.sec_policy_id),
            "sec_candidates": int(self.sec_candidates),
            "sec_cert_iters": int(self.sec_cert_iters),
            "sec_cert_seed": int(self.sec_cert_seed),
            "sec_cert_seeds": int(self.sec_cert_seeds),
            "sec_cert_iters_scale": float(self.sec_cert_iters_scale),
            "sec_refine_top": int(self.sec_refine_top),
            "sec_refine_iters": int(self.sec_refine_iters),
            "sec_cert_iters_budget": int(self.sec_cert_iters_budget),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SecPolicyConfig":
        """``None``/``{}`` -> the default twobasis policy. A v3.4 artifact has no
        ``sec_policy`` block at all; it is refused by the snapshot/resume check,
        never silently re-measured under the new policy."""

        raw = dict(data or {})
        if not raw:
            return cls()
        return cls(
            sec_policy_id=str(raw.get("sec_policy_id", cls.sec_policy_id)),
            sec_candidates=int(raw.get("sec_candidates", cls.sec_candidates)),
            sec_cert_iters=int(raw.get("sec_cert_iters", cls.sec_cert_iters)),
            sec_cert_seed=int(raw.get("sec_cert_seed", cls.sec_cert_seed)),
            sec_cert_seeds=int(raw.get("sec_cert_seeds", cls.sec_cert_seeds)),
            sec_cert_iters_scale=float(raw.get("sec_cert_iters_scale", cls.sec_cert_iters_scale)),
            sec_refine_top=int(raw.get("sec_refine_top", cls.sec_refine_top)),
            sec_refine_iters=int(raw.get("sec_refine_iters", cls.sec_refine_iters)),
            sec_cert_iters_budget=int(raw.get("sec_cert_iters_budget", cls.sec_cert_iters_budget)),
        )


DEFAULT_SEC_POLICY = SecPolicyConfig()
# Historical LRC policy: P=64 colour permutations, ISD budget 100 (02 §3).
# verification/circuit.py's LRC_SEAM_CANDIDATES / LRC_SEAM_ISD_ITERS must
# equal these two numbers (pinned by tests).
LRC_SEC_POLICY = SecPolicyConfig(
    sec_policy_id="lrc_ranked_v1", sec_candidates=64, sec_cert_iters=100, sec_cert_seed=0
)


def sec_policy_for_id(policy_id: str) -> SecPolicyConfig:
    """The frozen parameter set of a policy id (CLI --sec-policy)."""

    value = str(policy_id)
    if value == LRC_SEC_POLICY.sec_policy_id:
        return LRC_SEC_POLICY
    if value == DEFAULT_SEC_POLICY.sec_policy_id:
        return DEFAULT_SEC_POLICY
    raise ValueError(f"unknown sec_policy_id {value!r}; valid: {list(SEC_POLICY_IDS)}")


@dataclass(frozen=True)
class RacingConfig:
    """Anchor-p double-threshold racing (03 §8).

    The shard stream is pre-registered as batch-size shards up to shots_max;
    racing only decides consumption depth, never the RNG protocol. A candidate
    survives a round while its per-logical-per-round (A5) CI lower bound is
    below the hall-of-fame best upper bound OR the baseline upper bound
    (thresholds are snapshot at plan entry and stay static within the batch);
    when both available thresholds are crossed it stops with
    ``stopped_by="racing"``. Survivors top up to shots_confirm per basis;
    max_errors (per candidate x basis) can stop earlier, and shots_max sizes
    the pre-registered stream (shots_confirm <= shots_max, so the confirm
    check always fires first). Sequential-stopping bias is not corrected —
    disclosed in the methods section. epsilon is a threshold slack multiplier
    (effective threshold = ub * (1 + epsilon)); v1 keeps it at 0 and records
    it in the protocol snapshot."""

    enabled: bool = False
    shots_pilot: int = 5000
    batch: int = 5000
    shots_confirm: int = 20000
    shots_max: int = 50000
    max_errors: int | None = None
    epsilon: float = 0.0
    # 2026-09-11 (pilot10x_v1): baseline_arm=False makes the hall-of-fame best
    # the ONLY elimination reference ("10x the current best code");
    # policy_id names the frozen parameter set (formal_racing()).
    baseline_arm: bool = True
    policy_id: str = ""
    # pilot10x_maxerr100_v3: evaluate the max_errors cap at every shard
    # boundary, including before the pilot completes (the 10x arm still
    # waits for the pilot). Absent in v1/v2 blocks -> False.
    max_errors_in_pilot: bool = False
    # pilot10x_maxerr100_cost_v4 (2026-09-14): a candidate whose decoder costs
    # more than cost_cap_s_per_shot_per_qubit x n CPU-seconds per shot (never
    # less than cost_cap_floor_s_per_shot), measured over its landed shards once
    # cost_cap_min_shots shots per basis are in, is decode_infeasible for this
    # protocol -- a terminal finding about the code (recorded, excluded, never
    # re-proposed), the same verdict v3.4 gave a watchdog timeout. The rule is
    # arm-neutral and the baseline arm is exempt. None = no cap (v1-v3).
    cost_cap_s_per_shot_per_qubit: float | None = None
    cost_cap_floor_s_per_shot: float = 8.0
    cost_cap_min_shots: int = 250
    # 2026-11-10 (paper alignment, Appendix D3): "If the candidate's lower bound EXCEEDS both upper bounds by
    # a factor of ten" -> strict ``>`` (the earlier policies stop at ``>=``), and "the upper bound of the
    # current BEST MEASURED code" -> the upper bound of the hall-of-fame row with the lowest point estimate
    # ("best_estimate"; the earlier policies take the smallest upper bound of any hall-of-fame row,
    # "min_ub"). Both fields are left out of to_dict() at their legacy values, so the racing block of every
    # earlier policy (and every snapshot frozen under one) is byte-identical to what it was.
    strict_exceed: bool = False
    best_bound: str = "min_ub"

    def __post_init__(self) -> None:
        if str(self.best_bound) not in RACING_BEST_BOUND_RULES:
            raise ValueError(f"racing best_bound must be one of {RACING_BEST_BOUND_RULES} (got {self.best_bound!r})")
        if self.cost_cap_s_per_shot_per_qubit is not None and float(self.cost_cap_s_per_shot_per_qubit) <= 0:
            raise ValueError("racing cost_cap_s_per_shot_per_qubit must be > 0 when set")
        if int(self.cost_cap_min_shots) < 1 or float(self.cost_cap_floor_s_per_shot) < 0:
            raise ValueError("racing cost_cap_min_shots must be >= 1 and cost_cap_floor_s_per_shot >= 0")
        if int(self.batch) < 1 or int(self.shots_pilot) < 1:
            raise ValueError("racing batch and shots_pilot must be >= 1")
        if int(self.shots_max) < int(self.shots_pilot):
            raise ValueError("racing shots_max must be >= shots_pilot")
        if int(self.shots_confirm) > int(self.shots_max):
            raise ValueError("racing shots_confirm must be <= shots_max")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if not bool(self.strict_exceed):
            out.pop("strict_exceed")
        if str(self.best_bound) == "min_ub":
            out.pop("best_bound")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RacingConfig":
        raw = dict(data or {})
        return cls(
            strict_exceed=bool(raw.get("strict_exceed", False)),
            best_bound=str(raw.get("best_bound") or "min_ub"),
            enabled=bool(raw.get("enabled", False)),
            shots_pilot=int(raw.get("shots_pilot", 5000)),
            batch=int(raw.get("batch", 5000)),
            shots_confirm=int(raw.get("shots_confirm", 20000)),
            shots_max=int(raw.get("shots_max", 50000)),
            max_errors=(None if raw.get("max_errors") is None else int(raw["max_errors"])),
            epsilon=float(raw.get("epsilon", 0.0)),
            baseline_arm=bool(raw.get("baseline_arm", True)),
            policy_id=str(raw.get("policy_id") or ""),
            max_errors_in_pilot=bool(raw.get("max_errors_in_pilot", False)),
            cost_cap_s_per_shot_per_qubit=(
                None if raw.get("cost_cap_s_per_shot_per_qubit") is None else float(raw["cost_cap_s_per_shot_per_qubit"])
            ),
            cost_cap_floor_s_per_shot=float(raw.get("cost_cap_floor_s_per_shot", 8.0)),
            cost_cap_min_shots=int(raw.get("cost_cap_min_shots", 250)),
        )


def formal_racing_v1(search_shots: int) -> RacingConfig:
    """pilot10x_v1: pilot 2000 (or the whole stage when it is smaller),
    250-shot shards, confirm = max = the stage's shots, 10x threshold against
    the hall-of-fame best, no baseline arm, no max_errors cap."""

    shots = int(search_shots)
    if shots < 1:
        raise ValueError("search_shots must be >= 1")
    return RacingConfig(
        enabled=True,
        shots_pilot=min(int(RACING_PILOT_SHOTS), shots),
        batch=min(int(RACING_BATCH_SHOTS), shots),
        shots_confirm=shots,
        shots_max=shots,
        max_errors=None,
        epsilon=float(RACING_EPSILON),
        baseline_arm=False,
        policy_id=RACING_POLICY_V1_ID,
    )


def formal_racing_v2(search_shots: int) -> RacingConfig:
    """pilot10x_maxerr100_v2: pilot10x_v1 plus a max_errors cap of
    RACING_MAX_ERRORS per basis, checked from the pilot boundary on -- a
    candidate that has already produced that many failures in every basis
    stops there, confirmed (its estimate is unbiased; only its CI is wider
    than a full measurement's). The final stage is fixed-shot and unaffected."""

    return dataclasses.replace(
        formal_racing_v1(search_shots),
        max_errors=int(RACING_MAX_ERRORS),
        policy_id=RACING_POLICY_V2_ID,
    )


def formal_racing_v3(search_shots: int) -> RacingConfig:
    """pilot10x_maxerr100_v3 (2026-09-14 afternoon): v2 with the max_errors cap
    checked at every landed shard boundary, the pilot included. A code that
    reaches the cap in both bases before the pilot ends stops there
    (confirmed, from its first 250-shot shards); every other code is measured
    exactly as under v2."""

    return dataclasses.replace(
        formal_racing_v2(search_shots),
        max_errors_in_pilot=True,
        policy_id=RACING_POLICY_V3_ID,
    )


def formal_racing_v3_maxerr30(search_shots: int) -> RacingConfig:
    """pilot10x_maxerr30_v3 (2026-09-29, large-code tier variant): v3 with
    max_errors = 30 per basis. Every candidate that never reaches 30 failures
    in a basis is measured exactly as under v3; a candidate that does stops
    earlier with a wider CI (its estimate stays unbiased). Only the >= 288
    points run under it (user decision)."""

    return dataclasses.replace(
        formal_racing_v3(search_shots),
        max_errors=int(RACING_MAX_ERRORS_LARGE),
        policy_id=RACING_POLICY_V3_MAXERR30_ID,
    )


def formal_racing_v4(search_shots: int) -> RacingConfig:
    """pilot10x_maxerr100_cost_v4 (2026-09-14 night, withdrawn 2026-09-15
    morning -- see RACING_POLICY_ID): v3 plus a decoder-cost cap. A candidate whose decoder
    costs more than 0.10 CPU-s per shot per data qubit (never less than 8 s
    per shot), measured over its landed shards once 250 shots per basis are
    in, is decode_infeasible: a terminal finding about the code, recorded and
    excluded, the same verdict v3.4 gave a watchdog timeout. Calibration
    (NSCC, telescoping_cpu, p=0.5%): BB144 3.8 s/shot, BB n~180 ~7 s/shot,
    i.e. 0.025-0.04 s/shot/qubit, so the cap is ~3x the BB trend; the
    high-k LP codes that stalled tier-3 turns (>12 s/shot at n=150-175)
    are above it. Every candidate below the cap is measured exactly as
    under v3; the baseline arm is exempt; the rule is the same for all arms."""

    return dataclasses.replace(
        formal_racing_v3(search_shots),
        cost_cap_s_per_shot_per_qubit=RACING_COST_CAP_S_PER_SHOT_PER_QUBIT,
        cost_cap_floor_s_per_shot=RACING_COST_CAP_FLOOR_S_PER_SHOT,
        cost_cap_min_shots=RACING_COST_CAP_MIN_SHOTS,
        policy_id=RACING_POLICY_V4_ID,
    )


def formal_racing(search_shots: int) -> RacingConfig:
    """The frozen early-stop policy (RACING_POLICY_ID): pilot10x_maxerr100_v3
    since 2026-09-15 morning (v4's decoder-cost cap withdrawn, see
    RACING_POLICY_ID). A code that reaches the 100-error cap in both bases
    before the pilot ends stops there; every other code is measured as under
    v2; no decoder-cost verdicts."""

    return formal_racing_v3(search_shots)


def paper_racing(search_shots: int) -> RacingConfig:
    """The paper's search-stage sampling and early stop (Appendix D3; RACING_POLICY_PAPER_ID).

    pilot = 2 000 shots per basis, batches of 250, at most ``search_shots`` (25 000) per basis, stop once both
    bases hold 30 failures (decided at the pilot boundary and after every batch, never inside the pilot), and
    the Wilson 10x rule against BOTH the best measured code and the applicable reference code
    (baseline_arm=True, strict ``>``, the best code's own upper bound). No decoder-cost cap."""

    shots = int(search_shots)
    if shots < 1:
        raise ValueError("search_shots must be >= 1")
    return RacingConfig(
        enabled=True,
        shots_pilot=min(int(RACING_PILOT_SHOTS), shots),
        batch=min(int(RACING_BATCH_SHOTS), shots),
        shots_confirm=shots,
        shots_max=shots,
        max_errors=int(RACING_MAX_ERRORS_PAPER),
        epsilon=float(RACING_EPSILON),
        baseline_arm=True,
        policy_id=RACING_POLICY_PAPER_ID,
        max_errors_in_pilot=False,
        strict_exceed=True,
        best_bound="best_estimate",
    )


FORMAL_RACING_POLICIES: dict[str, Any] = {
    RACING_POLICY_V1_ID: formal_racing_v1,
    RACING_POLICY_V2_ID: formal_racing_v2,
    RACING_POLICY_V3_ID: formal_racing_v3,
    RACING_POLICY_V4_ID: formal_racing_v4,
    RACING_POLICY_V3_MAXERR30_ID: formal_racing_v3_maxerr30,
    RACING_POLICY_PAPER_ID: paper_racing,
}


# Forward-only rank of the formal policies for the artifact validator: a
# resumed chain may only move to a policy of equal or higher rank. v4 ranks
# WITH v3: it was withdrawn on 2026-09-15 and every chain that ran a turn
# under it went back to v3 (a recorded rollback, not a backward move).
# pilot10x_maxerr30_v3 (2026-09-29) is the large-code variant reached from v3.
RACING_POLICY_RANK: dict[str, int] = {
    RACING_POLICY_V1_ID: 0,
    RACING_POLICY_V2_ID: 1,
    RACING_POLICY_V3_ID: 2,
    RACING_POLICY_V4_ID: 2,
    RACING_POLICY_V3_MAXERR30_ID: 3,
    RACING_POLICY_PAPER_ID: 4,
}


def racing_policy_moves_forward(previous_id: str, current_id: str) -> bool:
    """True when a chain may run ``current_id`` after ``previous_id``."""

    return RACING_POLICY_RANK[str(current_id)] >= RACING_POLICY_RANK[str(previous_id)]


def formal_racing_for_max_errors(search_shots: int, max_errors: int | None) -> RacingConfig:
    """The formal policy for an explicit --racing-max-errors: None / 100 = the
    frozen pilot10x_maxerr100_v3, 30 = the large-code variant
    pilot10x_maxerr30_v3. Any other value is not a registered policy."""

    if max_errors is None or int(max_errors) == int(RACING_MAX_ERRORS):
        return formal_racing(search_shots)
    if int(max_errors) == int(RACING_MAX_ERRORS_LARGE):
        return formal_racing_v3_maxerr30(search_shots)
    raise ValueError(
        f"--racing-max-errors {max_errors} is not a registered formal policy "
        f"(valid: {RACING_MAX_ERRORS} -> {RACING_POLICY_V3_ID}, "
        f"{RACING_MAX_ERRORS_LARGE} -> {RACING_POLICY_V3_MAXERR30_ID})"
    )


def formal_racing_policy_id(racing: dict[str, Any] | None) -> str | None:
    """The formal policy id a racing block IS (byte for byte, for its own
    shots_confirm), or None when it is not one of the formal policies."""

    if not isinstance(racing, dict) or not racing.get("enabled"):
        return None
    try:
        shots = int(racing.get("shots_confirm") or 0)
    except (TypeError, ValueError):
        return None
    if shots < 1:
        return None
    # Blocks written before a field existed lack it: compare after parsing
    # (from_dict fills the defaults), so a v1/v2 block frozen last week still
    # resolves to its policy.
    normalized = RacingConfig.from_dict(racing).to_dict()
    for policy_id, factory in FORMAL_RACING_POLICIES.items():
        if factory(shots).to_dict() == normalized:
            return policy_id
    return None


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "deepseek"
    model_env: str = "DEEPSEEK_MODEL"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url_env: str = "DEEPSEEK_BASE_URL"
    default_model: str = "deepseek-chat"
    default_base_url: str = "https://api.deepseek.com"
    temperature: float = 0.45
    # deepseek-v4-pro is a thinking model whose reasoning tokens count against
    # the completion budget. 131072 is the maximum the API accepts (probed
    # 2026-06-12); the model bills only what it generates, so there is no cost
    # to granting the full window.
    max_tokens: int = 131072
    # Thinking models need more wall-clock per call than chat models; Opus 4.8
    # emitting up to 24 construct programs (+ reasoning) wants a generous window.
    # Overridable per experiment; transient timeouts retry with backoff.
    timeout_s: float = 900.0
    # A hung TCP connect must not spend the whole read budget: reaching the
    # provider is fast or not happening at all.
    connect_timeout_s: float = 20.0
    # Transient API failures (5xx, network, rate limit) retry with exponential
    # backoff so one hiccup does not waste a whole generation's proposals.
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    # Hard wall-clock ceiling for one logical call including every retry.
    # Transport timeouts can fail to fire (half-closed socket, trickling
    # server), and without a ceiling the retries stack: a campaign once sat
    # two hours inside a single orchestrator call. Failing here costs one
    # turn, which --resume replays; hanging costs the whole job.
    total_deadline_s: float = 1800.0

    @property
    def model(self) -> str:
        return os.environ.get(self.model_env, self.default_model)

    @property
    def base_url(self) -> str:
        return os.environ.get(self.base_url_env, self.default_base_url)

    def require_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"missing API key in ${self.api_key_env}")
        return key

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model"] = self.model
        data["base_url"] = self.base_url
        data["api_key_present"] = bool(os.environ.get(self.api_key_env, "").strip())
        return data


@dataclass(frozen=True)
class HardwareConfig:
    target: HardwareTarget
    connectivity_rule: str = "all_to_all"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "connectivity_rule": self.connectivity_rule,
            "notes": self.notes,
        }


# Fig.3 Phase B (2026-10-12): what the slow loop measures. "memory" = the Fig.2
# two-basis memory experiment; "cnot_dense" = two blocks of the code with one
# transversal logical CNOT between them after every QEC round of the same
# T = max(12, d_search) rounds, decoded jointly, reported in the Fig.2 plpr
# convention over the circuit's 2k logical qubits and T rounds
# (evaluation/task_circuits.py). Part of the protocol snapshot.
TASK_IDS: tuple[str, ...] = ("memory", "cnot_dense")
# S3 ablation (2026-10-23, appendix S3): which search feedback the LLM roles
# receive. "full" = the formal flow; "no_slow" = fast-loop-only (no
# circuit-level measurement during the search, the generator sees the
# structural layer only, the final stage measures the structural front);
# "no_feedback" = every turn is proposed from the task/constraint prompt alone
# (no archive, parents, orchestrator, reflector or measurement; the final
# stage ranks the whole legal pool by the structural proxy afterwards). A
# protocol VARIANT recorded in the snapshot (key `ablation`, absent = full);
# the generator's feedback_mode follows it (ABLATION_FEEDBACK_MODE).
ABLATION_IDS: tuple[str, ...] = ("full", "no_slow", "no_feedback")
ABLATION_FEEDBACK_MODE: dict[str, str] = {"full": "both", "no_slow": "fast", "no_feedback": "none"}
# The plan policy ids a strict turn audit accepts per ablation mode.
ABLATION_PLAN_POLICIES: dict[str, tuple[str, ...]] = {
    "full": ("agentic_v1",),
    "no_slow": ("agentic_v1",),
    "no_feedback": ("ablation_no_feedback_v1",),
}


def ablation_of(protocol: "dict[str, Any] | None") -> str:
    """The ablation mode frozen in a protocol snapshot (absent key = full)."""

    value = str((protocol or {}).get("ablation") or "full")
    if value not in ABLATION_IDS:
        raise ValueError(f"unknown ablation mode {value!r} (expected one of {ABLATION_IDS})")
    return value


# Search baselines without an LLM (2026-11-06, user decision; omniqec_agentic/baselines). The Full search's
# construction, verification, slow measurement and final stage stay; a fixed sampling prior replaces the generator
# and a uniform ("random") or Gaussian-process ("bo") choice from the cumulative legal pool replaces the slow-queue
# selection. A protocol VARIANT recorded in the snapshot (key `search_baseline`, absent = the LLM search).
SEARCH_METHODS: tuple[str, ...] = ("llm", "random", "bo")
BASELINE_PROVIDER = "baseline_sampler"
BASELINE_PLAN_POLICY_ID = "baseline_fixed_turn_v1"
# 4 generations x 960 draws = 3840 raw proposals per turn, the Full search's 4 x 24 programs x 40 seeds.
BASELINE_PROPOSALS_PER_GENERATION = 960
BASELINE_GENERATIONS_PER_TURN = 4
# bo: turns 0..3 nominate uniformly from the pool (the random baseline's rule), the model from turn 4.
BASELINE_INIT_RANDOM_TURNS = 4
# The search baselines run the 30 iterations of every discovery run (paper appendix D4/D5): run-campaign's --turns
# default for --search-method random / bo.
BASELINE_TURNS = 30


#
# Paper alignment (2026-11-10, docs/paper_alignment_search.md): Methods "Search configurations" and appendix D1:
# "All main searches require valid CSS codes, at least one logical qubit, a connected Tanner graph and a maximum
# stabilizer-check weight wmax <= 8. These constraints apply to both circuit tasks." Appendix B2 lists the cap among
# the fast-loop checks that run BEFORE distance estimation, so it is the default construction-time cap of
# VerificationConfig (the fast loop rejects a heavier code; the slow-loop gate promotion_max_check_weight stays as a
# redundant guard). None (CLI --max-check-weight 0) is the v3.7 as-run protocol, which had no fast-loop cap.
SEARCH_MAX_CHECK_WEIGHT = 8
# Appendix D1: the five manually written seed programs "are executed only during the first iteration. Subsequent
# iterations draw on the programs and evaluation records accumulated during the search, without repeating this
# initialization." Protocol snapshot key `seed_programs`; a snapshot WITHOUT the key was written by the earlier
# code, which ran the seeds at the start of every turn ("every_turn"), and keeps that behaviour on resume.
SEED_PROGRAMS_FIRST_TURN = "first_turn"
SEED_PROGRAMS_EVERY_TURN = "every_turn"
SEED_PROGRAMS_POLICIES: tuple[str, ...] = (SEED_PROGRAMS_FIRST_TURN, SEED_PROGRAMS_EVERY_TURN)


def seed_programs_policy_of(protocol: "dict[str, Any] | None") -> str:
    """The seed-program policy frozen in a protocol snapshot (absent key = the legacy every_turn)."""

    value = str((protocol or {}).get("seed_programs") or SEED_PROGRAMS_EVERY_TURN)
    if value not in SEED_PROGRAMS_POLICIES:
        raise ValueError(f"unknown seed_programs policy {value!r} (expected one of {SEED_PROGRAMS_POLICIES})")
    return value


def seed_programs_run_this_turn(protocol: "dict[str, Any] | None", turn: int) -> bool:
    """Whether the hand-written seed programs are executed in this turn's fast loop."""

    if seed_programs_policy_of(protocol) == SEED_PROGRAMS_FIRST_TURN:
        return int(turn) == 0
    return True


def search_method_of(protocol: "dict[str, Any] | None") -> str:
    """The search method frozen in a protocol snapshot: "llm" when the snapshot has no baseline key."""

    baseline = (protocol or {}).get("search_baseline")
    if not baseline:
        return "llm"
    method = str(baseline.get("method") or "")
    if method not in SEARCH_METHODS or method == "llm":
        raise ValueError(f"unknown search baseline {method!r} (expected one of {SEARCH_METHODS[1:]})")
    return method
# v2 (2026-10-15, the formal Fig.3 run): + measured_twin_feedback. The Phase B
# v1 runs (telescoping_structural_v3_7_cnot, 2026-10-12..15) are pilot data.
FIG3_CNOT_PROFILE_ID = "telescoping_structural_v3_7_cnot_v2"


def fig3_cnot_structural_config(base: "StructuralSearchConfig | None" = None) -> "StructuralSearchConfig":
    """The Fig.3 structural profile: v3.7 plus a third structural slow slot
    for the front's lightest cell (3S + 4R = 7 measurements per turn), the
    not-competitive prompt block (ratio 2) and, from v2, the measured/twin
    feedback to the generator. Q = k d^2, the d_search floor, the parent
    window and the final shortlist are the v3.7 values. The distinct
    profile_id keeps these artifacts from pooling with formal Fig.2 ones."""

    import dataclasses as _dc

    base = base or StructuralSearchConfig()
    return _dc.replace(
        base,
        profile_id=FIG3_CNOT_PROFILE_ID,
        slow_candidate_cap=int(base.slow_candidate_cap) + 1,
        structural_slow_slots=int(base.structural_slow_slots) + 1,
        lightest_endpoint_slot=True,
        not_competitive_ratio=2.0,
        measured_twin_feedback=True,
    )


@dataclass(frozen=True)
class VerificationConfig:
    # Per-seed proxy effort; the search protocol runs it once per entry of
    # StructuralSearchConfig.proxy_seeds on each of the X/Z sectors.
    proxy_iters: int = 120
    proxy_n_min: int | None = None
    proxy_n_max: int | None = None
    # telescoping_structural_v3_6: ONE decoder everywhere (search, runtime
    # hall of fame, fixed-T final audit). No LSD/Beam fallback, no per-family
    # tuning, no per-code minimum over several decoders.
    decoders: tuple[DecoderName, ...] = (DecoderName.TELESCOPING,)
    p_values: tuple[float, ...] = (0.005,)
    # New code should use ``noise``. ``p_values`` remains as a synchronized
    # compatibility alias: it names points of the CURRENT uniform protocol
    # (uniform_paper_v1 since v3.5; uniform_legacy_v1 only via
    # NoiseSweepConfig.uniform_legacy / --noise-profile uniform_legacy_v1).
    noise: NoiseSweepConfig = field(default_factory=_default_noise_sweep)
    # Small-budget policy search shots per basis (2026-09-10: 25k; was 20k).
    shots: int = 25_000
    rounds: int | None = None
    # Memory bases to decode. ("X","Z") = both (combined LER). ("Z",) = Z-memory
    # only (focus on correcting X errors) — a standard single-basis study, ~2x
    # faster; the resident BB baseline is measured with the SAME bases, so the
    # comparison stays apples-to-apples. Champions get re-confirmed on both bases.
    bases: tuple[str, ...] = ("X", "Z")
    max_errors: int | None = 80
    # Frozen slow-queue cap (§4.3): 2026-09-10 4 structural slots + 2 runtime
    # slots (was 3 + 1). The policy may name at most runtime_slow_slots
    # candidates and each CONSUMES a runtime slot; nothing adds slots beyond
    # this cap. Must equal StructuralSearchConfig.slow_candidate_cap.
    promote_top: int = 6
    # Legacy monolithic worker timeout. Old serialized artifacts contain only
    # this field; verification_from_dict maps it into each typed timeout below.
    # New v3.3 artifacts freeze the typed fields explicitly, so a slow distance
    # probe cannot impose a decode cutoff (or vice versa).
    circuit_timeout_s: float = 600.0
    build_timeout_s: float = 3600.0
    # v3.4: decode_shard_timeout_s is the FLOOR of the decode-shard watchdog.
    # The effective watchdog is max(floor, decode_shot_budget_s * shard shots)
    # (pool.effective_decode_shard_timeout_s): a 1250-shot search shard gets
    # 10,000 s and a 1562-shot final shard 12,500 s, where the flat 1800 s of
    # v3.3 could never be met by the slow lp_abelian codes (5.2-5.5 s/shot).
    # A non-positive budget falls back to the flat floor alone.
    decode_shard_timeout_s: float = 1800.0
    decode_shot_budget_s: float = 8.0
    # 2026-09-24: hard CEILING on the final-stage decode-shard watchdog (the
    # search stage is untouched). The telescoping worst-case term lets one
    # non-converging high-distance shard run for a day, which is longer than
    # a 24 h link; capped shards expire as plain ``timeout`` and the candidate
    # ends decode_infeasible through the fixed_shot_timeout_only gate. A
    # protocol RUNTIME key (checkpoint.RESUME_RUNTIME_KEYS): never serialised
    # by to_dict, recorded in slow_run_config.json / decode_budget_model.
    # None = no ceiling (the v3.6/v3.7 behaviour).
    final_decode_shard_timeout_cap_s: float | None = None
    # 2026-09-25: the same ceiling for the SEARCH-stage slow queue. Under the
    # frozen pilot10x_maxerr100_v3 policy (no decoder-cost cap) one
    # non-converging b360 candidate held a 128-core link for 11+ h; the next
    # link replays the turn and stalls on the same code. Runtime key too.
    search_decode_shard_timeout_cap_s: float | None = None
    distance_timeout_s: float = 1800.0
    # Number of retries after the first timeout. The identical request object
    # (including its pre-registered seed) is replayed; 2 means 3 total attempts.
    # v3.4 freezes this at 0: a deterministic decode that exceeded its per-shot
    # budget replays identically, so a retry only burns the watchdog again.
    timeout_retries: int = 0
    # Identical replay of a request whose worker died (no parseable JSON, e.g.
    # an OOM SIGKILL) or lost its pickled DEM (worker status dem_load_error).
    # Never applies to watchdog timeouts.
    transient_retries: int = 1
    # Ranking semantics (reconcile design): distance reward is capped at the
    # verifiable regime, and a ver1-style d/sqrt(n) trust filter screens
    # proxy-inflated candidates out of the fast tier.
    # 2026-09-10: OFF. The cap existed because every candidate was measured
    # at the fixed T_c=12; under rounds policy max_Tc_d a d>12 code is
    # measured at T=d, so capping its Q_eff distance at 12 would undercut the
    # very codes the policy exists to measure. `d_cap` is kept for --d-cap.
    d_cap_enabled: bool = False
    # 12 = the verifiable-regime ceiling under the fixed T_c=12 search rounds
    # (2026-08-16; was 8 — which under-credited genuine d>8 codes in the
    # fast-loop Q_eff ranking and starved them of promotion slots).
    # v3.2: d_cap and d_eff belong to the runtime/MAP channel ONLY. The
    # structural reducer never reads either.
    d_cap: int = 12
    # v3.2: the d/sqrt(n) screen is a runtime/acquisition screen and an audit
    # field. It no longer decides structural eligibility — at n_data=288 it
    # rejects d_search=4, which is exactly the high-k endpoint the structural
    # channel exists to keep.
    dsqrtn_floor: float = 0.25
    # Per-check stabilizer weight cap of the FAST loop (construction-time
    # structural check, max_check_weight_exceeded). Paper alignment
    # (2026-11-10): SEARCH_MAX_CHECK_WEIGHT = 8 for every main search, both
    # tasks (Methods "Search configurations", appendix B2/D1). v3.7
    # (2026-09-12) had no cap here (None); a snapshot written by that code
    # carries None and is read back as None. Hardware runs may still tighten
    # it (CampaignSpec.weight_limit).
    max_check_weight: int | None = SEARCH_MAX_CHECK_WEIGHT
    # Fig.4 c (2026-10-23): exact maximum check-weight contract. When set, the
    # structural check rejects a code whose max row weight over H_X / H_Z
    # differs from it (check_weight_mismatch); max_check_weight must be None or
    # >= it (the CLI sets both to the same value; an exact contract under the
    # default cap 8 is consistent with it). Serialized only when set, so every
    # earlier snapshot keeps its bytes.
    exact_check_weight: int | None = None
    ler_ci_z: float = 1.96
    # LER-slope -> d_circ calibration offset. UNCALIBRATED under the LRC
    # protocol (03 D8): the previous 0.958 was calibrated on the ver1
    # single-sided memory model (2026-06-10) and became invalid with the
    # 02 §2.0 experiment-model upgrade; ver1's 1.622 belongs to ver1's sinter
    # protocol. None keeps the d_circ_ler_estimate channel explicitly
    # unavailable ("uncalibrated_noise_profile") until `calibrate-slope-offset`
    # re-runs on the first final_slow survivors + BB baseline (03 §15).
    slope_offset: float | None = None
    decoder_build: DecoderBuildConfig = field(default_factory=DecoderBuildConfig)
    exact_distance: ExactDistanceConfig = field(default_factory=ExactDistanceConfig)
    structural: StructuralSearchConfig = field(default_factory=StructuralSearchConfig)
    # v3.5 SEC policy (twobasis_nonint_v1 by default; lrc_ranked_v1 is the
    # historical circuit and a different protocol). Serialized with the
    # config; an artifact without the key reads back as the default and is
    # then refused by the snapshot/resume check, never silently re-measured.
    sec_policy: SecPolicyConfig = field(default_factory=SecPolicyConfig)
    # Fig.3 Phase B: the slow-loop task, one of TASK_IDS (see above).
    task: str = "memory"
    # Rounds policy, one of ROUNDS_POLICY_IDS: max_Tc_d (formal Fig.2: T =
    # max(T_c, d_search)) or fixed_Tc (Fig.3 formal run: T = T_c for every code).
    rounds_policy: str = ROUNDS_POLICY_ID

    def __post_init__(self) -> None:
        import dataclasses as _dc

        from .verification.noise import UNIFORM_PROFILES, NoiseSweepConfig

        if int(self.timeout_retries) < 0:
            raise ValueError("timeout_retries must be >= 0")
        if int(self.transient_retries) < 0:
            raise ValueError("transient_retries must be >= 0")
        if str(self.task) not in TASK_IDS:
            raise ValueError(f"task must be one of {TASK_IDS}; got {self.task!r}")
        if str(self.rounds_policy) not in ROUNDS_POLICY_IDS:
            raise ValueError(f"rounds_policy must be one of {ROUNDS_POLICY_IDS}; got {self.rounds_policy!r}")
        if self.exact_check_weight is not None:
            if int(self.exact_check_weight) < 3:
                raise ValueError("exact_check_weight must be >= 3 when set")
            if self.max_check_weight is not None and int(self.max_check_weight) < int(self.exact_check_weight):
                raise ValueError(
                    "max_check_weight must be None or >= exact_check_weight "
                    f"({self.max_check_weight} < {self.exact_check_weight})"
                )
        if self.max_check_weight is not None and int(self.max_check_weight) < 1:
            raise ValueError("max_check_weight must be >= 1 when set (None = no fast-loop cap)")

        # ONE source of truth for proxy effort (v3.2 audit fix): the executors
        # read structural.proxy_iters_per_seed everywhere; proxy_iters is a
        # synchronized compatibility alias (same pattern as p_values/noise).
        # A non-protocol effort also rewrites profile_id, so a run cannot
        # DECLARE 120 iters/seed while EXECUTING something else — the identity
        # the strict validator checks changes with the effort.
        default_structural = StructuralSearchConfig()
        if int(self.proxy_iters) != int(self.structural.proxy_iters_per_seed):
            if self.structural == default_structural:
                object.__setattr__(
                    self,
                    "structural",
                    _dc.replace(
                        self.structural,
                        proxy_iters_per_seed=int(self.proxy_iters),
                    ),
                )
            elif int(self.proxy_iters) == int(default_structural.proxy_iters_per_seed):
                object.__setattr__(
                    self, "proxy_iters", int(self.structural.proxy_iters_per_seed)
                )
            else:
                raise ValueError(
                    "proxy_iters and structural.proxy_iters_per_seed disagree "
                    f"({self.proxy_iters} != {self.structural.proxy_iters_per_seed}); "
                    "set the effort in exactly one place"
                )

        # p_values <-> noise sync. BOTH uniform profiles (paper v3.5 default,
        # legacy v3.4 idle-noise) are "uniform" here: a non-default alias
        # against a family-default uniform sweep means the caller named the
        # points through p_values, so the alias drives the sweep's values and
        # the sweep keeps its profile; otherwise the sweep is the truth and
        # the alias mirrors it.
        default_values = (0.005,)
        uniform_defaults = (NoiseSweepConfig.uniform(), NoiseSweepConfig.uniform_legacy())
        if self.noise in uniform_defaults and tuple(self.p_values) != default_values:
            object.__setattr__(
                self,
                "noise",
                _dc.replace(
                    self.noise, values=tuple(float(value) for value in self.p_values)
                ),
            )
        else:
            object.__setattr__(self, "p_values", tuple(float(value) for value in self.noise.values))
        # 0.958 is the fast-loop proxy calibration of the uniform family (kept
        # for both uniform profiles; out of v3.5 scope); every other profile
        # is uncalibrated (03 D8).
        if self.noise.profile not in UNIFORM_PROFILES and self.slope_offset == 0.958:
            object.__setattr__(self, "slope_offset", None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy_iters": self.proxy_iters,
            "proxy_n_min": self.proxy_n_min,
            "proxy_n_max": self.proxy_n_max,
            "decoders": [decoder.value for decoder in self.decoders],
            "p_values": list(self.p_values),
            "noise": self.noise.to_dict(),
            "shots": self.shots,
            "rounds": self.rounds,
            "bases": list(self.bases),
            "max_errors": self.max_errors,
            "promote_top": self.promote_top,
            "circuit_timeout_s": self.circuit_timeout_s,
            "build_timeout_s": self.build_timeout_s,
            "decode_shard_timeout_s": self.decode_shard_timeout_s,
            "decode_shot_budget_s": self.decode_shot_budget_s,
            "distance_timeout_s": self.distance_timeout_s,
            "timeout_retries": self.timeout_retries,
            "transient_retries": self.transient_retries,
            "d_cap_enabled": self.d_cap_enabled,
            "d_cap": self.d_cap,
            "dsqrtn_floor": self.dsqrtn_floor,
            "max_check_weight": self.max_check_weight,
            **(
                {"exact_check_weight": int(self.exact_check_weight)}
                if self.exact_check_weight is not None
                else {}
            ),
            "ler_ci_z": self.ler_ci_z,
            "slope_offset": self.slope_offset,
            "decoder_build": self.decoder_build.to_dict(),
            "exact_distance": self.exact_distance.to_dict(),
            "structural": self.structural.to_dict(),
            "sec_policy": self.sec_policy.to_dict(),
            "task": self.task,
            "rounds_policy": self.rounds_policy,
        }


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    n_budget: int
    hardware: HardwareConfig
    out_dir: str
    llm: LLMConfig = field(default_factory=LLMConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    proposal_count: int = 8
    allowed_families: tuple[CodeFamily, ...] = DEFAULT_FAMILIES
    deformation_mode: DeformationMode = DeformationMode.NONE
    lp_shape_pin: LpShapePin = field(default_factory=LpShapePin)
    # 消融开关:both=fast+slow 一起反馈(默认,现状);fast/slow=prompt 里只给一种反馈;
    # none=不给任何反馈(不显归档/父代/历史,每代等于冷启动)。用于证明反馈循环有用。
    feedback_mode: str = "both"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            # n_budget is the TOTAL qubit budget: N_data + one measurement
            # ancilla per check row (BB72 costs 144; Quantinuum Helios is 98).
            "n_budget": self.n_budget,
            "budget_kind": "total",
            "hardware": self.hardware.to_dict(),
            "out_dir": self.out_dir,
            "llm": self.llm.to_dict(),
            "verification": self.verification.to_dict(),
            "proposal_count": self.proposal_count,
            "allowed_families": [family.value for family in self.allowed_families],
            "deformation_mode": self.deformation_mode.value,
            "lp_shape_pin": self.lp_shape_pin.to_dict(),
            "feedback_mode": self.feedback_mode,
        }


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load simple KEY=VALUE lines from a .env file without adding a dependency."""

    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_env_value(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def default_run_config(
    *,
    run_id: str,
    n_budget: int,
    hardware: str | HardwareTarget,
    out_dir: str,
) -> RunConfig:
    target = hardware if isinstance(hardware, HardwareTarget) else HardwareTarget(str(hardware))
    return RunConfig(
        run_id=run_id,
        n_budget=int(n_budget),
        hardware=HardwareConfig(target=target),
        out_dir=out_dir,
    )
