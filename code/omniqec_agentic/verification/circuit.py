"""Memory-experiment seam: sec/ synthesis behind the ver1 entry point.

v3.5: the SEC policy is selected by ``SecPolicyConfig`` — ``twobasis_nonint_v1``
(default; both bases every round, non-interleaved, certified intra-basis
orders, 02 §9.4) or the historical ``lrc_ranked_v1`` left-right circuit."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .noise import NoisePoint, NoiseProfile

try:  # config.py gains SecPolicyConfig in the same v3.5 change
    from omniqec_agentic.config import DEFAULT_SEC_POLICY, LRC_SEC_POLICY, SecPolicyConfig
except ImportError:  # pragma: no cover - local fallback until config.py lands
    from omniqec_agentic.sec.twobasis_rank import (  # noqa: F401
        DEFAULT_SEC_POLICY,
        LRC_SEC_POLICY,
        SecPolicyConfig,
    )


def detector_kind_counts(detector_kinds: Any) -> dict[str, int]:
    """{"X": n, "Z": n} of the emitter's per-detector check kinds."""

    kinds = [str(kind).upper() for kind in detector_kinds]
    return {"X": int(kinds.count("X")), "Z": int(kinds.count("Z"))}


@dataclass(frozen=True)
class MemoryExperiment:
    basis: str
    rounds: int
    p: float
    circuit: Any
    detector_error_model: Any
    schedule: Any  # TwoBasisSchedule (sec/twobasis_rank.py) or LRCSchedule (sec/schedule.py)
    num_data_qubits: int
    num_checks: int
    num_logicals: int
    noise_point: NoisePoint | None = None
    deformation: str = "none"
    h_mask: tuple[int, ...] = ()
    # v3.6 (docs/06 §1.1): the check kind ("X"/"Z") of every detector, in
    # detector-id order, recorded by the SEC emitter while it appends each
    # DETECTOR. Consumed by the telescoping decoder (GARI X/Z mask); ()
    # on legacy paths.
    detector_kinds: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        import stim

        data = {
            "basis": self.basis,
            "rounds": int(self.rounds),
            "num_data_qubits": int(self.num_data_qubits),
            "num_checks": int(self.num_checks),
            "num_logicals": int(self.num_logicals),
            "num_detectors": int(self.detector_error_model.num_detectors),
            "num_observables": int(self.detector_error_model.num_observables),
            "num_ticks": int(self.circuit.num_ticks),
            "schedule": self.schedule.to_metadata(),
            "stim_version": str(stim.__version__),
            "noisy_circuit_sha256": _sha256_text(self.circuit),
            "dem_sha256": _sha256_text(self.detector_error_model),
        }
        if self.noise_point is None:
            data["p"] = float(self.p)
        else:
            data[self.noise_point.axis] = float(self.noise_point.value)
        if self.noise_point is not None:
            data["noise_point"] = self.noise_point.to_dict()
        data["deformation"] = self.deformation
        data["detector_kind_counts"] = detector_kind_counts(self.detector_kinds)
        if any(self.h_mask):
            data["h_mask_weight"] = int(sum(self.h_mask))
            data["h_mask_sha256"] = hashlib.sha256(bytes(self.h_mask)).hexdigest()
        return data


@dataclass(frozen=True)
class MemoryExperimentPair:
    x_memory: MemoryExperiment
    z_memory: MemoryExperiment
    d_hint: int | None = None
    # RankedSchedule from sec synthesis (carries d_circ_ext evidence and the
    # chosen candidate's provenance); None only for legacy matrix-only paths.
    ranked: Any = None

    @property
    def sec_policy_id(self) -> str:
        if self.ranked is None:
            return "unknown"
        return str(self.ranked.schedule.source)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "d_hint": self.d_hint,
            "x_memory": self.x_memory.to_metadata(),
            "z_memory": self.z_memory.to_metadata(),
        }


# In-process synthesis cache: ranked schedules are a property of the code
# (matrices + partition/blocks + SEC policy params), independent of noise
# point and basis, so one candidate is synthesized once per process (02
# §7.1). The optional ``synth_cache_dir`` adds a cross-process disk layer so
# the job pool (one subprocess per job) also synthesizes once per candidate.
# Keys embed the policy id, K, iters, seed and (twobasis) the layer_blocks
# hash, so lrc_ranked_v1 and twobasis_nonint_v1 never share an entry.
_SYNTH_CACHE: dict[str, Any] = {}
_LRC_SYNTH_CACHE = _SYNTH_CACHE  # historical alias
# Full LRC policy parameters (02 §3: P=64 color permutations, ISD budget 100)
# — mirrored by LRC_SEC_POLICY (asserted below).
LRC_SEAM_CANDIDATES = 64
LRC_SEAM_ISD_ITERS = 100
assert (LRC_SEC_POLICY.sec_candidates, LRC_SEC_POLICY.sec_cert_iters) == (
    LRC_SEAM_CANDIDATES,
    LRC_SEAM_ISD_ITERS,
)

# The message token every layer (worker dicts, subprocess stderr, ranking
# rows) can recognize a catastrophic schedule by — the exception type itself
# does not survive a subprocess boundary.
SCHEDULE_CATASTROPHIC_TOKEN = "schedule_catastrophic"


class ScheduleCatastrophicError(ValueError):
    """No sampled schedule avoids a single-fault undetected logical (02 §3.3).

    v3.2 §4.4: this is the ONE hard exclusion — the candidate cannot be
    measured at all. It is a physics verdict about the code, not an
    infrastructure failure, and downstream layers treat it as a successfully
    identified terminal state."""


def is_schedule_catastrophic_reason(reason: object) -> bool:
    return SCHEDULE_CATASTROPHIC_TOKEN in str(reason or "")


def _synth_disk_load(cache_dir: str | None, key: str) -> Any | None:
    if not cache_dir:
        return None
    import pickle

    path = Path(cache_dir) / f"synth_{key}.pkl"
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _synth_disk_store(cache_dir: str | None, key: str, ranked: Any) -> None:
    if not cache_dir:
        return
    import pickle
    import tempfile as _tempfile

    directory = Path(cache_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _tempfile.NamedTemporaryFile(
            "wb", dir=str(directory), prefix=f"synth_{key}.", delete=False
        ) as handle:
            pickle.dump(ranked, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp_name = handle.name
        os.replace(temp_name, directory / f"synth_{key}.pkl")
    except Exception:
        pass  # cache miss on the next job is correct, just slower


# M5 (docs/06 §3): synthesis of one candidate is guarded by a lock file in
# the synth cache directory so concurrent pool jobs (distance_job + one
# build_dem per noise point) share ONE synthesis instead of racing 1+N times.
# A waiter polls for the finished pickle for at most SYNTH_LOCK_WAIT_S; a
# lock older than SYNTH_LOCK_STALE_S is treated as abandoned (a killed
# worker) and taken over. Without a cache dir nothing changes.
SYNTH_LOCK_WAIT_S = 3600.0
SYNTH_LOCK_STALE_S = 3600.0
SYNTH_LOCK_POLL_S = 0.5


def _synth_lock_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"synth_{key}.lock"


def _synth_progress_path(cache_dir: str | None, key: str) -> Path | None:
    """Certification checkpoint of one synthesis (None without a cache dir)."""

    if not cache_dir:
        return None
    return Path(cache_dir) / f"synth_{key}.progress.json"


def _synth_lock_try_acquire(cache_dir: str, key: str) -> bool:
    """os.open(O_CREAT|O_EXCL) on <key>.lock; True when this process owns it."""

    path = _synth_lock_path(cache_dir, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:  # stale lock from a killed worker: take it over
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        if age > SYNTH_LOCK_STALE_S:
            try:
                path.unlink()
            except OSError:
                pass
            return _synth_lock_try_acquire(cache_dir, key)
        return False
    except OSError:
        return True  # unusable cache dir: behave as if unguarded
    try:
        os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode())
    finally:
        os.close(fd)
    return True


def _synth_lock_release(cache_dir: str, key: str) -> None:
    try:
        _synth_lock_path(cache_dir, key).unlink()
    except OSError:
        pass


def _synth_lock_wait(cache_dir: str, key: str) -> Any | None:
    """Wait (bounded) for another process' synthesis; returns its result or
    None when the wait expired / the lock vanished without a pickle."""

    deadline = time.monotonic() + SYNTH_LOCK_WAIT_S
    path = _synth_lock_path(cache_dir, key)
    while time.monotonic() < deadline:
        ranked = _synth_disk_load(cache_dir, key)
        if ranked is not None:
            return ranked
        if not path.exists():
            return _synth_disk_load(cache_dir, key)
        time.sleep(SYNTH_LOCK_POLL_S)
    return None


def _synthesize_with_lock(cache_dir: str | None, key: str, compute) -> Any:
    """Disk-cached synthesis shared across processes (see M5 above)."""

    if not cache_dir:
        return compute()

    def _owned() -> Any:
        try:
            ranked = _synth_disk_load(cache_dir, key)
            if ranked is None:
                ranked = compute()
                _synth_disk_store(cache_dir, key, ranked)
            return ranked
        finally:
            _synth_lock_release(cache_dir, key)

    if _synth_lock_try_acquire(cache_dir, key):
        return _owned()
    ranked = _synth_lock_wait(cache_dir, key)
    if ranked is not None:
        return ranked
    # The holder vanished without a pickle, or the bounded wait expired:
    # one more attempt to own the lock, else synthesize locally.
    if _synth_lock_try_acquire(cache_dir, key):
        return _owned()
    ranked = compute()
    _synth_disk_store(cache_dir, key, ranked)
    return ranked


def _with_recomputed_degradation(ranked: Any, d_hint: int | None) -> Any:
    """TB-4: a synth cache hit carries the FIRST caller's d_code_proxy in its
    metadata; recompute schedule_degraded_x/z (and d_code_proxy) for THIS
    caller's d_hint on a copy, never trusting or mutating the cached blob."""

    metadata = getattr(ranked, "metadata", None)
    if not isinstance(metadata, dict) or "schedule_degraded_x" not in metadata:
        return ranked
    proxy = None if d_hint is None else int(d_hint)
    d_x = getattr(ranked, "d_circ_ext_x", metadata.get("d_ext_x"))
    d_z = getattr(ranked, "d_circ_ext_z", metadata.get("d_ext_z"))
    new_meta = dict(metadata)
    new_meta["d_code_proxy"] = proxy
    new_meta["schedule_degraded_x"] = bool(
        d_x is not None and proxy is not None and int(d_x) < proxy
    )
    new_meta["schedule_degraded_z"] = bool(
        d_z is not None and proxy is not None and int(d_z) < proxy
    )
    if new_meta == metadata:
        return ranked
    import dataclasses as _dc

    try:
        return _dc.replace(ranked, metadata=new_meta)
    except Exception:
        return ranked


def build_memory_experiment_pair(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    p: float,
    rounds: int | None = None,
    d_hint: int | None = None,
    prefer_coloring: bool = True,
    decompose_errors: bool = False,
    noise_point: NoisePoint | None = None,
    h_mask: np.ndarray | list[int] | tuple[int, ...] | None = None,
    deformation: str = "none",
    sector_sizes: list[int] | None = None,
    synth_cache_dir: str | None = None,
    sec_policy: SecPolicyConfig | None = None,
    layer_blocks: dict[str, Any] | None = None,
    rounds_policy: str = "max_Tc_d",
) -> MemoryExperimentPair:
    """Build X/Z memory experiments and undecomposed DEMs for one CSS code.

    ``rounds_policy`` (2026-10-27 fix) is the verification's rounds policy,
    handed to ``select_memory_rounds`` together with ``rounds`` (the floor
    T_c) and ``d_hint``: ``max_Tc_d`` (default, the formal Fig.2 rule) gives
    T = max(T_c, d_hint), ``fixed_Tc`` gives T = T_c for every code. Before
    the fix the policy never reached this call, so a ``fixed_Tc`` evaluation
    built its circuits at T = d_hint whenever d_hint > T_c while the pool
    normalised the rates by T_c. The SEC synthesis cache below is keyed
    without the rounds on purpose: the certified schedule does not depend on
    T, and every circuit / DEM is rebuilt at ``selected_rounds`` per call.

    ``sec_policy`` (default ``DEFAULT_SEC_POLICY`` = twobasis_nonint_v1)
    selects the synthesis: the two-basis non-interleaved circuit with
    certified intra-basis orders (02 §9.4; ``layer_blocks`` from the compiled
    code gives the structured layers, None falls back to Tanner colouring),
    or the historical ranked left-right synthesis (``lrc_ranked_v1``, needs
    ``sector_sizes``). ``prefer_coloring`` is accepted for signature
    compatibility and ignored. Profiles lower per 02 §6.3 (uniform / si1000 /
    helios); interim limit until 02 §6.4 lands: trivial H-mask only.
    """

    import hashlib as _hashlib
    import json as _json

    policy = sec_policy or DEFAULT_SEC_POLICY

    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    if hx.ndim != 2 or hz.ndim != 2 or hx.shape[1] != hz.shape[1]:
        raise ValueError("hx/hz must be 2D matrices with the same number of columns")
    mask = _normalize_h_mask(h_mask, int(hx.shape[1]))
    if np.any(mask):
        raise NotImplementedError(
            "sector_h under LRC awaits the (check_type, sector) gate table (02 §6.4)"
        )
    if policy.is_lrc and (sector_sizes is None or len(sector_sizes) != 2):
        raise ValueError("LRC synthesis needs the natural two-sector metadata (sector_sizes)")

    selected_rounds = select_memory_rounds(
        rounds=rounds, d_hint=d_hint, policy=str(rounds_policy or "max_Tc_d")
    )
    legacy_p = float(noise_point.value) if noise_point is not None else float(p)

    digest = _hashlib.sha256()
    digest.update(hx.tobytes() + hz.tobytes())
    digest.update(
        f"|sec:{policy.sec_policy_id}|K{policy.sec_candidates}"
        f"|I{policy.sec_cert_iters}|S{policy.sec_cert_seed}|".encode()
    )
    # v3.6: EVERY SecPolicyConfig field enters the key, so the certification
    # fields (sec_cert_seeds, sec_cert_iters_scale, sec_refine_*) can never
    # hit a v3.5 entry synthesized under a different certification.
    digest.update(
        b"|policy:"
        + _json.dumps(policy.to_dict(), sort_keys=True, default=str).encode()
    )
    if policy.is_lrc:
        digest.update(bytes(int(v) % 251 for v in sector_sizes))
    else:
        blocks_hash = (
            _hashlib.sha256(
                _json.dumps(layer_blocks, sort_keys=True).encode()
            ).hexdigest()[:16]
            if layer_blocks
            else "none"
        )
        digest.update(b"|blocks:" + blocks_hash.encode())
    key = digest.hexdigest()
    ranked = _SYNTH_CACHE.get(key)
    if ranked is None:
        ranked = _synth_disk_load(synth_cache_dir, key)
    if ranked is None:

        def _compute() -> Any:
            if policy.is_lrc:
                from omniqec_agentic.sec import rank_schedules, split_sectors

                sectors = split_sectors(hx, hz, list(sector_sizes))
                return rank_schedules(
                    sectors,
                    hx,
                    hz,
                    candidates=int(policy.sec_candidates),
                    isd_iters=int(policy.sec_cert_iters),
                    seed=int(key[:8], 16),
                )
            from omniqec_agentic.sec.twobasis_rank import (
                CertificationProgress,
                rank_twobasis_schedules,
            )

            # Review fix (docs/05 §4 M1 cost): every finished ISD probe is
            # checkpointed next to the cache entry, so a build killed by the
            # watchdog resumes the certification instead of restarting it
            # (identical values: seeds are per candidate / seed slot). The
            # checkpoint is removed once the full result is stored.
            progress = CertificationProgress(_synth_progress_path(synth_cache_dir, key))
            ranked_local = rank_twobasis_schedules(
                hx, hz, layer_blocks=layer_blocks, policy=policy, d_code_proxy=d_hint,
                progress=progress,
            )
            progress.discard()
            return ranked_local

        # M5: one synthesis per candidate across concurrent workers.
        ranked = _synthesize_with_lock(synth_cache_dir, key, _compute)
    _SYNTH_CACHE[key] = ranked
    # TB-4: schedule_degraded_x/z follow THIS caller's d_hint on every cache
    # hit (in-process, disk, or a disk hit inside the M5 lock); a fresh
    # synthesis already carries them and passes through unchanged.
    ranked = _with_recomputed_degradation(ranked, d_hint)
    if ranked.catastrophic:
        raise ScheduleCatastrophicError(
            "schedule_catastrophic: no schedule among the sampled candidates avoids "
            "a single-fault undetected logical (02 §3.3)"
        )
    schedule = ranked.schedule

    if policy.is_lrc:
        from omniqec_agentic.sec.audits import audit_static as _audit
        from omniqec_agentic.sec.stim_builder import build_lrc_memory_circuit as _build
    else:
        from omniqec_agentic.sec.audits import audit_twobasis_static as _audit
        from omniqec_agentic.sec.twobasis_builder import (
            build_twobasis_memory_circuit as _build,
        )

    kinds_x: list[str] = []
    kinds_z: list[str] = []
    circuit_x = _build(
        hx, hz, schedule, basis="X", rounds=selected_rounds, p=legacy_p,
        noise_point=noise_point, detector_kinds_out=kinds_x,
    )
    circuit_z = _build(
        hx, hz, schedule, basis="Z", rounds=selected_rounds, p=legacy_p,
        noise_point=noise_point, detector_kinds_out=kinds_z,
    )
    if len(kinds_x) != int(circuit_x.num_detectors) or len(kinds_z) != int(
        circuit_z.num_detectors
    ):
        raise RuntimeError("emitter recorded a detector kind count != num_detectors")
    # The static audit parses uniform CX/RX/MX structure; for lowered profiles
    # it runs on the uniform emission of the SAME schedule (profile-free).
    audit_circuit = circuit_x
    if noise_point is not None and noise_point.profile != NoiseProfile.UNIFORM_LEGACY:
        audit_circuit = _build(hx, hz, schedule, basis="X", rounds=selected_rounds, p=1e-3)
    static = _audit(audit_circuit, hx, hz, schedule, rounds=selected_rounds)
    if not static["valid"]:
        raise ValueError(f"sec static audit failed: {static['failures']}")
    dem_kwargs: dict[str, Any] = {"decompose_errors": decompose_errors}
    if noise_point is not None and noise_point.profile in (
        NoiseProfile.HELIOS,
        NoiseProfile.HELIOS_SYMMETRIZED,
        NoiseProfile.HELIOS_PAULI_SHARPENED,
    ):
        # PAULI_CHANNEL_2 components are validated <= this threshold in
        # HeliosNoiseConfig, so the disjoint approximation is sound.
        dem_kwargs["approximate_disjoint_errors"] = float(
            noise_point.approximate_disjoint_errors
        )
    dem_x = circuit_x.detector_error_model(**dem_kwargs)
    dem_z = circuit_z.detector_error_model(**dem_kwargs)
    return MemoryExperimentPair(
        x_memory=MemoryExperiment(
            basis="X",
            rounds=selected_rounds,
            p=legacy_p,
            circuit=circuit_x,
            detector_error_model=dem_x,
            schedule=schedule,
            num_data_qubits=int(hx.shape[1]),
            num_checks=int(hx.shape[0]),
            num_logicals=int(dem_x.num_observables),
            noise_point=noise_point,
            deformation=deformation,
            h_mask=tuple(int(value) for value in mask),
            detector_kinds=tuple(kinds_x),
        ),
        z_memory=MemoryExperiment(
            basis="Z",
            rounds=selected_rounds,
            p=legacy_p,
            circuit=circuit_z,
            detector_error_model=dem_z,
            schedule=schedule,
            num_data_qubits=int(hz.shape[1]),
            num_checks=int(hz.shape[0]),
            num_logicals=int(dem_z.num_observables),
            noise_point=noise_point,
            deformation=deformation,
            h_mask=tuple(int(value) for value in mask),
            detector_kinds=tuple(kinds_z),
        ),
        d_hint=d_hint,
        ranked=ranked,
    )


# Fixed search-phase round count (fairness protocol, 2026-08-16): a single-T
# per-round rate carries an O(1/T) boundary-transient underestimate, so
# per-candidate T (the old max(3, min(d, 8))) systematically flatters small-d
# codes in cross-candidate rankings. Every search-phase candidate AND the
# resident baseline now share one fixed T per campaign (>= the experiment's
# max plausible distance; 12 = the BB144 reference ceiling, matching d_cap);
# final_slow re-measures survivors at the protocol-frozen fixed T
# (BB paper: "Nc=d ... common practice").
DEFAULT_SEARCH_ROUNDS = 12


def select_memory_rounds(*, rounds: int | None, d_hint: int | None, policy: str = "max_Tc_d") -> int:
    """Rounds policy ``max_Tc_d`` (config.ROUNDS_POLICY_ID, 2026-09-10):
    T = max(T_c, d_search). ``rounds`` is the campaign floor T_c
    (DEFAULT_SEARCH_ROUNDS when None); a candidate whose proxy distance
    exceeds the floor is measured at T = d_search so its time-like errors are
    not truncated, while every candidate with d <= T_c shares the same depth
    (the v3.3 fairness rule). Search and final stages apply the same rule.

    ``fixed_Tc`` (2026-10-15, the Fig.3 formal run): T = T_c for every
    candidate, whatever its d_search -- the task circuit has a fixed depth."""

    floor = int(DEFAULT_SEARCH_ROUNDS) if rounds is None else int(rounds)
    if floor < 1:
        raise ValueError("rounds must be >= 1")
    if policy == "fixed_Tc":
        return floor
    if policy != "max_Tc_d":
        raise ValueError(f"unknown rounds policy {policy!r}")
    if d_hint is None:
        return floor
    return max(floor, int(d_hint))


def _normalize_h_mask(
    h_mask: np.ndarray | list[int] | tuple[int, ...] | None,
    n_data: int,
) -> np.ndarray:
    if h_mask is None:
        return np.zeros(int(n_data), dtype=np.uint8)
    raw = np.asarray(h_mask).reshape(-1)
    if raw.size != int(n_data):
        raise ValueError("Hadamard mask length must equal N_data")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("Hadamard mask must be binary")
    return raw.astype(np.uint8)



def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
