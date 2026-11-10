"""Observable decoder adapters for circuit-level DEM verification."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.config import (
    TELESCOPING_VENDORED_COMMIT,
    BposdDecoderConfig,
    DecoderName,
    LsdDecoderConfig,
    RelayBpDecoderConfig,
    TelescopingDecoderConfig,
    VerificationConfig,
)

from .dem import (
    DemMatrices,
    dem_file_to_matrices,
    observables_from_error,
    sample_error,
    sample_error_batch,
)


@dataclass(frozen=True)
class DecoderDependencyReport:
    status: str
    modules: dict[str, bool]
    missing: list[str]
    notes: list[str]
    # v3.6: installed versions of the decoder dependencies (the vendored
    # telescoping-decoder package + commit, relay_bp, scipy, ldpc, stim) for
    # the protocol snapshot.
    versions: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "modules": dict(self.modules),
            "missing": list(self.missing),
            "notes": list(self.notes),
            "versions": dict(self.versions),
        }


@dataclass(frozen=True)
class DemLerResult:
    status: str
    decoder: DecoderName
    shots: int
    failures: int
    logical_error_rate: float | None
    runtime_s: float
    failure_reason: str = ""
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decoder": self.decoder.value,
            "shots": int(self.shots),
            "failures": int(self.failures),
            "logical_error_rate": self.logical_error_rate,
            "runtime_s": float(self.runtime_s),
            "failure_reason": self.failure_reason,
            "metrics": dict(self.metrics or {}),
        }


class BposdObservableDecoder:
    """BP+OSD decoder that predicts logical observable parities from DEM syndromes."""

    decoder_name = DecoderName.BPOSD

    def __init__(
        self,
        matrices: DemMatrices,
        config: BposdDecoderConfig | None = None,
    ):
        config = config or BposdDecoderConfig()
        try:
            from ldpc import BpOsdDecoder
        except Exception as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(f"ldpc BpOsdDecoder unavailable: {exc}") from exc

        self.matrices = matrices
        self.config = config
        self.decoder = BpOsdDecoder(
            matrices.H,
            channel_probs=matrices.probs,
            max_iter=int(config.max_iter),
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=int(config.osd_order),
        )

    @classmethod
    def from_dem_file(
        cls,
        path: str | Path,
        config: BposdDecoderConfig | None = None,
    ) -> "BposdObservableDecoder":
        return cls(dem_file_to_matrices(path), config=config)

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        correction = self.decoder.decode(np.asarray(syndrome, dtype=np.uint8))
        return observables_from_error(self.matrices, correction)

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        decoded = [self.decode_observables(syndrome) for syndrome in syndromes]
        if not decoded:
            return np.zeros((0, self.matrices.num_observables), dtype=np.uint8)
        return np.vstack(decoded).astype(np.uint8)


class LsdObservableDecoder:
    """BP+LSD decoder — the decoder ver1's d_circ_ler protocol used."""

    decoder_name = DecoderName.LSD

    def __init__(
        self,
        matrices: DemMatrices,
        config: LsdDecoderConfig | None = None,
    ):
        config = config or LsdDecoderConfig()
        try:
            from ldpc import BpLsdDecoder
        except Exception as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(f"ldpc BpLsdDecoder unavailable: {exc}") from exc

        self.matrices = matrices
        self.config = config
        self.decoder = BpLsdDecoder(
            matrices.H,
            channel_probs=matrices.probs,
            max_iter=int(config.max_iter),
            bp_method=config.bp_method,
            # lsd_method must be explicit: ldpc 2.4.1 otherwise silently
            # resets lsd_order to 0 (LSD_0), discarding the requested order.
            lsd_method=config.lsd_method,
            lsd_order=int(config.lsd_order),
            ms_scaling_factor=float(config.ms_scaling_factor),
        )

    @classmethod
    def from_dem_file(
        cls,
        path: str | Path,
        config: LsdDecoderConfig | None = None,
    ) -> "LsdObservableDecoder":
        return cls(dem_file_to_matrices(path), config=config)

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        correction = self.decoder.decode(np.asarray(syndrome, dtype=np.uint8))
        return observables_from_error(self.matrices, correction)

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        decoded = [self.decode_observables(syndrome) for syndrome in syndromes]
        if not decoded:
            return np.zeros((0, self.matrices.num_observables), dtype=np.uint8)
        return np.vstack(decoded).astype(np.uint8)


class BeamObservableDecoder:
    """Beam search decoder (Ye-Wecker-Delfosse) — the default primary decoder.

    Deterministic (no RNG; 02 §11.1); consumes the DEM check matrix with
    per-mechanism priors directly. Implementation lives in
    third_party/BeamSearchDecoder (path wired via a site-packages .pth)."""

    decoder_name = DecoderName.BEAM

    def __init__(
        self,
        matrices: DemMatrices,
        config: "BeamDecoderConfig | None" = None,
    ):
        from omniqec_agentic.config import BeamDecoderConfig

        config = config or BeamDecoderConfig()
        try:
            from beam_search_decoder import BeamSearchDecoder
        except Exception as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(
                f"beam_search_decoder unavailable (third_party/BeamSearchDecoder): {exc}"
            ) from exc

        self.matrices = matrices
        self.config = config
        self.decoder = BeamSearchDecoder(
            matrices.H,
            error_channel=list(np.asarray(matrices.probs, dtype=float)),
            beam_width=int(config.beam_width),
            max_rounds=int(config.max_rounds),
            initial_iters=int(config.initial_iters),
            iters_per_round=int(config.iters_per_round),
            num_results=int(config.num_results),
        )

    @classmethod
    def from_dem_file(
        cls,
        path: str | Path,
        config: "BeamDecoderConfig | None" = None,
    ) -> "BeamObservableDecoder":
        return cls(dem_file_to_matrices(path), config=config)

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        correction = self.decoder.decode(np.asarray(syndrome, dtype=np.uint8))
        return observables_from_error(self.matrices, correction)

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        decoded = [self.decode_observables(syndrome) for syndrome in syndromes]
        if not decoded:
            return np.zeros((0, self.matrices.num_observables), dtype=np.uint8)
        return np.vstack(decoded).astype(np.uint8)


class RelayBpObservableDecoder:
    """RelayBP decoder that predicts logical observable parities from DEM syndromes."""

    decoder_name = DecoderName.RELAYBP

    def __init__(
        self,
        matrices: DemMatrices,
        config: RelayBpDecoderConfig | None = None,
    ):
        config = config or RelayBpDecoderConfig()
        try:
            relay_bp = importlib.import_module("relay_bp")
        except Exception as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(f"relay_bp unavailable: {exc}") from exc

        self.matrices = matrices
        self.config = config
        self.decoder = relay_bp.RelayDecoderF64(
            matrices.H,
            error_priors=matrices.probs,
            alpha=float(config.alpha),
            gamma0=float(config.gamma0),
            pre_iter=int(config.pre_iter),
            num_sets=int(config.num_sets),
            set_max_iter=int(config.set_max_iter),
            gamma_dist_interval=tuple(config.gamma_dist_interval),
            stop_nconv=int(config.stop_nconv),
        )
        self.runner = relay_bp.ObservableDecoderRunner(
            self.decoder,
            matrices.L,
            include_decode_result=False,
        )

    @classmethod
    def from_dem_file(
        cls,
        path: str | Path,
        config: RelayBpDecoderConfig | None = None,
    ) -> "RelayBpObservableDecoder":
        return cls(dem_file_to_matrices(path), config=config)

    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        decoded = self.runner.decode_observables_batch(
            syndromes,
            parallel=bool(self.config.parallel),
        )
        return np.asarray(decoded, dtype=np.uint8)

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        return self.decode_observables_batch(np.asarray([syndrome], dtype=np.uint8))[0]


TELESCOPING_STAGES = ("S3A", "S3B", "S3C", "NC")
TELESCOPING_DECODE_STAGES = ("search", "final")
# In-process S3 driver state (n_procs == 1): which materialized system the
# package's process-local S3 module currently holds. One system at a time per
# process; switching reloads it (the worker decodes X then Z sequentially).
_S3_INPROCESS_STATE: dict[str, Any] = {"npz": None, "cfg_key": None}
# Relative directory (next to the DEM bundle) holding the materialized
# telescoping systems of one candidate: <bundle stem>_telescoping/<basis>/.
TELESCOPING_ARTIFACT_SUFFIX = "_telescoping"


def is_x_detector_mask(detector_kinds: Any) -> np.ndarray:
    """Bool mask (True = X-type) from the SEC emitter's per-detector kinds."""

    kinds = [str(kind).upper() for kind in detector_kinds]
    bad = sorted({kind for kind in kinds if kind not in ("X", "Z")})
    if bad:
        raise ValueError(f"detector kinds must be 'X' or 'Z'; got {bad}")
    return np.asarray([kind == "X" for kind in kinds], dtype=bool)


def _import_telescoping():
    try:
        return importlib.import_module("telescoping_decoder")
    except Exception as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError(
            "telescoping_decoder unavailable (pip install "
            "third_party/telescoping-decoder into this environment): "
            f"{exc}"
        ) from exc


def build_decoding_system(
    circuit_text: str,
    detector_kinds: Any,
    basis: str,
    *,
    workdir: str | os.PathLike[str] | None,
    use_gari: bool = True,
    verify: bool = True,
    verify_samples: int = 1024,
):
    """The package's ``DecodingSystem`` (original + GARI) for one noisy
    memory circuit. The X/Z detector mask comes from the SEC emitter's
    recorded kinds and is cross-checked against the package's own
    circuit-derived mask (a mismatch is an error, never a guess)."""

    import stim

    td = _import_telescoping()
    basis = str(basis).upper()
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if detector_kinds is None:
        raise ValueError(
            "telescoping_cpu needs the per-detector check kinds "
            "(MemoryExperiment.detector_kinds / bundle['detector_kinds'])"
        )
    circuit = stim.Circuit(str(circuit_text))
    mask = is_x_detector_mask(detector_kinds)
    if mask.shape[0] != int(circuit.num_detectors):
        raise ValueError(
            f"detector kinds ({mask.shape[0]}) do not match the circuit's "
            f"{int(circuit.num_detectors)} detectors"
        )
    package_mask = np.asarray(td.circuit_xz_detector_mask(circuit), dtype=bool)
    if not np.array_equal(mask, package_mask):
        raise ValueError(
            "the SEC emitter's detector kinds disagree with the telescoping "
            "package's circuit-derived X/Z detector mask"
        )
    dem = td.circuit_to_dem(circuit)
    return td.DecodingSystem.from_dem(
        dem,
        is_x_detector=mask,
        init_basis=basis,
        use_gari=bool(use_gari),
        verify=bool(verify),
        verify_samples=int(verify_samples),
        workdir=None if workdir is None else str(workdir),
    )


def telescoping_artifact_dir(matrices_path: str | os.PathLike[str]) -> str:
    stem, _ext = os.path.splitext(str(matrices_path))
    return stem + TELESCOPING_ARTIFACT_SUFFIX


def materialize_telescoping_artifacts(
    circuit_texts: dict[str, str],
    detector_kinds: dict[str, Any],
    matrices_path: str | os.PathLike[str],
    *,
    config: TelescopingDecoderConfig | None = None,
) -> dict[str, Any]:
    """Build the telescoping systems of both bases ONCE per candidate (at
    DEM-build time) and write the GARI npz next to the DEM bundle, so every
    decode shard loads it instead of re-running the GARI transform (~4 s
    for BB144, ~10 s for Mitten200 per basis). Paths are recorded relative
    to the bundle directory (a relocated run directory still resolves)."""

    config = config or TelescopingDecoderConfig()
    base_dir = os.path.dirname(os.path.abspath(str(matrices_path)))
    out_dir = telescoping_artifact_dir(matrices_path)
    result: dict[str, Any] = {}
    for basis in ("X", "Z"):
        workdir = os.path.join(out_dir, basis)
        os.makedirs(workdir, exist_ok=True)
        started = time.perf_counter()
        system = build_decoding_system(
            circuit_texts[basis],
            detector_kinds[basis],
            basis,
            workdir=workdir,
            use_gari=str(config.s3_system) == "gari",
        )
        entry: dict[str, Any] = {
            "package": str(config.package),
            "source_fingerprint": str(system.source_fingerprint),
            "n_detectors": int(system.n_detectors),
            "n_obs": int(system.n_obs),
            "original_columns": int(system.H.shape[1]),
            "build_s": round(time.perf_counter() - started, 3),
        }
        if system.gari is not None:
            gari_path = system.npz_path_for("gari")
            entry["gari"] = os.path.relpath(gari_path, base_dir)
            entry["gari_shape"] = [int(v) for v in system.gari.h.shape]
            entry["gari_nnz"] = int(system.gari.h.nnz)
        else:
            entry["matrices"] = os.path.relpath(system.npz_path_for("original"), base_dir)
        result[basis] = entry
    return result


def resolve_telescoping_artifacts(
    entry: dict[str, Any] | None, base_dir: str | os.PathLike[str]
) -> dict[str, Any] | None:
    """Absolute artifact paths of one basis entry (None when unusable)."""

    if not isinstance(entry, dict):
        return None
    resolved: dict[str, Any] = {}
    for key in ("gari", "matrices"):
        rel = entry.get(key)
        if not rel:
            continue
        path = rel if os.path.isabs(str(rel)) else os.path.join(str(base_dir), str(rel))
        if os.path.isfile(path):
            resolved[key] = path
    if not resolved:
        return None
    resolved["source_fingerprint"] = entry.get("source_fingerprint")
    return resolved


class TelescopingObservableDecoder:
    """``telescoping_cpu_v1`` (telescoping_structural_v3_6, docs/06 §1).

    Thin adapter around the Mitten paper's open-source telescoping decoder
    (``telescoping_decoder.TelescopingDecoder``, vendored package) run in its
    CPU configuration on the GARI transform of the TWO-BASIS DEM of one
    memory basis: S1/S2 (GPU) off, S3-A/B/C on ``s3_system`` with the
    package's variant tables, S4 (Gurobi) off, and a shot that no stage
    accepts (``Stage.NC``) is DECLARED a logical error (``nc_policy="fail"``;
    the estimator never reports such a shot as a success).

    The package derives every S3 seed from ``(base_seed, shot id)``; the
    adapter uses the shard seed as ``base_seed`` and the shot's index within
    the shard as its id, so a decode is a pure function of (matrices, shard
    seed, shot index) -- bit-identical across machines, pool sizes and
    reruns for the same (seed, shot_shards) protocol.

    Systems: ``artifacts`` (the GARI npz materialized at DEM-build time by
    ``materialize_telescoping_artifacts``) when given, else built from the
    noisy ``circuit_text`` in a private temporary directory. Both paths
    cross-check the emitter's detector kinds against the package's mask and
    the system's detector / observable counts against ``matrices``.

    Per-shot bookkeeping: ``last_stages`` / ``last_declared_failures`` for the
    most recent batch; cumulative ``stage_counts``, ``stage_failures`` (from
    ``record_batch_outcome``), ``stage_time_s`` (package diagnostics),
    ``declared_failures`` and the package's per-shot labels.

    Process model: with the protocol's ``n_procs=1`` the S3 stages run IN
    THIS PROCESS through the package's own per-shot stage functions
    (``telescoping_decoder.s3._s3_a/b/c_one_shot`` after its worker
    initializer ``_install_cfg`` / ``_ensure_init`` / ``_pool_init``) -- the
    same pure functions the package's pool workers call, driven by the same
    S3-A -> S3-B -> S3-C loop as ``TelescopingDecoder.decode``, so the result
    is identical by construction (tested bit-for-bit against the package
    pool) while no child process, forkserver or ``Pool.terminate`` race is
    involved: the campaign's slow pool already runs one shard per core.
    ``n_procs > 1`` uses the package's pool instead (local benchmarking);
    that path needs a non-daemonic caller with an importable ``__main__``."""

    decoder_name = DecoderName.TELESCOPING

    def __init__(
        self,
        matrices: DemMatrices,
        config: TelescopingDecoderConfig | None = None,
        *,
        detector_kinds: Any,
        basis: str,
        seed: int = 0,
        stage: str = "search",
        circuit_text: str | None = None,
        artifacts: dict[str, Any] | None = None,
        workdir: str | os.PathLike[str] | None = None,
    ):
        td = _import_telescoping()
        config = config or TelescopingDecoderConfig()
        basis = str(basis).upper()
        if basis not in ("X", "Z"):
            raise ValueError("telescoping_cpu needs the memory basis ('X' or 'Z')")
        stage = str(stage)
        if stage not in TELESCOPING_DECODE_STAGES:
            raise ValueError(f"stage must be one of {TELESCOPING_DECODE_STAGES}, got {stage!r}")
        self.matrices = matrices
        self.config = config
        self.basis = basis
        self.seed = int(seed)
        self.stage = stage
        self._tmpdir: str | None = None
        self._dec = None
        artifacts = dict(artifacts or {})
        gari_npz = artifacts.get("gari")
        matrices_npz = artifacts.get("matrices")
        if gari_npz or matrices_npz:
            system = td.DecodingSystem.from_npz(
                gari_npz=gari_npz if gari_npz and os.path.isfile(str(gari_npz)) else None,
                matrices_npz=matrices_npz if matrices_npz and os.path.isfile(str(matrices_npz)) else None,
            )
            self.system_source = "artifacts"
        else:
            if circuit_text is None:
                raise ValueError(
                    "telescoping_cpu needs the materialized GARI artifact "
                    "(bundle['telescoping']) or the noisy circuit text"
                )
            self._tmpdir = tempfile.mkdtemp(
                prefix="telescoping_", dir=None if workdir is None else str(workdir)
            )
            system = build_decoding_system(
                circuit_text,
                detector_kinds,
                basis,
                workdir=self._tmpdir,
                use_gari=str(config.s3_system) == "gari",
            )
            self.system_source = "circuit"
        if int(system.n_detectors) != int(matrices.H.shape[0]):
            self.close()
            raise ValueError(
                f"telescoping system has {int(system.n_detectors)} detectors, the DEM "
                f"matrices {int(matrices.H.shape[0])}"
            )
        if int(system.n_obs) != int(matrices.num_observables):
            self.close()
            raise ValueError(
                f"telescoping system has {int(system.n_obs)} observables, the DEM "
                f"matrices {int(matrices.num_observables)}"
            )
        if detector_kinds is not None and system.is_x_detector is not None:
            if not np.array_equal(is_x_detector_mask(detector_kinds), np.asarray(system.is_x_detector, dtype=bool)):
                self.close()
                raise ValueError(
                    "the telescoping system's X/Z detector mask disagrees with the "
                    "SEC emitter's detector kinds"
                )
        if system.init_basis is not None and str(system.init_basis).upper() != basis:
            self.close()
            raise ValueError(
                f"telescoping system was built for the {system.init_basis}-memory, "
                f"requested {basis}"
            )
        self._dec = td.TelescopingDecoder(system, config.to_telescope_config(base_seed=self.seed))
        self._dec.prepare()
        self._stage_enum = td.Stage
        self._inprocess = int(config.n_procs) == 1
        self._s3mod = importlib.import_module("telescoping_decoder.s3") if self._inprocess else None
        self.resolved_systems = {k: str(v) for k, v in dict(self._dec._resolved).items()}
        self.last_stages: list[str] = []
        self.last_declared_failures: np.ndarray = np.zeros(0, dtype=bool)
        self.stage_counts: dict[str, int] = {name: 0 for name in TELESCOPING_STAGES}
        self.stage_failures: dict[str, int] = {name: 0 for name in TELESCOPING_STAGES}
        self.stage_time_s: dict[str, float] = {name: 0.0 for name in TELESCOPING_STAGES}
        self.declared_failures = 0
        self.label_counts: dict[str, int] = {}
        self._shot_offset = 0

    # ------------------------------------------------------------------ api
    def decode_observables_batch(self, syndromes: np.ndarray) -> np.ndarray:
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        count = int(syndromes.shape[0])
        if count == 0:
            self.last_stages = []
            self.last_declared_failures = np.zeros(0, dtype=bool)
            return np.zeros((0, self.matrices.num_observables), dtype=np.uint8)
        shot_ids = np.uint64(self._shot_offset) + np.arange(count, dtype=np.uint64)
        self._shot_offset += count
        if self._inprocess:
            result = self._decode_inprocess(syndromes, shot_ids)
        else:
            result = self._dec.decode(syndromes, shot_ids=shot_ids)
        stage_enum = self._stage_enum
        stages = [stage_enum(int(code)).name for code in result.stage]
        declared = np.asarray(result.stage, dtype=np.int64) == int(stage_enum.NC)
        for name in stages:
            self.stage_counts[name] = self.stage_counts.get(name, 0) + 1
        for label in result.label:
            key = str(label) or "?"
            self.label_counts[key] = self.label_counts.get(key, 0) + 1
        for key, name in (("s1", "S1"), ("s2", "S2"), ("s3a", "S3A"), ("s3b", "S3B"), ("s3c", "S3C"), ("s4", "S4")):
            diag = result.diagnostics.get(key)
            if isinstance(diag, dict):
                self.stage_time_s[name] = self.stage_time_s.get(name, 0.0) + float(
                    diag.get("wall_s", diag.get("decode_s", 0.0)) or 0.0
                )
        self.declared_failures += int(declared.sum())
        self.last_stages = stages
        self.last_declared_failures = declared
        return np.asarray(result.obs_pred, dtype=np.uint8)

    def decode_observables(self, syndrome: np.ndarray) -> np.ndarray:
        return self.decode_observables_batch(np.asarray([syndrome], dtype=np.uint8))[0]

    def record_batch_outcome(self, mismatch: np.ndarray) -> None:
        """Attribute the wrong shots of the LAST batch to their stages."""

        mismatch = np.asarray(mismatch, dtype=bool).reshape(-1)
        for stage, wrong in zip(self.last_stages, mismatch):
            if wrong:
                self.stage_failures[stage] = self.stage_failures.get(stage, 0) + 1

    def stage_metrics(self) -> dict[str, Any]:
        system = self._dec.system if self._dec is not None else None
        gari: dict[str, Any] = {}
        if system is not None:
            gari = {"n_detectors": int(system.n_detectors), "n_obs": int(system.n_obs)}
            if getattr(system, "gari", None) is not None:
                gari["shape"] = [int(v) for v in system.gari.h.shape]
                gari["nnz"] = int(system.gari.h.nnz)
        return {
            "stage_counts": {k: int(v) for k, v in self.stage_counts.items()},
            "stage_failures": {k: int(v) for k, v in self.stage_failures.items()},
            "stage_time_s": {k: round(float(v), 4) for k, v in self.stage_time_s.items()},
            "declared_failures": int(self.declared_failures),
            "nc_policy": str(self.config.nc_policy),
            "decode_stage": self.stage,
            "systems": dict(self.resolved_systems),
            "system_source": str(self.system_source),
            "s3_driver": "inprocess" if self._inprocess else "package_pool",
            "package": str(self.config.package),
            "gari": gari,
            "label_counts": dict(self.label_counts),
        }

    # ------------------------------------------------ in-process S3 driver
    def _activate_s3_state(self) -> None:
        """Make the package's process-local S3 module hold THIS decoder's
        system and flat config (its worker initializer, run here)."""

        s3 = self._s3mod
        dec = self._dec
        flat = dec._flat_cfg
        npz = str(dec._s3_npz)
        # Everything but base_seed shapes the loaded state (alpha schedules,
        # matrices); the seed is read per shot from the installed cfg.
        cfg_key = repr(sorted((k, v) for k, v in vars(flat).items() if k != "base_seed"))
        state = _S3_INPROCESS_STATE
        if not s3._G.get("_ready") or state.get("npz") != npz or state.get("cfg_key") != cfg_key:
            s3._G.clear()
            s3._B.clear()
            s3._install_cfg(flat)
            s3._ensure_init(npz)
            s3._pool_init()
            state["npz"] = npz
            state["cfg_key"] = cfg_key
        else:
            s3._install_cfg(flat)

    def _decode_inprocess(self, syndromes: np.ndarray, shot_ids: np.ndarray):
        """``TelescopingDecoder.decode`` for the CPU protocol (S1/S2/S4 off)
        with the S3-A/B/C per-shot functions called in this process."""

        s3 = self._s3mod
        stage_enum = self._stage_enum
        self._activate_s3_state()
        count = int(syndromes.shape[0])
        n_obs = int(self._dec.system.n_obs)
        stage = np.zeros(count, dtype=np.int8)
        label = np.array([""] * count, dtype=object)
        obs_pred = np.zeros((count, n_obs), dtype=np.uint8)
        diagnostics: dict[str, Any] = {"n_shots": count}
        remaining = list(range(count))
        for key, fn, code in (
            ("s3a", s3._s3_a_one_shot, stage_enum.S3A),
            ("s3b", s3._s3_b_one_shot, stage_enum.S3B),
            ("s3c", s3._s3_c_one_shot, stage_enum.S3C),
        ):
            if not remaining:
                break
            started = time.perf_counter()
            n_in = len(remaining)
            deferred: list[int] = []
            for row in remaining:
                result = fn((syndromes[row], None, int(shot_ids[row])))
                if result.get("conv"):
                    stage[row] = int(code)
                    label[row] = str(result.get("label", ""))
                    if result.get("obs_pred") is not None:
                        obs_pred[row] = np.asarray(result["obs_pred"], dtype=np.uint8)
                else:
                    deferred.append(row)
            remaining = deferred
            diagnostics[key] = {
                "n_in": n_in,
                "n_accepted": n_in - len(remaining),
                "wall_s": time.perf_counter() - started,
            }
        diagnostics["s3"] = {"system": self.resolved_systems.get("s3")}
        for row in remaining:
            label[row] = "NC_no_ip"
        diagnostics["n_nc"] = len(remaining)
        from types import SimpleNamespace

        return SimpleNamespace(
            obs_pred=obs_pred, stage=stage, label=label, converged=stage != int(stage_enum.NC),
            le=None, diagnostics=diagnostics,
        )

    def close(self) -> None:
        """Release the package's worker pools (graceful close + join, never
        a bare ``terminate`` -- that raced with an idle worker into a hang
        under load) and drop the temp systems."""

        dec = self._dec
        if dec is not None:
            for attr in ("_s3_pool", "_s4_pool"):
                pool = getattr(dec, attr, None)
                if pool is None:
                    continue
                try:
                    pool.close()
                    pool.join()
                except Exception:  # pragma: no cover - best effort
                    try:
                        pool.terminate()
                        pool.join()
                    except Exception:
                        pass
                setattr(dec, attr, None)
            try:
                dec.close()
            except Exception:  # pragma: no cover - best effort
                pass
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self) -> "TelescopingObservableDecoder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def build_observable_decoder(
    decoder: DecoderName | str,
    matrices: DemMatrices,
    verification: VerificationConfig | None = None,
    *,
    detector_kinds: Any = None,
    basis: str | None = None,
    stage: str = "search",
    seed: int = 0,
    circuit_text: str | None = None,
    telescoping_artifacts: dict[str, Any] | None = None,
    workdir: str | os.PathLike[str] | None = None,
):
    """``detector_kinds`` / ``basis`` / ``seed`` / ``circuit_text`` /
    ``telescoping_artifacts`` are consumed by the v3.6 telescoping decoder
    only (a clear error when the system cannot be built); ``stage`` is
    recorded; the historical decoders ignore them all. Close a telescoping
    decoder (``close()``) when done: it owns a worker process."""

    verification = verification or VerificationConfig()
    name = decoder if isinstance(decoder, DecoderName) else DecoderName(str(decoder))
    if name is DecoderName.BEAM:
        return BeamObservableDecoder(matrices, verification.decoder_build.beam)
    if name is DecoderName.LSD:
        return LsdObservableDecoder(matrices, verification.decoder_build.lsd)
    if name is DecoderName.BPOSD:
        return BposdObservableDecoder(matrices, verification.decoder_build.bposd)
    if name is DecoderName.RELAYBP:
        return RelayBpObservableDecoder(matrices, verification.decoder_build.relaybp)
    if name is DecoderName.TELESCOPING:
        if basis is None:
            raise ValueError("telescoping_cpu needs basis= ('X' or 'Z')")
        if not telescoping_artifacts and detector_kinds is None:
            raise ValueError(
                "telescoping_cpu needs detector_kinds= (the per-detector check kinds "
                "recorded by the v3.6 SEC emitter; a pre-v3.6 DEM bundle has none and "
                "cannot be decoded by the telescoping decoder) or materialized artifacts"
            )
        return TelescopingObservableDecoder(
            matrices,
            verification.decoder_build.telescoping,
            detector_kinds=detector_kinds,
            basis=basis,
            seed=int(seed),
            stage=stage,
            circuit_text=circuit_text,
            artifacts=telescoping_artifacts,
            workdir=workdir,
        )
    raise ValueError(f"unsupported decoder: {decoder}")


def _dist_version(name: str) -> str | None:
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return None


def decoder_dependency_report() -> DecoderDependencyReport:
    modules = {
        "numpy": _has_module("numpy"),
        "scipy": _has_module("scipy"),
        "stim": _has_module("stim"),
        "ldpc": _has_module("ldpc"),
        "relay_bp": _has_module("relay_bp"),
        "telescoping_decoder": _has_module("telescoping_decoder"),
    }
    missing = [name for name, ok in modules.items() if not ok]
    notes: list[str] = []
    if not modules["stim"]:
        notes.append("stim is required to load or generate circuit-level DEMs")
    if not modules["ldpc"]:
        notes.append("ldpc is required for the dev-only bposd / lsd decoders")
    if not modules["relay_bp"]:
        notes.append("relay_bp is required for the dev-only relaybp decoder")
    if not modules["telescoping_decoder"]:
        notes.append(
            "telescoping_decoder (third_party/telescoping-decoder) is required for "
            "telescoping_cpu, the formal telescoping_structural_v3_6 decoder"
        )
    compiler = shutil.which(os.environ.get("CC") or "gcc") or shutil.which("cc")
    if modules["telescoping_decoder"] and compiler is None:
        notes.append(
            "no C compiler found (gcc/cc): the telescoping S3 kernels compile on "
            "first use unless prebuilt .so files are pointed to by CHECKSERIAL_BP_SO / "
            "RELAY_MEM_BP_SO"
        )
    versions = {
        "telescoping_decoder": _dist_version("telescoping-decoder") or _dist_version("telescoping_decoder"),
        "telescoping_commit": TELESCOPING_VENDORED_COMMIT,
        "relay_bp": _dist_version("relay_bp") or _dist_version("relay-bp"),
        "scipy": _dist_version("scipy"),
        "ldpc": _dist_version("ldpc"),
        "stim": _dist_version("stim"),
    }
    return DecoderDependencyReport(
        status="ok" if not missing else "missing_optional_dependencies",
        modules=modules,
        missing=missing,
        notes=notes,
        versions=versions,
    )


def _batch_bookkeeping(decoder, mismatch: np.ndarray, book: dict[str, Any] | None):
    """v3.6: honour declared failures (never a success) and attribute wrong
    shots to the telescoping stage that resolved them. Returns the corrected
    mismatch vector; ``book`` accumulates per-call stage counters."""

    declared = getattr(decoder, "last_declared_failures", None)
    if declared is not None and np.asarray(declared).shape[0] == mismatch.shape[0]:
        mismatch = mismatch | np.asarray(declared, dtype=bool)
    stages = getattr(decoder, "last_stages", None)
    if book is not None and stages is not None and len(stages) == mismatch.shape[0]:
        counts = book.setdefault("stage_counts", {})
        fails = book.setdefault("stage_failures", {})
        for stage, wrong in zip(stages, mismatch):
            counts[stage] = counts.get(stage, 0) + 1
            if wrong:
                fails[stage] = fails.get(stage, 0) + 1
        if declared is not None:
            book["declared_failures"] = int(book.get("declared_failures", 0)) + int(
                np.asarray(declared, dtype=bool).sum()
            )
        if hasattr(decoder, "record_batch_outcome"):
            decoder.record_batch_outcome(mismatch)
    return mismatch


def _finish_stage_metrics(decoder, metrics: dict[str, Any], book: dict[str, Any] | None, time_before: dict[str, float] | None) -> None:
    if book is None or not hasattr(decoder, "stage_counts"):
        return
    names = list(getattr(decoder, "stage_counts", {}).keys()) or list(book.get("stage_counts", {}))
    metrics["stage_counts"] = {k: int(book.get("stage_counts", {}).get(k, 0)) for k in names}
    metrics["stage_failures"] = {k: int(book.get("stage_failures", {}).get(k, 0)) for k in names}
    after = getattr(decoder, "stage_time_s", {}) or {}
    before = time_before or {}
    metrics["stage_time_s"] = {
        k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 4) for k in names
    }
    metrics["declared_failures"] = int(book.get("declared_failures", 0))
    if hasattr(decoder, "stage_metrics"):
        extra = decoder.stage_metrics()
        for key, value in extra.items():
            if key not in ("stage_counts", "stage_failures", "stage_time_s", "declared_failures"):
                metrics[key] = value


def estimate_dem_logical_error_rate(
    decoder,
    *,
    shots: int,
    seed: int = 0,
    batch_size: int = 1024,
    max_errors: int | None = None,
) -> DemLerResult:
    """Estimate DEM logical error rate by sampling mechanisms and decoding observables."""

    start = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    matrices = decoder.matrices
    failures = 0
    done = 0
    book: dict[str, Any] | None = {} if hasattr(decoder, "last_stages") else None
    time_before = dict(getattr(decoder, "stage_time_s", {}) or {})
    try:
        while done < int(shots):
            size = min(int(batch_size), int(shots) - done)
            if size <= 1:
                syndrome, true_observables, _ = sample_error(matrices, rng)
                predicted = decoder.decode_observables(syndrome)
                mismatch = np.asarray([not np.array_equal(predicted, true_observables)], dtype=bool)
                mismatch = _batch_bookkeeping(decoder, mismatch, book)
                failures += int(mismatch.sum())
                done += 1
            else:
                batch = sample_error_batch(matrices, rng, size)
                predicted = decoder.decode_observables_batch(batch.syndromes)
                mismatch = np.any(predicted != batch.observables, axis=1)
                mismatch = _batch_bookkeeping(decoder, mismatch, book)
                failures += int(mismatch.sum())
                done += int(size)
            if max_errors is not None and failures >= int(max_errors):
                break
    except Exception as exc:
        return DemLerResult(
            status="decoder_error",
            decoder=decoder.decoder_name,
            shots=done,
            failures=failures,
            logical_error_rate=None,
            runtime_s=time.perf_counter() - start,
            failure_reason=str(exc),
            metrics=matrices.to_metadata(),
        )

    ler = failures / done if done else None
    metrics = matrices.to_metadata()
    metrics["requested_shots"] = int(shots)
    metrics["actual_shots"] = int(done)
    metrics["sampling_complete"] = bool(done == int(shots))
    metrics["stopped_by_max_errors"] = bool(
        done < int(shots)
        and max_errors is not None
        and failures >= int(max_errors)
    )
    _finish_stage_metrics(decoder, metrics, book, time_before)
    return DemLerResult(
        status="ok",
        decoder=decoder.decoder_name,
        shots=done,
        failures=failures,
        logical_error_rate=ler,
        runtime_s=time.perf_counter() - start,
        metrics=metrics,
    )


def normalize_shot_range(shot_range, shots: int) -> tuple[int, int]:
    """``shot_range`` -> ``(start, stop)`` within ``[0, shots]``; None = all."""

    total = int(shots)
    if shot_range is None:
        return 0, total
    start, stop = (int(shot_range[0]), int(shot_range[1]))
    if start < 0 or stop < start or stop > total:
        raise ValueError(f"shot_range {list(shot_range)} is not within [0, {total}]")
    return start, stop


def estimate_circuit_logical_error_rate(
    decoder,
    *,
    circuit_text: str,
    shots: int,
    seed: int = 0,
    batch_size: int = 1024,
    max_errors: int | None = None,
    shot_range: tuple[int, int] | list[int] | None = None,
) -> DemLerResult:
    """Estimate LER by sampling the NOISY CIRCUIT and decoding with the DEM.

    Replaces DEM-mechanism sampling as the slow protocol's sampling source
    (2026-08-15): the Sinter migration gate showed the independent-mechanism
    DEM approximation underestimates LER by up to ~50% on hook-rich codes
    (correlated multi-detector faults), which would systematically distort
    exactly the schedule-quality signal the framework ranks on. Circuit
    sampling is exact physics; the decoder still consumes the DEM matrices.
    Deterministic given (circuit, seed): stim's sampler is seeded per shard.

    ``shot_range=(start, stop)`` (shard splitting, 2026-09-05): decode only
    the shots with index in ``[start, stop)`` of this shard's ``shots``-shot
    stream. The sampler still draws EVERY batch of the stream in the same
    ``batch_size`` sequence (a batch outside the range is drawn and
    discarded), so the syndromes of shot ``i`` are those the unsplit shard
    would decode, and a decoder that seeds per shot (the telescoping
    adapter's ``_shot_offset``) is told shot ``i``'s index explicitly. The
    union of the parts of one shard is therefore bit-identical to the
    unsplit shard: same syndromes, same per-shot seeds, same verdicts; only
    which process computes each shot changes. ``shots`` / ``failures`` count
    the decoded range; ``metrics.shot_range`` records it.
    """

    import stim

    start = time.perf_counter()
    matrices = decoder.matrices
    failures = 0
    done = 0       # decoded shots (within the range)
    sampled = 0    # shots drawn from the stream so far
    range_start, range_stop = normalize_shot_range(shot_range, int(shots))
    range_size = range_stop - range_start
    # v3.6: a decoder exposing per-shot stages (the telescoping decoder) gets
    # its declared failures honoured (never reported as success) and
    # per-stage counts / wrong-shot attribution recorded in the metrics.
    book: dict[str, Any] | None = {} if hasattr(decoder, "last_stages") else None
    time_before = dict(getattr(decoder, "stage_time_s", {}) or {})
    try:
        sampler = stim.Circuit(circuit_text).compile_detector_sampler(seed=int(seed))
        while sampled < int(shots):
            size = min(int(batch_size), int(shots) - sampled)
            detectors, observables = sampler.sample(size, separate_observables=True)
            batch_start = sampled
            sampled += int(size)
            if sampled <= range_start:
                continue            # entirely before the range: drawn, not decoded
            if batch_start >= range_stop:
                break               # nothing left to decode; later batches are never needed
            lo = max(0, range_start - batch_start)
            hi = min(int(size), range_stop - batch_start)
            if hasattr(decoder, "_shot_offset"):
                # per-shot seeds are a function of the shot's index within the
                # shard, never of how the batches were cut
                decoder._shot_offset = int(batch_start + lo)
            predicted = decoder.decode_observables_batch(
                np.asarray(detectors[lo:hi], dtype=np.uint8)
            )
            mismatch = np.any(predicted != np.asarray(observables[lo:hi], dtype=np.uint8), axis=1)
            mismatch = _batch_bookkeeping(decoder, mismatch, book)
            failures += int(mismatch.sum())
            done += int(hi - lo)
            if max_errors is not None and failures >= int(max_errors):
                break
    except Exception as exc:
        return DemLerResult(
            status="decoder_error",
            decoder=decoder.decoder_name,
            shots=done,
            failures=failures,
            logical_error_rate=None,
            runtime_s=time.perf_counter() - start,
            failure_reason=str(exc),
            metrics=matrices.to_metadata(),
        )

    ler = failures / done if done else None
    metrics = matrices.to_metadata()
    metrics["requested_shots"] = int(range_size)
    metrics["actual_shots"] = int(done)
    metrics["sampling_complete"] = bool(done == int(range_size))
    metrics["stopped_by_max_errors"] = bool(
        done < int(range_size) and max_errors is not None and failures >= int(max_errors)
    )
    if shot_range is not None:
        metrics["shot_range"] = [int(range_start), int(range_stop)]
        metrics["shard_shots"] = int(shots)
    _finish_stage_metrics(decoder, metrics, book, time_before)
    return DemLerResult(
        status="ok",
        decoder=decoder.decoder_name,
        shots=done,
        failures=failures,
        logical_error_rate=ler,
        runtime_s=time.perf_counter() - start,
        metrics=metrics,
    )


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --------------------------------------------------- decoder_unsupported
# v3.6.1 (2026-09-05, GPT b360 turn 0): the telescoping package refuses a
# DEM in which two error mechanisms have identical detector support but
# different logical-observable support -- a property of the CANDIDATE (its
# circuit-level distance is 2: two faults that no detector tells apart flip
# different logicals), not of the infrastructure. Like schedule_catastrophic
# (d_circ = 1) and decode_infeasible (frozen budget) it is a terminal
# candidate verdict: recorded, excluded from ranking and from every later
# slow queue, never a measurement_blocked stop of the whole turn.
DECODER_UNSUPPORTED_MARKERS = ("identical detector support", "different logical-observable")


def is_decoder_unsupported_reason(reason: object) -> bool:
    """True when a decoder build failure is a verdict on the candidate's DEM
    (every marker in DECODER_UNSUPPORTED_MARKERS present), not a missing
    package, a mask mismatch or any other infrastructure error."""

    text = str(reason or "")
    return all(marker in text for marker in DECODER_UNSUPPORTED_MARKERS)
