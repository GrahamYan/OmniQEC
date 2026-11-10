"""Subprocess entry point for hard-timeout slow verification (03 §7).

Modes:
  (default)        full single-candidate verification (in-process path)
  "build_dem"      build the candidate's X/Z DEMs once and pickle them
  "shard_dem"      decode one shard from the pickled DEM (per-unit jobs)
  "distance_job"   exact/ISD/d_circ_ext distance evidence for one candidate

Invoked as ``python -m omniqec_agentic.evaluation.worker`` with one JSON
request on stdin and one JSON payload on stdout (last line wins)."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

from omniqec_agentic.config import DecoderName
from omniqec_agentic.construction import build_compiled_code, compiled_code_metadata
from omniqec_agentic.evaluation.task_circuits import task_metadata
from omniqec_agentic.verification.circuit import build_memory_experiment_pair
from omniqec_agentic.verification.decoders import (
    build_observable_decoder,
    estimate_circuit_logical_error_rate,
    materialize_telescoping_artifacts,
    resolve_telescoping_artifacts,
)
from omniqec_agentic.verification.distance import (
    certify_css_code_distance,
    exact_circuit_distance_from_pair,
    isd_circuit_distance_from_pair,
)

from .common import (
    decoder_name,
    int_or_none,
    noise_result_fields,
    resolve_deformation_mode,
    verification_from_dict,
)
from .dem_cache import dump_dem_bundle, load_dem_bundle
from omniqec_agentic.verification.dem import dem_to_matrices
from omniqec_agentic.verification.circuit import is_schedule_catastrophic_reason
from omniqec_agentic.verification.decoders import is_decoder_unsupported_reason

from .evidence import sec_ext_evidence
from .sampling import shard_seed


def telescoping_build_failure(exc: BaseException) -> dict[str, Any]:
    """Status fields for a failed telescoping system build. A DEM the
    decoder cannot represent (identical detector support, different logical
    support) is the candidate's own terminal verdict, ``decoder_unsupported``
    (v3.6.1); anything else stays an infrastructure ``build_error``."""

    reason = f"telescoping system build failed: {exc}"
    if is_decoder_unsupported_reason(exc):
        return {"status": "decoder_unsupported", "decoder_unsupported": True, "failure_reason": reason}
    return {"status": "build_error", "failure_reason": reason}


def run_build_dem_request(request: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: build the X/Z memory DEMs ONCE for a candidate and pickle
    them so every decode job just LOADS the matrices instead of rebuilding.

    This is the "build once per candidate" half of the architecture: the
    expensive circuit + DEM build happens a single time per (candidate, noise
    point); the per-(decoder, basis, shard) decode jobs each load the pickle
    and run one small slice, so no job rebuilds and no job is the 6-decoder
    bundle that overran the timeout."""

    verification = verification_from_dict(request.get("verification"))
    spec = request["spec"]
    noise_index = int(request.get("noise_index", request.get("p_index", 0)) or 0)
    noise_point = verification.noise.point(noise_index)
    noise_value = float(request.get("noise_value", request.get("p", noise_point.value)))
    base = {"schema": "build_dem_result", "matrices_path": request.get("matrices_path")}
    try:
        compiled = build_compiled_code(
            spec,
            deformation_mode=resolve_deformation_mode(request.get("deformation_mode")),
        )
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
        pair = build_memory_experiment_pair(
            hx, hz, p=noise_value, rounds=verification.rounds,
            d_hint=int_or_none(request.get("d_hint")),
            prefer_coloring=bool(request.get("prefer_coloring", True)),
            decompose_errors=False,
            noise_point=noise_point,
            h_mask=compiled.h_mask,
            deformation=compiled.deformation,
            sector_sizes=compiled.sector_sizes,
            synth_cache_dir=request.get("synth_cache_dir"),
            # v3.5 SEC: the policy travels in the verification config, the
            # block structure on the compiled code (None for hgp -> fallback
            # colouring). Absent sec_policy in the request -> default twobasis.
            sec_policy=verification.sec_policy,
            layer_blocks=compiled.layer_blocks,
            # 2026-10-27 fix: the SAME rounds policy as the pool's ctx["rounds"]
            # (fixed_Tc -> T = T_c for every code); without it a fixed_Tc
            # candidate with d_hint > T_c was built at T = d_hint.
            rounds_policy=str(getattr(verification, "rounds_policy", "max_Tc_d")),
        )
        # Fig.3 Phase B: the slow loop measures the gate-dense two-block
        # circuit built on this schedule and these rounds (task_circuits.py).
        if str(getattr(verification, "task", "memory")) == "cnot_dense":
            from omniqec_agentic.evaluation.task_circuits import task_experiment_pair

            pair = task_experiment_pair(pair, hx, hz, noise_point=noise_point)
    except Exception as exc:
        # A catastrophic schedule is a physics verdict, not a build failure:
        # mark it so the parent records a candidate-level terminal state
        # instead of an infrastructure error (v3.2 §4.4).
        if is_schedule_catastrophic_reason(exc):
            return {
                **base,
                "status": "schedule_catastrophic",
                "schedule_catastrophic": True,
                "failure_reason": str(exc),
            }
        return {**base, "status": "build_error", "failure_reason": str(exc)}
    # v3.6 (telescoping_structural_v3_6): materialize the telescoping
    # decoder's systems (original + GARI npz) ONCE per candidate next to the
    # bundle, so no decode shard re-runs the GARI transform; a system that
    # cannot be built (package missing, detector-kind mask mismatch) fails
    # the build here instead of in every shard.
    telescoping_info: dict[str, Any] = {"status": "skipped", "bases": {}}
    if DecoderName.TELESCOPING in tuple(verification.decoders):
        try:
            telescoping_info = {
                "status": "ok",
                "bases": materialize_telescoping_artifacts(
                    {"X": str(pair.x_memory.circuit), "Z": str(pair.z_memory.circuit)},
                    {
                        "X": list(pair.x_memory.detector_kinds),
                        "Z": list(pair.z_memory.detector_kinds),
                    },
                    request["matrices_path"],
                    config=verification.decoder_build.telescoping,
                ),
            }
        except Exception as exc:
            return {**base, **telescoping_build_failure(exc)}
    try:
        dump_dem_bundle(
            request["matrices_path"], pair, noise_point, extra={"telescoping": telescoping_info}
        )
    except Exception as exc:
        return {**base, "status": "pickle_error", "failure_reason": str(exc)}
    return {
        **base, "status": "ok",
        "telescoping": {
            "status": telescoping_info.get("status"),
            "bases": {
                basis: {k: v for k, v in entry.items() if k != "source_fingerprint"}
                for basis, entry in (telescoping_info.get("bases") or {}).items()
            },
        },
        "rounds_x": int(pair.x_memory.rounds), "rounds_z": int(pair.z_memory.rounds),
        # Fig.3 Phase B: the logical qubits of the MEASURED circuit (k for the
        # memory task, 2k for cnot_dense); the pool's A4/A5 combination roots
        # the block failure over these, not over the code's k.
        "num_logicals": {"X": int(pair.x_memory.num_logicals), "Z": int(pair.z_memory.num_logicals)},
        "task": task_metadata(str(getattr(verification, "task", "memory")), int(pair.x_memory.rounds)),
        "noise_point": noise_point.to_dict(),
        # v3.6 (docs/06 §1.2 / B5): ORIGINAL two-basis DEM column count per
        # basis; the pool scales the decode-shard watchdog by
        # max(1, cols / DECODE_BUDGET_SCALE_COLS) and the telescoping
        # worst-case shot model (the GARI matrix is ~1.25x wider; its shape
        # is reported above and per shard under metrics["gari"]).
        "num_matrix_columns": {
            "X": int(dem_to_matrices(pair.x_memory.detector_error_model).H.shape[1]),
            "Z": int(dem_to_matrices(pair.z_memory.detector_error_model).H.shape[1]),
        },
    }


# Persistent workers (2026-09-14, ``--serve``): one process decodes many
# shot-range chunks of the same candidate, so the pickled bundle is kept in a
# tiny in-process cache keyed by (path, mtime, size). A one-shot worker sees a
# single miss, exactly as before. The bundle is read-only for the shard path.
_BUNDLE_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_BUNDLE_CACHE_MAX = 4


def _load_bundle_cached(path: str) -> dict[str, Any]:
    stat = os.stat(path)
    key = (int(stat.st_mtime_ns), int(stat.st_size))
    hit = _BUNDLE_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    bundle = load_dem_bundle(path)
    if len(_BUNDLE_CACHE) >= _BUNDLE_CACHE_MAX:
        _BUNDLE_CACHE.pop(next(iter(_BUNDLE_CACHE)))
    _BUNDLE_CACHE[path] = (key, bundle)
    return bundle


def run_shard_dem_request(request: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: decode one shard from a PRE-BUILT pickled DEM (no rebuild).

    The candidate's X/Z DEM matrices are built ONCE by ``run_build_dem_request``
    and pickled; this job LOADS the pickle (cheap) and runs only the explicit
    (decoder, basis) ``units`` it is handed. The parent splits a candidate into
    per-(decoder, basis, shard) jobs so each is small (well under the job
    timeout). Each unit carries its GLOBAL decoder/basis index, so its seed is
    ``shard_seed(base_seed + decoder_index*100 + basis_index, shard, shards)`` —
    identical to the build-per-job path; with the exact pickled probs the result
    is bit-identical for the same (seed, shot_shards) protocol under the
    deterministic beam decoder. Caveat (measured 2026-08-15): ldpc's
    BpLsdDecoder LSD cluster tie-breaking is address-nondeterministic — two
    fresh instances on identical shots can differ by ~1/2000 failures — so LSD
    cells reproduce statistically, not bitwise."""

    verification = verification_from_dict(request.get("verification"))
    shard = int(request.get("shard") or 0)
    shot_shards = int(request.get("shot_shards") or 1)
    shots = int(request["shots"])
    base_seed = int(request["base_seed"])
    noise_index = int(request.get("noise_index", request.get("p_index", 0)) or 0)
    noise_point = verification.noise.point(noise_index)
    rounds = {"X": int(request["rounds_x"]), "Z": int(request["rounds_z"])}
    units = request["units"]
    base = {
        "schema": "shard_result",
        **noise_result_fields(noise_point),
        "shard": shard,
        "noise_index": noise_index,
    }
    try:
        bundle = _load_bundle_cached(request["matrices_path"])
    except Exception as exc:
        return {**base, "status": "dem_load_error", "failure_reason": str(exc), "results": {}}
    meta = bundle.get("meta") or {}
    max_errors = int_or_none(request.get("max_errors"))
    # v3.6: per-detector check kinds recorded at build time (None for a
    # pre-v3.6 bundle), the materialized telescoping systems (None -> the
    # shard builds them from the circuit text) and the decode stage
    # ("search" | "final"), recorded per cell (docs/06 §1.1).
    detector_kinds_all = bundle.get("detector_kinds") or {}
    telescoping_all = (bundle.get("telescoping") or {}).get("bases") or {}
    bundle_dir = os.path.dirname(os.path.abspath(str(request["matrices_path"])))
    stage = str(request.get("stage") or "search")
    results: dict[str, dict[str, Any]] = {}
    for unit in units:
        decoder = decoder_name(unit["decoder"])
        basis = str(unit["basis"])
        seed = shard_seed(
            base_seed + int(unit["decoder_index"]) * 100 + int(unit["basis_index"]),
            shard, shot_shards,
        )
        built = None
        try:
            built = build_observable_decoder(
                decoder, bundle[basis], verification,
                detector_kinds=detector_kinds_all.get(basis),
                basis=basis, stage=stage, seed=int(seed),
                circuit_text=bundle["circuit_text"][basis],
                telescoping_artifacts=resolve_telescoping_artifacts(
                    telescoping_all.get(basis), bundle_dir
                ),
            )
            res = estimate_circuit_logical_error_rate(
                built,
                circuit_text=bundle["circuit_text"][basis],
                shots=shots, seed=seed,
                batch_size=int(request.get("batch_size") or 1024), max_errors=max_errors,
                # shard splitting (2026-09-05): this job decodes only
                # [start, stop) of the shard's stream; the pool folds the
                # parts back into one shard cell (sampling.merge_subshard_results).
                shot_range=request.get("shot_range"),
            ).to_dict()
            res["rounds"] = int(rounds[basis])
            res["decode_stage"] = stage
            res["memory_experiment"] = meta.get(basis)
            if request.get("shot_range") is not None:
                res["shot_range"] = [int(request["shot_range"][0]), int(request["shot_range"][1])]
        except Exception as exc:
            res = {"status": "decoder_error", "decoder": decoder.value, "shots": 0,
                   "failures": 0, "logical_error_rate": None, "runtime_s": 0.0,
                   "failure_reason": str(exc), "metrics": bundle[basis].to_metadata()}
        finally:
            # The telescoping decoder owns a worker process: release it.
            if built is not None and hasattr(built, "close"):
                built.close()
        results.setdefault(decoder.value, {})[basis] = res
    return {**base, "status": "ok", "results": results}


def run_distance_job_request(request: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: exact/ISD/d_circ_ext distance evidence for one candidate."""

    verification = verification_from_dict(request.get("verification"))
    spec = request["spec"]
    evidence: dict[str, Any] = {
        "schema": "distance_job_result",
        "code": None,
        "circuit": {
            "schema": "exact_circuit_distance_evidence",
            "status": "not_available",
            "d_circ_exact": None,
            "certified": False,
        },
        "circuit_isd": {
            "schema": "isd_dem_circuit_distance_evidence",
            "status": "not_available",
            "d_circ_isd_upper_bound": None,
            "certified": False,
        },
        "circuit_ext": sec_ext_evidence(None),
    }
    try:
        compiled = build_compiled_code(
            spec,
            deformation_mode=resolve_deformation_mode(request.get("deformation_mode")),
        )
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
    except Exception as exc:
        evidence["status"] = "build_error"
        evidence["failure_reason"] = str(exc)
        return evidence
    d_hint = int_or_none(request.get("d_hint"))
    evidence["code"] = certify_css_code_distance(hx, hz, d_proxy=d_hint, config=verification.exact_distance)
    if not compiled.is_css:
        evidence["code"]["distance_semantics"] = "local_clifford_inherited"
        evidence["code"]["css_ancestor_hash"] = compiled_code_metadata(compiled)[
            "css_ancestor_hash"
        ]
    evidence["compiled_code"] = compiled_code_metadata(compiled)
    try:
        noise_index = int(request.get("noise_index", request.get("p_index", 0)) or 0)
        noise_point = verification.noise.point(noise_index)
        noise_value = float(request.get("noise_value", request.get("p", noise_point.value)))
        pair = build_memory_experiment_pair(
            hx,
            hz,
            p=noise_value,
            rounds=verification.rounds,
            d_hint=d_hint,
            prefer_coloring=bool(request.get("prefer_coloring", True)),
            decompose_errors=False,
            noise_point=noise_point,
            h_mask=compiled.h_mask,
            deformation=compiled.deformation,
            sector_sizes=compiled.sector_sizes,
            synth_cache_dir=request.get("synth_cache_dir"),
            sec_policy=verification.sec_policy,
            layer_blocks=compiled.layer_blocks,
            # 2026-10-27 fix: distance evidence of the circuit actually measured
            rounds_policy=str(getattr(verification, "rounds_policy", "max_Tc_d")),
        )
    except Exception as exc:
        if is_schedule_catastrophic_reason(exc):
            evidence["status"] = "schedule_catastrophic"
            evidence["circuit_ext"] = {
                "schema": "sec_d_circ_ext_evidence",
                "status": "catastrophic",
                "d_circ_ext": None,
                "schedule_catastrophic": True,
            }
        else:
            evidence["status"] = "circuit_error"
        evidence["failure_reason"] = str(exc)
        return evidence
    evidence["circuit"] = exact_circuit_distance_from_pair(pair, config=verification.exact_distance)
    evidence["circuit_isd"] = isd_circuit_distance_from_pair(pair, config=verification.exact_distance)
    evidence["circuit_ext"] = sec_ext_evidence(pair.ranked)
    evidence["status"] = "ok"
    return evidence


def _dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Route one request to its handler (shared by the one-shot and serve modes).

    ``ping`` / ``sleep`` / ``crash`` are test-only modes for the persistent
    pool's dispatch, watchdog and respawn paths; the campaign never sends them."""

    mode = str(request.get("mode") or "candidate")
    if mode == "ping":
        return {"schema": "ping_result", "status": "ok", "echo": request.get("echo"), "pid": os.getpid()}
    if mode == "sleep":
        time.sleep(float(request.get("seconds") or 0.0))
        return {"schema": "sleep_result", "status": "ok", "pid": os.getpid()}
    if mode == "crash":
        sys.stderr.write("worker: test crash requested\n")
        sys.stderr.flush()
        os._exit(int(request.get("code") or 3))
    if mode == "build_dem":
        payload = run_build_dem_request(request)
    elif mode == "shard_dem":
        payload = run_shard_dem_request(request)
    elif mode == "distance_job":
        payload = run_distance_job_request(request)
    else:
        from .pipeline import evaluate_slow_spec_inprocess

        verification = verification_from_dict(request.get("verification"))
        payload = evaluate_slow_spec_inprocess(
            program_id=str(request["program_id"]),
            spec=request["spec"],
            verification=verification,
            hardware=request.get("hardware"),
            seed=int(request.get("seed") or 0),
            batch_size=int(request.get("batch_size") or 1024),
            prefer_coloring=bool(request.get("prefer_coloring", True)),
            promotion_record=request.get("promotion_record"),
            deformation_mode=request.get("deformation_mode"),
            # v3.6: the same stage tag as the shard requests (S4 time limit).
            stage=str(request.get("stage") or "search"),
        )
    # v3.6 M4: peak RSS of this worker process (bytes; Linux reports KiB) so
    # the pool can record it per job and size the worker count by memory.
    return payload


def _with_rss(payload: dict[str, Any]) -> dict[str, Any]:
    # v3.6 M4: peak RSS of this worker process (bytes; Linux reports KiB) so
    # the pool can record it per job and size the worker count by memory.
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform.startswith("linux"):
            rss *= 1024
        if isinstance(payload, dict) and "ru_maxrss" not in payload:
            payload["ru_maxrss"] = rss
    except Exception:
        pass
    return payload


def _emit(payload: dict[str, Any], out: Any) -> None:
    out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    out.write("\n")
    out.flush()


def serve() -> int:
    """Persistent worker (2026-09-14): one JSON request per stdin line, one
    JSON payload per stdout line, until stdin closes. The process keeps its
    module state between requests (the vendored decoder's S3 state, the
    bundle cache), which is what makes 4-shot decode chunks affordable: the
    one-shot worker paid ~1.5-9 s of startup per chunk. Library prints are
    redirected to stderr so nothing but payloads reaches the protocol
    stream; an exception inside one request is answered as a
    ``worker_exception`` payload and the loop keeps serving."""

    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except Exception as exc:
            _emit({"schema": "worker_error", "status": "bad_request", "failure_reason": str(exc)[:300]}, real_stdout)
            continue
        try:
            payload = _dispatch(request)
        except BaseException as exc:  # noqa: BLE001 - keep serving, report the failure
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            payload = {
                "schema": "worker_error",
                "status": "worker_exception",
                "failure_reason": f"{type(exc).__name__}: {exc}"[:300],
            }
        _emit(_with_rss(payload if isinstance(payload, dict) else {"status": "bad_payload"}), real_stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--serve" in args:
        return serve()
    request = json.loads(sys.stdin.read())
    payload = _with_rss(_dispatch(request))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
