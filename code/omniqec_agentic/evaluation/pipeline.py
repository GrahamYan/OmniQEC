"""Single-candidate slow pipeline: build -> sec synthesis (cached) -> DEM ->
decode -> evidence (03 §7). Mechanically extracted from slow.py."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.config import HardwareTarget, VerificationConfig
from omniqec_agentic.construction import build_compiled_code, compiled_code_metadata
from omniqec_agentic.verification.circuit import (
    build_memory_experiment_pair,
    is_schedule_catastrophic_reason,
)
from omniqec_agentic.verification.decoders import (
    build_observable_decoder,
    estimate_circuit_logical_error_rate,
)
from omniqec_agentic.verification.dem import dem_to_matrices
from omniqec_agentic.verification.distance import (
    certify_css_code_distance,
    exact_circuit_distance_from_pair,
    isd_circuit_distance_from_pair,
)
from omniqec_agentic.verification.fast import evaluate_proxy
from omniqec_agentic.verification.ler import combine_basis_ler_results
from omniqec_agentic.verification.structural import structural_check_spec

from .common import (
    SLOW_SCHEMA_VERSION,
    decoder_name,
    hardware_value,
    noise_result_fields,
    resolve_deformation_mode,
    tail,
)
from .evidence import annotate_slow_checks, sec_ext_evidence

WORKER_MODULE = "omniqec_agentic.evaluation.worker"


def evaluate_slow_spec(
    *,
    program_id: str,
    spec: dict[str, Any],
    verification: VerificationConfig | None = None,
    hardware: HardwareTarget | str | None = None,
    seed: int = 0,
    batch_size: int = 1024,
    prefer_coloring: bool = True,
    promotion_record: dict[str, Any] | None = None,
    isolated: bool = True,
    deformation_mode: str | None = None,
    stage: str = "search",
) -> dict[str, Any]:
    """Run slow verification with a ver1-style hard timeout boundary.

    ``stage`` ("search" | "final", v3.6) is recorded on every basis cell
    exactly as the job pool's shard requests do; the serial path used to
    hard-code "search", so a final stage that landed on it (workers=1)
    silently measured with the search limit."""

    verification = verification or VerificationConfig()
    stage = _check_stage(stage)
    if not isolated or float(verification.circuit_timeout_s or 0.0) <= 0.0:
        return evaluate_slow_spec_inprocess(
            program_id=program_id,
            spec=spec,
            verification=verification,
            hardware=hardware,
            seed=seed,
            batch_size=batch_size,
            prefer_coloring=prefer_coloring,
            promotion_record=promotion_record,
            deformation_mode=deformation_mode,
            stage=stage,
        )
    return _evaluate_slow_spec_subprocess(
        program_id=program_id,
        spec=spec,
        verification=verification,
        hardware=hardware,
        seed=seed,
        batch_size=batch_size,
        prefer_coloring=prefer_coloring,
        promotion_record=promotion_record,
        deformation_mode=deformation_mode,
        stage=stage,
    )


DECODE_STAGES = ("search", "final")


def _check_stage(stage: str | None) -> str:
    value = str(stage or "search")
    if value not in DECODE_STAGES:
        raise ValueError(f"stage must be one of {DECODE_STAGES} (got {stage!r})")
    return value


def evaluate_slow_spec_inprocess(
    *,
    program_id: str,
    spec: dict[str, Any],
    verification: VerificationConfig | None = None,
    hardware: HardwareTarget | str | None = None,
    seed: int = 0,
    batch_size: int = 1024,
    prefer_coloring: bool = True,
    promotion_record: dict[str, Any] | None = None,
    deformation_mode: str | None = None,
    stage: str = "search",
) -> dict[str, Any]:
    """Run circuit-level noisy sampling for one concrete CSS spec."""

    verification = verification or VerificationConfig()
    stage = _check_stage(stage)
    deformation_mode = resolve_deformation_mode(deformation_mode, promotion_record)
    start = time.perf_counter()
    structural = structural_check_spec(
        program_id,
        spec,
        require_positive_k=True,
        deformation_mode=deformation_mode,
    )
    # ONE proxy protocol source (audit fix #2): the slow re-check runs the
    # SAME frozen seeds/effort/algorithm as the search, from the same config.
    fast = evaluate_proxy(
        spec,
        iters=int(verification.structural.proxy_iters_per_seed),
        seeds=tuple(verification.structural.proxy_seeds),
        algorithm_id=str(verification.structural.proxy_algorithm_id),
        dsqrtn_floor=float(verification.dsqrtn_floor),
        deformation_mode=deformation_mode,
    )
    payload: dict[str, Any] = {
        "schema": "slow_verification_run",
        "schema_version": SLOW_SCHEMA_VERSION,
        "isolation": "inprocess",
        "program_id": program_id,
        "status": "started",
        "seed": int(seed),
        "hardware": hardware_value(hardware),
        "verification": verification.to_dict(),
        "promotion_record": promotion_record,
        "deformation_mode": deformation_mode or "none",
        "stage": stage,
        "spec": spec,
        "structural": structural.to_dict(),
        "fast": fast,
        "results": [],
    }
    if not structural.valid:
        payload["status"] = "structural_reject"
        payload["runtime_s"] = time.perf_counter() - start
        return payload

    try:
        compiled = build_compiled_code(
            spec,
            program_id=program_id,
            deformation_mode=deformation_mode,
        )
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
        payload["compiled_code"] = compiled_code_metadata(compiled)
        payload["matrices"] = {
            "hx": hx.astype(int).tolist(),
            "hz": hz.astype(int).tolist(),
        }
    except Exception as exc:
        payload["status"] = "build_error"
        payload["failure_reason"] = str(exc)
        payload["runtime_s"] = time.perf_counter() - start
        return payload

    d_hint = int(fast["d"]) if isinstance(fast, dict) and "d" in fast else None
    exact_distance: dict[str, Any] = {
        "schema": "exact_distance_evidence",
        "schema_version": SLOW_SCHEMA_VERSION,
        "code": certify_css_code_distance(
            hx,
            hz,
            d_proxy=d_hint,
            config=verification.exact_distance,
        ),
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
    if not compiled.is_css:
        exact_distance["code"]["distance_semantics"] = "local_clifford_inherited"
        exact_distance["code"]["css_ancestor_hash"] = compiled_code_metadata(compiled)[
            "css_ancestor_hash"
        ]
    for noise_index, noise_value in enumerate(verification.noise.values):
        noise_point = verification.noise.point(noise_index)
        try:
            pair = build_memory_experiment_pair(
                hx,
                hz,
                p=float(noise_value),
                rounds=verification.rounds,
                d_hint=d_hint,
                prefer_coloring=prefer_coloring,
                decompose_errors=False,
                noise_point=noise_point,
                h_mask=compiled.h_mask,
                deformation=compiled.deformation,
                sector_sizes=compiled.sector_sizes,
                sec_policy=verification.sec_policy,
                layer_blocks=compiled.layer_blocks,
                # 2026-10-27 fix: same rounds policy as the job-pool path
                rounds_policy=str(getattr(verification, "rounds_policy", "max_Tc_d")),
            )
        except Exception as exc:
            catastrophic = is_schedule_catastrophic_reason(exc)
            if catastrophic:
                # v3.2 §4.4: physics verdict, not an infrastructure error —
                # record it on the evidence channel so the candidate gets a
                # terminal row downstream instead of resurfacing every turn.
                payload["schedule_catastrophic"] = True
                if exact_distance["circuit_ext"].get("status") == "not_available":
                    exact_distance["circuit_ext"] = {
                        "schema": "sec_d_circ_ext_evidence",
                        "status": "catastrophic",
                        "d_circ_ext": None,
                        "schedule_catastrophic": True,
                    }
            payload["results"].append(
                {
                    "status": "schedule_catastrophic" if catastrophic else "circuit_error",
                    **noise_result_fields(noise_point),
                    "failure_reason": str(exc),
                    "schedule_catastrophic": catastrophic,
                }
            )
            continue
        if exact_distance["circuit"].get("status") == "not_available":
            exact_distance["circuit"] = exact_circuit_distance_from_pair(
                pair,
                config=verification.exact_distance,
            )
        if exact_distance["circuit_isd"].get("status") == "not_available":
            exact_distance["circuit_isd"] = isd_circuit_distance_from_pair(
                pair,
                config=verification.exact_distance,
            )
        if exact_distance["circuit_ext"].get("status") == "not_available":
            exact_distance["circuit_ext"] = sec_ext_evidence(pair.ranked)

        basis_payloads: dict[str, dict[str, Any]] = {}
        basis_matrices = {
            "X": dem_to_matrices(pair.x_memory.detector_error_model),
            "Z": dem_to_matrices(pair.z_memory.detector_error_model),
        }
        basis_metadata = {
            "X": pair.x_memory.to_metadata(),
            "Z": pair.z_memory.to_metadata(),
        }
        basis_circuits = {
            "X": str(pair.x_memory.circuit),
            "Z": str(pair.z_memory.circuit),
        }
        for decoder_index, decoder_entry in enumerate(verification.decoders):
            decoder = decoder_name(decoder_entry)
            decoder_payload = {
                "status": "started",
                **noise_result_fields(noise_point),
                "decoder": decoder.value,
                "sampling_source": "noisy_circuit",
                "rounds": pair.x_memory.rounds,
                "basis": {},
                "combined": {},
            }
            # Honor verification.bases (the pool path already does); GLOBAL
            # basis index (X=0, Z=1) so a Z-only run's seeds match the Z leg
            # of a both-basis run.
            for basis in tuple(verification.bases):
                basis_index = {"X": 0, "Z": 1}[basis]
                matrices = basis_matrices[basis]
                basis_seed = int(seed) + noise_index * 10_000 + decoder_index * 100 + basis_index
                built_decoder = None
                try:
                    # v3.6: the telescoping decoder consumes the per-detector
                    # check kinds recorded by the SEC emitter (GARI X/Z mask,
                    # docs/06 §1.1) and builds its systems from the circuit
                    # text (no materialized artifacts on the in-process path).
                    built_decoder = build_observable_decoder(
                        decoder, matrices, verification,
                        detector_kinds=(pair.x_memory if basis == "X" else pair.z_memory).detector_kinds,
                        basis=basis, stage=stage, seed=basis_seed,
                        circuit_text=basis_circuits[basis],
                    )
                    result = estimate_circuit_logical_error_rate(
                        built_decoder,
                        circuit_text=basis_circuits[basis],
                        shots=int(verification.shots),
                        seed=basis_seed,
                        batch_size=int(batch_size),
                        max_errors=verification.max_errors,
                    )
                    result_payload = result.to_dict()
                    result_payload["decode_stage"] = stage
                    result_payload["memory_experiment"] = basis_metadata[basis]
                except Exception as exc:
                    result_payload = {
                        "status": "decoder_error",
                        "decoder": decoder.value,
                        "shots": 0,
                        "failures": 0,
                        "logical_error_rate": None,
                        "runtime_s": 0.0,
                        "failure_reason": str(exc),
                        "memory_experiment": basis_metadata[basis],
                        "metrics": matrices.to_metadata(),
                    }
                finally:
                    if built_decoder is not None and hasattr(built_decoder, "close"):
                        built_decoder.close()
                decoder_payload["basis"][basis] = result_payload
            decoder_payload["combined"] = combine_basis_ler_results(
                decoder_payload["basis"],
                rounds=pair.x_memory.rounds,
                k=structural.k or 0,
                ci_z=float(verification.ler_ci_z),
            )
            decoder_payload["status"] = (
                "ok"
                if decoder_payload["combined"].get("block_ler") is not None
                else "incomplete"
            )
            basis_payloads[decoder.value] = decoder_payload
        payload["results"].extend(basis_payloads.values())

    payload["exact_distance"] = exact_distance
    annotate_slow_checks(payload, verification=verification)
    if any(result.get("status") == "ok" for result in payload["results"]):
        payload["status"] = "ok"
    elif payload.get("schedule_catastrophic"):
        payload["status"] = "schedule_catastrophic"
    else:
        payload["status"] = "no_valid_results"
    payload["runtime_s"] = time.perf_counter() - start
    return payload


def _evaluate_slow_spec_subprocess(
    *,
    program_id: str,
    spec: dict[str, Any],
    verification: VerificationConfig,
    hardware: HardwareTarget | str | None = None,
    seed: int = 0,
    batch_size: int = 1024,
    prefer_coloring: bool = True,
    promotion_record: dict[str, Any] | None = None,
    deformation_mode: str | None = None,
    stage: str = "search",
) -> dict[str, Any]:
    start = time.perf_counter()
    request = {
        "program_id": program_id,
        "spec": spec,
        "verification": verification.to_dict(),
        "hardware": hardware_value(hardware),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "prefer_coloring": bool(prefer_coloring),
        "promotion_record": promotion_record,
        "deformation_mode": resolve_deformation_mode(deformation_mode, promotion_record),
        "stage": _check_stage(stage),
    }
    cmd = [sys.executable, "-m", WORKER_MODULE]
    process = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    timeout_s = float(verification.circuit_timeout_s)
    try:
        stdout, stderr = process.communicate(json.dumps(request, default=float), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        from .pool import kill_process_group

        kill_process_group(process)
        return subprocess_failure_payload(
            program_id=program_id,
            verification=verification,
            hardware=hardware,
            promotion_record=promotion_record,
            status="timeout",
            failure_reason=f"slow verifier exceeded circuit_timeout_s={timeout_s:g}",
            runtime_s=time.perf_counter() - start,
            extra={"killed": True, "subprocess_timeout_s": timeout_s},
        )

    runtime_s = time.perf_counter() - start
    from .pool import parse_worker_json

    parsed = parse_worker_json(stdout)
    if parsed is None:
        return subprocess_failure_payload(
            program_id=program_id,
            verification=verification,
            hardware=hardware,
            promotion_record=promotion_record,
            status="subprocess_error",
            failure_reason="slow worker produced no parseable JSON",
            runtime_s=runtime_s,
            extra={
                "returncode": process.returncode,
                "stdout_tail": tail(stdout),
                "stderr_tail": tail(stderr),
            },
        )
    if process.returncode not in (0, None):
        parsed.setdefault("worker_returncode", process.returncode)
        parsed.setdefault("worker_stderr_tail", tail(stderr))
    parsed["isolation"] = "subprocess"
    parsed["worker_returncode"] = process.returncode
    parsed.setdefault("runtime_s", runtime_s)
    return parsed


def subprocess_failure_payload(
    *,
    program_id: str,
    verification: VerificationConfig,
    hardware: HardwareTarget | str | None,
    promotion_record: dict[str, Any] | None,
    status: str,
    failure_reason: str,
    runtime_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "slow_verification_run",
        "schema_version": SLOW_SCHEMA_VERSION,
        "isolation": "subprocess",
        "program_id": program_id,
        "status": status,
        "hardware": hardware_value(hardware),
        "verification": verification.to_dict(),
        "promotion_record": promotion_record,
        "failure_reason": failure_reason,
        "results": [],
        "runtime_s": float(runtime_s),
    }
    if extra:
        payload.update(extra)
    return payload
