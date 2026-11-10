"""Thin Stim-circuit export on the sec/LRC seam (design doc 03 §12).

Replaces the deleted postsearch export path: for each candidate spec, write
the noisy X/Z memory circuits, their noiseless variants, the undecomposed
DEMs, and a JSON metadata sidecar. Artifact names follow the old convention
(``*_noisy.stim`` / ``*_noiseless.stim`` / ``*.dem`` / ``*_stim_export.json``)
so downstream packaging keeps working.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import SecPolicyConfig
from .construction import build_compiled_code, compiled_code_metadata
from .verification.circuit import build_memory_experiment_pair
from .verification.fast import evaluate_proxy
from .verification.noise import NoiseSweepConfig
from .verification.structural import structural_check_spec

EXPORT_SCHEMA_VERSION = 1


def export_stim_circuits_for_spec(
    *,
    program_id: str,
    spec: dict[str, Any],
    out_dir: str | Path,
    p: float = 0.005,
    rounds: int | None = None,
    proxy_iters: int = 120,
    noise: NoiseSweepConfig | None = None,
    sec_policy: SecPolicyConfig | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if noise is None:
        noise = NoiseSweepConfig.uniform((float(p),))
    if len(noise.values) != 1:
        raise ValueError("Stim export requires exactly one noise point")
    noise_point = noise.point(0)
    structural = structural_check_spec(program_id, spec, require_positive_k=True)
    payload: dict[str, Any] = {
        "schema": "stim_export",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "program_id": str(program_id),
        "status": "started",
        "spec": spec,
        "structural": structural.to_dict(),
        "fast": evaluate_proxy(spec, iters=int(proxy_iters)),
        "noise_point": noise_point.to_dict(),
        "rounds": rounds,
        "artifact_paths": {},
    }
    if not structural.valid:
        payload["status"] = "structural_reject"
        payload["runtime_s"] = time.perf_counter() - start
        return payload
    try:
        compiled = build_compiled_code(spec, program_id=program_id)
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
        payload["compiled_code"] = compiled_code_metadata(compiled)
        pair = build_memory_experiment_pair(
            hx, hz, p=float(noise_point.value), rounds=rounds,
            d_hint=payload["fast"].get("d"), noise_point=noise_point,
            sector_sizes=compiled.sector_sizes,
            sec_policy=sec_policy, layer_blocks=compiled.layer_blocks,
        )
        noiseless = build_memory_experiment_pair(
            hx, hz, p=0.0, rounds=rounds,
            d_hint=payload["fast"].get("d"),
            sector_sizes=compiled.sector_sizes,
            sec_policy=sec_policy, layer_blocks=compiled.layer_blocks,
        )
    except Exception as exc:
        payload["status"] = "build_error"
        payload["failure_reason"] = str(exc)
        payload["runtime_s"] = time.perf_counter() - start
        return payload
    for basis, experiment, clean in (
        ("X", pair.x_memory, noiseless.x_memory),
        ("Z", pair.z_memory, noiseless.z_memory),
    ):
        stem = f"{program_id}_{basis.lower()}"
        paths = {
            "noisy": out / f"{stem}_noisy.stim",
            "noiseless": out / f"{stem}_noiseless.stim",
            "dem": out / f"{stem}.dem",
        }
        paths["noisy"].write_text(str(experiment.circuit) + "\n", encoding="utf-8")
        paths["noiseless"].write_text(str(clean.circuit) + "\n", encoding="utf-8")
        paths["dem"].write_text(str(experiment.detector_error_model) + "\n", encoding="utf-8")
        payload["artifact_paths"][basis] = {k: str(v) for k, v in paths.items()}
        payload.setdefault("memory_experiments", {})[basis] = experiment.to_metadata()
    payload["sec_policy"] = (sec_policy.to_dict() if sec_policy is not None else None)
    payload["sec_policy_id"] = getattr(pair, "sec_policy_id", None)
    payload["status"] = "ok"
    payload["runtime_s"] = time.perf_counter() - start
    sidecar = out / f"{program_id}_stim_export.json"
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifact_paths"]["sidecar"] = str(sidecar)
    return payload


def export_stim_circuits_for_queue(
    *,
    queue_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    p: float = 0.005,
    rounds: int | None = None,
    noise: NoiseSweepConfig | None = None,
    sec_policy: SecPolicyConfig | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(queue_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    results = []
    for index, row in enumerate(rows, start=1):
        program_id = str(row.get("program_id") or f"queue_{index:04d}")
        spec = row.get("spec")
        if not isinstance(spec, dict):
            results.append({"program_id": program_id, "status": "missing_spec"})
            continue
        results.append(
            export_stim_circuits_for_spec(
                program_id=f"{index:04d}_{program_id}",
                spec=spec,
                out_dir=out / f"{index:04d}_{program_id}",
                p=p,
                rounds=rounds,
                noise=noise,
                sec_policy=sec_policy,
            )
        )
    summary = {
        "schema": "stim_export_queue",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "queue_path": str(queue_path),
        "out_dir": str(out),
        "count": len(results),
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "results": [
            {k: item.get(k) for k in ("program_id", "status", "failure_reason")}
            for item in results
        ],
    }
    (out / "stim_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
