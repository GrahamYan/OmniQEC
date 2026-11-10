"""Decoder benchmark on OmniQEC-discovered codes (spec JSON -> circuit -> DEM).

Loads code spec JSONs from previous OmniQEC experiments (champion_test_kit /
reference_figure2 queue format), rebuilds HX/HZ with the construction plugins,
generates the memory circuit + DEM exactly as the slow-verify path does
(uniform legacy noise), and benchmarks the decoder suite on identical shots.

Run from OmniQEC root with the qec conda env python:
    python third_party/bench_code_decoders.py --codes <a.json> <b.json> \
        --p 0.005 --shots 1000 --basis Z
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parent
ROOT = THIRD_PARTY.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIRD_PARTY / "BeamSearchDecoder" / "decoder"))
sys.path.insert(0, str(THIRD_PARTY / "bpotf_pkg"))

import numpy as np

from omniqec_agentic.construction.build import build_compiled_code
from omniqec_agentic.verification.circuit import build_memory_experiment_pair
from omniqec_agentic.verification.dem import dem_to_matrices

from bench_decoders import build_beam, build_bpotf, build_incumbents, run_one


def load_spec(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "spec" in data:
        return data
    raise ValueError(f"{path}: no 'spec' field")


def bench_code(path: Path, args) -> list[dict]:
    data = load_spec(path)
    spec = data["spec"]
    d_hint = data.get("d_proxy")
    if d_hint is None:
        match = re.search(r"\[\[\d+,\s*\d+,\s*(\d+)\]\]", str(data.get("code", "")))
        if match:
            d_hint = int(match.group(1))
    compiled = build_compiled_code(spec)
    hx, hz = compiled.hx, compiled.hz
    pair = build_memory_experiment_pair(
        hx, hz, p=args.p, rounds=args.rounds, d_hint=d_hint
    )
    experiment = pair.z_memory if args.basis == "Z" else pair.x_memory
    dem = experiment.detector_error_model
    matrices = dem_to_matrices(dem.flattened())
    meta = matrices.to_metadata()
    label = data.get("code") or data.get("figure2_label") or path.stem
    print(
        f"\n### {path.name}  {label}  family={spec.get('family')}"
        f"  rounds={experiment.rounds}  basis={args.basis}  p={args.p}"
    )
    print(
        f"DEM: {meta['num_detectors']} detectors,"
        f" {meta['num_matrix_columns']} mechanisms,"
        f" {meta['num_observables']} observables"
    )

    sampler = experiment.circuit.compile_detector_sampler(seed=args.seed)
    dets, obs = sampler.sample(args.shots, separate_observables=True)
    syndromes = dets.astype(np.uint8)
    observables = obs.astype(np.uint8)

    decoders = {}
    incumbent = [
        d for d in args.decoders if d in ("bposd", "lsd", "relaybp", "relaybp_strong")
    ]
    decoders.update(build_incumbents(matrices, incumbent))
    if "beam" in args.decoders:
        decoders.update(build_beam(matrices, args.beam_widths))
    if "bpotf" in args.decoders:
        try:
            decoders.update(build_bpotf(matrices, args.p))
        except Exception as exc:
            print(f"[warn] BPOTF unavailable: {exc}")

    results = []
    for name, entry in decoders.items():
        res = run_one(name, entry, syndromes, observables, matrices)
        res["code_file"] = path.name
        res["code_label"] = str(label)
        results.append(res)
        print(
            f"  {name:<18} LER={res['ler']:.2e} ({res['failures']}/{res['shots']})"
            f"  mean={res['mean_ms']:.2f}ms  p99={res['p99_ms']:.2f}ms"
            f"  total={res['total_s']:.1f}s"
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="+", required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--basis", choices=["X", "Z"], default="Z")
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["bposd", "lsd", "relaybp", "beam", "bpotf"],
    )
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    all_results = []
    for code_path in args.codes:
        try:
            all_results.extend(bench_code(Path(code_path), args))
        except Exception as exc:
            print(f"[error] {code_path}: {exc}")

    if args.out:
        payload = {
            "p": args.p,
            "shots": args.shots,
            "seed": args.seed,
            "basis": args.basis,
            "results": all_results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
