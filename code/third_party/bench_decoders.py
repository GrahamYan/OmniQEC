"""Decoder speed/accuracy benchmark on a shared circuit-level DEM.

Compares OmniQEC's incumbent decoders (BP+OSD, BP+LSD, Relay-BP, with
omniqec_agentic config defaults) against Beam Search (IonQ) and BPOTF on the
same detector error model and the same sampled syndromes, timing each shot.

Run from OmniQEC root with the qec conda env python:
    python third_party/bench_decoders.py --circuit <path.stim> --shots 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parent
ROOT = THIRD_PARTY.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIRD_PARTY / "BeamSearchDecoder" / "decoder"))
sys.path.insert(0, str(THIRD_PARTY / "bpotf_pkg"))

import numpy as np
import scipy.sparse
import stim

from omniqec_agentic.verification.dem import dem_to_matrices


def build_incumbents(matrices, which):
    """Build OmniQEC's existing decoders with omniqec_agentic config defaults."""
    decoders = {}
    if "bposd" in which:
        from ldpc import BpOsdDecoder

        decoders["bposd(osd_cs3)"] = {
            "decode": BpOsdDecoder(
                matrices.H,
                channel_probs=matrices.probs,
                max_iter=12,
                bp_method="product_sum",
                osd_method="osd_cs",
                osd_order=3,
            ).decode,
            "returns": "error",
        }
    if "lsd" in which:
        from ldpc import BpLsdDecoder

        decoders["lsd(order8)"] = {
            # lsd_method is mandatory in ldpc 2.4.1: without it, lsd_order is
            # silently reset to 0 — earlier "lsd(order8)" bench rows were
            # actually BP+LSD-0 with ms_scaling_factor=1.0.
            "decode": BpLsdDecoder(
                matrices.H,
                channel_probs=matrices.probs,
                max_iter=50,
                bp_method="ms",
                lsd_method="lsd_cs",
                lsd_order=8,
                ms_scaling_factor=0.625,
            ).decode,
            "returns": "error",
        }
    relay_variants = {
        # OmniQEC production default (config.py RelayBpDecoderConfig, "2d_reduced")
        "relaybp": ("relaybp(60sets)", dict(pre_iter=12, num_sets=60, set_max_iter=12, stop_nconv=1)),
        # Upstream example strength: collect 5 solutions, long legs
        "relaybp_strong": ("relaybp(strong)", dict(pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=5)),
    }
    for key, (label, params) in relay_variants.items():
        if key not in which:
            continue
        import relay_bp

        core = relay_bp.RelayDecoderF64(
            matrices.H,
            error_priors=matrices.probs,
            alpha=1.0,
            gamma0=0.15,
            gamma_dist_interval=(-0.226, 0.622),
            **params,
        )
        runner = relay_bp.ObservableDecoderRunner(
            core, matrices.L, include_decode_result=False
        )

        def relay_decode(syndrome, _runner=runner):
            return _runner.decode_observables_batch(
                syndrome.reshape(1, -1), parallel=False
            )[0]

        decoders[label] = {"decode": relay_decode, "returns": "obs"}
    return decoders


def build_beam(matrices, widths):
    from beam_search_decoder import BeamSearchDecoder

    decoders = {}
    for width in widths:
        dec = BeamSearchDecoder(
            matrices.H,
            error_channel=matrices.probs,
            beam_width=int(width),
            max_rounds=10,
            num_results=1,
            initial_iters=30,
            iters_per_round=20,
        )
        decoders[f"beam(B={width})"] = {"decode": dec.decode, "returns": "error"}
    return decoders


def build_bpotf(matrices, base_p):
    import BPOTF as bpotf_mod

    pcm = scipy.sparse.csc_matrix(matrices.H)
    # E_CLN requires a DemData carrying priors + dense uint8 observables matrix
    # (without phen/transfer matrices it runs plain OTF on the DEM pcm itself).
    dem_data = bpotf_mod.DemData()
    dem_data.priors = np.asarray(matrices.probs, dtype=np.float64)
    dem_data.obs_matrix = matrices.L.toarray().astype(np.uint8, order="F")
    dec = bpotf_mod.OBPOTF(
        pcm,
        float(base_p),
        bpotf_mod.NoiseType.E_CLN,
        None,
        None,
        dem_data,
    )
    return {"bpotf(E_CLN)": {"decode": dec.decode, "returns": "auto"}}


def run_one(name, entry, syndromes, observables, matrices):
    n_cols = matrices.H.shape[1]
    num_obs = matrices.num_observables
    L = matrices.L
    failures = 0
    times = np.zeros(len(syndromes), dtype=np.float64)
    returns = entry["returns"]
    decode = entry["decode"]
    for i, syndrome in enumerate(syndromes):
        t0 = time.perf_counter()
        out = decode(syndrome)
        out = np.asarray(out)
        if returns == "error" or (returns == "auto" and out.shape[-1] == n_cols):
            predicted = (L @ (out.astype(np.uint8) % 2)) % 2
        else:
            predicted = out.astype(np.uint8).reshape(-1)[:num_obs] % 2
        times[i] = time.perf_counter() - t0
        failures += int(not np.array_equal(predicted, observables[i]))
    ms = times * 1e3
    return {
        "decoder": name,
        "shots": len(syndromes),
        "failures": failures,
        "ler": failures / len(syndromes),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.percentile(ms, 50)),
        "p90_ms": float(np.percentile(ms, 90)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "total_s": float(times.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-p", type=float, default=0.003)
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["bposd", "lsd", "relaybp", "beam", "bpotf"],
    )
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    circuit = stim.Circuit.from_file(args.circuit)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    matrices = dem_to_matrices(dem)
    meta = matrices.to_metadata()
    print(f"circuit: {Path(args.circuit).name}")
    print(
        f"DEM: {meta['num_detectors']} detectors, {meta['num_matrix_columns']} error"
        f" mechanisms, {meta['num_observables']} observables"
    )

    sampler = circuit.compile_detector_sampler(seed=args.seed)
    dets, obs = sampler.sample(args.shots, separate_observables=True)
    syndromes = dets.astype(np.uint8)
    observables = obs.astype(np.uint8)
    print(f"sampled {args.shots} shots (seed={args.seed})\n")

    decoders = {}
    incumbent = [
        d for d in args.decoders if d in ("bposd", "lsd", "relaybp", "relaybp_strong")
    ]
    decoders.update(build_incumbents(matrices, incumbent))
    if "beam" in args.decoders:
        decoders.update(build_beam(matrices, args.beam_widths))
    if "bpotf" in args.decoders:
        try:
            decoders.update(build_bpotf(matrices, args.base_p))
        except Exception as exc:
            print(f"[warn] BPOTF unavailable: {exc}")

    results = []
    for name, entry in decoders.items():
        print(f"running {name} ...", flush=True)
        res = run_one(name, entry, syndromes, observables, matrices)
        results.append(res)
        print(
            f"  LER={res['ler']:.2e} ({res['failures']}/{res['shots']})"
            f"  mean={res['mean_ms']:.2f}ms  p99={res['p99_ms']:.2f}ms"
            f"  max={res['max_ms']:.1f}ms  total={res['total_s']:.1f}s"
        )

    header = (
        f"{'decoder':<18}{'LER':>10}{'fails':>7}{'mean ms':>10}{'median':>9}"
        f"{'p90':>9}{'p99':>10}{'max':>10}{'total s':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for res in sorted(results, key=lambda r: r["mean_ms"]):
        print(
            f"{res['decoder']:<18}{res['ler']:>10.2e}{res['failures']:>7d}"
            f"{res['mean_ms']:>10.2f}{res['median_ms']:>9.2f}{res['p90_ms']:>9.2f}"
            f"{res['p99_ms']:>10.2f}{res['max_ms']:>10.1f}{res['total_s']:>9.1f}"
        )

    if args.out:
        payload = {
            "circuit": str(args.circuit),
            "shots": args.shots,
            "seed": args.seed,
            "dem": meta,
            "results": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
