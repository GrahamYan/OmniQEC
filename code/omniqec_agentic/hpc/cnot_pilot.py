"""Fig.3 stages 0 and 1, phase A: cross-evaluate existing codes on logical operations.

Each candidate code C is built as several blocks of the SAME code side by side
and measured in experiments with identical everything else:

  idle   2 blocks, rounds_before + rounds_after QEC rounds, no gate
  cnot   2 blocks, rounds_before rounds | transversal CNOT 0 -> 1 | rounds_after rounds
  idle3  3 blocks, rounds_before + rounds_after QEC rounds, no gate
  fanout 3 blocks in the memory basis, fan-out CNOT 0 -> 1, 0 -> 2 (product-state variant)

in both memory bases (Z: every block |0>, X: every block |+>), and on the
general engine ``sec/logical_program.py`` (stages 1, 2 and 3):

  ghzidle   |+>|0>|0>, rounds_before + 1 + rounds_after rounds, no gate
  ghz       |+>|0>|0>, CNOT 0 -> 1 | QEC round | CNOT 0 -> 2 | rounds_after rounds;
            observables = the GHZ stabilizers Z0Z1, Z0Z2 (basis Z) and X0X1X2 (basis X)
  idle1     1 block, rounds_before + 1 + rounds_after rounds
  teleport  3 blocks: Bell pair (1, 2) after rounds_before rounds | QEC round |
            CNOT 0 -> 1 and Bell measurement; output block 2 runs rounds_after
            more and is checked with the Pauli frame of the Bell outcomes
  hidle     control code A and target code B (a candidate's ``child``) side by side
  hcnot     the same with the transversal CNOT A -> B after rounds_before The measurement stack is Fig.2's:
the SEC schedule comes from ``build_memory_experiment_pair`` under the source
run's ``sec_policy`` (its single-block circuit is checked equal to our
one-block emission), the noise is the source run's uniform_paper point, the
decoder is ``telescoping_cpu`` (final-stage config) on the GARI transform of
each two-block DEM, sampled from the noisy circuit.

Objective per logical index j (logical j of every block): a shot fails j in
basis W when any block's W-observable j is decoded wrong (a shot the decoder
declares undecodable fails every j); p_pair(j) = 1 - (1 - p_X(j)) (1 - p_Z(j));
p_task = mean_j p_pair(j). Each operation is compared with its idle baseline
(cnot/idle, fanout/idle3, ghz/ghzidle, teleport/idle1, hcnot/hidle): the
memory reference of the same length, size and inputs.

usage:
  python -m omniqec_agentic.hpc.cnot_pilot --candidates C.json --source-config slow_run_config.json \\
      --out DIR [--rounds-before 6 --rounds-after 6 --shots 25000 --shards 32 --workers 120 --only a,b]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.sec.multiblock_builder import EXPERIMENT_BASELINES, EXPERIMENT_LAYOUTS

PROGRAM_EXPERIMENTS = ("ghzidle", "ghz", "idle1", "teleport", "hidle", "hcnot")
EXPERIMENTS = tuple(EXPERIMENT_LAYOUTS) + PROGRAM_EXPERIMENTS
BASELINES = {**EXPERIMENT_BASELINES, "ghz": "ghzidle", "teleport": "idle1", "hcnot": "hidle"}


def child_code(hx: np.ndarray, hz: np.ndarray, child: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The stage 3 target code of a candidate: ``freeze_qubit`` q drops the Z
    checks on data qubit q and adds the weight-1 X check X_q."""

    if child.get("construction") != "freeze_qubit":
        raise ValueError(f"unknown child construction {child.get('construction')!r}")
    q = int(child["freeze_qubit"])
    x_q = np.zeros((1, hx.shape[1]), dtype=np.uint8)
    x_q[0, q] = 1
    return np.vstack([hx, x_q]), hz[hz[:, q] == 0]
BASES = ("X", "Z")


def _shard_seed(base: int, label: str, exp: str, basis: str, shard: int) -> int:
    digest = hashlib.sha256(f"{base}|{label}|{exp}|{basis}|{shard}".encode()).hexdigest()
    return int(digest[:12], 16)


def _wilson(f: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = f / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------- build phase
def build_candidate(job: dict[str, Any]) -> dict[str, Any]:
    """SEC synthesis + circuits + DEMs + GARI artifacts for one code."""

    import stim

    from omniqec_agentic.construction.build import build_compiled_code
    from omniqec_agentic.evaluation.common import verification_from_dict
    from omniqec_agentic.sec.logical_program import program_experiment_circuits
    from omniqec_agentic.sec.multiblock_builder import (
        build_twobasis_multiblock_circuit,
        cnot_experiment_circuits,
        logical_pair_count,
    )
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair
    from omniqec_agentic.verification.decoders import build_decoding_system

    label, out = job["label"], Path(job["out"]) / job["label"]
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        verification = verification_from_dict(job["verification"])
        point = verification.noise.point(0)
        compiled = build_compiled_code(job["spec"])
        if str(compiled.code_hash)[:8] != str(job["code_hash8"])[:8]:
            raise ValueError(f"rebuilt hash {compiled.code_hash[:8]} != {job['code_hash8']}")
        hx = np.asarray(compiled.hx, dtype=np.uint8)
        hz = np.asarray(compiled.hz, dtype=np.uint8)
        rounds_total = int(job["rounds_before"]) + int(job["rounds_after"])
        pair = build_memory_experiment_pair(
            hx, hz, p=float(point.value), rounds=rounds_total, d_hint=job.get("d_search"),
            noise_point=point, sec_policy=verification.sec_policy,
            layer_blocks=getattr(compiled, "layer_blocks", None),
            synth_cache_dir=job.get("synth_cache_dir"),
        )
        schedule = pair.x_memory.schedule
        # the one-block, gate-free emission IS the Fig.2 memory circuit (compared at the
        # rounds the pair builder chose: it floors the requested rounds at Fig.2's T)
        for basis, mem in (("X", pair.x_memory), ("Z", pair.z_memory)):
            one = build_twobasis_multiblock_circuit(
                hx, hz, schedule, basis=basis, rounds_before=int(mem.rounds), n_blocks=1, noise_point=point
            )
            if str(one) != str(mem.circuit):
                raise ValueError(f"one-block {basis} circuit differs from the Fig.2 memory circuit")
        experiments = tuple(job.get("experiments") or EXPERIMENTS)
        layout_exps = tuple(e for e in experiments if e not in PROGRAM_EXPERIMENTS)
        program_exps = tuple(e for e in experiments if e in PROGRAM_EXPERIMENTS)
        if job.get("cnot_every_round"):
            cnot_rounds = tuple(range(1, rounds_total))
        elif job.get("cnot_rounds"):
            cnot_rounds = tuple(int(r) for r in job["cnot_rounds"])
        else:
            cnot_rounds = None
        circuits = cnot_experiment_circuits(
            hx, hz, schedule, noise_point=point,
            rounds_before=int(job["rounds_before"]), rounds_after=int(job["rounds_after"]),
            experiments=layout_exps, cnot_rounds=cnot_rounds,
            cnot_noise_scale=float(job.get("cnot_noise_scale", 1.0)),
        ) if layout_exps else {}
        if program_exps:
            if job.get("cnot_every_round") or job.get("cnot_rounds") or float(job.get("cnot_noise_scale", 1.0)) != 1.0:
                raise ValueError("the logical programs run the plain protocol only")
            codes = [(hx, hz, schedule)]
            if job.get("child"):
                hx_b, hz_b = child_code(hx, hz, job["child"])
                pair_b = build_memory_experiment_pair(
                    hx_b, hz_b, p=float(point.value), rounds=rounds_total,
                    d_hint=min(int(job["child"].get("proxy_d_x") or 0), int(job["child"].get("proxy_d_z") or 0)) or None,
                    noise_point=point, sec_policy=verification.sec_policy, layer_blocks=None,
                    synth_cache_dir=job.get("synth_cache_dir"),
                )
                codes.append((hx_b, hz_b, pair_b.x_memory.schedule))
            circuits.update(program_experiment_circuits(
                codes, noise_point=point, rounds_before=int(job["rounds_before"]),
                rounds_after=int(job["rounds_after"]), experiments=program_exps,
            ))
        files: dict[str, Any] = {}
        for exp in experiments:
            for basis in BASES:
                circuit, kinds = circuits[exp][basis]
                stem = out / f"{exp}_{basis}"
                stem.with_suffix(".stim").write_text(str(circuit))
                dem = circuit.detector_error_model()
                stem.with_suffix(".dem").write_text(str(dem))
                stem.with_suffix(".kinds.json").write_text(json.dumps(list(kinds)))
                workdir = out / f"tele_{exp}_{basis}"
                workdir.mkdir(exist_ok=True)
                system = build_decoding_system(str(circuit), kinds, basis, workdir=str(workdir), use_gari=True)
                files[f"{exp}_{basis}"] = {
                    "stim": str(stem.with_suffix(".stim")),
                    "dem": str(stem.with_suffix(".dem")),
                    "kinds": str(stem.with_suffix(".kinds.json")),
                    "gari": str(system.npz_path_for("gari")) if system.gari is not None else None,
                    "num_detectors": int(circuit.num_detectors),
                    "num_observables": int(circuit.num_observables),
                    "gari_shape": [int(v) for v in system.gari.h.shape] if system.gari is not None else None,
                }
        k = logical_pair_count(hx, hz)
        info = {
            "label": label, "status": "ok", "k": k, "n": int(hx.shape[1]),
            "max_check_weight": int(max(hx.sum(axis=1).max(), hz.sum(axis=1).max())),
            "sec_policy": pair.sec_policy_id, "schedule": schedule.to_metadata(),
            "files": files, "build_s": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001 - report per candidate
        info = {"label": label, "status": "build_error", "error": f"{type(exc).__name__}: {exc}",
                "build_s": round(time.perf_counter() - started, 1)}
    (out / "build.json").write_text(json.dumps(info, indent=2, default=str))
    return info


# ---------------------------------------------------------------- decode phase
def decode_shard(task: dict[str, Any]) -> dict[str, Any]:
    import stim

    from omniqec_agentic.config import TelescopingDecoderConfig
    from omniqec_agentic.verification.decoders import TelescopingObservableDecoder
    from omniqec_agentic.verification.dem import dem_file_to_matrices

    started = time.perf_counter()
    f = task["files"]
    k = int(task["k"])
    n_obs = int(f["num_observables"])
    blocks = n_obs // k
    matrices = dem_file_to_matrices(f["dem"])
    kinds = json.loads(Path(f["kinds"]).read_text())
    circuit_text = Path(f["stim"]).read_text()
    decoder = TelescopingObservableDecoder(
        matrices, TelescopingDecoderConfig(), detector_kinds=kinds, basis=task["basis"],
        seed=int(task["seed"]), stage="final", circuit_text=circuit_text,
        artifacts={"gari": f["gari"]} if f.get("gari") else None,
    )
    obs_fail = np.zeros(n_obs, dtype=np.int64)
    pair_fail = np.zeros(k, dtype=np.int64)
    any_fail = declared = done = 0
    try:
        sampler = stim.Circuit(circuit_text).compile_detector_sampler(seed=int(task["seed"]))
        shots, batch = int(task["shots"]), int(task.get("batch_size", 1024))
        while done < shots:
            size = min(batch, shots - done)
            det, obs = sampler.sample(size, separate_observables=True)
            decoder._shot_offset = int(done)
            pred = decoder.decode_observables_batch(np.asarray(det, dtype=np.uint8))
            wrong = np.asarray(pred, dtype=np.uint8) != np.asarray(obs, dtype=np.uint8)
            dec = getattr(decoder, "last_declared_failures", None)
            if dec is not None and np.asarray(dec).shape[0] == size:
                dec = np.asarray(dec, dtype=bool)
                wrong = wrong | dec[:, None]
                declared += int(dec.sum())
            obs_fail += wrong.sum(axis=0)
            pair_fail += wrong.reshape(size, blocks, k).any(axis=1).sum(axis=0)
            any_fail += int(wrong.any(axis=1).sum())
            done += size
    finally:
        decoder.close()
    return {
        "label": task["label"], "exp": task["exp"], "basis": task["basis"], "shard": task["shard"],
        "shots": done, "any_fail": any_fail, "declared": declared,
        "obs_fail": obs_fail.tolist(), "pair_fail": pair_fail.tolist(),
        "runtime_s": round(time.perf_counter() - started, 2),
    }


# ---------------------------------------------------------------- summary
def summarize(builds: dict[str, dict], shard_rows: list[dict], candidates: dict[str, dict],
              experiments: tuple[str, ...] = EXPERIMENTS) -> dict[str, Any]:
    agg: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in shard_rows:
        key = (r["label"], r["exp"], r["basis"])
        a = agg.setdefault(key, {"shots": 0, "any_fail": 0, "declared": 0, "pair_fail": None, "runtime_s": 0.0})
        a["shots"] += r["shots"]; a["any_fail"] += r["any_fail"]; a["declared"] += r["declared"]
        a["runtime_s"] += r["runtime_s"]
        pf = np.asarray(r["pair_fail"], dtype=np.int64)
        a["pair_fail"] = pf if a["pair_fail"] is None else a["pair_fail"] + pf
    out: dict[str, Any] = {}
    for label, b in builds.items():
        if b.get("status") != "ok":
            out[label] = {"status": b.get("status"), "error": b.get("error")}
            continue
        k = int(b["k"])
        row: dict[str, Any] = {"status": "ok", "k": k, "n": b["n"], "w": b["max_check_weight"],
                               **{key: candidates[label].get(key) for key in ("code_hash8", "d_search", "family", "source", "fig2_plpr", "child")}}
        for exp in experiments:
            per_basis = {}
            for basis in BASES:
                a = agg.get((label, exp, basis))
                if not a or not a["shots"]:
                    per_basis = None
                    break
                p_j = a["pair_fail"] / a["shots"]
                per_basis[basis] = {"shots": a["shots"], "any_fail": a["any_fail"], "declared": a["declared"],
                                    "pair_fail": a["pair_fail"].tolist(), "p_pair_mean": float(p_j.mean()),
                                    "runtime_s": round(a["runtime_s"], 1)}
            if per_basis is None:
                row[exp] = None
                continue
            px = np.asarray(per_basis["X"]["pair_fail"]) / per_basis["X"]["shots"]
            pz = np.asarray(per_basis["Z"]["pair_fail"]) / per_basis["Z"]["shots"]
            p_task = float(np.mean(1 - (1 - px) * (1 - pz)))
            # CI: Wilson per basis on the pair-averaged failure count, combined
            lo_hi = {}
            for basis in BASES:
                pb = per_basis[basis]
                lo_hi[basis] = _wilson(float(np.mean(pb["pair_fail"])), pb["shots"])
            ci = (1 - (1 - lo_hi["X"][0]) * (1 - lo_hi["Z"][0]), 1 - (1 - lo_hi["X"][1]) * (1 - lo_hi["Z"][1]))
            row[exp] = {"p_task": p_task, "ci": [float(ci[0]), float(ci[1])], "bases": per_basis}
        for op, base in BASELINES.items():
            if row.get(base) and row.get(op):
                row[f"{op}_over_{base}"] = row[op]["p_task"] / row[base]["p_task"] if row[base]["p_task"] else None
                row[f"{op}_minus_{base}"] = row[op]["p_task"] - row[base]["p_task"]
        out[label] = row
    return out


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"  # a zero-failure baseline has no ratio


def write_report(summary: dict[str, Any], path: Path, meta: dict[str, Any]) -> None:
    L = [f"# Fig.3 pilot: logical operations vs idle (blocks of the same code)", "",
         "```", json.dumps(meta, indent=1, default=str), "```", "",
         "p_task = mean over logical indices j of 1-(1-p_X(j))(1-p_Z(j)); j fails in a basis when any block's observable j is wrong."]
    for op, base in BASELINES.items():
        ok = {k: v for k, v in summary.items() if v.get("status") == "ok" and v.get(base) and v.get(op)}
        if not ok:
            continue
        base_rank = {lab: i + 1 for i, lab in enumerate(sorted(ok, key=lambda x: ok[x][base]["p_task"]))}
        op_rank = {lab: i + 1 for i, lab in enumerate(sorted(ok, key=lambda x: ok[x][op]["p_task"]))}
        L += ["", f"## {op} vs {base}", "",
              f"| code | source | [[n,k,d]] w | Fig.2 plpr | {base} p_task [95% CI] | rank | {op} p_task [95% CI] | rank | {op}/{base} | declared ({base}/{op}) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lab in sorted(ok, key=lambda x: ok[x][op]["p_task"]):
            v = ok[lab]
            dec_b = sum(b["declared"] for b in v[base]["bases"].values())
            dec_o = sum(b["declared"] for b in v[op]["bases"].values())
            L.append(f"| {lab} | {v.get('source') or ''} | [[{v['n']},{v['k']},{v.get('d_search')}]] w{v['w']} | "
                     f"{(v.get('fig2_plpr') or float('nan')):.2e} | {v[base]['p_task']:.3e} [{v[base]['ci'][0]:.2e}, {v[base]['ci'][1]:.2e}] | {base_rank[lab]} | "
                     f"{v[op]['p_task']:.3e} [{v[op]['ci'][0]:.2e}, {v[op]['ci'][1]:.2e}] | {op_rank[lab]} | "
                     f"{_ratio(v.get(f'{op}_over_{base}'))} | {dec_b}/{dec_o} |")
        if len(ok) >= 3:
            ranks = np.array([[base_rank[x], op_rank[x]] for x in ok], dtype=float)
            d = ranks[:, 0] - ranks[:, 1]
            m = len(ok)
            rho = 1 - 6 * float((d ** 2).sum()) / (m * (m * m - 1))
            L += ["", f"Spearman rank correlation {base} vs {op}: {rho:.3f} over {m} codes."]
    bad = {k: v for k, v in summary.items() if v.get("status") != "ok"}
    if bad:
        L += ["", "Failed builds:"] + [f"- {k}: {v.get('error')}" for k, v in bad.items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--source-config", required=True, help="a Fig.2 final_slow/T12/slow_run_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds-before", type=int, default=6)
    ap.add_argument("--rounds-after", type=int, default=6)
    ap.add_argument("--shots", type=int, default=25000, help="shots per basis per experiment")
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", default="")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished builds and skip shards already in shards.jsonl (same seeds -> identical results)")
    ap.add_argument("--heavy-w", type=int, default=0,
                    help="codes with max check weight >= this get --heavy-shots per basis (0 = off)")
    ap.add_argument("--heavy-shots", type=int, default=2500)
    ap.add_argument("--experiments", default="idle,cnot", help=f"comma-separated subset of {','.join(EXPERIMENTS)}")
    ap.add_argument("--cnot-every-round", action="store_true",
                    help="a transversal CNOT tick after every round 1..R-1 instead of one after --rounds-before")
    ap.add_argument("--cnot-rounds", default="",
                    help="rounds after which a transversal CNOT tick is applied, '+' or ',' separated (e.g. 3+6+9; "
                         "the gate-density sweep: p_task vs number of gates gives the error per logical gate)")
    ap.add_argument("--cnot-noise-scale", type=float, default=1.0,
                    help="multiplies the DEPOLARIZE2 rate of the transversal CNOT layers only")
    ap.add_argument("--synth-cache", default="", help="SEC synthesis cache dir to reuse (e.g. a stage-0 run's _synth_cache)")
    ap.add_argument("--exclude-w", type=int, default=0, help="skip codes with max check weight >= this (0 = keep all)")
    args = ap.parse_args(argv)
    experiments = tuple(e for e in args.experiments.split(",") if e)
    cnot_rounds = sorted({int(r) for r in args.cnot_rounds.replace("+", ",").split(",") if r.strip()})
    if cnot_rounds and args.cnot_every_round:
        ap.error("--cnot-rounds and --cnot-every-round are exclusive")
    if cnot_rounds and not all(1 <= r <= args.rounds_before + args.rounds_after - 1 for r in cnot_rounds):
        ap.error(f"--cnot-rounds must lie in 1..{args.rounds_before + args.rounds_after - 1}")
    if not experiments or any(e not in EXPERIMENTS for e in experiments):
        ap.error(f"--experiments must be a subset of {EXPERIMENTS}")

    src = json.loads(Path(args.source_config).read_text())
    cands = json.loads(Path(args.candidates).read_text())
    if args.only:
        keep = set(args.only.split(","))
        cands = [c for c in cands if c["label"] in keep]
    if args.exclude_w:
        cands = [c for c in cands if int(c.get("w") or 0) < int(args.exclude_w)]
    by_label = {c["label"]: c for c in cands}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source_config": str(args.source_config), "seed": src["seed"], "noise": src["verification"]["noise"],
            "sec_policy": src["verification"]["sec_policy"]["sec_policy_id"], "decoder": "telescoping_cpu (final)",
            "rounds_before": args.rounds_before, "rounds_after": args.rounds_after, "shots_per_basis": args.shots,
            "shards": args.shards, "codes": len(cands),
            "heavy_w": args.heavy_w, "heavy_shots": args.heavy_shots if args.heavy_w else None,
            "experiments": list(experiments), "cnot_every_round": bool(args.cnot_every_round),
            "cnot_rounds": cnot_rounds, "cnot_noise_scale": args.cnot_noise_scale, "exclude_w": args.exclude_w}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    ctx = mp.get_context("spawn")
    jobs = [{"label": c["label"], "spec": c["spec"], "code_hash8": c["code_hash8"], "d_search": c.get("d_search"),
             "verification": src["verification"], "rounds_before": args.rounds_before,
             "rounds_after": args.rounds_after, "out": str(out),
             "synth_cache_dir": args.synth_cache or str(out / "_synth_cache"),
             "experiments": list(experiments), "cnot_every_round": bool(args.cnot_every_round),
             "cnot_rounds": cnot_rounds, "cnot_noise_scale": float(args.cnot_noise_scale), "child": c.get("child")}
            for c in cands]
    t0 = time.perf_counter()
    builds: dict[str, dict] = {}
    if args.resume:
        for job in jobs:
            path = out / job["label"] / "build.json"
            if path.exists():
                b = json.loads(path.read_text())
                if b.get("status") == "ok" and all(Path(f["stim"]).exists() for f in b["files"].values()):
                    builds[b["label"]] = b
        print(f"[build] resume: reusing {len(builds)} finished builds", flush=True)
    todo = [job for job in jobs if job["label"] not in builds]
    if todo:
        with ctx.Pool(min(args.workers, len(todo))) as pool:
            builds.update({b["label"]: b for b in pool.imap_unordered(build_candidate, todo)})
    for b in builds.values():
        print(f"[build] {b['label']}: {b['status']} {b.get('error', '')} {b.get('build_s')}s", flush=True)
    print(f"[build] done in {time.perf_counter() - t0:.0f}s", flush=True)

    def is_heavy(label: str) -> bool:
        return bool(args.heavy_w) and int(by_label[label].get("w") or 0) >= int(args.heavy_w)

    tasks = []
    plan: set[tuple] = set()
    for label, b in builds.items():
        if b.get("status") != "ok":
            continue
        total = int(args.heavy_shots) if is_heavy(label) else int(args.shots)
        per = int(math.ceil(total / args.shards))
        for exp in experiments:
            for basis in BASES:
                for shard in range(args.shards):
                    shots = min(per, total - shard * per)
                    if shots <= 0:
                        continue
                    plan.add((label, exp, basis, shard, shots))
                    tasks.append({"label": label, "exp": exp, "basis": basis, "shard": shard, "shots": shots,
                                  "k": b["k"], "files": b["files"][f"{exp}_{basis}"], "heavy": is_heavy(label),
                                  "seed": _shard_seed(int(src["seed"]), label, exp, basis, shard)})
    rows: list[dict] = []
    shards_path = out / "shards.jsonl"
    if args.resume and shards_path.exists():
        for line in shards_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by a killed job: that shard simply re-runs
                # only shards of the current plan count (a changed shot plan re-draws)
                if (r["label"], r["exp"], r["basis"], r["shard"], r["shots"]) in plan:
                    rows.append(r)
        done = {(r["label"], r["exp"], r["basis"], r["shard"], r["shots"]) for r in rows}
        tasks = [t for t in tasks if (t["label"], t["exp"], t["basis"], t["shard"], t["shots"]) not in done]
        print(f"[decode] resume: {len(rows)} shards kept, {len(tasks)} to run", flush=True)
    # light codes first (their operation/baseline pair completes early), longest first within a group
    tasks.sort(key=lambda t: (t["heavy"], -(t["files"]["num_detectors"] * (2 if t["exp"] in BASELINES else 1))))
    t1 = time.perf_counter()
    with shards_path.open("a" if args.resume else "w") as fh, ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(decode_shard, tasks), 1):
            rows.append(r)
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % max(1, len(tasks) // 40) == 0 or i == len(tasks):
                print(f"[decode] {i}/{len(tasks)} shards, {time.perf_counter() - t1:.0f}s", flush=True)
    summary = summarize(builds, rows, by_label, experiments)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(summary, out / "report.md", meta)
    print((out / "report.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
