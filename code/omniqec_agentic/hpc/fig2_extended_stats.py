"""Fig.2 extended statistics: more shots for the codes the campaigns chose, same measurement.

WHY (2026-10-14). The formal final stage measures every finalist at 25k shots per
basis. At N >= 220 the best codes fail 0-9 times in 50k shots, so their plpr has a
95 % interval 7-30x wide and the N = 360 champions have no lower bound at all. The
user wants each of these codes known to about +-25 % (>= 30 failures per basis).

WHAT. One code per job. The verification config is rebuilt from the SOURCE run's
protocol snapshot exactly as ``final_slow_node`` does (same noise, SEC policy,
decoder, timeouts, rounds floor T_c -> T = max(T_c, d_search) per candidate), only
``shots`` differs; the queue runs through ``evaluate_promotion_queue`` with
``stage="final"`` and a fresh seed namespace (``extended_stats_v1``, disjoint from
the search base 0 and the final ``final_confirmation_v1``), so the sample is new and
independent of the formal one. No shard ceiling is applied (the GPT condition of the
formal runs); the v3.4 per-shard watchdog still bounds a stuck shard.

The result is a SUPPLEMENT, not a campaign product: nothing under the 27 run
directories changes, the plan item and the slow row are written to
``<out_root>/<key>/result.json`` and merged into a table afterwards
(``--merge``), where the formal 25k counts and the new counts are pooled (both are
fixed-shot independent samples of the same code under the same setting).

USAGE
    python -m omniqec_agentic.hpc.fig2_extended_stats --plan PLAN.json --key b288_deepseek_efac54e4 \
        --runs-root runs --out-root runs/fig2_extended_stats --workers 127
    python -m omniqec_agentic.hpc.fig2_extended_stats --plan PLAN.json --merge --out-root ... > table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


def _plan_item(plan: dict[str, Any], key: str) -> dict[str, Any]:
    for item in plan["items"]:
        if item["key"] == key:
            return item
    raise SystemExit(f"key {key!r} not in plan ({len(plan['items'])} items)")


def measure(args: argparse.Namespace) -> int:
    from omniqec_agentic.evaluation import evaluate_promotion_queue
    from omniqec_agentic.workflow.graph import (
        _protocol_final_rounds,
        _verification_from_protocol,
        final_rng_base_seed,
    )

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    item = _plan_item(plan, args.key)
    # the source run's protocol snapshot + campaign spec; several roots because the 27 formal
    # runs live on three NSCC accounts (a root may hold just those two files of a run)
    run_dir = next(
        (Path(root) / item["source_run"] for root in args.runs_root
         if (Path(root) / item["source_run"] / "protocol_snapshot.json").is_file()),
        None,
    )
    if run_dir is None:
        raise SystemExit(f"{item['source_run']}/protocol_snapshot.json not found under {args.runs_root}")
    protocol = json.loads((run_dir / "protocol_snapshot.json").read_text(encoding="utf-8"))
    spec = json.loads((run_dir / "campaign_spec.json").read_text(encoding="utf-8"))
    final_rounds = _protocol_final_rounds(protocol)
    shots = int(args.shots_override or item["shots_per_basis"])
    p_values = tuple(float(v) for v in protocol["final_p_values"])
    verification = _verification_from_protocol(
        protocol, spec, p_values=p_values, shots=shots, rounds=final_rounds
    )
    if verification.final_decode_shard_timeout_cap_s is not None:
        raise SystemExit("the protocol verification carries a shard ceiling; this supplement runs without one")
    # ~300 shots per shard (the formal final: 25k / 128 = 195), never fewer shards than the formal final
    shot_shards = int(args.shot_shards or max(int(protocol.get("final_shot_shards") or 128), 2 ** math.ceil(math.log2(max(1.0, shots / 300.0)))))
    # a top-up round of the same code carries its own namespace: the same namespace (and queue
    # index 0) would reproduce the first round's shots exactly
    seed_namespace = str(item.get("seed_namespace") or plan["seed_namespace"])
    seed = final_rng_base_seed(seed_namespace)
    workers = int(args.workers if args.workers is not None else (protocol.get("slow_workers") or 0))
    out_dir = Path(args.out_root) / item["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / f"queue_T{final_rounds}.jsonl"
    record = {
        "program_id": item["program_id"],
        "spec": item["spec"],
        "code_hash": item.get("code_hash"),
        "selection_channels": ["extended_stats"] + (["baseline"] if item["role"] == "reference" else []),
        "is_baseline": item["role"] == "reference",
        "retry_at_final": False,
        "canonical_hash": None,
    }
    queue_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    config = {
        "schema": "fig2_extended_stats_config",
        "plan_version": plan.get("version"),
        "key": item["key"],
        "source_run": item["source_run"],
        "source_run_dir": str(run_dir),
        "protocol_profile": protocol.get("protocol_profile"),
        "required_model": protocol.get("required_model"),
        "shots_per_basis": shots,
        "rounds_floor": final_rounds,
        "rounds_policy": protocol.get("rounds_policy"),
        "p_values": list(p_values),
        "seed_namespace": seed_namespace,
        "seed": seed,
        "base_key": item.get("base_key") or item["key"],
        "shot_shards": shot_shards,
        "workers": workers,
        "final_decode_shard_timeout_cap_s": None,
        "verification": verification.to_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "extended_stats_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[ext] {item['key']} {item['program_id']} [[{item['n']},{item['k']},{item['d_search']}]] "
        f"shots/basis={shots} rounds_floor={final_rounds} shards={shot_shards} workers={workers} seed={seed}",
        flush=True,
    )
    t0 = time.time()
    summary = evaluate_promotion_queue(
        queue_path=queue_path,
        out_dir=out_dir / f"T{final_rounds}",
        verification=verification,
        include_baseline=False,
        resume_candidates=True,
        workers=workers,
        shot_shards=shot_shards,
        seed=seed,
        stage="final",
    )
    rankings_path = out_dir / f"T{final_rounds}" / "rankings.json"
    rows = json.loads(rankings_path.read_text(encoding="utf-8")).get("rankings") or []
    row = rows[0] if rows else None
    result = {
        "schema": "fig2_extended_stats_result",
        "plan_item": item,
        "config": config,
        "queue_status": summary.get("status"),
        "runtime_s": round(time.time() - t0, 1),
        "row": row,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if row is None:
        print(f"[ext] {item['key']} NO ROW (queue status {summary.get('status')})", flush=True)
        return 2
    print(
        f"[ext] {item['key']} status={row.get('status')} rounds={row.get('rounds')} failures X/Z="
        f"{row.get('failures_x')}/{row.get('failures_z')} of {row.get('shots_x')}/{row.get('shots_z')} "
        f"plpr={row.get('per_logical_per_round')} ci={row.get('per_logical_per_round_ci')} in {result['runtime_s']} s",
        flush=True,
    )
    return 0 if row.get("status") == "ok" else 3


def merge(args: argparse.Namespace) -> int:
    """One CSV row per code (plan ``base_key``): formal counts, the extension rounds summed, pooled estimate."""

    from omniqec_agentic.verification.ler import combine_basis_ler_results

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in plan["items"]:
        groups.setdefault(item.get("base_key") or item["key"], []).append(item)
    fields = [
        "key", "budget", "arm", "role", "code_hash8", "family", "n", "k", "d_search", "max_check_weight", "rounds",
        "formal_shots_per_basis", "formal_failures_x", "formal_failures_z", "formal_plpr",
        "ext_status", "ext_rounds", "ext_shots_per_basis", "ext_failures_x", "ext_failures_z", "ext_plpr", "ext_plpr_ci_low", "ext_plpr_ci_high",
        "pooled_shots_per_basis", "pooled_failures_x", "pooled_failures_z", "pooled_plpr", "pooled_plpr_ci_low", "pooled_plpr_ci_high",
        "ext_runtime_s", "seeds", "program_id", "source_run",
    ]

    def combined(fx: int, fz: int, sx: int, sz: int, k: int, rounds: int) -> tuple[float | None, float | None, float | None]:
        res = combine_basis_ler_results(
            {"X": {"logical_error_rate": fx / sx, "failures": fx, "shots": sx},
             "Z": {"logical_error_rate": fz / sz, "failures": fz, "shots": sz}},
            rounds=rounds, k=k,
        )
        ci = res.get("per_logical_per_round_ci") or [None, None]
        return res.get("per_logical_per_round"), ci[0], ci[1]

    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for base_key, items in groups.items():
        first = items[0]
        out: dict[str, Any] = {
            "key": base_key, "budget": first["budget"], "arm": first["arm"], "role": first["role"],
            "code_hash8": (first.get("code_hash") or "")[:8] or base_key.rsplit("_", 1)[-1], "family": first["family"],
            "n": first["n"], "k": first["k"], "d_search": first["d_search"], "max_check_weight": first["max_check_weight"],
            "rounds": first["formal_rounds"], "formal_shots_per_basis": first["formal_shots_per_basis"],
            "formal_failures_x": first.get("formal_failures_x"), "formal_failures_z": first.get("formal_failures_z"),
            "formal_plpr": first["formal_plpr"], "program_id": first["program_id"], "source_run": first["source_run"],
        }
        rows, statuses, seeds, runtime = [], [], [], 0.0
        for item in items:
            path = Path(args.out_root) / item["key"] / "result.json"
            if not path.is_file():
                statuses.append("pending")
                continue
            res = json.loads(path.read_text(encoding="utf-8"))
            row = res.get("row") or {}
            statuses.append(str(row.get("status") or res.get("queue_status")))
            seeds.append(str((res.get("config") or {}).get("seed")))
            runtime += float(res.get("runtime_s") or 0.0)
            if row.get("status") == "ok" and row.get("shots_x"):
                rows.append(row)
        out["ext_status"] = "+".join(statuses)
        out["ext_rounds"] = len(rows)
        out["seeds"] = "+".join(seeds)
        out["ext_runtime_s"] = round(runtime, 1) if seeds else None
        if rows:
            # the pooled estimate roots over the circuit's logical qubits x rounds: k for the memory
            # experiment, 2k for the Fig.3 two-block task (the plan item carries ``logicals``)
            k = int(first.get("logicals") or first["k"]); T = int(rows[0].get("rounds") or first["formal_rounds"])
            efx = sum(int(r["failures_x"]) for r in rows); efz = sum(int(r["failures_z"]) for r in rows)
            esx = sum(int(r["shots_x"]) for r in rows); esz = sum(int(r["shots_z"]) for r in rows)
            e = combined(efx, efz, esx, esz, k, T)
            out.update({"ext_shots_per_basis": esx, "ext_failures_x": efx, "ext_failures_z": efz,
                        "ext_plpr": e[0], "ext_plpr_ci_low": e[1], "ext_plpr_ci_high": e[2]})
            if first.get("formal_failures_x") is not None:
                # pooled: fixed-shot independent samples of the same code under the same setting -> add the counts
                fx = int(first["formal_failures_x"]) + efx; fz = int(first["formal_failures_z"]) + efz
                sx = int(first["formal_shots_per_basis"]) + esx; sz = int(first["formal_shots_per_basis"]) + esz
                pooled = combined(fx, fz, sx, sz, k, T)
                out.update({"pooled_shots_per_basis": sx, "pooled_failures_x": fx, "pooled_failures_z": fz,
                            "pooled_plpr": pooled[0], "pooled_plpr_ci_low": pooled[1], "pooled_plpr_ci_high": pooled[2]})
            else:  # a reference code: no formal counts in the plan, the extension alone is the estimate
                out.update({"pooled_shots_per_basis": esx, "pooled_failures_x": efx, "pooled_failures_z": efz,
                            "pooled_plpr": e[0], "pooled_plpr_ci_low": e[1], "pooled_plpr_ci_high": e[2]})
        writer.writerow(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--merge", action="store_true", help="write the merged CSV to stdout instead of measuring")
    parser.add_argument("--key", help="plan item to measure")
    parser.add_argument("--runs-root", action="append", help="directory holding formal run directories (repeatable, searched in order)")
    parser.add_argument("--workers", type=int, default=None, help="default: the protocol's slow_workers")
    parser.add_argument("--shot-shards", type=int, default=None)
    parser.add_argument("--shots-override", type=int, default=None, help="smoke tests only")
    args = parser.parse_args(argv)
    if args.merge:
        return merge(args)
    if not args.key or not args.runs_root:
        parser.error("--key and --runs-root are required to measure")
    return measure(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
