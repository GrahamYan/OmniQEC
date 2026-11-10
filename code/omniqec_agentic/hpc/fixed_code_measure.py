"""Fixed-code measurements under a frozen task protocol (2026-10-23).

WHY. Fig.4 a/b, the S4 noise sweep and every supplement of the plans of 2026-10-23 measure a
GIVEN code (a spec, never an LLM-generated program) under a task protocol that already exists
as a frozen ``protocol_snapshot.json`` of a formal run: the Fig.2 memory experiment
(``telescoping_structural_v3_7``) or the Fig.3 two-block transversal-CNOT task
(``telescoping_structural_v3_7_cnot_v2``). Only a handful of knobs differ from the formal
final stage, and every one of them is explicit in the plan item:

  task            which protocol family the item belongs to; the protocol's own task must match
  p               the physical error rate (S4 sweeps it; everything else stays at 0.005)
  shots_per_basis fixed shots, X and Z, no racing, no early stop, no decoder-cost cap
  rounds_floor    T_c (12 everywhere in these plans)
  rounds_policy   ``fixed_Tc`` -> T = T_c for every code (the Fig.4 a cross-evaluation and
                  every CNOT measurement); ``max_Tc_d`` -> T = max(T_c, d_search), the native
                  Fig.2 memory rule (Fig.4 b confirmations, S4)
  max_check_weight None: a fixed-code evaluation never applies a SEARCH weight cap (the Fig.3
                  snapshot carries cap 6; Fig.2 w7/w8 codes measured under the CNOT task must
                  not be rejected by it)
  seed_namespace  one per item: the sample of every condition is a fresh, auditable RNG stream
                  (``final_rng_base_seed``), disjoint from the search base and from the formal
                  final (``final_confirmation_v1``) and extended-stats (``extended_stats_v1``)
                  namespaces

Everything else -- noise profile, SEC policy, decoder preset, timeouts, bases -- comes from
the protocol snapshot unchanged. The measurement path is the formal final stage's
(``evaluate_promotion_queue(stage="final")`` with ``racing=None``), exactly as
``hpc/fig2_extended_stats.py``. The watchdog of the final stage (8 CPU-s per shot, floor
1800 s per shard) still applies: a condition that expires lands as ``decode_infeasible`` and
is reported as such, never silently dropped; ``--decode-shot-budget-s`` re-runs it with a
larger budget (recorded in its config).

USAGE
    python -m omniqec_agentic.hpc.fixed_code_measure --plan PLAN.json --protocol-root ROOT \
        --out-root OUT --key KEY [--key KEY2 ...] [--workers 127]
    python -m omniqec_agentic.hpc.fixed_code_measure --plan PLAN.json --protocol-root ROOT \
        --out-root OUT --keys-file keys.txt            # several conditions, one job, resumable
    python -m omniqec_agentic.hpc.fixed_code_measure --plan PLAN.json --out-root OUT --list-keys [--pending]
    python -m omniqec_agentic.hpc.fixed_code_measure --plan PLAN.json --out-root OUT --merge --out table.csv
    python -m omniqec_agentic.hpc.fixed_code_measure --plan PLAN.json --out-root OUT --audit-rounds

ROUNDS CHECK (2026-10-27): a condition is done only when its measured X/Z circuits
(candidate ``memory_experiment.rounds``), its row and its plan agree on T; ``--audit-rounds`` prints the
verdict per item, ``--merge`` reports a disagreement as status ``rounds_mismatch``, and a key whose
directory already holds circuits at the wrong T is refused (measure it into a new ``--out-root``).

PLAN: ``hpc/fixed_code_plan.py`` writes it (schema ``fixed_code_plan_v1``); the protocol root
holds ``<protocol_run>/protocol_snapshot.json`` and ``campaign_spec.json`` of the formal runs
(the package ships them under ``protocols/``).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "fixed_code_plan_v1"
RESULT_SCHEMA = "fixed_code_result_v1"
TASK_PROTOCOL_PROFILE = {
    "memory": "telescoping_structural_v3_7",
    "cnot_dense": "telescoping_structural_v3_7_cnot_v2",
}
# 2026-11-06: the structural profiles whose snapshots may serve a task. The search profile does not
# enter a fixed-code measurement (no search runs); the task, noise, SEC policy and decoder do.
TASK_PROTOCOL_PROFILES = {
    "memory": ("telescoping_structural_v3_7", "telescoping_structural_v3_8"),
    "cnot_dense": ("telescoping_structural_v3_7_cnot_v2",),
}

MERGE_FIELDS = [
    "key", "block", "measurement_id", "task", "budget", "code_hash", "family", "n", "k", "d_search", "w_max",
    "p", "rounds_floor", "rounds_policy", "rounds", "expected_rounds", "logicals", "logicals_measured",
    "shots_per_basis_planned", "shots_x", "shots_z", "failures_x", "failures_z", "ler_x", "ler_z", "block_ler",
    "plpr", "plpr_ci_low", "plpr_ci_high", "status", "failure_reason", "stopped_by",
    "seed_namespace", "seed", "runtime_s", "protocol_run", "protocol_profile", "noise_profile", "noise_config_hash",
    "sec_policy_id", "decoder", "decoder_preset", "circuit_sha256_x", "circuit_sha256_z", "dem_sha256_x", "dem_sha256_z",
    "circuit_rounds_x", "circuit_rounds_z",
    "circuit_data_qubits", "symplectic_stabilizer_sha256", "h_mask_sha256", "wl_hash", "canonical_hash", "max_check_weight_measured",
    "formal_shots_per_basis", "formal_failures_x", "formal_failures_z",
    "pooled_shots_per_basis", "pooled_failures_x", "pooled_failures_z", "pooled_plpr", "pooled_plpr_ci_low", "pooled_plpr_ci_high",
    "origin", "result_path",
]


# --------------------------------------------------------------------------- plan access
def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"{path}: schema {plan.get('schema')!r} is not {PLAN_SCHEMA!r}")
    keys = [item["key"] for item in plan["items"]]
    if len(keys) != len(set(keys)):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise SystemExit(f"{path}: duplicate item keys {dup[:5]}")
    return plan


def plan_item(plan: dict[str, Any], key: str) -> dict[str, Any]:
    for item in plan["items"]:
        if item["key"] == key:
            return item
    raise SystemExit(f"key {key!r} not in plan ({len(plan['items'])} items)")


def result_path(out_root: str | Path, key: str) -> Path:
    return Path(out_root) / key / "result.json"


def load_result(out_root: str | Path, key: str) -> dict[str, Any] | None:
    path = result_path(out_root, key)
    if not path.is_file():
        return None
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return res if isinstance(res, dict) else None


# --------------------------------------------------------------------------- rounds check (2026-10-27)
# Before the fix of 2026-10-27 the build worker ignored rounds_policy: a fixed_Tc condition whose
# proxy distance exceeded T_c was MEASURED at T = d (the candidate's X/Z memory_experiment.rounds)
# while the row, and its plpr, reported T_c. A condition counts as done only when the measured
# circuits, the row and the plan agree on T.
def expected_rounds_of(item: dict[str, Any]) -> int | None:
    """The T the plan requires: ``expected_rounds``, else T_c under ``fixed_Tc``, else unknown."""

    if item.get("expected_rounds") is not None:
        return int(item["expected_rounds"])
    if str(item.get("rounds_policy") or "") == "fixed_Tc" and item.get("rounds_floor") is not None:
        return int(item["rounds_floor"])
    return None


def candidate_file(out_root: str | Path, key: str, res: dict[str, Any]) -> Path | None:
    """The measured candidate's file (``T<floor>/candidates/<program_id>.json``)."""

    cfg = res.get("config") or {}
    row = res.get("row") or {}
    cand_dir = Path(out_root) / key / f"T{cfg.get('rounds_floor')}" / "candidates"
    if row.get("program_id") and (cand_dir / f"{row['program_id']}.json").is_file():
        return cand_dir / f"{row['program_id']}.json"
    if cand_dir.is_dir():
        files = sorted(cand_dir.glob("*.json"))
        return files[0] if files else None
    return None


def circuit_rounds(cand: dict[str, Any]) -> dict[str, int | None]:
    """T of the measured X / Z circuits (``results[*].basis.<B>.memory_experiment.rounds``);
    None when absent or not one value across the candidate's results."""

    found: dict[str, set[int]] = {"X": set(), "Z": set()}
    for res in cand.get("results") or []:
        if not isinstance(res, dict):
            continue
        for name in ("X", "Z"):
            meta = ((res.get("basis") or {}).get(name) or {}).get("memory_experiment") or {}
            if meta.get("rounds") is not None:
                found[name].add(int(meta["rounds"]))
    return {name: (values.pop() if len(values) == 1 else None) for name, values in found.items()}


def rounds_errors(item: dict[str, Any], res: dict[str, Any], cand: dict[str, Any] | None) -> list[str]:
    """Disagreements between the measured circuits' T, the row's T and the plan's T (empty = consistent)."""

    row = res.get("row") or {}
    row_rounds = None if row.get("rounds") is None else int(row["rounds"])
    expected = expected_rounds_of(item)
    errors: list[str] = []
    if row_rounds is None:
        errors.append("row.rounds missing")
    if expected is not None and row_rounds is not None and row_rounds != expected:
        errors.append(f"row.rounds={row_rounds}, plan expects {expected}")
    if cand is None:
        errors.append("candidate file missing (measured circuit T unknown)")
        return errors
    for name, value in circuit_rounds(cand).items():
        if value is None:
            errors.append(f"candidate.{name}.memory_experiment.rounds missing")
            continue
        if row_rounds is not None and value != row_rounds:
            errors.append(f"candidate.{name}.memory_experiment.rounds={value} != row.rounds={row_rounds}")
        if expected is not None and value != expected:
            errors.append(f"candidate.{name}.memory_experiment.rounds={value}, plan expects {expected}")
    return errors


def rounds_verdict(out_root: str | Path, key: str, item: dict[str, Any] | None = None) -> dict[str, Any]:
    """``pending`` (no result.json) | ``not_ok`` (row status not ok) | ``rounds_mismatch`` | ``valid``."""

    res = load_result(out_root, key)
    if res is None:
        return {"verdict": "pending", "key": key, "errors": []}
    item = item or res.get("plan_item") or {"key": key}
    row = res.get("row") or {}
    info: dict[str, Any] = {
        "key": key, "status": row.get("status") or res.get("queue_status") or "no_row",
        "rounds_policy": item.get("rounds_policy"), "expected_rounds": expected_rounds_of(item),
        "row_rounds": row.get("rounds"), "circuit_rounds_x": None, "circuit_rounds_z": None, "errors": [],
    }
    path = candidate_file(out_root, key, res)
    cand = None
    if path is not None:
        try:
            cand = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cand = None
    if cand is not None:
        rounds = circuit_rounds(cand)
        info["circuit_rounds_x"], info["circuit_rounds_z"] = rounds["X"], rounds["Z"]
    if info["status"] != "ok":
        info["verdict"] = "not_ok"
        return info
    info["errors"] = rounds_errors(item, res, cand)
    info["verdict"] = "rounds_mismatch" if info["errors"] else "valid"
    return info


def stale_measurement_errors(out_root: str | Path, item: dict[str, Any]) -> list[str]:
    """Candidate files already in this condition's directory whose circuits ran at a T the plan does not
    allow (a pre-fix measurement). Measuring over them would reuse them (final-stage resume keeps any
    candidate of an identical run config) or overwrite raw evidence, so the condition is refused."""

    expected = expected_rounds_of(item)
    if expected is None:
        return []
    cand_dir = Path(out_root) / item["key"] / f"T{int(item.get('rounds_floor') or 12)}" / "candidates"
    errors: list[str] = []
    for path in sorted(cand_dir.glob("*.json")) if cand_dir.is_dir() else []:
        try:
            cand = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, value in circuit_rounds(cand).items():
            if value is not None and value != expected:
                errors.append(f"{path.name}: {name} circuit T={value}, plan expects T={expected}")
    return errors


def is_done(out_root: str | Path, key: str, item: dict[str, Any] | None = None) -> bool:
    """result.json says ok AND the measured circuits, the row and the plan agree on T (2026-10-27)."""

    return rounds_verdict(out_root, key, item)["verdict"] == "valid"


def select_keys(plan: dict[str, Any], out_root: str | Path, *, pending: bool, chunk: str | None) -> list[str]:
    keys = [item["key"] for item in plan["items"]]
    if pending:
        keys = [item["key"] for item in plan["items"] if not is_done(out_root, item["key"], item)]
    if chunk:
        index, count = (int(x) for x in chunk.split("/"))
        if not (1 <= index <= count):
            raise SystemExit(f"--chunk {chunk}: index must be in 1..{count}")
        size = math.ceil(len(keys) / count) if keys else 0
        keys = keys[(index - 1) * size: index * size]
    return keys


def find_protocol_dir(roots: list[str], name: str) -> Path:
    for root in roots:
        candidate = Path(root) / name
        if (candidate / "protocol_snapshot.json").is_file() and (candidate / "campaign_spec.json").is_file():
            return candidate
    raise SystemExit(f"{name}/protocol_snapshot.json + campaign_spec.json not found under {roots}")


# --------------------------------------------------------------------------- verification
def build_verification(protocol: dict[str, Any], campaign_spec: dict[str, Any], item: dict[str, Any], *,
                       decode_shot_budget_s: float | None = None):
    """The protocol's verification with the item's explicit knobs applied (see the module doc)."""

    from omniqec_agentic.config import ROUNDS_POLICY_IDS
    from omniqec_agentic.workflow.graph import _protocol_final_rounds, _verification_from_protocol

    task = str(item["task"])
    if task not in TASK_PROTOCOL_PROFILE:
        raise SystemExit(f"item {item['key']}: unknown task {task!r}")
    policy = str(item["rounds_policy"])
    if policy not in ROUNDS_POLICY_IDS:
        raise SystemExit(f"item {item['key']}: rounds_policy {policy!r} not in {ROUNDS_POLICY_IDS}")
    rounds_floor = int(item.get("rounds_floor") or _protocol_final_rounds(protocol))
    p = float(item["p"])
    if not (0.0 < p < 0.5):
        raise SystemExit(f"item {item['key']}: p={p} out of range")
    shots = int(item["shots_per_basis"])
    if shots < 1:
        raise SystemExit(f"item {item['key']}: shots_per_basis must be >= 1")
    verification = _verification_from_protocol(protocol, campaign_spec, p_values=(p,), shots=shots, rounds=rounds_floor)
    protocol_task = str(getattr(verification, "task", "memory"))
    if protocol_task != task:
        raise SystemExit(
            f"item {item['key']}: task {task!r} but protocol {item.get('protocol_run')!r} measures {protocol_task!r}"
        )
    profile = str(protocol.get("protocol_profile") or "")
    if profile not in TASK_PROTOCOL_PROFILES[task]:
        raise SystemExit(f"item {item['key']}: protocol profile {profile!r} is not a {task} profile {TASK_PROTOCOL_PROFILES[task]!r}")
    replacements: dict[str, Any] = {
        "rounds_policy": policy,
        # a fixed-code evaluation never applies a search weight cap (plan v28 §4; the item may
        # still name one explicitly, e.g. for a hardware study)
        "max_check_weight": (None if item.get("max_check_weight") is None else int(item["max_check_weight"])),
        # fixed shots: no racing, no max_errors stop, no shard ceiling
        "max_errors": None,
        "final_decode_shard_timeout_cap_s": None,
        "search_decode_shard_timeout_cap_s": None,
    }
    if decode_shot_budget_s is not None:
        if float(decode_shot_budget_s) <= 0:
            raise SystemExit("--decode-shot-budget-s must be > 0")
        replacements["decode_shot_budget_s"] = float(decode_shot_budget_s)
    return dataclasses.replace(verification, **replacements)


def shot_shards_for(protocol: dict[str, Any], shots: int, override: int | None) -> int:
    if override:
        return max(1, int(override))
    # ~300 shots per shard, never fewer shards than the formal final (fig2_extended_stats rule)
    return int(max(int(protocol.get("final_shot_shards") or 128), 2 ** math.ceil(math.log2(max(1.0, shots / 300.0)))))


# --------------------------------------------------------------------------- measure
def measure_one(plan: dict[str, Any], item: dict[str, Any], args: argparse.Namespace) -> int:
    from omniqec_agentic.evaluation import evaluate_promotion_queue
    from omniqec_agentic.workflow.graph import final_rng_base_seed

    protocol_dir = find_protocol_dir(args.protocol_root, item["protocol_run"])
    protocol = json.loads((protocol_dir / "protocol_snapshot.json").read_text(encoding="utf-8"))
    campaign_spec = json.loads((protocol_dir / "campaign_spec.json").read_text(encoding="utf-8"))
    verification = build_verification(protocol, campaign_spec, item, decode_shot_budget_s=args.decode_shot_budget_s)
    rounds_floor = int(verification.rounds)
    shots = int(verification.shots)
    shards = shot_shards_for(protocol, shots, args.shot_shards)
    namespace = str(item["seed_namespace"])
    seed = final_rng_base_seed(namespace)
    workers = int(args.workers if args.workers is not None else (protocol.get("slow_workers") or 0))
    out_dir = Path(args.out_root) / item["key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / f"queue_T{rounds_floor}.jsonl"
    record = {
        "program_id": item.get("program_id") or item["key"],
        "spec": item["spec"],
        "code_hash": item.get("code_hash"),
        "selection_channels": ["fixed_code", str(item.get("block") or plan.get("block") or "")],
        "is_baseline": False,
        "retry_at_final": False,
        "canonical_hash": None,
    }
    queue_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    config = {
        "schema": "fixed_code_measure_config",
        "plan_schema": plan.get("schema"),
        "plan_version": plan.get("version"),
        "block": item.get("block") or plan.get("block"),
        "key": item["key"],
        "measurement_id": item.get("measurement_id"),
        "task": item["task"],
        "protocol_run": item["protocol_run"],
        "protocol_dir": str(protocol_dir),
        "protocol_profile": protocol.get("protocol_profile"),
        "p": float(item["p"]),
        "shots_per_basis": shots,
        "rounds_floor": rounds_floor,
        "rounds_policy": str(verification.rounds_policy),
        "max_check_weight": verification.max_check_weight,
        "decode_shot_budget_s": float(verification.decode_shot_budget_s),
        "seed_namespace": namespace,
        "seed": seed,
        "shot_shards": shards,
        "workers": workers,
        "racing": None,
        "final_decode_shard_timeout_cap_s": None,
        "verification": verification.to_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "fixed_code_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[fixed] {item['key']} task={item['task']} p={item['p']} shots/basis={shots} T_c={rounds_floor} "
        f"policy={verification.rounds_policy} shards={shards} workers={workers} seed={seed}",
        flush=True,
    )
    t0 = time.time()
    summary = evaluate_promotion_queue(
        queue_path=queue_path,
        out_dir=out_dir / f"T{rounds_floor}",
        verification=verification,
        include_baseline=False,
        resume_candidates=True,
        workers=workers,
        shot_shards=shards,
        seed=seed,
        stage="final",
    )
    rankings_path = out_dir / f"T{rounds_floor}" / "rankings.json"
    rows = json.loads(rankings_path.read_text(encoding="utf-8")).get("rankings") or [] if rankings_path.is_file() else []
    row = rows[0] if rows else None
    result = {
        "schema": RESULT_SCHEMA,
        "plan_item": item,
        "config": config,
        "queue_status": summary.get("status"),
        "runtime_s": round(time.time() - t0, 1),
        "row": row,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rounds_problems: list[str] = []
    if row is not None and row.get("status") == "ok":
        cand_path = candidate_file(args.out_root, item["key"], result)
        cand = json.loads(cand_path.read_text(encoding="utf-8")) if cand_path is not None else None
        rounds_problems = rounds_errors(item, result, cand)
        result["rounds_check"] = {
            "circuit_rounds": circuit_rounds(cand) if cand is not None else None,
            "row_rounds": row.get("rounds"), "expected_rounds": expected_rounds_of(item), "errors": rounds_problems,
        }
    result_path(args.out_root, item["key"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if row is None:
        print(f"[fixed] {item['key']} NO ROW (queue status {summary.get('status')})", flush=True)
        return 2
    if rounds_problems:
        print(f"[fixed] {item['key']} ROUNDS MISMATCH, not a valid measurement: {'; '.join(rounds_problems)}", flush=True)
        return 4
    print(
        f"[fixed] {item['key']} status={row.get('status')} T={row.get('rounds')} failures X/Z="
        f"{row.get('failures_x')}/{row.get('failures_z')} of {row.get('shots_x')}/{row.get('shots_z')} "
        f"plpr={row.get('per_logical_per_round')} ci={row.get('per_logical_per_round_ci')} in {result['runtime_s']} s",
        flush=True,
    )
    return 0 if row.get("status") == "ok" else 3


def measure(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    keys = list(args.key or [])
    if args.keys_file:
        keys += [line.strip() for line in Path(args.keys_file).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not keys:
        raise SystemExit("--key or --keys-file is required to measure")
    worst = 0
    for key in keys:
        item = plan_item(plan, key)
        if is_done(args.out_root, key, item) and not args.force:
            print(f"[fixed] {key} already measured (result.json status ok, rounds consistent); skipping", flush=True)
            continue
        stale = stale_measurement_errors(args.out_root, item)
        if stale:
            # never measure over (or resume from) a pre-fix measurement: use a new --out-root
            print(f"[fixed] {key} REFUSED: {args.out_root}/{key} holds a measurement at the wrong T "
                  f"({'; '.join(stale)}); measure it into a new --out-root", flush=True)
            worst = max(worst, 5)
            if args.stop_on_error:
                return 5
            continue
        rc = measure_one(plan, item, args)
        worst = max(worst, rc)
        if rc and args.stop_on_error:
            return rc
    return worst


# --------------------------------------------------------------------------- merge
def combined_plpr(fx: int, fz: int, sx: int, sz: int, logicals: int, rounds: int) -> tuple[Any, Any, Any]:
    from omniqec_agentic.verification.ler import combine_basis_ler_results

    res = combine_basis_ler_results(
        {"X": {"logical_error_rate": fx / sx, "failures": fx, "shots": sx},
         "Z": {"logical_error_rate": fz / sz, "failures": fz, "shots": sz}},
        rounds=int(rounds), k=int(logicals),
    )
    ci = res.get("per_logical_per_round_ci") or [None, None]
    return res.get("per_logical_per_round"), ci[0], ci[1]


def merge_row(item: dict[str, Any], out_root: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {f: None for f in MERGE_FIELDS}
    out.update({
        "key": item["key"], "block": item.get("block"), "measurement_id": item.get("measurement_id"), "task": item["task"],
        "budget": item.get("budget"), "code_hash": item.get("code_hash"), "family": item.get("family") or (item.get("spec") or {}).get("family"),
        "n": item.get("n"), "k": item.get("k"), "d_search": item.get("d_search"), "w_max": item.get("w_max"), "p": item.get("p"),
        "rounds_floor": item.get("rounds_floor"), "rounds_policy": item.get("rounds_policy"), "expected_rounds": item.get("expected_rounds"),
        "logicals": item.get("logicals"), "shots_per_basis_planned": item.get("shots_per_basis"), "seed_namespace": item.get("seed_namespace"),
        "protocol_run": item.get("protocol_run"), "origin": json.dumps(item.get("origin") or {}, ensure_ascii=False, sort_keys=True),
        "formal_shots_per_basis": item.get("formal_shots_per_basis"), "formal_failures_x": item.get("formal_failures_x"),
        "formal_failures_z": item.get("formal_failures_z"), "status": "pending",
    })
    path = result_path(out_root, item["key"])
    out["result_path"] = str(path)
    if not path.is_file():
        return out
    res = json.loads(path.read_text(encoding="utf-8"))
    cfg = res.get("config") or {}
    row = res.get("row") or {}
    ver = cfg.get("verification") or {}
    out.update({
        "status": row.get("status") or res.get("queue_status") or "no_row", "failure_reason": row.get("failure_reason"),
        "stopped_by": row.get("stopped_by"), "seed": cfg.get("seed"), "runtime_s": res.get("runtime_s"),
        "protocol_profile": cfg.get("protocol_profile"), "noise_profile": row.get("noise_profile") or ((ver.get("noise") or {}).get("profile")),
        "noise_config_hash": row.get("noise_config_hash"), "sec_policy_id": (ver.get("sec_policy") or {}).get("sec_policy_id"),
        "decoder": row.get("decoder"), "decoder_preset": ((ver.get("decoder_build") or {}).get("telescoping") or {}).get("preset"),
        "rounds": row.get("rounds"), "shots_x": row.get("shots_x"), "shots_z": row.get("shots_z"),
        "failures_x": row.get("failures_x"), "failures_z": row.get("failures_z"), "ler_x": row.get("ler_x"), "ler_z": row.get("ler_z"),
        "block_ler": row.get("block_ler"), "plpr": row.get("per_logical_per_round"),
        "plpr_ci_low": (row.get("per_logical_per_round_ci") or [None, None])[0],
        "plpr_ci_high": (row.get("per_logical_per_round_ci") or [None, None])[1],
        "wl_hash": row.get("wl_hash"), "canonical_hash": row.get("canonical_hash"), "max_check_weight_measured": row.get("max_check_weight"),
    })
    if out["n"] is None:
        out["n"] = row.get("n")
    if out["k"] is None:
        out["k"] = row.get("k")
    if out["d_search"] is None:
        out["d_search"] = row.get("d_search")
    if out["family"] is None:
        out["family"] = row.get("family")
    # the candidate file: code identity hashes and (new worker fields) the measured circuit hashes
    cand_path = candidate_file(out_root, item["key"], res)
    cand = None
    if cand_path is not None:
        cand = json.loads(cand_path.read_text(encoding="utf-8"))
        compiled = cand.get("compiled_code") or {}
        out["symplectic_stabilizer_sha256"] = compiled.get("symplectic_stabilizer_sha256")
        out["h_mask_sha256"] = compiled.get("h_mask_sha256")
        # the measured circuits: results[0].basis.<B>.memory_experiment is MemoryExperiment.to_metadata()
        # (noisy_circuit_sha256, dem_sha256, num_logicals, num_data_qubits = blocks x n, rounds)
        results = cand.get("results") or []
        basis = (results[0].get("basis") or {}) if results and isinstance(results[0], dict) else {}
        logicals = set()
        for name in ("X", "Z"):
            meta = (basis.get(name) or {}).get("memory_experiment") or {}
            out[f"circuit_sha256_{name.lower()}"] = meta.get("noisy_circuit_sha256")
            out[f"dem_sha256_{name.lower()}"] = meta.get("dem_sha256")
            if meta.get("num_logicals") is not None:
                logicals.add(int(meta["num_logicals"]))
            if meta.get("num_data_qubits") is not None:
                out["circuit_data_qubits"] = int(meta["num_data_qubits"])
        out["logicals_measured"] = logicals.pop() if len(logicals) == 1 else None
        measured = circuit_rounds(cand)
        out["circuit_rounds_x"], out["circuit_rounds_z"] = measured["X"], measured["Z"]
    # 2026-10-27: an ok row whose measured circuits ran at another T than the row / plan is not a result
    if out["status"] == "ok":
        problems = rounds_errors(item, res, cand)
        if problems:
            out["status"] = "rounds_mismatch"
            out["failure_reason"] = "; ".join(problems)
    # pooled with the formal counts (supplements): fixed-shot independent samples of one code under one setting
    if (out["status"] == "ok" and item.get("formal_failures_x") is not None and item.get("formal_shots_per_basis")
            and out["shots_x"] and out["shots_z"]):
        logicals = int(out["logicals_measured"] or item.get("logicals") or out["k"])
        rounds = int(out["rounds"])
        fx = int(item["formal_failures_x"]) + int(out["failures_x"]); fz = int(item["formal_failures_z"]) + int(out["failures_z"])
        sx = int(item["formal_shots_per_basis"]) + int(out["shots_x"]); sz = int(item["formal_shots_per_basis"]) + int(out["shots_z"])
        pooled = combined_plpr(fx, fz, sx, sz, logicals, rounds)
        out.update({"pooled_shots_per_basis": sx, "pooled_failures_x": fx, "pooled_failures_z": fz,
                    "pooled_plpr": pooled[0], "pooled_plpr_ci_low": pooled[1], "pooled_plpr_ci_high": pooled[2]})
    return out


def merge(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    rows = [merge_row(item, args.out_root) for item in plan["items"]]
    handle = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=MERGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            handle.close()
    done = sum(1 for r in rows if r["status"] == "ok")
    print(f"[fixed] merged {len(rows)} items: {done} ok, {sum(1 for r in rows if r['status'] == 'pending')} pending, "
          f"{len(rows) - done - sum(1 for r in rows if r['status'] == 'pending')} other", file=sys.stderr)
    return 0


def audit_rounds(args: argparse.Namespace) -> int:
    """One tab-separated line per plan item; the counts go to stderr."""

    plan = load_plan(args.plan)
    counts: dict[str, int] = {}
    print("#verdict\tkey\trounds_policy\texpected_T\trow_T\tcircuit_T_X\tcircuit_T_Z\tstatus\terrors")
    for item in plan["items"]:
        info = rounds_verdict(args.out_root, item["key"], item)
        counts[info["verdict"]] = counts.get(info["verdict"], 0) + 1
        fields = [info["verdict"], item["key"], item.get("rounds_policy"), expected_rounds_of(item), info.get("row_rounds"),
                  info.get("circuit_rounds_x"), info.get("circuit_rounds_z"), info.get("status"), "; ".join(info.get("errors") or [])]
        print("\t".join("" if f is None else str(f) for f in fields))
    print(f"[fixed] rounds audit of {len(plan['items'])} items: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--protocol-root", action="append", default=[], help="directory holding <protocol_run>/protocol_snapshot.json (repeatable)")
    parser.add_argument("--key", action="append", help="plan item to measure (repeatable)")
    parser.add_argument("--keys-file", help="one key per line; finished keys are skipped")
    parser.add_argument("--workers", type=int, default=None, help="default: the protocol's slow_workers (0 = cpu_count-1)")
    parser.add_argument("--shot-shards", type=int, default=None)
    parser.add_argument("--decode-shot-budget-s", type=float, default=None, help="raise the per-shot watchdog budget for a re-run of a timed-out condition")
    parser.add_argument("--force", action="store_true", help="measure even when result.json says ok (a new result overwrites)")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--list-keys", action="store_true", help="print the plan's keys and exit")
    parser.add_argument("--pending", action="store_true", help="with --list-keys: only keys without an ok result")
    parser.add_argument("--chunk", help="with --list-keys: i/n, the i-th of n equal slices")
    parser.add_argument("--audit-rounds", action="store_true",
                        help="print verdict (valid | rounds_mismatch | not_ok | pending), key, policy, plan/row/circuit T per plan item")
    parser.add_argument("--merge", action="store_true", help="write the merged CSV instead of measuring")
    parser.add_argument("--out", help="with --merge: CSV path (default stdout)")
    args = parser.parse_args(argv)
    if args.list_keys:
        for key in select_keys(load_plan(args.plan), args.out_root, pending=args.pending, chunk=args.chunk):
            print(key)
        return 0
    if args.audit_rounds:
        return audit_rounds(args)
    if args.merge:
        return merge(args)
    if not args.protocol_root:
        parser.error("--protocol-root is required to measure")
    return measure(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
