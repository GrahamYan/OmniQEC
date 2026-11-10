"""Top-5 extension of a finished run's final stage (final protocol of Fig.2 and Fig.3, 2026-11-06).

2026-11-10 (paper alignment): LEGACY TOOL. A run launched by the current code carries the paper's own final
stage in its protocol snapshot (``final_stage.id = paper_top5_cumulative_v1``, evaluation/final_stage.py):
the campaign itself confirms the five leading distinct codes by search-stage pLER, tops every one up to the
Table VII limit and names the champion (final_slow/final_ranking.json). This tool is for runs whose final
stage is the earlier 2 structural + 4 runtime shortlist at 25 000 shots; it refuses a paper-final run. Its
rule below (Top-5 among those finalists by FINAL pLPR, an adaptive target) is not the paper's rule; the caps
file is Table VII and is pinned to config.FINAL_CUMULATIVE_SHOT_CAPS by the tests.

RULE (fixed before any extension is measured)
  * Top-5: the five best measurable non-reference finalists of the run's final stage, ordered by the
    final pLPR, ties broken by the smaller upper end of the 95 % interval. A run with fewer measurable
    finalists contributes all of them.
  * A Top-5 code with fewer than 30 failures in a basis is extended ONCE. With f = min(F_X, F_Z) of its
    final-stage measurement (25 000 shots per basis), the target total number of shots per basis is

        target = cap(task, N)                                               if f = 0
        target = min(cap(task, N), 50 000 * ceil(45 * 25 000 / f / 50 000))   if 0 < f < 30

    and the extension is the target minus the shots already measured for that code (its final stage and
    any earlier extension passed with --prior-merged), never negative.
  * cap(task, N): the per-basis cap of the budget point, read from a caps file
    (``top5_extension_caps.json`` next to this module).
  * Measurement: ``hpc/fixed_code_measure.py`` under the run's own protocol snapshot, fixed shots, no
    racing, an independent seed namespace per part (``top5_extension_v1/<key>``), at the run's own rounds
    policy (``rounds_policy.final`` of its snapshot: fixed_Tc, T = 12 for every code, in the final
    protocols of both figures; a snapshot without the key is a memory run at max_Tc_d, T = max(12, d_search)).
    An extension larger than the per-job part size is split into parts, each with its own seed.
  * Champion: the Top-5 code with the lowest pooled pLPR (final stage + extension, failures and shots
    added per basis), ties broken by the smaller upper interval end. There is no second round: a code
    still below 30 failures at its cap is reported as measured (``ge30_each_basis = NO``).

USAGE
    python -m omniqec_agentic.hpc.top5_extension plan --runs-root RUNS --run NAME [NAME ...] \
        [--caps CAPS.json] [--prior-merged OLD.csv] --out top5_plan.json
    python -m omniqec_agentic.hpc.fixed_code_measure --plan top5_plan.json --protocol-root RUNS \
        --out-root OUT --keys-file keys.txt                      # measure (one or several keys per job)
    python -m omniqec_agentic.hpc.top5_extension select --runs-root RUNS --run NAME [NAME ...] \
        --plan top5_plan.json --out-root OUT [--prior-merged OLD.csv] --out top5_pooled.csv

``select`` refuses to name a champion for a run whose planned extension is not complete
(missing, failed or at the wrong T) unless ``--allow-incomplete`` is given; such a run is reported with
``run_status = incomplete``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "fixed_code_plan_v1"
CAPS_SCHEMA = "top5_extension_caps_v1"
DEFAULT_CAPS = Path(__file__).with_name("top5_extension_caps.json")
NAMESPACE_ROOT = "top5_extension_v1"
TOP = 5
TARGET_FAILURES = 30
EXPECTED_FAILURES = 45.0
ROUND_TO = 50_000
FORMAL_SHOTS = 25_000
DEFAULT_PART_CAP = {"memory": 900_000, "cnot_dense": 150_000}
RULE = (
    "Top-5 finalists by final pLPR (ties: smaller CI upper end); f = min(final F_X, F_Z); f >= 30 -> no extension; "
    "f = 0 -> target = cap(task, N); else target = min(cap(task, N), 50000 * ceil(45 * 25000 / f / 50000)); "
    "extension = target - shots already measured; one round; champion = lowest pooled pLPR (ties: smaller CI upper end)"
)


# --------------------------------------------------------------------------- inputs
def load_caps(path: str | Path) -> dict[str, dict[int, int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != CAPS_SCHEMA:
        raise SystemExit(f"{path}: schema {data.get('schema')!r}, expected {CAPS_SCHEMA!r}")
    caps: dict[str, dict[int, int]] = {}
    for task, table in (data.get("caps") or {}).items():
        caps[str(task)] = {int(budget): int(value) for budget, value in table.items()}
        if any(value < FORMAL_SHOTS for value in caps[str(task)].values()):
            raise SystemExit(f"{path}: every cap must be >= the final-stage shots ({FORMAL_SHOTS})")
    return caps


def find_run(roots: list[str], name: str) -> Path:
    for root in roots:
        candidate = Path(root) / name
        if (candidate / "protocol_snapshot.json").is_file():
            return candidate
    raise SystemExit(f"{name}: protocol_snapshot.json not found under {roots}")


def _final_dir(run_dir: Path) -> Path:
    dirs = sorted(p.parent for p in (run_dir / "final_slow").glob("T*/rankings.json"))
    if len(dirs) != 1:
        raise SystemExit(f"{run_dir.name}: expected one final_slow/T*/rankings.json, found {len(dirs)} (final stage not finished?)")
    return dirs[0]


def _ci_high(row: dict[str, Any]) -> float:
    ci = row.get("per_logical_per_round_ci") or [None, None]
    try:
        return float(ci[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")


def run_info(run_dir: Path) -> dict[str, Any]:
    protocol = json.loads((run_dir / "protocol_snapshot.json").read_text(encoding="utf-8"))
    task = str((protocol.get("verification") or {}).get("task") or "memory")
    budget = protocol.get("n_budget")
    if budget is None:
        match = re.search(r"_b(\d+)_", run_dir.name)
        if not match:
            raise SystemExit(f"{run_dir.name}: no n_budget in the snapshot and none in the run name")
        budget = match.group(1)
    if isinstance(protocol.get("final_stage"), dict):
        raise SystemExit(
            f"{run_dir.name}: final stage {protocol['final_stage'].get('id')!r} already confirms its finalists "
            "inside the campaign (final_slow/final_ranking.json); this legacy tool does not apply"
        )
    rounds = protocol.get("rounds_policy") or {}
    policy = str(rounds.get("final") or ("fixed_Tc" if task == "cnot_dense" else "max_Tc_d"))
    if rounds.get("search") and str(rounds["search"]) != policy:
        raise SystemExit(f"{run_dir.name}: search rounds policy {rounds['search']!r} differs from the final {policy!r}")
    return {"task": task, "budget": int(budget), "profile": protocol.get("protocol_profile"), "rounds_policy": policy}


def top5(run_dir: Path, top: int = TOP) -> list[dict[str, Any]]:
    """The run's Top-5 finalists, each with the spec its final stage measured.

    The spec comes from the final queue, matched by code hash: two finalists that share a program id
    overwrite each other's candidate file, the queue keeps both."""

    final_dir = _final_dir(run_dir)
    data = json.loads((final_dir / "rankings.json").read_text(encoding="utf-8"))
    rows = data.get("rankings") if isinstance(data, dict) else data
    ok = [r for r in rows or [] if isinstance(r, dict) and r.get("status") == "ok" and not r.get("is_baseline")
          and r.get("per_logical_per_round") is not None]
    ok.sort(key=lambda r: (float(r["per_logical_per_round"]), _ci_high(r)))
    queue_files = sorted((run_dir / "final_slow").glob("queue_T*.jsonl"))
    if len(queue_files) != 1:
        raise SystemExit(f"{run_dir.name}: expected one final_slow/queue_T*.jsonl, found {len(queue_files)}")
    specs: dict[str, dict[str, Any]] = {}
    for line in queue_files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            specs[str(entry.get("code_hash"))] = entry.get("spec")
    out = []
    for rank, row in enumerate(ok[:top], start=1):
        spec = specs.get(str(row.get("code_hash")))
        if not spec:
            raise SystemExit(f"{run_dir.name}: no queued spec for finalist {row.get('code_hash')}")
        out.append({**row, "final_rank": rank, "spec": spec})
    return out


def read_prior(path: str | Path | None) -> dict[tuple[str, str], dict[str, int]]:
    """Earlier extension measurements (a fixed_code_measure --merge table): status ok rows only."""

    prior: dict[tuple[str, str], dict[str, int]] = {}
    if not path:
        return prior
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            if str(row.get("shots_x")) != str(row.get("shots_z")):
                raise SystemExit(f"{path}: {row.get('key')} has unequal X/Z shots")
            key = (str(row["protocol_run"]), str(row["code_hash"]))
            acc = prior.setdefault(key, {"shots": 0, "failures_x": 0, "failures_z": 0, "parts": 0})
            acc["shots"] += int(row["shots_x"])
            acc["failures_x"] += int(row["failures_x"])
            acc["failures_z"] += int(row["failures_z"])
            acc["parts"] += 1
    return prior


# --------------------------------------------------------------------------- rule
def target_total(f_min: int, cap: int) -> int:
    """Target shots per basis (final stage included) for a code whose rarer basis failed f_min times."""

    if f_min >= TARGET_FAILURES:
        return 0
    if f_min <= 0:
        return int(cap)
    need = EXPECTED_FAILURES * FORMAL_SHOTS / float(f_min)
    return int(min(cap, ROUND_TO * math.ceil(need / ROUND_TO)))


def split_parts(total: int, part_cap: int) -> list[int]:
    if total <= 0:
        return []
    n = math.ceil(total / part_cap)
    base = total // n
    parts = [base] * n
    parts[0] += total - base * n
    return parts


# --------------------------------------------------------------------------- plan
def build_plan(roots: list[str], runs: list[str], caps: dict[str, dict[int, int]], *, prior: dict | None = None,
               part_cap: dict[str, int] | None = None, check_specs: bool = True) -> tuple[dict[str, Any], list[str]]:
    part_cap = {**DEFAULT_PART_CAP, **(part_cap or {})}
    prior = prior or {}
    plan: dict[str, Any] = {"schema": PLAN_SCHEMA, "block": "top5_extension", "version": "top5_extension_v1",
                            "seed_namespace_root": NAMESPACE_ROOT, "rule": {"rule": RULE, "caps": {t: {str(b): v for b, v in c.items()} for t, c in caps.items()},
                                                                            "part_cap": part_cap}, "items": []}
    report: list[str] = []
    for name in runs:
        run_dir = find_run(roots, name)
        info = run_info(run_dir)
        task, budget = info["task"], info["budget"]
        if task not in caps or budget not in caps[task]:
            raise SystemExit(f"{name}: no cap for task {task!r} at N = {budget}")
        cap = caps[task][budget]
        for row in top5(run_dir):
            code_hash = str(row.get("code_hash"))
            if check_specs:
                from omniqec_agentic.construction.build import build_compiled_code

                compiled = build_compiled_code(row["spec"])
                if compiled.code_hash != code_hash:
                    raise SystemExit(f"{name}: the queued spec of {code_hash} compiles to {compiled.code_hash}")
            fx, fz = int(row.get("failures_x") or 0), int(row.get("failures_z") or 0)
            sx, sz = int(row.get("shots_x") or 0), int(row.get("shots_z") or 0)
            if sx != sz:
                raise SystemExit(f"{name}: finalist {code_hash} has unequal final X/Z shots")
            f_min = min(fx, fz)
            target = target_total(f_min, cap)
            earlier = prior.get((name, code_hash), {}).get("shots", 0)
            extension = max(0, target - sx - earlier)
            parts = split_parts(extension, part_cap[task])
            k = int(row.get("k") or 0)
            report.append(f"{name} #{row['final_rank']} {code_hash[:8]} [[{row.get('n')},{k},{row.get('d_search')}]] "
                          f"w{row.get('max_check_weight')} final {fx}/{fz} @ {sx} -> target {target}, have {sx + earlier}, "
                          f"extend {extension} in {len(parts)} part(s)")
            for index, shots in enumerate(parts, start=1):
                key = f"T5_{name}_r{row['final_rank']}_{code_hash[:8]}_p{index}"
                plan["items"].append({
                    "key": key, "block": "top5_extension", "measurement_id": key, "task": task, "budget": budget,
                    "protocol_run": name, "spec": row["spec"], "code_hash": code_hash, "program_id": row.get("program_id"),
                    "family": row.get("family"), "n": row.get("n"), "k": k, "d_search": row.get("d_search"),
                    "w_max": row.get("max_check_weight"), "logicals": 2 * k if task == "cnot_dense" else k, "p": 0.005,
                    "shots_per_basis": int(shots), "rounds_floor": 12, "rounds_policy": info["rounds_policy"],
                    "expected_rounds": int(row.get("rounds") or 12), "max_check_weight": None,
                    "seed_namespace": f"{NAMESPACE_ROOT}/{key}",
                    "formal_shots_per_basis": sx, "formal_failures_x": fx, "formal_failures_z": fz,
                    "formal_plpr": row.get("per_logical_per_round"),
                    "origin": {"run": name, "final_rank": row["final_rank"], "program_id": row.get("program_id"), "f_min": f_min,
                               "cap": cap, "target_total": target, "earlier_extension": earlier, "extension_total": extension,
                               "part": index, "parts": len(parts)},
                })
    if len({item["key"] for item in plan["items"]}) != len(plan["items"]):
        raise SystemExit("duplicate plan keys")
    return plan, report


# --------------------------------------------------------------------------- select
def _pooled(fx: int, fz: int, shots: int, logicals: int, rounds: int) -> tuple[float, float, float]:
    from omniqec_agentic.verification.ler import combine_basis_ler_results

    res = combine_basis_ler_results({"X": {"logical_error_rate": fx / shots, "failures": fx, "shots": shots},
                                     "Z": {"logical_error_rate": fz / shots, "failures": fz, "shots": shots}},
                                    rounds=int(rounds), k=int(logicals))
    ci = res.get("per_logical_per_round_ci") or [None, None]
    return float(res["per_logical_per_round"]), float(ci[0]), float(ci[1])


def select(roots: list[str], runs: list[str], plan: dict[str, Any], out_roots: list[str], *, prior: dict | None = None,
           allow_incomplete: bool = False) -> list[dict[str, Any]]:
    from omniqec_agentic.hpc import fixed_code_measure as fcm

    prior = prior or {}
    items_by_code: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in plan.get("items") or []:
        items_by_code.setdefault((str(item["origin"]["run"]), str(item["code_hash"])), []).append(item)

    def result_of(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        for root in out_roots:
            res = fcm.load_result(root, item["key"])
            if res is None:
                continue
            row = res.get("row") or {}
            verdict = fcm.rounds_verdict(root, item["key"], item)
            if res.get("queue_status") != "ok" or row.get("status") != "ok" or row.get("stopped_by"):
                return None, f"{item['key']}: status {row.get('status') or res.get('queue_status')}"
            if verdict.get("verdict") != "valid":
                return None, f"{item['key']}: {verdict.get('verdict')} {verdict.get('errors') or ''}".rstrip()
            if int(row.get("shots_x") or 0) != int(item["shots_per_basis"]) or int(row.get("shots_z") or 0) != int(item["shots_per_basis"]):
                return None, f"{item['key']}: shots differ from the plan"
            if str(row.get("code_hash")) != str(item["code_hash"]):
                return None, f"{item['key']}: code hash differs from the plan"
            return row, "ok"
        return None, f"{item['key']}: not measured"

    out: list[dict[str, Any]] = []
    for name in runs:
        run_dir = find_run(roots, name)
        info = run_info(run_dir)
        rows, problems = [], []
        for row in top5(run_dir):
            code_hash = str(row.get("code_hash"))
            fx, fz, shots = int(row["failures_x"]), int(row["failures_z"]), int(row["shots_x"])
            ext_x = ext_z = ext_shots = 0
            parts = items_by_code.get((name, code_hash), [])
            for item in parts:
                res, why = result_of(item)
                if res is None:
                    problems.append(why)
                    continue
                ext_x += int(res["failures_x"]); ext_z += int(res["failures_z"]); ext_shots += int(res["shots_x"])
            earlier = prior.get((name, code_hash), {"shots": 0, "failures_x": 0, "failures_z": 0, "parts": 0})
            px, pz = fx + ext_x + earlier["failures_x"], fz + ext_z + earlier["failures_z"]
            ps = shots + ext_shots + earlier["shots"]
            k = int(row.get("k") or 0)
            logicals = 2 * k if info["task"] == "cnot_dense" else k
            plpr, lo, hi = _pooled(px, pz, ps, logicals, int(row.get("rounds") or 12))
            target = parts[0]["origin"]["target_total"] if parts else None
            rows.append({
                "run": name, "task": info["task"], "budget": info["budget"], "protocol_profile": info["profile"],
                "final_rank": row["final_rank"], "code_hash": code_hash, "code_hash8": code_hash[:8], "family": row.get("family"),
                "n": row.get("n"), "k": k, "d_search": row.get("d_search"), "w_max": row.get("max_check_weight"),
                "rounds": row.get("rounds"), "logicals": logicals,
                "final_failures_x": fx, "final_failures_z": fz, "final_shots_per_basis": shots,
                "final_plpr": row.get("per_logical_per_round"),
                "earlier_extension_shots": earlier["shots"], "extension_parts": len(parts), "extension_shots_per_basis": ext_shots,
                "extension_failures_x": ext_x, "extension_failures_z": ext_z, "target_total": target,
                "pooled_shots_per_basis": ps, "pooled_failures_x": px, "pooled_failures_z": pz,
                "pooled_plpr": plpr, "pooled_plpr_ci_low": lo, "pooled_plpr_ci_high": hi,
                "ge30_each_basis": "YES" if min(px, pz) >= TARGET_FAILURES else "NO",
                "extension_status": ("not_needed" if not parts and min(fx, fz) >= TARGET_FAILURES else
                                     "target_already_met" if not parts else "ok"),
            })
        complete = not problems
        for r in rows:
            if r["extension_parts"] and any(p.startswith(f"T5_{name}_r{r['final_rank']}_{r['code_hash8']}_") for p in problems):
                r["extension_status"] = "incomplete"
        order = sorted(rows, key=lambda r: (r["pooled_plpr"], r["pooled_plpr_ci_high"]))
        for rank, r in enumerate(order, start=1):
            r["pooled_rank"] = rank
            r["champion"] = "YES" if rank == 1 and (complete or allow_incomplete) else ""
            r["run_status"] = "complete" if complete else "incomplete"
        if problems:
            print(f"{name}: extension incomplete -- " + "; ".join(problems), file=sys.stderr)
        out.extend(sorted(rows, key=lambda r: r["final_rank"]))
    return out


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "select"):
        p = sub.add_parser(name)
        p.add_argument("--runs-root", action="append", required=True, help="directory holding the run directories (repeatable)")
        p.add_argument("--run", nargs="+", required=True, help="run directory names")
        p.add_argument("--prior-merged", default=None, help="fixed_code_measure --merge table of earlier extensions of these runs")
        p.add_argument("--out", required=True)
    plan_p = sub.choices["plan"]
    plan_p.add_argument("--caps", default=str(DEFAULT_CAPS))
    plan_p.add_argument("--part-cap-memory", type=int, default=DEFAULT_PART_CAP["memory"])
    plan_p.add_argument("--part-cap-cnot", type=int, default=DEFAULT_PART_CAP["cnot_dense"])
    plan_p.add_argument("--no-spec-check", action="store_true", help="skip recompiling the queued specs (no package needed)")
    sel_p = sub.choices["select"]
    sel_p.add_argument("--plan", required=True)
    sel_p.add_argument("--out-root", action="append", required=True, help="fixed_code_measure output root (repeatable)")
    sel_p.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)
    prior = read_prior(args.prior_merged)
    if args.cmd == "plan":
        plan, report = build_plan(args.runs_root, args.run, load_caps(args.caps), prior=prior,
                                  part_cap={"memory": args.part_cap_memory, "cnot_dense": args.part_cap_cnot},
                                  check_specs=not args.no_spec_check)
        Path(args.out).write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("\n".join(report))
        print(f"{len(plan['items'])} items -> {args.out}")
        return 0
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("block") != "top5_extension":
        raise SystemExit(f"{args.plan}: not a top5_extension plan")
    rows = select(args.runs_root, args.run, plan, args.out_root, prior=prior, allow_incomplete=args.allow_incomplete)
    if not rows:
        raise SystemExit("no rows")
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["champion"]:
            print(f"{r['run']}: champion {r['code_hash8']} pooled {r['pooled_plpr']:.3g} "
                  f"[{r['pooled_plpr_ci_low']:.3g}, {r['pooled_plpr_ci_high']:.3g}] {r['pooled_failures_x']}/{r['pooled_failures_z']} "
                  f"@ {r['pooled_shots_per_basis']}")
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
