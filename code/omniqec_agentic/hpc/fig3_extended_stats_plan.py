#!/usr/bin/env python3
"""Fig.3 extended-statistics plan: the Fig.2 supplement rule applied to the Phase B gate-task finals.

Fig.2 rule (docs/fig2_protocol_number_sources_20261015.md section 6, user decisions 2026-10-14/15):
every chosen champion is known to >= 30 failures per basis; the shots that get there are set
from the formal 25k counts (expected 45 failures in the weakest basis, rounded up to 50k, 0
failures -> the cap), then UNIFIED PER POINT (every arm of a point gets the point's maximum);
the formal 25k sample and the supplement are pooled afterwards. Champions are not reselected.
References (the resident baselines BB72 at N = 144, BB144 at N = 288) get a fixed amount and
only the supplement is reported, as in Fig.2.

The plan has the Fig.2 plan schema, so ``hpc/fig2_extended_stats.py`` measures and merges it
unchanged; every item carries ``logicals`` (= 2k, the two-block circuit's logical qubits) so
the merge pools in the task convention (root over 2k x T).

Stdlib only, Python 3.6 compatible: it runs on the NSCC login node where the runs live.

    python3 fig3_extended_stats_plan.py --runs-root /scratch/.../fig3_formal --points 60 98 \
        --out fig3_extended_stats_plan.json [--run-tag f3cnot_v2] [--allow-partial]
"""
import argparse
import json
import math
import os
import sys

ARMS = (("claudesub", "Claude"), ("gptsub", "GPT"), ("deepseek", "DeepSeek"))
REFERENCE = {144: ("bb", 72, 12), 288: ("bb", 144, 12)}  # the resident baselines (family, n, k)


def _rankings(run_dir):
    path = os.path.join(run_dir, "final_slow", "T12", "rankings.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    rows = data.get("rankings") if isinstance(data, dict) else data
    return [r for r in rows or [] if isinstance(r, dict)]


def _candidate_spec(row):
    path = row.get("candidate_path")
    if path and os.path.isfile(path):
        with open(path) as fh:
            cand = json.load(fh)
        spec = cand.get("spec")
        if spec:
            return spec, "final candidate file", cand.get("program_id") or row.get("program_id")
    return row.get("spec"), "rankings row", row.get("program_id")


def _ci_high(row):
    ci = row.get("per_logical_per_round_ci") or [None, None]
    try:
        return float(ci[1])
    except Exception:
        return float("inf")


def _champion(rows):
    ok = [r for r in rows if r.get("status") == "ok" and not r.get("is_baseline")
          and r.get("per_logical_per_round") is not None]
    if not ok:
        return None
    # Fig.2 rule (provenance/build_tables.py): lowest plpr, ties -> smaller interval upper bound
    return min(ok, key=lambda r: (float(r["per_logical_per_round"]), _ci_high(r)))


def _reference_row(rows, point):
    want = REFERENCE.get(point)
    if not want:
        return None
    fam, n, k = want
    for r in rows:
        if r.get("is_baseline"):
            return r
    for r in rows:  # the baseline was skipped as a duplicate finalist: the finalist is the same code
        if r.get("family") == fam and int(r.get("n") or 0) == n and int(r.get("k") or 0) == k and r.get("status") == "ok":
            return r
    return None


def _shots_for(f_min, expected, round_to, cap, formal_shots):
    if f_min <= 0:
        return cap
    need = expected / (float(f_min) / float(formal_shots))
    return int(min(cap, math.ceil(need / round_to) * round_to))


def build(args):
    plan = {
        "schema": "fig2_extended_stats_plan",
        "version": args.version,
        "created": args.created,
        "target": "at least %d failures per basis (Fig.2 rule), shots unified per point" % args.target,
        "run_tag": args.run_tag,
        "seed_namespace": args.namespace,
        "task": "cnot_dense",
        "rule": {"expected_failures": args.expected, "round_to": args.round_to, "cap": args.cap,
                 "formal_shots_per_basis": args.formal_shots, "reference_shots": dict(args.reference_shots)},
        "items": [],
    }
    report = []
    for point in args.points:
        arms = {}
        for tag, label in ARMS:
            name = "fig2_%s_b%d_%s" % (tag, point, args.run_tag)
            run_dir = next((os.path.join(root, name) for root in args.runs_root
                            if os.path.isdir(os.path.join(root, name))), None)
            rows = _rankings(run_dir) if run_dir else None
            if not rows:
                arms[label] = None
                continue
            arms[label] = (name, rows, _champion(rows))
        missing = [label for label, v in arms.items() if v is None or v[2] is None]
        if missing and not args.allow_partial:
            report.append("b%d: skipped, finals missing for %s" % (point, ", ".join(missing)))
            continue
        per_arm = {}
        for label, v in arms.items():
            if v is None or v[2] is None:
                continue
            name, rows, champ = v
            fx, fz = int(champ.get("failures_x") or 0), int(champ.get("failures_z") or 0)
            f_min = min(fx, fz)
            shots = 0 if f_min >= args.target else _shots_for(f_min, args.expected, args.round_to, args.cap, args.formal_shots)
            per_arm[label] = (name, champ, fx, fz, shots)
        point_shots = max(v[4] for v in per_arm.values()) if per_arm else 0
        for label, (name, champ, fx, fz, own) in sorted(per_arm.items()):
            n, k, d, w = champ.get("n"), champ.get("k"), champ.get("d_search"), champ.get("max_check_weight")
            line = "b%d %-8s [[%s,%s,%s]] w%s %s plpr=%.3g fails=%d/%d -> own %dk, point %dk" % (
                point, label, n, k, d, w, (champ.get("code_hash") or "")[:8], float(champ["per_logical_per_round"]),
                fx, fz, own // 1000, point_shots // 1000)
            report.append(line)
            if point_shots <= 0:
                continue
            spec, spec_source, program_id = _candidate_spec(champ)
            plan["items"].append({
                "key": "b%d_%s_%s" % (point, name.split("_")[1], (champ.get("code_hash") or "")[:8]),
                "budget": point, "arm": label, "source_run": name, "role": "champion",
                "program_id": program_id, "code_hash": champ.get("code_hash"), "family": champ.get("family"),
                "n": n, "k": k, "d_search": d, "max_check_weight": w, "logicals": 2 * int(k),
                "spec": spec, "spec_source": spec_source,
                "formal_failures_x": fx, "formal_failures_z": fz, "formal_shots_per_basis": args.formal_shots,
                "formal_plpr": champ.get("per_logical_per_round"), "formal_rounds": int(champ.get("rounds") or 12),
                "shots_per_basis": point_shots,
                "expected_failures_per_basis": round(point_shots * min(fx, fz) / float(args.formal_shots), 1),
                "own_rule_shots": own,
            })
        ref_shots = dict(args.reference_shots).get(point)
        if ref_shots and per_arm:
            # the reference measured in the gate task, from the first arm that has it
            for label, (name, champ, fx, fz, own) in sorted(per_arm.items()):
                row = _reference_row(arms[label][1], point)
                if row is None:
                    continue
                spec, spec_source, program_id = _candidate_spec(row)
                fam, n, k = REFERENCE[point]
                plan["items"].append({
                    "key": "b%d_ref_%s" % (point, (row.get("code_hash") or "")[:8]),
                    "budget": point, "arm": "reference", "source_run": name, "role": "reference",
                    "program_id": program_id or "baseline_bb_%d_%d_%s" % (n, k, row.get("d_search")),
                    "code_hash": row.get("code_hash"), "family": fam, "n": n, "k": k,
                    "d_search": row.get("d_search"), "max_check_weight": row.get("max_check_weight"),
                    "logicals": 2 * k, "spec": spec, "spec_source": spec_source,
                    "formal_failures_x": None, "formal_failures_z": None, "formal_shots_per_basis": args.formal_shots,
                    "formal_plpr": row.get("per_logical_per_round"), "formal_rounds": int(row.get("rounds") or 12),
                    "shots_per_basis": int(ref_shots), "expected_failures_per_basis": None,
                    "note": "reference: only the supplement is reported (Fig.2 convention); formal gate-task value %s" % row.get("per_logical_per_round"),
                })
                report.append("b%d reference [[%d,%d,%s]] %s -> %dk" % (point, n, k, row.get("d_search"), (row.get("code_hash") or "")[:8], int(ref_shots) // 1000))
                break
    return plan, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", action="append", required=True)
    ap.add_argument("--points", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-tag", default="f3cnot_v2", help="run directory suffix (formal run: f3cnot_v2; Phase B pilot: f3cnot_v1)")
    ap.add_argument("--target", type=int, default=30, help="failures per basis every champion must reach")
    ap.add_argument("--expected", type=float, default=45.0, help="expected failures in the weakest basis the shots aim at")
    ap.add_argument("--round-to", type=int, default=50000)
    ap.add_argument("--cap", type=int, default=500000, help="shots per basis for a 0-failure code (a later round tops up)")
    ap.add_argument("--formal-shots", type=int, default=25000)
    ap.add_argument("--reference-shots", type=lambda s: (int(s.split("=")[0]), int(s.split("=")[1])),
                    action="append", default=[(144, 50000), (288, 100000)], help="N=shots (default 144=50000 288=100000)")
    ap.add_argument("--namespace", default="extended_stats_v1")
    ap.add_argument("--version", default="fig3_extended_stats_v1")
    ap.add_argument("--created", default="2026-10-15")
    ap.add_argument("--allow-partial", action="store_true", help="emit items for a point whose arms are not all final (per-point unification is then provisional)")
    args = ap.parse_args(argv)
    plan, report = build(args)
    for line in report:
        print(line)
    with open(args.out, "w") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print("%d items -> %s" % (len(plan["items"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
