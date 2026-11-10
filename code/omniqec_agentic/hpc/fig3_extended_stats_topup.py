#!/usr/bin/env python3
"""Fig.3 extended statistics: the top-up round after a supplement round left a champion under target.

The rule is the one ``fig3_extended_stats_plan.py`` applies to the formal 25k counts, now applied to the
pooled counts (formal + every supplement round measured so far): a champion whose weakest basis has fewer
than 30 failures gets the shots that are expected to bring that basis to 45 failures, rounded up to 50k
(0 failures -> the cap); the top-up is UNIFIED PER POINT, so every champion of that point gets the largest
top-up of the point. Each round has its own seed namespace (``<plan namespace>_r<round>``), otherwise it would
re-measure the same shots; the new items carry ``base_key`` so that ``fig2_extended_stats.py --merge`` sums
every round of a code. References (BB72, BB144) keep their fixed amount and are never topped up.

The output plan holds the input items plus the new ones: measure the new keys with
``nscc_fig3_extended_stats.pbs`` (PLAN = the output) and merge with the output plan.

Stdlib only, Python 3.6 compatible (NSCC login node).

    python3 fig3_extended_stats_topup.py --plan fig3_extended_stats_plan_b360.json \\
        --out-root /scratch/.../fig3_formal_ext --round 2 --date 2026-10-24 --out fig3_extended_stats_plan_b360_r2.json

First used 2026-10-24 for b360 (Claude Z 27 failures in 525k after round 1 -> +350k for all three arms).
"""
import argparse
import collections
import copy
import json
import math
import os
import sys


def _result(out_root, key):
    path = os.path.join(out_root, key, "result.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _topup(f_min, total_shots, expected, round_to, cap):
    if f_min <= 0:
        return cap
    need = expected * total_shots / float(f_min) - total_shots
    return int(min(cap, max(0, math.ceil(need / round_to) * round_to)))


def build(args):
    with open(args.plan) as fh:
        plan = json.load(fh)
    namespace = "%s_r%d" % (plan["seed_namespace"], args.round)
    if any(it.get("seed_namespace") == namespace for it in plan["items"]):
        raise SystemExit("plan already has round %d items (namespace %s)" % (args.round, namespace))
    groups = collections.OrderedDict()
    for it in plan["items"]:
        groups.setdefault(it.get("base_key") or it["key"], []).append(it)
    report, per_point = [], collections.OrderedDict()
    for base_key, items in groups.items():
        first = items[0]
        if first.get("role") != "champion":
            continue
        fx, fz = int(first["formal_failures_x"]), int(first["formal_failures_z"])
        shots = int(first["formal_shots_per_basis"])
        for it in items:
            res = _result(args.out_root, it["key"])
            if res is None or res.get("queue_status") != "ok" or (res.get("row") or {}).get("status") != "ok":
                raise SystemExit("%s: no finished result under %s -- the round is not complete" % (it["key"], args.out_root))
            row = res["row"]
            if int(row["shots_x"]) != int(row["shots_z"]):
                raise SystemExit("%s: unequal shots per basis" % it["key"])
            fx, fz, shots = fx + int(row["failures_x"]), fz + int(row["failures_z"]), shots + int(row["shots_x"])
        f_min = min(fx, fz)
        own = 0 if f_min >= args.target else _topup(f_min, shots, args.expected, args.round_to, args.cap)
        per_point.setdefault(int(first["budget"]), []).append((base_key, first, fx, fz, shots, own))
        report.append("b%s %-8s %s pooled X/Z %d/%d of %d per basis -> own top-up %dk" % (
            first["budget"], first["arm"], base_key, fx, fz, shots, own // 1000))
    for point, codes in per_point.items():
        top = max(c[5] for c in codes)
        report.append("b%d: point top-up %dk per basis" % (point, top // 1000))
        if top <= 0:
            continue
        short = ["%s %s %d" % (c[1]["arm"], "X" if c[2] <= c[3] else "Z", min(c[2], c[3])) for c in codes if min(c[2], c[3]) < args.target]
        for base_key, first, fx, fz, shots, own in codes:
            item = copy.deepcopy(first)
            item.update({
                "key": "%s_r%d" % (base_key, args.round), "base_key": base_key, "seed_namespace": namespace,
                "shots_per_basis": top, "expected_failures_per_basis": None, "own_rule_shots": own,
                "note": "top-up round %d (%s): after the earlier rounds %s failures (< %d); per-point uniform top-up" % (
                    args.round, args.date, ", ".join(short), args.target),
            })
            plan["items"].append(item)
        rounds = plan.setdefault("rounds", {})
        line = "%s b%d: +%dk per basis for every champion (namespace %s): %s < %d failures" % (
            args.date, point, top // 1000, namespace, ", ".join(short), args.target)
        rounds[str(args.round)] = "; ".join(filter(None, [rounds.get(str(args.round)), line]))
    return plan, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="the plan of the rounds measured so far")
    ap.add_argument("--out-root", required=True, help="directory with one <key>/result.json per measured item")
    ap.add_argument("--round", type=int, required=True, help="number of the new round (2 for the first top-up)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--expected", type=float, default=45.0)
    ap.add_argument("--round-to", type=int, default=50000)
    ap.add_argument("--cap", type=int, default=500000)
    args = ap.parse_args(argv)
    plan, report = build(args)
    for line in report:
        print(line)
    if os.path.exists(args.out):
        raise SystemExit("refusing to overwrite %s" % args.out)
    with open(args.out, "w") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print("%d items -> %s" % (len(plan["items"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
