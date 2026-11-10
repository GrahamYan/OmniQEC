#!/usr/bin/env python3
"""Fig.3 confirmation pool for one budget point (2026-10-13).

The Phase B search ranks codes on 25k shots per basis with a 100-error racing stop
(CI half-width ~20%), so it cannot tell a 10-20% gap. Standard practice for a
search on a noisy objective is to re-measure the few elites independently with
high statistics; this script picks those elites and writes them in the pilot
pipeline's candidate-pool format (hpc/cnot_pilot.py, docs/fig3_cnot_pilot_candidates_*.json):

  memory_champion      the Fig.2 memory champion of the point (lowest final-stage plpr over
                       every Fig.2 source given)
  gate_champion        the Phase B gate-task champion (lowest task plpr over the three arms;
                       final-stage rankings when the run has them, else the search ledger)
  gate_arm_champion:*  each arm's own champion when it belongs to a (w, k) class not yet in
                       the pool (so every class that won an arm gets measured)
  --fig2-extra N / --phaseb-extra N add the next N finalists of a class not yet in the pool.

Sources (any mix; stdlib only and Python 3.6 compatible, so it runs on a login node):
  --phaseb-run DIR   a Phase B run dir (summary.csv + evidence_rows.jsonl [+ final_slow/T12/rankings.json])
  --fig2-run DIR     a Fig.2 run dir (same layout)
  --fig2-pool JSON   a pilot pool or a --fig2-only emission; entries whose source starts with
                     'fig2' or 'final' and carry fig2_plpr are Fig.2 finalists
  --fig2-only        emit only the Fig.2 entries found (to carry a point's Fig.2 finalists
                     from another account into the final merge)

Every entry keeps the fields cnot_pilot.py reads (label, spec, code_hash8, d_search, w, n, k,
family, source) plus fig2_plpr / fig2_plpr_ci (Fig.2 final value if known), task_plpr /
task_plpr_ci (Phase B value), roles and the measuring stage.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)
FAMILY_PREFIX = {"bb": "bb", "hgp": "hgp", "lp_abelian": "lp_ab", "lp_nonabelian": "lp_no"}
ARM_NAMES = {"claudesub": "Claude", "claude": "Claude", "gptsub": "GPT", "gpt": "GPT", "deepseek": "DeepSeek"}
MISSING = (None, "", "nan", "None")


def label_of(family, hash8):
    return "{}_{}".format(FAMILY_PREFIX.get(family, family), hash8)


def arm_of(run):
    parts = run.name.split("_")
    return ARM_NAMES.get(parts[1], parts[1]) if len(parts) > 1 else run.name


def _float(value):
    if value in MISSING:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ci(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = _float(value[0]), _float(value[1])
        return [lo, hi] if lo is not None and hi is not None else None
    if isinstance(value, str) and value.startswith("["):
        try:
            return _ci(json.loads(value))
        except json.JSONDecodeError:
            return None
    return None


def read_summary(run):
    """code_hash8 -> ledger row (the final-stage row when the run exported one)."""
    rows = {}
    path = run / "summary.csv"
    if not path.exists():
        return rows
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            hash8 = row.get("code_hash8")
            if not hash8:
                continue
            prev = rows.get(hash8)
            if prev is None or (row.get("turn") == "final" and prev.get("turn") != "final"):
                rows[hash8] = row
    return rows


def read_weights(run):
    """code_hash8 -> max check weight from evidence_rows.jsonl (v3.7.1 rows) or structural_ledger.jsonl
    (every structural row; the pre-v3.7.1 evidence rows of the Fig.2 runs carry no weight)."""
    weights = {}
    for name in ("evidence_rows.jsonl", "structural_ledger.jsonl"):
        path = run / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("code_hash") and row.get("max_check_weight") is not None:
                weights.setdefault(str(row["code_hash"])[:8], int(row["max_check_weight"]))
    return weights


def weight_from_spec(spec, hash8):
    """Rebuild the code and read its max check weight (needs the omniqec_agentic package on the path)."""
    try:
        from omniqec_agentic.construction.build import build_compiled_code  # optional, --fill-weights only
    except ModuleNotFoundError:  # run as a file: the repo root is two levels above omniqec_agentic/hpc/
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from omniqec_agentic.construction.build import build_compiled_code

    compiled = build_compiled_code(spec)
    if str(compiled.code_hash)[:8] != str(hash8)[:8]:
        raise ValueError("rebuilt hash {} != {}".format(str(compiled.code_hash)[:8], hash8))
    rows = [sum(int(x) for x in row) for row in list(compiled.hx) + list(compiled.hz)]
    return max(rows)


def read_final(run):
    """Final-stage rows (status ok) of final_slow/T12/rankings.json, best first."""
    path = run / "final_slow" / "T12" / "rankings.json"
    if not path.exists():
        return []
    blob = json.loads(path.read_text())
    rows = blob.get("rankings") if isinstance(blob, dict) else blob
    out = []
    for row in rows or []:
        if str(row.get("status")) != "ok" or _float(row.get("per_logical_per_round")) is None:
            continue
        out.append(row)
    out.sort(key=lambda r: float(r["per_logical_per_round"]))
    return out


def entries_from_run(run, kind):
    """Measured codes of one run, best first; kind = 'fig2' | 'phaseb'."""
    arm = arm_of(run)
    summary = read_summary(run)
    weights = read_weights(run)
    final = read_final(run)
    ranked = []
    if final:
        stage = "final"
        for row in final:
            ranked.append((float(row["per_logical_per_round"]), str(row["code_hash"])[:8],
                           _ci(row.get("per_logical_per_round_ci"))))
    else:
        stage = "search"
        for hash8, row in summary.items():
            value = _float(row.get("plpr"))
            if value is None:
                continue
            lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
            ranked.append((value, hash8, [lo, hi] if lo is not None and hi is not None else None))
        ranked.sort(key=lambda t: t[0])
    entries = []
    for value, hash8, ci in ranked:
        row = summary.get(hash8)
        if row is None or row.get("spec") in MISSING:
            continue  # cannot be rebuilt without its spec
        spec = row["spec"]
        if isinstance(spec, str):
            spec = json.loads(spec)
        d_search = _float(row.get("d_search_proxy"))
        entries.append({
            "label": label_of(row["family"], hash8),
            "spec": spec,
            "code_hash8": hash8,
            "d_search": int(d_search) if d_search is not None else None,
            "family": row["family"],
            "n": int(row["n_data"]),
            "k": int(row["k"]),
            "w": weights.get(hash8),
            "source": "{}_{}:{}".format(kind, stage, arm),
            "stage": stage,
            "arm": arm,
            "kind": kind,
            "value": value,
            "value_ci": ci,
        })
    return entries


def entries_from_pool(path):
    """Fig.2 finalists carried in a pilot pool / a --fig2-only emission."""
    entries = []
    for item in json.loads(path.read_text()):
        source = str(item.get("source", ""))
        if not (source.startswith("fig2") or source.startswith("final")):
            continue
        value = _float(item.get("fig2_plpr"))
        if value is None:
            continue
        arm = source.split(":", 1)[1] if ":" in source else "?"
        stage = "final" if "final" in source else "search"
        entries.append({
            "label": item["label"], "spec": item["spec"], "code_hash8": item["code_hash8"],
            "d_search": item.get("d_search"), "family": item.get("family"), "n": item.get("n"),
            "k": item.get("k"), "w": item.get("w"), "source": "fig2_{}:{}".format(stage, arm), "stage": stage,
            "arm": arm, "kind": "fig2", "value": value, "value_ci": _ci(item.get("fig2_plpr_ci")),
        })
    return entries


def cls(entry):
    return (entry.get("w"), entry.get("k"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", type=int, required=True, help="budget point N (for the report only)")
    ap.add_argument("--out", required=True, help="pool JSON to write")
    ap.add_argument("--phaseb-run", nargs="*", default=[], help="Phase B run dirs (the three arms)")
    ap.add_argument("--fig2-run", nargs="*", default=[], help="Fig.2 run dirs")
    ap.add_argument("--fig2-pool", nargs="*", default=[], help="pilot pools / --fig2-only emissions")
    ap.add_argument("--fig2-only", action="store_true", help="emit the Fig.2 finalists only")
    ap.add_argument("--fig2-extra", type=int, default=0, help="extra Fig.2 finalists of new classes")
    ap.add_argument("--phaseb-extra", type=int, default=0, help="extra Phase B finalists of new classes")
    ap.add_argument("--max-codes", type=int, default=6)
    ap.add_argument("--all-arm-champions", action="store_true",
                    help="measure every arm's champion even when its (w, k) class is already in the pool")
    ap.add_argument("--include", nargs="*", default=[],
                    help="HASH8[:ROLE] of codes to add from any source (e.g. the pilot's gate champion "
                         "8e8ada5d:pilot_gate_champion); default role 'included'")
    ap.add_argument("--fill-weights", action="store_true",
                    help="rebuild codes without a recorded check weight from their spec (needs omniqec_agentic importable)")
    args = ap.parse_args(argv)

    fig2 = []
    for run in args.fig2_run:
        fig2.extend(entries_from_run(Path(run), "fig2"))
    for pool in args.fig2_pool:
        fig2.extend(entries_from_pool(Path(pool)))
    if args.fill_weights:
        for entry in fig2:
            if entry.get("w") is None:
                entry["w"] = weight_from_spec(entry["spec"], entry["code_hash8"])
    # one row per code: keep the best (final-stage preferred) measurement
    best_fig2 = {}
    for entry in fig2:
        prev = best_fig2.get(entry["code_hash8"])
        if prev is None or (entry["stage"], -entry["value"]) > (prev["stage"], -prev["value"]):
            best_fig2[entry["code_hash8"]] = entry
    fig2 = sorted(best_fig2.values(), key=lambda e: e["value"])

    def public(entry, roles):
        fig2_match = best_fig2.get(entry["code_hash8"])
        out = {k: entry[k] for k in ("label", "spec", "code_hash8", "d_search", "family", "n", "k", "w", "source")}
        out["fig2_plpr"] = fig2_match["value"] if fig2_match else (entry["value"] if entry["kind"] == "fig2" else None)
        out["fig2_plpr_ci"] = fig2_match["value_ci"] if fig2_match else (entry["value_ci"] if entry["kind"] == "fig2" else None)
        out["task_plpr"] = entry["value"] if entry["kind"] == "phaseb" else None
        out["task_plpr_ci"] = entry["value_ci"] if entry["kind"] == "phaseb" else None
        out["stage"] = entry["stage"]
        out["roles"] = roles
        return out

    if args.fig2_only:
        pool = [public(e, ["fig2_finalist"]) for e in fig2]
        Path(args.out).write_text(json.dumps(pool, indent=1))
        print("b{}: {} Fig.2 finalists -> {}".format(args.point, len(pool), args.out))
        for e in pool[:8]:
            print("  {:.3e} {} [[{},{},{}]] w{} {}".format(e["fig2_plpr"], e["label"], e["n"], e["k"], e["d_search"], e["w"], e["source"]))
        return 0

    by_arm = {}
    for run in args.phaseb_run:
        entries = entries_from_run(Path(run), "phaseb")
        if entries:
            if args.fill_weights:
                for entry in entries:
                    if entry.get("w") is None:
                        entry["w"] = weight_from_spec(entry["spec"], entry["code_hash8"])
            by_arm[entries[0]["arm"]] = entries
    if not fig2 and not by_arm:
        print("nothing to pool: no Fig.2 or Phase B sources found", file=sys.stderr)
        return 2

    pool = []
    roles = {}

    def add(entry, role):
        roles.setdefault(entry["code_hash8"], []).append(role)
        if all(p["code_hash8"] != entry["code_hash8"] for p in pool):
            pool.append(entry)

    if fig2:
        add(fig2[0], "memory_champion")
    phaseb_all = sorted((e for ents in by_arm.values() for e in ents), key=lambda e: e["value"])
    if phaseb_all:
        add(phaseb_all[0], "gate_champion")
    for arm, entries in sorted(by_arm.items()):
        top = entries[0]
        if (args.all_arm_champions or cls(top) not in {cls(p) for p in pool}
                or top["code_hash8"] == phaseb_all[0]["code_hash8"]):
            add(top, "gate_arm_champion:{}".format(arm))
    for budget, source, tag in ((args.fig2_extra, fig2, "fig2_extra"), (args.phaseb_extra, phaseb_all, "phaseb_extra")):
        added = 0
        for entry in source:
            if added >= budget:
                break
            if cls(entry) in {cls(p) for p in pool}:
                continue
            add(entry, tag)
            added += 1
    for item in args.include:
        hash8, _, role = item.partition(":")
        role = role or "included"
        match = next((e for e in fig2 + phaseb_all if e["code_hash8"] == hash8), None)
        if match is None:
            print("--include {}: no source carries that code; skipped".format(hash8), file=sys.stderr)
            continue
        add(match, role)
    if len(pool) > args.max_codes:
        print("pool has {} codes > --max-codes {}; keeping the first {}".format(len(pool), args.max_codes, args.max_codes),
              file=sys.stderr)
        pool = pool[: args.max_codes]

    out = [public(e, roles[e["code_hash8"]]) for e in pool]
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("b{}: {} codes -> {}".format(args.point, len(out), args.out))
    print("| role | label | [[n,k,d]] w | Fig.2 plpr (final) | Phase B task plpr | source |")
    print("|---|---|---|---|---|---|")
    for e in out:
        f2 = "{:.3e}".format(e["fig2_plpr"]) if e["fig2_plpr"] is not None else "-"
        tp = "{:.3e}".format(e["task_plpr"]) if e["task_plpr"] is not None else "-"
        print("| {} | {} | [[{},{},{}]] w{} | {} | {} | {} |".format(
            ", ".join(e["roles"]), e["label"], e["n"], e["k"], e["d_search"], e["w"], f2, tp, e["source"]))
    missing_w = [e["label"] for e in out if e["w"] is None]
    if missing_w:
        print("WARNING: no check weight recorded for {} (evidence_rows.jsonl missing?)".format(missing_w), file=sys.stderr)
    for arm, entries in sorted(by_arm.items()):
        print("  {}: {} stage, {} measured codes, top-3 classes {}".format(
            arm, entries[0]["stage"], len(entries), ", ".join("w{}k{}".format(e["w"], e["k"]) for e in entries[:3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
