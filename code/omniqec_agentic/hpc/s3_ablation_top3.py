"""S3 ablation summary (2026-10-24): one row per (task, budget, group) with the bar = Top-3 mean of the final stage.

Rule (user decision 2026-10-23): every point is ONE search; its bar is the arithmetic mean of the three lowest
final-stage LER among the eligible candidates (status ok, not a baseline/reference row), ordered by
(plpr, ci_high); the formal 25k/basis final values only, no supplement pooling. Full = the original runs
(Fig.2 v3.7 Claude b98/180/260, Fig.3 formal_v2 Claude b98/180/260). seed_* programs are not treated
specially (same rule as Full); the table flags whether one sits in the Top-3.

    python -m omniqec_agentic.hpc.s3_ablation_top3 --run full:memory:98:<dir> ... --out table.csv
A run dir needs final_slow/final_ranking.json (the paper's top-5 cumulative final stage, Table VII) or, for runs
made before 2026-11-10, final_slow/T12/rankings.json; plus protocol_snapshot.json for the task/T check.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def final_rows(run: Path):
    paper_final = run / "final_slow" / "final_ranking.json"
    if paper_final.exists():
        rows = json.loads(paper_final.read_text())["rows"]
    else:
        j = json.loads((run / "final_slow" / "T12" / "rankings.json").read_text())
        rows = j["rankings"] if isinstance(j, dict) else j
    out = []
    for r in rows:
        plpr = r.get("per_logical_per_round")
        ci = r.get("per_logical_per_round_ci") or [None, None]
        out.append({
            "program_id": r.get("program_id"), "code_hash": r.get("code_hash"), "family": r.get("family"),
            "n": r.get("n"), "k": r.get("k"), "d_search": r.get("d_search"), "w": r.get("max_check_weight"),
            "plpr": plpr, "ci_low": ci[0], "ci_high": ci[1], "failures_x": r.get("failures_x"),
            "failures_z": r.get("failures_z"), "shots_x": r.get("shots_x"), "shots_z": r.get("shots_z"),
            "status": r.get("status"), "is_baseline": bool(r.get("is_baseline")), "rounds": r.get("rounds"),
            "channels": ";".join(r.get("selection_channels") or []),
        })
    return out


def summarize(label: str, run: Path):
    task, group, budget = label.split(":")[0:3] if False else (None, None, None)
    rows = final_rows(run)
    elig = [r for r in rows if r["status"] == "ok" and not r["is_baseline"] and r["plpr"] is not None]
    elig.sort(key=lambda r: (r["plpr"], r["ci_high"] if r["ci_high"] is not None else float("inf")))
    top = elig[:3]
    snap = {}
    sp = run / "protocol_snapshot.json"
    if sp.exists():
        snap = json.loads(sp.read_text())
    rec = {
        "run": run.name, "final_rows": len(rows), "eligible": len(elig), "top3_n": len(top),
        "top3_mean_plpr": statistics.mean(r["plpr"] for r in top) if top else None,
        "top3_has_seed": any(str(r["program_id"]).startswith("seed_") for r in top),
        "champion_plpr": top[0]["plpr"] if top else None,
        "champion_ci_high": top[0]["ci_high"] if top else None,
        "ablation": snap.get("ablation") or "full", "task": snap.get("task") or snap.get("verification", {}).get("task"),
        "rounds": top[0]["rounds"] if top else None,
    }
    for i, r in enumerate(top, 1):
        rec[f"top{i}"] = f"{r['program_id']} {r['family']} [[{r['n']},{r['k']},{r['d_search']}]] w{r['w']} plpr={r['plpr']:.4g} ci_high={r['ci_high']:.3g} F={r['failures_x']}/{r['failures_z']}"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="group:task:budget:/path/to/run (group = full|no_slow|no_feedback)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = []
    for item in a.run:
        group, task, budget, path = item.split(":", 3)
        rec = {"group": group, "task_label": task, "budget": int(budget)}
        rec.update(summarize(item, Path(path)))
        if group != "full" and rec["ablation"] != group:
            raise SystemExit(f"{path}: snapshot ablation={rec['ablation']} but label says {group}")
        recs.append(rec)
    recs.sort(key=lambda r: (r["task_label"], r["budget"], ["full", "no_slow", "no_feedback"].index(r["group"]) if r["group"] in ("full", "no_slow", "no_feedback") else 9))
    keys = list(dict.fromkeys(k for r in recs for k in r))
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    for r in recs:
        m = r["top3_mean_plpr"]
        print(f"{r['task_label']:7s} b{r['budget']:<4d} {r['group']:12s} eligible={r['eligible']} top3_mean={m if m is None else f'{m:.3g}'} champion={r['champion_plpr'] if r['champion_plpr'] is None else f'{r['champion_plpr']:.3g}'} seed_in_top3={r['top3_has_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
