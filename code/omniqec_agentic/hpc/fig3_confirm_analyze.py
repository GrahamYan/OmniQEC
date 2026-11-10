#!/usr/bin/env python3
"""Fig.3 confirmation analysis (2026-10-13): memory champion vs gate champion(s) at one point.

Reads the shard files written by hpc/cnot_pilot.py for the confirmation pool
(hpc/fig3_confirm_pool.py) and reports, in Fig.2's per-logical-qubit-per-round
convention (Ruiz et al. 2025; user decision 10-12):

  block  = 1 - (1 - p_anyX)(1 - p_anyZ)        any observable wrong = the shot failed
  p_log  = 1 - (1 - block)^(1 / (n_obs * T))   n_obs = logical qubits read out (2k for two blocks)
  eps    = (p_log(gate) - p_log(idle)) * T / n_gates   added error per CNOT per logical qubit

Statistics: shard bootstrap (default 2000 draws; every shard carries the same number of
shots, so shards are exchangeable units). Pairwise order probabilities between the
memory champion M and every gate-role code G:

  P_gate = P[p_gate(G) < p_gate(M)]     P_idle = P[p_idle(M) < p_idle(G)]

Pre-registered verdict (docs/fig3_confirm_protocol_20261013.md):
  different      P_gate >= resolve and P_idle >= resolve   (gate-optimal is another code)
  same           P_gate <= 1 - resolve                      (M also wins the gate task)
  extend         neither, and shots < --cap                 (run the next stage)
  tie_at_cap     neither at the cap: report the gap with its CI as a crossover point

usage: fig3_confirm_analyze.py --pool POOL.json --run DIR [--gate-exp cnot] [--extra r6=DIR r2=DIR]
                               --out-md FILE --out-json FILE [--resolve 0.99] [--cap 200000] [--nboot 2000]
DIR holds the idle and gate experiments of the same job (EXPERIMENTS=idle+cnot); --extra adds
other gate spacings measured in their own run dirs (same pool).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

BASES = ("X", "Z")


def load_meta(run: Path) -> dict:
    return json.loads((run / "meta.json").read_text())


def rounds_of(meta: dict) -> int:
    return int(meta["rounds_before"]) + int(meta["rounds_after"])


def gates_of(meta: dict) -> int:
    if meta.get("cnot_every_round"):
        return rounds_of(meta) - 1
    return len(meta.get("cnot_rounds") or []) or 1


def load_shards(run: Path, exp: str, labels: set[str]):
    """label -> {basis: [(shots, any_fail), ...]} and label -> n_obs; CPU seconds and shots per label."""
    data: dict[str, dict[str, list]] = collections.defaultdict(lambda: {b: [] for b in BASES})
    nobs: dict[str, int] = {}
    cost: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    path = run / "shards.jsonl"
    if not path.exists():
        return data, nobs, cost
    seen: set[tuple] = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut by a killed job
        if row.get("exp") != exp or row.get("label") not in labels:
            continue
        key = (row["label"], row["basis"], row["shard"], row["shots"])
        if key in seen:
            continue
        seen.add(key)
        data[row["label"]][row["basis"]].append((int(row["shots"]), int(row["any_fail"])))
        nobs[row["label"]] = len(row.get("obs_fail") or [])
        cost[row["label"]][0] += float(row.get("runtime_s") or 0.0)
        cost[row["label"]][1] += int(row["shots"])
    return data, nobs, cost


def plog(sh: dict[str, list], nobs: int, rounds: int) -> float:
    block = 1.0
    for basis in BASES:
        shots = sum(n for n, _ in sh[basis])
        fails = sum(f for _, f in sh[basis])
        block *= 1.0 - fails / shots
    return 1.0 - block ** (1.0 / (nobs * rounds))


def boot_plog(sh: dict[str, list], nobs: int, rounds: int, nboot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap draws of p_log: resample shards within each basis."""
    out = np.empty(nboot)
    arrays = {b: np.asarray(sh[b], dtype=float) for b in BASES}
    for i in range(nboot):
        block = 1.0
        for basis in BASES:
            arr = arrays[basis]
            idx = rng.integers(0, len(arr), len(arr))
            pick = arr[idx]
            block *= 1.0 - pick[:, 1].sum() / pick[:, 0].sum()
        out[i] = 1.0 - block ** (1.0 / (nobs * rounds))
    return out


def summarize_code(sh: dict[str, list], nobs: int, rounds: int, nboot: int, rng) -> dict:
    draws = boot_plog(sh, nobs, rounds, nboot, rng)
    shots = {b: sum(n for n, _ in sh[b]) for b in BASES}
    fails = {b: sum(f for _, f in sh[b]) for b in BASES}
    return {
        "p": plog(sh, nobs, rounds),
        "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "draws": draws,
        "shots_per_basis": min(shots.values()),
        "failures": fails,
        "shards": min(len(sh[b]) for b in BASES),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--run", required=True, help="run dir with the idle and the gate experiment")
    ap.add_argument("--idle-exp", default="idle")
    ap.add_argument("--gate-exp", default="cnot")
    ap.add_argument("--extra", nargs="*", default=[], help="name=DIR of extra gate-spacing runs (gate exp)")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--resolve", type=float, default=0.99, help="order probability that settles a pair")
    ap.add_argument("--cap", type=int, default=200_000, help="shots per basis at which an unsettled pair is a tie")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--min-shards", type=int, default=20, help="codes with fewer shards per basis are skipped")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    pool = json.loads(Path(args.pool).read_text())
    cands = {c["label"]: c for c in pool}
    run = Path(args.run)
    meta = load_meta(run)
    rounds = rounds_of(meta)
    n_gates = gates_of(meta)
    idle, nobs_idle, _ = load_shards(run, args.idle_exp, set(cands))
    gate, nobs_gate, cost = load_shards(run, args.gate_exp, set(cands))
    extras = {}
    for item in args.extra:
        name, path = item.split("=", 1)
        extra_run = Path(path)
        extra_meta = load_meta(extra_run)
        sh, nb, _ = load_shards(extra_run, args.gate_exp, set(cands))
        extras[name] = (sh, nb, rounds_of(extra_meta), gates_of(extra_meta))

    results: dict[str, dict] = {}
    for label, cand in cands.items():
        ok = all(len(idle[label][b]) >= args.min_shards and len(gate[label][b]) >= args.min_shards for b in BASES)
        if not ok:
            results[label] = {"status": "insufficient", "roles": cand.get("roles", [])}
            continue
        r_idle = summarize_code(idle[label], nobs_idle[label], rounds, args.nboot, rng)
        r_gate = summarize_code(gate[label], nobs_gate[label], rounds, args.nboot, rng)
        eps_draws = (r_gate["draws"] - r_idle["draws"]) * rounds / n_gates
        res = {
            "status": "ok", "roles": cand.get("roles", []), "w": cand.get("w"), "k": cand["k"], "n": cand["n"],
            "d_search": cand.get("d_search"), "fig2_plpr": cand.get("fig2_plpr"), "task_plpr": cand.get("task_plpr"),
            "idle": {k: v for k, v in r_idle.items() if k != "draws"},
            "gate": {k: v for k, v in r_gate.items() if k != "draws"},
            "eps_per_cnot": (r_gate["p"] - r_idle["p"]) * rounds / n_gates,
            "eps_ci": [float(np.percentile(eps_draws, 2.5)), float(np.percentile(eps_draws, 97.5))],
            "cpu_s_per_shot": (cost[label][0] / cost[label][1]) if cost[label][1] else None,
            "_idle_draws": r_idle["draws"], "_gate_draws": r_gate["draws"],
        }
        for name, (sh, nb, t_extra, g_extra) in extras.items():
            if all(len(sh[label][b]) >= args.min_shards for b in BASES):
                r_x = summarize_code(sh[label], nb[label], t_extra, args.nboot, rng)
                res[f"gate_{name}"] = {k: v for k, v in r_x.items() if k != "draws"}
                res[f"gate_{name}"]["n_gates"] = g_extra
                res[f"gate_{name}"]["eps_per_cnot"] = (r_x["p"] - r_idle["p"]) * t_extra / g_extra
        results[label] = res

    measured = [l for l, r in results.items() if r["status"] == "ok"]
    memory = [l for l in measured if "memory_champion" in results[l]["roles"]]
    gate_roles = [l for l in measured if any(role.startswith("gate") for role in results[l]["roles"])]
    pairs = []
    for m in memory[:1]:
        for g in gate_roles:
            if g == m:
                pairs.append({"memory": m, "gate": g, "verdict": "same_code",
                              "note": "the gate champion is the memory champion itself"})
                continue
            rm, rg = results[m], results[g]
            p_gate = float(np.mean(rg["_gate_draws"] < rm["_gate_draws"]))
            p_idle = float(np.mean(rm["_idle_draws"] < rg["_idle_draws"]))
            gap_draws = (rm["_gate_draws"] - rg["_gate_draws"]) / rg["_gate_draws"]
            shots = min(rm["gate"]["shots_per_basis"], rg["gate"]["shots_per_basis"])
            if p_gate >= args.resolve and p_idle >= args.resolve:
                verdict = "different"
            elif p_gate <= 1.0 - args.resolve:
                verdict = "same"
            elif shots >= args.cap:
                verdict = "tie_at_cap"
            else:
                verdict = "extend"
            pairs.append({
                "memory": m, "gate": g, "P_gate_G_better": p_gate, "P_idle_M_better": p_idle,
                "gate_gap_M_over_G": float(np.mean(gap_draws)),
                "gate_gap_ci": [float(np.percentile(gap_draws, 2.5)), float(np.percentile(gap_draws, 97.5))],
                "shots_per_basis": shots, "verdict": verdict,
            })

    # the Phase B gate champion against reference gate champions (e.g. the pilot's): same
    # circuit, so only the gate-circuit order matters
    champion = next((l for l in measured if "gate_champion" in results[l]["roles"]), None)
    for ref in [l for l in measured if any("pilot_gate_champion" in r or r == "included" for r in results[l]["roles"])]:
        if champion is None or ref == champion:
            continue
        rc, rg = results[ref], results[champion]
        p_new = float(np.mean(rg["_gate_draws"] < rc["_gate_draws"]))
        gap_draws = (rc["_gate_draws"] - rg["_gate_draws"]) / rg["_gate_draws"]
        shots = min(rc["gate"]["shots_per_basis"], rg["gate"]["shots_per_basis"])
        if p_new >= args.resolve:
            verdict = "new_better"
        elif p_new <= 1.0 - args.resolve:
            verdict = "reference_better"
        elif shots >= args.cap:
            verdict = "tie_at_cap"
        else:
            verdict = "extend"
        pairs.append({
            "memory": ref, "gate": champion, "kind": "gate_vs_reference_gate",
            "P_gate_G_better": p_new, "P_idle_M_better": float(np.mean(rc["_idle_draws"] < rg["_idle_draws"])),
            "gate_gap_M_over_G": float(np.mean(gap_draws)),
            "gate_gap_ci": [float(np.percentile(gap_draws, 2.5)), float(np.percentile(gap_draws, 97.5))],
            "shots_per_basis": shots, "verdict": verdict,
        })

    # projected cost of the next stage (doubling the shots) from the measured CPU-s/shot
    next_hours = 0.0
    for l in measured:
        c = results[l]["cpu_s_per_shot"]
        if c:
            next_hours += c * results[l]["gate"]["shots_per_basis"] * 2 * 2 / 3600.0 / 128.0  # both bases, idle + gate
    out = {
        "point_rounds": rounds, "n_gates": n_gates, "resolve": args.resolve, "cap": args.cap,
        "codes": {l: {k: v for k, v in r.items() if not k.startswith("_")} for l, r in results.items()},
        "pairs": pairs, "projected_next_stage_node_hours": next_hours,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=float))

    lines = [f"# Fig.3 confirmation: T={rounds} rounds, {n_gates} transversal CNOT(s); per logical qubit per round "
             f"(Fig.2 / Ruiz convention); shard bootstrap x{args.nboot}", ""]
    lines.append("| role | code | [[n,k,d]] w | idle p_log [95%] | gate p_log [95%] | gate/idle | eps per CNOT [95%] "
                 "| shots/basis | failures X/Z (gate) | CPU-s/shot |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for l in sorted(measured, key=lambda x: results[x]["gate"]["p"]):
        r = results[l]
        cpu = f"{r['cpu_s_per_shot']:.2f}" if r["cpu_s_per_shot"] else "-"
        lines.append(
            f"| {', '.join(r['roles'])} | {l} | [[{r['n']},{r['k']},{r['d_search']}]] w{r['w']} "
            f"| {r['idle']['p']:.3e} [{r['idle']['ci'][0]:.2e},{r['idle']['ci'][1]:.2e}] "
            f"| {r['gate']['p']:.3e} [{r['gate']['ci'][0]:.2e},{r['gate']['ci'][1]:.2e}] "
            f"| {r['gate']['p'] / r['idle']['p']:.2f} | {r['eps_per_cnot']:+.2e} [{r['eps_ci'][0]:+.1e},{r['eps_ci'][1]:+.1e}] "
            f"| {r['gate']['shots_per_basis']} | {r['gate']['failures']['X']}/{r['gate']['failures']['Z']} "
            f"| {cpu} |")
    skipped = [l for l, r in results.items() if r["status"] != "ok"]
    if skipped:
        lines.append(f"\nskipped (insufficient shards): {', '.join(skipped)}")
    if extras:
        lines.append("\n| code | " + " | ".join(f"{n}: p_log (gates, eps/CNOT)" for n in extras) + " |")
        lines.append("|---|" + "---|" * len(extras))
        for l in measured:
            cells = []
            for n in extras:
                x = results[l].get(f"gate_{n}")
                cells.append(f"{x['p']:.3e} ({x['n_gates']}, {x['eps_per_cnot']:+.1e})" if x else "-")
            lines.append(f"| {l} | " + " | ".join(cells) + " |")
    lines.append("\n## Pre-registered pair tests (memory champion M vs gate-role code G)\n")
    lines.append("| M | G | P[G better in gate] | P[M better in idle] | gate gap M/G - 1 [95%] | shots/basis | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in pairs:
        if p["verdict"] == "same_code":
            lines.append(f"| {p['memory']} | {p['gate']} | - | - | - | - | same_code |")
            continue
        left = p["memory"] + (" (reference gate champion)" if p.get("kind") == "gate_vs_reference_gate" else "")
        lines.append(f"| {left} | {p['gate']} | {p['P_gate_G_better']:.4f} | {p['P_idle_M_better']:.4f} "
                     f"| {p['gate_gap_M_over_G']:+.1%} [{p['gate_gap_ci'][0]:+.1%},{p['gate_gap_ci'][1]:+.1%}] "
                     f"| {p['shots_per_basis']} | **{p['verdict']}** |")
    lines.append(f"\nverdicts: different = gate-optimal is another code (both orders >= {args.resolve}); same = M also wins "
                 f"the gate task; extend = run the next stage; tie_at_cap = unresolved at {args.cap} shots/basis (crossover).")
    lines.append(f"projected node-hours for the next stage (shots x2, idle + gate, both bases): {next_hours:.1f}")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
