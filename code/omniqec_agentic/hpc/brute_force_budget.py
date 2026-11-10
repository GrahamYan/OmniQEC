"""Brute-force BB / abelian-LP search at one total-qubit budget (2026-09-15).

    python -m omniqec_agentic.hpc.brute_force_budget --budget 48 --out runs/brute_b48 --workers 127

Three stages, every artefact under --out:

  0. enumerate  every BB spec (ell*m <= budget/4, |A|,|B| <= --max-terms, weight 4..8,
                (0,0) in both A and B -- monomial shifts are permutations) and every
                lp_abelian spec (abelian groups of order 2..budget//4, shapes from the
                family pool, row-0 / column-0 cells contain the identity -- row and
                column scalings are permutations, per-cell weight <= --max-cell) whose
                compiled code fits the budget, has k >= 1 and passes the structural
                gate; distinct codes are kept by canonical (pynauty) hash, then the
                d_search proxy of the campaign is run -> enumerated.csv
  1. measure    every distinct code with d_search >= --min-d dynamically: up to
                --screen-shots per basis, stopping as soon as every basis has
                --screen-max-errors logical errors (user decision 2026-09-15: 30-50
                errors is enough, no precise stage); the exact code distance (MILP) is
                recorded by the slow stage -> stage1/
  2. confirm    OPTIONAL (--confirm-top / --confirm-kd2-top > 0): re-measure the best
                by plpr and the largest k*d^2 at --confirm-shots, no early stop -> stage2/

Noise: pauli_label_uniform_v1 at p=0.001 (measurement / reset / resonator-idle p,
X,Y,Z each p on both qubits of every CX, no ordinary idle) on the v3.7 measurement
protocol otherwise (telescoping_cpu, twobasis SEC, T=12, no check-weight cap).

Outputs: all_codes.csv (one row per distinct code, best available measurement),
top5_lowest_plpr.csv, top_kd2_vs_plpr.csv (largest k*d^2 with their plpr and the
ratio to the best plpr).
"""
from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import dataclasses
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------- enumeration


def _abelian_groups(max_order: int) -> list[list[int]]:
    """Invariant-factor lists of every abelian group of order 2..max_order."""

    def partitions_of_order(order: int) -> list[list[int]]:
        # factor lists q1 | q2 | ... with product == order (invariant factors)
        out: list[list[int]] = []

        def rec(remaining: int, prev: int, acc: list[int]) -> None:
            if remaining == 1:
                if acc:
                    out.append(list(acc))
                return
            for q in range(2, remaining + 1):
                if remaining % q == 0 and (not acc or q % prev == 0):
                    rec(remaining // q, q, acc + [q])

        rec(order, 1, [])
        # a list is invariant-factor form when each divides the next
        return [f for f in out if all(f[i + 1] % f[i] == 0 for i in range(len(f) - 1))]

    groups: list[list[int]] = []
    for order in range(2, max_order + 1):
        groups.extend(partitions_of_order(order))
    return groups


def _group_elements(factors: list[int]) -> list[list[int]]:
    return [list(t) for t in itertools.product(*[range(q) for q in factors])]


def enumerate_bb(budget: int, max_terms: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for ell in range(1, budget // 4 + 1):
        for m in range(ell, budget // 4 + 1):
            if 4 * ell * m > budget:
                break
            monomials = [(i, j) for i in range(ell) for j in range(m)]
            rest = [t for t in monomials if t != (0, 0)]
            sides: list[tuple[tuple[int, int], ...]] = []
            for extra in range(0, max_terms):
                for combo in itertools.combinations(rest, extra):
                    sides.append(((0, 0),) + combo)
            for ai, a in enumerate(sides):
                for b in sides[ai:]:  # [A|B] and [B|A] are the same code
                    w = len(a) + len(b)
                    if not 4 <= w <= 8:
                        continue
                    specs.append({"family": "bb", "ell": ell, "m": m,
                                  "A": [list(t) for t in a], "B": [list(t) for t in b]})
    return specs


def enumerate_lp_abelian(budget: int, max_cell: int, max_cell_2x2: int) -> list[dict[str, Any]]:
    shapes = ((1, 1), (1, 2), (2, 1), (2, 2))
    specs: list[dict[str, Any]] = []
    for factors in _abelian_groups(budget // 4):
        order = math.prod(factors)
        elems = _group_elements(factors)
        identity = [0] * len(factors)
        others = [e for e in elems if e != identity]

        def cells_for(shape: tuple[int, int], cap: int) -> list[list[list[list[int]]]]:
            rows, cols = shape
            per_cell: list[list[list[list[int]]]] = []  # candidate lists per cell position
            for r in range(rows):
                for c in range(cols):
                    pinned = r == 0 or c == 0  # row-0 / column-0 cells contain the identity
                    opts: list[list[list[int]]] = []
                    if pinned:
                        for extra in range(0, cap):
                            for combo in itertools.combinations(others, extra):
                                opts.append([identity] + [list(x) for x in combo])
                    else:
                        for size in range(1, cap + 1):
                            for combo in itertools.combinations(elems, size):
                                opts.append([list(x) for x in combo])
                    per_cell.append(opts)
            matrices: list[list[list[list[int]]]] = []
            for choice in itertools.product(*per_cell):
                matrix = [[choice[r * cols + c] for c in range(cols)] for r in range(rows)]
                matrices.append(matrix)
            return matrices

        for a_shape in shapes:
            for b_shape in shapes:
                if (sum(a_shape)) * (sum(b_shape)) * order > budget:
                    continue
                if a_shape == (1, 1) and b_shape == (1, 1) and len(factors) <= 2:
                    continue  # a 1x1 lift over C_l x C_m IS the bb family
                cap_a = max_cell_2x2 if a_shape == (2, 2) else max_cell
                cap_b = max_cell_2x2 if b_shape == (2, 2) else max_cell
                a_mats = cells_for(a_shape, cap_a)
                b_mats = cells_for(b_shape, cap_b)
                for a in a_mats:
                    wa = sum(len(cell) for row in a for cell in row)
                    for b in b_mats:
                        wb = sum(len(cell) for row in b for cell in row)
                        if wa + wb > 8 * max(a_shape[0], b_shape[1]):  # loose: the compiled row weight is checked later
                            continue
                        specs.append({"family": "lp_abelian", "group": {"factors": list(factors)}, "A": a, "B": b})
    return specs


# ----------------------------------------------------------------------------- stage 0 workers

_BUDGET = 0


def _init(budget: int) -> None:
    global _BUDGET
    _BUDGET = int(budget)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _structural(item: tuple[int, dict[str, Any]]) -> dict[str, Any] | None:
    index, spec = item
    from omniqec_agentic.verification.structural import structural_check_spec

    res = structural_check_spec(f"brute_{index:07d}", spec, n_budget=_BUDGET, total_budget=_BUDGET,
                                require_positive_k=True, max_check_weight=None)
    if not res.valid or not res.k or int(res.k) < 1:
        return None
    n_total = int(res.n_data or 0) + int(res.n_checks_x or 0) + int(res.n_checks_z or 0)
    if n_total > _BUDGET:
        return None
    return {"index": index, "spec": spec, "n": int(res.n_data), "k": int(res.k), "w": int(res.max_check_weight or 0),
            "n_total": n_total, "code_hash": res.code_hash, "canonical_hash": res.canonical_hash or res.code_hash,
            "family": spec["family"]}


def _proxy(row: dict[str, Any]) -> dict[str, Any]:
    from omniqec_agentic.verification.fast import evaluate_proxy

    metrics = evaluate_proxy(row["spec"], iters=120)
    row["d_search"] = metrics.get("d_search")
    row["proxy_valid"] = bool(metrics.get("proxy_valid"))
    return row


# ----------------------------------------------------------------------------- measurement


def _verification(shots: int, rounds: int, p: float):
    from omniqec_agentic.config import VerificationConfig
    from omniqec_agentic.verification.noise import NoiseSweepConfig

    base = VerificationConfig()
    noise = NoiseSweepConfig.pauli_label_uniform((float(p),))
    return dataclasses.replace(base, noise=noise, p_values=(float(p),), shots=int(shots), rounds=int(rounds),
                               max_check_weight=None, max_errors=None)


def _racing(shots: int, max_errors: int | None):
    from omniqec_agentic.config import formal_racing

    if max_errors is None:
        return None
    r = formal_racing(25000)
    return dataclasses.replace(r, shots_pilot=min(int(r.shots_pilot), int(shots)), shots_confirm=int(shots),
                               shots_max=int(shots), max_errors=int(max_errors), max_errors_in_pilot=True)


def _measure(rows: list[dict[str, Any]], out_dir: Path, *, shots: int, rounds: int, p: float, workers: int,
             max_errors: int | None, seed: int) -> dict[str, dict[str, str]]:
    from omniqec_agentic.evaluation import evaluate_promotion_queue

    out_dir.mkdir(parents=True, exist_ok=True)
    queue = out_dir / "queue.jsonl"
    queue.write_text("".join(json.dumps({"program_id": r["program_id"], "spec": r["spec"], "code_hash": r["code_hash"],
                                         "canonical_hash": r["canonical_hash"], "selection_channels": ["brute_force"],
                                         "is_baseline": False, "retry_at_final": False}) + "\n" for r in rows))
    t = time.perf_counter()
    summary = evaluate_promotion_queue(queue_path=queue, out_dir=out_dir, verification=_verification(shots, rounds, p),
                                       hardware="ion", seed=seed, batch_size=1024, include_baseline=False, workers=workers,
                                       shot_shards=16, deformation_mode="none", racing=_racing(shots, max_errors),
                                       stage="search")
    print(f"[brute] measured {len(rows)} codes at {shots} shots/basis in {time.perf_counter() - t:.0f}s "
          f"status={summary.get('status')}", flush=True)
    results = {}
    with (out_dir / "slow_results.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            results[r["program_id"]] = r
    return results


# ----------------------------------------------------------------------------- tables

COLS = ["rank_plpr", "program_id", "family", "n", "k", "d_search", "d_exact", "check_weight", "kd2", "n_total", "stage",
        "status", "plpr", "ci_low", "ci_high", "shots_x", "failures_x", "shots_z", "failures_z", "stopped_by",
        "code_hash8", "canonical_hash8", "spec"]


def _f(x: str | None) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except ValueError:
        return None


def _row(code: dict[str, Any], meas: dict[str, str] | None, stage: str) -> dict[str, Any]:
    d_exact = _f(meas.get("exact_code_distance")) if meas else None
    d = int(d_exact) if d_exact else (int(code["d_search"]) if code.get("d_search") else None)
    plpr = _f(meas.get("per_logical_per_round")) if meas else None
    return {"program_id": code["program_id"], "family": code["family"], "n": code["n"], "k": code["k"],
            "d_search": code.get("d_search"), "d_exact": int(d_exact) if d_exact else "",
            "check_weight": code["w"], "kd2": code["k"] * d * d if d else "", "n_total": code["n_total"],
            "stage": stage, "status": meas.get("status") if meas else "unmeasured", "plpr": plpr,
            "ci_low": _f(meas.get("per_logical_per_round_ci_low")) if meas else None,
            "ci_high": _f(meas.get("per_logical_per_round_ci_high")) if meas else None,
            "shots_x": meas.get("shots_x") if meas else "", "failures_x": meas.get("failures_x") if meas else "",
            "shots_z": meas.get("shots_z") if meas else "", "failures_z": meas.get("failures_z") if meas else "",
            "stopped_by": meas.get("stopped_by") if meas else "", "code_hash8": str(code["code_hash"])[:8],
            "canonical_hash8": str(code["canonical_hash"])[:8], "spec": json.dumps(code["spec"], separators=(",", ":"))}


def _write(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=48, help="total qubit budget (data + checks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--max-terms", type=int, default=4, help="BB: max monomials per side")
    ap.add_argument("--max-cell", type=int, default=3, help="LP: max group elements per cell (1x1/1x2/2x1)")
    ap.add_argument("--max-cell-2x2", type=int, default=2, help="LP: max group elements per cell (2x2 shapes)")
    ap.add_argument("--min-d", type=int, default=4, help="screen only codes with d_search >= this")
    ap.add_argument("--screen-max", type=int, default=8000,
                    help="cap on the number of screened codes (largest d_search, then k*d^2, first)")
    ap.add_argument("--screen-shots", type=int, default=25000, help="shots per basis cap")
    ap.add_argument("--screen-max-errors", type=int, default=50, help="stop once every basis has this many errors")
    ap.add_argument("--confirm-shots", type=int, default=25000)
    ap.add_argument("--confirm-top", type=int, default=0, help="0 = no confirmation stage")
    ap.add_argument("--confirm-kd2-top", type=int, default=0)
    ap.add_argument("--limit-specs", type=int, default=0, help="debug: only the first N enumerated specs")
    ap.add_argument("--families", default="bb,lp_abelian")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    families = set(args.families.split(","))
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    # ---- stage 0: enumerate + structural gate + dedupe + proxy distance ----
    t0 = time.perf_counter()
    specs: list[dict[str, Any]] = []
    if "bb" in families:
        specs += enumerate_bb(args.budget, args.max_terms)
    if "lp_abelian" in families:
        specs += enumerate_lp_abelian(args.budget, args.max_cell, args.max_cell_2x2)
    if args.limit_specs:
        specs = specs[: args.limit_specs]
    print(f"[brute] budget={args.budget} enumerated {len(specs)} specs "
          f"(bb={sum(s['family'] == 'bb' for s in specs)}, lp_abelian={sum(s['family'] == 'lp_abelian' for s in specs)})",
          flush=True)
    distinct: dict[str, dict[str, Any]] = {}
    with Pool(workers, initializer=_init, initargs=(args.budget,)) as pool:
        for res in pool.imap_unordered(_structural, enumerate(specs), chunksize=64):
            if res is None:
                continue
            key = res["canonical_hash"]
            prev = distinct.get(key)
            if prev is None or res["index"] < prev["index"]:
                distinct[key] = res
        print(f"[brute] {len(distinct)} distinct valid codes (k>=1, n_total<={args.budget}) after "
              f"{time.perf_counter() - t0:.0f}s; proxy distance ...", flush=True)
        codes = list(pool.imap_unordered(_proxy, list(distinct.values()), chunksize=8))
    codes.sort(key=lambda r: r["index"])
    for i, r in enumerate(codes):
        r["program_id"] = f"b{args.budget}_{r['family']}_{i:05d}"
    enum_rows = [_row(c, None, "enumerated") for c in codes]
    _write(out / "enumerated.csv", enum_rows, COLS)
    print(f"[brute] stage 0 done in {time.perf_counter() - t0:.0f}s: {len(codes)} codes -> enumerated.csv", flush=True)

    # ---- stage 1: screen ----
    screen = [c for c in codes if c.get("d_search") and int(c["d_search"]) >= args.min_d]
    screen.sort(key=lambda c: (-int(c["d_search"]), -int(c["k"]) * int(c["d_search"]) ** 2, c["index"]))
    if len(screen) > args.screen_max:
        print(f"[brute] {len(screen)} codes pass d_search>={args.min_d}; screening the {args.screen_max} with the "
              "largest d_search / k*d^2", flush=True)
        screen = screen[: args.screen_max]
    print(f"[brute] stage 1: measuring {len(screen)} codes with d_search>={args.min_d}: up to {args.screen_shots} "
          f"shots/basis, stop at {args.screen_max_errors} errors per basis", flush=True)
    s1 = _measure(screen, out / "stage1", shots=args.screen_shots, rounds=args.rounds, p=args.p, workers=workers,
                  max_errors=args.screen_max_errors, seed=0)
    by_id = {c["program_id"]: c for c in codes}
    rows1 = [_row(c, s1.get(c["program_id"]), "screen") for c in screen]
    ok1 = [r for r in rows1 if r["plpr"] is not None]
    ok1.sort(key=lambda r: r["plpr"])
    _write(out / "stage1_screen.csv", ok1 + [r for r in rows1 if r["plpr"] is None], COLS)

    # ---- stage 2: confirm the best by plpr and the largest kd^2 ----
    for r in ok1:
        r["kd2_num"] = float(r["kd2"]) if r["kd2"] != "" else 0.0
    confirm_ids = [r["program_id"] for r in ok1[: args.confirm_top]]
    for r in sorted(ok1, key=lambda r: -r["kd2_num"])[: args.confirm_kd2_top]:
        if r["program_id"] not in confirm_ids:
            confirm_ids.append(r["program_id"])
    confirm = [by_id[pid] for pid in confirm_ids]
    s2: dict[str, dict[str, str]] = {}
    if confirm:
        print(f"[brute] stage 2: confirming {len(confirm)} codes at {args.confirm_shots} shots/basis (no early stop)",
              flush=True)
        s2 = _measure(confirm, out / "stage2", shots=args.confirm_shots, rounds=args.rounds, p=args.p, workers=workers,
                      max_errors=None, seed=1)

    # ---- tables ----
    final: dict[str, dict[str, Any]] = {}
    for r in rows1:
        final[r["program_id"]] = r
    for c in confirm:
        m = s2.get(c["program_id"])
        if m and _f(m.get("per_logical_per_round")) is not None:
            final[c["program_id"]] = _row(c, m, "confirm")
    for c in codes:
        final.setdefault(c["program_id"], _row(c, None, "enumerated"))
    all_rows = sorted(final.values(), key=lambda r: (r["plpr"] is None, r["plpr"] if r["plpr"] is not None else 0.0))
    for i, r in enumerate(all_rows, 1):
        r["rank_plpr"] = i if r["plpr"] is not None else ""
    _write(out / "all_codes.csv", all_rows, COLS)
    confirmed = [r for r in all_rows if r["stage"] == "confirm"] or [r for r in all_rows if r["plpr"] is not None]
    top5 = confirmed[:5]
    _write(out / "top5_lowest_plpr.csv", top5, COLS)
    best = top5[0]["plpr"] if top5 else None
    kd2_rows = [r for r in all_rows if r["plpr"] is not None and r["kd2"] != ""]
    kd2_rows.sort(key=lambda r: (-float(r["kd2"]), r["plpr"]))
    for r in kd2_rows:
        r["plpr_over_best"] = (r["plpr"] / best) if best else ""
    _write(out / "top_kd2_vs_plpr.csv", kd2_rows[: max(args.confirm_kd2_top, 20)], COLS + ["plpr_over_best"])
    print("[brute] top-5 lowest plpr:", flush=True)
    for r in top5:
        print(f"   [[{r['n']},{r['k']},{r['d_exact'] or r['d_search']}]] {r['family']} w={r['check_weight']} "
              f"kd2={r['kd2']} plpr={r['plpr']:.3e} [{r['ci_low']:.2e}, {r['ci_high']:.2e}] {r['program_id']}", flush=True)
    print(f"[brute] done in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
