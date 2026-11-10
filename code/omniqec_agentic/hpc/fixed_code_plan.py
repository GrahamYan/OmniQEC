"""Plans for the fixed-code measurements of 2026-10-23 (schema ``fixed_code_plan_v1``).

One plan per block; ``hpc/fixed_code_measure.py`` measures and merges every one of them the
same way. The rules here are the plans' (Fig4 v28, appendix S1-S4), nothing is decided at
measurement time:

  fig4-ab        Fig4 a: the 293 new T=12 cross-task conditions (A_new_T12_measurements_293.csv,
                 both tasks at T=12, rounds policy fixed_Tc) and Fig4 b: the 5 extra native
                 Memory confirmations (B_extra_native_confirmations_5.csv, T = max(12, d_search),
                 rounds policy max_Tc_d). 25k shots per basis, p = 0.005, no search weight cap.
  s4             S4: the 20 Fig.2 codes (18 champions N <= 260 + BB72 + BB144) at the 8 new
                 points of the p grid (the p = 0.005 anchor is reused, not re-measured), memory
                 task, native T (= 12 for all 20 codes). ``--stage pilot`` fixes a small shot
                 count per condition to estimate q_b and the decode cost; ``--stage formal``
                 sizes S_b = min(cap, 100 / q_b) from the merged pilot, rounded up, with an
                 anchor-based extrapolation when the pilot saw no failure. Separate seed
                 namespaces (s4_pilot_v1 / s4_formal_v1): pilot shots never enter the result.
  a-inset        Fig4 a inset: the frozen pair (two codes, one N) re-measured independently in
                 both tasks, S = 25000 * ceil(max(25000, 400 / q) / 25000) per code and basis
                 from the exploratory counts, q = F_X/s_X + F_Z/s_Z (both zero: q = 0.5/(s_X+1) +
                 0.5/(s_Z+1)); the two codes of a task share the larger S.
  c-supplement   champion supplementation by the Fig.3 rule, for any set of finished runs:
                 f = min(final X, final Z failures); f >= 30 -> 0; f = 0 -> 500k; otherwise
                 min(500k, 50k * ceil((45 * 25k / f) / 50k)); unified to the maximum within a
                 group (the same N); pooled with the formal 25k counts at merge time. ``--top K``
                 takes the K best finalists of every run (S3 Top-3 re-measurement).

The plan builder needs the package only to count n and k of a spec (``--no-structural``
skips that; the measurement fills them in). Every item carries its own seed namespace.

    python -m omniqec_agentic.hpc.fixed_code_plan fig4-ab --a-csv A_new_T12_measurements_293.csv \
        --b-csv B_extra_native_confirmations_5.csv --out fig4_ab_plan.json
    python -m omniqec_agentic.hpc.fixed_code_plan s4 --points-csv fig2_points_current.csv \
        --refs-plan fig2_extended_stats_plan_20261014.json --stage pilot --pilot-shots 20000 --out s4_pilot_plan.json
    python -m omniqec_agentic.hpc.fixed_code_plan s4 ... --stage formal --pilot-results s4_pilot_merged.csv \
        --cap 1000000 --out s4_formal_plan.json
    python -m omniqec_agentic.hpc.fixed_code_plan a-inset --a-results fig4_ab_merged.csv --budget 144 \
        --codes <hash8> <hash8> --out fig4_a_inset_plan.json
    python -m omniqec_agentic.hpc.fixed_code_plan c-supplement --runs-root RUNS --run-dir NAME [NAME ...] \
        [--top 3] --out c_supplement_plan.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from omniqec_agentic.hpc.fixed_code_measure import PLAN_SCHEMA

TASK_OF = {"Memory": "memory", "CNOT": "cnot_dense", "memory": "memory", "cnot_dense": "cnot_dense"}
PROTOCOL_RUN_OF = {"memory": "fig2_claudesub_b{N}_v37", "cnot_dense": "fig2_claudesub_b{N}_f3cnot_v2"}
S4_P_GRID = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01)
S4_ANCHOR_P = 0.005
S4_ARMS = ("Claude", "GPT", "DeepSeek")
S4_BUDGETS = (60, 98, 144, 180, 220, 260)


# --------------------------------------------------------------------------- helpers
def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_plan(path: str | Path, plan: dict[str, Any]) -> None:
    keys = [item["key"] for item in plan["items"]]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"duplicate keys: {sorted({k for k in keys if keys.count(k) > 1})[:5]}")
    namespaces = [item["seed_namespace"] for item in plan["items"]]
    if len(namespaces) != len(set(namespaces)):
        raise SystemExit("duplicate seed namespaces")
    Path(path).write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def new_plan(block: str, version: str, namespace_root: str, rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA, "block": block, "version": version, "created": time.strftime("%Y-%m-%d %H:%M"),
        "seed_namespace_root": namespace_root, "rule": rule, "items": [],
    }


def hash8(code_hash: str | None, spec: dict[str, Any] | None = None) -> str:
    if code_hash:
        return str(code_hash)[:8]
    import hashlib

    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def structural_nk(key: str, spec: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """n, k and the observed check weight of a spec (the package builds the matrices)."""

    if not enabled:
        return {}
    from omniqec_agentic.verification.structural import structural_check_spec

    res = structural_check_spec(key, spec, require_positive_k=True)
    if not res.valid:
        raise SystemExit(f"{key}: spec does not build or is structurally invalid: {res.failure_reasons}")
    return {"n": int(res.n_data), "k": int(res.k), "w_max_observed": int(res.max_check_weight)}


def round_up(value: float, step: int) -> int:
    return int(math.ceil(float(value) / float(step)) * int(step))


# --------------------------------------------------------------------------- fig4 a/b
def build_fig4_ab(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]], *, structural: bool = True,
                  namespace_root: str = "fixed_code_v1", shots_default: int = 25000) -> dict[str, Any]:
    plan = new_plan("fig4_ab", "fig4_v28", namespace_root, {
        "a": "both tasks at T=12 (rounds_policy fixed_Tc), p=0.005, 25k shots per basis, no racing, no search weight cap",
        "b": "native Memory confirmation, T = max(12, d_search) (rounds_policy max_Tc_d), 25k shots per basis",
    })
    for row in a_rows:
        task = TASK_OF[row["target_task"]]
        spec = json.loads(row["spec"])
        budget = int(row["budget"])
        key = f"A_N{budget}_{'mem' if task == 'memory' else 'cnot'}_{hash8(row.get('code_hash'), spec)}"
        rounds = int(row.get("target_rounds") or 12)
        if rounds != 12:
            raise SystemExit(f"{key}: Fig4 a is the T=12 cross-evaluation, got target_rounds={rounds}")
        if str(row.get("racing", "False")).lower() == "true" or str(row.get("search_weight_cap_applies", "False")).lower() == "true":
            raise SystemExit(f"{key}: a row asks for racing or a search weight cap; the fixed-code evaluation has neither")
        nk = structural_nk(key, spec, structural)
        if nk and row.get("w_max") and int(row["w_max"]) != nk["w_max_observed"]:
            raise SystemExit(f"{key}: w_max {row['w_max']} in the manifest but the built code has check weight {nk['w_max_observed']}")
        item = {
            "key": key, "block": "fig4_a", "measurement_id": row.get("measurement_id"), "task": task, "budget": budget,
            "protocol_run": PROTOCOL_RUN_OF[task].format(N=budget), "spec": spec, "code_hash": row.get("code_hash"),
            "program_id": f"fixed_{hash8(row.get('code_hash'), spec)}", "family": spec.get("family"),
            "n": nk.get("n"), "k": nk.get("k"), "d_search": None, "w_max": int(row["w_max"]) if row.get("w_max") else nk.get("w_max_observed"),
            "logicals": (None if not nk else (nk["k"] if task == "memory" else 2 * nk["k"])),
            "p": float(row.get("noise_p") or 0.005), "shots_per_basis": int(row.get("initial_proposed_shots_per_basis") or shots_default),
            "rounds_floor": 12, "rounds_policy": "fixed_Tc", "expected_rounds": 12, "max_check_weight": None,
            "seed_namespace": f"{namespace_root}/fig4_a/{key}",
            "origin": {k: row.get(k) for k in ("membership_status", "selection_tasks", "selection_models", "spec_source",
                                                "measurement_action", "reason", "existing_source_file", "rng_seed")},
        }
        plan["items"].append(item)
    for row in b_rows:
        task = TASK_OF[row["task"]]
        if task != "memory":
            raise SystemExit("Fig4 b confirmations are native Memory measurements")
        spec = json.loads(row["spec"])
        budget = int(row["budget"])
        key = f"B_N{budget}_{row['model']}_{hash8(row.get('code_hash'), spec)}"
        nk = structural_nk(key, spec, structural)
        item = {
            "key": key, "block": "fig4_b", "measurement_id": row.get("confirmation_id"), "task": "memory", "budget": budget,
            "protocol_run": PROTOCOL_RUN_OF["memory"].format(N=budget), "spec": spec, "code_hash": row.get("code_hash"),
            "program_id": f"fixed_{hash8(row.get('code_hash'), spec)}", "family": spec.get("family"),
            "n": nk.get("n"), "k": nk.get("k"), "d_search": None, "w_max": int(row["w_max"]) if row.get("w_max") else nk.get("w_max_observed"),
            "logicals": nk.get("k"), "p": 0.005, "shots_per_basis": int(row.get("shots_per_basis") or shots_default),
            "rounds_floor": 12, "rounds_policy": "max_Tc_d", "expected_rounds": int(row["native_rounds"]) if row.get("native_rounds") else None,
            "max_check_weight": None, "seed_namespace": f"{namespace_root}/fig4_b/{key}",
            "origin": {k: row.get(k) for k in ("model", "rank", "source_run", "source_file", "search_failures_x", "search_failures_z",
                                                "search_upper_ler", "status", "selection_rule", "rng_seed", "shared_with_A_T12")},
        }
        plan["items"].append(item)
    return plan


# --------------------------------------------------------------------------- S4
def s4_codes(points_rows: list[dict[str, str]], refs_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The 20 S4 codes: Fig.2 champions N <= 260 (one per arm) + the two references."""

    codes = []
    for row in points_rows:
        if row.get("arm") not in S4_ARMS or int(row["budget"]) not in S4_BUDGETS:
            continue
        if not row.get("spec"):
            raise SystemExit(f"points row b{row['budget']} {row['arm']}: no spec")
        shots = int(float(row.get("extension_shots_per_basis") or 0)) + 25000
        fx = int(float(row.get("final_failures_x") or 0)) + int(float(row.get("extension_failures_x") or 0))
        fz = int(float(row.get("final_failures_z") or 0)) + int(float(row.get("extension_failures_z") or 0))
        codes.append({
            "code_id": f"b{row['budget']}_{row['arm'].lower()}_{row['code_hash8']}", "budget": int(row["budget"]), "arm": row["arm"],
            "role": "champion", "code_hash": row["code_hash8"], "spec": json.loads(row["spec"]), "family": row.get("family"),
            "n": int(row["n"]), "k": int(row["k"]), "d_search": int(row["d_search"]), "w_max": int(row["max_check_weight"]),
            "rounds": int(row["rounds"]), "source_run": row.get("source_run"),
            "anchor": {"p": S4_ANCHOR_P, "shots_per_basis": shots, "failures_x": fx, "failures_z": fz,
                       "plpr": float(row["plotted_plpr"]) if row.get("plotted_plpr") else None, "measurement": row.get("measurement")},
        })
    for item in refs_plan["items"]:
        if item.get("role") != "reference":
            continue
        codes.append({
            "code_id": f"b{item['budget']}_ref_{str(item['code_hash'])[:8]}", "budget": int(item["budget"]), "arm": "reference",
            "role": "reference", "code_hash": item["code_hash"], "spec": item["spec"], "family": item["family"],
            "n": int(item["n"]), "k": int(item["k"]), "d_search": int(item["d_search"]), "w_max": int(item["max_check_weight"]),
            "rounds": int(item.get("formal_rounds") or 12), "source_run": item.get("source_run"),
            "anchor": {"p": S4_ANCHOR_P, "shots_per_basis": None, "failures_x": None, "failures_z": None, "plpr": item.get("formal_plpr"),
                       "measurement": "reference anchor: extension alone (fig2_extended_stats.csv)"},
        })
    if len(codes) != 20:
        raise SystemExit(f"S4 expects 20 codes (18 champions + 2 references), found {len(codes)}")
    for code in codes:
        if code["d_search"] > 12:
            raise SystemExit(f"{code['code_id']}: d_search {code['d_search']} > 12, native T would not be 12")
    return codes


def s4_formal_shots(pilot: dict[str, Any] | None, anchor: dict[str, Any], d_search: int, p: float, *,
                    target_failures: int, cap: int, round_to: int) -> tuple[int, str]:
    """S_b = min(cap, target / q_b) for the weaker basis, rounded up to ``round_to``.

    q_b from the pilot when it saw failures; otherwise an upper-tail extrapolation of the
    anchor (p = 0.005) rate with the code's distance, q(p) ~ q_anchor * (p / p_anchor)^ceil((d+1)/2),
    which is the leading-order scaling below threshold (recorded as the rule used)."""

    q_candidates: list[tuple[float, str]] = []
    if pilot and int(pilot.get("shots_x") or 0) > 0 and int(pilot.get("shots_z") or 0) > 0:
        for basis in ("x", "z"):
            f = int(pilot.get(f"failures_{basis}") or 0)
            if f > 0:
                q_candidates.append((f / float(pilot[f"shots_{basis}"]), f"pilot_{basis}"))
    if q_candidates:
        q, rule = min(q_candidates)
        rule = f"pilot:{rule}"
    else:
        shots = anchor.get("shots_per_basis")
        fx, fz = anchor.get("failures_x"), anchor.get("failures_z")
        if shots and fx is not None and fz is not None and (fx + fz) > 0:
            q_anchor = max(1, min(fx, fz)) / float(shots)
        elif anchor.get("plpr"):
            # a reference anchor (extension alone): block rate from the per-logical rate, k*T logicals x rounds
            q_anchor = float(anchor["plpr"]) * float(anchor.get("logicals_rounds") or 1)
        else:
            raise SystemExit("no pilot failures and no anchor counts to extrapolate from")
        exponent = math.ceil((int(d_search) + 1) / 2)
        q = q_anchor * (float(p) / S4_ANCHOR_P) ** exponent
        rule = f"anchor_extrapolation:q_anchor={q_anchor:.3g},exp={exponent}"
    if q <= 0:
        return int(cap), rule + ";q=0->cap"
    need = target_failures / q
    shots = min(int(cap), round_up(need, round_to))
    return int(shots), rule + (";capped" if need > cap else "")


def build_s4(points_rows: list[dict[str, str]], refs_plan: dict[str, Any], *, stage: str, p_grid: tuple[float, ...] = S4_P_GRID,
             pilot_shots: int = 20000, pilot_results: list[dict[str, str]] | None = None, target_failures: int = 100,
             cap: int = 1_000_000, round_to: int = 50000, namespace_root: str | None = None) -> dict[str, Any]:
    if stage not in ("pilot", "formal"):
        raise SystemExit("--stage must be pilot or formal")
    namespace_root = namespace_root or f"s4_{stage}_v1"
    plan = new_plan(f"s4_{stage}", "appendix_s4_20261023", namespace_root, {
        "task": "memory (Fig.2 protocol), native T = 12 for every S4 code, uniform_paper_v1, only p changes",
        "p_grid": list(p_grid), "anchor_p_reused": S4_ANCHOR_P,
        "stage": stage, "pilot_shots_per_basis": pilot_shots if stage == "pilot" else None,
        "formal_rule": None if stage == "pilot" else f"S_b = min({cap}, {target_failures} / q_b) rounded up to {round_to}; q_b from the pilot, anchor extrapolation when the pilot saw no failure",
    })
    pilot_by_key = {r["key"]: r for r in (pilot_results or [])}
    for code in s4_codes(points_rows, refs_plan):
        anchor = dict(code["anchor"])
        anchor["logicals_rounds"] = code["k"] * code["rounds"]
        for p in p_grid:
            if abs(p - S4_ANCHOR_P) < 1e-12:
                continue
            p_tag = f"{p:g}".replace(".", "p")
            base_key = f"S4_{code['code_id']}_p{p_tag}"
            key = base_key if stage == "formal" else f"{base_key}_pilot"
            if stage == "pilot":
                shots, rule = int(pilot_shots), "pilot_fixed"
            else:
                pilot = pilot_by_key.get(f"{base_key}_pilot")
                if pilot is not None and pilot.get("status") not in ("ok",):
                    raise SystemExit(f"{base_key}: pilot status {pilot.get('status')!r}; fix or drop it before sizing the formal run")
                shots, rule = s4_formal_shots(pilot, anchor, code["d_search"], p, target_failures=target_failures, cap=cap, round_to=round_to)
            plan["items"].append({
                "key": key, "block": f"s4_{stage}", "measurement_id": key, "task": "memory", "budget": code["budget"],
                "protocol_run": PROTOCOL_RUN_OF["memory"].format(N=code["budget"]), "spec": code["spec"], "code_hash": code["code_hash"],
                "program_id": f"fixed_{str(code['code_hash'])[:8]}", "family": code["family"], "n": code["n"], "k": code["k"],
                "d_search": code["d_search"], "w_max": code["w_max"], "logicals": code["k"], "p": float(p), "shots_per_basis": int(shots),
                "rounds_floor": 12, "rounds_policy": "max_Tc_d", "expected_rounds": 12, "max_check_weight": None,
                "seed_namespace": f"{namespace_root}/{key}", "shots_rule": rule,
                "origin": {"arm": code["arm"], "role": code["role"], "source_run": code["source_run"], "anchor": code["anchor"],
                           "pilot_key": f"{base_key}_pilot" if stage == "formal" else None},
            })
    return plan


# --------------------------------------------------------------------------- Fig4 a inset
def inset_shots(fx: int, fz: int, sx: int, sz: int, *, target: int = 400, base: int = 25000) -> int:
    """S = base * ceil(max(base, target / q) / base), q = F_X/s_X + F_Z/s_Z (both zero: 0.5/(s_X+1) + 0.5/(s_Z+1))."""

    if fx == 0 and fz == 0:
        q = 0.5 / (sx + 1) + 0.5 / (sz + 1)
    else:
        q = fx / float(sx) + fz / float(sz)
    return int(base * math.ceil(max(base, target / q) / base))


def build_a_inset(a_results: list[dict[str, str]], *, budget: int, codes: list[str], target: int = 400, base: int = 25000,
                  namespace_root: str = "fig4_a_inset_v1", structural: bool = True) -> dict[str, Any]:
    if len(codes) != 2:
        raise SystemExit("--codes takes exactly two code hashes (8+ hex chars)")
    plan = new_plan("fig4_a_inset", "fig4_v28", namespace_root, {
        "rule": f"S = {base} * ceil(max({base}, {target} / q) / {base}), q = F_X/s_X + F_Z/s_Z from the exploratory counts; both codes of a task share the larger S",
        "budget": budget, "codes": codes, "note": f"{target} is the expected total failures over both bases, not a stopping rule",
    })
    per_task: dict[str, list[dict[str, Any]]] = {"memory": [], "cnot_dense": []}
    for row in a_results:
        if int(row["budget"]) != budget or row["task"] not in per_task:
            continue
        if not any(str(row["code_hash"]).startswith(c) for c in codes):
            continue
        if row.get("status") != "ok":
            raise SystemExit(f"{row['key']}: exploratory measurement status {row.get('status')!r}, cannot size the inset")
        fx, fz, sx, sz = int(row["failures_x"]), int(row["failures_z"]), int(row["shots_x"]), int(row["shots_z"])
        per_task[row["task"]].append({"row": row, "own_shots": inset_shots(fx, fz, sx, sz, target=target, base=base)})
    for task, entries in per_task.items():
        if len(entries) != 2:
            raise SystemExit(f"task {task}: expected the two codes' exploratory rows at N={budget}, found {len(entries)}")
        shots = max(e["own_shots"] for e in entries)
        for e in entries:
            row = e["row"]
            spec = json.loads(row["origin"]).get("spec") if row.get("origin") and "spec" in row["origin"] else None
            if spec is None:
                raise SystemExit(f"{row['key']}: the merged a table does not carry the spec; pass the plan-derived table")
            key = f"INSET_N{budget}_{'mem' if task == 'memory' else 'cnot'}_{str(row['code_hash'])[:8]}"
            plan["items"].append({
                "key": key, "block": "fig4_a_inset", "measurement_id": key, "task": task, "budget": budget,
                "protocol_run": PROTOCOL_RUN_OF[task].format(N=budget), "spec": spec, "code_hash": row["code_hash"],
                "program_id": f"fixed_{str(row['code_hash'])[:8]}", "family": row.get("family"), "n": int(row["n"]) if row.get("n") else None,
                "k": int(row["k"]) if row.get("k") else None, "d_search": int(row["d_search"]) if row.get("d_search") else None,
                "w_max": int(row["w_max"]) if row.get("w_max") else None,
                "logicals": int(row["logicals"]) if row.get("logicals") else None, "p": 0.005, "shots_per_basis": int(shots),
                "rounds_floor": 12, "rounds_policy": "fixed_Tc", "expected_rounds": 12, "max_check_weight": None,
                "seed_namespace": f"{namespace_root}/{key}", "shots_rule": f"own {e['own_shots']}, task max {shots}",
                "origin": {"exploratory_key": row["key"], "exploratory_counts": {"failures_x": row["failures_x"], "failures_z": row["failures_z"],
                                                                                "shots_x": row["shots_x"], "shots_z": row["shots_z"]}},
            })
    return plan


# --------------------------------------------------------------------------- c supplement (Fig.3 rule)
def supplement_shots(f_min: int, *, target: int = 30, expected: float = 45.0, round_to: int = 50000, cap: int = 500000,
                     formal_shots: int = 25000) -> int:
    if f_min >= target:
        return 0
    if f_min <= 0:
        return int(cap)
    need = expected / (float(f_min) / float(formal_shots))
    return int(min(cap, round_up(need, round_to)))


def _ci_high(row: dict[str, Any]) -> float:
    ci = row.get("per_logical_per_round_ci") or [None, None]
    try:
        return float(ci[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")


def finalists(run_dir: Path, top: int) -> list[dict[str, Any]]:
    path = run_dir / "final_slow" / "T12" / "rankings.json"
    if not path.is_file():
        raise SystemExit(f"{run_dir.name}: no final_slow/T12/rankings.json (final stage not finished)")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rankings") if isinstance(data, dict) else data
    ok = [r for r in rows or [] if isinstance(r, dict) and r.get("status") == "ok" and not r.get("is_baseline")
          and r.get("per_logical_per_round") is not None]
    ok.sort(key=lambda r: (float(r["per_logical_per_round"]), _ci_high(r)))
    return ok[:top]


def _candidate_spec(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """The spec the final stage measured for this finalist, matched by code hash.

    The final queue keeps every finalist; a candidate file is named by program id, and two finalists
    that share a program id overwrite each other's file, so a file is used only when its compiled
    code hash is the finalist's."""

    code_hash = str(row.get("code_hash") or "")
    for queue in sorted((run_dir / "final_slow").glob("queue_T*.jsonl")):
        for line in queue.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if str(entry.get("code_hash") or "") == code_hash and entry.get("spec"):
                    return entry["spec"]
    local = run_dir / "final_slow" / "T12" / "candidates" / f"{row.get('program_id')}.json"
    for path in (local, Path(str(row.get("candidate_path") or ""))):
        if path.is_file():
            cand = json.loads(path.read_text(encoding="utf-8"))
            if cand.get("spec") and str((cand.get("compiled_code") or {}).get("code_hash") or "") == code_hash:
                return cand["spec"]
    if row.get("spec"):
        return row["spec"]
    raise SystemExit(f"{run_dir.name}: no spec for finalist {row.get('program_id')} ({code_hash})")


def build_c_supplement(runs_root: list[str], run_names: list[str], *, top: int = 1, target: int = 30, expected: float = 45.0,
                       round_to: int = 50000, cap: int = 500000, formal_shots: int = 25000,
                       namespace_root: str = "c_supplement_v1") -> tuple[dict[str, Any], list[str]]:
    plan = new_plan("c_supplement", "fig4_v28_s3", namespace_root, {
        "rule": f"f = min(final X, final Z failures); f >= {target} -> 0; f = 0 -> {cap}; else min({cap}, {round_to} * ceil(({expected} * {formal_shots} / f) / {round_to})); unified to the group (same N) maximum",
        "top": top, "pooling": "formal 25k counts + supplement, merged by fixed_code_measure --merge (pooled_* columns)",
    })
    report: list[str] = []
    entries: list[dict[str, Any]] = []
    for name in run_names:
        run_dir = next((Path(root) / name for root in runs_root if (Path(root) / name / "protocol_snapshot.json").is_file()), None)
        if run_dir is None:
            raise SystemExit(f"{name}: protocol_snapshot.json not found under {runs_root}")
        protocol = json.loads((run_dir / "protocol_snapshot.json").read_text(encoding="utf-8"))
        task = str((protocol.get("verification") or {}).get("task") or "memory")
        match = re.search(r"_b(\d+)_", name)
        budget = int(match.group(1)) if match else int(protocol.get("n_budget") or 0)
        for rank, row in enumerate(finalists(run_dir, top), start=1):
            fx, fz = int(row.get("failures_x") or 0), int(row.get("failures_z") or 0)
            own = supplement_shots(min(fx, fz), target=target, expected=expected, round_to=round_to, cap=cap, formal_shots=formal_shots)
            entries.append({"run": name, "run_dir": run_dir, "task": task, "budget": budget, "rank": rank, "row": row, "own": own, "fx": fx, "fz": fz})
    groups: dict[int, int] = {}
    for e in entries:
        groups[e["budget"]] = max(groups.get(e["budget"], 0), e["own"])
    for e in entries:
        row, shots = e["row"], groups[e["budget"]]
        report.append(f"{e['run']} #{e['rank']} [[{row.get('n')},{row.get('k')},{row.get('d_search')}]] w{row.get('max_check_weight')} "
                      f"{str(row.get('code_hash') or '')[:8]} fails={e['fx']}/{e['fz']} -> own {e['own'] // 1000}k, group {shots // 1000}k")
        if shots <= 0:
            continue
        key = f"SUP_{e['run']}_r{e['rank']}_{str(row.get('code_hash') or '')[:8]}"
        k = int(row.get("k") or 0)
        plan["items"].append({
            "key": key, "block": "c_supplement", "measurement_id": key, "task": e["task"], "budget": e["budget"],
            "protocol_run": e["run"], "spec": _candidate_spec(e["run_dir"], row), "code_hash": row.get("code_hash"),
            "program_id": row.get("program_id"), "family": row.get("family"), "n": row.get("n"), "k": k, "d_search": row.get("d_search"),
            "w_max": row.get("max_check_weight"), "logicals": (2 * k if e["task"] == "cnot_dense" else k), "p": 0.005,
            "shots_per_basis": int(shots), "rounds_floor": 12, "rounds_policy": "fixed_Tc" if e["task"] == "cnot_dense" else "max_Tc_d",
            "expected_rounds": int(row.get("rounds") or 12), "max_check_weight": None, "seed_namespace": f"{namespace_root}/{key}",
            "shots_rule": f"own {e['own']}, group max {shots}",
            "formal_shots_per_basis": int(row.get("shots_x") or formal_shots), "formal_failures_x": e["fx"], "formal_failures_z": e["fz"],
            "formal_plpr": row.get("per_logical_per_round"),
            "origin": {"run": e["run"], "final_rank": e["rank"], "program_id": row.get("program_id")},
        })
    return plan, report


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ab = sub.add_parser("fig4-ab")
    ab.add_argument("--a-csv", required=True)
    ab.add_argument("--b-csv", required=True)
    ab.add_argument("--out", required=True)
    ab.add_argument("--no-structural", action="store_true", help="do not build the codes to fill n/k (stdlib only)")
    ab.add_argument("--namespace-root", default="fixed_code_v1")

    s4 = sub.add_parser("s4")
    s4.add_argument("--points-csv", required=True, help="fig2_points_current.csv (consolidated_20261019)")
    s4.add_argument("--refs-plan", required=True, help="fig2_extended_stats_plan_20261014.json (BB72/BB144 specs)")
    s4.add_argument("--stage", choices=["pilot", "formal"], required=True)
    s4.add_argument("--p-grid", default=",".join(f"{p:g}" for p in S4_P_GRID))
    s4.add_argument("--pilot-shots", type=int, default=20000)
    s4.add_argument("--pilot-results", help="merged CSV of the pilot plan (fixed_code_measure --merge)")
    s4.add_argument("--target-failures", type=int, default=100)
    s4.add_argument("--cap", type=int, default=1_000_000, help="shots per basis ceiling for one condition")
    s4.add_argument("--round-to", type=int, default=50000)
    s4.add_argument("--namespace-root", default=None)
    s4.add_argument("--out", required=True)

    inset = sub.add_parser("a-inset")
    inset.add_argument("--a-results", required=True, help="merged CSV of the fig4-ab plan")
    inset.add_argument("--budget", type=int, required=True)
    inset.add_argument("--codes", nargs=2, required=True)
    inset.add_argument("--target", type=int, default=400)
    inset.add_argument("--out", required=True)

    sup = sub.add_parser("c-supplement")
    sup.add_argument("--runs-root", action="append", required=True)
    sup.add_argument("--run-dir", nargs="+", required=True, help="run directory names (final stage finished)")
    sup.add_argument("--top", type=int, default=1)
    sup.add_argument("--target", type=int, default=30)
    sup.add_argument("--expected", type=float, default=45.0)
    sup.add_argument("--round-to", type=int, default=50000)
    sup.add_argument("--cap", type=int, default=500000)
    sup.add_argument("--namespace-root", default="c_supplement_v1")
    sup.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "fig4-ab":
        plan = build_fig4_ab(read_csv(args.a_csv), read_csv(args.b_csv), structural=not args.no_structural, namespace_root=args.namespace_root)
    elif args.command == "s4":
        pilot = read_csv(args.pilot_results) if args.pilot_results else None
        if args.stage == "formal" and pilot is None:
            raise SystemExit("--stage formal needs --pilot-results")
        plan = build_s4(read_csv(args.points_csv), json.loads(Path(args.refs_plan).read_text(encoding="utf-8")), stage=args.stage,
                        p_grid=tuple(float(x) for x in args.p_grid.split(",")), pilot_shots=args.pilot_shots, pilot_results=pilot,
                        target_failures=args.target_failures, cap=args.cap, round_to=args.round_to, namespace_root=args.namespace_root)
    elif args.command == "a-inset":
        plan = build_a_inset(read_csv(args.a_results), budget=args.budget, codes=list(args.codes), target=args.target)
    else:
        plan, report = build_c_supplement(args.runs_root, args.run_dir, top=args.top, target=args.target, expected=args.expected,
                                          round_to=args.round_to, cap=args.cap, namespace_root=args.namespace_root)
        for line in report:
            print(line)
    write_plan(args.out, plan)
    total = sum(2 * int(i["shots_per_basis"]) for i in plan["items"])
    print(f"{len(plan['items'])} items, {total:,} shots in total -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
