"""Best-code tables with the algebraic form of every code (2026-09-15).

Reads ``<export>/fig2_summary.csv`` and ``<export>/fig3_summary.csv`` (the
concatenated slow-loop measurements with the ``spec`` column), picks the top-5
distinct codes (d_search >= 5, lowest plpr) per budget point and arm, rebuilds
each code from its spec with the campaign's own family constructors so the
printed algebra is the code that was measured (the rebuilt CSS hash must equal
the CSV's ``code_hash8``), and writes

    <export>/fig2_best.md   one table per budget point (Arm | rank | [[n,k,d]] |
                            grammar | class | G | A | B | w | plpr | 95% CI | hash)
    <export>/fig3_best.md   one table per platform tag
    (the <sub>_best.csv files are left untouched)

usage: PYTHONPATH=. python omniqec_agentic/hpc/best_code_tables.py runs_export/<dir>
       PYTHONPATH=. python omniqec_agentic/hpc/best_code_tables.py --brute runs_export/brute_b48_<date>
         (brute-force search exports: writes top5_lowest_plpr_algebra.csv and
          top_kd2_vs_plpr_algebra.csv, the source csv files are left untouched)
"""

from __future__ import annotations

import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from omniqec_agentic.construction.build import build_compiled_code

ARM_NAME = {"claudesub": "Claude", "gptsub": "GPT", "deepseek": "DeepSeek"}
ARM_ORDER = {"Claude": 0, "GPT": 1, "DeepSeek": 2}
TOP = 5


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- algebra text
def _term(exps, variables) -> str:
    parts = []
    for e, v in zip(exps, variables):
        e = int(e)
        if e == 0:
            continue
        parts.append(v if e == 1 else f"{v}^{e}")
    return " ".join(parts) if parts else "1"


def _poly(elements, variables) -> str:
    return " + ".join(_term(e, variables) for e in elements)


def _matrix_or_poly(cells, variables) -> tuple[str, str]:
    """(text, shape) for an lp A/B block: a 1x1 block prints as its polynomial,
    a bigger block as [[p11, p12]; [p21, p22]]."""
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    if rows == 1 and cols == 1:
        return _poly(cells[0][0], variables), "1x1"
    text = "; ".join("[" + ", ".join(_poly(cell, variables) for cell in row) + "]" for row in cells)
    return f"[{text}]", f"{rows}x{cols}"


def _is_ring(matrix) -> bool:
    m = np.asarray(matrix, dtype=int) % 2
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    n = m.shape[0]
    return all(int(m[i].sum()) == 2 and m[i, i] == 1 and m[i, (i + 1) % n] == 1 for i in range(n))


def algebra(spec: dict) -> dict:
    fam = spec.get("family")
    if fam == "bb":
        ell, m = int(spec["ell"]), int(spec["m"])
        a = _poly(spec["A"], ("x", "y"))
        b = _poly(spec["B"], ("x", "y"))
        if math.gcd(ell, m) == 1:
            cls = f"BB, (l,m)=({ell},{m}) coprime, G ≅ C{ell * m}"
        else:
            cls = f"BB, (l,m)=({ell},{m})"
        return {"grammar": "bb", "class": cls, "G": f"C{ell}xC{m}", "A": a, "B": b}
    if fam == "lp_abelian":
        factors = [int(q) for q in (spec.get("group") or {}).get("factors", [])]
        variables = ("x", "y", "z", "u")[: len(factors)]
        g = "x".join(f"C{q}" for q in factors)
        a, sa = _matrix_or_poly(spec["A"], variables)
        b, sb = _matrix_or_poly(spec["B"], variables)
        if sa == "1x1" and sb == "1x1":
            cls = "2BGA (= GB)" if len(factors) == 1 else "2BGA (abelian G)"
        else:
            cls = f"LP {sa.replace('x', '×')}/{sb.replace('x', '×')}"
        return {"grammar": "lp_abelian", "class": cls, "G": g, "A": a, "B": b}
    if fam == "lp_nonabelian":
        g = str(spec.get("group"))
        variables = ("r", "s", "t", "u")
        a, sa = _matrix_or_poly(spec["A"], variables)
        b, sb = _matrix_or_poly(spec["B"], variables)
        cls = "2BGA (non-abelian)" if sa == "1x1" and sb == "1x1" else f"LP {sa.replace('x', '×')}/{sb.replace('x', '×')} (non-abelian)"
        return {"grammar": "lp_nonabelian", "class": cls, "G": g, "A": a, "B": b}
    if fam == "hgp":
        seeds = spec.get("seeds") or {}
        h1, h2 = np.asarray(seeds["H1"], dtype=int), np.asarray(seeds["H2"], dtype=int)

        def seed_text(name, h):
            if _is_ring(h):
                return f"{name} = ring({h.shape[0]})"
            return f"{name} = {h.shape[0]}x{h.shape[1]} (row w {int(h.sum(axis=1).max())})"

        if _is_ring(h1) and _is_ring(h2):
            cls = f"toric {h1.shape[0]}x{h2.shape[0]}"
        else:
            cls = f"HGP {h1.shape[0]}x{h1.shape[1]} / {h2.shape[0]}x{h2.shape[1]}"
        return {"grammar": "hgp", "class": cls, "G": "—", "A": seed_text("H1", h1), "B": seed_text("H2", h2)}
    return {"grammar": str(fam), "class": "?", "G": "?", "A": "?", "B": "?"}


def rebuild(spec: dict) -> dict:
    code = build_compiled_code(spec)
    hx, hz = np.asarray(code.hx), np.asarray(code.hz)
    return {
        "hash8": code.code_hash[:8],
        "w_x": int(hx.sum(axis=1).max()),
        "w_z": int(hz.sum(axis=1).max()),
        "n": int(hx.shape[1]),
    }


# ---------------------------------------------------------------- selection
def select_best(summary_csv: Path):
    rows = list(csv.DictReader(open(summary_csv, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        if _f(r.get("plpr")) is None or (_f(r.get("d_search_proxy")) or 0) < 5:
            continue
        by[(int(r["point"]), r["tag"], r["arm"])].append(r)
    picked = {}
    for key, rs in by.items():
        rs.sort(key=lambda r: _f(r["plpr"]))
        seen, top = set(), []
        for r in rs:
            h = r.get("code_hash8") or r.get("program_id")
            if h in seen:
                continue
            seen.add(h)
            top.append(r)
            if len(top) == TOP:
                break
        picked[key] = top
    # progress per run (turns done / finished) from the summary's run column is not available; the
    # caller passes it via runs_meta
    return picked


def runs_meta(export: Path, sub: str) -> dict:
    meta = {}
    for d in sorted((export / sub).glob("fig2_*")):
        parts = d.name.split("_")
        arm, pt, tag = ARM_NAME.get(parts[1], parts[1]), int(parts[2][1:]), "_".join(parts[3:])
        prog = d / "progress.csv"
        turns = (sum(1 for _ in open(prog)) - 1) if prog.is_file() and prog.stat().st_size > 0 else 0
        meta[(pt, tag, arm)] = {"turns": turns, "done": (d / "campaign_report.json").is_file()}
    return meta


def build_tables(export: Path, sub: str, title: str, section_by: str):
    summary = export / f"{sub}_summary.csv"
    if not summary.is_file():
        print(f"{summary}: missing")
        return
    picked = select_best(summary)
    meta = runs_meta(export, sub)
    ov = export / "spec_overrides.json"
    overrides = json.loads(ov.read_text(encoding="utf-8")) if ov.is_file() else {}
    out_rows = []
    for (pt, tag, arm), top in picked.items():
        for rank, r in enumerate(top, 1):
            spec = json.loads(r["spec"]) if r.get("spec") else {}
            if not spec:
                # a summary row whose spec column is empty (same-name candidate overwrite,
                # see memory fig2-final-stage-candidate-overwrite): take it from
                # <export>/spec_overrides.json {code_hash8: spec} when provided
                spec = overrides.get((r.get("code_hash8") or "")[:8], {})
            alg = algebra(spec) if spec else {"grammar": r["family"], "class": "?", "G": "?", "A": "?", "B": "?"}
            try:
                rb = rebuild(spec) if spec else {}
            except Exception as exc:  # noqa: BLE001 - report, do not abort the table
                rb = {"error": str(exc)[:60]}
            hash_ok = bool(rb.get("hash8")) and rb["hash8"] == (r.get("code_hash8") or "")[:8]
            # Fig.3 ion runs may apply a candidate-level H mask that enters the recorded hash;
            # the CSS matrices themselves still rebuild from the spec
            ion_mask = (not hash_ok) and bool(spec) and "ion" in str(tag)
            out_rows.append(
                {
                    "point": pt, "tag": tag, "arm": arm, "rank": rank,
                    "n": r["n_data"], "k": r["k"], "d": r["d_search_proxy"], "grammar": alg["grammar"],
                    "class": alg["class"], "G": alg["G"], "A": alg["A"], "B": alg["B"],
                    "w": max(rb.get("w_x") or 0, rb.get("w_z") or 0) or "",
                    "w_x": rb.get("w_x", ""), "w_z": rb.get("w_z", ""),
                    "plpr": r["plpr"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "hash": (r.get("code_hash8") or "")[:8],
                    "hash_ok": "yes" if hash_ok else ("ion H-mask" if ion_mask else ("no" if spec else "no spec")),
                    "turn": r["turn"], "program_id": r["program_id"], "rebuild_error": rb.get("error", ""),
                    "turns_done": meta.get((pt, tag, arm), {}).get("turns", ""),
                    "finished": "yes" if meta.get((pt, tag, arm), {}).get("done") else "",
                }
            )
    out_rows.sort(key=lambda x: (x["point"], x["tag"], ARM_ORDER.get(x["arm"], 9), x["rank"]))
    cols = ["point", "tag", "arm", "rank", "n", "k", "d", "grammar", "class", "G", "A", "B", "w", "w_x", "w_z",
            "plpr", "ci_low", "ci_high", "hash", "hash_ok", "turn", "program_id", "turns_done", "finished", "rebuild_error"]
    # the <sub>_best.csv files are owned by the export step (plain columns); this script only writes the .md
    # markdown: one table per section (budget point for fig2, platform tag for fig3)
    lines = [f"# {title}", "",
             "Every code is rebuilt from its `spec` with the campaign's family constructors; `hash` is the CSV's "
             "`code_hash8` and a row is only listed as verified when the rebuilt CSS hash matches it (all rows "
             "below match unless marked ✗). `w` = max check weight of the rebuilt HX/HZ; `d` = d_search proxy; "
             "top-5 distinct codes per arm with d ≥ 5, lowest plpr first.", ""]
    sections = defaultdict(list)
    for x in out_rows:
        sections[x["point"] if section_by == "point" else x["tag"]].append(x)
    for key in sorted(sections, key=lambda k: (k if isinstance(k, int) else 0, str(k))):
        xs = sections[key]
        prog = "; ".join(
            f"{arm} {meta.get((xs[0]['point'], xs[0]['tag'], arm), {}).get('turns', 0)} turns"
            + (" (done)" if meta.get((xs[0]['point'], xs[0]['tag'], arm), {}).get("done") else "")
            for arm in ("Claude", "GPT", "DeepSeek") if (xs[0]["point"], xs[0]["tag"], arm) in meta
        )
        head = f"N_total budget {key}" if section_by == "point" else f"{key} (budget {xs[0]['point']})"
        lines += [f"### {head}, top-{TOP} per arm — {prog}", "",
                  "| Arm | rank | [[n,k,d]] | grammar | class | G | A | B | w | plpr | 95% CI | turn | hash |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in xs:
            mark = "" if x["hash_ok"] == "yes" else (" (H-mask)" if x["hash_ok"] == "ion H-mask" else " ✗")
            lines.append(
                f"| {x['arm']} | {x['rank']} | [[{x['n']},{x['k']},{x['d']}]] | {x['grammar']} | {x['class']} | {x['G']} | "
                f"{x['A']} | {x['B']} | {x['w']} | {_f(x['plpr']):.2e} | [{_f(x['ci_low']):.2e}, {_f(x['ci_high']):.2e}] | "
                f"{x['turn']} | {x['hash']}{mark} |"
            )
        lines.append("")
    (export / f"{sub}_best.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = [x for x in out_rows if x["hash_ok"] != "yes"]
    print(f"{sub}: {len(out_rows)} rows, {len(sections)} sections, hash mismatches/unverified: {len(bad)}")
    for x in bad[:10]:
        print("   ", x["point"], x["arm"], x["rank"], x["program_id"], x["hash"], x.get("rebuild_error"))


# ---------------------------------------------------------------- brute-force exports
BRUTE_TABLES = ("top5_lowest_plpr", "top_kd2_vs_plpr")
BRUTE_COLS = ["rank", "program_id", "code", "family", "grammar", "class", "G", "A", "B", "|A|", "|B|",
              "w_x", "w_z", "kd2", "plpr", "ci_low", "ci_high", "plpr_over_best", "failures_x", "failures_z",
              "shots_x", "shots_z", "code_hash8", "hash_check", "spec"]


def _brute_algebra(spec: dict) -> dict:
    alg = algebra(spec)
    if spec.get("family") == "bb" and int(spec["ell"]) == 1:
        # l = 1: x is trivial, the code is a two-block group-algebra code over the cyclic group C_m
        m = int(spec["m"])
        alg["class"] = f"BB with l=1 (univariate), G = C{m}"
        alg["G"] = f"C{m}"
    return alg


def brute_algebra_csvs(out_dir: Path) -> None:
    """Writes <table>_algebra.csv next to top5_lowest_plpr.csv / top_kd2_vs_plpr.csv of a
    hpc/brute_force_budget.py run (the source csv files are left untouched)."""

    for name in BRUTE_TABLES:
        src = out_dir / f"{name}.csv"
        if not src.exists():
            continue
        rows_out = []
        with src.open(encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh), 1):
                spec = json.loads(row["spec"])
                alg = _brute_algebra(spec)
                built = rebuild(spec)
                d = row.get("d_exact") or row.get("d_search")
                if spec.get("family") == "bb":
                    wa, wb = len(spec["A"]), len(spec["B"])
                else:
                    wa = sum(len(cell) for r in spec["A"] for cell in r)
                    wb = sum(len(cell) for r in spec["B"] for cell in r)
                rows_out.append({
                    "rank": i, "program_id": row["program_id"], "code": f"[[{row['n']},{row['k']},{d}]]",
                    "family": row["family"], **alg, "|A|": wa, "|B|": wb, "w_x": built["w_x"], "w_z": built["w_z"],
                    "kd2": row["kd2"], "plpr": row["plpr"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                    "plpr_over_best": row.get("plpr_over_best", ""), "failures_x": row["failures_x"],
                    "failures_z": row["failures_z"], "shots_x": row["shots_x"], "shots_z": row["shots_z"],
                    "code_hash8": row["code_hash8"],
                    "hash_check": "ok" if built["hash8"] == row["code_hash8"] else f"MISMATCH rebuilt {built['hash8']}",
                    "spec": row["spec"],
                })
        dst = out_dir / f"{name}_algebra.csv"
        with dst.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=BRUTE_COLS)
            writer.writeheader()
            writer.writerows(rows_out)
        bad = [r["program_id"] for r in rows_out if r["hash_check"] != "ok"]
        print(f"{dst}: {len(rows_out)} rows, hash mismatches: {bad or 'none'}")


def main(argv: list[str]) -> int:
    if len(argv) > 2 and argv[1] == "--brute":
        brute_algebra_csvs(Path(argv[2]).resolve())
        return 0
    export = Path(argv[1] if len(argv) > 1 else "runs_export").resolve()
    build_tables(export, "fig2", "Fig.2 (telescoping_structural_v3_7, RUN_TAG v37): top codes per budget point and arm, with their algebraic form", "point")
    build_tables(export, "fig3", "Fig.3 b98 with Claude Fable 5.1 (protocol variant, 20 turns): top codes per platform, with their algebraic form", "tag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
