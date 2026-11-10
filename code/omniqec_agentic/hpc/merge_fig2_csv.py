"""Merge every Fig.2 campaign's summary/progress CSV into one file each.

Usage:
    python -m omniqec_agentic.hpc.merge_fig2_csv [runs_dir] [--top N] [--no-refresh]

Writes three files per LLM model into <runs_dir> (the model is the campaign's
protocol_snapshot.json required_model, e.g. gpt-5.6-sol, claude-opus-4-1,
deepseek/deepseek-v4-pro -> slug deepseek-v4-pro):
  fig2_all_summary_<model>.csv   every measured code of every run (summary.csv rows)
  fig2_all_progress_<model>.csv  every turn of every run (progress.csv rows)
  fig2_all_best_<model>.csv      "what did each point find": per run the BB baseline
                                 row (rank 0) then the N best non-baseline codes
                                 (rank 1..N, default 5), same columns as summary.csv
                                 so `spec` says which code each row is.
Runs of different models never share a file, so the three arms' exports can sit
in one folder without overwriting each other; `--suffix none` writes the old
un-suffixed trio with every model in one file. Each row is prefixed with
arm/budget/site/model/complete so one file answers "how is everything doing"
across arms and budget points -- the per-run CSVs stay where they were, this
only reads them.

Each run's per-dir CSVs are re-exported first (subprocess, best effort) so the
merged view is always current; the exporters only read finished turn artifacts
and rewrite the two derived files summary.csv / progress.csv, so this is safe
while campaigns run. `--no-refresh` skips the re-export (tests, or a runs/ tree
copied off the cluster without its code).

`--site v36` merges only the run directories whose site/tag is exactly that
(the dashboard passes its --tag here): a canary or a variant chain left in the
same runs/ tree (`fig2_gptsub_b60_v36_canary2`, `..._v36x60`) is a different
experiment and must not sit in the formal tables next to the real point --
2026-09-12 the GPT b60 canary2 rows (site=v36_canary2) were merged into the
formal GPT file and looked like extra turns/codes of the formal run.

Columns are aligned BY NAME, not by position: progress.csv's header depends on
which code families a run has proposed so far (one mix_<family> column each),
and a run exported by an older exporter may lack a column. The merged header
is the union in first-seen order and missing cells are empty -- a point is
never dropped because its header differs. A run is skipped only when its CSV
is missing or has no data rows, and the stdout line says which and why (the
wrapup job discards stderr).
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

_RUN_RE = re.compile(r"^fig2_(?P<arm>.+)_b(?P<budget>\d+)_(?P<site>[^_]+.*)$")
PREFIX = ("arm", "budget", "site", "model", "complete")
DEFAULT_TOP = 5
UNKNOWN_MODEL = "unknown-model"

Rows = list[list[str]]
Selector = Callable[[list[str], Rows], Rows]


def _refresh(run_dir: Path) -> None:
    for module in ("omniqec_agentic.summary_csv", "omniqec_agentic.progress_csv"):
        subprocess.run(
            [sys.executable, "-m", module, str(run_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )


def _run_model(run_dir: Path) -> str:
    """The campaign's LLM model string (protocol_snapshot.json required_model)."""

    from omniqec_agentic.summary_csv import _campaign_model

    model = _campaign_model(run_dir)
    return model if model and model != "?" else ""


def model_slug(model: str) -> str:
    """File-name form of a model string: the segment after the provider slash,
    anything outside [A-Za-z0-9._-] folded to '-'. Empty/unknown -> unknown-model."""

    tail = model.rsplit("/", 1)[-1] if model else ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.")
    return slug or UNKNOWN_MODEL


def _read_csv(path: Path) -> tuple[list[str], Rows] | None:
    """Header + data rows, or None for a missing/empty file."""

    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return None
        return header, [row for row in reader if row]


def _align(header: list[str], file_header: list[str], rows: Rows) -> Rows:
    """Re-order one file's rows onto the merged header; absent columns read ""."""

    index = {name: i for i, name in enumerate(file_header)}
    return [
        [row[index[name]] if name in index and index[name] < len(row) else "" for name in header]
        for row in rows
    ]


def _plpr(header: list[str], row: list[str]) -> float:
    try:
        return float(row[header.index("plpr")])
    except (ValueError, IndexError):
        return float("inf")


def _best_rows(header: list[str], rows: Rows, top: int) -> Rows:
    """Baseline first (rank 0), then the `top` best other codes (rank 1..top).

    summary.csv is already plpr-sorted, but sort again: the ranking must not
    depend on how the file happened to be written. The baseline is the BB code
    re-measured every turn; listing it first is what makes vs_bb readable
    without opening the full summary.
    """

    pid = header.index("program_id")
    ordered = sorted(rows, key=lambda row: _plpr(header, row))
    baseline = [row for row in ordered if row[pid].startswith("baseline_")]
    codes = [row for row in ordered if not row[pid].startswith("baseline_")]
    out: Rows = []
    if baseline:
        out.append(["0"] + baseline[0])
    for rank, row in enumerate(codes[:top], start=1):
        out.append([str(rank)] + row)
    return out


def merge(runs_dir: Path, *, refresh: bool = True, top: int = DEFAULT_TOP,
          suffix: str = "model", site: str | None = None) -> None:
    """Merge the per-run CSVs under runs_dir; `site` keeps only runs whose
    site/tag equals it (None = every fig2_* run in the tree)."""

    if suffix not in ("model", "none"):
        raise ValueError(f"suffix must be 'model' or 'none', got {suffix!r}")
    found = [
        (m, d)
        for d in sorted(runs_dir.iterdir())
        if d.is_dir() and (m := _RUN_RE.match(d.name))
        and (site is None or m["site"] == site)
    ]
    # Numeric budget order within an arm; a lexical sort puts b120 before b60
    # and the merged file then reads as if a point were missing.
    found.sort(key=lambda pair: (pair[0]["arm"], int(pair[0]["budget"]), pair[0]["site"]))

    if refresh:
        for m, d in found:
            _refresh(d)

    models = {d: _run_model(d) for _, d in found}
    # One output trio per model slug (first-seen order); `none` = one trio for all.
    groups: dict[str, list[tuple[re.Match[str], Path]]] = {}
    for m, d in found:
        key = model_slug(models[d]) if suffix == "model" else ""
        groups.setdefault(key, []).append((m, d))
    if not groups:
        groups[""] = []
    for key, members in groups.items():
        _merge_group(runs_dir, members, models, top=top, name_suffix=f"_{key}" if key else "")


def _merge_group(runs_dir: Path, found: list[tuple[re.Match[str], Path]],
                 models: dict[Path, str], *, top: int, name_suffix: str) -> None:
    # (source per-run file, merged output name, extra prefix columns, needed columns, row selector)
    views: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...], Selector | None], ...] = (
        ("summary", "fig2_all_summary", (), (), None),
        ("progress", "fig2_all_progress", (), (), None),
        ("summary", "fig2_all_best", ("rank",), ("program_id", "plpr"),
         lambda header, rows: _best_rows(header, rows, top)),
    )
    for kind, out_name, extra, needed, select in views:
        out_path = runs_dir / f"{out_name}{name_suffix}.csv"
        loaded: list[tuple[re.Match[str], Path, list[str], Rows]] = []
        skipped: list[str] = []
        for m, d in found:
            got = _read_csv(d / f"{kind}.csv")
            if got is None:
                skipped.append(f"{d.name}: 缺 {kind}.csv")
                continue
            file_header, rows = got
            if not rows:
                skipped.append(f"{d.name}: {kind}.csv 无数据行")
                continue
            missing = [name for name in needed if name not in file_header]
            if missing:
                skipped.append(f"{d.name}: {kind}.csv 缺列 {'/'.join(missing)}")
                continue
            loaded.append((m, d, file_header, rows))
        if not loaded:
            print(f"{out_name}: nothing to merge", file=sys.stderr)
            continue

        # A per-run column that is also a prefix column (summary.csv carries
        # `model`) folds into the prefix instead of appearing twice.
        header: list[str] = []
        for _, _, file_header, _ in loaded:
            header.extend(name for name in file_header if name not in header and name not in PREFIX)
        rows_out: Rows = []
        for m, d, file_header, rows in loaded:
            aligned = _align(header, file_header, rows)
            if select is not None:
                aligned = select(header, aligned)
            prefix = [
                m["arm"], m["budget"], m["site"], models.get(d) or UNKNOWN_MODEL,
                "yes" if (d / "campaign_report.json").exists() else "no",
            ]
            rows_out.extend(prefix + row for row in aligned)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(PREFIX) + list(extra) + header)
            writer.writerows(rows_out)
        note = f"  (跳过 {len(skipped)}: {'; '.join(skipped)})" if skipped else ""
        print(f"{out_path}  {len(rows_out)} 行{note}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs_dir", nargs="?", type=Path, default=Path("runs"))
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"codes per point in fig2_all_best_<model>.csv (default {DEFAULT_TOP})")
    parser.add_argument("--no-refresh", action="store_true",
                        help="do not re-export each run's summary/progress.csv first")
    parser.add_argument("--suffix", choices=("model", "none"), default="model",
                        help="model: one fig2_all_*_<model>.csv trio per LLM model (default); "
                             "none: the un-suffixed trio with every model in one file")
    parser.add_argument("--site", default=None,
                        help="only runs whose site/tag is exactly this (e.g. v36); "
                             "default: every fig2_* run in the tree, canaries included")
    args = parser.parse_args(argv)
    if not args.runs_dir.is_dir():
        raise SystemExit(f"not a directory: {args.runs_dir}")
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    merge(args.runs_dir, refresh=not args.no_refresh, top=args.top, suffix=args.suffix,
          site=args.site)


if __name__ == "__main__":
    main()
