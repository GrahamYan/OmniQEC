"""Campaign leaderboard export: one row per code (its BEST measurement),
sorted by plpr — the quick-look table the full per-stage slow_results.csv
(50+ protocol columns) is too heavy for.

Usage:
    python -m omniqec_agentic.summary_csv <run_dir> [-o out.csv]

Re-run any time (reads finished artifacts only; safe while a campaign runs).
Columns (user-specified 2026-08-17): turn, program_id, family, n, k, d,
plpr, ci_low, ci_high, vs_bb, block_ler, failures, shots, p, rounds,
decoder, code_hash8 — plus `spec` (2026-09-05): the code's constructive
parameters as compact JSON, so a row says WHICH code it is and not just a
hash (BB `A`/`B` polynomials + `ell`/`m`; HGP seed matrices; LP group + lists).
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(2**31 - 1)  # slow_results rows may carry long failure_reason / spec fields
import json
import re
from pathlib import Path

# `d_search_proxy` is named for what it is: the heuristic 4-seed search
# distance, never a certified one. A column called `d` next to n and k reads as
# an [[n,k,d]] claim to anyone opening the file.
SUMMARY_COLUMNS = (
    "turn", "program_id", "family", "n_data", "n_checks_x", "n_checks_z", "k",
    "d_search_proxy", "max_check_weight", "selection_channels",
    "plpr", "ci_low", "ci_high", "vs_bb",
    "block_ler", "failures", "shots", "p", "rounds",
    "decoder", "model", "code_hash8",
    # 2026-09-10: tokens the campaign had spent when this row was measured --
    # tok_turn = the measuring turn's billable tokens (input + output, from
    # its token_usage.json), tok_cum = the campaign total through that turn.
    # Final-stage rows carry the campaign total as tok_cum and no tok_turn.
    "tok_turn", "tok_cum",
    "spec",
)

_TURN_RE = re.compile(r"turn_(\d+)")


def _turn_label(path: Path) -> str:
    match = _TURN_RE.search(str(path))
    if match:
        return str(int(match.group(1)))
    return "final" if "final_slow" in str(path) else "?"


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _campaign_model(run_dir: Path) -> str:
    """LLM model used by this campaign (a per-run constant, e.g. gpt-5.6-sol).

    The authority is protocol_snapshot.json's required_model — the value the
    strict audit pins every recorded call against. run_config.json's llm block
    is NOT trustworthy for this: it serializes the LLMConfig defaults object,
    which carries DeepSeek's env names and "model": "deepseek-chat" regardless
    of the provider actually in use (GPT/Claude arms exported deepseek-chat in
    every summary.csv until 2026-08-31). It remains only as the fallback for
    pre-snapshot runs whose protocol has no required_model.
    """

    try:
        snapshot = json.loads(
            (run_dir / "protocol_snapshot.json").read_text(encoding="utf-8")
        )
        model = snapshot.get("required_model")
        if model:
            return str(model)
    except Exception:
        pass
    for cfg in sorted(run_dir.glob("turns/*/cycle/fast/run_config.json")):
        try:
            llm = json.loads(cfg.read_text(encoding="utf-8")).get("llm") or {}
            return str(llm.get("model") or llm.get("provider") or "?")
        except Exception:
            continue
    return "?"


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _artifact_hash(data: dict) -> str:
    """The code_hash a candidate artifact says it belongs to ("" for legacy files)."""

    for block in ("structural", "promotion_record"):
        value = data.get(block)
        if isinstance(value, dict) and value.get("code_hash"):
            return str(value["code_hash"])
    return ""


def _spec_json(data: dict | None) -> str:
    if data is None or not isinstance(data.get("spec"), dict):
        return ""
    return json.dumps(data["spec"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _SpecIndex:
    """Finds the constructive `spec` of a measured code.

    The slow queue writes every measured candidate's artifact next to its
    slow_results.csv (evaluation/pool.py: <out_dir>/candidates/<program_id>.json)
    and that artifact's `spec` block is the seed-grammar record the code was
    built from. The CSV row itself carries only family/n/k/hash, which is why
    the leaderboard could not say what a code IS until this column existed.

    Two traps, both handled here:
    * The artifact is found relative to the CSV, not via the row's
      `artifact_path`: that path is relative to the repo root the campaign ran
      in and breaks once a run directory is rsynced elsewhere.
    * program_ids restart every turn (funsearch: g<gen>_p<idx>), so two
      DIFFERENT codes can share one id. Per-turn stage dirs each have their own
      candidates/, but the final stage pools finalists from many turns into one
      final_slow/T<r>/ dir, where the second write overwrites the first and both
      CSV rows then point at one file. The artifact therefore has to prove it is
      the row's code (structural/promotion_record code_hash); on a mismatch the
      spec is looked up run-wide by code_hash -- the losing finalist still has
      its own turn-stage artifact. Legacy artifacts without a hash are accepted.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._by_path: dict[Path, dict | None] = {}
        self._by_hash: dict[str, str] | None = None

    def _artifact(self, path: Path) -> dict | None:
        if path not in self._by_path:
            self._by_path[path] = _load_json(path)
        return self._by_path[path]

    def _run_wide(self, code_hash: str) -> str:
        if self._by_hash is None:
            index: dict[str, str] = {}
            for path in sorted(self._run_dir.rglob("candidates/*.json")):
                data = self._artifact(path)
                if data is None:
                    continue
                digest, spec = _artifact_hash(data), _spec_json(data)
                if digest and spec and digest not in index:
                    index[digest] = spec
            self._by_hash = index
        return self._by_hash.get(code_hash, "")

    def lookup(self, csv_path: Path, program_id: str, code_hash: str) -> str:
        if not program_id:
            return ""
        data = self._artifact(csv_path.parent / "candidates" / f"{program_id}.json")
        if data is not None:
            digest = _artifact_hash(data)
            if not digest or not code_hash or digest == code_hash:
                return _spec_json(data)
        return self._run_wide(code_hash) if code_hash else ""


def _tokens_by_turn(run_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    """turn -> billable tokens of that turn, and turn -> running campaign total."""

    per_turn: dict[int, int] = {}
    for path in sorted(run_dir.glob("turns/turn_*/token_usage.json")):
        match = _TURN_RE.search(str(path.parent))
        data = _load_json(path)
        if match is None or data is None:
            continue
        per_turn[int(match.group(1))] = int((data.get("total") or {}).get("total_tokens") or 0)
    cumulative: dict[int, int] = {}
    running = 0
    for turn in sorted(per_turn):
        running += per_turn[turn]
        cumulative[turn] = running
    return per_turn, cumulative


def collect_best_rows(run_dir: Path) -> list[dict[str, object]]:
    model = _campaign_model(run_dir)
    best: dict[str, dict[str, object]] = {}
    specs = _SpecIndex(run_dir)
    tok_turn, tok_cum = _tokens_by_turn(run_dir)
    tok_campaign = max(tok_cum.values()) if tok_cum else None
    for csv_path in sorted(run_dir.rglob("slow_results.csv")):
        turn = _turn_label(csv_path)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "ok":
                    continue
                plpr = _float(raw.get("per_logical_per_round"))
                if plpr is None:
                    continue
                program_id = str(raw.get("program_id") or "")
                real_hash = str(raw.get("code_hash") or "")
                code_hash = real_hash or program_id
                row = {
                    "turn": turn,
                    "program_id": raw.get("program_id"),
                    "family": raw.get("family"),
                    "n_data": raw.get("n_data"),
                    "n_checks_x": raw.get("n_checks_x"),
                    "n_checks_z": raw.get("n_checks_z"),
                    "k": raw.get("k"),
                    "d_search_proxy": raw.get("d_search") or raw.get("d_proxy"),
                    "max_check_weight": raw.get("max_check_weight"),
                    "selection_channels": raw.get("selection_channels"),
                    "plpr": plpr,
                    "ci_low": _float(raw.get("per_logical_per_round_ci_low")),
                    "ci_high": _float(raw.get("per_logical_per_round_ci_high")),
                    "vs_bb": None,  # filled after the baseline best is known
                    "block_ler": _float(raw.get("block_ler")),
                    "failures": raw.get("total_failures"),
                    "shots": raw.get("total_shots"),
                    "p": raw.get("p") or raw.get("noise_value"),
                    "rounds": raw.get("rounds"),
                    "decoder": raw.get("decoder"),
                    "model": model,
                    "code_hash8": str(code_hash)[:8],
                    "tok_turn": tok_turn.get(int(turn)) if turn.isdigit() else None,
                    "tok_cum": tok_cum.get(int(turn)) if turn.isdigit() else tok_campaign,
                    "spec": specs.lookup(csv_path, program_id, real_hash),
                }
                held = best.get(str(code_hash))
                if held is None or float(row["plpr"]) < float(held["plpr"]):
                    best[str(code_hash)] = row

    rows = sorted(best.values(), key=lambda r: float(r["plpr"]))
    baseline = [r for r in rows if str(r["program_id"] or "").startswith("baseline_")]
    bb_plpr = float(baseline[0]["plpr"]) if baseline else None
    for row in rows:
        if bb_plpr:
            row["vs_bb"] = round(float(row["plpr"]) / bb_plpr, 3)
    return rows


def write_summary_csv(run_dir: Path, out: Path | None = None) -> tuple[Path, int]:
    """Export <run_dir>/summary.csv atomically (tmp + rename). Called by the
    campaign after every turn and after the final stage (workflow.graph)."""

    run_dir = Path(run_dir)
    rows = collect_best_rows(run_dir)
    out = Path(out) if out is not None else (run_dir / "summary.csv")
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(out)
    return out, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args()
    rows = collect_best_rows(args.run_dir)
    out = args.out or (args.run_dir / "summary.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"{len(rows)} codes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
