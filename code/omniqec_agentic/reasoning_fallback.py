"""The reasoning-saturation rule of the final protocols (2026-11-06, user decision; docs/final_protocol_20261106.md).

Every arm starts a figure at the model's default reasoning setting. When the first turn (turn 0) of a budget point
cannot be completed because the model spends its whole output budget on reasoning -- a generation whose every
completion attempt returns an empty answer after ``max_tokens`` output tokens -- the arm restarts every budget point
of that figure from scratch at the ``medium`` level (the CLIProxyAPI model-name suffix ``(medium)``). The rule reads
only whether the model produced an answer, never which codes it proposed or how they measured.

Where each part lives:
- the generator diagnoses a saturated generation of turn 0 (funsearch.py; SATURATION_MARKER in the strict-abort
  message);
- the launcher (hpc/nscc_claude_grid_v36.pbs, PRESET runs) raises the arm's flag (``raise``), moves the run aside
  (``retire``) and restarts the point at the flag's model, recording why in the new run's reasoning_fallback.json
  (``record``);
- a run of the same arm still at the default level stops at the start of its next turn
  (graph.orchestrator_turn -> check_fallback_request; FALLBACK_REQUEST_MARKER) and is restarted the same way;
- the validator (final profiles) requires the record on a fallback run and refuses a default run once the arm's
  flag exists.

The fallback is defined for the Claude arm only: the other two arms have no tested level suffix on their routes, so a
saturated start there stops the point for a human decision.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

FALLBACK_LEVEL = "medium"
FALLBACK_ARMS = ("claude",)
FLAG_ENV = "OMNIQEC_REASONING_FALLBACK_FLAG"
FLAG_SCHEMA = "reasoning_fallback_flag_v1"
RECORD_NAME = "reasoning_fallback.json"
RECORD_SCHEMA = "reasoning_fallback_record_v1"
RETIRED_DIRNAME = "aborted_reasoning"
SATURATION_MARKER = "reasoning saturated at the start of the run"
FALLBACK_REQUEST_MARKER = "reasoning fallback requested for this arm"

_SUFFIX = re.compile(r"^(?P<base>[^()]+)\((?P<level>[^()]*)\)$")


def effort_suffix(model: str) -> str | None:
    """The level in a ``model(level)`` name, else None."""

    match = _SUFFIX.match(str(model or "").strip())
    return match.group("level") if match else None


def base_model(model: str) -> str:
    match = _SUFFIX.match(str(model or "").strip())
    return match.group("base") if match else str(model or "").strip()


def fallback_model(model: str) -> str:
    return f"{base_model(model)}({FALLBACK_LEVEL})"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def saturated_attempt(row: Any) -> bool:
    """One completion attempt that produced no answer because the output budget went to reasoning: the call
    succeeded, its visible answer is empty, no program was parsed, and it used all ``max_tokens`` output tokens."""

    if not isinstance(row, dict) or row.get("status") != "ok":
        return False
    if int(row.get("proposal_count_parsed") or 0) != 0 or str(row.get("response") or "").strip():
        return False
    info = row.get("engine_info") or {}
    cap = int(info.get("max_tokens") or 0)
    used = int((info.get("usage") or {}).get("output_tokens") or 0)
    return cap > 0 and used >= cap


def saturated_generation(rows: Iterable[dict[str, Any]], *, attempts: int) -> bool:
    """Every one of a generation's ``attempts`` completion attempts was saturated."""

    rows = list(rows)
    return len(rows) >= int(attempts) and all(saturated_attempt(row) for row in rows)


def saturation_message(*, turn: int, gen: int, rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    cap = max((int((row.get("engine_info") or {}).get("max_tokens") or 0) for row in rows), default=0)
    return (
        f"{SATURATION_MARKER}: turn {int(turn)} generation {int(gen)}: {len(rows)} of {len(rows)} completion "
        f"attempts returned an empty answer after {cap} output tokens"
    )


def read_flag(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    flag = json.loads(path.read_text(encoding="utf-8"))
    if flag.get("schema") != FLAG_SCHEMA or not flag.get("from_model") or not flag.get("to_model"):
        raise ValueError(f"{path}: not a {FLAG_SCHEMA} flag")
    return flag


def check_fallback_request(protocol: dict[str, Any] | None) -> None:
    """Called at the start of every turn: a run of an arm whose flag was raised by another budget point stops here
    while it is still at the flag's ``from_model``."""

    flag = read_flag(os.environ.get(FLAG_ENV, "").strip() or None)
    if flag is None:
        return
    model = str((protocol or {}).get("required_model") or "")
    if model and model == str(flag["from_model"]):
        raise RuntimeError(
            f"{FALLBACK_REQUEST_MARKER}: {flag.get('trigger_point')} ({flag.get('trigger_run')}) moved the arm to "
            f"{flag['to_model']!r}; this run is frozen at {model!r} and restarts from scratch"
        )


def current_model(flag_path: str | Path | None, model: str) -> str:
    """The model this arm runs from now on: the flag's ``to_model`` once the flag exists, else ``model``."""

    flag = read_flag(flag_path)
    if flag is None:
        return str(model)
    if base_model(str(flag["from_model"])) != base_model(model):
        raise ValueError(f"flag model {flag['from_model']!r} is not this arm's model {model!r}")
    return str(flag["to_model"])


def frozen_model(run_dir: str | Path) -> str | None:
    snapshot = Path(run_dir) / "protocol_snapshot.json"
    if not snapshot.is_file():
        return None
    return str((json.loads(snapshot.read_text(encoding="utf-8")) or {}).get("required_model") or "") or None


def is_stale(run_dir: str | Path, flag_path: str | Path | None) -> bool:
    """A run frozen at the flag's ``from_model``: the default level of an arm that has switched."""

    flag = read_flag(flag_path)
    return flag is not None and frozen_model(run_dir) == str(flag["from_model"])


def saturation_evidence(run_dir: str | Path) -> list[dict[str, Any]]:
    """The turn-0 generations of a run whose completion attempts were all saturated."""

    from .funsearch import MAX_COMPLETION_ATTEMPTS

    out = []
    generations = Path(run_dir) / "turns" / "turn_0000" / "cycle" / "fast" / "generations"
    for raw in sorted(generations.glob("g*/llm_raw.jsonl")):
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        if saturated_generation(rows, attempts=MAX_COMPLETION_ATTEMPTS):
            out.append({
                "generation": raw.parent.name,
                "attempts": len(rows),
                "max_tokens": [int((row.get("engine_info") or {}).get("max_tokens") or 0) for row in rows],
                "output_tokens": [int(((row.get("engine_info") or {}).get("usage") or {}).get("output_tokens") or 0)
                                  for row in rows],
                "model": [str((row.get("engine_info") or {}).get("model") or "") for row in rows],
            })
    return out


def raise_flag(flag_path: str | Path, *, arm: str, run_tag: str, model: str, point: str, run_dir: str | Path,
               job: str, log_line: str = "") -> dict[str, Any]:
    """Raise the arm's flag once (the first budget point to trigger wins; later triggers read the existing flag).

    Raises SystemExit(3) when the rule has no lower level (another arm, or the model already carries a level) and
    SystemExit(4) when the run directory does not confirm the trigger."""

    if str(arm) not in FALLBACK_ARMS or effort_suffix(model) is not None:
        raise SystemExit(3)
    evidence = saturation_evidence(run_dir)
    if not evidence:
        raise SystemExit(4)
    existing = read_flag(flag_path)
    if existing is not None:
        return existing
    payload = {
        "schema": FLAG_SCHEMA,
        "arm": str(arm),
        "run_tag": str(run_tag),
        "from_model": str(model),
        "to_model": fallback_model(model),
        "level": FALLBACK_LEVEL,
        "trigger_point": str(point),
        "trigger_run": Path(run_dir).name,
        "trigger_job": str(job),
        "raised_at": _now(),
        "log_line": str(log_line),
        "evidence": evidence,
    }
    path = Path(flag_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return read_flag(path) or payload
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def retire(run_dir: str | Path, out_root: str | Path) -> Path:
    """Move a run aside under ``out_root/aborted_reasoning`` (kept for the record)."""

    run_dir = Path(run_dir)
    dest_root = Path(out_root) / RETIRED_DIRNAME
    dest_root.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = dest_root / f"{run_dir.name}.{stamp}"
    if dest.exists():
        raise FileExistsError(dest)
    shutil.move(str(run_dir), str(dest))
    return dest


def write_record(run_dir: str | Path, *, flag_path: str | Path, model: str, retired: str | None = None) -> dict | None:
    """reasoning_fallback.json of a run at the fallback level: the arm's flag and the run it replaced. A no-op for
    a run at the default level; an existing record is kept."""

    if effort_suffix(model) is None:
        return None
    flag = read_flag(flag_path)
    if flag is None or str(flag["to_model"]) != str(model):
        raise ValueError(f"run at {model!r} without a matching flag at {flag_path}")
    path = Path(run_dir) / RECORD_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    record = {
        "schema": RECORD_SCHEMA,
        "run": Path(run_dir).name,
        "model": str(model),
        "flag": flag,
        "replaces": str(retired) if retired else None,
        "written_at": _now(),
    }
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m omniqec_agentic.reasoning_fallback")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("model", help="print the model this arm runs from now on")
    p.add_argument("--flag", required=True)
    p.add_argument("--model", required=True)
    p = sub.add_parser("stale", help="exit 0 when RUN_DIR is frozen at the flag's from_model")
    p.add_argument("--flag", required=True)
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("raise", help="raise the arm's flag and print its to_model")
    for name in ("--flag", "--arm", "--run-tag", "--model", "--point", "--run-dir", "--job"):
        p.add_argument(name, required=True)
    p.add_argument("--log", default=None)
    p = sub.add_parser("retire", help="move RUN_DIR aside and print where")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-root", required=True)
    p = sub.add_parser("record", help="write reasoning_fallback.json for a run at the fallback level")
    for name in ("--flag", "--run-dir", "--model"):
        p.add_argument(name, required=True)
    p.add_argument("--retired", default="")
    args = parser.parse_args(argv)
    if args.cmd == "model":
        print(current_model(args.flag, args.model))
    elif args.cmd == "stale":
        return 0 if is_stale(args.run_dir, args.flag) else 1
    elif args.cmd == "raise":
        line = ""
        if args.log and Path(args.log).is_file():
            for text in Path(args.log).read_text(encoding="utf-8", errors="replace").splitlines():
                if SATURATION_MARKER in text:
                    line = text[text.index(SATURATION_MARKER):][:400]
                    break
        flag = raise_flag(args.flag, arm=args.arm, run_tag=args.run_tag, model=args.model, point=args.point,
                          run_dir=args.run_dir, job=args.job, log_line=line)
        print(flag["to_model"])
    elif args.cmd == "retire":
        print(retire(args.run_dir, args.out_root))
    elif args.cmd == "record":
        write_record(args.run_dir, flag_path=args.flag, model=args.model, retired=args.retired or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
