"""Per-turn reflection (Reflector role) — the qualitative memory channel.

The ledger already carries quantitative learning: a claim becomes a rule only
after the deterministic backtester rules on generations that postdate it. That
channel is deliberately slow and cannot hold the things a turn teaches that are
not yet claims — a direction that stopped paying, a construction that behaved
unlike its neighbours, a suspicion worth testing later.

This module fills that gap without weakening the other one. A reflection is
advice, never evidence: it is stored apart from the evidence rows, it is shown
to later turns as prose, and it expires. A quantitative regularity found here
still has to go through `ledger_propose` and survive prospective backtesting
before anything may cite it.

Expiry is the whole design. Reflections are rendered under a decaying token
budget (`render_reflections`): the newest few in full, older ones down to a
single line, and past the window nothing at all. An observation still worth
having after that window should have become a ledger entry; if it did not, it
was never load-bearing. Without decay this file is where context collapse
starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omniqec_agentic.config import LLMConfig

from .prompts import load_prompt

# Caps are applied when the reflection is written, not when it is rendered, so
# a verbose model cannot grow every later prompt.
MAX_ITEMS = 3
MAX_ITEM_CHARS = 200
MAX_ADVICE_CHARS = 400
LIST_FIELDS = ("what_worked", "what_failed", "surprises", "candidate_hypotheses")

# Rendering window: full text for the newest, one advice line for the rest.
FULL_WINDOW = 3
LINE_WINDOW = 10

# The reflection contract caps its own output at roughly 900 tokens (3 items x
# 200 chars per list, plus a 400-char advice line). Asking for the campaign's
# full ceiling on top of that is not free: OpenRouter reserves credit against
# `max_tokens`, not against what a call actually uses, so a 64,000-token
# request was refused with HTTP 402 on a key that could still afford 49,543 —
# for a call that would have spent about 1,800. Headroom for a thinking
# model's preamble, and no more.
REFLECTION_MAX_TOKENS = 8000

# Paper alignment (2026-11-10, appendix B4): "At the end of each iteration ...
# the same LLM backend is called to review the completed search." A review is
# part of every iteration, so a failed call is retried -- bounded, with
# backoff -- instead of being skipped after the client's own short transport
# retries. The archived runs show why one try was not enough: proxy
# auth_unavailable 503s, 525 / 1016 gateway errors, connection resets and
# timeouts (78 of 2236 turns failed; 17 more returned no JSON object). After
# the last attempt the turn keeps a FAILED record (reflection.json,
# reflection_call.jsonl with one row per attempt, the strict turn audit and the
# campaign audit summary); the measurements of the turn are never touched.
# Frozen in the protocol snapshot as `reflection_policy` (reflection_policy()).
REFLECTION_ATTEMPTS = 3
REFLECTION_BACKOFF_S: tuple[float, ...] = (60.0, 300.0)
# No new attempt starts once this much wall time has passed since the first
# one (a timed-out call alone can take the client's 900 s deadline).
REFLECTION_WALL_BUDGET_S = 1800.0
REFLECTION_POLICY_ID = "bounded_retry_v1"

REFLECTIONS_FILENAME = "reflections.jsonl"


def reflection_policy() -> dict[str, Any]:
    """The frozen retry policy of the per-iteration review (protocol snapshot key)."""

    return {
        "policy_id": REFLECTION_POLICY_ID,
        "attempts": int(REFLECTION_ATTEMPTS),
        "backoff_s": [float(value) for value in REFLECTION_BACKOFF_S],
        "wall_budget_s": float(REFLECTION_WALL_BUDGET_S),
        "max_tokens": int(REFLECTION_MAX_TOKENS),
        # A reply without any JSON object is a failed attempt, not an empty review.
        "unparseable_is_failure": True,
    }


def _clip_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:MAX_ITEM_CHARS])
        if len(out) >= MAX_ITEMS:
            break
    return out


def normalize_reflection(payload: Any, *, turn: int) -> dict[str, Any]:
    """Coerce a model reply into the fixed-size record that gets stored."""

    source = payload if isinstance(payload, dict) else {}
    record: dict[str, Any] = {"schema": "turn_reflection", "turn": int(turn)}
    for field in LIST_FIELDS:
        record[field] = _clip_list(source.get(field))
    advice = source.get("next_turn_advice")
    record["next_turn_advice"] = str(advice).strip()[:MAX_ADVICE_CHARS] if advice else ""
    return record


# Fields that explain why a measured code did or did not reach the frontier.
# Without them the hall of fame is a filtered list with the filter invisible,
# and the reflector reads a deliberate fault-tolerance exclusion as a
# bookkeeping bug (observed on the first live run: the turn's lowest plpr came
# from a code whose verified circuit distance was 2, correctly barred).
ADMISSION_FIELDS = (
    "program_id", "family", "per_logical_per_round", "d_proxy",
    "verified_circuit_distance", "degraded_ext", "stopped_by", "status",
)


def measured_rows(slow_rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Every measurement this turn, admitted or not, with the deciding fields."""

    out: list[dict[str, Any]] = []
    for row in slow_rows or []:
        if not isinstance(row, dict) or row.get("is_baseline"):
            continue
        out.append({key: row.get(key) for key in ADMISSION_FIELDS})
    return out


def reflection_input(
    *,
    turn: int,
    digest: dict[str, Any],
    plan: dict[str, Any] | None,
    coverage: dict[str, Any] | None,
    slow_rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """What the reflector sees: this turn's outcome, its plan, and the gaps.

    ``extra`` (Fig.3 Phase B): the task name and the not-competitive tables the
    generator is shown, so a note can point at them. Its ``fast_screening``
    entry (paper B4 alignment, 2026-11-10) becomes the top-level
    ``fast_loop_screening`` block: the iteration's screening results.

    Specs are dropped — the reflector reasons about regions and outcomes, and
    a full construction spec per row would crowd out the rows themselves.
    """

    slow_top = []
    for row in list(digest.get("slow_top_k") or [])[:8]:
        if isinstance(row, dict):
            slow_top.append({k: v for k, v in row.items() if k != "spec"})
    baseline = digest.get("baseline") or {}
    payload = {
        "turn": int(turn),
        "cycle_status": digest.get("cycle_status"),
        "runtime_hall_of_fame": slow_top,
        # The structural ledger travels with the runtime one (v3.2 §5.2):
        # shown only one of them, the reflector describes the campaign as if
        # the other did not exist, and its advice reaches the planner.
        "structural_frontier": list(digest.get("structural_frontier") or [])[:16],
        "baseline": {k: v for k, v in baseline.items() if k != "spec"} if baseline else None,
        "fast_top_k": list(digest.get("fast_top_k") or [])[:8],
        # The hall of fame above is FILTERED (d < 3 is barred as unable to
        # correct a single fault). These are the raw measurements, so an
        # absence there is explainable rather than mysterious.
        "measured_this_turn": measured_rows(slow_rows),
        # Fig.3 v2: why the slow queue measured what it measured (slots
        # filled, niche winners that were already-measured codes or their
        # isomorphic twins, candidates over the check-weight gate). A turn
        # that measured nothing is then "nothing new was proposed", not
        # "the direction failed".
        "slow_queue": digest.get("slow_queue"),
        "admission_rule": (
            "two independent ledgers: structural retention keeps the "
            "(k, d_search) Pareto elites and reads no runtime quantity; the "
            "runtime hall of fame admits a measured candidate only if "
            "d_search >= 3 AND verified_circuit_distance >= 3, and a lower "
            "plpr does not override that floor"
        ),
        "learning": digest.get("learning") or {},
        "plan_that_produced_this": {
            key: (plan or {}).get(key)
            for key in ("family_mix", "strategy_mix", "directives",
                        "explore_temperature", "proposal_count")
        },
        "structural_coverage": coverage or {},
    }
    if extra:
        extra = dict(extra)
        # Paper B4 (2026-11-10 alignment): the review also receives the
        # fast-loop screening results of the iteration.
        screening = extra.pop("fast_screening", None)
        if screening:
            payload["fast_loop_screening"] = screening
        if extra:
            payload["task_evidence"] = extra
    return json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, default=str)


def append_reflection(manual_dir: Path, record: dict[str, Any]) -> Path:
    """Append to the campaign-wide reflection log (one line per turn).

    Idempotent under crash-resume: a turn already present is rewritten in
    place rather than duplicated, so a re-entered reconcile cannot stack two
    reflections for the same turn.
    """

    path = Path(manual_dir) / "dynamic" / REFLECTIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_reflections(manual_dir) if int(row.get("turn", -1)) != int(record["turn"])]
    rows.append(record)
    rows.sort(key=lambda row: int(row.get("turn", -1)))
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def read_reflections(manual_dir: Path) -> list[dict[str, Any]]:
    path = Path(manual_dir) / "dynamic" / REFLECTIONS_FILENAME
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def render_reflections(
    rows: list[dict[str, Any]],
    *,
    full_window: int = FULL_WINDOW,
    line_window: int = LINE_WINDOW,
) -> str:
    """Decaying render: newest in full, then advice-only, then nothing."""

    ordered = sorted(rows, key=lambda row: int(row.get("turn", -1)), reverse=True)
    if not ordered:
        return "(no reflections yet)"
    lines: list[str] = []
    for index, row in enumerate(ordered[:line_window]):
        turn = row.get("turn")
        if index < full_window:
            lines.append(f"turn {turn}:")
            for field in LIST_FIELDS:
                items = list(row.get(field) or [])
                if items:
                    lines.append(f"  {field}:")
                    lines.extend(f"    - {item}" for item in items)
            advice = str(row.get("next_turn_advice") or "").strip()
            if advice:
                lines.append(f"  advice: {advice}")
        else:
            advice = str(row.get("next_turn_advice") or "").strip()
            lines.append(f"turn {turn}: {advice}" if advice else f"turn {turn}: (nothing recorded)")
    return "\n".join(lines) if lines else "(no reflections yet)"


def latest_advice(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row.get("turn", -1)), reverse=True)
    for row in ordered:
        advice = str(row.get("next_turn_advice") or "").strip()
        if advice:
            return advice
    return ""


def _parse_reflection_payload(response: str) -> Any:
    """The first JSON object in a reply (thinking models wrap it), or None."""

    try:
        payload = json.loads(response)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(response or ""):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response, start)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _backoff_for(attempt: int, backoff_s: tuple[float, ...]) -> float:
    """Wait before attempt ``attempt + 1`` (attempt counts from 1)."""

    if not backoff_s:
        return 0.0
    return float(backoff_s[min(int(attempt) - 1, len(backoff_s) - 1)])


def run_reflection(
    *,
    provider: str,
    llm_config: LLMConfig | None,
    turn: int,
    digest: dict[str, Any],
    plan: dict[str, Any] | None,
    coverage: dict[str, Any] | None,
    turn_dir: Path,
    manual_dir: Path,
    slow_rows: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    attempts: int | None = None,
    backoff_s: tuple[float, ...] | None = None,
    wall_budget_s: float | None = None,
    sleep: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    """The iteration review, with a bounded retry. Never raises.

    Reflection is an advisory channel, so a provider hiccup must not fail a
    turn that already produced valid measurements -- but it is also part of
    every iteration (paper B4), so a failed call is retried up to
    ``attempts`` times (default REFLECTION_ATTEMPTS) with the backoff
    REFLECTION_BACKOFF_S, and no new attempt starts after ``wall_budget_s``.
    Every attempt is one row of reflection_call.jsonl; the stored record
    carries ``status`` ("ok" / "failed"), ``attempts`` and, when failed, the
    last ``error``. ``sleep`` / ``clock`` are injectable for tests.
    """

    import time

    from omniqec_agentic.llm.factory import create_completion_client

    attempts = max(1, int(REFLECTION_ATTEMPTS if attempts is None else attempts))
    backoff = tuple(REFLECTION_BACKOFF_S if backoff_s is None else backoff_s)
    budget = float(REFLECTION_WALL_BUDGET_S if wall_budget_s is None else wall_budget_s)
    sleep = sleep or time.sleep
    clock = clock or time.monotonic

    system_text, prompt_hash = load_prompt("reflector")
    user_text = reflection_input(
        turn=turn, digest=digest, plan=plan, coverage=coverage, slow_rows=slow_rows, extra=extra
    )
    turn_dir = Path(turn_dir)
    turn_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = normalize_reflection(None, turn=turn)
    error = ""
    used = 0
    started = clock()
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            if clock() - started >= budget:
                break
            sleep(_backoff_for(attempt - 1, backoff))
        used = attempt
        usage: dict[str, Any] = {}
        error = ""
        try:
            config = llm_config or LLMConfig()
            if int(config.max_tokens) > REFLECTION_MAX_TOKENS:
                import dataclasses

                config = dataclasses.replace(config, max_tokens=REFLECTION_MAX_TOKENS)
            client = create_completion_client(provider, config)
            response = client.complete(system=system_text, user=user_text)
            info = dict(client.info())
            usage = dict(info.get("usage") or {})
            payload = _parse_reflection_payload(response)
            if payload is None:
                error = "UnparseableReflection: the reply contains no JSON object"
            else:
                record = normalize_reflection(payload, turn=turn)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Reflection calls are recorded apart from llm_calls.jsonl on purpose:
        # the strict turn audit checks that file for planner provenance, and an
        # advisory call has no business changing what that audit sees.
        with (turn_dir / "reflection_call.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "turn": int(turn), "role": "reflector",
                "prompt_id": f"reflector:{prompt_hash}",
                "attempt": int(attempt), "max_attempts": int(attempts),
                "status": "ok" if not error else "failed", "error": error, "usage": usage,
            }, ensure_ascii=False, sort_keys=True) + "\n")
        if not error:
            break
    if error:
        record = normalize_reflection(None, turn=turn)
    record["status"] = "ok" if not error else "failed"
    record["attempts"] = int(used)
    record["policy_id"] = REFLECTION_POLICY_ID
    if error:
        record["error"] = error

    (turn_dir / "reflection.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_reflection(manual_dir, record)
    return record
