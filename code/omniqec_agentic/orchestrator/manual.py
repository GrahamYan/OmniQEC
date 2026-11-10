"""Technical manual rendering (03 §5): static / dynamic / skills.

ACE mechanics adopted: incremental per-file updates, never a wholesale
rewrite (anti context-collapse); grow-and-refine — retired material moves to
the ledger graveyard, not deletion; supported insights render in FULL (no
summarization, anti brevity-bias). The curator is the deterministic
backtester: dynamic/ admission = backtest verdict, the LLM only proposes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omniqec_agentic.memory.bandit import StrategyBandit
from omniqec_agentic.memory.ledger import Ledger

RANKING_SEMANTICS = """# Two ledgers (telescoping_structural_v3_7)

The campaign keeps TWO independent records. Neither ranks the other, and
neither may delete an entry from the other.

## 1. Structural retention — "is this code structurally strong?"
- Objectives: absolute `k`, absolute `d_search` (the frozen 4-seed x {X,Z}
  minimum heuristic distance) and, since v3.7, the check weight
  `max_check_weight` (lighter is better). A candidate is retained while it
  holds a non-dominated (k, d_search, -weight) position at its budget point:
  a heavier code survives exactly when it reaches a k or a d_search that no
  lighter code reaches; a lighter code with the same (k, d_search) is not
  discarded. k >= 2 and d_search >= 5 are floors.
- The reducer keeps the max-k and the max-d_search endpoints of every Pareto
  layer, then fills by crowding distance over the three axes. `n`, budget
  utilisation, family and degree are recorded but are NOT axes.
- It reads NOTHING from the runtime ledger: no plpr, no d_eff, no d_cap, no
  degraded_* flag, no circuit distance. A structurally strong code that
  measured badly stays retained and stays parent-eligible.
- Two of the six slow slots are filled from it, mechanically: high-k
  endpoint and high-d_search endpoint. They are not promotions, and no
  policy decision reorders or vetoes them.

## 2. Runtime measurement — "how did it run under the frozen decoder?"
- One decoder everywhere: the telescoping decoder (telescoping_cpu_v1).
- per_logical_per_round (paper A4/A5) at the anchor point; overlapping CIs are
  statistical ties. This orders the runtime hall of fame. It is not a
  lineage rule.
- The FOUR runtime slow slots are ordered by the campaign's expected plpr for
  each unmeasured candidate: the mean measured plpr of codes with the same
  check weight and the same d_search, shrunk toward the same-weight mean and
  then the global mean when a class has few measurements (expected_plpr_v1).
  Before the first measurement the order is Q_eff = k * d_search^2. A named
  promotion consumes one of the four slots.
- Physics the measurements already contain: a check of weight w is measured
  with w CNOTs, hook errors grow with w, and the frozen decoder prices that
  cost into per_logical_per_round. check weight = |A| + |B| for bb and 1x1
  lifted products; an odd check weight requires |A| != |B|.
- distance evidence: exact code/circuit distance when configured, sec
  d_circ_ext probes (extended-matrix ISD upper bounds, SBB Thm 1), DEM-ISD
  upper bounds, and (once recalibrated) LER-slope estimates. All auxiliary.
- acquisition_score(ids) is the deterministic baseline for the FOUR runtime
  slots; departing from it needs a cited per-item rationale.

## Hard exclusion (the only one)
schedule_catastrophic: no sampled schedule avoids a single-fault logical, so
the candidate cannot be measured at all. `degraded_*` is a DIAGNOSTIC LABEL,
not a quarantine: its cost is already priced into the measured plpr.
"""

FEEDBACK_VOCABULARY = """# Feedback vocabulary (what a field means)

These are observations. None of them decides retention, parent eligibility or
promotion — the machine's retention rules do.

schedule_catastrophic: this check structure admits a single-fault logical
  under ALL P sampled schedules. A structural property, and the one condition
  that makes a candidate unmeasurable.
stopped_by=racing: the candidate's plpr CI fell clearly above both incumbent
  thresholds at the anchor p and sampling stopped early; a low-precision
  reading, not a settled result.
stopped_by=max_errors: enough failures accumulated for a precise estimate —
  the measurement is complete, not truncated.
d_circ_ext < d_search (degraded_ext): the circuit-level distance evidence is
  below the heuristic value. The runtime cost of that gap is already in the
  measured plpr; the code keeps its structural standing.
degraded_isd: a constructive undetectable logical error of weight below
  d_search exists in the circuit — same reading as degraded_ext.
trust screen (d_search / sqrt(n_data) below the floor): a runtime/acquisition
  screen and an audit field. It does not remove a candidate from structural
  retention; at large n it fires on exactly the high-k, moderate-d region.
low Delta-min schedules across all P: the schedule-hook signature, a
  structural property of the check overlap rather than a parameter accident.
"""


def render_static(manual_dir: Path) -> None:
    """Static section: tool catalog (docstring-rendered) + semantics pages."""

    from .registry import render_tool_docs

    static = manual_dir / "static"
    static.mkdir(parents=True, exist_ok=True)
    (static / "tools.md").write_text(render_tool_docs(), encoding="utf-8")
    (static / "ranking.md").write_text(RANKING_SEMANTICS, encoding="utf-8")
    (static / "feedback_vocab.md").write_text(FEEDBACK_VOCABULARY, encoding="utf-8")
    index = [
        "# Manual index",
        "",
        "- static/tools.md — tool catalog (registry-rendered; docstrings are the source of truth)",
        "- static/ranking.md — the two ledgers: structural retention and runtime measurement",
        "- static/feedback_vocab.md — feedback field semantics (what each field means)",
        "- dynamic/insights.md — supported insights (full text) + active hypotheses",
        "- dynamic/bandit.md — strategy-arm suggestion snapshot",
        "- dynamic/digest.md — latest turn digest (fast + slow top-k)",
        "- skills/ — frontier-producing construct programs with evidence pointers",
        "",
    ]
    (static / "INDEX.md").write_text("\n".join(index), encoding="utf-8")


REFUTED_RENDER_LIMIT = 12


def render_dynamic(
    manual_dir: Path,
    *,
    ledger: Ledger,
    bandit: StrategyBandit,
    digest: dict[str, Any] | None,
) -> None:
    """Dynamic section: backtester-gated insights, bandit snapshot, digest."""

    dynamic = manual_dir / "dynamic"
    dynamic.mkdir(parents=True, exist_ok=True)
    lines = ["# Insights (backtester-adjudicated)", ""]
    supported = ledger.supported()
    if supported:
        lines.append("## Supported (safe to exploit; cite by id)")
        lines.append("")
        for entry in supported:
            lines.append(f"- {entry.render_line()}")
        lines.append("")
    hypotheses = ledger.active_hypotheses()
    if hypotheses:
        lines.append("## Active hypotheses (testable; cite by id)")
        lines.append("")
        for entry in hypotheses:
            lines.append(f"- {entry.render_line()}")
        lines.append("")
    # Paper Table I (2026-11-10 alignment): the hypothesis list carries every
    # proposed relationship WITH its screening status, so the refuted ones are
    # listed too (newest first, bounded); they stay in the graveyard and
    # cannot be re-proposed.
    refuted = sorted(
        (entry for entry in ledger.entries.values() if entry.status == "refuted"),
        key=lambda entry: -int(entry.id),
    )[:REFUTED_RENDER_LIMIT]
    if refuted:
        lines.append("## Refuted (screened against later measurements; cannot be re-proposed)")
        lines.append("")
        for entry in refuted:
            lines.append(f"- {entry.render_line()}")
        lines.append("")
    if not supported and not hypotheses and not refuted:
        lines.append("(none yet — propose hypotheses via ledger_propose)")
        lines.append("")
    (dynamic / "insights.md").write_text("\n".join(lines), encoding="utf-8")

    suggestion = bandit.suggestion()
    bandit_lines = ["# Strategy bandit snapshot", ""]
    for arm, weight in sorted(suggestion["mix"].items(), key=lambda kv: -kv[1]):
        stat = suggestion["stats"].get(arm) or {}
        mean = stat.get("mean_reward")
        bandit_lines.append(
            f"- {arm}: suggested {weight:.2f} (pulls={stat.get('pulls', 0):.0f}, "
            f"mean_reward={'-' if mean is None else f'{mean:.3f}'})"
        )
    bandit_lines += ["", f"epsilon_arm floor: {suggestion['epsilon_arm']}", ""]
    (dynamic / "bandit.md").write_text("\n".join(bandit_lines), encoding="utf-8")

    if digest is not None:
        (dynamic / "digest.md").write_text(
            "# Latest turn digest\n\n```json\n"
            + json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )


def record_skill(
    manual_dir: Path,
    *,
    program_id: str,
    construct_source: str,
    evidence: dict[str, Any],
) -> None:
    """Skills section (Voyager-style): a frontier-producing construct program
    plus its evidence pointer. Skill = executable program; validation = slow
    evidence, never narrative."""

    skills = manual_dir / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in program_id)[:80]
    (skills / f"{stem}.py").write_text(construct_source, encoding="utf-8")
    (skills / f"{stem}.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
