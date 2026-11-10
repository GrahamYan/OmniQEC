"""Deterministic prospective backtester (03 §6.1) — the ledger's only judge.

Anti-double-dipping: supported/refuted verdicts use ONLY slow-evidence rows
whose generation is strictly AFTER the entry's propose_gen; earlier rows can
at most keep it a hypothesis. Requires N >= MIN_PROSPECTIVE rows in scope;
effect size = difference of group means of log10(metric) between
predicate-false and predicate-true rows (positive = predicate-true group has
LOWER metric, matching claim direction "lower"); bootstrap 95% CI must
exclude zero with a consistent sign. Runs incrementally at reconcile; the RNG
is seeded from the entry id so re-runs are bit-identical."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .ledger import Insight, Ledger

MIN_PROSPECTIVE = 20
BOOTSTRAP_SAMPLES = 2000
CI_LEVEL = 0.95


def _log_metric(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return math.log10(value)


def adjudicate_entry(
    entry: Insight,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """One entry against the accumulated evidence rows. Each row must carry a
    ``gen`` stamp (campaign generation of its slow measurement)."""

    prospective = [
        row
        for row in rows
        if row.get("gen") is not None
        and int(row["gen"]) > int(entry.provenance.get("propose_gen") or 0)
        and entry.scope.contains(row)
    ]
    true_vals: list[float] = []
    false_vals: list[float] = []
    last_gen = None
    for row in prospective:
        holds = entry.predicate_holds(row)
        if holds is None:
            continue
        value = _log_metric(row, entry.claim.metric)
        if value is None:
            continue
        (true_vals if holds else false_vals).append(value)
        last_gen = max(last_gen or 0, int(row["gen"]))
    n = len(true_vals) + len(false_vals)
    evidence: dict[str, Any] = {
        "support_n": len(true_vals),
        "contra_n": len(false_vals),
        "effect_size": None,
        "ci": None,
        "last_gen": last_gen,
    }
    if n < MIN_PROSPECTIVE or not true_vals or not false_vals:
        return {"status": "hypothesis", "evidence": evidence}

    sign = 1.0 if entry.claim.direction == "lower" else -1.0
    true_arr = np.asarray(true_vals)
    false_arr = np.asarray(false_vals)
    effect = sign * (float(false_arr.mean()) - float(true_arr.mean()))
    rng = np.random.default_rng(int(entry.id) * 7919 + 13)
    samples = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        bt = true_arr[rng.integers(0, len(true_arr), len(true_arr))]
        bf = false_arr[rng.integers(0, len(false_arr), len(false_arr))]
        samples[i] = sign * (bf.mean() - bt.mean())
    alpha = (1.0 - CI_LEVEL) / 2.0
    ci = (float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)))
    evidence.update(effect_size=float(effect), ci=[ci[0], ci[1]])

    min_effect = float(entry.claim.min_effect or 0.0)
    if ci[0] > min_effect:
        return {"status": "supported", "evidence": evidence}
    if ci[1] < -min_effect:
        return {"status": "refuted", "evidence": evidence}
    return {"status": "hypothesis", "evidence": evidence}


def run_backtest(ledger: Ledger, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Adjudicate every active entry; write verdicts through the ledger.

    Supported entries are re-adjudicated as evidence accumulates (a later
    reversal demotes/refutes them — the graveyard applies only to refuted)."""

    verdicts: dict[int, dict[str, Any]] = {}
    for entry in list(ledger.entries.values()):
        if entry.status not in ("hypothesis", "supported"):
            continue
        verdicts[entry.id] = adjudicate_entry(entry, rows)
    ledger.apply_adjudications(verdicts)
    return {
        "adjudicated": len(verdicts),
        "supported": sum(1 for v in verdicts.values() if v["status"] == "supported"),
        "refuted": sum(1 for v in verdicts.values() if v["status"] == "refuted"),
        "protocol": {
            "min_prospective": MIN_PROSPECTIVE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "ci_level": CI_LEVEL,
        },
    }


def insight_precision(
    ledger: Ledger,
    holdout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Endogenous learning metric (03 D7 / §6.2): fraction of supported rules
    whose direction re-validates on a held-out generation (no extra runs)."""

    checked = 0
    correct = 0
    for entry in ledger.supported():
        verdict = adjudicate_entry(entry, holdout_rows)
        effect = (verdict.get("evidence") or {}).get("effect_size")
        if effect is None:
            continue
        checked += 1
        if effect > 0:
            correct += 1
    return {
        "supported_total": len(ledger.supported()),
        "checked": checked,
        "correct": correct,
        "precision": (correct / checked) if checked else None,
    }
