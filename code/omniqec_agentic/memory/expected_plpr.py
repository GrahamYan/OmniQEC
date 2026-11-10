"""Expected plpr of an UNMEASURED candidate from the campaign's own
measurements (telescoping_structural_v3_7, 2026-09-12).

The runtime slow slots used to be ordered by Q_eff = k * d_search^2 — a
structure-only proxy that never learns from the slow loop. Under it a
weight-8 [[50,10,5]] (Q = 250) kept outranking a weight-6 [[48,4,6]]
(Q = 144) after ten turns had measured the first class at ~5e-3 and the
second at ~1e-3. This model closes that loop with the smallest thing that
does the job: the mean log10 plpr of the codes measured so far with the same
check weight and the same d_search, shrunk toward the same-weight mean and
then toward the global mean when a class has few measurements (n0 = 3). No
fitted slopes, no family term: those need more data than a 30-turn run has
and would be one more place to hide a preference. With NO measurements the
model predicts nothing and the archive order falls back to Q_eff, so turn 0
is unchanged.

Deterministic; a pure function of the evidence rows. Heavy codes are not
penalised by rule: if the heavy class measures well, its prediction improves
and it is promoted again.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

EXPECTED_PLPR_MODEL_ID = "expected_plpr_v1"
SHRINKAGE_N0 = 3.0


@dataclass(frozen=True)
class EvidencePoint:
    code_hash: str
    plpr: float
    max_check_weight: int | None
    d_search: int | None
    k: int | None
    family: str | None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evidence_points(rows: Iterable[dict[str, Any]]) -> list[EvidencePoint]:
    """One point per code (lowest plpr wins on a re-measurement), rows without
    a usable plpr skipped. Accepts evidence_rows.jsonl rows, slow rows and
    archive parent rows: the plpr key is `per_logical_per_round` or `plpr`,
    the distance key `d_search` or `d_proxy`."""

    by_hash: dict[str, EvidencePoint] = {}
    for row in rows or []:
        if not isinstance(row, dict) or row.get("is_baseline"):
            continue
        raw = row.get("per_logical_per_round")
        if raw is None:
            raw = row.get("plpr")
        try:
            plpr = float(raw)
        except (TypeError, ValueError):
            continue
        if not (plpr > 0.0) or not math.isfinite(plpr):
            continue
        code_hash = str(row.get("code_hash") or row.get("program_id") or "")
        if not code_hash:
            continue
        d_search = _int_or_none(row.get("d_search"))
        if d_search is None:
            d_search = _int_or_none(row.get("d_proxy"))
        point = EvidencePoint(
            code_hash=code_hash,
            plpr=plpr,
            max_check_weight=_int_or_none(row.get("max_check_weight")),
            d_search=d_search,
            k=_int_or_none(row.get("k")),
            family=str(row["family"]) if row.get("family") else None,
        )
        held = by_hash.get(code_hash)
        if held is None or point.plpr < held.plpr:
            by_hash[code_hash] = point
    return list(by_hash.values())


class ExpectedPlprModel:
    """Shrunken class means of log10 plpr: (weight, d_search) -> weight -> global."""

    model_id = EXPECTED_PLPR_MODEL_ID

    def __init__(self, points: Iterable[EvidencePoint], *, n0: float = SHRINKAGE_N0) -> None:
        self.points = [p for p in points if p.plpr > 0.0]
        self.n0 = float(n0)
        logs = [math.log10(p.plpr) for p in self.points]
        self.n_measured = len(logs)
        self.global_mean = statistics.fmean(logs) if logs else None
        self._by_weight: dict[int, list[float]] = {}
        self._by_weight_d: dict[tuple[int, int], list[float]] = {}
        for point, value in zip(self.points, logs):
            if point.max_check_weight is None:
                continue
            self._by_weight.setdefault(point.max_check_weight, []).append(value)
            if point.d_search is not None:
                self._by_weight_d.setdefault((point.max_check_weight, point.d_search), []).append(value)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, n0: float = SHRINKAGE_N0) -> "ExpectedPlprModel":
        return cls(evidence_points(rows), n0=n0)

    def _shrink(self, values: list[float] | None, prior: float) -> float:
        if not values:
            return prior
        n = float(len(values))
        return (n * statistics.fmean(values) + self.n0 * prior) / (n + self.n0)

    def predict_log10(
        self,
        *,
        max_check_weight: int | None,
        d_search: int | None,
        k: int | None = None,
        family: str | None = None,
    ) -> float | None:
        if self.global_mean is None:
            return None
        weight = _int_or_none(max_check_weight)
        if weight is None:
            return self.global_mean
        by_weight = self._shrink(self._by_weight.get(weight), self.global_mean)
        d = _int_or_none(d_search)
        if d is None:
            return by_weight
        return self._shrink(self._by_weight_d.get((weight, d)), by_weight)

    def predict(
        self,
        *,
        max_check_weight: int | None,
        d_search: int | None,
        k: int | None = None,
        family: str | None = None,
    ) -> float | None:
        value = self.predict_log10(
            max_check_weight=max_check_weight, d_search=d_search, k=k, family=family
        )
        return None if value is None else 10.0 ** value

    def by_weight_table(self) -> list[dict[str, Any]]:
        """Per check-weight class: measured count, best and median plpr (facts)."""

        out: list[dict[str, Any]] = []
        for weight in sorted(self._by_weight):
            values = sorted(10.0 ** v for v in self._by_weight[weight])
            out.append(
                {
                    "max_check_weight": weight,
                    "measured": len(values),
                    "best_plpr": values[0],
                    "median_plpr": statistics.median(values),
                }
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_measured": self.n_measured,
            "shrinkage_n0": self.n0,
            "global_mean_log10": self.global_mean,
            "by_weight": self.by_weight_table(),
        }
