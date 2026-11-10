"""Insight ledger (03 §6.1): externalized, backtested campaign learning.

Entries are strict pydantic models persisted as one JSON file per campaign.
The policy agent has PROPOSAL RIGHTS ONLY: status transitions to
supported/refuted are made exclusively by the deterministic prospective
backtester (memory/backtester.py). Refuted entries move to the graveyard —
kept forever, excluded from prompts, and their predicate hash blocks
re-proposal.

Display ids follow 03a: ``H-<n>`` while hypothesis, ``I-<n>`` once supported
(same number; both resolve to the entry for citation validation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LEDGER_SCHEMA_VERSION = 1
MAX_ACTIVE_HYPOTHESES = 16  # concurrency cap (03 §6.1 multiplicity control)

# Fields a predicate may reference: computed candidate/evidence columns only.
PREDICATE_FIELDS = (
    "family",
    "n",
    "k",
    "d_proxy",
    "fast_Q",
    "per_logical_per_round",
    "d_circ_ext",
    "degraded_ext",
    "degraded_isd",
    "verified_circuit_distance",
    "max_check_weight",
    "stopped_by",
    # Structural coordinates (construction/structure.py). Without these the
    # only expressible regularities are about size and score; a claim like
    # "direct-product groups beat dihedral at this budget" — the kind the
    # search most needs — could not be written down, let alone adjudicated.
    "group_class",
    "group_order",
    "group_order_band",
    "shape_a",
    "shape_b",
)
CLAIM_METRICS = ("per_logical_per_round",)


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float | int | str | bool

    @field_validator("field")
    @classmethod
    def _known_field(cls, value: str) -> str:
        if value not in PREDICATE_FIELDS:
            raise ValueError(f"predicate field '{value}' not a computed column {PREDICATE_FIELDS}")
        return value

    def evaluate(self, row: dict[str, Any]) -> bool | None:
        actual = row.get(self.field)
        if actual is None:
            return None
        try:
            if self.op == "==":
                return actual == self.value
            if self.op == "!=":
                return actual != self.value
            left, right = float(actual), float(self.value)
        except (TypeError, ValueError):
            return None
        return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[self.op]


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    families: tuple[str, ...] = ("bb", "hgp", "lp_abelian", "lp_nonabelian")
    budget_min: int | None = None
    budget_max: int | None = None
    profiles: tuple[str, ...] = ()

    def contains(self, row: dict[str, Any]) -> bool:
        if self.families and str(row.get("family")) not in self.families:
            return False
        # v3.6 (docs/06 §3 minor): the scope is a TOTAL-qubit budget window,
        # so it reads n_total (data + ancillas) and falls back to the data
        # count only for rows that never recorded n_total. Reading `n` alone
        # put [[200,40,12]] (n_total 360) outside every b>=288 hypothesis.
        n = row.get("n_total")
        if n is None:
            n = row.get("n")
        if self.budget_min is not None and (n is None or int(n) < self.budget_min):
            return False
        if self.budget_max is not None and (n is None or int(n) > self.budget_max):
            return False
        if self.profiles and str(row.get("noise_profile")) not in self.profiles:
            return False
        return True


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metric: str = "per_logical_per_round"
    direction: Literal["lower", "higher"] = "lower"
    min_effect: float | None = None

    @field_validator("metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in CLAIM_METRICS:
            raise ValueError(f"claim metric '{value}' not in {CLAIM_METRICS}")
        return value


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    support_n: int = 0
    contra_n: int = 0
    effect_size: float | None = None
    ci: tuple[float, float] | None = None
    last_gen: int | None = None


class Insight(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    status: Literal["hypothesis", "supported", "refuted", "retired"] = "hypothesis"
    scope: Scope = Field(default_factory=Scope)
    predicate: tuple[Comparison, ...]  # conjunction
    claim: Claim = Field(default_factory=Claim)
    evidence: Evidence = Field(default_factory=Evidence)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @property
    def display_id(self) -> str:
        return f"I-{self.id}" if self.status == "supported" else f"H-{self.id}"

    @property
    def predicate_hash(self) -> str:
        canonical = json.dumps(
            {
                "scope": self.scope.model_dump(),
                "predicate": [c.model_dump() for c in self.predicate],
                "claim": self.claim.model_dump(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def predicate_holds(self, row: dict[str, Any]) -> bool | None:
        """Conjunction; None when any referenced field is missing."""

        result = True
        for comparison in self.predicate:
            value = comparison.evaluate(row)
            if value is None:
                return None
            result = result and value
        return result

    def render_line(self) -> str:
        pred = " AND ".join(f"{c.field}{c.op}{c.value}" for c in self.predicate)
        claim = f"{self.claim.metric} {self.claim.direction}"
        scope = ",".join(self.scope.families)
        line = f"{self.display_id} [{self.status}, scope={scope}]: IF {pred} THEN {claim}"
        if self.evidence.effect_size is not None:
            line += f" (effect={self.evidence.effect_size:+.3f} log10, N={self.evidence.support_n + self.evidence.contra_n})"
        return line


class Ledger:
    """One JSON file per campaign; every mutation writes through."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[int, Insight] = {}
        self._next_id = 1
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("entries") or []:
                insight = Insight.model_validate(raw)
                self.entries[insight.id] = insight
            self._next_id = int(data.get("next_id") or (max(self.entries, default=0) + 1))

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema": "insight_ledger",
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "next_id": self._next_id,
                    "entries": [entry.model_dump(mode="json") for entry in self.entries.values()],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def active_hypotheses(self) -> list[Insight]:
        return [e for e in self.entries.values() if e.status == "hypothesis"]

    def supported(self) -> list[Insight]:
        return [e for e in self.entries.values() if e.status == "supported"]

    def resolve(self, display_id: str) -> Insight | None:
        try:
            prefix, number = display_id.split("-", 1)
            entry = self.entries.get(int(number))
        except (ValueError, AttributeError):
            return None
        if entry is None or prefix not in ("I", "H"):
            return None
        return entry

    def propose(
        self,
        *,
        scope: dict[str, Any] | None,
        predicate: list[dict[str, Any]],
        claim: dict[str, Any] | None,
        proposer: str,
        turn: int,
        propose_gen: int,
        rationale: str,
    ) -> dict[str, Any]:
        """Validated proposal. Rejections: cap exceeded, graveyard duplicate,
        schema violation. Returns {accepted, id?, reason?}."""

        if len(self.active_hypotheses()) >= MAX_ACTIVE_HYPOTHESES:
            return {
                "accepted": False,
                "reason": f"active hypothesis cap ({MAX_ACTIVE_HYPOTHESES}) reached; retract or await adjudication",
            }
        try:
            insight = Insight(
                id=self._next_id,
                scope=Scope.model_validate(scope or {}),
                predicate=tuple(Comparison.model_validate(c) for c in predicate),
                claim=Claim.model_validate(claim or {}),
                provenance={
                    "proposer": proposer,
                    "turn": int(turn),
                    "propose_gen": int(propose_gen),
                    "rationale": str(rationale),
                    "cited_by": 0,
                },
            )
        except Exception as exc:
            return {"accepted": False, "reason": f"schema: {exc}"}
        if not insight.predicate:
            return {"accepted": False, "reason": "empty predicate"}
        graveyard = {
            e.predicate_hash for e in self.entries.values() if e.status in ("refuted", "retired")
        }
        if insight.predicate_hash in graveyard:
            return {"accepted": False, "reason": "predicate previously refuted/retired (graveyard)"}
        if insight.predicate_hash in {
            e.predicate_hash for e in self.entries.values() if e.status in ("hypothesis", "supported")
        }:
            return {"accepted": False, "reason": "duplicate of an active entry"}
        self.entries[insight.id] = insight
        self._next_id += 1
        self._write()
        return {"accepted": True, "id": insight.display_id}

    def retract(self, display_id: str, *, turn: int) -> dict[str, Any]:
        entry = self.resolve(display_id)
        if entry is None:
            return {"accepted": False, "reason": f"unknown id {display_id}"}
        if entry.status not in ("hypothesis", "supported"):
            return {"accepted": False, "reason": f"{display_id} already {entry.status}"}
        entry.status = "retired"
        entry.provenance["retired_turn"] = int(turn)
        self._write()
        return {"accepted": True, "id": display_id}

    def apply_adjudications(self, verdicts: dict[int, dict[str, Any]]) -> None:
        """Backtester write-back (the ONLY path to supported/refuted)."""

        for entry_id, verdict in verdicts.items():
            entry = self.entries.get(entry_id)
            if entry is None or entry.status not in ("hypothesis", "supported"):
                continue
            entry.evidence = Evidence.model_validate(verdict.get("evidence") or {})
            new_status = verdict.get("status")
            if new_status in ("hypothesis", "supported", "refuted"):
                entry.status = new_status
        self._write()
