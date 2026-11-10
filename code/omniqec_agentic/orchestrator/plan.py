"""TurnPlan schema + rejection-style validator (03a §3.3).

Citation discipline is schema-bound, not prompted-for: `cites` may only
contain ids that exist in THIS turn's context — S-* block ids, candidate
code_hash prefixes (>= 8 chars, resolved against the current rows), or
ledger ids I-*/H-*. Unknown ids, empty load-bearing cites, or out-of-cap
values reject the payload with structured errors (one repair round is the
caller's job)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omniqec_agentic.memory.bandit import STRATEGY_ARMS

_HASH_RE = re.compile(r"^[0-9a-f]{8,64}$")

# Advisory context blocks: readable, citable for colour, but never sufficient
# on their own for a load-bearing claim (v3.2 §5.2). S-10 is the reflection
# window — prose a previous turn wrote about itself. Letting it satisfy the
# evidence requirement turns "cite your source" into "cite something".
ADVISORY_SOURCE_IDS = frozenset({"S-10"})


class Cited(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    cites: list[str] = Field(default_factory=list)


class Promotion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_hash: str
    budget_class: Literal["pilot", "confirm"] = "pilot"
    rationale: Cited | None = None


class TurnPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Cited] = Field(default_factory=list)
    ledger_ops: list[dict[str, Any]] = Field(default_factory=list)  # record only; live ops go through tools
    rationale: Cited
    family_mix: dict[str, float] = Field(default_factory=dict)
    strategy_mix: dict[str, float] = Field(default_factory=dict)
    program_count: int = 8
    promotions: list[Promotion] = Field(default_factory=list)
    directives: str = ""
    # How hard this turn exploits the current elite. 0 keeps the deterministic
    # top-ranked parents; higher values soften parent choice toward a Boltzmann
    # draw so the search can leave a neighbourhood it has already mined. The
    # explore/exploit balance is the policy's call, not a fixed constant.
    explore_temperature: float = 0.0
    finalize: bool = False


def _check_cites(
    cites: list[str],
    *,
    source_ids: set[str],
    known_hashes: set[str],
    ledger_ids: set[str],
    errors: list[str],
    where: str,
    require_nonempty: bool,
) -> None:
    if require_nonempty and not cites:
        errors.append(f"{where}: empty cites on a load-bearing claim")
    if require_nonempty and cites and set(cites) <= ADVISORY_SOURCE_IDS:
        errors.append(
            f"{where}: {sorted(ADVISORY_SOURCE_IDS & set(cites))} is advisory prose from "
            "an earlier turn, not evidence — cite a measurement or structural block too"
        )
    for cite in cites:
        if cite in source_ids or cite in ledger_ids:
            continue
        if _HASH_RE.match(cite) and any(h.startswith(cite) for h in known_hashes):
            continue
        errors.append(f"{where}: unknown citation id '{cite}'")


def validate_turn_plan(
    raw: dict[str, Any],
    *,
    source_ids: set[str],
    known_hashes: set[str],
    ledger_ids: set[str],
    families: tuple[str, ...],
    program_cap: int,
    promote_cap: int,
    quarantined_hashes: set[str],
    acquisition_order: list[str] | None = None,
) -> tuple[TurnPlan | None, list[str]]:
    errors: list[str] = []
    try:
        plan = TurnPlan.model_validate(raw)
    except Exception as exc:
        return None, [f"schema: {exc}"]

    _check_cites(
        plan.rationale.cites,
        source_ids=source_ids, known_hashes=known_hashes, ledger_ids=ledger_ids,
        errors=errors, where="rationale", require_nonempty=True,
    )
    # Observations: cite ids are validated when present, but empty cites are
    # legal (03a R3 hard-requires cites on rationale fields only; forcing them
    # on every observation makes forward-looking statements impossible).
    for index, observation in enumerate(plan.observations):
        _check_cites(
            observation.cites,
            source_ids=source_ids, known_hashes=known_hashes, ledger_ids=ledger_ids,
            errors=errors, where=f"observations[{index}]", require_nonempty=False,
        )

    unknown_families = set(plan.family_mix) - set(families)
    if unknown_families:
        errors.append(f"family_mix: unknown families {sorted(unknown_families)}")
    if plan.family_mix:
        total = sum(plan.family_mix.values())
        if any(v < 0 for v in plan.family_mix.values()) or not 0.99 <= total <= 1.01:
            errors.append(f"family_mix: fractions must be >=0 and sum to 1 (got {total:.3f})")
    unknown_arms = set(plan.strategy_mix) - set(STRATEGY_ARMS)
    if unknown_arms:
        errors.append(f"strategy_mix: unknown arms {sorted(unknown_arms)}")
    if not 0.0 <= float(plan.explore_temperature) <= 5.0:
        errors.append("explore_temperature: must be in [0.0, 5.0]")

    if not 1 <= int(plan.program_count) <= int(program_cap):
        errors.append(f"program_count: must be in [1, {program_cap}]")
    if len(plan.promotions) > int(promote_cap):
        errors.append(
            f"promotions: {len(plan.promotions)} exceeds the runtime-slot cap "
            f"{promote_cap}. A named candidate CONSUMES a runtime slot; the "
            "structural slots are machine-selected and cannot be named here"
        )

    baseline = list(acquisition_order or [])
    for index, promotion in enumerate(plan.promotions):
        prefix = str(promotion.code_hash)
        if len(prefix) < 8 or not _HASH_RE.match(prefix):
            errors.append(f"promotions[{index}]: code_hash must be a >=8-char hex prefix")
            continue
        matches = [h for h in known_hashes if h.startswith(prefix)]
        if not matches:
            errors.append(f"promotions[{index}]: '{prefix}' matches no known candidate")
            continue
        if any(h in quarantined_hashes for h in matches):
            errors.append(
                f"promotions[{index}]: '{prefix}' is schedule_catastrophic — no "
                "sampled schedule avoids a single-fault logical, so it cannot be "
                "measured at all"
            )
            continue
        if baseline:
            in_baseline_head = any(
                b.startswith(prefix) or prefix.startswith(b)
                for b in baseline[: len(plan.promotions)]
            )
            if not in_baseline_head and promotion.rationale is None:
                # Name the exact shape to write. The bare rule was restated up
                # to the repair limit without the model ever producing the
                # field, which killed a 22-turn campaign; an error that cannot
                # be acted on is worth no more than no error at all.
                head = str(baseline[0])[:12]
                errors.append(
                    f"promotions[{index}]: '{prefix}' departs from the acquisition "
                    f"baseline order (the baseline head is '{head}'; measured "
                    "candidates are never in the baseline). Either promote "
                    f'{{"code_hash": "{head}", "budget_class": "pilot"}} with no '
                    f"rationale, or keep '{prefix}' and add a per-item rationale, "
                    f'e.g. {{"code_hash": "{prefix}", "budget_class": "pilot", '
                    '"rationale": {"text": "why this candidate before the baseline '
                    'head", "cites": ["S-4"]}}.'
                )
            elif promotion.rationale is not None:
                _check_cites(
                    promotion.rationale.cites,
                    source_ids=source_ids, known_hashes=known_hashes,
                    ledger_ids=ledger_ids, errors=errors,
                    where=f"promotions[{index}].rationale", require_nonempty=True,
                )
    if errors:
        return None, errors
    return plan, []
