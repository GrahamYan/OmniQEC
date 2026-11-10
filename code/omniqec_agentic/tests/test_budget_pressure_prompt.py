"""Generator prompt: what unspent budget costs, stated without a recipe.

Half the b288 winners spent barely half the qubit budget, and nothing in the
prompt or the archive table made that visible: ver1's Q = k*d^2/n was
scale-free, so a small code read exactly as well as a full-size one (the score
is k*d^2 since 2026-09-10, but the budget-use column stays). These tests pin the fix,
and equally pin what the fix must NOT become — a prescribed size band is a
recipe, and recipes are what collapse the search onto one shape.
"""

from __future__ import annotations

import re

from omniqec_agentic.config import CodeFamily, HardwareConfig, HardwareTarget, RunConfig
from omniqec_agentic.funsearch import (
    _budget_use,
    _funsearch_system_prompt,
    _funsearch_user_prompt,
    _parent_construction_lines,
)
from omniqec_agentic.memory import ArchiveMemory


def _config(budget: int = 288) -> RunConfig:
    return RunConfig(
        run_id="prompt_probe",
        n_budget=budget,
        hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused",
        proposal_count=4,
        allowed_families=tuple(CodeFamily),
    )


def _user_prompt(**context) -> str:
    return _funsearch_user_prompt(
        memory=ArchiveMemory(),
        config=_config(),
        gen=1,
        proposal_count=4,
        n_recent=4,
        prompt_context=dict(context),
    )


def test_unspent_budget_is_stated_as_a_cost() -> None:
    text = _user_prompt()
    assert "WASTED-BUDGET WARNING" in text
    assert "judged at N=288" in text
    # It must remain a cost statement the model can weigh, not an order: a
    # small code stays legal when it is a deliberate choice.
    assert "deliberate choice" in text


def test_no_prescribed_size_band_or_construction_recipe() -> None:
    text = _user_prompt() + _funsearch_system_prompt(
        allowed_families=[f.value for f in CodeFamily]
    )
    # A numeric target range would tell the model what to build; the whole
    # design is that only failure is specified and success is discovered.
    assert not re.search(r"target .{0,40}\[\s*\d+\s*,\s*\d+\s*\]", text, re.I)
    assert "aim to" not in text.lower()
    # No sentence may name the mutation to apply. (A syntax hint such as HGP's
    # "independent seed design is encouraged" is about legal shapes, not about
    # which codes are good, and is deliberately not covered here.)
    for recipe in ("scaling them up", "larger ell/m", "more proto rows"):
        assert recipe not in text


def test_objective_states_no_quantity_to_optimise() -> None:
    """v3.2 §5: the selection pressure is in the machine's retention rules.

    The objective paragraph may say what a legal candidate is and who decides
    lineage; it may not name a quantity to maximise, because such a sentence
    outranks every block below it."""

    system = _funsearch_system_prompt(allowed_families=["bb"])
    assert "retention rules" in system
    for banned in ("larger k wins", "PLPR decides", "lower logical error rate"):
        assert banned not in system
    # ...and still says nothing about HOW to build anything.
    assert "lift" not in system.split("GENOTYPE")[0].lower()


def test_distance_floor_is_stated_as_legality_not_as_a_target() -> None:
    """d_search >= 3 is what makes a code error-correcting, so it is stated.

    The old "KNOWN DEAD END: shrinking n ..." line went with it: it was an
    instruction justified by a slow-stage rule ("those gains are discarded")
    that the retention rules no longer implement."""

    system = _funsearch_system_prompt(allowed_families=["bb"])
    assert "Legality floor" in system
    assert "cannot correct a single fault" in system
    assert "KNOWN DEAD END" not in system


def test_budget_use_renders_spend_over_budget() -> None:
    assert _budget_use({"n": 114, "n_checks": 114, "n_budget": 288}) == "228/288"
    assert _budget_use({"n": 78, "n_checks": 78, "n_budget": 288}) == "156/288"
    # Unknown check counts must degrade, never crash or fake a number.
    assert _budget_use({"n": 78, "n_budget": 288}) == "?"
    assert _budget_use({"n": 78, "n_checks": 78}) == "?"


def test_parents_carry_their_spend_without_being_told_to_grow() -> None:
    lines = _parent_construction_lines(
        {
            "next_parent_specs": [
                {"program_id": "g0001_p0015", "family": "lp_nonabelian",
                 "n": 78, "n_checks": 78, "n_budget": 288, "k": 4, "d_proxy": 8,
                 "Q": 3.28, "rng_seed": 0, "spec": {"family": "lp_nonabelian"},
                 "construct_source": "def construct(rng):\n    return {}"},
            ]
        },
        limit=6,
    )
    body = "\n".join(lines)
    assert "[spends 156/288]" in body
    assert "scaling them up" not in body  # fact, not exhortation
