"""Regressions for the literature-vs-catalog group naming footgun (01 §4.2).

The catalog indexes dihedral groups by ROTATION order (``D3`` has six
elements); the Mitten and Lin-Pryadko tables index them by GROUP order and
write the order-6 group as ``S3``. Transcribing a group name out of a paper
therefore fails in two different ways, and the silent one is the dangerous
one:

  ``C5xS3``   -- no such catalog name, so it raised a bare "unknown group";
  ``C4xD10``  -- a *legal* name for a different group (order 80, not 40),
                 which only the size gate stopped, and only by complaining
                 about size rather than identity.

Both now name the spelling the caller wanted. Aliasing was deliberately not
used: ``construction/lp.py`` echoes ``group.name`` back into the normalized
spec, so a second accepted spelling would put one group into the archive under
two names.
"""

from __future__ import annotations

import pytest

from omniqec_agentic.construction.groups.catalog import (
    catalog_summary,
    get_group,
    group_order_budget,
)
from omniqec_agentic.verification.structural import structural_check_spec


@pytest.mark.parametrize(
    ("literature", "canonical"),
    [("S3", "D3"), ("C5xS3", "C5xD3"), ("C11xS3", "C11xD3")],
)
def test_symmetric_spelling_points_at_the_catalog_name(literature, canonical):
    with pytest.raises(ValueError) as excinfo:
        get_group(literature)
    message = str(excinfo.value)
    assert f"'{canonical}'" in message, message
    # The pointer has to be actionable on its own: the caller should be able to
    # copy the quoted name straight back in.
    assert get_group(canonical).name == canonical


def test_mitten_host_group_resolves_under_its_catalog_spelling():
    # Mitten's flagship [[150,30,10]] host group, written "C5 x S3" there.
    group = get_group("C5xD3")
    assert (group.order, group.gap_id, group.is_abelian) == (30, (30, 1), False)


def test_dihedral_index_collision_is_named_not_just_rejected():
    # Literature "C4 x D10" is order 40 == our C4xD5; the same string parses
    # here as C4 x D_10 of order 80. It used to be stopped only by the order
    # cap, which meant it complained about size rather than identity; the
    # spelling is now reserved outright, so the verdict no longer depends on
    # where the cap happens to sit.
    with pytest.raises(ValueError) as excinfo:
        get_group("C4xD10")
    message = str(excinfo.value)
    assert "'C4xD5'" in message, message
    assert "order 80" in message, message
    assert get_group("C4xD5").gap_id == (40, 5)


@pytest.mark.parametrize("budget", [60, 80, 98, 120, 144, 288])
def test_reserved_spelling_outranks_the_budget_verdict(budget):
    """A reserved name must never be reported as merely too big.

    ``C4xD10`` reads literally as order 80, so its smallest LP code costs
    4*80 = 320 total qubits -- over budget at every point in the live grid.
    When the budget pre-gate trusted that literal reading it answered
    ``total_qubit_budget_violation`` and the "use 'C4xD5'" pointer never
    reached the caller, which is the silent mis-transcription this whole file
    exists to prevent. ``catalog_order`` therefore abstains on reserved
    spellings so the naming diagnostic wins.
    """

    spec = {
        "family": "lp_nonabelian",
        "group": "C4xD10",
        "A": [[[[0, 0, 0], [1, 0, 0]]]],
        "B": [[[[0, 0, 0], [0, 1, 0]]]],
    }
    result = structural_check_spec("naming", spec, total_budget=budget)
    reasons = " ".join(result.failure_reasons)
    assert "'C4xD5'" in reasons, reasons
    assert "total_qubit_budget_violation" not in reasons, reasons


@pytest.mark.parametrize("name", ["C7xD9", "C7xD10"])
def test_hint_stays_off_when_the_halved_reading_is_not_a_catalog_entry(name):
    # The hint rides on the order-cap error and is evidence-gated on
    # KNOWN_GAP_IDS, so it cannot invent a literature reading for an arbitrary
    # oversized dihedral product: C7xD9 has an odd index to halve, and
    # C7xD10 halves cleanly to a C7xD5 that is not a catalog entry.
    # v3.6 M3: the cap is budget-derived (n_budget // 4); b288 gives the 72
    # the v3.5 constant used to hard-code.
    with group_order_budget(288), pytest.raises(ValueError) as excinfo:
        get_group(name)
    message = str(excinfo.value)
    assert "literature" not in message, message
    assert "exceeds the budget cap 72 (= n_budget // 4 for n_budget 288)" in message, message


def test_prompt_grammar_states_the_indexing_convention():
    # catalog_summary() is rendered into the generator prompt verbatim
    # (construction/registry.py), so this is the only place the model can learn
    # that the paper's spelling is not the one the validator accepts.
    summary = catalog_summary()
    assert "S3" in summary and "D3" in summary
    assert "D10" in summary and "D5" in summary


def test_unknown_names_still_get_the_plain_grammar_error():
    with pytest.raises(ValueError, match="unknown non-abelian group"):
        get_group("Q8")


@pytest.mark.parametrize("name", ["C10xD6", "C11xD6"])
def test_hint_does_fire_for_a_real_literature_reading(name):
    # Companion to the test above: without this, "hint stays off" would pass
    # even if the hint were dead code. C{m}xD6 halves to C{m}xD3, which IS in
    # KNOWN_GAP_IDS, so the hint must appear.
    with group_order_budget(288), pytest.raises(ValueError) as excinfo:
        get_group(name)
    assert "literature" in str(excinfo.value)


def test_catalog_order_matches_what_construction_yields():
    """The size gate reads |G| off the name before building anything.

    That ordering is what makes an absurd name cheap to refuse (``D600`` went
    from ~28s of Cayley-table construction to microseconds), but it is only
    safe while ``catalog_order`` agrees with ``_construct`` on every grammar
    branch -- a number that is too large would reject a legal group without
    ever constructing it.
    """

    from omniqec_agentic.construction.groups.catalog import _construct, catalog_order

    names = ["A4", "S4"]
    names += [f"D{i}" for i in range(3, 25)]
    names += [f"C{m}x{h}" for m in range(2, 8) for h in ["A4", "S4"] + [f"D{i}" for i in range(3, 8)]]
    names += [f"C{m}:C{k}({t})" for m in (5, 7, 9, 13) for k in (2, 3, 4, 5) for t in range(2, 6)]

    checked = 0
    for name in names:
        try:
            group = _construct(name)
        except ValueError:
            continue  # not a legal presentation; the cap never sees it
        checked += 1
        assert catalog_order(name) == group.order, name
    assert checked > 50, checked
