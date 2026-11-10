"""Regressions for the autonomy gaps found in the 2026-08-20 audit.

Each test pins one mechanism the search needs to explore a mixed space and
learn structure-level regularities from measured performance. The recurring
theme of the audit was silence: a steering channel nobody executed, a coverage
question nothing could answer, a censored measurement that looked settled.
"""

from __future__ import annotations

import json
import random

from omniqec_agentic.construction.structure import (
    STRUCTURE_FIELDS,
    order_band,
    structure_descriptor,
    structure_key,
)
from omniqec_agentic.memory.coverage import catalog_gaps, coverage_report
from omniqec_agentic.memory.ledger import PREDICATE_FIELDS
from omniqec_agentic.workflow.graph import _family_quota


# ---- structural vocabulary -------------------------------------------------

def test_nonabelian_branches_are_distinguished():
    """The 85%-dihedral skew is invisible unless branches have distinct names."""

    cases = {
        "D6": ("dihedral", 12),
        "C7:C3(2)": ("metacyclic", 21),
        "C3xD3": ("direct_product", 18),
        "A4": ("alternating_symmetric", 12),
    }
    for group, (expected_class, expected_order) in cases.items():
        descriptor = structure_descriptor({"family": "lp_nonabelian", "group": group})
        assert descriptor["group_class"] == expected_class, group
        assert descriptor["group_order"] == expected_order, group


def test_descriptor_always_returns_every_field():
    for spec in (
        {"family": "bb", "ell": 6, "m": 6},
        {"family": "hgp", "seeds": {"H1": [[1, 0], [0, 1]]}},
        {"family": "lp_abelian", "group": {"factors": [3, 3]}},
        {"family": "lp_nonabelian", "group": "D6"},
        {},
        None,
    ):
        descriptor = structure_descriptor(spec)
        assert set(descriptor) == set(STRUCTURE_FIELDS)


def test_lp_shapes_are_recorded():
    spec = {
        "family": "lp_nonabelian",
        "group": "D6",
        "A": [[1, 2]],
        "B": [[1], [2]],
    }
    descriptor = structure_descriptor(spec)
    assert descriptor["shape_a"] == "1x2"
    assert descriptor["shape_b"] == "2x1"


def test_order_bands_bucket_and_stay_open_ended():
    assert order_band(12) == "12-24"
    assert order_band(21) == "12-24"
    assert order_band(144) == "100+"
    assert order_band(None) is None


def test_structure_key_is_a_readable_region_label():
    key = structure_key({"family": "lp_nonabelian", "group": "C3xD3"})
    assert key == "lp_nonabelian/direct_product/12-24"


# ---- coverage instrument ---------------------------------------------------

def _archive_row(code_hash: str, group: str, family: str = "lp_nonabelian"):
    return {"code_hash": code_hash, "family": family,
            "spec": {"family": family, "group": group}}


def test_coverage_separates_censored_from_measured():
    """Racing-stopped is 'unresolved', not 'bad' — collapsing them would teach
    the search to write off a region it never actually measured."""

    archive = [_archive_row("aa" * 8, "D6"), _archive_row("bb" * 8, "C3xD3")]
    slow = [
        {"code_hash": "aa" * 8, "per_logical_per_round": 0.01, "stopped_by": "confirm"},
        {"code_hash": "bb" * 8, "per_logical_per_round": 0.09, "stopped_by": "racing"},
    ]
    report = coverage_report(archive, slow)
    regions = {row["region"]: row for row in report["regions"]}
    dihedral = regions["lp_nonabelian/dihedral/12-24"]
    direct = regions["lp_nonabelian/direct_product/12-24"]
    assert dihedral["measured"] == 1 and dihedral["censored"] == 0
    assert direct["censored"] == 1


def test_coverage_reports_regions_with_no_measurement():
    archive = [_archive_row("cc" * 8, "D8")]
    report = coverage_report(archive, [])
    assert "lp_nonabelian/dihedral/12-24" in report["unmeasured_regions"]
    assert report["total_proposed"] == 1


def test_coverage_ignores_the_resident_baseline():
    archive = [_archive_row("dd" * 8, "D6")]
    slow = [{"code_hash": "dd" * 8, "per_logical_per_round": 0.001,
             "is_baseline": True, "stopped_by": "confirm"}]
    report = coverage_report(archive, slow)
    assert report["regions"][0]["measured"] == 0


def test_catalog_gaps_names_untried_branches():
    """The audit's headline finding, made detectable: a run can spend every
    proposal in one branch and nothing in a ranked list would say so."""

    archive = [_archive_row(f"{i:016x}" * 2, "D6") for i in range(9)]
    archive.append(_archive_row("ff" * 8, "C7:C3(2)"))
    gaps = catalog_gaps(archive)
    assert gaps["by_branch"]["dihedral"] == 9
    assert gaps["by_branch"]["metacyclic"] == 1
    assert set(gaps["never_proposed_branches"]) == {
        "direct_product", "alternating_symmetric"
    }


def test_coverage_survives_rows_without_specs():
    report = coverage_report([{"code_hash": "ab" * 8}], [])
    assert report["total_proposed"] == 1  # counted as 'unknown', never crashes


# ---- adjudicated hypothesis vocabulary -------------------------------------

def test_structure_fields_are_addressable_by_a_hypothesis():
    """Without these columns a claim about group structure is unwritable, so
    the backtester could never rule on the regularities that matter most."""

    for field in ("group_class", "group_order", "group_order_band", "shape_a", "shape_b"):
        assert field in PREDICATE_FIELDS


def test_structural_predicate_validates_and_evaluates():
    from omniqec_agentic.memory.ledger import Comparison

    comparison = Comparison(field="group_class", op="==", value="direct_product")
    assert comparison.evaluate({"group_class": "direct_product"}) is True
    assert comparison.evaluate({"group_class": "dihedral"}) is False
    assert comparison.evaluate({}) is None


# ---- execution: the family allocation actually allocates --------------------

def test_family_quota_sums_to_the_batch_size():
    quota = _family_quota({
        "family_mix": {"bb": 0.25, "hgp": 0.25, "lp_abelian": 0.25, "lp_nonabelian": 0.25},
        "proposal_count": 24,
    })
    assert sum(quota.values()) == 24
    assert set(quota) == {"bb", "hgp", "lp_abelian", "lp_nonabelian"}


def test_family_quota_handles_uneven_fractions_without_losing_slots():
    quota = _family_quota({
        "family_mix": {"bb": 0.5, "hgp": 0.2, "lp_nonabelian": 0.3},
        "proposal_count": 24,
    })
    assert sum(quota.values()) == 24


def test_family_quota_is_empty_when_the_policy_asked_for_nothing():
    assert _family_quota({"family_mix": {}, "proposal_count": 24}) == {}
    assert _family_quota({"family_mix": {"bb": 1.0}, "proposal_count": 0}) == {}


def test_quota_reaches_the_generation_prompt_as_a_requirement():
    from omniqec_agentic.funsearch import _merge_prompt_context_overlay

    merged = _merge_prompt_context_overlay({}, {"family_quota": {"lp_nonabelian": 12}})
    assert merged["family_quota"] == {"lp_nonabelian": 12}


# ---- selection: prefix diversity and the exploration knob ------------------

def test_boltzmann_order_is_identity_at_zero_temperature():
    from omniqec_agentic.memory.archive import _boltzmann_order

    ranked = [(f"cell{i}", None) for i in range(8)]
    assert _boltzmann_order(ranked, temperature=0.0) == ranked


def test_boltzmann_order_can_reorder_when_warm():
    from omniqec_agentic.memory.archive import _boltzmann_order

    ranked = [(f"cell{i}", None) for i in range(12)]
    shuffled = _boltzmann_order(ranked, temperature=2.0, rng=random.Random(3))
    assert sorted(cell for cell, _ in shuffled) == sorted(cell for cell, _ in ranked)
    assert shuffled != ranked  # a warm draw must actually explore


def test_explore_temperature_is_a_plan_field_the_policy_controls():
    from omniqec_agentic.orchestrator.plan import TurnPlan

    plan = TurnPlan(rationale={"text": "warm up", "cites": ["S-3"]}, explore_temperature=1.5)
    assert plan.explore_temperature == 1.5
    assert TurnPlan(rationale={"text": "x", "cites": ["S-3"]}).explore_temperature == 0.0


def test_explore_temperature_is_range_checked():
    from omniqec_agentic.orchestrator.plan import validate_turn_plan

    raw = {
        "rationale": {"text": "too hot", "cites": ["S-3"]},
        "program_count": 4,
        "explore_temperature": 99.0,
    }
    _, errors = validate_turn_plan(
        raw, source_ids={"S-3"}, known_hashes=set(), ledger_ids=set(),
        families=("bb",), program_cap=24, promote_cap=8,
        quarantined_hashes=set(),
    )
    assert any("explore_temperature" in e for e in errors)


# ---- feedback honesty ------------------------------------------------------

def test_racing_censoring_is_disclosed_to_the_generator():
    """A pilot-depth stop used to render identically to a settled measurement."""

    from omniqec_agentic.funsearch import _slow_result_line

    row = {"program_id": "g1", "n": 72, "k": 12, "d_proxy": 6, "family": "bb",
           "decoder": "beam", "p": 0.005, "per_logical_per_round": 0.004}
    settled = _slow_result_line(dict(row, stopped_by="confirm"))
    censored = _slow_result_line(dict(row, stopped_by="racing"))
    assert "racing_stopped_early" in censored
    assert "racing_stopped_early" not in settled


# ---- curriculum ------------------------------------------------------------

def test_seed_pool_covers_more_than_one_nonabelian_branch():
    from omniqec_agentic.examples.seed_programs import SEEDS
    from omniqec_agentic.sandbox import run_construct_source

    branches = set()
    for index, source in enumerate(SEEDS):
        spec = run_construct_source(source, program_id=f"seed_{index}").constructed.spec
        if spec.get("family") == "lp_nonabelian":
            branches.add(structure_descriptor(spec)["group_class"])
    assert len(branches) >= 2, f"nonabelian seeds still monocultural: {branches}"


def test_every_seed_still_constructs():
    from omniqec_agentic.examples.seed_programs import SEEDS
    from omniqec_agentic.sandbox import run_construct_source

    for index, source in enumerate(SEEDS):
        spec = run_construct_source(source, program_id=f"seed_{index}").constructed.spec
        assert spec.get("family")


def test_syntax_examples_cover_every_nonabelian_branch():
    """A single dihedral example taught that every group takes 2-tuples."""

    from omniqec_agentic.construction.registry import FAMILY_REGISTRY

    text = FAMILY_REGISTRY["lp_nonabelian"].syntax_example_line
    for marker in ("D4", "C7:C3(2)", "C3xD3", "A4"):
        assert marker in text, f"grammar branch {marker} has no worked example"


# ---- evidence rows carry the vocabulary ------------------------------------

def test_evidence_rows_inherit_structure_from_the_archive(tmp_path):
    from omniqec_agentic.workflow.graph import _structure_by_hash

    queue = tmp_path / "all_archive_queue.jsonl"
    queue.write_text(
        json.dumps({"code_hash": "ab" * 8,
                    "spec": {"family": "lp_nonabelian", "group": "C3xD3"}}) + "\n",
        encoding="utf-8",
    )
    state = {"last_cycle": {"artifact_paths": {"all_archive_queue": str(queue)}}}
    mapping = _structure_by_hash(state)
    assert mapping["ab" * 8]["group_class"] == "direct_product"


def test_structure_join_is_absent_not_fatal_when_the_queue_is_missing():
    assert _structure_by_hash_safe() == {}


def _structure_by_hash_safe():
    from omniqec_agentic.workflow.graph import _structure_by_hash

    return _structure_by_hash({"last_cycle": {"artifact_paths": {}}})


# ---- slow-verification ceiling --------------------------------------------

def test_circuit_timeout_is_settable_per_run():
    """A turn whose candidates ALL time out is rejected by the strict audit and
    then repeats forever on resume, so the ceiling must be tunable per budget
    rather than a fixed default (b288 died at turn 3 this way, 2026-08-21)."""

    from omniqec_agentic.config import VerificationConfig

    assert VerificationConfig(circuit_timeout_s=3600.0).circuit_timeout_s == 3600.0
    assert VerificationConfig().circuit_timeout_s > 0


def test_any_timeout_blocks_the_queue_instead_of_censoring_a_candidate():
    """A cheap success must not hide an expensive candidate's timeout."""

    from omniqec_agentic.evaluation.writeback import slow_queue_status

    assert slow_queue_status(
        record_count=3, timeout_candidate_count=3,
        decoder_attempt_candidate_count=0, ok_result_count=0,
    ) == "measurement_blocked"
    assert slow_queue_status(
        record_count=3, timeout_candidate_count=2,
        decoder_attempt_candidate_count=1, ok_result_count=1,
    ) == "measurement_blocked"


def test_default_decoder_is_the_frozen_telescoping_profile():
    """telescoping_structural_v3_7 (docs/06 §1, user decision 2026-09-03)
    measures with exactly one decoder everywhere -- the Mitten paper's
    telescoping decoder (CPU configuration, GARI system) -- at one frozen
    configuration. Changing either is a protocol change, so it must be an
    explicit run parameter rather than an edit to the shared default."""

    from omniqec_agentic.config import (
        TELESCOPING_PROTOCOL_FIELDS,
        DecoderName,
        VerificationConfig,
    )

    verification = VerificationConfig()
    assert verification.decoders == (DecoderName.TELESCOPING,)
    telescoping = verification.decoder_build.telescoping.to_dict()
    assert telescoping == TELESCOPING_PROTOCOL_FIELDS
    beam_only = VerificationConfig(decoders=(DecoderName.BEAM,))
    assert beam_only.decoders == (DecoderName.BEAM,)
    assert beam_only.to_dict()["decoders"] == ["beam"]
