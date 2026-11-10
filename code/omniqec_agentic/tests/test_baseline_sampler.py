"""Proposal sampler of the no-LLM baselines (baselines/sampler.py, plan B §3, 2026-11-06; the paper-literal
stages of 2026-11-10 are pinned in test_baseline_sampler_literal.py)."""

from __future__ import annotations

import json
import time
from collections import Counter

import pytest

from omniqec_agentic.baselines import sampler as S
from omniqec_agentic.construction.build import build_matrices
from omniqec_agentic.construction.grammars.classical_seed import validate_classical_seed
from omniqec_agentic.construction.groups.catalog import get_group, group_order_budget
from omniqec_agentic.errors import ProgramError
from omniqec_agentic.sandbox import run_construct


def _turn(n_budget, families, *, seed_key="test/r00", turn=0, count=48, gens=4):
    return [
        draw
        for gen in range(1, gens + 1)
        for draw in S.draw_generation(
            n_budget=n_budget, families=families, seed_key=seed_key, turn=turn, gen=gen,
            count=count, generations_per_turn=gens,
        )
    ]


def _n_total(spec):
    hx, hz = build_matrices(spec)
    return hx.shape[1] + hx.shape[0] + hz.shape[0]


def test_draws_are_deterministic_and_keyed_by_the_seed_key() -> None:
    first = [d.to_dict() for d in _turn(144, S.FAMILIES, count=24)]
    again = [d.to_dict() for d in _turn(144, S.FAMILIES, count=24)]
    other = [d.to_dict() for d in _turn(144, S.FAMILIES, count=24, seed_key="test/r01")]
    assert first == again
    assert first != other
    json.dumps(first)  # proposals.jsonl rows


def test_a_generation_does_not_depend_on_which_generations_were_drawn_before() -> None:
    kwargs = dict(n_budget=144, families=S.FAMILIES, seed_key="k", turn=3, count=8, generations_per_turn=4)
    alone = [d.to_dict() for d in S.draw_generation(gen=3, **kwargs)]
    for gen in (1, 2):
        S.draw_generation(gen=gen, **kwargs)
    after = [d.to_dict() for d in S.draw_generation(gen=3, **kwargs)]
    assert alone == after
    assert [d["index"] for d in alone] == list(range(3 * 32 + 16, 3 * 32 + 24))  # global index within the run


def test_families_are_stratified_per_turn_and_shuffled() -> None:
    labels = S.turn_family_plan(families=S.FAMILIES, per_turn=3840, seed_key="k", turn=0)
    assert Counter(labels) == {family: 960 for family in S.FAMILIES}
    assert len(set(labels[:960])) == len(S.FAMILIES)  # shuffled, not one family per generation
    draws = _turn(360, S.FAMILIES, count=960)
    assert Counter(d.family for d in draws) == {family: 960 for family in S.FAMILIES}
    assert [d.index for d in draws] == list(range(3840))
    assert [(d.gen, d.slot) for d in draws[958:962]] == [(1, 958), (1, 959), (2, 0), (2, 1)]
    # remainder goes to the first families in FAMILIES order
    assert Counter(S.turn_family_plan(families=S.FAMILIES, per_turn=5, seed_key="k", turn=0)) == {
        "bb": 2, "hgp": 1, "lp_abelian": 1, "lp_nonabelian": 1,
    }


def test_the_callers_family_order_does_not_change_the_draws() -> None:
    forward = [d.to_dict() for d in _turn(98, ("bb", "hgp", "lp_abelian", "lp_nonabelian"), count=16)]
    backward = [d.to_dict() for d in _turn(98, ("lp_nonabelian", "lp_abelian", "hgp", "bb"), count=16)]
    assert forward == backward
    with pytest.raises(ValueError):
        S.canonical_families(("bb", "gb"))
    with pytest.raises(ValueError):
        S.canonical_families(())


def test_bb_only_runs_draw_bb_only() -> None:
    draws = _turn(260, ("bb",), count=96)
    assert {d.family for d in draws} == {"bb"}
    assert all(d.spec is not None for d in draws)  # every BB parameter choice is grammatical
    for draw in draws:
        spec = draw.spec
        assert spec["ell"] <= spec["m"] and 4 * spec["ell"] * spec["m"] == draw.resource
        assert 4 <= len(spec["A"]) + len(spec["B"]) <= 8  # paper appendix D4
        assert draw.skeleton["W"] == len(spec["A"]) + len(spec["B"])
        assert len({tuple(t) for t in spec["A"]}) == len(spec["A"])


@pytest.mark.parametrize("n_budget", [60, 98, 144, 260, 360])
def test_successful_specs_build_with_the_declared_resource(n_budget) -> None:
    draws = _turn(n_budget, S.FAMILIES, count=48)
    built = Counter()
    with group_order_budget(n_budget):
        for draw in draws:
            assert draw.resource is not None and draw.resource <= n_budget
            if draw.spec is None:
                continue
            assert draw.spec["family"] == draw.family
            assert _n_total(draw.spec) == draw.resource
            built[draw.family] += 1
    assert {"bb", "lp_abelian", "lp_nonabelian"} <= set(built)


def test_failed_draws_carry_a_reason_and_fail_in_the_sandbox() -> None:
    draws = _turn(144, ("hgp", "lp_nonabelian"), count=48)
    failed = [d for d in draws if d.spec is None]
    assert failed, "the HGP column-degree check fails often at these seed sizes"
    for draw in failed[:10]:
        assert draw.failure and draw.resource is not None and draw.skeleton
        with pytest.raises(ProgramError) as excinfo:
            run_construct(S.program_source(draw))
        assert f"baseline sampler: {draw.failure}" in str(excinfo.value)
    reasons = {d.failure.split(": ", 1)[-1] for d in failed}
    assert reasons <= {
        "column degrees outside [2, 4]", "duplicate rows", "supports of A and B generate a proper subgroup",
    }


@pytest.mark.parametrize("n_budget", [144, 360])
def test_program_source_round_trips_through_the_sandbox(n_budget) -> None:
    draws = [d for d in _turn(n_budget, S.FAMILIES, count=24) if d.spec is not None]
    with group_order_budget(n_budget):
        seen = set()
        for draw in draws:
            if draw.family in seen and len(seen) < 4:
                continue
            seen.add(draw.family)
            assert run_construct(S.program_source(draw)) == draw.spec
    assert seen >= {"bb", "lp_abelian", "lp_nonabelian"}


def test_hgp_specs_respect_the_classical_seed_grammar() -> None:
    draws = [d for d in _turn(180, ("hgp",), count=480) if d.spec is not None]
    assert draws
    for draw in draws:
        for label in ("H1", "H2"):
            seed = draw.spec["seeds"][label]
            assert validate_classical_seed(seed, label=label) == seed
            rows, cols = len(seed), len(seed[0])
            assert [rows, cols] == draw.skeleton[label]
            assert sum(map(sum, seed)) == draw.skeleton[f"{label}_edges"]


def test_lp_cells_respect_the_grammar() -> None:
    draws = [d for d in _turn(360, ("lp_abelian", "lp_nonabelian"), count=480)]
    weights = Counter()
    for draw in draws:
        order = draw.skeleton["group_order"]
        for key in ("cell_weights_A", "cell_weights_B"):
            for row in draw.skeleton[key]:
                for w in row:
                    assert 1 <= w <= min(8, order)
                    weights[w] += 1
        if draw.spec is not None:
            assert [len(draw.spec["A"]), len(draw.spec["A"][0])] == draw.skeleton["shape_A"]
            assert all(cell for row in draw.spec["A"] + draw.spec["B"] for cell in row)
    assert set(weights) == set(range(1, 8))  # lifted w_max <= 8 leaves at most 7 per cell (literal_uniform_v2)


def test_every_feasible_resource_is_reachable() -> None:
    for family, n_budget, count in (("bb", 60, 1200), ("lp_nonabelian", 98, 1200), ("hgp", 98, 1200)):
        draws = _turn(n_budget, (family,), count=count, gens=1)
        assert {d.resource for d in draws} == set(S.resource_catalog(family, n_budget))


def test_catalogue_skeletons_have_their_resource() -> None:
    n_budget = 180
    for r, items in S.resource_catalog("bb", n_budget).items():
        for item in items:
            assert item["ell"] <= item["m"] and 4 * item["ell"] * item["m"] == r <= n_budget
    for r, items in S.resource_catalog("hgp", n_budget).items():
        for item in items:
            (r1, n1), (r2, n2) = item["H1"], item["H2"]
            assert (r1 + n1) * (r2 + n2) == r <= n_budget
    for family in ("lp_abelian", "lp_nonabelian"):
        for r, items in S.resource_catalog(family, n_budget).items():
            for item in items:
                size = (sum(item["shape_A"])) * (sum(item["shape_B"]))
                assert size * item["group_order"] == r <= n_budget
    assert min(S.resource_catalog("hgp", 60)) == 36  # two 3x3 seeds
    assert set(S.resource_catalog("bb", 60)) == {4 * cells for cells in range(2, 16)}


@pytest.mark.parametrize(
    "rows, cols, feasible",
    [(2, 2, False), (2, 5, False), (5, 2, False), (3, 3, True), (4, 3, True), (5, 3, False),
     (3, 7, False), (4, 8, True), (8, 4, False), (7, 4, True), (5, 10, True), (11, 5, False), (12, 20, True)],
)
def test_classical_seed_feasibility(rows, cols, feasible) -> None:
    witness = S.classical_seed_witness(rows, cols)
    assert (witness is not None) == feasible
    if witness is not None:
        assert validate_classical_seed(witness, label="H") == witness
        assert (len(witness), len(witness[0])) == (rows, cols)


def test_abelian_catalogue_is_one_entry_per_isomorphism_class() -> None:
    by_order = Counter(S._product(factors) for factors in S.abelian_catalog(72))
    assert by_order[8] == 3 and by_order[12] == 2 and by_order[16] == 5 and by_order[64] == 11 and by_order[72] == 6
    for factors in S.abelian_catalog(72):
        assert all(b % a == 0 for a, b in zip(factors, factors[1:]))


def test_nonabelian_catalogue_is_one_resolvable_name_per_isomorphism_class() -> None:
    names = S.nonabelian_catalog(90)
    groups = [get_group(name, max_order=90) for name in names]
    assert all(not group.is_abelian and group.name == name for group, name in zip(groups, names))
    buckets: dict = {}
    for group in groups:
        buckets.setdefault(S.group_isomorphism_invariants(group), []).append(group)
    for bucket in buckets.values():
        for i, first in enumerate(bucket):
            for second in bucket[i + 1:]:
                assert not S.groups_isomorphic(first, second), (first.name, second.name)
    by_order = lambda order: {g.name for g in groups if g.order == order}  # noqa: E731
    assert by_order(12) == {"D6", "A4", "C3:C4(2)"}
    assert by_order(20) == {"D10", "C5:C4(2)", "C5:C4(4)"}
    assert by_order(16) == {"D8", "C2xD4", "C4:C4(3)", "C8:C2(3)", "C8:C2(5)"}
    # a known alias the catalogue grammar accepts twice: C3 x D6 = C3 x C2 x D3 = C6 x D3
    assert S.groups_isomorphic(get_group("C3xD6"), get_group("C6xD3"))
    assert ("C3xD6" in names) != ("C6xD3" in names)
    assert not S.groups_isomorphic(get_group("D12"), get_group("C2xD6"))


def test_catalog_digest_is_stable_and_covers_the_inputs() -> None:
    digest = S.catalog_digest(S.FAMILIES, 144)
    assert digest == S.catalog_digest(list(reversed(S.FAMILIES)), 144)
    assert digest != S.catalog_digest(("bb",), 144)
    assert digest != S.catalog_digest(S.FAMILIES, 180)


def test_a_generation_of_960_draws_is_fast_once_the_catalogues_exist() -> None:
    _turn(360, S.FAMILIES, count=8, gens=1)  # build the catalogues and groups
    start = time.perf_counter()
    S.draw_generation(n_budget=360, families=S.FAMILIES, seed_key="speed", turn=1, gen=2, count=960,
                      generations_per_turn=4)
    assert time.perf_counter() - start < 3.0
