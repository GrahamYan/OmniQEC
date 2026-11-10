"""Paper-literal stages of the baseline sampler (baselines/sampler.py literal_uniform_v2, 2026-11-10).

Appendix D5 (Random) and D4 (BO) name the stages of a draw; the user's decision is to read them literally: first
uniform over the sizes, then uniform over the groups. Every stage is checked against an exact enumeration of its
admissible set, and the draws against that set with a chi-square test (the draws are deterministic, so a test either
always passes or always fails)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from scipy.stats import chisquare

from omniqec_agentic.baselines import sampler as S
from omniqec_agentic.construction.grammars.ga_seed import DEFAULT_ALLOWED_SHAPES
from omniqec_agentic.construction.groups.catalog import get_group, group_order_budget

P_MIN = 1e-4


def _skeletons(family: str, n_budget: int, draws: int, seed: int = 0) -> list[tuple[int, dict]]:
    rng = np.random.default_rng(seed)
    return [S._draw_skeleton(family, n_budget, rng) for _ in range(draws)]


def _assert_uniform(observed: Counter, support: list, label: str) -> None:
    assert set(observed) <= set(support), (label, set(observed) - set(support))
    counts = [observed.get(item, 0) for item in support]
    total = sum(counts)
    assert total > 0, label
    if len(support) == 1:
        return
    stat, p_value = chisquare(counts, [total / len(support)] * len(support))
    assert p_value > P_MIN, (label, p_value, counts)


def _factor_pairs(cells: int) -> list[tuple[int, int]]:
    return [(ell, cells // ell) for ell in range(1, cells + 1) if cells % ell == 0 and ell <= cells // ell]


def _shape_size(shape) -> int:
    return int(shape[0]) + int(shape[1])


def _abelian_number(n: int) -> bool:
    """Every group of order n is abelian iff n = prod p_i^a_i with a_i <= 2 and p_i does not divide p_j^a_j - 1."""

    factors: dict[int, int] = {}
    rest, p = n, 2
    while p * p <= rest:
        while rest % p == 0:
            factors[p] = factors.get(p, 0) + 1
            rest //= p
        p += 1
    if rest > 1:
        factors[rest] = factors.get(rest, 0) + 1
    if any(a > 2 for a in factors.values()):
        return False
    return not any((q ** a - 1) % p == 0 for p in factors for q, a in factors.items() if q != p)


# ------------------------------------------------------------------ id, family split
def test_the_sampler_id_is_bumped_and_an_older_run_cannot_resume() -> None:
    from omniqec_agentic.config import VerificationConfig, formal_racing
    from omniqec_agentic.workflow.checkpoint import validate_resume_protocol
    from omniqec_agentic.workflow.state import build_protocol_snapshot

    assert S.SAMPLER_ID == "literal_uniform_v2"
    assert S.PRIOR_CONSTANTS["bb_total_weight"] == [4, 8]

    def snapshot(sampler_id: str, digest: str) -> dict:
        baseline = {"method": "random", "families": list(S.FAMILIES), "sampler_id": sampler_id,
                    "sampler_catalog_sha256": digest, "seed_key": f"{sampler_id}|memory|N98|r01"}
        return build_protocol_snapshot(
            verification=VerificationConfig(), racing=formal_racing(25_000), max_turns=30, proposal_count=960,
            generations_per_turn=4, slow_workers=8, n_budget=98, required_provider="baseline_sampler",
            required_model=sampler_id, search_baseline=baseline,
        )

    current = snapshot(S.SAMPLER_ID, S.catalog_digest(S.FAMILIES, 98))
    validate_resume_protocol(current, current)
    with pytest.raises(ValueError, match="required_model.*search_baseline|search_baseline.*required_model"):
        validate_resume_protocol(snapshot("fixed_prior_v2", "0" * 64), current)


def test_the_digest_and_the_draws_depend_on_the_sampler_id(monkeypatch) -> None:
    digest = S.catalog_digest(S.FAMILIES, 98)
    draws = [d.to_dict() for d in S.draw_generation(n_budget=98, families=S.FAMILIES, seed_key="k", turn=0, gen=1,
                                                    count=16, generations_per_turn=4)]
    monkeypatch.setattr(S, "SAMPLER_ID", "fixed_prior_v2")
    assert S.catalog_digest(S.FAMILIES, 98) != digest
    older = [d.to_dict() for d in S.draw_generation(n_budget=98, families=S.FAMILIES, seed_key="k", turn=0, gen=1,
                                                    count=16, generations_per_turn=4)]
    assert [d["skeleton"] for d in older] != [d["skeleton"] for d in draws]


def test_a_run_gives_each_family_one_quarter_of_its_115200_attempts() -> None:
    """D5: "One quarter of these attempts is allocated to each construction family" (30 x 3,840)."""

    totals = Counter()
    for turn in range(30):
        labels = S.turn_family_plan(families=S.FAMILIES, per_turn=3840, seed_key="r01", turn=turn)
        assert Counter(labels) == {family: 960 for family in S.FAMILIES}
        totals.update(labels)
    assert totals == {family: 28_800 for family in S.FAMILIES} and sum(totals.values()) == 115_200


# ------------------------------------------------------------------ BB (and BO)
@pytest.mark.parametrize("n_budget", [60, 98, 260])
def test_bb_pairs_are_every_admissible_pair_once(n_budget) -> None:
    expected = [(ell, m) for cells in range(2, n_budget // 4 + 1) for ell, m in _factor_pairs(cells)]
    pairs = [(p["ell"], p["m"]) for p in S.stage_catalog("bb", n_budget)["pairs"]]
    assert pairs == expected and len(set(pairs)) == len(pairs)
    assert all(ell <= m and 4 * ell * m <= n_budget and ell * m >= 2 for ell, m in pairs)


@pytest.mark.parametrize("n_budget", [60, 260])
def test_bb_dimensions_are_uniform_over_the_admissible_pairs(n_budget) -> None:
    """D4 / D5: (l, m) uniform over the pairs with 4 l m <= N (l <= m, the validator's canonical form) -- not R first."""

    pairs = [(p["ell"], p["m"]) for p in S.stage_catalog("bb", n_budget)["pairs"]]
    observed = Counter((sk["ell"], sk["m"]) for resource, sk in _skeletons("bb", n_budget, 200 * len(pairs)))
    _assert_uniform(observed, pairs, f"bb N={n_budget}")
    # the old R-first rule gives every R the same mass; here R = 4 l m carries mass proportional to its pair count
    by_resource = Counter(4 * ell * m for ell, m in pairs)
    assert max(by_resource.values()) >= 3


def test_bb_term_numbers_are_uniform_over_the_admissible_splits() -> None:
    """D4: the numbers of terms of A and B, a combined total between four and eight: (wA, wB) uniform over the
    admissible pairs (so the total W = 4..8 has mass 3 : 4 : 5 : 6 : 7)."""

    assert S._bb_weight_pairs(2) == [(2, 2)]
    assert S._bb_weight_pairs(3) == [(1, 3), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    big = S._bb_weight_pairs(36)
    assert big == [(a, b) for a in range(1, 8) for b in range(1, 8) if 4 <= a + b <= 8] and len(big) == 25
    draws = [
        d for gen in range(1, 5)
        for d in S.draw_generation(n_budget=60, families=("bb",), seed_key="w", turn=0, gen=gen, count=960,
                                   generations_per_turn=4)
    ]
    assert all(d.spec is not None for d in draws)
    big_cells = [d for d in draws if d.skeleton["ell"] * d.skeleton["m"] >= 7]
    observed = Counter((d.skeleton["wA"], d.skeleton["wB"]) for d in big_cells)
    _assert_uniform(observed, big, "bb (wA, wB)")
    for d in draws:
        assert (len(d.spec["A"]), len(d.spec["B"])) == (d.skeleton["wA"], d.skeleton["wB"])


def test_bb_supports_are_uniform_subsets() -> None:
    cells = 2 * 3
    rng = np.random.default_rng(5)
    observed = Counter()
    for _ in range(6000):
        spec, chosen = S._draw_bb({"ell": 2, "m": 3}, rng)
        if chosen["wA"] == 2:
            observed[tuple(sorted(tuple(t) for t in spec["A"]))] += 1
    support = sorted({tuple(sorted(c)) for c in (
        tuple((t // 3, t % 3) for t in combo) for combo in combinations(range(cells), 2))})
    assert len(support) == comb(cells, 2)
    _assert_uniform(observed, support, "bb A support")


def test_bo_draws_its_bb_codes_from_the_same_stages() -> None:
    """The BO baseline's proposals are the BB-only runs of the same sampler (funsearch._baseline_sampler_batch with
    the run's families), so its (l, m) draw is the same uniform rule."""

    n_budget = 144
    pairs = [(p["ell"], p["m"]) for p in S.stage_catalog("bb", n_budget)["pairs"]]
    draws = [
        d for gen in range(1, 5)
        for d in S.draw_generation(n_budget=n_budget, families=("bb",), seed_key="bo", turn=2, gen=gen, count=960,
                                   generations_per_turn=4)
    ]
    _assert_uniform(Counter((d.spec["ell"], d.spec["m"]) for d in draws), pairs, "bo (l, m)")


# ------------------------------------------------------------------ HGP
def test_hgp_seed_dimensions_are_uniform_over_the_budget_admissible_pairs() -> None:
    n_budget = 98
    pairs = [(tuple(p["H1"]), tuple(p["H2"])) for p in S.stage_catalog("hgp", n_budget)["pairs"]]
    shapes = sorted({shape for pair in pairs for shape in pair})
    feasible = [s for s in shapes if S.classical_seed_witness(*s) is not None]
    assert feasible == shapes
    expected = sorted((a, b) for a in shapes for b in shapes if _shape_size(a) * _shape_size(b) <= n_budget)
    assert sorted(pairs) == expected and len(set(pairs)) == len(pairs)
    observed = Counter((tuple(sk["H1"]), tuple(sk["H2"])) for resource, sk in _skeletons("hgp", n_budget, 40 * len(pairs)))
    _assert_uniform(observed, pairs, "hgp seed shapes")


@pytest.mark.parametrize("rows, cols", [(3, 4), (4, 3), (5, 6)])
def test_hgp_check_weights_are_uniform_over_the_admissible_sequences(rows, cols) -> None:
    """Row degrees in [2, min(4, cols)] whose sum the columns can carry (2 cols <= sum <= min(4, rows) cols)."""

    cap = min(4, cols)
    support = [
        seq for seq in product(range(2, cap + 1), repeat=rows)
        if 2 * cols <= sum(seq) <= min(4, rows) * cols
    ]
    rng = np.random.default_rng(rows * 10 + cols)
    observed = Counter(tuple(S._check_weights(rng, rows, cols)) for _ in range(60 * len(support)))
    _assert_uniform(observed, support, f"hgp check weights {rows}x{cols}")


def test_hgp_entries_are_uniform_given_the_check_weights() -> None:
    rng = np.random.default_rng(3)
    observed = Counter(tuple(S._seed_entries(rng, 5, [2])[0]) for _ in range(5000))
    support = [tuple(1 if c in combo else 0 for c in range(5)) for combo in combinations(range(5), 2)]
    _assert_uniform(observed, support, "hgp row support")


def test_hgp_draws_record_their_stages_and_fail_only_on_the_grammar() -> None:
    draws = [d for gen in range(1, 5) for d in S.draw_generation(
        n_budget=98, families=("hgp",), seed_key="h", turn=0, gen=gen, count=240, generations_per_turn=4)]
    reasons = Counter((d.failure or "").split(": ", 1)[-1] for d in draws if d.spec is None)
    assert set(reasons) <= {"column degrees outside [2, 4]", "duplicate rows"}
    assert any(d.spec is not None for d in draws)
    for d in draws:
        for label in ("H1", "H2"):
            degrees = d.skeleton[f"{label}_row_degrees"]
            rows, cols = d.skeleton[label]
            assert len(degrees) == rows and all(2 <= x <= min(4, cols) for x in degrees)
            assert 2 * cols <= sum(degrees) == d.skeleton[f"{label}_edges"] <= min(4, rows) * cols
            assert sum(d.skeleton[f"{label}_col_degrees"]) == sum(degrees)
            if d.spec is not None:
                assert [sum(row) for row in d.spec["seeds"][label]] == degrees


# ------------------------------------------------------------------ LP
@pytest.mark.parametrize("family", ["lp_abelian", "lp_nonabelian"])
@pytest.mark.parametrize("n_budget", [60, 98, 260])
def test_lp_stages_are_the_admissible_sets(family, n_budget) -> None:
    stages = S.stage_catalog(family, n_budget)
    max_order = n_budget // 4
    groups_by_order = {int(order): groups for order, groups in stages["groups_by_order"].items()}
    if family == "lp_abelian":
        assert sorted(groups_by_order) == list(range(2, max_order + 1))  # every order has an abelian group
        for order, groups in groups_by_order.items():
            assert all(S._product(g["factors"]) == order for g in groups)
            assert len(groups) == sum(1 for f in S.abelian_catalog(max_order) if S._product(f) == order)
    else:
        # the orders left out are exactly those all of whose groups are abelian (catalogue complete to |G| = 65)
        assert sorted(groups_by_order) == [g for g in range(2, max_order + 1) if not _abelian_number(g)]
        for order, groups in groups_by_order.items():
            assert sorted(groups) == sorted(n for n in S.nonabelian_catalog(max_order)
                                            if get_group(n, max_order=max_order).order == order)
    orders = sorted(groups_by_order)
    expected_pairs = []
    for a in DEFAULT_ALLOWED_SHAPES:
        for b in DEFAULT_ALLOWED_SHAPES:
            factor = _shape_size(a) * _shape_size(b)
            admissible = [g for g in orders if factor * g <= n_budget]
            if admissible:
                expected_pairs.append({"shape_A": list(a), "shape_B": list(b), "factor": factor, "orders": admissible})
    assert stages["shape_pairs"] == expected_pairs
    if n_budget >= 98:
        assert len(expected_pairs) == 16  # the paper's points: every shape pair is admissible for both families


def test_the_catalogue_misses_order_75_for_non_abelian_groups() -> None:
    """The one catalogue gap up to |G| = 90: (C5 x C5) : C3 is neither dihedral, metacyclic nor a listed product, so
    at N >= 300 the non-abelian order 75 is left out (not an order the paper's Random points N = 98 / 180 / 260 reach)."""

    groups = S.stage_catalog("lp_nonabelian", 360)["groups_by_order"]
    missing = [g for g in range(2, 91) if str(g) not in groups and not _abelian_number(g)]
    assert missing == [75]


@pytest.mark.parametrize("family", ["lp_abelian", "lp_nonabelian"])
def test_lp_shape_then_order_then_group_are_each_uniform(family) -> None:
    n_budget = 98
    stages = S.stage_catalog(family, n_budget)
    pairs = [(tuple(p["shape_A"]), tuple(p["shape_B"])) for p in stages["shape_pairs"]]
    orders_of = {(tuple(p["shape_A"]), tuple(p["shape_B"])): p["orders"] for p in stages["shape_pairs"]}
    draws = _skeletons(family, n_budget, 24_000, seed=11)
    shapes = Counter((tuple(sk["shape_A"]), tuple(sk["shape_B"])) for _, sk in draws)
    _assert_uniform(shapes, pairs, f"{family} shape pairs")
    for pair in [((1, 1), (1, 1)), ((1, 2), (2, 1)), ((2, 2), (2, 2))]:
        orders = Counter(sk["group_order"] for _, sk in draws if (tuple(sk["shape_A"]), tuple(sk["shape_B"])) == pair)
        _assert_uniform(orders, orders_of[pair], f"{family} orders of {pair}")
    for order in (12, 16, 24):
        support = [str(g) if isinstance(g, str) else tuple(g["factors"]) for g in stages["groups_by_order"][str(order)]]
        assert len(support) >= 2
        groups = Counter(
            sk["group"] if isinstance(sk["group"], str) else tuple(sk["group"]["factors"])
            for _, sk in draws if sk["group_order"] == order
        )
        _assert_uniform(groups, support, f"{family} groups of order {order}")
    for resource, sk in draws:
        assert resource == _shape_size(sk["shape_A"]) * _shape_size(sk["shape_B"]) * sk["group_order"] <= n_budget


def _admissible_weights(shape_a, shape_b, cap):
    """Brute force: every joint (A, B) cell-weight assignment in 1..cap whose lifted w_max is <= 8."""

    out = []
    na, nb = shape_a[0] * shape_a[1], shape_b[0] * shape_b[1]
    for flat in product(range(1, cap + 1), repeat=na + nb):
        wa = tuple(tuple(flat[r * shape_a[1]:(r + 1) * shape_a[1]]) for r in range(shape_a[0]))
        wb = tuple(tuple(flat[na + r * shape_b[1]:na + (r + 1) * shape_b[1]]) for r in range(shape_b[0]))
        rows_a = [sum(r) for r in wa]
        cols_a = [sum(r[j] for r in wa) for j in range(shape_a[1])]
        rows_b = [sum(r) for r in wb]
        cols_b = [sum(r[j] for r in wb) for j in range(shape_b[1])]
        if max(max(rows_a) + max(cols_b), max(rows_b) + max(cols_a)) <= 8:
            out.append((wa, wb))
    return out


@pytest.mark.parametrize("shape_a, shape_b, order", [
    ((1, 1), (1, 1), 30), ((1, 1), (1, 1), 3), ((1, 2), (2, 1), 30), ((2, 1), (1, 1), 12), ((2, 2), (2, 2), 3),
    ((1, 1), (2, 2), 5), ((2, 2), (1, 2), 4),
])
def test_lp_cell_weight_admissible_set_is_every_assignment_with_w_at_most_8(shape_a, shape_b, order) -> None:
    """User decision 2026-11-10: the cell weights are drawn inside the w_max <= 8 constraint (decided by the weights
    alone, as BB's 4..8 terms); the admissible set is checked against brute-force enumeration."""

    cap = min(8, order)
    assert S.lp_cell_weight_count(shape_a, shape_b, order) == len(_admissible_weights(shape_a, shape_b, cap))
    assert S.lp_cell_weight_count((1, 1), (1, 1), 30) == 28  # 1 <= wA, wB and wA + wB <= 8


@pytest.mark.parametrize("shape_a, shape_b, order", [((1, 1), (1, 1), 30), ((1, 2), (2, 1), 30), ((1, 1), (2, 1), 4)])
def test_lp_cell_weights_are_uniform_over_the_admissible_joint_assignments(shape_a, shape_b, order) -> None:
    support = _admissible_weights(shape_a, shape_b, min(8, order))
    rng = np.random.default_rng(len(support))
    observed = Counter()
    for _ in range(80 * len(support)):
        wa, wb = S._draw_cell_weights(shape_a, shape_b, order, rng)
        observed[(tuple(map(tuple, wa)), tuple(map(tuple, wb)))] += 1
    _assert_uniform(observed, support, f"lp cell weights {shape_a} {shape_b} |G|={order}")


def test_lp_draws_respect_the_check_weight_and_the_elements_are_uniform() -> None:
    from omniqec_agentic.verification.structural import structural_check_spec

    n_budget = 260
    draws = [d for gen in range(1, 5) for d in S.draw_generation(
        n_budget=n_budget, families=("lp_abelian", "lp_nonabelian"), seed_key="c", turn=0, gen=gen, count=480,
        generations_per_turn=4)]
    checked = 0
    with group_order_budget(n_budget):
        for d in draws:
            order = d.skeleton["group_order"]
            wa, wb = d.skeleton["cell_weights_A"], d.skeleton["cell_weights_B"]
            assert all(1 <= w <= min(8, order) for row in wa + wb for w in row)
            assert S._lp_max_check_weight(wa, wb) <= 8
            if d.spec is None:
                continue
            assert [len(d.spec["A"]), len(d.spec["A"][0])] == d.skeleton["shape_A"]
            assert [[len(cell) for cell in row] for row in d.spec["A"]] == wa
            if checked < 150:
                result = structural_check_spec("t", d.spec, total_budget=n_budget, max_check_weight=None)
                assert result.max_check_weight == S._lp_max_check_weight(wa, wb) <= 8
                checked += 1
    assert checked == 150
    group = get_group("D3")
    rng = np.random.default_rng(2)
    observed = Counter(tuple(sorted(tuple(e) for e in S._draw_cells(group, [[2]], rng)[0][0])) for _ in range(3000))
    support = sorted(tuple(sorted(pair)) for pair in combinations(
        [tuple(e) for e in group.elements], 2))
    _assert_uniform(observed, support, "lp cell elements")


def test_lp_draws_build_with_their_resource() -> None:
    from omniqec_agentic.construction.build import build_matrices

    for family in ("lp_abelian", "lp_nonabelian"):
        draws = S.draw_generation(n_budget=180, families=(family,), seed_key="b", turn=0, gen=1, count=200,
                                  generations_per_turn=1)
        with group_order_budget(180):
            built = 0
            for d in draws:
                if d.spec is None:
                    assert d.failure == "supports of A and B generate a proper subgroup"
                    continue
                hx, hz = build_matrices(d.spec)
                assert hx.shape[1] + hx.shape[0] + hz.shape[0] == d.resource
                built += 1
        assert built > 100
