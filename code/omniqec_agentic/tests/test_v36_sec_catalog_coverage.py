"""v3.6 (docs/06 §3 M1 / M3 / minor, §4.3): staged SEC certification, the
budget-derived group-order cap, the metacyclic direct-product dedup, the
coverage instrument with shapes / utilisation bands, and the Mitten rows of
the known-code table.

Evidence behind each block: docs/05_large_code_fairness_audit_20260903.md
§2.3 (single-seed ISD does not converge on n >= 150 extended kernels and the
argmax then selects noise), §4 M1 / M3 / 次要, §5 item 3.
"""

from __future__ import annotations

import inspect
import json
import re

import numpy as np
import pytest

from omniqec_agentic.config import (
    MAX_GROUP_ORDER,
    RunConfig,
    CodeFamily,
    HardwareConfig,
    HardwareTarget,
    SecPolicyConfig,
    group_order_cap_for_budget,
)
from omniqec_agentic.construction import build_compiled_code, build_matrices
from omniqec_agentic.construction.groups import catalog
from omniqec_agentic.construction.groups.catalog import (
    get_group,
    group_invariants,
    group_order_budget,
    metacyclic_direct_product_name,
)
from omniqec_agentic.construction.registry import FAMILY_REGISTRY, format_contract_lines_for_budget
from omniqec_agentic.construction.structure import (
    UTILISATION_LABELS,
    spec_total_qubits,
    structure_key,
    utilisation_band,
)
from omniqec_agentic.errors import ProgramError
from omniqec_agentic.memory.coverage import (
    catalog_gaps,
    coverage_fact_lines,
    coverage_report,
    feasible_branch_shape_pairs,
)
from omniqec_agentic.sec import twobasis_rank as tr
from omniqec_agentic.sec.layers import build_layer_sets
from omniqec_agentic.sec.twobasis_rank import (
    basis_residual_suffixes,
    certify_basis_order,
    rank_twobasis_schedules,
)
from test_prompt_neutrality import BANNED_PHRASES, GOAL_RE

BB_SMALL_SPEC = {
    "family": "bb", "ell": 3, "m": 3,
    "A": [[1, 0], [0, 1], [2, 2]], "B": [[0, 2], [2, 0], [1, 1]],
}
MITTEN_150 = {
    "family": "lp_nonabelian", "group": "C5xD3",
    "A": [[[[0, 0, 0], [1, 2, 0], [2, 1, 1]], [[0, 0, 0], [1, 0, 0], [0, 1, 1]]]],
    "B": [[[[4, 2, 0], [3, 1, 0], [1, 2, 0]], [[0, 0, 0], [4, 0, 0], [1, 1, 1]]]],
}
MITTEN_200 = {
    "family": "lp_nonabelian", "group": "C4xD5",
    "A": [[[[2, 1, 0], [1, 3, 1], [1, 2, 1]], [[0, 0, 0], [1, 2, 0], [2, 2, 0]]]],
    "B": [[[[3, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 0, 0], [0, 2, 0], [3, 3, 1]]]],
}


def _mats(compiled):
    return np.array(compiled.hx, dtype=np.uint8), np.array(compiled.hz, dtype=np.uint8)


# ---------------------------------------------------------------------------
# M1: staged certification on a deterministic fake ISD
# ---------------------------------------------------------------------------


class _FakeIsd:
    """A ``_proxy_side`` stand-in that answers per (basis, candidate, seed slot)
    from a table and records every call. The ranker seeds
    ``default_rng(sec_cert_seed + index*7 + side + 1000*s)``; the first draw
    of such a generator identifies the seed exactly."""

    def __init__(self, table: dict[str, dict[int, list[int | None]]], *, seed0: int, candidates: int, seeds: int):
        self.table = table
        self.calls: list[tuple[str, int, int, int]] = []
        self.by_draw: dict[int, tuple[str, int, int]] = {}
        for kind, side in (("X", 0), ("Z", 1)):
            for index in range(candidates):
                for s in range(2 * seeds):
                    seed = seed0 + index * 7 + side + 1000 * s
                    draw = int(np.random.default_rng(seed).integers(0, 2**31))
                    self.by_draw[draw] = (kind, index, s)

    def __call__(self, kernel, logical_parity, iters, rng):
        kind, index, s = self.by_draw[int(rng.integers(0, 2**31))]
        self.calls.append((kind, index, s, int(iters)))
        value = self.table[kind][index][s]
        return 10**9 if value is None else int(value)


def _policy(**overrides) -> SecPolicyConfig:
    base = dict(
        sec_candidates=2, sec_cert_iters=5, sec_cert_seed=0, sec_cert_seeds=3,
        sec_cert_iters_scale=0.0, sec_refine_top=2, sec_refine_iters=9,
    )
    base.update(overrides)
    return SecPolicyConfig(**base)


@pytest.fixture(scope="module")
def bb_small():
    compiled = build_compiled_code(BB_SMALL_SPEC)
    hx, hz = _mats(compiled)
    return compiled, hx, hz


def test_inflated_single_seed_estimate_does_not_win(bb_small, monkeypatch):
    """docs/05 §2.3: one lucky 400-iteration run reported d_ext 19 > d = 12 on
    Mitten200 and the argmax picked it. Candidate 0 here has an inflated
    first seed (9) but a true upper bound of 4 on its other seeds; candidate
    1 answers 6 on every seed. Under the v3.5 single-seed reading 0 wins;
    under the v3.6 multi-seed MIN, 1 wins."""

    compiled, hx, hz = bb_small
    table = {
        "X": {0: [9, 4, 4, 9, 4, 4], 1: [6, 6, 6, 6, 6, 6]},
        "Z": {0: [9, 4, 4, 9, 4, 4], 1: [6, 6, 6, 6, 6, 6]},
    }
    fake = _FakeIsd(table, seed0=0, candidates=2, seeds=3)
    monkeypatch.setattr(tr, "_proxy_side", fake)
    monkeypatch.setattr(tr, "_delta_one_hit", lambda context, suffixes_w: False)

    staged = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=_policy())
    assert (staged.schedule.candidate_index_x, staged.schedule.candidate_index_z) == (1, 1)
    assert (staged.d_circ_ext_x, staged.d_circ_ext_z) == (6, 6)
    assert staged.metadata["per_candidate"]["X"] == [[0, 4, 4, False], [1, 6, 6, False]]
    assert staged.metadata["converged_x"] is True

    fake.calls.clear()
    single = rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks,
        policy=_policy(sec_cert_seeds=1, sec_refine_top=0),
    )
    assert (single.schedule.candidate_index_x, single.schedule.candidate_index_z) == (0, 0)
    assert single.d_circ_ext_x == 9  # the inflated value the v3.5 reading recorded
    assert single.metadata["per_candidate"]["X"] == [[0, 9, None, False], [1, 6, None, False]]
    assert single.metadata["converged_x"] is None  # nothing refined


def test_stage1_iterations_scale_with_the_kernel_dimension(bb_small, monkeypatch):
    compiled, hx, hz = bb_small
    table = {"X": {0: [5] * 6, 1: [5] * 6}, "Z": {0: [5] * 6, 1: [5] * 6}}
    fake = _FakeIsd(table, seed0=0, candidates=2, seeds=3)
    monkeypatch.setattr(tr, "_proxy_side", fake)
    monkeypatch.setattr(tr, "_delta_one_hit", lambda context, suffixes_w: False)
    ranked = rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks,
        policy=_policy(sec_cert_iters=5, sec_cert_iters_scale=4.0, sec_refine_iters=9),
    )
    for kind in ("X", "Z"):
        kernel_dim = ranked.metadata["kernel_dim"][kind]
        assert kernel_dim > 1
        assert ranked.metadata["stage1_iters"][kind] == max(5, 4 * kernel_dim)
    stage1 = {c[3] for c in fake.calls if c[2] < 3}
    refine = {c[3] for c in fake.calls if c[2] >= 3}
    assert stage1 == {max(5, 4 * kd) for kd in ranked.metadata["kernel_dim"].values()}
    assert refine == {9}  # refine iterations are NOT scaled
    # every stage-1 candidate saw all three seeds; refine used the fresh ones
    assert {c[2] for c in fake.calls if c[3] == 9} == {3, 4, 5}


def test_d_ext_is_clamped_to_the_code_distance_and_convergence_is_recorded(bb_small, monkeypatch):
    compiled, hx, hz = bb_small
    # candidate 0 is chosen (10 on every seed); its refine seeds disagree in Z
    table = {
        "X": {0: [10, 10, 10, 10, 10, 10], 1: [3, 3, 3, 3, 3, 3]},
        "Z": {0: [10, 10, 10, 10, 9, 10], 1: [3, 3, 3, 3, 3, 3]},
    }
    monkeypatch.setattr(tr, "_proxy_side", _FakeIsd(table, seed0=0, candidates=2, seeds=3))
    monkeypatch.setattr(tr, "_delta_one_hit", lambda context, suffixes_w: False)
    ranked = rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks, policy=_policy(), d_code_proxy=4,
    )
    md = ranked.metadata
    assert (ranked.d_circ_ext_x, ranked.d_circ_ext_z) == (4, 4)
    assert (md["d_ext_x_unclamped"], md["d_ext_z_unclamped"]) == (10, 9)
    assert md["clamped_x"] is True and md["clamped_z"] is True
    assert md["schedule_degraded_x"] is False and md["schedule_degraded_z"] is False
    assert md["converged_x"] is True and md["converged_z"] is False
    sched = ranked.schedule.to_metadata()
    assert sched["d_ext_x"] == 4 and sched["converged_z"] is False
    assert json.loads(json.dumps(md)) == md
    # without a proxy nothing is clamped and degraded is undefined (False)
    plain = rank_twobasis_schedules(hx, hz, layer_blocks=compiled.layer_blocks, policy=_policy())
    assert plain.d_circ_ext_x == 10 and plain.metadata["clamped_x"] is False


def test_delta_one_elimination_and_catastrophic_verdict_are_unchanged(monkeypatch):
    """[[4,1]] toy: X check 1111, Z checks 1100/0011 — the canonical X order
    leaves the suffix 0011 = a logical X undetected (docs/02 §9.4)."""

    hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    hz = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    ranked = rank_twobasis_schedules(
        hx, hz, layer_blocks=None,
        policy=SecPolicyConfig(sec_candidates=16, sec_cert_iters=10, sec_cert_seeds=2, sec_refine_top=2, sec_refine_iters=10),
    )
    assert not ranked.catastrophic and ranked.delta_one_eliminated >= 1
    assert ranked.metadata["per_candidate"]["X"][0] == [0, None, None, True]
    only = rank_twobasis_schedules(
        hx, hz, layer_blocks=None,
        policy=SecPolicyConfig(sec_candidates=1, sec_cert_iters=10),
    )
    assert only.catastrophic and only.schedule.candidate_index_x == 0
    assert only.d_circ_ext_x is None and only.d_circ_ext_z is None
    assert only.metadata["reason"] == "all_candidates_have_single_fault_logical"
    assert only.metadata["converged_x"] is None


def test_candidate_independent_work_is_hoisted(bb_small, monkeypatch):
    """TB-5 (docs/05 §4 次要): logicals() and the stabilizer span reduction
    used to run once per candidate; now once per rank call / per basis."""

    compiled, hx, hz = bb_small
    counts = {"logicals": 0, "row_reduce": 0}
    real_logicals, real_reduce = tr.logicals, tr._row_reduce

    def counting_logicals(*args, **kwargs):
        counts["logicals"] += 1
        return real_logicals(*args, **kwargs)

    def counting_reduce(*args, **kwargs):
        counts["row_reduce"] += 1
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(tr, "logicals", counting_logicals)
    monkeypatch.setattr(tr, "_row_reduce", counting_reduce)
    rank_twobasis_schedules(
        hx, hz, layer_blocks=compiled.layer_blocks,
        policy=_policy(sec_candidates=6, sec_cert_seeds=1, sec_refine_top=0),
    )
    assert counts == {"logicals": 1, "row_reduce": 2}


def test_certify_basis_order_keeps_its_signature_and_multi_seed_is_a_tighter_bound():
    from omniqec_agentic.evaluation import BASELINE_BB_SPEC

    compiled = build_compiled_code(BASELINE_BB_SPEC)  # bb72: every order survives delta-one
    hx, hz = _mats(compiled)
    base = build_layer_sets(hx, hz, layer_blocks=compiled.layer_blocks)["X"]
    # the first candidate order without a single-fault logical
    for index in range(16):
        layers = tr._candidate_layers(base, kind="X", index=index)
        suffixes = basis_residual_suffixes(layers, m_w=hx.shape[0], n_data=hx.shape[1])
        one, hit = certify_basis_order(hx, hz, kind="X", suffixes_w=suffixes, iters=15, seed=3)
        if not hit:
            break
    assert not hit and one is not None
    many, _ = certify_basis_order(hx, hz, kind="X", suffixes_w=suffixes, iters=15, seed=3, seeds=3)
    assert many <= one  # the min over seeds includes seed slot 0
    context = tr.basis_cert_context(hx, hz, kind="X")
    again, _ = certify_basis_order(hx, hz, kind="X", suffixes_w=suffixes, iters=15, seed=3, context=context)
    assert again == one
    with pytest.raises(ValueError, match="kind"):
        certify_basis_order(hx, hz, kind="Z", suffixes_w=suffixes, iters=15, seed=3, context=context)


# ---------------------------------------------------------------------------
# M3: budget-derived group-order cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["D45", "C9xD5"])
def test_order_90_groups_build_at_360_and_are_rejected_at_288(name):
    assert group_order_cap_for_budget(360) == 90 and group_order_cap_for_budget(288) == 72
    with group_order_budget(360):
        assert get_group(name).order == 90
    with group_order_budget(288), pytest.raises(ValueError) as excinfo:
        get_group(name)
    message = str(excinfo.value)
    assert "order 90 exceeds the budget cap 72 (= n_budget // 4 for n_budget 288)" in message
    # the LP validator surfaces the same verdict as a ProgramError signature
    spec = {"family": "lp_nonabelian", "group": name,
            "A": [[[[0, 0], [1, 0]]]] if name == "D45" else [[[[0, 0, 0], [1, 0, 0]]]],
            "B": [[[[0, 0], [0, 1]]]] if name == "D45" else [[[[0, 0, 0], [0, 1, 0]]]]}
    with group_order_budget(288), pytest.raises(ProgramError, match="exceeds the budget cap 72"):
        FAMILY_REGISTRY["lp_nonabelian"].validate(spec)


def test_resource_guard_still_rejects_order_over_200():
    assert MAX_GROUP_ORDER == 200
    for budget in (None, 1000):
        with group_order_budget(budget), pytest.raises(ValueError) as excinfo:
            get_group("D101")  # order 202
        assert "order 202 exceeds the resource guard 200" in str(excinfo.value)
    # a budget above the guard never raises the cap past it
    assert group_order_cap_for_budget(10**6) == MAX_GROUP_ORDER
    with group_order_budget(None):
        assert get_group("D100").order == 200


def test_budget_context_is_scoped_and_reported():
    assert catalog.get_group_order_budget() in (None, catalog.get_group_order_budget())
    before = catalog.get_group_order_budget()
    with group_order_budget(144):
        assert catalog.get_group_order_budget() == 144
        assert catalog.effective_group_order_cap() == 36
        assert catalog.effective_group_order_cap(max_order=10) == 10
        with pytest.raises(ValueError, match="explicit cap 10"):
            get_group("D6", max_order=10)
    assert catalog.get_group_order_budget() == before


def test_run_funsearch_sets_the_budget_before_any_validation():
    from omniqec_agentic import funsearch

    source = inspect.getsource(funsearch.run_funsearch)
    first_call = source.index("set_group_order_budget(int(config.n_budget))")
    assert first_call < source.index("memory.evaluate_and_record(")
    # re-asserted where the batch's fast evaluation begins
    assert source.count("set_group_order_budget(int(config.n_budget))") >= 2


def test_format_contract_states_the_cap_at_this_budget():
    from omniqec_agentic.funsearch import _build_funsearch_prompt, _funsearch_system_prompt
    from omniqec_agentic.memory import ArchiveMemory

    plugin = FAMILY_REGISTRY["lp_nonabelian"]
    lines = format_contract_lines_for_budget(plugin, 288)
    assert lines[: len(plugin.format_contract_lines)] == tuple(plugin.format_contract_lines)
    assert any("group order must be <= n_budget // 4 (= 72 at this budget)" in line for line in lines)
    assert any("(= 90 at this budget)" in line for line in format_contract_lines_for_budget(plugin, 360))
    unbudgeted = format_contract_lines_for_budget(plugin, None)
    with group_order_budget(None):
        assert any("n_budget // 4" in line and "200" in line for line in unbudgeted)
    assert FAMILY_REGISTRY["bb"].budget_contract_lines is None
    system = _funsearch_system_prompt(allowed_families=["lp_nonabelian"], n_budget=360)
    assert "(= 90 at this budget)" in system
    config = RunConfig(
        run_id="cap", n_budget=360, hardware=HardwareConfig(target=HardwareTarget.ION),
        out_dir="unused", proposal_count=4, allowed_families=tuple(CodeFamily),
    )
    prompt = _build_funsearch_prompt(memory=ArchiveMemory(), config=config, gen=1, proposal_count=4, prompt_context={})
    assert "(= 90 at this budget)" in prompt["system_prompt"]
    assert "(= 72 at this budget)" not in prompt["system_prompt"]


def test_failure_action_for_the_cap_is_neutral():
    from omniqec_agentic.funsearch import _failure_actions_for_signature

    signature = "build:lp_nonabelian: group 'D45' has order 90: order 90 exceeds the budget cap 72 (= n_budget // 4 for n_budget 288)"
    actions = _failure_actions_for_signature(signature)
    assert len(actions) == 1 and "n_budget // 4" in actions[0]
    assert "fewer blocks" in actions[0] and "group" in actions[0]
    text = actions[0]
    assert GOAL_RE.search(text) is None
    for phrase in BANNED_PHRASES:
        assert phrase not in text
    for verb in ("prefer", "maximi", "should"):
        assert verb not in text.lower()
    assert _failure_actions_for_signature("order 202 exceeds the resource guard 200") == actions


# ---------------------------------------------------------------------------
# minor: isomorphism dedup of metacyclic re-spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("respelling", "canonical"),
    [("C4:C10(3)", "C5xD4"), ("C20:C2(11)", "C5xD4"), ("C5:C6(4)", "C3xD5"), ("C15:C2(11)", "C5xD3")],
)
def test_metacyclic_respellings_point_at_the_direct_product(respelling, canonical):
    with pytest.raises(ValueError) as excinfo:
        get_group(respelling)
    assert f"use '{canonical}'" in str(excinfo.value)
    assert get_group(canonical).name == canonical


@pytest.mark.parametrize("name", ["C3:C4(2)", "C4:C4(3)", "C8:C2(5)", "C8:C2(3)", "C7:C3(2)", "C5:C8(4)"])
def test_genuine_metacyclic_groups_are_still_accepted(name):
    assert get_group(name).name == name


def test_dedup_rule_is_consistent_with_the_invariants():
    """Every (m, n, k) the rule rewrites must at least agree with its target
    on the element-order histogram and the centre size (the invariants the
    audit named); a rewrite that failed this would be a wrong isomorphism."""

    from math import gcd

    from omniqec_agentic.construction.groups.base import compose, cycle_perm, group_from_generators, pad_perm

    checked = 0
    for m in range(3, 40):
        for n in (2, 6, 10):
            if m * n > 120:
                continue
            for k in range(2, m):
                if gcd(k, m) != 1 or pow(k, n, m) != 1 or (n == 2 and k == m - 1):
                    continue
                if catalog.canonical_twist(m, n, k) != k:
                    continue
                target = metacyclic_direct_product_name(m, n, k)
                if target is None:
                    continue
                c = pow(k, -1, m)
                total = m + n
                rotation = pad_perm(cycle_perm(m), offset=0, total=total)
                action = tuple((c * i) % m for i in range(m))
                twist = compose(pad_perm(action, offset=0, total=total), pad_perm(cycle_perm(n), offset=m, total=total))
                presented = group_from_generators(
                    name="presented", kind="metacyclic", generator_perms=[rotation, twist],
                    generator_orders=(m, n), gap_id=None,
                )
                assert group_invariants(presented) == group_invariants(get_group(target)), (m, n, k, target)
                checked += 1
    assert checked >= 20
    assert group_invariants(get_group("C5xD4"))["centre_size"] == 10


# ---------------------------------------------------------------------------
# §4.3 coverage: shapes, utilisation bands, gaps at this budget, fact lines
# ---------------------------------------------------------------------------


def test_structure_key_keeps_the_v35_label_and_adds_shapes_and_utilisation():
    assert structure_key({"family": "lp_nonabelian", "group": "C3xD3"}) == "lp_nonabelian/direct_product/12-24"
    assert structure_key(MITTEN_200) == "lp_nonabelian/direct_product/40-60/1x2|1x2"
    assert spec_total_qubits(MITTEN_200) == 360 and spec_total_qubits(MITTEN_150) == 270
    assert structure_key(MITTEN_200, 360) == "lp_nonabelian/direct_product/40-60/1x2|1x2/util0.9-1.0"
    assert structure_key(MITTEN_150, 360).endswith("/util0.75-0.9")
    assert structure_key(BB_SMALL_SPEC, 288) == "bb/bb_square/0-12/util<0.5"
    assert spec_total_qubits(BB_SMALL_SPEC) == 36
    hgp = {"family": "hgp", "seeds": {"H1": [[1, 1, 0], [0, 1, 1]], "H2": [[1, 1], [1, 1]]}}
    assert spec_total_qubits(hgp) == 20 and structure_key(hgp, 40) == "hgp/hgp/2x3|2x2/util0.5-0.75"
    assert utilisation_band(None, 288) is None and utilisation_band(100, None) is None
    assert [utilisation_band(v, 100) for v in (0, 49, 50, 74, 75, 89, 90, 100)] == [
        "util<0.5", "util<0.5", "util0.5-0.75", "util0.5-0.75", "util0.75-0.9", "util0.75-0.9",
        "util0.9-1.0", "util0.9-1.0",
    ]
    assert tuple(UTILISATION_LABELS) == ("util<0.5", "util0.5-0.75", "util0.75-0.9", "util0.9-1.0")
    assert structure_key(None) == "unknown"


def _row(spec, code_hash, **extra):
    return {"spec": spec, "code_hash": code_hash, "family": spec.get("family"), **extra}


def test_catalog_gaps_names_never_proposed_branch_shapes_at_this_budget():
    rows = [
        _row(MITTEN_200, "a" * 32, k=40, d_search=12, n_budget=360),
        _row(MITTEN_150, "b" * 32, k=30, d_search=10, n_budget=360),
        _row({"family": "lp_nonabelian", "group": "D6", "A": [[[[0, 0], [1, 0]]]], "B": [[[[0, 0], [0, 1]]]]},
             "c" * 32, k=2, d_search=3, n_budget=360),
        _row(BB_SMALL_SPEC, "d" * 32, k=4, d_search=3, n_budget=360),
    ]
    gaps = catalog_gaps(rows, 360)
    assert gaps["n_budget"] == 360 and gaps["lp_nonabelian_proposals"] == 3
    assert gaps["by_branch"] == {"dihedral": 1, "metacyclic": 0, "direct_product": 2, "alternating_symmetric": 0}
    never = {(g["family"], g["branch"], g["shapes"]) for g in gaps["never_proposed_branch_shapes"]}
    assert ("lp_nonabelian", "direct_product", "1x2|1x2") not in never
    assert ("lp_nonabelian", "dihedral", "1x1|1x1") not in never
    assert ("lp_nonabelian", "dihedral", "1x2|1x2") in never
    assert ("lp_nonabelian", "metacyclic", "1x1|1x1") in never
    assert ("lp_abelian", "cyclic", "1x1|1x1") in never
    # a 2x2|2x2 direct product needs 16*16 = 256 qubits with its smallest group: feasible at 360
    assert ("lp_nonabelian", "direct_product", "2x2|2x2") in never
    # ...and infeasible at 144, so it is not a gap there
    assert ("lp_nonabelian", "direct_product", "2x2|2x2") not in {
        (g["family"], g["branch"], g["shapes"]) for g in catalog_gaps(rows, 144)["never_proposed_branch_shapes"]
    }
    assert gaps["feasible_branch_shape_pairs"] == len(feasible_branch_shape_pairs(360))
    assert len(feasible_branch_shape_pairs(144)) < len(feasible_branch_shape_pairs(360)) <= len(feasible_branch_shape_pairs(None))
    regions = {r["region"]: r for r in gaps["regions"]}
    mitten = regions["lp_nonabelian/direct_product/40-60/1x2|1x2/util0.9-1.0"]
    assert (mitten["proposals"], mitten["best_k"], mitten["best_d_search"], mitten["max_total_qubits"]) == (1, 40, 12, 360)
    assert regions["bb/bb_square/0-12/util<0.5"]["best_k"] == 4
    assert json.loads(json.dumps(gaps)) == gaps
    # the v3.5 call shape still works (no budget: utilisation from the rows)
    legacy = catalog_gaps(rows)
    assert legacy["n_budget"] is None and legacy["by_branch"] == gaps["by_branch"]


def test_coverage_report_regions_carry_shapes_and_bands_from_the_rows():
    rows = [_row(MITTEN_200, "a" * 32, n_budget=360), _row(MITTEN_200, "e" * 32)]
    report = coverage_report(rows, [{"code_hash": "a" * 32, "per_logical_per_round": 0.1, "stopped_by": "confirm"}])
    regions = {r["region"]: r for r in report["regions"]}
    assert regions["lp_nonabelian/direct_product/40-60/1x2|1x2/util0.9-1.0"]["measured"] == 1
    assert regions["lp_nonabelian/direct_product/40-60/1x2|1x2"]["proposed"] == 1
    forced = coverage_report(rows, [], n_budget=360)
    assert forced["regions"][0]["proposed"] == 2


def test_coverage_fact_lines_are_neutral_facts():
    rows = [
        _row(MITTEN_200, "a" * 32, k=40, d_search=12, n_budget=360),
        _row(BB_SMALL_SPEC, "d" * 32, k=4, d_search=3, n_budget=360),
    ]
    lines = coverage_fact_lines(rows, 360, limit=6)
    assert lines[0].startswith("Regions never proposed at this budget (n_budget=360): ")
    assert "lp_nonabelian/dihedral/1x1|1x1" in lines[0] and "more)" in lines[0]
    assert "Region lp_nonabelian/direct_product/40-60/1x2|1x2/util0.9-1.0: proposals 1, best k 40, best d_search 12" in lines
    assert "Region bb/bb_square/0-12/util<0.5: proposals 1, best k 4, best d_search 3" in lines
    text = "\n".join(lines)
    assert GOAL_RE.search(text) is None, GOAL_RE.search(text)
    for phrase in BANNED_PHRASES:
        assert phrase not in text
    assert re.search(r"\b(prefer|maximi[sz]e|should|must|try)\b", text, re.I) is None
    # no rows: still a complete, neutral sentence
    empty = coverage_fact_lines([], None)
    assert len(empty) == 1 and "n_budget unknown" in empty[0]
    # every feasible pair proposed -> the 'none' wording
    full = [_row({"family": fam, "group": ("D6" if fam == "lp_nonabelian" else {"factors": [3]}),
                  "A": [[[[0, 0]]] * c for _ in range(r)] if fam == "lp_nonabelian" else [[[[0]]] * c for _ in range(r)],
                  "B": [[[[0, 0]]] * c2 for _ in range(r2)] if fam == "lp_nonabelian" else [[[[0]]] * c2 for _ in range(r2)]},
                 f"{i:032x}")
            for i, (fam, r, c, r2, c2) in enumerate(
                (fam, r, c, r2, c2)
                for fam in ("lp_nonabelian", "lp_abelian")
                for r, c in ((1, 1), (1, 2), (2, 1), (2, 2))
                for r2, c2 in ((1, 1), (1, 2), (2, 1), (2, 2))
            )]
    # dihedral / cyclic only: the other branches stay unproposed
    gaps = catalog_gaps(full, 24)
    assert all(g["branch"] not in ("dihedral", "cyclic") for g in gaps["never_proposed_branch_shapes"])


def test_s9_block_renders_the_fact_lines(tmp_path):
    from omniqec_agentic.orchestrator.loop import _coverage_payload
    from omniqec_agentic.tools.context import ToolContext

    queue = tmp_path / "all_archive_queue.jsonl"
    queue.write_text(
        json.dumps({"spec": MITTEN_200, "code_hash": "a" * 32, "k": 40, "d_search": 12, "family": "lp_nonabelian"}) + "\n",
        encoding="utf-8",
    )
    state = {
        "spec": {"qubit_budget": 360},
        "last_cycle": {"artifact_paths": {"all_archive_queue": str(queue)}},
    }
    ctx = ToolContext(run_dir=tmp_path, manual_dir=tmp_path / "manual", state=state, turn=1)
    payload = _coverage_payload(ctx)
    assert payload["lp_nonabelian_catalog"]["n_budget"] == 360
    assert payload["fact_lines"] == coverage_fact_lines(ctx.archive_rows, 360, limit=12)
    assert payload["fact_lines"][0].startswith("Regions never proposed at this budget (n_budget=360)")
    assert any(line.startswith("Region lp_nonabelian/direct_product/40-60/1x2|1x2/util0.9-1.0: proposals 1") for line in payload["fact_lines"])
    assert payload["regions"][0]["region"].endswith("/util0.9-1.0")
    text = json.dumps(payload)
    assert GOAL_RE.search(text) is None
    # no budget in the state: the block still renders
    ctx_nb = ToolContext(run_dir=tmp_path, manual_dir=tmp_path / "manual", state={"last_cycle": state["last_cycle"]}, turn=1)
    assert _coverage_payload(ctx_nb)["lp_nonabelian_catalog"]["n_budget"] is None


# ---------------------------------------------------------------------------
# known codes: the Mitten rows
# ---------------------------------------------------------------------------


def test_known_code_table_carries_the_mitten_rows_and_is_consistent():
    from omniqec_agentic.memory.known_codes import (
        known_code_lookup,
        known_code_names,
        load_known_codes,
        verify_known_code_table,
    )

    names = known_code_names()
    assert any("[[150,30,10]]" in name for name in names)
    assert any("[[200,40,12]]" in name for name in names)
    rows = {int(e["n"]): e for e in load_known_codes() if e.get("family") == "lp_nonabelian" and int(e["n"]) in (150, 200)}
    assert rows[150]["spec"] == MITTEN_150 and rows[200]["spec"] == MITTEN_200
    assert (rows[150]["k"], rows[150]["d"], rows[200]["k"], rows[200]["d"]) == (30, 10, 40, 12)
    assert "2607.28795" in rows[150]["source"] and "2607.28795" in rows[200]["source"]
    assert verify_known_code_table() == []
    for spec, n, k in ((MITTEN_150, 150, 30), (MITTEN_200, 200, 40)):
        hx, hz = build_matrices(spec)
        hx, hz = np.asarray(hx, dtype=np.uint8), np.asarray(hz, dtype=np.uint8)
        assert hx.shape[1] == n and int(hx.sum(1).max()) == 9 and int(hz.sum(1).max()) == 9
        match = known_code_lookup(hx, hz)
        assert match is not None and match["nkd"] == [n, k, rows[n]["d"]]
        assert match["match_tier"] in ("wl", "canonical")
        assert get_group(spec["group"]).gap_id == ((30, 1) if n == 150 else (40, 5))
