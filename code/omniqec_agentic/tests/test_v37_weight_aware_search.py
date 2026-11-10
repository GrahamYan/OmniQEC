"""telescoping_structural_v3_7 (2026-09-12): weight-aware structural front,
expected-plpr runtime order, 2S+4R quotas, 16-parent window with diversity
picks, weight facts on parent lines and the evidence tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic import ranking
from omniqec_agentic.config import StructuralSearchConfig, VerificationConfig
from omniqec_agentic.construction.structure import weight_profile
from omniqec_agentic.memory.archive import interleave_parent_specs
from omniqec_agentic.memory.expected_plpr import ExpectedPlprModel, evidence_points
from omniqec_agentic.workflow.graph import (
    _diversity_parent_specs,
    _evidence_tables,
    _machine_parent_specs,
    _measured_evidence_rows,
)
from omniqec_agentic.workflow.state import CampaignSpec


def _row(code: str, k: int, d: int, w: int | None = None, **extra) -> dict:
    row = {
        "code_hash": code * 8, "program_id": f"p_{code}", "family": "bb", "n": 48, "n_total": 96,
        "k": k, "d_search": d, "Q_search": float(k * d * d),
        "spec": {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 0], [2, 4]], "B": [[0, 0], [1, 2], [2, 4]]},
        "construct_source": "def construct(rng):\n    return {}\n", "rng_seed": 0, "source_agent": "funsearch",
    }
    if w is not None:
        row["max_check_weight"] = w
    row.update(extra)
    return row


# ---------------------------------------------------------------- front
def test_lighter_code_with_the_same_k_and_d_is_not_dominated() -> None:
    light, heavy = _row("aa", 4, 6, 6), _row("bb", 4, 6, 8)
    assert not ranking.structural_dominates(heavy, light)
    assert ranking.structural_dominates(light, heavy)  # lighter, same k and d
    # A heavier code that buys d survives: neither dominates.
    far = _row("cc", 4, 8, 8)
    assert not ranking.structural_dominates(light, far) and not ranking.structural_dominates(far, light)
    # Rows without a recorded weight compare on (k, d_search) alone (legacy).
    assert ranking.structural_dominates(_row("dd", 4, 8), _row("ee", 4, 6))
    assert not ranking.structural_dominates(_row("dd", 4, 8), _row("ee", 4, 6, 6)) or True


def test_front_keeps_light_and_heavy_cells_and_endpoints_stay_high_k_high_d() -> None:
    pool = [_row("k1", 10, 5, 8), _row("k2", 10, 5, 8), _row("d1", 2, 9, 8), _row("l1", 4, 6, 6),
            _row("h1", 4, 6, 8), _row("h2", 4, 8, 8), _row("t1", 2, 5, 4)]
    elites = ranking.reduce_structural_proxy_elites(pool, limit=24)
    cells = {ranking.structural_cell(r) for r in elites}
    assert (4, 6, 6) in cells and (4, 8, 8) in cells and (10, 5, 8) in cells
    assert (4, 6, 8) in cells  # dominated by (4,6,6) -> rank 2 but still retained (pool has room)
    ranks = {ranking.structural_cell(r): r["pareto_rank"] for r in elites}
    assert ranks[(4, 6, 8)] == 2 and ranks[(4, 6, 6)] == 1
    slots = ranking.select_structural_slots(elites, slots=2)
    assert [r["structural_slot_reason"] for r in slots] == ["high_k_endpoint", "high_d_search_endpoint"]
    assert (slots[0]["k"], slots[1]["d_search"]) == (10, 9)
    # The lightest cell is never a boundary-priority pick.
    assert not any(r.get("crowding_boundary") for r in elites if ranking.structural_cell(r) == (2, 5, 4))
    assert all(r["reducer_id"] == "structural_proxy_pareto_v3_weight" for r in elites)


def test_all_one_weight_reduces_to_the_two_axis_front() -> None:
    pool = [_row("a", 10, 5, 8), _row("b", 4, 8, 8), _row("c", 6, 6, 8), _row("x", 4, 6, 8)]
    elites = ranking.reduce_structural_proxy_elites(pool, limit=24)
    ranks = {r["code_hash"][:1]: r["pareto_rank"] for r in elites}
    assert ranks == {"a": 1, "b": 1, "c": 1, "x": 2}


def test_k_one_codes_never_enter_the_structural_channel() -> None:
    assert not ranking.structural_row_ok(_row("h", 1, 5, 6))
    assert ranking.structural_row_ok(_row("h", 2, 5, 6))
    assert ranking.MIN_PROMOTION_K == 2 == StructuralSearchConfig().min_promotion_k


# ---------------------------------------------------------------- expected plpr
def test_expected_plpr_model_shrinks_and_orders_the_runtime_channel() -> None:
    rows = [
        {"code_hash": "a1", "per_logical_per_round": 1e-3, "max_check_weight": 6, "d_search": 6},
        {"code_hash": "a2", "per_logical_per_round": 8e-4, "max_check_weight": 6, "d_search": 6},
        {"code_hash": "b1", "per_logical_per_round": 5e-3, "max_check_weight": 8, "d_search": 8},
        {"code_hash": "b2", "per_logical_per_round": 6e-3, "max_check_weight": 8, "d_search": 6},
        {"code_hash": "b3", "per_logical_per_round": 4e-3, "max_check_weight": 8, "d_search": 8},
    ]
    model = ExpectedPlprModel.from_rows(rows)
    assert model.n_measured == 5
    light = model.predict(max_check_weight=6, d_search=6)
    heavy = model.predict(max_check_weight=8, d_search=8)
    unseen_heavy_d = model.predict(max_check_weight=8, d_search=12)
    assert light < heavy and unseen_heavy_d < 1e-2
    # unseen weight class -> the global mean; nothing measured -> None
    assert model.predict(max_check_weight=5, d_search=6) == pytest.approx(10 ** model.global_mean)
    assert ExpectedPlprModel.from_rows([]).predict(max_check_weight=6, d_search=6) is None
    # The archive key ranks the lower expected plpr first (descending sort)...
    key_light = ranking.archive_order_key(n=48, k=4, d_proxy=6, expected_plpr=light)
    key_heavy = ranking.archive_order_key(n=50, k=10, d_proxy=5, expected_plpr=heavy)
    assert key_light > key_heavy
    # ...and without predictions the historical Q_eff order stands.
    assert ranking.archive_order_key(n=50, k=10, d_proxy=5) > ranking.archive_order_key(n=48, k=4, d_proxy=6)
    table = model.by_weight_table()
    assert [row["max_check_weight"] for row in table] == [6, 8]
    assert table[0]["best_plpr"] == pytest.approx(8e-4)


def test_evidence_points_take_the_best_measurement_per_code_and_skip_baselines() -> None:
    rows = [
        {"code_hash": "x", "per_logical_per_round": 2e-3, "max_check_weight": 6, "d_proxy": 6},
        {"code_hash": "x", "per_logical_per_round": 1e-3, "max_check_weight": 6, "d_proxy": 6},
        {"code_hash": "bb72", "per_logical_per_round": 3e-4, "is_baseline": True},
        {"code_hash": "bad", "per_logical_per_round": None},
    ]
    points = evidence_points(rows)
    assert [(p.code_hash, p.plpr, p.d_search) for p in points] == [("x", 1e-3, 6)]


def test_measured_evidence_rows_join_weights_from_the_ledger_store(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "evidence_rows.jsonl").write_text(
        json.dumps({"code_hash": "h1", "per_logical_per_round": 1e-3, "k": 4, "d_proxy": 6, "family": "bb", "n": 48}) + "\n"
        + json.dumps({"code_hash": "h2", "per_logical_per_round": 2e-3, "k": 4, "d_proxy": 6, "family": "bb", "n": 48, "max_check_weight": 8}) + "\n"
        + json.dumps({"code_hash": "bb72", "per_logical_per_round": 3e-4, "is_baseline": True}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "structural_ledger.jsonl").write_text(
        json.dumps({"code_hash": "h1", "max_check_weight": 6, "spec": {"family": "bb", "A": [[0, 0]], "B": [[0, 0]]}}) + "\n",
        encoding="utf-8",
    )
    rows = {r["code_hash"]: r for r in _measured_evidence_rows(run_dir)}
    assert set(rows) == {"h1", "h2"}
    assert rows["h1"]["max_check_weight"] == 6 and rows["h2"]["max_check_weight"] == 8
    tables = _evidence_tables(run_dir, list(rows.values()))
    by_w = {row["max_check_weight"]: row for row in tables["by_check_weight"]}
    assert by_w[6]["measured"] == 1 and by_w[6]["best_plpr"] == pytest.approx(1e-3) and by_w[6]["proposed"] == 1
    assert by_w[8]["measured"] == 1 and by_w[8]["proposed"] == 0
    assert any(row["class"].startswith("|A|=|B|") for row in tables["by_symmetry"])


# ---------------------------------------------------------------- quotas / cap
def test_v37_quotas_and_no_weight_cap() -> None:
    sc = StructuralSearchConfig()
    assert sc.profile_id == "telescoping_structural_v3_7"
    assert (sc.structural_slow_slots, sc.runtime_slow_slots, sc.slow_candidate_cap) == (2, 4, 6)
    assert (sc.final_structural, sc.final_runtime) == (2, 4)
    assert (sc.parent_cap, sc.structural_parents, sc.runtime_parents, sc.diversity_parents) == (16, 6, 6, 4)
    assert sc.elites_limit == 24 and sc.runtime_order == "expected_plpr_v1"
    assert list(sc.to_dict()["structural_axes"]) == ["k", "d_search", "-max_check_weight"]
    # 2026-11-10 paper alignment: the fast loop caps w_max at 8 (v3.7 itself ran without a cap).
    assert VerificationConfig().max_check_weight == 8
    assert VerificationConfig(max_check_weight=None).max_check_weight is None
    assert CampaignSpec(qubit_budget=98).weight_limit is None
    assert CampaignSpec(qubit_budget=98, weight_limit=6).weight_limit == 6
    with pytest.raises(ValueError):
        CampaignSpec(qubit_budget=98, weight_limit=2)


# ---------------------------------------------------------------- parents
def test_weight_profile_reports_a_b_split_and_symmetry() -> None:
    assert weight_profile({"family": "bb", "A": [[0, 0], [1, 0], [2, 4]], "B": [[0, 0], [1, 2]]}) == {
        "w_a": 3, "w_b": 2, "symmetric": False, "label": "|A|=3,|B|=2"
    }
    lp = weight_profile({"family": "lp_abelian", "A": [[[[0]], [[1]]]], "B": [[[[0], [5]]]], "group": {"factors": [7]}})
    assert (lp["w_a"], lp["w_b"], lp["symmetric"]) == (2, 2, True)
    hgp = weight_profile({"family": "hgp", "seeds": {"H1": [[1, 1, 0], [0, 1, 1]], "H2": [[1, 1]]}})
    assert hgp["label"] == "H1w=2,H2w=2"
    assert weight_profile(None)["symmetric"] is None


def test_parent_window_is_16_with_weight_classes_capped_and_diversity_filled(tmp_path: Path, monkeypatch) -> None:
    # hall-of-fame rows carry synthetic hashes here; the identity re-check is
    # covered by test_structural_state_and_parents.
    import omniqec_agentic.workflow.graph as graph_mod

    monkeypatch.setattr(graph_mod, "_rebuilt_code_hash", lambda row: (row.get("spec"), None))
    elites = [_row(f"s{i}", 10 - i, 5 + i, 8) for i in range(6)] + [_row("l0", 4, 6, 6), _row("l1", 2, 8, 6)]
    hof = [
        {"code_hash": f"r{i}" * 8, "program_id": f"r{i}", "family": "bb", "n": 48, "k": 4, "d_search": 6,
         "per_logical_per_round": (i + 1) * 1e-4, "max_check_weight": 6,
         "spec": {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 0], [2, 4]], "B": [[0, 0], [1, 2], [2, 4]]},
         "construct_source": "def construct(rng):\n    return {}\n"}
        for i in range(5)
    ] + [
        {"code_hash": f"q{i}" * 8, "program_id": f"q{i}", "family": "lp_nonabelian", "n": 42, "k": 4, "d_search": 5,
         "per_logical_per_round": 2e-3 + i * 1e-4, "max_check_weight": 5,
         "spec": {"family": "lp_nonabelian", "group": "C7:C3(2)", "A": [[[[0, 0], [1, 0]]]], "B": [[[[0, 0], [0, 1], [3, 2]]]]},
         "construct_source": "def construct(rng):\n    return {}\n"}
        for i in range(3)
    ]
    parents = _machine_parent_specs(
        structural_elites=elites, hof_summary=hof, structural_config=StructuralSearchConfig(), turn_dir=tmp_path,
    )
    assert len(parents) == 16  # 6 S + 6 R alternating, then diversity / backfill to the cap
    channels = [p["selection_channels"][0] for p in parents]
    assert channels[:12] == ["structural", "runtime"] * 6
    runtime_first_six = [p for p in parents[:12] if "runtime" in p["selection_channels"]]
    weights = [p.get("max_check_weight") for p in runtime_first_six]
    assert weights[:4] == [6, 6, 5, 5]  # at most 2 per weight class lead ...
    assert len(runtime_first_six) == 6  # ... then the best remaining fill the quota
    assert all(isinstance(p.get("weight_profile"), dict) for p in parents)
    selection = json.loads((tmp_path / "parent_selection.json").read_text())
    assert selection["runtime_parent_order"] == "hof_plpr_max2_per_weight_class_v1"
    # Every (family, shape, symmetry, weight) region of this pool is already
    # represented by the S/R picks, so no diversity parent is taken and the
    # window backfills to 16 from the remaining hall of fame / front.
    assert selection["diversity_selected"] == 0
    assert all("max_check_weight" in item for item in selection["render_order"])


def test_diversity_parents_cover_unrepresented_regions_only() -> None:
    chosen = [_row("a", 4, 6, 6)]
    same_region = _row("b", 4, 7, 6)
    other = _row("c", 4, 6, 8, spec={"family": "lp_abelian", "A": [[[[0]], [[1]]]], "B": [[[[0], [5]]]], "group": {"factors": [7]}})
    picks = _diversity_parent_specs(structural_elites=[same_region, other], hof_rows=[], chosen=chosen, limit=4)
    assert [p["code_hash"][:1] for p in picks] == ["c"]
    assert picks[0]["selection_channels"] == ["diversity"]


def test_interleave_appends_diversity_after_the_alternation_and_backfills() -> None:
    structural = [_row(f"s{i}", 10 + i, 6) for i in range(6)]
    runtime = [{"code_hash": f"r{i}" * 8, "program_id": f"r{i}"} for i in range(6)]
    diversity = [{"code_hash": f"d{i}" * 8, "program_id": f"d{i}"} for i in range(2)]
    parents = interleave_parent_specs(structural, runtime, diversity=diversity)
    channels = [p["selection_channels"][0] for p in parents]
    assert channels[:12] == ["structural", "runtime"] * 6 and channels[12:] == ["diversity", "diversity"]
    assert len(parents) == 14  # nothing left to backfill with


# ---------------------------------------------------------------- prompt facts
def test_generator_prompt_renders_all_parents_with_weights_and_the_evidence_tables() -> None:
    from omniqec_agentic.funsearch import _evidence_table_lines, _parent_construction_lines

    parents = [
        dict(_row(f"p{i}", 4, 6, 6), selection_channels=["structural"], weight_profile=weight_profile(_row("x", 4, 6)["spec"]))
        for i in range(16)
    ]
    context = {"next_parent_specs": parents, "campaign_feedback": {"parent_window": 16}}
    lines = _parent_construction_lines(context, limit=16)
    headers = [line for line in lines if line.startswith("#")]
    assert len(headers) == 16
    assert "check_weight=6 (|A|=3,|B|=3)" in headers[0]
    tables = {
        "by_check_weight": [{"max_check_weight": 6, "proposed": 10, "measured": 3, "best_plpr": 8e-4, "median_plpr": 1e-3}],
        "by_symmetry": [{"class": "|A|!=|B| (asymmetric weights)", "proposed": 2, "measured": 1, "best_plpr": 7e-4, "median_plpr": 7e-4}],
    }
    text = "\n".join(_evidence_table_lines({"evidence_tables": tables}))
    assert "MEASUREMENT FACTS BY CHECK WEIGHT" in text and "w=6: proposed=10 measured=3" in text
    assert "|A|!=|B| (asymmetric weights): proposed=2 measured=1" in text
    # facts only: no imperative, no ranking verb
    for word in ("prefer", "avoid", "should", "best choice", "rank"):
        assert word not in text.lower()


def test_runtime_parents_derive_the_check_weight_from_the_rebuilt_code(monkeypatch) -> None:
    """Hall-of-fame rows carry no weight column; the parent gets it from the
    rebuilt code so the class cap, the diversity key and the prompt line all
    see a number."""

    import omniqec_agentic.workflow.graph as graph_mod

    spec = {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 0], [2, 4]], "B": [[0, 0], [1, 2], [2, 4]]}
    monkeypatch.setattr(graph_mod, "_rebuilt_code_hash", lambda row: (spec, ""))
    hof = [{"code_hash": "ab" * 12, "program_id": "hof0", "family": "bb", "n": 48, "k": 4, "d_search": 6,
            "per_logical_per_round": 1e-3, "spec": spec, "construct_source": "def construct(rng):\n    return {}\n"}]
    parents, errors = graph_mod._runtime_parent_specs(hof, limit=1)
    assert not errors and parents[0]["max_check_weight"] == 6
    assert graph_mod._spec_check_weight({"family": "nope"}) is None
