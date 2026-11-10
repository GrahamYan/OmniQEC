"""v3.7.1 (2026-09-29): the check weight reaches every stage that v3.7 says
uses it.

v3.7 made the structural front, the runtime parent window and the evidence
tables weight-aware, but the weight itself was missing from four carriers: the
light rows of the cross-turn reduction (so the three-axis front silently ran as
the pre-v3.7 two-axis front), every slow result row (so the cross-turn hall of
fame and the planner's blocks read null), the evidence rows (so a weight
hypothesis could never be adjudicated) and the construction census (so a weight
class held out by the d_search floor was indistinguishable from one nobody
tried). These tests pin the carriers, not any selection rule: the quotas, the
endpoints, the reducer and Q are untouched.
"""

from __future__ import annotations

import json

from omniqec_agentic import ranking
from omniqec_agentic.workflow.graph import (
    _STRUCTURAL_LIGHT_FIELDS,
    _append_structural_store,
    _evidence_tables,
    _reduce_elites_from_store,
    _structural_frontier_rows,
    _weight_census,
)


def _row(code: str, k: int, d: int, w: int, *, n_total: int = 96) -> dict:
    return {
        "code_hash": code * 12,
        "program_id": f"p_{code}",
        "family": "bb",
        "n": 48,
        "n_total": n_total,
        "k": k,
        "d_search": d,
        "d_proxy": d,
        "Q_search": float(k * d * d),
        "max_check_weight": w,
        "max_qubit_degree": 4,
        "spec": {"family": "bb", "ell": 3, "m": 8, "A": [[0, 0], [1, 0]], "B": [[0, 0], [2, 4]]},
        "construct_source": "def construct(rng):\n    return {}\n",
        "rng_seed": 0,
        "source_agent": "funsearch",
    }


# ------------------------------------------------- the cross-turn reduction
def test_light_fields_carry_the_weight_axis() -> None:
    """The reducer compares the third axis only when both cells record a
    weight, so the field list is what decides whether v3.7 is in effect."""

    assert ranking.STRUCTURAL_WEIGHT_AXIS in _STRUCTURAL_LIGHT_FIELDS


def test_store_reduction_keeps_a_light_code_the_two_axis_front_would_drop(tmp_path) -> None:
    """A (k=2, d=6, w=4) code is dominated on (k, d_search) alone by a
    (k=8, d=8, w=6) code and survives on the three-axis front. Before v3.7.1
    the store path dropped it on every reconcile."""

    light = _row("aa", 2, 6, 4)
    heavy = _row("bb", 8, 8, 6)
    middle = _row("cc", 4, 7, 6)
    store = tmp_path / "structural_ledger.jsonl"
    _append_structural_store(store, [light, heavy, middle], set())

    hashes = {row["code_hash"] for row in _reduce_elites_from_store(store, limit=16)}
    assert light["code_hash"] in hashes, "the lightest code must survive the store reduction"
    assert heavy["code_hash"] in hashes

    # ...and the reduction is the weight-aware one: the winners' rows keep the
    # weight, so the parent round-robin sees (k, d_search, weight) cells.
    reduced = {row["code_hash"]: row for row in _reduce_elites_from_store(store, limit=16)}
    assert reduced[light["code_hash"]]["max_check_weight"] == 4


def test_store_reduction_matches_the_in_memory_reduction(tmp_path) -> None:
    """The cross-turn path and the intra-turn ArchiveMemory path must reduce
    the same pool to the same elites; they diverged while one was weight-blind."""

    pool = [_row("aa", 2, 6, 4), _row("bb", 8, 8, 6), _row("cc", 4, 7, 6), _row("dd", 4, 7, 8)]
    store = tmp_path / "structural_ledger.jsonl"
    _append_structural_store(store, pool, set())

    from_store = sorted(row["code_hash"] for row in _reduce_elites_from_store(store, limit=16))
    in_memory = sorted(
        row["code_hash"]
        for row in ranking.reduce_structural_proxy_elites([dict(r) for r in pool], limit=16)
    )
    assert from_store == in_memory


# ------------------------------------------------- the rows the LLMs are shown
def test_structural_frontier_rows_show_the_weight() -> None:
    rows = _structural_frontier_rows([_row("aa", 2, 6, 4)])
    assert rows[0]["max_check_weight"] == 4


def test_planner_blocks_request_the_weight_column() -> None:
    """S-3 (structural frontier) and S-4 (slow digest) are the planner's only
    view of retention and measurement; both omitted the weight."""

    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "orchestrator" / "loop.py"
    text = source.read_text(encoding="utf-8")
    s3 = text.split('"S-3", "structural_frontier"', 1)[1].split('"S-4"', 1)[0]
    s4 = text.split('"S-4", "digest_slow"', 1)[1].split('"S-5"', 1)[0]
    assert '"max_check_weight"' in s3
    assert '"max_check_weight"' in s4


def test_slow_rows_carry_the_weight() -> None:
    """writeback builds every ranking row; the weight sits in the structural
    result and was never copied onto the row."""

    from omniqec_agentic.evaluation.writeback import ranking_rows

    payload = {
        "program_id": "p_aa",
        "status": "ok",
        "structural": {"code_hash": "aa" * 12, "k": 2, "n_data": 48, "max_check_weight": 4,
                       "max_qubit_degree": 3},
        "fast": {"family": "bb", "n": 48, "k": 2, "d": 6, "d_search": 6},
        "promotion_record": {},
    }
    rows = ranking_rows(payload, __import__("pathlib").Path("candidate.json"))
    assert rows, "a payload with no results still yields the terminal row"
    assert all(row.get("max_check_weight") == 4 for row in rows)


def test_results_csv_has_the_weight_column() -> None:
    from omniqec_agentic.results_csv import RESULTS_CSV_COLUMNS

    assert "max_check_weight" in RESULTS_CSV_COLUMNS


# ------------------------------------------------- the construction census
def test_archive_memory_counts_constructions_by_weight() -> None:
    """A weight class whose codes all fall below the d_search floor must be
    countable; it never reaches the structural ledger."""

    from omniqec_agentic.memory.archive import ArchiveMemory

    memory = ArchiveMemory()
    memory.attempts = [
        {"max_check_weight": 4, "structural_eligible": False, "below_min_promotion_distance": True},
        {"max_check_weight": 4, "structural_eligible": False, "below_min_promotion_distance": True},
        {"max_check_weight": 6, "structural_eligible": True, "below_min_promotion_distance": False},
    ]
    census = {row["max_check_weight"]: row for row in memory.weight_census()}
    assert census[4]["constructed"] == 2
    assert census[4]["below_min_promotion_distance"] == 2
    assert census[4]["structural_eligible"] == 0
    assert census[6]["structural_eligible"] == 1


def test_weight_census_is_summed_over_turns(tmp_path) -> None:
    for turn, count in ((0, 3), (1, 5)):
        cycle = tmp_path / "turns" / f"turn_{turn:04d}" / "cycle" / "fast"
        cycle.mkdir(parents=True)
        (cycle / "weight_census.json").write_text(
            json.dumps(
                {
                    "schema": "weight_census_v1",
                    "rows": [
                        {"max_check_weight": 4, "constructed": count,
                         "below_min_promotion_distance": count},
                        {"max_check_weight": 6, "constructed": 1,
                         "below_min_promotion_distance": 0},
                    ],
                }
            ),
            encoding="utf-8",
        )
    totals, below = _weight_census(tmp_path)
    assert totals[4] == 8 and below[4] == 8
    assert totals[6] == 2 and below[6] == 0


def test_evidence_table_reports_a_class_that_never_passed_the_floor(tmp_path) -> None:
    """The w=4 class of the Fig.3 runs: written by the generator, never
    structurally eligible, therefore absent from the old table entirely."""

    cycle = tmp_path / "turns" / "turn_0000" / "cycle" / "fast"
    cycle.mkdir(parents=True)
    (cycle / "weight_census.json").write_text(
        json.dumps(
            {
                "schema": "weight_census_v1",
                "rows": [{"max_check_weight": 4, "constructed": 7,
                          "below_min_promotion_distance": 7}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "structural_ledger.jsonl").write_text(
        json.dumps(_row("bb", 4, 7, 6)) + "\n", encoding="utf-8"
    )
    tables = _evidence_tables(tmp_path, [])
    by_weight = {row["max_check_weight"]: row for row in tables["by_check_weight"]}
    assert 4 in by_weight, "a class with no eligible code must still be a row"
    assert by_weight[4]["proposed"] == 0
    assert by_weight[4]["constructed"] == 7
    assert by_weight[4]["below_distance_floor"] == 7
    assert by_weight[6]["proposed"] == 1


def test_evidence_rows_field_list_carries_the_weight() -> None:
    """ledger.PREDICATE_FIELDS has listed max_check_weight since v3.7; the
    backtester can only rule on a column the rows actually carry."""

    from pathlib import Path

    from omniqec_agentic.memory.ledger import PREDICATE_FIELDS

    assert "max_check_weight" in PREDICATE_FIELDS
    source = Path(__file__).resolve().parents[1] / "workflow" / "graph.py"
    block = source.read_text(encoding="utf-8").split("evidence_rows.jsonl", 1)[1]
    fields = block.split("fields = (", 1)[1].split(")", 1)[0]
    assert '"max_check_weight"' in fields


# ------------------------------------------------- the planner's query tool
def test_query_archive_exposes_and_filters_the_weight() -> None:
    import inspect

    from omniqec_agentic.tools.queries import query_archive

    fn = getattr(query_archive, "__wrapped__", query_archive)
    assert "max_check_weight" in inspect.signature(fn).parameters
    assert "max_check_weight" in inspect.getsource(fn)


# ------------------------------------------------- the prompts tell the truth
def test_prompts_no_longer_claim_a_two_axis_front() -> None:
    from pathlib import Path

    prompts = Path(__file__).resolve().parents[1] / "orchestrator" / "prompts"
    for name in ("orchestrator_system.md", "reflector.md"):
        text = (prompts / name).read_text(encoding="utf-8")
        assert "max_check_weight" in text, name
        assert "the (k, d_search) proxy-Pareto elites" not in text, name
