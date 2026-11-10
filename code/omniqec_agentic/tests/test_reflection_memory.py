"""Reflection + relay-note memory: fixed-size records, decaying window.

The point of these tests is the DECAY. A memory channel that only grows is how
this kind of system dies — every later prompt pays for every earlier turn — so
the caps and the window are the behaviour worth pinning, not the happy path.
"""

from __future__ import annotations

import json

from omniqec_agentic.funsearch import (
    NOTES_MAX_CHARS,
    NOTES_MAX_LINES,
    _extract_notes,
    _generation_notes_lines,
)
from omniqec_agentic.orchestrator.reflector import (
    FULL_WINDOW,
    LINE_WINDOW,
    MAX_ITEMS,
    MAX_ITEM_CHARS,
    append_reflection,
    latest_advice,
    normalize_reflection,
    read_reflections,
    reflection_input,
    render_reflections,
)


def _reflection(turn: int, advice: str = "") -> dict:
    return normalize_reflection(
        {
            "what_worked": [f"t{turn} worked"],
            "what_failed": [f"t{turn} failed"],
            "surprises": [],
            "next_turn_advice": advice or f"advice from turn {turn}",
            "candidate_hypotheses": [f"t{turn} hypothesis"],
        },
        turn=turn,
    )


def test_reflection_is_clipped_when_written_not_when_rendered() -> None:
    record = normalize_reflection(
        {
            "what_worked": [f"item {i}" for i in range(10)],
            "what_failed": ["x" * 5000],
            "next_turn_advice": "y" * 5000,
            "candidate_hypotheses": None,
        },
        turn=4,
    )
    assert len(record["what_worked"]) == MAX_ITEMS
    assert len(record["what_failed"][0]) == MAX_ITEM_CHARS
    assert len(record["next_turn_advice"]) == 400
    assert record["candidate_hypotheses"] == []


def test_a_garbage_reply_still_yields_a_well_formed_record() -> None:
    record = normalize_reflection("not a dict at all", turn=2)
    assert record["turn"] == 2
    assert all(record[field] == [] for field in
               ("what_worked", "what_failed", "surprises", "candidate_hypotheses"))
    assert record["next_turn_advice"] == ""


def test_render_decays_with_age_and_drops_past_the_window() -> None:
    rows = [_reflection(turn) for turn in range(20)]
    text = render_reflections(rows)

    # Newest few carry their full body...
    assert "t19 worked" in text and "t19 hypothesis" in text
    # ...older ones are reduced to their advice line...
    assert "advice from turn 15" in text
    assert "t15 worked" not in text
    # ...and anything past the window is gone entirely.
    assert "turn 5" not in text and "advice from turn 5" not in text

    turns_shown = [line for line in text.splitlines() if line.startswith("turn ")]
    assert len(turns_shown) == LINE_WINDOW
    assert sum(1 for row in rows[-FULL_WINDOW:] if f"t{row['turn']} worked" in text) == FULL_WINDOW


def test_render_is_bounded_no_matter_how_long_the_campaign_runs() -> None:
    short = render_reflections([_reflection(t) for t in range(12)])
    long = render_reflections([_reflection(t) for t in range(200)])
    assert len(long) <= len(short) * 1.1  # window, not history, sets the size


def test_append_is_idempotent_for_a_re_entered_turn(tmp_path) -> None:
    append_reflection(tmp_path, _reflection(0))
    append_reflection(tmp_path, _reflection(1, advice="first write"))
    append_reflection(tmp_path, _reflection(1, advice="after crash-resume"))

    rows = read_reflections(tmp_path)
    assert [row["turn"] for row in rows] == [0, 1]
    assert rows[1]["next_turn_advice"] == "after crash-resume"
    assert latest_advice(rows) == "after crash-resume"


def test_reflection_input_drops_specs_but_keeps_outcomes() -> None:
    digest = {
        "cycle_status": "ok",
        "slow_top_k": [{"code_hash": "abc", "per_logical_per_round": 1e-3,
                        "spec": {"family": "bb", "A": [[1, 2]]}}],
        "structural_frontier": [{"code_hash": "def", "k": 20, "d_search": 4}],
        "baseline": {"per_logical_per_round": 7e-3, "spec": {"family": "bb"}},
        "learning": {"evidence_rows_total": 12},
    }
    text = reflection_input(turn=3, digest=digest, plan={"family_mix": {"bb": 1.0}},
                            coverage={"regions": []})
    payload = json.loads(text)
    assert payload["runtime_hall_of_fame"][0]["per_logical_per_round"] == 1e-3
    assert "spec" not in payload["runtime_hall_of_fame"][0]
    assert "spec" not in payload["baseline"]
    # Both ledgers travel together: shown only the runtime one, the reflector
    # writes advice for the planner as if structural retention did not exist.
    assert payload["structural_frontier"][0]["d_search"] == 4
    assert payload["plan_that_produced_this"]["family_mix"] == {"bb": 1.0}


# ---- generator relay notes ------------------------------------------------

def test_notes_are_extracted_and_capped() -> None:
    reply = (
        "```python\ndef construct(rng): return {}\n```\n\n"
        "NOTES-FOR-NEXT-GENERATION:\n"
        + "\n".join(f"- line {i} " + "z" * 500 for i in range(25))
    )
    notes = _extract_notes(reply)
    assert len(notes) == NOTES_MAX_LINES
    assert all(len(note) <= NOTES_MAX_CHARS for note in notes)
    assert notes[0].startswith("line 0")


def test_absent_or_fenced_notes_never_leak_code() -> None:
    assert _extract_notes("```python\ndef construct(rng): return {}\n```") == []
    fenced = "NOTES-FOR-NEXT-GENERATION:\n- tried: D4 lifts\n```python\nsecret\n```"
    assert _extract_notes(fenced) == ["tried: D4 lifts"]


def test_notes_block_ends_at_the_first_blank_line_after_content() -> None:
    reply = "NOTES-FOR-NEXT-GENERATION:\n\n- ruled out: C2xD4\n\nunrelated trailing prose"
    assert _extract_notes(reply) == ["ruled out: C2xD4"]


def test_notes_render_newest_first_and_are_empty_without_input() -> None:
    assert _generation_notes_lines({}) == []
    lines = _generation_notes_lines({
        "recent_generation_notes": [
            {"turn": 7, "notes": ["next: A4 branch untouched"]},
            {"turn": 6, "notes": ["ruled out: metacyclic at ell>8"]},
        ]
    })
    body = "\n".join(lines)
    assert body.index("turn 7") < body.index("turn 6")
    assert "A4 branch untouched" in body


def test_reflector_sees_why_a_low_plpr_code_was_barred() -> None:
    """Regression from the first live run.

    The turn's lowest plpr came from a code with verified_circuit_distance=2,
    which the fault-tolerance floor correctly keeps out of the hall of fame.
    The reflector was shown only the FILTERED frontier, so it reported the
    absence as "likely a bookkeeping bug" — a confident wrong conclusion drawn
    from a true observation, because the deciding field was never in view.
    """

    from omniqec_agentic.orchestrator.reflector import measured_rows

    slow_rows = [
        {"program_id": "seed_0004", "family": "lp_nonabelian", "status": "ok",
         "per_logical_per_round": 0.06123, "d_proxy": 3,
         "verified_circuit_distance": 2, "degraded_ext": True, "stopped_by": None},
        {"program_id": "g0001_p0008", "family": "lp_abelian", "status": "ok",
         "per_logical_per_round": 0.06647, "d_proxy": 4,
         "verified_circuit_distance": 3, "degraded_ext": True, "stopped_by": None},
        {"program_id": "baseline_bb_72_12_6", "is_baseline": True,
         "per_logical_per_round": 0.004, "d_proxy": 6,
         "verified_circuit_distance": 6, "degraded_ext": False, "stopped_by": None},
    ]

    rows = measured_rows(slow_rows)
    assert [row["program_id"] for row in rows] == ["seed_0004", "g0001_p0008"]
    barred = rows[0]
    assert barred["per_logical_per_round"] < rows[1]["per_logical_per_round"]
    assert barred["verified_circuit_distance"] == 2  # the reason, now visible

    digest = {
        "cycle_status": "ok",
        "slow_top_k": [{"program_id": "g0001_p0008", "per_logical_per_round": 0.06647}],
        "learning": {},
    }
    payload = json.loads(reflection_input(
        turn=0, digest=digest, plan={}, coverage={}, slow_rows=slow_rows
    ))
    assert len(payload["measured_this_turn"]) == 2
    assert "verified_circuit_distance >= 3" in payload["admission_rule"]


def test_reflection_call_record_lands_where_the_accounting_looks(tmp_path) -> None:
    """Regression: the reflector's own cost was dropped from every aggregate.

    reflector.py wrote turn_dir/reflection_call.jsonl while token_usage.py read
    turn_dir/orchestrator/reflection_call.jsonl, so the one role whose cost the
    accounting was added to capture reported zero calls forever.
    """

    import inspect

    from omniqec_agentic import token_usage
    from omniqec_agentic.orchestrator import reflector

    writer = inspect.getsource(reflector.run_reflection)
    reader = inspect.getsource(token_usage.turn_usage)
    assert 'turn_dir / "reflection_call.jsonl"' in writer
    assert 'turn_dir / "reflection_call.jsonl"' in reader

    turn_dir = tmp_path / "turns" / "turn_0000"
    turn_dir.mkdir(parents=True)
    (turn_dir / "reflection_call.jsonl").write_text(
        json.dumps({"role": "reflector", "usage": {
            "input_tokens": 900, "output_tokens": 300, "raw": {"cost": 0.0125}}}) + "\n",
        encoding="utf-8",
    )
    payload = token_usage.turn_usage(turn_dir)
    assert payload["by_role"]["reflector"]["calls"] == 1
    assert payload["by_role"]["reflector"]["cost_usd"] == 0.0125
    assert payload["total"]["total_tokens"] == 1200


def test_reflection_does_not_reserve_the_whole_campaign_ceiling(monkeypatch, tmp_path) -> None:
    """Regression: a 64,000-token request was refused on a key with 49,543 left.

    OpenRouter reserves credit against `max_tokens`, not against actual usage,
    so inheriting the campaign ceiling made a ~1,800-token reflection fail with
    HTTP 402 while the campaign around it kept running.
    """

    from omniqec_agentic.config import LLMConfig
    from omniqec_agentic.orchestrator import reflector as R

    seen: dict = {}

    class _Client:
        def complete(self, *, system, user):
            return '{"what_worked": ["ok"]}'

        def info(self):
            return {"provider": "fake", "model": "m"}

    def _factory(provider, config):
        seen["max_tokens"] = config.max_tokens
        return _Client()

    monkeypatch.setattr(
        "omniqec_agentic.llm.factory.create_completion_client", _factory
    )
    record = R.run_reflection(
        provider="fake", llm_config=LLMConfig(max_tokens=131072), turn=0,
        digest={}, plan={}, coverage={}, turn_dir=tmp_path, manual_dir=tmp_path,
    )
    assert record["status"] == "ok"
    assert seen["max_tokens"] == R.REFLECTION_MAX_TOKENS
    # Still comfortably above what the contract can emit.
    assert R.REFLECTION_MAX_TOKENS > (R.MAX_ITEMS * R.MAX_ITEM_CHARS * 4 + R.MAX_ADVICE_CHARS)


def test_a_smaller_configured_ceiling_is_respected(monkeypatch, tmp_path) -> None:
    from omniqec_agentic.config import LLMConfig
    from omniqec_agentic.orchestrator import reflector as R

    seen: dict = {}

    class _Client:
        def complete(self, *, system, user):
            return "{}"

        def info(self):
            return {}

    monkeypatch.setattr(
        "omniqec_agentic.llm.factory.create_completion_client",
        lambda provider, config: (seen.update(max_tokens=config.max_tokens), _Client())[1],
    )
    R.run_reflection(provider="fake", llm_config=LLMConfig(max_tokens=2000), turn=0,
                     digest={}, plan={}, coverage={}, turn_dir=tmp_path, manual_dir=tmp_path)
    assert seen["max_tokens"] == 2000
