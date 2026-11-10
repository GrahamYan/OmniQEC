"""hpc/fig3_extended_stats_topup.py: after a supplement round, a champion whose weakest basis (formal 25k +
every round so far) has fewer than 30 failures gets a top-up aimed at 45 failures, rounded up to 50k; the
top-up goes to every champion of the point (per-point uniform), in a new seed namespace, with base_key so the
merge sums the rounds; references are never topped up; an unfinished round or a repeated round is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniqec_agentic.hpc import fig3_extended_stats_topup as topup


def _item(key, arm, role="champion", fx=4, fz=0):
    return {"key": key, "budget": 360, "arm": arm, "role": role, "code_hash": key[-8:],
            "formal_failures_x": fx, "formal_failures_z": fz, "formal_shots_per_basis": 25000,
            "shots_per_basis": 500000, "spec": {"family": "bb"}}


def _result(root: Path, key, fx, fz, shots=500000):
    d = root / key
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"queue_status": "ok", "row": {
        "status": "ok", "failures_x": fx, "failures_z": fz, "shots_x": shots, "shots_z": shots}}))


@pytest.fixture
def setup(tmp_path):
    plan = {"seed_namespace": "extended_stats_v1", "items": [
        _item("b360_claudesub_aaaaaaaa", "Claude", fx=3, fz=0),
        _item("b360_gptsub_bbbbbbbb", "GPT", fx=4, fz=3),
        _item("b360_ref_cccccccc", "reference", role="reference", fx=None, fz=None),
    ]}
    plan["items"][2]["formal_failures_x"] = plan["items"][2]["formal_failures_z"] = None
    out_root = tmp_path / "ext"
    _result(out_root, "b360_claudesub_aaaaaaaa", 151, 27)
    _result(out_root, "b360_gptsub_bbbbbbbb", 121, 50)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, out_root, tmp_path


def _run(path, out_root, out, round_=2):
    return topup.main(["--plan", str(path), "--out-root", str(out_root), "--round", str(round_),
                       "--date", "2026-10-24", "--out", str(out)])


def test_per_point_uniform_topup(setup):
    path, out_root, tmp = setup
    assert _run(path, out_root, tmp / "r2.json") == 0
    plan = json.loads((tmp / "r2.json").read_text())
    new = [it for it in plan["items"] if it.get("seed_namespace") == "extended_stats_v1_r2"]
    # Claude Z: 27 failures in 525k -> 45 expected needs 875k -> +350k; GPT already has >= 30 but gets the same
    assert sorted(it["key"] for it in new) == ["b360_claudesub_aaaaaaaa_r2", "b360_gptsub_bbbbbbbb_r2"]
    assert {it["shots_per_basis"] for it in new} == {350000}
    assert {it["base_key"] for it in new} == {"b360_claudesub_aaaaaaaa", "b360_gptsub_bbbbbbbb"}
    assert {it["key"]: it["own_rule_shots"] for it in new} == {"b360_claudesub_aaaaaaaa_r2": 350000, "b360_gptsub_bbbbbbbb_r2": 0}
    assert len(plan["items"]) == 5  # the three input items are kept, the reference is not topped up
    assert "Claude Z 27" in plan["rounds"]["2"]


def test_no_topup_when_every_champion_has_target(setup, tmp_path):
    path, out_root, tmp = setup
    plan = json.loads(path.read_text())
    plan["items"] = plan["items"][1:]  # only GPT (>= 30 per basis) and the reference
    path.write_text(json.dumps(plan))
    _run(path, out_root, tmp / "r2.json")
    assert len(json.loads((tmp / "r2.json").read_text())["items"]) == 2


def test_refuses_unfinished_round_and_repeated_round(setup):
    path, out_root, tmp = setup
    result = out_root / "b360_gptsub_bbbbbbbb" / "result.json"
    saved = result.read_text()
    result.unlink()
    with pytest.raises(SystemExit):
        _run(path, out_root, tmp / "r2.json")
    result.write_text(saved)
    _run(path, out_root, tmp / "r2.json")
    with pytest.raises(SystemExit):
        _run(tmp / "r2.json", out_root, tmp / "r2b.json")  # round 2 items already present
