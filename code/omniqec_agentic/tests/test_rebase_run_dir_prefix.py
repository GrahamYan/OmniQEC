"""rebase_run_dir: a new run-dir name that STARTS WITH the old one (…_v36 ->
…_v36x60, the 2026-09-07 GPT b98 extension) must rewrite each path once, count
zero leftovers afterwards, and stay idempotent on a second pass."""

from __future__ import annotations

from omniqec_agentic.hpc.rebase_run_dir import count_prefix, replace_prefix

OLD = "/home/u/OmniQEC/runs/fig2_gptsub_b98_v36"
NEW = "/home/u/OmniQEC/runs/fig2_gptsub_b98_v36x60"


def test_prefix_of_itself_rewrites_once_and_leaves_no_leftovers() -> None:
    text = f'{{"a": "{OLD}/turns/turn_0001/x.json", "b": "{OLD}"}}'
    assert count_prefix(text, OLD, NEW) == 2
    once = replace_prefix(text, OLD, NEW)
    assert once == f'{{"a": "{NEW}/turns/turn_0001/x.json", "b": "{NEW}"}}'
    assert count_prefix(once, OLD, NEW) == 0
    assert replace_prefix(once, OLD, NEW) == once  # second pass: no _v36x60x60


def test_unrelated_new_prefix_behaves_like_plain_replace() -> None:
    new = "/scratch/u/OmniQEC_b/runs/fig2_gptsub_b98_v36"
    text = f"{OLD}/a {OLD}"
    assert count_prefix(text, OLD, new) == 2
    assert replace_prefix(text, OLD, new) == f"{new}/a {new}"
