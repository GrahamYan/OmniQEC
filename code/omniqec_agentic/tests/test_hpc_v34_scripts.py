"""Static guard for the version-agnostic diagnose helper (hpc/claude_v33_diagnose.py).

The v3.4 launcher / submitter / status trio and its runbook were historical
deployment files and are not part of the open-source tree; only the helper,
which classifies decode_infeasible as a terminal candidate verdict that never
stops a point, is still pinned here."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSE = PACKAGE_ROOT / "hpc" / "claude_v33_diagnose.py"


def test_diagnose_helper_classifies_decode_infeasible_as_terminal() -> None:
    text = DIAGNOSE.read_text(encoding="utf-8")
    assert "decode_infeasible (terminal)" in text
    assert "shard_runtime_s" in text
