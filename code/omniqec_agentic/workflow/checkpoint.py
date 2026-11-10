"""Checkpoint plumbing (03 §2.3): SqliteSaver in the run directory,
STRICT_MSGPACK, resolved-versions snapshot, resume protocol validation."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# CVE-2025-64439 hardening (03 §2.3): must be set before langgraph
# deserializes any checkpoint.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

CHECKPOINT_FILENAME = "checkpoint.sqlite3"
_VERSIONED_PACKAGES = (
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-checkpoint-sqlite",
    "langchain-core",
    "pydantic",
    "stim",
    "ldpc",
    "numpy",
    # v3.6 (docs/06 §1.2): the formal decoder is the vendored
    # telescoping-decoder package (third_party/telescoping-decoder); the
    # validator requires its key to be resolved. relay_bp / scipy stay
    # recorded for the dev-only decoders.
    "telescoping-decoder",
    "relay_bp",
    "scipy",
)


def open_checkpointer(run_dir: str | Path):
    """SqliteSaver on run_dir/checkpoint.sqlite3 (offline-HPC friendly)."""

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(run_dir) / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def resolved_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = "absent"
    # The decoder BACKEND, not just the package: `ldpc` can be installed while
    # BpOsdDecoder fails to import (wheel/ABI mismatch), and under a
    # single-decoder protocol that is the difference between a measured
    # campaign and no campaign at all.
    versions["bposd_backend"] = _bposd_backend()
    # The vendored commit of the telescoping decoder is protocol identity.
    from omniqec_agentic.config import TELESCOPING_VENDORED_COMMIT

    versions["telescoping_commit"] = TELESCOPING_VENDORED_COMMIT
    return versions


def _bposd_backend() -> str:
    try:
        from ldpc import BpOsdDecoder  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unresolved: {type(exc).__name__}"
    return "ldpc.BpOsdDecoder"


def write_versions_snapshot(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "resolved_versions.json"
    path.write_text(
        json.dumps(resolved_versions(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


# Snapshot keys that describe HOW the measurement is executed, not WHAT is
# measured (2026-09-05): every recorded number is a pure function of
# (seed, shard, shot index), so the worker count only changes wall-clock
# time. A campaign frozen at 63 workers may resume at 127 (a 128-core node)
# and vice versa; the value actually used is recorded per turn in
# job_config.workers, and `resume_runtime_overrides` says which keys to
# carry into the checkpointed protocol so the graph runs with them.
# 2026-09-24: final_decode_shard_timeout_cap_s -- the ceiling of the
# final-stage shard watchdog. Like slow_workers it changes how long a shard
# may run, not what is measured; a capped shard ends as a v3.4 watchdog
# timeout (decode_infeasible), the same verdict the frozen per-shot budget
# already hands out. Only present in the snapshot when set.
RESUME_RUNTIME_KEYS: frozenset[str] = frozenset(
    {
        "slow_workers",
        "final_decode_shard_timeout_cap_s",
        "search_decode_shard_timeout_cap_s",
        # 2026-09-29: the slow queue's stabilizer-weight gate (campaign._build_slow_queue).
        "promotion_max_check_weight",
    }
)
# Keys that may only be LOWERED on resume (2026-09-05, user decision: 60 -> 30
# turns for every arm). Lowering max_turns truncates the campaign -- the
# turns already run are untouched, the final stage runs at the new bound --
# so it is recorded like a runtime override; raising it would extend a
# campaign past its frozen bound and stays a protocol mismatch.
RESUME_LOWERABLE_KEYS: frozenset[str] = frozenset({"max_turns"})
# Keys the resume diff ignores: the runtime keys plus the override history the
# graph appends to the checkpointed protocol (a fresh launch never carries it).
RESUME_IGNORED_KEYS: frozenset[str] = RESUME_RUNTIME_KEYS | frozenset({"runtime_overrides"})
# 2026-09-11 (user decision, pilot10x_maxerr100_v2): the search-stage early
# stop policy may move from one FORMAL policy to another on resume. Both must
# be formal policies for the point's search shots (config.FORMAL_RACING_
# POLICIES), the switch lands at a turn boundary (a resume is one), and it is
# recorded in runtime_overrides like a worker-count change; every turn's own
# slow_run_config.json records the block that turn ran under, which is what
# the validator checks.
RESUME_POLICY_KEYS: frozenset[str] = frozenset({"racing"})


def _formal_racing_switch(stored_protocol: dict[str, Any], current_protocol: dict[str, Any]) -> bool:
    from omniqec_agentic.config import formal_racing_policy_id

    stored = stored_protocol.get("racing")
    current = current_protocol.get("racing")
    if stored == current or not isinstance(stored, dict) or not isinstance(current, dict):
        return False
    return (
        formal_racing_policy_id(stored) is not None
        and formal_racing_policy_id(current) is not None
        and int(stored.get("shots_confirm") or 0) == int(current.get("shots_confirm") or 0)
    )


def _lowered(stored_protocol: dict[str, Any], current_protocol: dict[str, Any], key: str) -> bool:
    try:
        return int(current_protocol[key]) < int(stored_protocol[key])
    except (KeyError, TypeError, ValueError):
        return False


def resume_runtime_overrides(
    stored_protocol: dict[str, Any], current_protocol: dict[str, Any]
) -> dict[str, Any]:
    """The runtime keys whose current value differs from the stored one, plus
    the lowerable keys the current launch lowers."""

    out = {
        key: current_protocol.get(key)
        for key in RESUME_RUNTIME_KEYS
        if key in current_protocol and stored_protocol.get(key) != current_protocol.get(key)
    }
    for key in RESUME_LOWERABLE_KEYS:
        if _lowered(stored_protocol, current_protocol, key):
            out[key] = current_protocol[key]
    if _formal_racing_switch(stored_protocol, current_protocol):
        out["racing"] = current_protocol["racing"]
    return out


def original_max_turns(protocol: dict[str, Any] | None) -> int | None:
    """The max_turns a campaign was frozen with before any lowering, read from
    the override history; None when max_turns was never lowered."""

    history = (protocol or {}).get("runtime_overrides") or []
    for entry in history:
        before = ((entry or {}).get("from") or {}).get("max_turns")
        if before is not None:
            return int(before)
    return None


def validate_resume_protocol(
    stored_protocol: dict[str, Any] | None,
    current_protocol: dict[str, Any],
) -> None:
    """--resume refuses when the frozen protocol snapshot changed (03 §2.3).

    Runtime keys (``RESUME_RUNTIME_KEYS``) are exempt: they never change a
    recorded number."""

    if not stored_protocol:
        raise ValueError("resume requested but the checkpoint has no protocol snapshot")
    if stored_protocol != current_protocol:
        changed = sorted(
            key
            for key in set(stored_protocol) | set(current_protocol)
            if stored_protocol.get(key) != current_protocol.get(key)
            and key not in RESUME_IGNORED_KEYS
            and not (key in RESUME_LOWERABLE_KEYS and _lowered(stored_protocol, current_protocol, key))
            and not (key in RESUME_POLICY_KEYS and _formal_racing_switch(stored_protocol, current_protocol))
        )
        if changed:
            raise ValueError(
                "protocol snapshot mismatch — refusing to resume with a different "
                f"measurement protocol; differing keys: {changed}"
            )
