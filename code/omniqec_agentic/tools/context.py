"""ToolContext: the read surface tools operate on (03 §4).

Bundles campaign-run pointers and lazily-loaded resources. Tools never write
protocol state; the only mutable members are the ledger (policy scope,
backtester-adjudicated) and the staged plan slot the loop consumes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from omniqec_agentic.memory.bandit import StrategyBandit
from omniqec_agentic.memory.ledger import Ledger


@dataclass
class ToolContext:
    run_dir: Path
    manual_dir: Path
    state: dict[str, Any]  # light CampaignState snapshot (hof/baseline/budget/protocol/spec)
    turn: int

    @cached_property
    def ledger(self) -> Ledger:
        return Ledger(self.run_dir / "ledger.json")

    @cached_property
    def bandit(self) -> StrategyBandit:
        return StrategyBandit(self.run_dir / "bandit.json")

    @cached_property
    def archive_rows(self) -> list[dict[str, Any]]:
        """Fast-archive queue rows from the LAST completed turn.

        The RUNTIME/MAP view, in acquisition order. Structural retention lives
        in `state["structural_proxy_elites"]` and is not derived from here."""

        paths = (self.state.get("last_cycle") or {}).get("artifact_paths") or {}
        queue = paths.get("all_archive_queue") or paths.get("promotion_queue")
        if not queue or not Path(queue).exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in Path(queue).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    @cached_property
    def slow_rows(self) -> list[dict[str, Any]]:
        paths = (self.state.get("last_cycle") or {}).get("artifact_paths") or {}
        rankings = paths.get("slow_rankings")
        if not rankings or not Path(rankings).exists():
            return []
        data = json.loads(Path(rankings).read_text(encoding="utf-8"))
        return [row for row in data.get("rankings") or [] if isinstance(row, dict)]

    def candidate_payload(self, code_hash_prefix: str) -> dict[str, Any] | None:
        """Full slow payload for a candidate by code_hash prefix (>= 8 chars)."""

        prefix = str(code_hash_prefix)
        if len(prefix) < 8:
            return None
        for row in self.slow_rows:
            full = str(row.get("code_hash") or "")
            if full.startswith(prefix):
                path = row.get("candidate_path")
                if path and Path(path).exists():
                    return json.loads(Path(path).read_text(encoding="utf-8"))
        return None

    # staged outputs consumed by the orchestrator loop -------------------
    staged_plan: dict[str, Any] | None = field(default=None)
