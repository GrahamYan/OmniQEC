"""UCB1 strategy bandit (03 §6.4).

Arms = strategy x (optional) family. Reward per turn = the arm's share of
generational contribution (new hall-of-fame entries / frontier improvement).
The suggestion is an INFORMED SIGNAL, not a command: the policy agent may
deviate with a cited rationale, and every arm keeps an epsilon_arm floor so
no strategy (or family, under mixed search) goes extinct. Mechanism follows
ShinkaEvolve's UCB1 (there used for LLM-ensemble selection)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

STRATEGY_ARMS = ("exploit", "test_hypothesis", "novelty", "transfer", "wildcat")
EPSILON_ARM = 0.05
UCB_C = math.sqrt(2.0)


class StrategyBandit:
    def __init__(self, path: str | Path, *, arms: tuple[str, ...] = STRATEGY_ARMS):
        self.path = Path(path)
        self.arms = tuple(arms)
        self.stats: dict[str, dict[str, float]] = {
            arm: {"pulls": 0.0, "reward": 0.0} for arm in self.arms
        }
        self.last_gen: int | None = None
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for arm, stat in (data.get("stats") or {}).items():
                if arm in self.stats:
                    self.stats[arm] = {
                        "pulls": float(stat.get("pulls") or 0.0),
                        "reward": float(stat.get("reward") or 0.0),
                    }
            if data.get("last_gen") is not None:
                self.last_gen = int(data["last_gen"])

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema": "strategy_bandit",
                    "arms": list(self.arms),
                    "stats": self.stats,
                    "last_gen": self.last_gen,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def update(self, rewards: dict[str, float], *, weight: float = 1.0, gen: int | None = None) -> None:
        """One turn's contribution shares per arm (missing arms pull 0).

        ``gen`` makes the update idempotent under crash-resume: the persisted
        file updates immediately inside this call, but the LangGraph
        checkpoint only commits AFTER the node returns, so a crash in between
        re-runs reconcile and would double-credit the turn without this guard.
        """

        if gen is not None and self.last_gen is not None and int(gen) <= self.last_gen:
            return
        for arm in self.arms:
            self.stats[arm]["pulls"] += float(weight)
            self.stats[arm]["reward"] += float(rewards.get(arm) or 0.0)
        if gen is not None:
            self.last_gen = int(gen)
        self._write()

    def suggestion(self) -> dict[str, Any]:
        """UCB1 scores -> normalized mix with the epsilon_arm floor."""

        total_pulls = sum(stat["pulls"] for stat in self.stats.values())
        scores: dict[str, float] = {}
        for arm, stat in self.stats.items():
            if stat["pulls"] <= 0:
                scores[arm] = float("inf")  # untried arms first
                continue
            mean = stat["reward"] / stat["pulls"]
            scores[arm] = mean + UCB_C * math.sqrt(
                math.log(max(math.e, total_pulls)) / stat["pulls"]
            )
        untried = [arm for arm, s in scores.items() if s == float("inf")]
        if untried:
            base = {arm: (1.0 if arm in untried else 0.0) for arm in self.arms}
        else:
            floor = min(scores.values())
            base = {arm: s - floor + 1e-9 for arm, s in scores.items()}
        norm = sum(base.values())
        # exact epsilon floor: reserve epsilon per arm, split the remainder
        # proportionally — every arm gets >= EPSILON_ARM by construction.
        remainder = 1.0 - EPSILON_ARM * len(self.arms)
        mix = {arm: EPSILON_ARM + remainder * v / norm for arm, v in base.items()}
        return {
            "mix": mix,
            "epsilon_arm": EPSILON_ARM,
            "stats": {
                arm: {
                    "pulls": stat["pulls"],
                    "mean_reward": (stat["reward"] / stat["pulls"]) if stat["pulls"] else None,
                }
                for arm, stat in self.stats.items()
            },
        }


def turn_rewards_from_tags(
    accepted_programs: list[dict[str, Any]],
    new_hof_hashes: set[str],
) -> dict[str, float]:
    """Credit assignment (03 §6.2): each new hall-of-fame entry credits the
    arm implied by its program's tags — an explicit ``strategy: transfer`` /
    ``strategy: wildcat`` wins, else ``exploits`` -> exploit, ``tests`` ->
    test_hypothesis, untagged -> novelty (free exploration). Without the
    explicit strategy tag the transfer/wildcat arms could never earn reward
    and would sit at the epsilon floor forever."""

    rewards = {arm: 0.0 for arm in STRATEGY_ARMS}
    if not new_hof_hashes:
        return rewards
    share = 1.0 / len(new_hof_hashes)
    for program in accepted_programs:
        if str(program.get("code_hash")) not in new_hof_hashes:
            continue
        strategy = str(program.get("strategy") or "").strip().lower()
        if strategy in ("transfer", "wildcat"):
            rewards[strategy] += share
        elif program.get("exploits"):
            rewards["exploit"] += share
        elif program.get("tests"):
            rewards["test_hypothesis"] += share
        else:
            rewards["novelty"] += share
    return rewards


def parse_hypothesis_tags(source: str) -> dict[str, Any]:
    """First-line structured comment (03a §2):
    ``# exploits: I-3, I-7; tests: H-12; strategy: transfer`` ->
    {"exploits": [...], "tests": [...], "strategy": "transfer"|None}.
    Untagged programs are legal (free exploration)."""

    tags: dict[str, Any] = {"exploits": [], "tests": [], "strategy": None}
    first = (source or "").lstrip().splitlines()[:1]
    if not first or not first[0].lstrip().startswith("#"):
        return tags
    text = first[0].lstrip()[1:]
    for clause in text.split(";"):
        if ":" not in clause:
            continue
        key, _, values = clause.partition(":")
        key = key.strip().lower()
        if key in ("exploits", "tests"):
            tags[key] = [v.strip() for v in values.split(",") if v.strip()]
        elif key == "strategy":
            value = values.strip().lower()
            tags["strategy"] = value if value in ("transfer", "wildcat") else None
    return tags
