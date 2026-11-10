"""Learning-layer regressions (03 §14.6): prospective backtest correctness on
synthetic data with a known rule, anti-double-dipping, concurrency cap,
graveyard, tag parsing, bandit mix sanity."""

from __future__ import annotations

from omniqec_agentic.memory.bandit import (
    StrategyBandit,
    parse_hypothesis_tags,
    turn_rewards_from_tags,
)
from omniqec_agentic.memory.backtester import run_backtest
from omniqec_agentic.memory.ledger import MAX_ACTIVE_HYPOTHESES, Ledger


def _rows(gen_start: int, count: int, *, bb_plpr: float, hgp_plpr: float) -> list[dict]:
    rows = []
    for i in range(count):
        rows.append({"gen": gen_start + i, "family": "bb", "n": 72,
                     "per_logical_per_round": bb_plpr * (1 + 0.05 * (i % 3))})
        rows.append({"gen": gen_start + i, "family": "hgp", "n": 72,
                     "per_logical_per_round": hgp_plpr * (1 + 0.05 * (i % 3))})
    return rows


def _propose_bb_rule(ledger: Ledger, *, gen: int) -> str:
    result = ledger.propose(
        scope={"families": ("bb", "hgp")},
        predicate=[{"field": "family", "op": "==", "value": "bb"}],
        claim={"metric": "per_logical_per_round", "direction": "lower"},
        proposer="test", turn=1, propose_gen=gen, rationale="synthetic",
    )
    assert result["accepted"], result
    return result["id"]


def test_known_rule_becomes_supported_prospectively(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    display = _propose_bb_rule(ledger, gen=5)
    assert display.startswith("H-")
    # bb genuinely 10x better, rows AFTER propose_gen
    summary = run_backtest(ledger, _rows(6, 15, bb_plpr=1e-3, hgp_plpr=1e-2))
    assert summary["supported"] == 1
    entry = ledger.resolve(display)
    assert entry.status == "supported"
    assert entry.display_id.startswith("I-")
    assert entry.evidence.effect_size > 0.9  # ~1 decade in log10


def test_reversed_rule_is_refuted_and_graveyarded(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    display = _propose_bb_rule(ledger, gen=5)
    run_backtest(ledger, _rows(6, 15, bb_plpr=1e-2, hgp_plpr=1e-3))  # bb WORSE
    assert ledger.resolve(display).status == "refuted"
    again = ledger.propose(
        scope={"families": ("bb", "hgp")},
        predicate=[{"field": "family", "op": "==", "value": "bb"}],
        claim={"metric": "per_logical_per_round", "direction": "lower"},
        proposer="test", turn=2, propose_gen=20, rationale="retry",
    )
    assert not again["accepted"] and "graveyard" in again["reason"]


def test_pre_propose_rows_never_adjudicate(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    display = _propose_bb_rule(ledger, gen=100)
    # Overwhelming evidence, but all of it BEFORE propose_gen
    run_backtest(ledger, _rows(1, 40, bb_plpr=1e-3, hgp_plpr=1e-2))
    assert ledger.resolve(display).status == "hypothesis"


def test_insufficient_n_stays_hypothesis(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    display = _propose_bb_rule(ledger, gen=5)
    run_backtest(ledger, _rows(6, 4, bb_plpr=1e-3, hgp_plpr=1e-2))  # 8 rows < 20
    assert ledger.resolve(display).status == "hypothesis"


def test_concurrency_cap(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    for i in range(MAX_ACTIVE_HYPOTHESES):
        result = ledger.propose(
            scope={}, predicate=[{"field": "n", "op": ">", "value": i}],
            claim={}, proposer="t", turn=1, propose_gen=1, rationale="r",
        )
        assert result["accepted"]
    overflow = ledger.propose(
        scope={}, predicate=[{"field": "k", "op": ">", "value": 0}],
        claim={}, proposer="t", turn=1, propose_gen=1, rationale="r",
    )
    assert not overflow["accepted"] and "cap" in overflow["reason"]


def test_predicate_field_whitelist(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    bad = ledger.propose(
        scope={}, predicate=[{"field": "secret_field", "op": ">", "value": 1}],
        claim={}, proposer="t", turn=1, propose_gen=1, rationale="r",
    )
    assert not bad["accepted"]


def test_tags_and_bandit() -> None:
    tags = parse_hypothesis_tags("# exploits: I-3, I-7; tests: H-12\ndef construct(rng): ...")
    assert tags == {"exploits": ["I-3", "I-7"], "tests": ["H-12"], "strategy": None}
    assert parse_hypothesis_tags("def construct(rng): ...") == {
        "exploits": [], "tests": [], "strategy": None,
    }
    assert parse_hypothesis_tags("# strategy: transfer\ndef construct(rng): ...")["strategy"] == "transfer"
    assert parse_hypothesis_tags("# strategy: nonsense\ndef construct(rng): ...")["strategy"] is None

    rewards = turn_rewards_from_tags(
        [{"code_hash": "a", "exploits": ["I-3"]}, {"code_hash": "b"}],
        new_hof_hashes={"a", "b"},
    )
    assert rewards["exploit"] == 0.5 and rewards["novelty"] == 0.5

    # An explicit strategy tag routes credit to the transfer/wildcat arms,
    # which are otherwise structurally unrewardable.
    rewards = turn_rewards_from_tags(
        [{"code_hash": "a", "exploits": ["I-3"], "strategy": "transfer"}, {"code_hash": "b", "strategy": "wildcat"}],
        new_hof_hashes={"a", "b"},
    )
    assert rewards["transfer"] == 0.5 and rewards["wildcat"] == 0.5 and rewards["exploit"] == 0.0


def test_bandit_gen_guard_is_idempotent(tmp_path) -> None:
    bandit = StrategyBandit(tmp_path / "bandit.json")
    bandit.update({"exploit": 1.0}, gen=3)
    bandit.update({"exploit": 1.0}, gen=3)  # crash-resume replay: no double credit
    reloaded = StrategyBandit(tmp_path / "bandit.json")
    assert reloaded.stats["exploit"] == {"pulls": 1.0, "reward": 1.0}
    reloaded.update({"exploit": 1.0}, gen=4)
    assert reloaded.stats["exploit"] == {"pulls": 2.0, "reward": 2.0}


def test_bandit_mix(tmp_path) -> None:
    bandit = StrategyBandit(tmp_path / "bandit.json")
    first = bandit.suggestion()
    assert abs(sum(first["mix"].values()) - 1.0) < 1e-9
    bandit.update({"exploit": 1.0})
    bandit.update({"novelty": 0.2})
    suggestion = StrategyBandit(tmp_path / "bandit.json").suggestion()  # persisted
    mix = suggestion["mix"]
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert all(v >= suggestion["epsilon_arm"] - 1e-9 for v in mix.values())
