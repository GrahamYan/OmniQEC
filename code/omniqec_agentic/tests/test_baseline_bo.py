"""BO adapted selector (baselines/bo.py, plan B §5): views, Jeffreys objective, seeded fallbacks, determinism,
chunking, warm start, failure handling. Small model sizes keep it fast; skipped without the BO packages."""

from __future__ import annotations

import copy
import itertools
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from omniqec_agentic.baselines import bo  # noqa: E402
from omniqec_agentic.baselines.bo_graphs import NODE_TYPES, VIEWS_INFO, code_graph  # noqa: E402
from omniqec_agentic.construction.build import build_matrices  # noqa: E402
from omniqec_agentic.verification.gf2 import rank2  # noqa: E402


def _small_config(**chunk: int) -> dict:
    config = copy.deepcopy(bo.BO_CONFIG)
    config["embedding"]["d_model"] = 8
    config["embedding"]["num_layers"] = 2
    config["training"]["max_steps"] = 5
    config["objective"]["posterior_samples"] = 256
    config["acquisition"]["mc_samples"] = 16
    if chunk:
        config["acquisition"]["chunk"] = int(chunk["chunk"])
    # the snapshot copy the integration passes: a JSON round trip with a leading bo_id
    return json.loads(json.dumps({"bo_id": bo.BO_ID, **config}))


@pytest.fixture(scope="module")
def codes():
    out = []
    terms = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 1)]
    partners = [[(0, 0), (1, 2), (2, 1)], [(0, 1), (1, 0), (2, 2)], [(1, 0), (0, 2), (2, 1)]]
    for ell, m in [(3, 3), (3, 4), (3, 5), (3, 6), (4, 4), (4, 6)]:
        for a in itertools.combinations(terms, 3):
            for b in partners:
                spec = {"family": "bb", "ell": ell, "m": m, "A": [list(t) for t in a], "B": [list(t) for t in b]}
                try:
                    hx, hz = build_matrices(spec)
                except ValueError:
                    continue
                k = int(hx.shape[1] - rank2(hx) - rank2(hz))
                if k > 0:
                    out.append((f"bb{ell}x{m}_{len(out):03d}", hx, hz, k))
    assert len(out) >= 30
    return out[:30]


def _observations(codes, count, seed=1):
    rng = np.random.default_rng(seed)
    return [
        bo.Observation(h, hx, hz, k, 12, 1, int(rng.integers(0, 40)), 25000, int(rng.integers(0, 40)), 25000)
        for h, hx, hz, k in codes[:count]
    ]


def _pool(codes, start):
    return [bo.PoolItem(h, hx, hz) for h, hx, hz, _k in codes[start:]]


def test_code_graph_has_the_three_views_of_the_authors(codes) -> None:
    code_hash, hx, hz, k = codes[0]
    graph = code_graph(code_hash, hx, hz)
    assert graph.sizes == {"DQ": hx.shape[1], "SX": hx.shape[0], "SZ": hz.shape[0], "LX": k, "LZ": k}
    assert [view["name"] for view in VIEWS_INFO] == ["decode", "xlogic", "zlogic"]
    assert set(NODE_TYPES) == {"DQ", "SX", "SZ", "LX", "LZ"}
    nnz = {"SX": int(hx.sum()), "SZ": int(hz.sum())}
    for tag in ("SX_DQ", "DQ_SX"):
        assert graph.edges[tag][0].size == nnz["SX"]
    for tag in ("SZ_DQ", "DQ_SZ"):
        assert graph.edges[tag][0].size == nnz["SZ"]
    # direction: SX_DQ sends check -> qubit, so its destinations are qubit indices
    dst, src = graph.edges["SX_DQ"]
    assert dst.max() < graph.sizes["DQ"] and src.max() < graph.sizes["SX"]
    dst, src = graph.edges["DQ_LX"]
    assert dst.max() < graph.sizes["LX"] and src.max() < graph.sizes["DQ"]


def test_jeffreys_objective_is_finite_at_zero_failures_and_monotone() -> None:
    kwargs = {"k": 4, "rounds": 12, "blocks": 1, "samples": 4096, "seed": 3}
    y0, v0 = bo.jeffreys_objective(0, 25000, 0, 25000, **kwargs)
    y5, v5 = bo.jeffreys_objective(5, 25000, 5, 25000, **kwargs)
    y50, _ = bo.jeffreys_objective(50, 25000, 50, 25000, **kwargs)
    assert np.isfinite(y0) and np.isfinite(v0) and v0 > 0
    assert y0 > y5 > y50
    assert v0 > v5  # a zero-failure code is the least certain one
    assert bo.jeffreys_objective(5, 25000, 5, 25000, **kwargs) == (y5, v5)
    # two blocks halve the per-logical rate: larger -log10
    y5_cnot, _ = bo.jeffreys_objective(5, 25000, 5, 25000, **{**kwargs, "blocks": 2})
    assert y5_cnot > y5


def test_seeded_uniform_choice_is_distinct_and_order_independent() -> None:
    hashes = [f"h{i:02d}" for i in range(20)]
    first = bo.seeded_uniform_choice(hashes, 6, seed=11)
    assert len(first) == len(set(first)) == 6
    assert bo.seeded_uniform_choice(list(reversed(hashes)), 6, seed=11) == first
    assert bo.seeded_uniform_choice(hashes, 50, seed=11) != [] and len(bo.seeded_uniform_choice(hashes, 50, 1)) == 20


def test_initial_design_and_fallback_nominate_uniformly(codes, tmp_path) -> None:
    config = _small_config()
    pool = _pool(codes, 10)
    picks, diag = bo.select_batch(
        observations=_observations(codes, 10), pool=pool, q=6, seed=5, state_dir=tmp_path, turn=2, config=config
    )
    assert diag["phase"] == "initial_design" and len(picks) == len(set(picks)) == 6
    assert not list(tmp_path.iterdir())  # no model state before the first BO round
    unusable = [bo.Observation(h, hx, hz, k, 12, 1, 0, 0, 0, 0) for h, hx, hz, k in codes[:5]]
    picks, diag = bo.select_batch(
        observations=_observations(codes, 1) + unusable[1:], pool=pool, q=6, seed=5, state_dir=tmp_path, turn=4,
        config=config,
    )
    assert diag["fallback"] is True and diag["n_obs_usable"] == 1 and len(picks) == 6
    assert set(picks) <= {item.code_hash for item in pool}


def test_bo_round_is_deterministic_distinct_and_json(codes, tmp_path) -> None:
    config = _small_config()
    obs, pool = _observations(codes, 10), _pool(codes, 10)
    first, diag = bo.select_batch(
        observations=obs, pool=pool, q=5, seed=9, state_dir=tmp_path / "a", turn=4, config=config
    )
    second, _ = bo.select_batch(
        observations=obs, pool=list(reversed(pool)), q=5, seed=9, state_dir=tmp_path / "b", turn=4, config=config
    )
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert set(first) <= {item.code_hash for item in pool}
    assert diag["phase"] == "bo" and diag["fallback"] is False and diag["warm_start"] is False
    assert [pick["rank"] for pick in diag["picks"]] == [1, 2, 3, 4, 5]
    json.dumps(diag)
    # observed codes never return as candidates
    picks, _ = bo.select_batch(
        observations=obs, pool=_pool(codes, 0), q=5, seed=9, state_dir=tmp_path / "c", turn=4, config=config
    )
    assert not set(picks) & {item.code_hash for item in obs}


def test_chunked_inference_matches_one_chunk(codes, tmp_path) -> None:
    obs, pool = _observations(codes, 10), _pool(codes, 10)
    whole, _ = bo.select_batch(
        observations=obs, pool=pool, q=4, seed=13, state_dir=tmp_path / "a", turn=4, config=_small_config(chunk=1024)
    )
    pieces, _ = bo.select_batch(
        observations=obs, pool=pool, q=4, seed=13, state_dir=tmp_path / "b", turn=4, config=_small_config(chunk=3)
    )
    assert whole == pieces
    parts = bo._torch_parts()
    torch.manual_seed(0)
    embedder = parts["ChainComplexEmbedder"](d_model=8, num_layers=2, num_bases=4, dropout=0.1)
    graphs = [code_graph(h, hx, hz) for h, hx, hz, _k in codes[:12]]
    one = bo.embed_graphs(embedder, graphs, chunk=1024)
    many = bo.embed_graphs(embedder, graphs, chunk=5)
    assert torch.allclose(one, many, atol=1e-5)


def test_warm_start_reads_the_previous_turn_and_ignores_a_foreign_state(codes, tmp_path) -> None:
    config = _small_config()
    obs, pool = _observations(codes, 10), _pool(codes, 10)
    bo.select_batch(observations=obs, pool=pool, q=3, seed=4, state_dir=tmp_path, turn=4, config=config)
    saved = torch.load(bo.state_path(tmp_path, 4), weights_only=True)
    assert saved["bo_id"] == bo.BO_ID and saved["turn"] == 4 and saved["round_index"] == 0
    embedder = bo._torch_parts()["ChainComplexEmbedder"](d_model=8, num_layers=2, num_bases=4, dropout=0.1)
    embedder.load_state_dict(saved["embed"])  # round trip of the saved embedder
    assert torch.isfinite(bo.embed_graphs(embedder, [code_graph(*codes[0][:3])], chunk=8)).all()
    _, diag = bo.select_batch(
        observations=_observations(codes, 12), pool=_pool(codes, 12), q=3, seed=4, state_dir=tmp_path, turn=5,
        config=config,
    )
    assert diag["warm_start"] is True and diag["warm_start_source"] == "bo_model_turn0004.pt"
    assert diag["training"]["round_index"] == 1 and diag["training"]["lengthscale_frozen"] is True
    assert bo.state_path(tmp_path, 5).exists()
    assert not bo.state_path(tmp_path, 3).exists() and bo.state_path(tmp_path, 4).exists()  # newest two kept
    other = copy.deepcopy(config)
    other["training"]["max_steps"] = 4
    _, diag = bo.select_batch(
        observations=_observations(codes, 12), pool=_pool(codes, 12), q=3, seed=4, state_dir=tmp_path, turn=6,
        config=other,
    )
    assert diag["warm_start"] is False and "ignored" in diag["warm_start_note"]
    assert not bo.state_path(tmp_path, 4).exists() and diag["state_files_pruned"] == ["bo_model_turn0004.pt"]
    bo.state_path(tmp_path, 7).write_bytes(b"not a torch file")
    _, diag = bo.select_batch(
        observations=_observations(codes, 12), pool=_pool(codes, 12), q=3, seed=4, state_dir=tmp_path, turn=8,
        config=config,
    )
    assert diag["warm_start"] is False and "unreadable" in diag["warm_start_note"]


def test_numerical_failure_retries_fresh_then_raises(codes, tmp_path, monkeypatch) -> None:
    config = _small_config()
    obs, pool = _observations(codes, 10), _pool(codes, 10)
    bo.select_batch(observations=obs, pool=pool, q=2, seed=4, state_dir=tmp_path, turn=4, config=config)
    real_train = bo._train
    calls = {"n": 0}

    def flaky(model, train_y, *, config, round_index):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FloatingPointError("injected")
        return real_train(model, train_y, config=config, round_index=round_index)

    monkeypatch.setattr(bo, "_train", flaky)
    picks, diag = bo.select_batch(observations=obs, pool=pool, q=2, seed=4, state_dir=tmp_path, turn=5, config=config)
    assert len(picks) == 2 and diag["warm_start"] is False and len(diag["retry_errors"]) == 1
    assert "warm" in diag["retry_errors"][0]

    def broken(model, train_y, *, config, round_index):
        raise FloatingPointError("injected")

    monkeypatch.setattr(bo, "_train", broken)
    with pytest.raises(RuntimeError, match="after a fresh retry"):
        bo.select_batch(observations=obs, pool=pool, q=2, seed=4, state_dir=tmp_path, turn=6, config=config)


def test_threads_come_from_the_environment_and_are_restored(codes, tmp_path, monkeypatch) -> None:
    before = torch.get_num_threads()
    monkeypatch.setenv("OMNIQEC_BO_THREADS", "3")
    _, diag = bo.select_batch(
        observations=_observations(codes, 10), pool=_pool(codes, 10), q=2, seed=4, state_dir=tmp_path, turn=4,
        config=_small_config(),
    )
    assert diag["torch_threads"] == 3 and torch.get_num_threads() == before
    monkeypatch.setenv("OMNIQEC_BO_THREADS", "0")
    assert bo.bo_threads() == 1
    monkeypatch.delenv("OMNIQEC_BO_THREADS")
    assert bo.bo_threads() == 16


def test_a_foreign_bo_id_is_refused(codes, tmp_path) -> None:
    config = _small_config()
    config["bo_id"] = "bo_something_else"
    with pytest.raises(ValueError, match="bo_id"):
        bo.select_batch(observations=[], pool=_pool(codes, 0), q=2, seed=1, state_dir=tmp_path, turn=4, config=config)
