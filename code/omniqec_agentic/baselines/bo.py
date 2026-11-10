"""BO adapted: Gaussian-process selection of the slow-loop codes (plan B §5, user decision 2026-11-06).

Source. Chengyu et al., "Bayesian Optimization for Quantum Error-Correcting Code Discovery" (arXiv:2601.18562) and
the authors' code, BO_for_QECcodes@dc4593fc8d0e993c68bce8eb6c9ce01bccdcf994 (MIT licence). This module is a
re-implementation of their method skeleton -- three-view chain-complex embedding, deep-kernel GP, expected
improvement -- not an import of their code. Values taken from their entry point ``BOplayground.py``:

- views (lines 734-747): ``decode`` (SZ, DQ, SX), ``xlogic`` (DQ, SX, LX), ``zlogic`` (SZ, DQ, LZ), see bo_graphs.
- embedder (lines 754-764 -> ``bayesian_optimization/chaincomplexembedding.py`` ``ChainComplexEmbedder``, lines
  700-1073, and ``ChainComplexMessagePassingLayer``, lines 473-683): d_model 128, 4 layers, a separate layer per
  view, per-type learned token as the initial node feature, PreNorm (LayerNorm) + bias-free residual projection per
  destination type, relation weights W_r with ``num_bases=4`` -- which their rule ``num_bases >= len(relations)``
  turns into one W_r per relation for the four relations of every view --, symmetric normalisation
  D_dst^-1/2 A D_src^-1/2, a scalar gate per relation, sum over relations, update MLP Linear(2d, max(2d, 64)) -
  GELU - Dropout - Linear(., d), output dropout, residual sum, views summed with a scalar gate per (layer, view);
  readout = per-type mean pool over DQ, LX, LZ, SX, SZ, concatenated, LayerNorm - Linear(5d, 5d) - GELU - Dropout -
  Linear(5d, d); dropout 0.1, no self loop, xavier-uniform initialisation.
- GP (lines 749-802, 916-922; ``bayesian_optimization/gp.py`` ``GaussianProcess_QEC``): ScaleKernel(Matern
  nu=1.5, ARD over the 128 embedding dimensions), LinearMean on the embedding.
- trainer (lines 924-956, ``GPTrainer`` lines 395-722): at most 80 AdamW steps per round on the exact marginal
  log-likelihood, parameter groups embed 4e-4 / mean 1e-3 / kernel 1e-3 / likelihood 2e-2, weight decay 1e-4
  (likelihood 1e-4 / 5), gradient-norm clip 2.0, ReduceLROnPlateau(factor 0.5, patience 5, min_lr 1e-6) stepped
  on the loss, early stop after 10 consecutive steps without improvement (paper appendix D4; their extra condition
  "and the relative improvement is below 1e-4" is not applied, deviation 8), best state restored, optimizer rebuilt
  every round, warm start across rounds, outputscale and noise rescaled by
  (old_sd / new_sd)^2 when the target scaler changes, base-kernel lengthscale frozen for rounds 0-2, LogNormal
  priors lengthscale (0, 0.5), outputscale (0, 0.5), noise (-4, 0.5), noise floor 1e-3.

Deviations (plan B §5 and this project's protocol; every one recorded in BO_CONFIG["deviations"]):

1. Objective: circuit-level measurement of this project (two bases, T = 12, b blocks) instead of their
   code-capacity simulation and decoder; the GP target is y = -log10(pLER) with
   pLER = 1 - [(1 - qX)(1 - qZ)]^(1 / (b k T)), estimated per code from the Jeffreys posterior
   q ~ Beta(f + 1/2, s - f + 1/2) of each basis with 4096 fixed-seed samples (sample mean = observation, sample
   variance = its noise), so a zero-failure code has a finite target with a wide variance instead of a point at zero.
   Their GP models z = standardised log(pL + 1e-8): ours is the same log transform up to sign, base and scale.
2. Likelihood: per-point fixed noise (the Jeffreys variances) plus one learnable residual noise (their prior
   LogNormal(-4, 0.5) and floor 1e-3, initialised at the prior median e^-4) instead of their single homoscedastic
   noise: racing stops make the measurement precision differ by orders of magnitude between codes. Targets and
   variances are standardised together.
3. Acquisition: qLogNoisyExpectedImprovement (BoTorch, 256 Sobol samples, fixed seed) on the latent-function
   posterior, instead of their analytic EI on the predictive posterior (which adds the observation noise once more).
4. Candidates: greedy sequential choice of the turn's slow slots over the run's cumulative pool of legal, unmeasured
   codes (pending picks condition the next pick), instead of their bit-flip hill climbing from random BB parameter
   vectors; the pool comes from the baseline sampler, which is the only proposal mechanism.
5. Input: any CSS code (HX, HZ; LX, LZ from verification.gf2.logicals) instead of their BB parameter vector; the
   embedding is computed from the matrices exactly as their RelationEncoder does.
6. Initial design: the first ``init_turns`` (4) turns nominate uniformly at random (24 Memory / 28 CNOT codes) instead
   of their 20 random initial points; each later turn is one BO round (round index = turn - init_turns).
7. Numerics: the embedder trains in float32 as theirs and is evaluated in float64 for the GP inputs, the GP runs in
   float64 (BoTorch's recommendation); the acquisition baseline drops repeated embedding rows (WL-equivalent codes
   embed identically, which made the joint covariance singular and the Cholesky jitter flip on float32 noise), exact
   ties go to the smallest code hash; a fixed torch
   thread count and seeded RNG make a selection reproducible; the gradient clip is computed over both dtypes.
8. Early stop (v2, paper alignment 2026-11-10): training stops after ``patience`` (10) consecutive steps without
   improvement, an improvement being a loss decrease of more than ``min_delta`` (1e-6), as the paper states it
   (appendix D4: "early stopping after ten steps without improvement"); their additional relative-improvement
   condition (tol 1e-4 on the last step) is dropped. It only differed when |loss| < 1e-2.

Selection API: ``select_batch`` returns the chosen code hashes and a JSON diagnostics dict. torch, gpytorch and
botorch are imported inside functions only, so importing this module needs numpy alone."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bo_graphs import NODE_TYPES, VIEWS_INFO, CodeGraph, code_graph, relation_endpoints

BO_ID = "bo_adapted_v2"
AUTHORS_SOURCE = {
    "paper": "arXiv:2601.18562",
    "repository": "https://github.com/Chengyuyihua/BO_for_QECcodes",
    "commit": "dc4593fc8d0e993c68bce8eb6c9ce01bccdcf994",
    "licence": "MIT",
    "entry_point": "BOplayground.py",
}

BO_CONFIG: dict[str, Any] = {
    "bo_id": BO_ID,
    "source": AUTHORS_SOURCE,
    "init_turns": 4,
    "min_observations": 2,
    "objective": {
        "target": "-log10(pLER)",
        "pler": "1 - [(1 - qX)(1 - qZ)]^(1/(b k T))",
        "posterior": "Jeffreys Beta(f + 1/2, s - f + 1/2) per basis",
        "posterior_samples": 4096,
        "standardise": "targets and variances together",
    },
    "embedding": {
        "views": [
            {"name": view["name"], "partite_classes": list(view["partite_classes"]), "relations": list(view["relations"])}
            for view in VIEWS_INFO
        ],
        "node_types": list(NODE_TYPES),
        "d_model": 128,
        "num_layers": 4,
        "num_bases": 4,
        "norm": "sym",
        "residual": True,
        "self_loop": False,
        "dropout": 0.1,
        "relation_aggr": "sum",
        "view_aggr": "sum",
        "activation": "gelu",
        "readout": "per-type mean pool, concat, LayerNorm-Linear-GELU-Dropout-Linear",
        "init": "xavier uniform, seeded",
        "dtype": "float32",
    },
    "gp": {
        "kernel": "ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=d_model))",
        "mean": "LinearMean(d_model)",
        "lengthscale_prior": [0.0, 0.5],
        "outputscale_prior": [0.0, 0.5],
        "residual_noise_prior": [-4.0, 0.5],
        "residual_noise_floor": 1e-3,
        "residual_noise_init": math.exp(-4.0),
        "observation_noise": "fixed per point (Jeffreys variance) + learnable residual",
        "dtype": "float64",
    },
    "training": {
        "max_steps": 80,
        "optimizer": "adamw",
        "lr": {"embed": 4e-4, "mean": 1e-3, "kernel": 1e-3, "like": 2e-2},
        "weight_decay": 1e-4,
        "likelihood_weight_decay_factor": 0.2,
        "grad_clip": 2.0,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 5, "min_lr": 1e-6},
        "early_stop": {"patience": 10, "min_delta": 1e-6},
        "warm_start": True,
        "recreate_optimizer_each_round": True,
        "save_best_state": True,
        "rescale_on_scaler_change": True,
        "freeze_lengthscale_until_round": 2,
    },
    "acquisition": {
        "name": "qLogNoisyExpectedImprovement",
        "mc_samples": 256,
        "sampler": "SobolQMCNormalSampler (seeded)",
        "posterior": "latent function (observation_noise=False)",
        "prune_baseline": False,
        "cache_root": True,
        "batch": "greedy sequential, picks of this turn as X_pending",
        "chunk": 1024,
        "inference_dtype": "float64",
        "baseline_dedupe": "repeated embedding rows dropped from X_baseline",
        "tie_tolerance": 1e-9,
        "tie_break": "smallest code_hash",
    },
    # The NSCC launcher exports OMP/MKL_NUM_THREADS=1 for the decoder workers; select_batch sets torch's intra-op
    # threads itself (the decoder pool is idle while the slow queue is built) and restores them afterwards.
    "threads": {"env": "OMNIQEC_BO_THREADS", "default": 16},
    # Model state files (33 MB each at d_model 128): a warm start reads the previous turn only, so the newest two
    # are kept (a retried turn t still finds t - 1); 0 keeps every file.
    "state_files_kept": 2,
    "deviations": [
        "circuit-level two-basis objective -log10(pLER) from the Jeffreys posterior instead of code-capacity pL",
        "fixed per-point noise + learnable residual noise instead of one homoscedastic noise",
        "qLogNoisyExpectedImprovement on the latent posterior instead of analytic EI on the predictive posterior",
        "greedy batch over the cumulative sampler pool instead of bit-flip hill climbing in BB parameter space",
        "CSS matrices (HX, HZ, LX, LZ) of any family as input instead of a BB parameter vector",
        "first 4 turns uniform random nominations instead of 20 random initial points",
        "embedder trained float32 / evaluated float64, GP float64, baseline rows deduplicated, ties to the smallest"
        " code hash, fixed thread count, gradient clip over both dtypes",
        "early stop after 10 steps without improvement (min_delta 1e-6), without their extra relative-tolerance"
        " condition",
    ],
}


@dataclass(frozen=True, eq=False)
class Observation:
    """One slow-measured code: its matrices, k, rounds T, blocks b and the per-basis failures and shots."""

    code_hash: str
    hx: np.ndarray
    hz: np.ndarray
    k: int
    rounds: int
    blocks: int
    fail_x: int
    shots_x: int
    fail_z: int
    shots_z: int

    def usable(self) -> bool:
        try:
            return (
                int(self.k) >= 1 and int(self.rounds) >= 1 and int(self.blocks) >= 1
                and int(self.shots_x) >= 1 and int(self.shots_z) >= 1
                and 0 <= int(self.fail_x) <= int(self.shots_x) and 0 <= int(self.fail_z) <= int(self.shots_z)
            )
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True, eq=False)
class PoolItem:
    """One legal, not yet measured candidate of the run's cumulative pool."""

    code_hash: str
    hx: np.ndarray
    hz: np.ndarray


def stable_seed(*parts: Any) -> int:
    """A 63-bit seed from the parts' text (independent of call order and of Python's hash randomisation)."""

    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def jeffreys_objective(
    fail_x: int,
    shots_x: int,
    fail_z: int,
    shots_z: int,
    *,
    k: int,
    rounds: int,
    blocks: int,
    samples: int = 4096,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean and variance of y = -log10(pLER) under the per-basis Jeffreys posteriors (plan B §5.4).

    pLER = 1 - [(1 - qX)(1 - qZ)]^(1/(b k T)) is evaluated as -expm1(c (log1p(-qX) + log1p(-qZ))), c = 1/(b k T),
    so a tiny q keeps its precision; samples are clipped to [1e-300, 1 - 1e-16] (a Beta(1/2, s) draw can underflow).
    """

    exponent = 1.0 / float(int(blocks) * int(k) * int(rounds))
    rng = np.random.default_rng(int(seed))
    qx = rng.beta(int(fail_x) + 0.5, int(shots_x) - int(fail_x) + 0.5, size=int(samples))
    qz = rng.beta(int(fail_z) + 0.5, int(shots_z) - int(fail_z) + 0.5, size=int(samples))
    qx = np.clip(qx, 1e-300, 1.0 - 1e-16)
    qz = np.clip(qz, 1e-300, 1.0 - 1e-16)
    pler = -np.expm1(exponent * (np.log1p(-qx) + np.log1p(-qz)))
    y = -np.log10(np.clip(pler, 1e-300, 1.0))
    return float(np.mean(y)), float(np.var(y, ddof=1))


def observation_target(observation: Observation, *, samples: int) -> tuple[float, float]:
    """The GP observation of one measured code; the posterior samples are seeded by its code hash alone, so a code
    keeps the same target in every turn."""

    return jeffreys_objective(
        observation.fail_x, observation.shots_x, observation.fail_z, observation.shots_z,
        k=observation.k, rounds=observation.rounds, blocks=observation.blocks,
        samples=samples, seed=stable_seed(BO_ID, "jeffreys", observation.code_hash),
    )


def seeded_uniform_choice(code_hashes: Sequence[str], q: int, seed: int) -> list[str]:
    """q distinct hashes uniformly without replacement; independent of the input order (sorted first)."""

    ordered = sorted({str(item) for item in code_hashes})
    take = min(max(0, int(q)), len(ordered))
    if take == 0:
        return []
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(len(ordered), size=take, replace=False)
    return [ordered[int(index)] for index in picks]


def state_path(state_dir: Path | str, turn: int) -> Path:
    return Path(state_dir) / f"bo_model_turn{int(turn):04d}.pt"


def bo_threads() -> int:
    """torch intra-op threads for a BO round: env OMNIQEC_BO_THREADS, default 16, at least 1."""

    raw = str(os.environ.get("OMNIQEC_BO_THREADS", "") or "").strip()
    try:
        value = int(raw) if raw else 16
    except ValueError:
        value = 16
    return max(1, value)


# ---------------------------------------------------------------------------------------------------------------
# torch parts (built on first use)
# ---------------------------------------------------------------------------------------------------------------

_TORCH_PARTS: dict[str, Any] = {}


def _torch_parts() -> dict[str, Any]:
    if _TORCH_PARTS:
        return _TORCH_PARTS
    import torch
    from torch import nn

    class ChainComplexLayer(nn.Module):
        """One view's message passing layer (the authors' ChainComplexMessagePassingLayer, sum aggregation)."""

        def __init__(self, relations: Sequence[str], d_model: int, num_bases: int | None, dropout: float) -> None:
            super().__init__()
            self.relations = list(relations)
            self.d_model = int(d_model)
            self.ln_in = nn.ModuleDict({t: nn.LayerNorm(self.d_model) for t in NODE_TYPES})
            self.res_proj = nn.ModuleDict({t: nn.Linear(self.d_model, self.d_model, bias=False) for t in NODE_TYPES})
            hidden = max(self.d_model * 2, 64)
            self.update_mlp = nn.ModuleDict({
                t: nn.Sequential(
                    nn.Linear(self.d_model * 2, hidden, bias=True),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, self.d_model, bias=True),
                )
                for t in NODE_TYPES
            })
            # The authors' rule: bases only when they are fewer than the relations (never, for 4 bases and 4
            # relations per view).
            if num_bases is None or int(num_bases) >= len(self.relations):
                self.weight_mode = "per_relation"
                self.rel_W = nn.ParameterDict({
                    r: nn.Parameter(torch.empty(self.d_model, self.d_model)) for r in self.relations
                })
                for param in self.rel_W.values():
                    nn.init.xavier_uniform_(param)
            else:
                self.weight_mode = "basis"
                bases = int(num_bases)
                self.bases = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.d_model, self.d_model)) for _ in range(bases)]
                )
                for param in self.bases:
                    nn.init.xavier_uniform_(param)
                self.rel_coeff = nn.ParameterDict({
                    r: nn.Parameter(torch.randn(bases) / (bases ** 0.5)) for r in self.relations
                })
            self.rel_gate = nn.ParameterDict({r: nn.Parameter(torch.tensor(1.0)) for r in self.relations})
            self.dropout_out = nn.Dropout(float(dropout))

        def _weight(self, relation: str):
            if self.weight_mode == "per_relation":
                return self.rel_W[relation]
            coeff = self.rel_coeff[relation]
            weight = torch.zeros_like(self.bases[0])
            for alpha, basis in zip(coeff, self.bases):
                weight = weight + alpha * basis
            return weight

        def forward(self, H: dict[str, Any], adjacency: dict[str, Any]) -> dict[str, Any]:
            proj: dict[str, Any] = {}
            agg: dict[str, Any] = {}
            for t in NODE_TYPES:
                x = H[t]
                proj[t] = self.res_proj[t](self.ln_in[t](x)) if x.shape[0] else x.new_zeros((0, self.d_model))
                agg[t] = x.new_zeros((x.shape[0], self.d_model))
            for relation in self.relations:
                src, dst = relation_endpoints(relation)
                if H[src].shape[0] == 0 or H[dst].shape[0] == 0:
                    continue
                message = torch.sparse.mm(adjacency[relation], H[src] @ self._weight(relation))
                agg[dst] = agg[dst] + message * self.rel_gate[relation]
            out: dict[str, Any] = {}
            for t in NODE_TYPES:
                if H[t].shape[0] == 0:
                    out[t] = proj[t]
                    continue
                update = self.dropout_out(self.update_mlp[t](torch.cat([proj[t], agg[t]], dim=-1)))
                out[t] = proj[t] + update
            return out

    class ChainComplexEmbedder(nn.Module):
        """Three views x num_layers layers, per-type mean pool and readout (the authors' ChainComplexEmbedder)."""

        def __init__(self, *, d_model: int, num_layers: int, num_bases: int | None, dropout: float) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.view_names = [str(view["name"]) for view in VIEWS_INFO]
            self.type_tokens = nn.ParameterDict({t: nn.Parameter(torch.empty(1, self.d_model)) for t in NODE_TYPES})
            for param in self.type_tokens.values():
                nn.init.xavier_uniform_(param)
            self.view_gates = nn.ParameterList([
                nn.ParameterDict({name: nn.Parameter(torch.tensor(1.0)) for name in self.view_names})
                for _ in range(int(num_layers))
            ])
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    str(view["name"]): ChainComplexLayer(view["relations"], self.d_model, num_bases, dropout)
                    for view in VIEWS_INFO
                })
                for _ in range(int(num_layers))
            ])
            read_dim = len(NODE_TYPES) * self.d_model
            hidden = max(read_dim, 128)
            self.readout_mlp = nn.Sequential(
                nn.LayerNorm(read_dim),
                nn.Linear(read_dim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.d_model),
            )

        def forward(self, packed: dict[str, Any]):
            totals = packed["totals"]
            H = {t: self.type_tokens[t].expand(int(totals[t]), -1) for t in NODE_TYPES}
            for layer_index, per_view in enumerate(self.layers):
                outputs = {name: per_view[name](H, packed["views"][name]) for name in self.view_names}
                H = {
                    t: sum(self.view_gates[layer_index][name] * outputs[name][t] for name in self.view_names)
                    for t in NODE_TYPES
                }
            pooled = []
            batch = int(packed["batch"])
            for t in NODE_TYPES:
                sums = H[t].new_zeros((batch, self.d_model)).index_add_(0, packed["segments"][t], H[t])
                pooled.append(sums / packed["counts"][t].clamp_min(1.0).unsqueeze(-1))
            return self.readout_mlp(torch.cat(pooled, dim=-1))

    _TORCH_PARTS.update({"ChainComplexLayer": ChainComplexLayer, "ChainComplexEmbedder": ChainComplexEmbedder})
    return _TORCH_PARTS


def pack_graphs(graphs: Sequence[CodeGraph], *, dtype: Any = None) -> dict[str, Any]:
    """Block-diagonal batch of graphs: per relation one sparse (|DST| x |SRC|) matrix with the symmetric
    normalisation folded into its values (D_dst^-1/2 A D_src^-1/2), per type the node-to-sample segment ids."""

    import torch

    value_dtype = torch.float32 if dtype is None else dtype
    batch = len(graphs)
    counts = {t: np.array([int(graph.sizes[t]) for graph in graphs], dtype=np.int64) for t in NODE_TYPES}
    offsets = {t: np.concatenate([[0], np.cumsum(counts[t])]).astype(np.int64) for t in NODE_TYPES}
    totals = {t: int(offsets[t][-1]) for t in NODE_TYPES}
    views: dict[str, dict[str, Any]] = {}
    for view in VIEWS_INFO:
        adjacency: dict[str, Any] = {}
        for relation in view["relations"]:
            src, dst = relation_endpoints(relation)
            dst_parts, src_parts = [], []
            for index, graph in enumerate(graphs):
                dst_idx, src_idx = graph.edges[relation]
                dst_parts.append(dst_idx + offsets[dst][index])
                src_parts.append(src_idx + offsets[src][index])
            dst_all = np.concatenate(dst_parts) if dst_parts else np.zeros(0, np.int64)
            src_all = np.concatenate(src_parts) if src_parts else np.zeros(0, np.int64)
            deg_dst = np.bincount(dst_all, minlength=totals[dst]).astype(np.float64)
            deg_src = np.bincount(src_all, minlength=totals[src]).astype(np.float64)
            values = (deg_dst[dst_all] ** -0.5) * (deg_src[src_all] ** -0.5) if dst_all.size else np.zeros(0)
            adjacency[relation] = torch.sparse_coo_tensor(
                torch.from_numpy(np.vstack([dst_all, src_all])),
                torch.from_numpy(values).to(value_dtype),
                size=(totals[dst], totals[src]),
            ).coalesce()
        views[str(view["name"])] = adjacency
    return {
        "batch": batch,
        "totals": totals,
        "views": views,
        "segments": {t: torch.from_numpy(np.repeat(np.arange(batch, dtype=np.int64), counts[t])) for t in NODE_TYPES},
        "counts": {t: torch.from_numpy(counts[t]).to(value_dtype) for t in NODE_TYPES},
    }


def embed_graphs(embedder: Any, graphs: Sequence[CodeGraph], *, chunk: int) -> Any:
    """Eval-mode embeddings of graphs in chunks (float64 rows, same order as graphs)."""

    import torch

    dtype = next(embedder.parameters()).dtype
    rows = []
    embedder.eval()
    with torch.no_grad():
        for start in range(0, len(graphs), max(1, int(chunk))):
            part = graphs[start:start + max(1, int(chunk))]
            rows.append(embedder(pack_graphs(part, dtype=dtype)).to(torch.float64))
    if not rows:
        return torch.zeros((0, int(embedder.d_model)), dtype=torch.float64)
    return torch.cat(rows, dim=0)


def _inference_dtype(config: dict[str, Any]) -> Any:
    import torch

    return torch.float64 if str(config["acquisition"].get("inference_dtype", "float64")) == "float64" else torch.float32


def _unique_rows(z: Any) -> Any:
    """X_baseline without repeated rows (first occurrence kept): the incumbent is a max, a repeat adds nothing."""

    import torch

    kept: list[int] = []
    seen: set[bytes] = set()
    rounded = torch.round(z * 1e9).to(torch.int64).cpu().numpy()
    for index in range(rounded.shape[0]):
        key = rounded[index].tobytes()
        if key not in seen:
            seen.add(key)
            kept.append(index)
    return z[kept]


def _build_model(config: dict[str, Any], train_y: Any, train_var: Any, packed: dict[str, Any]) -> Any:
    import gpytorch
    import torch
    from gpytorch.constraints import GreaterThan
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from gpytorch.means import LinearMean
    from gpytorch.priors import LogNormalPrior

    emb_cfg = config["embedding"]
    gp_cfg = config["gp"]
    d_model = int(emb_cfg["d_model"])
    embedder = _torch_parts()["ChainComplexEmbedder"](
        d_model=d_model, num_layers=int(emb_cfg["num_layers"]),
        num_bases=None if emb_cfg.get("num_bases") is None else int(emb_cfg["num_bases"]),
        dropout=float(emb_cfg["dropout"]),
    )
    ls_loc, ls_scale = gp_cfg["lengthscale_prior"]
    os_loc, os_scale = gp_cfg["outputscale_prior"]
    nz_loc, nz_scale = gp_cfg["residual_noise_prior"]
    likelihood = FixedNoiseGaussianLikelihood(
        noise=train_var,
        learn_additional_noise=True,
        noise_prior=LogNormalPrior(float(nz_loc), float(nz_scale)),
        noise_constraint=GreaterThan(float(gp_cfg["residual_noise_floor"])),
    ).to(torch.float64)
    likelihood.second_noise_covar.noise = torch.tensor(float(gp_cfg["residual_noise_init"]), dtype=torch.float64)

    class DeepKernelGP(gpytorch.models.ExactGP):
        """GP over the embedding; its inputs are the training rows' indices, its forward embeds their graphs."""

        def __init__(self) -> None:
            train_x = torch.arange(int(packed["batch"]), dtype=torch.float64).unsqueeze(-1)
            super().__init__(train_x, train_y, likelihood)
            self.embed = embedder
            self.mean_module = LinearMean(d_model).to(torch.float64)
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, ard_num_dims=d_model, lengthscale_prior=LogNormalPrior(float(ls_loc), float(ls_scale))),
                outputscale_prior=LogNormalPrior(float(os_loc), float(os_scale)),
            ).to(torch.float64)
            self._packed = packed

        def forward(self, x: Any) -> Any:
            z = self.embed(self._packed).to(torch.float64)
            return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))

    return DeepKernelGP()


def _clip_total_norm(parameters: list[Any], max_norm: float) -> float:
    """torch.nn.utils.clip_grad_norm_ over parameters of two dtypes (norm accumulated in float64)."""

    import torch

    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return 0.0
    total = math.sqrt(sum(float(torch.sum(grad.detach().to(torch.float64) ** 2)) for grad in grads))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for grad in grads:
            grad.mul_(scale)
    return total


def _load_warm_state(model: Any, state: dict[str, Any], *, new_sd: float, config: dict[str, Any]) -> None:
    import torch

    model.embed.load_state_dict(state["embed"])
    model.mean_module.load_state_dict(state["mean"])
    model.covar_module.load_state_dict(state["covar"])
    with torch.no_grad():
        model.likelihood.second_noise_covar.raw_noise.copy_(state["residual_raw_noise"])
    old_sd = float(state.get("y_sd") or 0.0)
    if bool(config["training"].get("rescale_on_scaler_change", True)) and old_sd > 0 and new_sd > 0:
        alpha_sq = (old_sd / new_sd) ** 2
        floor = float(config["gp"]["residual_noise_floor"])
        with torch.no_grad():
            model.covar_module.outputscale = model.covar_module.outputscale * alpha_sq
            noise = model.likelihood.second_noise_covar.noise * alpha_sq
            model.likelihood.second_noise_covar.noise = torch.clamp(noise, min=floor * (1.0 + 1e-9))


def _train(model: Any, train_y: Any, *, config: dict[str, Any], round_index: int) -> dict[str, Any]:
    """One training round of the authors' GPTrainer."""

    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_cfg = config["training"]
    lr = train_cfg["lr"]
    decay = float(train_cfg["weight_decay"])
    freeze = int(round_index) <= int(train_cfg["freeze_lengthscale_until_round"])
    for param in model.covar_module.base_kernel.parameters():
        param.requires_grad_(not freeze)
    groups = [
        {"params": list(model.embed.parameters()), "lr": float(lr["embed"]), "weight_decay": decay},
        {"params": list(model.mean_module.parameters()), "lr": float(lr["mean"]), "weight_decay": decay},
        {"params": list(model.covar_module.parameters()), "lr": float(lr["kernel"]), "weight_decay": decay},
        {
            "params": list(model.likelihood.parameters()),
            "lr": float(lr["like"]),
            "weight_decay": decay * float(train_cfg["likelihood_weight_decay_factor"]),
        },
    ]
    optimizer = torch.optim.AdamW(groups)
    sched_cfg = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]), min_lr=float(sched_cfg["min_lr"]),
    )
    all_params = [param for group in groups for param in group["params"]]
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.likelihood.train()
    train_x = model.train_inputs[0]
    stop_cfg = train_cfg["early_stop"]
    patience = int(stop_cfg["patience"])
    min_delta = float(stop_cfg.get("min_delta", 1e-6))
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history: list[float] = []
    early_stopped = False
    for _step in range(1, int(train_cfg["max_steps"]) + 1):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite marginal log-likelihood at step {_step}: {float(loss)}")
        loss.backward()
        _clip_total_norm(all_params, float(train_cfg["grad_clip"]))
        optimizer.step()
        current = float(loss.item())
        history.append(current)
        scheduler.step(current)
        if current + min_delta < best_loss:
            best_loss = current
            no_improve = 0
            if bool(train_cfg.get("save_best_state", True)):
                best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
        if no_improve >= patience:  # paper D4: ten steps without improvement
            early_stopped = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    for param in model.covar_module.base_kernel.parameters():
        param.requires_grad_(True)
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().reshape(-1).cpu().numpy()
    return {
        "steps": len(history),
        "history": [round(value, 6) for value in history],
        "best_loss": float(best_loss),
        "early_stopped": early_stopped,
        "lengthscale_frozen": freeze,
        "round_index": int(round_index),
        "outputscale": float(model.covar_module.outputscale.detach()),
        "residual_noise": float(model.likelihood.second_noise_covar.noise.detach().reshape(-1)[0]),
        "lengthscale": {
            "min": float(lengthscale.min()), "median": float(np.median(lengthscale)), "max": float(lengthscale.max()),
        },
    }


def _acquire(
    model: Any,
    z_obs: Any,
    y_std: Any,
    var_std: Any,
    z_pool: Any,
    *,
    q: int,
    seed: int,
    config: dict[str, Any],
) -> tuple[list[int], list[float], int]:
    """Greedy sequential qLogNEI over the pool rows; returns their indices, acquisition values and the number of
    numerical warnings (Cholesky jitter: codes with identical embeddings make the joint covariance singular)."""

    import warnings

    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.models import SingleTaskGP
    from botorch.sampling import SobolQMCNormalSampler

    residual = model.likelihood.second_noise_covar.noise.detach().reshape(-1)[0]
    acq_cfg = config["acquisition"]
    chunk = max(1, int(acq_cfg["chunk"]))
    picks: list[int] = []
    values: list[float] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surrogate = SingleTaskGP(
            train_X=z_obs,
            train_Y=y_std.unsqueeze(-1),
            train_Yvar=(var_std + residual).unsqueeze(-1),
            covar_module=copy.deepcopy(model.covar_module),
            mean_module=copy.deepcopy(model.mean_module),
            outcome_transform=None,
        )
        surrogate.eval()
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([int(acq_cfg["mc_samples"])]), seed=int(seed) % (2**31)
        )
        acquisition = qLogNoisyExpectedImprovement(
            model=surrogate,
            X_baseline=_unique_rows(z_obs),
            sampler=sampler,
            prune_baseline=bool(acq_cfg["prune_baseline"]),
            cache_root=bool(acq_cfg["cache_root"]),
        )
        available = torch.ones(z_pool.shape[0], dtype=torch.bool)
        with torch.no_grad():
            for _ in range(min(int(q), int(z_pool.shape[0]))):
                acquisition.set_X_pending(z_pool[picks] if picks else None)
                scores = torch.cat([
                    acquisition(z_pool[start:start + chunk].unsqueeze(1))
                    for start in range(0, int(z_pool.shape[0]), chunk)
                ])
                scores = torch.where(available, scores, torch.full_like(scores, -math.inf))
                best = torch.max(scores)
                tied = available & (scores >= best - float(acq_cfg.get("tie_tolerance", 0.0)))
                index = int(torch.nonzero(tied)[0]) if bool(torch.isfinite(best)) and bool(tied.any()) else int(torch.argmax(scores))
                if not available[index]:
                    break
                picks.append(index)
                values.append(float(scores[index]))
                available[index] = False
    numerical = sum(1 for item in caught if "NumericalWarning" in type(item.message).__name__)
    return picks, values, numerical


def select_batch(
    *,
    observations: Sequence[Observation],
    pool: Sequence[PoolItem],
    q: int,
    seed: int,
    state_dir: Path | str,
    turn: int,
    config: dict[str, Any] = BO_CONFIG,
) -> tuple[list[str], dict[str, Any]]:
    """Choose at most q distinct pool codes for this turn's slow slots.

    Turns before ``init_turns`` and turns with fewer than ``min_observations`` usable observations nominate
    uniformly (seeded, flagged in the diagnostics). Otherwise one BO round: warm start from the previous turn's state
    file (fresh seeded initialisation when there is none), train, save this turn's state, embed, greedy qLogNEI. A
    numerical failure retries once from the fresh initialisation with the same seed and then raises.

    ``config`` is a plain JSON dict (the protocol snapshot's copy of BO_CONFIG, possibly with a leading "bo_id");
    a "bo_id" other than this module's is refused."""

    started = time.perf_counter()
    if str(config.get("bo_id") or BO_ID) != BO_ID:
        raise ValueError(f"config bo_id {config.get('bo_id')!r} is not this module's {BO_ID!r}")
    threads = bo_threads()
    turn = int(turn)
    state_dir = Path(state_dir)
    observed = {str(item.code_hash) for item in observations}
    unique_pool: dict[str, PoolItem] = {}
    for item in pool:
        code_hash = str(item.code_hash)
        if code_hash not in observed and code_hash not in unique_pool:
            unique_pool[code_hash] = item
    ordered_pool = [unique_pool[key] for key in sorted(unique_pool)]
    usable: dict[str, Observation] = {}
    for item in sorted(observations, key=lambda obs: str(obs.code_hash)):
        if item.usable() and str(item.code_hash) not in usable:
            usable[str(item.code_hash)] = item
    usable_obs = list(usable.values())
    diagnostics: dict[str, Any] = {
        "schema": "bo_selection_v1",
        "bo_id": BO_ID,
        "turn": turn,
        "q": int(q),
        "seed": int(seed),
        "n_obs": len(observations),
        "n_obs_usable": len(usable_obs),
        "n_pool": len(ordered_pool),
        "n_pool_dropped_observed": sum(1 for item in pool if str(item.code_hash) in observed),
        "config_digest": config_digest(config),
        "torch_threads": threads,
        "fallback": False,
        "fallback_reason": None,
        "phase": "bo",
    }
    if not ordered_pool or int(q) <= 0:
        diagnostics.update({"phase": "empty_pool" if not ordered_pool else "no_slots", "picks": []})
        diagnostics["timings_s"] = {"total": round(time.perf_counter() - started, 3)}
        return [], diagnostics
    hashes = [item.code_hash for item in ordered_pool]
    if turn < int(config["init_turns"]):
        picks = seeded_uniform_choice(hashes, q, seed)
        diagnostics.update({"phase": "initial_design", "picks": [{"code_hash": h} for h in picks]})
        diagnostics["timings_s"] = {"total": round(time.perf_counter() - started, 3)}
        return picks, diagnostics
    if len(usable_obs) < int(config["min_observations"]):
        picks = seeded_uniform_choice(hashes, q, seed)
        diagnostics.update({
            "fallback": True,
            "fallback_reason": f"{len(usable_obs)} usable observations < {int(config['min_observations'])}",
            "phase": "fallback_uniform",
            "picks": [{"code_hash": h} for h in picks],
        })
        diagnostics["timings_s"] = {"total": round(time.perf_counter() - started, 3)}
        return picks, diagnostics

    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    diagnostics["torch_threads"] = int(torch.get_num_threads())
    diagnostics["torch_version"] = str(torch.__version__)
    try:
        t_graphs = time.perf_counter()
        obs_graphs = [code_graph(item.code_hash, item.hx, item.hz) for item in usable_obs]
        pool_graphs = [code_graph(item.code_hash, item.hx, item.hz) for item in ordered_pool]
        samples = int(config["objective"]["posterior_samples"])
        targets = [observation_target(item, samples=samples) for item in usable_obs]
        y = np.array([value for value, _ in targets], dtype=np.float64)
        var = np.array([variance for _, variance in targets], dtype=np.float64)
        y_mean = float(np.mean(y))
        y_sd = float(np.std(y))
        if not y_sd > 1e-12:
            y_sd = 1.0
        y_std = torch.tensor((y - y_mean) / y_sd, dtype=torch.float64)
        var_std = torch.tensor(np.maximum(var, 0.0) / (y_sd ** 2), dtype=torch.float64)
        packed_obs = pack_graphs(obs_graphs)
        diagnostics["timings_s"] = {"graphs": round(time.perf_counter() - t_graphs, 3)}
        diagnostics["targets"] = {
            "y_mean": y_mean, "y_sd": y_sd,
            "y_min": float(np.min(y)), "y_max": float(np.max(y)),
            "var_median": float(np.median(var)),
        }
        round_index = turn - int(config["init_turns"])
        previous = state_path(state_dir, turn - 1)
        digest = config_digest(config)
        warm_state = None
        warm_note = "no previous state"
        if bool(config["training"].get("warm_start", True)) and previous.exists():
            try:
                loaded = torch.load(previous, weights_only=True)
            except Exception as exc:  # states are written atomically; an unreadable one is recorded, not used
                loaded = None
                warm_note = f"{previous.name} unreadable: {type(exc).__name__}"
            if loaded is not None:
                if loaded.get("config_digest") == digest and loaded.get("bo_id") == BO_ID:
                    warm_state = loaded
                    warm_note = str(previous.name)
                else:
                    warm_note = f"{previous.name} ignored: config or bo_id differs"
        errors: list[str] = []
        result = None
        for attempt, use_warm in ((1, warm_state is not None), (2, False)):
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed) % (2**63))
                    model = _build_model(config, y_std, var_std, packed_obs)
                    if use_warm:
                        _load_warm_state(model, warm_state, new_sd=y_sd, config=config)
                    t_train = time.perf_counter()
                    training = _train(model, y_std, config=config, round_index=round_index)
                    train_s = time.perf_counter() - t_train
                    t_embed = time.perf_counter()
                    chunk = int(config["acquisition"]["chunk"])
                    evaluator = copy.deepcopy(model.embed).to(_inference_dtype(config))
                    z_obs = embed_graphs(evaluator, obs_graphs, chunk=chunk)
                    z_pool = embed_graphs(evaluator, pool_graphs, chunk=chunk)
                    embed_s = time.perf_counter() - t_embed
                    t_acq = time.perf_counter()
                    picks_idx, picks_val, numerical = _acquire(
                        model, z_obs, y_std, var_std, z_pool, q=int(q), seed=int(seed), config=config
                    )
                    acquire_s = time.perf_counter() - t_acq
                if not picks_idx or any(not math.isfinite(value) for value in picks_val):
                    raise FloatingPointError(f"acquisition produced no finite pick: {picks_val}")
                result = (model, training, picks_idx, picks_val, numerical, train_s, embed_s, acquire_s, use_warm)
                break
            except Exception as exc:  # numerical failure: retry from the fixed initialisation, then raise
                errors.append(f"attempt {attempt} ({'warm' if use_warm else 'fresh'}): {type(exc).__name__}: {exc}")
        if result is None:
            raise RuntimeError(f"BO adapted failed at turn {turn} after a fresh retry: " + " | ".join(errors))
        model, training, picks_idx, picks_val, numerical, train_s, embed_s, acquire_s, used_warm = result
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "bo_id": BO_ID,
            "config_digest": digest,
            "turn": turn,
            "round_index": round_index,
            "embed": model.embed.state_dict(),
            "mean": model.mean_module.state_dict(),
            "covar": model.covar_module.state_dict(),
            "residual_raw_noise": model.likelihood.second_noise_covar.raw_noise.detach().clone(),
            "y_mean": y_mean,
            "y_sd": y_sd,
        }
        out_path = state_path(state_dir, turn)
        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
        pruned: list[str] = []
        kept = int(config.get("state_files_kept") or 0)
        if kept > 0:
            for path in sorted(state_dir.glob("bo_model_turn*.pt")):
                try:
                    file_turn = int(path.stem[len("bo_model_turn"):])
                except ValueError:
                    continue
                if file_turn <= turn - kept:
                    path.unlink()
                    pruned.append(path.name)
        picks = [ordered_pool[index].code_hash for index in picks_idx]
        diagnostics.update({
            "warm_start": bool(used_warm),
            "warm_start_source": warm_note if used_warm else ("fresh" if not errors else "fresh after failure"),
            "warm_start_note": warm_note,
            "retry_errors": errors,
            "acquisition_numerical_warnings": int(numerical),
            "training": training,
            "state_file": out_path.name,
            "state_files_pruned": pruned,
            "picks": [
                {"code_hash": code_hash, "log_acquisition": round(value, 6), "rank": rank}
                for rank, (code_hash, value) in enumerate(zip(picks, picks_val), start=1)
            ],
        })
        diagnostics["timings_s"].update({
            "train": round(train_s, 3), "embed": round(embed_s, 3), "acquire": round(acquire_s, 3),
            "total": round(time.perf_counter() - started, 3),
        })
        return picks, diagnostics
    finally:
        torch.set_num_threads(previous_threads)
