"""Proposal sampler of the no-LLM search baselines (plan B §3, 2026-11-06; paper-literal stages since 2026-11-10).

The sampler replaces the LLM generator: every turn it makes the same number of raw proposals as a Full turn
(4 generations x 960 = 3 840) from a distribution that is frozen before the search and never reads a measurement.
It follows the paper's description (appendix D4 for BO, D5 for Random) stage by stage, in the paper's order, and at
every stage draws UNIFORMLY over that stage's admissible set: the values the physical-qubit budget N and the family's
grammar allow given the earlier stages, as far as they can be decided before the code is constructed (user decision
2026-11-10: "first uniform over the sizes, then uniform over the groups", read literally).

* family: stratified per turn (D5: "equal numbers of attempts allocated to each family") -- the turn's proposals are
  split equally over the run's families (960 each of the 3 840 for the four families; all of them for a BB-only run)
  and shuffled with the turn's RNG.
* bb (D4: "the two cyclic-group dimensions l and m subject to 4 l m <= N ... the numbers of terms in the two BB
  polynomials A and B, with a combined total between four and eight ... group elements from C_l x C_m"; D5: "the
  cyclic group dimensions, the weights of the two group-algebra polynomials and their monomial supports"):
  (ell, m) uniform over the pairs with ell <= m, 4 ell m <= N and ell m >= 2. ell <= m is the BB validator's
  canonical form (it swaps ell > m together with the torus axes), so (m, ell) is the same code and every unordered
  pair is one label. Then (wA, wB) uniform over the pairs with 1 <= wA, wB <= ell m and 4 <= wA + wB <= 8 (the LLM's
  BB constructor keeps its own 4..9 bound), then A and B uniform without replacement from Z_ell x Z_m.
* hgp (D5: "the dimensions and check weights of the two classical seed matrices, followed by their binary
  entries"): the two seed shapes (r1 x n1, r2 x n2) uniform over the ordered pairs with (r1 + n1)(r2 + n2) <= N
  whose shapes each admit a grammatical seed (classical_seed_witness); per seed, the check weights (row degrees)
  uniform over the sequences in [2, min(4, n)]^r whose sum the columns can carry (2 n <= sum <= min(4, r) n, exact
  counting, no rejection); then the binary entries: each row's support uniform among the subsets of its weight. A
  seed whose column degrees leave [2, 4] or that repeats a row fails the grammar and the draw is a failure.
* lp_abelian / lp_nonabelian (D5: "we first sample the seed-matrix dimensions and determine the admissible group
  orders from the physical-qubit budget. We then select an Abelian or non-Abelian group of an admissible order,
  respectively, and sample the weights and group elements of the seed-matrix entries"): (shape_A, shape_B) uniform
  over the ordered pairs of {1x1, 1x2, 2x1, 2x2} that admit an order; the admissible orders of that pair are the
  2 <= |G| <= N / (sA sB) (s = rows + cols of the shape, R = sA sB |G|) for which the catalogue holds a group of the
  family -- every order has an abelian group; the orders without a non-abelian catalogue group are left out, which up
  to |G| = 74 are exactly the orders all of whose groups are abelian (75 is the catalogue's first gap: (C5 x C5) : C3,
  reached only from N = 300); |G| uniform over them; G uniform among the family's catalogue groups of that order (one
  per isomorphism class: abelian by invariant factors, ``nonabelian_catalog`` for the rest); the cell weights of A
  and B jointly uniform over the assignments with every cell in 1..min(8, |G|) whose lifted maximum check weight
  max(row(A) + col(B), row(B) + col(A)) is <= 8 (user decision 2026-11-10: the weight constraint is decided by the
  cell weights, as BB's 4..8 terms, so it belongs to the admissible set); then every cell's elements uniform without
  replacement. The joint support of A and B must generate G (the grammar's check; otherwise a failure).

R = data qubits + one ancilla per check row is the budget the constructors and the structural check count; every
admissible set above keeps R <= N. Every draw is one raw proposal: a draw whose random parameters the grammar cannot
use is a recorded failure, not re-drawn, and the constraints only the constructed code shows (CSS validity, k >= 1,
Tanner connectivity) are the shared fast loop's checks, exactly as for the LLM programs. The maximum check weight
<= 8 holds by construction in every family (BB at most 8 terms, HGP degrees <= 4, LP cell weights drawn inside it);
the fast loop still checks it for every code.
Each draw has its own RNG, derived from (SAMPLER_ID, seed_key, turn, index), so no draw depends on another draw or
on the order the fast loop evaluates them.

History: fixed_prior_v1 / fixed_prior_v2 (2026-11-06 / 11-10, BB total 4..8 from v2) drew the total size R first,
uniform over the distinct feasible R, then the skeleton uniformly among those of that R; BB W uniform then the split;
HGP E uniform, row and column degree sequences, stub matching; LP cell weights with q(w) proportional to 2^-(w-1).
literal_uniform_v1 (2026-11-10) had the present stages with every LP cell weight independently uniform in
1..min(8, |G|), leaving w_max <= 8 to the fast loop.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, product
from math import gcd, isqrt
from typing import Any, Sequence

import numpy as np

from ..construction.bb import BB_MAX_TOTAL_WEIGHT, BB_MIN_TOTAL_WEIGHT
from ..construction.grammars.classical_seed import DEFAULT_MAX_DEGREE, DEFAULT_MIN_DEGREE
from ..construction.grammars.ga_seed import DEFAULT_ALLOWED_SHAPES, DEFAULT_MAX_CELL_WEIGHT
from ..construction.groups.abelian import abelian_group
from ..construction.groups.base import Group
from ..construction.groups.catalog import canonical_twist, get_group

# literal_uniform_v1 (2026-11-10, paper appendix D5 read literally): every stage uniform over its admissible set, in
# the paper's order (sizes first, then groups). Replaces fixed_prior_v2 (R first).
# literal_uniform_v2 (user decision 2026-11-10): the LP cell weights are uniform over the joint assignments whose
# lifted maximum check weight is <= 8 (decidable from the weights, as BB's 4..8 terms). Older runs cannot resume.
SAMPLER_ID = "literal_uniform_v2"
FAMILIES: tuple[str, ...] = ("bb", "hgp", "lp_abelian", "lp_nonabelian")

# BB total-weight range of the baseline sampler (paper appendix D4: "a combined total between four and eight"). Only
# the sampler uses it: the BB constructor (construction/bb.py, used by the LLM search) keeps BB_MAX_TOTAL_WEIGHT.
BASELINE_BB_MIN_TOTAL_WEIGHT = BB_MIN_TOTAL_WEIGHT
BASELINE_BB_MAX_TOTAL_WEIGHT = 8
assert BB_MIN_TOTAL_WEIGHT <= BASELINE_BB_MIN_TOTAL_WEIGHT <= BASELINE_BB_MAX_TOTAL_WEIGHT <= BB_MAX_TOTAL_WEIGHT
# The paper's maximum check-weight constraint (config.SEARCH_MAX_CHECK_WEIGHT, the fast loop's cap). The LP cell
# weights are drawn inside it because it is decided by the weights alone; the fast loop still checks every code.
BASELINE_MAX_CHECK_WEIGHT = 8

# The sampler's own constants; recorded in catalog_digest so a change shows up in the protocol snapshot.
PRIOR_CONSTANTS: dict[str, Any] = {
    "bb_total_weight": [BASELINE_BB_MIN_TOTAL_WEIGHT, BASELINE_BB_MAX_TOTAL_WEIGHT],
    "hgp_degree": [DEFAULT_MIN_DEGREE, DEFAULT_MAX_DEGREE],
    "lp_shapes": [list(shape) for shape in DEFAULT_ALLOWED_SHAPES],
    "lp_max_cell_weight": DEFAULT_MAX_CELL_WEIGHT,
    "lp_max_check_weight": BASELINE_MAX_CHECK_WEIGHT,
    "family_stratification": "equal per turn, shuffled",
    "bb_stages": "(ell <= m) uniform over 4 ell m <= N; (wA, wB) uniform over 4 <= wA + wB <= 8; supports uniform",
    "hgp_stages": "seed-shape pair uniform over (r1 + n1)(r2 + n2) <= N; row-degree sequence uniform over the "
                  "column-feasible ones; each row's support uniform",
    "lp_stages": "shape pair uniform; |G| uniform over the admissible orders with a catalogue group of the family; "
                 "G uniform among them; cell weights uniform over the joint assignments in 1..min(8, |G|) with "
                 "lifted w_max <= 8; elements uniform",
}

# Representative preference inside one isomorphism class of the non-abelian catalogue.
_KIND_RANK = {"dihedral": 0, "alternating": 1, "symmetric": 1, "direct_product": 2, "metacyclic": 3}


@dataclass(frozen=True)
class Draw:
    """One raw proposal: the chosen resource and skeleton, and the spec or the reason there is none."""

    index: int
    turn: int
    gen: int
    slot: int
    family: str
    resource: int | None
    skeleton: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] | None = None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "baseline_sampler_draw",
            "sampler_id": SAMPLER_ID,
            "index": self.index,
            "turn": self.turn,
            "gen": self.gen,
            "slot": self.slot,
            "family": self.family,
            "resource": self.resource,
            "skeleton": self.skeleton,
            "spec": self.spec,
            "failure": self.failure,
        }


class _DrawFailure(Exception):
    """A random parameter choice the constructor grammar cannot use (the draw is recorded as failed)."""


# ---------------------------------------------------------------- public API

def canonical_families(families: Sequence[str]) -> tuple[str, ...]:
    """The run's families in the fixed FAMILIES order (draws must not depend on how a caller listed them)."""

    wanted = {str(family) for family in families}
    unknown = sorted(wanted - set(FAMILIES))
    if unknown or not wanted:
        raise ValueError(f"baseline sampler families must be a non-empty subset of {FAMILIES}, got {sorted(wanted)}")
    return tuple(family for family in FAMILIES if family in wanted)


def turn_family_plan(*, families: Sequence[str], per_turn: int, seed_key: str, turn: int) -> list[str]:
    """The family label of each of the turn's ``per_turn`` proposals: equal shares (a remainder goes to the
    first families in FAMILIES order), shuffled with the turn's RNG."""

    ordered = canonical_families(families)
    base, extra = divmod(int(per_turn), len(ordered))
    labels: list[str] = []
    for position, family in enumerate(ordered):
        labels.extend([family] * (base + (1 if position < extra else 0)))
    rng = np.random.default_rng(_seed_int(seed_key, turn, "plan"))
    order = rng.permutation(len(labels))
    return [labels[int(i)] for i in order]


def draw_generation(
    *,
    n_budget: int,
    families: Sequence[str],
    seed_key: str,
    turn: int,
    gen: int,
    count: int,
    generations_per_turn: int,
) -> list[Draw]:
    """The ``count`` raw proposals of generation ``gen`` (1-based) of ``turn``; exactly ``count`` Draws."""

    count, generations_per_turn, gen, turn = int(count), int(generations_per_turn), int(gen), int(turn)
    if count < 1 or generations_per_turn < 1 or not 1 <= gen <= generations_per_turn or turn < 0:
        raise ValueError(f"bad generation request: turn={turn} gen={gen}/{generations_per_turn} count={count}")
    per_turn = count * generations_per_turn
    ordered = canonical_families(families)
    for family in ordered:
        if not _has_skeleton(family, int(n_budget)):
            raise ValueError(f"family {family!r} has no feasible skeleton at n_budget={n_budget}")
    labels = turn_family_plan(families=ordered, per_turn=per_turn, seed_key=seed_key, turn=turn)
    draws: list[Draw] = []
    for slot in range(count):
        offset = (gen - 1) * count + slot
        index = turn * per_turn + offset
        family = labels[offset]
        rng = np.random.default_rng(_seed_int(seed_key, turn, f"index={index}"))
        draws.append(_draw_one(family=family, n_budget=int(n_budget), rng=rng, index=index, turn=turn, gen=gen, slot=slot))
    return draws


def program_source(draw: Draw) -> str:
    """The construct(rng) program the fast loop evaluates for one draw.

    A failed draw still becomes a program, so a generation always holds exactly ``count`` programs and every raw
    proposal is one attempt of the archive. The sandbox exposes no exception classes (sandbox._SAFE_BUILTINS), so
    the failure is raised by a missing-key lookup whose key is the reason; the archive records it as
    ``construct() raised: 'baseline sampler: <reason>'``."""

    header = f"def construct(rng):\n    # baseline sampler {SAMPLER_ID}, draw {draw.index} ({draw.family})\n"
    if draw.spec is not None:
        return header + "    return " + repr(draw.spec) + "\n"
    reason = f"baseline sampler: {draw.failure or 'no spec'}"
    return header + "    return {}[" + repr(reason) + "]\n"


def stage_catalog(family: str, n_budget: int) -> dict[str, Any]:
    """The admissible sets the draws of one family index into at this budget (a JSON-safe copy).

    bb: ``{"pairs": [{"ell", "m"}, ...]}``; hgp: ``{"pairs": [{"H1": [r, n], "H2": [r, n]}, ...]}``; LP:
    ``{"shape_pairs": [{"shape_A", "shape_B", "factor", "orders"}, ...], "groups_by_order": {"<|G|>": [...]}}``."""

    stages = _stages(str(family), int(n_budget))
    if "groups_by_order" in stages:
        return {
            "shape_pairs": copy.deepcopy(list(stages["shape_pairs"])),
            "groups_by_order": {str(order): copy.deepcopy(list(groups)) for order, groups in stages["groups_by_order"].items()},
        }
    return {"pairs": copy.deepcopy(list(stages["pairs"]))}


def resource_catalog(family: str, n_budget: int) -> dict[int, list[dict[str, Any]]]:
    """R -> the family's admissible skeletons of total size R at this budget (a view of ``stage_catalog``; the draws
    do not choose R first)."""

    out: dict[int, list[dict[str, Any]]] = {}
    for resource, skeleton in _skeletons(str(family), int(n_budget)):
        out.setdefault(resource, []).append(skeleton)
    return dict(sorted(out.items()))


def catalog_digest(families: Sequence[str], n_budget: int) -> str:
    """sha256 over the sampler id, the sampler constants and the frozen stage catalogues of the run's families."""

    ordered = canonical_families(families)
    payload = {
        "sampler_id": SAMPLER_ID,
        "prior": PRIOR_CONSTANTS,
        "n_budget": int(n_budget),
        "families": {family: stage_catalog(family, int(n_budget)) for family in ordered},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- RNG helpers

def _seed_int(seed_key: str, turn: int, tag: str) -> int:
    text = f"{SAMPLER_ID}|{seed_key}|turn={int(turn)}|{tag}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def _randbelow(rng: np.random.Generator, bound: int) -> int:
    """Exact uniform integer in [0, bound) for arbitrarily large Python ints."""

    bound = int(bound)
    if bound <= 0:
        raise ValueError("bound must be positive")
    if bound <= 2**62:
        return int(rng.integers(0, bound))
    bits = bound.bit_length()
    words = (bits + 31) // 32
    while True:
        value = 0
        for word in rng.integers(0, 2**32, size=words, dtype=np.uint64):
            value = (value << 32) | int(word)
        value >>= words * 32 - bits
        if value < bound:
            return value


def _choose_weighted(rng: np.random.Generator, weights: Sequence[int]) -> int:
    """Index i with probability weights[i] / sum(weights) (exact, integer weights)."""

    total = sum(int(w) for w in weights)
    pick = _randbelow(rng, total)
    for position, weight in enumerate(weights):
        pick -= int(weight)
        if pick < 0:
            return position
    raise AssertionError("unreachable")


def _uniform(rng: np.random.Generator, items: Sequence[Any]) -> Any:
    return items[int(rng.integers(0, len(items)))]


# ---------------------------------------------------------------- one draw

def _draw_skeleton(family: str, n_budget: int, rng: np.random.Generator) -> tuple[int, dict[str, Any]]:
    """The size stages of one draw, each uniform over its admissible set: BB (ell, m); HGP the two seed shapes; LP
    the shape pair, then |G| among the pair's admissible orders, then G among the catalogue groups of that order.
    Returns (R, skeleton)."""

    stages = _stages(family, n_budget)
    if family == "bb":
        pair = _uniform(rng, stages["pairs"])
        ell, m = int(pair["ell"]), int(pair["m"])
        return 4 * ell * m, {"ell": ell, "m": m}
    if family == "hgp":
        pair = _uniform(rng, stages["pairs"])
        (r1, n1), (r2, n2) = pair["H1"], pair["H2"]
        return (int(r1) + int(n1)) * (int(r2) + int(n2)), {"H1": [int(r1), int(n1)], "H2": [int(r2), int(n2)]}
    pair = _uniform(rng, stages["shape_pairs"])
    order = int(_uniform(rng, pair["orders"]))
    group_field = copy.deepcopy(_uniform(rng, stages["groups_by_order"][order]))
    skeleton = {"shape_A": list(pair["shape_A"]), "shape_B": list(pair["shape_B"]), "group": group_field, "group_order": order}
    return int(pair["factor"]) * order, skeleton


def _draw_one(*, family: str, n_budget: int, rng: np.random.Generator, index: int, turn: int, gen: int, slot: int) -> Draw:
    resource, skeleton = _draw_skeleton(family, n_budget, rng)
    try:
        if family == "bb":
            spec, chosen = _draw_bb(skeleton, rng)
        elif family == "hgp":
            spec, chosen = _draw_hgp(skeleton, rng)
        else:
            spec, chosen = _draw_lp(family, skeleton, rng)
        failure = None
    except _DrawFailure as exc:
        spec, chosen, failure = None, dict(getattr(exc, "chosen", {}) or {}), str(exc)
    skeleton.update(chosen)
    return Draw(
        index=index, turn=turn, gen=gen, slot=slot, family=family, resource=resource,
        skeleton=skeleton, spec=spec, failure=failure,
    )


def _fail(reason: str, chosen: dict[str, Any]) -> _DrawFailure:
    exc = _DrawFailure(reason)
    exc.chosen = chosen  # type: ignore[attr-defined]
    return exc


# BB ------------------------------------------------------------------------

def _bb_weight_pairs(cells: int) -> list[tuple[int, int]]:
    """The admissible (wA, wB): both polynomials non-empty, at most ell m terms each, 4 <= wA + wB <= 8."""

    top = min(int(cells), BASELINE_BB_MAX_TOTAL_WEIGHT)
    return [
        (wa, wb)
        for wa in range(1, top + 1)
        for wb in range(1, top + 1)
        if BASELINE_BB_MIN_TOTAL_WEIGHT <= wa + wb <= BASELINE_BB_MAX_TOTAL_WEIGHT
    ]


def _draw_bb(skeleton: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    ell, m = int(skeleton["ell"]), int(skeleton["m"])
    cells = ell * m
    wa, wb = _uniform(rng, _bb_weight_pairs(cells))
    a_terms = sorted(rng.choice(cells, size=wa, replace=False).tolist())
    b_terms = sorted(rng.choice(cells, size=wb, replace=False).tolist())
    spec = {
        "family": "bb",
        "ell": ell,
        "m": m,
        "A": [[int(t) // m, int(t) % m] for t in a_terms],
        "B": [[int(t) // m, int(t) % m] for t in b_terms],
    }
    return spec, {"W": wa + wb, "wA": wa, "wB": wb}


# HGP -----------------------------------------------------------------------

def _row_degree_cap(cols: int) -> int:
    return min(DEFAULT_MAX_DEGREE, int(cols))


def _edge_range(rows: int, cols: int) -> tuple[int, int]:
    """Feasible edge counts of a rows x cols seed: row degrees in [2, min(4, cols)], column degrees in
    [2, min(4, rows)]."""

    lo = max(DEFAULT_MIN_DEGREE * rows, DEFAULT_MIN_DEGREE * cols)
    hi = min(rows * _row_degree_cap(cols), cols * _row_degree_cap(rows))
    return lo, hi


@lru_cache(maxsize=None)
def _sequence_count(length: int, total: int, lo: int, hi: int) -> int:
    """Number of integer sequences of ``length`` entries in [lo, hi] with the given sum."""

    if length == 0:
        return 1 if total == 0 else 0
    if total < lo * length or total > hi * length:
        return 0
    return sum(_sequence_count(length - 1, total - d, lo, hi) for d in range(lo, hi + 1))


def _uniform_sequence(rng: np.random.Generator, length: int, total: int, lo: int, hi: int) -> list[int]:
    """A sequence uniform among all ``length``-sequences in [lo, hi] with sum ``total`` (sequential exact
    sampling over the counts; no rejection)."""

    out: list[int] = []
    remaining = int(total)
    for position in range(length, 0, -1):
        options = list(range(lo, hi + 1))
        weights = [_sequence_count(position - 1, remaining - d, lo, hi) for d in options]
        if sum(weights) == 0:
            raise AssertionError("degree sequence counting is inconsistent")
        d = options[_choose_weighted(rng, weights)]
        out.append(d)
        remaining -= d
    return out


def _check_weights(rng: np.random.Generator, rows: int, cols: int) -> list[int]:
    """Row degrees uniform over the admissible sequences: every entry in [2, min(4, cols)] and a sum the columns can
    carry with degrees in [2, min(4, rows)] (the sum is drawn with weight = its number of sequences, then the
    sequence uniformly among those of that sum)."""

    lo, hi = _edge_range(rows, cols)
    cap = _row_degree_cap(cols)
    totals = list(range(lo, hi + 1))
    weights = [_sequence_count(rows, total, DEFAULT_MIN_DEGREE, cap) for total in totals]
    edges = totals[_choose_weighted(rng, weights)]
    return _uniform_sequence(rng, rows, edges, DEFAULT_MIN_DEGREE, cap)


def _seed_entries(rng: np.random.Generator, cols: int, row_degrees: Sequence[int]) -> list[list[int]]:
    """Binary entries given the row degrees: each row's support uniform among the ``cols``-subsets of its size."""

    matrix: list[list[int]] = []
    for degree in row_degrees:
        row = [0] * int(cols)
        for col in rng.choice(int(cols), size=int(degree), replace=False).tolist():
            row[int(col)] = 1
        matrix.append(row)
    return matrix


def _draw_hgp(skeleton: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    shapes = {label: (int(skeleton[label][0]), int(skeleton[label][1])) for label in ("H1", "H2")}
    chosen: dict[str, Any] = {}
    degrees: dict[str, list[int]] = {}
    for label, (rows, cols) in shapes.items():  # the check weights of both seeds, then their entries
        degrees[label] = _check_weights(rng, rows, cols)
        chosen[f"{label}_row_degrees"] = degrees[label]
        chosen[f"{label}_edges"] = sum(degrees[label])
    seeds: dict[str, list[list[int]]] = {}
    for label, (_, cols) in shapes.items():
        seeds[label] = _seed_entries(rng, cols, degrees[label])
        chosen[f"{label}_col_degrees"] = [int(v) for v in np.asarray(seeds[label], dtype=np.int64).sum(axis=0)]
    for label in shapes:
        col_degrees = chosen[f"{label}_col_degrees"]
        if min(col_degrees) < DEFAULT_MIN_DEGREE or max(col_degrees) > DEFAULT_MAX_DEGREE:
            raise _fail(f"{label}: column degrees outside [{DEFAULT_MIN_DEGREE}, {DEFAULT_MAX_DEGREE}]", chosen)
        if len({tuple(row) for row in seeds[label]}) != len(seeds[label]):
            raise _fail(f"{label}: duplicate rows", chosen)
    return {"family": "hgp", "seeds": {"H1": seeds["H1"], "H2": seeds["H2"]}}, chosen


# LP ------------------------------------------------------------------------

def _shape_size(shape: Sequence[int]) -> int:
    return int(shape[0]) + int(shape[1])


def _lp_max_check_weight(weights_a: Sequence[Sequence[int]], weights_b: Sequence[Sequence[int]]) -> int:
    """The lifted product's maximum check weight from the cell weights alone: an X check (i, t) has weight
    row_i(A) + col_t(B), a Z check (j, s) row_s(B) + col_j(A) (the elements of a cell are distinct, so the lift keeps
    every cell weight; construction/lp.lp_code)."""

    rows_a, cols_a = _line_sums(weights_a)
    rows_b, cols_b = _line_sums(weights_b)
    return max(max(rows_a) + max(cols_b), max(rows_b) + max(cols_a))


def _line_sums(weights: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    rows = [sum(int(w) for w in row) for row in weights]
    cols = [sum(int(row[j]) for row in weights) for j in range(len(weights[0]))]
    return rows, cols


@lru_cache(maxsize=None)
def _cell_weight_matrices(shape: tuple[int, int], cap: int) -> dict[tuple[int, int], tuple[tuple[tuple[int, ...], ...], ...]]:
    """Every weight matrix of one base shape with cells in 1..cap, grouped by (max row sum, max column sum)."""

    rows, cols = int(shape[0]), int(shape[1])
    groups: dict[tuple[int, int], list[tuple[tuple[int, ...], ...]]] = {}
    for flat in product(range(1, int(cap) + 1), repeat=rows * cols):
        matrix = tuple(tuple(flat[r * cols:(r + 1) * cols]) for r in range(rows))
        row_sums, col_sums = _line_sums(matrix)
        groups.setdefault((max(row_sums), max(col_sums)), []).append(matrix)
    return {key: tuple(groups[key]) for key in sorted(groups)}


@lru_cache(maxsize=None)
def _cell_weight_table(shape_a: tuple[int, int], shape_b: tuple[int, int], cap: int) -> tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], tuple[int, ...]]:
    """The admissible joint cell-weight assignments of (A, B), as (key_A, key_B) blocks with their sizes: cells in
    1..cap and max(row(A) + col(B), row(B) + col(A)) <= BASELINE_MAX_CHECK_WEIGHT."""

    groups_a = _cell_weight_matrices(shape_a, cap)
    groups_b = _cell_weight_matrices(shape_b, cap)
    blocks, sizes = [], []
    for key_a, members_a in groups_a.items():
        for key_b, members_b in groups_b.items():
            if key_a[0] + key_b[1] <= BASELINE_MAX_CHECK_WEIGHT and key_b[0] + key_a[1] <= BASELINE_MAX_CHECK_WEIGHT:
                blocks.append((key_a, key_b))
                sizes.append(len(members_a) * len(members_b))
    return tuple(blocks), tuple(sizes)


def lp_cell_weight_count(shape_a: Sequence[int], shape_b: Sequence[int], order: int) -> int:
    """Number of admissible joint cell-weight assignments (the set the draw is uniform over)."""

    cap = min(DEFAULT_MAX_CELL_WEIGHT, int(order))
    return sum(_cell_weight_table(tuple(shape_a), tuple(shape_b), cap)[1])


def _draw_cell_weights(shape_a: Sequence[int], shape_b: Sequence[int], order: int, rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]]]:
    """Cell weights of A and B uniform over the admissible joint assignments: every cell in 1..min(8, |G|) and the
    lifted product's maximum check weight <= 8 (user decision 2026-11-10, as BB's 4..8 terms). A block (key_A, key_B)
    is drawn with weight = its number of assignments, then A and B uniformly inside it: exact, no rejection."""

    cap = min(DEFAULT_MAX_CELL_WEIGHT, int(order))
    shape_a, shape_b = (int(shape_a[0]), int(shape_a[1])), (int(shape_b[0]), int(shape_b[1]))
    blocks, sizes = _cell_weight_table(shape_a, shape_b, cap)
    key_a, key_b = blocks[_choose_weighted(rng, sizes)]
    weights_a = _uniform(rng, _cell_weight_matrices(shape_a, cap)[key_a])
    weights_b = _uniform(rng, _cell_weight_matrices(shape_b, cap)[key_b])
    return [list(row) for row in weights_a], [list(row) for row in weights_b]


def _draw_cells(group: Group, weights: Sequence[Sequence[int]], rng: np.random.Generator) -> list[list[list[list[int]]]]:
    """The group elements of every cell: uniform without replacement, given the cell weights."""

    cells: list[list[list[list[int]]]] = []
    for weight_row in weights:
        row: list[list[list[int]]] = []
        for w in weight_row:
            members = sorted(int(i) for i in rng.choice(group.order, size=int(w), replace=False))
            row.append([list(group.elements[i]) for i in members])
        cells.append(row)
    return cells


def _draw_lp(family: str, skeleton: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    group_field = skeleton["group"]
    group = abelian_group(tuple(int(q) for q in group_field["factors"])) if family == "lp_abelian" else get_group(
        str(group_field), max_order=int(skeleton["group_order"])
    )
    a_sizes, b_sizes = _draw_cell_weights(skeleton["shape_A"], skeleton["shape_B"], group.order, rng)
    a_cells = _draw_cells(group, a_sizes, rng)
    b_cells = _draw_cells(group, b_sizes, rng)
    chosen = {"cell_weights_A": a_sizes, "cell_weights_B": b_sizes}
    support = {group.element_index(entry) for cells in (a_cells, b_cells) for row in cells for cell in row for entry in cell}
    if not group.generates(support):
        raise _fail("supports of A and B generate a proper subgroup", chosen)
    spec = {"family": family, "group": copy.deepcopy(group_field), "A": a_cells, "B": b_cells}
    return spec, chosen


# ---------------------------------------------------------------- catalogues

@lru_cache(maxsize=None)
def _stages(family: str, n_budget: int) -> dict[str, Any]:
    """The frozen admissible sets of one family (read-only; ``stage_catalog`` returns copies)."""

    if family == "bb":
        return {"pairs": _bb_pairs(n_budget)}
    if family == "hgp":
        return {"pairs": _hgp_pairs(n_budget)}
    if family in ("lp_abelian", "lp_nonabelian"):
        return _lp_stages(family, n_budget)
    raise ValueError(f"unknown family {family!r}")


def _skeletons(family: str, n_budget: int) -> list[tuple[int, dict[str, Any]]]:
    """Every admissible skeleton of the family with its total size R (JSON-safe copies)."""

    stages = _stages(family, n_budget)
    if family == "bb":
        return [(4 * p["ell"] * p["m"], dict(p)) for p in stages["pairs"]]
    if family == "hgp":
        return [((p["H1"][0] + p["H1"][1]) * (p["H2"][0] + p["H2"][1]), copy.deepcopy(p)) for p in stages["pairs"]]
    out: list[tuple[int, dict[str, Any]]] = []
    for pair in stages["shape_pairs"]:
        for order in pair["orders"]:
            for group_field in stages["groups_by_order"][order]:
                skeleton = {
                    "shape_A": list(pair["shape_A"]), "shape_B": list(pair["shape_B"]),
                    "group": copy.deepcopy(group_field), "group_order": int(order),
                }
                out.append((int(pair["factor"]) * int(order), skeleton))
    return out


def _has_skeleton(family: str, n_budget: int) -> bool:
    stages = _stages(family, n_budget)
    return bool(stages.get("pairs") or stages.get("shape_pairs"))


def _bb_pairs(n_budget: int) -> tuple[dict[str, int], ...]:
    """(ell, m) with ell <= m, 4 ell m <= N and ell m >= 2 (so that 4 <= wA + wB fits), by (ell m, ell)."""

    pairs: list[dict[str, int]] = []
    for cells in range(2, n_budget // 4 + 1):
        if not _bb_weight_pairs(cells):
            continue
        pairs.extend({"ell": ell, "m": cells // ell} for ell in range(1, isqrt(cells) + 1) if cells % ell == 0)
    return tuple(pairs)


def classical_seed_witness(rows: int, cols: int) -> list[list[int]] | None:
    """One rows x cols matrix the classical-seed grammar accepts (row and column degrees in [2, 4], no duplicate
    rows), or None when none exists. min(rows, cols) >= 5: the incidence matrix of a cycle plus i -> i+2 chords
    (exists iff max <= 2 min); smaller seeds: exhaustive search."""

    rows, cols = int(rows), int(cols)
    if rows < 2 or cols < 2:
        return None
    lo, hi = _edge_range(rows, cols)
    if lo > hi:
        return None
    small = min(rows, cols)
    if small >= 5:
        vertices, edges = (cols, rows) if rows >= cols else (rows, cols)
        pairs = [(i, (i + 1) % vertices) for i in range(vertices)] + [(i, (i + 2) % vertices) for i in range(edges - vertices)]
        incidence = [[1 if v in pair else 0 for v in range(vertices)] for pair in pairs]  # edges x vertices
        if rows >= cols:
            return incidence
        return [[incidence[e][v] for e in range(edges)] for v in range(vertices)]
    return _seed_search(rows, cols)


def _seed_search(rows: int, cols: int) -> list[list[int]] | None:
    cap_row, cap_col = _row_degree_cap(cols), _row_degree_cap(rows)
    patterns = [combo for w in range(DEFAULT_MIN_DEGREE, cap_row + 1) for combo in combinations(range(cols), w)]
    degrees = [0] * cols
    chosen: list[tuple[int, ...]] = []

    def feasible(remaining: int) -> bool:
        deficit = sum(max(0, DEFAULT_MIN_DEGREE - d) for d in degrees)
        return deficit <= remaining * cap_row

    def search(start: int) -> bool:
        if len(chosen) == rows:
            return all(d >= DEFAULT_MIN_DEGREE for d in degrees)
        remaining = rows - len(chosen)
        for position in range(start, len(patterns)):
            if len(patterns) - position < remaining:
                break
            pattern = patterns[position]
            if any(degrees[c] >= cap_col for c in pattern):
                continue
            for c in pattern:
                degrees[c] += 1
            chosen.append(pattern)
            if feasible(remaining - 1) and search(position + 1):
                return True
            chosen.pop()
            for c in pattern:
                degrees[c] -= 1
        return False

    if not search(0):
        return None
    return [[1 if c in pattern else 0 for c in range(cols)] for pattern in chosen]


@lru_cache(maxsize=None)
def _classical_seed_feasible(rows: int, cols: int) -> bool:
    return classical_seed_witness(rows, cols) is not None


def _hgp_pairs(n_budget: int) -> tuple[dict[str, list[int]], ...]:
    """Ordered pairs of seed shapes (r1 x n1, r2 x n2) with (r1 + n1)(r2 + n2) <= N, each shape admitting a
    grammatical seed (2 min(r, n) >= max(r, n) and a witness), by (R, H1, H2)."""

    smallest = None
    seeds: dict[int, list[tuple[int, int]]] = {}
    for size in range(4, n_budget + 1):
        for rows in range(2, size - 1):
            cols = size - rows
            if 2 * min(rows, cols) < max(rows, cols):
                continue
            if _classical_seed_feasible(rows, cols):
                seeds.setdefault(size, []).append((rows, cols))
        if seeds and smallest is None:
            smallest = min(seeds)
        if smallest is not None and size * smallest > n_budget:
            break
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for s1, shapes1 in seeds.items():
        for s2, shapes2 in seeds.items():
            if s1 * s2 > n_budget:
                continue
            pairs.extend((s1 * s2, first, second) for first in shapes1 for second in shapes2)
    pairs.sort()
    return tuple({"H1": list(first), "H2": list(second)} for _, first, second in pairs)


def _lp_stages(family: str, n_budget: int) -> dict[str, Any]:
    """LP stages: the ordered shape pairs with at least one admissible order, each with its admissible orders
    (2 <= |G| <= N / (sA sB) and a catalogue group of the family of that order), and the catalogue groups by order
    (abelian: invariant factors; non-abelian: one resolvable name per isomorphism class)."""

    shapes = [tuple(shape) for shape in DEFAULT_ALLOWED_SHAPES]
    max_order = n_budget // min(_shape_size(shape) for shape in shapes) ** 2
    groups_by_order: dict[int, list[Any]] = {}
    if family == "lp_abelian":
        for factors in abelian_catalog(max_order):
            groups_by_order.setdefault(_product(factors), []).append({"factors": list(factors)})
    else:
        for name in nonabelian_catalog(max_order):
            groups_by_order.setdefault(int(get_group(name, max_order=max_order).order), []).append(name)
    orders = sorted(groups_by_order)
    shape_pairs: list[dict[str, Any]] = []
    for shape_a in shapes:
        for shape_b in shapes:
            factor = _shape_size(shape_a) * _shape_size(shape_b)
            admissible = [order for order in orders if factor * order <= n_budget]
            if admissible:
                shape_pairs.append(
                    {"shape_A": list(shape_a), "shape_B": list(shape_b), "factor": factor, "orders": admissible}
                )
    return {
        "shape_pairs": tuple(shape_pairs),
        "groups_by_order": {order: tuple(groups_by_order[order]) for order in orders},
    }


def _product(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _divisors(value: int) -> list[int]:
    small = [d for d in range(1, isqrt(value) + 1) if value % d == 0]
    return sorted(set(small + [value // d for d in small]))


def _invariant_chains(order: int, base: int) -> list[tuple[int, ...]]:
    """Ascending invariant-factor lists d1 | d2 | ... | dt with base | d1, product ``order``, every d >= 2."""

    out: list[tuple[int, ...]] = []
    if order >= 2 and order % base == 0:
        out.append((order,))
    for d in _divisors(order):
        if 2 <= d < order and d % base == 0 and (order // d) % d == 0:
            out.extend((d,) + rest for rest in _invariant_chains(order // d, d))
    return out


@lru_cache(maxsize=None)
def abelian_catalog(max_order: int) -> tuple[tuple[int, ...], ...]:
    """Every abelian group of order 2..max_order, once per isomorphism class, as its invariant factors."""

    groups: list[tuple[int, ...]] = []
    for order in range(2, int(max_order) + 1):
        groups.extend(sorted(_invariant_chains(order, 1)))
    return tuple(groups)


def _nonabelian_names(max_order: int) -> list[str]:
    """Every name of the catalogue grammar with |G| <= max_order (aliases included; get_group decides)."""

    names: list[str] = []
    names += [f"D{n}" for n in range(3, max_order // 2 + 1)]
    names += [name for name, order in (("A4", 12), ("S4", 24)) if order <= max_order]
    for m in range(3, max_order // 2 + 1):
        for n in range(2, max_order // m + 1):
            for k in range(2, m):
                if gcd(k, m) != 1 or pow(k, n, m) != 1 or (n == 2 and k == m - 1):
                    continue
                if canonical_twist(m, n, k) == k:
                    names.append(f"C{m}:C{n}({k})")
    inner = [(f"D{n}", 2 * n) for n in range(3, max_order // 4 + 1)] + [("A4", 12), ("S4", 24)]
    for m in range(2, max_order // 6 + 1):
        for inner_name, inner_order in inner:
            if m * inner_order > max_order:
                continue
            if m == 2 and inner_name.startswith("D") and int(inner_name[1:]) % 2 == 1:
                continue  # C2 x D_odd = D_2odd (the catalogue refuses this spelling)
            names.append(f"C{m}x{inner_name}")
    return names


@lru_cache(maxsize=None)
def nonabelian_catalog(max_order: int) -> tuple[str, ...]:
    """The non-abelian groups get_group resolves with |G| <= max_order, one name per isomorphism class.

    Every grammar name is resolved through get_group (which already refuses the catalogue's known aliases and
    normalises metacyclic twists); the remaining duplicates are removed by an exact isomorphism test inside
    buckets of equal invariants (order, the multiset of (element order, centraliser size), derived-subgroup and
    centre sizes). The kept name is the first by (order, kind: dihedral < A4/S4 < direct product < metacyclic,
    name length, name)."""

    resolved: list[tuple[tuple[Any, ...], str, Group]] = []
    seen_names: set[str] = set()
    for name in _nonabelian_names(int(max_order)):
        try:
            group = get_group(name, max_order=int(max_order))
        except ValueError:
            continue
        if group.is_abelian or group.name in seen_names:
            continue
        seen_names.add(group.name)
        key = (group.order, _KIND_RANK.get(group.kind, 9), len(group.name), group.name)
        resolved.append((key, group.name, group))
    resolved.sort(key=lambda item: item[0])
    kept: dict[tuple[Any, ...], list[Group]] = {}
    names: list[str] = []
    for _, name, group in resolved:
        bucket = kept.setdefault(group_isomorphism_invariants(group), [])
        if any(groups_isomorphic(group, other) for other in bucket):
            continue
        bucket.append(group)
        names.append(name)
    return tuple(names)


# ---------------------------------------------------------------- group isomorphism

def _identity_index(group: Group) -> int:
    return int(group.element_index(tuple(0 for _ in group.generator_orders)))


# Group is a frozen dataclass with dict / array fields (unhashable): per-group caches key on (kind, name), which
# the catalogue and abelian_group make unique per presentation.
_ORDERS_CACHE: dict[tuple[str, str], tuple[int, ...]] = {}
_INVARIANTS_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}


def _element_orders(group: Group) -> tuple[int, ...]:
    key = (group.kind, group.name)
    if key not in _ORDERS_CACHE:
        _ORDERS_CACHE[key] = _compute_element_orders(group)
    return _ORDERS_CACHE[key]


def _compute_element_orders(group: Group) -> tuple[int, ...]:
    size = group.order
    identity = _identity_index(group)
    orders = np.zeros(size, dtype=np.int64)
    current = np.arange(size)
    every = np.arange(size)
    for power in range(1, size + 1):
        hit = (current == identity) & (orders == 0)
        orders[hit] = power
        if bool((orders > 0).all()):
            break
        current = group.cayley[current, every]
    return tuple(int(v) for v in orders)


def group_isomorphism_invariants(group: Group) -> tuple[Any, ...]:
    """Invariants equal for isomorphic groups (a bucket key; equality proves nothing)."""

    key = (group.kind, group.name)
    if key not in _INVARIANTS_CACHE:
        _INVARIANTS_CACHE[key] = _compute_invariants(group)
    return _INVARIANTS_CACHE[key]


def _compute_invariants(group: Group) -> tuple[Any, ...]:
    cayley = np.asarray(group.cayley)
    commuting = cayley == cayley.T
    centraliser = commuting.sum(axis=1)
    orders = _element_orders(group)
    profile = tuple(sorted(Counter(zip(orders, (int(c) for c in centraliser))).items()))
    inverse = np.asarray(group.inverse)
    left = cayley[inverse[:, None], inverse[None, :]]  # x^-1 y^-1
    right = cayley  # x y
    commutators = {int(v) for v in np.unique(cayley[left, right])}
    derived = len(group.subgroup_closure(commutators))
    centre = int(commuting.all(axis=1).sum())
    return (int(group.order), profile, derived, centre)


def _generating_set(group: Group) -> list[int]:
    orders = _element_orders(group)
    by_order = sorted(range(group.order), key=lambda g: (-orders[g], g))
    generators: list[int] = []
    closure = {_identity_index(group)}
    while len(closure) < group.order:
        best, best_size = None, len(closure)
        for g in by_order:
            if g in closure:
                continue
            size = len(group.subgroup_closure(generators + [g]))
            if size > best_size:
                best, best_size = g, size
                if size == group.order:
                    break
        if best is None:
            raise AssertionError("generating set search stalled")
        generators.append(best)
        closure = group.subgroup_closure(generators)
    return generators


def groups_isomorphic(first: Group, second: Group) -> bool:
    """Exact isomorphism test: map a generating set of ``first`` to every tuple of same-order elements of
    ``second`` (pruned by the orders of pairwise products) and extend along the Cayley graph; an isomorphism exists
    iff some extension is a consistent bijection."""

    if first.order != second.order or group_isomorphism_invariants(first) != group_isomorphism_invariants(second):
        return False
    size = first.order
    gens = _generating_set(first)
    orders_a, orders_b = _element_orders(first), _element_orders(second)
    cayley_a, cayley_b = np.asarray(first.cayley), np.asarray(second.cayley)
    identity_a, identity_b = _identity_index(first), _identity_index(second)
    candidates = [[h for h in range(size) if orders_b[h] == orders_a[g]] for g in gens]

    def extends(images: list[int]) -> bool:
        mapping = {identity_a: identity_b}
        frontier = [identity_a]
        while frontier:
            x = frontier.pop()
            hx = mapping[x]
            for g, h in zip(gens, images):
                y = int(cayley_a[x, g])
                hy = int(cayley_b[hx, h])
                held = mapping.get(y)
                if held is None:
                    mapping[y] = hy
                    frontier.append(y)
                elif held != hy:
                    return False
        return len(mapping) == size and len(set(mapping.values())) == size

    def assign(position: int, images: list[int]) -> bool:
        if position == len(gens):
            return extends(images)
        g = gens[position]
        for h in candidates[position]:
            ok = True
            for earlier, h_earlier in zip(gens[:position], images):
                if orders_a[int(cayley_a[earlier, g])] != orders_b[int(cayley_b[h_earlier, h])]:
                    ok = False
                    break
            if ok and assign(position + 1, images + [h]):
                return True
        return False

    return assign(0, [])
