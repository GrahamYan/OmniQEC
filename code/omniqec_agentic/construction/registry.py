"""Family plugin registry — the single place a construction family is defined.

Four canonical search grammars (design doc docs/01_code_family_design.md):
bb, hgp, lp_abelian, lp_nonabelian. A plugin bundles spec validation, matrix
construction, natural sector sizes (consumed by the family-agnostic
deformation module), optional compile metadata, and the prompt-facing
contract. The sandbox, the builder, and the prompt assembler all dispatch
through this registry; everything downstream only ever sees HX/HZ plus
metadata and is family-blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

from omniqec_agentic.errors import ProgramError

from .bb import bb_code, bb_layer_blocks, bb_sector_sizes, validate_bb_spec
from .grammars.ga_seed import DEFAULT_ALLOWED_SHAPES, DEFAULT_MAX_CELL_WEIGHT
from .groups.catalog import catalog_summary, get_group_order_budget
from .hgp import hgp_code, hgp_sector_sizes, validate_hgp_spec
from .lp import (
    abelian_group_order,
    lp_code,
    lp_group_metadata,
    lp_layer_blocks,
    lp_sector_sizes,
    lp_total_qubits,
    nonabelian_group_order,
    resolve_abelian_group,
    resolve_nonabelian_group,
    validate_lp_spec,
)


@dataclass(frozen=True)
class FamilyPlugin:
    name: str
    validate: Callable[[dict[str, Any]], dict[str, Any]]
    build: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]
    # Natural two-sector data-qubit split, consumed by deformation/sector_h.
    sector_sizes: Callable[[dict[str, Any]], list[int]]
    genotype_line: str
    syntax_example_line: str
    format_contract_lines: tuple[str, ...]
    # Budgets are TOTAL qubits: data plus one measurement ancilla per check
    # row (BB72 costs 144 total). Size formulas and contracts speak total.
    size_formula_line: str
    user_contract_lines: tuple[str, ...]
    extra_system_notes: tuple[str, ...] = field(default_factory=tuple)
    compile_metadata: Callable[[dict[str, Any], np.ndarray, np.ndarray], dict[str, Any]] | None = None
    # v3.6 (docs/06 §3 M3): format-contract lines that depend on the run's
    # total-qubit budget (the LP group-order cap = n_budget // 4). The prompt
    # assembler calls this with RunConfig.n_budget and appends the result to
    # ``format_contract_lines``; None for families with no budget-dependent
    # rule. Pure text: nothing here reads the catalog's budget context.
    budget_contract_lines: Callable[[int | None], tuple[str, ...]] | None = None
    # v3.5 SEC: JSON-safe (check_block, data_block, perm) block structure of
    # hx/hz from the VALIDATED spec (schemas.CompiledCode.layer_blocks). None
    # for families without a block lift (hgp) -> fallback edge colouring.
    layer_blocks: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    # Arithmetic reading of ``size_formula_line``: total qubits from the RAW
    # spec, with no construction. Lets the budget gate reject before the build
    # instead of after it (verification/structural.py), which is what makes the
    # LP group-order cap scale with the budget rather than sit at a constant.
    # Returns None whenever the spec cannot be read with certainty, so an
    # unpredictable spec still gets the builder's own diagnostic.


def _guarded(family: str, validate: Callable[[dict[str, Any]], dict[str, Any]]):
    def wrapper(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate(spec)
        except ProgramError:
            raise
        except Exception as exc:
            raise ProgramError(f"{family}: {exc}") from exc

    return wrapper


_SHAPE_POOL = ", ".join(f"{r}x{c}" for r, c in DEFAULT_ALLOWED_SHAPES)

_LP_SHARED_FORMAT_LINES = (
    "- LP A and B are rectangular cell matrices; each shape is chosen INDEPENDENTLY "
    f"from {{{_SHAPE_POOL}}}.",
    "- Every cell is a non-empty list of exponent tuples (one tuple per group element; "
    f"identity is all zeros); no duplicate tuples in a cell; cell weight <= {DEFAULT_MAX_CELL_WEIGHT}.",
    "- The elements appearing across A and B together must generate the WHOLE group "
    "(otherwise the lift degenerates to a subgroup and is rejected).",
)

# Rendered only when the run has NO construction-time check-weight cap (the v3.7
# as-run protocol). Under a cap -- the paper's w_max <= 8 since the 2026-11-10
# alignment -- funsearch._construction_contract_lines swaps it for
# LP_CAPPED_WEIGHT_LINE, so the contract never says "not capped" next to the
# CHECK WEIGHT CAP line.
LP_UNCAPPED_WEIGHT_LINE = (
    "LP: the compiled per-check stabilizer weight is recorded, not capped (v3.7 removed the cap); "
    "cell weight <= 8 is the only hard limit."
)
LP_CAPPED_WEIGHT_LINE = (
    "LP: cell weight <= 8 is a grammar limit; the compiled per-check stabilizer weight must also "
    "satisfy the CHECK WEIGHT line of this contract."
)

_LP_SHARED_CONTRACT_LINES = (
    "LP: pick A/B shapes and the group so (nA+mA)*(nB+mB)*|G| <= the total qubit budget.",
    LP_UNCAPPED_WEIGHT_LINE,
    "LP: exponent tuples must contain plain Python ints; exponents reduce modulo the generator orders.",
)


def lp_group_order_contract_line(n_budget: int | None) -> str:
    """The budget-derived group-order cap, rendered for the generator prompt.

    Reads ``config.group_order_cap_for_budget`` through the catalog so the
    number in the prompt is the number ``get_group`` enforces (v3.6 M3: b288
    -> 72, b360 -> 90). Without a budget the line states the rule and the
    resource guard."""

    from omniqec_agentic.config import MAX_GROUP_ORDER, group_order_cap_for_budget

    if n_budget is None:
        return (
            "- lp_nonabelian group order must be <= n_budget // 4 (the absolute "
            f"resource guard is {int(MAX_GROUP_ORDER)}); larger groups are rejected "
            "before construction."
        )
    cap = int(group_order_cap_for_budget(int(n_budget)))
    return (
        f"- lp_nonabelian group order must be <= n_budget // 4 (= {cap} at this "
        "budget); larger groups are rejected before construction."
    )


def _lp_nonabelian_budget_lines(n_budget: int | None) -> tuple[str, ...]:
    return (lp_group_order_contract_line(n_budget),)


def _lp_compile_metadata(resolver):
    def compile_metadata(spec, hx, hz):
        return lp_group_metadata(spec, resolve_group=resolver)

    return compile_metadata


_BB = FamilyPlugin(
    name="bb",
    validate=_guarded("bb", validate_bb_spec),
    build=bb_code,
    sector_sizes=bb_sector_sizes,
    layer_blocks=bb_layer_blocks,
    genotype_line=(
        "For bb: {'family':'bb', 'ell': <int>, 'm': <int>, 'A': [[a,b],...], 'B': [[a,b],...]} "
        "(bivariate bicycle: each [a,b] is a monomial x^a y^b on the Z_ell x Z_m torus)."
    ),
    syntax_example_line=(
        "BB example: {'family':'bb','ell':3,'m':5,'A':[[1,0],[0,1],[2,2]],'B':[[0,2],[2,0],[1,1]]}."
    ),
    format_contract_lines=(
        "- BB ell and m are positive Python ints; A and B are non-empty lists of two-int pairs.",
        "- BB total weight len(A)+len(B) must lie in [4, 8] (the w_max <= 8 constraint: a BB check has weight "
        "len(A)+len(B)); duplicate terms (after mod "
        "reduction) are invalid.",
        "- BB is canonicalized to ell <= m; the validator swaps the axes automatically if needed.",
    ),
    size_formula_line="- bb: N_data = 2*ell*m, checks = 2*ell*m, total qubits = 4*ell*m.",
    user_contract_lines=(
        "BB: choose ell,m so 4*ell*m <= the total qubit budget before returning.",
        "BB: keep 4 <= len(A)+len(B) <= 8 and avoid degenerate term choices that "
        "disconnect the Tanner graph.",
    ),
    extra_system_notes=(
        "RNG-driven BB style is also valid, for example building A and B terms as "
        "[int(rng.integers(ell)), int(rng.integers(m))] so each term stays a two-int pair.",
    ),
)


_HGP = FamilyPlugin(
    name="hgp",
    validate=_guarded("hgp", validate_hgp_spec),
    build=hgp_code,
    sector_sizes=hgp_sector_sizes,
    genotype_line=(
        "For hgp: {'family':'hgp', 'seeds': {'H1': <binary matrix>, 'H2': <binary matrix>}} "
        "(hypergraph product of two independently designed classical parity-check seeds)."
    ),
    syntax_example_line=(
        "HGP example: {'family':'hgp','seeds':{'H1':[[1,1,0],[0,1,1],[1,0,1]],"
        "'H2':[[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]]}}."
    ),
    format_contract_lines=(
        "- HGP seeds are rectangular 0/1 matrices; every row AND column degree must lie in [2, 4].",
        "- HGP seeds may not contain duplicate rows; H1 == H2 is allowed but independent "
        "seed design is encouraged.",
        "- HGP seeds need not be full row rank; only k > 0 of the product code is required.",
    ),
    size_formula_line=(
        "- hgp: for H1 m1-by-n1 and H2 m2-by-n2, N_data = n1*n2 + m1*m2, "
        "checks = m1*n2 + n1*m2, total qubits = (n1+m1)*(n2+m2)."
    ),
    user_contract_lines=(
        "HGP: choose both seed shapes so (n1+m1)*(n2+m2) <= the total qubit budget.",
        "HGP: seed shapes are otherwise free — the budget is the only size limit.",
    ),
)


_LP_ABELIAN = FamilyPlugin(
    name="lp_abelian",
    validate=_guarded(
        "lp_abelian",
        partial(validate_lp_spec, family="lp_abelian", resolve_group=resolve_abelian_group),
    ),
    build=partial(lp_code, resolve_group=resolve_abelian_group),
    sector_sizes=partial(lp_sector_sizes, resolve_group=resolve_abelian_group),
    layer_blocks=partial(lp_layer_blocks, resolve_group=resolve_abelian_group),
    genotype_line=(
        "For lp_abelian: {'family':'lp_abelian', 'group': {'factors': [q1,q2,...]}, "
        "'A': <cell matrix>, 'B': <cell matrix>} (lifted product over the abelian group "
        "C_q1 x C_q2 x ...; an exponent tuple [e1,e2,...] means g1^e1 * g2^e2 * ...)."
    ),
    syntax_example_line=(
        "LP-abelian example: {'family':'lp_abelian','group':{'factors':[2,3]},"
        "'A':[[[[0,0],[1,1]],[[0,1],[1,0]]]],'B':[[[[0,0],[1,2]]]]}."
    ),
    format_contract_lines=(
        "- lp_abelian group factors are Python ints >= 2; any finite abelian group is allowed, "
        "not just cyclic.",
    )
    + _LP_SHARED_FORMAT_LINES,
    size_formula_line=(
        "- lp_abelian/lp_nonabelian: if A is mA-by-nA and B is mB-by-nB, "
        "N_data = (nA*nB + mA*mB)*|G|, checks = (mA*nB + nA*mB)*|G|, "
        "total qubits = (nA+mA)*(nB+mB)*|G|."
    ),
    user_contract_lines=_LP_SHARED_CONTRACT_LINES,
    compile_metadata=_lp_compile_metadata(resolve_abelian_group),
)


_LP_NONABELIAN = FamilyPlugin(
    name="lp_nonabelian",
    validate=_guarded(
        "lp_nonabelian",
        partial(validate_lp_spec, family="lp_nonabelian", resolve_group=resolve_nonabelian_group),
    ),
    build=partial(lp_code, resolve_group=resolve_nonabelian_group),
    sector_sizes=partial(lp_sector_sizes, resolve_group=resolve_nonabelian_group),
    layer_blocks=partial(lp_layer_blocks, resolve_group=resolve_nonabelian_group),
    genotype_line=(
        "For lp_nonabelian: {'family':'lp_nonabelian', 'group': '<catalog name>', "
        "'A': <cell matrix>, 'B': <cell matrix>} (same seed grammar as lp_abelian; only "
        "the group source differs)."
    ),
    syntax_example_line=(
        # One example per grammar branch, because the branches differ in TUPLE
        # ARITY and a single dihedral example silently taught that every group
        # takes 2-tuples. These are format illustrations, not good codes.
        "LP-nonabelian examples, one per catalog branch (note the differing "
        "tuple arities): dihedral {'family':'lp_nonabelian','group':'D4',"
        "'A':[[[[0,0],[1,0],[0,1]]]],'B':[[[[0,0],[2,1]],[[1,1],[3,0]]]]}; "
        "metacyclic {'family':'lp_nonabelian','group':'C7:C3(2)',"
        "'A':[[[[0,0],[1,0]]]],'B':[[[[0,0],[0,1],[3,2]]]]} (tuples [i,j] = r^i s^j); "
        "direct product {'family':'lp_nonabelian','group':'C3xD3',"
        "'A':[[[[0,0,0],[1,1,0]]]],'B':[[[[0,0,0],[0,0,1],[2,1,0]]]]} "
        "(tuples are [outer]+[inner...], so C{m}xD{n} takes 3 entries); "
        "A4/S4 {'family':'lp_nonabelian','group':'A4','A':[[[[0,0,0],[1,0,0]]]],"
        "'B':[[[[0,0,0],[0,1,0],[0,0,1]]]]} (tuples along the composition series)."
    ),
    format_contract_lines=(
        f"- lp_nonabelian group names follow the catalog grammar: {catalog_summary()}.",
        "- A metacyclic twist k is normalized to the canonical representative of its power "
        "orbit; known-isomorphic presentations (e.g. C2xD5 = D10) are rejected with the "
        "canonical name.",
    )
    + _LP_SHARED_FORMAT_LINES,
    size_formula_line=_LP_ABELIAN.size_formula_line,
    user_contract_lines=_LP_SHARED_CONTRACT_LINES
    + (
        "LP-nonabelian: the group must be genuinely used — supports generating a proper "
        "(possibly abelian) subgroup are rejected.",
    ),
    compile_metadata=_lp_compile_metadata(resolve_nonabelian_group),
    budget_contract_lines=_lp_nonabelian_budget_lines,
)


FAMILY_REGISTRY: dict[str, FamilyPlugin] = {
    plugin.name: plugin for plugin in (_BB, _HGP, _LP_ABELIAN, _LP_NONABELIAN)
}


def get_family(name: str) -> FamilyPlugin | None:
    return FAMILY_REGISTRY.get(str(name))


def format_contract_lines_for_budget(plugin: FamilyPlugin, n_budget: int | None) -> tuple[str, ...]:
    """``plugin.format_contract_lines`` plus its budget-dependent lines at
    ``n_budget`` (the RunConfig budget; falls back to the catalog's current
    budget context, then to the budget-free wording)."""

    if n_budget is None:
        n_budget = get_group_order_budget()
    extra = plugin.budget_contract_lines(n_budget) if plugin.budget_contract_lines else ()
    return tuple(plugin.format_contract_lines) + tuple(extra)
