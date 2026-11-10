"""Non-abelian group catalog for the lp_nonabelian family (design doc §4.2).

Catalog v1 (A5 removed — the only non-solvable candidate, no exponent normal
form; the Mitten-paper groups are all solvable):

  D{n}            dihedral, n >= 3, |G| = 2n, tuples [i,j] = r^i s^j
  A4 / S4         |G| = 12 / 24, tuples along the composition series
  C{m}:C{n}({k})  metacyclic C_m x| C_n with s^-1 r s = r^k, |G| = m*n
  C{m}x{H}        direct products C_m x H, H in {D_n, A4, S4}

Groups are constructed on demand from the name grammar and cached. Two size
gates realize the budget-parameterized catalog without a hard-coded list:
``catalog_order`` reads |G| straight off the name and rejects before
``_construct`` runs (building |G|=1200 costs ~30s, and the cap used to fire
only afterwards); the limit itself is the ``config.MAX_GROUP_ORDER`` resource
guard, and (v3.6 M3) the binding per-run bound is the budget-derived
``n_budget // 4`` read from ``set_group_order_budget``. Metacyclic twists are
normalized to the canonical representative min{k^j : gcd(j, n) = 1} — a valid
non-canonical twist is accepted and rewritten, not rejected. Known-isomorphic
presentations are rejected with a pointer to the canonical name (conservative
rule-based dedup; residual cross-category isomorphs are accepted and dedupe at
code-hash level). GAP SmallGroup ids are attached where the literature pins
them (Mitten arXiv:2607.28795 Table XIII; Lin-Pryadko arXiv:2306.16400).
"""

from __future__ import annotations

import re
from functools import lru_cache
from math import gcd

from contextlib import contextmanager
from typing import Iterator

from omniqec_agentic.config import MAX_GROUP_ORDER, group_order_cap_for_budget

from .base import Group, Perm, compose, cycle_perm, group_from_generators, pad_perm

# v3.6 (docs/06 §3 M3): the binding group-order bound is BUDGET-DERIVED. The
# run sets its total-qubit budget here once (``set_group_order_budget``, called
# at the start of ``funsearch.run_funsearch`` and before every fast-evaluation
# batch) and every ``get_group`` call of the turn — validation, build, layer
# blocks, metadata — sees ``group_order_cap_for_budget(n_budget)`` =
# n_budget // 4. With no budget in scope the absolute resource guard
# ``MAX_GROUP_ORDER`` applies, which exists only to keep an absurd name from
# costing seconds of Cayley-table construction.
_GROUP_ORDER_BUDGET: int | None = None


def set_group_order_budget(n_budget: int | None) -> None:
    """Set the run's total-qubit budget for the catalog's order cap
    (``None`` -> resource guard only)."""

    global _GROUP_ORDER_BUDGET
    _GROUP_ORDER_BUDGET = None if n_budget is None else int(n_budget)


def get_group_order_budget() -> int | None:
    return _GROUP_ORDER_BUDGET


@contextmanager
def group_order_budget(n_budget: int | None) -> Iterator[None]:
    """Scoped ``set_group_order_budget`` (tests, one-off tools)."""

    previous = _GROUP_ORDER_BUDGET
    set_group_order_budget(n_budget)
    try:
        yield
    finally:
        set_group_order_budget(previous)


def effective_group_order_cap(max_order: int | None = None) -> int:
    """The cap ``get_group`` enforces right now: an explicit ``max_order``,
    else the budget-derived cap, else the resource guard."""

    if max_order is not None:
        return int(max_order)
    return int(group_order_cap_for_budget(_GROUP_ORDER_BUDGET))


def _cap_message(order: int, limit: int, max_order: int | None) -> str:
    budget = _GROUP_ORDER_BUDGET
    if max_order is not None:
        return f"order {order} exceeds the explicit cap {limit}"
    if budget is not None and limit < int(MAX_GROUP_ORDER):
        return (
            f"order {order} exceeds the budget cap {limit} "
            f"(= n_budget // 4 for n_budget {budget})"
        )
    if budget is not None:
        return (
            f"order {order} exceeds the resource guard {limit} "
            f"(MAX_GROUP_ORDER; the budget cap for n_budget {budget} is "
            f"{int(budget) // 4}, itself capped at the guard)"
        )
    return f"order {order} exceeds the resource guard {limit} (MAX_GROUP_ORDER)"

# GAP SmallGroup(order, index) ids confirmed in the literature; unlisted
# catalog entries stay None (non-blocking backfill, design doc §11).
KNOWN_GAP_IDS: dict[str, tuple[int, int]] = {
    "A4": (12, 3),
    "S4": (24, 12),
    "C5xD3": (30, 1),      # Mitten "C5 x S3"
    "C4xD5": (40, 5),      # Mitten "C4 x D10" (order-10 dihedral)
    "C10xD3": (60, 11),
    "C11xD3": (66, 1),
    "C5:C8(4)": (40, 1),   # Lin-Pryadko Table I, [[80,8,10]] host group
    "C5:C20(2)": (100, 9),
    "C5xS4": (120, 37),
    "C13xA4": (156, 13),
    "C7xS4": (168, 45),
    "C13:C15(3)": (195, 1),
}

_DIHEDRAL_RE = re.compile(r"^D(\d+)$")
_METACYCLIC_RE = re.compile(r"^C(\d+):C(\d+)\((\d+)\)$")
_DIRECT_RE = re.compile(r"^C(\d+)x(D\d+|A4|S4)$")

# Literature spellings that denote a catalog group under another name. Rejected
# with a pointer rather than accepted as an alias: construction/lp.py echoes
# ``group.name`` back into the normalized spec, so one group must have exactly
# one spelling in the archive -- the same rule that sends C2xD{n} to D{2n}.
_SYMMETRIC_ALIASES = {"S3": "D3"}
_LITERATURE_RE = re.compile(r"^(C\d+x)?(S3)$")

# Literature indexes dihedral groups by ORDER (its D10 has 10 elements); this
# catalog indexes by rotation order (D10 here has 20). A transcribed name
# therefore resolves to a real but *different* group, and the only thing that
# usually stops it is the size gate -- a confusing place to discover a naming
# convention, so the gate says so when the literature reading is a catalog entry.
_DIHEDRAL_NAME_RE = re.compile(r"^(C\d+x)?D(\d+)$")

# The dihedral-index collision in the one form the cited tables actually print:
# Mitten Table XIII writes "C4 x D10" for the order-40 group spelled C4xD5 here
# (KNOWN_GAP_IDS above). This has to be an explicit list rather than a rule over
# the name, because the rule cannot tell a mis-transcription from a legitimate
# request: C5xD6 halves to the catalog entry C5xD3 too, yet it is a perfectly
# good order-60 group nobody mistyped. So only printed literature spellings are
# claimed; every other even-index dihedral product stays constructible and gets
# the non-blocking ``_dihedral_index_hint`` if it trips the size gate.
#
# This is the name-level check docs/01 §13 required before the order cap could
# be raised past 80: the cap used to be the only thing standing between
# "C4xD10" and a silently different group.
_LITERATURE_DIHEDRAL_SPELLINGS: dict[str, str] = {"C4xD10": "C4xD5"}


def _literal_order(name: str) -> int | None:
    """|G| read straight off the name grammar, reserved spellings included.

    Only two callers want this reading: ``catalog_order`` (which then filters
    the reserved names out) and ``_literature_notation_error`` (which quotes it
    to say what the name would have meant if taken literally). Everyone else
    wants ``catalog_order``.
    """

    text = str(name)
    if text == "A4":
        return 12
    if text == "S4":
        return 24
    match = _DIHEDRAL_RE.match(text)
    if match:
        return 2 * int(match.group(1))
    match = _METACYCLIC_RE.match(text)
    if match:
        return int(match.group(1)) * int(match.group(2))
    match = _DIRECT_RE.match(text)
    if match:
        inner = _literal_order(match.group(2))
        return None if inner is None else int(match.group(1)) * inner
    return None


def catalog_order(name: str) -> int | None:
    """|G| for a name this catalog will actually resolve, without constructing.

    ``None`` means "no opinion, ask ``_construct``" — the name is outside the
    grammar, OR it is a reserved literature spelling that resolves to no group
    at all. Both must abstain rather than answer, because every caller uses a
    number here to reject *early*, and an early rejection replaces the real
    diagnostic with a size verdict. That is not hypothetical: ``C4xD10`` reads
    literally as order 80, so answering 80 made the budget pre-gate in
    ``verification/structural.py`` report ``total_qubit_budget_violation`` at
    every budget below 320 and the "use 'C4xD5'" pointer never surfaced.

    A returned order is exact for every grammar branch — pinned against
    ``_construct`` in tests/test_group_catalog_naming.py — because the size gate
    runs *before* construction and a wrong number would reject a legal group.
    """

    text = str(name)
    # Reserved spellings resolve to nothing, so they have no order to report.
    if text in _LITERATURE_DIHEDRAL_SPELLINGS or _LITERATURE_RE.match(text):
        return None
    return _literal_order(text)


def _dihedral_index_hint(name: str) -> str:
    match = _DIHEDRAL_NAME_RE.match(name)
    if not match:
        return ""
    n = int(match.group(2))
    halved = f"{match.group(1) or ''}D{n // 2}"
    if n % 2 or halved not in KNOWN_GAP_IDS:
        return ""
    return (
        f"; if this name came from the literature, its D{n} is the order-{n} "
        f"dihedral group, which this catalog spells '{halved}'"
    )


def _literature_notation_error(name: str) -> str | None:
    """Message for a name the literature prints for a *different* group."""

    match = _LITERATURE_RE.match(name)
    if match:
        canonical = f"{match.group(1) or ''}{_SYMMETRIC_ALIASES[match.group(2)]}"
        return (
            f"'{name}' is literature notation: S3 is the order-6 dihedral group, "
            f"which this catalog indexes by rotation order as D3 — use '{canonical}'"
        )
    canonical = _LITERATURE_DIHEDRAL_SPELLINGS.get(name)
    if canonical is None:
        return None
    index = int(_DIHEDRAL_NAME_RE.match(name).group(2))
    return (
        f"'{name}' is literature notation: its D{index} is the order-{index} "
        f"dihedral group, which this catalog indexes by rotation order as "
        f"D{index // 2} — use '{canonical}'. Read literally the name is a "
        f"different group of order {_literal_order(name)}, so the spelling is "
        f"reserved rather than silently accepted"
    )


def get_group(name: str, *, max_order: int | None = None) -> Group:
    """Resolve a catalog name to a Group; the result's ``.name`` is canonical.

    The size limit is checked against ``catalog_order(name)`` BEFORE anything is
    built. That ordering matters: construction is roughly O(|G|^2), so the old
    construct-then-check spent ~30s on a name like ``D600`` only to reject it.

    The limit is ``max_order`` when given, else the budget-derived cap
    ``n_budget // 4`` of the budget set with ``set_group_order_budget`` (v3.6
    M3: b288 -> 72, b360 -> 90), else the ``MAX_GROUP_ORDER`` resource guard.
    """

    text = str(name)
    limit = effective_group_order_cap(max_order)
    predicted = catalog_order(text)
    if predicted is not None and predicted > limit:
        raise ValueError(
            f"group '{text}' has order {predicted}: "
            f"{_cap_message(predicted, limit, max_order)}"
            f"{_dihedral_index_hint(text)}"
        )
    group = _construct(text)
    # Unreachable for any parseable name -- catalog_order is exact for every
    # grammar branch (pinned in tests/test_group_catalog_naming.py) and twist
    # normalization preserves m*n. Kept so the invariant stays enforced rather
    # than assumed if the two ever drift apart.
    if group.order > limit:
        raise ValueError(
            f"group '{group.name}' has order {group.order}: "
            f"{_cap_message(group.order, limit, max_order)}"
            f"{_dihedral_index_hint(group.name)}"
        )
    return group


def catalog_summary() -> str:
    """Prompt-facing description of the name grammar (not an enumeration)."""

    return (
        "D{n} dihedral (n>=3, |G|=2n, tuples [i,j]=r^i s^j); "
        "A4 (|G|=12, tuples [i,j,l]); S4 (|G|=24, tuples [i,j,l,p]); "
        "metacyclic C{m}:C{n}({k}) with m>=3, n>=2, gcd(k,m)=1, k^n=1 mod m, "
        "k!=1 (|G|=m*n, tuples [i,j]=r^i s^j, relation s^-1 r s = r^k); "
        "direct products C{m}x{H} for H in D{n}/A4/S4, m>=2 "
        "(|G|=m*|H|, tuples [e]+H-tuple). "
        "These are catalog spellings, not literature ones: dihedral groups are "
        "indexed by rotation order, so the order-6 group often written S3 is D3 "
        "here and the order-10 one often written D10 is D5"
    )


@lru_cache(maxsize=None)
def _construct(name: str) -> Group:
    # Identity before syntax: a literature spelling is a legal-looking name for
    # a group the caller did not mean, so it must lose to nothing below.
    literature = _literature_notation_error(name)
    if literature:
        raise ValueError(literature)
    if name == "A4":
        return _a4()
    if name == "S4":
        return _s4()
    match = _DIHEDRAL_RE.match(name)
    if match:
        return _dihedral(int(match.group(1)))
    match = _METACYCLIC_RE.match(name)
    if match:
        return _metacyclic(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DIRECT_RE.match(name)
    if match:
        return _direct_product(int(match.group(1)), match.group(2))
    raise ValueError(
        f"unknown non-abelian group '{name}' (catalog grammar: {catalog_summary()})"
    )


def _named(kind: str, name: str, perms: list[Perm], orders: tuple[int, ...]) -> Group:
    return group_from_generators(
        name=name,
        kind=kind,
        generator_perms=perms,
        generator_orders=orders,
        gap_id=KNOWN_GAP_IDS.get(name),
    )


def _dihedral_perms(n: int) -> list[Perm]:
    rotation = cycle_perm(n)
    reflection = tuple((n - i) % n for i in range(n))
    return [rotation, reflection]


def _dihedral(n: int) -> Group:
    if n < 3:
        raise ValueError("dihedral D_n requires n >= 3")
    return _named("dihedral", f"D{n}", _dihedral_perms(n), (n, 2))


# A4/S4 generator conventions (design doc §4.3): V4 = <b, c>, a of order 3,
# t a transposition; normal forms follow the composition series.
_A4_PERMS: list[Perm] = [(1, 2, 0, 3), (1, 0, 3, 2), (3, 2, 1, 0)]
_S4_PERMS: list[Perm] = [(1, 0, 2, 3)] + _A4_PERMS


def _a4() -> Group:
    return _named("alternating", "A4", _A4_PERMS, (3, 2, 2))


def _s4() -> Group:
    return _named("symmetric", "S4", _S4_PERMS, (2, 3, 2, 2))


def canonical_twist(m: int, n: int, k: int) -> int:
    """min over the twist orbit {k^j mod m : gcd(j, n) = 1} (isomorphic groups)."""

    return min(pow(k, j, m) for j in range(1, n + 1) if gcd(j, n) == 1)


def _prime_power_factors(m: int) -> list[tuple[int, int]]:
    factors: list[tuple[int, int]] = []
    q, rest = 2, int(m)
    while q * q <= rest:
        if rest % q == 0:
            a = 0
            while rest % q == 0:
                rest //= q
                a += 1
            factors.append((q, a))
        q += 1
    if rest > 1:
        factors.append((rest, 1))
    return factors


def metacyclic_direct_product_name(m: int, n: int, k: int) -> str | None:
    """Catalog spelling of a metacyclic presentation that is secretly a
    direct product ``C{c}xD{m1}`` (v3.6 minor: isomorphism dedup, docs/05 §4
    "40 阶 11 个名字只覆盖 8 个同构类").

    Rule (provable, conservative): let the twist ``x -> k x`` on Z_m square to
    the identity and act on every prime-power factor q^a of m either
    trivially (k = 1 mod q^a, the central part m2) or by inversion (k = -1
    mod q^a, the dihedral part m1; a lone factor 2 counts as inverted). If
    n/2 is odd then C_n = C_2 x C_{n/2} with s^2 central, so
    G = D_{m1} x C_{m2} x C_{n/2}; when gcd(m2, n/2) = 1 the two central
    cyclic factors merge into C_{m2*n/2} and the group is the catalog entry
    ``C{m2*n/2}xD{m1}`` (``D{2*m1}`` when that cyclic factor is C_2 and m1 is
    odd — the existing C2xD{odd} rule). Examples: C4:C10(3) = C20:C2(11) =
    C5xD4 (k inverts the C4 and fixes the C5); C5:C6(4) = C3xD5. Returns
    None when the rule does not apply (the presentation is then accepted as
    its own catalog name; residual isomorphs dedupe at code-hash level)."""

    m, n, k = int(m), int(n), int(k)
    if n % 2 or pow(k, 2, m) != 1 or k % m == 1:
        return None
    m1 = m2 = 1
    for q, a in _prime_power_factors(m):
        qa = q**a
        residue = k % qa
        if q == 2 and a == 1:
            m1 *= qa  # k odd: 1 = -1 mod 2, file it with the inverted part
        elif residue == 1:
            m2 *= qa
        elif residue == qa - 1:
            m1 *= qa
        else:
            return None  # not +-1 on this factor: no dihedral split
    if m1 < 3:
        return None
    half = n // 2
    if half % 2 == 0 or gcd(m2, half) != 1:
        return None
    cyclic = m2 * half
    if cyclic == 1:
        return f"D{m1}"
    if cyclic == 2 and m1 % 2 == 1:
        return f"D{2 * m1}"
    return f"C{cyclic}xD{m1}"


def group_invariants(group: Group) -> dict[str, object]:
    """Cheap isomorphism invariants: element-order histogram and centre size.

    Equal invariants do not prove isomorphism, so this never drives a
    rejection on its own; it is the consistency check behind the rule-based
    dedup (a claimed isomorphism must at least agree here) and a diagnostic
    for the archive."""

    cayley = group.cayley
    order = int(group.order)
    identity = int(group.element_index(tuple(0 for _ in group.generator_orders)))
    histogram: dict[int, int] = {}
    for g in range(order):
        power, count = g, 1
        while power != identity:
            power = int(cayley[power, g])
            count += 1
        histogram[count] = histogram.get(count, 0) + 1
    centre = int(sum(1 for g in range(order) if all(cayley[g, h] == cayley[h, g] for h in range(order))))
    return {
        "order": order,
        "element_order_histogram": dict(sorted(histogram.items())),
        "centre_size": centre,
    }


def _metacyclic(m: int, n: int, k: int) -> Group:
    if m < 3 or n < 2:
        raise ValueError("metacyclic C_m:C_n requires m >= 3 and n >= 2")
    if not 2 <= k <= m - 1 or gcd(k, m) != 1:
        raise ValueError(f"metacyclic twist k={k} must be in [2, m-1] with gcd(k, m) = 1")
    if pow(k, n, m) != 1:
        raise ValueError(f"metacyclic twist must satisfy k^n = 1 mod m (k={k}, m={m}, n={n})")
    if n == 2 and k == m - 1:
        raise ValueError(f"C{m}:C2({k}) is the dihedral group; use D{m}")
    canonical = canonical_twist(m, n, k)
    if canonical != k:
        return _construct(f"C{m}:C{n}({canonical})")
    split = metacyclic_direct_product_name(m, n, k)
    if split is not None:
        raise ValueError(
            f"C{m}:C{n}({k}) is isomorphic to the direct product {split} "
            f"(the twist inverts a coprime cyclic factor and the rest is central); "
            f"use '{split}'"
        )
    # s^-1 r s = r^k  <=>  s r s^-1 = r^c with c = k^-1 mod m; realize r as the
    # +1 cycle and s as (x -> c*x) on m points plus its own n-cycle so that s
    # keeps order n even when the action order ord_m(k) properly divides n.
    c = pow(k, -1, m)
    total = m + n
    rotation = pad_perm(cycle_perm(m), offset=0, total=total)
    action = tuple((c * i) % m for i in range(m))
    twist = compose(
        pad_perm(action, offset=0, total=total),
        pad_perm(cycle_perm(n), offset=m, total=total),
    )
    return _named("metacyclic", f"C{m}:C{n}({k})", [rotation, twist], (m, n))


def _direct_product(m: int, inner_name: str) -> Group:
    if m < 2:
        raise ValueError("direct product C_m x H requires m >= 2")
    if inner_name == "A4":
        inner_perms, inner_orders = _A4_PERMS, (3, 2, 2)
    elif inner_name == "S4":
        inner_perms, inner_orders = _S4_PERMS, (2, 3, 2, 2)
    else:
        n = int(inner_name[1:])
        if n < 3:
            raise ValueError("direct product inner dihedral requires D_n with n >= 3")
        if m == 2 and n % 2 == 1:
            raise ValueError(f"C2xD{n} is isomorphic to D{2 * n}; use D{2 * n}")
        inner_perms, inner_orders = _dihedral_perms(n), (n, 2)
    inner_points = len(inner_perms[0])
    total = m + inner_points
    perms = [pad_perm(cycle_perm(m), offset=0, total=total)] + [
        pad_perm(perm, offset=m, total=total) for perm in inner_perms
    ]
    return _named("direct_product", f"C{m}x{inner_name}", perms, (m,) + inner_orders)
