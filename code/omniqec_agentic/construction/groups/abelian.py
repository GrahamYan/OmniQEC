"""Finite abelian groups ``C_q1 x C_q2 x ...`` for the lp_abelian family.

The spec declares the group as a plain factor list; construction uses the
declared factors verbatim (exponent tuples are read against them), while
``invariant_factors`` provides the isomorphism-invariant identity recorded in
metadata (design doc §4.1).
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from .base import Group, cycle_perm, group_from_generators, pad_perm


@lru_cache(maxsize=None)
def abelian_group(factors: tuple[int, ...]) -> Group:
    clean = tuple(int(q) for q in factors)
    if not clean or any(q < 2 for q in clean):
        raise ValueError("abelian group factors must be a non-empty list of ints >= 2")
    total = sum(clean)
    generators = []
    offset = 0
    for q in clean:
        generators.append(pad_perm(cycle_perm(q), offset=offset, total=total))
        offset += q
    return group_from_generators(
        name="x".join(f"C{q}" for q in clean),
        kind="abelian",
        generator_perms=generators,
        generator_orders=clean,
    )


def invariant_factors(factors: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Canonical invariant-factor form ``d1 | d2 | ... | dk`` (ascending)."""

    prime_powers: dict[int, list[int]] = defaultdict(list)
    for q in factors:
        for prime, exponent in _prime_factorization(int(q)):
            prime_powers[prime].append(prime**exponent)
    slots = max(len(powers) for powers in prime_powers.values())
    combined = []
    for position in range(slots):
        factor = 1
        for prime in prime_powers:
            ranked = sorted(prime_powers[prime], reverse=True)
            if position < len(ranked):
                factor *= ranked[position]
        combined.append(factor)
    return tuple(sorted(combined))


def _prime_factorization(value: int) -> list[tuple[int, int]]:
    if value < 2:
        raise ValueError("factors must be >= 2")
    result = []
    divisor = 2
    while divisor * divisor <= value:
        exponent = 0
        while value % divisor == 0:
            value //= divisor
            exponent += 1
        if exponent:
            result.append((divisor, exponent))
        divisor += 1
    if value > 1:
        result.append((value, 1))
    return result
