"""Known-good construct programs for the four canonical families.

The five manually written seed programs of the paper (appendix D1: one BB, one
HGP, one Abelian LP and two non-Abelian LP examples). A campaign executes them
in the FIRST iteration only (protocol key seed_programs=first_turn,
config.seed_programs_run_this_turn), separately from the LLM program budget;
later iterations keep what they produced through the discovery memory. They
are the cold-start seeds when no warm-start seed_sources are given. None of
them is a benchmark: the BB seed is deliberately NOT the BB72
[[72,12,6]] baseline (benchmark blinding); the LP-nonabelian seed is the
Lin-Pryadko D6 [[24,8,3]] code (arXiv:2306.16400 Table III, m=6 row,
a = 1+r^4, b = 1+sr^4+r^3+r^4+sr^2+r) converted to this package's
rotation-first exponent tuples (their s^e r^k = our r^{-k mod 6} s^e) and
MILP-recertified at d=3 here; the LP-abelian seed is a [[27,2,4]] C3xC3
lifted product (MILP-certified d=4, found by in-house sweep 2026-08-15).
"""

SEED_BB = '''
def construct(rng):
    return {"family": "bb", "ell": 3, "m": 5,
            "A": [[1, 0], [0, 1], [2, 2]],
            "B": [[0, 2], [2, 0], [1, 1]]}
'''

SEED_HGP = '''
def construct(rng):
    circulant = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]
    return {"family": "hgp", "seeds": {"H1": circulant, "H2": circulant}}
'''

SEED_LP_ABELIAN = '''
def construct(rng):
    return {"family": "lp_abelian",
            "group": {"factors": [3, 3]},
            "A": [[[[2, 0], [0, 0]], [[1, 0], [0, 1]]]],
            "B": [[[[0, 2], [1, 2], [0, 0]]]]}
'''

SEED_LP_NONABELIAN = '''
def construct(rng):
    return {"family": "lp_nonabelian",
            "group": "D6",
            "A": [[[[0, 0], [4, 0]]]],
            "B": [[[[0, 0], [2, 1], [3, 0], [4, 0], [4, 1], [1, 0]]]]}
'''

# The nonabelian catalog has four grammar branches, and for a long time every
# exemplar here was dihedral — so the search only ever mutated inside that one
# branch (a measured 85% of proposals). This seed is metacyclic: a different
# branch, a different tuple arity, and deliberately NOT from the literature
# (found by an in-house random probe, [[42,4,5]] weight 5).
SEED_LP_METACYCLIC = '''
def construct(rng):
    return {"family": "lp_nonabelian",
            "group": "C7:C3(2)",
            "A": [[[[0, 0], [1, 0]]]],
            "B": [[[[0, 0], [0, 1], [3, 2]]]]}
'''

SEEDS = [
    SEED_BB,
    SEED_HGP,
    SEED_LP_ABELIAN,
    SEED_LP_NONABELIAN,
    SEED_LP_METACYCLIC,
]
