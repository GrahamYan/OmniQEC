"""Deterministic construction compilers for candidate specs.

The construction layer executes supported constructed specs into a unified
Hx/Hz representation. It does not score or rank candidates. Natural sector
sizes are a per-family plugin export consumed via ``build_compiled_code``.
"""

from .build import (
    build_compiled_code,
    build_matrices,
    compiled_code_hash,
    compiled_code_metadata,
    symplectic_stabilizers,
)

__all__ = [
    "build_compiled_code",
    "build_matrices",
    "compiled_code_hash",
    "compiled_code_metadata",
    "symplectic_stabilizers",
]
