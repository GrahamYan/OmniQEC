"""Seed grammar layer (design doc §5)."""

from .classical_seed import validate_classical_seed
from .ga_seed import DEFAULT_ALLOWED_SHAPES, require_supports_generate, validate_ga_matrix

__all__ = [
    "DEFAULT_ALLOWED_SHAPES",
    "require_supports_generate",
    "validate_classical_seed",
    "validate_ga_matrix",
]
