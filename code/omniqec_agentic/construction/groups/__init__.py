"""Group layer: exponent-tuple encoded finite groups (design doc §4)."""

from .abelian import abelian_group, invariant_factors
from .base import Group
from .catalog import catalog_summary, get_group

__all__ = ["Group", "abelian_group", "catalog_summary", "get_group", "invariant_factors"]
