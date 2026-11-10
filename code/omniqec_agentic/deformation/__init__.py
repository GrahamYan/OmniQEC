"""Run-level local-Clifford deformation (design doc §7).

Family-agnostic: modes consume only the natural sector metadata every family
plugin exports. Controlled exclusively by run config (``deformation_mode``,
default off); never part of the genotype or the prompt.
"""

from .api import resolve_h_mask

__all__ = ["resolve_h_mask"]
