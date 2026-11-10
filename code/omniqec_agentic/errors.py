"""Shared exception types (kept dependency-free to avoid import cycles)."""

from __future__ import annotations


class ProgramError(Exception):
    """Raised when a generated construct program or its spec is invalid."""
