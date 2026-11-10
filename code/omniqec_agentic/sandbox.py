"""Restricted execution for ver1-style `construct(rng)` programs."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ProgramError
from .schemas import ConstructedSpec, family_from_spec

DEFAULT_CONSTRUCT_TIMEOUT_S = 10.0


@contextmanager
def _time_limit(seconds: float):
    """SIGALRM-based wall-clock limit for untrusted construct programs.

    Only armed in the main thread (signal restriction); worker threads fall
    back to no timeout, which is acceptable because non-main-thread callers
    only re-run sources that already passed a timed evaluation once."""

    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise ProgramError(f"construct timed out after {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)

_SAFE_BUILTIN_NAMES = (
    "range",
    "len",
    "int",
    "float",
    "list",
    "dict",
    "tuple",
    "set",
    "min",
    "max",
    "abs",
    "sum",
    "sorted",
    "enumerate",
    "zip",
    "map",
    "filter",
    "round",
    "bool",
    "all",
    "any",
    "reversed",
    "print",
)

_BUILTINS = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
_SAFE_BUILTINS = {name: _BUILTINS[name] for name in _SAFE_BUILTIN_NAMES}


@dataclass(frozen=True)
class SandboxResult:
    program_id: str
    constructed: ConstructedSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "constructed": self.constructed.to_dict(),
        }


def run_construct_source(
    source: str,
    *,
    program_id: str,
    seed: int = 0,
    timeout_s: float = DEFAULT_CONSTRUCT_TIMEOUT_S,
) -> SandboxResult:
    """Execute a ver1-style construct program and return a normalized spec."""

    spec = run_construct(source, seed=seed, timeout_s=timeout_s)
    constructed = ConstructedSpec(
        program_id=program_id,
        spec=spec,
        family=family_from_spec(spec),
        rng_seed=int(seed),
        normalized=True,
    )
    return SandboxResult(program_id=program_id, constructed=constructed)


def run_construct(
    source: str,
    *,
    seed: int = 0,
    timeout_s: float = DEFAULT_CONSTRUCT_TIMEOUT_S,
) -> dict[str, Any]:
    """Exec `source`, call `construct(default_rng(seed))`, and validate the spec."""

    namespace = {"np": np, "numpy": np, "__builtins__": _SAFE_BUILTINS}
    try:
        with _time_limit(timeout_s):
            exec(compile(source, "<construct>", "exec"), namespace)
    except ProgramError:
        raise
    except Exception as exc:
        raise ProgramError(f"compile/exec failed: {exc}") from exc

    construct = namespace.get("construct")
    if not callable(construct):
        raise ProgramError("no callable construct(rng) defined")

    try:
        with _time_limit(timeout_s):
            spec = construct(np.random.default_rng(seed))
    except ProgramError:
        raise
    except Exception as exc:
        raise ProgramError(f"construct() raised: {exc}") from exc
    return validate_spec(spec)


def validate_spec(spec: Any) -> dict[str, Any]:
    """Validate and normalize a spec via the family plugin registry."""

    from .construction.registry import get_family

    if not isinstance(spec, dict) or "family" not in spec:
        raise ProgramError("spec must be a dict with a 'family' key")

    plugin = get_family(str(spec["family"]))
    if plugin is None:
        raise ProgramError(f"unknown family '{spec['family']}'")
    return plugin.validate(spec)

