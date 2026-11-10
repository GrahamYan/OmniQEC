"""Agentic CSS QEC discovery framework.

The package is independent from the historical version directories in this
workspace. Reference code can be ported into this package, but runtime imports
should stay inside `omniqec_agentic` or third-party dependencies.
"""

from .config import (
    BposdDecoderConfig,
    CodeFamily,
    DecoderBuildConfig,
    DecoderName,
    DeformationMode,
    HardwareTarget,
    LLMConfig,
    RelayBpDecoderConfig,
    RunConfig,
    StructuralSearchConfig,
    VerificationConfig,
)
from .sandbox import ProgramError, run_construct, run_construct_source, validate_spec
from .funsearch import run_funsearch
from .evaluation import evaluate_slow_spec
from .campaign import run_agentic_cycle

__all__ = [
    "CodeFamily",
    "BposdDecoderConfig",
    "DecoderBuildConfig",
    "DecoderName",
    "DeformationMode",
    "HardwareTarget",
    "LLMConfig",
    "RelayBpDecoderConfig",
    "RunConfig",
    "StructuralSearchConfig",
    "VerificationConfig",
    "ProgramError",
    "run_construct",
    "run_construct_source",
    "validate_spec",
    "evaluate_slow_spec",
    "run_agentic_cycle",
    "run_funsearch",
]
