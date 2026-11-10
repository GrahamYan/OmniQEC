"""Evaluator library (design doc 03 §7): framework-free slow verification.

Module map (mechanical split of the former slow.py, semantics unchanged):
  pipeline.py   single-candidate pipeline (build -> sec synthesis -> DEM ->
                decode -> evidence) + subprocess isolation boundary
  pool.py       work-stealing job pool + queue-level driver
  dem_cache.py  DEM bundle pickled once per (candidate, noise point)
  sampling.py   shard partition/seed/merge (anchor-p racing lands here, 03 §8)
  evidence.py   d_circ_ext / DEM-ISD / slope merging, degraded_* flags
  writeback.py  slow_results / rankings / CSV rows
  feedback.py   prompt-facing slow feedback + parent selection
  baseline.py   resident BB anchors
  worker.py     subprocess entry (python -m omniqec_agentic.evaluation.worker)
  common.py     schema version, config deserialization, small helpers

A4/A5 LER combination stays in verification/ler.py (03 §7 + 2026-08-16 对齐); decoder
session classes stay in verification/decoders.py (single source, shared with
the qualification harness)."""

from .common import SLOW_SCHEMA_VERSION, verification_from_dict
from .baseline import (
    BASELINE_BB144_PROGRAM_ID,
    BASELINE_BB144_SPEC,
    BASELINE_BB_PROGRAM_ID,
    BASELINE_BB_SPEC,
    select_resident_baseline,
)
from .evidence import (
    VER1_D_CIRC_MIN_ERRORS,
    VER1_D_CIRC_MIN_POINTS,
    VER1_LER_SLOPE_OFFSET,
)
from .feedback import build_slow_feedback, make_slow_aware_prompt_context
from .pipeline import evaluate_slow_spec, evaluate_slow_spec_inprocess
from .pool import evaluate_promotion_queue, load_promotion_queue
from .sampling import RacingThresholds, anchor_p_for_profile
from .worker import (
    run_build_dem_request,
    run_distance_job_request,
    run_shard_dem_request,
)
from .writeback import write_slow_result

__all__ = [
    "BASELINE_BB144_PROGRAM_ID",
    "BASELINE_BB144_SPEC",
    "BASELINE_BB_PROGRAM_ID",
    "BASELINE_BB_SPEC",
    "SLOW_SCHEMA_VERSION",
    "VER1_D_CIRC_MIN_ERRORS",
    "VER1_D_CIRC_MIN_POINTS",
    "VER1_LER_SLOPE_OFFSET",
    "RacingThresholds",
    "anchor_p_for_profile",
    "build_slow_feedback",
    "evaluate_promotion_queue",
    "evaluate_slow_spec",
    "evaluate_slow_spec_inprocess",
    "load_promotion_queue",
    "make_slow_aware_prompt_context",
    "run_build_dem_request",
    "run_distance_job_request",
    "run_shard_dem_request",
    "select_resident_baseline",
    "verification_from_dict",
    "write_slow_result",
]
