"""Work-stealing job pool + queue-level driver (03 §7, semantics unchanged
from slow.py). Every (candidate, p, decoder, basis, shot-shard) decode job and
every distance job runs as its own hard-timeout subprocess consumed by one
global pool; SEC synthesis is hoisted once-per-candidate via the shared
synth-cache directory (02 §7.1)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omniqec_agentic import ranking
from omniqec_agentic.config import (
    DECODE_BUDGET_SCALE_COLS,
    TELESCOPING_COST_S_PER_COLUMN_ITER,
    TELESCOPING_GARI_COLUMN_FACTOR,
    DecoderName,
    HardwareTarget,
    RacingConfig,
    VerificationConfig,
)
from omniqec_agentic.construction import build_compiled_code, compiled_code_metadata
from omniqec_agentic.results_csv import write_results_csv
from omniqec_agentic.verification.circuit import select_memory_rounds
from omniqec_agentic.verification.fast import evaluate_proxy
from omniqec_agentic.verification.ler import combine_basis_ler_results
from omniqec_agentic.verification.structural import structural_check_spec

from .baseline import with_resident_baseline
from .common import (
    SLOW_SCHEMA_VERSION,
    decoder_name,
    hardware_value,
    int_or_none,
    noise_result_fields,
    record_to_slow_payload_input,
    resolve_deformation_mode,
    tail,
)
from .dem_cache import dem_bundle_path
from .evidence import (
    annotate_slow_checks,
    best_combined_result,
    distance_evidence_feedback,
    sec_ext_evidence,
    VER1_D_CIRC_MIN_ERRORS,
    VER1_D_CIRC_MIN_POINTS,
    VER1_LER_SLOPE_OFFSET,
)
from .feedback import build_slow_feedback
from .pipeline import WORKER_MODULE, evaluate_slow_spec
from .worker_pool import PersistentWorkerPool, persistent_workers_enabled
from .sampling import (
    join_failure_reasons,
    RacingThresholds,
    decoder_cost_decision,
    merge_shard_results,
    merge_subshard_results,
    racing_decision,
    racing_pilot_shards,
    racing_shard_stream,
    shard_partition,
)
from .writeback import (
    compact_slow_payload,
    has_decoder_attempt,
    ranking_rows,
    slow_csv_rows,
    slow_queue_status,
    summarize_evidence_status,
    write_slow_result,
)

_WORKER_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_worker_request(
    request: dict[str, Any],
    *,
    timeout_s: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run one worker request in an isolated subprocess.

    Library-internal thread pools are capped to one thread per worker: the job
    pool provides the parallelism, and N workers x M BLAS threads oversubscribes
    large machines.  ``None`` or a non-positive timeout disables the
    application-level timeout while retaining process isolation.
    """

    env = dict(os.environ)
    env.update(_WORKER_THREAD_CAPS)
    process = subprocess.Popen(
        [sys.executable, "-m", WORKER_MODULE],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
        env=env,
    )
    communicate_timeout = (
        None if timeout_s is None or float(timeout_s) <= 0.0 else float(timeout_s)
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps(request, default=float),
            timeout=communicate_timeout,
        )
    except subprocess.TimeoutExpired:
        kill_process_group(process)
        return None, {
            "timeout": True,
            "timeout_s": communicate_timeout,
            "returncode": process.returncode,
        }
    parsed = parse_worker_json(stdout)
    return parsed, {
        "timeout": False,
        "returncode": process.returncode,
        "stderr_tail": tail(stderr),
        "stdout_tail": tail(stdout) if parsed is None else "",
    }


def release_shared_pickle(path: str | None, owner: int, owners: dict[str, set[int]]) -> bool:
    """Drop ``owner``'s claim on the DEM pickle at ``path`` and delete the file once
    nobody else holds one. Returns True when the file was removed.

    Two queue entries of the same code map to the same ``dem_bundle_path`` (family,
    candidate hash, noise hash): in the b144 Phase B final of 2026-10-13 the BB
    reference row and the LLM's own copy of that code ran side by side, the first to
    finish unlinked the pickle, and the other's remaining shards ended in
    ``dem_load_error`` -> ``measurement_blocked`` -> a strict-final FATAL_STOP.
    A path never registered in ``owners`` (older callers) is released unconditionally."""

    if not path:
        return False
    holders = owners.get(path)
    if holders is not None:
        holders.discard(owner)
        if holders:
            return False
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def decode_budget_scale(num_columns: int | None) -> float:
    """DEM-size factor of the decode budget (v3.6, docs/06 §1.4 / §3 B5):
    ``max(1, num_matrix_columns / DECODE_BUDGET_SCALE_COLS)``. ``None`` (size
    unknown, e.g. before the build landed) is factor 1."""

    if num_columns is None:
        return 1.0
    try:
        columns = float(num_columns)
    except (TypeError, ValueError):
        return 1.0
    if columns <= 0.0:
        return 1.0
    return max(1.0, columns / float(DECODE_BUDGET_SCALE_COLS))


def telescoping_worst_case_shot_s(
    verification: VerificationConfig, num_columns: int | None, stage: str = "search"
) -> float | None:
    """Worst-case wall time of ONE telescoping shot (docs/06 §1.4 / B5).

    The S3 stages are iteration-bounded, so their worst case (every stage
    exhausted, the shot ends NC) is a cost model on the GARI system size:
    ``TELESCOPING_COST_S_PER_COLUMN_ITER x (num_columns x
    TELESCOPING_GARI_COLUMN_FACTOR) x worst_case_iterations()``. ``stage`` is
    accepted for the job-config record only (no stage-dependent limit under
    the CPU protocol). None when the telescoping decoder is not among the
    decoders, the size is unknown or the model is disabled (constant <= 0).
    """

    decoders = tuple(verification.decoders)
    if DecoderName.TELESCOPING not in decoders and DecoderName.TELESCOPING.value not in [
        str(getattr(d, "value", d)) for d in decoders
    ]:
        return None
    if num_columns is None:
        return None
    try:
        columns = float(num_columns)
    except (TypeError, ValueError):
        return None
    if columns <= 0.0 or float(TELESCOPING_COST_S_PER_COLUMN_ITER) <= 0.0:
        return None
    telescoping = verification.decoder_build.telescoping
    iterations = int(telescoping.worst_case_iterations())
    gari_columns = columns * float(TELESCOPING_GARI_COLUMN_FACTOR)
    return float(TELESCOPING_COST_S_PER_COLUMN_ITER) * gari_columns * float(iterations)


def decode_budget_model(
    verification: VerificationConfig, shots: int, num_columns: int | None, stage: str = "search"
) -> dict[str, Any]:
    """The two watchdog terms of a decode shard, for job_config / audit."""

    worst = telescoping_worst_case_shot_s(verification, num_columns, stage)
    budget = float(getattr(verification, "decode_shot_budget_s", 0.0) or 0.0)
    return {
        "stage": str(stage),
        "num_matrix_columns": None if num_columns is None else int(num_columns),
        "decode_budget_scale": decode_budget_scale(num_columns),
        "flat_term_s": (
            budget * float(max(0, int(shots))) * decode_budget_scale(num_columns)
            if budget > 0.0 else None
        ),
        "telescoping_worst_case_shot_s": worst,
        "telescoping_term_s": None if worst is None else worst * float(max(0, int(shots))),
        "decode_shard_timeout_cap_s": decode_shard_timeout_cap_s(verification, stage),
        # kept for the 2026-09-24 records: the final-stage ceiling by its own name
        "final_decode_shard_timeout_cap_s": (
            decode_shard_timeout_cap_s(verification, "final") if str(stage) == "final" else None
        ),
        "cost_s_per_column_iter": float(TELESCOPING_COST_S_PER_COLUMN_ITER),
        "gari_column_factor": float(TELESCOPING_GARI_COLUMN_FACTOR),
    }


# Smallest per-chunk share of a shard-watchdog ceiling (2026-09-29): a chunk
# is a few shots, so the honest budget is seconds; 300 s absorbs load skew.
CHUNK_CAP_FLOOR_S = 300.0


def effective_decode_shard_timeout_s(
    verification: VerificationConfig,
    shots: int,
    num_columns: int | None = None,
    stage: str = "search",
    chunk_shots: int | None = None,
) -> float | None:
    """Decode-shard watchdog for a shard of ``shots`` shots (v3.4, v3.6).

    ``decode_shard_timeout_s`` is the FLOOR; when ``decode_shot_budget_s`` is
    positive the watchdog is ``max(floor, budget * shots * scale)`` so a
    shard's allowance scales with its work (1250 shots -> 10,000 s at
    8 s/shot; the final 200000/128 = 1562-shot shard -> 12,500 s) and, since
    v3.6, with the DEM size: ``scale = max(1, num_columns /
    DECODE_BUDGET_SCALE_COLS)`` where ``num_columns`` is the two-basis DEM's
    ``num_matrix_columns``.

    Review fix (2026-09-03): with the telescoping decoder the watchdog is
    ALSO at least ``shots x telescoping_worst_case_shot_s(...)`` -- the flat
    term (16.5 s/shot for a 124k-column Mitten DEM) is below a shot that
    exhausts S3-B/S3-C (~100-300 s measured on Mitten200), so it would have
    killed exactly the large codes the protocol sets out to measure. The
    watchdog now bounds what the protocol's own iteration counts allow; the
    real cost of a code is governed by the budget policy's shots.

    A non-positive budget keeps the flat floor alone; a non-positive result
    disables the watchdog (``None``), as before. ``num_columns=None`` keeps
    the v3.4 value exactly (no model term without a size).
    """

    floor = float(verification.decode_shard_timeout_s)
    budget = float(getattr(verification, "decode_shot_budget_s", 0.0) or 0.0)
    if budget > 0.0:
        timeout = max(
            floor,
            budget * float(max(0, int(shots))) * decode_budget_scale(num_columns),
        )
    else:
        timeout = floor
    worst = telescoping_worst_case_shot_s(verification, num_columns, stage)
    if worst is not None and budget > 0.0:
        timeout = max(timeout, worst * float(max(0, int(shots))))
    # 2026-09-24: the final stage may carry a hard ceiling (runtime key
    # final_decode_shard_timeout_cap_s). Without it a single non-converging
    # d=14-18 shard ran past the 24 h link on every attempt (DeepSeek
    # b144/b220/b260 finals), so the confirmation could never finish.
    cap = decode_shard_timeout_cap_s(verification, stage)
    if cap is not None:
        # 2026-09-29: the ceiling is a per-SHARD allowance. A split shard's
        # chunk (shot_range of ``chunk_shots`` of the shard's ``shots``) gets
        # its proportional share, never below CHUNK_CAP_FLOOR_S -- with the
        # full ceiling per 3-shot chunk (64 chunks x 128 shards x 2 bases) a
        # non-converging final candidate ran 48 h without a single timeout.
        share = 1.0
        if chunk_shots is not None and int(shots) > 0 and 0 < int(chunk_shots) < int(shots):
            share = float(chunk_shots) / float(shots)
        timeout = min(timeout, max(CHUNK_CAP_FLOOR_S, cap * share))
    return timeout if timeout > 0.0 else None


def decode_shard_timeout_cap_s(verification: VerificationConfig, stage: str) -> float | None:
    """The stage's runtime ceiling on the decode-shard watchdog, or None.

    ``final_decode_shard_timeout_cap_s`` (2026-09-24) for the fixed-T final,
    ``search_decode_shard_timeout_cap_s`` (2026-09-25) for the per-turn slow
    queue. Both are protocol RUNTIME keys (checkpoint.RESUME_RUNTIME_KEYS)."""

    key = {
        "final": "final_decode_shard_timeout_cap_s",
        "search": "search_decode_shard_timeout_cap_s",
    }.get(str(stage))
    cap = getattr(verification, key, None) if key else None
    if cap is None:
        return None
    return float(cap) if float(cap) > 0.0 else None


def effective_build_timeout_s(
    verification: VerificationConfig, n_data: int | None = None
) -> float | None:
    """Build-job watchdog (review fix, docs/05 §4 M1 cost).

    ``build_timeout_s`` is the floor. The twobasis SEC certification of a
    Mitten-class n=200 code schedules ~340k ISD iterations (~2.6 CPU-hours),
    so a flat hour turned it into a build timeout -> slow-failure strike ->
    exclusion. With the data-qubit count known the watchdog is at least the
    certification work model (``twobasis_rank.certification_work_model``:
    iterations from the policy plan x a size-scaled per-iteration cost with
    a 2x margin). None n_data / LRC policy / non-positive floor: unchanged.
    """

    floor = float(verification.build_timeout_s)
    if floor <= 0.0:
        return None
    if n_data is None:
        return floor
    policy = verification.sec_policy
    if policy is None or getattr(policy, "is_lrc", False):
        return floor
    try:
        from omniqec_agentic.sec.twobasis_rank import certification_work_model

        model = certification_work_model(policy, n_data=int(n_data))
    except Exception:
        return floor
    return max(floor, float(model["worst_case_s"]))


def worker_timeout_for(
    request: dict[str, Any], verification: VerificationConfig
) -> float | None:
    """Resolve the typed watchdog for one worker kind.

    A non-positive value disables only the application watchdog; the isolated
    subprocess boundary remains. Old artifacts are mapped onto these fields by
    ``verification_from_dict`` before reaching the pool. The decode-shard
    watchdog scales with the request's ``shots`` (v3.4, see
    ``effective_decode_shard_timeout_s``).
    """

    mode = str(request.get("mode") or "candidate")
    if mode == "build_dem":
        return effective_build_timeout_s(verification, request.get("n_data"))
    elif mode == "shard_dem":
        shot_range = request.get("shot_range")
        chunk_shots = None
        if isinstance(shot_range, (list, tuple)) and len(shot_range) == 2:
            try:
                chunk_shots = max(0, int(shot_range[1]) - int(shot_range[0]))
            except (TypeError, ValueError):
                chunk_shots = None
        return effective_decode_shard_timeout_s(
            verification,
            int(request.get("shots") or 0),
            request.get("num_matrix_columns"),
            str(request.get("stage") or "search"),
            chunk_shots,
        )
    elif mode == "distance_job":
        configured = float(verification.distance_timeout_s)
    else:
        configured = float(verification.circuit_timeout_s)
    return configured if configured > 0.0 else None


def run_worker_request_with_retries(
    request: dict[str, Any], *, verification: VerificationConfig
) -> dict[str, Any]:
    """Run one immutable request, replaying it in place on retryable events.

    The exact same mapping is passed on every attempt, so the pre-registered
    seed, decoder/basis indices and shard identity cannot drift. Two separate
    replay budgets apply (v3.4):

    * ``timeout_retries`` replays a watchdog timeout (frozen at 0 by the
      protocol: a deterministic decode that exceeded its per-shot budget
      replays identically, so the retry only burns the watchdog again);
    * ``transient_retries`` replays a worker that died without parseable JSON
      (killed, e.g. OOM SIGKILL -> ``subprocess_error``) or that lost its
      pickled DEM (parsed status ``dem_load_error``). It never applies to
      timeouts.

    Any other parsed worker status is returned immediately: replay is a
    recovery policy, not a way to hide deterministic worker errors. The
    ``job_execution`` record always carries the total attempts, both retry
    budgets and their usage, the effective ``timeout_s`` and the
    ``returncode`` of the LAST attempt.
    """

    timeout_s = worker_timeout_for(request, verification)
    timeout_limit = max(0, int(verification.timeout_retries))
    transient_limit = max(0, int(getattr(verification, "transient_retries", 0)))
    mode = str(request.get("mode") or "candidate")
    attempts = 0
    timeout_retries_used = 0
    transient_retries_used = 0
    while True:
        attempts += 1
        parsed, meta = run_worker_request(request, timeout_s=timeout_s)
        returncode = meta.get("returncode")
        execution = {
            "attempts": int(attempts),
            "retries_used": int(timeout_retries_used),
            "retry_limit": int(timeout_limit),
            "transient_retry_limit": int(transient_limit),
            "transient_retries_used": int(transient_retries_used),
            "timeout_s": timeout_s,
            "returncode": int(returncode) if isinstance(returncode, int) else None,
        }
        if parsed is not None:
            if (
                str(parsed.get("status") or "") == "dem_load_error"
                and transient_retries_used < transient_limit
            ):
                transient_retries_used += 1
                continue
            # v3.6 (M4): a worker that reports its peak RSS (resource
            # ru_maxrss, bytes) has it recorded next to the attempt bookkeeping;
            # a worker that does not report it leaves the key absent.
            rss = parsed.get("ru_maxrss")
            if isinstance(rss, (int, float)) and not isinstance(rss, bool) and rss >= 0:
                execution = {**execution, "ru_maxrss": int(rss)}
            return {**parsed, "job_execution": execution}
        if meta.get("timeout"):
            if timeout_retries_used < timeout_limit:
                timeout_retries_used += 1
                continue
            # v3.4.1: a watchdog expiry that FOLLOWED a transient replay (a
            # killed worker / lost DEM on an earlier attempt of this very
            # request) is not evidence of a deterministic per-shot-budget
            # overrun — the node was already misbehaving — so it carries the
            # DISTINCT status ``timeout_after_transient`` that the
            # decode_infeasible gate (``fixed_shot_timeout_only``) never
            # accepts; the candidate ends measurement_blocked (infrastructure
            # suspected). Only a pure first-attempt expiry is ``timeout``.
            # Distance jobs keep ``uncertified_timeout`` (auxiliary evidence,
            # never a shard cell); the reason records the replay in all modes.
            if mode == "distance_job":
                status = "uncertified_timeout"
            elif transient_retries_used > 0:
                status = "timeout_after_transient"
            else:
                status = "timeout"
            failure_reason = (
                f"{mode} worker exceeded timeout_s={timeout_s!r} on all "
                f"{attempts} attempts"
            )
            if transient_retries_used > 0:
                failure_reason += f" after {transient_retries_used} transient replay(s)"
            return {
                "status": status,
                "failure_reason": failure_reason,
                "job_execution": {**execution, "timed_out": True},
            }
        if transient_retries_used < transient_limit:
            transient_retries_used += 1
            continue
        return {
            "status": "subprocess_error",
            "failure_reason": str(
                meta.get("stderr_tail")
                or f"worker produced no parseable JSON (returncode={returncode})"
            )[-300:],
            "returncode": execution["returncode"],
            "job_execution": execution,
        }


def fixed_shot_completion_errors(
    result: dict[str, Any],
    *,
    required_bases: tuple[str, ...],
    expected_shots: int,
    expected_shards: int,
) -> list[str]:
    """Return every reason a fixed-shot decoder result is not publishable."""

    errors: list[str] = []
    basis_payloads = result.get("basis") if isinstance(result.get("basis"), dict) else {}
    for basis in required_bases:
        item = basis_payloads.get(basis)
        if not isinstance(item, dict):
            errors.append(f"missing_{basis}_basis")
            continue
        if str(item.get("status") or "") != "ok":
            errors.append(f"{basis}_status={item.get('status')!r}")
        shots = int(item.get("shots") or 0)
        if shots != int(expected_shots):
            errors.append(f"{basis}_shots={shots}!={int(expected_shots)}")
        shard_statuses = item.get("shard_statuses")
        if not isinstance(shard_statuses, list) or len(shard_statuses) != int(expected_shards):
            count = len(shard_statuses) if isinstance(shard_statuses, list) else 0
            errors.append(f"{basis}_shards={count}!={int(expected_shards)}")
        elif any(str(status) != "ok" for status in shard_statuses):
            errors.append(f"{basis}_shard_incomplete")
    combined = result.get("combined") if isinstance(result.get("combined"), dict) else {}
    if combined.get("per_logical_per_round") is None:
        errors.append("missing_combined_plpr")
    return errors


def fixed_shot_timeout_only(
    result: dict[str, Any], *, required_bases: tuple[str, ...]
) -> bool:
    """True when the ONLY defect of an incomplete fixed-shot result is a PURE
    first-attempt watchdog expiry (v3.4, tightened in v3.4.1): every required
    basis is present, at least one of its shard jobs actually returned
    (``shard_attempts`` is not all ``None``), and every non-``ok`` shard

    * has status exactly ``"timeout"`` — a shard whose expiry followed a
      transient replay carries the distinct ``"timeout_after_transient"``
      status (see ``run_worker_request_with_retries``) and is rejected here;
    * actually returned: its ``shard_attempts`` slot is not ``None``. The
      merge stamps ``"timeout"`` on a slot whose job never landed (e.g. the
      build failed, so no decode job was ever submitted); such placeholders
      are not evidence about the decoder.

    With at least one such shard the result is ``decode_infeasible`` — the
    decoder exceeded the frozen per-shot budget on a deterministic replay, a
    terminal finding about the code — otherwise it stays
    ``measurement_blocked`` (an infrastructure defect such as a killed worker,
    a lost DEM pickle or a job that never returned, which pauses the queue).
    A result without a ``shard_attempts`` record is never infeasible.
    """

    basis_payloads = result.get("basis") if isinstance(result.get("basis"), dict) else {}
    timeouts = 0
    for basis in required_bases:
        item = basis_payloads.get(basis)
        if not isinstance(item, dict):
            return False
        shard_statuses = item.get("shard_statuses")
        shard_attempts = item.get("shard_attempts")
        if not isinstance(shard_statuses, list) or not isinstance(shard_attempts, list):
            return False
        if len(shard_attempts) != len(shard_statuses):
            return False
        if not any(attempt is not None for attempt in shard_attempts):
            # No shard job of this basis ever returned (e.g. the build
            # failed): the "timeout" placeholders say nothing about decode.
            return False
        for status, attempt in zip(shard_statuses, shard_attempts):
            if str(status) == "ok":
                continue
            if str(status) != "timeout" or attempt is None:
                return False
            timeouts += 1
    return timeouts > 0


def decode_infeasible_failure_reason(
    results: list[dict[str, Any]],
    *,
    required_bases: tuple[str, ...],
    timeout_s: float | None,
) -> str:
    """Candidate-level reason for a ``decode_infeasible`` verdict: per basis,
    how many of the shards hit the (effective) decode watchdog."""

    infeasible = [
        result
        for result in results
        if isinstance(result, dict) and result.get("status") == "decode_infeasible"
    ]
    parts: list[str] = []
    for result in infeasible:
        basis_payloads = result.get("basis") if isinstance(result.get("basis"), dict) else {}
        prefix = ""
        if len(infeasible) > 1:
            prefix = f"{result.get('decoder')}@{result.get('noise_value')} "
        for basis in required_bases:
            item = basis_payloads.get(basis) if isinstance(basis_payloads.get(basis), dict) else {}
            statuses = item.get("shard_statuses") if isinstance(item.get("shard_statuses"), list) else []
            n_timeout = sum(1 for status in statuses if str(status) == "timeout")
            parts.append(f"{prefix}{basis}:{n_timeout}/{len(statuses)} shards")
    return (
        "decode exceeded the frozen per-shot budget: "
        + ", ".join(parts)
        + f" (timeout_s={timeout_s!r})"
    )


def count_incomplete_candidates(candidate_payloads: list[dict[str, Any]]) -> int:
    """Count non-terminal candidates whose measurement did not finish.

    ``schedule_catastrophic`` is a valid structural terminal finding, and so
    is ``decode_infeasible`` (v3.4: the decoder exceeded the frozen per-shot
    budget on a deterministic replay). Every other non-``ok`` status is
    incomplete evidence, including failures before a decoder was ever
    attempted (input/build/subprocess errors).
    """

    return sum(
        1
        for item in candidate_payloads
        if item.get("status") not in {"ok", "schedule_catastrophic", "decode_infeasible", "decoder_unsupported"}
        and not item.get("schedule_catastrophic")
        and not item.get("decode_infeasible")
        and not item.get("decoder_unsupported")
    )


def check_build_rounds(outcome: Any, expected_rounds: int) -> Any:
    """2026-10-27: a build whose X/Z circuits were emitted at a T other than
    the one this pool normalises the candidate's rates with (ctx["rounds"],
    from the verification's rounds policy) is turned into a ``build_error``
    with the reason, never decoded. Before the fix of that date the build
    worker ignored ``rounds_policy`` (fixed_Tc codes with d_hint > T_c ran at
    T = d_hint while the row reported T_c); this keeps any such mismatch
    loud. A consistent or non-ok outcome passes through unchanged."""

    if not isinstance(outcome, dict) or outcome.get("status") != "ok":
        return outcome
    if "rounds_x" not in outcome and "rounds_z" not in outcome:
        return outcome
    built = {"X": outcome.get("rounds_x"), "Z": outcome.get("rounds_z")}
    if all(value is not None and int(value) == int(expected_rounds) for value in built.values()):
        return outcome
    return {
        **outcome,
        "status": "build_error",
        "rounds_mismatch": True,
        "failure_reason": (
            f"rounds mismatch: circuits built at T_X={built['X']}, T_Z={built['Z']} "
            f"but the rounds policy gives T={int(expected_rounds)}"
        ),
    }


def _num_logicals(build_outcome: dict[str, Any]) -> int | None:
    """Fig.3 Phase B: the logical qubits of the measured circuit reported by
    the build worker (``num_logicals``: int or per-basis dict); None for a
    worker that does not report it (the code's k is used)."""

    raw = build_outcome.get("num_logicals")
    if raw is None:
        return None
    try:
        if isinstance(raw, dict):
            values = {int(v) for v in raw.values() if v is not None}
            if len(values) != 1:
                raise ValueError(f"num_logicals differs between bases: {raw}")
            return values.pop()
        return int(raw)
    except (TypeError, ValueError):
        return None


def _num_matrix_columns(build_outcome: dict[str, Any]) -> int | None:
    """DEM column count reported by a build_dem worker (v3.6 B5): an int, or
    a per-basis mapping (max over bases); ``None`` when not reported."""

    raw = build_outcome.get("num_matrix_columns")
    if raw is None:
        meta = build_outcome.get("meta")
        if isinstance(meta, dict):
            raw = {
                basis: (blob or {}).get("num_matrix_columns")
                for basis, blob in meta.items()
                if isinstance(blob, dict)
            }
    if isinstance(raw, dict):
        values = [int(v) for v in raw.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return max(values) if values else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    return None


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=20)
    except Exception:
        pass


def parse_worker_json(stdout: str | None) -> dict[str, Any] | None:
    for line in reversed((stdout or "").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def without_relaybp_parallel(verification: VerificationConfig) -> VerificationConfig:
    """Disable RelayBP's internal Rust threads when the job pool is wide:
    pool-level parallelism already saturates the cores."""

    import dataclasses

    return dataclasses.replace(
        verification,
        decoder_build=dataclasses.replace(
            verification.decoder_build,
            relaybp=dataclasses.replace(verification.decoder_build.relaybp, parallel=False),
        ),
    )


# v3.6 (M4): per-worker peak-RSS allowance used to cap the pool width by the
# memory actually available. 2 GiB is conservative for the telescoping decoder (GARI system)
# (measured < 0.5 GB per worker after the two-basis DEM shrink, docs/05 §4
# M4); the launcher can lower it (OMNIQEC_WORKER_RSS_BYTES) and can pin the
# memory the scheduler granted (OMNIQEC_TOTAL_MEM_BYTES) when the cgroup
# limit is not readable.
DEFAULT_WORKER_RSS_BYTES = 2 * 1024 ** 3
# Chunks per shard under persistent workers when OMNIQEC_SHARD_SPLIT is unset
# (a 250-shot racing shard -> ~4-shot chunks).
PERSISTENT_DEFAULT_SHARD_SPLIT = 64


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(float(str(raw).strip()))
    except ValueError:
        return None
    return value if value > 0 else None


def total_memory_bytes() -> int | None:
    """Memory available to this job: the smallest of OMNIQEC_TOTAL_MEM_BYTES,
    the cgroup (v2 / v1) limit and the physical RAM; ``None`` when nothing is
    readable (the cap is then skipped)."""

    candidates: list[int] = []
    override = _env_int("OMNIQEC_TOTAL_MEM_BYTES")
    if override is not None:
        candidates.append(override)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit():
            value = int(text)
            # cgroup v1 reports a huge sentinel when unlimited.
            if 0 < value < 1 << 60:
                candidates.append(value)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            candidates.append(int(pages) * int(page_size))
    except (AttributeError, OSError, ValueError):
        pass
    return min(candidates) if candidates else None


def worker_memory_cap(
    total_bytes: int | None = None, per_worker_bytes: int | None = None
) -> int | None:
    """floor(total / per_worker) -- the widest pool the memory allows (>= 1),
    or ``None`` when the total is unknown or the per-worker allowance is
    disabled (OMNIQEC_WORKER_RSS_BYTES=0)."""

    if per_worker_bytes is None:
        raw = os.environ.get("OMNIQEC_WORKER_RSS_BYTES")
        if raw is not None and str(raw).strip():
            try:
                per_worker_bytes = int(float(str(raw).strip()))
            except ValueError:
                per_worker_bytes = DEFAULT_WORKER_RSS_BYTES
        else:
            per_worker_bytes = DEFAULT_WORKER_RSS_BYTES
    if int(per_worker_bytes) <= 0:
        return None
    total = total_memory_bytes() if total_bytes is None else int(total_bytes)
    if total is None or total <= 0:
        return None
    return max(1, int(total) // int(per_worker_bytes))


def shard_split_factor(workers: int | None = None) -> int:
    """How many contiguous shot-range jobs each decode shard is cut into
    (shard splitting, 2026-09-05). ``OMNIQEC_SHARD_SPLIT`` (>= 1) wins;
    otherwise 1 (the unsplit v3.4-v3.6 layout).

    Why: a shard is one process decoding its ~1250 shots in sequence, and a
    BP-hostile candidate costs 5-7 CPU-s per shot, so its 32 shards run for
    hours while the other workers idle (b140 turn 1: 205 core-hours of decode
    on 63 workers, 3.2 h wall). Splitting each shard into ``split`` ranges
    lets every worker help: the parts are bit-identical to the unsplit shard
    (see ``estimate_circuit_logical_error_rate``) and are folded back into
    the 16 shard cells the protocol records (``merge_subshard_results``), so
    nothing the validator or the audit reads changes. Not a protocol key.
    """

    raw = os.environ.get("OMNIQEC_SHARD_SPLIT")
    if raw is None or not str(raw).strip():
        return 1
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"OMNIQEC_SHARD_SPLIT must be an integer >= 1, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"OMNIQEC_SHARD_SPLIT must be >= 1, got {value}")
    return value


def resolve_workers(workers: int) -> int:
    """Pool width: ``0`` = cpu_count - 1, else the requested count; v3.6 (M4)
    caps either by the memory-derived width ``floor(total_mem /
    per_worker_rss)`` so 127 workers x a multi-GB decoder cannot OOM-kill
    shards into ``measurement_blocked``. The cap never lowers a serial run
    below 1 and is disabled with OMNIQEC_WORKER_RSS_BYTES=0."""

    if int(workers) == 0:
        resolved = max(1, (os.cpu_count() or 2) - 1)
    else:
        resolved = max(1, int(workers))
    cap = worker_memory_cap()
    if cap is not None and resolved > cap:
        resolved = max(1, cap)
    return resolved


def load_promotion_queue(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    queue_path = Path(path)
    if not queue_path.exists():
        raise FileNotFoundError(str(queue_path))
    if queue_path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        records = data.get("queue", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("promotion queue must be a JSON list or an object with a queue list")
    records = [record for record in records if isinstance(record, dict)]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


# slow_run_config.json keys that describe WHERE/HOW MANY WORKERS a queue ran,
# not WHAT was measured: they may differ between two attempts of
# the same queue without changing what a candidate measurement means. The two
# shard-watchdog ceilings are protocol runtime keys as well
# (checkpoint.RESUME_RUNTIME_KEYS): a link may raise the ceiling and still
# reuse the candidates an earlier, capped attempt finished (2026-10-14).
_RUN_CONFIG_RUNTIME_KEYS = (
    "queue_path",
    "out_dir",
    "workers",
    "final_decode_shard_timeout_cap_s",
    "search_decode_shard_timeout_cap_s",
)


def _reusable_candidates(
    run_config: dict[str, Any],
    run_config_path: Path,
    candidate_dir: Path,
    records: list[dict[str, Any]],
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Candidates an EARLIER attempt of this very queue finished (2026-09-24,
    final-stage resume): index -> payload from ``candidates/<id>.json``.

    Reuse is allowed only when the previous ``slow_run_config.json`` equals
    the current one on every measurement key (seed, shots, shards, stage,
    verification, racing, ...) and the candidate file names the same
    program_id, the same spec and the index-derived seed the protocol froze
    for that queue position. Anything else is measured afresh."""

    from .common import safe_file_stem

    if not run_config_path.is_file():
        return {}
    try:
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(previous, dict):
        return {}

    def _measurement_keys(config: dict[str, Any]) -> dict[str, Any]:
        normalised = json.loads(json.dumps(config, sort_keys=True, ensure_ascii=False))
        return {k: v for k, v in normalised.items() if k not in _RUN_CONFIG_RUNTIME_KEYS}

    if _measurement_keys(previous) != _measurement_keys(run_config):
        return {}
    reusable: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        program_id = str(record.get("program_id") or f"promotion_{index + 1:04d}")
        path = candidate_dir / f"{safe_file_stem(program_id)}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or str(payload.get("program_id")) != program_id:
            continue
        if payload.get("status") in (None, "started"):
            continue
        if (payload.get("promotion_record") or {}).get("spec") != record.get("spec"):
            continue
        expected_seed = int(seed) + index * 1_000_000
        if payload.get("seed") is not None and int(payload["seed"]) != expected_seed:
            continue
        reusable[index] = payload
    return reusable


def evaluate_promotion_queue(
    *,
    queue_path: str | Path,
    out_dir: str | Path,
    verification: VerificationConfig | None = None,
    hardware: HardwareTarget | str | None = None,
    limit: int | None = None,
    seed: int = 0,
    batch_size: int = 1024,
    prefer_coloring: bool = True,
    include_baseline: bool = True,
    baseline_spec: dict[str, Any] | None = None,
    baseline_program_id: str | None = None,
    workers: int = 1,
    shot_shards: int = 1,
    deformation_mode: str | None = None,
    racing: RacingConfig | None = None,
    racing_thresholds: RacingThresholds | None = None,
    stage: str = "search",
    resume_candidates: bool = False,
) -> dict[str, Any]:
    """Run slow verification for selected promotion-queue records.

    When `include_baseline` is set, the BB [[72,12,6]] reference runs through
    the identical protocol as a resident comparator (skipped if an identical
    code is already in the queue). Baseline rows are flagged `is_baseline` and
    never become slow parents.

    `workers > 1` switches to a fine-grained job pool: every
    (candidate, p, decoder, basis, shot-shard) decode job and every distance
    job runs as its own hard-timeout subprocess, consumed by one global
    work-stealing pool — uneven job durations cannot strand cores. Seeds are
    index-derived, so a given (workers, shot_shards) protocol is deterministic
    regardless of worker count; `shot_shards=1` reproduces the serial path's
    numbers exactly for the deterministic beam decoder (LSD reproduces only to
    within ldpc's tie-break jitter; see worker.run_shard_dem_request).
    `workers=0` auto-sizes to cpu_count-1. `shot_shards` is part of the
    measurement protocol (changing it re-partitions the RNG).

    `stage` (v3.6) is "search" (per-turn slow stage) or "final" (fixed-T
    confirmation): every shard request carries it so the worker applies the
    stage's decode budgets; a queue record flagged `retry_at_final`
    runs its jobs with the final budgets whatever the stage."""

    verification = verification or VerificationConfig()
    stage = str(stage or "search")
    if stage not in ("search", "final"):
        raise ValueError(f"stage must be 'search' or 'final' (got {stage!r})")
    racing_mode = bool(racing and racing.enabled)
    if racing_mode:
        if resolve_workers(workers) <= 1:
            raise ValueError("racing needs the job pool (workers > 1); serial racing is not supported")
        if len(verification.noise.values) != 1:
            raise ValueError(
                "racing runs at the anchor point only (03 §8): configure a single "
                "noise value; the full p sweep belongs to the final pipeline"
            )
    out_path = Path(out_dir)
    candidate_dir = out_path / "candidates"
    out_path.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    records = load_promotion_queue(queue_path, limit=limit)
    if include_baseline:
        records = with_resident_baseline(
            records,
            baseline_spec=baseline_spec,
            baseline_program_id=baseline_program_id,
        )
    results_jsonl = out_path / "slow_results.jsonl"
    rankings_path = out_path / "rankings.json"
    results_csv_path = out_path / "slow_results.csv"
    distance_evidence_path = out_path / "distance_evidence.json"
    slow_feedback_path = out_path / "slow_feedback.json"
    slow_parent_specs_path = out_path / "slow_parent_specs.json"
    summary_path = out_path / "slow_summary.json"
    run_config_path = out_path / "slow_run_config.json"
    run_config: dict[str, Any] = {
        "schema": "slow_queue_run_config",
        "schema_version": SLOW_SCHEMA_VERSION,
        "queue_path": str(queue_path),
        "out_dir": str(out_path),
        "hardware": hardware_value(hardware),
        "limit": limit,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "prefer_coloring": bool(prefer_coloring),
        "workers": resolve_workers(workers),
        "shot_shards": max(1, int(shot_shards)),
        "deformation_mode": deformation_mode,
        "stage": stage,
        "racing": racing.to_dict() if racing is not None else None,
        "racing_thresholds": (
            racing_thresholds.to_dict() if racing_thresholds is not None else None
        ),
        "verification": verification.to_dict(),
        # runtime ceiling of this stage's shard watchdog (never in to_dict)
        "final_decode_shard_timeout_cap_s": (
            decode_shard_timeout_cap_s(verification, "final") if stage == "final" else None
        ),
        "search_decode_shard_timeout_cap_s": (
            decode_shard_timeout_cap_s(verification, "search") if stage == "search" else None
        ),
    }
    # 2026-09-24 final-stage resume: candidates an earlier attempt of THIS
    # queue finished are taken from candidates/<id>.json instead of being
    # decoded again (a 24 h link cannot always hold six d>=14 codes). The
    # previous run config must match on every measurement key first.
    preloaded: dict[int, dict[str, Any]] = {}
    if resume_candidates:
        preloaded = _reusable_candidates(run_config, run_config_path, candidate_dir, records, int(seed))
        if preloaded:
            print(
                f"  [slow] resume: reusing {len(preloaded)}/{len(records)} candidate(s) "
                f"measured by an earlier attempt of this queue",
                flush=True,
            )
    results_jsonl.write_text("", encoding="utf-8")
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def _persist_candidate(index: int, payload: dict[str, Any]) -> None:
        # Written as soon as the pool assembles the candidate, so a link
        # killed at walltime keeps it for the next attempt (resume above).
        from .common import safe_file_stem

        program_id = str(payload.get("program_id") or f"promotion_{index + 1:04d}")
        write_slow_result(payload, candidate_dir / f"{safe_file_stem(program_id)}.json")

    start = time.perf_counter()

    def _evaluate_one(index: int, record: dict[str, Any]) -> dict[str, Any]:
        program_id = str(record.get("program_id") or f"promotion_{index + 1:04d}")
        candidate_payload = record_to_slow_payload_input(record, program_id=program_id)
        if "error" in candidate_payload:
            return {
                "schema": "slow_verification_run",
                "schema_version": SLOW_SCHEMA_VERSION,
                "program_id": program_id,
                "status": "input_error",
                "hardware": hardware_value(hardware),
                "promotion_record": record,
                "failure_reason": candidate_payload["error"],
                "results": [],
            }
        return evaluate_slow_spec(
            program_id=program_id,
            spec=candidate_payload["spec"],
            verification=verification,
            hardware=hardware,
            seed=int(seed) + index * 1_000_000,
            batch_size=int(batch_size),
            prefer_coloring=prefer_coloring,
            promotion_record=record,
            deformation_mode=(
                resolve_deformation_mode(None, record)
                if record.get("deformation_mode") is not None or record.get("deformation") is not None
                else deformation_mode
            ),
            # v3.6: the serial path applies the SAME stage rule as the job
            # path (final stage, or a retry_at_final record -> final budgets).
            stage=("final" if (stage == "final" or bool(record.get("retry_at_final"))) else "search"),
        )

    workers = resolve_workers(workers)
    if workers > 1:
        candidate_payloads = evaluate_queue_with_jobs(
            records,
            verification=verification,
            hardware=hardware,
            seed=int(seed),
            batch_size=int(batch_size),
            prefer_coloring=prefer_coloring,
            workers=workers,
            shot_shards=max(1, int(shot_shards)),
            deformation_mode=deformation_mode,
            racing=racing if racing_mode else None,
            racing_thresholds=racing_thresholds,
            stage=stage,
            preloaded=preloaded or None,
            on_candidate_done=_persist_candidate if resume_candidates else None,
        )
    else:
        candidate_payloads = []
        for index, record in enumerate(records):
            if index in preloaded:
                candidate_payloads.append(preloaded[index])
                continue
            payload = _evaluate_one(index, record)
            if resume_candidates:
                _persist_candidate(index, payload)
            candidate_payloads.append(payload)

    candidate_summaries: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []
    for index, payload in enumerate(candidate_payloads):
        program_id = str(payload.get("program_id") or f"promotion_{index + 1:04d}")
        from .common import safe_file_stem

        candidate_path = candidate_dir / f"{safe_file_stem(program_id)}.json"
        write_slow_result(payload, candidate_path)
        with results_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_slow_payload(payload, candidate_path), ensure_ascii=False, sort_keys=True) + "\n")
        candidate_summaries.append(compact_slow_payload(payload, candidate_path))
        all_ranking_rows.extend(ranking_rows(payload, candidate_path))

    distance_evidence = distance_evidence_feedback(candidate_payloads)
    rankings = ranking.rank_slow_rows(all_ranking_rows)
    write_results_csv(results_csv_path, slow_csv_rows(rankings))
    rankings_path.write_text(
        json.dumps(
            {
                "schema": "slow_rankings",
                "schema_version": SLOW_SCHEMA_VERSION,
                "rank_by": "per_logical_per_round_with_per_logical_block_ci",
                "ci_z": float(verification.ler_ci_z),
                "rankings": rankings,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    distance_evidence_path.write_text(
        json.dumps(
            {
                "schema": "slow_distance_evidence",
                "schema_version": SLOW_SCHEMA_VERSION,
                "method": "exact_distance_isd_then_ler_slope_estimate",
                "exact_code_distance_status": summarize_evidence_status(distance_evidence, "exact_code_distance_status"),
                "exact_d_circ_status": summarize_evidence_status(distance_evidence, "exact_d_circ_status"),
                "d_circ_isd_upper_bound_status": summarize_evidence_status(distance_evidence, "d_circ_isd_upper_bound_status"),
                "d_circ_ext_status": summarize_evidence_status(distance_evidence, "d_circ_ext_status"),
                "ver1_reference": {
                    "fit": "log10(per_logical_per_round) vs log10(p), weighted by sqrt(total_failures)",
                    "slope_offset": VER1_LER_SLOPE_OFFSET,
                    "minimum_failures_per_point": VER1_D_CIRC_MIN_ERRORS,
                    "minimum_points": VER1_D_CIRC_MIN_POINTS,
                },
                "evidence": distance_evidence,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    slow_feedback = build_slow_feedback(
        queue_path=str(queue_path),
        hardware=hardware_value(hardware),
        rankings=rankings,
        candidate_payloads=candidate_payloads,
        distance_evidence=distance_evidence,
    )
    slow_feedback_path.write_text(
        json.dumps(slow_feedback, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    slow_parent_specs_path.write_text(
        json.dumps(slow_feedback["slow_parent_specs"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ok_candidate_count = sum(1 for item in candidate_summaries if item.get("status") == "ok")
    timeout_candidate_count = sum(1 for item in candidate_summaries if item.get("status") == "timeout")
    measurement_blocked_candidate_count = sum(
        1
        for item in candidate_summaries
        if item.get("status") == "measurement_blocked"
    )
    catastrophic_candidate_count = sum(
        1
        for item in candidate_payloads
        if item.get("schedule_catastrophic") or item.get("status") == "schedule_catastrophic"
    )
    decode_infeasible_candidate_count = sum(
        1
        for item in candidate_payloads
        if item.get("decode_infeasible") or item.get("status") == "decode_infeasible"
    )
    decoder_unsupported_candidate_count = sum(
        1
        for item in candidate_payloads
        if item.get("decoder_unsupported") or item.get("status") == "decoder_unsupported"
    )
    incomplete_candidate_count = count_incomplete_candidates(candidate_payloads)
    decoder_attempt_candidate_count = sum(1 for item in candidate_payloads if has_decoder_attempt(item))
    result_count = len(all_ranking_rows)
    ok_rankings = [row for row in rankings if row.get("status") == "ok"]
    ok_result_count = len(ok_rankings)
    summary = {
        "schema": "slow_queue_summary",
        "schema_version": SLOW_SCHEMA_VERSION,
        "status": slow_queue_status(
            record_count=len(records),
            timeout_candidate_count=timeout_candidate_count,
            decoder_attempt_candidate_count=decoder_attempt_candidate_count,
            ok_result_count=ok_result_count,
            catastrophic_candidate_count=catastrophic_candidate_count,
            measurement_blocked_candidate_count=measurement_blocked_candidate_count,
            incomplete_candidate_count=incomplete_candidate_count,
            decode_infeasible_candidate_count=decode_infeasible_candidate_count,
            decoder_unsupported_candidate_count=decoder_unsupported_candidate_count,
        ),
        "queue_path": str(queue_path),
        "out_dir": str(out_path),
        "hardware": hardware_value(hardware),
        "candidate_count": len(records),
        "ok_candidate_count": ok_candidate_count,
        "timeout_candidate_count": timeout_candidate_count,
        "measurement_blocked_candidate_count": measurement_blocked_candidate_count,
        "incomplete_candidate_count": incomplete_candidate_count,
        "catastrophic_candidate_count": catastrophic_candidate_count,
        "decode_infeasible_candidate_count": decode_infeasible_candidate_count,
        "decoder_unsupported_candidate_count": decoder_unsupported_candidate_count,
        "decoder_attempt_candidate_count": decoder_attempt_candidate_count,
        "result_count": result_count,
        "ok_result_count": ok_result_count,
        "runtime_s": time.perf_counter() - start,
        "artifact_paths": {
            "run_config": str(run_config_path),
            "results_jsonl": str(results_jsonl),
            "results_csv": str(results_csv_path),
            "rankings": str(rankings_path),
            "distance_evidence": str(distance_evidence_path),
            "slow_feedback": str(slow_feedback_path),
            "slow_parent_specs": str(slow_parent_specs_path),
            "summary": str(summary_path),
            "candidates_dir": str(candidate_dir),
        },
        "best": ok_rankings[0] if ok_rankings else None,
        "slow_parent_count": len(slow_feedback["slow_parent_specs"]),
        "candidates": candidate_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_queue_with_jobs(
    records: list[dict[str, Any]],
    *,
    verification: VerificationConfig,
    hardware: HardwareTarget | str | None,
    seed: int,
    batch_size: int,
    prefer_coloring: bool,
    workers: int,
    shot_shards: int,
    deformation_mode: str | None = None,
    racing: RacingConfig | None = None,
    racing_thresholds: RacingThresholds | None = None,
    stage: str = "search",
    preloaded: dict[int, dict[str, Any]] | None = None,
    on_candidate_done: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fine-grained execution: flatten all candidates into decode/distance
    jobs, run them through one global pool, reassemble candidate payloads with
    the exact schema of the serial path.

    With ``racing`` enabled (03 §8) the pre-registered batch-size shard stream
    replaces ``shot_shards``: each candidate consumes pilot shards first, then
    one batch shard per round until the pure ``racing_decision`` (static
    threshold snapshot, cumulative counts only) stops it. The decision runs on
    the PRIMARY decoder's per-logical-per-round (A5) CI; all configured
    decoders and both bases consume the same shards in lockstep. The resident
    baseline uses the same stream with thresholds disabled (straight to
    shots_confirm)."""

    from concurrent.futures import ThreadPoolExecutor

    stage = str(stage or "search")
    if stage not in ("search", "final"):
        raise ValueError(f"stage must be 'search' or 'final' (got {stage!r})")
    job_verification = without_relaybp_parallel(verification)
    decoder_names = [decoder_name(decoder) for decoder in verification.decoders]
    # ---- per-candidate prepare (cheap, in-process) -------------------------
    contexts: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        program_id = str(record.get("program_id") or f"promotion_{index + 1:04d}")
        if preloaded and index in preloaded:
            # 2026-09-24 final-stage resume: this candidate was measured by an
            # earlier attempt of the SAME queue (evaluate_promotion_queue
            # checked program_id / spec / seed); keep its index so the seeds
            # of the remaining candidates are the ones the protocol froze.
            contexts.append({"payload": preloaded[index], "ready": True, "reused": True})
            continue
        start = time.perf_counter()
        payload: dict[str, Any] = {
            "schema": "slow_verification_run",
            "schema_version": SLOW_SCHEMA_VERSION,
            "isolation": "subprocess_jobs",
            "program_id": program_id,
            "status": "started",
            "seed": int(seed) + index * 1_000_000,
            "hardware": hardware_value(hardware),
            "verification": verification.to_dict(),
            "promotion_record": record,
            "results": [],
        }
        candidate_input = record_to_slow_payload_input(record, program_id=program_id)
        if "error" in candidate_input:
            payload["status"] = "input_error"
            payload["failure_reason"] = candidate_input["error"]
            payload["runtime_s"] = time.perf_counter() - start
            contexts.append({"payload": payload, "ready": True})
            continue
        spec = candidate_input["spec"]
        candidate_deformation = (
            resolve_deformation_mode(None, record)
            if record.get("deformation_mode") is not None or record.get("deformation") is not None
            else deformation_mode
        )
        payload["spec"] = spec
        payload["deformation_mode"] = candidate_deformation or "none"
        structural = structural_check_spec(
            program_id,
            spec,
            require_positive_k=True,
            deformation_mode=candidate_deformation,
        )
        # ONE proxy protocol source (audit fix #2): same frozen
        # seeds/effort/algorithm as the search phase.
        fast = evaluate_proxy(
            spec,
            iters=int(verification.structural.proxy_iters_per_seed),
            seeds=tuple(verification.structural.proxy_seeds),
            algorithm_id=str(verification.structural.proxy_algorithm_id),
            dsqrtn_floor=float(verification.dsqrtn_floor),
            deformation_mode=candidate_deformation,
        )
        payload["structural"] = structural.to_dict()
        payload["fast"] = fast
        if not structural.valid:
            payload["status"] = "structural_reject"
            payload["runtime_s"] = time.perf_counter() - start
            contexts.append({"payload": payload, "ready": True})
            continue
        try:
            compiled = build_compiled_code(
                spec,
                program_id=program_id,
                deformation_mode=candidate_deformation,
            )
            payload["compiled_code"] = compiled_code_metadata(compiled)
            payload["matrices"] = {
                "hx": np.asarray(compiled.hx, dtype=np.uint8).astype(int).tolist(),
                "hz": np.asarray(compiled.hz, dtype=np.uint8).astype(int).tolist(),
            }
        except Exception as exc:
            payload["status"] = "build_error"
            payload["failure_reason"] = str(exc)
            payload["runtime_s"] = time.perf_counter() - start
            contexts.append({"payload": payload, "ready": True})
            continue
        d_hint = int_or_none(fast.get("d")) if isinstance(fast, dict) else None
        # v3.6 (B5): a candidate the graph flagged for its one final-stage
        # retry (decode_infeasible in search) runs with the final budgets --
        # the stage tag reaches every shard request of this candidate.
        candidate_stage = (
            "final" if (stage == "final" or bool(record.get("retry_at_final"))) else "search"
        )
        payload["stage"] = candidate_stage
        contexts.append(
            {
                "payload": payload,
                "ready": False,
                "spec": spec,
                "stage": candidate_stage,
                "num_matrix_columns": None,
                # Review fix: the build watchdog scales with the SEC
                # certification work, which the data-qubit count predicts.
                "n_data": int(np.asarray(compiled.hx).shape[1]),
                "family": compiled.family,
                "candidate_hash": str(structural.code_hash or record.get("code_hash") or program_id),
                "d_hint": d_hint,
                "k": structural.k or 0,
                "rounds": select_memory_rounds(
                    rounds=verification.rounds, d_hint=d_hint,
                    policy=str(getattr(verification, "rounds_policy", "max_Tc_d")),
                ),
                "candidate_seed": int(seed) + index * 1_000_000,
                "deformation_mode": candidate_deformation,
                "prepare_s": time.perf_counter() - start,
            }
        )

    # ---- interleaved pool: EVERY candidate's jobs in one pool --------------
    # Per candidate/noise-point: ONE build job (build the DEM once -> pickle),
    # plus ONE distance
    # job (ISD ~90s, single-core); when the build lands, its decode jobs are
    # submitted, split per (decoder, basis, shard) -> each loads the pickle and
    # runs ~900s (well under circuit_timeout_s), no rebuild and no 6-decoder
    # bundle (the old build-once-per-shard bundle overran the 1h timeout on big
    # codes; see PROGRESS L6). One candidate's slow distance/relaybp jobs overlap
    # other candidates' decode, so the cores stay saturated. Candidates are
    # assembled + logged ([slow]) as their last job lands, so results appear one
    # by one. Bit-identical to the legacy run_shard_request path (same seeds +
    # exact pickled probs; verify_pkl.py).
    from concurrent.futures import FIRST_COMPLETED, wait

    verification_dict = job_verification.to_dict()
    first_noise_value = float(verification.noise.values[0])
    racing_mode = bool(racing and racing.enabled)
    # Speculative prefetch (2026-08-28): racing decisions are a pure function
    # of the consumed prefix and per-shard seeds are index-derived
    # (sampling.py), so shards beyond the next decision boundary may decode
    # ahead without changing any recorded number — the decision loop below
    # still walks prefixes strictly in order, and the stop point truncates
    # everything past it at assembly. Lookahead 1 reproduces the original
    # lockstep schedule exactly; the window ramps up as a candidate survives
    # rounds, so early-eliminated candidates waste at most a shard or two.
    race_lookahead = 1
    if racing_mode:
        try:
            race_lookahead = max(1, int(os.environ.get("OMNIQEC_RACE_LOOKAHEAD", "128")))
        except ValueError:
            race_lookahead = 128
        print(f"[race] speculative prefetch lookahead={race_lookahead}", flush=True)
    if racing_mode:
        race_shard_shots = racing_shard_stream(racing)
        pilot_shards = racing_pilot_shards(racing)
        # 2026-11-07: pilot shards in flight ahead of the next decision. The whole pilot used to go out at once
        # (the default, unchanged); a candidate that reaches the max-errors bar on its first shard -- nearly every
        # CNOT candidate of the no-LLM baselines -- then decoded the rest of the pilot for nothing. A smaller window
        # trades that waste for pilot latency. Recorded numbers are the same for every value (decisions consume the
        # prefix in order, seeds are index-derived); only the timing-based decoder-cost cap can see the schedule.
        pilot_window = pilot_shards
        try:
            pilot_window = max(1, min(pilot_shards, int(os.environ.get("OMNIQEC_RACE_PILOT_WINDOW", pilot_shards))))
        except ValueError:
            pilot_window = pilot_shards
        if not bool(getattr(racing, "max_errors_in_pilot", False)):
            # 2026-11-10: a policy that decides nothing inside the pilot (the paper policy, v1/v2) needs the
            # whole pilot in flight -- a smaller window would wait forever for a decision boundary whose
            # shards were never submitted, and could not save a shard anyway.
            pilot_window = pilot_shards
        if pilot_window != pilot_shards:
            print(f"[race] pilot window {pilot_window} of {pilot_shards} shards", flush=True)
        # Survivors stop at shots_confirm, so in the common no-timeout march
        # no decision boundary past ceil(confirm/batch) is ever reached —
        # speculating beyond it is pure waste. Only when decoded shots lag
        # (failed/timed-out shards push boundaries past confirm_shards) does
        # the window open past this cap.
        confirm_shards = -(-int(racing.shots_confirm) // int(racing.batch))
        n_shards = len(race_shard_shots)
        shard_shots = race_shard_shots  # request sizing below
        shard_max_errors = None  # caps are enforced at round boundaries
    else:
        n_shards = shot_shards
        shard_shots = shard_partition(int(verification.shots), shot_shards)
        shard_max_errors = (
            None
            if verification.max_errors is None
            else max(1, -(-int(verification.max_errors) // shot_shards))
        )
    # v3.4: the decode-shard watchdog scales with the shard's shots; the
    # largest shard of this run defines the value recorded in job_config and
    # quoted by a decode_infeasible verdict.
    run_shard_timeout_s = effective_decode_shard_timeout_s(
        job_verification, max(shard_shots) if shard_shots else 0, None, stage
    )
    # Shard splitting (2026-09-05; racing mode too since 2026-09-11): a
    # shard's shot stream is decoded as shard_split contiguous shot-range
    # jobs and folded back into the one shard cell the protocol records
    # (bit-identical). Racing consumes whole shards at its decision
    # boundaries, so a split shard is folded the moment its last part lands
    # (below) and only then counts as landed for the race. Without this a
    # 250-shot pilot shard of a 3 s/shot code kept ONE worker busy for 13
    # minutes while 111 idled. Each part keeps the FULL shard's watchdog
    # (request["shots"] stays the shard's shots), so a split never produces
    # a timeout the unsplit shard would not have.
    shard_split = shard_split_factor(workers)
    # Persistent decode workers (2026-09-14, worker_pool.py): shot-range
    # chunks go to long-lived ``--serve`` processes instead of one
    # subprocess per part, so the split can be much finer (default 64 ->
    # ~4-shot chunks of a 250-shot shard) and a heavy-tailed shot no longer
    # holds a whole 31-shot part while the node idles. Bit-identical: the
    # chunks decode the same shot indices with the same per-shot seeds and
    # are folded back into the same shard cells. Launcher knob
    # OMNIQEC_PERSISTENT_WORKERS=1; OMNIQEC_SHARD_SPLIT still overrides.
    use_persistent = persistent_workers_enabled()
    if use_persistent and not str(os.environ.get("OMNIQEC_SHARD_SPLIT", "")).strip():
        shard_split = PERSISTENT_DEFAULT_SHARD_SPLIT
    if shard_split > 1:
        print(f"[pool] shard splitting: {shard_split} shot-range jobs per shard", flush=True)

    def _run_job(request: dict[str, Any]) -> dict[str, Any]:
        return run_worker_request_with_retries(
            request,
            verification=job_verification,
        )

    # 2026-09-26 candidate-level abort. A capped watchdog (search_/final_
    # decode_shard_timeout_cap_s) bounds ONE request, but a shard is split
    # into up to 64 chunks and a candidate has 2 x 16..128 shards: a code
    # whose decoder does not converge burned its watchdog on every chunk,
    # ~8 h (search) to ~64 h (final) per candidate, so the ceiling never
    # ended the candidate. Now the FIRST capped timeout of a candidate aborts
    # it: its queued chunks resolve at once as pure ``timeout`` outcomes
    # (persistent pool: cancel_where; thread pool: the guard below), racing
    # stops with ``shard_timeout_cap`` and the assembly hands out the same
    # decode_infeasible verdict a v3.4 watchdog timeout gets.
    aborted_ctx: set[int] = set()

    def _abort_outcome(ci: int, timeout_s: float | None) -> dict[str, Any]:
        return {
            "status": "timeout",
            "timeout": True,
            "failure_reason": (
                "cancelled: the candidate's first capped watchdog timeout aborted its "
                "remaining shard chunks (decode_shard_timeout_cap)"
            ),
            "returncode": None,
            "job_execution": {
                "attempts": 1, "retries_used": 0, "retry_limit": 0,
                "transient_retries_used": 0, "timeout": True,
                "timeout_s": timeout_s, "returncode": None, "cancelled": True,
            },
        }

    def _run_job_for_ctx(ci: int, request: dict[str, Any]) -> dict[str, Any]:
        if ci in aborted_ctx:
            return _abort_outcome(ci, worker_timeout_for(request, job_verification))
        return _run_job(request)

    # Build the DEM ONCE per (candidate, noise point) -> pickle; every decode job loads it
    # and runs a single (decoder, basis, shard) unit, so each job stays small
    # (well under circuit_timeout_s) while the expensive build happens once. The
    # GLOBAL (decoder, basis) indices are passed explicitly so the per-unit seed
    # matches the build-per-job path exactly (bit-identical for same seed/shards).
    # The dem_dir doubles as the cross-process SEC synthesis cache: the first
    # job to need a candidate's ranked schedule writes it, the rest load it.
    dem_dir = tempfile.mkdtemp(prefix="omniqec_dem_")
    # basis_index is the GLOBAL position (X=0, Z=1) so the per-unit seed is the
    # same whether a run decodes one basis or both -> a Z-only run's Z results
    # are bit-identical to a both-basis run's Z results.
    _BASIS_INDEX = {"X": 0, "Z": 1}
    bases = tuple(verification.bases)
    unit_index = [
        {"decoder": d.value, "basis": basis, "decoder_index": di, "basis_index": _BASIS_INDEX[basis]}
        for di, d in enumerate(decoder_names)
        for basis in bases
    ]

    pending: dict[int, int] = {}
    matrices_paths: dict[tuple[int, int], str] = {}
    # 2026-10-13: contexts that share a DEM pickle (the same code queued twice, e.g.
    # the BB reference row and the LLM's own copy of it in a final queue) each hold
    # a claim on the path; the file goes only when the last claim is released.
    pickle_owners: dict[str, set[int]] = {}
    for ctx_index, ctx in enumerate(contexts):
        if ctx["ready"]:
            continue
        if racing_mode:
            record = ctx["payload"].get("promotion_record") or {}
            ctx["race"] = {
                "consumed": 0,
                "outstanding": 0,
                "submitted": 0,
                "landed": {},
                # v3 (max_errors_in_pilot): decide at every landed shard from
                # the first one; v1/v2 decide first at the pilot boundary.
                "next_decision_at": 1 if bool(getattr(racing, "max_errors_in_pilot", False)) else pilot_shards,
                "done": False,
                "stopped_by": None,
                "confirmed": False,
                "stages": [],
                "is_baseline": bool(record.get("is_baseline")),
            }
        # outstanding jobs before assembly = 1 distance + one build per noise point; each
        # build, when it lands, adds its own decode jobs to this count.
        pending[ctx_index] = 1 + len(verification.noise.values)
        for noise_index in range(len(verification.noise.values)):
            noise_hash = verification.noise.point(noise_index).config_hash
            matrices_paths[(ctx_index, noise_index)] = dem_bundle_path(
                dem_dir,
                family=ctx["family"],
                candidate_hash=ctx["candidate_hash"],
                noise_hash=noise_hash,
                noise_index=noise_index,
            )
            pickle_owners.setdefault(matrices_paths[(ctx_index, noise_index)], set()).add(ctx_index)

    by_ctx: dict[int, dict[str, Any]] = {
        i: {"distance": None, "decode": {}, "decode_parts": {}, "decode_parts_landed": {}, "job_failures": []}
        for i in pending
    }

    def _assemble(ctx_index: int) -> None:
        ctx = contexts[ctx_index]
        payload = ctx["payload"]
        slot = by_ctx[ctx_index]
        distance = slot["distance"] if isinstance(slot["distance"], dict) else {}
        exact_distance = {
            "schema": "exact_distance_evidence",
            "schema_version": SLOW_SCHEMA_VERSION,
            "code": distance.get("code")
            or {"schema": "exact_code_distance_evidence", "status": "not_available", "d_exact": None, "certified": False},
            "circuit": distance.get("circuit")
            or {"schema": "exact_circuit_distance_evidence", "status": "not_available", "d_circ_exact": None, "certified": False},
            "circuit_isd": distance.get("circuit_isd")
            or {"schema": "isd_dem_circuit_distance_evidence", "status": "not_available", "d_circ_isd_upper_bound": None, "certified": False},
            "circuit_ext": distance.get("circuit_ext") or sec_ext_evidence(None),
        }
        # A failed distance job (e.g. circuit_error) carries the diagnostic
        # the sub-dicts don't; surface it instead of dropping it.
        if distance.get("status") not in (None, "ok"):
            exact_distance["status"] = distance.get("status")
            if distance.get("failure_reason"):
                exact_distance["failure_reason"] = distance.get("failure_reason")
        runtime = float(ctx.get("prepare_s") or 0.0)
        race = ctx.get("race")
        # Shard splitting: fold every shard's parts into the single cell the
        # per-basis merge below expects (bit-identical to an unsplit shard).
        for key, per_shard in slot["decode_parts"].items():
            cells = slot["decode"].setdefault(key, [None] * n_shards)
            for shard_index, parts in per_shard.items():
                cells[int(shard_index)] = merge_subshard_results(parts, decoder=key[1])
        slot["decode_parts"] = {}
        # Racing: only the consumed prefix of the pre-registered stream exists;
        # merging the full stream would misread unconsumed shards as timeouts.
        merge_upto = (race["consumed"] if race else n_shards) or n_shards
        for noise_index, _noise_value in enumerate(verification.noise.values):
            noise_point = verification.noise.point(noise_index)
            for decoder in decoder_names:
                basis_payloads = {
                    basis: merge_shard_results(
                        slot["decode"].get((noise_index, decoder.value, basis), [None] * n_shards)[:merge_upto],
                        decoder=decoder.value,
                    )
                    for basis in bases
                }
                runtime += sum(float(basis_payloads[b].get("runtime_s") or 0.0) for b in bases)
                decoder_payload = {
                    "status": "started",
                    **noise_result_fields(noise_point),
                    "decoder": decoder.value,
                    "sampling_source": "noisy_circuit",
                    "rounds": int(ctx["rounds"]),
                    "basis": basis_payloads,
                    "combined": combine_basis_ler_results(
                        basis_payloads,
                        rounds=int(ctx["rounds"]),
                        k=int(ctx.get("k_eff") or ctx["k"]),
                        ci_z=float(verification.ler_ci_z),
                    ),
                }
                fixed_shot_mode = race is None and verification.max_errors is None
                if fixed_shot_mode:
                    completion_errors = fixed_shot_completion_errors(
                        decoder_payload,
                        required_bases=bases,
                        expected_shots=int(verification.shots),
                        expected_shards=int(n_shards),
                    )
                    decoder_payload["measurement_complete"] = not completion_errors
                    if completion_errors:
                        # Preserve partial counts for infrastructure diagnosis,
                        # but never expose a censored point estimate as a
                        # publishable/rankable measurement.
                        decoder_payload["partial_combined"] = decoder_payload["combined"]
                        decoder_payload["combined"] = {
                            "block_ler": None,
                            "per_logical_per_round": None,
                            "failure_reason": "fixed_shot_measurement_incomplete",
                        }
                        # v3.4: when the ONLY defect is watchdog timeouts the
                        # decoder exceeded the frozen per-shot budget — a
                        # terminal finding (decode_infeasible), not an
                        # infrastructure block worth re-queueing. v3.4.1: only
                        # PURE first-attempt expiries of jobs that actually
                        # returned count; a timeout_after_transient shard or a
                        # never-returned slot (failed build) keeps the result
                        # measurement_blocked, consistent with the
                        # candidate-level build_error priority below.
                        decoder_payload["status"] = (
                            "decode_infeasible"
                            if fixed_shot_timeout_only(decoder_payload, required_bases=bases)
                            else "measurement_blocked"
                        )
                        decoder_payload["failure_reason"] = join_failure_reasons(list(completion_errors))
                    else:
                        decoder_payload["status"] = "ok"
                else:
                    decoder_payload["status"] = (
                        "ok"
                        if decoder_payload["combined"].get("block_ler") is not None
                        else "incomplete"
                    )
                if race is not None:
                    decoder_payload["stopped_by"] = race["stopped_by"]
                payload["results"].append(decoder_payload)
        if race is not None:
            payload["racing"] = {
                "stopped_by": race["stopped_by"],
                "confirmed": race["confirmed"],
                "shards_consumed": race["consumed"],
                "shard_shots": int(racing.batch),
                # Actual decoded shots (min across bases); the requested
                # stream prefix is kept separately for diagnosis.
                "shots_per_basis": race.get(
                    "shots_actual", sum(race_shard_shots[: race["consumed"]])
                ),
                "shots_requested": sum(race_shard_shots[: race["consumed"]]),
                "is_baseline_arm": race["is_baseline"],
                "thresholds": (
                    racing_thresholds.to_dict() if racing_thresholds is not None else None
                ),
                "stages": race["stages"],
            }
        payload["exact_distance"] = exact_distance
        annotate_slow_checks(payload, verification=verification)
        all_measurements_complete = bool(payload["results"]) and all(
            result.get("status") == "ok" for result in payload["results"]
        )
        measurement_blocked = any(
            result.get("status") == "measurement_blocked"
            for result in payload["results"]
        )
        decode_infeasible = any(
            result.get("status") == "decode_infeasible"
            for result in payload["results"]
        )
        build_error = slot.get("build_error")
        # v3.2 §4.4: either job (build or distance) may be the one that hits
        # the catastrophic verdict; surface it as a candidate-level terminal
        # state rather than a decode failure.
        catastrophic = bool(
            (isinstance(build_error, dict) and build_error.get("schedule_catastrophic"))
            or exact_distance.get("circuit_ext", {}).get("schedule_catastrophic")
        )
        if catastrophic:
            payload["schedule_catastrophic"] = True
        if catastrophic:
            payload["status"] = "schedule_catastrophic"
            reason = (
                (build_error or {}).get("failure_reason")
                if isinstance(build_error, dict)
                else None
            ) or exact_distance.get("failure_reason")
            if reason:
                payload["failure_reason"] = str(reason)
        elif isinstance(build_error, dict) and (
            build_error.get("decoder_unsupported") or build_error.get("status") == "decoder_unsupported"
        ):
            # v3.6.1: the decoder cannot be built for THIS candidate's DEM
            # (identical detector support, different logical support = a
            # circuit-distance-2 code). A terminal candidate verdict like
            # schedule_catastrophic / decode_infeasible: recorded, excluded,
            # never a measurement_blocked stop of the turn.
            payload["status"] = "decoder_unsupported"
            payload["decoder_unsupported"] = True
            payload["failure_reason"] = str(
                build_error.get("failure_reason") or "decoder cannot be built for this candidate's DEM"
            )
        elif isinstance(build_error, dict):
            # The build stage failed: surface its real status + reason instead
            # of letting the empty decode slots masquerade as shard timeouts.
            payload["status"] = str(build_error.get("status") or "build_error")
            if build_error.get("failure_reason"):
                payload["failure_reason"] = str(build_error["failure_reason"])
        elif race is not None and race.get("stopped_by") == "shard_timeout_cap":
            # 2026-09-26: the candidate's first capped watchdog timeout aborted
            # it (see aborted_ctx). Same terminal verdict as decoder_cost /
            # a v3.4 watchdog timeout: recorded, excluded, never re-queued.
            info = dict(slot.get("aborted") or {})
            payload["status"] = "decode_infeasible"
            payload["decode_infeasible"] = True
            payload["shard_timeout_cap"] = info
            payload["failure_reason"] = (
                f"shard_timeout_cap: a {info.get('stage', 'search')}-stage decode shard exceeded the "
                f"{float(info.get('cap_s') or 0.0):.0f} s watchdog ceiling (shard {info.get('shard')}, "
                f"part {info.get('part')}); the candidate was aborted"
            )
        elif race is not None and race.get("stopped_by") == "decoder_cost":
            # pilot10x_maxerr100_cost_v4: the decoder cost more per shot than
            # the protocol allows for a code of this size. A terminal finding
            # about the code like a v3.4 watchdog timeout: recorded, excluded,
            # never re-queued, never a FATAL_STOP; the consumed shards are kept
            # in the results for diagnosis but are not a measurement.
            info = dict(race.get("decoder_cost") or {})
            payload["status"] = "decode_infeasible"
            payload["decode_infeasible"] = True
            payload["decoder_cost"] = info
            payload["failure_reason"] = (
                f"decoder_cost: {float(info.get('cost_s_per_shot') or 0.0):.1f} CPU-s/shot over "
                f"{int(info.get('shots_per_basis') or 0)} shots/basis exceeds the protocol cap "
                f"{float(info.get('cost_cap_s_per_shot') or 0.0):.1f} CPU-s/shot for n={int(info.get('n_data') or 0)} "
                f"({racing.cost_cap_s_per_shot_per_qubit} s/shot/qubit, floor {racing.cost_cap_floor_s_per_shot} s/shot; "
                f"policy {racing.policy_id})"
            )
        elif all_measurements_complete:
            # Distance evidence is auxiliary. In particular, an
            # uncertified_timeout there cannot erase a complete X/Z PLPR.
            payload["status"] = "ok"
        elif measurement_blocked:
            payload["status"] = "measurement_blocked"
            payload["failure_reason"] = "one or more required fixed-shot cells are incomplete"
        elif decode_infeasible:
            # v3.4 candidate-level terminal state: every defect across the
            # required bases is a watchdog timeout, i.e. BPOSD exceeded the
            # frozen per-shot budget on a deterministic replay. Recorded as
            # a finding about the code (never re-queued, never a FATAL_STOP);
            # a mixed defect (any non-timeout shard, incl. v3.4.1's
            # timeout_after_transient) stays measurement_blocked.
            payload["status"] = "decode_infeasible"
            payload["decode_infeasible"] = True
            payload["failure_reason"] = decode_infeasible_failure_reason(
                payload["results"],
                required_bases=bases,
                timeout_s=ctx.get("shard_timeout_s", run_shard_timeout_s),
            )
        else:
            payload["status"] = "no_valid_results"
        payload["runtime_s"] = runtime
        if slot.get("job_failures"):
            payload["job_failures"] = list(slot["job_failures"])
        payload["job_config"] = {
            "workers": int(workers),
            "shard_split": int(shard_split),
            "worker_mode": "persistent" if use_persistent else "subprocess",
            "shot_shards": int(shot_shards),
            "build_timeout_s": float(verification.build_timeout_s),
            "decode_shard_timeout_s": float(verification.decode_shard_timeout_s),
            "decode_shot_budget_s": float(verification.decode_shot_budget_s),
            # v3.6 (B5): the candidate's own watchdog once its DEM size is
            # known (budget x shots x max(1, columns / DECODE_BUDGET_SCALE_COLS));
            # the run-level value below is the size-1 baseline.
            "effective_decode_shard_timeout_s": ctx.get("shard_timeout_s", run_shard_timeout_s),
            "base_decode_shard_timeout_s": run_shard_timeout_s,
            "num_matrix_columns": ctx.get("num_matrix_columns"),
            "decode_budget_scale": decode_budget_scale(ctx.get("num_matrix_columns")),
            # The ORIGINAL two-basis DEM column count (the build's
            # num_matrix_columns, max over bases) is the scaling basis -- not
            # the larger GARI matrix the telescoping decoder decodes on.
            "decode_budget_scale_basis": "two_basis_dem_num_matrix_columns",
            # Review fix: both watchdog terms (flat v3.4 term and the telescoping
            # worst-case shot model) and the build watchdog actually applied.
            "decode_budget_model": decode_budget_model(
                job_verification,
                max(shard_shots) if shard_shots else 0,
                ctx.get("num_matrix_columns"),
                ctx.get("stage", stage),
            ),
            "effective_build_timeout_s": effective_build_timeout_s(
                job_verification, ctx.get("n_data")
            ),
            "n_data": ctx.get("n_data"),
            "stage": ctx.get("stage", stage),
            "distance_timeout_s": float(verification.distance_timeout_s),
            "timeout_retries": int(verification.timeout_retries),
            "transient_retries": int(verification.transient_retries),
            "persistent_shard_ledger": False,
        }
        best = best_combined_result(payload)
        _fast = payload.get("fast") or {}
        _struct = payload.get("structural") or {}
        _n = _fast.get("n") or _struct.get("n_data")
        _k = _fast.get("k") or _struct.get("k")
        _d = _fast.get("d")
        _fam = (payload.get("spec") or {}).get("family") or _fast.get("family") or "?"
        print(
            f"  [slow] {payload['program_id']} [[{_n},{_k},{_d}]] {_fam} {payload['status']} "
            f"plpr={best.get('per_logical_per_round')} by={best.get('decoder')}",
            flush=True,
        )
        if on_candidate_done is not None:
            # Persist the finished candidate NOW (final-stage resume): a link
            # killed at walltime keeps every assembled candidate.
            on_candidate_done(ctx_index, payload)

    def _decode_jobs_for(
        ci: int,
        noise_index: int,
        build_outcome: dict[str, Any],
        shards: range | list[int] | None = None,
    ) -> list[dict[str, Any]]:
        rounds_x = int(build_outcome["rounds_x"])
        rounds_z = int(build_outcome["rounds_z"])
        base_seed = int(contexts[ci]["candidate_seed"]) + noise_index * 10_000
        out: list[dict[str, Any]] = []
        for shard in (range(n_shards) if shards is None else shards):
            if shard_shots[shard] <= 0:
                continue
            # shard splitting: contiguous shot ranges of this shard's stream;
            # a range that would be empty (fewer shots than parts) is skipped
            # and the shard has fewer parts
            part_sizes = [n for n in shard_partition(int(shard_shots[shard]), shard_split) if n > 0]
            ranges: list[tuple[int, int] | None] = [None] if len(part_sizes) <= 1 else []
            if len(part_sizes) > 1:
                cursor = 0
                for n in part_sizes:
                    ranges.append((cursor, cursor + n))
                    cursor += n
            for unit in unit_index:
                for part, shot_range in enumerate(ranges):
                    request = {
                        "mode": "shard_dem", "matrices_path": matrices_paths[(ci, noise_index)],
                        "noise_value": float(verification.noise.values[noise_index]),
                        "noise_index": noise_index,
                        "shard": shard, "shot_shards": n_shards,
                        "shots": int(shard_shots[shard]), "max_errors": shard_max_errors,
                        "base_seed": base_seed, "rounds_x": rounds_x, "rounds_z": rounds_z,
                        "units": [unit], "batch_size": int(batch_size),
                        "verification": verification_dict,
                        # v3.6: the worker records the decode stage (search | final) by
                        # stage; the DEM size scales this shard's watchdog.
                        "stage": contexts[ci].get("stage", stage),
                        "num_matrix_columns": contexts[ci].get("num_matrix_columns"),
                    }
                    if shot_range is not None:
                        request["shot_range"] = [int(shot_range[0]), int(shot_range[1])]
                    out.append({
                        "ctx": ci, "kind": "shard", "noise_index": noise_index, "shard": shard,
                        "part": part, "parts": len(ranges),
                        "request": request,
                    })
        return out

    def _cleanup_pickles(ci: int) -> None:
        for noise_index in range(len(verification.noise.values)):
            release_shared_pickle(matrices_paths.get((ci, noise_index)), ci, pickle_owners)

    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    ppool: PersistentWorkerPool | None = None
    if use_persistent:
        ppool = PersistentWorkerPool(
            max(1, int(workers)), verification=job_verification, timeout_for=worker_timeout_for
        )
        print(f"[pool] persistent decode workers: {ppool.n_workers} (chunks per shard: {shard_split})", flush=True)

    def _submit_decode(nj: dict[str, Any]):
        """Shard-decode jobs go to the persistent workers when enabled (keyed
        by candidate / noise point / unit so a worker keeps the same decoder
        state), everything else to the one-shot subprocess pool."""

        if ppool is None:
            return pool.submit(_run_job_for_ctx, nj["ctx"], nj["request"])
        units = tuple(
            (str(u.get("decoder")), str(u.get("basis"))) for u in (nj["request"].get("units") or [])
        )
        return ppool.submit(nj["request"], key=(nj["ctx"], nj.get("noise_index"), units))
    fut_to_job: dict[Any, dict[str, Any]] = {}
    pending_futs: set[Any] = set()

    def _build_request(ctx_index: int, noise_index: int) -> dict[str, Any]:
        ctx = contexts[ctx_index]
        return {
            "mode": "build_dem", "spec": ctx["spec"],
            "noise_value": float(verification.noise.values[noise_index]),
            "noise_index": noise_index,
            "d_hint": ctx["d_hint"], "prefer_coloring": bool(prefer_coloring),
            "deformation_mode": ctx.get("deformation_mode"),
            "matrices_path": matrices_paths[(ctx_index, noise_index)],
            "synth_cache_dir": dem_dir,
            "verification": verification_dict,
            # Review fix: sizes the build watchdog (certification work model).
            "n_data": ctx.get("n_data"),
        }

    def _distance_request(ctx_index: int) -> dict[str, Any]:
        ctx = contexts[ctx_index]
        return {
            "mode": "distance_job", "spec": ctx["spec"],
            "noise_value": first_noise_value, "noise_index": 0,
            "d_hint": ctx["d_hint"], "prefer_coloring": bool(prefer_coloring),
            "deformation_mode": ctx.get("deformation_mode"),
            "synth_cache_dir": dem_dir,
            "verification": verification_dict,
        }

    def _submit_after_first_build(ctx_index: int) -> None:
        """v3.6 (M5): the distance job and the remaining noise points' builds
        are submitted only once the noise-0 build has landed, so the SEC
        synthesis (the shared synth-cache entry) is computed ONCE per
        candidate instead of 1+N times concurrently (docs/05 §4 M5: a Mitten
        synthesis takes 8 min alone, 16-18 min under contention)."""

        ctx = contexts[ctx_index]
        if ctx.get("followups_submitted"):
            return
        ctx["followups_submitted"] = True
        nf = pool.submit(_run_job, _distance_request(ctx_index))
        fut_to_job[nf] = {"ctx": ctx_index, "kind": "distance"}
        pending_futs.add(nf)
        for noise_index in range(1, len(verification.noise.values)):
            nf = pool.submit(_run_job, _build_request(ctx_index, noise_index))
            fut_to_job[nf] = {"ctx": ctx_index, "kind": "build", "noise_index": noise_index}
            pending_futs.add(nf)

    try:
        # Seed the pool with each candidate's noise-0 build ONLY (v3.6 M5);
        # its distance job and the other noise points' builds follow when it
        # lands, and decode jobs are submitted as each build lands (dynamic,
        # so the pool never blocks on a barrier and the long single-core
        # distance jobs overlap other candidates' decode work).
        for ctx_index, ctx in enumerate(contexts):
            if ctx["ready"]:
                continue
            fut_to_job[pool.submit(_run_job, _build_request(ctx_index, 0))] = {
                "ctx": ctx_index, "kind": "build", "noise_index": 0,
            }

        def _submit_race_shards(ci: int, shards: list[int]) -> None:
            ctx = contexts[ci]
            race = ctx["race"]
            new_jobs = _decode_jobs_for(ci, 0, ctx["build_outcome"], shards)
            if shards:
                race["submitted"] = max(race["submitted"], max(shards) + 1)
            race["outstanding"] += len(new_jobs)
            pending[ci] += len(new_jobs)
            for nj in new_jobs:
                nf = _submit_decode(nj)
                fut_to_job[nf] = nj
                pending_futs.add(nf)

        def _race_window(boundary: int) -> int:
            # 2026-09-11 (pilot10x_v1): eliminations cluster at the pilot
            # decision, so nothing is speculated before it. A survivor first
            # fills a second checkpoint (2x the pilot, one more pilot-sized
            # wave); past that it marches to shots_confirm as ONE parallel
            # wave (window = race_lookahead, default 128 = a whole node), so
            # a slow survivor is not serialised into 250-shot steps. A
            # mid-race elimination can therefore waste at most that window.
            if int(boundary) < 2 * int(pilot_shards):
                return max(1, int(pilot_shards))
            return max(1, int(race_lookahead))

        def _race_advance(ci: int) -> None:
            """Walk decision boundaries strictly in order as their prefix of
            shards finishes landing; top the speculative window up after every
            continue. Landed-but-undecided results past a stop point are
            simply never consumed (merge_upto truncates at race["consumed"]),
            which is what keeps every recorded number identical to the
            lockstep schedule."""

            race = contexts[ci]["race"]
            per_shard = len(unit_index)
            while not race["done"]:
                boundary = race["next_decision_at"]
                if any(race["landed"].get(s, 0) < per_shard for s in range(boundary)):
                    return  # prefix not fully landed yet; a later landing re-enters
                race["consumed"] = boundary
                decision = _race_round_decision(ci)
                if decision["action"] == "continue" and boundary < n_shards:
                    race["next_decision_at"] = boundary + 1
                    target = min(n_shards, boundary + _race_window(boundary))
                    if boundary < confirm_shards:
                        target = min(target, confirm_shards)
                    if boundary < pilot_shards:
                        # a pre-pilot decision (v3) never speculates past the
                        # pilot: the pilot shards are already in flight and
                        # the 10x decision still happens at the pilot boundary
                        target = min(target, pilot_shards, boundary + pilot_window)
                    if target > race["submitted"]:
                        _submit_race_shards(ci, list(range(race["submitted"], target)))
                elif decision["action"] == "continue":
                    # Pre-registered stream exhausted below the confirm bar:
                    # only possible when shards failed/timed out (actual
                    # decoded shots lag the requested prefix). Stop and
                    # surface it — indexing past the stream would crash.
                    race["done"] = True
                    race["stopped_by"] = "stream_exhausted"
                    race["confirmed"] = False
                else:
                    race["done"] = True
                    race["stopped_by"] = decision["stopped_by"]
                    race["confirmed"] = decision["confirmed"]

        def _race_round_decision(ci: int) -> dict[str, Any]:
            """Decide the boundary the caller just set as race["consumed"]:
            pure decision on cumulative counts of that prefix
            (order-independent). Records the stage and returns the decision;
            acting on it is _race_advance's job."""

            ctx = contexts[ci]
            race = ctx["race"]
            slot = by_ctx[ci]
            primary = decoder_names[0]
            consumed = race["consumed"]
            merged = {
                basis: merge_shard_results(
                    slot["decode"].get((0, primary.value, basis), [None] * n_shards)[:consumed],
                    decoder=primary.value,
                )
                for basis in bases
            }
            combined = combine_basis_ler_results(
                merged,
                rounds=int(ctx["rounds"]),
                k=int(ctx.get("k_eff") or ctx["k"]),
                ci_z=float(verification.ler_ci_z),
            )
            plpr_ci = combined.get("per_logical_per_round_ci")
            shots_requested = sum(race_shard_shots[:consumed])
            # Confirm/pilot bookkeeping must count DECODED shots: a timed-out
            # or crashed shard contributes zero real shots, and counting the
            # requested stream prefix would mark "confirmed" short of
            # shots_confirm. min across bases = every basis reached the bar.
            shots_used = min(int(merged[b].get("shots") or 0) for b in bases)
            race["shots_actual"] = shots_used
            decision = racing_decision(
                shots=shots_used,
                basis_failures={b: int(merged[b].get("failures") or 0) for b in bases},
                plpr_ci_low=(plpr_ci[0] if plpr_ci else None),
                config=racing,
                thresholds=None if race["is_baseline"] else racing_thresholds,
            )
            # pilot10x_maxerr100_cost_v4: decoder cost over the consumed prefix,
            # CPU-seconds per decoded shot with both bases pooled (each worker
            # request is single-threaded, so its runtime_s is CPU time). The
            # baseline arm is exempt; a stop here becomes decode_infeasible in
            # _assemble. Timing never enters the recorded stages (they must stay
            # bit-identical across schedules); it is recorded only on a stop.
            cost_runtime_s = sum(float(merged[b].get("runtime_s") or 0.0) for b in bases)
            cost_shots = sum(int(merged[b].get("shots") or 0) for b in bases)
            cost_s_per_shot = (cost_runtime_s / cost_shots) if cost_shots > 0 else None
            if decision["action"] == "continue" and not race["is_baseline"]:
                cost_decision = decoder_cost_decision(
                    cost_s_per_shot=cost_s_per_shot,
                    shots=shots_used,
                    n=int(ctx.get("n_data") or 0),
                    config=racing,
                )
                if cost_decision is not None:
                    decision = cost_decision
                    race["decoder_cost"] = {
                        "cost_s_per_shot": cost_decision["cost_s_per_shot"],
                        "cost_cap_s_per_shot": cost_decision["cost_cap_s_per_shot"],
                        "shots_per_basis": shots_used,
                        "n_data": int(ctx.get("n_data") or 0),
                    }
            race["stages"].append(
                {
                    "shards_consumed": consumed,
                    "shots_per_basis": shots_used,
                    "shots_requested": shots_requested,
                    "failures": {b: int(merged[b].get("failures") or 0) for b in bases},
                    "per_logical_per_round": combined.get("per_logical_per_round"),
                    "per_logical_block_ci": combined.get("per_logical_block_ci"),
                    "plpr_ci": plpr_ci,
                    "action": decision["action"],
                    "stopped_by": decision["stopped_by"],
                }
            )
            return decision

        pending_futs.update(fut_to_job.keys())
        while pending_futs:
            done, pending_futs = wait(pending_futs, return_when=FIRST_COMPLETED)
            for fut in done:
                j = fut_to_job.pop(fut)
                outcome = fut.result()
                ci = j["ctx"]
                if j["kind"] == "build":
                    outcome = check_build_rounds(outcome, int(contexts[ci]["rounds"]))
                slot = by_ctx[ci]
                outcome_status = str((outcome or {}).get("status") or "")
                if outcome_status != "ok":
                    # v3.4 (BUG-1/BUG-2): EVERY non-ok job outcome is recorded
                    # — dem_load_error, subprocess_error (incl. a killed
                    # worker, with its returncode inside job_execution),
                    # build/pickle errors, distance failures — not only the
                    # timeout statuses, so a lost pickle or an OOM kill can no
                    # longer vanish from the candidate's diagnostics. The
                    # top-level returncode (v3.4.1) mirrors the outcome's own
                    # (subprocess_error) or the last attempt's, so a diagnosis
                    # tool never has to regex it out of the reason text.
                    job_execution = (outcome or {}).get("job_execution")
                    returncode = (outcome or {}).get("returncode")
                    if returncode is None and isinstance(job_execution, dict):
                        returncode = job_execution.get("returncode")
                    slot["job_failures"].append(
                        {
                            "kind": j["kind"],
                            "noise_index": j.get("noise_index"),
                            "shard": j.get("shard"),
                            "status": outcome_status or "unknown",
                            "failure_reason": (outcome or {}).get("failure_reason"),
                            "returncode": returncode if isinstance(returncode, int) else None,
                            "job_execution": job_execution,
                        }
                    )
                if j["kind"] == "distance":
                    slot["distance"] = outcome
                    pending[ci] -= 1
                elif j["kind"] == "build":
                    if int(j.get("noise_index") or 0) == 0:
                        # v3.6 (M5): whatever the outcome, the synthesis cache
                        # entry has been written (or the build failed for a
                        # reason the distance job will report too).
                        _submit_after_first_build(ci)
                    if isinstance(outcome, dict) and outcome.get("status") == "ok":
                        # Fig.3 Phase B: the measured circuit's logical qubits
                        # (2k for the cnot_dense task) root the A4/A5 rate.
                        k_eff = _num_logicals(outcome)
                        if k_eff is not None and int(j.get("noise_index") or 0) == 0:
                            contexts[ci]["k_eff"] = int(k_eff)
                        # v3.6 (B5): the two-basis DEM size scales this
                        # candidate's decode watchdog. The worker reports
                        # num_matrix_columns (an int, or per-basis dict);
                        # absent -> factor 1 (v3.4 value).
                        columns = _num_matrix_columns(outcome)
                        if columns is not None and int(j.get("noise_index") or 0) == 0:
                            contexts[ci]["num_matrix_columns"] = columns
                            contexts[ci]["shard_timeout_s"] = effective_decode_shard_timeout_s(
                                job_verification,
                                max(shard_shots) if shard_shots else 0,
                                columns,
                                contexts[ci].get("stage", stage),
                            )
                        if racing_mode:
                            contexts[ci]["build_outcome"] = outcome
                            pending[ci] -= 1
                            _submit_race_shards(ci, list(range(pilot_window)))
                            if pending[ci] == 0:  # zero-job edge: assemble below
                                _assemble(ci)
                                _cleanup_pickles(ci)
                            continue
                        new_jobs = _decode_jobs_for(ci, j["noise_index"], outcome)
                        pending[ci] += len(new_jobs)
                        for nj in new_jobs:
                            nf = _submit_decode(nj)
                            fut_to_job[nf] = nj
                            pending_futs.add(nf)
                    else:
                        # Keep the diagnostic: without it the missing decode
                        # cells read as shard timeouts, so a physics verdict
                        # (schedule_catastrophic) or a worker crash (OOM,
                        # import error) would be reported as an
                        # infrastructure timeout with no failure_reason.
                        slot["build_error"] = (
                            outcome
                            if isinstance(outcome, dict)
                            else {
                                "status": "timeout",
                                "failure_reason": "build_dem worker timed out",
                            }
                        )
                    pending[ci] -= 1
                else:  # shard decode
                    results = (outcome or {}).get("results") if isinstance(outcome, dict) else None
                    job_parts = int(j.get("parts") or 1)
                    # 2026-09-26: first capped watchdog timeout -> abort the candidate
                    _stage_ci = contexts[ci].get("stage", stage)
                    _cap_ci = decode_shard_timeout_cap_s(job_verification, _stage_ci)
                    if (
                        _cap_ci is not None
                        and ci not in aborted_ctx
                        and outcome_status in ("timeout", "timeout_after_transient")
                        and not (isinstance(outcome, dict) and (outcome.get("job_execution") or {}).get("cancelled"))
                    ):
                        aborted_ctx.add(ci)
                        slot["aborted"] = {
                            "stage": _stage_ci, "cap_s": float(_cap_ci),
                            "shard": j.get("shard"), "part": j.get("part"),
                            "first_status": outcome_status,
                        }
                        race_ci = contexts[ci].get("race")
                        if race_ci is not None and not race_ci.get("done"):
                            race_ci["done"] = True
                            race_ci["stopped_by"] = "shard_timeout_cap"
                            race_ci["confirmed"] = False
                        n_cancelled = 0
                        if ppool is not None:
                            n_cancelled = ppool.cancel_where(
                                lambda key, _ci=ci: isinstance(key, tuple) and key and key[0] == _ci,
                                lambda item, _ci=ci: _abort_outcome(
                                    _ci, worker_timeout_for(item["request"], job_verification)
                                ),
                            )
                        print(
                            f"  [slow] {contexts[ci]['payload'].get('program_id')} aborted after its first capped "
                            f"watchdog timeout ({_stage_ci} cap {float(_cap_ci):.0f} s, shard {j.get('shard')} "
                            f"part {j.get('part')}); {n_cancelled} queued chunk(s) cancelled -> decode_infeasible",
                            flush=True,
                        )
                    # An unsplit job completes its shard by itself; a split
                    # job completes it when its last part lands (folded below).
                    shard_complete = job_parts <= 1
                    requested_units = {
                        (str(unit.get("decoder")), str(unit.get("basis")))
                        for unit in ((j.get("request") or {}).get("units") or [])
                        if isinstance(unit, dict)
                    }
                    for decoder in decoder_names:
                        for basis in bases:
                            cell = (results or {}).get(decoder.value, {}).get(basis) if results else None
                            # v3.4 (BUG-1): ANY non-ok shard outcome (timeout,
                            # subprocess_error, dem_load_error, ...) fills the
                            # requested cell with ITS status, so a lost DEM
                            # pickle no longer masquerades as a watchdog
                            # timeout. A None slot only remains when the job
                            # never returned at all.
                            if (
                                cell is None
                                and outcome_status != "ok"
                                and (decoder.value, basis) in requested_units
                            ):
                                cell = {
                                    "status": outcome_status or "unknown",
                                    "decoder": decoder.value,
                                    "shots": 0,
                                    "failures": 0,
                                    "logical_error_rate": None,
                                    "runtime_s": 0.0,
                                    "failure_reason": (outcome or {}).get("failure_reason"),
                                    "job_execution": (outcome or {}).get("job_execution"),
                                }
                            if cell is not None:
                                if isinstance(cell, dict) and (outcome or {}).get("job_execution"):
                                    cell = {
                                        **cell,
                                        "job_execution": (outcome or {}).get("job_execution"),
                                    }
                                if job_parts > 1:
                                    # shard splitting: keep the part; the shard
                                    # cell is folded when its last part lands
                                    # (racing needs whole shards at once) or,
                                    # for anything left over, at assembly
                                    key = (j["noise_index"], decoder.value, basis)
                                    per_shard = slot["decode_parts"].setdefault(key, {})
                                    per_shard.setdefault(j["shard"], [None] * job_parts)[int(j["part"])] = cell
                                else:
                                    slot["decode"].setdefault(
                                        (j["noise_index"], decoder.value, basis), [None] * n_shards
                                    )[j["shard"]] = cell
                    if job_parts > 1:
                        # every unit of this (noise point, shard) is split into
                        # job_parts jobs: the shard is whole once all of them
                        # landed, and only then is it folded into the cells the
                        # race decision and the assembly read
                        counter_key = (j["noise_index"], j["shard"])
                        landed_parts = slot["decode_parts_landed"].get(counter_key, 0) + 1
                        slot["decode_parts_landed"][counter_key] = landed_parts
                        if landed_parts >= job_parts * len(unit_index):
                            shard_complete = True
                            for key, per_shard in list(slot["decode_parts"].items()):
                                if key[0] != j["noise_index"]:
                                    continue
                                parts = per_shard.pop(j["shard"], None)
                                if parts is not None:
                                    slot["decode"].setdefault(key, [None] * n_shards)[j["shard"]] = (
                                        merge_subshard_results(parts, decoder=key[1])
                                    )
                    pending[ci] -= 1
                    if racing_mode:
                        race = contexts[ci]["race"]
                        race["outstanding"] -= 1
                        if shard_complete:
                            # landed counts UNITS per shard (per_shard = len(unit_index)
                            # in _race_advance): one per unsplit job, all at once
                            # when a split shard folds
                            race["landed"][j["shard"]] = race["landed"].get(j["shard"], 0) + (
                                len(unit_index) if job_parts > 1 else 1
                            )
                        if not race["done"]:
                            _race_advance(ci)
                if pending[ci] == 0:
                    _assemble(ci)
                    _cleanup_pickles(ci)
    finally:
        pool.shutdown()
        if ppool is not None:
            ppool.shutdown()
        shutil.rmtree(dem_dir, ignore_errors=True)

    return [ctx["payload"] for ctx in contexts]
