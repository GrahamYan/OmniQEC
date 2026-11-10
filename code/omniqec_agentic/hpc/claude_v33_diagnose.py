#!/usr/bin/env python
"""Explain why a v3.3 / v3.4 / v3.5 run dir stopped (measurement_blocked / FATAL_STOP)
and which v3.4 candidates ended as decode_infeasible (terminal, not a stop).

    python omniqec_agentic/hpc/claude_v33_diagnose.py <run_dir> [--turn N] [--log-tail 40]

Standard library only, no package imports: it must run on a login node even
before the conda env is activated. Reads protocol_snapshot.json, FATAL_STOP.json
and every turns/turn_XXXX/cycle/slow/{slow_summary.json,candidates/*.json},
and prints per candidate / per basis which shards finished, which timed out or
crashed, on how many attempts, against which watchdog — then a one-line
classification per candidate so the human can decide whether the watchdog is
too tight for this code size, the worker died (OOM), or the build never
finished. v3.4 artifacts additionally carry per-shard runtimes
(`shard_runtime_s`, shown as max/mean), per-shard attempts (`shard_attempts`),
the per-shot budget / transient-retry policy in `job_config`, and the
candidate status `decode_infeasible` (only defect = watchdog timeouts under
the frozen per-shot budget): that one is a terminal outcome — excluded from
ranking/final and never re-queued — not a measurement_blocked stop.
v3.4.1 tightened that gate: a shard whose watchdog expired only AFTER a
transient replay (killed worker / lost DEM on an earlier attempt) carries the
distinct shard status `timeout_after_transient` and, like `subprocess_error`
and `dem_load_error`, is infrastructure (measurement_blocked), never
infeasible; returncodes are read from `job_failures[].returncode` /
`job_failures[].job_execution.returncode`, not only from the reason text.
v3.5 (bposd_structural_v3_5) changed only the noise model (paper SI C
uniform_paper_v1, no idle noise; v3.4 ran uniform_legacy_v1 with idle
noise): the protocol line below prints the snapshot's noise_profile /
noise_values so a v3.4 and a v3.5 run dir are never read as one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        return {"_unreadable": f"{type(exc).__name__}: {exc}"}


def _short(text, limit=160):
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


_ATTEMPTS = re.compile(r"on all (\d+) attempts")
_TIMEOUT = re.compile(r"timeout_s=([0-9.]+)")
_RETCODE = re.compile(r"returncode[=: ]+(-?\d+)")


def _summarize_reason(reason: str) -> str:
    """Collapse 'msg; msg; msg' into 'msg (xN)' groups, keeping attempt/timeout info."""
    parts = [p.strip() for p in str(reason or "").split(";") if p.strip()]
    if not parts:
        return ""
    counts = Counter(parts)
    out = []
    for msg, n in counts.most_common():
        out.append(f"{_short(msg, 140)}" + (f" (x{n})" if n > 1 else ""))
    return " | ".join(out[:4]) + (f" | ... {len(out) - 4} more distinct" if len(out) > 4 else "")


_KILLED_RETURNCODES = (-9, 137)
# A watchdog expiry SIGKILLs the worker's process group by design, so a
# timeout entry legitimately carries returncode -9: it is not an OOM signal.
_TIMEOUT_STATUSES = ("timeout", "timeout_after_transient", "uncertified_timeout")


def _job_returncodes(
    candidate: dict, kind: str | None = None, *, exclude_timeouts: bool = True
) -> list[int]:
    """Returncodes recorded for the candidate's non-ok jobs (v3.4 job_failures):
    the entry's top-level `returncode` (v3.4.1; the outcome's own for a
    subprocess_error) or `job_execution.returncode` (the LAST attempt's).
    `kind` narrows to "shard" / "build" / "distance"; watchdog-timeout entries
    are skipped by default (their -9 is the watchdog's own kill)."""

    out: list[int] = []
    for entry in candidate.get("job_failures") or []:
        if not isinstance(entry, dict):
            continue
        if kind is not None and str(entry.get("kind") or "") != kind:
            continue
        if exclude_timeouts and str(entry.get("status") or "") in _TIMEOUT_STATUSES:
            continue
        rc = entry.get("returncode")
        if rc is None:
            execution = entry.get("job_execution")
            if isinstance(execution, dict):
                rc = execution.get("returncode")
        if isinstance(rc, int) and not isinstance(rc, bool):
            out.append(rc)
    return out


def _killed_hint(candidate: dict, kind: str, reason: str) -> str:
    """OOM/SIGKILL hint from the recorded returncodes (v3.4.1), falling back to
    the regex over the merged reason text for pre-v3.4 artifacts."""

    killed = [rc for rc in _job_returncodes(candidate, kind) if rc in _KILLED_RETURNCODES]
    if killed:
        return (
            f" — SIGKILL/137 on {len(killed)} {kind} job(s) (job_failures returncode): "
            "worker killed, most likely OOM"
        )
    rc = _RETCODE.search(reason)
    if rc and rc.group(1) in ("-9", "137"):
        return " — SIGKILL/137: worker killed, most likely OOM"
    return ""


def _classify(candidate: dict) -> str:
    status = str(candidate.get("status") or "")
    if status == "ok":
        return "ok"
    if status == "schedule_catastrophic" or (candidate.get("distance_evidence") or {}).get("schedule_catastrophic"):
        return "schedule_catastrophic (terminal, not a blocker)"
    if status == "decode_infeasible" or candidate.get("decode_infeasible"):
        # v3.4: every required basis returned and the ONLY non-ok shard status is
        # the watchdog timeout under the frozen per-shot budget. The candidate
        # is dropped from ranking/final and never re-queued; the queue stays ok.
        return (
            "decode_infeasible (terminal): "
            + _short(candidate.get("failure_reason") or "decode exceeded the frozen per-shot budget", 200)
            + " — not a stop; excluded from ranking/final, not re-queued"
        )
    reasons = []
    for result in candidate.get("results") or []:
        for basis, bv in (result.get("basis") or {}).items():
            statuses = Counter(str(s) for s in (bv.get("shard_statuses") or []))
            reason = str(bv.get("failure_reason") or "")
            if statuses.get("timeout"):
                attempts = _ATTEMPTS.search(reason)
                tmo = _TIMEOUT.search(reason)
                reasons.append(
                    f"{basis}: {statuses['timeout']}/{sum(statuses.values())} shards hit the decode watchdog"
                    + (f" ({tmo.group(1)}s)" if tmo else "")
                    + (f" on all {attempts.group(1)} attempts" if attempts else "")
                )
            if statuses.get("timeout_after_transient"):
                # v3.4.1: the watchdog expired only after this very request had
                # already been replayed for a killed worker / lost DEM, so the
                # expiry is not evidence about the decoder's per-shot cost.
                reasons.append(
                    f"{basis}: {statuses['timeout_after_transient']}/{sum(statuses.values())} shards "
                    "timeout_after_transient — watchdog expiry after a transient replay "
                    "(killed worker / lost DEM on an earlier attempt) — treat as infrastructure, "
                    "not infeasible"
                )
            other = {
                k: v for k, v in statuses.items()
                if k not in ("ok", "timeout", "timeout_after_transient")
            }
            if other:
                hint = _killed_hint(candidate, "shard", reason)
                if other.get("subprocess_error"):
                    hint += (
                        " — subprocess_error: worker died without parseable JSON "
                        "(infrastructure; transient retry exhausted)"
                    )
                if other.get("dem_load_error"):
                    hint += (
                        " — dem_load_error: worker could not load its DEM "
                        "(infrastructure; transient retry exhausted)"
                    )
                reasons.append(f"{basis}: non-ok shards {dict(other)}{hint}")
            if not statuses and str(bv.get("status") or "") not in ("ok", ""):
                reasons.append(f"{basis}: basis status={bv.get('status')} with no shard record — build/DEM stage never produced shards?")
    if not reasons:
        reasons.append(f"status={status}: {_short(candidate.get('failure_reason'), 120)}")
        # A candidate that never reached decode (build/distance failure):
        # the kill hint comes from those jobs' recorded returncodes.
        for kind in ("build", "distance"):
            hint = _killed_hint(candidate, kind, "")
            if hint:
                reasons.append(f"{kind} job(s){hint}")
    return "; ".join(reasons)


def _job_failures_line(candidate: dict) -> str:
    """One line over the v3.4 job_failures records: counts per (kind, status)
    and the returncodes seen, so an OOM kill or a replayed-then-timed-out
    shard is visible without opening every basis."""

    entries = [e for e in (candidate.get("job_failures") or []) if isinstance(e, dict)]
    if not entries:
        return ""
    groups: dict[tuple[str, str], list] = {}
    for entry in entries:
        key = (str(entry.get("kind") or "?"), str(entry.get("status") or "?"))
        rc = entry.get("returncode")
        if rc is None and isinstance(entry.get("job_execution"), dict):
            rc = entry["job_execution"].get("returncode")
        groups.setdefault(key, []).append(rc)
    parts = []
    for (kind, status), rcs in sorted(groups.items()):
        seen = sorted({str(rc) for rc in rcs})
        parts.append(f"{kind}/{status} x{len(rcs)} (returncode {', '.join(seen)})")
    return "     job_failures: " + "; ".join(parts)


def describe_candidate(path: Path, *, verbose: bool) -> str:
    cand = _load(path)
    if not isinstance(cand, dict):
        return f"  {path.name}: unreadable ({cand})"
    fast = cand.get("fast") or {}
    promo = cand.get("promotion_record") or {}
    n = fast.get("n") or promo.get("n")
    k = fast.get("k") or promo.get("k")
    d = fast.get("d_search") or promo.get("d_search") or fast.get("d")
    n_total = fast.get("n_total") or promo.get("n_total")
    family = fast.get("family") or promo.get("family") or (cand.get("compiled_code") or {}).get("family")
    job = cand.get("job_config") or {}
    lines = [
        f"  ── {cand.get('program_id')}  [[{n},{k},d_search={d}]] n_total={n_total} {family}  "
        f"status={cand.get('status')}  runtime={float(cand.get('runtime_s') or 0):.0f}s"
    ]
    if job:
        v34 = ""
        if job.get("decode_shot_budget_s") is not None or job.get("transient_retries") is not None:
            v34 = (
                f" shot_budget={job.get('decode_shot_budget_s')}s/shot "
                f"effective_decode_shard={job.get('effective_decode_shard_timeout_s')}s "
                f"transient_retries={job.get('transient_retries')}"
            )
        lines.append(
            f"     job_config: shards={job.get('shot_shards')} build={job.get('build_timeout_s')}s "
            f"decode_shard={job.get('decode_shard_timeout_s')}s distance={job.get('distance_timeout_s')}s "
            f"retries={job.get('timeout_retries')} workers={job.get('workers')}{v34}"
        )
    if cand.get("failure_reason"):
        lines.append(f"     failure_reason: {_short(cand.get('failure_reason'), 200)}")
    failures_line = _job_failures_line(cand)
    if failures_line:
        lines.append(failures_line)
    for result in cand.get("results") or []:
        lines.append(
            f"     result: status={result.get('status')} measurement_complete={result.get('measurement_complete')} "
            f"decoder={result.get('decoder')} p={result.get('p')} rounds={result.get('rounds')}"
        )
        combined = result.get("combined") or {}
        if combined.get("failure_reason"):
            lines.append(f"       combined.failure_reason={combined.get('failure_reason')}")
        for basis, bv in sorted((result.get("basis") or {}).items()):
            statuses = Counter(str(s) for s in (bv.get("shard_statuses") or []))
            metrics = bv.get("metrics") or {}
            me = bv.get("memory_experiment") or {}
            shots = bv.get("shots")
            requested = metrics.get("requested_shots")
            shard_count = bv.get("shard_count") or len(bv.get("shard_statuses") or [])
            per_shard = (int(requested) // int(shard_count)) if requested and shard_count else None
            ok_shards = statuses.get("ok", 0)
            runtime = float(bv.get("runtime_s") or 0)
            rate = ""
            if shots and runtime > 0:
                # runtime_s is the decode-loop sum over ok shards only, so this
                # is the mean cost per decoded shot on this node (startup,
                # unpickle and decoder construction are not included).
                rate = f" ≈{runtime / float(shots):.2f}s/shot"
                if ok_shards:
                    rate += f", {runtime / ok_shards:.0f}s per ok shard"
            lines.append(
                f"       basis {basis}: status={bv.get('status')} shards={dict(statuses) or '{}'} "
                f"shots={shots}/{requested} (~{per_shard} per shard) runtime={runtime:.0f}s{rate} "
                f"failures={bv.get('failures')}"
            )
            # v3.4 merge_shard_results: per-shard wall times (None for a shard
            # that never completed) and attempts. max vs the watchdog says how
            # close the slowest finished shard came to the per-shot budget.
            shard_rt = [float(x) for x in (bv.get("shard_runtime_s") or []) if isinstance(x, (int, float))]
            if shard_rt:
                attempts = [int(a) for a in (bv.get("shard_attempts") or []) if isinstance(a, (int, float))]
                lines.append(
                    f"         shard_runtime_s: max={max(shard_rt):.0f}s mean={sum(shard_rt) / len(shard_rt):.0f}s "
                    f"over {len(shard_rt)}/{len(bv.get('shard_runtime_s') or [])} finished shards"
                    + (f"; attempts max={max(attempts)}" if attempts else "")
                )
            if me:
                lines.append(
                    f"         circuit: data_qubits={me.get('num_data_qubits')} checks={me.get('num_checks')} "
                    f"detectors={me.get('num_detectors')} ticks={me.get('num_ticks')} rounds={me.get('rounds')}"
                )
            if metrics.get("num_error_mechanisms") is not None:
                lines.append(
                    f"         dem: error_mechanisms={metrics.get('num_error_mechanisms')} "
                    f"nnz_H={metrics.get('num_nonzero_H')} actual_shots={metrics.get('actual_shots')} "
                    f"sampling_complete={metrics.get('sampling_complete')}"
                )
            if bv.get("failure_reason"):
                lines.append(f"         failure_reason: {_summarize_reason(bv.get('failure_reason'))}")
    checks = cand.get("slow_checks") or {}
    dist = checks.get("distance_evidence") or cand.get("distance_evidence") or {}
    if dist:
        lines.append(
            f"     distance: status={dist.get('status')} d_circ_ext={dist.get('d_circ_ext')} "
            f"exact_code_distance={dist.get('exact_code_distance')} catastrophic={dist.get('schedule_catastrophic')}"
        )
    lines.append(f"     ⇒ {_classify(cand)}")
    if verbose:
        lines.append("     structural: " + _short(json.dumps({k: (cand.get('structural') or {}).get(k) for k in ('status', 'valid', 'failure_reasons', 'n_data', 'max_check_weight')}), 200))
    return "\n".join(lines)


def describe_turn(turn_dir: Path, *, verbose: bool) -> str:
    slow = turn_dir / "cycle" / "slow"
    out = [f"=== {turn_dir.name} ==="]
    audit = _load(turn_dir / "turn_audit.json")
    if isinstance(audit, dict):
        out.append(f"  turn_audit: status={audit.get('status')} " + _short(json.dumps(audit.get('errors') or audit.get('checks') or ''), 200))
    summary = _load(slow / "slow_summary.json")
    if not isinstance(summary, dict):
        out.append(f"  (no slow_summary.json under {slow} — slow stage never wrote a summary; check the attempt log for build/queue errors)")
        return "\n".join(out)
    keys = (
        "status", "candidate_count", "ok_candidate_count", "timeout_candidate_count",
        "measurement_blocked_candidate_count", "incomplete_candidate_count",
        "decode_infeasible_candidate_count",
        "catastrophic_candidate_count", "ok_result_count", "runtime_s",
    )
    out.append("  slow_summary: " + "  ".join(f"{k}={summary.get(k) if k != 'runtime_s' else round(float(summary.get(k) or 0))}" for k in keys if k in summary))
    cands = sorted((slow / "candidates").glob("*.json")) if (slow / "candidates").is_dir() else []
    if not cands:
        out.append("  (no candidate payloads)")
    for path in cands:
        out.append(describe_candidate(path, verbose=verbose))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--turn", type=int, default=None, help="only this turn index (default: every turn dir)")
    ap.add_argument("--log-tail", type=int, default=40, help="lines of the FATAL_STOP attempt log to show (0 = none)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    print(f"run_dir: {run_dir}")

    snap = _load(run_dir / "protocol_snapshot.json")
    if isinstance(snap, dict):
        ver = snap.get("verification") or {}
        print(
            f"protocol: profile={snap.get('protocol_profile')} "
            f"noise_profile={snap.get('noise_profile')} noise_values={snap.get('noise_values')} "
            f"shot_shards={snap.get('shot_shards')} "
            f"final_shot_shards={snap.get('final_shot_shards')} "
            f"slow_workers={snap.get('slow_workers')} max_turns={snap.get('max_turns')} final_shots={snap.get('final_shots')}"
        )
        print(
            f"verification: shots={ver.get('shots')} rounds={ver.get('rounds')} decoders={ver.get('decoders')} "
            f"build={ver.get('build_timeout_s')}s decode_shard={ver.get('decode_shard_timeout_s')}s "
            f"distance={ver.get('distance_timeout_s')}s retries={ver.get('timeout_retries')}"
            + (
                f" shot_budget={ver.get('decode_shot_budget_s')}s/shot transient_retries={ver.get('transient_retries')}"
                if ver.get("decode_shot_budget_s") is not None or ver.get("transient_retries") is not None
                else " (pre-v3.4 snapshot: flat shard watchdog, no per-shot budget)"
            )
        )

        def _watchdog(shots_per_shard: int) -> str:
            # v3.4 pool.effective_decode_shard_timeout_s: max(floor, budget x shots);
            # budget <= 0 / absent -> floor alone (v3.3 flat watchdog).
            try:
                floor = float(ver.get("decode_shard_timeout_s") or 0)
                budget = float(ver.get("decode_shot_budget_s") or 0)
            except (TypeError, ValueError):
                return ""
            eff = max(floor, budget * shots_per_shard) if budget > 0 else floor
            return f", effective shard watchdog {eff:g}s" if eff > 0 else ", shard watchdog disabled"

        shards = snap.get("shot_shards")
        final_shards = snap.get("final_shot_shards") or shards
        if ver.get("shots") and shards:
            per = int(ver["shots"]) // int(shards)
            print(f"  → search stage: {per} shots per shard per basis ({shards} shards{_watchdog(per)})")
        if snap.get("final_shots") and final_shards:
            per = int(snap["final_shots"]) // int(final_shards)
            print(f"  → final stage:  {per} shots per shard per basis ({final_shards} shards{_watchdog(per)})")
    else:
        print("protocol_snapshot.json: missing/unreadable")

    fatal = _load(run_dir / "FATAL_STOP.json")
    if isinstance(fatal, dict):
        print("FATAL_STOP.json:")
        for key, value in fatal.items():
            print(f"  {key}: {_short(json.dumps(value) if not isinstance(value, str) else value, 300)}")
    else:
        print("FATAL_STOP.json: (absent)")

    turn_dirs = sorted((run_dir / "turns").glob("turn_*")) if (run_dir / "turns").is_dir() else []
    if args.turn is not None:
        turn_dirs = [d for d in turn_dirs if d.name == f"turn_{args.turn:04d}"]
    if not turn_dirs and (run_dir / "slow_summary.json").exists():
        # A bare slow-stage directory (slow-verify-queue --out, or a
        # turns/turn_XXXX/cycle/slow path passed directly).
        print()
        print(f"=== slow stage dir {run_dir.name} ===")
        summary = _load(run_dir / "slow_summary.json") or {}
        print("  slow_summary: " + "  ".join(f"{k}={summary.get(k)}" for k in ("status", "candidate_count", "ok_candidate_count", "measurement_blocked_candidate_count", "incomplete_candidate_count", "decode_infeasible_candidate_count") if k in summary))
        for path in sorted((run_dir / "candidates").glob("*.json")) if (run_dir / "candidates").is_dir() else []:
            print(describe_candidate(path, verbose=args.verbose))
        return 0
    if not turn_dirs:
        print("no turn directories found")
    for turn_dir in turn_dirs:
        print()
        print(describe_turn(turn_dir, verbose=args.verbose))

    log_path = None
    if isinstance(fatal, dict) and fatal.get("attempt_log"):
        log_path = Path(str(fatal["attempt_log"]))
    if log_path and args.log_tail > 0:
        print()
        print(f"=== tail -{args.log_tail} {log_path} ===")
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            interesting = [
                ln for ln in lines
                if not ln.startswith('{"decoder_build"')
            ]
            for ln in interesting[-args.log_tail:]:
                print("  " + _short(ln, 220))
        except OSError as exc:
            print(f"  (cannot read: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
