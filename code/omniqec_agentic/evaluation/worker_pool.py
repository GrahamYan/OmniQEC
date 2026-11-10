"""Persistent decode workers (2026-09-14).

Why. The slow loop decodes a candidate's shot stream as small shot-range
chunks so that the heavy-tailed per-shot cost (an S3-C exhaustion is 20-40 s
against 0.01-0.1 s for a converging shot) does not leave 126 cores waiting for
the slowest 31-shot part. Small chunks are only affordable when the worker
process persists: a one-shot ``python -m worker`` paid 1.5 s alone and ~9 s
under 127-way contention per part (interpreter, bundle load, decoder build),
while a persistent process re-uses its module state and builds the next
decoder of the same candidate in 0.01 s (measured 2026-09-14, b98).

What. ``PersistentWorkerPool`` owns ``n_workers`` long-lived
``python -m omniqec_agentic.evaluation.worker --serve`` processes, one
dispatcher thread each. ``submit(request, key=...)`` returns a
``concurrent.futures.Future`` resolving to the same outcome mapping
``pool.run_worker_request_with_retries`` produces for a one-shot job (status,
payload, ``job_execution`` with attempts / retry budgets / timeout_s), so the
slow loop's bookkeeping, watchdog verdicts (``timeout`` -> decode_infeasible
gate) and transient-replay policy are unchanged. A request that exceeds its
watchdog kills and respawns that worker; a worker that dies mid-request is
respawned and the request replayed within ``transient_retries``.

Every recorded number is identical to the one-shot path: a chunk decodes the
same shot indices with the same per-shot seeds whichever process runs it
(``estimate_circuit_logical_error_rate(shot_range=...)``), and the pool folds
chunks back into shard cells exactly as for split parts.

Affinity. A worker prefers the queued request whose ``key`` (candidate,
basis) matches the one it served last, so the vendored decoder's in-process
S3 state is switched as rarely as possible; otherwise it takes the oldest
request (FIFO), which keeps the racing prefixes landing in order.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Any

from omniqec_agentic.config import VerificationConfig

from .pipeline import WORKER_MODULE

_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_STDERR_TAIL_LINES = 40


def persistent_workers_enabled() -> bool:
    """``OMNIQEC_PERSISTENT_WORKERS=1`` (launcher knob, not a protocol key)."""

    raw = str(os.environ.get("OMNIQEC_PERSISTENT_WORKERS", "")).strip().lower()
    return raw in ("1", "true", "yes", "on")


class _WorkerProcess:
    """One ``--serve`` subprocess with a line reader thread (so a dispatcher
    can wait on a response with a timeout) and a bounded stderr tail."""

    def __init__(self, env: dict[str, str]):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, "--serve"],
            cwd=str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            text=True,
            bufsize=1,
            env=env,
        )
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_reader.start()

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self._lines.put(line)
        except Exception:
            pass
        self._lines.put(None)  # EOF

    def _read_stderr(self) -> None:
        try:
            for line in self.proc.stderr:  # type: ignore[union-attr]
                self._stderr.append(line.rstrip("\n"))
        except Exception:
            pass

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, request: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(request, default=float))
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()

    def receive(self, timeout_s: float | None) -> tuple[dict[str, Any] | None, str]:
        """(payload, kind): kind is 'ok' | 'timeout' | 'eof' | 'garbage'."""

        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None, "timeout"
            if line is None:
                return None, "eof"
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue  # a stray print on fd 1: skip it, the payload follows
            if isinstance(payload, dict):
                return payload, "ok"

    def kill(self) -> None:
        try:
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), 9)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.kill()


class PersistentWorkerPool:
    def __init__(
        self,
        n_workers: int,
        *,
        verification: VerificationConfig,
        timeout_for: Any,
        env: dict[str, str] | None = None,
    ):
        self.n_workers = max(1, int(n_workers))
        self.verification = verification
        self._timeout_for = timeout_for
        self._env = dict(os.environ if env is None else env)
        self._env.update(_THREAD_CAPS)
        self._lock = threading.Condition()
        self._order: deque[dict[str, Any]] = deque()
        self._by_key: dict[Any, deque[dict[str, Any]]] = {}
        self._closing = False
        self._threads: list[threading.Thread] = []
        self._idle: dict[int, Any] = {}
        self._scan_window = 256
        self._procs: list[_WorkerProcess | None] = [None] * self.n_workers
        self.stats = {"submitted": 0, "completed": 0, "respawns": 0, "timeouts": 0, "affinity_hits": 0}
        for index in range(self.n_workers):
            thread = threading.Thread(target=self._loop, args=(index,), name=f"omniqec-worker-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    # ------------------------------------------------------------ submit
    def submit(self, request: dict[str, Any], key: Any = None) -> "Future[dict[str, Any]]":
        future: "Future[dict[str, Any]]" = Future()
        item = {"request": request, "future": future, "key": key, "taken": False}
        with self._lock:
            if self._closing:
                raise RuntimeError("PersistentWorkerPool is shut down")
            self._order.append(item)
            if key is not None:
                self._by_key.setdefault(key, deque()).append(item)
            self.stats["submitted"] += 1
            self._lock.notify_all()
        return future

    def cancel_where(self, predicate: Any, outcome: Any) -> int:
        """2026-09-26: resolve every QUEUED (not yet taken) request whose
        ``key`` satisfies ``predicate`` with ``outcome(item)`` instead of
        running it, and return how many were resolved. Used by the pool when
        a candidate's first capped watchdog timeout lands: its remaining
        chunks would each burn their own watchdog for nothing. Requests a
        worker already took run to completion (their own timeout bounds them);
        the resolved futures land through the normal outcome path."""

        cancelled = 0
        with self._lock:
            for item in list(self._order):
                if item["taken"] or item["future"].done():
                    continue
                if not predicate(item["key"]):
                    continue
                item["taken"] = True
                item["cancelled"] = True
                dq = self._by_key.get(item["key"])
                if dq is not None:
                    try:
                        dq.remove(item)
                    except ValueError:
                        pass
                    if not dq:
                        self._by_key.pop(item["key"], None)
                cancelled += 1
                item["future"].set_result(outcome(item))
            while self._order and self._order[0]["taken"]:
                self._order.popleft()
            self.stats["cancelled"] = int(self.stats.get("cancelled", 0)) + cancelled
        return cancelled

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            self._lock.notify_all()
        for proc in self._procs:
            if proc is not None:
                proc.kill()
        for thread in self._threads:
            thread.join(timeout=5)
        with self._lock:
            for item in list(self._order):
                if not item["taken"] and not item["future"].done():
                    item["future"].set_result(
                        {"status": "subprocess_error", "failure_reason": "worker pool shut down before the job ran",
                         "job_execution": {"attempts": 0}}
                    )
            self._order.clear()
            self._by_key.clear()

    # -------------------------------------------------------- scheduling
    def _next(self, index: int, prefer_key: Any) -> dict[str, Any] | None:
        """Block until an item is available or the pool is closing (None).

        Affinity: this worker first takes a queued request with its own last
        key; otherwise it takes the oldest request whose key is not the last
        key of ANOTHER idle worker (that worker will pick it up, keeping its
        decoder state), scanning a bounded window so nothing starves; if every
        request in the window is claimed it takes the oldest anyway."""

        with self._lock:
            self._idle[index] = prefer_key
            try:
                while True:
                    if self._closing:
                        return None
                    if prefer_key is not None:
                        dq = self._by_key.get(prefer_key)
                        while dq:
                            item = dq.popleft()
                            if not item["taken"]:
                                item["taken"] = True
                                self.stats["affinity_hits"] += 1
                                return item
                        if dq is not None and not dq:
                            self._by_key.pop(prefer_key, None)
                    claimed = {k for i, k in self._idle.items() if i != index and k is not None}
                    fallback = None
                    scanned = 0
                    for item in self._order:
                        if item["taken"]:
                            continue
                        scanned += 1
                        if item["key"] is None or item["key"] not in claimed or item["key"] == prefer_key:
                            fallback = item
                            break
                        if scanned >= self._scan_window:
                            break
                    if fallback is None:
                        # every visible request is claimed by an idle worker
                        # that has not woken up yet: take the oldest instead
                        # of idling (a claimed key still gets served, just
                        # by a worker that has to switch state)
                        for item in self._order:
                            if not item["taken"]:
                                fallback = item
                                break
                    if fallback is not None:
                        fallback["taken"] = True
                        while self._order and self._order[0]["taken"]:
                            self._order.popleft()
                        return fallback
                    self._lock.wait(timeout=1.0)
            finally:
                self._idle.pop(index, None)

    def _spawn(self, index: int) -> _WorkerProcess:
        proc = _WorkerProcess(self._env)
        self._procs[index] = proc
        return proc

    def _loop(self, index: int) -> None:
        proc: _WorkerProcess | None = None
        last_key: Any = None
        while True:
            item = self._next(index, last_key)
            if item is None:
                break
            if proc is None or not proc.alive():
                if proc is not None:
                    self.stats["respawns"] += 1
                proc = self._spawn(index)
            request = item["request"]
            last_key = item["key"]
            try:
                outcome, proc = self._run_one(index, proc, request)
            except Exception as exc:  # pragma: no cover - defensive: never lose a future
                outcome = {"status": "subprocess_error", "failure_reason": f"dispatcher error: {exc}"[:300],
                           "job_execution": {"attempts": 1}}
            self.stats["completed"] += 1
            item["future"].set_result(outcome)
        if proc is not None:
            proc.kill()

    def _run_one(self, index: int, proc: _WorkerProcess, request: dict[str, Any]) -> tuple[dict[str, Any], _WorkerProcess]:
        """Mirror of ``pool.run_worker_request_with_retries`` on a persistent
        process: same retry budgets, same statuses, same job_execution."""

        verification = self.verification
        timeout_s = self._timeout_for(request, verification)
        timeout_limit = max(0, int(verification.timeout_retries))
        transient_limit = max(0, int(getattr(verification, "transient_retries", 0)))
        mode = str(request.get("mode") or "candidate")
        attempts = 0
        timeout_retries_used = 0
        transient_retries_used = 0
        while True:
            attempts += 1
            if not proc.alive():
                self.stats["respawns"] += 1
                proc = self._spawn(index)
            try:
                proc.send(request)
                payload, kind = proc.receive(timeout_s)
            except (BrokenPipeError, OSError, ValueError):
                payload, kind = None, "eof"
            execution = {
                "attempts": int(attempts),
                "retries_used": int(timeout_retries_used),
                "retry_limit": int(timeout_limit),
                "transient_retry_limit": int(transient_limit),
                "transient_retries_used": int(transient_retries_used),
                "timeout_s": timeout_s,
                "returncode": proc.proc.returncode if proc.proc.returncode is not None else None,
                "worker_mode": "persistent",
            }
            if kind == "ok" and payload is not None:
                status = str(payload.get("status") or "")
                if status == "dem_load_error" and transient_retries_used < transient_limit:
                    transient_retries_used += 1
                    continue
                if status == "worker_exception":
                    # the request raised inside the worker: deterministic, not
                    # replayed (a one-shot worker would have died without JSON
                    # and been replayed once; here the reason is known)
                    return (
                        {"status": "subprocess_error", "failure_reason": str(payload.get("failure_reason") or "")[-300:],
                         "returncode": None, "job_execution": execution},
                        proc,
                    )
                rss = payload.get("ru_maxrss")
                if isinstance(rss, (int, float)) and not isinstance(rss, bool) and rss >= 0:
                    execution = {**execution, "ru_maxrss": int(rss)}
                return {**payload, "job_execution": execution}, proc
            if kind == "timeout":
                self.stats["timeouts"] += 1
                proc.kill()
                self.stats["respawns"] += 1
                proc = self._spawn(index)
                if timeout_retries_used < timeout_limit:
                    timeout_retries_used += 1
                    continue
                if mode == "distance_job":
                    status = "uncertified_timeout"
                elif transient_retries_used > 0:
                    status = "timeout_after_transient"
                else:
                    status = "timeout"
                reason = f"{mode} worker exceeded timeout_s={timeout_s!r} on all {attempts} attempts"
                if transient_retries_used > 0:
                    reason += f" after {transient_retries_used} transient replay(s)"
                return {"status": status, "failure_reason": reason, "job_execution": {**execution, "timed_out": True}}, proc
            # eof / garbage: the worker died (OOM kill, segfault) or closed its stream
            tail = proc.stderr_tail()
            returncode = proc.proc.poll()
            proc.kill()
            self.stats["respawns"] += 1
            proc = self._spawn(index)
            if transient_retries_used < transient_limit:
                transient_retries_used += 1
                continue
            execution["returncode"] = int(returncode) if isinstance(returncode, int) else None
            return (
                {"status": "subprocess_error",
                 "failure_reason": str(tail or f"worker produced no parseable JSON (returncode={returncode})")[-300:],
                 "returncode": execution["returncode"], "job_execution": execution},
                proc,
            )
