"""Run a command as a child subreaper and reap every orphan it leaves behind.

    python omniqec_agentic/hpc/subreaper.py <command> [args...]

Why (2026-09-06, a training-Pod b60 fatal): a container whose PID 1 never calls
``wait()`` (that training Pod's PID 1 is ``sleep``) keeps every orphaned
process as a permanent zombie, and zombies count against the container's
``pids.max``. Each decode/build worker used to leave two of them (the
telescoping package's forkserver + resource_tracker helpers, see
``config.py::to_telescope_config``), so ~15k worker jobs exhausted a
30000-pid quota and the next ``fork`` failed with ``EAGAIN``.

With ``PR_SET_CHILD_SUBREAPER`` (Linux) every process orphaned anywhere
below the command is re-parented to this wrapper instead of PID 1, and the
wrapper reaps it. The wrapper is transparent otherwise:

* stdio and the environment are inherited unchanged;
* SIGTERM / SIGINT / SIGHUP are forwarded to the command;
* the exit status is the command's own (``128 + signal`` when it died by a
  signal, exactly what ``bash`` reports for ``wait``), so callers that
  interpret exit codes (the Pod grid launcher) see no difference;
* after the command exits, remaining orphans are reaped once without
  blocking and the wrapper exits — daemons meant to outlive the command
  (mihomo) are left alone.

On platforms without ``prctl`` (macOS) the command simply runs unwrapped
with the same exit-status contract. Stdlib only: it runs under the Pod's
base interpreter before any conda environment is activated.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys

PR_SET_CHILD_SUBREAPER = 36
_FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def become_subreaper() -> bool:
    """Mark this process as a child subreaper. ``False`` where unsupported."""

    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, OSError):
        return False
    return int(result) == 0


def _exit_status(status: int) -> int:
    if os.WIFSIGNALED(status):
        return 128 + int(os.WTERMSIG(status))
    if os.WIFEXITED(status):
        return int(os.WEXITSTATUS(status))
    return 1


def reap_nonblocking() -> int:
    """Reap every already-exited child without blocking; returns the count."""

    reaped = 0
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        except InterruptedError:
            continue
        if pid == 0:
            return reaped
        reaped += 1


def run(argv: list[str]) -> int:
    become_subreaper()
    child = subprocess.Popen(argv)

    def forward(signum, _frame):
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    for sig in _FORWARDED:
        signal.signal(sig, forward)

    rc: int | None = None
    while rc is None:
        try:
            pid, status = os.waitpid(-1, 0)
        except InterruptedError:
            continue
        except ChildProcessError:
            # The command was reaped by something else (cannot happen with
            # Popen), report failure rather than hang.
            rc = child.poll() if child.poll() is not None else 1
            break
        if pid == child.pid:
            child.returncode = _exit_status(status)  # keep Popen consistent
            rc = child.returncode
    reap_nonblocking()
    return int(rc)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: subreaper.py <command> [args...]", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
