"""hpc/subreaper.py: transparent exit-status / signal contract everywhere,
orphan reaping on Linux (a training Pod whose PID 1 is ``sleep`` never reaps)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SUBREAPER = Path(__file__).resolve().parents[1] / "hpc" / "subreaper.py"


def _run(*cmd: str, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUBREAPER), *cmd], capture_output=True, text=True, timeout=30, **kw
    )


def test_exit_status_is_the_commands_own() -> None:
    assert _run("sh", "-c", "exit 0").returncode == 0
    assert _run("sh", "-c", "exit 7").returncode == 7
    assert _run("--", "sh", "-c", "exit 3").returncode == 3


def test_signal_death_is_reported_like_bash_wait() -> None:
    assert _run("sh", "-c", "kill -9 $$").returncode == 137


def test_no_command_is_a_usage_error() -> None:
    proc = _run()
    assert proc.returncode == 2 and "usage" in proc.stderr


def test_stdio_and_environment_pass_through(tmp_path: Path) -> None:
    proc = _run("sh", "-c", 'echo "out $SUBREAPER_TEST_VAR"; echo err >&2',
                env={**os.environ, "SUBREAPER_TEST_VAR": "42"})
    assert proc.stdout == "out 42\n" and proc.stderr == "err\n" and proc.returncode == 0


def test_sigterm_is_forwarded_to_the_command() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SUBREAPER), "sh", "-c", 'trap "exit 9" TERM; sleep 3 & wait'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 9


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_CHILD_SUBREAPER is Linux-only")
def test_orphans_below_the_command_are_reaped_not_left_as_zombies() -> None:
    # The subshell exits at once, orphaning `sleep 0.2`; it is re-parented to
    # the wrapper (not PID 1) and reaped when it ends, before `sh` returns.
    proc = _run("sh", "-c", "( sleep 0.2 & echo $! ); sleep 1; exit 5")
    assert proc.returncode == 5
    orphan = int(proc.stdout.strip())
    assert not Path(f"/proc/{orphan}").exists(), "orphan still present (zombie?) after the wrapper exited"


def test_become_subreaper_reports_platform_support() -> None:
    sys.path.insert(0, str(SUBREAPER.parent))
    try:
        import subreaper  # noqa: WPS433 (test-local import of the script module)
    finally:
        sys.path.pop(0)
    supported = subreaper.become_subreaper()
    assert supported == sys.platform.startswith("linux")
