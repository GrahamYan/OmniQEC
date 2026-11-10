"""Shared timeout and retry policy for provider API calls."""

from __future__ import annotations

import json

import threading
import time
from typing import Any, Callable

_NON_RETRYABLE_MARKERS = ("api key", "authentication", "invalid_request", "permission")
# 2026-09-12: a 5xx from the gateway usually means a short upstream episode
# (Anthropic "Overloaded" -> CLIProxyAPI puts the credential in a ~1 minute
# cooldown and answers 503 auth_unavailable; Cloudflare 52x/530 at the
# edge). The default 2/4/8 s schedule ended inside the episode and the whole
# strict turn restarted, then parked in a 15-minute quota poll. Server
# errors and empty completions therefore retry on this longer schedule
# (OMNIQEC_LLM_5XX_BACKOFF_S="10,30,60,120" to change it).
_SERVER_ERROR_BACKOFF_S = (10.0, 30.0, 60.0, 120.0)


def server_error_backoff_s() -> tuple[float, ...]:
    import os

    raw = os.environ.get("OMNIQEC_LLM_5XX_BACKOFF_S", "").strip()
    if not raw:
        return _SERVER_ERROR_BACKOFF_S
    try:
        values = tuple(float(x) for x in raw.split(",") if x.strip())
    except ValueError:
        return _SERVER_ERROR_BACKOFF_S
    return values or _SERVER_ERROR_BACKOFF_S


def is_server_episode(exc: BaseException) -> bool:
    """A failure worth the long schedule: HTTP 5xx from the gateway or a 2xx
    body without choices (both are the upstream having a bad minute)."""

    if isinstance(exc, EmptyCompletion):
        return True
    status = getattr(exc, "status_code", None)
    try:
        return status is not None and int(status) >= 500
    except (TypeError, ValueError):
        return False

# Grace between the transport's own read timeout and the hard deadline that
# supersedes it, so a healthy call that merely runs long is never abandoned by
# the watchdog before the SDK gets its chance to time out and retry.
_DEADLINE_GRACE_S = 120.0


class EmptyCompletion(RuntimeError):
    """A 2xx completion whose body carries no `choices` (2026-09-11).

    Seen on the ChatGPT arm through the Worker -> CLIProxyAPI gateway: the
    SDK parsed the body into a ChatCompletion with ``choices=None`` and the
    client's ``response.choices[0]`` died with a bare
    ``TypeError: 'NoneType' object is not subscriptable`` that aborted the
    whole strict turn. Raised INSIDE the retried callable so the transport
    retries the call like any other transient failure; the message carries
    the body so the artifact says what the gateway actually returned.
    """


def require_choices(response: Any, *, provider: str) -> Any:
    """Return `response` if it has at least one choice, else raise EmptyCompletion."""
    choices = getattr(response, "choices", None)
    if choices:
        return response
    snippet = ""
    try:
        dump = getattr(response, "model_dump", None)
        payload = dump() if callable(dump) else response
        if isinstance(payload, dict):
            # The error field is what the operator needs; the SDK pads the
            # parsed body with null fields that would otherwise eat the budget.
            ordered = {k: v for k, v in payload.items() if k == "error"}
            ordered.update({k: v for k, v in payload.items() if k != "error" and v not in (None, [], {})})
            payload = ordered
        snippet = json.dumps(payload, default=str, ensure_ascii=False)[:600]
    except Exception:
        snippet = repr(response)[:600]
    raise EmptyCompletion(
        f"{provider} returned a completion without choices (choices={choices!r}); body: {snippet}"
    )


def build_timeout(config: Any) -> Any:
    """Transport timeout: a short connect, the configured budget for reads.

    A single float would apply the read budget to connect as well, so an
    unreachable endpoint would burn the full thinking-model window before
    failing.
    """
    connect = float(getattr(config, "connect_timeout_s", 20.0))
    read = float(config.timeout_s)
    try:
        import httpx
    except ImportError:  # transport not granular; the watchdog still applies
        return read
    return httpx.Timeout(read, connect=connect)


def per_call_deadline_s(config: Any) -> float:
    """Hard ceiling for one attempt, used when the transport timeout fails."""
    return float(config.timeout_s) + _DEADLINE_GRACE_S


class LLMCallTimeout(TimeoutError):
    """A provider call overran its hard deadline and was abandoned.

    Raised by `call_with_deadline` when the worker thread is still running past
    the deadline. The HTTP request keeps going in that thread until the socket
    gives up; the caller does not wait for it.
    """


def call_with_deadline(fn: Callable[[], Any], *, deadline_s: float | None) -> Any:
    """Run `fn`, abandoning it if it overruns `deadline_s`.

    Transport-level timeouts are necessary but not sufficient: an HTTP read
    timeout only fires when *nothing* arrives for the timeout window, so a
    server that trickles bytes — or a half-closed socket the client never
    notices — blocks the caller indefinitely. A campaign lost two hours to
    exactly that (2026-08-19, b60: the process sat in poll() on a CLOSE-WAIT
    socket while its 900s timeout never fired), so the wall clock gets the
    final say here.

    The worker is a daemon thread: an abandoned call cannot keep the process
    alive at exit, and it releases its connection once the socket errors out.
    """
    if not deadline_s or deadline_s <= 0:
        return fn()

    box: dict[str, Any] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # forwarded to the caller verbatim
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run, name="llm-call", daemon=True).start()
    if not done.wait(deadline_s):
        raise LLMCallTimeout(
            f"provider call exceeded its {deadline_s:.0f}s hard deadline and was abandoned"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def call_with_retries(
    fn: Callable[[], Any],
    *,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    per_call_deadline_s: float | None = None,
    total_deadline_s: float | None = None,
) -> Any:
    """Run `fn`, retrying transient failures with exponential backoff.

    Configuration errors (missing/invalid key, bad request) re-raise
    immediately: retrying them only burns time.

    `per_call_deadline_s` caps any single attempt and `total_deadline_s` caps
    the whole sequence, so a logical call cannot quietly stack N attempts of
    the transport timeout into hours of dead wall clock. Provider SDKs should
    be constructed with their own retries disabled — otherwise each attempt
    here silently becomes several, multiplying both budgets.
    """

    started = time.monotonic()
    server_schedule = server_error_backoff_s()
    # Server episodes get the longer schedule and at least as many attempts
    # as that schedule has delays; everything else keeps max_retries.
    max_attempts = max(0, int(max_retries)) + 1
    server_attempts = max(max_attempts, len(server_schedule) + 1)

    def _budget_left() -> float | None:
        if not total_deadline_s or total_deadline_s <= 0:
            return None
        return total_deadline_s - (time.monotonic() - started)

    for attempt in range(server_attempts):
        remaining = _budget_left()
        deadline = per_call_deadline_s
        if remaining is not None:
            if remaining <= 0:
                raise LLMCallTimeout(
                    f"provider call budget of {total_deadline_s:.0f}s exhausted "
                    f"after {attempt} attempt(s)"
                )
            deadline = remaining if deadline is None else min(deadline, remaining)
        try:
            return call_with_deadline(fn, deadline_s=deadline)
        except Exception as exc:
            text = str(exc).lower()
            # An EmptyCompletion is retried whatever its body says: 2026-09-11
            # the gateway wrapped a ChatGPT stream disconnect ("stream closed
            # before response.completed") in a 200 body typed
            # invalid_request_error, and the marker match below turned that
            # transient into an immediate strict-turn abort.
            non_retryable = isinstance(exc, (ImportError, KeyboardInterrupt)) or (
                not isinstance(exc, EmptyCompletion)
                and any(marker in text for marker in _NON_RETRYABLE_MARKERS)
            )
            episode = is_server_episode(exc)
            last_attempt = attempt >= (server_attempts - 1 if episode else max_attempts - 1)
            if non_retryable or last_attempt:
                raise
            if episode:
                delay = float(server_schedule[min(attempt, len(server_schedule) - 1)])
            else:
                delay = float(backoff_s) * (2**attempt)
            remaining = _budget_left()
            if remaining is not None:
                if remaining <= 0:
                    raise
                delay = min(delay, remaining)
            time.sleep(delay)
    raise RuntimeError("unreachable")
