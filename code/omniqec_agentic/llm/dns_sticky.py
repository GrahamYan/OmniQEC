"""Sticky DNS for the LLM gateway hosts (2026-09-11).

NSCC's resolvers (and Cloudflare's edge, seen as HTTP 530 "error code:
1016" through the Worker) intermittently fail to resolve the Tailscale Funnel
name of the Lab CLIProxyAPI, in windows that last minutes. Every layer above
then reports "Connection error": the openai SDK raises APIConnectionError,
the strict turn aborts, three retries burn in a couple of minutes and the
link parks in a 15-minute quota poll -- for a name that resolved fine a
quarter of an hour earlier and whose ingress IPs did not change.

`install(hosts)` wraps socket.getaddrinfo for the named hosts only:

* a successful answer is remembered (per host / family / type / proto);
* when resolution fails, the last good answer is served, re-addressed to the
  requested port;
* with no good answer yet, `OMNIQEC_DNS_PINS` ("host=ip1,ip2;host2=ip") seeds
  the fallback, so a process started inside an outage still connects.

TLS is unaffected: the SNI and certificate check still use the hostname, the
pin only replaces the A/AAAA lookup. IPv4 answers are ordered first because
the cluster has no IPv6 route. Hosts not listed resolve exactly as before.
"""

from __future__ import annotations

import os
import socket
import threading
from typing import Any, Iterable

_lock = threading.Lock()
_hosts: set[str] = set()
_good: dict[tuple[str, int, int, int], list[tuple]] = {}  # (host, family, type, proto) -> entries
_original: Any = None
_installed = False

PINS_ENV = "OMNIQEC_DNS_PINS"


def parse_pins(text: str | None) -> dict[str, list[str]]:
    pins: dict[str, list[str]] = {}
    for chunk in (text or "").replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        host, _, ips = chunk.partition("=")
        host = host.strip().lower().rstrip(".")
        addrs = [ip.strip() for ip in ips.split(",") if ip.strip()]
        if host and addrs:
            pins[host] = addrs
    return pins


def _readdress(entries: list[tuple], port: int) -> list[tuple]:
    out = []
    for family, type_, proto, canon, sockaddr in entries:
        if family == socket.AF_INET6 and len(sockaddr) == 4:
            sockaddr = (sockaddr[0], port, sockaddr[2], sockaddr[3])
        else:
            sockaddr = (sockaddr[0], port)
        out.append((family, type_, proto, canon, sockaddr))
    return out


def _pinned_entries(host: str, port: int, family: int, type_: int, proto: int) -> list[tuple]:
    entries: list[tuple] = []
    for ip in parse_pins(os.environ.get(PINS_ENV)).get(host, []):
        try:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            if family not in (0, socket.AF_UNSPEC, fam):
                continue
            t = type_ or socket.SOCK_STREAM
            p = proto or (socket.IPPROTO_TCP if t == socket.SOCK_STREAM else 0)
            sockaddr = (ip, port, 0, 0) if fam == socket.AF_INET6 else (ip, port)
            entries.append((fam, t, p, "", sockaddr))
        except Exception:
            continue
    return entries


def _v4_first(entries: list[tuple]) -> list[tuple]:
    return sorted(entries, key=lambda e: 0 if e[0] == socket.AF_INET else 1)


def _sticky_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002 - socket's signature
    name = host.decode() if isinstance(host, bytes) else host
    name = (name or "").lower().rstrip(".")
    if name not in _hosts:
        return _original(host, port, family, type, proto, flags)
    port_int = int(port or 0)
    key = (name, int(family), int(type), int(proto))
    try:
        result = _original(host, port, family, type, proto, flags)
    except (socket.gaierror, OSError) as exc:
        with _lock:
            cached = _good.get(key)
        if cached:
            return _readdress(cached, port_int)
        pinned = _pinned_entries(name, port_int, int(family), int(type), int(proto))
        if pinned:
            return _v4_first(pinned)
        raise exc
    result = _v4_first(list(result))
    with _lock:
        _good[key] = list(result)
    return result


def install(hosts: Iterable[str]) -> list[str]:
    """Make resolution sticky for `hosts` (idempotent). Returns the host set."""
    global _original, _installed
    with _lock:
        for h in hosts:
            h = (h or "").strip().lower().rstrip(".")
            if h and not h.replace(".", "").isdigit():  # skip bare IPv4 literals
                _hosts.add(h)
        for h in parse_pins(os.environ.get(PINS_ENV)):
            _hosts.add(h)
        if not _installed:
            _original = socket.getaddrinfo
            socket.getaddrinfo = _sticky_getaddrinfo
            _installed = True
        return sorted(_hosts)


def host_of(url: str) -> str:
    from urllib.parse import urlsplit

    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def _reset_for_tests() -> None:
    global _installed, _original
    with _lock:
        if _installed and _original is not None:
            socket.getaddrinfo = _original
        _installed = False
        _original = None
        _hosts.clear()
        _good.clear()
