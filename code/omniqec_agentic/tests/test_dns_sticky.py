"""Sticky DNS for the gateway host (2026-09-11): a name that resolved a
moment ago keeps resolving through a resolver outage, and a process born
inside the outage falls back to OMNIQEC_DNS_PINS."""

from __future__ import annotations

import socket

import pytest

from omniqec_agentic.llm import dns_sticky

HOST = "cliproxy.example.net"


@pytest.fixture(autouse=True)
def _clean():
    dns_sticky._reset_for_tests()
    yield
    dns_sticky._reset_for_tests()


def _answer(ip: str, port: int):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


def test_last_good_answer_survives_a_resolver_outage(monkeypatch):
    calls = {"n": 0}

    def fake(host, port, family=0, type=0, proto=0, flags=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _answer("10.0.0.5", port)
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    dns_sticky.install([HOST])
    assert socket.getaddrinfo(HOST, 443)[0][4] == ("10.0.0.5", 443)
    # outage: the cached answer comes back, re-addressed to the requested port
    assert socket.getaddrinfo(HOST, 8443)[0][4] == ("10.0.0.5", 8443)
    assert calls["n"] == 2


def test_pins_seed_a_process_born_inside_the_outage(monkeypatch):
    def always_fails(host, port, family=0, type=0, proto=0, flags=0):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", always_fails)
    monkeypatch.setenv(dns_sticky.PINS_ENV, f"{HOST}=2001:db8::1,203.0.113.7")
    dns_sticky.install([HOST])
    res = socket.getaddrinfo(HOST, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
    # IPv4 first: the cluster has no IPv6 route
    assert res[0][0] == socket.AF_INET and res[0][4] == ("203.0.113.7", 443)
    assert res[1][0] == socket.AF_INET6 and res[1][4][:2] == ("2001:db8::1", 443)


def test_unlisted_hosts_and_unpinned_failures_are_untouched(monkeypatch):
    def always_fails(host, port, family=0, type=0, proto=0, flags=0):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", always_fails)
    monkeypatch.delenv(dns_sticky.PINS_ENV, raising=False)
    dns_sticky.install([HOST])
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("other.example", 443)
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo(HOST, 443)  # listed, but nothing good and no pin


def test_parse_pins_and_host_of():
    assert dns_sticky.parse_pins("a.example=1.2.3.4, 5.6.7.8; b.example=9.9.9.9") == {
        "a.example": ["1.2.3.4", "5.6.7.8"],
        "b.example": ["9.9.9.9"],
    }
    assert dns_sticky.host_of("https://gateway.example.org/v1") == "gateway.example.org"


def test_openai_client_installs_the_hook_for_its_base_url(monkeypatch):
    from omniqec_agentic.config import LLMConfig
    from omniqec_agentic.llm.openai_compat import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"https://{HOST}/v1")
    OpenAIClient(LLMConfig())
    assert HOST in dns_sticky._hosts
    assert socket.getaddrinfo is dns_sticky._sticky_getaddrinfo
