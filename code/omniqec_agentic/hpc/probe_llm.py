"""Provider reachability probe — the exact client stack a campaign uses.

A curl to the gateway proves the firewall is passable; it does not prove the
campaign can talk to the model. This probe goes through
`llm.factory.create_completion_client`, so it exercises the same base_url
resolution, timeout policy, retry wrapper and usage parsing that the run will,
and a green result here means the run's LLM path is genuinely wired.

Usage:
    python -m omniqec_agentic.hpc.probe_llm deepseek openai
    python -m omniqec_agentic.hpc.probe_llm --all

Exit code is the number of providers that failed, so it drops straight into a
PBS script: `python -m omniqec_agentic.hpc.probe_llm deepseek || exit 3`.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from typing import Any

from omniqec_agentic.config import LLMConfig
from omniqec_agentic.llm.factory import SUPPORTED_LLM_PROVIDERS

# One cheap round trip. Reasoning models bill thinking to the completion
# budget, so a tight max_tokens returns EMPTY content and a probe that checks
# for text would fail a working provider (this killed six jobs on 2026-08-20).
PROBE_PROMPT = "Reply with exactly: OK"
PROBE_MAX_TOKENS = 512

# Which env vars actually decide where a provider's traffic goes. Printed so a
# failure shows the routing that produced it instead of hiding it.
ROUTING_ENV = {
    "deepseek": ("DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "DEEPSEEK_API_KEY"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"),
    "claude": ("ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "ANTHROPIC_API_KEY"),
    "openrouter": ("OPENROUTER_BASE_URL", "OPENROUTER_MODEL",
                   "OPENROUTER_MAX_TOKENS", "OPENROUTER_API_KEY"),
}


def _redact(name: str, value: str) -> str:
    if not value:
        return "(unset)"
    if name.endswith("_API_KEY"):
        return f"set, {len(value)} chars, ...{value[-4:]}"
    return value


def probe(provider: str, *, max_tokens: int = PROBE_MAX_TOKENS) -> dict[str, Any]:
    from pathlib import Path

    from omniqec_agentic.config import load_dotenv

    load_dotenv()
    # OpenRouter keeps its credential in its own file so it is never mixed with
    # another provider's; the client loads it too, and a probe that did not
    # would report the key as unset on a call that just succeeded.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env.openrouter")
    result: dict[str, Any] = {"provider": provider, "ok": False}
    result["routing"] = {
        name: _redact(name, os.environ.get(name, "").strip())
        for name in ROUTING_ENV.get(provider, ())
    }
    started = time.perf_counter()
    try:
        config = LLMConfig(provider=provider, max_tokens=max_tokens)
        client = create_client(provider, config)
        text = client.complete(system="You are a connectivity probe.", user=PROBE_PROMPT)
        info = dict(client.info())
        result["ok"] = True
        result["model"] = info.get("model")
        result["reply"] = (text or "").strip()[:80]
        usage = info.get("usage") or {}
        if usage:
            result["tokens"] = {
                "in": usage.get("input_tokens"),
                "out": usage.get("output_tokens"),
            }
            if usage.get("cost_usd") is not None:
                result["cost_usd"] = usage.get("cost_usd")
        # A router can answer from a different model than the one requested.
        if info.get("served_model") and info["served_model"] != info.get("model"):
            result["served"] = f"{info['served_model']} via {info.get('served_upstream')}"
        elif info.get("served_upstream"):
            result["served"] = f"upstream={info['served_upstream']}"
        # An empty reply is a real answer from a reasoning model that spent its
        # budget thinking, not a failure: the call completed.
        if not result["reply"]:
            result["note"] = "empty content (reasoning model consumed max_tokens)"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def create_client(provider: str, config: LLMConfig):
    from omniqec_agentic.llm.factory import create_completion_client

    return create_completion_client(provider, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("providers", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="probe every supported provider")
    parser.add_argument("--max-tokens", type=int, default=PROBE_MAX_TOKENS)
    args = parser.parse_args()

    providers = list(SUPPORTED_LLM_PROVIDERS) if args.all else (args.providers or ["deepseek"])
    print(f"host={socket.gethostname()}  python={sys.version.split()[0]}")
    failures = 0
    for provider in providers:
        outcome = probe(provider, max_tokens=args.max_tokens)
        mark = "OK  " if outcome["ok"] else "FAIL"
        print(f"\n[{mark}] {provider}  ({outcome['seconds']}s)")
        for name, value in (outcome.get("routing") or {}).items():
            print(f"        {name} = {value}")
        if outcome["ok"]:
            print(f"        model   = {outcome.get('model')}")
            print(f"        reply   = {outcome.get('reply')!r}")
            if outcome.get("tokens"):
                print(f"        tokens  = {outcome['tokens']}")
            if outcome.get("cost_usd") is not None:
                print(f"        cost    = ${outcome['cost_usd']:.6f}")
            if outcome.get("served"):
                print(f"        served  = {outcome['served']}")
            if outcome.get("note"):
                print(f"        note    = {outcome['note']}")
        else:
            print(f"        error   = {outcome.get('error')}")
            failures += 1
    print(f"\n{len(providers) - failures}/{len(providers)} providers reachable")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
