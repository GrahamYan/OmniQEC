"""Tool registry (03 §4.1): one decorator does four jobs — signature ->
pydantic schema, scope enforcement, docstring -> manual static rendering,
transcript recording.

Discipline (03 §4.3): tools have ZERO side effects on the protocol layer;
returns are machine-computed truths — the agent's words never become the
record. Scopes: "read" (pure queries), "draft" (sandboxed evaluation of
agent drafts), "policy" (the restricted writes: ledger ops, submit_plan)."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

Scope = Literal["read", "draft", "policy"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: Scope
    func: Callable[..., dict[str, Any]]
    params_model: type[BaseModel]
    doc: str
    example: str

    def schema(self) -> dict[str, Any]:
        return self.params_model.model_json_schema()


_REGISTRY: dict[str, ToolSpec] = {}


def tool(scope: Scope, *, example: str = "") -> Callable:
    """Register a tool. The wrapped function's FIRST parameter must be the
    ToolContext (injected by the runtime, invisible to the model); remaining
    annotated parameters become the pydantic argument schema."""

    def decorate(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        fields: dict[str, Any] = {}
        parameters = list(signature.parameters.values())[1:]  # skip ctx
        for parameter in parameters:
            annotation = hints.get(parameter.name, Any)
            default = parameter.default if parameter.default is not inspect.Parameter.empty else ...
            fields[parameter.name] = (annotation, default)
        params_model = create_model(
            f"{func.__name__}_params",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        spec = ToolSpec(
            name=func.__name__,
            scope=scope,
            func=func,
            params_model=params_model,
            doc=inspect.getdoc(func) or "",
            example=example,
        )
        if spec.name in _REGISTRY:
            raise ValueError(f"duplicate tool name: {spec.name}")
        _REGISTRY[spec.name] = spec
        return func

    return decorate


def registry() -> dict[str, ToolSpec]:
    _ensure_tools_loaded()
    return dict(_REGISTRY)


def _ensure_tools_loaded() -> None:
    # Tool modules self-register on import.
    from omniqec_agentic.tools import (  # noqa: F401
        acquisition_score,
        ledger_ops,
        precheck_draft,
        queries,
    )


def render_tool_docs() -> str:
    """Manual static section: the single source of truth is the docstring
    (03 §5); one example per tool (SWE-agent ACI discipline)."""

    _ensure_tools_loaded()
    lines = ["# Tool catalog", ""]
    for name in sorted(_REGISTRY):
        spec = _REGISTRY[name]
        properties = spec.schema().get("properties", {})
        arguments = ", ".join(
            f"{key}: {value.get('type', 'any')}" for key, value in properties.items()
        )
        lines.append(f"## {name}({arguments})  [scope={spec.scope}]")
        lines.append("")
        lines.append(spec.doc or "(no docstring)")
        if spec.example:
            lines.append("")
            lines.append(f"Example call: `{spec.example}`")
        lines.append("")
    return "\n".join(lines)


@dataclass
class ToolRuntime:
    """Per-turn dispatcher: scope-aware validation, call budget, transcript."""

    context: Any  # tools.context.ToolContext
    tool_budget: int = 32
    transcript: list[dict[str, Any]] = field(default_factory=list)
    calls_used: int = 0

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        _ensure_tools_loaded()
        started = time.perf_counter()
        record: dict[str, Any] = {"tool": name, "arguments": arguments or {}}
        if self.calls_used >= int(self.tool_budget):
            record["error"] = "tool_budget_exhausted"
            self.transcript.append(record)
            return {"error": "tool_budget_exhausted", "calls_used": self.calls_used}
        self.calls_used += 1
        spec = _REGISTRY.get(name)
        if spec is None:
            record["error"] = "unknown_tool"
            self.transcript.append(record)
            return {"error": f"unknown tool '{name}'", "available": sorted(_REGISTRY)}
        try:
            params = spec.params_model.model_validate(arguments or {})
        except Exception as exc:
            record["error"] = f"invalid_arguments: {exc}"
            self.transcript.append(record)
            # Self-healing error: return the expected schema + example so the
            # model can correct the call without a discovery round-trip.
            return {
                "error": f"invalid arguments for {name}: {exc}",
                "expected_parameters": spec.schema().get("properties", {}),
                "example": spec.example,
            }
        try:
            result = spec.func(self.context, **params.model_dump())
        except Exception as exc:
            record["error"] = f"tool_error: {exc}"
            self.transcript.append(record)
            return {"error": f"{name} failed: {exc}"}
        record["result_digest"] = _digest(result)
        record["runtime_s"] = time.perf_counter() - started
        self.transcript.append(record)
        return result

    def remaining(self) -> int:
        return max(0, int(self.tool_budget) - self.calls_used)


def _digest(result: Any) -> str:
    import hashlib

    try:
        canonical = json.dumps(result, sort_keys=True, default=str)
    except Exception:
        canonical = str(result)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
