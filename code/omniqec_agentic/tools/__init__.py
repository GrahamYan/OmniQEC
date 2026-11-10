"""Policy-agent tool suite (03 §4): read / draft / policy scopes.

Modules self-register with orchestrator.registry on import; the registry's
``render_tool_docs`` turns their docstrings into the manual's static tool
catalog (single source of truth = docstring)."""

from .context import ToolContext

__all__ = ["ToolContext"]
