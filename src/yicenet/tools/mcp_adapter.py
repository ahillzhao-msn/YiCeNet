"""MCPAdapter — PlatformAdapter for the MCP server (daemon process model).

MCP tools supply structured arguments rather than raw Claude Code hook
payloads, so field extraction is simple pass-through.  assistant_response
is populated only via yicenet_turn_complete (explicit, not inferred from
transcript).
"""
from __future__ import annotations


class MCPAdapter:
    """Translates MCP tool arguments to HookOrchestrator domain primitives."""

    platform_id = "claude-code-mcp"
    process_model = "daemon"

    def session_id(self, payload: dict) -> str:
        return payload.get("session_id", "")

    def turn_id(self, payload: dict) -> int:
        return int(payload.get("turn_id", 0))

    def prompt(self, payload: dict) -> str:
        return payload.get("task_brief", "") or payload.get("prompt", "")

    def assistant_response(self, payload: dict) -> str:
        return payload.get("response_snippet", "")

    def platform_signals(self, payload: dict):
        signals = payload.get("signals")
        return signals if isinstance(signals, dict) else None
