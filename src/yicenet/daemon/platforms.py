"""Per-platform hook semantics for the daemon.

Every agent CLI (Claude Code, Kimi Code, ...) calls the same dumb client
(`yicenet-hook <platform> <event>`, native binary or Python fallback), which
forwards the raw stdin bytes to `POST /hook/<event>?platform=<platform>` and
copies the response body to stdout verbatim.  Everything that differs between
platforms — event names, payload fields, what goes to stdout — lives here.

A handler takes the decoded payload and returns the exact stdout bytes
(b"" for "print nothing").
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Dict

Handler = Callable[[dict], bytes]

DEFAULT_PLATFORM = "claude-code"

_routes: Dict[str, Dict[str, Handler]] = {}
_lock = threading.Lock()


class UnknownRoute(LookupError):
    pass


def _json(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _configure_memory(adapter) -> None:
    # The MemoryBank is a process singleton: the first platform served configures it.
    from yicenet.memory_bank import configure_memory_bank_for
    configure_memory_bank_for(adapter)


# ── Claude Code ──────────────────────────────────────────────────────────────


def _claude_routes() -> Dict[str, Handler]:
    from yicenet.tools.claude_hook import ClaudeCodeAdapter

    adapter = ClaudeCodeAdapter(process_model="daemon")
    _configure_memory(adapter)

    def pre(payload: dict) -> bytes:
        # UserPromptSubmit: plain stdout is appended to the model's context.
        result = adapter.predict_for_turn_payload(payload)
        return b"[YiCeNet] " + _json(result if result is not None else {})

    def post_tool(payload: dict) -> bytes:
        if adapter.ctx is not None:
            adapter.ctx.sniff_tool(
                name=payload.get("tool_name", ""),
                exit_code=payload.get("exit_code", 0),
                duration_ms=payload.get("duration_ms", 0),
                result_size_bytes=payload.get("result_size", 0),
            )
        return b""

    def stop(payload: dict) -> bytes:
        adapter.stop(payload)
        return b""

    return {"pre": pre, "post_tool": post_tool, "stop": stop}


# ── Kimi Code ────────────────────────────────────────────────────────────────


def _kimi_routes() -> Dict[str, Handler]:
    from yicenet.tools import kimi_code_hook as kimi

    # The module-level handlers all go through kimi._adapter; swap in a daemon-lifetime one.
    kimi._adapter = kimi.KimiCodeAdapter(process_model="daemon")
    _configure_memory(kimi._adapter)

    def pre_message_send(payload: dict) -> bytes:
        result = kimi._adapter.predict_for_turn_payload(payload)
        return _json(result) if result is not None else b""

    def observe(fn: Callable[[dict], None]) -> Handler:
        def handler(payload: dict) -> bytes:
            try:
                fn(payload)
            except SystemExit:  # CLI-style handlers end with sys.exit(0)
                pass
            return b""
        return handler

    routes = {name: observe(fn) for name, fn in kimi._COMMANDS.items()}
    routes["pre_message_send"] = pre_message_send
    return routes


_FACTORIES: Dict[str, Callable[[], Dict[str, Handler]]] = {
    "claude-code": _claude_routes,
    "kimi-code": _kimi_routes,
}


def platforms() -> list:
    return sorted(_FACTORIES)


def handler_for(platform: str, event: str) -> Handler:
    """Resolve (platform, event) to a handler; builds the platform adapter on first use."""
    factory = _FACTORIES.get(platform)
    if factory is None:
        raise UnknownRoute(f"unknown platform {platform!r}")
    routes = _routes.get(platform)
    if routes is None:
        with _lock:
            routes = _routes.get(platform)
            if routes is None:
                routes = _routes[platform] = factory()
    handler = routes.get(event)
    if handler is None:
        raise UnknownRoute(f"unknown event {event!r} for {platform}")
    return handler
