"""HTTP side-channel for hook IPC in hybrid mode (Mode 3).

The MCP server starts this in a background daemon thread at session open.
The thin hook client (ipc_hook.py) connects to it instead of cold-starting
the engine, keeping hook latency under 100ms.

Endpoints:
  POST /hook/pre   — run predict_for_turn_payload; return JSON for stdout injection
  POST /hook/stop  — run on_turn_complete; return {"ok": true}

Port selection: YICENET_DAEMON_PORT env var → config → DEFAULT_PORT (7788).
Port is written to PORT_FILE so the hook client can discover it without env.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DEFAULT_PORT = 7788
PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"

_server: HTTPServer | None = None
_server_lock = threading.Lock()

# Lazy singleton — created on first IPC request, not at import time.
# Uses process_model="daemon" so HookOrchestrator never flushes MemoryBank
# sessions between IPC calls (the daemon is the long-lived owner of state).
_adapter = None
_adapter_lock = threading.Lock()


def _hook_adapter():
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                from yicenet.tools.claude_hook import ClaudeCodeAdapter
                _adapter = ClaudeCodeAdapter(process_model="daemon")
    return _adapter


class _HookHandler(BaseHTTPRequestHandler):

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            payload: dict = json.loads(body) if body else {}
        except Exception:
            self._send(400, {"error": "bad request"})
            return

        if self.path == "/hook/pre":
            self._handle_pre(payload)
        elif self.path == "/hook/stop":
            self._handle_stop(payload)
        else:
            self._send(404, {"error": "not found"})

    def _handle_pre(self, payload: dict) -> None:
        try:
            result = _hook_adapter().predict_for_turn_payload(payload)
            self._send(200, result if result is not None else {})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def _handle_stop(self, payload: dict) -> None:
        try:
            _hook_adapter().stop(payload)
            self._send(200, {"ok": True})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass  # suppress access log; errors still go to stderr


def start_hook_server(port: int = 0) -> int:
    """Start the HTTP hook server in a daemon thread.

    Safe to call multiple times — idempotent after the first call.
    Returns the actual bound port (0 if startup failed).
    """
    global _server
    with _server_lock:
        if _server is not None:
            return _server.server_address[1]

        if port == 0:
            try:
                from yicenet.config import get_platform_config
                cfg = get_platform_config("claude-code")
                port = int(cfg.get("daemon", {}).get("port", 0))
            except Exception:
                port = 0
        if port == 0:
            port = int(os.environ.get("YICENET_DAEMON_PORT", DEFAULT_PORT))

        try:
            srv = HTTPServer(("127.0.0.1", port), _HookHandler)
        except OSError:
            # Port in use — let OS pick a free one
            try:
                srv = HTTPServer(("127.0.0.1", 0), _HookHandler)
            except OSError:
                return 0

        actual_port = srv.server_address[1]
        _server = srv

        try:
            PORT_FILE.write_text(str(actual_port), encoding="utf-8")
        except Exception:
            pass

        t = threading.Thread(target=srv.serve_forever, name="yicenet-hook-server", daemon=True)
        t.start()
        return actual_port


def stop_hook_server() -> None:
    """Shut down the hook server and remove the port file."""
    global _server
    with _server_lock:
        if _server is not None:
            _server.shutdown()
            _server = None
    try:
        PORT_FILE.unlink(missing_ok=True)
    except Exception:
        pass
