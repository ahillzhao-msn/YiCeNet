"""YiCeNet daemon — independent HTTP server for hook IPC.

Runs as a standalone background process, NOT a thread inside MCP.
Spawned by daemon.launcher on first hook request; self-terminates
after IDLE_TIMEOUT_S seconds of inactivity.

Endpoints:
  GET  /health         — liveness probe (returns {"ok": true})
  POST /hook/pre       — run predict_for_turn_payload; return JSON
  POST /hook/post_tool — feed tool data into context collector
  POST /hook/stop      — run on_turn_complete

Port: YICENET_DAEMON_PORT env var → config → DEFAULT_PORT (7788).
Port written to PORT_FILE; PID written to PID_FILE.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DEFAULT_PORT = 7788
IDLE_TIMEOUT_S = 1800  # 30 minutes
PORT_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.port"
PID_FILE = Path(tempfile.gettempdir()) / "yicenet-daemon.pid"

_last_request_time: float = time.monotonic()
_request_lock = threading.Lock()

# Lazy singleton adapter — created on first IPC request.
_adapter = None
_adapter_lock = threading.Lock()


def _touch_activity() -> None:
    global _last_request_time
    with _request_lock:
        _last_request_time = time.monotonic()


def _hook_adapter():
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                from yicenet.tools.claude_hook import ClaudeCodeAdapter
                from yicenet.memory_bank import configure_memory_bank_for
                adapter = ClaudeCodeAdapter(process_model="daemon")
                configure_memory_bank_for(adapter)
                _adapter = adapter
    return _adapter


class _HookHandler(BaseHTTPRequestHandler):

    def do_GET(self) -> None:
        if self.path == "/health":
            _touch_activity()
            self._send(200, {"ok": True, "pid": os.getpid()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        _touch_activity()
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            payload: dict = json.loads(body) if body else {}
        except Exception:
            self._send(400, {"error": "bad request"})
            return

        if self.path == "/hook/pre":
            self._handle_pre(payload)
        elif self.path == "/hook/post_tool":
            self._handle_post_tool(payload)
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

    def _handle_post_tool(self, payload: dict) -> None:
        try:
            adapter = _hook_adapter()
            if adapter.ctx is not None:
                adapter.ctx.sniff_tool(
                    name=payload.get("tool_name", ""),
                    exit_code=payload.get("exit_code", 0),
                    duration_ms=payload.get("duration_ms", 0),
                    result_size_bytes=payload.get("result_size", 0),
                )
            self._send(200, {"ok": True})
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
        pass


# ── Idle watchdog ────────────────────────────────────────────────────────────


def _idle_watchdog(timeout_s: int) -> None:
    """Background thread: exit the process after idle timeout."""
    while True:
        time.sleep(60)
        with _request_lock:
            idle = time.monotonic() - _last_request_time
        if idle >= timeout_s:
            sys.stderr.write(
                f"[YiCeNet daemon] idle {idle:.0f}s >= {timeout_s}s, shutting down\n"
            )
            _cleanup()
            os._exit(0)


# ── Lifecycle ────────────────────────────────────────────────────────────────


def _resolve_port() -> int:
    port = int(os.environ.get("YICENET_DAEMON_PORT", 0))
    if port:
        return port
    try:
        from yicenet.config import get_platform_config
        cfg = get_platform_config("claude-code")
        port = int(cfg.get("daemon", {}).get("port", 0))
    except Exception:
        pass
    return port or DEFAULT_PORT


def _cleanup() -> None:
    for f in (PORT_FILE, PID_FILE):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


def run_standalone(port: int = 0, idle_timeout: int = IDLE_TIMEOUT_S) -> None:
    """Run the hook server as a standalone daemon process.

    Writes PID and port files, starts idle watchdog, serves until killed
    or idle timeout expires.
    """
    if port == 0:
        port = _resolve_port()

    try:
        srv = HTTPServer(("127.0.0.1", port), _HookHandler)
    except OSError:
        try:
            srv = HTTPServer(("127.0.0.1", 0), _HookHandler)
        except OSError:
            sys.exit(1)

    actual_port = srv.server_address[1]

    # Write identity files
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        PORT_FILE.write_text(str(actual_port), encoding="utf-8")
    except Exception:
        pass

    atexit.register(_cleanup)

    # Start idle watchdog
    wd = threading.Thread(
        target=_idle_watchdog,
        args=(idle_timeout,),
        name="yicenet-idle-watchdog",
        daemon=True,
    )
    wd.start()

    sys.stderr.write(
        f"[YiCeNet daemon] pid={os.getpid()} port={actual_port} "
        f"idle_timeout={idle_timeout}s\n"
    )

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        _cleanup()


if __name__ == "__main__":
    # Reconfigure stdout/stderr to UTF-8 for CJK on Windows.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    run_standalone()
