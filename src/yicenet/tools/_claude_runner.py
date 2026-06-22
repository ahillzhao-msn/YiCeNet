"""YiCeNet Claude Code hook runner.
Dispatched by UserPromptSubmit / PostToolUse / Stop hooks.
Reads event from YICENET_HOOK_EVENT env var or first argv.

Mode priority:
  - YICENET_MODE=daemon (default) → IPC path (Mode 3)
  - YICENET_MODE=subprocess → subprocess path (Mode 1)
  - YICENET_MODE=auto|hybrid → try IPC, fall back to subprocess

This file is the source of truth for what gets installed at
~/.claude/hooks/yicenet_claude_hook.py by ClaudeCodeInstaller.
Edit here; re-run install to deploy.
"""
import sys
import os
import json

# Reconfigure stdout/stderr to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

event = os.environ.get("YICENET_HOOK_EVENT") or (sys.argv[1] if len(sys.argv) > 1 else "pre")
mode = os.environ.get("YICENET_MODE", "daemon")  # "daemon" | "subprocess" | "auto" | "hybrid"

# Read stdin once and pass explicitly — avoids double-read across mode paths.
_raw = ""
try:
    _raw = sys.stdin.read().strip()
except Exception:
    pass
_payload: dict = {}
try:
    _payload = json.loads(_raw) if _raw else {}
except Exception:
    pass

# ── Daemon / auto / hybrid: try IPC first, fall back to subprocess ──────────

if mode != "subprocess":
    from yicenet.tools.ipc_hook import pre_message_send_ipc, stop_ipc, post_tool_ipc

    if event == "pre":
        if pre_message_send_ipc(_payload):
            sys.exit(0)
        # Daemon unreachable — fall through to subprocess path below.
    elif event == "post_tool":
        if post_tool_ipc(_payload):
            sys.exit(0)
        # Daemon unreachable — fall through to subprocess path below.
    elif event == "stop":
        if stop_ipc(_payload):
            sys.exit(0)
        # Daemon unreachable — fall through to subprocess path below.

# ── Subprocess mode (Mode 1) ────────────────────────────────────────────────

from yicenet.tools.claude_hook import ClaudeCodeAdapter
from yicenet.memory_bank import configure_memory_bank_for

_adapter = ClaudeCodeAdapter()
configure_memory_bank_for(_adapter)

if event == "pre":
    _adapter.pre_message_send(_payload)
elif event == "post_tool":
    from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
    sid = _adapter.session_id(_payload)
    tid = _adapter.turn_id(_payload)
    col = SubprocessContextCollector(sid, tid)
    col.sniff_tool(
        name=_payload.get("tool_name", ""),
        exit_code=_payload.get("exit_code", 0),
        duration_ms=_payload.get("duration_ms", 0),
        result_size_bytes=_payload.get("result_size", 0),
    )
elif event == "stop":
    from yicenet.hook_engine.collector.subprocess import SubprocessContextCollector
    sid = _adapter.session_id(_payload)
    tid = _adapter.turn_id(_payload)
    _adapter._ctx = SubprocessContextCollector(sid, tid)
    _adapter.stop(_payload)
