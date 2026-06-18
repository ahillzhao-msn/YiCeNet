"""YiCeNet Claude Code hook runner.
Dispatched by UserPromptSubmit / Stop hooks.
Reads event from YICENET_HOOK_EVENT env var or first argv.

Mode is auto-detected at runtime:
  - daemon port file exists and daemon responds → IPC path (Mode 3)
  - otherwise → subprocess path (Mode 1)

YICENET_MODE=subprocess forces subprocess regardless of daemon state.
YICENET_MODE=hybrid is accepted but treated the same as auto-detect.

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
mode = os.environ.get("YICENET_MODE", "auto")  # "auto" | "subprocess" | "hybrid"

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

# ── Auto / hybrid: try IPC first, fall back to subprocess ────────────────────

if mode != "subprocess":
    from yicenet.tools.ipc_hook import pre_message_send_ipc, stop_ipc

    if event == "pre":
        if pre_message_send_ipc(_payload):
            sys.exit(0)
        # Daemon unreachable — fall through to subprocess path below.
    elif event == "stop":
        stop_ipc(_payload)  # best-effort; subprocess stop below if needed
        sys.exit(0)

# ── Subprocess mode (Mode 1) ──────────────────────────────────────────────────

from yicenet.tools.claude_hook import ClaudeCodeAdapter
from yicenet.memory_bank import configure_memory_bank_for

_adapter = ClaudeCodeAdapter()
configure_memory_bank_for(_adapter)

if event == "pre":
    _adapter.pre_message_send(_payload)
elif event == "post_tool":
    pass  # no-op: feedback requires next user message; handled at pre time
elif event == "stop":
    _adapter.stop(_payload)
