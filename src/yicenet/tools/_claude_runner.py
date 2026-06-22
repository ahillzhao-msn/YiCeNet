"""YiCeNet Claude Code hook runner.
Dispatched by UserPromptSubmit / PostToolUse / Stop hooks.
Reads event from YICENET_HOOK_EVENT env var or first argv.

Modes:
  - YICENET_MODE=daemon (default) → IPC to independent daemon; auto-spawns
  - YICENET_MODE=subprocess → cold-start engine in-process (slow fallback)

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
mode = os.environ.get("YICENET_MODE", "daemon")

# Read stdin once and pass explicitly.
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

# ── Daemon mode (default): IPC to independent daemon, auto-spawn ────────────

if mode == "daemon":
    from yicenet.tools.ipc_hook import pre_message_send_ipc, stop_ipc, post_tool_ipc

    if event == "pre":
        if not pre_message_send_ipc(_payload):
            # Daemon spawn failed — emit degraded status
            _degraded = {
                "yicenet": {
                    "session_id": "",
                    "turn_id": 0,
                    "label": "[YiCeNet daemon unavailable]",
                    "hexagram": "",
                    "action": "",
                    "env_confidence": 0.0,
                    "context_status": "thin",
                    "prescription": {},
                }
            }
            print(json.dumps(_degraded, ensure_ascii=False), flush=True)
            sys.stderr.write("[YiCeNet] daemon unavailable\n")
            sys.stderr.flush()
    elif event == "post_tool":
        post_tool_ipc(_payload)
    elif event == "stop":
        stop_ipc(_payload)

    sys.exit(0)

# ── Subprocess mode: cold-start engine in-process ───────────────────────────

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
