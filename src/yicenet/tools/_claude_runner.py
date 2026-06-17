"""YiCeNet Claude Code hook runner.
Dispatched by UserPromptSubmit / PostToolUse / Stop hooks.
Reads event from YICENET_HOOK_EVENT env var or first argv.

This file is the source of truth for what gets installed at
~/.claude/hooks/yicenet_claude_hook.py by ClaudeCodeInstaller.
Edit here; re-run install to deploy.
"""
import sys
import os

# Reconfigure stdout/stderr to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

event = os.environ.get("YICENET_HOOK_EVENT") or (sys.argv[1] if len(sys.argv) > 1 else "pre")

if event == "pre":
    from yicenet.tools.claude_hook import pre_message_send
    pre_message_send()
elif event == "post_tool":
    from yicenet.tools.claude_hook import post_tool_use
    post_tool_use()
elif event == "stop":
    from yicenet.tools.claude_hook import stop
    stop()
