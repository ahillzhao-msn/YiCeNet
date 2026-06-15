#!/usr/bin/env bash
# install-claudecode-hooks.sh — Full Claude Code integration for YiCeNet.
#
# Delegates core setup (pip install, checkpoints, mcpServers entry) to
# yicenet-bootstrap --target claude-code, then adds the Claude Code-specific
# lifecycle hooks (PostToolUse, Stop) for flywheel reward signal collection.
#
# Usage:
#   bash scripts/install/install-claudecode-hooks.sh
#   YICENET_PATH=/custom/path bash scripts/install/install-claudecode-hooks.sh

set -euo pipefail

# ── 1. Detect YiCeNet path ──
echo ""
echo "── Detecting YiCeNet ──"

if [ -n "${YICENET_PATH:-}" ]; then
    echo "Using YICENET_PATH=$YICENET_PATH"
elif [ -f "$(pwd)/pyproject.toml" ] && grep -q "yicenet" "$(pwd)/pyproject.toml" 2>/dev/null; then
    YICENET_PATH="$(pwd)"
    echo "Detected from current directory: $YICENET_PATH"
elif [ -d "$HOME/YiCeNet" ]; then
    YICENET_PATH="$HOME/YiCeNet"
    echo "Detected at $YICENET_PATH"
else
    echo "ERROR: Cannot find YiCeNet directory."
    echo "Set YICENET_PATH=/path/to/YiCeNet and re-run."
    exit 1
fi

# ── 2. Core setup via bootstrap ──
# Handles: pip install yicenet[mcp], checkpoints, mcpServers entry in settings.json
echo ""
echo "── Running yicenet-bootstrap --target claude-code ──"
YICENET_HOME="$YICENET_PATH" python3 "$YICENET_PATH/src/yicenet/bootstrap.py" \
    --target claude-code --auto --skip-cron
echo "✓ Core setup done"

# ── 3. Write Claude Code lifecycle hooks ──
# Bootstrap writes mcpServers only. This step adds the PostToolUse + Stop hooks
# so that every tool call outcome feeds the YiCeNet flywheel automatically.
echo ""
echo "── Writing Claude Code hooks ──"

CLAUDE_SETTINGS="$HOME/.claude/settings.json"

python3 - <<PYEOF
import json, os

settings_path = os.path.expanduser("~/.claude/settings.json")
yicenet_path = "$YICENET_PATH"

with open(settings_path, "r", encoding="utf-8") as f:
    try:
        settings = json.load(f)
    except json.JSONDecodeError:
        settings = {}

# Session ID: stable hash of cwd + date — no Claude API needed.
# Recomputed on every hook invocation; same value within one coding day.
_sid = (
    'python3 -c "import hashlib,os,datetime; '
    "print(hashlib.sha256((os.getcwd()+datetime.date.today().isoformat()).encode()).hexdigest()[:12])"
    '"'
)

# PostToolUse: detect exit_code from hook JSON payload → submit reward signal
post_tool_cmd = (
    "YICENET_SID=$(" + _sid + "); "
    "python3 -c \""
    "import json,sys,os; "
    "os.environ.setdefault('YICENET_HOME', '" + yicenet_path + "'); "
    "from yicenet.flywheel import submit_trajectory; "
    "d=json.load(sys.stdin) if not sys.stdin.isatty() else {}; "
    "ok=d.get('exit_code',0)==0; "
    "submit_trajectory({'producer':'claude-code','version':1,"
    "'conversation_id':'$YICENET_SID',"
    "'trajectory':{'completed':ok,'abandoned':not ok,'continued':False,"
    "'corrected':False,'praised':False,'token_cost':0,'token_efficiency':0.0}})"
    "\" 2>/dev/null || true"
)

# Stop: flush MemoryBank at session end to free per-session encoder cache
stop_cmd = (
    "YICENET_SID=$(" + _sid + "); "
    "python3 -c \""
    "import os; os.environ.setdefault('YICENET_HOME', '" + yicenet_path + "'); "
    "from yicenet.memory_bank import get_memory_bank; "
    "get_memory_bank().flush_session('$YICENET_SID')"
    "\" 2>/dev/null || true"
)

settings.setdefault("hooks", {})

def _upsert_hook(section: str, cmd: str):
    """Add or replace yicenet hook in a section, avoiding duplicates."""
    hooks = settings["hooks"].setdefault(section, [])
    settings["hooks"][section] = [
        h for h in hooks
        if not any("yicenet" in str(hh.get("command", "")) for hh in h.get("hooks", []))
    ]
    settings["hooks"][section].append({
        "matcher": "",
        "hooks": [{"type": "command", "command": cmd}],
    })

_upsert_hook("PostToolUse", post_tool_cmd)
_upsert_hook("Stop", stop_cmd)

with open(settings_path, "w", encoding="utf-8") as f:
    json.dump(settings, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("✓ hooks.PostToolUse written  (flywheel reward signal per tool call)")
print("✓ hooks.Stop written         (MemoryBank session flush on exit)")
PYEOF

# ── 4. Verify ──
echo ""
echo "── Verifying ──"
python3 -c "
import os; os.environ.setdefault('YICENET_HOME', '$YICENET_PATH')
from yicenet.mcp_server import mcp
print('✓ yicenet.mcp_server importable')
" || { echo "ERROR: yicenet.mcp_server import failed"; exit 1; }

which yicenet-serve >/dev/null 2>&1 \
    && echo "✓ yicenet-serve on PATH: $(which yicenet-serve)" \
    || echo "⚠  yicenet-serve not on PATH — ensure pip install target is in PATH"

echo ""
echo "Done. YiCeNet is fully wired into Claude Code."
echo ""
echo "  MCP server:  yicenet-serve  (stdio — Claude Code manages the process)"
echo "  Tools:       yicenet_attend · yicenet_predict · yicenet_feedback · yicenet_switch"
echo "  Hooks:       PostToolUse → flywheel,  Stop → MemoryBank flush"
echo "  Session ID:  sha256(cwd + YYYY-MM-DD)[:12]"
echo ""
echo "  Restart Claude Code to activate."
