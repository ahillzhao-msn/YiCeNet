#!/usr/bin/env bash
# install-mcp-server.sh — General MCP server installer for YiCeNet.
#
# Registers yicenet-serve as an MCP stdio server in the target tool's config.
# Decoupled from hook installation — hooks and MCP are independent integrations.
#
# Usage:
#   bash scripts/install/install-mcp-server.sh --target claude-code
#   bash scripts/install/install-mcp-server.sh --target hermes
#   bash scripts/install/install-mcp-server.sh --target cursor
#
# Targets:
#   claude-code  — writes mcpServers entry to ~/.claude/settings.json
#   hermes       — writes mcpServers entry to ~/.hermes/config.json
#   cursor       — writes mcpServers entry to ~/.cursor/mcp.json
#   (more targets can be added below)

set -euo pipefail

TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Usage: bash install-mcp-server.sh --target <claude-code|hermes|cursor>"
    exit 1
fi

SERVE_CMD="$(command -v yicenet-serve 2>/dev/null || echo "")"
if [ -z "$SERVE_CMD" ]; then
    echo "ERROR: yicenet-serve not found on PATH. Run pip install yicenet first."
    exit 1
fi

echo "── Installing YiCeNet MCP server for: $TARGET ──"
echo "   Command: $SERVE_CMD"

register_mcp() {
    local config_file="$1"
    python3 - <<PYEOF
import json, os, sys
path = os.path.expanduser("$config_file")
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    cfg = json.loads(open(path).read()) if os.path.exists(path) else {}
except Exception:
    cfg = {}
cfg.setdefault("mcpServers", {})["yicenet"] = {
    "command": "$SERVE_CMD",
    "args": [],
    "type": "stdio",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"  ✓ mcpServers.yicenet written to {path}")
PYEOF
}

case "$TARGET" in
    claude-code)
        register_mcp "~/.claude/settings.json"
        echo "  Restart Claude Code to activate."
        ;;
    hermes)
        register_mcp "~/.hermes/config.json"
        echo "  Restart Hermes to activate."
        ;;
    cursor)
        register_mcp "~/.cursor/mcp.json"
        echo "  Restart Cursor to activate."
        ;;
    *)
        echo "ERROR: Unknown target '$TARGET'."
        echo "Supported: claude-code, hermes, cursor"
        exit 1
        ;;
esac

echo ""
echo "Done. MCP tools available: yicenet_attend · yicenet_predict · yicenet_feedback · yicenet_switch"
