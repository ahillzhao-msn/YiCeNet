#!/usr/bin/env bash
#
# YiCeNet Uninstall — v15.6.1
#
# Two methods:
#   1. Python CLI  (recommended):  yicenet-uninstall
#   2. This script (fallback):     bash scripts/uninstall-yicenet.sh
#
# The Python CLI handles multi-platform cleanup (Hermes plugin, Claude Code hooks,
# files, checkpoints). This script is a thin wrapper.
#
set -euo pipefail

echo "=== YiCeNet Uninstall ==="

# Check Python uninstall first
if command -v yicenet-uninstall &>/dev/null; then
    echo "→ Using yicenet-uninstall CLI..."
    exec yicenet-uninstall "$@"
fi

# Fallback: manual cleanup
YICENET_HOME="${YICENET_HOME:-$HOME/.yicenet}"
echo "→ Python CLI not found. Manual cleanup:"

# Remove Hermes plugin
HERMES_PLUGINS="$HOME/.hermes/plugins/yicenet-hooks"
if [ -d "$HERMES_PLUGINS" ]; then
    rm -rf "$HERMES_PLUGINS" && echo "  ✓ Removed Hermes plugin"
fi

# Remove Claude Code hooks
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
if [ -f "$CLAUDE_SETTINGS" ]; then
    echo "  ⚠  Please remove YiCeNet hooks from $CLAUDE_SETTINGS manually"
fi

# Remove data
if [ -d "$YICENET_HOME" ]; then
    echo "  To remove all data: rm -rf $YICENET_HOME"
    echo "  (Kept by default — may contain valuable logs/checkpoints)"
fi

echo "→ Done.  To fully remove package: pip uninstall yicenet"
