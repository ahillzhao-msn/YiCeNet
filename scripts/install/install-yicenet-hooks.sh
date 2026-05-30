#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# YiCeNet (易策网络) — Standalone Hermes Plugin Installer
#
# Wires YiCeNet's predict + feedback as native Hermes lifecycle
# hooks.  Works independently of LOOM.
#
# Three-channel flywheel:
#   1. Session DB scan (intrinsic foundation)
#   2. LOOM solidify → yicenet buffer (when LOOM present)
#   3. This plugin: post_llm_call → feedback() (first-hand)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

PLUGIN_NAME="yicenet-hooks"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/$PLUGIN_NAME"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══ Installing YiCeNet Hermes Plugin ═══"

# ── 1. Detect YiCeNet path ──
if [ -d "$HOME/YiCeNet" ]; then
    YICENET_PATH="$HOME/YiCeNet"
elif [ -d "$SCRIPT_DIR/.." ] && [ -f "$SCRIPT_DIR/../src/yicenet/hermes_tool.py" ]; then
    YICENET_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "⚠ YiCeNet not found at ~/YiCeNet or relative to this script."
    echo "  Please set YICENET_PATH manually and re-run."
    echo "  Example: YICENET_PATH=/path/to/YiCeNet $0"
    exit 1
fi
echo "✓ YiCeNet at: $YICENET_PATH"

# ── 2. Create plugin directory ──
mkdir -p "$PLUGIN_DIR"
echo "✓ Plugin dir: $PLUGIN_DIR"

# ── 3. Write plugin.yaml ──
cp "$SCRIPT_DIR/plugin.yaml" "$PLUGIN_DIR/plugin.yaml" 2>/dev/null || cat > "$PLUGIN_DIR/plugin.yaml" << 'YAML'
name: yicenet-hooks
version: 1.0.0
description: "YiCeNet lifecycle hooks: predict + feedback as native Hermes hooks. 3-channel flywheel."
author: Hermes Agent
hooks:
  - on_session_start
  - pre_llm_call
  - post_api_request
  - post_llm_call
  - on_session_end
YAML
echo "✓ plugin.yaml"

# ── 4. Write __init__.py ──
cp "$SCRIPT_DIR/__init__.py" "$PLUGIN_DIR/__init__.py" 2>/dev/null || {
    echo "⚠ __init__.py not found alongside this script."
    echo "  Expected at: $SCRIPT_DIR/__init__.py"
    echo "  Create manually or re-clone YiCeNet."
    exit 1
}
echo "✓ __init__.py"

# ── 5. Symlink Hermes tool (so yicenet_predict is available) ──
TOOL_LINK="$HERMES_HOME/hermes-agent/tools/yicenet_tool.py"
TOOL_SRC="$YICENET_PATH/src/yicenet/hermes_tool.py"
if [ ! -L "$TOOL_LINK" ]; then
    ln -sf "$TOOL_SRC" "$TOOL_LINK"
    echo "✓ Symlinked yicenet_tool.py → hermes-agent/tools/"
else
    echo "  yicenet_tool.py already linked"
fi

# ── 6. Enable plugin ──
echo ""
echo "── Enabling plugin ──"
if command -v hermes &>/dev/null; then
    hermes plugins enable "$PLUGIN_NAME" 2>&1 || {
        echo "⚠ 'hermes plugins enable' failed. Patching config.yaml directly..."
        CONFIG="$HERMES_HOME/config.yaml"
        python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
plugins = cfg.setdefault('plugins', {})
enabled = plugins.setdefault('enabled', [])
if '$PLUGIN_NAME' not in enabled:
    enabled.append('$PLUGIN_NAME')
with open('$CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('✓ Patched config.yaml')
"
    }
else
    echo "⚠ hermes CLI not found — manually add to config.yaml:"
    echo "  plugins:"
    echo "    enabled: [$PLUGIN_NAME]"
fi

# ── 7. Verify ──
echo ""
echo "── Verification ──"
ls -la "$PLUGIN_DIR/"
echo ""
echo "Config check:"
grep -A 2 "enabled:" "$HERMES_HOME/config.yaml" 2>/dev/null | head -5
echo ""

# Import test
"$HERMES_HOME/hermes-agent/venv/bin/python3" -c "
import sys
sys.path.insert(0, '$HOME/YiCeNet/src')
from yicenet.hermes_tool import yicenet_predict, yicenet_switch
print('✓ yicenet_predict importable')
print('✓ yicenet_switch importable')
print('Plugin ready for next Hermes session.')
" 2>&1

echo ""
echo "═══ Install complete ═══"
echo "To activate: restart Hermes (new session)"
echo "To verify:   hermes plugins list | grep yicenet"
echo "To remove:   hermes plugins disable yicenet-hooks"
echo "              rm -rf ~/.hermes/plugins/yicenet-hooks"
echo ""
echo "Three-channel flywheel:"
echo "  1. Session DB scan (intrinsic)  → every 6h"
echo "  2. LOOM → yicenet buffer         → when LOOM present"
echo "  3. This plugin: feedback()       → every post_llm_call"
