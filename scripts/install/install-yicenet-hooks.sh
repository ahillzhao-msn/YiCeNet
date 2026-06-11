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

# ── 2. Download checkpoints (if missing) ──
CKPT_DIR="$YICENET_PATH/checkpoints"
REQUIRED_FILES=("yicenet_v15.pt" "world_model_best.pt" "registry.json")
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CKPT_DIR/$f" ]; then
        MISSING=1
        break
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "⚠ Checkpoints incomplete. Required: ${REQUIRED_FILES[*]}"
    echo "  Release: https://github.com/ahillzhao-msn/YiCeNet/releases/tag/v18.0.0"
    echo ""
    read -r -p "  Auto-download from GitHub releases? [y/N] " REPLY
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$CKPT_DIR"
        RELEASE_URL="https://github.com/ahillzhao-msn/YiCeNet/releases/download/v18.0.0"
        for f in yicenet_v15.pt minimal.pt world_model_best.pt registry.json; do
            echo "    Downloading $f..."
            curl -sL "$RELEASE_URL/$f" -o "$CKPT_DIR/$f" &
        done
        wait
        echo "  ✓ Checkpoints downloaded to $CKPT_DIR"
    else
        echo "  ⚠ Skipping. Download manually from the release URL above."
        echo "    YiCeNetEngine will fail until checkpoints are in place."
    fi
else
    echo "✓ Checkpoints found"
fi

# ── 3. Create plugin directory ──
mkdir -p "$PLUGIN_DIR"
echo "✓ Plugin dir: $PLUGIN_DIR"

# ── 4. Write plugin.yaml ──
cp "$SCRIPT_DIR/plugin.yaml" "$PLUGIN_DIR/plugin.yaml" 2>/dev/null || cat > "$PLUGIN_DIR/plugin.yaml" << 'YAML'
name: yicenet-hooks
version: 2.0.0
description: "YiCeNet lifecycle hooks: predict + feedback as native Hermes hooks. 3-channel flywheel."
author: Hermes Agent
hooks:
  - on_session_start
  - pre_llm_call
  - pre_tool_call
  - post_tool_call
  - post_api_request
  - post_llm_call
  - on_session_end
YAML
echo "✓ plugin.yaml"

# ── 5. Write __init__.py ──
cp "$SCRIPT_DIR/__init__.py" "$PLUGIN_DIR/__init__.py" 2>/dev/null || {
    echo "⚠ __init__.py not found alongside this script."
    echo "  Expected at: $SCRIPT_DIR/__init__.py"
    echo "  Create manually or re-clone YiCeNet."
    exit 1
}
echo "✓ __init__.py"

# ── 6. Install Python package (ensure yicenet is importable) ──
echo ""
echo "── Installing Python package ──"
pip install -e "$YICENET_PATH" 2>/dev/null || pip3 install -e "$YICENET_PATH" 2>/dev/null || {
    echo "⚠ pip install failed. Make sure yicenet is on PYTHONPATH:"
    echo "  export PYTHONPATH=\$YICENET_PATH/src:\$PYTHONPATH"
}
echo "✓ Python package installed"

# ── 7. Enable plugin ──
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

# ── 8. Verify ──
echo ""
echo "── Verification ──"
ls -la "$PLUGIN_DIR/"
echo ""
echo "Config check:"
grep -A 2 "enabled:" "$HERMES_HOME/config.yaml" 2>/dev/null | head -5
echo ""

# Import test
python3 -c "
import sys
sys.path.insert(0, '$HOME/YiCeNet/src')
from yicenet.hermes_tool import yicenet_predict, yicenet_switch
print('✓ yicenet_predict importable')
print('✓ yicenet_switch importable')
" 2>&1 || python -c "
import sys
sys.path.insert(0, '$HOME/YiCeNet/src')
from yicenet.hermes_tool import yicenet_predict, yicenet_switch
print('✓ yicenet_predict importable')
print('✓ yicenet_switch importable')
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
