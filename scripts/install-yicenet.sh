#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# YiCeNet (易策网络) — pip install into Hermes venv (via uv)
#
# Install strategy:
#   1. Detect Hermes venv
#   2. Download YiCeNet wheel + model checkpoints from release
#   3. Install wheel into Hermes venv via uv (no source code)
#   4. Set YICENET_HOME for runtime data paths
#   5. Install/update yicenet-hooks plugin
# ──────────────────────────────────────────────────────────────
set -euo pipefail

REPO="ahillzhao-msn/YiCeNet"
PLUGIN_NAME="yicenet-hooks"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══ Installing YiCeNet into Hermes venv ═══"

# ── 1. Detect Hermes venv python ──
VENV_PY=""
if command -v hermes &>/dev/null; then
    VENV_PY="$(hermes -z 'which python' 2>/dev/null || true)"
fi
if [ -z "$VENV_PY" ]; then
    for cand in \
        "$HERMES_HOME/hermes-agent/venv/bin/python" \
        "$HERMES_HOME/.venv/bin/python" \
        "$HERMES_HOME/venv/bin/python"; do
        if [ -x "$cand" ]; then
            VENV_PY="$cand"
            break
        fi
    done
fi
if [ -z "$VENV_PY" ]; then
    echo "✗ Hermes venv not found. Install Hermes first."
    exit 1
fi
echo "✓ Hermes python: $VENV_PY"

UV="$(command -v uv || command -v pip3 || echo '')"
if [ -z "$UV" ]; then
    echo "✗ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓ uv: $UV"

# ── 2. Determine version ──
VERSION="${1:-latest}"
if [ "$VERSION" = "latest" ]; then
    echo "  → Fetching latest release..."
    LATEST="$(
        curl -sL "https://api.github.com/repos/$REPO/releases/latest" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])"
    )"
    VERSION="$LATEST"
fi
echo "  Version: $VERSION"

RELEASE_URL="https://github.com/$REPO/releases/download/$VERSION"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# ── 3. YiCeNet data directory (YICENET_HOME) ──
YICENET_DATA="${YICENET_HOME:-$HERMES_HOME/data/yicenet}"
mkdir -p "$YICENET_DATA/checkpoints"
mkdir -p "$YICENET_DATA/data"
echo "✓ YiCeNet data: $YICENET_DATA"

# ── 4. Download + install wheel ──
echo ""
echo "── Installing Python package ──"
echo "  → Downloading wheel list..."
WHEEL_FILE="$(
    curl -sL "https://api.github.com/repos/$REPO/releases/tags/$VERSION" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
for a in data.get('assets', []):
    if a['name'].endswith('.whl'):
        print(a['name'])
        break
"
)"
if [ -z "$WHEEL_FILE" ]; then
    echo "✗ No wheel found in release $VERSION"
    exit 1
fi

echo "  → Downloading $WHEEL_FILE..."
curl -sL "$RELEASE_URL/$WHEEL_FILE" -o "$TMPDIR/$WHEEL_FILE"

echo "  → Installing into Hermes venv..."
uv pip install --python "$VENV_PY" "$TMPDIR/$WHEEL_FILE"

# Verify
"$VENV_PY" -c "from yicenet import __version__; print(f'✓ YiCeNet v{__version__} installed')"

# ── 5. Download checkpoints from release ──
echo ""
echo "── Downloading checkpoints ──"
CKPT_DIR="$YICENET_DATA/checkpoints"
REQUIRED=("yicenet_v15.pt" "yicenet_v18.pt" "world_model_best.pt" "registry.json")
MISSING=0
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$CKPT_DIR/$f" ]; then
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "  Required checkpoints: ${REQUIRED[*]}"
    echo "  → Downloading from $RELEASE_URL ..."
    for f in "${REQUIRED[@]}"; do
        echo "    $f ..."
        curl -sL "$RELEASE_URL/$f" -o "$CKPT_DIR/$f" &
    done
    wait
    echo "✓ Checkpoints downloaded to $CKPT_DIR"
else
    echo "✓ All checkpoints present"
fi

# ── 6. Install/update yicenet-hooks plugin ──
echo ""
echo "── Installing Hermes plugin ──"
PLUGIN_DIR="$HERMES_HOME/plugins/$PLUGIN_NAME"
mkdir -p "$PLUGIN_DIR"

cat > "$PLUGIN_DIR/plugin.yaml" << 'YAML'
name: yicenet-hooks
version: 2.0.0
description: >
  YiCeNet (易策网络) -- standalone Hermes lifecycle hooks. 四鉤子:
    pre_llm_call -> yicenet_predict (hexagram context; auto-skip when LOOM)
    pre_tool_call -> hexagram calibration on code-gen tools (observe only)
    post_tool_call -> tool-level reward signal
    post_api_request -> accumulate token usage
    post_llm_call -> feedback() writes reward signal to flywheel buffer
  Works standalone (no LOOM) or alongside loom-hooks (auto-skip).
  Design doc: LOOM docs/hooks-evolution.md
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

cat > "$PLUGIN_DIR/__init__.py" << 'PYEOF'
"""YiCeNet (易策网络) -- standalone Hermes lifecycle hooks.

Installed via "uv pip install yicenet" into Hermes venv.
All imports resolve from the installed package (no source-path hacks).
YICENET_HOME env var points to runtime data (checkpoints, vocab).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.getLogger("yicenet-hooks").setLevel(logging.INFO)
logger = logging.getLogger("yicenet-hooks")

# Flywheel buffer goes under YICENET_HOME/data/
_YICENET_HOME = os.environ.get("YICENET_HOME", str(Path.home() / ".hermes" / "data" / "yicenet"))
_YICENET_BUFFER = os.path.join(_YICENET_HOME, "data", "flywheel_buffer.jsonl")
_session_usage: dict[str, dict[str, float]] = {}
_yicenet_predict = None


def _get_yicenet():
    global _yicenet_predict
    if _yicenet_predict is None:
        try:
            from yicenet.hermes_tool import yicenet_predict as yp
            _yicenet_predict = yp
        except ImportError as e:
            logger.warning("yicenet_predict not available: %s", e)
            _yicenet_predict = False
    return _yicenet_predict if _yicenet_predict else None


def feedback(session_id: str, response_chars: int, input_chars: int, n_turns: int,
             model: str, platform: str, success: bool = True) -> None:
    if not _YICENET_BUFFER:
        return
    usage = _session_usage.pop(session_id, {})
    token_cost = usage.get("total_tokens", 0) or int(response_chars * 0.25)
    token_efficiency = usage.get("efficiency", 0)
    if not token_efficiency:
        total = input_chars + response_chars + 1
        token_efficiency = response_chars / total if total > 0 else 0.5
    satisfaction = min(1.0, (0.6 if success else 0.2) + token_efficiency * 0.3)
    sample = {
        "user_text": f"[yicenet-hooks] sid={session_id[:12]}",
        "producer": "yicenet-hooks",
        "conversation_id": session_id,
        "hexagram_evolution": [],
        "timestamp": time.time(),
        "token_cost": int(token_cost),
        "token_efficiency": round(token_efficiency, 4),
        "continued": False, "corrected": False,
        "completed": n_turns > 0, "praised": False, "abandoned": False,
        "satisfaction": round(satisfaction, 4),
    }
    try:
        os.makedirs(os.path.dirname(_YICENET_BUFFER), exist_ok=True)
        with open(_YICENET_BUFFER, "a") as f:
            f.write(json.dumps(sample) + "\n")
    except Exception as e:
        logger.debug("feedback write failed: %s", e)


def _record_api_usage(session_id: str, usage_data: dict | None) -> None:
    if not session_id or not usage_data:
        return
    if session_id not in _session_usage:
        _session_usage[session_id] = {"total_tokens": 0, "api_calls": 0, "total_input": 0, "total_output": 0}
    acc = _session_usage[session_id]
    acc["total_tokens"] += usage_data.get("total_tokens", 0) or 0
    acc["api_calls"] += 1
    in_tok = usage_data.get("input_tokens", 0) or usage_data.get("prompt_tokens", 0) or 0
    out_tok = usage_data.get("output_tokens", 0) or usage_data.get("completion_tokens", 0) or 0
    acc["total_input"] += in_tok
    acc["total_output"] += out_tok
    total = acc["total_input"] + acc["total_output"]
    acc["efficiency"] = acc["total_output"] / total if total > 0 else 0.5


def _loom_hooks_active() -> bool:
    try:
        from hermes_cli.plugins import get_plugin_manager
        pm = get_plugin_manager()
        return 'loom-hooks' in pm._plugins
    except Exception:
        return False


def on_session_start(**kw: Any) -> None:
    if _loom_hooks_active():
        return
    yp = _get_yicenet()
    if not yp:
        return
    session_id = kw.get("session_id", "?")
    platform = kw.get("platform", "?")
    try:
        yp(f"Session start: {platform}", temperature=0.1, deterministic=True)
        logger.debug("yicenet baseline: session %s", session_id[:12])
    except Exception as e:
        logger.debug("yicenet baseline skipped: %s", e)


def pre_llm_call(**kw: Any) -> dict | str | None:
    if _loom_hooks_active():
        logger.debug("loom-hooks active -- yicenet-hooks pre_llm_call skipped")
        return None
    yp = _get_yicenet()
    if not yp:
        return None
    user_message = kw.get("user_message", "")
    session_id = kw.get("session_id", "")
    is_first = kw.get("is_first_turn", False)
    if not user_message or not user_message.strip():
        return None
    try:
        hx = yp(user_message[:200], temperature=0.1, deterministic=is_first)
        if hx and len(hx) > 10:
            return {"context": f"[YiCeNet hexagram]\n{hx[:300]}"}
    except Exception as e:
        logger.debug("yicenet predict skipped: %s", e)
    return None


def pre_tool_call(**kw: Any) -> None:
    if _loom_hooks_active():
        return
    yp = _get_yicenet()
    if not yp:
        return
    tool_name = kw.get("tool_name", "")
    args = kw.get("args", {})
    if tool_name not in ("write_file", "patch", "terminal"):
        return
    brief = f"{tool_name}: "
    if tool_name == "write_file":
        brief += args.get("path", "")[:100]
    elif tool_name == "patch":
        brief += args.get("path", "")[:100]
    elif tool_name == "terminal":
        brief += args.get("command", "")[:200]
    if not brief.strip():
        return
    try:
        hx = yp(brief, temperature=0.1, deterministic=True)
        logger.debug("yicenet pre_tool: %s -> %s", tool_name, str(hx)[:80])
    except Exception as e:
        logger.debug("yicenet pre_tool skipped: %s", e)


def post_tool_call(**kw: Any) -> None:
    if _loom_hooks_active():
        return
    tool_name = kw.get("tool_name", "")
    result = kw.get("result", "")
    duration_ms = kw.get("duration_ms", 0)
    if tool_name not in ("write_file", "patch", "terminal"):
        return
    success = True
    if isinstance(result, str):
        try:
            res = json.loads(result)
            if isinstance(res, dict) and "error" in res:
                success = False
        except (json.JSONDecodeError, TypeError):
            pass
    feedback(
        session_id=kw.get("session_id", ""),
        response_chars=0,
        input_chars=len(kw.get("args", {}).get("command", kw.get("args", {}).get("path", ""))),
        n_turns=0,
        model=kw.get("model", "unknown"),
        platform=kw.get("platform", ""),
        success=success,
    )


def post_api_request(**kw: Any) -> None:
    usage = kw.get("usage")
    if usage and isinstance(usage, dict):
        _record_api_usage(kw.get("session_id", ""), usage)


def post_llm_call(**kw: Any) -> None:
    user_message = kw.get("user_message", "")
    assistant_response = kw.get("assistant_response", "")
    session_id = kw.get("session_id", "")
    model = kw.get("model", "unknown")
    platform = kw.get("platform", "")
    history = kw.get("conversation_history", [])
    n_turns = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "assistant")
    if not assistant_response or not assistant_response.strip():
        return
    feedback(session_id, len(assistant_response), len(user_message or ""),
             n_turns, model, platform)


def on_session_end(**kw: Any) -> None:
    session_id = kw.get("session_id", "?")
    logger.debug("yicenet session ended: %s", session_id[:12])


def register(ctx) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    logger.info("yicenet-hooks: registered 7 hooks")
PYEOF

echo "✓ Plugin written to $PLUGIN_DIR"

# ── 7. Enable plugin ──
echo ""
echo "── Enabling plugin ──"
if command -v hermes &>/dev/null; then
    hermes plugins enable "$PLUGIN_NAME" 2>&1 || echo "⚠ hermes plugins enable failed — enable manually in config.yaml"
else
    echo "⚠ hermes CLI not found"
fi

echo ""
echo "═══ YiCeNet install complete ═══"
echo "  Package: installed in Hermes venv"
echo "  Data:    $YICENET_DATA"
echo "  Checkpoints: $CKPT_DIR"
echo "  Plugin:  $PLUGIN_DIR"
echo "  Env:     export YICENET_HOME=$YICENET_DATA"
echo ""
echo "  To activate: restart Hermes"
echo "  To verify:   hermes plugins list | grep yicenet"
echo "  To remove:   hermes plugins disable $PLUGIN_NAME && rm -rf $PLUGIN_DIR"
