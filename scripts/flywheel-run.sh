#!/usr/bin/env bash
# YiCeNet Flywheel — environment-adaptive launcher
#
# Auto-detects the Python environment:
#   1. Hermes Agent venv    (hermes python)
#   2. pip-installed yicenet (python3)
#   3. fallback python
#
# Logs to ~/.yicenet/logs/flywheel.log
#
set -euo pipefail

_YH="${YICENET_HOME:-$HOME/.yicenet}"
_LOG="$_YH/logs/flywheel.log"
mkdir -p "$(dirname "$_LOG")"

echo "[$(date -Iseconds)] Flywheel run starting — ${0}" >> "$_LOG"

_ran=0
_launch() {
    local py="$1"
    if "$py" -m yicenet.flywheel >> "$_LOG" 2>&1; then
        _ran=1
        return 0
    fi
    return 1
}

# 1) Hermes Agent venv
_HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
_HERMES_PY="$_HERMES_HOME/hermes-agent/venv/bin/python"
if [ -x "$_HERMES_PY" ]; then
    _launch "$_HERMES_PY" && { echo "[$(date -Iseconds)] Flywheel run finished (exit=0, hermes)" >> "$_LOG"; exit 0; }
fi

# 2) System python3
if command -v python3 &>/dev/null; then
    _launch "python3" && { echo "[$(date -Iseconds)] Flywheel run finished (exit=0, python3)" >> "$_LOG"; exit 0; }
fi

# 3) Fallback python
if command -v python &>/dev/null; then
    _launch "python" && { echo "[$(date -Iseconds)] Flywheel run finished (exit=0, python)" >> "$_LOG"; exit 0; }
fi

echo "[$(date -Iseconds)] Flywheel run failed — no usable python found" >> "$_LOG"
exit 1
