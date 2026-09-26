#!/usr/bin/env bash
# Build the native hook client (native/yicenet-hook) and install it to ~/.yicenet/bin.
# Uses CMake when available, otherwise compiles the single source file with c++ directly.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$HOME/.yicenet/bin}"
SRC="$PROJECT_DIR/native/yicenet-hook"
BUILD="$PROJECT_DIR/build/yicenet-hook"
mkdir -p "$OUT_DIR" "$BUILD"
if command -v cmake >/dev/null 2>&1; then
    cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD" >/dev/null
    BIN="$BUILD/yicenet-hook"
else
    CXX="${CXX:-c++}"
    "$CXX" -std=c++17 -O2 -o "$BUILD/yicenet-hook" "$SRC/yicenet_hook.cpp"
    BIN="$BUILD/yicenet-hook"
fi
install -m 0755 "$BIN" "$OUT_DIR/yicenet-hook"
echo "Installed $OUT_DIR/yicenet-hook"
