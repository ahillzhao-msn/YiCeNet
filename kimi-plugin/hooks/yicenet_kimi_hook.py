#!/usr/bin/env python3
"""
YiCeNet Kimi Code plugin hook dispatcher.

This script is invoked by Kimi Code's hook system:
  python ./hooks/yicenet_kimi_hook.py pre_message_send
  python ./hooks/yicenet_kimi_hook.py stop
  python ./hooks/yicenet_kimi_hook.py post_tool_use

It reads the Kimi Code hook payload from stdin, delegates to the
KimiCodeAdapter, and writes any context injection to stdout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Reconfigure to UTF-8 so CJK characters print on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _ensure_yicenet_home() -> None:
    """Set a sane default for YICENET_HOME in hook subprocesses."""
    if not os.environ.get("YICENET_HOME"):
        os.environ.setdefault("YICENET_HOME", str(Path.home() / ".yicenet"))


def main() -> None:
    _ensure_yicenet_home()

    try:
        from yicenet.tools.kimi_code_hook import main as _adapter_main
    except ImportError as exc:
        print(
            f"[YiCeNet hook] ERROR: cannot import yicenet: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    _adapter_main(sys.argv)


if __name__ == "__main__":
    main()
