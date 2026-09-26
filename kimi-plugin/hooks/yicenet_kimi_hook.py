#!/usr/bin/env python3
"""
YiCeNet Kimi Code plugin hook dispatcher (Python fallback of the native client).

Invoked by Kimi Code's hook system:
  python ./hooks/yicenet_kimi_hook.py pre_message_send
  python ./hooks/yicenet_kimi_hook.py stop
  python ./hooks/yicenet_kimi_hook.py post_tool_use

Forwards the raw payload to the YiCeNet daemon exactly like the native client
(`yicenet-hook kimi-code <event>`, which can replace `python <this script>` in
kimi.plugin.json when it is installed) and writes the daemon's reply to stdout.
If the daemon cannot be reached, runs the KimiCodeAdapter in-process instead.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    if not os.environ.get("YICENET_HOME"):
        os.environ.setdefault("YICENET_HOME", str(Path.home() / ".yicenet"))
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    body = sys.stdin.buffer.read()

    try:
        from yicenet.tools.ipc_hook import forward
    except ImportError as exc:
        print(f"[YiCeNet hook] ERROR: cannot import yicenet: {exc}", file=sys.stderr)
        sys.exit(0)

    out = forward("kimi-code", event, body)
    if out is not None:
        if out:
            os.write(1, out)
        return

    # Daemon unavailable: cold-start the adapter in this process.
    from yicenet.tools import kimi_code_hook as kimi
    handler = kimi._COMMANDS.get(event)
    if handler is None:
        sys.exit(0)
    try:
        text = body.decode("utf-8", errors="replace").strip()
        payload = json.loads(text) if text else {}
    except Exception:
        payload = {}
    handler(payload if isinstance(payload, dict) else {})


if __name__ == "__main__":
    main()
