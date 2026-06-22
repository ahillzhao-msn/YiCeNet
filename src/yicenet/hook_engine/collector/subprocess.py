"""SubprocessContextCollector — file-based accumulator for cross-process hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .interface import ContextCollector


class SubprocessContextCollector(ContextCollector):
    """File-based accumulator for cross-process hook invocations.

    Each CC subprocess hook:
      1. Creates/appends events to a JSONL side-channel file
      2. On build_vector(), loads all events and replays them through
         DaemonContextCollector for computation logic reuse
    """

    _BASE = Path.home() / ".yicenet" / "data" / "cc-ctx"

    def __init__(self, session_id: str, turn_id: int) -> None:
        self._session_id = session_id
        self._turn_id = turn_id
        self._events: list[dict] = []
        self._loaded = False

    def sniff_user(self, text: str, is_first_turn: bool = False) -> None:
        self._append_event({"type": "user", "text": text, "is_first": is_first_turn})

    def sniff_api(self, prompt_tokens: int, completion_tokens: int,
                  duration_ms: float = 0.0) -> None:
        self._append_event({"type": "api", "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "duration_ms": duration_ms})

    def sniff_tool(self, name: str, exit_code: int,
                   duration_ms: float, result_size_bytes: int = 0) -> None:
        self._append_event({"type": "tool", "name": name, "exit_code": exit_code,
                            "duration_ms": duration_ms, "result_size": result_size_bytes})

    def sniff_response(self, text: str) -> None:
        self._append_event({"type": "response", "text": text})

    def sniff_hexagram(self, q_max: float, q_gap: float,
                       entropy: float) -> None:
        self._append_event({"type": "hexagram", "q_max": q_max,
                            "q_gap": q_gap, "entropy": entropy})

    def sniff_timing(self, user_interval_sec: Optional[float] = None,
                     prev_metadata: Optional[dict] = None) -> None:
        self._append_event({"type": "timing", "interval_sec": user_interval_sec})

    def build_vector(self, prev_metadata: Optional[dict] = None) -> dict:
        self._load()
        from .daemon import DaemonContextCollector
        inner = DaemonContextCollector()
        for e in self._events:
            t = e.get("type")
            if t == "user":
                inner.sniff_user(e["text"], e.get("is_first", False))
            elif t == "api":
                inner.sniff_api(e["prompt_tokens"], e["completion_tokens"],
                                e.get("duration_ms", 0.0))
            elif t == "tool":
                inner.sniff_tool(e["name"], e["exit_code"],
                                 e["duration_ms"], e.get("result_size", 0))
            elif t == "response":
                inner.sniff_response(e["text"])
            elif t == "hexagram":
                inner.sniff_hexagram(e["q_max"], e["q_gap"], e["entropy"])
            elif t == "timing":
                inner.sniff_timing(e.get("interval_sec"))
        return inner.build_vector(prev_metadata=prev_metadata)

    def cleanup(self) -> None:
        try:
            self._path().unlink()
        except OSError:
            pass

    # -- internal -------------------------------------------------------------

    def _append_event(self, event: dict) -> None:
        self._events.append(event)
        self._flush()

    def _flush(self) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._events = []

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._path()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._events.append(json.loads(line))

    def _path(self) -> Path:
        return self._BASE / f"{self._session_id}.{self._turn_id}.jsonl"
