"""
FlywheelBufferSource — consumes ~/.yicenet/data/flywheel_buffer.jsonl.

This file is the universal drop-zone: any producer (Hermes plugin,
Claude Code PostToolUse hook, MCP yicenet_feedback tool, Loom, …) appends
structured trajectory records via submit_trajectory().

Unlike HermesDataSource and ClaudeCodeDataSource, the buffer already contains
pre-computed boolean signals and satisfaction scores — no external_metrics
re-processing needed. FlywheelBufferSource just deserialises and filters.

Thread / process safety:
  Writes use O_APPEND (atomic at the OS level for lines < PIPE_BUF ≈ 4KB).
  Reads are sequential from last-read byte offset; no lock needed for reading
  since JSONL is append-only and we never truncate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from yicenet.config import yicenet_data_dir
from . import DataSource, Sample


def _default_buffer_path() -> Path:
    return yicenet_data_dir() / "flywheel_buffer.jsonl"


class FlywheelBufferSource(DataSource):
    """
    Reads pre-computed reward trajectories from flywheel_buffer.jsonl.

    Incremental scan: tracks the byte offset of the last line consumed so
    repeated calls don't re-process already-seen entries.  The offset is
    reset when the file shrinks (rotated / cleared by maintenance tooling).
    """

    def __init__(self, buffer_path: Optional[Path] = None):
        self._path = Path(buffer_path) if buffer_path else _default_buffer_path()
        self._last_offset: int = 0

    @property
    def source_id(self) -> str:
        return "buffer"

    def is_available(self) -> bool:
        return self._path.exists()

    def scan_since(self, timestamp: float) -> list[Sample]:
        if not self._path.exists():
            return []

        try:
            file_size = self._path.stat().st_size
        except OSError:
            return []

        # File was rotated / truncated — reset offset
        if file_size < self._last_offset:
            self._last_offset = 0

        samples: list[Sample] = []
        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._last_offset)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    entry_ts = float(obj.get("timestamp", 0.0))
                    if entry_ts < timestamp:
                        continue  # older than requested window

                    conv_id = obj.get("conversation_id", "")
                    producer = obj.get("producer", "unknown")

                    samples.append(Sample(
                        conversation_id=f"{producer}:{conv_id}" if conv_id else producer,
                        source=self.source_id,
                        user_text=obj.get("user_text", ""),
                        assistant_text="",
                        next_user_text="",
                        timestamp=entry_ts,
                        satisfaction=float(obj.get("satisfaction", 0.0)),
                        token_cost=float(obj.get("token_cost", 0)),
                        response_length=float(obj.get("token_efficiency", 0)),
                        continued=bool(obj.get("continued", False)),
                        corrected=bool(obj.get("corrected", False)),
                        completed=bool(obj.get("completed", False)),
                        praised=bool(obj.get("praised", False)),
                        abandoned=bool(obj.get("abandoned", False)),
                        embedding=obj.get("embedding", []),
                        source_msg_id="",
                    ))

                self._last_offset = fh.tell()
        except OSError:
            pass

        return samples

    def reset_offset(self) -> None:
        """Force a full re-scan from the beginning of the buffer file."""
        self._last_offset = 0
