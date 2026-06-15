"""
YiCeNet DataSource — pluggable scanner abstraction for the flywheel.

Every platform that feeds the flywheel implements DataSource.scan_since().
The flywheel calls all registered sources, merges the samples, and trains on
the unified stream — no platform-specific logic inside flywheel.py itself.

Standard sources:
  HermesDataSource      — Hermes state.db (SQLite, existing schema)
  ClaudeCodeDataSource  — ~/.claude/projects/**/*.jsonl (JSONL transcripts)
  FlywheelBufferSource  — ~/.yicenet/data/flywheel_buffer.jsonl
                          (written by submit_trajectory() from any producer)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    """Standardised training sample consumed by the flywheel."""

    # Identity
    conversation_id: str
    source: str                  # "hermes" | "claude-code" | "buffer" | …

    # Text payload (may be truncated for large responses)
    user_text: str
    assistant_text: str
    next_user_text: str = ""

    # Timing
    timestamp: float = 0.0

    # Scalar signals (mirrors flywheel_buffer.jsonl schema)
    satisfaction: float = 0.0
    token_cost: float = 0.0       # normalised 0-1
    response_length: float = 0.0  # normalised 0-1

    # Boolean signals
    continued: bool = False
    corrected: bool = False
    completed: bool = False
    praised: bool = False
    abandoned: bool = False

    # Optional pre-computed embedding (384-d from TinyEncoder or external)
    embedding: list[float] = field(default_factory=list)

    # Source-specific opaque ID (e.g. Hermes msg_id) for deduplication
    source_msg_id: str = ""


class DataSource(ABC):
    """Abstract scanner: returns new samples since a Unix timestamp."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Short identifier, e.g. 'hermes', 'claude-code', 'buffer'."""

    @abstractmethod
    def scan_since(self, timestamp: float) -> list[Sample]:
        """Return all samples with timestamp > given Unix epoch float."""

    def is_available(self) -> bool:
        """Return False if the underlying store doesn't exist on this machine."""
        return True
