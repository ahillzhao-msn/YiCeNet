"""
YiCeNet MemoryBank — per-session turn storage for cross-attention.

Stores TinyEncoder 384d outputs per turn, scoped by session_id.
Append-only, session-volatile (flushed on conversation close).

Storage cost: ~1.5KB per turn (384 × float32).
1,000 turns = ~1.5MB. 10,000 turns = ~15MB. Not explosive.

Designed for Phase 1 of the Cross-Attention Memory Cortex architecture.
See docs/cross-attention-memory-cortex.md.

Usage:
    bank = MemoryBank()
    bank.store_turn(session_id, turn_id, encoder_384d, hexagram_id, summary)
    keys, meta = bank.get_session_keys(session_id)  # for attention
    bank.flush_session(session_id)                   # session end
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnRecord:
    """A single turn's encoder output and metadata."""
    turn_id: int
    encoder_output: np.ndarray  # (384,) float32
    hexagram_id: int
    summary: str = ""
    timestamp: float = 0.0


@dataclass
class SessionBuffer:
    """In-memory buffer for one session's turns."""
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    
    def add(self, record: TurnRecord) -> None:
        self.turns.append(record)
    
    def get_keys(self) -> np.ndarray:
        """Return all encoder outputs as (N, 384) matrix."""
        if not self.turns:
            return np.empty((0, 384), dtype=np.float32)
        return np.stack([t.encoder_output for t in self.turns])
    
    def get_metadata(self) -> list[dict]:
        """Return turn metadata for prescription generation."""
        return [
            {
                "turn_id": t.turn_id,
                "hexagram_id": t.hexagram_id,
                "summary": t.summary,
            }
            for t in self.turns
        ]
    
    def size_bytes(self) -> int:
        """Approximate memory usage of this buffer."""
        if not self.turns:
            return 0
        vec_bytes = len(self.turns) * 384 * 4  # float32
        meta_bytes = sum(len(t.summary) for t in self.turns)
        return vec_bytes + meta_bytes
    
    def clear(self) -> None:
        self.turns.clear()


class MemoryBank:
    """Per-session turn storage. Implements IMemoryBank.

    One instance per engine; manages all sessions in-process.
    Sessions are isolated — no cross-session leakage.
    """
    
    def __init__(self, max_turns_per_session: int = 5000):
        self._sessions: dict[str, SessionBuffer] = {}
        self._max_turns = max_turns_per_session
    
    def init_session(self, session_id: str) -> None:
        """Create or reset a session buffer."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(session_id=session_id)
    
    def store_turn(
        self,
        session_id: str,
        turn_id: int,
        encoder_output: np.ndarray,
        hexagram_id: int,
        summary: str = "",
        timestamp: float = 0.0,
    ) -> None:
        """Store one turn's encoder output.
        
        Args:
            session_id: LOOM session identifier
            turn_id: sequential turn number within session
            encoder_output: (384,) float32 array from TinyEncoder
            hexagram_id: YiCeNet hexagram index (0-63)
            summary: optional short text summary for LLM injection
            timestamp: unix timestamp
        """
        if session_id not in self._sessions:
            self.init_session(session_id)
        
        buf = self._sessions[session_id]
        if len(buf.turns) >= self._max_turns:
            # FIFO eviction: remove oldest turn
            buf.turns.pop(0)
        
        record = TurnRecord(
            turn_id=turn_id,
            encoder_output=encoder_output.astype(np.float32).copy(),
            hexagram_id=hexagram_id,
            summary=summary,
            timestamp=timestamp or 0.0,
        )
        buf.add(record)
    
    def get_session_keys(self, session_id: str) -> tuple[np.ndarray, list[dict]]:
        """Get all stored encoder outputs and metadata for a session.
        
        Returns:
            keys: (N, 384) matrix of encoder outputs
            metadata: list of dicts with turn_id, hexagram_id, summary
        """
        buf = self._sessions.get(session_id)
        if buf is None or not buf.turns:
            return np.empty((0, 384), dtype=np.float32), []
        return buf.get_keys(), buf.get_metadata()
    
    def get_turn_count(self, session_id: str) -> int:
        """Number of turns stored for a session."""
        buf = self._sessions.get(session_id)
        return len(buf.turns) if buf else 0

    def get_hexagram_history(self, session_id: str) -> list[int]:
        """Return ordered list of hexagram IDs for this session (oldest first).

        Excludes turns where hexagram_id < 0 (attend() calls without routing).
        Used by the engine to compute hexagram chain dynamics for env_vec.
        """
        buf = self._sessions.get(session_id)
        if buf is None:
            return []
        return [t.hexagram_id for t in buf.turns if t.hexagram_id >= 0]
    
    def flush_session(self, session_id: str) -> list[TurnRecord]:
        """Clear session buffer and return the records (for optional solidify).
        
        Called when LOOM closes a conversation.
        """
        buf = self._sessions.pop(session_id, None)
        if buf is None:
            return []
        records = list(buf.turns)
        buf.clear()
        return records
    
    def get_active_sessions(self) -> list[str]:
        """Return list of session IDs currently in memory."""
        return list(self._sessions.keys())
    
    def memory_usage(self) -> dict[str, int]:
        """Approximate memory usage per session."""
        return {
            sid: buf.size_bytes()
            for sid, buf in self._sessions.items()
        }
    
    def clear_all(self) -> None:
        """Clear all session buffers."""
        for buf in self._sessions.values():
            buf.clear()
        self._sessions.clear()


# Global singleton (one per process)
_default_bank: Optional[MemoryBank] = None


def get_memory_bank() -> MemoryBank:
    """Get or create the global MemoryBank singleton."""
    global _default_bank
    if _default_bank is None:
        _default_bank = MemoryBank()
    return _default_bank
