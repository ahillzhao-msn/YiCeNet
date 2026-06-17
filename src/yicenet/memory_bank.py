"""
YiCeNet MemoryBank — per-session turn storage for cross-attention.

Stores TinyEncoder 384d outputs per turn, scoped by session_id.
Append-only within a session; flushed on conversation close.

Architecture
------------
MemoryBank is the single business abstraction for cross-turn state.
An optional PersistenceBackend makes it survive process restarts —
required for Claude Code hooks (short-lived subprocesses) and optional
for long-lived daemons (Hermes, Loom) that want crash recovery.

Storage cost: ~1.5KB per turn (384 × float32) + negligible metadata.

Usage:
    bank = get_memory_bank()          # configured singleton
    bank.store_turn(sid, tid, vec, hid, summary, metadata={"key": "val"})
    bank.update_turn_metadata(sid, tid, {"response_snippet": "..."})
    keys, meta = bank.get_session_keys(sid)
    last = bank.get_last_turn(sid)
    bank.flush_session(sid)           # clears memory; backend file kept
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


# ── Domain objects ────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """A single turn's encoder output and metadata."""
    turn_id: int
    encoder_output: np.ndarray      # (384,) float32
    hexagram_id: int
    summary: str = ""
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)
    """Flexible per-turn extras: response_snippet, response_char_count,
    platform_signals, etc.  Populated incrementally across hooks."""


@dataclass
class SessionBuffer:
    """In-memory buffer for one session's turns."""
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)

    def add(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def get_keys(self) -> np.ndarray:
        if not self.turns:
            return np.empty((0, 384), dtype=np.float32)
        return np.stack([t.encoder_output for t in self.turns])

    def get_metadata(self) -> list[dict]:
        return [
            {"turn_id": t.turn_id, "hexagram_id": t.hexagram_id, "summary": t.summary}
            for t in self.turns
        ]

    def size_bytes(self) -> int:
        if not self.turns:
            return 0
        return len(self.turns) * 384 * 4 + sum(len(t.summary) for t in self.turns)

    def clear(self) -> None:
        self.turns.clear()


# ── Persistence layer ─────────────────────────────────────────────────────────

@runtime_checkable
class PersistenceBackend(Protocol):
    """Storage interface for MemoryBank.  Implementations must be safe for
    single-writer access per session (one hook fires at a time per session)."""

    def append(self, session_id: str, record: TurnRecord) -> None: ...
    def update(self, session_id: str, turn_id: int, metadata: dict) -> None: ...
    def load(self, session_id: str) -> list[TurnRecord]: ...
    def delete(self, session_id: str) -> None: ...
    def list_sessions(self) -> list[str]: ...


class FileBackend:
    """JSONL file-per-session persistence.

    Layout: <data_dir>/memory/<session_id>.jsonl
    One JSON line per turn.  Vectors are stored only when store_vectors=True
    (daemon processes with full attention-history replay needs).

    Subprocess hooks (Claude Code) set store_vectors=False: they need only
    metadata (hexagram_id, response_snippet) for feedback extraction and
    the overhead of base64-encoding 384d vectors per turn is unnecessary.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        store_vectors: bool = False,
    ) -> None:
        from yicenet.config import yicenet_data_dir
        self._dir = Path(base_dir) if base_dir else yicenet_data_dir() / "memory"
        self._store_vectors = store_vectors

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.jsonl"

    def append(self, session_id: str, record: TurnRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {
            "turn_id": record.turn_id,
            "hexagram_id": record.hexagram_id,
            "summary": record.summary,
            "timestamp": record.timestamp,
            "metadata": record.metadata,
        }
        if self._store_vectors and record.encoder_output is not None:
            import base64
            entry["vector_b64"] = base64.b64encode(
                record.encoder_output.astype(np.float32).tobytes()
            ).decode("ascii")
        with open(self._path(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update(self, session_id: str, turn_id: int, metadata: dict) -> None:
        """Merge metadata into a specific turn's entry (rewrites the file)."""
        path = self._path(session_id)
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        updated = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("turn_id") == turn_id:
                    obj.setdefault("metadata", {}).update(metadata)
            except (json.JSONDecodeError, KeyError):
                pass
            updated.append(json.dumps(obj, ensure_ascii=False))
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def load(self, session_id: str) -> list[TurnRecord]:
        path = self._path(session_id)
        if not path.exists():
            return []
        records: list[TurnRecord] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vector = np.zeros(384, dtype=np.float32)
                if "vector_b64" in obj:
                    import base64
                    vector = np.frombuffer(
                        base64.b64decode(obj["vector_b64"]), dtype=np.float32
                    ).copy()
                records.append(TurnRecord(
                    turn_id=int(obj["turn_id"]),
                    encoder_output=vector,
                    hexagram_id=int(obj["hexagram_id"]),
                    summary=obj.get("summary", ""),
                    timestamp=float(obj.get("timestamp", 0.0)),
                    metadata=obj.get("metadata", {}),
                ))
            except Exception:
                continue
        return records

    def delete(self, session_id: str) -> None:
        try:
            self._path(session_id).unlink(missing_ok=True)
        except Exception:
            pass

    def list_sessions(self) -> list[str]:
        try:
            return [p.stem for p in self._dir.glob("*.jsonl")]
        except Exception:
            return []


# ── MemoryBank ────────────────────────────────────────────────────────────────

class MemoryBank:
    """Per-session turn storage.  Implements IMemoryBank.

    One instance per process; manages all sessions.
    Sessions are isolated — no cross-session leakage.

    backend (optional): when set, every store_turn() write goes through to
    disk and init_session() loads prior turns from disk.  flush_session()
    clears in-memory state but deliberately does NOT delete the backend file
    so that the next process (e.g. the next hook subprocess) can reload it.
    """

    def __init__(
        self,
        max_turns_per_session: int = 5000,
        backend: Optional[PersistenceBackend] = None,
    ) -> None:
        self._sessions: dict[str, SessionBuffer] = {}
        self._max_turns = max_turns_per_session
        self._backend = backend

    # ── session lifecycle ─────────────────────────────────────────────────────

    def init_session(self, session_id: str) -> None:
        """Create or restore a session buffer.

        When a backend is configured and the session is not yet in memory,
        loads prior turns from the backend file (cross-process restore).
        """
        if session_id in self._sessions:
            return
        buf = SessionBuffer(session_id=session_id)
        self._sessions[session_id] = buf
        if self._backend:
            for record in self._backend.load(session_id):
                buf.add(record)

    def flush_session(self, session_id: str) -> list[TurnRecord]:
        """Clear in-memory buffer; return the records.

        Does NOT delete the backend file — intentional.  Hook subprocesses
        die after flush_session(); the file survives for the next hook.
        Long-lived daemons call this at conversation close.
        """
        buf = self._sessions.pop(session_id, None)
        if buf is None:
            return []
        records = list(buf.turns)
        buf.clear()
        return records

    def delete_session(self, session_id: str) -> None:
        """Permanently remove session from memory AND backend."""
        self._sessions.pop(session_id, None)
        if self._backend:
            self._backend.delete(session_id)

    # ── turn storage ──────────────────────────────────────────────────────────

    def store_turn(
        self,
        session_id: str,
        turn_id: int,
        encoder_output: np.ndarray,
        hexagram_id: int,
        summary: str = "",
        timestamp: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> None:
        if session_id not in self._sessions:
            self.init_session(session_id)
        buf = self._sessions[session_id]
        if len(buf.turns) >= self._max_turns:
            buf.turns.pop(0)
        record = TurnRecord(
            turn_id=turn_id,
            encoder_output=encoder_output.astype(np.float32).copy(),
            hexagram_id=hexagram_id,
            summary=summary,
            timestamp=timestamp or 0.0,
            metadata=dict(metadata) if metadata else {},
        )
        buf.add(record)
        if self._backend:
            self._backend.append(session_id, record)

    def update_turn_metadata(
        self, session_id: str, turn_id: int, metadata: dict
    ) -> None:
        """Merge metadata into an existing turn (in-memory + backend)."""
        buf = self._sessions.get(session_id)
        if buf:
            for r in buf.turns:
                if r.turn_id == turn_id:
                    r.metadata.update(metadata)
                    break
        if self._backend:
            self._backend.update(session_id, turn_id, metadata)

    # ── queries ───────────────────────────────────────────────────────────────

    def get_last_turn(self, session_id: str) -> Optional[TurnRecord]:
        """Return the most recently stored TurnRecord, or None."""
        buf = self._sessions.get(session_id)
        if buf and buf.turns:
            return buf.turns[-1]
        return None

    def get_session_keys(
        self, session_id: str
    ) -> tuple[np.ndarray, list[dict]]:
        buf = self._sessions.get(session_id)
        if buf is None or not buf.turns:
            return np.empty((0, 384), dtype=np.float32), []
        return buf.get_keys(), buf.get_metadata()

    def get_turn_count(self, session_id: str) -> int:
        buf = self._sessions.get(session_id)
        return len(buf.turns) if buf else 0

    def get_hexagram_history(self, session_id: str) -> list[int]:
        buf = self._sessions.get(session_id)
        if buf is None:
            return []
        return [t.hexagram_id for t in buf.turns if t.hexagram_id >= 0]

    def get_active_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def memory_usage(self) -> dict[str, int]:
        return {sid: buf.size_bytes() for sid, buf in self._sessions.items()}

    def clear_all(self) -> None:
        for buf in self._sessions.values():
            buf.clear()
        self._sessions.clear()


# ── Singleton + factory ───────────────────────────────────────────────────────

_default_bank: Optional[MemoryBank] = None


def configure_memory_bank_for(adapter) -> None:
    """Configure the global MemoryBank singleton for the given platform adapter.

    Must be called once at process startup, before any code imports
    get_memory_bank().  Idempotent: subsequent calls are no-ops.

    The adapter's process_model property ('subprocess' | 'daemon') determines
    whether a FileBackend is attached:
      - 'subprocess': FileBackend always attached (CC hooks need cross-process state)
      - 'daemon':     FileBackend attached only when persist_daemon_sessions=True
    """
    global _default_bank
    if _default_bank is not None:
        return

    try:
        from yicenet.config import load_user_config
        cfg = load_user_config()
    except Exception:
        cfg = {}

    need_backend = adapter.process_model == "subprocess" or bool(
        cfg.get("memory", {}).get("persist_daemon_sessions", False)
    )
    store_vectors = bool(
        cfg.get("memory", {}).get("store_vectors", False)
    )
    backend = FileBackend(store_vectors=store_vectors) if need_backend else None
    _default_bank = MemoryBank(backend=backend)


def get_memory_bank() -> MemoryBank:
    """Get or create the global MemoryBank singleton.

    If configure_memory_bank_for() was not called first (e.g. engine used
    directly without a hook runner), returns a plain in-memory bank with
    no persistence backend.
    """
    global _default_bank
    if _default_bank is None:
        _default_bank = MemoryBank()
    return _default_bank
