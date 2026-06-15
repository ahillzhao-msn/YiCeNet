"""
ClaudeCodeDataSource — scans Claude Code session transcripts.

Claude Code stores conversations as JSONL files under:
  ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl

Each line is a transcript entry. The fields we care about:

  {
    "type": "say",
    "message": {
      "role": "user" | "assistant",
      "content": "<string>" | [{"type":"text","text":"..."},...]
    },
    "timestamp": "2025-06-15T10:23:00.000Z",   # ISO-8601
    "sessionId": "<uuid>",
    "cwd": "/path/to/project"
  }

Tool-use entries (type="tool_use", "tool_result") are skipped for signal
extraction — we only look at the narrative user↔assistant message pairs.

Signal extraction reuses external_metrics.py (same patterns as Hermes path)
so the flywheel sees homogeneous Sample objects regardless of source.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from yicenet.external_metrics import (
    compute_satisfaction,
    estimate_token_cost,
    estimate_response_length,
    _check_patterns,
    _CORRECTION_PATTERNS,
    _COMPLETION_PATTERNS,
    _PRAISE_PATTERNS,
    _ABANDON_PATTERNS,
)
from . import DataSource, Sample

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _parse_iso(ts_str: str) -> float:
    """Parse ISO-8601 timestamp (with or without Z suffix) → Unix float."""
    try:
        return datetime.fromisoformat(
            ts_str.replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _extract_text(content) -> str:
    """Normalise Claude Code content field: str or list[block] → plain str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def _get_role(entry: dict) -> str:
    """Extract role from a transcript entry (handles nested message.role)."""
    msg = entry.get("message")
    if isinstance(msg, dict):
        return msg.get("role", "")
    return entry.get("role", "")


def _get_content_text(entry: dict) -> str:
    """Extract plain text from a transcript entry."""
    msg = entry.get("message")
    if isinstance(msg, dict):
        return _extract_text(msg.get("content", ""))
    return _extract_text(entry.get("content", ""))


def _is_narrative(entry: dict) -> bool:
    """True for user/assistant say entries; False for tool_use/tool_result."""
    t = entry.get("type", "")
    if t not in ("say", "", "message"):
        return False
    role = _get_role(entry)
    return role in ("user", "assistant")


class ClaudeCodeDataSource(DataSource):
    """
    Scans ~/.claude/projects/ for new conversation turns since a timestamp.

    Pairs each user message with its immediate assistant response, then looks
    one step further for the follow-up user message to extract a reward signal
    — mirroring exactly what HermesDataSource does with the Hermes SQL query.

    Efficient incremental scanning:
      1. Skip JSONL files whose mtime < since (filesystem-level filter)
      2. Skip individual entries with timestamp < since (line-level filter)
      3. Only iterate forward once per file (O(n) per file, O(files) total)
    """

    def __init__(self, projects_dir: Optional[Path] = None):
        self._dir = Path(projects_dir) if projects_dir else CLAUDE_PROJECTS_DIR

    @property
    def source_id(self) -> str:
        return "claude-code"

    def is_available(self) -> bool:
        return self._dir.exists()

    def scan_since(self, timestamp: float) -> list[Sample]:
        if not self.is_available():
            return []

        samples: list[Sample] = []
        for jsonl_path in self._dir.glob("**/*.jsonl"):
            # Fast pre-filter: skip files not touched since `timestamp`
            try:
                if jsonl_path.stat().st_mtime < timestamp - 1:
                    continue
            except OSError:
                continue
            try:
                samples.extend(self._parse_file(jsonl_path, since=timestamp))
            except Exception:
                continue

        return samples

    # ── private ──────────────────────────────────────────────────────────────

    def _parse_file(self, path: Path, since: float) -> list[Sample]:
        """Parse one Claude Code JSONL file into (user→assistant) Sample pairs."""
        # Read all narrative entries in one pass
        entries: list[dict] = []
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and _is_narrative(obj):
                    entries.append(obj)
        except OSError:
            return []

        if not entries:
            return []

        # Derive session_id: prefer sessionId field, fall back to filename stem
        session_id = entries[0].get("sessionId", path.stem)

        samples: list[Sample] = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            if _get_role(entry) != "user":
                i += 1
                continue

            ts = _parse_iso(entry.get("timestamp", ""))
            if ts < since:
                i += 1
                continue

            user_text = _get_content_text(entry)
            if not user_text.strip():
                i += 1
                continue

            # Find the immediately following assistant entry
            j = i + 1
            while j < len(entries) and _get_role(entries[j]) != "assistant":
                j += 1
            if j >= len(entries):
                break  # incomplete pair at end of file — skip

            asst_text = _get_content_text(entries[j])

            # Look one more step for the follow-up user message (signal source)
            k = j + 1
            while k < len(entries) and _get_role(entries[k]) != "user":
                k += 1
            next_text = _get_content_text(entries[k]) if k < len(entries) else ""

            satisfaction  = compute_satisfaction(next_text or None, user_text)
            token_cost    = estimate_token_cost(asst_text or user_text)
            response_len  = estimate_response_length(next_text)
            corrected     = _check_patterns(next_text, _CORRECTION_PATTERNS)
            completed     = _check_patterns(next_text, _COMPLETION_PATTERNS)
            praised       = _check_patterns(next_text, _PRAISE_PATTERNS)
            abandoned     = _check_patterns(next_text, _ABANDON_PATTERNS) or not next_text
            continued     = bool(next_text) and not abandoned

            samples.append(Sample(
                conversation_id=f"claude-code:{session_id}",
                source=self.source_id,
                user_text=user_text,
                assistant_text=asst_text,
                next_user_text=next_text,
                timestamp=ts,
                satisfaction=satisfaction,
                token_cost=token_cost,
                response_length=response_len,
                continued=continued,
                corrected=corrected,
                completed=completed,
                praised=praised,
                abandoned=abandoned,
                source_msg_id=entry.get("uuid", ""),
            ))

            i = j + 1  # advance past the assistant entry we consumed

        return samples
