"""
HermesDataSource — scans the Hermes SQLite session database.

Extracts the existing scan_new_messages() logic from flywheel.py into a
self-contained DataSource so flywheel.py no longer contains Hermes-specific SQL.

Schema assumed:
  messages(id INTEGER, content TEXT, role TEXT, session_id TEXT, timestamp REAL)
  role values: 'user' | 'assistant'
"""

from __future__ import annotations

import sqlite3
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

# Default Hermes DB location (overridable via HERMES_HOME env var)
import os as _os
_HERMES_HOME = _os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
DEFAULT_DB_PATH = Path(_HERMES_HOME) / "state.db"

# SQL: fetch user→assistant pairs with the immediate next user follow-up.
# Uses the same join logic as the original scan_new_messages() in flywheel.py.
_QUERY = """
    SELECT m1.id, m1.content, m1.session_id, m1.timestamp,
           m2.content AS asst_content,
           (SELECT content FROM messages m3
            WHERE m3.session_id = m1.session_id
              AND m3.role = 'user'
              AND m3.timestamp > m2.timestamp
            ORDER BY m3.timestamp LIMIT 1) AS next_user_text
    FROM messages m1
    JOIN messages m2
      ON m2.session_id = m1.session_id
     AND m2.role = 'assistant'
     AND m2.timestamp > m1.timestamp
     AND m2.timestamp = (
         SELECT MIN(timestamp) FROM messages
         WHERE session_id = m1.session_id
           AND role = 'assistant'
           AND timestamp > m1.timestamp
     )
    WHERE m1.role = 'user'
      AND m1.timestamp > ?
      AND m1.content IS NOT NULL
      AND length(m1.content) > 3
    ORDER BY m1.id
    LIMIT 200
"""


class HermesDataSource(DataSource):
    """Reads user→assistant conversation pairs from the Hermes SQLite DB."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db = Path(db_path) if db_path else DEFAULT_DB_PATH

    @property
    def source_id(self) -> str:
        return "hermes"

    def is_available(self) -> bool:
        return self._db.exists()

    def scan_since(self, timestamp: float) -> list[Sample]:
        if not self.is_available():
            return []

        try:
            conn = sqlite3.connect(str(self._db))
            rows = conn.execute(_QUERY, (timestamp,)).fetchall()
            conn.close()
        except sqlite3.Error:
            return []

        samples = []
        for msg_id, content, session_id, ts, asst_text, next_text in rows:
            asst_text = asst_text or ""
            next_text = next_text or ""

            satisfaction  = compute_satisfaction(next_text or None, content)
            token_cost    = estimate_token_cost(asst_text or content)
            response_len  = estimate_response_length(next_text)
            corrected     = _check_patterns(next_text, _CORRECTION_PATTERNS)
            completed     = _check_patterns(next_text, _COMPLETION_PATTERNS)
            praised       = _check_patterns(next_text, _PRAISE_PATTERNS)
            abandoned     = _check_patterns(next_text, _ABANDON_PATTERNS) or not next_text
            continued     = bool(next_text) and not abandoned

            samples.append(Sample(
                conversation_id=f"hermes:{session_id}",
                source=self.source_id,
                user_text=content,
                assistant_text=asst_text,
                next_user_text=next_text,
                timestamp=float(ts),
                satisfaction=satisfaction,
                token_cost=token_cost,
                response_length=response_len,
                continued=continued,
                corrected=corrected,
                completed=completed,
                praised=praised,
                abandoned=abandoned,
                source_msg_id=str(msg_id),
            ))

        return samples
