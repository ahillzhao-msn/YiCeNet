"""
Metrics logger for YiCeNet — bridges inference engine ↔ SQLite dashboard.

Call from inference engine after each predict:
    from metrics import MetricsLogger
    MetricsLogger().log_trajectory(...)

Call from training worker:
    MetricsLogger().log_evaluation(...)
    MetricsLogger().log_hexagram_usage(...)
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional


class MetricsLogger:
    """Thread-safe single-instance logger to SQLite."""

    _instance: Optional["MetricsLogger"] = None

    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._db_path = db_path or os.environ.get(
                "DB_PATH",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "metrics.db")
            )
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    hexagram_id INTEGER,
                    candidate_values TEXT,
                    action_id INTEGER,
                    reward REAL,
                    terminal_type TEXT DEFAULT 'active',
                    latency_ms REAL,
                    token_cost REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT,
                    avg_reward REAL,
                    win_rate REAL,
                    episodes INTEGER,
                    duration_sec REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hexagram_usage (
                    date TEXT,
                    hexagram_id INTEGER,
                    count INTEGER DEFAULT 0,
                    avg_q_value REAL,
                    PRIMARY KEY (date, hexagram_id)
                )
            """)

    def log_trajectory(
        self,
        session_id: str,
        hexagram_id: int,
        candidate_values: list,
        action_id: int,
        reward: float,
        terminal_type: str = "active",
        latency_ms: float = 0.0,
        token_cost: float = 0.0,
    ):
        """Log a single inference trajectory."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO trajectories
                   (session_id, hexagram_id, candidate_values, action_id,
                    reward, terminal_type, latency_ms, token_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, hexagram_id,
                 json.dumps(candidate_values), action_id,
                 reward, terminal_type, latency_ms, token_cost)
            )

    def log_evaluation(
        self,
        version: str,
        avg_reward: float,
        win_rate: float,
        episodes: int,
        duration_sec: float,
    ):
        """Log a training evaluation event."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO evaluations
                   (version, avg_reward, win_rate, episodes, duration_sec)
                   VALUES (?, ?, ?, ?, ?)""",
                (version, avg_reward, win_rate, episodes, duration_sec)
            )

    def log_hexagram_usage(self, hexagram_id: int, q_value: float):
        """Increment daily hexagram usage counter."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO hexagram_usage (date, hexagram_id, count, avg_q_value)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(date, hexagram_id) DO UPDATE SET
                       count = count + 1,
                       avg_q_value = (avg_q_value + ?) / 2.0""",
                (today, hexagram_id, q_value, q_value)
            )

    def get_stats(self) -> dict:
        """Quick summary for health check."""
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE terminal_type='success'"
            ).fetchone()[0]
            abandon = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE terminal_type='abandoned'"
            ).fetchone()[0]
        return {
            "total_trajectories": total,
            "success": success,
            "abandoned": abandon,
            "success_rate": success / max(total, 1),
        }
