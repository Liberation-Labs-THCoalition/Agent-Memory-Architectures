"""SQLite Persistence — durable storage for the memory journal.

Lightweight adapter that backs the MemoryJournal with SQLite
for longitudinal analysis. Stores geometry readings and
consolidation snapshots.

Author: Operator (Coalition)
Original design: Oracle Harness (Liberation Labs)
Adapted for public release: 2026-04-18
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DEFAULT_DB = os.path.expanduser("~/.oracle-memory/journal.db")


class PersistentStore:
    """SQLite-backed persistence for geometry readings and consolidation.

    Usage::

        store = PersistentStore()

        # Record a geometry reading
        store.record_geometry("snap_001", "turn_3", {
            "effective_rank": 60.8,
            "spectral_entropy": 20.2,
        })

        # Query trends
        trend = store.get_trend(hours=24)
        print(f"Average rank: {trend['avg_rank']}")

        # Save consolidation snapshot
        store.save_consolidation(trend)
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS geometry_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT NOT NULL,
                    checkpoint TEXT,
                    timestamp REAL,
                    effective_rank REAL,
                    spectral_entropy REAL,
                    norm_per_token REAL,
                    top_sv_ratio REAL,
                    key_norm REAL,
                    extra TEXT
                );

                CREATE TABLE IF NOT EXISTS consolidation_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    window_hours REAL,
                    total_readings INTEGER,
                    avg_rank REAL,
                    avg_entropy REAL,
                    drift_score REAL,
                    drift_direction TEXT,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_geo_time
                    ON geometry_readings(timestamp);
                CREATE INDEX IF NOT EXISTS idx_consol_time
                    ON consolidation_snapshots(timestamp);
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_geometry(
        self, snapshot_id: str, checkpoint: str, geometry: dict
    ):
        """Store a geometry reading."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO geometry_readings
                   (snapshot_id, checkpoint, timestamp, effective_rank,
                    spectral_entropy, norm_per_token, top_sv_ratio,
                    key_norm, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id, checkpoint, time.time(),
                    geometry.get("effective_rank"),
                    geometry.get("spectral_entropy"),
                    geometry.get("norm_per_token"),
                    geometry.get("top_sv_ratio"),
                    geometry.get("key_norm"),
                    json.dumps({
                        k: v for k, v in geometry.items()
                        if k not in (
                            "effective_rank", "spectral_entropy",
                            "norm_per_token", "top_sv_ratio", "key_norm",
                        )
                    }) if geometry else None,
                ),
            )

    def get_geometry_history(self, last_n: int = 200) -> list[dict]:
        """Get recent geometry readings."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM geometry_readings
                   ORDER BY timestamp DESC LIMIT ?""",
                (last_n,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_trend(self, hours: float = 24.0) -> dict:
        """Compute geometry trend over a time window."""
        cutoff = time.time() - (hours * 3600)
        with self._conn() as conn:
            stats = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    AVG(effective_rank) as avg_rank,
                    AVG(spectral_entropy) as avg_entropy,
                    AVG(norm_per_token) as avg_norm
                FROM geometry_readings WHERE timestamp > ?""",
                (cutoff,),
            ).fetchone()

        total = stats["total"] if stats else 0
        return {
            "window_hours": hours,
            "total_readings": total,
            "avg_rank": stats["avg_rank"] if stats and total > 0 else None,
            "avg_entropy": stats["avg_entropy"] if stats and total > 0 else None,
            "avg_norm": stats["avg_norm"] if stats and total > 0 else None,
        }

    def save_consolidation(self, trend: dict, notes: str = ""):
        """Persist a consolidation snapshot."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO consolidation_snapshots
                   (timestamp, window_hours, total_readings,
                    avg_rank, avg_entropy, drift_score,
                    drift_direction, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    trend.get("window_hours"),
                    trend.get("total_readings"),
                    trend.get("avg_rank"),
                    trend.get("avg_entropy"),
                    None, "stable", notes,
                ),
            )
