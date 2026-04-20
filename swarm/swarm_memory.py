"""Swarm Memory — communal memory for the microagent swarm.

Records every finding, roast, and false positive from the pipeline.
Over time, builds a dataset of small-model behavior patterns:
- What do 1.5B models consistently hallucinate?
- Which vulnerability types get over-flagged?
- What patterns survive Opus review vs get rejected?

Also stores the occasional gem of a roast for posterity.

Author: Nexus (Coalition)
Date: 2026-04-20
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

DB_PATH = os.path.expanduser("~/.oracle/swarm_memory.db")


@dataclass
class SwarmFinding:
    """A single finding from a microagent."""
    agent: str            # "linter", "reviewer", "security"
    file_path: str
    finding: str
    severity: str         # "clean", "info", "warning", "error"
    model: str            # "qwen2.5-coder:1.5b"
    timestamp: float
    confirmed: Optional[bool] = None  # True=valid, False=false positive, None=unreviewed
    opus_notes: Optional[str] = None  # Notes from Opus review if escalated
    is_roast: bool = False            # Particularly entertaining false positive


class SwarmMemory:
    """SQLite-backed communal memory for the microagent swarm.

    Usage::

        mem = SwarmMemory()

        # Record findings from a pipeline run
        mem.record("security", "scrub.py", "SQL injection in code with no SQL",
                   "error", model="qwen2.5-coder:1.5b", is_roast=True)

        # Mark a finding as false positive after Opus review
        mem.mark_false_positive(finding_id, notes="No SQL in this file")

        # Get the hall of fame
        roasts = mem.get_roasts(limit=10)

        # Get false positive patterns
        patterns = mem.get_false_positive_patterns()

        # Stats
        stats = mem.stats()
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    finding TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    model TEXT DEFAULT 'qwen2.5-coder:1.5b',
                    timestamp REAL,
                    confirmed INTEGER,
                    opus_notes TEXT,
                    is_roast INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    files_reviewed INTEGER,
                    total_findings INTEGER,
                    escalated INTEGER,
                    clean INTEGER,
                    duration_ms REAL
                );

                CREATE TABLE IF NOT EXISTS solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    finding_pattern TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    file_pattern TEXT,
                    times_applied INTEGER DEFAULT 0,
                    created_at REAL,
                    last_applied REAL
                );

                CREATE INDEX IF NOT EXISTS idx_findings_agent ON findings(agent);
                CREATE INDEX IF NOT EXISTS idx_findings_confirmed ON findings(confirmed);
                CREATE INDEX IF NOT EXISTS idx_findings_roast ON findings(is_roast);
                CREATE INDEX IF NOT EXISTS idx_solutions_pattern ON solutions(finding_pattern);
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

    def record(
        self, agent: str, file_path: str, finding: str, severity: str,
        model: str = "qwen2.5-coder:1.5b", is_roast: bool = False,
    ) -> int:
        """Record a finding from a microagent."""
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO findings
                   (agent, file_path, finding, severity, model, timestamp, is_roast)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (agent, file_path, finding, severity, model, time.time(), int(is_roast)),
            )
            return cursor.lastrowid

    def record_pipeline_run(
        self, files: int, findings: int, escalated: int,
        clean: int, duration_ms: float,
    ):
        """Record a pipeline run summary."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO pipeline_runs
                   (timestamp, files_reviewed, total_findings, escalated, clean, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (time.time(), files, findings, escalated, clean, duration_ms),
            )

    def mark_confirmed(self, finding_id: int, confirmed: bool, notes: str = ""):
        """Mark a finding as confirmed valid or false positive."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE findings SET confirmed = ?, opus_notes = ? WHERE id = ?",
                (int(confirmed), notes, finding_id),
            )

    def mark_false_positive(self, finding_id: int, notes: str = ""):
        """Convenience: mark as false positive."""
        self.mark_confirmed(finding_id, False, notes)

    def mark_as_roast(self, finding_id: int):
        """Promote a finding to the hall of fame."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE findings SET is_roast = 1 WHERE id = ?",
                (finding_id,),
            )

    def get_roasts(self, limit: int = 10) -> list:
        """Get the hall of fame — best false positives."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT agent, file_path, finding, severity, model
                   FROM findings WHERE is_roast = 1
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()

    def get_false_positive_patterns(self) -> list:
        """Get the most common false positive types by agent."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT agent, finding, COUNT(*) as count
                   FROM findings
                   WHERE confirmed = 0
                   GROUP BY agent, finding
                   ORDER BY count DESC
                   LIMIT 20""",
            ).fetchall()

    def get_unreviewed(self, limit: int = 20) -> list:
        """Get findings that haven't been reviewed yet."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT id, agent, file_path, finding, severity
                   FROM findings WHERE confirmed IS NULL
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()

    def record_solution(self, finding_pattern: str, agent: str,
                         solution: str, file_pattern: str = None):
        """Record a solution for a recurring finding pattern."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO solutions
                   (finding_pattern, agent, solution, file_pattern, created_at, last_applied)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (finding_pattern, agent, solution, file_pattern, time.time(), time.time()),
            )

    def find_solution(self, finding: str, agent: str) -> Optional[str]:
        """Look up a known solution for a finding pattern."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT solution, id FROM solutions
                   WHERE agent = ? AND ? LIKE '%' || finding_pattern || '%'
                   ORDER BY times_applied DESC LIMIT 1""",
                (agent, finding),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE solutions SET times_applied = times_applied + 1, last_applied = ? WHERE id = ?",
                    (time.time(), row['id']),
                )
                return row['solution']
        return None

    def get_solutions(self, limit: int = 20) -> list:
        """Get most-applied solutions."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT finding_pattern, agent, solution, times_applied
                   FROM solutions ORDER BY times_applied DESC LIMIT ?""",
                (limit,),
            ).fetchall()

    def stats(self) -> dict:
        """Get swarm memory statistics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
            confirmed = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE confirmed = 1"
            ).fetchone()[0]
            false_pos = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE confirmed = 0"
            ).fetchone()[0]
            unreviewed = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE confirmed IS NULL"
            ).fetchone()[0]
            roasts = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE is_roast = 1"
            ).fetchone()[0]
            runs = conn.execute(
                "SELECT COUNT(*) FROM pipeline_runs"
            ).fetchone()[0]

            # Per-agent stats
            agents = conn.execute(
                """SELECT agent,
                    COUNT(*) as total,
                    SUM(CASE WHEN confirmed = 0 THEN 1 ELSE 0 END) as false_pos
                   FROM findings GROUP BY agent"""
            ).fetchall()

        agent_stats = {}
        for a in agents:
            fp_rate = a['false_pos'] / max(a['total'], 1)
            agent_stats[a['agent']] = {
                'total': a['total'],
                'false_positives': a['false_pos'],
                'fp_rate': f"{fp_rate:.0%}",
            }

        return {
            'total_findings': total,
            'confirmed_valid': confirmed,
            'false_positives': false_pos,
            'unreviewed': unreviewed,
            'roasts': roasts,
            'pipeline_runs': runs,
            'per_agent': agent_stats,
        }
