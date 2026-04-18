"""Memory Journal — append-only event log for cache recordings.

Logs cache snapshots, geometry extractions, memory operations, and
consolidation events. General-purpose event journal — no inference
pipeline specifics.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from oracle_memory.src.types import (
    GeometrySummary,
    JournalEntry,
    JournalEventType,
)

logger = logging.getLogger(__name__)


class MemoryJournal:
    """Append-only event log for memory operations.

    Tracks cache snapshots, geometry readings, memory stores, and
    consolidation passes. Optional JSONL persistence.
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._entries: list[JournalEntry] = []
        self._persist_dir: Optional[Path] = None

        if persist_dir:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    def record(self, entry: JournalEntry) -> None:
        """Record a journal entry."""
        self._entries.append(entry)
        if self._persist_dir:
            self._persist_to_disk(entry)

    def record_snapshot(self, snapshot_id: str, label: str, n_tokens: int = 0) -> None:
        """Record a cache snapshot event."""
        self.record(JournalEntry.create(
            JournalEventType.CACHE_SNAPSHOT,
            snapshot_id=snapshot_id, label=label, n_tokens=n_tokens,
        ))

    def record_geometry(self, snapshot_id: str, geometry: GeometrySummary) -> None:
        """Record a geometry extraction from a cache snapshot."""
        self.record(JournalEntry.create(
            JournalEventType.GEOMETRY_EXTRACTED,
            snapshot_id=snapshot_id,
            key_norm=geometry.key_norm,
            effective_rank=geometry.effective_rank,
            spectral_entropy=geometry.spectral_entropy,
            top_sv_ratio=geometry.top_sv_ratio,
            n_tokens=geometry.n_tokens,
        ))

    def record_memory_stored(self, memory_id: str, memory_type: str, n_links: int) -> None:
        """Record a memory being stored in the consolidated store."""
        self.record(JournalEntry.create(
            JournalEventType.MEMORY_STORED,
            memory_id=memory_id, memory_type=memory_type, n_links=n_links,
        ))

    def record_consolidation(self, report_summary: dict) -> None:
        """Record a consolidation pass."""
        self.record(JournalEntry.create(
            JournalEventType.CONSOLIDATION_RUN,
            **report_summary,
        ))

    # ── Queries ──

    def get_recent(self, n: int = 50) -> list[JournalEntry]:
        """Most recent N entries."""
        return self._entries[-n:]

    def get_by_event(self, event: str, last_n: int = 100) -> list[JournalEntry]:
        """Get entries of a specific event type."""
        matches = [e for e in self._entries if e.event == event]
        return matches[-last_n:]

    def get_geometry_history(self, last_n: int = 50) -> list[dict]:
        """Get recent geometry readings as dicts."""
        geo_entries = self.get_by_event(JournalEventType.GEOMETRY_EXTRACTED, last_n)
        return [e.data for e in geo_entries]

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    # ── Persistence ──

    def _persist_to_disk(self, entry: JournalEntry) -> None:
        """Append entry to JSONL log."""
        if not self._persist_dir:
            return
        try:
            path = self._persist_dir / "journal.jsonl"
            record = {
                "entry_id": entry.entry_id,
                "event": entry.event,
                "timestamp": entry.timestamp,
                "data": entry.data,
            }
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Journal persist failed: %s", e)
