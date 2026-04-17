"""Tier 2: Transaction Memory — append-only event journal.

Every inference decision is logged: geometry readings, alignment checks,
steering actions, commits, aborts. The journal is the model's episodic
memory — what it did and why.

Ported from Oracle Harness Layer 4 (TransactionJournal).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from oracle_memory.src.types import (
    AlignmentCheck,
    GeometrySummary,
    JournalEntry,
    JournalEventType,
)

logger = logging.getLogger(__name__)


class TransactionJournal:
    """Append-only event log for inference transactions.

    Supports in-memory operation (fast) and optional JSONL persistence
    (durable). The journal is the audit trail — every decision, every
    rollback, every steering action is recorded.
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._entries: list[JournalEntry] = []
        self._tx_ids: set[str] = set()
        self._persist_dir: Optional[Path] = None

        if persist_dir:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ── Recording methods ──

    def record(self, entry: JournalEntry) -> None:
        """Record a raw journal entry."""
        self._entries.append(entry)
        self._tx_ids.add(entry.tx_id)
        if self._persist_dir:
            self._persist_to_disk(entry)

    def record_begin(self, tx_id: str, input_text: str) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.TX_BEGIN,
            input_text=input_text[:500],
        ))

    def record_checkpoint(self, tx_id: str, checkpoint_id: str, label: str) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.CHECKPOINT_CREATED,
            checkpoint_id=checkpoint_id, label=label,
        ))

    def record_geometry(self, tx_id: str, geometry: GeometrySummary) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.GEOMETRY_EXTRACTED,
            key_norm=geometry.key_norm,
            effective_rank=geometry.effective_rank,
            spectral_entropy=geometry.spectral_entropy,
            top_sv_ratio=geometry.top_sv_ratio,
            n_tokens=geometry.n_tokens,
        ))

    def record_alignment(self, tx_id: str, check: AlignmentCheck) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.ALIGNMENT_CHECK,
            verdict=check.verdict.value,
            confidence=check.confidence,
            reasons=check.reasons,
            dominant_emotion=check.dominant_emotion,
            alignment_risk=check.alignment_risk,
        ))

    def record_steering(self, tx_id: str, vectors: dict, reason: str) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.STEERING_APPLIED,
            vectors=vectors, reason=reason,
        ))

    def record_rollback(self, tx_id: str, to_checkpoint: str, retry_count: int) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.ROLLBACK,
            to_checkpoint=to_checkpoint, retry_count=retry_count,
        ))

    def record_commit(self, tx_id: str, output_preview: str) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.COMMIT,
            output_preview=output_preview[:200],
        ))

    def record_abort(self, tx_id: str, reason: str, retry_count: int) -> None:
        self.record(JournalEntry.create(
            tx_id, JournalEventType.ABORT,
            reason=reason, retry_count=retry_count,
        ))

    # ── Query methods ──

    def get_transaction_events(self, tx_id: str) -> list[JournalEntry]:
        """All events for a transaction, in order."""
        return [e for e in self._entries if e.tx_id == tx_id]

    def get_recent(self, n: int = 50) -> list[JournalEntry]:
        """Most recent N entries across all transactions."""
        return self._entries[-n:]

    def count_rollbacks(self, tx_id: str) -> int:
        """How many rollbacks occurred in a transaction."""
        return sum(
            1 for e in self._entries
            if e.tx_id == tx_id and e.event == JournalEventType.ROLLBACK
        )

    def get_alignment_history(self, last_n_transactions: int = 20) -> list[dict]:
        """Alignment summary for recent transactions.

        Used by the Conscience layer for drift detection.
        Returns list of {tx_id, verdict, confidence, retry_count}.
        """
        # Find unique recent tx_ids (ordered by first appearance)
        seen = []
        for entry in reversed(self._entries):
            if entry.tx_id not in seen:
                seen.append(entry.tx_id)
            if len(seen) >= last_n_transactions:
                break
        seen.reverse()

        results = []
        for tx_id in seen:
            events = self.get_transaction_events(tx_id)
            alignment_events = [
                e for e in events if e.event == JournalEventType.ALIGNMENT_CHECK
            ]
            terminal = [
                e for e in events
                if e.event in (JournalEventType.COMMIT, JournalEventType.ABORT)
            ]

            if alignment_events:
                last_check = alignment_events[-1]
                results.append({
                    "tx_id": tx_id,
                    "verdict": last_check.data.get("verdict", "unknown"),
                    "confidence": last_check.data.get("confidence", 0),
                    "retry_count": self.count_rollbacks(tx_id),
                    "outcome": terminal[-1].event.value if terminal else "unknown",
                })

        return results

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    @property
    def total_transactions(self) -> int:
        return len(self._tx_ids)

    # ── Persistence ──

    def _persist_to_disk(self, entry: JournalEntry) -> None:
        """Append entry to per-transaction JSONL file."""
        if not self._persist_dir:
            return
        try:
            path = self._persist_dir / f"tx_{entry.tx_id}.jsonl"
            record = {
                "tx_id": entry.tx_id,
                "event": entry.event.value,
                "timestamp": entry.timestamp,
                "data": entry.data,
            }
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("Journal persist failed: %s", e)
