"""Append-only temporal decision log.

Every significant system event is recorded immutably in the
``temporal_memories`` table, providing a complete audit trail for
governance, debugging, and memory archaeology.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


class Category(str, enum.Enum):
    """Standard event categories for the temporal log."""

    KINTSUGI = "KINTSUGI"
    SECURITY = "SECURITY"
    DECISION = "DECISION"
    SKILL_CHIP = "SKILL_CHIP"
    MODIFICATION = "MODIFICATION"
    MEMORY = "MEMORY"
    GOVERNANCE = "GOVERNANCE"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TemporalEvent:
    """A single event retrieved from the temporal log."""

    id: str
    category: str
    message: str
    metadata: dict
    created_at: datetime


# ---------------------------------------------------------------------------
# TemporalLog
# ---------------------------------------------------------------------------


class TemporalLog:
    """Manages the append-only temporal event log."""

    async def log_event(
        self,
        org_id: str,
        category: str,
        message: str,
        metadata: dict,
        session: AsyncSession,
    ) -> str:
        """Record an event in the temporal log.

        Args:
            org_id: Organisation scope.
            category: One of :class:`Category` values.
            message: Human-readable event description.
            metadata: Arbitrary structured metadata.
            session: Active async database session.

        Returns:
            String UUID of the created record.
        """
        from kintsugi.models.base import TemporalMemory

        row = TemporalMemory(
            org_id=org_id,
            category=category,
            message=message,
            metadata_json=metadata,
        )
        session.add(row)
        await session.flush()

        logger.info(
            "Temporal event [%s] org=%s: %s",
            category,
            org_id,
            message[:120],
        )
        return str(row.id)

    async def query_events(
        self,
        org_id: str,
        category: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        keyword: str | None = None,
        limit: int = 50,
        session: AsyncSession | None = None,
    ) -> list[TemporalEvent]:
        """Query the temporal log with optional filters.

        Args:
            org_id: Organisation scope.
            category: Filter by event category.
            start: Earliest ``created_at`` (inclusive).
            end: Latest ``created_at`` (inclusive).
            keyword: Substring match on ``message``.
            limit: Maximum rows to return.
            session: Active async database session.

        Returns:
            List of matching :class:`TemporalEvent` objects.
        """
        if session is None:
            raise ValueError("session is required")

        from kintsugi.models.base import TemporalMemory

        stmt = select(TemporalMemory).where(TemporalMemory.org_id == org_id)

        if category is not None:
            stmt = stmt.where(TemporalMemory.category == category)
        if start is not None:
            stmt = stmt.where(TemporalMemory.created_at >= start)
        if end is not None:
            stmt = stmt.where(TemporalMemory.created_at <= end)
        if keyword is not None:
            stmt = stmt.where(TemporalMemory.message.ilike(f"%{keyword}%"))

        stmt = stmt.order_by(TemporalMemory.created_at.desc()).limit(limit)

        result = await session.execute(stmt)
        rows = result.scalars().all()

        return [
            TemporalEvent(
                id=str(r.id),
                category=r.category,
                message=r.message,
                metadata=r.metadata_json or {},
                created_at=r.created_at,
            )
            for r in rows
        ]
