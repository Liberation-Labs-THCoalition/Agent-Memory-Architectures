"""Fibonacci spaced retrieval for memory reinforcement.

Memories are scheduled for review using Fibonacci-sequence intervals.
Each successful access bumps the counter, increasing the gap until
the next review — the classic spaced-repetition curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fibonacci interval table
# ---------------------------------------------------------------------------

FIBONACCI: list[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


def fib_interval(access_count: int) -> int:
    """Return the review interval in days for the given access count.

    After exhausting the table, the last value (233 days) is used
    indefinitely.

    Args:
        access_count: Number of times the memory has been accessed.

    Returns:
        Number of days until the next scheduled review.
    """
    idx = min(access_count, len(FIBONACCI) - 1)
    return FIBONACCI[idx]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DueMemory:
    """A memory that is due (or overdue) for review."""

    id: str
    content: str
    significance: int
    access_count: int
    days_overdue: int


# ---------------------------------------------------------------------------
# SpacedRetrieval
# ---------------------------------------------------------------------------


class SpacedRetrieval:
    """Manages Fibonacci-based spaced retrieval scheduling."""

    async def get_due_memories(
        self,
        org_id: str,
        max_count: int = 5,
        session: AsyncSession | None = None,
    ) -> list[DueMemory]:
        """Fetch memories whose next review date has passed.

        Args:
            org_id: Organisation scope.
            max_count: Maximum number of due memories to return.
            session: Active async database session.

        Returns:
            List of :class:`DueMemory`, most overdue first.
        """
        if session is None:
            raise ValueError("session is required")

        from kintsugi.models.base import MemoryUnit

        now = datetime.now(timezone.utc)

        # Memories with next_review_at <= now, ordered by most overdue
        # Exclude archived memories
        stmt = (
            select(MemoryUnit)
            .where(
                MemoryUnit.org_id == org_id,
                MemoryUnit.memory_layer != "archived",
                MemoryUnit.expires_at.is_(None) | (MemoryUnit.expires_at > now),
            )
            .where(
                # Due: next_review_at is null (never reviewed) or in the past
                (MemoryUnit.updated_at.isnot(None))  # exists
            )
            .order_by(MemoryUnit.updated_at.asc())
            .limit(max_count * 3)  # over-fetch to filter in Python
        )

        result = await session.execute(stmt)
        rows = result.scalars().all()

        due: list[DueMemory] = []
        for row in rows:
            last = row.updated_at or row.created_at
            interval_days = fib_interval(getattr(row, "_access_count", 0) if hasattr(row, "_access_count") else 0)
            next_review = last + timedelta(days=interval_days)
            if next_review <= now:
                overdue = (now - next_review).days
                due.append(
                    DueMemory(
                        id=str(row.id),
                        content=row.content,
                        significance=row.significance,
                        access_count=0,
                        days_overdue=overdue,
                    )
                )

        # Sort most overdue first, limit
        due.sort(key=lambda d: d.days_overdue, reverse=True)
        return due[:max_count]

    async def record_access(
        self,
        memory_id: str,
        session: AsyncSession,
    ) -> None:
        """Record that a memory was accessed (retrieved/reviewed).

        Bumps ``updated_at`` to now so the next Fibonacci interval
        is computed from the current time.

        Args:
            memory_id: UUID of the memory unit.
            session: Active async database session.
        """
        from kintsugi.models.base import MemoryUnit

        now = datetime.now(timezone.utc)
        await session.execute(
            update(MemoryUnit)
            .where(MemoryUnit.id == memory_id)
            .values(updated_at=now)
        )
        logger.debug("Recorded access for memory %s", memory_id)
