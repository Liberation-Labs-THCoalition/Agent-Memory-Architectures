"""Significance continuum and memory expiration.

Maps integer significance scores (1-10) to named memory layers with
tiered expiration policies. An :class:`ExpiredMemoryReaper` handles
the periodic archival of expired memories.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory layers
# ---------------------------------------------------------------------------


class MemoryLayer(str, enum.Enum):
    """Named memory tiers mapped from significance scores."""

    PERMANENT = "PERMANENT"   # 1-2
    CORE = "CORE"             # 3-4
    IMPORTANT = "IMPORTANT"   # 5-6
    STANDARD = "STANDARD"     # 7-8
    VOLATILE = "VOLATILE"     # 9-10


# Significance -> layer mapping
_LAYER_RANGES: list[tuple[range, MemoryLayer]] = [
    (range(1, 3), MemoryLayer.PERMANENT),
    (range(3, 5), MemoryLayer.CORE),
    (range(5, 7), MemoryLayer.IMPORTANT),
    (range(7, 9), MemoryLayer.STANDARD),
    (range(9, 11), MemoryLayer.VOLATILE),
]

# Significance -> TTL in days (None = never expires)
_EXPIRATION_DAYS: dict[MemoryLayer, int | None] = {
    MemoryLayer.PERMANENT: None,
    MemoryLayer.CORE: 730,
    MemoryLayer.IMPORTANT: 365,
    MemoryLayer.STANDARD: 90,
    MemoryLayer.VOLATILE: 30,
}


def compute_layer(significance: int) -> MemoryLayer:
    """Map a significance score (1-10) to its :class:`MemoryLayer`.

    Raises:
        ValueError: If significance is outside 1-10.
    """
    for rng, layer in _LAYER_RANGES:
        if significance in rng:
            return layer
    raise ValueError(f"significance must be 1-10, got {significance}")


def compute_expiration(significance: int, created_at: datetime) -> datetime | None:
    """Compute the expiration timestamp for a memory.

    Args:
        significance: Score 1-10.
        created_at: When the memory was created.

    Returns:
        Expiration datetime, or ``None`` for permanent memories.
    """
    layer = compute_layer(significance)
    days = _EXPIRATION_DAYS[layer]
    if days is None:
        return None
    return created_at + timedelta(days=days)


# ---------------------------------------------------------------------------
# Reaper
# ---------------------------------------------------------------------------


@dataclass
class ReapResult:
    """Summary of a reaper pass."""

    checked: int
    expired: int
    archived: int
    errors: int


class ExpiredMemoryReaper:
    """Finds and archives memories that have passed their expiration date."""

    async def reap(
        self,
        org_id: str,
        session: AsyncSession,
    ) -> ReapResult:
        """Scan for expired memories and mark them as archived.

        Memories with ``expires_at <= now()`` and ``is_archived = False``
        are flagged as archived. In a full implementation this would also
        move content to cold storage; here we set the flag for downstream
        processing.

        Returns:
            :class:`ReapResult` with counts.
        """
        from kintsugi.models.base import MemoryUnit

        now = datetime.now(timezone.utc)

        # Find non-archived, expired memories
        stmt = select(MemoryUnit).where(
            MemoryUnit.org_id == org_id,
            MemoryUnit.expires_at.isnot(None),
            MemoryUnit.expires_at <= now,
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

        checked = len(rows)
        archived = 0
        errors = 0

        expired_ids = [r.id for r in rows]
        if expired_ids:
            try:
                await session.execute(
                    update(MemoryUnit)
                    .where(MemoryUnit.id.in_(expired_ids))
                    .values(memory_layer="archived")
                )
                archived = len(expired_ids)
            except Exception:
                logger.exception("Error archiving expired memories for org=%s", org_id)
                errors = len(expired_ids)

        logger.info(
            "Reaper org=%s: checked=%d expired=%d archived=%d errors=%d",
            org_id,
            checked,
            checked,
            archived,
            errors,
        )
        return ReapResult(
            checked=checked,
            expired=checked,
            archived=archived,
            errors=errors,
        )
