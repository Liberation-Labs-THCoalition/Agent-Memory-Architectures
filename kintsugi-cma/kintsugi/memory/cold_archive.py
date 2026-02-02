"""CMA Cold Archive — sub-threshold compressed storage.

Windows that fall below the entropy threshold are compressed with gzip,
integrity-hashed with SHA-256, and stored in the ``memory_archives`` table
for potential future retrieval.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kintsugi.memory.cma_stage1 import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ArchivedWindow:
    """A decompressed archived window returned to callers."""

    id: str
    content: str
    entropy_score: float
    content_hash: str
    archived_at: datetime


@dataclass
class IntegrityReport:
    """Result of an archive integrity verification pass."""

    total_checked: int
    passed: int
    failed: int
    failed_ids: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window_text(window: Window) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in window.turns)


def _compress(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"), compresslevel=6)


def _decompress(data: bytes) -> str:
    return gzip.decompress(data).decode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# ColdArchive
# ---------------------------------------------------------------------------


class ColdArchive:
    """Manages the compressed cold-storage tier for sub-threshold windows."""

    async def archive_window(
        self,
        org_id: str,
        window: Window,
        entropy_score: float,
        session: AsyncSession,
    ) -> str:
        """Compress and store a window in the archive.

        Args:
            org_id: Organisation scope identifier.
            window: The :class:`Window` to archive.
            entropy_score: Pre-computed entropy for the window.
            session: Active async database session.

        Returns:
            String UUID of the newly created archive row.
        """
        from kintsugi.models.base import MemoryArchive

        text = _window_text(window)
        compressed = _compress(text)
        content_hash = _sha256(compressed)

        row = MemoryArchive(
            org_id=org_id,
            content_compressed=compressed,
            entropy_score=entropy_score,
            content_hash=content_hash,
        )
        session.add(row)
        await session.flush()

        logger.info(
            "Archived window [%d-%d] for org=%s  hash=%s",
            window.start_idx,
            window.end_idx,
            org_id,
            content_hash[:12],
        )
        return str(row.id)

    async def retrieve_archive(
        self,
        org_id: str,
        date_range: tuple[datetime, datetime],
        session: AsyncSession,
    ) -> list[ArchivedWindow]:
        """Retrieve archived windows within a date range.

        Args:
            org_id: Organisation scope.
            date_range: ``(start, end)`` inclusive datetime bounds.
            session: Active async database session.

        Returns:
            List of :class:`ArchivedWindow` with decompressed content.
        """
        from kintsugi.models.base import MemoryArchive

        start, end = date_range
        stmt = (
            select(MemoryArchive)
            .where(
                MemoryArchive.org_id == org_id,
                MemoryArchive.archived_at >= start,
                MemoryArchive.archived_at <= end,
            )
            .order_by(MemoryArchive.archived_at)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

        return [
            ArchivedWindow(
                id=str(r.id),
                content=_decompress(r.content_compressed),
                entropy_score=r.entropy_score,
                content_hash=r.content_hash,
                archived_at=r.archived_at,
            )
            for r in rows
        ]

    async def verify_integrity(
        self,
        org_id: str,
        session: AsyncSession,
    ) -> IntegrityReport:
        """Recompute SHA-256 hashes and compare against stored values.

        Returns:
            :class:`IntegrityReport` summarising the check.
        """
        from kintsugi.models.base import MemoryArchive

        stmt = select(MemoryArchive).where(MemoryArchive.org_id == org_id)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        passed = 0
        failed = 0
        failed_ids: list[str] = []

        for row in rows:
            computed = _sha256(row.content_compressed)
            if computed == row.content_hash:
                passed += 1
            else:
                failed += 1
                failed_ids.append(str(row.id))
                logger.error(
                    "Integrity failure for archive %s: expected=%s got=%s",
                    row.id,
                    row.content_hash,
                    computed,
                )

        total = passed + failed
        logger.info(
            "Integrity check for org=%s: %d checked, %d passed, %d failed",
            org_id,
            total,
            passed,
            failed,
        )
        return IntegrityReport(
            total_checked=total,
            passed=passed,
            failed=failed,
            failed_ids=failed_ids,
        )
