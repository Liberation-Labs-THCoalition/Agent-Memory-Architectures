"""Tests for kintsugi.memory.spaced module."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kintsugi.memory.spaced import (
    FIBONACCI,
    DueMemory,
    SpacedRetrieval,
    fib_interval,
)


# ---------------------------------------------------------------------------
# FIBONACCI table and fib_interval
# ---------------------------------------------------------------------------


class TestFibInterval:
    def test_table_values(self):
        assert FIBONACCI == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

    @pytest.mark.parametrize("idx,expected", list(enumerate(FIBONACCI)))
    def test_all_indices(self, idx, expected):
        assert fib_interval(idx) == expected

    def test_beyond_table(self):
        assert fib_interval(100) == 233
        assert fib_interval(len(FIBONACCI)) == 233

    def test_zero(self):
        assert fib_interval(0) == 1


# ---------------------------------------------------------------------------
# DueMemory dataclass
# ---------------------------------------------------------------------------


class TestDueMemory:
    def test_fields(self):
        d = DueMemory(id="x", content="hi", significance=5, access_count=3, days_overdue=7)
        assert d.id == "x"
        assert d.days_overdue == 7


# ---------------------------------------------------------------------------
# SpacedRetrieval — patch select/update to avoid SQLAlchemy validation
# ---------------------------------------------------------------------------

def _make_spaced_session(rows):
    """Helper: create mock session returning given rows from execute."""
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


def _spaced_patches():
    """Context manager patches for spaced retrieval tests."""
    mock_mu = MagicMock()
    # Make expires_at support comparison with datetime (used in query building)
    mock_mu.expires_at.__gt__ = MagicMock(return_value=MagicMock())
    mock_mu.expires_at.is_ = MagicMock(return_value=MagicMock())
    mock_mu.expires_at.__or__ = MagicMock(return_value=MagicMock())
    # select() returns a chainable mock
    mock_select_chain = MagicMock()
    mock_select = MagicMock(return_value=mock_select_chain)
    mock_update = MagicMock(return_value=MagicMock(return_value=MagicMock()))
    return (
        patch.dict(sys.modules, {
            "kintsugi.models.base": MagicMock(MemoryUnit=mock_mu),
            "kintsugi.models": MagicMock(),
        }),
        patch("kintsugi.memory.spaced.select", mock_select),
        patch("kintsugi.memory.spaced.update", mock_update),
    )


class TestSpacedRetrieval:
    @pytest.mark.asyncio
    async def test_get_due_memories_no_session(self):
        sr = SpacedRetrieval()
        with pytest.raises(ValueError, match="session is required"):
            await sr.get_due_memories("org-1", session=None)

    @pytest.mark.asyncio
    async def test_get_due_memories_returns_overdue(self):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.id = "m1"
        row.content = "test memory"
        row.significance = 5
        row.updated_at = now - timedelta(days=10)
        row.created_at = now - timedelta(days=20)
        # Ensure hasattr check for _access_count returns False
        del row._access_count

        mock_session = _make_spaced_session([row])
        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            result = await sr.get_due_memories("org-1", session=mock_session)

        assert len(result) == 1
        assert result[0].id == "m1"
        assert result[0].days_overdue >= 8

    @pytest.mark.asyncio
    async def test_get_due_memories_not_yet_due(self):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.id = "m1"
        row.content = "test"
        row.significance = 5
        row.updated_at = now
        row.created_at = now - timedelta(days=5)
        del row._access_count

        mock_session = _make_spaced_session([row])
        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            result = await sr.get_due_memories("org-1", session=mock_session)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_due_memories_uses_created_at_fallback(self):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.id = "m2"
        row.content = "test"
        row.significance = 3
        row.updated_at = None
        row.created_at = now - timedelta(days=10)
        del row._access_count

        mock_session = _make_spaced_session([row])
        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            result = await sr.get_due_memories("org-1", session=mock_session)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_due_memories_respects_max_count(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(10):
            r = MagicMock()
            r.id = f"m{i}"
            r.content = f"mem {i}"
            r.significance = 5
            r.updated_at = now - timedelta(days=10 + i)
            r.created_at = now - timedelta(days=20)
            del r._access_count
            rows.append(r)

        mock_session = _make_spaced_session(rows)
        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            result = await sr.get_due_memories("org-1", max_count=3, session=mock_session)

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_get_due_memories_sorted_most_overdue_first(self):
        now = datetime.now(timezone.utc)
        row_recent = MagicMock()
        row_recent.id = "recent"
        row_recent.content = "r"
        row_recent.significance = 5
        row_recent.updated_at = now - timedelta(days=5)
        row_recent.created_at = now - timedelta(days=20)
        del row_recent._access_count

        row_old = MagicMock()
        row_old.id = "old"
        row_old.content = "o"
        row_old.significance = 5
        row_old.updated_at = now - timedelta(days=50)
        row_old.created_at = now - timedelta(days=60)
        del row_old._access_count

        mock_session = _make_spaced_session([row_recent, row_old])
        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            result = await sr.get_due_memories("org-1", session=mock_session)

        assert result[0].id == "old"

    @pytest.mark.asyncio
    async def test_record_access(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        p1, p2, p3 = _spaced_patches()
        with p1, p2, p3:
            sr = SpacedRetrieval()
            await sr.record_access("mem-123", mock_session)

        mock_session.execute.assert_called_once()
