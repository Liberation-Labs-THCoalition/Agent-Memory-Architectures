"""Tests for kintsugi.memory.temporal module."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from kintsugi.memory.temporal import Category, TemporalEvent, TemporalLog


# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------

class TestCategory:
    def test_all_values(self):
        expected = {"KINTSUGI", "SECURITY", "DECISION", "SKILL_CHIP",
                    "MODIFICATION", "MEMORY", "GOVERNANCE"}
        assert {c.value for c in Category} == expected

    def test_count(self):
        assert len(Category) == 7

    def test_is_str_enum(self):
        assert isinstance(Category.KINTSUGI, str)
        assert Category.SECURITY == "SECURITY"


# ---------------------------------------------------------------------------
# TemporalEvent dataclass
# ---------------------------------------------------------------------------

class TestTemporalEvent:
    def test_create(self):
        now = datetime.now()
        evt = TemporalEvent(id="abc", category="DECISION", message="hi",
                            metadata={"k": "v"}, created_at=now)
        assert evt.id == "abc"
        assert evt.category == "DECISION"
        assert evt.message == "hi"
        assert evt.metadata == {"k": "v"}
        assert evt.created_at == now


# ---------------------------------------------------------------------------
# TemporalLog
# ---------------------------------------------------------------------------

def _mock_session_for_add():
    """Create an AsyncMock session where add() is a plain MagicMock (not async)."""
    session = AsyncMock()
    session.add = MagicMock()  # add is sync
    return session


class TestTemporalLogEvent:
    @pytest.mark.asyncio
    async def test_log_event_returns_id(self):
        session = _mock_session_for_add()
        mock_row = MagicMock()
        mock_row.id = "uuid-123"

        mock_model = MagicMock()
        mock_base = MagicMock()
        mock_base.TemporalMemory = mock_model
        mock_model.return_value = mock_row

        with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
            log = TemporalLog()
            result = await log.log_event(
                org_id="org1",
                category="DECISION",
                message="test message",
                metadata={"key": "val"},
                session=session,
            )

        assert result == "uuid-123"
        session.add.assert_called_once_with(mock_row)
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_log_event_passes_fields_to_model(self):
        session = _mock_session_for_add()

        mock_model = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "x"
        mock_model.return_value = mock_row
        mock_base = MagicMock()
        mock_base.TemporalMemory = mock_model

        with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
            log = TemporalLog()
            await log.log_event("org_a", "MEMORY", "msg", {"a": 1}, session)

            mock_model.assert_called_once_with(
                org_id="org_a",
                category="MEMORY",
                message="msg",
                metadata_json={"a": 1},
            )


class TestTemporalLogQuery:
    @pytest.mark.asyncio
    async def test_query_events_no_session_raises(self):
        log = TemporalLog()
        with pytest.raises(ValueError, match="session is required"):
            await log.query_events(org_id="org1", session=None)

    @pytest.mark.asyncio
    async def test_query_events_basic(self):
        now = datetime.now()
        mock_row = MagicMock()
        mock_row.id = "id1"
        mock_row.category = "SECURITY"
        mock_row.message = "alert"
        mock_row.metadata_json = {"level": "high"}
        mock_row.created_at = now

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]
        session.execute.return_value = mock_result

        # Need to patch select() so it doesn't choke on MagicMock model
        with patch("kintsugi.memory.temporal.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            # Chain: .where().where()...order_by().limit()
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt
            mock_stmt.limit.return_value = mock_stmt

            mock_base = MagicMock()
            with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
                log = TemporalLog()
                events = await log.query_events(org_id="org1", session=session)

        assert len(events) == 1
        assert events[0].id == "id1"
        assert events[0].category == "SECURITY"
        assert events[0].metadata == {"level": "high"}

    @pytest.mark.asyncio
    async def test_query_events_empty(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.temporal.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt
            mock_stmt.limit.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                log = TemporalLog()
                events = await log.query_events(org_id="org1", session=session)

        assert events == []

    @pytest.mark.asyncio
    async def test_query_events_null_metadata(self):
        mock_row = MagicMock()
        mock_row.id = "id2"
        mock_row.category = "DECISION"
        mock_row.message = "m"
        mock_row.metadata_json = None
        mock_row.created_at = datetime.now()

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.temporal.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt
            mock_stmt.limit.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                log = TemporalLog()
                events = await log.query_events(org_id="org1", session=session)

        assert events[0].metadata == {}

    @pytest.mark.asyncio
    async def test_query_events_with_all_filters(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.temporal.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt
            mock_stmt.limit.return_value = mock_stmt

            mock_base = MagicMock()
            # Make model attributes support >= and <= comparisons with datetime
            mock_tm = mock_base.TemporalMemory
            mock_tm.created_at.__ge__ = MagicMock(return_value=MagicMock())
            mock_tm.created_at.__le__ = MagicMock(return_value=MagicMock())
            mock_tm.message.ilike = MagicMock(return_value=MagicMock())

            with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
                log = TemporalLog()
                events = await log.query_events(
                    org_id="org1",
                    category="KINTSUGI",
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 12, 31),
                    keyword="test",
                    limit=10,
                    session=session,
                )

        assert events == []
        session.execute.assert_awaited_once()
        # where() should be called multiple times for all filters
        assert mock_stmt.where.call_count >= 2

    @pytest.mark.asyncio
    async def test_query_events_multiple_rows(self):
        now = datetime.now()
        rows = []
        for i in range(3):
            r = MagicMock()
            r.id = f"id{i}"
            r.category = "MEMORY"
            r.message = f"msg{i}"
            r.metadata_json = {}
            r.created_at = now
            rows.append(r)

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.temporal.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt
            mock_stmt.limit.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                log = TemporalLog()
                events = await log.query_events(org_id="org1", session=session)

        assert len(events) == 3
        assert [e.id for e in events] == ["id0", "id1", "id2"]
