"""Tests for kintsugi.memory.cold_archive module."""

from __future__ import annotations

import gzip
import hashlib
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from kintsugi.memory.cold_archive import (
    _compress,
    _decompress,
    _sha256,
    _window_text,
    ArchivedWindow,
    ColdArchive,
    IntegrityReport,
)
from kintsugi.memory.cma_stage1 import Turn, Window


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestCompress:
    def test_round_trip(self):
        text = "Hello, world! This is a test."
        assert _decompress(_compress(text)) == text

    def test_round_trip_unicode(self):
        text = "Unicode test"
        assert _decompress(_compress(text)) == text

    def test_compress_returns_bytes(self):
        assert isinstance(_compress("test"), bytes)

    def test_decompress_returns_str(self):
        assert isinstance(_decompress(_compress("x")), str)

    def test_empty_string(self):
        assert _decompress(_compress("")) == ""

    def test_large_string(self):
        text = "x" * 10000
        assert _decompress(_compress(text)) == text


class TestSha256:
    def test_known_hash(self):
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert _sha256(data) == expected

    def test_deterministic(self):
        data = b"test data"
        assert _sha256(data) == _sha256(data)

    def test_different_inputs(self):
        assert _sha256(b"a") != _sha256(b"b")

    def test_length(self):
        assert len(_sha256(b"x")) == 64


class TestWindowText:
    def test_formats_turns(self):
        now = datetime.now()
        turns = [
            Turn(role="user", content="hi", timestamp=now),
            Turn(role="assistant", content="hello", timestamp=now),
        ]
        window = Window(turns=turns, start_idx=0, end_idx=1)
        result = _window_text(window)
        assert result == "user: hi\nassistant: hello"

    def test_single_turn(self):
        now = datetime.now()
        window = Window(turns=[Turn(role="user", content="x", timestamp=now)],
                        start_idx=0, end_idx=0)
        assert _window_text(window) == "user: x"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_archived_window(self):
        now = datetime.now()
        aw = ArchivedWindow(id="1", content="c", entropy_score=0.5,
                            content_hash="abc", archived_at=now)
        assert aw.entropy_score == 0.5
        assert aw.id == "1"

    def test_integrity_report(self):
        r = IntegrityReport(total_checked=3, passed=2, failed=1, failed_ids=["x"])
        assert r.failed == 1
        assert r.total_checked == 3


# ---------------------------------------------------------------------------
# ColdArchive async methods — need to patch select() to avoid SQLAlchemy errors
# ---------------------------------------------------------------------------

def _mock_session_for_add():
    session = AsyncMock()
    session.add = MagicMock()
    return session


class TestArchiveWindow:
    @pytest.mark.asyncio
    async def test_archive_window_returns_id(self):
        now = datetime.now()
        turns = [Turn(role="user", content="test", timestamp=now)]
        window = Window(turns=turns, start_idx=0, end_idx=0)

        session = _mock_session_for_add()
        mock_row = MagicMock()
        mock_row.id = "archive-uuid"

        mock_model = MagicMock()
        mock_model.return_value = mock_row
        mock_base = MagicMock()
        mock_base.MemoryArchive = mock_model

        with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
            ca = ColdArchive()
            result = await ca.archive_window("org1", window, 0.3, session)

        assert result == "archive-uuid"
        session.add.assert_called_once()
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_archive_window_stores_compressed_content(self):
        now = datetime.now()
        turns = [Turn(role="user", content="data", timestamp=now)]
        window = Window(turns=turns, start_idx=0, end_idx=0)

        session = _mock_session_for_add()
        mock_row = MagicMock()
        mock_row.id = "id"

        mock_model = MagicMock()
        mock_model.return_value = mock_row
        mock_base = MagicMock()
        mock_base.MemoryArchive = mock_model

        with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
            ca = ColdArchive()
            await ca.archive_window("org1", window, 0.5, session)

            call_kwargs = mock_model.call_args[1]
            assert isinstance(call_kwargs["content_compressed"], bytes)
            assert call_kwargs["org_id"] == "org1"
            assert call_kwargs["entropy_score"] == 0.5
            assert call_kwargs["content_hash"] == _sha256(call_kwargs["content_compressed"])
            # Verify decompressed content matches
            assert _decompress(call_kwargs["content_compressed"]) == "user: data"


class TestRetrieveArchive:
    @pytest.mark.asyncio
    async def test_retrieve_decompresses(self):
        text = "user: hello"
        compressed = _compress(text)
        now = datetime.now()

        mock_row = MagicMock()
        mock_row.id = "r1"
        mock_row.content_compressed = compressed
        mock_row.entropy_score = 0.4
        mock_row.content_hash = _sha256(compressed)
        mock_row.archived_at = now

        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt

            mock_base = MagicMock()
            mock_ma = mock_base.MemoryArchive
            mock_ma.archived_at.__ge__ = MagicMock(return_value=MagicMock())
            mock_ma.archived_at.__le__ = MagicMock(return_value=MagicMock())

            with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                results = await ca.retrieve_archive(
                    "org1", (datetime(2024, 1, 1), datetime(2024, 12, 31)), session
                )

        assert len(results) == 1
        assert results[0].content == text
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_retrieve_empty(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt
            mock_stmt.order_by.return_value = mock_stmt

            mock_base = MagicMock()
            mock_ma = mock_base.MemoryArchive
            mock_ma.archived_at.__ge__ = MagicMock(return_value=MagicMock())
            mock_ma.archived_at.__le__ = MagicMock(return_value=MagicMock())

            with patch.dict("sys.modules", {"kintsugi.models.base": mock_base, "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                results = await ca.retrieve_archive(
                    "org1", (datetime(2024, 1, 1), datetime(2024, 12, 31)), session
                )
        assert results == []


class TestVerifyIntegrity:
    def _setup_session(self, rows):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = rows
        session.execute.return_value = mock_result
        return session

    @pytest.mark.asyncio
    async def test_all_pass(self):
        compressed = _compress("data")
        row = MagicMock()
        row.id = "id1"
        row.content_compressed = compressed
        row.content_hash = _sha256(compressed)

        session = self._setup_session([row])

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                report = await ca.verify_integrity("org1", session)

        assert report.total_checked == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.failed_ids == []

    @pytest.mark.asyncio
    async def test_integrity_failure(self):
        compressed = _compress("data")
        row = MagicMock()
        row.id = "bad-id"
        row.content_compressed = compressed
        row.content_hash = "wrong-hash"

        session = self._setup_session([row])

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                report = await ca.verify_integrity("org1", session)

        assert report.failed == 1
        assert "bad-id" in report.failed_ids

    @pytest.mark.asyncio
    async def test_empty_archive(self):
        session = self._setup_session([])

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                report = await ca.verify_integrity("org1", session)

        assert report.total_checked == 0

    @pytest.mark.asyncio
    async def test_mixed_results(self):
        good_data = _compress("good")
        bad_data = _compress("bad")

        good_row = MagicMock()
        good_row.id = "g1"
        good_row.content_compressed = good_data
        good_row.content_hash = _sha256(good_data)

        bad_row = MagicMock()
        bad_row.id = "b1"
        bad_row.content_compressed = bad_data
        bad_row.content_hash = "corrupted"

        session = self._setup_session([good_row, bad_row])

        with patch("kintsugi.memory.cold_archive.select") as mock_select:
            mock_stmt = MagicMock()
            mock_select.return_value = mock_stmt
            mock_stmt.where.return_value = mock_stmt

            with patch.dict("sys.modules", {"kintsugi.models.base": MagicMock(), "kintsugi.models": MagicMock()}):
                ca = ColdArchive()
                report = await ca.verify_integrity("org1", session)

        assert report.total_checked == 2
        assert report.passed == 1
        assert report.failed == 1
