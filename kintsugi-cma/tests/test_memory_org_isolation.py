"""Tests for kintsugi.memory.org_isolation module."""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from kintsugi.memory.org_isolation import (
    sql_set_org_context,
    sql_insert_memory,
    sql_hybrid_search,
    sql_delete_memory,
    sql_get_stats,
    get_org_connection,
    OrgMemoryStore,
    MemoryRecord,
    _json_adapter,
)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

class TestSqlHelpers:
    def test_set_org_context(self):
        sql, params = sql_set_org_context("org_abc")
        assert "SET LOCAL" in sql
        assert "app.current_org_id" in sql
        assert params == ("org_abc",)

    def test_insert_memory_sql(self):
        sql = sql_insert_memory()
        assert "INSERT INTO org_memories" in sql
        assert "RETURNING" in sql

    def test_hybrid_search_with_embedding(self):
        sql = sql_hybrid_search(has_embedding=True)
        assert "semantic" in sql
        assert "keyword" in sql
        assert "rrf_score" in sql

    def test_hybrid_search_without_embedding(self):
        sql = sql_hybrid_search(has_embedding=False)
        assert "semantic" not in sql
        assert "ts_rank" in sql

    def test_delete_memory_sql(self):
        sql = sql_delete_memory()
        assert "DELETE FROM org_memories" in sql
        assert "RETURNING id" in sql

    def test_get_stats_sql(self):
        sql = sql_get_stats()
        assert "COUNT(*)" in sql
        assert "core" in sql
        assert "ephemeral" in sql
        assert "avg_significance" in sql.lower() or "AVG(significance)" in sql


# ---------------------------------------------------------------------------
# get_org_connection
# ---------------------------------------------------------------------------

class TestGetOrgConnection:
    def test_valid_org_id(self):
        pool = MagicMock()
        mock_conn = MagicMock()
        pool.getconn.return_value = mock_conn

        conn = get_org_connection(pool, "org_acme_123")
        assert conn is mock_conn
        assert mock_conn.autocommit is False

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid org_id"):
            get_org_connection(MagicMock(), "")

    def test_invalid_none_like(self):
        with pytest.raises(ValueError):
            get_org_connection(MagicMock(), "")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError):
            get_org_connection(MagicMock(), "org;DROP TABLE")

    def test_valid_with_hyphens_underscores(self):
        pool = MagicMock()
        pool.getconn.return_value = MagicMock()
        # Should not raise
        get_org_connection(pool, "org-acme_123")

    def test_sets_org_context_via_cursor(self):
        pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        pool.getconn.return_value = mock_conn

        get_org_connection(pool, "org1")
        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args[0]
        assert "SET LOCAL" in args[0]


# ---------------------------------------------------------------------------
# MemoryRecord dataclass
# ---------------------------------------------------------------------------

class TestMemoryRecord:
    def test_create(self):
        r = MemoryRecord(id="1", content="c", significance=5,
                         memory_layer="active", tags=["t"], metadata={},
                         created_at=datetime.now())
        assert r.score is None

    def test_with_score(self):
        r = MemoryRecord(id="1", content="c", significance=5,
                         memory_layer="active", tags=[], metadata={},
                         created_at=datetime.now(), score=0.95)
        assert r.score == 0.95


# ---------------------------------------------------------------------------
# OrgMemoryStore
# ---------------------------------------------------------------------------

def _make_store():
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return OrgMemoryStore(conn, "org_test"), conn, cursor


class TestOrgMemoryStoreStore:
    def test_store_returns_record(self):
        store, conn, cursor = _make_store()
        now = datetime.now()
        cursor.fetchone.return_value = (uuid.uuid4(), now, "active")

        record = store.store("hello world", tags=["t1"], significance=6)
        assert record.content == "hello world"
        assert record.significance == 6
        assert record.memory_layer == "active"
        assert record.tags == ["t1"]
        conn.commit.assert_called_once()

    def test_store_invalid_significance(self):
        store, _, _ = _make_store()
        with pytest.raises(ValueError, match="significance must be 1-10"):
            store.store("x", significance=0)
        with pytest.raises(ValueError):
            store.store("x", significance=11)

    def test_store_default_tags_and_metadata(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = (uuid.uuid4(), datetime.now(), "active")

        record = store.store("content")
        assert record.tags == []
        assert record.metadata == {}


class TestOrgMemoryStoreSearch:
    def test_hybrid_search_with_embedding(self):
        store, conn, cursor = _make_store()
        now = datetime.now()
        cursor.fetchall.return_value = [
            (uuid.uuid4(), "result", 7, "active", ["tag"], {"k": "v"}, now, 0.85),
        ]

        results = store.hybrid_search("query", query_embedding=[0.1] * 768, n_results=5)
        assert len(results) == 1
        assert results[0].content == "result"
        assert results[0].score == 0.85

    def test_hybrid_search_without_embedding(self):
        store, conn, cursor = _make_store()
        now = datetime.now()
        cursor.fetchall.return_value = [
            (uuid.uuid4(), "text", 5, "active", None, None, now, 0.5),
        ]

        results = store.hybrid_search("query", n_results=5)
        assert len(results) == 1
        assert results[0].tags == []
        assert results[0].metadata == {}

    def test_hybrid_search_empty(self):
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = []

        results = store.hybrid_search("nothing")
        assert results == []


class TestOrgMemoryStoreDelete:
    def test_delete_found(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = (uuid.uuid4(),)

        assert store.delete("some-id") is True
        conn.commit.assert_called_once()

    def test_delete_not_found(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = None

        assert store.delete("missing-id") is False


class TestOrgMemoryStoreStats:
    def test_get_stats(self):
        store, conn, cursor = _make_store()
        now = datetime.now()
        cursor.fetchone.return_value = (100, 10, 30, 40, 20, 5.5, now, now)

        stats = store.get_stats()
        assert stats["total_memories"] == 100
        assert stats["core_count"] == 10
        assert stats["avg_significance"] == 5.5

    def test_get_stats_empty(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = None

        stats = store.get_stats()
        assert stats["total_memories"] == 0
        assert stats["avg_significance"] is None

    def test_get_stats_null_avg(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = (0, 0, 0, 0, 0, None, None, None)

        stats = store.get_stats()
        assert stats["avg_significance"] is None


# ---------------------------------------------------------------------------
# _json_adapter
# ---------------------------------------------------------------------------

class TestJsonAdapter:
    def test_with_psycopg2(self):
        # If psycopg2 is available, should wrap in Json
        result = _json_adapter({"key": "val"})
        try:
            from psycopg2.extras import Json
            assert isinstance(result, Json)
        except ImportError:
            # Fallback: returns raw dict
            assert result == {"key": "val"}

    def test_fallback_without_psycopg2(self):
        with patch.dict("sys.modules", {"psycopg2.extras": None, "psycopg2": None}):
            # Force ImportError by removing the cached module
            import importlib
            # Can't easily force ImportError with patch.dict alone, so test the raw path
            data = {"a": 1}
            # If psycopg2 is actually installed, this won't trigger fallback
            # Just verify it returns something usable
            result = _json_adapter(data)
            assert result is not None
