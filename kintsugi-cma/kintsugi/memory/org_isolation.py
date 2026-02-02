"""
Organization-scoped memory isolation using PostgreSQL Row-Level Security (RLS).

Ensures complete tenant isolation at the database level. Every query is scoped
to an org_id via PostgreSQL's current_setting() mechanism, meaning even if
application logic has bugs, RLS policies prevent cross-org data leakage.

Architecture:
    1. Each connection sets `app.current_org_id` before any operation.
    2. The RLS policy on org_memories enforces org_id = current_setting('app.current_org_id').
    3. OrgMemoryStore wraps all operations so callers never touch raw SQL.

Usage:
    conn = get_org_connection(pool, "org_acme_123")
    store = OrgMemoryStore(conn, "org_acme_123")
    store.store("Project deadline moved to Q3", tags=["planning"], significance=7)
    results = store.hybrid_search("deadline", query_embedding=vec, n_results=5)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# psycopg2 typing — used for type hints only, no runtime dependency required
try:
    from psycopg2.extensions import connection as PgConnection, cursor as PgCursor
except ImportError:
    PgConnection = Any  # type: ignore[assignment,misc]
    PgCursor = Any  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ORG_MEMORIES_SCHEMA = """
-- Organization-scoped memory table with hybrid search support.
-- Requires: pgvector extension, pg_trgm extension.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS org_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          VARCHAR(100) NOT NULL,
    content         TEXT NOT NULL,
    embedding_768   vector(768),
    embedding_1536  vector(1536),
    significance    INTEGER CHECK (significance >= 1 AND significance <= 10),
    memory_layer    VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN significance >= 8 THEN 'core'
            WHEN significance >= 5 THEN 'active'
            WHEN significance >= 3 THEN 'background'
            ELSE 'ephemeral'
        END
    ) STORED,
    tags            TEXT[] DEFAULT '{}',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tsv             TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

-- Org partition index for fast tenant lookups
CREATE INDEX IF NOT EXISTS idx_org_memories_org_id
    ON org_memories (org_id);

-- HNSW indexes for approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_org_memories_embedding_768
    ON org_memories USING hnsw (embedding_768 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_org_memories_embedding_1536
    ON org_memories USING hnsw (embedding_1536 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_org_memories_tsv
    ON org_memories USING gin (tsv);

-- GIN index for tag filtering
CREATE INDEX IF NOT EXISTS idx_org_memories_tags
    ON org_memories USING gin (tags);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_org_memories_metadata
    ON org_memories USING gin (metadata jsonb_path_ops);

-- Significance + created_at for retention policy scans
CREATE INDEX IF NOT EXISTS idx_org_memories_significance
    ON org_memories (org_id, significance, created_at);

-- Row-Level Security
ALTER TABLE org_memories ENABLE ROW LEVEL SECURITY;

-- Force RLS even for table owners (defense in depth)
ALTER TABLE org_memories FORCE ROW LEVEL SECURITY;

CREATE POLICY org_isolation_select ON org_memories
    FOR SELECT
    USING (org_id = current_setting('app.current_org_id', true));

CREATE POLICY org_isolation_insert ON org_memories
    FOR INSERT
    WITH CHECK (org_id = current_setting('app.current_org_id', true));

CREATE POLICY org_isolation_update ON org_memories
    FOR UPDATE
    USING (org_id = current_setting('app.current_org_id', true))
    WITH CHECK (org_id = current_setting('app.current_org_id', true));

CREATE POLICY org_isolation_delete ON org_memories
    FOR DELETE
    USING (org_id = current_setting('app.current_org_id', true));
"""


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def sql_set_org_context(org_id: str) -> tuple[str, tuple[str]]:
    """Generate a parameterized SET for the org context.

    Returns:
        Tuple of (sql_template, params) safe for cursor.execute().
    """
    return "SET LOCAL app.current_org_id = %s", (org_id,)


def sql_insert_memory() -> str:
    """Return the INSERT statement for a new org memory."""
    return """
        INSERT INTO org_memories (id, org_id, content, embedding_768, significance, tags, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at, memory_layer
    """


def sql_hybrid_search(*, has_embedding: bool = True) -> str:
    """Return a hybrid search query combining vector similarity and full-text.

    The query uses Reciprocal Rank Fusion (RRF) to merge vector and keyword
    scores into a single ranking.

    Args:
        has_embedding: If True, include vector similarity in ranking.
                       If False, fall back to full-text only.
    """
    if has_embedding:
        return """
            WITH semantic AS (
                SELECT id, 1.0 / (60 + RANK() OVER (ORDER BY embedding_768 <=> %s)) AS rrf_score
                FROM org_memories
                WHERE embedding_768 IS NOT NULL
                ORDER BY embedding_768 <=> %s
                LIMIT %s
            ),
            keyword AS (
                SELECT id, 1.0 / (60 + RANK() OVER (ORDER BY ts_rank(tsv, query) DESC)) AS rrf_score
                FROM org_memories, plainto_tsquery('english', %s) query
                WHERE tsv @@ query
                LIMIT %s
            ),
            fused AS (
                SELECT COALESCE(s.id, k.id) AS id,
                       COALESCE(s.rrf_score, 0) + COALESCE(k.rrf_score, 0) AS combined_score
                FROM semantic s
                FULL OUTER JOIN keyword k ON s.id = k.id
            )
            SELECT m.id, m.content, m.significance, m.memory_layer, m.tags,
                   m.metadata, m.created_at, f.combined_score
            FROM fused f
            JOIN org_memories m ON m.id = f.id
            ORDER BY f.combined_score DESC
            LIMIT %s
        """
    return """
        SELECT m.id, m.content, m.significance, m.memory_layer, m.tags,
               m.metadata, m.created_at,
               ts_rank(m.tsv, query) AS score
        FROM org_memories m, plainto_tsquery('english', %s) query
        WHERE m.tsv @@ query
        ORDER BY score DESC
        LIMIT %s
    """


def sql_delete_memory() -> str:
    """Return DELETE statement (RLS ensures org scoping)."""
    return "DELETE FROM org_memories WHERE id = %s RETURNING id"


def sql_get_stats() -> str:
    """Return aggregate stats query scoped by RLS."""
    return """
        SELECT
            COUNT(*)                                    AS total_memories,
            COUNT(*) FILTER (WHERE memory_layer = 'core')       AS core_count,
            COUNT(*) FILTER (WHERE memory_layer = 'active')     AS active_count,
            COUNT(*) FILTER (WHERE memory_layer = 'background') AS background_count,
            COUNT(*) FILTER (WHERE memory_layer = 'ephemeral')  AS ephemeral_count,
            AVG(significance)::NUMERIC(4,2)             AS avg_significance,
            MIN(created_at)                             AS oldest_memory,
            MAX(created_at)                             AS newest_memory
        FROM org_memories
    """


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_org_connection(pool: Any, org_id: str) -> PgConnection:
    """Obtain a connection from the pool and set the org context.

    This should be used as the sole entry point for getting connections.
    The org context is set via SET LOCAL so it only lasts for the current
    transaction, providing automatic cleanup.

    Args:
        pool: A psycopg2 connection pool (e.g., ThreadedConnectionPool).
        org_id: The organization identifier to scope all queries to.

    Returns:
        A psycopg2 connection with app.current_org_id set.

    Raises:
        ValueError: If org_id is empty or contains disallowed characters.
    """
    if not org_id or not org_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            f"Invalid org_id: {org_id!r}. "
            "Must be non-empty and contain only alphanumerics, hyphens, underscores."
        )

    conn = pool.getconn()
    conn.autocommit = False
    with conn.cursor() as cur:
        sql, params = sql_set_org_context(org_id)
        cur.execute(sql, params)
    return conn


# ---------------------------------------------------------------------------
# OrgMemoryStore
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """A single memory record returned from the store."""
    id: str
    content: str
    significance: int
    memory_layer: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: Any
    score: Optional[float] = None


class OrgMemoryStore:
    """Organization-scoped memory store with hybrid search.

    All operations are scoped to the org_id set at construction time.
    The underlying connection must have been obtained via get_org_connection()
    or have app.current_org_id set manually.

    Thread safety: Instances are NOT thread-safe. Use one per thread/request.

    Example:
        conn = get_org_connection(pool, "org_acme_123")
        store = OrgMemoryStore(conn, "org_acme_123")
        record = store.store(
            content="Sprint retrospective: improve CI pipeline",
            tags=["engineering", "retro"],
            significance=6,
        )
        results = store.hybrid_search("CI pipeline improvements", embedding)
    """

    def __init__(self, conn: PgConnection, org_id: str) -> None:
        """Initialize the store.

        Args:
            conn: A psycopg2 connection with org context already set.
            org_id: The organization ID (must match the connection context).
        """
        self._conn = conn
        self._org_id = org_id

    def _set_org_context(self) -> None:
        """Re-set the org context on the connection.

        Call this if you suspect the context was lost (e.g., after a rollback).
        Uses SET LOCAL so the setting is transaction-scoped.
        """
        with self._conn.cursor() as cur:
            sql, params = sql_set_org_context(self._org_id)
            cur.execute(sql, params)

    def store(
        self,
        content: str,
        tags: Optional[Sequence[str]] = None,
        significance: int = 5,
        embedding_768: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Store a new memory scoped to this organization.

        Args:
            content: The text content of the memory.
            tags: Optional list of string tags for categorization.
            significance: Importance score from 1 (ephemeral) to 10 (core).
            embedding_768: Optional 768-dimensional embedding vector.
            metadata: Optional JSON-serializable metadata dict.

        Returns:
            A MemoryRecord with the id, created_at, and memory_layer populated.

        Raises:
            ValueError: If significance is outside [1, 10].
        """
        if not 1 <= significance <= 10:
            raise ValueError(f"significance must be 1-10, got {significance}")

        memory_id = str(uuid.uuid4())
        resolved_tags = list(tags) if tags else []
        resolved_metadata = metadata or {}

        with self._conn.cursor() as cur:
            self._set_org_context()
            cur.execute(
                sql_insert_memory(),
                (
                    memory_id,
                    self._org_id,
                    content,
                    embedding_768,
                    significance,
                    resolved_tags,
                    _json_adapter(resolved_metadata),
                ),
            )
            row = cur.fetchone()

        self._conn.commit()

        return MemoryRecord(
            id=str(row[0]),
            content=content,
            significance=significance,
            memory_layer=row[2],
            tags=resolved_tags,
            metadata=resolved_metadata,
            created_at=row[1],
        )

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[list[float]] = None,
        n_results: int = 10,
    ) -> list[MemoryRecord]:
        """Search memories using hybrid vector + full-text ranking.

        Uses Reciprocal Rank Fusion (RRF) when an embedding is provided,
        falling back to pure full-text search otherwise. RLS ensures only
        this org's memories are returned.

        Args:
            query_text: Natural language search query.
            query_embedding: Optional 768-dim vector for semantic search.
            n_results: Maximum number of results to return.

        Returns:
            List of MemoryRecord sorted by relevance (best first).
        """
        has_embedding = query_embedding is not None
        sql = sql_hybrid_search(has_embedding=has_embedding)

        with self._conn.cursor() as cur:
            self._set_org_context()

            if has_embedding:
                # RRF hybrid: semantic limit + keyword limit + final limit
                semantic_limit = n_results * 3
                keyword_limit = n_results * 3
                cur.execute(sql, (
                    query_embedding,    # semantic ORDER BY
                    query_embedding,    # semantic WHERE (same param used twice)
                    semantic_limit,
                    query_text,         # keyword query
                    keyword_limit,
                    n_results,          # final LIMIT
                ))
            else:
                cur.execute(sql, (query_text, n_results))

            rows = cur.fetchall()

        return [
            MemoryRecord(
                id=str(row[0]),
                content=row[1],
                significance=row[2],
                memory_layer=row[3],
                tags=row[4] or [],
                metadata=row[5] or {},
                created_at=row[6],
                score=float(row[7]) if row[7] is not None else None,
            )
            for row in rows
        ]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID (org-scoped via RLS).

        Args:
            memory_id: UUID of the memory to delete.

        Returns:
            True if a row was deleted, False if not found (or wrong org).
        """
        with self._conn.cursor() as cur:
            self._set_org_context()
            cur.execute(sql_delete_memory(), (memory_id,))
            deleted = cur.fetchone() is not None

        self._conn.commit()
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics for this organization's memories.

        Returns:
            Dict with keys: total_memories, core_count, active_count,
            background_count, ephemeral_count, avg_significance,
            oldest_memory, newest_memory.
        """
        with self._conn.cursor() as cur:
            self._set_org_context()
            cur.execute(sql_get_stats())
            row = cur.fetchone()

        if row is None:
            return {
                "total_memories": 0,
                "core_count": 0,
                "active_count": 0,
                "background_count": 0,
                "ephemeral_count": 0,
                "avg_significance": None,
                "oldest_memory": None,
                "newest_memory": None,
            }

        return {
            "total_memories": row[0],
            "core_count": row[1],
            "active_count": row[2],
            "background_count": row[3],
            "ephemeral_count": row[4],
            "avg_significance": float(row[5]) if row[5] is not None else None,
            "oldest_memory": row[6],
            "newest_memory": row[7],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_adapter(obj: dict[str, Any]) -> Any:
    """Wrap a dict for psycopg2 JSONB insertion.

    Returns a psycopg2 Json adapter if available, otherwise the raw dict
    (which works with recent psycopg2 versions with register_default_jsonb).
    """
    try:
        from psycopg2.extras import Json
        return Json(obj)
    except ImportError:
        return obj
