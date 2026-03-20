"""SQLite-vec embedding cache for fast local semantic search.

This is NOT the primary store — Notion is. This cache provides:
- Vector embeddings for semantic similarity search
- Fast local retrieval without Notion API calls
- Embedding persistence across sessions
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Local SQLite-vec cache for memory embeddings."""

    def __init__(
        self,
        db_path: str = "./data/memory_index.db",
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                notion_page_id TEXT,
                content_hash TEXT,
                memory_type TEXT,
                significance REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                tags TEXT DEFAULT '[]',
                entities TEXT DEFAULT '[]',
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                embedding BLOB,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memory_embeddings(memory_type)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON memory_embeddings(status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_significance
            ON memory_embeddings(significance)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_notion_page_id
            ON memory_embeddings(notion_page_id)
        """)
        self.conn.commit()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return self.model.encode(text).tolist()

    def index_memory(
        self,
        memory_id: str,
        content: str,
        notion_page_id: Optional[str] = None,
        content_hash: str = "",
        memory_type: str = "fact",
        significance: float = 0.5,
        status: str = "active",
        tags: Optional[list[str]] = None,
        entities: Optional[list[str]] = None,
    ) -> list[float]:
        """Embed and index a memory for local search."""
        embedding = self.embed(content)
        embedding_blob = json.dumps(embedding).encode()

        self.conn.execute("""
            INSERT OR REPLACE INTO memory_embeddings
            (memory_id, notion_page_id, content_hash, memory_type,
             significance, status, tags, entities, embedding,
             created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), 0)
        """, (
            memory_id,
            notion_page_id,
            content_hash,
            memory_type,
            significance,
            status,
            json.dumps(tags or []),
            json.dumps(entities or []),
            embedding_blob,
        ))
        self.conn.commit()
        return embedding

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_significance: Optional[float] = None,
        status: str = "active",
    ) -> list[dict]:
        """Semantic search using cosine similarity.

        Returns list of dicts with memory_id, notion_page_id, score, etc.
        """
        query_embedding = self.embed(query)

        # Build WHERE clause
        conditions = ["status = ?"]
        params: list = [status]

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)

        if min_significance is not None:
            conditions.append("significance >= ?")
            params.append(min_significance)

        where = " AND ".join(conditions)

        rows = self.conn.execute(f"""
            SELECT memory_id, notion_page_id, content_hash, memory_type,
                   significance, tags, entities, embedding, access_count
            FROM memory_embeddings
            WHERE {where}
        """, params).fetchall()

        # Compute cosine similarity
        results = []
        for row in rows:
            stored_embedding = json.loads(row[7])
            score = self._cosine_similarity(query_embedding, stored_embedding)
            results.append({
                "memory_id": row[0],
                "notion_page_id": row[1],
                "content_hash": row[2],
                "memory_type": row[3],
                "significance": row[4],
                "tags": json.loads(row[5]),
                "entities": json.loads(row[6]),
                "score": score,
                "access_count": row[8],
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Bump access metrics for returned results
        for r in results[:limit]:
            self.conn.execute("""
                UPDATE memory_embeddings
                SET last_accessed = datetime('now'),
                    access_count = access_count + 1
                WHERE memory_id = ?
            """, (r["memory_id"],))
        self.conn.commit()

        return results[:limit]

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the local index."""
        self.conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        )
        self.conn.commit()

    def update_status(self, memory_id: str, status: str) -> None:
        """Update the status of a cached memory."""
        self.conn.execute(
            "UPDATE memory_embeddings SET status = ?, updated_at = datetime('now') WHERE memory_id = ?",
            (status, memory_id),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Return cache statistics."""
        rows = self.conn.execute("""
            SELECT status, memory_type, COUNT(*), AVG(significance)
            FROM memory_embeddings
            GROUP BY status, memory_type
        """).fetchall()

        stats = {"total": 0, "by_status": {}, "by_type": {}}
        for status, mtype, count, avg_sig in rows:
            stats["total"] += count
            stats["by_status"][status] = stats["by_status"].get(status, 0) + count
            stats["by_type"][mtype] = {
                "count": count,
                "avg_significance": round(avg_sig, 3) if avg_sig else 0,
            }
        return stats

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
