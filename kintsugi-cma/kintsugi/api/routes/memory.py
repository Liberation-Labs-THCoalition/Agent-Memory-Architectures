"""Memory search and store endpoints — hybrid CMA Stage 3 pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kintsugi.config.settings import settings
from kintsugi.db import get_session
from kintsugi.memory.cma_stage3 import ScoredResult, estimate_complexity, retrieve
from kintsugi.memory.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_uuid(value: str, field: str = "org_id") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid UUID for {field}: {value!r}")


def _vec_literal(vec) -> str:
    """Convert a numpy array to a pgvector literal string like '[0.1,0.2,...]'."""
    return "[" + ",".join(str(float(v)) for v in vec) + "]"


def _rows_to_scored(rows, source: str) -> list[ScoredResult]:
    results: list[ScoredResult] = []
    for row in rows:
        results.append(ScoredResult(
            id=str(row.id),
            content=row.content,
            score=float(row.score),
            source=source,
            metadata={
                "significance": row.significance,
                "memory_layer": row.memory_layer,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            },
        ))
    return results


def _get_provider():
    kwargs: dict[str, Any] = {}
    if settings.EMBEDDING_MODE == "api":
        kwargs["api_key"] = settings.OPENAI_API_KEY
    return get_embedding_provider(mode=settings.EMBEDDING_MODE, **kwargs)


# ---------------------------------------------------------------------------
# GET /api/memory/search
# ---------------------------------------------------------------------------

@router.get("/search")
async def memory_search(
    q: str = Query(..., min_length=1, description="Search query"),
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
) -> dict:
    oid = _parse_uuid(org_id)

    # --- embed query ---
    try:
        provider = _get_provider()
        query_vec = await provider.embed(q)
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding failure: {exc}")

    vec_lit = _vec_literal(query_vec)

    # --- three parallel searches ---
    dense_sql = text("""
        SELECT mu.id, mu.content, mu.significance, mu.memory_layer, mu.created_at,
               1 - (me.embedding <=> :query_vec::vector) AS score
        FROM memory_units mu
        JOIN memory_embeddings me ON me.memory_id = mu.id
        WHERE mu.org_id = :org_id
        ORDER BY me.embedding <=> :query_vec::vector
        LIMIT :limit
    """)

    lexical_sql = text("""
        SELECT mu.id, mu.content, mu.significance, mu.memory_layer, mu.created_at,
               ts_rank(ml.tsv, plainto_tsquery('english', :query)) AS score
        FROM memory_units mu
        JOIN memory_lexical ml ON ml.memory_id = mu.id
        WHERE mu.org_id = :org_id AND ml.tsv @@ plainto_tsquery('english', :query)
        ORDER BY score DESC
        LIMIT :limit
    """)

    symbolic_sql = text("""
        SELECT mu.id, mu.content, mu.significance, mu.memory_layer, mu.created_at,
               (10 - mu.significance)::float / 10.0 AS score
        FROM memory_units mu
        WHERE mu.org_id = :org_id
        ORDER BY mu.significance ASC, mu.created_at DESC
        LIMIT :limit
    """)

    params_dense = {"query_vec": vec_lit, "org_id": str(oid), "limit": limit}
    params_lex = {"query": q, "org_id": str(oid), "limit": limit}
    params_sym = {"org_id": str(oid), "limit": limit}

    dense_task = session.execute(dense_sql, params_dense)
    lexical_task = session.execute(lexical_sql, params_lex)
    symbolic_task = session.execute(symbolic_sql, params_sym)

    dense_res, lexical_res, symbolic_res = await asyncio.gather(
        dense_task, lexical_task, symbolic_task
    )

    dense_hits = _rows_to_scored(dense_res.fetchall(), "dense")
    lexical_hits = _rows_to_scored(lexical_res.fetchall(), "lexical")
    symbolic_hits = _rows_to_scored(symbolic_res.fetchall(), "symbolic")

    # --- CMA Stage 3 fusion ---
    profile = estimate_complexity(q)
    fused = retrieve(
        query=q,
        dense_results=dense_hits,
        lexical_results=lexical_hits,
        symbolic_results=symbolic_hits,
        n_results=limit,
    )

    return {
        "results": [
            {
                "id": r.id,
                "content": r.content,
                "score": round(r.score, 6),
                "source": r.source,
                "metadata": r.metadata,
            }
            for r in fused
        ],
        "count": len(fused),
        "query": q,
        "org_id": str(oid),
        "query_profile": {
            "complexity": profile.complexity,
            "weights": {
                "dense": profile.dense_weight,
                "lexical": profile.lexical_weight,
                "symbolic": profile.symbolic_weight,
            },
        },
    }


# ---------------------------------------------------------------------------
# POST /api/memory/store
# ---------------------------------------------------------------------------

class MemoryStoreRequest(BaseModel):
    content: str
    org_id: str
    significance: int = 5
    entity_type: str = "general"


@router.post("/store")
async def memory_store(
    body: MemoryStoreRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    oid = _parse_uuid(body.org_id)
    memory_id = uuid.uuid4()

    # --- generate embedding ---
    try:
        provider = _get_provider()
        vec = await provider.embed(body.content)
    except Exception as exc:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding failure: {exc}")

    vec_lit = _vec_literal(vec)

    # --- insert memory unit ---
    await session.execute(
        text("""
            INSERT INTO memory_units (id, org_id, content, significance, memory_layer)
            VALUES (:id, :org_id, :content, :significance, :memory_layer)
        """),
        {
            "id": str(memory_id),
            "org_id": str(oid),
            "content": body.content,
            "significance": body.significance,
            "memory_layer": "core",
        },
    )

    # --- insert embedding ---
    await session.execute(
        text("""
            INSERT INTO memory_embeddings (memory_id, embedding, model)
            VALUES (:memory_id, :embedding::vector, :model)
        """),
        {
            "memory_id": str(memory_id),
            "embedding": vec_lit,
            "model": settings.EMBEDDING_MODE,
        },
    )

    # --- insert lexical ---
    await session.execute(
        text("""
            INSERT INTO memory_lexical (memory_id, tsv)
            VALUES (:memory_id, to_tsvector('english', :content))
        """),
        {"memory_id": str(memory_id), "content": body.content},
    )

    # --- insert metadata ---
    await session.execute(
        text("""
            INSERT INTO memory_metadata (memory_id, entity_type, significance, extra)
            VALUES (:memory_id, :entity_type, :significance, :extra::jsonb)
        """),
        {
            "memory_id": str(memory_id),
            "entity_type": body.entity_type,
            "significance": body.significance,
            "extra": "{}",
        },
    )

    await session.commit()

    return {
        "id": str(memory_id),
        "org_id": str(oid),
        "status": "created",
    }
