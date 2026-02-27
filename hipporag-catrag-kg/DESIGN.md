# HippoRAG-Inspired Knowledge Graph Extension for Agent Memory Systems

**Status:** Draft
**Author:** TH Coalition / Liberation Labs
**Date:** 2026-02-26
**Target:** Agent-Memory-Architectures monorepo, extending kintsugi-cma

---

## 1. Problem Statement

Modern agent memory systems built on embedding-based retrieval (dense vector search, BM25 full-text) are effective at **direct recall** -- queries like "what happened on Tuesday?" or "what did user X say about budget?" match well because the answer shares lexical and semantic overlap with the query.

They fail at **associative recall** -- queries like "what connects project Alpha to the budget shortfall?" or "who else was involved in conversations about both hiring and the Q3 report?" These require reasoning over *relationships between* memories, which are implicit in a flat embedding store. The retriever has no structural representation of how entities, events, and concepts relate to each other.

### The Gap in Our Current System

The kintsugi-cma architecture provides:

- **Stage 1:** Atomic fact extraction from conversations
- **Stage 2:** 3-phase consolidation (temporal decay, deduplication, affinity clustering)
- **Stage 3:** Hybrid retrieval with adaptive fusion (dense + lexical + symbolic scoring)
- **10 MCP tools** for agent interaction
- **Per-org isolation** via row-level policies
- **768-dim embeddings** from `all-mpnet-base-v2` via sentence-transformers

What it lacks: any explicit representation of *how entities mentioned across memories relate to each other*. Two memories might mention "Alice" and "Project Aurora" in different contexts, but the system has no structure connecting them. The only way to discover this is to hope that embedding similarity surfaces both memories for the same query, which is unreliable.

### What We Want

A lightweight knowledge graph layer that:

1. Extracts named entities from memories as they are stored
2. Builds explicit (subject, predicate, object) triples linking those entities
3. Uses graph traversal (Personalized PageRank) to answer associative queries
4. Merges graph-based results with existing hybrid search via Reciprocal Rank Fusion
5. Runs on modest hardware without external graph databases

---

## 2. Research Foundations

### 2.1 HippoRAG 2 (OSU-NLP-Group, ICML 2025)

**Paper:** arXiv 2502.14802

HippoRAG 2 models long-term memory after the hippocampal indexing theory from neuroscience. The core idea: the hippocampus does not store memories directly -- it maintains an *index* of associations that points to distributed cortical representations.

In HippoRAG 2, this translates to:

- **Neocortex analog:** A passage store (in our case, the existing `memory_units` table)
- **Parahippocampal cortex analog:** An entity extraction layer (NER) that identifies salient concepts
- **Hippocampal index analog:** An open knowledge graph of (subject, predicate, object) triples linking extracted entities
- **Retrieval via Personalized PageRank (PPR):** Given a query, extract query entities, find matching nodes in the KG, then run PPR from those seed nodes. The PPR scores propagate through the graph, surfacing memories that are *structurally connected* to the query even if they do not share direct semantic overlap.

Key results from the paper:
- PPR-based retrieval significantly outperforms dense-only retrieval on multi-hop questions
- The KG enables the system to answer questions that require connecting information across multiple documents
- Open KG construction (triples extracted from text) is more flexible than predefined schemas

### 2.2 CatRAG (arXiv 2602.01965, February 2026)

CatRAG extends HippoRAG 2 by addressing a critical failure mode: **semantic drift toward hub nodes.**

The problem: In a knowledge graph, some entities are mentioned far more frequently than others (e.g., a project name, an organization, a common location). Standard PPR treats all edges equally, so high-degree hub nodes attract disproportionate PageRank mass. A query about "Alice's role in Project Aurora" might drift toward "Project Aurora" (a hub with 50 connections) and surface memories about Aurora that have nothing to do with Alice.

CatRAG's solution: **query-adaptive edge weighting.** Before running PPR, CatRAG reweights edges based on their relevance to the *current query*:

1. Compute the embedding of the query
2. For each edge in the neighborhood of the seed nodes, compute the cosine similarity between the query embedding and the edge's predicate/context embedding
3. Scale edge weights by this similarity -- edges that are semantically relevant to the query get higher weight, irrelevant edges get suppressed
4. Run PPR on the reweighted graph

This prevents the "hub node drift" problem: even if Project Aurora has 50 edges, only the edges relevant to the current query (e.g., those involving Alice or roles) receive high weight.

### 2.3 Why These Papers Matter for Agent Memory

Agent memory differs from standard RAG in important ways:

- **Incremental construction:** Memories arrive one at a time over weeks/months, not as a static corpus
- **Entity re-occurrence:** The same people, projects, and concepts appear repeatedly across conversations
- **Relational queries are natural:** "What do we know about how Alice and Bob interact?" is a question agents actually receive
- **Scale is modest:** Hundreds to low thousands of memories, not millions of documents

This makes the KG approach particularly well-suited: the graph is small enough for exact PPR (no approximation needed), entities recur enough to form meaningful structure, and the queries agents receive genuinely benefit from associative retrieval.

---

## 3. Architecture Overview

```
                     +------------------+
                     |   Agent Query    |
                     +--------+---------+
                              |
                    +---------v----------+
                    |  Query Classifier  |
                    |  (existing Stage3) |
                    +---------+----------+
                              |
              +---------------+---------------+
              |               |               |
     +--------v------+ +-----v-------+ +-----v--------+
     | Dense Search   | | Lexical     | | Graph Search |
     | (pgvector)     | | (tsvector)  | | (NEW)        |
     +--------+------+ +-----+-------+ +-----+--------+
              |               |               |
              +---------------+---------------+
                              |
                    +---------v----------+
                    |   Reciprocal Rank  |
                    |   Fusion (RRF)     |
                    +---------+----------+
                              |
                    +---------v----------+
                    |   Ranked Results   |
                    +--------------------+


    Graph Search internals:
    +---------------------------------------------------------+
    |  1. Extract entities from query (spaCy NER)             |
    |  2. Find seed nodes in kg_entities                      |
    |  3. Compute query-aware edge weights (CatRAG)           |
    |  4. Run Personalized PageRank from seeds                |
    |  5. Map high-PPR entities -> source memories            |
    |  6. Return scored memory list                           |
    +---------------------------------------------------------+


    Storage pipeline (on memory ingest):
    +------------------+     +------------------+     +------------------+
    | Stage 1:         | --> | Stage 2:         | --> | KG Extraction    |
    | Fact Extraction  |     | Consolidation    |     | (NEW)            |
    +------------------+     +------------------+     +------------------+
                                                             |
                                                      +------v------+
                                                      | kg_entities |
                                                      | kg_triples  |
                                                      | kg_entity_  |
                                                      |  mentions   |
                                                      +-------------+
```

### Design Principles

1. **Additive, not invasive.** The KG layer adds new tables and a new retrieval path. It does not modify existing tables or break existing retrieval. If the KG is empty or disabled, the system behaves exactly as before.

2. **PostgreSQL-native.** No Neo4j, no separate graph database. The KG lives in the same PostgreSQL instance as everything else. At our scale (hundreds to low thousands of entities/triples), SQL-based graph traversal is fast enough and eliminates operational complexity.

3. **CPU-friendly NER.** Entity extraction uses spaCy with small/medium English models that run on CPU. No GPU required for extraction. The existing sentence-transformers pipeline (which does use GPU when available) handles embeddings.

4. **Graceful degradation.** If spaCy extraction misses entities, the graph is incomplete but the system still works -- dense and lexical search remain available. The KG is a *supplementary* retrieval signal, not a replacement.

---

## 4. Schema Design

### 4.1 New Tables

```sql
-- Migration: 002_knowledge_graph.py
-- Depends on: 001_initial

-- Enable pgvector if not already enabled (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- kg_entities: Canonical entity nodes in the knowledge graph
-- ============================================================
CREATE TABLE kg_entities (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    name_lower    TEXT NOT NULL GENERATED ALWAYS AS (lower(name)) STORED,
    entity_type   VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
        -- Expected types: PERSON, ORG, GPE, EVENT, PRODUCT, CONCEPT, DATE, UNKNOWN
        -- Aligned with spaCy entity labels
    embedding     vector(768),
        -- Embedding of the entity name, for fuzzy matching during seed node lookup.
        -- Computed via the same sentence-transformers model used for memories.
    first_seen    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata      JSONB DEFAULT '{}',

    -- Per-org uniqueness: same entity name within an org resolves to one node
    CONSTRAINT uq_entity_org_name UNIQUE (org_id, name_lower)
);

-- Fast lookup by name within an org
CREATE INDEX ix_kg_entities_org_name ON kg_entities (org_id, name_lower);

-- HNSW index for embedding-based fuzzy entity matching
CREATE INDEX ix_kg_entities_embedding_hnsw ON kg_entities
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- For finding high-degree hub nodes (useful for CatRAG diagnostics)
CREATE INDEX ix_kg_entities_mention_count ON kg_entities (org_id, mention_count DESC);


-- ============================================================
-- kg_triples: Directed edges in the knowledge graph
-- ============================================================
CREATE TABLE kg_triples (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    subject_entity_id UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    predicate         TEXT NOT NULL,
        -- Free-form predicate text, e.g. "works_on", "discussed", "mentioned_with"
        -- Not restricted to a fixed schema -- this is an open KG.
    object_entity_id  UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    source_memory_id  UUID NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
        -- The memory from which this triple was extracted.
        -- Enables provenance tracking: "why does this edge exist?"
    confidence        REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
        -- Extraction confidence. 1.0 for co-occurrence-based edges,
        -- variable for LLM-extracted edges (future upgrade path).
    predicate_embedding vector(768),
        -- Embedding of the predicate text, used for CatRAG query-aware weighting.
        -- Can be NULL if edge weighting is disabled.
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Prevent exact duplicate triples from the same source memory
    CONSTRAINT uq_triple_source UNIQUE (org_id, subject_entity_id, predicate, object_entity_id, source_memory_id)
);

-- Adjacency list traversal: "give me all edges from entity X"
CREATE INDEX ix_kg_triples_subject ON kg_triples (org_id, subject_entity_id);
CREATE INDEX ix_kg_triples_object ON kg_triples (org_id, object_entity_id);

-- Provenance lookup: "what triples came from memory Y?"
CREATE INDEX ix_kg_triples_source ON kg_triples (source_memory_id);


-- ============================================================
-- kg_entity_mentions: Links entities to the memories where they appear
-- ============================================================
CREATE TABLE kg_entity_mentions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id       UUID NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    memory_id       UUID NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
    mention_context TEXT,
        -- A short snippet (sentence or clause) surrounding the mention.
        -- Useful for disambiguation and provenance display.
    char_offset     INTEGER,
        -- Character offset within the memory content where the mention starts.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_entity_mention UNIQUE (entity_id, memory_id)
);

-- "Which memories mention entity X?" -- critical for PPR -> memory mapping
CREATE INDEX ix_kg_mentions_entity ON kg_entity_mentions (entity_id);

-- "Which entities appear in memory Y?" -- used during KG construction
CREATE INDEX ix_kg_mentions_memory ON kg_entity_mentions (memory_id);
```

### 4.2 Alembic Migration

```python
"""Knowledge graph tables for HippoRAG-style associative retrieval.

Revision ID: 002
Revises: 001
Create Date: 2026-02-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "kg_entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("name_lower", sa.Text, sa.Computed("lower(name)", persisted=True), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=False, server_default="UNKNOWN"),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("first_seen", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_seen", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("mention_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("metadata", JSONB, server_default="{}"),
        sa.UniqueConstraint("org_id", "name_lower", name="uq_entity_org_name"),
    )
    op.create_index("ix_kg_entities_org_name", "kg_entities", ["org_id", "name_lower"])
    op.execute(
        "CREATE INDEX ix_kg_entities_embedding_hnsw ON kg_entities "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.create_index("ix_kg_entities_mention_count", "kg_entities", ["org_id", sa.text("mention_count DESC")])

    op.create_table(
        "kg_triples",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("org_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_entity_id", UUID(as_uuid=True), sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("predicate", sa.Text, nullable=False),
        sa.Column("object_entity_id", UUID(as_uuid=True), sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_memory_id", UUID(as_uuid=True), sa.ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("predicate_embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "org_id", "subject_entity_id", "predicate", "object_entity_id", "source_memory_id",
            name="uq_triple_source",
        ),
        sa.CheckConstraint("confidence BETWEEN 0.0 AND 1.0", name="ck_triple_confidence"),
    )
    op.create_index("ix_kg_triples_subject", "kg_triples", ["org_id", "subject_entity_id"])
    op.create_index("ix_kg_triples_object", "kg_triples", ["org_id", "object_entity_id"])
    op.create_index("ix_kg_triples_source", "kg_triples", ["source_memory_id"])

    op.create_table(
        "kg_entity_mentions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("entity_id", UUID(as_uuid=True), sa.ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False),
        sa.Column("memory_id", UUID(as_uuid=True), sa.ForeignKey("memory_units.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mention_context", sa.Text, nullable=True),
        sa.Column("char_offset", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("entity_id", "memory_id", name="uq_entity_mention"),
    )
    op.create_index("ix_kg_mentions_entity", "kg_entity_mentions", ["entity_id"])
    op.create_index("ix_kg_mentions_memory", "kg_entity_mentions", ["memory_id"])


def downgrade() -> None:
    op.drop_table("kg_entity_mentions")
    op.drop_table("kg_triples")
    op.drop_table("kg_entities")
```

### 4.3 Entity-Relationship Diagram

```
organizations
    |
    | 1:N
    v
kg_entities --------+---------- kg_entity_mentions ---------> memory_units
    |                |                                              ^
    |  subject       |  object                                     |
    v                v                                             |
kg_triples ------- (subject_entity_id, object_entity_id) ---------+
                        |                                (source_memory_id)
                        |
                  predicate (open text)
```

---

## 5. Entity Extraction Pipeline

### 5.1 spaCy NER Approach

We use spaCy for named entity recognition because it runs locally on CPU with zero API cost. The tradeoff is lower extraction quality compared to LLM-based extraction -- spaCy will miss implicit entities, abstract concepts, and context-dependent references. This is acceptable for a v1 that can be upgraded later.

**Model selection:**

| Model | Size | Accuracy (OntoNotes F1) | RAM | Load Time |
|-------|------|------------------------|-----|-----------|
| `en_core_web_sm` | 12 MB | ~85% | ~50 MB | <1s |
| `en_core_web_md` | 40 MB | ~86% | ~100 MB | ~1s |
| `en_core_web_trf` | 400 MB | ~90% | ~1 GB | ~5s |

**Recommendation:** Start with `en_core_web_md`. The medium model includes word vectors (useful for entity disambiguation) and fits comfortably in 8 GB RAM alongside the rest of the stack. The transformer model is better but its memory footprint is problematic on constrained hardware.

### 5.2 Extraction Code

```python
"""kg_extractor.py -- Entity and triple extraction from memory content."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import spacy

logger = logging.getLogger(__name__)

# Lazy-load spaCy model (same pattern as sentence-transformers in embeddings.py)
_nlp: spacy.Language | None = None

# Entity types we care about. Others (CARDINAL, ORDINAL, etc.) are too noisy.
RELEVANT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC", "NORP", "LOC", "LAW"}


def _load_spacy(model_name: str = "en_core_web_md") -> spacy.Language:
    """Load spaCy model, downloading if necessary."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        _nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
        # We only need NER and tokenization. Disabling parser and lemmatizer
        # saves ~30% processing time.
    except OSError:
        logger.warning("spaCy model %s not found, downloading...", model_name)
        from spacy.cli import download
        download(model_name)
        _nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    return _nlp


@dataclass(frozen=True)
class ExtractedEntity:
    """An entity extracted from text."""
    name: str
    entity_type: str  # spaCy label: PERSON, ORG, GPE, etc.
    char_start: int
    char_end: int
    context: str  # Surrounding sentence or clause


@dataclass(frozen=True)
class ExtractedTriple:
    """A (subject, predicate, object) triple extracted from text."""
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass
class ExtractionResult:
    """Complete extraction result for a single memory."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    triples: list[ExtractedTriple] = field(default_factory=list)


def extract_entities(text: str, model_name: str = "en_core_web_md") -> list[ExtractedEntity]:
    """Extract named entities from text using spaCy NER.

    Args:
        text: The memory content to extract entities from.
        model_name: spaCy model to use.

    Returns:
        List of extracted entities with types and context.
    """
    nlp = _load_spacy(model_name)
    doc = nlp(text)

    entities = []
    seen_names: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITY_TYPES:
            continue

        # Normalize: strip whitespace, skip very short entities
        name = ent.text.strip()
        if len(name) < 2:
            continue

        # Deduplicate within same extraction (case-insensitive)
        name_key = name.lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        # Extract surrounding context (the sentence containing the entity)
        sent = ent.sent
        context = sent.text.strip() if sent else text[max(0, ent.start_char - 50):ent.end_char + 50]

        entities.append(ExtractedEntity(
            name=name,
            entity_type=ent.label_,
            char_start=ent.start_char,
            char_end=ent.end_char,
            context=context,
        ))

    return entities


def extract_triples_cooccurrence(
    entities: list[ExtractedEntity],
    window_chars: int = 200,
) -> list[ExtractedTriple]:
    """Infer relationships from entity co-occurrence within a text window.

    This is the simplest relationship extraction strategy: if two entities
    appear within `window_chars` characters of each other, we create a
    "co_occurs_with" triple. The confidence is inversely proportional to
    the distance between them.

    Args:
        entities: Entities extracted from the same memory.
        window_chars: Maximum character distance for co-occurrence.

    Returns:
        List of co-occurrence triples with distance-based confidence.
    """
    triples = []
    for e1, e2 in combinations(entities, 2):
        distance = abs(e1.char_start - e2.char_start)
        if distance <= window_chars:
            # Confidence: 1.0 when adjacent, decaying linearly to 0.5 at max distance
            confidence = 1.0 - 0.5 * (distance / window_chars)
            triples.append(ExtractedTriple(
                subject=e1.name,
                predicate="co_occurs_with",
                object=e2.name,
                confidence=round(confidence, 3),
            ))
    return triples


def extract_triples_embedding(
    entities: list[ExtractedEntity],
    embedder,  # EmbeddingProvider from kintsugi.memory.embeddings
    similarity_threshold: float = 0.65,
) -> list[ExtractedTriple]:
    """Infer relationships from embedding similarity of entity contexts.

    For each pair of entities, compute cosine similarity between their
    context embeddings. If similarity exceeds the threshold, create a
    "semantically_related" triple.

    This captures relationships that co-occurrence misses -- entities that
    are contextually related even if they appear far apart in the text.

    Note: This is an async function in production (embedding is async).
    Shown synchronously here for clarity.
    """
    # Implementation sketch -- actual code would be async
    # contexts = [e.context for e in entities]
    # embeddings = await embedder.embed_batch(contexts)
    # for (i, e1), (j, e2) in combinations(enumerate(entities), 2):
    #     sim = cosine_similarity(embeddings[i], embeddings[j])
    #     if sim >= similarity_threshold:
    #         triples.append(ExtractedTriple(
    #             subject=e1.name,
    #             predicate="semantically_related",
    #             object=e2.name,
    #             confidence=round(float(sim), 3),
    #         ))
    raise NotImplementedError("See async version in kg_extractor_async.py")


def run_extraction(text: str, model_name: str = "en_core_web_md") -> ExtractionResult:
    """Full extraction pipeline: entities + co-occurrence triples.

    Args:
        text: Memory content.
        model_name: spaCy model name.

    Returns:
        ExtractionResult with entities and triples.
    """
    entities = extract_entities(text, model_name)
    triples = extract_triples_cooccurrence(entities)
    return ExtractionResult(entities=entities, triples=triples)
```

### 5.3 Storage Integration

The extraction pipeline hooks into the existing memory storage flow. When a new memory is stored via `cc_store_memory`, the KG extractor runs *after* Stage 1 (fact extraction) and Stage 2 (consolidation):

```python
"""kg_store.py -- Persist extracted entities and triples to PostgreSQL."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def upsert_entity(
    session: AsyncSession,
    org_id: UUID,
    name: str,
    entity_type: str,
    embedding: list[float] | None = None,
) -> UUID:
    """Insert or update a KG entity. Returns the entity ID.

    Uses PostgreSQL ON CONFLICT for atomic upsert. If the entity already
    exists (same org + lowercase name), we update last_seen and increment
    mention_count.
    """
    stmt = pg_insert(kg_entities_table).values(
        org_id=org_id,
        name=name,
        entity_type=entity_type,
        embedding=embedding,
    ).on_conflict_on_constraint("uq_entity_org_name").do_update(
        set_={
            "last_seen": text("now()"),
            "mention_count": kg_entities_table.c.mention_count + 1,
            "entity_type": entity_type,  # Update type if extraction improves
        }
    ).returning(kg_entities_table.c.id)

    result = await session.execute(stmt)
    return result.scalar_one()


async def insert_triple(
    session: AsyncSession,
    org_id: UUID,
    subject_entity_id: UUID,
    predicate: str,
    object_entity_id: UUID,
    source_memory_id: UUID,
    confidence: float = 1.0,
    predicate_embedding: list[float] | None = None,
) -> UUID | None:
    """Insert a KG triple. Returns triple ID, or None if duplicate."""
    stmt = pg_insert(kg_triples_table).values(
        org_id=org_id,
        subject_entity_id=subject_entity_id,
        predicate=predicate,
        object_entity_id=object_entity_id,
        source_memory_id=source_memory_id,
        confidence=confidence,
        predicate_embedding=predicate_embedding,
    ).on_conflict_on_constraint("uq_triple_source").do_nothing().returning(
        kg_triples_table.c.id
    )

    result = await session.execute(stmt)
    row = result.first()
    return row[0] if row else None


async def link_entity_to_memory(
    session: AsyncSession,
    entity_id: UUID,
    memory_id: UUID,
    mention_context: str | None = None,
    char_offset: int | None = None,
) -> None:
    """Record that an entity was mentioned in a specific memory."""
    stmt = pg_insert(kg_entity_mentions_table).values(
        entity_id=entity_id,
        memory_id=memory_id,
        mention_context=mention_context,
        char_offset=char_offset,
    ).on_conflict_on_constraint("uq_entity_mention").do_nothing()

    await session.execute(stmt)


async def process_memory_for_kg(
    session: AsyncSession,
    org_id: UUID,
    memory_id: UUID,
    content: str,
    embedder=None,
) -> dict:
    """Full pipeline: extract entities and triples from a memory, persist to KG.

    Args:
        session: Active database session.
        org_id: Organization ID (for isolation).
        memory_id: The memory_units.id being processed.
        content: The memory text content.
        embedder: Optional EmbeddingProvider for entity embeddings.

    Returns:
        Summary dict with counts of entities and triples created.
    """
    from .kg_extractor import run_extraction

    result = run_extraction(content)

    entity_id_map: dict[str, UUID] = {}
    for entity in result.entities:
        # Compute entity name embedding if embedder is available
        emb = None
        if embedder is not None:
            emb_array = await embedder.embed(entity.name)
            emb = emb_array.tolist()

        eid = await upsert_entity(
            session, org_id, entity.name, entity.entity_type, emb
        )
        entity_id_map[entity.name] = eid

        await link_entity_to_memory(
            session, eid, memory_id, entity.context, entity.char_start
        )

    triples_created = 0
    for triple in result.triples:
        subj_id = entity_id_map.get(triple.subject)
        obj_id = entity_id_map.get(triple.object)
        if subj_id and obj_id:
            # Compute predicate embedding for CatRAG weighting
            pred_emb = None
            if embedder is not None:
                pred_array = await embedder.embed(triple.predicate)
                pred_emb = pred_array.tolist()

            tid = await insert_triple(
                session, org_id, subj_id, triple.predicate, obj_id,
                memory_id, triple.confidence, pred_emb
            )
            if tid is not None:
                triples_created += 1

    await session.flush()

    return {
        "entities_processed": len(result.entities),
        "triples_created": triples_created,
        "entity_names": [e.name for e in result.entities],
    }
```

### 5.4 Entity Resolution

A critical challenge: the same real-world entity may be referred to by different names ("Alice", "Alice Chen", "A. Chen"). Our v1 approach uses case-insensitive exact match on the `name_lower` column, which handles capitalization but not aliases.

**Future improvements (not in v1):**

1. **Embedding-based fuzzy match:** Before creating a new entity, query `kg_entities` for entities with similar embeddings (cosine similarity > 0.9). If found, merge.
2. **Alias table:** Track known aliases (e.g., "NYC" -> "New York City") and resolve on insert.
3. **LLM-assisted resolution:** Use a lightweight LLM call to determine if two entity names refer to the same thing.

For v1, the exact-match approach is sufficient. Most agent conversations use consistent entity names within an organization's context.

---

## 6. Retrieval Algorithm

### 6.1 Overview

Graph-based retrieval follows these steps:

1. **Query entity extraction:** Run spaCy NER on the query to find entity mentions
2. **Seed node identification:** Match query entities to `kg_entities` nodes (exact + fuzzy)
3. **Query-aware edge weighting (CatRAG):** Reweight graph edges by relevance to the query
4. **Personalized PageRank:** Propagate importance from seed nodes through the weighted graph
5. **Memory mapping:** Convert high-PPR entities to ranked memory lists via `kg_entity_mentions`
6. **Fusion:** Merge graph-based results with dense + lexical results via RRF

### 6.2 Personalized PageRank Implementation

PPR is the standard PageRank algorithm with a modification: instead of distributing the teleportation probability uniformly, it concentrates it on a specific set of "seed" nodes. This biases the random walk toward nodes that are structurally close to the seeds.

```python
"""kg_retrieval.py -- Graph-based retrieval with PPR and CatRAG weighting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank."""
    alpha: float = 0.15        # Teleportation (restart) probability
    max_iterations: int = 50   # Maximum PPR iterations
    tolerance: float = 1e-6    # Convergence threshold
    top_k: int = 20            # Number of top entities to return


@dataclass
class GraphRetrievalResult:
    """A memory scored by graph-based retrieval."""
    memory_id: UUID
    score: float
    source_entities: list[str]  # Entity names that contributed to this score
    hops: int  # Minimum graph distance from a seed node


async def find_seed_nodes(
    session: AsyncSession,
    org_id: UUID,
    query_entities: list[str],
    query_embedding: np.ndarray | None = None,
    fuzzy_threshold: float = 0.85,
) -> dict[UUID, float]:
    """Find KG nodes matching query entities. Returns {entity_id: seed_weight}.

    Strategy:
    1. Exact match on name_lower (weight 1.0)
    2. If query_embedding is provided and exact match fails, fall back to
       embedding similarity search (weight = similarity score)
    """
    seeds: dict[UUID, float] = {}

    for entity_name in query_entities:
        # Try exact match first
        stmt = select(
            text("id")
        ).select_from(
            text("kg_entities")
        ).where(
            text("org_id = :org_id AND name_lower = :name_lower")
        ).params(org_id=org_id, name_lower=entity_name.lower())

        result = await session.execute(stmt)
        row = result.first()
        if row:
            seeds[row[0]] = 1.0
            continue

        # Fall back to fuzzy embedding match
        if query_embedding is not None:
            stmt = text("""
                SELECT id, 1 - (embedding <=> :qemb) AS similarity
                FROM kg_entities
                WHERE org_id = :org_id
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> :qemb
                LIMIT 3
            """).params(org_id=org_id, qemb=str(query_embedding.tolist()))

            result = await session.execute(stmt)
            for row in result:
                if row[1] >= fuzzy_threshold:
                    seeds[row[0]] = float(row[1])

    return seeds


async def load_adjacency(
    session: AsyncSession,
    org_id: UUID,
) -> tuple[dict[UUID, int], np.ndarray, list[dict]]:
    """Load the full adjacency structure for an organization's KG.

    Returns:
        - node_index: mapping from entity UUID to integer index
        - adjacency: sparse-like adjacency matrix (N x N numpy array)
        - edge_metadata: list of dicts with predicate, confidence, predicate_embedding
          for each edge (indexed same as adjacency nonzero entries)

    Note: At our target scale (~1000 entities, ~5000 edges), loading the full
    adjacency into memory is fine. For larger graphs, use sparse matrices.
    """
    # Load all entities for this org
    stmt = text("SELECT id FROM kg_entities WHERE org_id = :org_id ORDER BY id")
    result = await session.execute(stmt.params(org_id=org_id))
    entity_ids = [row[0] for row in result]

    node_index = {eid: i for i, eid in enumerate(entity_ids)}
    n = len(entity_ids)

    if n == 0:
        return node_index, np.zeros((0, 0)), []

    adjacency = np.zeros((n, n), dtype=np.float64)
    edge_meta = []

    # Load all triples
    stmt = text("""
        SELECT subject_entity_id, object_entity_id, predicate,
               confidence, predicate_embedding
        FROM kg_triples
        WHERE org_id = :org_id
    """)
    result = await session.execute(stmt.params(org_id=org_id))

    for row in result:
        subj_idx = node_index.get(row[0])
        obj_idx = node_index.get(row[1])
        if subj_idx is not None and obj_idx is not None:
            adjacency[subj_idx, obj_idx] = row[3]  # confidence as base weight
            adjacency[obj_idx, subj_idx] = row[3]  # undirected for PPR
            edge_meta.append({
                "subject_idx": subj_idx,
                "object_idx": obj_idx,
                "predicate": row[2],
                "confidence": row[3],
                "predicate_embedding": row[4],
            })

    return node_index, adjacency, edge_meta


def apply_catrag_weighting(
    adjacency: np.ndarray,
    edge_metadata: list[dict],
    query_embedding: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """Apply CatRAG-style query-adaptive edge weighting.

    For each edge, compute cosine similarity between the query embedding
    and the edge's predicate embedding. Use softmax-scaled similarity to
    reweight the edge.

    This suppresses edges that are irrelevant to the current query,
    preventing PPR from drifting toward hub nodes along unrelated paths.

    Args:
        adjacency: Base adjacency matrix (N x N).
        edge_metadata: Per-edge metadata including predicate_embedding.
        query_embedding: Embedding of the current query (768-dim).
        temperature: Softmax temperature. Lower = more aggressive filtering.

    Returns:
        Reweighted adjacency matrix.
    """
    weighted = adjacency.copy()
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

    for edge in edge_metadata:
        pred_emb = edge.get("predicate_embedding")
        if pred_emb is None:
            continue  # Keep original weight if no predicate embedding

        pred_vec = np.array(pred_emb, dtype=np.float64)
        pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-9)

        # Cosine similarity between query and predicate
        sim = float(np.dot(query_norm, pred_norm))

        # Softmax-style scaling: e^(sim/T) / e^(1/T)
        # Normalizes so that a perfectly matching edge gets weight ~= base weight
        # and irrelevant edges get suppressed
        scale = np.exp(sim / temperature) / np.exp(1.0 / temperature)
        scale = np.clip(scale, 0.01, 2.0)  # Prevent zeroing out or exploding

        i, j = edge["subject_idx"], edge["object_idx"]
        weighted[i, j] *= scale
        weighted[j, i] *= scale

    return weighted


def personalized_pagerank(
    adjacency: np.ndarray,
    seed_indices: dict[int, float],
    config: PPRConfig | None = None,
) -> np.ndarray:
    """Compute Personalized PageRank scores.

    Args:
        adjacency: Weighted adjacency matrix (N x N). Can be pre-weighted
                   by CatRAG. Values represent edge weights (higher = stronger connection).
        seed_indices: Dict mapping node index -> seed weight (teleportation bias).
        config: PPR hyperparameters.

    Returns:
        Array of PPR scores for each node (N,).
    """
    if config is None:
        config = PPRConfig()

    n = adjacency.shape[0]
    if n == 0:
        return np.array([])

    # Build transition matrix (column-stochastic)
    col_sums = adjacency.sum(axis=0)
    col_sums[col_sums == 0] = 1.0  # Avoid division by zero for dangling nodes
    transition = adjacency / col_sums

    # Build personalization vector (seed distribution)
    personalization = np.zeros(n, dtype=np.float64)
    total_seed_weight = sum(seed_indices.values())
    if total_seed_weight > 0:
        for idx, weight in seed_indices.items():
            personalization[idx] = weight / total_seed_weight
    else:
        personalization[:] = 1.0 / n  # Uniform if no seeds

    # Power iteration
    scores = personalization.copy()
    alpha = config.alpha

    for iteration in range(config.max_iterations):
        prev = scores.copy()
        scores = (1 - alpha) * (transition @ scores) + alpha * personalization

        # Check convergence
        diff = np.abs(scores - prev).sum()
        if diff < config.tolerance:
            logger.debug("PPR converged in %d iterations (diff=%.2e)", iteration + 1, diff)
            break

    return scores


async def graph_retrieve(
    session: AsyncSession,
    org_id: UUID,
    query: str,
    query_embedding: np.ndarray,
    config: PPRConfig | None = None,
    enable_catrag: bool = True,
    catrag_temperature: float = 0.1,
) -> list[GraphRetrievalResult]:
    """Full graph-based retrieval pipeline.

    1. Extract entities from query
    2. Find seed nodes
    3. Load adjacency + apply CatRAG weighting
    4. Run PPR
    5. Map top entities to memories

    Args:
        session: Database session.
        org_id: Organization ID.
        query: The query text.
        query_embedding: Pre-computed query embedding (768-dim).
        config: PPR configuration.
        enable_catrag: Whether to apply query-aware edge weighting.
        catrag_temperature: CatRAG softmax temperature.

    Returns:
        List of GraphRetrievalResult, sorted by score descending.
    """
    from .kg_extractor import extract_entities

    if config is None:
        config = PPRConfig()

    # Step 1: Extract entities from query
    query_entities = extract_entities(query)
    entity_names = [e.name for e in query_entities]

    if not entity_names:
        # No entities found in query -- graph retrieval cannot help
        logger.debug("No entities extracted from query, skipping graph retrieval")
        return []

    # Step 2: Find seed nodes
    seeds = await find_seed_nodes(session, org_id, entity_names, query_embedding)

    if not seeds:
        logger.debug("No seed nodes found for entities: %s", entity_names)
        return []

    # Step 3: Load adjacency
    node_index, adjacency, edge_meta = await load_adjacency(session, org_id)

    if adjacency.size == 0:
        return []

    # Step 4: CatRAG weighting
    if enable_catrag and edge_meta:
        adjacency = apply_catrag_weighting(
            adjacency, edge_meta, query_embedding, catrag_temperature
        )

    # Step 5: Run PPR
    seed_indices = {}
    for entity_id, weight in seeds.items():
        if entity_id in node_index:
            seed_indices[node_index[entity_id]] = weight

    ppr_scores = personalized_pagerank(adjacency, seed_indices, config)

    # Step 6: Map top-PPR entities to memories
    index_to_entity = {i: eid for eid, i in node_index.items()}
    top_indices = np.argsort(ppr_scores)[::-1][:config.top_k]

    # Collect entity IDs with significant PPR scores
    top_entity_ids = []
    entity_scores = {}
    for idx in top_indices:
        score = ppr_scores[idx]
        if score < 1e-8:
            break
        eid = index_to_entity[idx]
        top_entity_ids.append(eid)
        entity_scores[eid] = score

    if not top_entity_ids:
        return []

    # Query kg_entity_mentions to find memories linked to top entities
    stmt = text("""
        SELECT DISTINCT em.memory_id, em.entity_id, e.name
        FROM kg_entity_mentions em
        JOIN kg_entities e ON e.id = em.entity_id
        WHERE em.entity_id = ANY(:entity_ids)
        ORDER BY em.memory_id
    """)
    result = await session.execute(stmt.params(entity_ids=top_entity_ids))

    # Aggregate: each memory's score = sum of PPR scores of its linked entities
    memory_scores: dict[UUID, float] = {}
    memory_entities: dict[UUID, list[str]] = {}

    for row in result:
        mid, eid, ename = row[0], row[1], row[2]
        score = entity_scores.get(eid, 0.0)
        memory_scores[mid] = memory_scores.get(mid, 0.0) + score
        memory_entities.setdefault(mid, []).append(ename)

    # Build results sorted by score
    results = []
    for mid, score in sorted(memory_scores.items(), key=lambda x: x[1], reverse=True):
        results.append(GraphRetrievalResult(
            memory_id=mid,
            score=score,
            source_entities=memory_entities.get(mid, []),
            hops=0,  # TODO: compute actual hop distance from seeds
        ))

    return results
```

### 6.3 Reciprocal Rank Fusion with Graph Results

The existing Stage 3 retrieval uses RRF to combine dense and lexical results. We extend this to include graph results as a third signal:

```python
"""Extending the existing fuse_rrf function to include graph results."""


def fuse_rrf(
    dense_results: list,      # Existing: from pgvector cosine search
    lexical_results: list,    # Existing: from tsvector BM25 search
    graph_results: list,      # NEW: from PPR graph retrieval
    k: int = 60,              # RRF constant (standard default)
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    graph_weight: float = 0.8,  # Slightly lower default -- graph is supplementary
) -> list:
    """Reciprocal Rank Fusion across three retrieval signals.

    RRF score for document d:
        score(d) = sum over all rankers r of: weight_r / (k + rank_r(d))

    If a document does not appear in a ranker's list, it gets no contribution
    from that ranker (not penalized, just not rewarded).
    """
    fused_scores: dict[str, float] = {}  # memory_id -> fused score
    fused_meta: dict[str, dict] = {}     # memory_id -> metadata

    for rank, result in enumerate(dense_results):
        mid = str(result.memory_id)
        fused_scores[mid] = fused_scores.get(mid, 0.0) + dense_weight / (k + rank + 1)
        fused_meta.setdefault(mid, {"sources": []})["sources"].append("dense")

    for rank, result in enumerate(lexical_results):
        mid = str(result.memory_id)
        fused_scores[mid] = fused_scores.get(mid, 0.0) + lexical_weight / (k + rank + 1)
        fused_meta.setdefault(mid, {"sources": []})["sources"].append("lexical")

    for rank, result in enumerate(graph_results):
        mid = str(result.memory_id)
        fused_scores[mid] = fused_scores.get(mid, 0.0) + graph_weight / (k + rank + 1)
        meta = fused_meta.setdefault(mid, {"sources": []})
        meta["sources"].append("graph")
        meta["graph_entities"] = result.source_entities

    # Sort by fused score descending
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(mid, score, fused_meta.get(mid, {})) for mid, score in ranked]
```

### 6.4 Query-Aware Edge Weighting (CatRAG) -- Detailed Explanation

The CatRAG mechanism deserves deeper explanation because it is the key differentiator from vanilla HippoRAG.

**The hub node problem, concretely:**

Imagine a community organization's memory system. The entity "Community Center" appears in 40 out of 100 memories -- it is a hub node. Without CatRAG, any query that touches the Community Center node will pull in memories about *everything* that happened there: budget meetings, volunteer schedules, event planning, maintenance issues.

A query like "What did Alice discuss about the garden project?" might:
1. Extract entities: "Alice", "garden project"
2. Find seed nodes for both
3. PPR propagates from Alice -> Community Center (because Alice is mentioned there often) -> budget meetings, volunteer schedules, etc.
4. The result is dominated by high-degree nodes, not Alice-garden-project connections

**CatRAG fix:**

Before PPR, we reweight edges:
1. Compute embedding of the query: "What did Alice discuss about the garden project?"
2. For each edge adjacent to seed nodes, compare the query embedding to the edge's predicate embedding
3. Edge "Alice -- discussed_at -- Community Center" with predicate "discussed_at" has moderate similarity to the query
4. Edge "Alice -- worked_on -- garden project" with predicate "worked_on" has HIGH similarity
5. Edge "Community Center -- budget_review -- Finance Committee" has LOW similarity to the query
6. PPR now preferentially follows Alice -> garden project paths over Alice -> Community Center -> unrelated paths

**Temperature parameter:**

The `temperature` parameter controls how aggressively we filter:
- `temperature = 0.05`: Very aggressive. Only highly relevant edges survive. Good for precise queries.
- `temperature = 0.1`: Moderate. Default. Balances precision and exploration.
- `temperature = 0.5`: Gentle. Most edges retain significant weight. Good for exploratory queries.

---

## 7. MCP Tool Design

### 7.1 cc_graph_query

The primary new tool for associative retrieval.

```python
"""MCP tool definition for graph-based associative retrieval."""

from dataclasses import dataclass


TOOL_DEFINITION = {
    "name": "cc_graph_query",
    "description": (
        "Associative memory retrieval using knowledge graph traversal. "
        "Use this when you need to find connections between entities, "
        "discover how people/topics/events relate to each other, or "
        "answer multi-hop questions that require linking information "
        "across multiple memories. For direct factual recall, prefer "
        "cc_recall_memory instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The associative query. Works best with named entities present.",
            },
            "org_id": {
                "type": "string",
                "description": "Organization UUID for memory isolation.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
                "default": 10,
            },
            "enable_catrag": {
                "type": "boolean",
                "description": "Enable query-aware edge weighting (CatRAG). Default true.",
                "default": True,
            },
            "include_graph_context": {
                "type": "boolean",
                "description": "Include entity relationship paths in results. Default false.",
                "default": False,
            },
            "merge_with_hybrid": {
                "type": "boolean",
                "description": (
                    "Merge graph results with dense+lexical hybrid search via RRF. "
                    "Default true. Set false for pure graph retrieval."
                ),
                "default": True,
            },
        },
        "required": ["query", "org_id"],
    },
}


async def handle_cc_graph_query(
    query: str,
    org_id: str,
    max_results: int = 10,
    enable_catrag: bool = True,
    include_graph_context: bool = False,
    merge_with_hybrid: bool = True,
    # Injected dependencies:
    session=None,
    embedder=None,
) -> dict:
    """Handler for cc_graph_query MCP tool.

    Returns:
        {
            "memories": [
                {
                    "id": "uuid",
                    "content": "memory text",
                    "score": 0.85,
                    "sources": ["graph", "dense"],  # which retrievers found it
                    "graph_entities": ["Alice", "Project Aurora"],  # if from graph
                    "graph_path": "Alice -> worked_on -> Aurora -> discussed_in -> Budget Meeting"
                },
                ...
            ],
            "query_entities": ["Alice", "Project Aurora"],
            "seed_nodes_found": 2,
            "total_kg_entities": 150,
            "total_kg_triples": 420,
        }
    """
    from uuid import UUID
    from .kg_retrieval import graph_retrieve, PPRConfig
    from .kg_extractor import extract_entities

    org_uuid = UUID(org_id)

    # Extract query entities for the response metadata
    query_ents = extract_entities(query)
    query_entity_names = [e.name for e in query_ents]

    # Compute query embedding
    query_emb = await embedder.embed(query)

    # Graph retrieval
    ppr_config = PPRConfig(top_k=max_results * 2)  # Over-fetch for RRF
    graph_results = await graph_retrieve(
        session, org_uuid, query, query_emb,
        config=ppr_config,
        enable_catrag=enable_catrag,
    )

    if merge_with_hybrid:
        # Run existing hybrid search in parallel
        # (This would call the existing Stage 3 retrieve function)
        # dense_results = await dense_search(session, org_uuid, query_emb, max_results * 2)
        # lexical_results = await lexical_search(session, org_uuid, query, max_results * 2)
        # fused = fuse_rrf(dense_results, lexical_results, graph_results)
        pass  # Integration point with existing code

    # Build response
    memories = []
    for result in graph_results[:max_results]:
        memory_data = {
            "id": str(result.memory_id),
            "score": round(result.score, 4),
            "sources": ["graph"],
            "graph_entities": result.source_entities,
        }
        # Fetch actual memory content
        # content = await fetch_memory_content(session, result.memory_id)
        # memory_data["content"] = content
        memories.append(memory_data)

    return {
        "memories": memories,
        "query_entities": query_entity_names,
        "seed_nodes_found": len(query_entity_names),  # Simplified
    }
```

### 7.2 cc_graph_stats

A diagnostic tool for inspecting the KG state.

```python
TOOL_DEFINITION = {
    "name": "cc_graph_stats",
    "description": (
        "Returns statistics about the knowledge graph: entity counts, "
        "triple counts, most-connected entities, entity types breakdown. "
        "Use for diagnostics and understanding what the agent knows structurally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "org_id": {
                "type": "string",
                "description": "Organization UUID.",
            },
            "include_top_entities": {
                "type": "boolean",
                "description": "Include the top 20 most-connected entities. Default true.",
                "default": True,
            },
        },
        "required": ["org_id"],
    },
}


async def handle_cc_graph_stats(
    org_id: str,
    include_top_entities: bool = True,
    session=None,
) -> dict:
    """Handler for cc_graph_stats MCP tool.

    Returns:
        {
            "total_entities": 150,
            "total_triples": 420,
            "total_mentions": 680,
            "entity_type_counts": {
                "PERSON": 45,
                "ORG": 30,
                "GPE": 25,
                ...
            },
            "top_entities": [
                {"name": "Community Center", "type": "ORG", "mentions": 40, "edges": 85},
                {"name": "Alice Chen", "type": "PERSON", "mentions": 25, "edges": 42},
                ...
            ],
            "graph_density": 0.037,  # edges / possible_edges
            "avg_degree": 5.6,
        }
    """
    from uuid import UUID
    from sqlalchemy import text

    org_uuid = UUID(org_id)

    # Entity count
    result = await session.execute(
        text("SELECT COUNT(*) FROM kg_entities WHERE org_id = :oid").params(oid=org_uuid)
    )
    total_entities = result.scalar()

    # Triple count
    result = await session.execute(
        text("SELECT COUNT(*) FROM kg_triples WHERE org_id = :oid").params(oid=org_uuid)
    )
    total_triples = result.scalar()

    # Mention count
    result = await session.execute(text("""
        SELECT COUNT(*) FROM kg_entity_mentions em
        JOIN kg_entities e ON e.id = em.entity_id
        WHERE e.org_id = :oid
    """).params(oid=org_uuid))
    total_mentions = result.scalar()

    # Entity type breakdown
    result = await session.execute(text("""
        SELECT entity_type, COUNT(*) FROM kg_entities
        WHERE org_id = :oid GROUP BY entity_type ORDER BY COUNT(*) DESC
    """).params(oid=org_uuid))
    type_counts = {row[0]: row[1] for row in result}

    stats = {
        "total_entities": total_entities,
        "total_triples": total_triples,
        "total_mentions": total_mentions,
        "entity_type_counts": type_counts,
        "graph_density": (
            (2 * total_triples) / (total_entities * (total_entities - 1))
            if total_entities > 1 else 0.0
        ),
        "avg_degree": (
            (2 * total_triples) / total_entities
            if total_entities > 0 else 0.0
        ),
    }

    if include_top_entities:
        result = await session.execute(text("""
            SELECT e.name, e.entity_type, e.mention_count,
                   (SELECT COUNT(*) FROM kg_triples t
                    WHERE t.subject_entity_id = e.id OR t.object_entity_id = e.id) AS edge_count
            FROM kg_entities e
            WHERE e.org_id = :oid
            ORDER BY e.mention_count DESC
            LIMIT 20
        """).params(oid=org_uuid))
        stats["top_entities"] = [
            {"name": row[0], "type": row[1], "mentions": row[2], "edges": row[3]}
            for row in result
        ]

    return stats
```

---

## 8. Hardware Constraints and Performance Budget

### Target Hardware

| Component | Spec | Constraint |
|-----------|------|------------|
| CPU | i5-12400F (6C/12T) | NER extraction runs here |
| RAM | 8 GB | Must fit spaCy + sentence-transformers + PostgreSQL |
| GPU | GTX 1660 SUPER 6 GB | sentence-transformers embedding; spaCy does NOT use this |
| Storage | SSD | PostgreSQL data |

### Memory Budget

| Component | Estimated RAM Usage |
|-----------|-------------------|
| PostgreSQL (shared_buffers) | ~1 GB |
| sentence-transformers (all-mpnet-base-v2) | ~500 MB |
| spaCy (en_core_web_md) | ~100 MB |
| Python application + FastAPI | ~200 MB |
| OS + other services | ~2 GB |
| **Headroom** | **~4 GB** |

The KG adjacency matrix for PPR is the main additional cost. At 1000 entities, a dense float64 matrix is 1000 x 1000 x 8 bytes = 8 MB. At 5000 entities, it is 200 MB. Beyond ~3000 entities, switch to scipy sparse matrices.

### Latency Budget

| Operation | Target | Notes |
|-----------|--------|-------|
| spaCy NER extraction (per memory) | <50 ms | CPU-only, small model |
| Entity embedding (per entity) | ~10 ms | Batched on GPU |
| KG storage (upsert entities + triples) | <20 ms | PostgreSQL, indexed |
| **Total ingest overhead** | **<100 ms per memory** | Acceptable |
| PPR computation (1000 nodes) | <5 ms | NumPy matrix ops |
| CatRAG edge weighting | <10 ms | Vectorized cosine similarity |
| Seed node lookup | <5 ms | Indexed query |
| Entity-to-memory mapping | <10 ms | Indexed join |
| **Total graph retrieval** | **<30 ms** | Well within interactive budget |

### Scaling Boundary

This design works well for **up to ~5,000 entities and ~20,000 triples** (approximately 2,000-5,000 memories depending on entity density). Beyond that:

1. Switch adjacency matrix to `scipy.sparse.csr_matrix`
2. Consider approximate PPR (e.g., push-based local PPR algorithms)
3. Or: upgrade to a dedicated graph database if the use case demands it

For our target of ~1,000 memories, we are well within budget.

---

## 9. Trade-off Analysis

### 9.1 spaCy NER vs. LLM-Based Extraction

| Factor | spaCy (en_core_web_md) | LLM (Claude API) |
|--------|----------------------|-------------------|
| **Cost per memory** | $0.00 | ~$0.002-0.01 (depending on length) |
| **Latency** | <50 ms | 500-2000 ms |
| **Quality (entities)** | Good for named entities (PERSON, ORG, GPE). Misses abstract concepts, implicit references | Excellent. Can extract implicit entities, relationships, concepts |
| **Quality (triples)** | Co-occurrence only. No real predicate extraction | Can extract meaningful predicates ("manages", "blocked_by", "collaborates_with") |
| **Hardware** | CPU only, 100 MB RAM | Requires API access and internet |
| **Offline capability** | Fully offline | Requires network |
| **Determinism** | Deterministic | Non-deterministic |

**Decision: Start with spaCy, design for swappability.**

The extraction pipeline is behind a clean interface (`extract_entities()` and `extract_triples_cooccurrence()`). To upgrade to LLM extraction later, implement the same interface with Claude API calls. The KG storage layer does not care how entities were extracted.

**Upgrade path:**

```python
# Future: LLM-based extraction (interface-compatible with spaCy version)
async def extract_entities_llm(text: str, llm_client) -> list[ExtractedEntity]:
    """Extract entities using Claude API for higher quality.

    Prompt asks for structured JSON output with entity names, types,
    and relationships. Falls back to spaCy if API is unavailable.
    """
    prompt = f"""Extract all named entities and their relationships from this text.
Return JSON: {{"entities": [{{"name": "...", "type": "PERSON|ORG|...", "context": "..."}}],
               "triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}

Text: {text}"""

    try:
        response = await llm_client.complete(prompt)
        parsed = json.loads(response)
        return [ExtractedEntity(**e) for e in parsed["entities"]]
    except Exception:
        # Graceful fallback to spaCy
        return extract_entities(text)
```

### 9.2 PostgreSQL vs. Neo4j

| Factor | PostgreSQL (our approach) | Neo4j |
|--------|--------------------------|-------|
| **Operational complexity** | Zero. Already running. | New service to deploy, monitor, back up |
| **Query language** | SQL (familiar) | Cypher (new skill) |
| **Graph traversal** | Application-level PPR (NumPy) | Native graph algorithms, optimized traversal |
| **Scale sweet spot** | <10K entities | 10K-10M+ entities |
| **Cost** | Free (already provisioned) | Free Community or paid Enterprise |
| **Joins with memory tables** | Native (same DB, same transaction) | Cross-database joins (complex) |
| **Backup/restore** | Single pg_dump | Separate backup tooling |

**Decision: PostgreSQL.**

At our scale, the performance difference is negligible. The operational simplicity of keeping everything in one database is decisive. If we ever grow past 10K entities (which would require ~5,000+ memories), we can revisit.

### 9.3 Open KG vs. Fixed Schema

We use an **open knowledge graph** where predicates are free-form text strings ("co_occurs_with", "discussed", "works_on") rather than a fixed ontology (HAS_ROLE, WORKS_AT, REPORTS_TO).

**Why open:**
- Agent memory content is unpredictable. A fixed schema cannot anticipate all relationship types.
- spaCy co-occurrence extraction does not produce typed predicates anyway.
- HippoRAG 2 demonstrated that open KGs work well for retrieval (the PPR algorithm does not require typed edges).
- When we upgrade to LLM extraction, the LLM can produce richer predicates without schema changes.

**Downside:** Open predicates make it harder to query the graph with structured questions ("find all REPORTS_TO relationships"). But our primary use case is retrieval-augmented generation, not structured querying.

---

## 10. Consolidation and Lifecycle

### 10.1 KG Maintenance During Memory Consolidation

The existing Stage 2 consolidation pipeline merges and deduplicates memories. When memories are consolidated, the KG must be updated:

1. **Memory deletion:** When a memory is deleted (expired, deduplicated), its triples survive if other memories reference the same entities. The `source_memory_id` foreign key cascades, so triples from the deleted memory are removed, but entities and their other triples persist.

2. **Memory merging:** When two memories are merged into one during consolidation, run KG extraction on the merged content and associate the results with the new memory ID. Old mention links are cleaned up by the cascade.

3. **Entity pruning:** Periodically check for orphaned entities (entities with zero mentions). These can be safely deleted:

```sql
-- Find and remove orphaned entities (no remaining mention links)
DELETE FROM kg_entities
WHERE id IN (
    SELECT e.id FROM kg_entities e
    LEFT JOIN kg_entity_mentions em ON em.entity_id = e.id
    WHERE em.id IS NULL
      AND e.org_id = :org_id
);
```

### 10.2 Incremental KG Construction

The KG is built incrementally: each new memory adds entities and triples. Over time, the graph becomes richer as entities recur and new connections form. This is a natural fit for agent memory, where information arrives continuously.

**No batch rebuild required.** Unlike some RAG systems that require periodic full re-indexing, the KG is always up to date because extraction happens on the ingest path.

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
"""test_kg_extractor.py -- Tests for entity and triple extraction."""

import pytest
from kg_extractor import extract_entities, extract_triples_cooccurrence, run_extraction


class TestExtractEntities:
    def test_extracts_person(self):
        entities = extract_entities("Alice met with Bob at the community center.")
        names = {e.name for e in entities}
        assert "Alice" in names
        assert "Bob" in names

    def test_extracts_org(self):
        entities = extract_entities("Microsoft announced a partnership with OpenAI.")
        types = {e.name: e.entity_type for e in entities}
        assert types.get("Microsoft") == "ORG"

    def test_empty_text(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        entities = extract_entities("The weather was nice today.")
        # May or may not extract "today" depending on model -- either way is fine
        assert isinstance(entities, list)

    def test_deduplication(self):
        entities = extract_entities("Alice said hello. Then Alice said goodbye.")
        alice_count = sum(1 for e in entities if e.name.lower() == "alice")
        assert alice_count == 1

    def test_context_captured(self):
        entities = extract_entities("Alice presented the quarterly report to the board.")
        alice = next(e for e in entities if e.name == "Alice")
        assert "quarterly report" in alice.context or "presented" in alice.context


class TestExtractTriplesCooccurrence:
    def test_nearby_entities_linked(self):
        from kg_extractor import ExtractedEntity
        e1 = ExtractedEntity(name="Alice", entity_type="PERSON", char_start=0, char_end=5, context="")
        e2 = ExtractedEntity(name="Bob", entity_type="PERSON", char_start=15, char_end=18, context="")
        triples = extract_triples_cooccurrence([e1, e2], window_chars=200)
        assert len(triples) == 1
        assert triples[0].subject == "Alice"
        assert triples[0].object == "Bob"

    def test_distant_entities_not_linked(self):
        from kg_extractor import ExtractedEntity
        e1 = ExtractedEntity(name="Alice", entity_type="PERSON", char_start=0, char_end=5, context="")
        e2 = ExtractedEntity(name="Bob", entity_type="PERSON", char_start=500, char_end=503, context="")
        triples = extract_triples_cooccurrence([e1, e2], window_chars=200)
        assert len(triples) == 0

    def test_confidence_decreases_with_distance(self):
        from kg_extractor import ExtractedEntity
        e1 = ExtractedEntity(name="Alice", entity_type="PERSON", char_start=0, char_end=5, context="")
        e_near = ExtractedEntity(name="Bob", entity_type="PERSON", char_start=10, char_end=13, context="")
        e_far = ExtractedEntity(name="Carol", entity_type="PERSON", char_start=180, char_end=185, context="")
        triples = extract_triples_cooccurrence([e1, e_near, e_far], window_chars=200)
        near_triple = next(t for t in triples if t.object == "Bob")
        far_triple = next(t for t in triples if t.object == "Carol")
        assert near_triple.confidence > far_triple.confidence


class TestRunExtraction:
    def test_full_pipeline(self):
        result = run_extraction("Alice and Bob discussed the Aurora project at the community center.")
        assert len(result.entities) >= 2
        # At least Alice and Bob should produce a co-occurrence triple
        assert len(result.triples) >= 1
```

### 11.2 PPR Tests

```python
"""test_kg_retrieval.py -- Tests for PPR and graph retrieval."""

import numpy as np
import pytest
from kg_retrieval import personalized_pagerank, apply_catrag_weighting, PPRConfig


class TestPersonalizedPageRank:
    def test_single_node(self):
        adj = np.array([[0.0]])
        scores = personalized_pagerank(adj, {0: 1.0})
        assert scores[0] == pytest.approx(1.0)

    def test_two_connected_nodes_seed_bias(self):
        adj = np.array([[0.0, 1.0],
                        [1.0, 0.0]])
        scores = personalized_pagerank(adj, {0: 1.0})
        # Node 0 (seed) should have higher score than node 1
        assert scores[0] > scores[1]

    def test_disconnected_node_gets_no_score(self):
        # Nodes 0-1 connected, node 2 isolated
        adj = np.array([[0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
        scores = personalized_pagerank(adj, {0: 1.0})
        assert scores[2] < 1e-6

    def test_hub_node_accumulates_score(self):
        # Star graph: node 0 connected to nodes 1, 2, 3
        n = 4
        adj = np.zeros((n, n))
        for i in range(1, n):
            adj[0, i] = 1.0
            adj[i, 0] = 1.0
        scores = personalized_pagerank(adj, {1: 1.0})
        # Hub (node 0) should have high score via propagation
        assert scores[0] > scores[2]
        assert scores[0] > scores[3]

    def test_convergence(self):
        np.random.seed(42)
        n = 50
        adj = (np.random.rand(n, n) > 0.8).astype(float)
        adj = (adj + adj.T) / 2  # Symmetric
        np.fill_diagonal(adj, 0)
        scores = personalized_pagerank(adj, {0: 1.0}, PPRConfig(max_iterations=200))
        assert abs(scores.sum() - 1.0) < 0.01  # Scores approximately sum to 1

    def test_empty_graph(self):
        scores = personalized_pagerank(np.zeros((0, 0)), {})
        assert len(scores) == 0


class TestCatRAGWeighting:
    def test_relevant_edge_gets_higher_weight(self):
        adj = np.array([[0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0]])

        # Edge 0->1 has predicate embedding similar to query
        # Edge 0->2 has predicate embedding dissimilar to query
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 765)
        edge_meta = [
            {"subject_idx": 0, "object_idx": 1, "predicate": "related",
             "confidence": 1.0, "predicate_embedding": [0.9, 0.1, 0.0] + [0.0] * 765},
            {"subject_idx": 0, "object_idx": 2, "predicate": "unrelated",
             "confidence": 1.0, "predicate_embedding": [0.0, 0.0, 1.0] + [0.0] * 765},
        ]

        weighted = apply_catrag_weighting(adj, edge_meta, query_emb, temperature=0.1)

        # Edge 0->1 (relevant) should be stronger than edge 0->2 (irrelevant)
        assert weighted[0, 1] > weighted[0, 2]

    def test_no_predicate_embedding_preserves_weight(self):
        adj = np.array([[0.0, 1.0],
                        [1.0, 0.0]])
        query_emb = np.zeros(768)
        edge_meta = [
            {"subject_idx": 0, "object_idx": 1, "predicate": "test",
             "confidence": 1.0, "predicate_embedding": None},
        ]
        weighted = apply_catrag_weighting(adj, edge_meta, query_emb)
        assert weighted[0, 1] == adj[0, 1]
```

---

## 12. Rollout Plan

### Phase 1: Schema + Extraction (Week 1-2)

- [ ] Run Alembic migration 002 to create KG tables
- [ ] Implement `kg_extractor.py` with spaCy NER + co-occurrence triples
- [ ] Implement `kg_store.py` with PostgreSQL upsert logic
- [ ] Hook extraction into the `cc_store_memory` pipeline
- [ ] Write unit tests for extraction and storage
- [ ] Backfill: run extraction on all existing memories to seed the KG

### Phase 2: Retrieval (Week 3-4)

- [ ] Implement `kg_retrieval.py` with PPR and CatRAG weighting
- [ ] Implement `cc_graph_query` MCP tool
- [ ] Implement `cc_graph_stats` MCP tool
- [ ] Integrate graph results into Stage 3 RRF fusion
- [ ] Write retrieval unit tests and integration tests
- [ ] Manual testing with real queries

### Phase 3: Tuning + Hardening (Week 5-6)

- [ ] Tune PPR alpha, CatRAG temperature, RRF graph_weight on real queries
- [ ] Add entity pruning to consolidation pipeline
- [ ] Add observability: log extraction counts, PPR latencies, graph density over time
- [ ] Load test with 1000+ synthetic memories
- [ ] Document the upgrade path to LLM-based extraction

### Future: Phase 4 (Not Scheduled)

- LLM-based entity and triple extraction (Claude API)
- Entity alias resolution
- Temporal graph evolution (entity relationships that change over time)
- Graph visualization endpoint for debugging

---

## 13. Configuration

All KG-related settings are exposed via environment variables with sensible defaults, following the same pattern as existing kintsugi configuration:

```python
"""Configuration for the KG extension."""

from pydantic_settings import BaseSettings


class KGSettings(BaseSettings):
    """Knowledge Graph configuration."""

    # Extraction
    kg_enabled: bool = True
    kg_spacy_model: str = "en_core_web_md"
    kg_cooccurrence_window: int = 200  # chars
    kg_min_entity_length: int = 2

    # Retrieval
    kg_ppr_alpha: float = 0.15  # Teleportation probability
    kg_ppr_max_iter: int = 50
    kg_ppr_tolerance: float = 1e-6
    kg_ppr_top_k: int = 20

    # CatRAG
    kg_catrag_enabled: bool = True
    kg_catrag_temperature: float = 0.1

    # Fusion
    kg_rrf_weight: float = 0.8  # Weight for graph signal in RRF

    # Maintenance
    kg_prune_orphans_interval_hours: int = 24

    model_config = {"env_prefix": "KINTSUGI_"}
```

---

## 14. Open Questions

1. **Entity embedding granularity.** Should we embed entity *names* (short, may lack context) or entity *descriptions* (richer, but where does the description come from)? Current design embeds names. May want to embed the concatenation of all mention contexts for better disambiguation.

2. **Cross-org entity linking.** Currently, entities are strictly org-isolated. Should there be a mechanism for recognizing that "US Congress" in org A and "US Congress" in org B refer to the same real-world entity? For now, no -- org isolation is a security boundary.

3. **Predicate vocabulary.** Co-occurrence produces only "co_occurs_with" predicates. This limits CatRAG's ability to differentiate edges. The embedding-based semantic relation extraction could help, as could the LLM upgrade path.

4. **Graph versioning.** If we want to answer "what did the knowledge graph look like a month ago?" we would need temporal versioning of edges. Not in scope for v1, but the `created_at` columns on triples provide a foundation.

5. **Bidirectional vs. directed edges.** Current PPR treats the graph as undirected (adjacency is symmetrized). Some relationships are naturally directed ("Alice manages Bob"). Worth revisiting if we upgrade to LLM extraction that produces directed predicates.

---

## 15. References

1. **HippoRAG 2:** Bernal Jimenez Gutierrez, Yiheng Shu, Michihiro Yasunaga, Yu Su. "HippoRAG 2: From Retrieval-Augmented Generation to Agentic Knowledge Management." arXiv:2502.14802, ICML 2025.

2. **CatRAG:** (arXiv:2602.01965, February 2026). Extends HippoRAG 2 with query-aware dynamic edge weighting to address semantic drift toward hub nodes during Personalized PageRank traversal.

3. **Personalized PageRank:** Haveliwala, T. "Topic-Sensitive PageRank." WWW 2002. The foundational algorithm for biased graph traversal from seed nodes.

4. **Reciprocal Rank Fusion:** Cormack, G., Clarke, C., Buettcher, S. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." SIGIR 2009.

5. **spaCy NER:** Honnibal, M., Montani, I. "spaCy: Industrial-Strength Natural Language Processing." https://spacy.io.

6. **Hippocampal Indexing Theory:** Teyler, T.J., DiScenna, P. "The hippocampal memory indexing theory." Behavioral Neuroscience, 1986. The neuroscience foundation for HippoRAG's architecture.

---

## Appendix A: Full Dependency Additions

```toml
# Add to pyproject.toml [project.dependencies]
"spacy>=3.7,<4",

# After install, download the model:
# python -m spacy download en_core_web_md
```

No new infrastructure services required. The KG lives entirely within the existing PostgreSQL instance.

---

## Appendix B: Backfill Script

For populating the KG from existing memories:

```python
"""backfill_kg.py -- Populate knowledge graph from existing memories."""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

logger = logging.getLogger(__name__)


async def backfill(database_url: str, org_id: str, batch_size: int = 50):
    """Process all existing memories through KG extraction.

    Args:
        database_url: PostgreSQL connection string.
        org_id: Organization UUID to process.
        batch_size: Number of memories to process per batch.
    """
    engine = create_async_engine(database_url)

    async with AsyncSession(engine) as session:
        # Count total memories
        result = await session.execute(
            text("SELECT COUNT(*) FROM memory_units WHERE org_id = :oid").params(oid=org_id)
        )
        total = result.scalar()
        logger.info("Backfilling KG for %d memories", total)

        offset = 0
        processed = 0

        while offset < total:
            result = await session.execute(text("""
                SELECT id, content FROM memory_units
                WHERE org_id = :oid
                ORDER BY created_at
                LIMIT :limit OFFSET :offset
            """).params(oid=org_id, limit=batch_size, offset=offset))

            rows = result.fetchall()
            if not rows:
                break

            for memory_id, content in rows:
                try:
                    from kg_store import process_memory_for_kg
                    stats = await process_memory_for_kg(
                        session, org_id, memory_id, content
                    )
                    processed += 1
                    if processed % 100 == 0:
                        logger.info("Processed %d/%d memories", processed, total)
                except Exception:
                    logger.exception("Failed to process memory %s", memory_id)

            await session.commit()
            offset += batch_size

        logger.info("Backfill complete: %d memories processed", processed)

    await engine.dispose()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    # Usage: python backfill_kg.py <database_url> <org_id>
    asyncio.run(backfill(sys.argv[1], sys.argv[2]))
```
