# HippoRAG/CatRAG Knowledge Graph Extension

Knowledge graph layer for associative memory retrieval in agent memory systems. Adds structured entity-relationship tracking to an existing PostgreSQL + pgvector memory architecture, enabling multi-hop and relational queries that pure embedding search cannot answer.

## What This Solves

Standard hybrid retrieval (dense vectors + BM25) handles direct recall well: "what happened Tuesday?" matches memories with overlapping semantics. It fails at associative queries: "what connects Alice to the budget shortfall?" requires traversing *relationships between* memories that share no direct textual overlap.

This extension builds a lightweight knowledge graph from memory content and uses Personalized PageRank to surface structurally connected memories.

## Research Basis

- **HippoRAG 2** (arXiv:2502.14802, ICML 2025): Neurobiologically-inspired open knowledge graph + Personalized PageRank for retrieval
- **CatRAG** (arXiv:2602.01965, Feb 2026): Query-aware dynamic edge weighting to prevent semantic drift toward hub nodes

## Architecture at a Glance

```
Memory Ingest:
  content -> spaCy NER -> entities + co-occurrence triples -> PostgreSQL KG tables

Query:
  query -> extract entities -> find seed nodes -> CatRAG edge weighting -> PPR -> ranked memories
         \-> dense search (pgvector) ----\
         \-> lexical search (tsvector) ---+--> Reciprocal Rank Fusion --> final results
         \-> graph search (PPR) ---------/
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entity extraction | spaCy (en_core_web_md) | Zero API cost, runs on CPU, ~100 MB RAM |
| Graph database | PostgreSQL (same instance) | No new infrastructure, SQL-based graph queries suffice at <5K entities |
| Relationship inference | Co-occurrence + embedding similarity | No LLM calls needed, upgradeable to Claude API later |
| Edge weighting | CatRAG query-adaptive | Prevents hub-node drift in PPR, critical for real-world entity distributions |
| Result fusion | RRF with 3 signals | Additive -- graph retrieval supplements but never replaces existing search |

## New Schema

Three tables added to the existing PostgreSQL database:

- **kg_entities** -- Canonical entity nodes (PERSON, ORG, GPE, etc.) with 768-dim embeddings
- **kg_triples** -- Directed edges: (subject, predicate, object) with source memory provenance
- **kg_entity_mentions** -- Links entities to the memories where they appear

## New MCP Tools

- **cc_graph_query** -- Associative retrieval via knowledge graph traversal
- **cc_graph_stats** -- KG diagnostics: entity counts, type breakdown, top entities, graph density

## Hardware Requirements

Runs on modest hardware alongside the existing stack:

- spaCy NER: CPU-only, ~100 MB RAM, <50 ms per memory
- PPR computation: <5 ms for 1000-node graphs (NumPy)
- Total graph retrieval: <30 ms end-to-end

Tested target: i5-12400F, 8 GB RAM, GTX 1660 SUPER 6 GB.

## Documentation

See [DESIGN.md](./DESIGN.md) for the full design document including:

- Complete SQL schema with indexes
- Alembic migration code
- Python implementation of extraction, storage, PPR, and CatRAG
- MCP tool definitions and handlers
- Trade-off analysis (spaCy vs LLM, PostgreSQL vs Neo4j)
- Unit test examples
- Phased rollout plan
- Hardware performance budget

## Status

**Implemented and deployed.** KG engine running in production on CC-Memory MCP server. Backfill of 100 memories yielded 788 entities and 25,290 co-occurrence triples. Entity embedding backfill for full CatRAG adaptive weighting is pending.
