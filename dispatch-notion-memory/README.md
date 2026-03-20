# Dispatch Notion Memory

A persistent memory MCP for Claude Desktop (Cowork/Dispatch) that uses Notion as the primary store and human interface, with SQLite-vec as a local embedding cache for fast semantic search.

## Architecture

```
Discord Voice/Text → n8n → Notion Inbox
                              ↓
                     Memory MCP (this service)
                     ├── Notion API (read/write)
                     ├── SQLite-vec (embeddings + local index)
                     └── spaCy NER (entity extraction)
                              ↓
                     Notion PARA databases
                     (Inbox → Projects/Areas/Resources → Archive)
```

**Primary store:** Notion Second Brain (PARA structure)
**Embedding cache:** SQLite-vec (local, fast semantic search)
**Entity extraction:** spaCy NER (KG-ready from Phase 2)
**Transport:** MCP (stdio or HTTP)

## Notion Schema

Maps to existing PARA databases with added fields:

| Notion Database | Memory Role | Key Fields |
|---|---|---|
| Inbox | New unprocessed memories | Name, Content, Source, Tags, Confidence, Captured At |
| Projects | Active project context | + Significance, Entities, Priority, Due Date |
| Areas | Ongoing life/work areas | + Significance, Entities |
| Resources | Reference material | + Significance, Entities, Topic |
| Archive | Decayed memories | + Archived Date, Original Category |

**New fields to add:**
- `Significance` (number, 0.0–1.0) — governs decay behavior
- `Entities` (multi-select) — extracted named entities for KG readiness

## Phases

### Phase 1 — Connect and Validate
Get the service running as a Cowork MCP. Read/write to Notion. Basic store, search, retrieve, consolidate.

### Phase 2 — Enrich (current target)
Significance scoring, typed memories, entity extraction on ingest, significance-aware decay, dream consolidation with Notion as the lifecycle manager.

### Phase 3 — Knowledge Graph
Triple generation from entity co-occurrence, graph traversal queries, Personalized PageRank retrieval. Follows the [hipporag-catrag-kg](../hipporag-catrag-kg/) design.

## MCP Tools

| Tool | Phase | Description |
|---|---|---|
| `memory_store` | 1 | Create a memory in Notion + embed locally |
| `memory_search` | 1 | Semantic search via SQLite-vec |
| `memory_retrieve` | 1 | Fetch by ID, type, time range |
| `memory_update` | 1 | Modify content or metadata |
| `memory_delete` | 1 | Remove memory |
| `memory_consolidate` | 1 | Trigger dream consolidation cycle |
| `memory_recall` | 2 | Query by type, entities, significance, compound filters |
| `memory_refresh` | 2 | Bump access metrics without modifying |
| `memory_compress` | 2 | Significance-aware decay pass |
| `memory_status` | 2 | Stats by type, status, significance |
| `memory_bootstrap` | 2 | Curated context payload for session init |
| `memory_extract_triples` | 3 | Generate KG triples |
| `memory_traverse` | 3 | Graph query via PageRank |
| `memory_entity_search` | 3 | Find memories by entity |
| `memory_graph_stats` | 3 | KG connectivity metrics |

## Setup

```bash
cd dispatch-notion-memory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your Notion API key and database IDs

# Run as MCP
python -m dispatch_memory.server
```

## Configuration

See `config/config.example.yaml` for required settings:
- Notion integration token
- Database IDs for each PARA database
- SQLite-vec database path
- Consolidation schedule
- Significance defaults by memory type

## Related Architectures

- [kintsugi-cma](../kintsugi-cma/) — Three-stage CMA with BDI governance (significance scoring, consolidation patterns)
- [hipporag-catrag-kg](../hipporag-catrag-kg/) — Associative KG retrieval with PageRank (Phase 3 target)
