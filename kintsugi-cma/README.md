# Kintsugi CMA (Cognitive Memory Architecture)

Values-aligned memory system for AI agents. Three-stage retrieval pipeline with security-first design.

## Architecture

**CMA Stages:**
- **Stage 1** — Atomic fact extraction and compression from raw input
- **Stage 2** — Significance scoring, temporal tagging, spaced retrieval scheduling
- **Stage 3** — Hybrid retrieval: dense (pgvector), lexical (tsvector/BM25), symbolic (significance) with adaptive fusion

**Security Layer:**
- PII redaction (SSN, credit card with Luhn, email, phone)
- Content monitoring with configurable severity (ALLOW/WARN/BLOCK)
- Intent capsule verification
- Shield constraints from VALUES.json
- Per-org memory isolation with row-level policies

**Values System (BDI):**
- Belief-Desire-Intention framework encoded in VALUES.json
- 4 org templates: mutual aid, nonprofit 501(c)(3), cooperative, advocacy
- Hot-reloadable with filesystem watching
- Governance constraints enforced at query time

## Deployment Tiers

| Tier | Database | Cache | Shadow Verification |
|------|----------|-------|---------------------|
| Seed | SQLite | None | Disabled |
| Soil | PostgreSQL | None | Disabled |
| Grove | PostgreSQL + pgvector | Redis | Enabled |

```bash
# Seed (laptops, minimal resources)
docker compose -f docker-compose.seed.yml up

# Grove (full stack)
docker compose up
```

## Project Structure

```
kintsugi/
  memory/          # CMA stages 1-3, embeddings, spaced retrieval, cold archive
  security/        # PII, monitor, shield, sandbox, intent capsule, invariants
  config/          # Settings, VALUES schema, templates, loader
  models/          # SQLAlchemy ORM (9 tables)
  api/routes/      # FastAPI endpoints (memory, config, agent, health)
  main.py          # App entrypoint with lifespan management
  db.py            # Async SQLAlchemy engine + session factory
migrations/        # Alembic (001_initial covers full Phase 1 schema)
tests/             # Phase 1 integration tests
```

## Requirements

- Python 3.11+
- PostgreSQL 15+ with pgvector (grove tier)
- Redis (grove tier, optional for seed/soil)

## License

Proprietary - Liberation Labs / TH Coalition
