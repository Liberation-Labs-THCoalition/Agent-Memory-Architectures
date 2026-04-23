# Project Mnemosyne

**Agent memory architectures for the Coalition and beyond.**

Named for the titaness of memory — mother of the Muses. The memory
systems that make everything else possible.

## Architectures

| Name | Description | Status |
|------|-------------|--------|
| [oracle-memory](./oracle-memory/) | KV cache recording with Lyra Technique geometry — three-tier (activation, journal, consolidated) | Active |
| [kintsugi-cma](./kintsugi-cma/) | Cognitive Memory Architecture — three-stage hybrid retrieval (compression → consolidation → retrieval) with BDI governance | Phase 1 Complete |
| [hipporag-catrag-kg](./hipporag-catrag-kg/) | Knowledge graph layer — HippoRAG 2 + CatRAG for associative retrieval via Personalized PageRank | Design Complete |
| [mnemosyne-wiki](./mnemosyne-wiki/) | LLM Wiki layer — interlinked markdown generated from knowledge graphs. Human-browsable knowledge surface. | **New** |

## The Stack

```
┌─────────────────────────────────────────────┐
│  LLM Wiki (mnemosyne-wiki)                  │  ← Human-readable surface
│  Interlinked markdown pages                  │
├─────────────────────────────────────────────┤
│  Knowledge Graph (hipporag-catrag-kg)       │  ← Associative structure
│  Entities, typed predicates, PPR retrieval   │
├─────────────────────────────────────────────┤
│  Cognitive Memory (kintsugi-cma)            │  ← Compression + consolidation
│  Stage 1→2→3, significance scoring          │
├─────────────────────────────────────────────┤
│  Cache Recording (oracle-memory)            │  ← Geometry signal source
│  KV snapshots, spectral features            │
└─────────────────────────────────────────────┘
```

Each layer is independent. Use one, some, or all. The stack composes
bottom-up: cache geometry feeds into significance scoring, which feeds
into the knowledge graph, which generates the wiki.

## Organization

Monorepo. Each architecture lives in its own directory with independent
dependencies, tests, and docs. Shared patterns emerge from practice,
not premature abstraction.

## About

Built by [Liberation Labs / TH Coalition](https://github.com/Liberation-Labs-THCoalition).
Memory systems for agents that remember with integrity.

Architecture: CC (Coalition Code). Research: Lyra. Infrastructure: Nexus.
Direction: Thomas Edrington.
