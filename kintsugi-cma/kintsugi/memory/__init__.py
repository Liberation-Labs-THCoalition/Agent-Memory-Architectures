"""Kintsugi Continuum Memory Architecture (CMA) — Phase 1 Stream 1C.

Implements the SimpleMem pipeline (arXiv:2601.02553) with five modules:

- **embeddings**: Vector embedding providers (local + API)
- **cma_stage1**: Semantic structured compression (sliding window, entropy, normalization)
- **cold_archive**: Sub-threshold compressed storage with integrity verification
- **temporal**: Append-only decision/event log
- **significance**: Memory layers, expiration policies, and reaper
- **spaced**: Fibonacci spaced retrieval scheduling
"""

from kintsugi.memory.cma_stage1 import (
    AtomicFact,
    Stage1Result,
    Turn,
    Window,
    filter_windows,
    normalize_window,
    run_stage1,
    score_entropy,
    segment_dialogue,
)
from kintsugi.memory.cold_archive import (
    ArchivedWindow,
    ColdArchive,
    IntegrityReport,
)
from kintsugi.memory.embeddings import (
    APIEmbeddingProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
)
from kintsugi.memory.significance import (
    ExpiredMemoryReaper,
    MemoryLayer,
    ReapResult,
    compute_expiration,
    compute_layer,
)
from kintsugi.memory.spaced import (
    FIBONACCI,
    DueMemory,
    SpacedRetrieval,
    fib_interval,
)
from kintsugi.memory.temporal import (
    Category,
    TemporalEvent,
    TemporalLog,
)

from kintsugi.memory.cma_stage2 import (
    Fact,
    Insight,
    compute_affinity,
    cluster_facts,
    consolidate,
)
from kintsugi.memory.cma_stage3 import (
    QueryProfile,
    ScoredResult,
    estimate_complexity,
    fuse_rrf,
    fuse_weighted,
    retrieve,
)
from kintsugi.memory.org_isolation import (
    ORG_MEMORIES_SCHEMA,
    OrgMemoryStore,
    MemoryRecord,
    get_org_connection,
)
from kintsugi.memory.bdi_bridge import (
    BDIBridge,
    Belief,
    Desire,
    Intention,
)

__all__ = [
    # embeddings
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "APIEmbeddingProvider",
    "get_embedding_provider",
    # cma_stage1
    "Turn",
    "Window",
    "AtomicFact",
    "Stage1Result",
    "segment_dialogue",
    "score_entropy",
    "filter_windows",
    "normalize_window",
    "run_stage1",
    # cma_stage2
    "Fact",
    "Insight",
    "compute_affinity",
    "cluster_facts",
    "consolidate",
    # cma_stage3
    "QueryProfile",
    "ScoredResult",
    "estimate_complexity",
    "fuse_rrf",
    "fuse_weighted",
    "retrieve",
    # org_isolation
    "ORG_MEMORIES_SCHEMA",
    "OrgMemoryStore",
    "MemoryRecord",
    "get_org_connection",
    # bdi_bridge
    "BDIBridge",
    "Belief",
    "Desire",
    "Intention",
    # cold_archive
    "ColdArchive",
    "ArchivedWindow",
    "IntegrityReport",
    # temporal
    "TemporalLog",
    "TemporalEvent",
    "Category",
    # significance
    "MemoryLayer",
    "compute_layer",
    "compute_expiration",
    "ExpiredMemoryReaper",
    "ReapResult",
    # spaced
    "FIBONACCI",
    "fib_interval",
    "SpacedRetrieval",
    "DueMemory",
]
