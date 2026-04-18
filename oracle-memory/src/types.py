"""Core types for the Oracle Memory architecture.

Portable — no framework dependencies. Uses dataclasses and enums only.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════
# Geometry — the vital signs of inference
# (Lyra Technique spectral features)
# ═══════════════════════════════════════════


@dataclass
class GeometrySummary:
    """Compressed geometry reading from a KV cache snapshot.

    These are the spectral features the Lyra Technique extracts:
    SVD of the key cache matrix → singular values → derived metrics.

    The key insight: these metrics encode cognitive state. Different
    modes of reasoning (factual, creative, confabulating) produce
    geometrically distinct spectral signatures in the cache.
    """
    key_norm: float = 0.0           # Frobenius norm of key cache
    norm_per_token: float = 0.0     # Normalized by sequence length
    effective_rank: float = 0.0     # Dims for 90% cumulative variance
    spectral_entropy: float = 0.0   # Shannon entropy of σ² distribution
    top_sv_ratio: float = 0.0       # σ₁ / Σσᵢ — dominance of first axis
    angular_spread: float = 0.0     # Geometric spread across layers
    n_layers: int = 0
    n_tokens: int = 0
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════
# Cache state — the KV cache snapshot
# ═══════════════════════════════════════════


@dataclass
class CacheState:
    """A snapshot of the KV cache at a point in time.

    Backend-agnostic: cache_data can be torch tensors, numpy arrays,
    or raw bytes depending on the backend. The memory architecture
    doesn't interpret the raw data — it stores, labels, and restores.

    The geometry field holds the spectral features extracted from
    the cache at snapshot time. This is what makes cache recording
    useful: you can track how the geometry evolves over a conversation.
    """
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    label: str = ""                  # "turn_0", "turn_5", "post_topic_shift", etc.
    cache_data: Any = None           # Backend-specific tensor data
    geometry: Optional[GeometrySummary] = None
    n_layers: int = 0
    n_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════
# Journal types — event log
# ═══════════════════════════════════════════


class JournalEventType:
    """Event types for the memory journal."""
    CACHE_SNAPSHOT = "cache_snapshot"
    GEOMETRY_EXTRACTED = "geometry_extracted"
    MEMORY_STORED = "memory_stored"
    MEMORY_LINKED = "memory_linked"
    CONSOLIDATION_RUN = "consolidation_run"
    CUSTOM = "custom"


@dataclass
class JournalEntry:
    """A single event in the memory journal."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    event: str = ""
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    @classmethod
    def create(cls, event: str, **data) -> JournalEntry:
        return cls(event=event, data=data)


# ═══════════════════════════════════════════
# Consolidated memory types
# ═══════════════════════════════════════════


@dataclass
class MemoryLink:
    """Directional link between two memories."""
    source_id: str
    target_id: str
    relationship: str       # "updates", "extends", "contradicts", "supersedes"
    strength: float = 1.0   # Decays over time
    created_at: float = field(default_factory=time.time)


@dataclass
class ConsolidatedMemory:
    """A single entry in long-term consolidated memory.

    Richer than a journal entry — has embeddings, links, and decay.
    """
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str = ""
    memory_type: str = "observation"  # observation, pattern, drift, geometry
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None  # 768-dim semantic embedding
    geometry_summary: Optional[GeometrySummary] = None
    links: list[MemoryLink] = field(default_factory=list)
    significance: float = 0.5       # 0.0-1.0, affects retention
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsolidationReport:
    """Output of a consolidation pass."""
    window_hours: float
    total_snapshots: int
    new_memories_created: int
    links_created: int
    entries_compressed: int
    geometry_drift_score: float     # 0.0 = stable, 1.0 = major drift
    drift_direction: str
    dominant_patterns: list[str]
    timestamp: float = field(default_factory=time.time)
