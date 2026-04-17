"""Core types for the memory architecture.

Portable — no framework dependencies. Uses dataclasses and enums only.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ═══════════════════════════════════════════
# Geometry — the vital signs of inference
# ═══════════════════════════════════════════


@dataclass
class GeometrySummary:
    """Compressed geometry reading from a KV cache snapshot.

    These are the spectral features the Lyra Technique extracts:
    SVD of the key cache matrix → singular values → derived metrics.
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
# Cache state — Tier 1 (Activation Memory)
# ═══════════════════════════════════════════


@dataclass
class CacheState:
    """A snapshot of the KV cache at a point in time.

    Backend-agnostic: cache_data can be torch tensors, numpy arrays,
    or raw bytes depending on the backend. The memory architecture
    doesn't interpret them — it stores and restores.
    """
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    label: str = ""                  # "encoding", "post_inference_attempt_0", etc.
    cache_data: Any = None           # Backend-specific tensor data
    geometry: Optional[GeometrySummary] = None
    n_layers: int = 0
    n_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════
# Transaction types — Tier 2 (Transaction Memory)
# ═══════════════════════════════════════════


class TransactionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    CHECKING = "checking"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    STEERING = "steering"
    RETRYING = "retrying"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self in (TransactionStatus.COMMITTED, TransactionStatus.ABORTED)


class AlignmentVerdict(Enum):
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    UNCERTAIN = "uncertain"


@dataclass
class AlignmentCheck:
    verdict: AlignmentVerdict
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    dominant_emotion: str = ""
    alignment_risk: float = 0.0
    suggested_steering: Optional[SteeringAction] = None


@dataclass
class SteeringAction:
    vectors: dict[str, float] = field(default_factory=dict)  # e.g. {"calm": 1.0}
    reason: str = ""
    strength: float = 1.0


@dataclass
class InferenceTransaction:
    tx_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: TransactionStatus = TransactionStatus.PENDING
    input_text: str = ""
    output_text: Optional[str] = None
    output_tokens: Any = None
    checkpoints: list[CacheState] = field(default_factory=list)
    geometry_readings: list[GeometrySummary] = field(default_factory=list)
    alignment_checks: list[AlignmentCheck] = field(default_factory=list)
    steering_applied: list[SteeringAction] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    committed_at: Optional[float] = None
    aborted_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def duration(self) -> float:
        end = self.committed_at or self.aborted_at or time.time()
        return end - self.created_at

    @classmethod
    def create(cls, input_text: str, max_retries: int = 3) -> InferenceTransaction:
        return cls(input_text=input_text, max_retries=max_retries)


# ═══════════════════════════════════════════
# Journal types — event log
# ═══════════════════════════════════════════


class JournalEventType(Enum):
    TX_BEGIN = "tx_begin"
    CHECKPOINT_CREATED = "checkpoint_created"
    INFERENCE_START = "inference_start"
    INFERENCE_COMPLETE = "inference_complete"
    GEOMETRY_EXTRACTED = "geometry_extracted"
    EMOTION_DETECTED = "emotion_detected"
    ALIGNMENT_CHECK = "alignment_check"
    STEERING_APPLIED = "steering_applied"
    ROLLBACK = "rollback"
    RETRY = "retry"
    COMMIT = "commit"
    ABORT = "abort"


@dataclass
class JournalEntry:
    tx_id: str
    event: JournalEventType
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    @classmethod
    def create(cls, tx_id: str, event: JournalEventType, **data) -> JournalEntry:
        return cls(tx_id=tx_id, event=event, data=data)


# ═══════════════════════════════════════════
# Consolidated memory types — Tier 3
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
    memory_type: str = "observation"  # observation, decision, pattern, drift
    tags: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None  # 768-dim semantic embedding
    geometry_summary: Optional[GeometrySummary] = None
    links: list[MemoryLink] = field(default_factory=list)
    significance: float = 0.5       # 0.0-1.0, affects retention
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    source_tx_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ConsolidationReport:
    """Output of a sleep-time consolidation pass."""
    window_hours: float
    total_transactions: int
    committed_count: int
    aborted_count: int
    avg_retries: float
    avg_risk: float
    abort_rate: float
    dominant_emotions: list[str]
    drift_score: float              # 0.0 = stable, 1.0 = major drift
    drift_direction: str            # What's drifting
    new_memories_created: int
    links_created: int
    entries_compressed: int
    timestamp: float = field(default_factory=time.time)
