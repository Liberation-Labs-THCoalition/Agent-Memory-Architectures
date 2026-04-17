"""Oracle Memory — three-tier cognitive memory for AI agents.

Tier 1: Activation Memory — KV cache snapshots (CheckpointManager)
Tier 2: Transaction Memory — ACID event journal (TransactionJournal)
Tier 3: Consolidated Memory — longitudinal store with retroactive linking

Ported from Oracle Harness Layer 4 with SOTA upgrades.
"""
from oracle_memory.src.types import (
    CacheState, GeometrySummary, InferenceTransaction, TransactionStatus,
    AlignmentVerdict, AlignmentCheck, SteeringAction,
    JournalEntry, JournalEventType,
    ConsolidatedMemory, ConsolidationReport, MemoryLink,
)
from oracle_memory.src.checkpoint import CheckpointManager
from oracle_memory.src.journal import TransactionJournal
from oracle_memory.src.consolidated import ConsolidatedStore
