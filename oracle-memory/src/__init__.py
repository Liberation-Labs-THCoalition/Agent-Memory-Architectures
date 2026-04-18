"""Oracle Memory — KV cache recording with Lyra Technique geometry.

A memory architecture that captures the geometric structure of the
KV cache during inference. The cache isn't just a performance
shortcut — its spectral features encode cognitive state.

Components:
  CheckpointManager — snapshot/restore/label KV cache states
  MemoryJournal — event log for cache recordings and geometry readings
  ConsolidatedStore — long-term memory with retroactive linking
"""
from oracle_memory.src.types import (
    CacheState, GeometrySummary,
    JournalEntry, JournalEventType,
    ConsolidatedMemory, ConsolidationReport, MemoryLink,
)
from oracle_memory.src.checkpoint import CheckpointManager
from oracle_memory.src.journal import MemoryJournal
from oracle_memory.src.consolidated import ConsolidatedStore
