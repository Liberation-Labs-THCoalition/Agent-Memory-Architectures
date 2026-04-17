"""Tier 1: Activation Memory — KV cache checkpoint manager.

Manages ordered snapshots of KV cache state. Think of it as git
for cognitive state: you can snapshot, label, restore, and prune.

Ported from Oracle Harness Layer 4 (CheckpointManager).
The KV cache recording innovation is the foundation of the Lyra
Technique — the geometry of these snapshots is the signal source
for alignment monitoring.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

from oracle_memory.src.types import CacheState

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Ordered KV cache state snapshots — git commits for cognitive state.

    Usage:
        mgr = CheckpointManager(max_checkpoints=10)

        # Snapshot after encoding
        encoding_state = backend.snapshot_cache()
        encoding_state.label = "encoding"
        mgr.create(encoding_state)

        # Snapshot after inference
        post_state = backend.snapshot_cache()
        post_state.label = "post_inference_attempt_0"
        mgr.create(post_state)

        # Rollback
        baseline = mgr.get_by_label("encoding")
        backend.restore_cache(baseline)

        # Prune old snapshots
        mgr.prune(keep_last=2)
    """

    def __init__(self, max_checkpoints: int = 10) -> None:
        self._checkpoints: OrderedDict[str, CacheState] = OrderedDict()
        self._max_checkpoints = max_checkpoints

    def create(self, cache: CacheState) -> str:
        """Store a checkpoint. Returns its snapshot_id."""
        if len(self._checkpoints) >= self._max_checkpoints:
            # Evict oldest, but never the initial checkpoint
            keys = list(self._checkpoints.keys())
            for key in keys:
                if key != keys[0]:  # Protect initial
                    del self._checkpoints[key]
                    break
            else:
                # All checkpoints are the initial — evict oldest anyway
                if len(keys) > 1:
                    del self._checkpoints[keys[0]]

        self._checkpoints[cache.snapshot_id] = cache
        logger.debug("Checkpoint created: %s (%s)", cache.snapshot_id[:8], cache.label)
        return cache.snapshot_id

    def get(self, checkpoint_id: str) -> Optional[CacheState]:
        """Retrieve a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def get_by_label(self, label: str) -> Optional[CacheState]:
        """Get most recent checkpoint with the given label."""
        for cache in reversed(self._checkpoints.values()):
            if cache.label == label:
                return cache
        return None

    @property
    def latest(self) -> Optional[CacheState]:
        """Most recently created checkpoint."""
        if self._checkpoints:
            return list(self._checkpoints.values())[-1]
        return None

    @property
    def initial(self) -> Optional[CacheState]:
        """First checkpoint (clean baseline)."""
        if self._checkpoints:
            return list(self._checkpoints.values())[0]
        return None

    def list_checkpoints(self) -> list[tuple[str, str, float]]:
        """List all checkpoints as (id, label, timestamp)."""
        return [
            (c.snapshot_id, c.label, c.timestamp)
            for c in self._checkpoints.values()
        ]

    def prune(self, keep_last: int = 2) -> int:
        """Remove old checkpoints, keeping the N most recent.

        Returns the number of checkpoints removed.
        """
        if len(self._checkpoints) <= keep_last:
            return 0

        keys = list(self._checkpoints.keys())
        to_remove = keys[:-keep_last] if keep_last > 0 else keys
        for key in to_remove:
            del self._checkpoints[key]

        logger.debug("Pruned %d checkpoints, %d remaining", len(to_remove), len(self._checkpoints))
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._checkpoints)
