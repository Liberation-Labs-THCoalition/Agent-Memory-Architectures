"""Tier 3: Consolidated Memory — longitudinal store with retroactive linking.

This is the new tier that doesn't exist in the original Oracle Harness.
It implements three SOTA patterns:

1. A-MEM retroactive linking (NeurIPS 2025):
   New memories trigger updates to related historical memories.
   The memory network continuously refines its own representations.

2. MemOS three-tier integration:
   Consolidated memory sits alongside activation memory (KV cache)
   and transaction memory (journal). Each tier serves a different
   temporal scale.

3. LightMem sleep-time consolidation:
   Periodic analysis decoupled from inference. Compresses old
   journal entries, detects drift, and surfaces longitudinal patterns.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional, Callable

from oracle_memory.src.types import (
    ConsolidatedMemory,
    ConsolidationReport,
    MemoryLink,
)

logger = logging.getLogger(__name__)


class ConsolidatedStore:
    """Long-term memory with retroactive linking and consolidation.

    This is the "slow thinking" layer. While the journal records
    everything in real-time, the consolidated store distills
    patterns over time.

    Usage:
        store = ConsolidatedStore(persist_dir="/path/to/memories")

        # Store a new memory with automatic back-linking
        mem = ConsolidatedMemory(
            content="Abort rate spiked during coding prompts",
            memory_type="pattern",
            tags=["drift", "coding"],
        )
        links = store.store(mem)  # Returns IDs of linked memories

        # Search
        results = store.search("coding prompt alignment", top_k=5)

        # Sleep-time consolidation
        report = store.consolidate(journal, window_hours=24)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        similarity_threshold: float = 0.7,
    ) -> None:
        self._memories: dict[str, ConsolidatedMemory] = {}
        self._persist_dir: Optional[Path] = None
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold

        if persist_dir:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ── Store with retroactive linking ──

    def store(self, memory: ConsolidatedMemory) -> list[str]:
        """Store a memory and create retroactive links to related memories.

        This is the A-MEM innovation: storing isn't just appending.
        New memories update the network by linking to (and being
        linked from) related historical memories.

        Returns IDs of memories that were linked.
        """
        # Generate embedding if we have an embed function and content
        if self._embed_fn and memory.content and not memory.embedding:
            memory.embedding = self._embed_fn(memory.content)

        # Find related memories and create bidirectional links
        linked_ids = []
        if memory.embedding:
            similar = self._find_similar(memory.embedding, top_k=5)
            for old_mem, similarity in similar:
                if similarity >= self._similarity_threshold:
                    # Forward link: new → old
                    memory.links.append(MemoryLink(
                        source_id=memory.memory_id,
                        target_id=old_mem.memory_id,
                        relationship="extends",
                        strength=similarity,
                    ))
                    # Backward link: old → new (retroactive update)
                    old_mem.links.append(MemoryLink(
                        source_id=old_mem.memory_id,
                        target_id=memory.memory_id,
                        relationship="extended_by",
                        strength=similarity,
                    ))
                    linked_ids.append(old_mem.memory_id)
                    self._persist_memory(old_mem)  # Save updated old memory

        self._memories[memory.memory_id] = memory
        self._persist_memory(memory)

        logger.info(
            "Stored memory %s (%s) with %d links",
            memory.memory_id[:8], memory.memory_type, len(linked_ids),
        )
        return linked_ids

    def get(self, memory_id: str) -> Optional[ConsolidatedMemory]:
        """Retrieve a memory by ID. Updates access tracking."""
        mem = self._memories.get(memory_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = time.time()
        return mem

    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[ConsolidatedMemory]:
        """Search memories by semantic similarity and/or filters."""
        candidates = list(self._memories.values())

        # Filter by type
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]

        # Filter by tags
        if tags:
            tag_set = set(tags)
            candidates = [m for m in candidates if tag_set & set(m.tags)]

        # Semantic ranking (if embed function available)
        if self._embed_fn and query:
            query_emb = self._embed_fn(query)
            scored = []
            for mem in candidates:
                if mem.embedding:
                    sim = self._cosine_similarity(query_emb, mem.embedding)
                    scored.append((mem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in scored[:top_k]]

        # Fallback: sort by recency
        candidates.sort(key=lambda m: m.created_at, reverse=True)
        return candidates[:top_k]

    def get_linked(self, memory_id: str) -> list[tuple[ConsolidatedMemory, str]]:
        """Get all memories linked to a given memory.

        Returns list of (memory, relationship) tuples.
        """
        mem = self._memories.get(memory_id)
        if not mem:
            return []

        results = []
        for link in mem.links:
            target = self._memories.get(link.target_id)
            if target:
                results.append((target, link.relationship))
        return results

    # ── Sleep-time consolidation ──

    def consolidate(
        self,
        journal,  # MemoryJournal
        window_hours: float = 24.0,
    ) -> ConsolidationReport:
        """Run sleep-time consolidation over geometry history.

        Decoupled from inference — run periodically. Analyzes
        geometry readings over a time window to detect drift in
        the cache's spectral structure.
        """
        geo_history = journal.get_geometry_history(last_n=200)
        if not geo_history:
            return ConsolidationReport(
                window_hours=window_hours,
                total_snapshots=0,
                new_memories_created=0, links_created=0, entries_compressed=0,
                geometry_drift_score=0, drift_direction="stable",
                dominant_patterns=[],
            )

        total = len(geo_history)
        new_memories = 0
        total_links = 0

        # Detect geometry drift: is effective rank changing over time?
        ranks = [g.get("effective_rank", 0) for g in geo_history if g.get("effective_rank")]
        if len(ranks) >= 10:
            first_half = ranks[:len(ranks)//2]
            second_half = ranks[len(ranks)//2:]
            mean_first = sum(first_half) / len(first_half)
            mean_second = sum(second_half) / len(second_half)
            rank_shift = abs(mean_second - mean_first) / max(mean_first, 1.0)
            drift_score = min(rank_shift * 5, 1.0)  # Scale to 0-1
            drift_direction = (
                "rank_expanding" if mean_second > mean_first * 1.05
                else "rank_contracting" if mean_second < mean_first * 0.95
                else "stable"
            )
        else:
            drift_score = 0.0
            drift_direction = "insufficient_data"

        # Create memory for significant geometry shifts
        if drift_score > 0.3:
            mem = ConsolidatedMemory(
                content=f"Geometry drift detected (score={drift_score:.2f}, "
                        f"direction={drift_direction}). Cache effective rank "
                        f"shifted over {window_hours}h window.",
                memory_type="drift",
                tags=["geometry", "drift", "consolidation"],
                significance=drift_score,
            )
            links = self.store(mem)
            new_memories += 1
            total_links += len(links)

        # Detect spectral entropy patterns
        entropies = [g.get("spectral_entropy", 0) for g in geo_history if g.get("spectral_entropy")]
        patterns = []
        if entropies:
            mean_ent = sum(entropies) / len(entropies)
            if mean_ent > 22.0:
                patterns.append("high_entropy")
            if any(e > mean_ent * 1.3 for e in entropies[-10:]):
                patterns.append("recent_entropy_spike")

        return ConsolidationReport(
            window_hours=window_hours,
            total_snapshots=total,
            new_memories_created=new_memories,
            links_created=total_links,
            entries_compressed=0,
            geometry_drift_score=drift_score,
            drift_direction=drift_direction,
            dominant_patterns=patterns,
        )

    # ── Internal helpers ──

    def _find_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[ConsolidatedMemory, float]]:
        """Find memories most similar to the given embedding."""
        scored = []
        for mem in self._memories.values():
            if mem.embedding:
                sim = self._cosine_similarity(embedding, mem.embedding)
                scored.append((mem, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Persistence ──

    def _persist_memory(self, memory: ConsolidatedMemory) -> None:
        """Save a single memory to disk as JSON."""
        if not self._persist_dir:
            return
        try:
            path = self._persist_dir / f"{memory.memory_id}.json"
            data = {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "tags": memory.tags,
                "significance": memory.significance,
                "access_count": memory.access_count,
                "last_accessed": memory.last_accessed,
                "created_at": memory.created_at,
                "source_tx_id": memory.source_tx_id,
                "metadata": memory.metadata,
                "links": [
                    {
                        "source_id": l.source_id,
                        "target_id": l.target_id,
                        "relationship": l.relationship,
                        "strength": l.strength,
                        "created_at": l.created_at,
                    }
                    for l in memory.links
                ],
            }
            # Don't persist embedding (large, regeneratable)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Memory persist failed for %s: %s", memory.memory_id[:8], e)

    def _load_from_disk(self) -> None:
        """Load all memories from persist directory."""
        if not self._persist_dir:
            return
        for path in self._persist_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                mem = ConsolidatedMemory(
                    memory_id=data["memory_id"],
                    content=data["content"],
                    memory_type=data.get("memory_type", "observation"),
                    tags=data.get("tags", []),
                    significance=data.get("significance", 0.5),
                    access_count=data.get("access_count", 0),
                    last_accessed=data.get("last_accessed", 0),
                    created_at=data.get("created_at", 0),
                    source_tx_id=data.get("source_tx_id"),
                    metadata=data.get("metadata", {}),
                    links=[
                        MemoryLink(
                            source_id=l["source_id"],
                            target_id=l["target_id"],
                            relationship=l["relationship"],
                            strength=l.get("strength", 1.0),
                            created_at=l.get("created_at", 0),
                        )
                        for l in data.get("links", [])
                    ],
                )
                self._memories[mem.memory_id] = mem
            except Exception as e:
                logger.warning("Failed to load memory from %s: %s", path, e)

        logger.info("Loaded %d consolidated memories from disk", len(self._memories))

    def __len__(self) -> int:
        return len(self._memories)
