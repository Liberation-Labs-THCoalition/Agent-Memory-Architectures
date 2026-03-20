"""Dream-inspired memory consolidation with significance-aware decay.

Four-phase cycle inherited from Agentic-memory-service:
  1. Decay scoring — significance x recency x access frequency
  2. Association discovery — co-occurring entities flagged (Phase 3: triples)
  3. Compression — similar memories merged, originals archived
  4. Archival — below-threshold memories moved out of active index

Schedule: daily (light), weekly (full), monthly (deep compression + entity dedup).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from ..models import Memory, MemoryStatus, TTLClass

logger = logging.getLogger(__name__)

# Half-life in days for each TTL class
HALF_LIVES = {
    TTLClass.PERMANENT: None,   # No decay
    TTLClass.LONG: 90,
    TTLClass.MEDIUM: 30,
    TTLClass.SHORT: 7,
}


class Consolidator:
    """Significance-aware memory consolidation engine."""

    def __init__(
        self,
        archive_threshold: float = 0.2,
        forget_threshold: float = 0.1,
    ):
        self.archive_threshold = archive_threshold
        self.forget_threshold = forget_threshold

    def compute_decay_score(self, memory: Memory, now: Optional[datetime] = None) -> float:
        """Compute the current decay score for a memory.

        decay_score = significance x recency_weight(last_accessed) x log(1 + access_count)

        Returns a float where higher = more alive, lower = more decayed.
        """
        now = now or datetime.utcnow()

        # Permanent memories don't decay
        if memory.ttl_class == TTLClass.PERMANENT:
            return memory.significance * max(1.0, math.log(1 + memory.access_count))

        half_life_days = HALF_LIVES.get(memory.ttl_class, 30)
        if half_life_days is None:
            return memory.significance

        # Compute recency weight as exponential decay
        days_since_access = (now - memory.last_accessed).total_seconds() / 86400
        recency_weight = 0.5 ** (days_since_access / half_life_days)

        # Access frequency boost (logarithmic to prevent gaming)
        access_boost = math.log(1 + memory.access_count)

        # Combined score
        score = memory.significance * recency_weight * max(1.0, access_boost)

        return round(score, 4)

    def run_decay_pass(self, memories: list[Memory]) -> dict[str, list[Memory]]:
        """Phase 1: Score all memories and classify by decay state.

        Returns dict with keys: 'active', 'archive', 'forget'
        """
        now = datetime.utcnow()
        result: dict[str, list[Memory]] = {
            "active": [],
            "archive": [],
            "forget": [],
        }

        for memory in memories:
            if memory.status in (MemoryStatus.ARCHIVED, MemoryStatus.FORGOTTEN):
                continue

            score = self.compute_decay_score(memory, now)

            if score < self.forget_threshold:
                memory.status = MemoryStatus.FORGOTTEN
                result["forget"].append(memory)
                logger.debug(f"Memory {memory.id} scored {score:.4f} → FORGET")
            elif score < self.archive_threshold:
                memory.status = MemoryStatus.ARCHIVED
                result["archive"].append(memory)
                logger.debug(f"Memory {memory.id} scored {score:.4f} → ARCHIVE")
            else:
                result["active"].append(memory)

        logger.info(
            f"Decay pass: {len(result['active'])} active, "
            f"{len(result['archive'])} to archive, "
            f"{len(result['forget'])} to forget"
        )
        return result

    def discover_associations(self, memories: list[Memory]) -> list[tuple[str, str, list[str]]]:
        """Phase 2: Find memories with overlapping entities.

        Returns list of (memory_id_a, memory_id_b, shared_entities) tuples.
        In Phase 3, these will be promoted to KG triples.
        """
        # Build entity → memory_id index
        entity_index: dict[str, list[str]] = defaultdict(list)
        for memory in memories:
            for entity in memory.entities:
                entity_index[entity.name_lower].append(memory.id)

        # Find co-occurring pairs
        associations = []
        seen_pairs: set[tuple[str, str]] = set()

        for entity_name, memory_ids in entity_index.items():
            if len(memory_ids) < 2:
                continue
            for i in range(len(memory_ids)):
                for j in range(i + 1, len(memory_ids)):
                    pair = tuple(sorted([memory_ids[i], memory_ids[j]]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        # Find all shared entities for this pair
                        shared = self._shared_entities(
                            memory_ids[i], memory_ids[j], memories
                        )
                        if shared:
                            associations.append((pair[0], pair[1], shared))

        logger.info(f"Association discovery: found {len(associations)} linked pairs")
        return associations

    def identify_compression_candidates(
        self, memories: list[Memory], similarity_threshold: float = 0.7
    ) -> list[list[Memory]]:
        """Phase 3: Find clusters of similar archived memories to compress.

        Groups archived memories with overlapping entities and tags.
        In a full implementation, this would also use embedding similarity.
        """
        archived = [m for m in memories if m.status == MemoryStatus.ARCHIVED]
        if len(archived) < 2:
            return []

        clusters: list[list[Memory]] = []
        used: set[str] = set()

        for i, mem_a in enumerate(archived):
            if mem_a.id in used:
                continue

            cluster = [mem_a]
            used.add(mem_a.id)

            for j in range(i + 1, len(archived)):
                mem_b = archived[j]
                if mem_b.id in used:
                    continue

                overlap = self._entity_overlap(mem_a, mem_b)
                tag_overlap = len(set(mem_a.tags) & set(mem_b.tags))

                if overlap >= 2 or (overlap >= 1 and tag_overlap >= 1):
                    cluster.append(mem_b)
                    used.add(mem_b.id)

            if len(cluster) >= 2:
                clusters.append(cluster)

        logger.info(
            f"Compression: identified {len(clusters)} clusters "
            f"({sum(len(c) for c in clusters)} memories total)"
        )
        return clusters

    def consolidate(
        self,
        memories: list[Memory],
        mode: str = "full",
    ) -> dict:
        """Run a full consolidation cycle.

        Args:
            memories: All memories to process.
            mode: 'light' (decay only), 'full' (decay + associations + compression),
                  'deep' (full + entity dedup).

        Returns summary of actions taken.
        """
        summary = {"mode": mode, "actions": []}

        # Phase 1: Always run decay
        decay_result = self.run_decay_pass(memories)
        summary["decay"] = {
            "active": len(decay_result["active"]),
            "to_archive": len(decay_result["archive"]),
            "to_forget": len(decay_result["forget"]),
        }
        summary["actions"].append("decay_scoring")

        if mode in ("full", "deep"):
            # Phase 2: Association discovery
            associations = self.discover_associations(decay_result["active"])
            summary["associations"] = len(associations)
            summary["actions"].append("association_discovery")

            # Phase 3: Compression candidates
            all_memories = decay_result["active"] + decay_result["archive"]
            clusters = self.identify_compression_candidates(all_memories)
            summary["compression_clusters"] = len(clusters)
            summary["memories_compressible"] = sum(len(c) for c in clusters)
            summary["actions"].append("compression")

        if mode == "deep":
            # Phase 4: Entity dedup (placeholder for Phase 3 KG)
            summary["actions"].append("entity_dedup")

        # Phase 4: Archival summary
        summary["actions"].append("archival")

        logger.info(f"Consolidation complete ({mode}): {summary}")
        return summary

    @staticmethod
    def _shared_entities(
        id_a: str, id_b: str, memories: list[Memory]
    ) -> list[str]:
        """Find shared entity names between two memories."""
        mem_a = next((m for m in memories if m.id == id_a), None)
        mem_b = next((m for m in memories if m.id == id_b), None)
        if not mem_a or not mem_b:
            return []

        names_a = {e.name_lower for e in mem_a.entities}
        names_b = {e.name_lower for e in mem_b.entities}
        return list(names_a & names_b)

    @staticmethod
    def _entity_overlap(a: Memory, b: Memory) -> int:
        """Count shared entities between two memories."""
        names_a = {e.name_lower for e in a.entities}
        names_b = {e.name_lower for e in b.entities}
        return len(names_a & names_b)
