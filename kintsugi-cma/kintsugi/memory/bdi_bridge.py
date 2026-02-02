"""
Bridge between BDI (Beliefs-Desires-Intentions) architecture and Kintsugi memory.

The BDI model provides a cognitive architecture where:
    - **Beliefs** represent what the organization knows to be true, derived from
      high-significance memories that have been corroborated or reinforced.
    - **Desires** represent organizational values and goals that shape which
      memories are retained, boosted, or allowed to decay.
    - **Intentions** represent active plans/goals that influence which memories
      are prioritized during retrieval.

This bridge does NOT store BDI state itself — it translates between raw memory
records and the BDI abstractions, enabling higher-level reasoning systems to
work with structured cognitive primitives.

Usage:
    bridge = BDIBridge()

    # Extract beliefs from memory search results
    beliefs = bridge.extract_beliefs(memories, min_significance=7)

    # Bias search results toward organizational desires
    biased = bridge.apply_desire_bias(memories, active_desires)

    # Prioritize results based on active intentions
    ranked = bridge.prioritize_by_intentions(biased, active_intentions)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """An organizational belief derived from high-significance memories.

    Beliefs are the distilled knowledge an organization holds as true.
    Confidence reflects how well-supported the belief is (number of
    corroborating memories, their significance, recency).

    Attributes:
        id: Deterministic hash derived from content for deduplication.
        content: The belief statement in natural language.
        confidence: Confidence score between 0.0 and 1.0.
        source_memory_ids: IDs of memories that support this belief.
        tags: Inherited tags from source memories.
    """
    id: str
    content: str
    confidence: float
    source_memory_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")


@dataclass
class Desire:
    """An organizational desire or value that shapes memory retention.

    Desires influence which memories are boosted (aligned with values)
    and which are allowed to decay (misaligned or irrelevant).

    Attributes:
        id: Unique identifier for this desire.
        description: Human-readable description of the desire/value.
        priority: Priority weight between 0.0 and 1.0.
        related_tags: Tags that signal alignment with this desire.
        boost_factor: Multiplier applied to significance of aligned memories.
        decay_factor: Multiplier applied to significance of unaligned memories.
    """
    id: str
    description: str
    priority: float
    related_tags: list[str] = field(default_factory=list)
    boost_factor: float = 1.3
    decay_factor: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be 0-1, got {self.priority}")


@dataclass
class Intention:
    """An active organizational intention (goal/plan) that guides retrieval.

    Intentions connect beliefs (what we know) to desires (what we want)
    and represent committed courses of action.

    Attributes:
        id: Unique identifier for this intention.
        goal: Description of the intended outcome.
        status: One of 'active', 'completed', 'suspended'.
        belief_ids: Beliefs that justify this intention.
        desire_ids: Desires this intention aims to fulfill.
        priority_boost: Additional score boost for memories relevant to this intention.
    """
    id: str
    goal: str
    status: str
    belief_ids: list[str] = field(default_factory=list)
    desire_ids: list[str] = field(default_factory=list)
    priority_boost: float = 0.2

    VALID_STATUSES = ("active", "completed", "suspended")

    def __post_init__(self) -> None:
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"status must be one of {self.VALID_STATUSES}, got {self.status!r}"
            )


# ---------------------------------------------------------------------------
# BDI Bridge
# ---------------------------------------------------------------------------

class BDIBridge:
    """Translates between raw memory records and BDI cognitive primitives.

    This bridge is stateless — it operates on memory dicts as returned by
    OrgMemoryStore.hybrid_search() and produces BDI objects or re-ranked
    memory lists.

    Memory dict expected shape:
        {
            "id": str,
            "content": str,
            "significance": int,
            "memory_layer": str,
            "tags": list[str],
            "metadata": dict,
            "created_at": datetime | str,
            "score": float | None,
        }
    """

    # -- Belief extraction ---------------------------------------------------

    def extract_beliefs(
        self,
        memories: list[dict[str, Any]],
        min_significance: int = 2,
    ) -> list[Belief]:
        """Extract beliefs from memories that meet the significance threshold.

        Memories above the threshold are treated as belief candidates.
        Confidence is computed from significance (normalized to 0-1) weighted
        by the number of corroborating memories with overlapping content.

        Args:
            memories: List of memory dicts from the store.
            min_significance: Minimum significance to consider (inclusive).

        Returns:
            List of Belief objects, sorted by confidence descending.
        """
        candidates = [
            m for m in memories
            if m.get("significance", 0) >= min_significance
        ]

        if not candidates:
            return []

        beliefs: list[Belief] = []
        seen_hashes: set[str] = set()

        for mem in candidates:
            content = mem.get("content", "")
            content_hash = self._content_hash(content)

            if content_hash in seen_hashes:
                # Merge into existing belief as corroboration
                for b in beliefs:
                    if b.id == content_hash:
                        b.source_memory_ids.append(mem["id"])
                        b.confidence = min(1.0, b.confidence + 0.05)
                        break
                continue

            seen_hashes.add(content_hash)

            significance = mem.get("significance", 5)
            base_confidence = significance / 10.0

            all_tags = list(set(mem.get("tags", [])))

            beliefs.append(Belief(
                id=content_hash,
                content=content,
                confidence=round(base_confidence, 3),
                source_memory_ids=[mem["id"]],
                tags=all_tags,
            ))

        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs

    # -- Desire-based bias ---------------------------------------------------

    def apply_desire_bias(
        self,
        memories: list[dict[str, Any]],
        desires: list[Desire],
    ) -> list[dict[str, Any]]:
        """Adjust memory scores based on alignment with organizational desires.

        Memories whose tags overlap with a desire's related_tags get their
        score boosted by the desire's boost_factor (weighted by priority).
        Memories with no desire alignment get a mild decay.

        This produces a re-scored copy of the memory list — originals are
        not mutated.

        Args:
            memories: List of memory dicts (must have 'score' and 'tags').
            desires: Active organizational desires.

        Returns:
            New list of memory dicts with adjusted scores, sorted descending.
        """
        if not desires:
            return list(memories)

        # Build tag -> weighted boost mapping
        tag_boosts: dict[str, float] = {}
        for desire in desires:
            for tag in desire.related_tags:
                existing = tag_boosts.get(tag, 1.0)
                weighted_boost = 1.0 + (desire.boost_factor - 1.0) * desire.priority
                tag_boosts[tag] = max(existing, weighted_boost)

        # Compute aggregate decay for unaligned memories
        avg_decay = sum(d.decay_factor * d.priority for d in desires) / sum(
            d.priority for d in desires
        ) if any(d.priority > 0 for d in desires) else 1.0

        results: list[dict[str, Any]] = []
        for mem in memories:
            adjusted = dict(mem)
            base_score = mem.get("score") or 0.0
            mem_tags = set(mem.get("tags", []))

            # Find best boost from overlapping tags
            best_boost = 1.0
            aligned = False
            for tag in mem_tags:
                if tag in tag_boosts:
                    aligned = True
                    best_boost = max(best_boost, tag_boosts[tag])

            if aligned:
                adjusted["score"] = round(base_score * best_boost, 6)
                adjusted["_desire_aligned"] = True
            else:
                adjusted["score"] = round(base_score * avg_decay, 6)
                adjusted["_desire_aligned"] = False

            results.append(adjusted)

        results.sort(key=lambda m: m.get("score", 0), reverse=True)
        return results

    # -- Intention-based prioritization --------------------------------------

    def prioritize_by_intentions(
        self,
        results: list[dict[str, Any]],
        intentions: list[Intention],
    ) -> list[dict[str, Any]]:
        """Re-rank search results based on active intentions.

        Memories that are referenced by an intention's supporting beliefs
        get a score boost proportional to the intention's priority_boost.
        Only active intentions are considered.

        Args:
            results: Memory dicts (typically output of apply_desire_bias).
            intentions: All intentions (only 'active' ones are used).

        Returns:
            New list of memory dicts re-ranked by intention relevance.
        """
        active = [i for i in intentions if i.status == "active"]
        if not active:
            return list(results)

        # Collect all belief IDs referenced by active intentions
        relevant_belief_ids: set[str] = set()
        total_boost = 0.0
        belief_boost_map: dict[str, float] = {}

        for intention in active:
            for bid in intention.belief_ids:
                relevant_belief_ids.add(bid)
                current = belief_boost_map.get(bid, 0.0)
                belief_boost_map[bid] = current + intention.priority_boost

        # Boost memories whose IDs appear as source memories for relevant beliefs
        # or whose content hash matches a belief ID
        output: list[dict[str, Any]] = []
        for mem in results:
            adjusted = dict(mem)
            base_score = mem.get("score") or 0.0
            mem_id = mem.get("id", "")
            content_hash = self._content_hash(mem.get("content", ""))

            boost = 0.0
            # Check if this memory's content hash matches a relevant belief
            if content_hash in belief_boost_map:
                boost += belief_boost_map[content_hash]
            # Check if memory ID is directly referenced
            if mem_id in relevant_belief_ids:
                boost += max(i.priority_boost for i in active)

            if boost > 0:
                adjusted["score"] = round(base_score + boost, 6)
                adjusted["_intention_boosted"] = True
            else:
                adjusted["_intention_boosted"] = False

            output.append(adjusted)

        output.sort(key=lambda m: m.get("score", 0), reverse=True)
        return output

    # -- Convenience ---------------------------------------------------------

    def process_pipeline(
        self,
        memories: list[dict[str, Any]],
        desires: list[Desire],
        intentions: list[Intention],
        min_belief_significance: int = 2,
    ) -> tuple[list[Belief], list[dict[str, Any]]]:
        """Run the full BDI pipeline: extract beliefs, bias, prioritize.

        Convenience method that chains extract_beliefs -> apply_desire_bias
        -> prioritize_by_intentions.

        Args:
            memories: Raw memory dicts from search.
            desires: Active organizational desires.
            intentions: All intentions.
            min_belief_significance: Threshold for belief extraction.

        Returns:
            Tuple of (extracted_beliefs, re-ranked_memories).
        """
        beliefs = self.extract_beliefs(memories, min_significance=min_belief_significance)
        biased = self.apply_desire_bias(memories, desires)
        ranked = self.prioritize_by_intentions(biased, intentions)
        return beliefs, ranked

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _content_hash(content: str) -> str:
        """Produce a short deterministic hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
