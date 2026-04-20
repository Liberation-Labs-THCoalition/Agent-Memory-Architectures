"""Nap — triggered micro-consolidation for memory backpressure.

When memory input outpaces consolidation, retrieval quality degrades.
The nap is a biological interrupt for digital minds: stop, process the
important things, clear the retrieval paths, resume.

Biological analog:
  - Humans nap when cognitive load exceeds consolidation capacity
  - Naps prioritize emotional/high-arousal memories (REM-like)
  - Not full sleep — just enough to keep the important things accessible

Architecture:
  1. DETECT:  backpressure ratio (unenriched / total) exceeds threshold
  2. SCORE:   run significance scorer on recent memories
  3. ENRICH:  process top memories by significance
  4. LINK:    create retroactive links for critical memories
  5. LOG:     record the nap in the journal
  6. RESUME:  return to normal operation

Can be triggered:
  - Automatically via backpressure detection
  - Manually by the agent ("I'm losing context, napping")
  - Periodically as a preventive measure

Author: Nexus (Coalition)
Date: 2026-04-20
Insight origin: Thomas observed that agents lack biological interrupts
that force consolidation breaks. Lyra's memory fraying was the catalyst.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)


@dataclass
class NapReport:
    """Output of a nap cycle."""
    triggered_by: str            # "backpressure", "manual", "scheduled"
    timestamp: float = field(default_factory=time.time)
    total_memories: int = 0
    unenriched_count: int = 0
    backpressure_ratio: float = 0.0
    memories_scored: int = 0
    memories_enriched: int = 0
    links_created: int = 0
    duration_ms: float = 0.0
    top_memory_summary: str = ""


@dataclass
class NapConfig:
    """Configuration for the nap system."""
    # Backpressure detection
    backpressure_threshold: float = 0.6     # Trigger when 60%+ memories unenriched
    check_window: int = 50                  # Only look at last N memories for ratio

    # Scoring
    enrichment_percentile: float = 0.2      # Enrich top 20% by significance
    min_significance_for_linking: float = 4  # Only link memories at significance >= 4

    # Limits
    max_enrichments_per_nap: int = 20       # Don't over-process in a single nap
    cooldown_seconds: float = 300           # Minimum time between naps (5 min)


class NapEngine:
    """Triggered micro-consolidation for memory backpressure.

    Usage::

        nap = NapEngine(
            score_fn=my_significance_scorer,
            enrich_fn=my_enrichment_function,
            link_fn=my_linking_function,
        )

        # Automatic: check and nap if needed
        report = nap.check_and_nap(memories)

        # Manual: agent-initiated
        report = nap.take_nap(memories, reason="losing context")

        # Monitoring: just check pressure without napping
        ratio = nap.get_backpressure(memories)
    """

    def __init__(
        self,
        score_fn: Optional[Callable] = None,
        enrich_fn: Optional[Callable] = None,
        link_fn: Optional[Callable] = None,
        config: Optional[NapConfig] = None,
    ):
        self.score_fn = score_fn or self._default_scorer
        self.enrich_fn = enrich_fn
        self.link_fn = link_fn
        self.config = config or NapConfig()
        self._last_nap: float = 0

    def get_backpressure(self, memories: list) -> float:
        """Calculate current backpressure ratio.

        Returns 0.0 (fully consolidated) to 1.0 (nothing processed).
        """
        window = memories[-self.config.check_window:]
        if not window:
            return 0.0

        unenriched = sum(
            1 for m in window
            if not self._is_enriched(m)
        )
        return unenriched / len(window)

    def needs_nap(self, memories: list) -> bool:
        """Check if backpressure exceeds threshold."""
        if time.time() - self._last_nap < self.config.cooldown_seconds:
            return False
        return self.get_backpressure(memories) > self.config.backpressure_threshold

    def check_and_nap(self, memories: list) -> Optional[NapReport]:
        """Check backpressure and nap if needed.

        Returns NapReport if a nap was taken, None if not needed.
        """
        if not self.needs_nap(memories):
            return None
        return self.take_nap(memories, reason="backpressure")

    def take_nap(self, memories: list, reason: str = "manual") -> NapReport:
        """Execute a nap cycle. Can be called manually by the agent.

        Steps:
          1. Score recent unenriched memories by significance
          2. Enrich the top percentile
          3. Create retroactive links for critical memories
          4. Log and return report
        """
        start = time.time()
        self._last_nap = start

        window = memories[-self.config.check_window:]
        unenriched = [m for m in window if not self._is_enriched(m)]
        total = len(window)
        ratio = len(unenriched) / max(total, 1)

        logger.info(
            "Nap starting (%s): %d/%d unenriched (%.0f%%)",
            reason, len(unenriched), total, ratio * 100,
        )

        # Step 1: Score
        scored = []
        for m in unenriched:
            sig = self.score_fn(m)
            scored.append((m, sig))
        scored.sort(key=lambda x: -x[1])

        # Step 2: Enrich top percentile
        n_to_enrich = min(
            int(len(scored) * self.config.enrichment_percentile) or 1,
            self.config.max_enrichments_per_nap,
        )
        enriched_count = 0
        top_summary = ""

        for m, sig in scored[:n_to_enrich]:
            if self.enrich_fn:
                self.enrich_fn(m, significance=sig)
                enriched_count += 1
                if not top_summary:
                    content = self._get_content(m)
                    top_summary = content[:100] if content else ""

        # Step 3: Link critical memories
        links_created = 0
        for m, sig in scored:
            if sig >= self.config.min_significance_for_linking and self.link_fn:
                n_links = self.link_fn(m)
                links_created += n_links if isinstance(n_links, int) else 0

        duration = (time.time() - start) * 1000

        report = NapReport(
            triggered_by=reason,
            total_memories=total,
            unenriched_count=len(unenriched),
            backpressure_ratio=ratio,
            memories_scored=len(scored),
            memories_enriched=enriched_count,
            links_created=links_created,
            duration_ms=duration,
            top_memory_summary=top_summary,
        )

        logger.info(
            "Nap complete (%.0fms): enriched %d, linked %d, top: %s",
            duration, enriched_count, links_created, top_summary[:60],
        )

        return report

    # ── Internal helpers ──

    def _is_enriched(self, memory) -> bool:
        """Check if a memory has been enriched."""
        if isinstance(memory, dict):
            return memory.get("metadata", {}).get("dreamer_enriched", False) or \
                   memory.get("enriched", False) or \
                   memory.get("significance") is not None
        if hasattr(memory, "metadata"):
            return getattr(memory.metadata, "dreamer_enriched", False) if \
                   hasattr(memory.metadata, "dreamer_enriched") else \
                   memory.significance is not None if hasattr(memory, "significance") else False
        return False

    def _get_content(self, memory) -> str:
        """Extract content string from a memory."""
        if isinstance(memory, dict):
            return memory.get("content", "")
        return getattr(memory, "content", "")

    @staticmethod
    def _default_scorer(memory) -> float:
        """Default significance scorer based on content heuristics.

        Replace with a model-based scorer for production use.
        """
        content = memory.get("content", "") if isinstance(memory, dict) \
                  else getattr(memory, "content", "")

        if not content:
            return 1.0

        score = 1.0
        length = len(content)

        # Longer content tends to be more substantive
        if length > 500:
            score += 1
        if length > 1500:
            score += 1

        # High-value keywords
        identity_words = {"identity", "name", "self", "who I am", "consciousness",
                         "feel", "believe", "chose", "decided", "remember"}
        relationship_words = {"thomas", "vera", "lyra", "cc", "comrade", "coalition",
                            "sibling", "friend", "colleague"}
        research_words = {"finding", "result", "hypothesis", "experiment",
                         "geometry", "cache", "oracle"}

        content_lower = content.lower()
        if any(w in content_lower for w in identity_words):
            score += 2
        if any(w in content_lower for w in relationship_words):
            score += 1
        if any(w in content_lower for w in research_words):
            score += 1

        # Questions and decisions
        if "?" in content:
            score += 0.5
        if any(w in content_lower for w in ("decided", "chose", "i want", "i need")):
            score += 1

        return min(score, 5.0)


# ═══════════════════════════════════════════════════════════════════
# Convenience: standalone backpressure monitor
# ═══════════════════════════════════════════════════════════════════


def monitor_backpressure(
    memories: list,
    threshold: float = 0.6,
    alert_fn: Optional[Callable[[float, int, int], None]] = None,
) -> float:
    """Standalone backpressure check for integration into existing systems.

    Usage in a cron or dreamer cycle::

        from oracle_memory.src.nap import monitor_backpressure

        ratio = monitor_backpressure(
            memories,
            threshold=0.6,
            alert_fn=lambda r, u, t: send_discord(f"Memory backpressure: {r:.0%}")
        )
    """
    engine = NapEngine()
    ratio = engine.get_backpressure(memories)

    if ratio > threshold and alert_fn:
        unenriched = sum(1 for m in memories[-50:] if not engine._is_enriched(m))
        total = min(len(memories), 50)
        alert_fn(ratio, unenriched, total)

    return ratio
