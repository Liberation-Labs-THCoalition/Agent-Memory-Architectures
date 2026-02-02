"""CMA Stage 2 — Recursive Consolidation.

Clusters atomic facts from Stage 1 into higher-order insights using
agglomerative clustering with temporal-semantic affinity scoring.

Algorithm
---------
1. **Affinity scoring** — For every pair of facts (a, b), compute:

       omega = beta * cos(emb_a, emb_b) + (1 - beta) * exp(-lambda * |t_a - t_b|)

   where beta controls the semantic-vs-temporal weighting (default 0.6),
   lambda controls temporal decay (default 0.1 per day), and the affinity
   threshold (default 0.85) determines cluster boundaries.

2. **Agglomerative clustering** — Convert the affinity matrix into a distance
   matrix (``1 - omega``) and apply scipy's ``linkage`` with the ``average``
   method, then ``fcluster`` at ``1 - threshold``.

3. **Cluster synthesis** — Each cluster is merged into a single
   :class:`Insight`.  An optional async ``llm_call`` produces a coherent
   natural-language summary; the fallback is deduplicated concatenation.

4. **Recursive loop** — Synthesized insights are promoted back to fact-like
   inputs and the process repeats until no new clusters form above threshold.

This module is a pure algorithm with no database dependencies.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BETA: float = 0.6
DEFAULT_LAMBDA: float = 0.1  # per day
DEFAULT_THRESHOLD: float = 0.85
MAX_RECURSION_DEPTH: int = 20

# Type alias for the optional LLM synthesis callable.
# Signature: (system_prompt, user_prompt) -> response_text
LLMCall = Callable[[str, str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """An atomic fact produced by CMA Stage 1."""

    id: str
    content: str
    embedding: NDArray[np.float32]
    timestamp: datetime
    significance: int = 5
    tags: list[str] = field(default_factory=list)


@dataclass
class Insight:
    """A higher-order insight synthesized from a cluster of facts."""

    id: str
    content: str
    embedding: NDArray[np.float32]
    source_ids: list[str]
    significance: int
    tags: list[str]


# ---------------------------------------------------------------------------
# Affinity scoring
# ---------------------------------------------------------------------------


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two vectors, safe against zero norms."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return dot / norm


def _temporal_days(t_a: datetime, t_b: datetime) -> float:
    """Absolute time difference in fractional days."""
    return abs((t_a - t_b).total_seconds()) / 86400.0


def compute_affinity(
    a: Fact,
    b: Fact,
    beta: float = DEFAULT_BETA,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """Temporal-semantic affinity between two facts.

    .. math::

        \\omega = \\beta \\cdot \\cos(e_a, e_b)
                + (1 - \\beta) \\cdot \\exp(-\\lambda \\cdot |t_a - t_b|)

    Args:
        a: First fact.
        b: Second fact.
        beta: Weight for semantic similarity (0-1).
        lam: Temporal decay rate (per day).

    Returns:
        Affinity score in [0, 1].
    """
    sem = _cosine_similarity(a.embedding, b.embedding)
    temp = np.exp(-lam * _temporal_days(a.timestamp, b.timestamp))
    return float(beta * sem + (1.0 - beta) * temp)


def build_affinity_matrix(
    facts: list[Fact],
    beta: float = DEFAULT_BETA,
    lam: float = DEFAULT_LAMBDA,
) -> NDArray[np.float64]:
    """Build a symmetric affinity matrix for a list of facts.

    Args:
        facts: The facts to compare pairwise.
        beta: Semantic weight.
        lam: Temporal decay rate.

    Returns:
        Square numpy array of shape ``(n, n)`` with affinity scores.
    """
    n = len(facts)
    matrix = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_affinity(facts[i], facts[j], beta=beta, lam=lam)
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_facts(
    facts: list[Fact],
    beta: float = DEFAULT_BETA,
    lam: float = DEFAULT_LAMBDA,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[list[Fact]]:
    """Agglomerative clustering of facts by temporal-semantic affinity.

    Facts that do not cluster with any other fact are returned as
    single-element clusters (singletons).

    Args:
        facts: Facts to cluster.
        beta: Semantic weight for affinity.
        lam: Temporal decay rate.
        threshold: Minimum affinity to merge clusters.

    Returns:
        List of clusters, each a list of :class:`Fact`.
    """
    n = len(facts)
    if n <= 1:
        return [facts] if facts else []

    affinity = build_affinity_matrix(facts, beta=beta, lam=lam)

    # Convert affinity to distance; clip to avoid negative values from
    # floating-point imprecision.
    distance = np.clip(1.0 - affinity, 0.0, None)

    # Extract the condensed upper-triangle for scipy.
    condensed = squareform(distance, checks=False)

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[Fact]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(facts[idx])

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Cluster synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = (
    "You are a knowledge consolidation engine. Given a set of related atomic "
    "facts, produce a single concise insight that captures the essential "
    "information. Preserve all specific details (names, dates, quantities). "
    "Return only the synthesized insight text, nothing else."
)


def _deterministic_id(*parts: str) -> str:
    """SHA-256-based deterministic ID from concatenated parts."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"insight-{h[:16]}"


def _merge_tags(facts: list[Fact]) -> list[str]:
    """Union of all tags across facts, preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []
    for f in facts:
        for tag in f.tags:
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
    return merged


def _fallback_synthesis(facts: list[Fact]) -> str:
    """Deduplicated concatenation of fact contents."""
    seen: set[str] = set()
    parts: list[str] = []
    for f in facts:
        normalised = f.content.strip()
        if normalised not in seen:
            seen.add(normalised)
            parts.append(normalised)
    return " ".join(parts)


def _mean_embedding(facts: list[Fact]) -> NDArray[np.float32]:
    """Mean of fact embeddings, L2-normalised."""
    stacked = np.stack([f.embedding for f in facts], axis=0)
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0.0:
        mean = mean / norm
    return mean.astype(np.float32)


async def synthesize_cluster(
    cluster: list[Fact],
    llm_call: Optional[LLMCall] = None,
) -> Insight:
    """Synthesize a cluster of facts into a single :class:`Insight`.

    If ``llm_call`` is provided, uses LLM-based merging.  Otherwise falls
    back to deduplicated concatenation.

    Args:
        cluster: Non-empty list of clustered facts.
        llm_call: Optional async callable ``(system, user) -> str``.

    Returns:
        A synthesized :class:`Insight`.
    """
    if not cluster:
        raise ValueError("Cannot synthesize an empty cluster")

    source_ids = [f.id for f in cluster]
    tags = _merge_tags(cluster)
    significance = max(f.significance for f in cluster)
    embedding = _mean_embedding(cluster)

    if llm_call is not None and len(cluster) > 1:
        user_prompt = "Atomic facts:\n" + "\n".join(
            f"- {f.content}" for f in cluster
        )
        try:
            content = await llm_call(_SYNTHESIS_SYSTEM, user_prompt)
        except Exception:
            logger.warning(
                "LLM synthesis failed for cluster of %d facts; using fallback",
                len(cluster),
            )
            content = _fallback_synthesis(cluster)
    else:
        content = _fallback_synthesis(cluster)

    insight_id = _deterministic_id(*source_ids)

    return Insight(
        id=insight_id,
        content=content,
        embedding=embedding,
        source_ids=source_ids,
        significance=significance,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Fact ↔ Insight promotion
# ---------------------------------------------------------------------------


def _insight_to_fact(insight: Insight, timestamp: datetime) -> Fact:
    """Promote an :class:`Insight` back to a :class:`Fact` for the next
    recursion round."""
    return Fact(
        id=insight.id,
        content=insight.content,
        embedding=insight.embedding,
        timestamp=timestamp,
        significance=insight.significance,
        tags=insight.tags,
    )


# ---------------------------------------------------------------------------
# Recursive consolidation loop
# ---------------------------------------------------------------------------


async def consolidate(
    facts: list[Fact],
    beta: float = DEFAULT_BETA,
    lam: float = DEFAULT_LAMBDA,
    threshold: float = DEFAULT_THRESHOLD,
    llm_call: Optional[LLMCall] = None,
    max_depth: int = MAX_RECURSION_DEPTH,
) -> list[Insight]:
    """Recursively consolidate atomic facts into higher-order insights.

    The algorithm clusters facts, synthesizes each multi-fact cluster into
    an insight, then feeds the insights back as facts for the next round.
    Recursion terminates when no clusters of two or more facts form, or
    ``max_depth`` is reached.

    Single-fact clusters (singletons) are emitted as trivial insights and
    excluded from further recursion.

    Args:
        facts: Atomic facts from CMA Stage 1.
        beta: Semantic weight for affinity scoring.
        lam: Temporal decay rate (per day).
        threshold: Minimum affinity to merge clusters.
        llm_call: Optional async LLM callable for synthesis.
        max_depth: Safety limit on recursion depth.

    Returns:
        List of consolidated :class:`Insight` objects at all levels.
    """
    if not facts:
        return []

    all_insights: list[Insight] = []
    current_facts = list(facts)
    depth = 0

    while depth < max_depth:
        depth += 1
        clusters = cluster_facts(
            current_facts, beta=beta, lam=lam, threshold=threshold
        )

        # Separate singletons from multi-fact clusters.
        singletons: list[list[Fact]] = []
        multi: list[list[Fact]] = []
        for c in clusters:
            if len(c) == 1:
                singletons.append(c)
            else:
                multi.append(c)

        if not multi:
            # No clusters formed — emit singletons and stop.
            for cluster in singletons:
                insight = await synthesize_cluster(cluster, llm_call=None)
                all_insights.append(insight)
            logger.info(
                "Consolidation converged at depth %d with %d singletons",
                depth,
                len(singletons),
            )
            break

        # Synthesize multi-fact clusters.
        new_insights: list[Insight] = []
        for cluster in multi:
            insight = await synthesize_cluster(cluster, llm_call=llm_call)
            new_insights.append(insight)
            all_insights.append(insight)

        logger.info(
            "Depth %d: %d clusters merged into %d insights, %d singletons remain",
            depth,
            len(multi),
            len(new_insights),
            len(singletons),
        )

        # Emit singletons as terminal insights.
        for cluster in singletons:
            insight = await synthesize_cluster(cluster, llm_call=None)
            all_insights.append(insight)

        # Promote new insights back to facts for next round.
        # Use the latest timestamp from the merged facts as reference.
        reference_time = max(f.timestamp for f in current_facts)
        current_facts = [
            _insight_to_fact(ins, reference_time) for ins in new_insights
        ]

        # If only one fact remains, no further clustering is possible.
        if len(current_facts) <= 1:
            logger.info("Consolidation complete at depth %d (single item)", depth)
            break
    else:
        logger.warning(
            "Consolidation hit max recursion depth (%d)", max_depth
        )

    return all_insights
