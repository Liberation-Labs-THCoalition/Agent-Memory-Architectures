"""
Kintsugi CMA Stage 3: Adaptive Retrieval
=========================================

Hybrid search with query-adaptive weights for the Kintsugi memory system.

This module is a pure algorithm layer — no database or embedding dependencies.
It accepts pre-computed scored results from three retrieval views (dense,
lexical, symbolic) and fuses them into a single ranked list.

Key components:

    QueryProfile        Dataclass holding complexity class and per-view weights.
    ScoredResult        Uniform wrapper for results from any retrieval method.
    estimate_complexity Heuristic query analyser that maps a raw query string
                        to a QueryProfile (lookup / conceptual / balanced).
    fuse_weighted       Score-level fusion: S = w_d * dense + w_l * lexical
                        + w_s * symbolic, with weights from the profile.
    fuse_rrf            Reciprocal Rank Fusion — rank-level fusion that is
                        robust to score-scale mismatches across views.
    retrieve            High-level orchestrator combining estimation + fusion.

Typical usage::

    from kintsugi.memory.cma_stage3 import retrieve, ScoredResult

    ranked = retrieve(
        query="Why did the agent change strategy?",
        query_embedding=emb,
        dense_results=dense_hits,
        lexical_results=bm25_hits,
        symbolic_results=tag_hits,
        n_results=10,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, tuple[float, float, float]] = {
    #                  dense  lexical  symbolic
    "lookup":        (0.25,   0.55,    0.20),
    "conceptual":    (0.60,   0.15,    0.25),
    "balanced":      (0.40,   0.35,    0.25),
}

_LOOKUP_WORDS = frozenset({"who", "what", "when", "where", "which", "name", "list", "define"})
_CONCEPTUAL_WORDS = frozenset({"why", "how", "explain", "describe", "compare", "analyse", "analyze"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QueryProfile:
    """Encapsulates the estimated complexity class and per-view weights."""

    complexity: str          # "lookup" | "conceptual" | "balanced"
    dense_weight: float
    lexical_weight: float
    symbolic_weight: float

    def __post_init__(self) -> None:
        if self.complexity not in _PROFILES:
            raise ValueError(f"Unknown complexity class: {self.complexity!r}")


@dataclass(slots=True)
class ScoredResult:
    """Uniform container for a scored retrieval hit from any view."""

    id: str
    content: str
    score: float
    source: str              # "dense" | "lexical" | "symbolic" | "fused"
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        trunc = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"ScoredResult(id={self.id!r}, score={self.score:.4f}, source={self.source!r}, content={trunc!r})"


# ---------------------------------------------------------------------------
# Query complexity estimator
# ---------------------------------------------------------------------------

def estimate_complexity(query: str) -> QueryProfile:
    """Estimate query complexity and return an adaptive weight profile.

    Heuristics
    ----------
    * **Word count**: <= 3 words leans lookup; >= 10 leans conceptual.
    * **Question words**: presence of *who/what/when/where* → lookup;
      *why/how/explain* → conceptual.
    * **Entity signals**: capitalised words (not sentence-initial) and bare
      numbers hint at a lookup (the user wants a specific fact).

    Returns a ``QueryProfile`` whose weights sum to 1.0.
    """
    tokens = query.split()
    lower_tokens = [t.lower().strip("?.,!") for t in tokens]
    n_tokens = len(tokens)

    # --- signal accumulators (positive = lookup, negative = conceptual) ---
    signal: float = 0.0

    # word count
    if n_tokens <= 3:
        signal += 1.0
    elif n_tokens >= 10:
        signal -= 1.0

    # question words
    for tok in lower_tokens:
        if tok in _LOOKUP_WORDS:
            signal += 0.8
        elif tok in _CONCEPTUAL_WORDS:
            signal -= 0.8

    # entity presence — skip first token (sentence-initial caps)
    entity_count = sum(
        1 for t in tokens[1:]
        if t[0].isupper() or re.fullmatch(r"\d[\d.,]*", t)
    ) if n_tokens > 1 else 0
    signal += entity_count * 0.5

    # --- map signal to complexity class ---
    if signal >= 1.0:
        cls = "lookup"
    elif signal <= -0.8:
        cls = "conceptual"
    else:
        cls = "balanced"

    d, l, s = _PROFILES[cls]
    return QueryProfile(complexity=cls, dense_weight=d, lexical_weight=l, symbolic_weight=s)


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------

def _normalize_scores(results: Sequence[ScoredResult]) -> List[ScoredResult]:
    """Min-max normalise scores to [0, 1].  Returns new objects."""
    if not results:
        return []
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi != lo else 1.0
    return [
        ScoredResult(
            id=r.id,
            content=r.content,
            score=(r.score - lo) / span,
            source=r.source,
            metadata=r.metadata,
        )
        for r in results
    ]


def fuse_weighted(
    dense_results: Sequence[ScoredResult],
    lexical_results: Sequence[ScoredResult],
    symbolic_results: Sequence[ScoredResult],
    profile: QueryProfile,
) -> List[ScoredResult]:
    """Score-level weighted fusion.

    Each view's scores are min-max normalised independently, then combined::

        S_final = w_dense * S_dense + w_lexical * S_lexical + w_symbolic * S_symbolic

    Results appearing in multiple views accumulate scores additively.
    """
    norm_dense = _normalize_scores(dense_results)
    norm_lex = _normalize_scores(lexical_results)
    norm_sym = _normalize_scores(symbolic_results)

    # id -> accumulated score & best content/metadata
    bucket: Dict[str, ScoredResult] = {}

    def _accumulate(results: List[ScoredResult], weight: float) -> None:
        for r in results:
            contribution = weight * r.score
            if r.id in bucket:
                bucket[r.id] = ScoredResult(
                    id=r.id,
                    content=bucket[r.id].content,
                    score=bucket[r.id].score + contribution,
                    source="fused",
                    metadata={**bucket[r.id].metadata, **r.metadata},
                )
            else:
                bucket[r.id] = ScoredResult(
                    id=r.id,
                    content=r.content,
                    score=contribution,
                    source="fused",
                    metadata=dict(r.metadata),
                )

    _accumulate(norm_dense, profile.dense_weight)
    _accumulate(norm_lex, profile.lexical_weight)
    _accumulate(norm_sym, profile.symbolic_weight)

    return sorted(bucket.values(), key=lambda r: r.score, reverse=True)


def fuse_rrf(
    result_lists: Sequence[Sequence[ScoredResult]],
    k: int = 60,
) -> List[ScoredResult]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    For each result appearing at rank *r* in list *i*::

        RRF(d) = sum_i  1 / (k + r_i)

    The constant *k* (default 60) dampens the effect of high ranks and is
    the standard value from the original paper.

    Parameters
    ----------
    result_lists:
        One or more ranked result sequences (best-first ordering).
    k:
        Smoothing constant.

    Returns
    -------
    Fused results sorted by descending RRF score.
    """
    scores: Dict[str, float] = {}
    content_map: Dict[str, ScoredResult] = {}

    for rlist in result_lists:
        for rank, result in enumerate(rlist, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
            if result.id not in content_map:
                content_map[result.id] = result

    fused: List[ScoredResult] = []
    for rid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        base = content_map[rid]
        fused.append(ScoredResult(
            id=rid,
            content=base.content,
            score=score,
            source="rrf",
            metadata=dict(base.metadata),
        ))
    return fused


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    query_embedding: Optional[object] = None,
    dense_results: Sequence[ScoredResult] = (),
    lexical_results: Sequence[ScoredResult] = (),
    symbolic_results: Sequence[ScoredResult] = (),
    *,
    n_results: int = 10,
    method: str = "weighted",
    rrf_k: int = 60,
    profile_override: Optional[QueryProfile] = None,
) -> List[ScoredResult]:
    """Multi-view retrieval orchestrator.

    Parameters
    ----------
    query:
        Raw query string used for complexity estimation.
    query_embedding:
        Pre-computed embedding vector (reserved for future scoring use;
        not consumed by the current pure-fusion logic).
    dense_results:
        Scored hits from the dense (embedding) retrieval view.
    lexical_results:
        Scored hits from the lexical (BM25 / tsvector) retrieval view.
    symbolic_results:
        Scored hits from the symbolic (tag overlap / timestamp) view.
    n_results:
        Maximum number of results to return.
    method:
        ``"weighted"`` for score-level fusion (default) or ``"rrf"`` for
        Reciprocal Rank Fusion.
    rrf_k:
        Smoothing constant for RRF (only used when *method* is ``"rrf"``).
    profile_override:
        Supply an explicit ``QueryProfile`` to bypass the heuristic estimator.

    Returns
    -------
    Up to *n_results* ``ScoredResult`` objects in descending score order.
    """
    profile = profile_override if profile_override is not None else estimate_complexity(query)

    if method == "rrf":
        fused = fuse_rrf(
            [list(dense_results), list(lexical_results), list(symbolic_results)],
            k=rrf_k,
        )
    elif method == "weighted":
        fused = fuse_weighted(dense_results, lexical_results, symbolic_results, profile)
    else:
        raise ValueError(f"Unknown fusion method: {method!r}. Use 'weighted' or 'rrf'.")

    # Attach the profile used to every returned result's metadata.
    for r in fused:
        r.metadata.setdefault("query_profile", profile.complexity)

    return fused[:n_results]
