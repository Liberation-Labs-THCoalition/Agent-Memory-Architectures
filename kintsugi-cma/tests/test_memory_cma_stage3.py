"""Tests for CMA Stage 3 — Adaptive Retrieval."""

from __future__ import annotations

import pytest

from kintsugi.memory.cma_stage3 import (
    QueryProfile,
    ScoredResult,
    _normalize_scores,
    estimate_complexity,
    fuse_rrf,
    fuse_weighted,
    retrieve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sr(id: str, score: float, source: str = "dense", content: str = "c") -> ScoredResult:
    return ScoredResult(id=id, content=content, score=score, source=source)


# ---------------------------------------------------------------------------
# QueryProfile
# ---------------------------------------------------------------------------


class TestQueryProfile:
    def test_valid_creation(self):
        p = QueryProfile(complexity="lookup", dense_weight=0.25, lexical_weight=0.55, symbolic_weight=0.20)
        assert p.complexity == "lookup"

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError):
            QueryProfile(complexity="unknown", dense_weight=0.5, lexical_weight=0.3, symbolic_weight=0.2)

    def test_frozen(self):
        p = QueryProfile(complexity="balanced", dense_weight=0.4, lexical_weight=0.35, symbolic_weight=0.25)
        with pytest.raises(AttributeError):
            p.complexity = "lookup"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScoredResult
# ---------------------------------------------------------------------------


class TestScoredResult:
    def test_default_metadata(self):
        r = ScoredResult(id="x", content="hi", score=0.5, source="dense")
        assert r.metadata == {}

    def test_repr_truncation(self):
        long_content = "a" * 100
        r = ScoredResult(id="x", content=long_content, score=0.5, source="dense")
        assert "..." in repr(r)


# ---------------------------------------------------------------------------
# estimate_complexity
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    def test_lookup_who(self):
        p = estimate_complexity("Who is Alice?")
        assert p.complexity == "lookup"

    def test_lookup_short(self):
        p = estimate_complexity("Bob age")
        assert p.complexity == "lookup"

    def test_conceptual_why(self):
        p = estimate_complexity("Why did the agent change its strategy over time?")
        assert p.complexity == "conceptual"

    def test_conceptual_explain(self):
        p = estimate_complexity("Explain the reasoning behind the architecture decisions made here")
        assert p.complexity == "conceptual"

    def test_balanced_generic(self):
        p = estimate_complexity("tell me about the project progress")
        assert p.complexity == "balanced"

    def test_entity_signals_boost_lookup(self):
        p = estimate_complexity("Find John Smith 2025")
        assert p.complexity == "lookup"

    def test_weights_sum_to_one(self):
        for query in ["Who?", "Why does it work?", "general question about things"]:
            p = estimate_complexity(query)
            total = p.dense_weight + p.lexical_weight + p.symbolic_weight
            assert total == pytest.approx(1.0)

    def test_single_word(self):
        p = estimate_complexity("hello")
        # single word, no special signals -> lookup (<=3 tokens gives +1.0)
        assert p.complexity == "lookup"

    def test_empty_query(self):
        p = estimate_complexity("")
        # 0 tokens -> signal += 1.0 for <=3 words -> lookup
        assert p.complexity == "lookup"


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_empty(self):
        assert _normalize_scores([]) == []

    def test_single_item(self):
        results = _normalize_scores([_sr("a", 5.0)])
        assert results[0].score == 0.0  # (5-5)/(1) = 0

    def test_range(self):
        results = _normalize_scores([_sr("a", 1.0), _sr("b", 3.0), _sr("c", 5.0)])
        scores = {r.id: r.score for r in results}
        assert scores["a"] == pytest.approx(0.0)
        assert scores["c"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.5)

    def test_all_same_score(self):
        results = _normalize_scores([_sr("a", 3.0), _sr("b", 3.0)])
        # span=0 -> uses 1.0, so (3-3)/1 = 0
        assert all(r.score == 0.0 for r in results)


# ---------------------------------------------------------------------------
# fuse_weighted
# ---------------------------------------------------------------------------


class TestFuseWeighted:
    def test_empty_inputs(self):
        profile = QueryProfile(complexity="balanced", dense_weight=0.4, lexical_weight=0.35, symbolic_weight=0.25)
        result = fuse_weighted([], [], [], profile)
        assert result == []

    def test_single_view(self):
        profile = QueryProfile(complexity="lookup", dense_weight=0.25, lexical_weight=0.55, symbolic_weight=0.20)
        dense = [_sr("a", 1.0)]
        result = fuse_weighted(dense, [], [], profile)
        assert len(result) == 1
        assert result[0].source == "fused"

    def test_overlapping_ids_accumulate(self):
        profile = QueryProfile(complexity="balanced", dense_weight=0.4, lexical_weight=0.35, symbolic_weight=0.25)
        dense = [_sr("a", 1.0, "dense"), _sr("b", 0.5, "dense")]
        lex = [_sr("a", 0.8, "lexical"), _sr("c", 1.0, "lexical")]
        sym = [_sr("a", 0.5, "symbolic")]
        result = fuse_weighted(dense, lex, sym, profile)
        ids = [r.id for r in result]
        assert "a" in ids
        # "a" appears in all three views, should have highest score
        assert result[0].id == "a"

    def test_sorted_descending(self):
        profile = QueryProfile(complexity="balanced", dense_weight=0.4, lexical_weight=0.35, symbolic_weight=0.25)
        dense = [_sr("a", 0.1), _sr("b", 1.0)]
        result = fuse_weighted(dense, [], [], profile)
        assert result[0].score >= result[1].score


# ---------------------------------------------------------------------------
# fuse_rrf
# ---------------------------------------------------------------------------


class TestFuseRRF:
    def test_empty(self):
        assert fuse_rrf([]) == []

    def test_single_list(self):
        results = fuse_rrf([[_sr("a", 1.0), _sr("b", 0.5)]])
        assert len(results) == 2
        assert results[0].id == "a"  # rank 1 -> higher RRF score

    def test_overlapping_lists(self):
        l1 = [_sr("a", 1.0), _sr("b", 0.5)]
        l2 = [_sr("b", 1.0), _sr("a", 0.5)]
        results = fuse_rrf([l1, l2], k=60)
        # Both appear in both lists, scores accumulate
        assert len(results) == 2
        # a: 1/61 + 1/62, b: 1/62 + 1/61 -> same score
        assert results[0].score == pytest.approx(results[1].score)

    def test_rrf_source_tag(self):
        results = fuse_rrf([[_sr("a", 1.0)]])
        assert results[0].source == "rrf"

    def test_custom_k(self):
        results = fuse_rrf([[_sr("a", 1.0)]], k=10)
        assert results[0].score == pytest.approx(1.0 / 11)

    def test_three_lists(self):
        l1 = [_sr("a", 1.0)]
        l2 = [_sr("a", 1.0)]
        l3 = [_sr("a", 1.0)]
        results = fuse_rrf([l1, l2, l3], k=60)
        assert results[0].score == pytest.approx(3.0 / 61)


# ---------------------------------------------------------------------------
# retrieve (orchestrator)
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_weighted_method(self):
        dense = [_sr("a", 1.0, "dense")]
        results = retrieve("Who is Alice?", dense_results=dense, n_results=5)
        assert len(results) <= 5
        assert all(r.metadata.get("query_profile") for r in results)

    def test_rrf_method(self):
        dense = [_sr("a", 1.0, "dense")]
        lex = [_sr("b", 0.9, "lexical")]
        results = retrieve("test", dense_results=dense, lexical_results=lex, method="rrf")
        assert len(results) == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion method"):
            retrieve("test", method="invalid")

    def test_n_results_limit(self):
        dense = [_sr(f"d{i}", float(i), "dense") for i in range(20)]
        results = retrieve("test", dense_results=dense, n_results=5)
        assert len(results) == 5

    def test_profile_override(self):
        profile = QueryProfile(complexity="conceptual", dense_weight=0.6, lexical_weight=0.15, symbolic_weight=0.25)
        dense = [_sr("a", 1.0, "dense")]
        results = retrieve("Who?", dense_results=dense, profile_override=profile)
        assert results[0].metadata["query_profile"] == "conceptual"

    def test_empty_inputs(self):
        results = retrieve("test query")
        assert results == []

    def test_all_three_views(self):
        dense = [_sr("a", 0.9, "dense"), _sr("b", 0.7, "dense")]
        lex = [_sr("b", 0.8, "lexical"), _sr("c", 0.6, "lexical")]
        sym = [_sr("c", 0.5, "symbolic"), _sr("a", 0.3, "symbolic")]
        results = retrieve("balanced query here", dense_results=dense,
                           lexical_results=lex, symbolic_results=sym)
        ids = {r.id for r in results}
        assert ids == {"a", "b", "c"}

    def test_rrf_with_all_views(self):
        dense = [_sr("a", 0.9, "dense")]
        lex = [_sr("a", 0.8, "lexical")]
        sym = [_sr("a", 0.5, "symbolic")]
        results = retrieve("test", dense_results=dense, lexical_results=lex,
                           symbolic_results=sym, method="rrf", rrf_k=60)
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(3.0 / 61)
