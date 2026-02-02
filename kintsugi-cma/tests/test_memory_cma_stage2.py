"""Tests for CMA Stage 2 — Recursive Consolidation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from kintsugi.memory.cma_stage2 import (
    Fact,
    Insight,
    _cosine_similarity,
    _deterministic_id,
    _fallback_synthesis,
    _insight_to_fact,
    _mean_embedding,
    _merge_tags,
    _temporal_days,
    build_affinity_matrix,
    cluster_facts,
    compute_affinity,
    consolidate,
    synthesize_cluster,
)

DIM = 8


def _make_fact(
    id: str = "f1",
    content: str = "test fact",
    embedding: np.ndarray | None = None,
    timestamp: datetime | None = None,
    significance: int = 5,
    tags: list[str] | None = None,
) -> Fact:
    if embedding is None:
        rng = np.random.RandomState(hash(id) % 2**31)
        embedding = rng.randn(DIM).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-9
    return Fact(
        id=id,
        content=content,
        embedding=embedding,
        timestamp=timestamp or datetime(2025, 1, 1),
        significance=significance,
        tags=tags or [],
    )


def _make_similar_facts(n: int, base_vec: np.ndarray | None = None) -> list[Fact]:
    """Create n facts with very similar embeddings and close timestamps."""
    if base_vec is None:
        base_vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    facts = []
    for i in range(n):
        noise = np.random.RandomState(i).randn(DIM).astype(np.float32) * 0.01
        vec = base_vec + noise
        vec /= np.linalg.norm(vec)
        facts.append(_make_fact(
            id=f"f{i}",
            content=f"Similar fact {i}",
            embedding=vec,
            timestamp=datetime(2025, 1, 1) + timedelta(hours=i),
            tags=["common"],
        ))
    return facts


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_cosine_identical(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_zero(self):
        assert _cosine_similarity(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)) == 0.0

    def test_temporal_days(self):
        a = datetime(2025, 1, 1)
        b = datetime(2025, 1, 3, 12)  # 2.5 days later
        assert _temporal_days(a, b) == pytest.approx(2.5)
        assert _temporal_days(b, a) == pytest.approx(2.5)  # symmetric

    def test_deterministic_id(self):
        id1 = _deterministic_id("a", "b")
        id2 = _deterministic_id("a", "b")
        id3 = _deterministic_id("a", "c")
        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("insight-")

    def test_merge_tags_preserves_order(self):
        f1 = _make_fact(id="a", tags=["x", "y"])
        f2 = _make_fact(id="b", tags=["y", "z"])
        assert _merge_tags([f1, f2]) == ["x", "y", "z"]

    def test_merge_tags_empty(self):
        assert _merge_tags([]) == []

    def test_fallback_synthesis_dedup(self):
        f1 = _make_fact(id="a", content="Hello")
        f2 = _make_fact(id="b", content="Hello")
        f3 = _make_fact(id="c", content="World")
        assert _fallback_synthesis([f1, f2, f3]) == "Hello World"

    def test_mean_embedding_normalized(self):
        facts = _make_similar_facts(3)
        mean = _mean_embedding(facts)
        assert mean.dtype == np.float32
        assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_affinity
# ---------------------------------------------------------------------------


class TestComputeAffinity:
    def test_identical_facts_high_affinity(self):
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        t = datetime(2025, 1, 1)
        a = _make_fact(id="a", embedding=vec, timestamp=t)
        b = _make_fact(id="b", embedding=vec.copy(), timestamp=t)
        aff = compute_affinity(a, b)
        assert aff == pytest.approx(1.0, abs=1e-5)

    def test_distant_time_reduces_affinity(self):
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        a = _make_fact(id="a", embedding=vec, timestamp=datetime(2025, 1, 1))
        b = _make_fact(id="b", embedding=vec.copy(), timestamp=datetime(2025, 7, 1))
        aff = compute_affinity(a, b)
        assert aff < 1.0

    def test_beta_zero_ignores_semantics(self):
        a = _make_fact(id="a", timestamp=datetime(2025, 1, 1))
        b = _make_fact(id="b", timestamp=datetime(2025, 1, 1))
        aff = compute_affinity(a, b, beta=0.0)
        # Pure temporal, same time -> exp(0) = 1.0
        assert aff == pytest.approx(1.0, abs=1e-5)

    def test_beta_one_ignores_temporal(self):
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        a = _make_fact(id="a", embedding=vec, timestamp=datetime(2025, 1, 1))
        b = _make_fact(id="b", embedding=vec.copy(), timestamp=datetime(2030, 1, 1))
        aff = compute_affinity(a, b, beta=1.0)
        assert aff == pytest.approx(1.0, abs=1e-5)


class TestBuildAffinityMatrix:
    def test_shape_and_symmetry(self):
        facts = [_make_fact(id=f"f{i}") for i in range(4)]
        m = build_affinity_matrix(facts)
        assert m.shape == (4, 4)
        np.testing.assert_array_almost_equal(m, m.T)

    def test_diagonal_is_one(self):
        facts = [_make_fact(id=f"f{i}") for i in range(3)]
        m = build_affinity_matrix(facts)
        np.testing.assert_array_almost_equal(np.diag(m), np.ones(3))


# ---------------------------------------------------------------------------
# cluster_facts
# ---------------------------------------------------------------------------


class TestClusterFacts:
    def test_empty(self):
        assert cluster_facts([]) == []

    def test_single_fact(self):
        clusters = cluster_facts([_make_fact()])
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_similar_facts_cluster_together(self):
        facts = _make_similar_facts(4)
        clusters = cluster_facts(facts, threshold=0.5)
        # Very similar facts should end up in one cluster
        assert len(clusters) == 1

    def test_dissimilar_facts_separate(self):
        v1 = np.zeros(DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(DIM, dtype=np.float32)
        v2[1] = 1.0
        f1 = _make_fact(id="a", embedding=v1, timestamp=datetime(2025, 1, 1))
        f2 = _make_fact(id="b", embedding=v2, timestamp=datetime(2026, 1, 1))
        clusters = cluster_facts([f1, f2], threshold=0.99)
        assert len(clusters) == 2

    def test_two_facts_identical(self):
        vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        t = datetime(2025, 1, 1)
        f1 = _make_fact(id="a", embedding=vec, timestamp=t)
        f2 = _make_fact(id="b", embedding=vec.copy(), timestamp=t)
        clusters = cluster_facts([f1, f2], threshold=0.5)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# synthesize_cluster
# ---------------------------------------------------------------------------


class TestSynthesizeCluster:
    @pytest.mark.asyncio
    async def test_empty_cluster_raises(self):
        with pytest.raises(ValueError):
            await synthesize_cluster([])

    @pytest.mark.asyncio
    async def test_singleton_uses_fallback(self):
        f = _make_fact(id="f1", content="single fact", tags=["t1"])
        insight = await synthesize_cluster([f])
        assert isinstance(insight, Insight)
        assert insight.content == "single fact"
        assert insight.source_ids == ["f1"]
        assert insight.tags == ["t1"]

    @pytest.mark.asyncio
    async def test_with_llm_call(self):
        async def mock_llm(sys: str, usr: str) -> str:
            return "Synthesized insight"

        facts = _make_similar_facts(3)
        insight = await synthesize_cluster(facts, llm_call=mock_llm)
        assert insight.content == "Synthesized insight"
        assert len(insight.source_ids) == 3

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        async def failing_llm(sys: str, usr: str) -> str:
            raise RuntimeError("LLM down")

        facts = _make_similar_facts(2)
        insight = await synthesize_cluster(facts, llm_call=failing_llm)
        # Should use fallback concatenation
        assert "Similar fact 0" in insight.content

    @pytest.mark.asyncio
    async def test_significance_is_max(self):
        f1 = _make_fact(id="a", significance=3)
        f2 = _make_fact(id="b", significance=8)
        insight = await synthesize_cluster([f1, f2])
        assert insight.significance == 8

    @pytest.mark.asyncio
    async def test_singleton_with_llm_uses_fallback(self):
        """Single-element cluster should NOT call LLM (len(cluster) > 1 check)."""
        called = False

        async def mock_llm(sys: str, usr: str) -> str:
            nonlocal called
            called = True
            return "should not be used"

        f = _make_fact(id="f1", content="solo")
        insight = await synthesize_cluster([f], llm_call=mock_llm)
        assert not called
        assert insight.content == "solo"


# ---------------------------------------------------------------------------
# _insight_to_fact
# ---------------------------------------------------------------------------


class TestInsightToFact:
    def test_conversion(self):
        emb = np.ones(DIM, dtype=np.float32)
        ins = Insight(id="i1", content="insight", embedding=emb,
                      source_ids=["a"], significance=7, tags=["t"])
        f = _insight_to_fact(ins, datetime(2025, 6, 1))
        assert f.id == "i1"
        assert f.content == "insight"
        assert f.significance == 7
        assert f.timestamp == datetime(2025, 6, 1)


# ---------------------------------------------------------------------------
# consolidate (recursive)
# ---------------------------------------------------------------------------


class TestConsolidate:
    @pytest.mark.asyncio
    async def test_empty(self):
        assert await consolidate([]) == []

    @pytest.mark.asyncio
    async def test_single_fact_returns_singleton_insight(self):
        f = _make_fact()
        insights = await consolidate([f])
        assert len(insights) == 1

    @pytest.mark.asyncio
    async def test_similar_facts_consolidate(self):
        facts = _make_similar_facts(4)
        insights = await consolidate(facts, threshold=0.5)
        assert len(insights) >= 1

    @pytest.mark.asyncio
    async def test_max_depth_respected(self):
        facts = _make_similar_facts(5)
        insights = await consolidate(facts, threshold=0.5, max_depth=1)
        assert len(insights) >= 1

    @pytest.mark.asyncio
    async def test_with_llm(self):
        async def mock_llm(sys: str, usr: str) -> str:
            return "consolidated insight"

        facts = _make_similar_facts(3)
        insights = await consolidate(facts, threshold=0.5, llm_call=mock_llm)
        assert any(i.content == "consolidated insight" for i in insights)

    @pytest.mark.asyncio
    async def test_dissimilar_all_singletons(self):
        """Orthogonal, time-distant facts should not cluster at high threshold."""
        facts = []
        for i in range(3):
            v = np.zeros(DIM, dtype=np.float32)
            v[i] = 1.0
            facts.append(_make_fact(
                id=f"f{i}",
                embedding=v,
                timestamp=datetime(2025, 1, 1) + timedelta(days=365 * i),
            ))
        insights = await consolidate(facts, threshold=0.99)
        # All singletons
        assert len(insights) == 3
