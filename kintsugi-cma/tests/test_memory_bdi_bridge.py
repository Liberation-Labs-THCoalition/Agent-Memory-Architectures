"""Tests for kintsugi.memory.bdi_bridge module."""

from __future__ import annotations

import hashlib

import pytest

from kintsugi.memory.bdi_bridge import (
    BDIBridge,
    Belief,
    Desire,
    Intention,
)


# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


class TestBelief:
    def test_valid(self):
        b = Belief(id="x", content="sky is blue", confidence=0.8)
        assert b.confidence == 0.8
        assert b.source_memory_ids == []
        assert b.tags == []

    def test_confidence_too_high(self):
        with pytest.raises(ValueError, match="confidence must be 0-1"):
            Belief(id="x", content="a", confidence=1.5)

    def test_confidence_too_low(self):
        with pytest.raises(ValueError, match="confidence must be 0-1"):
            Belief(id="x", content="a", confidence=-0.1)

    def test_boundary_values(self):
        Belief(id="x", content="a", confidence=0.0)
        Belief(id="x", content="a", confidence=1.0)


class TestDesire:
    def test_valid(self):
        d = Desire(id="d1", description="grow", priority=0.5)
        assert d.boost_factor == 1.3
        assert d.decay_factor == 0.9

    def test_priority_invalid(self):
        with pytest.raises(ValueError, match="priority must be 0-1"):
            Desire(id="d1", description="x", priority=2.0)

    def test_priority_negative(self):
        with pytest.raises(ValueError):
            Desire(id="d1", description="x", priority=-0.1)


class TestIntention:
    def test_valid(self):
        i = Intention(id="i1", goal="launch", status="active")
        assert i.priority_boost == 0.2

    def test_invalid_status(self):
        with pytest.raises(ValueError, match="status must be one of"):
            Intention(id="i1", goal="x", status="invalid")

    @pytest.mark.parametrize("status", ["active", "completed", "suspended"])
    def test_valid_statuses(self, status):
        Intention(id="i1", goal="x", status=status)


# ---------------------------------------------------------------------------
# BDIBridge helpers
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        h1 = BDIBridge._content_hash("hello")
        h2 = BDIBridge._content_hash("hello")
        assert h1 == h2

    def test_different_content(self):
        assert BDIBridge._content_hash("a") != BDIBridge._content_hash("b")

    def test_length(self):
        assert len(BDIBridge._content_hash("test")) == 16


# ---------------------------------------------------------------------------
# extract_beliefs
# ---------------------------------------------------------------------------


class TestExtractBeliefs:
    def setup_method(self):
        self.bridge = BDIBridge()

    def _mem(self, id, content, significance=5, tags=None):
        return {"id": id, "content": content, "significance": significance, "tags": tags or []}

    def test_empty_input(self):
        assert self.bridge.extract_beliefs([]) == []

    def test_below_threshold(self):
        mems = [self._mem("1", "x", significance=1)]
        assert self.bridge.extract_beliefs(mems, min_significance=5) == []

    def test_basic_extraction(self):
        mems = [self._mem("1", "sky is blue", significance=8)]
        beliefs = self.bridge.extract_beliefs(mems)
        assert len(beliefs) == 1
        assert beliefs[0].confidence == 0.8
        assert beliefs[0].source_memory_ids == ["1"]

    def test_dedup_corroboration(self):
        mems = [
            self._mem("1", "sky is blue", significance=8),
            self._mem("2", "sky is blue", significance=7),
        ]
        beliefs = self.bridge.extract_beliefs(mems)
        assert len(beliefs) == 1
        assert "1" in beliefs[0].source_memory_ids
        assert "2" in beliefs[0].source_memory_ids
        assert abs(beliefs[0].confidence - 0.85) < 1e-9

    def test_sorted_by_confidence_desc(self):
        mems = [
            self._mem("1", "low", significance=3),
            self._mem("2", "high", significance=9),
        ]
        beliefs = self.bridge.extract_beliefs(mems)
        assert beliefs[0].confidence > beliefs[1].confidence

    def test_tags_inherited(self):
        mems = [self._mem("1", "x", significance=5, tags=["a", "b"])]
        beliefs = self.bridge.extract_beliefs(mems)
        assert set(beliefs[0].tags) == {"a", "b"}

    def test_confidence_capped_at_1(self):
        # 25 identical memories: 0.5 + 24*0.05 = 1.7 -> capped at 1.0
        mems = [self._mem(str(i), "same", significance=5) for i in range(25)]
        beliefs = self.bridge.extract_beliefs(mems)
        assert beliefs[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# apply_desire_bias
# ---------------------------------------------------------------------------


class TestApplyDesireBias:
    def setup_method(self):
        self.bridge = BDIBridge()

    def _mem(self, id="1", score=1.0, tags=None):
        return {"id": id, "content": "x", "score": score, "tags": tags or []}

    def test_no_desires(self):
        mems = [self._mem()]
        result = self.bridge.apply_desire_bias(mems, [])
        assert len(result) == 1
        assert result[0]["score"] == 1.0

    def test_aligned_boost(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["important"], boost_factor=2.0)
        mems = [self._mem(tags=["important"], score=1.0)]
        result = self.bridge.apply_desire_bias(mems, [desire])
        assert result[0]["score"] > 1.0
        assert result[0]["_desire_aligned"] is True

    def test_unaligned_decay(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["other"], decay_factor=0.5)
        mems = [self._mem(tags=["unrelated"], score=1.0)]
        result = self.bridge.apply_desire_bias(mems, [desire])
        assert result[0]["score"] < 1.0
        assert result[0]["_desire_aligned"] is False

    def test_sorted_by_score_desc(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["hot"], boost_factor=5.0)
        mems = [
            self._mem(id="low", score=1.0, tags=["cold"]),
            self._mem(id="high", score=0.5, tags=["hot"]),
        ]
        result = self.bridge.apply_desire_bias(mems, [desire])
        assert result[0]["id"] == "high"

    def test_no_mutation(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["a"])
        mem = self._mem(tags=["a"])
        self.bridge.apply_desire_bias([mem], [desire])
        assert "_desire_aligned" not in mem

    def test_score_none_treated_as_zero(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["a"])
        mem = {"id": "1", "content": "x", "score": None, "tags": ["a"]}
        result = self.bridge.apply_desire_bias([mem], [desire])
        assert result[0]["score"] == 0.0

    def test_empty_memories(self):
        desire = Desire(id="d1", description="x", priority=1.0, related_tags=["a"])
        assert self.bridge.apply_desire_bias([], [desire]) == []


# ---------------------------------------------------------------------------
# prioritize_by_intentions
# ---------------------------------------------------------------------------


class TestPrioritizeByIntentions:
    def setup_method(self):
        self.bridge = BDIBridge()

    def _mem(self, id="1", content="hello", score=1.0):
        return {"id": id, "content": content, "score": score}

    def test_no_active_intentions(self):
        intention = Intention(id="i1", goal="x", status="completed")
        mems = [self._mem()]
        result = self.bridge.prioritize_by_intentions(mems, [intention])
        assert len(result) == 1
        # No active intentions -> returns copy without modification
        assert result[0]["score"] == 1.0

    def test_no_intentions(self):
        mems = [self._mem()]
        result = self.bridge.prioritize_by_intentions(mems, [])
        assert result[0]["score"] == 1.0

    def test_boost_by_content_hash(self):
        content = "hello"
        content_hash = BDIBridge._content_hash(content)
        intention = Intention(
            id="i1", goal="x", status="active",
            belief_ids=[content_hash], priority_boost=0.5,
        )
        mems = [self._mem(content=content, score=1.0)]
        result = self.bridge.prioritize_by_intentions(mems, [intention])
        assert result[0]["score"] == 1.5
        assert result[0]["_intention_boosted"] is True

    def test_boost_by_memory_id(self):
        intention = Intention(
            id="i1", goal="x", status="active",
            belief_ids=["mem-1"], priority_boost=0.3,
        )
        mems = [self._mem(id="mem-1", score=1.0)]
        result = self.bridge.prioritize_by_intentions(mems, [intention])
        assert result[0]["score"] > 1.0
        assert result[0]["_intention_boosted"] is True

    def test_sorted_by_score(self):
        content_hash = BDIBridge._content_hash("special")
        intention = Intention(
            id="i1", goal="x", status="active",
            belief_ids=[content_hash], priority_boost=10.0,
        )
        mems = [
            self._mem(id="1", content="normal", score=5.0),
            self._mem(id="2", content="special", score=1.0),
        ]
        result = self.bridge.prioritize_by_intentions(mems, [intention])
        assert result[0]["id"] == "2"

    def test_empty_results(self):
        intention = Intention(id="i1", goal="x", status="active", belief_ids=["b1"])
        assert self.bridge.prioritize_by_intentions([], [intention]) == []

    def test_no_mutation(self):
        intention = Intention(id="i1", goal="x", status="active", belief_ids=["b1"])
        mem = self._mem()
        self.bridge.prioritize_by_intentions([mem], [intention])
        assert "_intention_boosted" not in mem


# ---------------------------------------------------------------------------
# process_pipeline
# ---------------------------------------------------------------------------


class TestProcessPipeline:
    def test_full_pipeline(self):
        bridge = BDIBridge()
        mems = [
            {"id": "1", "content": "important fact", "significance": 8, "tags": ["ai"], "score": 1.0},
            {"id": "2", "content": "trivial", "significance": 1, "tags": [], "score": 0.5},
        ]
        desires = [Desire(id="d1", description="focus on AI", priority=0.8, related_tags=["ai"])]
        intentions = [Intention(id="i1", goal="build AI", status="active", belief_ids=[])]

        beliefs, ranked = bridge.process_pipeline(mems, desires, intentions)
        assert len(beliefs) >= 1
        assert all(b.confidence > 0 for b in beliefs)
        assert len(ranked) == 2

    def test_pipeline_empty(self):
        bridge = BDIBridge()
        beliefs, ranked = bridge.process_pipeline([], [], [])
        assert beliefs == []
        assert ranked == []
