"""Tests for CMA Stage 1 — Semantic Structured Compression."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
import pytest_asyncio

from kintsugi.memory.cma_stage1 import (
    AtomicFact,
    Stage1Result,
    Turn,
    Window,
    _cosine_similarity,
    _window_text,
    filter_windows,
    normalize_window,
    run_stage1,
    score_entropy,
    segment_dialogue,
)
from kintsugi.memory.embeddings import EmbeddingProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


class MockEmbeddingProvider(EmbeddingProvider):
    """Returns deterministic embeddings based on text hash."""

    @property
    def dimension(self) -> int:
        return DIM

    async def embed(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


class ConstantEmbeddingProvider(EmbeddingProvider):
    """Always returns the same vector."""

    def __init__(self, vec: np.ndarray | None = None):
        self._vec = vec if vec is not None else np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)

    @property
    def dimension(self) -> int:
        return DIM

    async def embed(self, text: str) -> np.ndarray:
        return self._vec.copy()

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec.copy() for _ in texts]


def _make_turns(n: int, base_time: datetime | None = None) -> list[Turn]:
    base = base_time or datetime(2025, 1, 1)
    return [
        Turn(role="user" if i % 2 == 0 else "assistant",
             content=f"Message {i}",
             timestamp=base + timedelta(minutes=i))
        for i in range(n)
    ]


def _valid_llm_call(system: str, user: str) -> str:
    """LLM mock that returns valid JSON for atomic fact extraction."""
    if "fact extraction" in system.lower():
        return json.dumps([
            {"content": "Fact A from window", "entities": ["Alice"]},
            {"content": "Fact B from window", "entities": ["Bob"]},
        ])
    return user  # coreference / timestamp steps just pass through


# ---------------------------------------------------------------------------
# segment_dialogue
# ---------------------------------------------------------------------------


class TestSegmentDialogue:
    def test_empty_turns(self):
        assert segment_dialogue([]) == []

    def test_fewer_turns_than_window(self):
        turns = _make_turns(3)
        windows = segment_dialogue(turns, window_size=10, stride=5)
        assert len(windows) == 1
        assert windows[0].start_idx == 0
        assert windows[0].end_idx == 3
        assert len(windows[0].turns) == 3

    def test_exact_window_size(self):
        turns = _make_turns(10)
        windows = segment_dialogue(turns, window_size=10, stride=5)
        # First window covers all 10 turns, end==len so breaks immediately
        assert len(windows) == 1
        assert windows[0].end_idx == 10

    def test_overlapping_windows(self):
        turns = _make_turns(20)
        windows = segment_dialogue(turns, window_size=10, stride=5)
        # 0-10, 5-15, 10-20
        assert len(windows) == 3
        assert windows[0].start_idx == 0
        assert windows[1].start_idx == 5
        assert windows[2].start_idx == 10

    def test_stride_equals_window(self):
        turns = _make_turns(20)
        windows = segment_dialogue(turns, window_size=10, stride=10)
        assert len(windows) == 2

    def test_single_turn(self):
        turns = _make_turns(1)
        windows = segment_dialogue(turns, window_size=10, stride=5)
        assert len(windows) == 1
        assert windows[0].end_idx == 1

    def test_stride_one(self):
        turns = _make_turns(5)
        windows = segment_dialogue(turns, window_size=3, stride=1)
        assert len(windows) == 3  # 0-3, 1-4, 2-5


# ---------------------------------------------------------------------------
# _window_text / _cosine_similarity
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_window_text(self):
        turns = _make_turns(2)
        w = Window(turns=turns, start_idx=0, end_idx=2)
        text = _window_text(w)
        assert "user:" in text
        assert "assistant:" in text

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# score_entropy
# ---------------------------------------------------------------------------


class TestScoreEntropy:
    @pytest.mark.asyncio
    async def test_first_window_entropy_is_one(self):
        turns = _make_turns(3)
        w = Window(turns=turns, start_idx=0, end_idx=3)
        provider = MockEmbeddingProvider()
        score = await score_entropy(w, prev_embedding=None, embedding_provider=provider)
        assert score == 1.0
        assert w.entropy_score == 1.0
        assert w.embedding is not None

    @pytest.mark.asyncio
    async def test_identical_windows_low_entropy(self):
        turns = _make_turns(3)
        w1 = Window(turns=turns, start_idx=0, end_idx=3)
        w2 = Window(turns=turns, start_idx=0, end_idx=3)
        provider = ConstantEmbeddingProvider()
        await score_entropy(w1, prev_embedding=None, embedding_provider=provider)
        score = await score_entropy(w2, prev_embedding=w1.embedding, embedding_provider=provider)
        assert score == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_different_windows_positive_entropy(self):
        turns1 = _make_turns(3)
        turns2 = [Turn("user", "Completely different topic xyz", datetime(2025, 6, 1))]
        w1 = Window(turns=turns1, start_idx=0, end_idx=3)
        w2 = Window(turns=turns2, start_idx=3, end_idx=4)
        provider = MockEmbeddingProvider()
        await score_entropy(w1, None, provider)
        score = await score_entropy(w2, w1.embedding, provider)
        assert 0.0 < score <= 2.0  # 1 - cosine can be up to 2 for opposing vectors


# ---------------------------------------------------------------------------
# filter_windows
# ---------------------------------------------------------------------------


class TestFilterWindows:
    def test_empty(self):
        retained, archived = filter_windows([])
        assert retained == []
        assert archived == []

    def test_all_retained(self):
        w = Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.5)
        retained, archived = filter_windows([w], threshold=0.3)
        assert len(retained) == 1
        assert len(archived) == 0

    def test_all_archived(self):
        w = Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.1)
        retained, archived = filter_windows([w], threshold=0.5)
        assert len(retained) == 0
        assert len(archived) == 1

    def test_mixed(self):
        windows = [
            Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.9),
            Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.1),
            Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.5),
        ]
        retained, archived = filter_windows(windows, threshold=0.5)
        assert len(retained) == 2
        assert len(archived) == 1

    def test_none_entropy_treated_as_zero(self):
        w = Window(turns=[], start_idx=0, end_idx=0, entropy_score=None)
        retained, archived = filter_windows([w], threshold=0.1)
        assert len(archived) == 1

    def test_exact_threshold_retained(self):
        w = Window(turns=[], start_idx=0, end_idx=0, entropy_score=0.35)
        retained, archived = filter_windows([w], threshold=0.35)
        assert len(retained) == 1


# ---------------------------------------------------------------------------
# normalize_window
# ---------------------------------------------------------------------------


class TestNormalizeWindow:
    @pytest.mark.asyncio
    async def test_valid_json_extraction(self):
        turns = _make_turns(2)
        w = Window(turns=turns, start_idx=0, end_idx=2)
        facts = await normalize_window(w, _valid_llm_call)
        assert len(facts) == 2
        assert all(isinstance(f, AtomicFact) for f in facts)
        assert facts[0].entities == ["Alice"]
        assert facts[0].source_window_idx == 0

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self):
        def llm(sys, usr):
            if "fact extraction" in sys.lower():
                return '```json\n[{"content": "fenced fact", "entities": []}]\n```'
            return usr

        turns = _make_turns(2)
        w = Window(turns=turns, start_idx=0, end_idx=2)
        facts = await normalize_window(w, llm)
        assert len(facts) == 1
        assert facts[0].content == "fenced fact"

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        def llm(sys, usr):
            if "fact extraction" in sys.lower():
                return "not valid json at all {"
            return usr

        turns = _make_turns(2)
        w = Window(turns=turns, start_idx=5, end_idx=7)
        facts = await normalize_window(w, llm)
        assert len(facts) == 1
        assert facts[0].source_window_idx == 5

    @pytest.mark.asyncio
    async def test_non_dict_items_skipped(self):
        def llm(sys, usr):
            if "fact extraction" in sys.lower():
                return json.dumps([{"content": "ok", "entities": []}, "not a dict", 42])
            return usr

        turns = _make_turns(2)
        w = Window(turns=turns, start_idx=0, end_idx=2)
        facts = await normalize_window(w, llm)
        assert len(facts) == 1

    @pytest.mark.asyncio
    async def test_empty_turns_uses_utcnow(self):
        w = Window(turns=[], start_idx=0, end_idx=0)

        def llm(sys, usr):
            if "fact extraction" in sys.lower():
                return json.dumps([{"content": "fact", "entities": []}])
            return usr

        facts = await normalize_window(w, llm)
        assert len(facts) == 1
        # timestamp should be close to now
        assert (datetime.utcnow() - facts[0].timestamp).total_seconds() < 5


# ---------------------------------------------------------------------------
# run_stage1 (full pipeline)
# ---------------------------------------------------------------------------


class TestRunStage1:
    @pytest.mark.asyncio
    async def test_empty_turns(self):
        result = await run_stage1([], MockEmbeddingProvider(), _valid_llm_call)
        assert isinstance(result, Stage1Result)
        assert result.retained_facts == []
        assert result.archived_windows == []
        assert result.retained_windows == []

    @pytest.mark.asyncio
    async def test_basic_pipeline(self):
        turns = _make_turns(12)
        provider = MockEmbeddingProvider()
        result = await run_stage1(
            turns, provider, _valid_llm_call,
            window_size=5, stride=5, threshold=0.0,
        )
        # threshold=0 means all windows retained
        assert len(result.retained_facts) > 0
        assert len(result.archived_windows) == 0

    @pytest.mark.asyncio
    async def test_high_threshold_archives_most(self):
        turns = _make_turns(5)
        provider = ConstantEmbeddingProvider()
        result = await run_stage1(
            turns, provider, _valid_llm_call,
            window_size=3, stride=2, threshold=0.99,
        )
        # First window gets entropy=1.0 (no predecessor), rest ~0.0
        assert len(result.retained_windows) == 1
        assert len(result.archived_windows) >= 1

    @pytest.mark.asyncio
    async def test_single_turn(self):
        turns = _make_turns(1)
        provider = MockEmbeddingProvider()
        result = await run_stage1(
            turns, provider, _valid_llm_call,
            window_size=10, stride=5, threshold=0.0,
        )
        assert len(result.retained_windows) == 1
