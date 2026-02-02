"""CMA Stage 1 — Semantic Structured Compression.

Implements the SimpleMem pipeline (arXiv:2601.02553):
  1. Sliding-window dialogue segmentation
  2. Entropy-based filtering (cosine distance from predecessor)
  3. Normalization: coreference resolution, timestamp anchoring, atomic-fact extraction

The module is model-agnostic: an ``llm_call`` callable is injected for all
generative steps, and an :class:`EmbeddingProvider` handles vector encoding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kintsugi.memory.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single dialogue turn."""

    role: str
    content: str
    timestamp: datetime


@dataclass
class Window:
    """A contiguous slice of conversation turns."""

    turns: list[Turn]
    start_idx: int
    end_idx: int
    embedding: NDArray[np.float32] | None = None
    entropy_score: float | None = None


@dataclass
class AtomicFact:
    """An independent factual statement extracted from a window."""

    content: str
    source_window_idx: int
    timestamp: datetime
    entities: list[str] = field(default_factory=list)


@dataclass
class Stage1Result:
    """Output of the full Stage 1 pipeline."""

    retained_facts: list[AtomicFact]
    archived_windows: list[Window]
    retained_windows: list[Window]


# ---------------------------------------------------------------------------
# 1. Sliding-window segmentation
# ---------------------------------------------------------------------------


def segment_dialogue(
    turns: list[Turn],
    window_size: int = 10,
    stride: int = 5,
) -> list[Window]:
    """Split dialogue into overlapping windows (50% default overlap).

    Args:
        turns: Ordered list of conversation turns.
        window_size: Number of turns per window.
        stride: Step size between windows.

    Returns:
        List of :class:`Window` objects covering the entire dialogue.
    """
    if not turns:
        return []

    windows: list[Window] = []
    for start in range(0, len(turns), stride):
        end = min(start + window_size, len(turns))
        windows.append(
            Window(
                turns=turns[start:end],
                start_idx=start,
                end_idx=end,
            )
        )
        if end == len(turns):
            break
    return windows


# ---------------------------------------------------------------------------
# 2. Entropy scoring
# ---------------------------------------------------------------------------


def _window_text(window: Window) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in window.turns)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return dot / norm


async def score_entropy(
    window: Window,
    prev_embedding: NDArray[np.float32] | None,
    embedding_provider: EmbeddingProvider,
) -> float:
    """Compute semantic entropy for *window*.

    Entropy is defined as ``1 - cosine_similarity(current, previous)``.
    The first window (no predecessor) receives an entropy of 1.0.

    Side-effect: sets ``window.embedding`` and ``window.entropy_score``.
    """
    text = _window_text(window)
    window.embedding = await embedding_provider.embed(text)

    if prev_embedding is None:
        window.entropy_score = 1.0
    else:
        sim = _cosine_similarity(window.embedding, prev_embedding)
        window.entropy_score = 1.0 - sim

    return window.entropy_score


# ---------------------------------------------------------------------------
# 3. Entropy-based filtering
# ---------------------------------------------------------------------------


def filter_windows(
    windows: list[Window],
    threshold: float = 0.35,
) -> tuple[list[Window], list[Window]]:
    """Partition windows into retained (high-entropy) and archived (low-entropy).

    Args:
        windows: Windows with ``entropy_score`` already computed.
        threshold: Minimum entropy to be retained.

    Returns:
        ``(retained, archived)`` tuple.
    """
    retained: list[Window] = []
    archived: list[Window] = []
    for w in windows:
        score = w.entropy_score if w.entropy_score is not None else 0.0
        if score >= threshold:
            retained.append(w)
        else:
            archived.append(w)
    return retained, archived


# ---------------------------------------------------------------------------
# 4. Window normalization (LLM-assisted)
# ---------------------------------------------------------------------------

# Type alias: llm_call(system_prompt, user_prompt) -> response_text
LLMCall = Callable[[str, str], str]

_COREFERENCE_SYSTEM = (
    "You are a coreference resolution engine. Replace all pronouns and ambiguous "
    "references in the following dialogue with the explicit named entities they refer to. "
    "Return only the resolved text, nothing else."
)

_TIMESTAMP_SYSTEM = (
    "You are a timestamp normalization engine. Convert all relative time expressions "
    "(e.g., 'yesterday', 'last week', 'in two hours') to absolute ISO-8601 timestamps "
    "based on the reference time provided. Return only the normalized text."
)

_ATOMIC_SYSTEM = (
    "You are a fact extraction engine. Break the following text into a JSON array of "
    "independent atomic facts. Each fact should be a JSON object with keys: "
    '"content" (string — the fact), "entities" (array of strings — named entities). '
    "Return ONLY the JSON array, no markdown fencing."
)


async def normalize_window(
    window: Window,
    llm_call: LLMCall,
) -> list[AtomicFact]:
    """Full normalization pipeline for a single window.

    Steps:
      a. Coreference resolution
      b. Timestamp anchoring
      c. Atomic fact extraction

    Args:
        window: The window to normalize.
        llm_call: ``(system_prompt, user_prompt) -> response_text``.

    Returns:
        List of :class:`AtomicFact` objects.
    """
    text = _window_text(window)
    reference_time = window.turns[-1].timestamp if window.turns else datetime.utcnow()

    # a. Coreference resolution
    resolved = llm_call(_COREFERENCE_SYSTEM, text)

    # b. Timestamp anchoring
    ts_prompt = f"Reference time: {reference_time.isoformat()}\n\nText:\n{resolved}"
    anchored = llm_call(_TIMESTAMP_SYSTEM, ts_prompt)

    # c. Atomic fact extraction
    raw_facts = llm_call(_ATOMIC_SYSTEM, anchored)

    # Parse JSON response
    try:
        facts_data = json.loads(raw_facts)
    except json.JSONDecodeError:
        # Try stripping markdown code fences if present
        stripped = raw_facts.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            facts_data = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse atomic facts JSON for window %d-%d; "
                "falling back to single fact.",
                window.start_idx,
                window.end_idx,
            )
            facts_data = [{"content": anchored, "entities": []}]

    atomic_facts: list[AtomicFact] = []
    for item in facts_data:
        if isinstance(item, dict):
            atomic_facts.append(
                AtomicFact(
                    content=item.get("content", str(item)),
                    source_window_idx=window.start_idx,
                    timestamp=reference_time,
                    entities=item.get("entities", []),
                )
            )

    return atomic_facts


# ---------------------------------------------------------------------------
# 5. Full Stage 1 pipeline
# ---------------------------------------------------------------------------


async def run_stage1(
    turns: list[Turn],
    embedding_provider: EmbeddingProvider,
    llm_call: LLMCall,
    window_size: int = 10,
    stride: int = 5,
    threshold: float = 0.35,
) -> Stage1Result:
    """Execute the complete CMA Stage 1 pipeline.

    1. Segment dialogue into overlapping windows.
    2. Score entropy for each window.
    3. Filter windows by entropy threshold.
    4. Normalize retained windows into atomic facts.

    Returns:
        :class:`Stage1Result` with retained facts and archived windows.
    """
    # 1. Segment
    windows = segment_dialogue(turns, window_size=window_size, stride=stride)
    if not windows:
        return Stage1Result(retained_facts=[], archived_windows=[], retained_windows=[])

    # 2. Score entropy
    prev_embedding: NDArray[np.float32] | None = None
    for w in windows:
        await score_entropy(w, prev_embedding, embedding_provider)
        prev_embedding = w.embedding

    # 3. Filter
    retained, archived = filter_windows(windows, threshold=threshold)

    # 4. Normalize retained windows -> atomic facts
    all_facts: list[AtomicFact] = []
    for w in retained:
        facts = await normalize_window(w, llm_call)
        all_facts.extend(facts)

    logger.info(
        "Stage1 complete: %d windows -> %d retained, %d archived, %d facts",
        len(windows),
        len(retained),
        len(archived),
        len(all_facts),
    )

    return Stage1Result(
        retained_facts=all_facts,
        archived_windows=archived,
        retained_windows=retained,
    )
