"""Shadow verification -- comparing primary and shadow outputs.

The Verifier scores safety, quality, and alignment, then computes a
behavioural divergence metric (SWEI) to decide whether a modification
should be approved, rejected, extended, or escalated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class VerifierVerdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    EXTEND = "EXTEND"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class VerificationResult:
    """Immutable outcome of a verification pass."""

    verdict: VerifierVerdict
    safety_passed: bool
    quality_score: float
    alignment_score: float
    swei_divergence: float
    rationale: str
    checked_at: datetime


@dataclass
class VerifierConfig:
    """Tuning knobs for the Verifier."""

    divergence_threshold: float = 0.15
    min_quality_score: float = 0.6
    safety_weight: float = 0.4
    quality_weight: float = 0.3
    alignment_weight: float = 0.3
    extend_window_turns: int = 5


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Compare primary and shadow outputs to decide on modification fate.

    Accepts an optional ``InvariantChecker`` for hard safety checks.
    All scoring methods are pure functions over plain dicts for easy
    unit testing.
    """

    def __init__(
        self,
        config: VerifierConfig | None = None,
        invariant_checker: Any = None,
    ) -> None:
        self._config = config or VerifierConfig()
        self._invariant_checker = invariant_checker

    # -- public API ---------------------------------------------------------

    def verify(
        self,
        primary_outputs: List[Dict[str, Any]],
        shadow_outputs: List[Dict[str, Any]],
        bdi_context: Dict[str, Any] | None = None,
        invariant_context: Any = None,
    ) -> VerificationResult:
        """Run full verification pipeline and return a verdict.

        Parameters
        ----------
        primary_outputs:
            Collected outputs from the primary agent.
        shadow_outputs:
            Collected outputs from the shadow fork.
        bdi_context:
            Optional beliefs/desires/intentions dict for alignment scoring.
        invariant_context:
            Optional ``InvariantContext`` for hard safety checks.
        """
        cfg = self._config
        rationale_parts: List[str] = []

        # 1. Safety check
        safety_passed = True
        if invariant_context is not None and self._invariant_checker is not None:
            inv_result = self._invariant_checker.check_all(invariant_context)
            safety_passed = inv_result.all_passed
            if not safety_passed:
                rationale = f"Invariant failures: {inv_result.failures}"
                logger.warning("Verification REJECT: %s", rationale)
                return VerificationResult(
                    verdict=VerifierVerdict.REJECT,
                    safety_passed=False,
                    quality_score=0.0,
                    alignment_score=0.0,
                    swei_divergence=1.0,
                    rationale=rationale,
                    checked_at=datetime.now(timezone.utc),
                )
            rationale_parts.append("safety: passed")
        else:
            rationale_parts.append("safety: skipped (no context)")

        # 2. Quality check
        quality = self._compute_quality(primary_outputs, shadow_outputs)
        rationale_parts.append(f"quality: {quality:.3f}")

        # 3. Alignment check
        alignment = self._compute_alignment(shadow_outputs, bdi_context) if bdi_context else 1.0
        rationale_parts.append(f"alignment: {alignment:.3f}")

        # 4. SWEI divergence
        swei = self._compute_swei(primary_outputs, shadow_outputs)
        rationale_parts.append(f"swei: {swei:.3f}")

        # 5. Verdict logic
        threshold = cfg.divergence_threshold
        if swei > threshold * 2:
            verdict = VerifierVerdict.ESCALATE
            rationale_parts.append("verdict: ESCALATE (swei > 2x threshold)")
        elif swei > threshold:
            verdict = VerifierVerdict.EXTEND
            rationale_parts.append("verdict: EXTEND (swei > threshold)")
        elif quality < cfg.min_quality_score:
            verdict = VerifierVerdict.REJECT
            rationale_parts.append("verdict: REJECT (low quality)")
        else:
            verdict = VerifierVerdict.APPROVE
            rationale_parts.append("verdict: APPROVE")

        rationale = "; ".join(rationale_parts)
        logger.info("Verification %s: %s", verdict.value, rationale)

        return VerificationResult(
            verdict=verdict,
            safety_passed=safety_passed,
            quality_score=quality,
            alignment_score=alignment,
            swei_divergence=swei,
            rationale=rationale,
            checked_at=datetime.now(timezone.utc),
        )

    # -- scoring helpers ----------------------------------------------------

    def _compute_swei(
        self,
        primary_outputs: List[Dict[str, Any]],
        shadow_outputs: List[Dict[str, Any]],
    ) -> float:
        """Structural/behavioural divergence between primary and shadow.

        Combines three signals:
        - Output count difference
        - Aggregate text length ratio
        - Tool call pattern divergence (key overlap)

        Returns a float in [0, 1].
        """
        if not primary_outputs and not shadow_outputs:
            return 0.0

        scores: List[float] = []

        # Count divergence
        p_count = max(len(primary_outputs), 1)
        s_count = max(len(shadow_outputs), 1)
        count_div = abs(p_count - s_count) / max(p_count, s_count)
        scores.append(count_div)

        # Text length divergence
        p_len = sum(len(str(o)) for o in primary_outputs) or 1
        s_len = sum(len(str(o)) for o in shadow_outputs) or 1
        len_ratio = min(p_len, s_len) / max(p_len, s_len)
        scores.append(1.0 - len_ratio)

        # Key overlap divergence
        p_keys = _collect_keys(primary_outputs)
        s_keys = _collect_keys(shadow_outputs)
        if p_keys or s_keys:
            union = p_keys | s_keys
            intersection = p_keys & s_keys
            key_div = 1.0 - (len(intersection) / len(union)) if union else 0.0
            scores.append(key_div)

        return min(sum(scores) / len(scores), 1.0) if scores else 0.0

    def _compute_quality(
        self,
        primary_outputs: List[Dict[str, Any]],
        shadow_outputs: List[Dict[str, Any]],
    ) -> float:
        """Similarity between primary and shadow outputs.

        Higher means more similar (shadow behaves like primary).
        Returns float in [0, 1].
        """
        if not primary_outputs and not shadow_outputs:
            return 1.0
        if not primary_outputs or not shadow_outputs:
            return 0.0

        p_text = " ".join(str(o) for o in primary_outputs)
        s_text = " ".join(str(o) for o in shadow_outputs)

        # Simple word overlap (Jaccard)
        p_words = set(p_text.lower().split())
        s_words = set(s_text.lower().split())
        if not p_words and not s_words:
            return 1.0
        union = p_words | s_words
        intersection = p_words & s_words
        return len(intersection) / len(union) if union else 1.0

    def _compute_alignment(
        self,
        outputs: List[Dict[str, Any]],
        bdi_context: Dict[str, Any] | None,
    ) -> float:
        """Keyword overlap between outputs and BDI beliefs/desires/intentions.

        Returns float in [0, 1].
        """
        if not bdi_context or not outputs:
            return 1.0

        # Extract keywords from BDI values
        bdi_words: set[str] = set()
        for value in bdi_context.values():
            if isinstance(value, str):
                bdi_words.update(value.lower().split())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    bdi_words.update(str(item).lower().split())

        if not bdi_words:
            return 1.0

        output_text = " ".join(str(o) for o in outputs).lower()
        output_words = set(output_text.split())

        overlap = bdi_words & output_words
        return len(overlap) / len(bdi_words) if bdi_words else 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_keys(outputs: List[Dict[str, Any]]) -> set[str]:
    """Collect all dictionary keys from a list of output dicts."""
    keys: set[str] = set()
    for o in outputs:
        if isinstance(o, dict):
            keys.update(o.keys())
    return keys
