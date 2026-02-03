"""BDI drift classification from coherence score changes."""

from dataclasses import dataclass
from typing import List

from .coherence import CoherenceScore


@dataclass(frozen=True)
class DriftClassification:
    category: str  # healthy_adaptation | stale_beliefs | intention_drift | values_tension
    confidence: float
    evidence: tuple  # frozen needs immutable
    recommendation: str


_VALID_CATEGORIES = {
    "healthy_adaptation",
    "stale_beliefs",
    "intention_drift",
    "values_tension",
}


class BDIDriftClassifier:
    """Classifies drift from coherence score deltas."""

    def __init__(self) -> None:
        pass

    def classify(
        self,
        coherence_before: CoherenceScore,
        coherence_after: CoherenceScore,
        time_delta_days: float,
    ) -> DriftClassification:
        overall_delta = coherence_after.overall - coherence_before.overall
        bd_delta = (
            coherence_after.belief_desire_alignment
            - coherence_before.belief_desire_alignment
        )
        di_delta = (
            coherence_after.desire_intention_alignment
            - coherence_before.desire_intention_alignment
        )
        bi_delta = (
            coherence_after.belief_intention_alignment
            - coherence_before.belief_intention_alignment
        )

        new_issues = set(coherence_after.issues) - set(coherence_before.issues)
        evidence_items: List[str] = []

        # Healthy adaptation: overall improved or stable, no new issues
        if overall_delta >= 0 and not new_issues:
            evidence_items.append(f"Overall coherence delta: +{overall_delta:.4f}")
            return DriftClassification(
                category="healthy_adaptation",
                confidence=min(1.0, 0.6 + overall_delta),
                evidence=tuple(evidence_items),
                recommendation="Continue current trajectory. BDI coherence is stable or improving.",
            )

        # Stale beliefs: belief-related scores dropped + large time delta
        if (bd_delta < -0.05 or bi_delta < -0.05) and time_delta_days > 60:
            evidence_items.append(f"Belief-desire delta: {bd_delta:.4f}")
            evidence_items.append(f"Belief-intention delta: {bi_delta:.4f}")
            evidence_items.append(f"Time since last check: {time_delta_days:.0f} days")
            return DriftClassification(
                category="stale_beliefs",
                confidence=min(1.0, 0.5 + abs(bd_delta) + abs(bi_delta)),
                evidence=tuple(evidence_items),
                recommendation="Schedule a belief review session. Several beliefs may be outdated.",
            )

        # Intention drift: intention alignment dropped
        if di_delta < -0.05 or bi_delta < -0.05:
            evidence_items.append(f"Desire-intention delta: {di_delta:.4f}")
            evidence_items.append(f"Belief-intention delta: {bi_delta:.4f}")
            return DriftClassification(
                category="intention_drift",
                confidence=min(1.0, 0.5 + abs(di_delta) + abs(bi_delta)),
                evidence=tuple(evidence_items),
                recommendation="Review active intentions. Some may no longer serve current desires or beliefs.",
            )

        # Values tension: desire alignment dropped
        if bd_delta < -0.05:
            evidence_items.append(f"Belief-desire delta: {bd_delta:.4f}")
            return DriftClassification(
                category="values_tension",
                confidence=min(1.0, 0.5 + abs(bd_delta)),
                evidence=tuple(evidence_items),
                recommendation="Facilitate a values alignment session. Desires may have shifted away from core beliefs.",
            )

        # Default fallback
        evidence_items.append(f"Overall delta: {overall_delta:.4f}")
        if new_issues:
            evidence_items.append(f"New issues: {len(new_issues)}")
        return DriftClassification(
            category="values_tension",
            confidence=0.4,
            evidence=tuple(evidence_items),
            recommendation="Review BDI coherence. Minor tensions detected across layers.",
        )

    def classify_from_events(self, events: List[dict]) -> DriftClassification:
        """Classify drift from a list of drift event dicts."""
        if not events:
            return DriftClassification(
                category="healthy_adaptation",
                confidence=0.5,
                evidence=("No events provided.",),
                recommendation="No drift events to classify.",
            )

        category_counts: dict = {}
        evidence_items: List[str] = []
        for event in events:
            cat = event.get("category", "values_tension")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            desc = event.get("description", "")
            if desc:
                evidence_items.append(desc)

        # Pick the most frequent category
        dominant = max(category_counts, key=lambda k: category_counts[k])
        if dominant not in _VALID_CATEGORIES:
            dominant = "values_tension"

        total = sum(category_counts.values())
        confidence = category_counts[dominant] / total if total else 0.5

        recommendations = {
            "healthy_adaptation": "Continue current trajectory.",
            "stale_beliefs": "Schedule a belief review session.",
            "intention_drift": "Review active intentions for relevance.",
            "values_tension": "Facilitate a values alignment session.",
        }

        return DriftClassification(
            category=dominant,
            confidence=round(confidence, 4),
            evidence=tuple(evidence_items[:5]),
            recommendation=recommendations.get(dominant, "Review BDI coherence."),
        )
