"""
Feedback Collection - Stakeholder Feedback for EFE Tuning

This module provides mechanisms for collecting, aggregating, and analyzing
stakeholder feedback on AI decisions. Feedback is used by the EFE tuner
to improve weight configurations over time.

Feedback Types:
    - THUMBS_UP/THUMBS_DOWN: Simple binary feedback
    - RATING: 1-5 scale rating
    - TEXT: Qualitative text feedback
    - OUTCOME: Objective outcome measurement

The FeedbackCollector aggregates feedback by decision, computes weighted
scores based on stakeholder roles, and provides analysis for tuning.
"""

from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterator


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""

    THUMBS_UP = "thumbs_up"  # Simple positive feedback
    THUMBS_DOWN = "thumbs_down"  # Simple negative feedback
    RATING = "rating"  # 1-5 scale rating
    TEXT = "text"  # Qualitative text feedback
    OUTCOME = "outcome"  # Objective outcome measurement


class StakeholderRole(str, Enum):
    """
    Predefined stakeholder roles with default weights.

    Weights determine how much influence each stakeholder type
    has in aggregated feedback scores.
    """

    USER = "user"  # End users of the system
    ADMINISTRATOR = "administrator"  # System administrators
    SUBJECT = "subject"  # People affected by decisions
    AUDITOR = "auditor"  # External auditors
    ETHICS_BOARD = "ethics_board"  # Ethics board members
    DOMAIN_EXPERT = "domain_expert"  # Domain experts
    OPERATOR = "operator"  # System operators
    EXTERNAL = "external"  # External stakeholders


# Default weights for stakeholder roles
DEFAULT_ROLE_WEIGHTS: dict[str, float] = {
    StakeholderRole.USER.value: 1.0,
    StakeholderRole.ADMINISTRATOR.value: 1.2,
    StakeholderRole.SUBJECT.value: 1.5,  # Those affected get higher weight
    StakeholderRole.AUDITOR.value: 1.3,
    StakeholderRole.ETHICS_BOARD.value: 2.0,  # Ethics board highest weight
    StakeholderRole.DOMAIN_EXPERT.value: 1.4,
    StakeholderRole.OPERATOR.value: 1.0,
    StakeholderRole.EXTERNAL.value: 0.8,
}


@dataclass
class StakeholderWeight:
    """
    Weight configuration for a stakeholder.

    Attributes:
        stakeholder_id: Unique identifier for the stakeholder
        role: The stakeholder's role
        weight: Influence weight (higher = more influence)
        trust_score: Trust score from 0-1 (can modify weight)
        feedback_count: Number of feedbacks provided
        last_feedback: Timestamp of last feedback
    """

    stakeholder_id: str
    role: str
    weight: float = 1.0
    trust_score: float = 1.0
    feedback_count: int = 0
    last_feedback: datetime | None = None

    def effective_weight(self) -> float:
        """Compute effective weight considering trust score."""
        return self.weight * self.trust_score


@dataclass
class Feedback:
    """
    Individual feedback record from a stakeholder.

    Attributes:
        feedback_id: Unique identifier for this feedback
        decision_id: The decision being evaluated
        feedback_type: Type of feedback
        value: The feedback value (interpretation depends on type)
        stakeholder_id: Who provided the feedback
        stakeholder_role: Role of the stakeholder
        timestamp: When the feedback was provided
        metadata: Additional context about the feedback
    """

    decision_id: str
    feedback_type: FeedbackType
    value: float | str
    stakeholder_id: str
    stakeholder_role: str
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate feedback value based on type."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            if self.value not in (1, 1.0, True, "up"):
                self.value = 1.0
            else:
                self.value = 1.0
        elif self.feedback_type == FeedbackType.THUMBS_DOWN:
            if self.value not in (-1, -1.0, 0, False, "down"):
                self.value = -1.0
            else:
                self.value = -1.0
        elif self.feedback_type == FeedbackType.RATING:
            if isinstance(self.value, (int, float)):
                if not 1 <= self.value <= 5:
                    raise ValueError(f"Rating must be 1-5, got {self.value}")
            else:
                raise ValueError(f"Rating must be numeric, got {type(self.value)}")
        elif self.feedback_type == FeedbackType.OUTCOME:
            if isinstance(self.value, (int, float)):
                if not -1.0 <= self.value <= 1.0:
                    raise ValueError(f"Outcome must be -1.0 to 1.0, got {self.value}")
            else:
                raise ValueError(f"Outcome must be numeric, got {type(self.value)}")

    def normalized_score(self) -> float | None:
        """
        Get normalized score in range [-1, 1].

        Returns None for text feedback which cannot be normalized.
        """
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return 1.0
        elif self.feedback_type == FeedbackType.THUMBS_DOWN:
            return -1.0
        elif self.feedback_type == FeedbackType.RATING:
            # Convert 1-5 to -1 to 1
            return (float(self.value) - 3) / 2
        elif self.feedback_type == FeedbackType.OUTCOME:
            return float(self.value)
        else:
            return None


@dataclass
class FeedbackAggregation:
    """
    Aggregation method configuration.

    Attributes:
        method: Aggregation method name
        weight_by_role: Whether to weight by stakeholder role
        weight_by_trust: Whether to weight by trust score
        recency_decay: Decay factor for older feedback (0 = no decay)
        min_feedback_count: Minimum feedback required for valid aggregation
    """

    method: str = "weighted_mean"  # "mean", "weighted_mean", "median", "mode"
    weight_by_role: bool = True
    weight_by_trust: bool = True
    recency_decay: float = 0.0  # Half-life in days (0 = no decay)
    min_feedback_count: int = 1


@dataclass
class FeedbackSummary:
    """
    Summary of feedback for a decision.

    Attributes:
        decision_id: The decision summarized
        total_count: Total feedback count
        by_type: Count by feedback type
        by_role: Count by stakeholder role
        aggregated_score: Weighted aggregated score
        score_std_dev: Standard deviation of scores
        sentiment: Overall sentiment (positive/neutral/negative)
        text_feedbacks: List of text feedback content
        generated_at: When the summary was generated
    """

    decision_id: str
    total_count: int
    by_type: dict[str, int]
    by_role: dict[str, int]
    aggregated_score: float
    score_std_dev: float
    sentiment: str
    text_feedbacks: list[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackCollector:
    """
    Collects and aggregates feedback for tuning.

    The collector maintains feedback records organized by decision,
    computes weighted aggregations based on stakeholder roles and trust,
    and provides analysis for the EFE tuner.

    Example:
        >>> collector = FeedbackCollector()
        >>>
        >>> # Record feedback
        >>> feedback = Feedback(
        ...     decision_id="decision-123",
        ...     feedback_type=FeedbackType.RATING,
        ...     value=4,
        ...     stakeholder_id="user-456",
        ...     stakeholder_role="user",
        ... )
        >>> collector.record(feedback)
        >>>
        >>> # Get aggregated score
        >>> score = collector.aggregate_score("decision-123")
        >>> print(f"Score: {score}")
    """

    def __init__(
        self,
        aggregation: FeedbackAggregation | None = None,
        role_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the feedback collector.

        Args:
            aggregation: Aggregation configuration
            role_weights: Override default role weights
        """
        self._feedback: dict[str, list[Feedback]] = defaultdict(list)
        self._stakeholders: dict[str, StakeholderWeight] = {}
        self._aggregation = aggregation or FeedbackAggregation()
        self._role_weights = role_weights or dict(DEFAULT_ROLE_WEIGHTS)
        self._callbacks: list[Callable[[Feedback], None]] = []

    def register_callback(self, callback: Callable[[Feedback], None]) -> None:
        """
        Register a callback to be called when feedback is recorded.

        Args:
            callback: Function to call with each new feedback
        """
        self._callbacks.append(callback)

    def set_role_weight(self, role: str, weight: float) -> None:
        """
        Set the weight for a stakeholder role.

        Args:
            role: The role name
            weight: The weight value (typically 0.5 to 2.0)
        """
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        self._role_weights[role] = weight

    def set_stakeholder_trust(self, stakeholder_id: str, trust_score: float) -> None:
        """
        Set the trust score for a specific stakeholder.

        Args:
            stakeholder_id: The stakeholder identifier
            trust_score: Trust score from 0 to 1
        """
        if not 0 <= trust_score <= 1:
            raise ValueError(f"Trust score must be 0-1, got {trust_score}")

        if stakeholder_id in self._stakeholders:
            self._stakeholders[stakeholder_id].trust_score = trust_score
        else:
            self._stakeholders[stakeholder_id] = StakeholderWeight(
                stakeholder_id=stakeholder_id,
                role="unknown",
                trust_score=trust_score,
            )

    def record(self, feedback: Feedback) -> None:
        """
        Record a feedback entry.

        Args:
            feedback: The feedback to record
        """
        self._feedback[feedback.decision_id].append(feedback)

        # Update stakeholder info
        if feedback.stakeholder_id not in self._stakeholders:
            role_weight = self._role_weights.get(feedback.stakeholder_role, 1.0)
            self._stakeholders[feedback.stakeholder_id] = StakeholderWeight(
                stakeholder_id=feedback.stakeholder_id,
                role=feedback.stakeholder_role,
                weight=role_weight,
            )

        stakeholder = self._stakeholders[feedback.stakeholder_id]
        stakeholder.feedback_count += 1
        stakeholder.last_feedback = feedback.timestamp

        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(feedback)
            except Exception:
                pass  # Don't let callback errors break recording

    def get_for_decision(self, decision_id: str) -> list[Feedback]:
        """
        Get all feedback for a decision.

        Args:
            decision_id: The decision identifier

        Returns:
            List of feedback entries for the decision
        """
        return list(self._feedback.get(decision_id, []))

    def get_decisions_with_feedback(self) -> list[str]:
        """Get list of decision IDs that have feedback."""
        return list(self._feedback.keys())

    def aggregate_score(
        self,
        decision_id: str,
        aggregation: FeedbackAggregation | None = None,
    ) -> float:
        """
        Compute aggregated score for a decision.

        Args:
            decision_id: The decision to aggregate
            aggregation: Override aggregation settings

        Returns:
            Aggregated score in range [-1, 1], or 0 if no feedback
        """
        feedback_list = self._feedback.get(decision_id, [])
        if not feedback_list:
            return 0.0

        agg = aggregation or self._aggregation

        if len(feedback_list) < agg.min_feedback_count:
            return 0.0

        # Collect scores with weights
        weighted_scores: list[tuple[float, float]] = []
        now = datetime.now(timezone.utc)

        for fb in feedback_list:
            score = fb.normalized_score()
            if score is None:
                continue

            # Compute weight
            weight = 1.0

            if agg.weight_by_role:
                weight *= self._role_weights.get(fb.stakeholder_role, 1.0)

            if agg.weight_by_trust:
                stakeholder = self._stakeholders.get(fb.stakeholder_id)
                if stakeholder:
                    weight *= stakeholder.trust_score

            if agg.recency_decay > 0:
                age_days = (now - fb.timestamp).total_seconds() / 86400
                decay = 0.5 ** (age_days / agg.recency_decay)
                weight *= decay

            weighted_scores.append((score, weight))

        if not weighted_scores:
            return 0.0

        # Aggregate based on method
        if agg.method == "mean":
            return statistics.mean(s for s, _ in weighted_scores)
        elif agg.method == "weighted_mean":
            total_weight = sum(w for _, w in weighted_scores)
            if total_weight == 0:
                return 0.0
            return sum(s * w for s, w in weighted_scores) / total_weight
        elif agg.method == "median":
            return statistics.median(s for s, _ in weighted_scores)
        else:
            # Default to weighted mean
            total_weight = sum(w for _, w in weighted_scores)
            if total_weight == 0:
                return 0.0
            return sum(s * w for s, w in weighted_scores) / total_weight

    def get_stakeholder_weights(self) -> dict[str, float]:
        """
        Get effective weights for all stakeholders.

        Returns:
            Dictionary mapping stakeholder IDs to effective weights
        """
        return {
            sid: sw.effective_weight()
            for sid, sw in self._stakeholders.items()
        }

    def get_role_weights(self) -> dict[str, float]:
        """Get current role weight configuration."""
        return dict(self._role_weights)

    def summarize(self, decision_id: str) -> FeedbackSummary:
        """
        Generate a summary of feedback for a decision.

        Args:
            decision_id: The decision to summarize

        Returns:
            FeedbackSummary with aggregated statistics
        """
        feedback_list = self._feedback.get(decision_id, [])

        # Count by type
        by_type: dict[str, int] = defaultdict(int)
        for fb in feedback_list:
            by_type[fb.feedback_type.value] += 1

        # Count by role
        by_role: dict[str, int] = defaultdict(int)
        for fb in feedback_list:
            by_role[fb.stakeholder_role] += 1

        # Collect numeric scores
        scores = [
            fb.normalized_score()
            for fb in feedback_list
            if fb.normalized_score() is not None
        ]

        if scores:
            agg_score = self.aggregate_score(decision_id)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        else:
            agg_score = 0.0
            std_dev = 0.0

        # Determine sentiment
        if agg_score > 0.2:
            sentiment = "positive"
        elif agg_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Collect text feedback
        text_feedbacks = [
            str(fb.value)
            for fb in feedback_list
            if fb.feedback_type == FeedbackType.TEXT
        ]

        return FeedbackSummary(
            decision_id=decision_id,
            total_count=len(feedback_list),
            by_type=dict(by_type),
            by_role=dict(by_role),
            aggregated_score=agg_score,
            score_std_dev=std_dev,
            sentiment=sentiment,
            text_feedbacks=text_feedbacks,
        )

    def get_recent_feedback(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Feedback]:
        """
        Get recent feedback across all decisions.

        Args:
            limit: Maximum number of feedback entries
            since: Only return feedback after this timestamp

        Returns:
            List of recent feedback entries
        """
        all_feedback = []
        for feedbacks in self._feedback.values():
            all_feedback.extend(feedbacks)

        # Filter by timestamp if specified
        if since:
            all_feedback = [fb for fb in all_feedback if fb.timestamp > since]

        # Sort by timestamp descending
        all_feedback.sort(key=lambda x: x.timestamp, reverse=True)

        return all_feedback[:limit]

    def get_stakeholder_feedback(
        self,
        stakeholder_id: str,
        limit: int = 100,
    ) -> list[Feedback]:
        """
        Get all feedback from a specific stakeholder.

        Args:
            stakeholder_id: The stakeholder to query
            limit: Maximum number of entries

        Returns:
            List of feedback from the stakeholder
        """
        result = []
        for feedbacks in self._feedback.values():
            for fb in feedbacks:
                if fb.stakeholder_id == stakeholder_id:
                    result.append(fb)
                    if len(result) >= limit:
                        return result
        return result

    def analyze_stakeholder_consistency(
        self,
        stakeholder_id: str,
    ) -> dict:
        """
        Analyze consistency of a stakeholder's feedback.

        Args:
            stakeholder_id: The stakeholder to analyze

        Returns:
            Dictionary with consistency metrics
        """
        feedbacks = self.get_stakeholder_feedback(stakeholder_id)

        if not feedbacks:
            return {
                "stakeholder_id": stakeholder_id,
                "feedback_count": 0,
                "consistency_score": 1.0,
                "average_score": 0.0,
                "score_variance": 0.0,
            }

        scores = [
            fb.normalized_score()
            for fb in feedbacks
            if fb.normalized_score() is not None
        ]

        if len(scores) < 2:
            return {
                "stakeholder_id": stakeholder_id,
                "feedback_count": len(feedbacks),
                "consistency_score": 1.0,
                "average_score": scores[0] if scores else 0.0,
                "score_variance": 0.0,
            }

        variance = statistics.variance(scores)
        # Convert variance to consistency (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance * 4)

        return {
            "stakeholder_id": stakeholder_id,
            "feedback_count": len(feedbacks),
            "consistency_score": consistency,
            "average_score": statistics.mean(scores),
            "score_variance": variance,
        }

    def compute_inter_rater_agreement(self, decision_id: str) -> float:
        """
        Compute inter-rater agreement for a decision.

        Uses a simplified kappa-like measure based on variance.

        Args:
            decision_id: The decision to analyze

        Returns:
            Agreement score from 0 (no agreement) to 1 (perfect agreement)
        """
        feedback_list = self._feedback.get(decision_id, [])

        scores = [
            fb.normalized_score()
            for fb in feedback_list
            if fb.normalized_score() is not None
        ]

        if len(scores) < 2:
            return 1.0  # Perfect agreement with self

        variance = statistics.variance(scores)
        # Max possible variance for [-1, 1] range is 1.0
        # Convert to agreement measure
        return max(0.0, 1.0 - variance)

    def export_feedback(self, decision_id: str | None = None) -> list[dict]:
        """
        Export feedback data for external analysis.

        Args:
            decision_id: Specific decision, or None for all

        Returns:
            List of feedback dictionaries
        """
        if decision_id:
            feedback_list = self._feedback.get(decision_id, [])
        else:
            feedback_list = []
            for feedbacks in self._feedback.values():
                feedback_list.extend(feedbacks)

        return [
            {
                "feedback_id": fb.feedback_id,
                "decision_id": fb.decision_id,
                "feedback_type": fb.feedback_type.value,
                "value": fb.value,
                "normalized_score": fb.normalized_score(),
                "stakeholder_id": fb.stakeholder_id,
                "stakeholder_role": fb.stakeholder_role,
                "timestamp": fb.timestamp.isoformat(),
                "metadata": fb.metadata,
            }
            for fb in feedback_list
        ]

    def clear_feedback(self, decision_id: str | None = None) -> int:
        """
        Clear feedback data.

        Args:
            decision_id: Specific decision, or None for all

        Returns:
            Number of feedback entries cleared
        """
        if decision_id:
            count = len(self._feedback.get(decision_id, []))
            self._feedback.pop(decision_id, None)
            return count
        else:
            count = sum(len(v) for v in self._feedback.values())
            self._feedback.clear()
            return count

    def get_statistics(self) -> dict:
        """
        Get overall statistics about collected feedback.

        Returns:
            Dictionary with aggregate statistics
        """
        total_feedback = sum(len(v) for v in self._feedback.values())
        total_decisions = len(self._feedback)
        total_stakeholders = len(self._stakeholders)

        # Type distribution
        type_counts: dict[str, int] = defaultdict(int)
        role_counts: dict[str, int] = defaultdict(int)

        for feedbacks in self._feedback.values():
            for fb in feedbacks:
                type_counts[fb.feedback_type.value] += 1
                role_counts[fb.stakeholder_role] += 1

        # Average scores
        all_scores = []
        for feedbacks in self._feedback.values():
            for fb in feedbacks:
                score = fb.normalized_score()
                if score is not None:
                    all_scores.append(score)

        return {
            "total_feedback": total_feedback,
            "total_decisions": total_decisions,
            "total_stakeholders": total_stakeholders,
            "feedback_per_decision": (
                total_feedback / total_decisions if total_decisions > 0 else 0
            ),
            "type_distribution": dict(type_counts),
            "role_distribution": dict(role_counts),
            "average_score": statistics.mean(all_scores) if all_scores else 0.0,
            "score_std_dev": (
                statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            ),
        }
