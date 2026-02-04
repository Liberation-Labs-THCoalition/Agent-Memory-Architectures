"""Comprehensive tests for Kintsugi tuning module.

Tests cover:
- TuningStrategy enum values
- TuningOutcome dataclass creation and validation
- TuningConfig defaults and validation
- EFETuner creation, outcome recording, tuning logic, and weight management
- FeedbackType enum values
- Feedback dataclass creation and validation
- FeedbackCollector recording, aggregation, and stakeholder weights
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from kintsugi.tuning import (
    # Strategy and config
    TuningStrategy,
    TuningConfig,
    # Outcomes
    TuningOutcome,
    # EFE Tuner
    EFETuner,
    # Feedback system
    FeedbackType,
    Feedback,
    FeedbackCollector,
    StakeholderWeight,
)
from kintsugi.tuning.feedback import StakeholderRole
# Note: EFETuner uses dict[str, float] for weights, not EFEWeights class


# ===========================================================================
# TuningStrategy Tests (4 tests)
# ===========================================================================


class TestTuningStrategy:
    """Tests for TuningStrategy enum."""

    def test_gradient_strategy_exists(self):
        """TuningStrategy.GRADIENT exists with correct value."""
        assert TuningStrategy.GRADIENT.value == "gradient"

    def test_evolutionary_strategy_exists(self):
        """TuningStrategy.EVOLUTIONARY exists with correct value."""
        assert TuningStrategy.EVOLUTIONARY.value == "evolutionary"

    def test_bayesian_strategy_exists(self):
        """TuningStrategy.BAYESIAN exists with correct value."""
        assert TuningStrategy.BAYESIAN.value == "bayesian"

    def test_manual_strategy_exists(self):
        """TuningStrategy.MANUAL exists with correct value."""
        assert TuningStrategy.MANUAL.value == "manual"


# ===========================================================================
# TuningOutcome Tests (6 tests)
# ===========================================================================


class TestTuningOutcome:
    """Tests for TuningOutcome dataclass."""

    def test_creation_with_required_fields(self):
        """TuningOutcome can be created with required fields."""
        outcome = TuningOutcome(
            decision_id="dec_123",
            outcome_score=0.8,
            efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
        )
        assert outcome.decision_id == "dec_123"
        assert outcome.outcome_score == 0.8
        assert outcome.efe_weights_used["risk"] == 0.33

    def test_outcome_score_validation_max(self):
        """TuningOutcome raises ValueError for outcome_score > 1."""
        with pytest.raises(ValueError, match="outcome_score must be in"):
            TuningOutcome(
                decision_id="dec_123",
                outcome_score=1.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )

    def test_outcome_score_validation_min(self):
        """TuningOutcome raises ValueError for outcome_score < -1."""
        with pytest.raises(ValueError, match="outcome_score must be in"):
            TuningOutcome(
                decision_id="dec_123",
                outcome_score=-1.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )

    def test_default_timestamp_is_now(self):
        """TuningOutcome defaults timestamp to current time."""
        before = datetime.now(timezone.utc)
        outcome = TuningOutcome(
            decision_id="dec_123",
            outcome_score=0.5,
            efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
        )
        after = datetime.now(timezone.utc)
        assert before <= outcome.timestamp <= after

    def test_efe_weights_used_stored(self):
        """TuningOutcome correctly stores efe_weights_used."""
        weights = {"risk": 0.5, "ambiguity": 0.3, "epistemic": 0.2}
        outcome = TuningOutcome(
            decision_id="dec_456",
            outcome_score=0.7,
            efe_weights_used=weights,
        )
        assert outcome.efe_weights_used == weights
        assert outcome.efe_weights_used["risk"] == 0.5
        assert outcome.efe_weights_used["ambiguity"] == 0.3
        assert outcome.efe_weights_used["epistemic"] == 0.2

    def test_creation_with_custom_timestamp(self):
        """TuningOutcome can be created with custom timestamp."""
        custom_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        outcome = TuningOutcome(
            decision_id="dec_789",
            outcome_score=0.9,
            efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            timestamp=custom_time,
        )
        assert outcome.timestamp == custom_time


# ===========================================================================
# TuningConfig Tests (6 tests)
# ===========================================================================


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_default_values(self):
        """TuningConfig has sensible defaults."""
        config = TuningConfig()
        assert config.strategy == TuningStrategy.GRADIENT
        assert config.learning_rate == 0.01
        assert config.min_samples == 50
        assert config.require_consensus is True

    def test_learning_rate_default_is_0_01(self):
        """TuningConfig default learning_rate is 0.01."""
        config = TuningConfig()
        assert config.learning_rate == 0.01

    def test_min_samples_default_is_50(self):
        """TuningConfig default min_samples is 50."""
        config = TuningConfig()
        assert config.min_samples == 50

    def test_require_consensus_default_is_true(self):
        """TuningConfig default require_consensus is True."""
        config = TuningConfig()
        assert config.require_consensus is True

    def test_custom_configuration(self):
        """TuningConfig accepts custom values."""
        config = TuningConfig(
            strategy=TuningStrategy.BAYESIAN,
            learning_rate=0.05,
            min_samples=100,
            require_consensus=False,
        )
        assert config.strategy == TuningStrategy.BAYESIAN
        assert config.learning_rate == 0.05
        assert config.min_samples == 100
        assert config.require_consensus is False

    def test_learning_rate_validation(self):
        """TuningConfig.validate() reports invalid learning_rate."""
        config = TuningConfig(learning_rate=-0.01)
        errors = config.validate()
        assert len(errors) > 0
        assert any("learning_rate" in e for e in errors)


# ===========================================================================
# EFETuner Tests (15 tests)
# ===========================================================================


class TestEFETuner:
    """Tests for EFETuner class."""

    @pytest.fixture
    def tuner(self) -> EFETuner:
        """Create an EFETuner with default config and initial weights."""
        t = EFETuner()
        t.set_initial_weights({"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33})
        return t

    @pytest.fixture
    def tuner_low_min_samples(self) -> EFETuner:
        """Create an EFETuner with low min_samples for testing."""
        config = TuningConfig(min_samples=3)
        t = EFETuner(config=config)
        t.set_initial_weights({"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33})
        return t

    def test_creation_with_default_config(self, tuner):
        """EFETuner can be created with default config."""
        assert tuner._config.strategy == TuningStrategy.GRADIENT
        assert tuner._config.learning_rate == 0.01
        assert tuner._config.min_samples == 50

    def test_record_outcome_stores_outcome(self, tuner):
        """EFETuner.record_outcome() stores the outcome."""
        outcome = TuningOutcome(
            decision_id="dec_001",
            outcome_score=0.6,
            efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
        )
        tuner.record_outcome(outcome)
        assert len(tuner._outcomes) == 1
        assert tuner._outcomes[0] == outcome

    def test_should_tune_returns_false_with_few_samples(self, tuner):
        """EFETuner.should_tune() returns False with fewer than min_samples."""
        # Default min_samples is 50, add only 10
        for i in range(10):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=0.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )
            tuner.record_outcome(outcome)
        assert tuner.should_tune() is False

    def test_should_tune_returns_true_with_enough_samples(self, tuner_low_min_samples):
        """EFETuner.should_tune() returns True with enough samples."""
        for i in range(5):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=0.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )
            tuner_low_min_samples.record_outcome(outcome)
        assert tuner_low_min_samples.should_tune() is True

    def test_compute_gradients_returns_weight_changes(self, tuner_low_min_samples):
        """EFETuner.compute_gradients() returns weight change dictionary."""
        # Add outcomes with varying scores
        outcomes = [
            TuningOutcome(
                decision_id="dec_1",
                outcome_score=0.8,
                efe_weights_used={"risk": 0.4, "ambiguity": 0.3, "epistemic": 0.3},
            ),
            TuningOutcome(
                decision_id="dec_2",
                outcome_score=0.3,
                efe_weights_used={"risk": 0.2, "ambiguity": 0.4, "epistemic": 0.4},
            ),
            TuningOutcome(
                decision_id="dec_3",
                outcome_score=0.9,
                efe_weights_used={"risk": 0.5, "ambiguity": 0.25, "epistemic": 0.25},
            ),
        ]
        for outcome in outcomes:
            tuner_low_min_samples.record_outcome(outcome)

        gradients = tuner_low_min_samples.compute_gradients()
        assert "risk" in gradients
        assert "ambiguity" in gradients
        assert "epistemic" in gradients
        assert isinstance(gradients["risk"], float)

    def test_propose_weights_returns_new_weights(self, tuner_low_min_samples):
        """EFETuner.propose_weights() returns new weight dict."""
        for i in range(5):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=0.5 + i * 0.1,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )
            tuner_low_min_samples.record_outcome(outcome)

        proposed = tuner_low_min_samples.propose_weights()
        assert isinstance(proposed, dict)
        # Weights should still sum to ~1.0
        total = sum(proposed.values())
        assert math.isclose(total, 1.0, abs_tol=0.05)

    def test_apply_weights_updates_current_weights(self):
        """EFETuner.apply_weights() updates current weights."""
        config = TuningConfig(require_consensus=False)
        tuner = EFETuner(config=config)
        new_weights = {"risk": 0.4, "ambiguity": 0.35, "epistemic": 0.25}
        tuner.apply_weights(new_weights, approver="admin")

        assert tuner.current_weights == new_weights

    def test_apply_weights_requires_consensus_when_configured(self, tuner):
        """EFETuner.apply_weights() requires consensus when configured."""
        new_weights = {"risk": 0.4, "ambiguity": 0.35, "epistemic": 0.25}
        # Default config has require_consensus=True, so should raise without approval
        with pytest.raises(ValueError, match="[Cc]onsensus"):
            tuner.apply_weights(new_weights)

    def test_apply_weights_without_consensus_when_not_required(self):
        """EFETuner.apply_weights() works without consensus when not required."""
        config = TuningConfig(require_consensus=False)
        tuner = EFETuner(config=config)
        new_weights = {"risk": 0.4, "ambiguity": 0.35, "epistemic": 0.25}

        tuner.apply_weights(new_weights, approver="admin")
        assert tuner.current_weights == new_weights

    def test_get_tuning_report_returns_report_dict(self, tuner_low_min_samples):
        """EFETuner.get_tuning_report() returns dict with tuning info."""
        for i in range(5):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=0.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )
            tuner_low_min_samples.record_outcome(outcome)

        report = tuner_low_min_samples.get_tuning_report()
        assert isinstance(report, dict)
        assert report["metrics"]["total_outcomes"] == 5
        assert "current_weights" in report
        assert "average_outcome_score" in report["metrics"]

    def test_weight_history_tracking(self):
        """EFETuner tracks weight history internally."""
        config = TuningConfig(require_consensus=False)
        tuner = EFETuner(config=config)
        weights1 = {"risk": 0.4, "ambiguity": 0.35, "epistemic": 0.25}
        weights2 = {"risk": 0.35, "ambiguity": 0.35, "epistemic": 0.30}

        tuner.apply_weights(weights1, approver="admin1")
        tuner.apply_weights(weights2, approver="admin2")

        # Weight history tracked internally as list of (timestamp, weights) tuples
        assert len(tuner._weight_history) >= 2
        assert all(isinstance(h, tuple) and len(h) == 2 for h in tuner._weight_history)

    def test_weight_constraints_enforced(self, tuner_low_min_samples):
        """EFETuner enforces weight constraints (sum to 1, non-negative)."""
        # Add outcomes that would push weights to extremes
        for i in range(5):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=1.0 if i % 2 == 0 else -1.0,
                efe_weights_used={"risk": 0.9, "ambiguity": 0.05, "epistemic": 0.05},
            )
            tuner_low_min_samples.record_outcome(outcome)

        proposed = tuner_low_min_samples.propose_weights()
        # All weights should be non-negative
        assert proposed["risk"] >= 0
        assert proposed["ambiguity"] >= 0
        assert proposed["epistemic"] >= 0
        # Should still sum to ~1.0
        total = proposed["risk"] + proposed["ambiguity"] + proposed["epistemic"]
        assert math.isclose(total, 1.0, abs_tol=0.05)

    def test_outcomes_are_recorded(self, tuner_low_min_samples):
        """EFETuner stores outcomes internally."""
        for i in range(5):
            outcome = TuningOutcome(
                decision_id=f"dec_{i}",
                outcome_score=0.5,
                efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
            )
            tuner_low_min_samples.record_outcome(outcome)

        assert len(tuner_low_min_samples._outcomes) == 5

    def test_get_outcome_count_via_internal_list(self, tuner):
        """EFETuner tracks outcomes via internal _outcomes list."""
        assert len(tuner._outcomes) == 0

        outcome = TuningOutcome(
            decision_id="dec_1",
            outcome_score=0.5,
            efe_weights_used={"risk": 0.33, "ambiguity": 0.34, "epistemic": 0.33},
        )
        tuner.record_outcome(outcome)
        assert len(tuner._outcomes) == 1


# ===========================================================================
# FeedbackType Tests (3 tests)
# ===========================================================================


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_thumbs_up_type_exists(self):
        """FeedbackType.THUMBS_UP exists."""
        assert FeedbackType.THUMBS_UP.value == "thumbs_up"

    def test_thumbs_down_type_exists(self):
        """FeedbackType.THUMBS_DOWN exists."""
        assert FeedbackType.THUMBS_DOWN.value == "thumbs_down"

    def test_rating_type_exists(self):
        """FeedbackType.RATING exists."""
        assert FeedbackType.RATING.value == "rating"

    def test_outcome_type_exists(self):
        """FeedbackType.OUTCOME exists."""
        assert FeedbackType.OUTCOME.value == "outcome"


# ===========================================================================
# Feedback Tests (5 tests)
# ===========================================================================


class TestFeedback:
    """Tests for Feedback dataclass."""

    def test_creation_with_required_fields(self):
        """Feedback can be created with required fields."""
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_456",
            stakeholder_role="user",
        )
        assert feedback.decision_id == "dec_123"
        assert feedback.stakeholder_id == "user_456"
        assert feedback.feedback_type == FeedbackType.OUTCOME
        assert feedback.value == 0.8

    def test_rating_validation(self):
        """Feedback validates rating values are 1-5."""
        feedback = Feedback(
            decision_id="dec_1",
            feedback_type=FeedbackType.RATING,
            value=4,
            stakeholder_id="user_1",
            stakeholder_role="user",
        )
        assert feedback.value == 4

        # Invalid rating should raise
        with pytest.raises(ValueError):
            Feedback(
                decision_id="dec_2",
                feedback_type=FeedbackType.RATING,
                value=10,
                stakeholder_id="user_2",
                stakeholder_role="user",
            )

    def test_timestamp_default_is_now(self):
        """Feedback defaults timestamp to current time."""
        before = datetime.now(timezone.utc)
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.5,
            stakeholder_id="user_456",
            stakeholder_role="user",
        )
        after = datetime.now(timezone.utc)
        assert before <= feedback.timestamp <= after

    def test_creation_with_custom_timestamp(self):
        """Feedback can be created with custom timestamp."""
        custom_time = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.5,
            stakeholder_id="user_456",
            stakeholder_role="user",
            timestamp=custom_time,
        )
        assert feedback.timestamp == custom_time

    def test_creation_with_metadata(self):
        """Feedback can store additional metadata."""
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_456",
            stakeholder_role="user",
            metadata={"reason": "helpful response", "category": "quality"},
        )
        assert feedback.metadata["reason"] == "helpful response"
        assert feedback.metadata["category"] == "quality"


# ===========================================================================
# FeedbackCollector Tests (10 tests)
# ===========================================================================


class TestFeedbackCollector:
    """Tests for FeedbackCollector class."""

    @pytest.fixture
    def collector(self) -> FeedbackCollector:
        """Create a FeedbackCollector."""
        return FeedbackCollector()

    def test_record_stores_feedback(self, collector):
        """FeedbackCollector.record() stores feedback."""
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_456",
            stakeholder_role="user",
        )
        collector.record(feedback)
        assert len(collector._feedback) == 1

    def test_get_for_decision_returns_feedback_list(self, collector):
        """FeedbackCollector.get_for_decision() returns list of feedback."""
        # Add feedback for multiple decisions
        feedback1 = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_1",
            stakeholder_role="user",
        )
        feedback2 = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.RATING,
            value=4,
            stakeholder_id="user_2",
            stakeholder_role="user",
        )
        feedback3 = Feedback(
            decision_id="dec_456",
            feedback_type=FeedbackType.OUTCOME,
            value=0.9,
            stakeholder_id="user_1",
            stakeholder_role="user",
        )

        collector.record(feedback1)
        collector.record(feedback2)
        collector.record(feedback3)

        result = collector.get_for_decision("dec_123")
        assert len(result) == 2
        assert all(f.decision_id == "dec_123" for f in result)

    def test_aggregate_score_computes_weighted_score(self, collector):
        """FeedbackCollector.aggregate_score() computes weighted average."""
        feedback1 = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_1",
            stakeholder_role="user",
        )
        feedback2 = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.6,
            stakeholder_id="user_2",
            stakeholder_role="user",
        )

        collector.record(feedback1)
        collector.record(feedback2)

        score = collector.aggregate_score("dec_123")
        # Should be weighted average
        assert 0.6 <= score <= 0.8

    def test_get_stakeholder_weights_returns_weights(self, collector):
        """FeedbackCollector.get_stakeholder_weights() returns weight dictionary."""
        # First record some feedback to populate stakeholders
        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.RATING,
            value=4,
            stakeholder_id="user_1",
            stakeholder_role=StakeholderRole.USER.value,
        )
        collector.record(feedback)
        weights = collector.get_stakeholder_weights()
        assert isinstance(weights, dict)
        assert "user_1" in weights

    def test_multiple_stakeholder_roles(self, collector):
        """FeedbackCollector handles multiple stakeholder roles."""
        # Add feedback from different stakeholders with different roles
        feedback_user = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.RATING,
            value=4,
            stakeholder_id="user_1",
            stakeholder_role=StakeholderRole.USER.value,
        )
        feedback_admin = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.RATING,
            value=3,
            stakeholder_id="admin_1",
            stakeholder_role=StakeholderRole.ADMINISTRATOR.value,
        )
        feedback_subject = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.RATING,
            value=5,
            stakeholder_id="subject_1",
            stakeholder_role=StakeholderRole.SUBJECT.value,
        )

        collector.record(feedback_user)
        collector.record(feedback_admin)
        collector.record(feedback_subject)

        # Score should consider stakeholder weights
        score = collector.aggregate_score("dec_123")
        assert -1.0 <= score <= 1.0

    def test_aggregate_score_returns_zero_for_no_feedback(self, collector):
        """FeedbackCollector.aggregate_score() returns 0.0 for no feedback."""
        score = collector.aggregate_score("nonexistent_decision")
        assert score == 0.0

    def test_feedback_is_stored(self, collector):
        """FeedbackCollector stores feedback internally."""
        assert len(collector._feedback) == 0

        feedback = Feedback(
            decision_id="dec_123",
            feedback_type=FeedbackType.OUTCOME,
            value=0.8,
            stakeholder_id="user_456",
            stakeholder_role="user",
        )
        collector.record(feedback)
        assert len(collector._feedback) == 1

    def test_clear_feedback(self, collector):
        """FeedbackCollector.clear_feedback() removes feedback."""
        for i in range(5):
            feedback = Feedback(
                decision_id=f"dec_{i}",
                feedback_type=FeedbackType.OUTCOME,
                value=0.5,
                stakeholder_id=f"user_{i}",
                stakeholder_role="user",
            )
            collector.record(feedback)

        assert len(collector._feedback) == 5
        collector.clear_feedback()
        assert len(collector._feedback) == 0

    def test_get_decisions_with_feedback(self, collector):
        """FeedbackCollector.get_decisions_with_feedback() returns decision IDs."""
        decisions = ["dec_1", "dec_2", "dec_3"]
        for dec_id in decisions:
            feedback = Feedback(
                decision_id=dec_id,
                feedback_type=FeedbackType.OUTCOME,
                value=0.5,
                stakeholder_id="user_1",
                stakeholder_role="user",
            )
            collector.record(feedback)

        result = collector.get_decisions_with_feedback()
        assert len(result) == 3
        assert set(result) == set(decisions)


# ===========================================================================
# StakeholderRole Tests (3 additional tests)
# ===========================================================================


class TestStakeholderRole:
    """Tests for StakeholderRole enum."""

    def test_user_role_exists(self):
        """StakeholderRole.USER exists."""
        assert StakeholderRole.USER.value == "user"

    def test_administrator_role_exists(self):
        """StakeholderRole.ADMINISTRATOR exists."""
        assert StakeholderRole.ADMINISTRATOR.value == "administrator"

    def test_subject_role_exists(self):
        """StakeholderRole.SUBJECT exists."""
        assert StakeholderRole.SUBJECT.value == "subject"
