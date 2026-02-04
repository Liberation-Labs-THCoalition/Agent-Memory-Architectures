"""
Kintsugi CMA - Auto-Tuning Module

This module provides automated tuning capabilities for the Ethical Framing Engine (EFE).
The tuning system learns from decision outcomes and stakeholder feedback to optimize
EFE weights over time, while maintaining ethical guardrails and requiring consensus
for significant changes.

Key Components:
    - EFETuner: Main tuning engine with multiple strategy support
    - TuningOutcome: Recorded outcome data for learning
    - TuningConfig: Configuration for tuning behavior
    - TuningStrategy: Available tuning algorithms
    - FeedbackCollector: Collects and aggregates stakeholder feedback
    - Feedback: Individual feedback records
    - FeedbackType: Types of feedback supported

Example Usage:
    >>> from kintsugi.tuning import EFETuner, TuningConfig, TuningStrategy
    >>>
    >>> config = TuningConfig(
    ...     strategy=TuningStrategy.GRADIENT,
    ...     learning_rate=0.01,
    ...     min_samples=50,
    ... )
    >>> tuner = EFETuner(config)
    >>>
    >>> # Record outcomes from decisions
    >>> from kintsugi.tuning import TuningOutcome
    >>> outcome = TuningOutcome(
    ...     decision_id="decision-123",
    ...     efe_weights_used={"autonomy": 0.3, "beneficence": 0.4},
    ...     outcome_score=0.8,
    ... )
    >>> tuner.record_outcome(outcome)
    >>>
    >>> # Check if ready to tune
    >>> if tuner.should_tune():
    ...     proposed = tuner.propose_weights()
    ...     tuner.apply_weights(proposed, approver="ethics-board")

Tuning Strategies:
    - GRADIENT: Gradient descent optimization on outcome scores
    - EVOLUTIONARY: Evolutionary algorithm with population-based search
    - BAYESIAN: Bayesian optimization for sample-efficient learning
    - MANUAL: Human-in-the-loop only, no automatic updates

Safety Features:
    - Minimum sample requirements before tuning
    - Maximum weight change limits per cycle
    - Consensus requirements for weight changes
    - Full audit trail of weight history
    - Rollback capability to previous weights
"""

from kintsugi.tuning.efe_tuner import (
    EFETuner,
    TuningConfig,
    TuningOutcome,
    TuningStrategy,
    WeightProposal,
    TuningCycle,
    WeightConstraint,
    TuningMetrics,
)

from kintsugi.tuning.feedback import (
    Feedback,
    FeedbackCollector,
    FeedbackType,
    FeedbackAggregation,
    StakeholderWeight,
    FeedbackSummary,
)

__all__ = [
    # EFE Tuner
    "EFETuner",
    "TuningConfig",
    "TuningOutcome",
    "TuningStrategy",
    "WeightProposal",
    "TuningCycle",
    "WeightConstraint",
    "TuningMetrics",
    # Feedback
    "Feedback",
    "FeedbackCollector",
    "FeedbackType",
    "FeedbackAggregation",
    "StakeholderWeight",
    "FeedbackSummary",
]

# Module version
__version__ = "0.1.0"

# Module-level constants for default configurations
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MIN_SAMPLES = 50
DEFAULT_MAX_WEIGHT_CHANGE = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 0.0001

# Ethical weight bounds - these define hard limits
ETHICAL_WEIGHT_BOUNDS = {
    "autonomy": (0.05, 0.95),
    "beneficence": (0.05, 0.95),
    "non_maleficence": (0.10, 1.00),  # Higher minimum for do-no-harm
    "justice": (0.05, 0.95),
    "transparency": (0.05, 0.95),
    "privacy": (0.05, 0.95),
    "accountability": (0.05, 0.95),
    "sustainability": (0.05, 0.95),
}


def create_default_tuner() -> EFETuner:
    """
    Create an EFE tuner with default conservative configuration.

    Returns:
        EFETuner: Configured tuner with safe defaults

    Example:
        >>> tuner = create_default_tuner()
        >>> tuner.config.strategy
        <TuningStrategy.GRADIENT: 'gradient'>
    """
    config = TuningConfig(
        strategy=TuningStrategy.GRADIENT,
        learning_rate=DEFAULT_LEARNING_RATE,
        min_samples=DEFAULT_MIN_SAMPLES,
        max_weight_change=DEFAULT_MAX_WEIGHT_CHANGE,
        require_consensus=True,
    )
    return EFETuner(config)


def create_conservative_tuner() -> EFETuner:
    """
    Create a highly conservative tuner for sensitive deployments.

    This tuner requires more samples, allows smaller weight changes,
    and always requires consensus approval.

    Returns:
        EFETuner: Conservative tuner configuration
    """
    config = TuningConfig(
        strategy=TuningStrategy.GRADIENT,
        learning_rate=0.005,  # Half the default
        min_samples=100,  # Double the default
        max_weight_change=0.05,  # Half the default
        require_consensus=True,
    )
    return EFETuner(config)


def create_experimental_tuner() -> EFETuner:
    """
    Create a tuner for experimental/testing environments.

    This tuner is more aggressive and doesn't require consensus,
    suitable only for non-production testing.

    Returns:
        EFETuner: Experimental tuner configuration

    Warning:
        Do not use in production! This configuration bypasses
        safety checks designed for real-world deployments.
    """
    config = TuningConfig(
        strategy=TuningStrategy.BAYESIAN,
        learning_rate=0.05,
        min_samples=20,
        max_weight_change=0.2,
        require_consensus=False,
    )
    return EFETuner(config)
