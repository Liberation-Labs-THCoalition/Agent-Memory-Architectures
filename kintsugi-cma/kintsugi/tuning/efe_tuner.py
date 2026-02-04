"""
EFE Auto-Tuning - Ethical Framing Engine Weight Optimization

This module provides automated tuning of EFE weights based on decision outcomes
and stakeholder feedback. The tuner supports multiple optimization strategies
while maintaining ethical guardrails and audit trails.

The tuning process:
1. Records decision outcomes with the weights that were used
2. Collects stakeholder feedback on decisions
3. Computes gradients or uses other optimization methods
4. Proposes new weights within safety bounds
5. Requires consensus approval for significant changes
6. Applies weights and records history for rollback

Safety mechanisms prevent runaway optimization and ensure human oversight.
"""

from __future__ import annotations

import math
import random
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterator


class TuningStrategy(str, Enum):
    """Available tuning strategies for EFE weight optimization."""

    GRADIENT = "gradient"  # Gradient descent on outcome scores
    EVOLUTIONARY = "evolutionary"  # Evolutionary algorithm
    BAYESIAN = "bayesian"  # Bayesian optimization
    MANUAL = "manual"  # Human-in-the-loop only


@dataclass
class TuningOutcome:
    """
    Recorded outcome for tuning feedback.

    Captures the result of a decision along with the EFE weights
    that were used, enabling learning from outcomes.

    Attributes:
        decision_id: Unique identifier for the decision
        efe_weights_used: The EFE weights active during the decision
        outcome_score: Overall outcome quality (-1.0 to 1.0)
        stakeholder_feedback: Per-stakeholder feedback scores
        timestamp: When the outcome was recorded
        metadata: Additional context about the outcome
    """

    decision_id: str
    efe_weights_used: dict[str, float]
    outcome_score: float  # -1.0 to 1.0
    stakeholder_feedback: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate outcome score is in range."""
        if not -1.0 <= self.outcome_score <= 1.0:
            raise ValueError(f"outcome_score must be in [-1.0, 1.0], got {self.outcome_score}")


@dataclass
class WeightConstraint:
    """
    Constraint on a specific weight parameter.

    Attributes:
        name: Name of the weight parameter
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default_value: Default/initial value
        locked: If True, weight cannot be changed
    """

    name: str
    min_value: float = 0.0
    max_value: float = 1.0
    default_value: float = 0.5
    locked: bool = False

    def clamp(self, value: float) -> float:
        """Clamp value to constraint bounds."""
        if self.locked:
            return self.default_value
        return max(self.min_value, min(self.max_value, value))

    def is_valid(self, value: float) -> bool:
        """Check if value satisfies constraint."""
        if self.locked:
            return value == self.default_value
        return self.min_value <= value <= self.max_value


@dataclass
class WeightProposal:
    """
    Proposed weight changes from tuning.

    Attributes:
        proposal_id: Unique identifier for this proposal
        current_weights: Current EFE weights
        proposed_weights: Proposed new weights
        confidence: Confidence in the proposal (0.0 to 1.0)
        rationale: Explanation of why these changes are proposed
        expected_improvement: Expected improvement in outcome scores
        created_at: When the proposal was created
        approved_by: Who approved the proposal (if approved)
        approved_at: When the proposal was approved
    """

    proposal_id: str
    current_weights: dict[str, float]
    proposed_weights: dict[str, float]
    confidence: float
    rationale: str
    expected_improvement: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: str | None = None
    approved_at: datetime | None = None

    def get_changes(self) -> dict[str, tuple[float, float]]:
        """Get dict of weight changes as (old, new) tuples."""
        changes = {}
        all_keys = set(self.current_weights.keys()) | set(self.proposed_weights.keys())
        for key in all_keys:
            old = self.current_weights.get(key, 0.0)
            new = self.proposed_weights.get(key, 0.0)
            if old != new:
                changes[key] = (old, new)
        return changes

    def max_change_magnitude(self) -> float:
        """Get the maximum magnitude of any single weight change."""
        changes = self.get_changes()
        if not changes:
            return 0.0
        return max(abs(new - old) for old, new in changes.values())


@dataclass
class TuningCycle:
    """
    Record of a single tuning cycle.

    Attributes:
        cycle_id: Unique identifier for this cycle
        started_at: When the cycle started
        completed_at: When the cycle completed
        outcomes_used: Number of outcomes used for tuning
        strategy_used: Which strategy was employed
        weights_before: Weights before the cycle
        weights_after: Weights after the cycle
        improvement_achieved: Actual improvement measured
        notes: Additional notes about the cycle
    """

    cycle_id: str
    started_at: datetime
    completed_at: datetime | None = None
    outcomes_used: int = 0
    strategy_used: TuningStrategy = TuningStrategy.GRADIENT
    weights_before: dict[str, float] = field(default_factory=dict)
    weights_after: dict[str, float] = field(default_factory=dict)
    improvement_achieved: float | None = None
    notes: str = ""


@dataclass
class TuningMetrics:
    """
    Metrics from tuning operations.

    Attributes:
        total_outcomes: Total outcomes recorded
        outcomes_since_last_tune: Outcomes since last tuning
        average_outcome_score: Average of recent outcome scores
        outcome_score_trend: Trend direction of outcome scores
        cycles_completed: Number of tuning cycles completed
        last_tune_timestamp: When last tuning occurred
        weight_stability: How stable weights have been (0-1)
    """

    total_outcomes: int = 0
    outcomes_since_last_tune: int = 0
    average_outcome_score: float = 0.0
    outcome_score_trend: float = 0.0  # Positive = improving
    cycles_completed: int = 0
    last_tune_timestamp: datetime | None = None
    weight_stability: float = 1.0  # 1.0 = completely stable


@dataclass
class TuningConfig:
    """
    Configuration for EFE auto-tuning.

    Attributes:
        strategy: Which tuning strategy to use
        learning_rate: Learning rate for gradient-based methods
        min_samples: Minimum outcomes before tuning
        max_weight_change: Maximum change per tuning cycle
        require_consensus: Whether to require approval for changes
        momentum: Momentum factor for gradient descent
        weight_decay: L2 regularization strength
        exploration_rate: Exploration rate for evolutionary/Bayesian
        population_size: Population size for evolutionary algorithm
        elite_fraction: Fraction of population to keep as elite
    """

    strategy: TuningStrategy = TuningStrategy.GRADIENT
    learning_rate: float = 0.01
    min_samples: int = 50
    max_weight_change: float = 0.1
    require_consensus: bool = True
    momentum: float = 0.9
    weight_decay: float = 0.0001
    exploration_rate: float = 0.1
    population_size: int = 20
    elite_fraction: float = 0.2

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if self.min_samples < 1:
            errors.append(f"min_samples must be >= 1, got {self.min_samples}")
        if self.max_weight_change <= 0 or self.max_weight_change > 1:
            errors.append(f"max_weight_change must be in (0, 1], got {self.max_weight_change}")
        if self.momentum < 0 or self.momentum >= 1:
            errors.append(f"momentum must be in [0, 1), got {self.momentum}")
        if self.weight_decay < 0:
            errors.append(f"weight_decay must be >= 0, got {self.weight_decay}")
        return errors


class EFETuner:
    """
    Auto-tunes EFE weights based on outcome feedback.

    The tuner collects decision outcomes and stakeholder feedback,
    then uses the configured optimization strategy to propose
    improved weights. Safety mechanisms ensure changes are bounded
    and require consensus when configured.

    Example:
        >>> config = TuningConfig(strategy=TuningStrategy.GRADIENT)
        >>> tuner = EFETuner(config)
        >>> tuner.set_initial_weights({"autonomy": 0.3, "beneficence": 0.4})
        >>>
        >>> # Record outcomes over time
        >>> outcome = TuningOutcome(
        ...     decision_id="d1",
        ...     efe_weights_used={"autonomy": 0.3, "beneficence": 0.4},
        ...     outcome_score=0.7,
        ... )
        >>> tuner.record_outcome(outcome)
        >>>
        >>> # When ready, propose and apply new weights
        >>> if tuner.should_tune():
        ...     proposed = tuner.propose_weights()
        ...     tuner.apply_weights(proposed, approver="admin")
    """

    def __init__(self, config: TuningConfig | None = None):
        """
        Initialize the EFE tuner.

        Args:
            config: Tuning configuration. Uses defaults if None.
        """
        self._config = config or TuningConfig()
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        self._outcomes: list[TuningOutcome] = []
        self._current_weights: dict[str, float] = {}
        self._weight_history: list[tuple[datetime, dict[str, float]]] = []
        self._velocity: dict[str, float] = {}  # For momentum
        self._constraints: dict[str, WeightConstraint] = {}
        self._cycles: list[TuningCycle] = []
        self._pending_proposal: WeightProposal | None = None
        self._outcome_window: int = 200  # Keep last N outcomes for analysis

    @property
    def config(self) -> TuningConfig:
        """Get the tuning configuration."""
        return self._config

    @property
    def current_weights(self) -> dict[str, float]:
        """Get current EFE weights."""
        return dict(self._current_weights)

    @property
    def metrics(self) -> TuningMetrics:
        """Get current tuning metrics."""
        return self._compute_metrics()

    def set_initial_weights(self, weights: dict[str, float]) -> None:
        """
        Set initial EFE weights.

        Args:
            weights: Initial weight values
        """
        self._current_weights = dict(weights)
        self._weight_history.append((datetime.now(timezone.utc), dict(weights)))
        # Initialize velocity for momentum
        self._velocity = {k: 0.0 for k in weights}

    def set_constraint(self, constraint: WeightConstraint) -> None:
        """
        Set a constraint on a weight parameter.

        Args:
            constraint: The constraint to set
        """
        self._constraints[constraint.name] = constraint

    def get_constraint(self, name: str) -> WeightConstraint | None:
        """Get constraint for a weight, if any."""
        return self._constraints.get(name)

    def record_outcome(self, outcome: TuningOutcome) -> None:
        """
        Record a decision outcome for future tuning.

        Args:
            outcome: The outcome to record
        """
        self._outcomes.append(outcome)
        # Trim old outcomes to maintain window
        if len(self._outcomes) > self._outcome_window * 2:
            self._outcomes = self._outcomes[-self._outcome_window:]

    def should_tune(self) -> bool:
        """
        Check if we have enough data to tune.

        Returns:
            True if tuning should be performed
        """
        if self._config.strategy == TuningStrategy.MANUAL:
            return False

        outcomes_since_last = self._count_outcomes_since_last_tune()
        return outcomes_since_last >= self._config.min_samples

    def _count_outcomes_since_last_tune(self) -> int:
        """Count outcomes since the last tuning cycle."""
        if not self._cycles:
            return len(self._outcomes)

        last_cycle = self._cycles[-1]
        if last_cycle.completed_at is None:
            return 0

        count = 0
        for outcome in reversed(self._outcomes):
            if outcome.timestamp > last_cycle.completed_at:
                count += 1
            else:
                break
        return count

    def compute_gradients(self) -> dict[str, float]:
        """
        Compute weight gradients from outcomes.

        Uses finite differences to estimate how each weight
        affects outcome scores.

        Returns:
            Dictionary of weight names to gradient values
        """
        if len(self._outcomes) < 2:
            return {k: 0.0 for k in self._current_weights}

        # Get recent outcomes
        recent = self._outcomes[-self._outcome_window:]

        gradients = {}
        for weight_name in self._current_weights:
            gradient = self._compute_weight_gradient(weight_name, recent)
            gradients[weight_name] = gradient

        return gradients

    def _compute_weight_gradient(
        self, weight_name: str, outcomes: list[TuningOutcome]
    ) -> float:
        """
        Compute gradient for a single weight using correlation analysis.

        This estimates how changes in the weight correlate with outcome scores.
        """
        weight_values = []
        scores = []

        for outcome in outcomes:
            if weight_name in outcome.efe_weights_used:
                weight_values.append(outcome.efe_weights_used[weight_name])
                scores.append(outcome.outcome_score)

        if len(weight_values) < 3:
            return 0.0

        # Compute correlation coefficient as proxy for gradient
        mean_w = statistics.mean(weight_values)
        mean_s = statistics.mean(scores)

        numerator = sum(
            (w - mean_w) * (s - mean_s) for w, s in zip(weight_values, scores)
        )
        w_var = sum((w - mean_w) ** 2 for w in weight_values)
        s_var = sum((s - mean_s) ** 2 for s in scores)

        if w_var == 0 or s_var == 0:
            return 0.0

        correlation = numerator / math.sqrt(w_var * s_var)

        # Scale correlation to reasonable gradient magnitude
        return correlation * 0.1

    def propose_weights(self) -> dict[str, float]:
        """
        Propose new weights based on outcomes.

        Uses the configured strategy to compute weight updates.

        Returns:
            Dictionary of proposed weight values
        """
        if self._config.strategy == TuningStrategy.GRADIENT:
            return self._propose_gradient()
        elif self._config.strategy == TuningStrategy.EVOLUTIONARY:
            return self._propose_evolutionary()
        elif self._config.strategy == TuningStrategy.BAYESIAN:
            return self._propose_bayesian()
        else:
            # Manual strategy - no changes
            return dict(self._current_weights)

    def _propose_gradient(self) -> dict[str, float]:
        """Propose weights using gradient descent with momentum."""
        gradients = self.compute_gradients()
        proposed = {}

        for name, current in self._current_weights.items():
            gradient = gradients.get(name, 0.0)

            # Apply momentum
            velocity = self._velocity.get(name, 0.0)
            velocity = self._config.momentum * velocity + gradient
            self._velocity[name] = velocity

            # Compute update with weight decay
            update = (
                self._config.learning_rate * velocity
                - self._config.weight_decay * current
            )

            # Clamp update to max change
            update = max(-self._config.max_weight_change,
                        min(self._config.max_weight_change, update))

            new_value = current + update

            # Apply constraints
            constraint = self._constraints.get(name)
            if constraint:
                new_value = constraint.clamp(new_value)
            else:
                new_value = max(0.0, min(1.0, new_value))

            proposed[name] = new_value

        return proposed

    def _propose_evolutionary(self) -> dict[str, float]:
        """Propose weights using evolutionary algorithm."""
        # Generate population
        population = []
        for _ in range(self._config.population_size):
            individual = {}
            for name, current in self._current_weights.items():
                # Mutate with Gaussian noise
                mutation = random.gauss(0, self._config.exploration_rate)
                mutation = max(-self._config.max_weight_change,
                              min(self._config.max_weight_change, mutation))
                new_value = current + mutation

                constraint = self._constraints.get(name)
                if constraint:
                    new_value = constraint.clamp(new_value)
                else:
                    new_value = max(0.0, min(1.0, new_value))

                individual[name] = new_value
            population.append(individual)

        # Evaluate fitness based on similarity to high-scoring outcomes
        scored = []
        for individual in population:
            fitness = self._evaluate_fitness(individual)
            scored.append((fitness, individual))

        # Select elite
        scored.sort(reverse=True, key=lambda x: x[0])
        elite_count = max(1, int(self._config.elite_fraction * len(scored)))
        elite = [ind for _, ind in scored[:elite_count]]

        # Return average of elite
        proposed = {}
        for name in self._current_weights:
            values = [ind[name] for ind in elite]
            proposed[name] = statistics.mean(values)

        return proposed

    def _evaluate_fitness(self, weights: dict[str, float]) -> float:
        """Evaluate fitness of a weight configuration."""
        if not self._outcomes:
            return 0.0

        # Compute similarity to high-scoring outcomes
        total_score = 0.0
        total_weight = 0.0

        for outcome in self._outcomes[-self._outcome_window:]:
            # Weight by outcome score (positive outcomes matter more)
            outcome_weight = max(0, outcome.outcome_score + 1) / 2

            # Compute similarity
            similarity = self._weight_similarity(weights, outcome.efe_weights_used)
            total_score += similarity * outcome_weight * outcome.outcome_score
            total_weight += outcome_weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _weight_similarity(
        self, weights1: dict[str, float], weights2: dict[str, float]
    ) -> float:
        """Compute similarity between two weight configurations."""
        if not weights1 or not weights2:
            return 0.0

        common_keys = set(weights1.keys()) & set(weights2.keys())
        if not common_keys:
            return 0.0

        squared_diff = sum(
            (weights1[k] - weights2[k]) ** 2 for k in common_keys
        )
        distance = math.sqrt(squared_diff / len(common_keys))

        # Convert distance to similarity (0-1)
        return math.exp(-distance * 5)

    def _propose_bayesian(self) -> dict[str, float]:
        """
        Propose weights using Bayesian optimization.

        This is a simplified version that uses upper confidence bound
        acquisition on a Gaussian process approximation.
        """
        # For simplicity, use a hybrid approach:
        # Start with gradient estimate, add exploration bonus
        base_proposal = self._propose_gradient()

        # Add exploration based on uncertainty (variance of outcomes)
        if len(self._outcomes) >= 10:
            recent_scores = [o.outcome_score for o in self._outcomes[-50:]]
            uncertainty = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.5
        else:
            uncertainty = 0.5

        proposed = {}
        for name, base_value in base_proposal.items():
            # Add exploration noise proportional to uncertainty
            exploration = random.gauss(0, uncertainty * self._config.exploration_rate)
            exploration = max(-self._config.max_weight_change / 2,
                            min(self._config.max_weight_change / 2, exploration))

            new_value = base_value + exploration

            constraint = self._constraints.get(name)
            if constraint:
                new_value = constraint.clamp(new_value)
            else:
                new_value = max(0.0, min(1.0, new_value))

            proposed[name] = new_value

        return proposed

    def create_proposal(self) -> WeightProposal:
        """
        Create a formal weight proposal for review.

        Returns:
            WeightProposal with current and proposed weights
        """
        proposed = self.propose_weights()
        gradients = self.compute_gradients()

        # Generate rationale
        changes = []
        for name in self._current_weights:
            old = self._current_weights[name]
            new = proposed.get(name, old)
            if abs(new - old) > 0.001:
                direction = "increase" if new > old else "decrease"
                gradient = gradients.get(name, 0)
                changes.append(
                    f"{name}: {direction} by {abs(new - old):.3f} "
                    f"(gradient: {gradient:.4f})"
                )

        rationale = (
            f"Based on {len(self._outcomes)} outcomes using {self._config.strategy.value} "
            f"strategy. Changes: {'; '.join(changes) if changes else 'No changes'}"
        )

        # Estimate expected improvement
        current_fitness = self._evaluate_fitness(self._current_weights)
        proposed_fitness = self._evaluate_fitness(proposed)
        expected_improvement = proposed_fitness - current_fitness

        proposal = WeightProposal(
            proposal_id=str(uuid.uuid4()),
            current_weights=dict(self._current_weights),
            proposed_weights=proposed,
            confidence=min(1.0, len(self._outcomes) / (self._config.min_samples * 2)),
            rationale=rationale,
            expected_improvement=expected_improvement,
        )

        self._pending_proposal = proposal
        return proposal

    def apply_weights(
        self, weights: dict[str, float], approver: str | None = None
    ) -> None:
        """
        Apply new weights (may require consensus).

        Args:
            weights: New weight values to apply
            approver: Who approved the change (required if consensus needed)

        Raises:
            ValueError: If consensus required but no approver provided
        """
        if self._config.require_consensus and not approver:
            raise ValueError("Consensus required: must provide approver")

        # Validate all weights are in bounds
        for name, value in weights.items():
            constraint = self._constraints.get(name)
            if constraint and not constraint.is_valid(value):
                raise ValueError(
                    f"Weight {name}={value} violates constraint "
                    f"[{constraint.min_value}, {constraint.max_value}]"
                )

        # Record history
        now = datetime.now(timezone.utc)
        self._weight_history.append((now, dict(weights)))

        # Create cycle record
        cycle = TuningCycle(
            cycle_id=str(uuid.uuid4()),
            started_at=now,
            completed_at=now,
            outcomes_used=self._count_outcomes_since_last_tune(),
            strategy_used=self._config.strategy,
            weights_before=dict(self._current_weights),
            weights_after=dict(weights),
            notes=f"Approved by: {approver}" if approver else "",
        )
        self._cycles.append(cycle)

        # Apply weights
        self._current_weights = dict(weights)

        # Clear pending proposal
        if self._pending_proposal:
            self._pending_proposal.approved_by = approver
            self._pending_proposal.approved_at = now
            self._pending_proposal = None

    def rollback(self, steps: int = 1) -> dict[str, float]:
        """
        Rollback to previous weights.

        Args:
            steps: Number of steps to rollback

        Returns:
            The weights after rollback
        """
        if len(self._weight_history) <= steps:
            raise ValueError(f"Cannot rollback {steps} steps, only {len(self._weight_history)} in history")

        # Get target weights
        target_idx = -(steps + 1)
        _, target_weights = self._weight_history[target_idx]

        # Apply as new weights
        self._current_weights = dict(target_weights)
        self._weight_history.append((datetime.now(timezone.utc), dict(target_weights)))

        return dict(target_weights)

    def get_tuning_report(self) -> dict:
        """
        Generate report on tuning state and recommendations.

        Returns:
            Dictionary containing tuning status, metrics, and recommendations
        """
        metrics = self._compute_metrics()
        gradients = self.compute_gradients()

        # Generate recommendations
        recommendations = []
        if metrics.outcomes_since_last_tune < self._config.min_samples:
            needed = self._config.min_samples - metrics.outcomes_since_last_tune
            recommendations.append(f"Need {needed} more outcomes before tuning")

        if metrics.outcome_score_trend < -0.1:
            recommendations.append("Outcome scores trending down - consider reviewing weights")

        if metrics.weight_stability < 0.5:
            recommendations.append("Weights have been unstable - consider reducing learning rate")

        # Find weights with strongest gradients
        sorted_gradients = sorted(
            gradients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, grad in sorted_gradients[:3]:
            if abs(grad) > 0.02:
                direction = "increase" if grad > 0 else "decrease"
                recommendations.append(f"Consider {direction} for {name} (gradient: {grad:.4f})")

        return {
            "status": "ready" if self.should_tune() else "collecting",
            "strategy": self._config.strategy.value,
            "metrics": {
                "total_outcomes": metrics.total_outcomes,
                "outcomes_since_last_tune": metrics.outcomes_since_last_tune,
                "average_outcome_score": round(metrics.average_outcome_score, 4),
                "outcome_score_trend": round(metrics.outcome_score_trend, 4),
                "cycles_completed": metrics.cycles_completed,
                "weight_stability": round(metrics.weight_stability, 4),
            },
            "current_weights": dict(self._current_weights),
            "gradients": {k: round(v, 6) for k, v in gradients.items()},
            "recommendations": recommendations,
            "pending_proposal": self._pending_proposal is not None,
        }

    def _compute_metrics(self) -> TuningMetrics:
        """Compute current tuning metrics."""
        total = len(self._outcomes)
        since_last = self._count_outcomes_since_last_tune()

        # Compute average and trend
        if total > 0:
            recent = [o.outcome_score for o in self._outcomes[-50:]]
            avg_score = statistics.mean(recent)

            # Compute trend using linear regression
            if len(recent) >= 5:
                n = len(recent)
                x_mean = (n - 1) / 2
                y_mean = avg_score
                numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                trend = numerator / denominator if denominator != 0 else 0.0
            else:
                trend = 0.0
        else:
            avg_score = 0.0
            trend = 0.0

        # Compute weight stability
        if len(self._weight_history) >= 2:
            recent_weights = [w for _, w in self._weight_history[-10:]]
            if len(recent_weights) >= 2:
                variances = []
                for name in self._current_weights:
                    values = [w.get(name, 0) for w in recent_weights]
                    if len(values) > 1:
                        variances.append(statistics.variance(values))
                stability = 1.0 / (1.0 + sum(variances) * 10) if variances else 1.0
            else:
                stability = 1.0
        else:
            stability = 1.0

        return TuningMetrics(
            total_outcomes=total,
            outcomes_since_last_tune=since_last,
            average_outcome_score=avg_score,
            outcome_score_trend=trend,
            cycles_completed=len(self._cycles),
            last_tune_timestamp=self._cycles[-1].completed_at if self._cycles else None,
            weight_stability=stability,
        )

    def export_state(self) -> dict:
        """Export tuner state for persistence."""
        return {
            "config": {
                "strategy": self._config.strategy.value,
                "learning_rate": self._config.learning_rate,
                "min_samples": self._config.min_samples,
                "max_weight_change": self._config.max_weight_change,
                "require_consensus": self._config.require_consensus,
            },
            "current_weights": dict(self._current_weights),
            "velocity": dict(self._velocity),
            "outcomes": [
                {
                    "decision_id": o.decision_id,
                    "efe_weights_used": o.efe_weights_used,
                    "outcome_score": o.outcome_score,
                    "timestamp": o.timestamp.isoformat(),
                }
                for o in self._outcomes[-100:]  # Keep last 100
            ],
            "cycles_completed": len(self._cycles),
        }

    def import_state(self, state: dict) -> None:
        """Import tuner state from persistence."""
        if "current_weights" in state:
            self._current_weights = dict(state["current_weights"])
        if "velocity" in state:
            self._velocity = dict(state["velocity"])
        # Note: Full outcome restoration would require more complete serialization
