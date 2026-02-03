"""Tiered model allocation and cost tracking.

Maps abstract model tiers (FAST / BALANCED / POWERFUL) to concrete model IDs
based on the deployment tier and ``settings.MODEL_ROUTING`` configuration.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field

from kintsugi.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------


class ModelTier(str, enum.Enum):
    """Abstract capability tiers, independent of any provider."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


# ---------------------------------------------------------------------------
# Task-type → tier mapping
# ---------------------------------------------------------------------------

_TASK_TIER_MAP: dict[str, ModelTier] = {
    # Fast / haiku-class
    "coreference_resolution": ModelTier.FAST,
    "timestamp_anchoring": ModelTier.FAST,
    "fact_extraction": ModelTier.FAST,
    # Balanced / sonnet-class
    "consolidation_synthesis": ModelTier.BALANCED,
    "quality_judgment": ModelTier.BALANCED,
    "behavioral_profile": ModelTier.BALANCED,
    # Powerful / opus-class
    "architectural_reasoning": ModelTier.POWERFUL,
    "adversarial_generation": ModelTier.POWERFUL,
    "meta_analysis": ModelTier.POWERFUL,
}

# ---------------------------------------------------------------------------
# Rough per-token cost table (USD) — used only for estimation
# ---------------------------------------------------------------------------

_COST_PER_1K: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "claude-3-5-haiku-20241022": (0.001, 0.005),
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-opus-4-20250514": (0.015, 0.075),
    "local/default": (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class ModelRouter:
    """Resolve abstract tiers to concrete model identifiers.

    Parameters
    ----------
    routing:
        Explicit ``{haiku/sonnet/opus: model_id}`` mapping.  Falls back to
        ``settings.MODEL_ROUTING``.
    deployment_tier:
        One of ``seed``, ``sprout``, ``grove``.  Seed tier always resolves
        to a local model identifier.
    """

    _TIER_TO_ROUTING_KEY: dict[ModelTier, str] = {
        ModelTier.FAST: "haiku",
        ModelTier.BALANCED: "sonnet",
        ModelTier.POWERFUL: "opus",
    }

    def __init__(
        self,
        routing: dict[str, str] | None = None,
        deployment_tier: str = "sprout",
    ) -> None:
        self._routing = routing or dict(settings.MODEL_ROUTING)
        self._deployment_tier = deployment_tier or settings.DEPLOYMENT_TIER

    # -- public API ---------------------------------------------------------

    def resolve(self, tier: ModelTier) -> str:
        """Return the concrete model ID for *tier*.

        For the ``seed`` deployment tier every tier resolves to
        ``"local/default"`` so the engine can run fully offline.
        """
        if self._deployment_tier == "seed":
            return "local/default"
        key = self._TIER_TO_ROUTING_KEY[tier]
        return self._routing.get(key, "local/default")

    def resolve_for_task(self, task_type: str) -> str:
        """Map a task-type string to its concrete model ID."""
        tier = _TASK_TIER_MAP.get(task_type)
        if tier is None:
            logger.warning("Unknown task type %r — defaulting to BALANCED", task_type)
            tier = ModelTier.BALANCED
        return self.resolve(tier)

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return a rough USD cost estimate for a single call."""
        rates = _COST_PER_1K.get(model_id, (0.003, 0.015))
        return (input_tokens / 1000) * rates[0] + (output_tokens / 1000) * rates[1]


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


@dataclass
class CostTracker:
    """Cumulative cost tracking scoped to a session.

    Integrates with the Shield budget concept: calls to :meth:`record` will
    raise ``ValueError`` when the session budget is exhausted.

    Parameters
    ----------
    session_budget:
        Maximum USD spend for this session.  Defaults to
        ``settings.SHIELD_BUDGET_PER_SESSION``.
    """

    session_budget: float = field(default_factory=lambda: settings.SHIELD_BUDGET_PER_SESSION)
    _cumulative: float = field(default=0.0, init=False, repr=False)
    _records: list[dict] = field(default_factory=list, init=False, repr=False)

    @property
    def cumulative(self) -> float:
        """Total USD spent so far in this session."""
        return self._cumulative

    @property
    def remaining(self) -> float:
        """Budget remaining."""
        return max(0.0, self.session_budget - self._cumulative)

    def record(self, model_id: str, cost: float) -> None:
        """Record a cost entry.  Raises if budget would be exceeded."""
        if self._cumulative + cost > self.session_budget:
            raise ValueError(
                f"Session budget exhausted: {self._cumulative + cost:.4f} > "
                f"{self.session_budget:.4f}"
            )
        self._cumulative += cost
        self._records.append(
            {"model_id": model_id, "cost": cost, "ts": time.time()}
        )

    def summary(self) -> dict:
        """Return a summary dict suitable for logging / temporal memory."""
        return {
            "session_budget": self.session_budget,
            "cumulative": self._cumulative,
            "remaining": self.remaining,
            "call_count": len(self._records),
        }
