"""Tests for kintsugi.cognition.model_router."""

from __future__ import annotations

import pytest

from kintsugi.cognition.model_router import ModelTier, ModelRouter, CostTracker


# ---------------------------------------------------------------------------
# ModelTier enum
# ---------------------------------------------------------------------------

class TestModelTier:
    def test_values(self):
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.BALANCED.value == "balanced"
        assert ModelTier.POWERFUL.value == "powerful"

    def test_is_str(self):
        assert isinstance(ModelTier.FAST, str)


# ---------------------------------------------------------------------------
# ModelRouter.resolve
# ---------------------------------------------------------------------------

class TestModelRouterResolve:
    def test_seed_always_returns_local(self):
        router = ModelRouter(deployment_tier="seed")
        for tier in ModelTier:
            assert router.resolve(tier) == "local/default"

    def test_sprout_resolves_each_tier(self):
        routing = {"haiku": "h-model", "sonnet": "s-model", "opus": "o-model"}
        router = ModelRouter(routing=routing, deployment_tier="sprout")
        assert router.resolve(ModelTier.FAST) == "h-model"
        assert router.resolve(ModelTier.BALANCED) == "s-model"
        assert router.resolve(ModelTier.POWERFUL) == "o-model"

    def test_missing_key_falls_back_to_local(self):
        router = ModelRouter(routing={"sonnet": "s"}, deployment_tier="sprout")
        # "haiku" key not in routing, so FAST falls back to local/default
        assert router.resolve(ModelTier.FAST) == "local/default"


# ---------------------------------------------------------------------------
# ModelRouter.resolve_for_task
# ---------------------------------------------------------------------------

class TestResolveForTask:
    def test_known_tasks(self):
        routing = {"haiku": "h", "sonnet": "s", "opus": "o"}
        router = ModelRouter(routing=routing, deployment_tier="sprout")
        assert router.resolve_for_task("coreference_resolution") == "h"
        assert router.resolve_for_task("consolidation_synthesis") == "s"
        assert router.resolve_for_task("architectural_reasoning") == "o"

    def test_unknown_task_defaults_to_balanced(self):
        routing = {"haiku": "h", "sonnet": "s", "opus": "o"}
        router = ModelRouter(routing=routing, deployment_tier="sprout")
        assert router.resolve_for_task("something_unknown") == "s"


# ---------------------------------------------------------------------------
# ModelRouter.estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        router = ModelRouter(deployment_tier="sprout")
        cost = router.estimate_cost("local/default", 1000, 1000)
        assert cost == 0.0

    def test_unknown_model_uses_default_rates(self):
        router = ModelRouter(deployment_tier="sprout")
        cost = router.estimate_cost("unknown-model", 1000, 1000)
        # default rates: (0.003, 0.015)
        assert pytest.approx(cost) == 0.003 + 0.015

    def test_haiku_cost(self):
        router = ModelRouter(deployment_tier="sprout")
        cost = router.estimate_cost("claude-3-5-haiku-20241022", 2000, 1000)
        assert pytest.approx(cost) == 2 * 0.001 + 1 * 0.005


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_record_and_cumulative(self):
        ct = CostTracker(session_budget=10.0)
        ct.record("m1", 1.5)
        ct.record("m2", 2.0)
        assert pytest.approx(ct.cumulative) == 3.5

    def test_remaining(self):
        ct = CostTracker(session_budget=5.0)
        ct.record("m", 3.0)
        assert pytest.approx(ct.remaining) == 2.0

    def test_budget_exhausted_raises(self):
        ct = CostTracker(session_budget=1.0)
        ct.record("m", 0.8)
        with pytest.raises(ValueError, match="budget exhausted"):
            ct.record("m", 0.3)

    def test_summary(self):
        ct = CostTracker(session_budget=10.0)
        ct.record("m", 1.0)
        s = ct.summary()
        assert s["session_budget"] == 10.0
        assert s["cumulative"] == 1.0
        assert s["remaining"] == 9.0
        assert s["call_count"] == 1
