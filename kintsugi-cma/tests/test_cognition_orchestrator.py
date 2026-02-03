"""Tests for kintsugi.cognition.orchestrator."""

from __future__ import annotations

import pytest

from kintsugi.cognition.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    RoutingDecision,
)
from kintsugi.cognition.model_router import ModelRouter, ModelTier


# ---------------------------------------------------------------------------
# OrchestratorConfig defaults
# ---------------------------------------------------------------------------

class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.fallback_domain == "general"
        assert cfg.confidence_threshold == 0.6
        assert cfg.routing_table == {}


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

class TestKeywordMatching:
    @pytest.fixture()
    def orch(self):
        return Orchestrator()

    @pytest.mark.asyncio
    async def test_grants(self, orch):
        d = await orch.classify_request("We need to write a grant proposal")
        assert d.skill_domain == "grants"

    @pytest.mark.asyncio
    async def test_volunteers(self, orch):
        d = await orch.classify_request("volunteer recruitment plan")
        assert d.skill_domain == "volunteers"

    @pytest.mark.asyncio
    async def test_finance(self, orch):
        d = await orch.classify_request("Show me the budget and expense report")
        assert d.skill_domain == "finance"

    @pytest.mark.asyncio
    async def test_impact(self, orch):
        d = await orch.classify_request("What are the outcome metrics?")
        assert d.skill_domain == "impact"

    @pytest.mark.asyncio
    async def test_communications(self, orch):
        d = await orch.classify_request("Draft the newsletter email")
        assert d.skill_domain == "communications"

    @pytest.mark.asyncio
    async def test_fallback_general(self, orch):
        d = await orch.classify_request("Hello, how are you today?")
        assert d.skill_domain == "general"
        assert d.confidence == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_confidence_increases_with_hits(self, orch):
        d = await orch.classify_request("grant funding proposal funder")
        assert d.confidence > 0.6


# ---------------------------------------------------------------------------
# register_domain / get_routing_table
# ---------------------------------------------------------------------------

class TestDomainManagement:
    def test_register_and_retrieve(self):
        orch = Orchestrator()
        orch.register_domain("hr", ["hiring", "Salary"])
        table = orch.get_routing_table()
        assert table["hiring"] == "hr"
        assert table["salary"] == "hr"

    def test_get_routing_table_is_copy(self):
        orch = Orchestrator()
        t = orch.get_routing_table()
        t["foo"] = "bar"
        assert "foo" not in orch.get_routing_table()


# ---------------------------------------------------------------------------
# classify_request with LLM classifier
# ---------------------------------------------------------------------------

class TestLLMClassifier:
    @pytest.mark.asyncio
    async def test_llm_used_when_low_confidence(self):
        async def fake_llm(message, domains):
            return ("grants", 0.9)

        orch = Orchestrator(llm_classifier=fake_llm)
        d = await orch.classify_request("xyz nothing matches here")
        assert d.skill_domain == "grants"
        assert d.confidence == 0.9
        assert d.reasoning == "LLM classification"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self):
        async def bad_llm(message, domains):
            raise RuntimeError("boom")

        orch = Orchestrator(llm_classifier=bad_llm)
        d = await orch.classify_request("xyz nothing matches here")
        assert d.skill_domain == "general"

    @pytest.mark.asyncio
    async def test_llm_not_used_when_high_confidence(self):
        called = False

        async def spy_llm(message, domains):
            nonlocal called
            called = True
            return ("grants", 0.99)

        orch = Orchestrator(llm_classifier=spy_llm)
        d = await orch.classify_request("budget expense revenue invoice financial accounting")
        assert not called


# ---------------------------------------------------------------------------
# route() method
# ---------------------------------------------------------------------------

class TestRoute:
    @pytest.mark.asyncio
    async def test_route_returns_decision(self):
        orch = Orchestrator()
        d = await orch.route("Show the budget", org_id="org-1")
        assert isinstance(d, RoutingDecision)
        assert d.skill_domain == "finance"
