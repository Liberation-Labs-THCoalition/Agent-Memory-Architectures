"""Tests for kintsugi.config.values_schema."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from kintsugi.config.values_schema import (
    Belief,
    Desire,
    Intention,
    ImpactBenchmark,
    Principle,
    Organization,
    Beliefs,
    Desires,
    Intentions,
    Principles,
    KintsugiGovernance,
    Shield,
    OrganizationValues,
)


# ---------------------------------------------------------------------------
# BDI primitives
# ---------------------------------------------------------------------------

class TestBelief:
    def test_valid(self):
        b = Belief(content="X", confidence=0.5, source="obs")
        assert b.content == "X"
        assert b.last_verified is None

    def test_with_datetime(self):
        b = Belief(content="X", confidence=1.0, source="s", last_verified=datetime(2024, 1, 1))
        assert b.last_verified == datetime(2024, 1, 1)

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Belief(content="X", confidence=1.1, source="s")

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            Belief(content="X", confidence=-0.1, source="s")

    def test_empty_content(self):
        with pytest.raises(ValidationError):
            Belief(content="", confidence=0.5, source="s")

    def test_empty_source(self):
        with pytest.raises(ValidationError):
            Belief(content="X", confidence=0.5, source="")


class TestDesire:
    def test_valid(self):
        d = Desire(content="goal", priority=3)
        assert d.measurable is False
        assert d.metric is None

    def test_priority_bounds(self):
        Desire(content="a", priority=1)
        Desire(content="a", priority=5)
        with pytest.raises(ValidationError):
            Desire(content="a", priority=0)
        with pytest.raises(ValidationError):
            Desire(content="a", priority=6)

    def test_with_metric(self):
        d = Desire(content="a", priority=2, measurable=True, metric="count")
        assert d.measurable is True
        assert d.metric == "count"


class TestIntention:
    def test_defaults(self):
        i = Intention(content="do thing")
        assert i.status == "active"
        assert i.started is None
        assert i.deadline is None

    def test_statuses(self):
        for s in ("active", "paused", "completed"):
            Intention(content="x", status=s)

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            Intention(content="x", status="cancelled")


class TestImpactBenchmark:
    def test_valid_sdg(self):
        b = ImpactBenchmark(metric="m", target="t", sdg_alignment=[1, 17])
        assert b.sdg_alignment == [1, 17]

    def test_sdg_too_low(self):
        with pytest.raises(ValidationError):
            ImpactBenchmark(metric="m", target="t", sdg_alignment=[0])

    def test_sdg_too_high(self):
        with pytest.raises(ValidationError):
            ImpactBenchmark(metric="m", target="t", sdg_alignment=[18])

    def test_empty_sdg_ok(self):
        b = ImpactBenchmark(metric="m", target="t")
        assert b.sdg_alignment == []

    def test_current_optional(self):
        b = ImpactBenchmark(metric="m", target="t")
        assert b.current is None


class TestPrinciple:
    def test_valid(self):
        p = Principle(name="n", description="d")
        assert p.is_bright_line is False

    def test_bright_line(self):
        p = Principle(name="n", description="d", is_bright_line=True)
        assert p.is_bright_line is True

    def test_empty_name(self):
        with pytest.raises(ValidationError):
            Principle(name="", description="d")


# ---------------------------------------------------------------------------
# Top-level sections
# ---------------------------------------------------------------------------

class TestOrganization:
    def test_valid(self):
        o = Organization(name="Org", mission="Help")
        assert o.type == "other"
        assert o.size == "small"
        assert o.founded is None

    def test_all_types(self):
        for t in ("mutual_aid", "nonprofit_501c3", "cooperative", "advocacy", "other"):
            Organization(name="O", mission="M", type=t)

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            Organization(name="O", mission="M", type="forprofit")

    def test_invalid_size(self):
        with pytest.raises(ValidationError):
            Organization(name="O", mission="M", size="huge")


class TestBeliefs:
    def test_defaults(self):
        b = Beliefs()
        assert b.environment == []
        assert b.capabilities == []
        assert b.last_reviewed is None


class TestDesires:
    def test_defaults(self):
        d = Desires()
        assert d.values == []
        assert d.impact_benchmarks == []


class TestIntentions:
    def test_defaults(self):
        i = Intentions()
        assert i.active_strategies == []
        assert i.campaigns == []
        assert i.grants == []


class TestPrinciples:
    def test_defaults(self):
        p = Principles()
        assert p.transparency_level == "full"
        assert p.custom == []
        assert "equitable" in p.equity_mandate.lower()


class TestKintsugiGovernance:
    def test_defaults(self):
        k = KintsugiGovernance()
        assert k.shadow_verification is True
        assert k.divergence_threshold == 0.15
        assert "financial" in k.consensus_required_for
        assert k.bloom_schedule == "weekly"
        assert k.max_modification_scope == "tool_config"

    def test_divergence_bounds(self):
        KintsugiGovernance(divergence_threshold=0.0)
        KintsugiGovernance(divergence_threshold=1.0)
        with pytest.raises(ValidationError):
            KintsugiGovernance(divergence_threshold=-0.1)
        with pytest.raises(ValidationError):
            KintsugiGovernance(divergence_threshold=1.1)

    def test_invalid_bloom_schedule(self):
        with pytest.raises(ValidationError):
            KintsugiGovernance(bloom_schedule="daily")

    def test_invalid_mod_scope(self):
        with pytest.raises(ValidationError):
            KintsugiGovernance(max_modification_scope="kernel")


class TestShield:
    def test_defaults(self):
        s = Shield()
        assert s.budget_per_session == 5.0
        assert s.budget_per_day == 50.0
        assert s.egress_allowlist == []
        assert s.blocked_patterns == []

    def test_negative_budget(self):
        with pytest.raises(ValidationError):
            Shield(budget_per_session=-1)


class TestOrganizationValues:
    def _minimal_org(self):
        return {"name": "Test", "mission": "Do good"}

    def test_minimal(self):
        v = OrganizationValues(organization=self._minimal_org())
        assert v.organization.name == "Test"
        assert isinstance(v.beliefs, Beliefs)
        assert isinstance(v.shield, Shield)

    def test_missing_org_fails(self):
        with pytest.raises(ValidationError):
            OrganizationValues()
