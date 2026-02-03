"""Tests for kintsugi_engine.drift – Phase 3 Stream 3C."""

import pytest
from datetime import datetime, timedelta, timezone

from kintsugi.kintsugi_engine.drift import (
    DriftCategory,
    DriftConfig,
    DriftDetector,
    DriftEvent,
)


class TestDriftCategory:
    def test_values(self):
        assert DriftCategory.HEALTHY_ADAPTATION.value == "healthy_adaptation"
        assert DriftCategory.STALE_BELIEFS.value == "stale_beliefs"
        assert DriftCategory.INTENTION_DRIFT.value == "intention_drift"
        assert DriftCategory.VALUES_TENSION.value == "values_tension"

    def test_count(self):
        assert len(DriftCategory) == 4


class TestDriftEvent:
    def _make(self, **kw):
        defaults = dict(
            event_id="e1",
            category=DriftCategory.STALE_BELIEFS,
            severity="warning",
            description="d",
            evidence={},
            bdi_layer="beliefs",
        )
        defaults.update(kw)
        return DriftEvent(**defaults)

    def test_valid(self):
        e = self._make()
        assert e.severity == "warning"

    @pytest.mark.parametrize("bad", ["low", "high", "error", ""])
    def test_bad_severity(self, bad):
        with pytest.raises(ValueError, match="severity"):
            self._make(severity=bad)

    @pytest.mark.parametrize("bad", ["belief", "goal", "actions", ""])
    def test_bad_bdi_layer(self, bad):
        with pytest.raises(ValueError, match="bdi_layer"):
            self._make(bdi_layer=bad)


class TestDriftConfig:
    def test_defaults(self):
        c = DriftConfig()
        assert c.check_interval_hours == 168.0
        assert c.swei_threshold == 0.15
        assert c.staleness_days == 90
        assert c.min_observations == 10


class TestAnalyzeBehavioralPatterns:
    def setup_method(self):
        self.dd = DriftDetector()

    def test_values_tension_from_contradicts_beliefs(self):
        actions = [{"description": "some action", "contradicts_beliefs": ["b1"]}]
        bdi = {"beliefs": [{"id": "b1", "content": "integrity"}], "desires": [], "intentions": []}
        events = self.dd.analyze_behavioral_patterns(actions, bdi)
        cats = [e.category for e in events]
        assert DriftCategory.VALUES_TENSION in cats

    def test_stale_beliefs_from_old_last_reviewed(self):
        old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        beliefs = [{"id": "b1", "content": "old belief", "last_reviewed": old}]
        events = self.dd.analyze_behavioral_patterns([], {"beliefs": beliefs, "desires": [], "intentions": []})
        cats = [e.category for e in events]
        assert DriftCategory.STALE_BELIEFS in cats

    def test_intention_drift_unsupported_intentions(self):
        intentions = [{"id": "i1", "status": "active"}]
        actions = [{"description": "unrelated", "intention_ids": []}]
        events = self.dd.analyze_behavioral_patterns(
            actions, {"beliefs": [], "desires": [], "intentions": intentions}
        )
        cats = [e.category for e in events]
        assert DriftCategory.INTENTION_DRIFT in cats

    def test_healthy_adaptation_high_aligned_ratio(self):
        dd = DriftDetector(DriftConfig(min_observations=3))
        actions = [{"description": "a", "aligned": True} for _ in range(5)]
        events = dd.analyze_behavioral_patterns(
            actions, {"beliefs": [], "desires": [], "intentions": []}
        )
        cats = [e.category for e in events]
        assert DriftCategory.HEALTHY_ADAPTATION in cats

    def test_healthy_adaptation_not_enough_observations(self):
        dd = DriftDetector(DriftConfig(min_observations=100))
        actions = [{"description": "a", "aligned": True} for _ in range(5)]
        events = dd.analyze_behavioral_patterns(
            actions, {"beliefs": [], "desires": [], "intentions": []}
        )
        cats = [e.category for e in events]
        assert DriftCategory.HEALTHY_ADAPTATION not in cats


class TestClassifyDrift:
    def setup_method(self):
        self.dd = DriftDetector()

    def test_healthy(self):
        assert self.dd.classify_drift(0.05, 0.8) == DriftCategory.HEALTHY_ADAPTATION

    def test_values_tension(self):
        assert self.dd.classify_drift(0.20, 0.4) == DriftCategory.VALUES_TENSION

    def test_intention_drift(self):
        assert self.dd.classify_drift(0.10, 0.55) == DriftCategory.INTENTION_DRIFT

    def test_stale_beliefs(self):
        # swei >= threshold and alignment between 0.5-0.7 doesn't match other rules
        assert self.dd.classify_drift(0.16, 0.65) == DriftCategory.STALE_BELIEFS

    def test_boundary_healthy(self):
        # swei < threshold and alignment >= 0.7
        assert self.dd.classify_drift(0.14, 0.7) == DriftCategory.HEALTHY_ADAPTATION

    def test_boundary_not_healthy(self):
        assert self.dd.classify_drift(0.15, 0.7) != DriftCategory.HEALTHY_ADAPTATION


class TestGenerateReviewInvitation:
    def setup_method(self):
        self.dd = DriftDetector()

    def test_empty_events(self):
        inv = self.dd.generate_review_invitation([])
        assert inv["summary"] == "No drift events detected."
        assert inv["events"] == []

    def test_with_events(self):
        events = [
            DriftEvent("e1", DriftCategory.STALE_BELIEFS, "warning", "stale", {}, "beliefs", requires_review=True),
            DriftEvent("e2", DriftCategory.VALUES_TENSION, "critical", "tension", {}, "desires", requires_review=True),
        ]
        inv = self.dd.generate_review_invitation(events)
        assert "2 drift event(s)" in inv["summary"]
        assert sorted(inv["affected_layers"]) == ["beliefs", "desires"]
        assert len(inv["events"]) == 2
        assert any("critical" in a.lower() or "Urgent" in a for a in inv["recommended_actions"])


class TestGetSeverity:
    def setup_method(self):
        self.dd = DriftDetector()

    def test_healthy_always_info(self):
        assert self.dd.get_severity(DriftCategory.HEALTHY_ADAPTATION, 0.5) == "info"

    def test_values_tension_critical(self):
        assert self.dd.get_severity(DriftCategory.VALUES_TENSION, 0.30) == "critical"

    def test_values_tension_warning(self):
        assert self.dd.get_severity(DriftCategory.VALUES_TENSION, 0.20) == "warning"

    def test_stale_beliefs_warning(self):
        assert self.dd.get_severity(DriftCategory.STALE_BELIEFS, 0.5) == "warning"

    def test_intention_drift_warning(self):
        assert self.dd.get_severity(DriftCategory.INTENTION_DRIFT, 0.10) == "warning"

    def test_intention_drift_critical(self):
        assert self.dd.get_severity(DriftCategory.INTENTION_DRIFT, 0.30) == "critical"
