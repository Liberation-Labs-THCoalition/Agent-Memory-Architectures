"""Tests for bdi.coherence + bdi.drift_classifier – Phase 3 BDI."""

import pytest
from datetime import datetime, timezone
from dataclasses import FrozenInstanceError

from kintsugi.bdi.models import (
    BDIBelief,
    BDIDesire,
    BDIIntention,
    BDISnapshot,
    BeliefStatus,
    DesireStatus,
    IntentionStatus,
)
from kintsugi.bdi.coherence import CoherenceChecker, CoherenceScore
from kintsugi.bdi.drift_classifier import BDIDriftClassifier, DriftClassification, _VALID_CATEGORIES

NOW = datetime.now(timezone.utc)


def _belief(id="b1", content="community health improvement", tags=None, status=BeliefStatus.ACTIVE):
    return BDIBelief(
        id=id, content=content, confidence=0.9, status=status,
        source="user", tags=tags or ["health", "community"], created_at=NOW,
    )


def _desire(id="d1", content="improve community health outcomes", tags=None, status=DesireStatus.ACTIVE):
    return BDIDesire(
        id=id, content=content, priority=0.8, status=status,
        related_tags=tags or ["health", "community"], measurable=True, metric="m", created_at=NOW,
    )


def _intention(id="i1", belief_ids=None, desire_ids=None, status=IntentionStatus.ACTIVE):
    return BDIIntention(
        id=id, goal="run health program", status=status,
        belief_ids=belief_ids if belief_ids is not None else ["b1"],
        desire_ids=desire_ids if desire_ids is not None else ["d1"],
        created_at=NOW,
    )


def _snapshot(beliefs=None, desires=None, intentions=None):
    return BDISnapshot(
        org_id="org1",
        beliefs=beliefs or [],
        desires=desires or [],
        intentions=intentions or [],
        snapshot_at=NOW,
    )


# ------------------------------------------------------------------ coherence
class TestCoherenceScore:
    def test_frozen(self):
        cs = CoherenceScore(0.8, 0.7, 0.6, 0.7, ())
        with pytest.raises(FrozenInstanceError):
            cs.overall = 0.9

    def test_issues_tuple(self):
        cs = CoherenceScore(0.5, 0.5, 0.5, 0.5, ("issue1",))
        assert cs.issues == ("issue1",)


class TestCoherenceChecker:
    def setup_method(self):
        self.checker = CoherenceChecker()

    def test_fully_linked_high_scores(self):
        b = _belief()
        d = _desire()
        i = _intention()
        snap = _snapshot([b], [d], [i])
        score = self.checker.check_coherence(snap)
        assert score.overall > 0.5
        assert score.belief_desire_alignment > 0.3
        assert score.desire_intention_alignment > 0.3
        assert score.belief_intention_alignment > 0.3

    def test_unlinked_intentions_low_scores(self):
        b = _belief()
        d = _desire()
        i = _intention(belief_ids=[], desire_ids=[])
        snap = _snapshot([b], [d], [i])
        score = self.checker.check_coherence(snap)
        assert score.desire_intention_alignment == 0.0
        assert score.belief_intention_alignment == 0.0
        assert len(score.issues) > 0

    def test_empty_snapshot_neutral(self):
        snap = _snapshot()
        score = self.checker.check_coherence(snap)
        assert score.belief_desire_alignment == 0.5
        assert score.desire_intention_alignment == 0.5
        assert score.belief_intention_alignment == 0.5

    def test_tag_overlap_scoring(self):
        b = _belief(tags=["education", "youth"])
        d = _desire(tags=["education", "youth"], content="education youth programs")
        i = _intention(belief_ids=["b1"], desire_ids=["d1"])
        snap = _snapshot([b], [d], [i])
        score = self.checker.check_coherence(snap)
        assert score.belief_desire_alignment > 0.5

    def test_no_tag_overlap_low_bd(self):
        b = _belief(tags=["alpha"])
        d = _desire(tags=["beta"], content="completely unrelated topic xyzzy")
        i = _intention(belief_ids=["b1"], desire_ids=["d1"])
        snap = _snapshot([b], [d], [i])
        score = self.checker.check_coherence(snap)
        assert score.belief_desire_alignment < 0.5


# ------------------------------------------------------------------ drift classification
class TestDriftClassification:
    def test_frozen(self):
        dc = DriftClassification("healthy_adaptation", 0.8, ("e",), "ok")
        with pytest.raises(FrozenInstanceError):
            dc.category = "stale_beliefs"

    def test_valid_categories(self):
        assert "healthy_adaptation" in _VALID_CATEGORIES
        assert "stale_beliefs" in _VALID_CATEGORIES
        assert "intention_drift" in _VALID_CATEGORIES
        assert "values_tension" in _VALID_CATEGORIES


class TestBDIDriftClassifier:
    def setup_method(self):
        self.clf = BDIDriftClassifier()

    def _score(self, bd=0.8, di=0.8, bi=0.8, overall=0.8, issues=()):
        return CoherenceScore(bd, di, bi, overall, issues)

    def test_healthy(self):
        before = self._score(overall=0.7)
        after = self._score(overall=0.8)
        dc = self.clf.classify(before, after, 30)
        assert dc.category == "healthy_adaptation"

    def test_stale_beliefs(self):
        before = self._score(bd=0.8, bi=0.8, overall=0.8)
        after = self._score(bd=0.7, bi=0.7, overall=0.7, issues=("new issue",))
        dc = self.clf.classify(before, after, 90)
        assert dc.category == "stale_beliefs"

    def test_intention_drift(self):
        before = self._score(di=0.8, bi=0.8, overall=0.8)
        after = self._score(di=0.7, bi=0.7, overall=0.7, issues=("new issue",))
        dc = self.clf.classify(before, after, 10)  # short time -> not stale
        assert dc.category == "intention_drift"

    def test_values_tension(self):
        before = self._score(bd=0.8, di=0.8, bi=0.8, overall=0.8)
        after = self._score(bd=0.7, di=0.8, bi=0.8, overall=0.77, issues=("issue",))
        dc = self.clf.classify(before, after, 10)
        assert dc.category == "values_tension"

    def test_classify_from_events_with_events(self):
        events = [
            {"category": "stale_beliefs", "description": "belief is old"},
            {"category": "stale_beliefs", "description": "another old belief"},
            {"category": "intention_drift", "description": "drift"},
        ]
        dc = self.clf.classify_from_events(events)
        assert dc.category == "stale_beliefs"
        assert dc.confidence > 0.5

    def test_classify_from_events_empty(self):
        dc = self.clf.classify_from_events([])
        assert dc.category == "healthy_adaptation"
        assert dc.confidence == 0.5
