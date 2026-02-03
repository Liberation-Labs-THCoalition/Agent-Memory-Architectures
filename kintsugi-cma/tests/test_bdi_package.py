"""Tests for bdi.models + bdi.store – Phase 3 BDI."""

import pytest
from datetime import datetime, timezone

from kintsugi.bdi.models import (
    BDIBelief,
    BDIDesire,
    BDIIntention,
    BDISnapshot,
    BeliefStatus,
    DesireStatus,
    IntentionStatus,
)
from kintsugi.bdi.store import BDIStore


# ------------------------------------------------------------------ enums
class TestEnums:
    def test_belief_status(self):
        assert set(s.value for s in BeliefStatus) == {"active", "archived", "challenged", "stale"}

    def test_desire_status(self):
        assert set(s.value for s in DesireStatus) == {"active", "achieved", "suspended", "abandoned"}

    def test_intention_status(self):
        assert set(s.value for s in IntentionStatus) == {"active", "completed", "suspended", "failed"}


# ------------------------------------------------------------------ model validation
NOW = datetime.now(timezone.utc)


def _belief(**kw):
    defaults = dict(
        id="b1", content="test", confidence=0.8, status=BeliefStatus.ACTIVE,
        source="user", tags=["t"], created_at=NOW,
    )
    defaults.update(kw)
    return BDIBelief(**defaults)


def _desire(**kw):
    defaults = dict(
        id="d1", content="goal", priority=0.5, status=DesireStatus.ACTIVE,
        related_tags=["t"], measurable=True, metric="m", created_at=NOW,
    )
    defaults.update(kw)
    return BDIDesire(**defaults)


def _intention(**kw):
    defaults = dict(
        id="i1", goal="do it", status=IntentionStatus.ACTIVE,
        belief_ids=["b1"], desire_ids=["d1"], created_at=NOW,
    )
    defaults.update(kw)
    return BDIIntention(**defaults)


class TestBDIBelief:
    def test_valid(self):
        b = _belief()
        assert b.version == 1

    def test_confidence_too_high(self):
        with pytest.raises(ValueError):
            _belief(confidence=1.1)

    def test_confidence_negative(self):
        with pytest.raises(ValueError):
            _belief(confidence=-0.1)

    def test_version_zero(self):
        with pytest.raises(ValueError):
            _belief(version=0)


class TestBDIDesire:
    def test_valid(self):
        assert _desire().priority == 0.5

    def test_priority_too_high(self):
        with pytest.raises(ValueError):
            _desire(priority=1.5)

    def test_priority_negative(self):
        with pytest.raises(ValueError):
            _desire(priority=-0.1)


class TestBDIIntention:
    def test_valid(self):
        assert _intention().progress == 0.0

    def test_progress_too_high(self):
        with pytest.raises(ValueError):
            _intention(progress=1.1)

    def test_progress_negative(self):
        with pytest.raises(ValueError):
            _intention(progress=-0.1)


# ------------------------------------------------------------------ store
class TestBDIStore:
    def setup_method(self):
        self.store = BDIStore("org1")

    # beliefs
    def test_add_get_belief(self):
        b = _belief()
        self.store.add_belief(b)
        assert self.store.get_belief("b1") is b

    def test_list_beliefs(self):
        self.store.add_belief(_belief(id="b1"))
        self.store.add_belief(_belief(id="b2", status=BeliefStatus.ARCHIVED))
        assert len(self.store.list_beliefs()) == 2
        assert len(self.store.list_beliefs(status=BeliefStatus.ACTIVE)) == 1

    def test_archive_belief(self):
        self.store.add_belief(_belief())
        self.store.archive_belief("b1")
        assert self.store.get_belief("b1").status == BeliefStatus.ARCHIVED

    def test_update_belief_bumps_version(self):
        self.store.add_belief(_belief())
        self.store.update_belief("b1", confidence=0.9)
        b = self.store.get_belief("b1")
        assert b.version == 2
        assert b.confidence == 0.9
        assert b.last_reviewed is not None

    def test_update_belief_bad_attr(self):
        self.store.add_belief(_belief())
        with pytest.raises(AttributeError):
            self.store.update_belief("b1", nonexistent=True)

    # desires
    def test_add_get_desire(self):
        d = _desire()
        self.store.add_desire(d)
        assert self.store.get_desire("d1") is d

    def test_list_desires(self):
        self.store.add_desire(_desire(id="d1"))
        self.store.add_desire(_desire(id="d2", status=DesireStatus.SUSPENDED))
        assert len(self.store.list_desires()) == 2
        assert len(self.store.list_desires(status=DesireStatus.ACTIVE)) == 1

    def test_suspend_desire(self):
        self.store.add_desire(_desire())
        self.store.suspend_desire("d1")
        assert self.store.get_desire("d1").status == DesireStatus.SUSPENDED

    def test_update_desire_bumps_version(self):
        self.store.add_desire(_desire())
        self.store.update_desire("d1", priority=0.9)
        assert self.store.get_desire("d1").version == 2

    # intentions
    def test_add_get_intention(self):
        i = _intention()
        self.store.add_intention(i)
        assert self.store.get_intention("i1") is i

    def test_list_intentions(self):
        self.store.add_intention(_intention(id="i1"))
        self.store.add_intention(_intention(id="i2", status=IntentionStatus.COMPLETED))
        assert len(self.store.list_intentions()) == 2
        assert len(self.store.list_intentions(status=IntentionStatus.ACTIVE)) == 1

    def test_complete_intention(self):
        self.store.add_intention(_intention())
        self.store.complete_intention("i1")
        i = self.store.get_intention("i1")
        assert i.status == IntentionStatus.COMPLETED
        assert i.progress == 1.0

    def test_update_intention_bumps_version(self):
        self.store.add_intention(_intention())
        self.store.update_intention("i1", progress=0.5)
        assert self.store.get_intention("i1").version == 2

    # snapshot & history
    def test_get_snapshot(self):
        self.store.add_belief(_belief())
        self.store.add_desire(_desire())
        self.store.add_intention(_intention())
        snap = self.store.get_snapshot()
        assert isinstance(snap, BDISnapshot)
        assert len(snap.beliefs) == 1
        assert len(snap.desires) == 1
        assert len(snap.intentions) == 1
        assert snap.org_id == "org1"

    def test_get_revision_history(self):
        self.store.add_belief(_belief())
        self.store.update_belief("b1", confidence=0.5)
        history = self.store.get_revision_history("belief", "b1")
        assert len(history) == 2  # add + update

    def test_get_missing_returns_none(self):
        assert self.store.get_belief("missing") is None
        assert self.store.get_desire("missing") is None
        assert self.store.get_intention("missing") is None

    def test_update_missing_raises(self):
        with pytest.raises(KeyError):
            self.store.update_belief("missing", confidence=0.5)
