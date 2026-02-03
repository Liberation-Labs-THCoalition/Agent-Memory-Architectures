"""In-memory BDI store with revision history."""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    BDIBelief,
    BDIDesire,
    BDIIntention,
    BDISnapshot,
    BeliefStatus,
    DesireStatus,
    IntentionStatus,
)


class BDIStore:
    """Thread-unsafe in-memory store for BDI entities with revision tracking."""

    def __init__(self, org_id: str) -> None:
        self.org_id = org_id
        self._beliefs: Dict[str, BDIBelief] = {}
        self._desires: Dict[str, BDIDesire] = {}
        self._intentions: Dict[str, BDIIntention] = {}
        self._revisions: List[dict] = []

    # ------------------------------------------------------------------
    # Beliefs
    # ------------------------------------------------------------------

    def add_belief(self, belief: BDIBelief) -> None:
        self._record_revision("belief", belief.id, None, asdict(belief))
        self._beliefs[belief.id] = belief

    def update_belief(self, belief_id: str, **updates: Any) -> None:
        belief = self._beliefs.get(belief_id)
        if belief is None:
            raise KeyError(f"Belief {belief_id!r} not found")
        before = asdict(belief)
        for key, value in updates.items():
            if not hasattr(belief, key):
                raise AttributeError(f"BDIBelief has no attribute {key!r}")
            setattr(belief, key, value)
        belief.version += 1
        belief.last_reviewed = datetime.now(timezone.utc)
        self._record_revision("belief", belief_id, before, asdict(belief))

    def get_belief(self, belief_id: str) -> Optional[BDIBelief]:
        return self._beliefs.get(belief_id)

    def list_beliefs(self, status: Optional[BeliefStatus] = None) -> List[BDIBelief]:
        beliefs = list(self._beliefs.values())
        if status is not None:
            beliefs = [b for b in beliefs if b.status == status]
        return beliefs

    def archive_belief(self, belief_id: str) -> None:
        self.update_belief(belief_id, status=BeliefStatus.ARCHIVED)

    # ------------------------------------------------------------------
    # Desires
    # ------------------------------------------------------------------

    def add_desire(self, desire: BDIDesire) -> None:
        self._record_revision("desire", desire.id, None, asdict(desire))
        self._desires[desire.id] = desire

    def update_desire(self, desire_id: str, **updates: Any) -> None:
        desire = self._desires.get(desire_id)
        if desire is None:
            raise KeyError(f"Desire {desire_id!r} not found")
        before = asdict(desire)
        for key, value in updates.items():
            if not hasattr(desire, key):
                raise AttributeError(f"BDIDesire has no attribute {key!r}")
            setattr(desire, key, value)
        desire.version += 1
        desire.last_reviewed = datetime.now(timezone.utc)
        self._record_revision("desire", desire_id, before, asdict(desire))

    def get_desire(self, desire_id: str) -> Optional[BDIDesire]:
        return self._desires.get(desire_id)

    def list_desires(self, status: Optional[DesireStatus] = None) -> List[BDIDesire]:
        desires = list(self._desires.values())
        if status is not None:
            desires = [d for d in desires if d.status == status]
        return desires

    def suspend_desire(self, desire_id: str) -> None:
        self.update_desire(desire_id, status=DesireStatus.SUSPENDED)

    # ------------------------------------------------------------------
    # Intentions
    # ------------------------------------------------------------------

    def add_intention(self, intention: BDIIntention) -> None:
        self._record_revision("intention", intention.id, None, asdict(intention))
        self._intentions[intention.id] = intention

    def update_intention(self, intention_id: str, **updates: Any) -> None:
        intention = self._intentions.get(intention_id)
        if intention is None:
            raise KeyError(f"Intention {intention_id!r} not found")
        before = asdict(intention)
        for key, value in updates.items():
            if not hasattr(intention, key):
                raise AttributeError(f"BDIIntention has no attribute {key!r}")
            setattr(intention, key, value)
        intention.version += 1
        intention.last_reviewed = datetime.now(timezone.utc)
        self._record_revision("intention", intention_id, before, asdict(intention))

    def get_intention(self, intention_id: str) -> Optional[BDIIntention]:
        return self._intentions.get(intention_id)

    def list_intentions(
        self, status: Optional[IntentionStatus] = None
    ) -> List[BDIIntention]:
        intentions = list(self._intentions.values())
        if status is not None:
            intentions = [i for i in intentions if i.status == status]
        return intentions

    def complete_intention(self, intention_id: str) -> None:
        self.update_intention(
            intention_id, status=IntentionStatus.COMPLETED, progress=1.0
        )

    # ------------------------------------------------------------------
    # Snapshot & history
    # ------------------------------------------------------------------

    def get_snapshot(self) -> BDISnapshot:
        return BDISnapshot(
            org_id=self.org_id,
            beliefs=list(self._beliefs.values()),
            desires=list(self._desires.values()),
            intentions=list(self._intentions.values()),
            snapshot_at=datetime.now(timezone.utc),
        )

    def get_revision_history(
        self, entity_type: str, entity_id: str
    ) -> List[dict]:
        return [
            r
            for r in self._revisions
            if r["entity_type"] == entity_type and r["entity_id"] == entity_id
        ]

    def _record_revision(
        self,
        entity_type: str,
        entity_id: str,
        before: Optional[dict],
        after: dict,
    ) -> None:
        self._revisions.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "before": before,
                "after": after,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
