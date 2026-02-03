"""Consensus Gate -- approval queue for sensitive agent actions.

Every action that touches a governed category (financial, PII, external comms,
self-modification) must pass through the ConsensusGate before execution.  Items
sit in a pending queue until the required number of approvals is reached, the
item is explicitly rejected, or the timeout expires.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsensusPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConsentCategory(str, Enum):
    FINANCIAL = "FINANCIAL"
    PII = "PII"
    EXTERNAL_COMMS = "EXTERNAL_COMMS"
    SELF_MODIFICATION = "SELF_MODIFICATION"
    GENERAL = "GENERAL"


class ConsentStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ESCALATED = "ESCALATED"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ConsentItem:
    """A single action awaiting approval."""

    id: str
    org_id: str
    category: ConsentCategory
    priority: ConsensusPriority
    description: str
    action_payload: dict
    status: ConsentStatus = ConsentStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    rationale: Optional[str] = None
    timeout_hours: float = 24.0
    _approvals: list = field(default_factory=list, repr=False)


@dataclass
class ConsensusConfig:
    """Tuneable knobs for the consensus gate."""

    approval_thresholds: Dict[ConsentCategory, int] = field(default_factory=lambda: {
        ConsentCategory.FINANCIAL: 2,
        ConsentCategory.PII: 2,
        ConsentCategory.EXTERNAL_COMMS: 1,
        ConsentCategory.SELF_MODIFICATION: 2,
        ConsentCategory.GENERAL: 1,
    })
    default_timeout_hours: float = 24.0
    escalation_after_hours: float = 48.0
    auto_approve_categories: List[ConsentCategory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({
    ConsentStatus.APPROVED,
    ConsentStatus.REJECTED,
    ConsentStatus.EXPIRED,
})


class ConsensusGate:
    """In-memory approval queue for governed agent actions."""

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()
        self._items: Dict[str, ConsentItem] = {}

    # -- public API ---------------------------------------------------------

    def submit(
        self,
        org_id: str,
        category: ConsentCategory,
        description: str,
        action_payload: dict,
        priority: ConsensusPriority = ConsensusPriority.MEDIUM,
    ) -> ConsentItem:
        """Create a new pending consent item and return it."""
        item = ConsentItem(
            id=str(uuid.uuid4()),
            org_id=org_id,
            category=category,
            priority=priority,
            description=description,
            action_payload=action_payload,
            timeout_hours=self._config.default_timeout_hours,
        )
        if category in self._config.auto_approve_categories:
            item.status = ConsentStatus.APPROVED
            item.resolved_at = datetime.now(timezone.utc)
            item.resolved_by = "auto"
            item.rationale = "Auto-approved by category policy."
        self._items[item.id] = item
        return item

    def approve(self, item_id: str, approver: str, rationale: str = "") -> ConsentItem:
        """Record an approval.  Transitions to APPROVED once threshold is met."""
        item = self._get_or_raise(item_id)
        self._assert_actionable(item, "approve")

        item._approvals.append(approver)
        threshold = self._config.approval_thresholds.get(item.category, 1)
        if len(item._approvals) >= threshold:
            item.status = ConsentStatus.APPROVED
            item.resolved_at = datetime.now(timezone.utc)
            item.resolved_by = approver
            item.rationale = rationale or None
        return item

    def reject(self, item_id: str, rejector: str, rationale: str = "") -> ConsentItem:
        """Reject the item immediately."""
        item = self._get_or_raise(item_id)
        self._assert_actionable(item, "reject")

        item.status = ConsentStatus.REJECTED
        item.resolved_at = datetime.now(timezone.utc)
        item.resolved_by = rejector
        item.rationale = rationale or None
        return item

    def list_pending(self, org_id: str | None = None) -> list[ConsentItem]:
        """Return all items still in PENDING (or ESCALATED) state."""
        results: list[ConsentItem] = []
        for item in self._items.values():
            if item.status not in (ConsentStatus.PENDING, ConsentStatus.ESCALATED):
                continue
            if org_id is not None and item.org_id != org_id:
                continue
            results.append(item)
        return results

    def get_item(self, item_id: str) -> ConsentItem | None:
        return self._items.get(item_id)

    def check_expired(self) -> list[ConsentItem]:
        """Find items that have exceeded their timeout and mark them EXPIRED."""
        now = datetime.now(timezone.utc)
        expired: list[ConsentItem] = []
        for item in self._items.values():
            if item.status not in (ConsentStatus.PENDING, ConsentStatus.ESCALATED):
                continue
            elapsed_hours = (now - item.created_at).total_seconds() / 3600.0
            if elapsed_hours >= item.timeout_hours:
                item.status = ConsentStatus.EXPIRED
                item.resolved_at = now
                expired.append(item)
        return expired

    def escalate(self, item_id: str) -> ConsentItem:
        """Escalate a pending item for higher-priority review."""
        item = self._get_or_raise(item_id)
        if item.status != ConsentStatus.PENDING:
            raise ValueError(
                f"Cannot escalate item {item_id} with status {item.status.value}; "
                "only PENDING items can be escalated."
            )
        item.status = ConsentStatus.ESCALATED
        return item

    # -- internals ----------------------------------------------------------

    def _get_or_raise(self, item_id: str) -> ConsentItem:
        item = self._items.get(item_id)
        if item is None:
            raise ValueError(f"No consent item with id {item_id!r}")
        return item

    @staticmethod
    def _assert_actionable(item: ConsentItem, verb: str) -> None:
        if item.status in _TERMINAL_STATUSES:
            raise ValueError(
                f"Cannot {verb} item {item.id} -- already in terminal state "
                f"{item.status.value}."
            )
