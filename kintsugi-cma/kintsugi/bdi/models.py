"""BDI runtime models using pure dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class BeliefStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    CHALLENGED = "challenged"
    STALE = "stale"


class DesireStatus(Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    SUSPENDED = "suspended"
    ABANDONED = "abandoned"


class IntentionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    FAILED = "failed"


@dataclass
class BDIBelief:
    id: str
    content: str
    confidence: float
    status: BeliefStatus
    source: str
    tags: List[str]
    created_at: datetime
    last_reviewed: Optional[datetime] = None
    version: int = 1
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")


@dataclass
class BDIDesire:
    id: str
    content: str
    priority: float
    status: DesireStatus
    related_tags: List[str]
    measurable: bool
    metric: Optional[str]
    created_at: datetime
    last_reviewed: Optional[datetime] = None
    version: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.priority <= 1.0):
            raise ValueError(f"priority must be 0-1, got {self.priority}")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")


@dataclass
class BDIIntention:
    id: str
    goal: str
    status: IntentionStatus
    belief_ids: List[str]
    desire_ids: List[str]
    created_at: datetime
    last_reviewed: Optional[datetime] = None
    version: int = 1
    progress: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"progress must be 0-1, got {self.progress}")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")


@dataclass
class BDISnapshot:
    org_id: str
    beliefs: List[BDIBelief]
    desires: List[BDIDesire]
    intentions: List[BDIIntention]
    snapshot_at: datetime
    version: int = 1
