"""Evolutionary Pipeline for Kintsugi CMA -- Phase 3 Stream 3B.

Manages the lifecycle of modification proposals: queuing, activation,
evaluation, and generational tracking. Enforces sequential evaluation
(max 1 active at a time) per the Kintsugi specification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class ModificationScope(Enum):
    PROMPT = "PROMPT"
    TOOL_CONFIG = "TOOL_CONFIG"
    SKILL_CHIP = "SKILL_CHIP"
    ARCHITECTURE = "ARCHITECTURE"


@dataclass
class ModificationProposal:
    proposal_id: str
    scope: ModificationScope
    description: str
    modification: dict
    estimated_eval_turns: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "queued"
    parent_trace_id: Optional[str] = None
    # Populated after evaluation
    result_verdict: Optional[str] = None
    result_swei: Optional[float] = None


@dataclass
class EvolutionConfig:
    max_queue_size: int = 20
    max_active_evaluations: int = 1
    min_eval_turns: int = 5
    max_eval_turns: int = 50
    allowed_scopes: List[ModificationScope] = field(
        default_factory=lambda: [ModificationScope.PROMPT, ModificationScope.TOOL_CONFIG]
    )


class EvolutionManager:
    """Manages the evolutionary proposal pipeline with sequential evaluation."""

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self.config = config or EvolutionConfig()
        self._proposals: Dict[str, ModificationProposal] = {}
        self._generation: int = 0
        self._total_evaluated: int = 0
        self._total_approved: int = 0
        self._total_rejected: int = 0

    def _validate_scope(self, scope: ModificationScope) -> None:
        if scope not in self.config.allowed_scopes:
            raise ValueError(
                f"Scope {scope.value} not in allowed scopes: "
                f"{[s.value for s in self.config.allowed_scopes]}"
            )

    def submit_proposal(
        self,
        scope: ModificationScope,
        description: str,
        modification: dict,
        parent_trace_id: Optional[str] = None,
    ) -> ModificationProposal:
        self._validate_scope(scope)
        queued = [p for p in self._proposals.values() if p.status == "queued"]
        if len(queued) >= self.config.max_queue_size:
            raise ValueError(
                f"Queue full ({self.config.max_queue_size}). "
                "Discard or evaluate existing proposals first."
            )
        proposal = ModificationProposal(
            proposal_id=uuid.uuid4().hex[:12],
            scope=scope,
            description=description,
            modification=modification,
            parent_trace_id=parent_trace_id,
        )
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def get_queue(self) -> List[ModificationProposal]:
        queued = [p for p in self._proposals.values() if p.status == "queued"]
        return sorted(queued, key=lambda p: p.created_at)

    def get_active(self) -> Optional[ModificationProposal]:
        for p in self._proposals.values():
            if p.status == "active":
                return p
        return None

    def activate_next(self) -> Optional[ModificationProposal]:
        if self.get_active() is not None:
            return None
        queue = self.get_queue()
        if not queue:
            return None
        proposal = queue[0]
        proposal.status = "active"
        return proposal

    def complete_evaluation(
        self, proposal_id: str, verdict_str: str, swei: float
    ) -> ModificationProposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Proposal {proposal_id} not found")
        if proposal.status != "active":
            raise ValueError(
                f"Proposal {proposal_id} is '{proposal.status}', expected 'active'"
            )
        proposal.status = "evaluated"
        proposal.result_verdict = verdict_str
        proposal.result_swei = swei
        self._total_evaluated += 1
        verdict_upper = verdict_str.upper()
        if verdict_upper == "APPROVE":
            self._total_approved += 1
            self._generation += 1
        elif verdict_upper in ("REJECT", "ESCALATE"):
            self._total_rejected += 1
        return proposal

    def discard_proposal(self, proposal_id: str) -> ModificationProposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Proposal {proposal_id} not found")
        proposal.status = "discarded"
        return proposal

    def get_generation_info(self) -> dict:
        return {
            "generation": self._generation,
            "total_evaluated": self._total_evaluated,
            "total_approved": self._total_approved,
            "total_rejected": self._total_rejected,
            "queue_depth": len(self.get_queue()),
        }
