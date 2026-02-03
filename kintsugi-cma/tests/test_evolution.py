"""Tests for kintsugi.kintsugi_engine.evolution -- Phase 3 Stream 3B."""

import time
from datetime import datetime, timezone

import pytest

from kintsugi.kintsugi_engine.evolution import (
    EvolutionConfig,
    EvolutionManager,
    ModificationProposal,
    ModificationScope,
)


# --- ModificationScope enum ---

class TestModificationScope:
    def test_values(self):
        assert ModificationScope.PROMPT.value == "PROMPT"
        assert ModificationScope.TOOL_CONFIG.value == "TOOL_CONFIG"
        assert ModificationScope.SKILL_CHIP.value == "SKILL_CHIP"
        assert ModificationScope.ARCHITECTURE.value == "ARCHITECTURE"

    def test_member_count(self):
        assert len(ModificationScope) == 4


# --- ModificationProposal defaults ---

class TestModificationProposal:
    def test_defaults(self):
        p = ModificationProposal(
            proposal_id="abc",
            scope=ModificationScope.PROMPT,
            description="d",
            modification={},
        )
        assert p.status == "queued"
        assert p.estimated_eval_turns == 10
        assert p.parent_trace_id is None
        assert p.result_verdict is None
        assert p.result_swei is None
        assert isinstance(p.created_at, datetime)


# --- EvolutionConfig defaults ---

class TestEvolutionConfig:
    def test_defaults(self):
        c = EvolutionConfig()
        assert c.max_queue_size == 20
        assert c.max_active_evaluations == 1
        assert c.min_eval_turns == 5
        assert c.max_eval_turns == 50
        assert ModificationScope.PROMPT in c.allowed_scopes
        assert ModificationScope.TOOL_CONFIG in c.allowed_scopes
        assert len(c.allowed_scopes) == 2


# --- EvolutionManager ---

class TestEvolutionManager:
    def _mgr(self, **kwargs):
        return EvolutionManager(EvolutionConfig(**kwargs))

    def test_submit_proposal_creates_unique_id(self):
        mgr = EvolutionManager()
        p1 = mgr.submit_proposal(ModificationScope.PROMPT, "a", {"k": 1})
        p2 = mgr.submit_proposal(ModificationScope.PROMPT, "b", {"k": 2})
        assert p1.proposal_id != p2.proposal_id
        assert p1.status == "queued"

    def test_submit_disallowed_scope_raises(self):
        mgr = EvolutionManager()
        with pytest.raises(ValueError, match="not in allowed scopes"):
            mgr.submit_proposal(ModificationScope.ARCHITECTURE, "x", {})

    def test_submit_queue_full_raises(self):
        mgr = self._mgr(max_queue_size=2)
        mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.submit_proposal(ModificationScope.PROMPT, "b", {})
        with pytest.raises(ValueError, match="Queue full"):
            mgr.submit_proposal(ModificationScope.PROMPT, "c", {})

    def test_get_queue_sorted_by_created_at(self):
        mgr = EvolutionManager()
        p1 = mgr.submit_proposal(ModificationScope.PROMPT, "first", {})
        p2 = mgr.submit_proposal(ModificationScope.PROMPT, "second", {})
        queue = mgr.get_queue()
        assert queue[0].proposal_id == p1.proposal_id
        assert queue[1].proposal_id == p2.proposal_id

    def test_activate_next_pops_oldest(self):
        mgr = EvolutionManager()
        p1 = mgr.submit_proposal(ModificationScope.PROMPT, "first", {})
        mgr.submit_proposal(ModificationScope.PROMPT, "second", {})
        activated = mgr.activate_next()
        assert activated.proposal_id == p1.proposal_id
        assert activated.status == "active"
        assert len(mgr.get_queue()) == 1

    def test_activate_next_returns_none_when_active_exists(self):
        mgr = EvolutionManager()
        mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.submit_proposal(ModificationScope.PROMPT, "b", {})
        mgr.activate_next()
        assert mgr.activate_next() is None

    def test_activate_next_returns_none_on_empty_queue(self):
        mgr = EvolutionManager()
        assert mgr.activate_next() is None

    def test_get_active(self):
        mgr = EvolutionManager()
        assert mgr.get_active() is None
        mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.activate_next()
        assert mgr.get_active() is not None

    def test_complete_evaluation_stores_verdict_and_swei(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.activate_next()
        result = mgr.complete_evaluation(p.proposal_id, "APPROVE", 0.85)
        assert result.status == "evaluated"
        assert result.result_verdict == "APPROVE"
        assert result.result_swei == 0.85

    def test_complete_evaluation_non_active_raises(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        with pytest.raises(ValueError, match="expected 'active'"):
            mgr.complete_evaluation(p.proposal_id, "APPROVE", 0.5)

    def test_complete_evaluation_approve_increments_generation(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.activate_next()
        mgr.complete_evaluation(p.proposal_id, "APPROVE", 0.9)
        info = mgr.get_generation_info()
        assert info["generation"] == 1
        assert info["total_approved"] == 1
        assert info["total_rejected"] == 0

    def test_complete_evaluation_reject_increments_rejected(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.activate_next()
        mgr.complete_evaluation(p.proposal_id, "REJECT", 0.3)
        info = mgr.get_generation_info()
        assert info["generation"] == 0
        assert info["total_rejected"] == 1

    def test_complete_evaluation_escalate_increments_rejected(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.activate_next()
        mgr.complete_evaluation(p.proposal_id, "ESCALATE", 0.4)
        assert mgr.get_generation_info()["total_rejected"] == 1

    def test_discard_proposal(self):
        mgr = EvolutionManager()
        p = mgr.submit_proposal(ModificationScope.PROMPT, "a", {})
        mgr.discard_proposal(p.proposal_id)
        assert p.status == "discarded"
        assert len(mgr.get_queue()) == 0

    def test_get_generation_info(self):
        mgr = EvolutionManager()
        info = mgr.get_generation_info()
        assert info == {
            "generation": 0,
            "total_evaluated": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "queue_depth": 0,
        }

    def test_full_lifecycle(self):
        mgr = EvolutionManager()
        # Submit and evaluate first
        p1 = mgr.submit_proposal(ModificationScope.PROMPT, "first", {"v": 1})
        mgr.activate_next()
        mgr.complete_evaluation(p1.proposal_id, "APPROVE", 0.9)
        # Submit and activate second
        p2 = mgr.submit_proposal(ModificationScope.TOOL_CONFIG, "second", {"v": 2})
        activated = mgr.activate_next()
        assert activated.proposal_id == p2.proposal_id
        mgr.complete_evaluation(p2.proposal_id, "REJECT", 0.2)
        info = mgr.get_generation_info()
        assert info["generation"] == 1
        assert info["total_evaluated"] == 2
        assert info["total_approved"] == 1
        assert info["total_rejected"] == 1

    def test_complete_evaluation_unknown_id_raises(self):
        mgr = EvolutionManager()
        with pytest.raises(KeyError):
            mgr.complete_evaluation("nonexistent", "APPROVE", 0.5)

    def test_discard_unknown_id_raises(self):
        mgr = EvolutionManager()
        with pytest.raises(KeyError):
            mgr.discard_proposal("nonexistent")
