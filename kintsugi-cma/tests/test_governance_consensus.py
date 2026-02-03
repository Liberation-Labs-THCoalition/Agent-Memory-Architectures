"""Tests for kintsugi.governance.consensus module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from kintsugi.governance.consensus import (
    ConsentCategory,
    ConsentItem,
    ConsentStatus,
    ConsensusConfig,
    ConsensusGate,
    ConsensusPriority,
    _TERMINAL_STATUSES,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_consensus_priority_values(self):
        assert ConsensusPriority.LOW == "LOW"
        assert ConsensusPriority.MEDIUM == "MEDIUM"
        assert ConsensusPriority.HIGH == "HIGH"
        assert ConsensusPriority.CRITICAL == "CRITICAL"

    def test_consent_category_values(self):
        assert ConsentCategory.FINANCIAL == "FINANCIAL"
        assert ConsentCategory.PII == "PII"
        assert ConsentCategory.EXTERNAL_COMMS == "EXTERNAL_COMMS"
        assert ConsentCategory.SELF_MODIFICATION == "SELF_MODIFICATION"
        assert ConsentCategory.GENERAL == "GENERAL"

    def test_consent_status_values(self):
        assert ConsentStatus.PENDING == "PENDING"
        assert ConsentStatus.APPROVED == "APPROVED"
        assert ConsentStatus.REJECTED == "REJECTED"
        assert ConsentStatus.EXPIRED == "EXPIRED"
        assert ConsentStatus.ESCALATED == "ESCALATED"

    def test_terminal_statuses(self):
        assert ConsentStatus.APPROVED in _TERMINAL_STATUSES
        assert ConsentStatus.REJECTED in _TERMINAL_STATUSES
        assert ConsentStatus.EXPIRED in _TERMINAL_STATUSES
        assert ConsentStatus.PENDING not in _TERMINAL_STATUSES
        assert ConsentStatus.ESCALATED not in _TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# ConsensusConfig
# ---------------------------------------------------------------------------

class TestConsensusConfig:
    def test_defaults(self):
        cfg = ConsensusConfig()
        assert cfg.default_timeout_hours == 24.0
        assert cfg.escalation_after_hours == 48.0
        assert cfg.auto_approve_categories == []
        assert cfg.approval_thresholds[ConsentCategory.FINANCIAL] == 2
        assert cfg.approval_thresholds[ConsentCategory.GENERAL] == 1

    def test_custom_thresholds(self):
        cfg = ConsensusConfig(
            approval_thresholds={ConsentCategory.GENERAL: 3},
            default_timeout_hours=12.0,
        )
        assert cfg.approval_thresholds[ConsentCategory.GENERAL] == 3
        assert cfg.default_timeout_hours == 12.0


# ---------------------------------------------------------------------------
# ConsensusGate
# ---------------------------------------------------------------------------

class TestConsensusGateSubmit:
    def test_submit_creates_pending_item(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "test", {"key": "val"})
        assert item.status == ConsentStatus.PENDING
        assert item.org_id == "org1"
        assert item.category == ConsentCategory.GENERAL
        assert item.description == "test"
        assert item.action_payload == {"key": "val"}
        assert item.priority == ConsensusPriority.MEDIUM

    def test_submit_auto_approve(self):
        cfg = ConsensusConfig(auto_approve_categories=[ConsentCategory.GENERAL])
        gate = ConsensusGate(cfg)
        item = gate.submit("org1", ConsentCategory.GENERAL, "auto", {})
        assert item.status == ConsentStatus.APPROVED
        assert item.resolved_by == "auto"
        assert item.rationale == "Auto-approved by category policy."
        assert item.resolved_at is not None

    def test_submit_non_auto_category_stays_pending(self):
        cfg = ConsensusConfig(auto_approve_categories=[ConsentCategory.GENERAL])
        gate = ConsensusGate(cfg)
        item = gate.submit("org1", ConsentCategory.FINANCIAL, "fin", {})
        assert item.status == ConsentStatus.PENDING


class TestConsensusGateApprove:
    def test_single_approval_meets_threshold(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        result = gate.approve(item.id, "alice", "looks good")
        assert result.status == ConsentStatus.APPROVED
        assert result.resolved_by == "alice"
        assert result.rationale == "looks good"

    def test_multi_approval_threshold(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.FINANCIAL, "t", {})
        result = gate.approve(item.id, "alice")
        assert result.status == ConsentStatus.PENDING
        result = gate.approve(item.id, "bob", "ok")
        assert result.status == ConsentStatus.APPROVED
        assert result.resolved_by == "bob"

    def test_approve_no_rationale_sets_none(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        result = gate.approve(item.id, "alice")
        assert result.rationale is None

    def test_approve_already_rejected_raises(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.reject(item.id, "alice")
        with pytest.raises(ValueError, match="terminal state"):
            gate.approve(item.id, "bob")

    def test_approve_already_approved_raises(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.approve(item.id, "alice")
        with pytest.raises(ValueError, match="terminal state"):
            gate.approve(item.id, "bob")

    def test_approve_unknown_id_raises(self):
        gate = ConsensusGate()
        with pytest.raises(ValueError, match="No consent item"):
            gate.approve("nonexistent", "alice")


class TestConsensusGateReject:
    def test_reject(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        result = gate.reject(item.id, "alice", "bad idea")
        assert result.status == ConsentStatus.REJECTED
        assert result.resolved_by == "alice"
        assert result.rationale == "bad idea"

    def test_reject_no_rationale(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        result = gate.reject(item.id, "alice")
        assert result.rationale is None

    def test_reject_already_approved_raises(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.approve(item.id, "alice")
        with pytest.raises(ValueError, match="terminal state"):
            gate.reject(item.id, "bob")

    def test_reject_unknown_id_raises(self):
        gate = ConsensusGate()
        with pytest.raises(ValueError, match="No consent item"):
            gate.reject("nonexistent", "alice")


class TestConsensusGateListPending:
    def test_list_pending_all(self):
        gate = ConsensusGate()
        gate.submit("org1", ConsentCategory.GENERAL, "a", {})
        gate.submit("org2", ConsentCategory.GENERAL, "b", {})
        item_c = gate.submit("org1", ConsentCategory.GENERAL, "c", {})
        gate.approve(item_c.id, "alice")
        pending = gate.list_pending()
        assert len(pending) == 2

    def test_list_pending_with_org_filter(self):
        gate = ConsensusGate()
        gate.submit("org1", ConsentCategory.GENERAL, "a", {})
        gate.submit("org2", ConsentCategory.GENERAL, "b", {})
        assert len(gate.list_pending(org_id="org1")) == 1

    def test_list_pending_includes_escalated(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "a", {})
        gate.escalate(item.id)
        assert len(gate.list_pending()) == 1


class TestConsensusGateGetItem:
    def test_get_item_found(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        assert gate.get_item(item.id) is item

    def test_get_item_not_found(self):
        gate = ConsensusGate()
        assert gate.get_item("nope") is None


class TestConsensusGateCheckExpired:
    def test_check_expired_marks_items(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        # Move created_at back 25 hours
        item.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        expired = gate.check_expired()
        assert len(expired) == 1
        assert expired[0].status == ConsentStatus.EXPIRED
        assert expired[0].resolved_at is not None

    def test_check_expired_skips_non_pending(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.approve(item.id, "alice")
        item.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        assert gate.check_expired() == []

    def test_check_expired_skips_fresh_items(self):
        gate = ConsensusGate()
        gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        assert gate.check_expired() == []

    def test_check_expired_includes_escalated(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.escalate(item.id)
        item.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        expired = gate.check_expired()
        assert len(expired) == 1


class TestConsensusGateEscalate:
    def test_escalate_pending(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        result = gate.escalate(item.id)
        assert result.status == ConsentStatus.ESCALATED

    def test_escalate_non_pending_raises(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.approve(item.id, "alice")
        with pytest.raises(ValueError, match="only PENDING"):
            gate.escalate(item.id)

    def test_escalate_already_escalated_raises(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.escalate(item.id)
        with pytest.raises(ValueError, match="only PENDING"):
            gate.escalate(item.id)

    def test_escalate_unknown_id_raises(self):
        gate = ConsensusGate()
        with pytest.raises(ValueError, match="No consent item"):
            gate.escalate("nonexistent")

    def test_approve_escalated_item(self):
        gate = ConsensusGate()
        item = gate.submit("org1", ConsentCategory.GENERAL, "t", {})
        gate.escalate(item.id)
        # Escalated is not terminal, so approve should work
        result = gate.approve(item.id, "alice")
        assert result.status == ConsentStatus.APPROVED
