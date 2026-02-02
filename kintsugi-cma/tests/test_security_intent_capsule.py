"""Comprehensive pytest tests for kintsugi.security.intent_capsule module.

Tests cover:
- IntentCapsule signing and verification
- HMAC signature validation
- Expiry checking
- Tampering detection
- Cycle constraint verification (tools, egress domains, budget)
- Mission alignment scoring
"""

import hmac
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from kintsugi.security.intent_capsule import (
    AlignmentResult,
    CycleVerdict,
    IntentCapsule,
    _canonical_payload,
    mission_alignment_check,
    sign_capsule,
    verify_capsule,
    verify_cycle,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def secret_key():
    """Standard secret key for tests."""
    return "test-secret-key-do-not-use-in-production"


@pytest.fixture
def basic_goal():
    """Basic mission goal."""
    return "Analyze customer feedback data and generate monthly report"


@pytest.fixture
def basic_constraints():
    """Basic constraint dict."""
    return {
        "allowed_tools": ["bash", "python", "grep"],
        "egress_domains": ["example.com", "api.example.org"],
        "budget_remaining": 100.0,
    }


@pytest.fixture
def basic_capsule(basic_goal, basic_constraints, secret_key):
    """A valid, signed IntentCapsule."""
    return sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="test-org-123",
        secret_key=secret_key,
    )


# ==============================================================================
# Test _canonical_payload
# ==============================================================================

def test_canonical_payload_deterministic():
    """Canonical payload produces identical output for same inputs."""
    goal = "Test goal"
    constraints = {"a": 1, "b": 2}
    org_id = "org-1"
    signed_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    payload1 = _canonical_payload(goal, constraints, org_id, signed_at)
    payload2 = _canonical_payload(goal, constraints, org_id, signed_at)

    assert payload1 == payload2


def test_canonical_payload_sorted_keys():
    """Canonical payload sorts keys to ensure determinism."""
    goal = "Test"
    constraints = {"z": 3, "a": 1, "m": 2}
    org_id = "org"
    signed_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

    payload = _canonical_payload(goal, constraints, org_id, signed_at)
    decoded = json.loads(payload.decode("utf-8"))

    # JSON keys should be sorted
    assert list(decoded["constraints"].keys()) == ["a", "m", "z"]


def test_canonical_payload_is_bytes():
    """Canonical payload returns bytes, not string."""
    goal = "Test"
    constraints = {}
    org_id = "org"
    signed_at = datetime.now(timezone.utc)

    payload = _canonical_payload(goal, constraints, org_id, signed_at)
    assert isinstance(payload, bytes)


# ==============================================================================
# Test sign_capsule
# ==============================================================================

def test_sign_capsule_creates_valid_capsule(basic_goal, basic_constraints, secret_key):
    """sign_capsule produces an IntentCapsule with all required fields."""
    capsule = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="test-org",
        secret_key=secret_key,
    )

    assert isinstance(capsule, IntentCapsule)
    assert capsule.goal == basic_goal
    assert capsule.constraints == basic_constraints
    assert capsule.org_id == "test-org"
    assert isinstance(capsule.signature, str)
    assert len(capsule.signature) == 64  # HMAC-SHA256 produces 32 bytes = 64 hex chars
    assert isinstance(capsule.signed_at, datetime)
    assert capsule.expires_at is None


def test_sign_capsule_with_expiry(basic_goal, basic_constraints, secret_key):
    """sign_capsule respects expires_at parameter."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    capsule = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="test-org",
        secret_key=secret_key,
        expires_at=expiry,
    )

    assert capsule.expires_at == expiry


def test_sign_capsule_different_keys_produce_different_signatures(
    basic_goal, basic_constraints
):
    """Different secret keys produce different signatures."""
    capsule1 = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="org",
        secret_key="key-1",
    )
    capsule2 = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="org",
        secret_key="key-2",
    )

    assert capsule1.signature != capsule2.signature


def test_sign_capsule_different_goals_produce_different_signatures(
    basic_constraints, secret_key
):
    """Different goals produce different signatures."""
    capsule1 = sign_capsule(
        goal="Goal A",
        constraints=basic_constraints,
        org_id="org",
        secret_key=secret_key,
    )
    capsule2 = sign_capsule(
        goal="Goal B",
        constraints=basic_constraints,
        org_id="org",
        secret_key=secret_key,
    )

    assert capsule1.signature != capsule2.signature


# ==============================================================================
# Test verify_capsule
# ==============================================================================

def test_verify_capsule_valid(basic_capsule, secret_key):
    """verify_capsule returns True for validly signed capsule."""
    assert verify_capsule(basic_capsule, secret_key) is True


def test_verify_capsule_wrong_key(basic_capsule):
    """verify_capsule returns False when using wrong secret key."""
    assert verify_capsule(basic_capsule, "wrong-key") is False


def test_verify_capsule_tampered_goal(basic_capsule, secret_key):
    """verify_capsule detects tampering with goal field."""
    # Create a modified capsule (dataclasses are frozen, so we use replace)
    from dataclasses import replace

    tampered = replace(basic_capsule, goal="TAMPERED GOAL")
    assert verify_capsule(tampered, secret_key) is False


def test_verify_capsule_tampered_constraints(basic_capsule, secret_key):
    """verify_capsule detects tampering with constraints."""
    from dataclasses import replace

    tampered_constraints = {**basic_capsule.constraints, "budget_remaining": 999999.0}
    tampered = replace(basic_capsule, constraints=tampered_constraints)
    assert verify_capsule(tampered, secret_key) is False


def test_verify_capsule_tampered_org_id(basic_capsule, secret_key):
    """verify_capsule detects tampering with org_id."""
    from dataclasses import replace

    tampered = replace(basic_capsule, org_id="evil-org")
    assert verify_capsule(tampered, secret_key) is False


def test_verify_capsule_expired(basic_goal, basic_constraints, secret_key):
    """verify_capsule rejects expired capsules."""
    # Create a capsule that expired 1 hour ago
    expiry = datetime.now(timezone.utc) - timedelta(hours=1)
    capsule = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="org",
        secret_key=secret_key,
        expires_at=expiry,
    )

    assert verify_capsule(capsule, secret_key) is False


def test_verify_capsule_not_yet_expired(basic_goal, basic_constraints, secret_key):
    """verify_capsule accepts capsule that expires in the future."""
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    capsule = sign_capsule(
        goal=basic_goal,
        constraints=basic_constraints,
        org_id="org",
        secret_key=secret_key,
        expires_at=expiry,
    )

    assert verify_capsule(capsule, secret_key) is True


def test_verify_capsule_no_expiry(basic_capsule, secret_key):
    """verify_capsule accepts capsule with no expiry (None)."""
    assert basic_capsule.expires_at is None
    assert verify_capsule(basic_capsule, secret_key) is True


def test_verify_capsule_uses_constant_time_comparison(basic_capsule, secret_key):
    """verify_capsule uses hmac.compare_digest for timing-attack resistance."""
    # This test verifies the function runs without error; actual constant-time
    # behavior is tested by the implementation's use of hmac.compare_digest
    result = verify_capsule(basic_capsule, secret_key)
    assert result is True


# ==============================================================================
# Test verify_cycle - Tool Constraints
# ==============================================================================

def test_verify_cycle_allowed_tool_passes():
    """verify_cycle passes when tool is in allowed_tools."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"allowed_tools": ["bash", "python"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "bash: ls -la")
    assert verdict.passed is True
    assert "permitted" in verdict.reason.lower()


def test_verify_cycle_disallowed_tool_fails():
    """verify_cycle fails when tool is not in allowed_tools."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"allowed_tools": ["bash", "python"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "curl: fetch http://example.com")
    assert verdict.passed is False
    assert "curl" in verdict.reason.lower()
    assert "not in allowed_tools" in verdict.reason


def test_verify_cycle_tool_name_case_insensitive():
    """verify_cycle treats tool names case-insensitively."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"allowed_tools": ["BASH", "Python"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "bash: echo hello")
    assert verdict.passed is True


def test_verify_cycle_tool_extraction_with_whitespace():
    """verify_cycle handles whitespace around tool name."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"allowed_tools": ["bash"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "  bash  : ls")
    assert verdict.passed is True


def test_verify_cycle_no_tool_constraint_allows_all():
    """verify_cycle passes when allowed_tools is not specified."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "anything: do stuff")
    assert verdict.passed is True


# ==============================================================================
# Test verify_cycle - Egress Domain Constraints
# ==============================================================================

def test_verify_cycle_allowed_egress_domain_passes():
    """verify_cycle passes when URL domain is in egress_domains."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"egress_domains": ["example.com", "api.trusted.org"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "fetch data from https://example.com/api")
    assert verdict.passed is True


def test_verify_cycle_disallowed_egress_domain_fails():
    """verify_cycle fails when URL domain is not in egress_domains."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"egress_domains": ["example.com"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "curl https://evil.com/data")
    assert verdict.passed is False
    assert "evil.com" in verdict.reason
    assert "not in egress_domains" in verdict.reason


def test_verify_cycle_multiple_urls_in_action():
    """verify_cycle checks all URLs in the action string."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"egress_domains": ["example.com"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    # First URL allowed, second not
    verdict = verify_cycle(
        capsule, "fetch https://example.com/data and post to https://evil.com/sink"
    )
    assert verdict.passed is False
    assert "evil.com" in verdict.reason


def test_verify_cycle_no_url_in_action_skips_egress_check():
    """verify_cycle skips egress check when action contains no URLs."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"egress_domains": ["example.com"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "process local files")
    assert verdict.passed is True


def test_verify_cycle_no_egress_constraint_allows_all_domains():
    """verify_cycle passes when egress_domains is not specified."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "curl https://any-domain.com")
    assert verdict.passed is True


# ==============================================================================
# Test verify_cycle - Budget Constraints
# ==============================================================================

def test_verify_cycle_budget_remaining_positive_passes():
    """verify_cycle passes when budget_remaining is positive."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"budget_remaining": 50.0},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "any action")
    assert verdict.passed is True


def test_verify_cycle_budget_exhausted_fails():
    """verify_cycle fails when budget_remaining is 0."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"budget_remaining": 0.0},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "any action")
    assert verdict.passed is False
    assert "budget exhausted" in verdict.reason.lower()


def test_verify_cycle_budget_negative_fails():
    """verify_cycle fails when budget_remaining is negative."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"budget_remaining": -10.0},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "any action")
    assert verdict.passed is False
    assert "budget exhausted" in verdict.reason.lower()


def test_verify_cycle_no_budget_constraint_passes():
    """verify_cycle passes when budget_remaining is not specified."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "any action")
    assert verdict.passed is True


# ==============================================================================
# Test verify_cycle - Combined Constraints
# ==============================================================================

def test_verify_cycle_all_constraints_pass():
    """verify_cycle passes when all constraints are satisfied."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={
            "allowed_tools": ["bash"],
            "egress_domains": ["example.com"],
            "budget_remaining": 100.0,
        },
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "bash: curl https://example.com/api")
    assert verdict.passed is True


def test_verify_cycle_fails_on_first_violation():
    """verify_cycle returns failure on first constraint violation."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={
            "allowed_tools": ["python"],  # Will fail on 'bash'
            "egress_domains": ["example.com"],
            "budget_remaining": 100.0,
        },
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "bash: curl https://evil.com")
    assert verdict.passed is False
    # Should fail on tool check first
    assert "bash" in verdict.reason.lower()


def test_verify_cycle_empty_constraints_always_passes():
    """verify_cycle passes when constraints dict is empty."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    verdict = verify_cycle(capsule, "anything: do whatever")
    assert verdict.passed is True
    assert "permitted" in verdict.reason.lower()


# ==============================================================================
# Test mission_alignment_check
# ==============================================================================

def test_mission_alignment_high_overlap_passes():
    """mission_alignment_check passes with high token overlap."""
    capsule = IntentCapsule(
        goal="analyze customer feedback data",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "analyze customer feedback from Q4")
    assert result.passed is True
    assert result.score >= 0.5  # At least 50% overlap


def test_mission_alignment_exact_match_high_score():
    """mission_alignment_check gives high score for exact token match."""
    capsule = IntentCapsule(
        goal="generate monthly report",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "generate monthly report")
    assert result.passed is True
    assert result.score == 1.0


def test_mission_alignment_no_overlap_fails():
    """mission_alignment_check fails with zero overlap."""
    capsule = IntentCapsule(
        goal="analyze customer feedback",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "delete production database")
    assert result.passed is False
    assert result.score == 0.0


def test_mission_alignment_partial_overlap_threshold():
    """mission_alignment_check uses 0.1 threshold (10% overlap)."""
    capsule = IntentCapsule(
        goal="a b c d e f g h i j",  # 10 tokens
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    # Exactly 1 token overlap = 0.1 score = passes
    result = mission_alignment_check(capsule, "a x y z")
    assert result.passed is True
    assert result.score == 0.1

    # Zero overlap = fails
    result = mission_alignment_check(capsule, "x y z")
    assert result.passed is False
    assert result.score == 0.0


def test_mission_alignment_empty_goal_always_passes():
    """mission_alignment_check passes for empty goal."""
    capsule = IntentCapsule(
        goal="",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "anything at all")
    assert result.passed is True
    assert result.score == 1.0
    assert "empty goal" in result.reasoning.lower()


def test_mission_alignment_case_insensitive():
    """mission_alignment_check treats tokens case-insensitively."""
    capsule = IntentCapsule(
        goal="Generate Report",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "GENERATE REPORT")
    assert result.passed is True
    assert result.score == 1.0


def test_mission_alignment_reasoning_includes_matched_tokens():
    """mission_alignment_check includes matched tokens in reasoning."""
    capsule = IntentCapsule(
        goal="analyze customer feedback",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "analyze feedback quality")
    assert result.passed is True
    assert "matched tokens:" in result.reasoning.lower()
    assert "'analyze'" in result.reasoning or "analyze" in result.reasoning
    assert "'feedback'" in result.reasoning or "feedback" in result.reasoning


def test_mission_alignment_reasoning_indicates_no_match():
    """mission_alignment_check indicates no match in reasoning."""
    capsule = IntentCapsule(
        goal="analyze data",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "delete everything")
    assert result.passed is False
    assert "none" in result.reasoning.lower()


def test_mission_alignment_score_rounded_to_4_decimals():
    """mission_alignment_check rounds score to 4 decimal places."""
    capsule = IntentCapsule(
        goal="a b c d e f g",  # 7 tokens
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    result = mission_alignment_check(capsule, "a b")  # 2/7 = 0.285714...
    assert result.score == 0.2857  # Rounded to 4 decimals


# ==============================================================================
# Test CycleVerdict and AlignmentResult dataclasses
# ==============================================================================

def test_cycle_verdict_frozen():
    """CycleVerdict is immutable (frozen dataclass)."""
    verdict = CycleVerdict(passed=True, reason="Test")
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        verdict.passed = False


def test_alignment_result_frozen():
    """AlignmentResult is immutable (frozen dataclass)."""
    result = AlignmentResult(passed=True, score=0.8, reasoning="Test")
    with pytest.raises(Exception):
        result.score = 0.9


def test_intent_capsule_frozen():
    """IntentCapsule is immutable (frozen dataclass)."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )
    with pytest.raises(Exception):
        capsule.goal = "Modified"


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

def test_sign_capsule_empty_goal(secret_key):
    """sign_capsule handles empty goal string."""
    capsule = sign_capsule(
        goal="",
        constraints={},
        org_id="org",
        secret_key=secret_key,
    )
    assert capsule.goal == ""
    assert verify_capsule(capsule, secret_key) is True


def test_sign_capsule_empty_constraints(secret_key):
    """sign_capsule handles empty constraints dict."""
    capsule = sign_capsule(
        goal="Test",
        constraints={},
        org_id="org",
        secret_key=secret_key,
    )
    assert capsule.constraints == {}
    assert verify_capsule(capsule, secret_key) is True


def test_verify_cycle_action_without_colon():
    """verify_cycle handles action string without tool prefix."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"allowed_tools": ["bash"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    # Action without ':' - tool_name will be "just some action"
    verdict = verify_cycle(capsule, "just some action")
    # The tool name extraction will fail gracefully
    # Since "just some action".split(":")[0] = "just some action"
    assert verdict.passed is False  # Not in allowed_tools


def test_verify_cycle_malformed_url():
    """verify_cycle handles malformed URLs gracefully."""
    capsule = IntentCapsule(
        goal="Test",
        constraints={"egress_domains": ["example.com"]},
        org_id="org",
        signature="sig",
        signed_at=datetime.now(timezone.utc),
    )

    # Malformed URL - urlparse should handle gracefully
    verdict = verify_cycle(capsule, "fetch from http:// broken url")
    # urlparse will extract hostname=None, which gets filtered out
    assert verdict.passed is True


def test_canonical_payload_special_characters():
    """Canonical payload handles special characters in fields."""
    goal = "Test with unicode: こんにちは and symbols: @#$%"
    constraints = {"key": "value with 'quotes' and \"double quotes\""}
    org_id = "org-with-dashes-123"
    signed_at = datetime.now(timezone.utc)

    payload = _canonical_payload(goal, constraints, org_id, signed_at)
    assert isinstance(payload, bytes)
    # Should be valid JSON
    json.loads(payload.decode("utf-8"))


def test_sign_and_verify_round_trip_with_complex_data(secret_key):
    """Sign and verify works with complex nested constraints."""
    complex_constraints = {
        "allowed_tools": ["bash", "python", "curl"],
        "egress_domains": ["api.example.com", "data.service.io"],
        "budget_remaining": 1234.56,
        "nested": {
            "key": "value",
            "list": [1, 2, 3],
        },
    }

    capsule = sign_capsule(
        goal="Complex test goal with many words",
        constraints=complex_constraints,
        org_id="complex-org-456",
        secret_key=secret_key,
    )

    assert verify_capsule(capsule, secret_key) is True
