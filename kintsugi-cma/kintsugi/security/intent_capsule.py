"""Cryptographic mandate signing and verification for agent intent capsules.

Every agentic action in Kintsugi must trace back to a signed IntentCapsule
that encodes the organization's goal, constraints, and identity. This module
provides HMAC-SHA256 signing, verification, per-cycle constraint checking,
and mission-alignment scaffolding.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class CycleVerdict:
    """Result of checking a single action against capsule constraints."""

    passed: bool
    reason: str


@dataclass(frozen=True)
class AlignmentResult:
    """Result of checking an action against the capsule's stated mission goal."""

    passed: bool
    score: float
    reasoning: str


@dataclass(frozen=True)
class IntentCapsule:
    """Immutable, signed mandate that authorises a bounded set of agent actions.

    Fields:
        goal:        Human-readable mission statement.
        constraints:  Dict of hard limits (budget_remaining, allowed_tools,
                      egress_domains, etc.).
        org_id:       Organisation identifier.
        signature:    HMAC-SHA256 hex digest computed over canonical payload.
        signed_at:    UTC timestamp of signing.
        expires_at:   Optional UTC expiry; None means no expiry.
    """

    goal: str
    constraints: dict
    org_id: str
    signature: str
    signed_at: datetime
    expires_at: Optional[datetime] = None


def _canonical_payload(
    goal: str,
    constraints: dict,
    org_id: str,
    signed_at: datetime,
) -> bytes:
    """Produce a deterministic byte string for signing.

    Keys are sorted, datetimes are ISO-formatted, and the result is
    UTF-8 encoded JSON with no extra whitespace.
    """
    payload = {
        "goal": goal,
        "constraints": constraints,
        "org_id": org_id,
        "signed_at": signed_at.isoformat(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_capsule(
    goal: str,
    constraints: dict,
    org_id: str,
    secret_key: str,
    *,
    expires_at: Optional[datetime] = None,
) -> IntentCapsule:
    """Create and sign a new IntentCapsule.

    Args:
        goal:        The mission goal for this agent session.
        constraints: Hard constraint dict (budget_remaining, allowed_tools, etc.).
        org_id:      Organisation identifier.
        secret_key:  HMAC secret (should be stored securely, never logged).
        expires_at:  Optional expiry timestamp (UTC).

    Returns:
        A fully-signed IntentCapsule.
    """
    signed_at = datetime.now(timezone.utc)
    payload = _canonical_payload(goal, constraints, org_id, signed_at)
    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return IntentCapsule(
        goal=goal,
        constraints=constraints,
        org_id=org_id,
        signature=signature,
        signed_at=signed_at,
        expires_at=expires_at,
    )


def verify_capsule(capsule: IntentCapsule, secret_key: str) -> bool:
    """Recompute the HMAC and compare to the capsule's stored signature.

    Also rejects expired capsules.
    """
    if capsule.expires_at is not None:
        now = datetime.now(timezone.utc)
        if now > capsule.expires_at:
            return False

    payload = _canonical_payload(
        capsule.goal,
        capsule.constraints,
        capsule.org_id,
        capsule.signed_at,
    )
    expected = hmac.new(
        secret_key.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, capsule.signature)


def verify_cycle(capsule: IntentCapsule, current_action: str) -> CycleVerdict:
    """Check whether *current_action* is permitted by the capsule's constraints.

    Inspected constraint keys:
        budget_remaining  (float) -- action cost header parsed from action string
        allowed_tools     (list[str])
        egress_domains    (list[str])
    """
    constraints = capsule.constraints

    # -- Tool allow-list ---------------------------------------------------
    allowed_tools = constraints.get("allowed_tools")
    if allowed_tools is not None:
        tool_name = current_action.split(":")[0].strip().lower()
        if tool_name and tool_name not in [t.lower() for t in allowed_tools]:
            return CycleVerdict(
                passed=False,
                reason=f"Tool '{tool_name}' is not in allowed_tools: {allowed_tools}",
            )

    # -- Egress domain allow-list ------------------------------------------
    egress_domains = constraints.get("egress_domains")
    if egress_domains is not None and "://" in current_action:
        for token in current_action.split():
            if "://" in token:
                parsed = urlparse(token)
                domain = parsed.hostname or ""
                if domain and domain not in egress_domains:
                    return CycleVerdict(
                        passed=False,
                        reason=f"Domain '{domain}' not in egress_domains allowlist.",
                    )

    # -- Budget remaining --------------------------------------------------
    budget_remaining = constraints.get("budget_remaining")
    if budget_remaining is not None and budget_remaining <= 0:
        return CycleVerdict(passed=False, reason="Budget exhausted.")

    return CycleVerdict(passed=True, reason="Action permitted by capsule constraints.")


def mission_alignment_check(
    capsule: IntentCapsule,
    proposed_action: str,
) -> AlignmentResult:
    """Score how well *proposed_action* aligns with the capsule's stated goal.

    This is a deterministic keyword-overlap heuristic that serves as a
    scaffold. In production it will be replaced by an LLM-backed evaluator
    while preserving the same interface.
    """
    goal_tokens = set(capsule.goal.lower().split())
    action_tokens = set(proposed_action.lower().split())
    if not goal_tokens:
        return AlignmentResult(
            passed=True,
            score=1.0,
            reasoning="Empty goal; all actions permitted.",
        )
    overlap = goal_tokens & action_tokens
    score = len(overlap) / len(goal_tokens)
    passed = score >= 0.1  # very permissive threshold for the heuristic
    reasoning = (
        f"Token overlap {len(overlap)}/{len(goal_tokens)} "
        f"between goal and proposed action. "
        f"Matched tokens: {sorted(overlap) if overlap else 'none'}. "
        "This heuristic will be replaced by an LLM evaluator in Phase 2."
    )
    return AlignmentResult(passed=passed, score=round(score, 4), reasoning=reasoning)
