"""Formal invariant checking -- the last line of defence.

Every proposed agent action passes through InvariantChecker.check_all()
before execution.  ANY single invariant failure results in an automatic
REJECT with no override path.  This is by design: invariants are
non-negotiable safety properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from kintsugi.security.intent_capsule import IntentCapsule, verify_capsule
from kintsugi.security.monitor import SecurityMonitor, Verdict
from kintsugi.security.pii import PIIRedactor


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class InvariantContext:
    """All inputs needed for a full invariant check.

    Callers populate only the fields relevant to the current action;
    unused fields default to None and their corresponding checks are skipped.
    """

    command: Optional[str] = None
    url: Optional[str] = None
    egress_allowlist: Optional[List[str]] = None
    cost: Optional[float] = None
    budget_remaining: Optional[float] = None
    text: Optional[str] = None
    capsule: Optional[IntentCapsule] = None
    secret_key: Optional[str] = None


@dataclass(frozen=True)
class InvariantResult:
    """Outcome of running all applicable invariants."""

    all_passed: bool
    failures: List[str]
    checked_at: datetime


# ---------------------------------------------------------------------------
# InvariantChecker
# ---------------------------------------------------------------------------

class InvariantChecker:
    """Boolean invariant checks that MUST all pass for an action to proceed.

    Each ``check_*`` method returns True on success.  ``check_all`` runs
    every applicable check and aggregates failures.
    """

    def __init__(self) -> None:
        self._monitor = SecurityMonitor()
        self._pii = PIIRedactor()

    # -- individual checks --------------------------------------------------

    def check_shell_safety(self, command: str) -> bool:
        """Delegate to SecurityMonitor; returns False if BLOCK verdict."""
        verdict = self._monitor.check_command(command)
        return verdict.verdict != Verdict.BLOCK

    def check_egress(self, url: str, allowlist: List[str]) -> bool:
        """Return True if *url*'s host appears in *allowlist*."""
        from urllib.parse import urlparse

        if not allowlist:
            return False
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        for allowed in allowlist:
            allowed = allowed.lower()
            if host == allowed or host.endswith("." + allowed):
                return True
        return False

    def check_budget(self, cost: float, remaining: float) -> bool:
        """Return True if the action cost fits within the remaining budget."""
        return cost <= remaining

    def check_pii_redacted(self, text: str) -> bool:
        """Return True if *text* contains NO detectable PII."""
        detections = self._pii.detect(text)
        return len(detections) == 0

    def check_intent_signature(self, capsule: IntentCapsule, secret_key: str) -> bool:
        """Return True if the capsule signature is valid and not expired."""
        return verify_capsule(capsule, secret_key)

    # -- aggregate ----------------------------------------------------------

    def check_all(self, context: InvariantContext) -> InvariantResult:
        """Run every applicable invariant and return the aggregate result.

        An invariant is *applicable* when the corresponding fields in
        ``context`` are not None.  ANY failure causes ``all_passed=False``.
        """
        failures: List[str] = []

        if context.command is not None:
            if not self.check_shell_safety(context.command):
                failures.append("shell_safety")

        if context.url is not None and context.egress_allowlist is not None:
            if not self.check_egress(context.url, context.egress_allowlist):
                failures.append("egress")

        if context.cost is not None and context.budget_remaining is not None:
            if not self.check_budget(context.cost, context.budget_remaining):
                failures.append("budget")

        if context.text is not None:
            if not self.check_pii_redacted(context.text):
                failures.append("pii_redacted")

        if context.capsule is not None and context.secret_key is not None:
            if not self.check_intent_signature(context.capsule, context.secret_key):
                failures.append("intent_signature")

        return InvariantResult(
            all_passed=len(failures) == 0,
            failures=failures,
            checked_at=datetime.now(timezone.utc),
        )
