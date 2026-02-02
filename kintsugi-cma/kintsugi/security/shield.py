"""Hard constraint enforcement layer (Shield).

The Shield composes four independent enforcers -- budget, egress, rate-limit,
and circuit-breaker -- and produces a single ALLOW/BLOCK verdict for every
proposed agent action.  No soft overrides: if any enforcer blocks, the action
is rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class ShieldDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class ShieldVerdict:
    """Immutable result of a Shield check."""

    decision: ShieldDecision
    reason: str


# ---------------------------------------------------------------------------
# ShieldConfig
# ---------------------------------------------------------------------------

@dataclass
class ShieldConfig:
    """Parsed org-level shield settings.

    Expected keys in the source dict:
        budget_session_limit (float)
        budget_daily_limit   (float)
        egress_allowlist     (list[str])   -- allowed domains
        rate_limits          (dict[str, dict])  -- tool_name -> {rate, burst}
        circuit_breaker_threshold (int)
    """

    budget_session_limit: float = 10.0
    budget_daily_limit: float = 100.0
    egress_allowlist: List[str] = field(default_factory=list)
    rate_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    circuit_breaker_threshold: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShieldConfig:
        """Build a ShieldConfig from a plain dict (e.g. loaded from YAML/JSON)."""
        return cls(
            budget_session_limit=float(data.get("budget_session_limit", 10.0)),
            budget_daily_limit=float(data.get("budget_daily_limit", 100.0)),
            egress_allowlist=list(data.get("egress_allowlist", [])),
            rate_limits=dict(data.get("rate_limits", {})),
            circuit_breaker_threshold=int(data.get("circuit_breaker_threshold", 5)),
        )


# ---------------------------------------------------------------------------
# BudgetEnforcer
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """Tracks token/cost spend per session and per calendar day (UTC)."""

    def __init__(self, session_limit: float, daily_limit: float) -> None:
        self.session_limit: float = session_limit
        self.daily_limit: float = daily_limit
        self.session_spent: float = 0.0
        self.daily_spent: float = 0.0
        self.daily_reset_at: datetime = self._next_midnight()

    @staticmethod
    def _next_midnight() -> datetime:
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).__class__(
            now.year, now.month, now.day + 1, tzinfo=timezone.utc
        ) if now.hour or now.minute or now.second or now.microsecond else now

    def _maybe_reset_daily(self) -> None:
        now = datetime.now(timezone.utc)
        if now >= self.daily_reset_at:
            self.daily_spent = 0.0
            self.daily_reset_at = self._next_midnight()

    def check_budget(self, cost: float) -> bool:
        """Return True if *cost* fits within both session and daily limits."""
        self._maybe_reset_daily()
        if self.session_spent + cost > self.session_limit:
            return False
        if self.daily_spent + cost > self.daily_limit:
            return False
        return True

    def record_spend(self, cost: float) -> None:
        """Commit a spend after an action has been approved and executed."""
        self._maybe_reset_daily()
        self.session_spent += cost
        self.daily_spent += cost


# ---------------------------------------------------------------------------
# EgressValidator
# ---------------------------------------------------------------------------

class EgressValidator:
    """Domain-level allowlist for outbound network requests."""

    def __init__(self, allowlist: List[str]) -> None:
        self.allowlist: List[str] = [d.lower().strip() for d in allowlist]

    def check_egress(self, url: str) -> bool:
        """Return True if the URL's host is on the allowlist.

        If the allowlist is empty, all egress is denied by default (fail-closed).
        """
        if not self.allowlist:
            return False
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        for allowed in self.allowlist:
            if host == allowed or host.endswith("." + allowed):
                return True
        return False


# ---------------------------------------------------------------------------
# RateLimiter (token bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-tool token-bucket rate limiter.

    Each tool has its own bucket with a configurable *rate* (tokens/sec)
    and *burst* (max tokens).
    """

    def __init__(self, tool_configs: Dict[str, Dict[str, float]]) -> None:
        # tool_configs: {"tool_name": {"rate": 1.0, "burst": 5.0}}
        self._buckets: Dict[str, Dict[str, float]] = {}
        for tool, cfg in tool_configs.items():
            rate = float(cfg.get("rate", 1.0))
            burst = float(cfg.get("burst", 5.0))
            self._buckets[tool] = {
                "rate": rate,
                "burst": burst,
                "tokens": burst,
                "last": time.monotonic(),
            }

    def _refill(self, bucket: Dict[str, float]) -> None:
        now = time.monotonic()
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(
            bucket["burst"],
            bucket["tokens"] + elapsed * bucket["rate"],
        )
        bucket["last"] = now

    def check_rate(self, tool_name: str) -> bool:
        """Consume one token for *tool_name*; return False if exhausted.

        Unknown tools are allowed (no rate limit configured).
        """
        bucket = self._buckets.get(tool_name)
        if bucket is None:
            return True
        self._refill(bucket)
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Per-tool consecutive-failure circuit breaker.

    After *threshold* consecutive failures the circuit opens and the tool
    is blocked until a manual reset or a successful probe.
    """

    def __init__(self, threshold: int = 5) -> None:
        self.threshold: int = threshold
        self._failures: Dict[str, int] = {}

    def record_result(self, tool: str, success: bool) -> None:
        """Record the outcome of a tool invocation."""
        if success:
            self._failures[tool] = 0
        else:
            self._failures[tool] = self._failures.get(tool, 0) + 1

    def is_open(self, tool: str) -> bool:
        """Return True if the circuit for *tool* is open (i.e. blocked)."""
        return self._failures.get(tool, 0) >= self.threshold

    def reset(self, tool: str) -> None:
        """Manually reset the circuit for *tool*."""
        self._failures[tool] = 0


# ---------------------------------------------------------------------------
# Shield (compositor)
# ---------------------------------------------------------------------------

class Shield:
    """Top-level hard-constraint compositor.

    Composes BudgetEnforcer, EgressValidator, RateLimiter, and CircuitBreaker
    into a single ``check_action`` call.
    """

    def __init__(self, config: ShieldConfig) -> None:
        self.config = config
        self.budget = BudgetEnforcer(config.budget_session_limit, config.budget_daily_limit)
        self.egress = EgressValidator(config.egress_allowlist)
        self.rate_limiter = RateLimiter(config.rate_limits)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker_threshold)

    def check_action(
        self,
        action_type: str,
        cost: float = 0.0,
        url: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> ShieldVerdict:
        """Run all enforcers and return a single verdict.

        Args:
            action_type: Descriptive label for the action.
            cost:        Estimated cost for budget enforcement.
            url:         Outbound URL (if any) for egress check.
            tool:        Tool name for rate-limit and circuit-breaker checks.

        Returns:
            ShieldVerdict with ALLOW or BLOCK and a human-readable reason.
        """
        # Budget
        if cost > 0 and not self.budget.check_budget(cost):
            return ShieldVerdict(ShieldDecision.BLOCK, f"Budget exceeded for action '{action_type}' (cost={cost}).")

        # Egress
        if url is not None and not self.egress.check_egress(url):
            return ShieldVerdict(ShieldDecision.BLOCK, f"Egress blocked: domain not in allowlist for URL '{url}'.")

        # Circuit breaker (check before rate limiter so we don't waste tokens)
        if tool is not None and self.circuit_breaker.is_open(tool):
            return ShieldVerdict(ShieldDecision.BLOCK, f"Circuit breaker open for tool '{tool}'.")

        # Rate limiter
        if tool is not None and not self.rate_limiter.check_rate(tool):
            return ShieldVerdict(ShieldDecision.BLOCK, f"Rate limit exceeded for tool '{tool}'.")

        return ShieldVerdict(ShieldDecision.ALLOW, "All shield checks passed.")
