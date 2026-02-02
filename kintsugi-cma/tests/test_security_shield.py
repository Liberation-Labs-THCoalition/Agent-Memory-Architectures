"""Comprehensive test suite for kintsugi.security.shield module.

Tests all components: ShieldConfig, BudgetEnforcer, EgressValidator,
RateLimiter, CircuitBreaker, and the Shield compositor.

Target: >90% code coverage.
"""

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from kintsugi.security.shield import (
    ShieldDecision,
    ShieldVerdict,
    ShieldConfig,
    BudgetEnforcer,
    EgressValidator,
    RateLimiter,
    CircuitBreaker,
    Shield,
)


# =============================================================================
# ShieldDecision and ShieldVerdict Tests
# =============================================================================


class TestShieldDecision:
    """Test the ShieldDecision enum."""

    def test_decision_values(self):
        """Test enum values are correct."""
        assert ShieldDecision.ALLOW == "ALLOW"
        assert ShieldDecision.BLOCK == "BLOCK"

    def test_decision_str(self):
        """Test string representation."""
        # ShieldDecision is a str Enum, so value equals the string
        assert ShieldDecision.ALLOW.value == "ALLOW"
        assert ShieldDecision.BLOCK.value == "BLOCK"


class TestShieldVerdict:
    """Test the ShieldVerdict dataclass."""

    def test_verdict_creation(self):
        """Test creating a verdict."""
        verdict = ShieldVerdict(ShieldDecision.ALLOW, "Test passed")
        assert verdict.decision == ShieldDecision.ALLOW
        assert verdict.reason == "Test passed"

    def test_verdict_immutability(self):
        """Test that verdict is frozen (immutable)."""
        verdict = ShieldVerdict(ShieldDecision.BLOCK, "Test failed")
        with pytest.raises(Exception):  # FrozenInstanceError
            verdict.decision = ShieldDecision.ALLOW


# =============================================================================
# ShieldConfig Tests
# =============================================================================


class TestShieldConfig:
    """Test ShieldConfig dataclass and from_dict method."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ShieldConfig()
        assert config.budget_session_limit == 10.0
        assert config.budget_daily_limit == 100.0
        assert config.egress_allowlist == []
        assert config.rate_limits == {}
        assert config.circuit_breaker_threshold == 5

    def test_from_dict_empty(self):
        """Test from_dict with empty dict uses defaults."""
        config = ShieldConfig.from_dict({})
        assert config.budget_session_limit == 10.0
        assert config.budget_daily_limit == 100.0
        assert config.egress_allowlist == []
        assert config.rate_limits == {}
        assert config.circuit_breaker_threshold == 5

    def test_from_dict_partial(self):
        """Test from_dict with partial data uses some defaults."""
        config = ShieldConfig.from_dict({
            "budget_session_limit": 50.0,
            "egress_allowlist": ["example.com"],
        })
        assert config.budget_session_limit == 50.0
        assert config.budget_daily_limit == 100.0  # default
        assert config.egress_allowlist == ["example.com"]
        assert config.rate_limits == {}  # default
        assert config.circuit_breaker_threshold == 5  # default

    def test_from_dict_full(self):
        """Test from_dict with all fields specified."""
        config = ShieldConfig.from_dict({
            "budget_session_limit": 25.0,
            "budget_daily_limit": 200.0,
            "egress_allowlist": ["example.com", "api.test.com"],
            "rate_limits": {"tool1": {"rate": 2.0, "burst": 10.0}},
            "circuit_breaker_threshold": 3,
        })
        assert config.budget_session_limit == 25.0
        assert config.budget_daily_limit == 200.0
        assert config.egress_allowlist == ["example.com", "api.test.com"]
        assert config.rate_limits == {"tool1": {"rate": 2.0, "burst": 10.0}}
        assert config.circuit_breaker_threshold == 3

    def test_from_dict_type_coercion(self):
        """Test that from_dict properly coerces types."""
        config = ShieldConfig.from_dict({
            "budget_session_limit": "15.5",  # string to float
            "budget_daily_limit": 150,  # int to float
            "circuit_breaker_threshold": "7",  # string to int
        })
        assert config.budget_session_limit == 15.5
        assert config.budget_daily_limit == 150.0
        assert config.circuit_breaker_threshold == 7


# =============================================================================
# BudgetEnforcer Tests
# =============================================================================


class TestBudgetEnforcer:
    """Test the BudgetEnforcer class."""

    def test_initialization(self):
        """Test that enforcer initializes with correct values."""
        enforcer = BudgetEnforcer(session_limit=20.0, daily_limit=150.0)
        assert enforcer.session_limit == 20.0
        assert enforcer.daily_limit == 150.0
        assert enforcer.session_spent == 0.0
        assert enforcer.daily_spent == 0.0
        assert enforcer.daily_reset_at > datetime.now(timezone.utc)

    def test_check_budget_within_limits(self):
        """Test check_budget passes when within limits."""
        enforcer = BudgetEnforcer(session_limit=10.0, daily_limit=100.0)
        assert enforcer.check_budget(5.0) is True
        assert enforcer.check_budget(10.0) is True

    def test_check_budget_session_exceeded(self):
        """Test check_budget fails when session limit exceeded."""
        enforcer = BudgetEnforcer(session_limit=10.0, daily_limit=100.0)
        enforcer.record_spend(8.0)
        # The implementation uses > (greater than), so exactly at limit passes
        assert enforcer.check_budget(2.0) is True  # 8 + 2 = 10, at limit (passes)
        assert enforcer.check_budget(2.01) is False  # 8 + 2.01 = 10.01, over limit (fails)

    def test_check_budget_daily_exceeded(self):
        """Test check_budget fails when daily limit exceeded."""
        # Use high session limit so session check doesn't interfere
        enforcer = BudgetEnforcer(session_limit=200.0, daily_limit=100.0)
        enforcer.record_spend(95.0)
        # The implementation uses > (greater than), so exactly at limit passes
        assert enforcer.check_budget(5.0) is True  # 95 + 5 = 100, at limit (passes)
        assert enforcer.check_budget(5.01) is False  # 95 + 5.01 = 100.01, over limit (fails)

    def test_record_spend_accumulates(self):
        """Test that record_spend correctly accumulates costs."""
        enforcer = BudgetEnforcer(session_limit=100.0, daily_limit=500.0)
        assert enforcer.session_spent == 0.0
        assert enforcer.daily_spent == 0.0

        enforcer.record_spend(10.5)
        assert enforcer.session_spent == 10.5
        assert enforcer.daily_spent == 10.5

        enforcer.record_spend(25.3)
        assert enforcer.session_spent == 35.8
        assert enforcer.daily_spent == 35.8

    def test_daily_reset_logic(self):
        """Test that daily budget resets at midnight UTC."""
        enforcer = BudgetEnforcer(session_limit=50.0, daily_limit=100.0)
        enforcer.record_spend(80.0)
        assert enforcer.daily_spent == 80.0

        # Mock datetime to simulate passing midnight
        future_time = datetime.now(timezone.utc) + timedelta(days=1, hours=1)
        with patch('kintsugi.security.shield.datetime') as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # This should trigger daily reset
            enforcer._maybe_reset_daily()
            assert enforcer.daily_spent == 0.0
            assert enforcer.session_spent == 80.0  # session doesn't reset

    def test_zero_cost_check(self):
        """Test that zero cost checks always pass."""
        enforcer = BudgetEnforcer(session_limit=10.0, daily_limit=100.0)
        enforcer.record_spend(10.0)  # At session limit
        assert enforcer.check_budget(0.0) is True  # Zero cost should pass

    def test_next_midnight_calculation(self):
        """Test that _next_midnight correctly calculates next midnight UTC."""
        enforcer = BudgetEnforcer(session_limit=10.0, daily_limit=100.0)
        next_midnight = enforcer._next_midnight()

        # Next midnight should be in the future
        assert next_midnight > datetime.now(timezone.utc)

        # Should be at exactly midnight (00:00:00)
        assert next_midnight.hour == 0
        assert next_midnight.minute == 0
        assert next_midnight.second == 0
        assert next_midnight.microsecond == 0


# =============================================================================
# EgressValidator Tests
# =============================================================================


class TestEgressValidator:
    """Test the EgressValidator class."""

    def test_allowed_exact_domain(self):
        """Test that exact domain matches are allowed."""
        validator = EgressValidator(["example.com", "api.test.com"])
        assert validator.check_egress("https://example.com/path") is True
        assert validator.check_egress("https://api.test.com/api/v1") is True

    def test_subdomain_matching(self):
        """Test that subdomains are allowed when parent domain is in allowlist."""
        validator = EgressValidator(["example.com"])
        assert validator.check_egress("https://api.example.com/data") is True
        assert validator.check_egress("https://sub.api.example.com/data") is True

    def test_denied_domain(self):
        """Test that domains not in allowlist are denied."""
        validator = EgressValidator(["example.com"])
        assert validator.check_egress("https://evil.com/malware") is False
        assert validator.check_egress("https://notexample.com/") is False

    def test_empty_allowlist_denies_all(self):
        """Test that empty allowlist denies all egress (fail-closed)."""
        validator = EgressValidator([])
        assert validator.check_egress("https://example.com/") is False
        assert validator.check_egress("https://google.com/") is False

    def test_no_hostname(self):
        """Test that URLs without hostname are denied."""
        validator = EgressValidator(["example.com"])
        assert validator.check_egress("file:///local/path") is False
        assert validator.check_egress("data:text/plain,hello") is False
        assert validator.check_egress("") is False

    def test_case_insensitive_matching(self):
        """Test that domain matching is case-insensitive."""
        validator = EgressValidator(["Example.COM"])
        assert validator.check_egress("https://example.com/") is True
        assert validator.check_egress("https://EXAMPLE.COM/") is True
        assert validator.check_egress("https://ExAmPlE.cOm/") is True

    def test_various_url_formats(self):
        """Test various URL formats."""
        validator = EgressValidator(["example.com"])

        # With ports
        assert validator.check_egress("https://example.com:8080/api") is True

        # With auth
        assert validator.check_egress("https://user:pass@example.com/api") is True

        # Different schemes
        assert validator.check_egress("http://example.com/") is True
        assert validator.check_egress("ftp://example.com/files") is True

        # With query and fragment
        assert validator.check_egress("https://example.com/path?q=1#section") is True

    def test_similar_domain_not_matched(self):
        """Test that similar domains don't match if not a subdomain."""
        validator = EgressValidator(["example.com"])
        assert validator.check_egress("https://notexample.com/") is False
        assert validator.check_egress("https://example.com.evil.com/") is False

    def test_allowlist_normalization(self):
        """Test that allowlist entries are normalized (lowercased, stripped)."""
        validator = EgressValidator(["  Example.COM  ", " API.test.org"])
        assert "example.com" in validator.allowlist
        assert "api.test.org" in validator.allowlist


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Test the RateLimiter (token bucket) class."""

    def test_initialization(self):
        """Test that rate limiter initializes correctly."""
        configs = {
            "tool1": {"rate": 1.0, "burst": 5.0},
            "tool2": {"rate": 2.0, "burst": 10.0},
        }
        limiter = RateLimiter(configs)

        assert "tool1" in limiter._buckets
        assert "tool2" in limiter._buckets
        assert limiter._buckets["tool1"]["rate"] == 1.0
        assert limiter._buckets["tool1"]["burst"] == 5.0
        assert limiter._buckets["tool1"]["tokens"] == 5.0  # starts at burst

    def test_burst_allows_initial_calls(self):
        """Test that burst capacity allows initial rapid calls."""
        limiter = RateLimiter({"tool1": {"rate": 1.0, "burst": 3.0}})

        # Should allow 3 calls immediately (burst capacity)
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is True

        # 4th call should fail (tokens exhausted)
        assert limiter.check_rate("tool1") is False

    def test_exhaustion_blocks(self):
        """Test that exhausting tokens blocks further calls."""
        limiter = RateLimiter({"tool1": {"rate": 1.0, "burst": 2.0}})

        assert limiter.check_rate("tool1") is True  # 1 token left
        assert limiter.check_rate("tool1") is True  # 0 tokens left
        assert limiter.check_rate("tool1") is False  # blocked
        assert limiter.check_rate("tool1") is False  # still blocked

    def test_refill_after_time(self):
        """Test that tokens refill over time based on rate."""
        limiter = RateLimiter({"tool1": {"rate": 10.0, "burst": 2.0}})  # 10 tokens/sec

        # Exhaust tokens
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is False  # blocked

        # Wait for refill (0.15 sec should add 1.5 tokens)
        time.sleep(0.15)
        assert limiter.check_rate("tool1") is True  # should work now

        # Wait again
        time.sleep(0.15)
        assert limiter.check_rate("tool1") is True  # should work again

    def test_refill_capped_at_burst(self):
        """Test that tokens don't exceed burst capacity."""
        limiter = RateLimiter({"tool1": {"rate": 100.0, "burst": 5.0}})

        # Use one token
        assert limiter.check_rate("tool1") is True  # 4 tokens left

        # Wait long enough to refill way more than burst
        time.sleep(1.0)  # Would add 100 tokens if uncapped

        # Should only have burst capacity (5 tokens), not 104
        for i in range(5):
            assert limiter.check_rate("tool1") is True, f"Call {i+1} should succeed"

        # 6th call should fail
        assert limiter.check_rate("tool1") is False

    def test_unknown_tool_allowed(self):
        """Test that tools not in config are always allowed."""
        limiter = RateLimiter({"tool1": {"rate": 1.0, "burst": 1.0}})

        # Unknown tool should always pass
        assert limiter.check_rate("unknown_tool") is True
        assert limiter.check_rate("another_unknown") is True

        # Even when tool1 is exhausted
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is False  # exhausted
        assert limiter.check_rate("unknown_tool") is True  # still works

    def test_multiple_tools_independent(self):
        """Test that rate limits for different tools are independent."""
        limiter = RateLimiter({
            "tool1": {"rate": 1.0, "burst": 2.0},
            "tool2": {"rate": 1.0, "burst": 2.0},
        })

        # Exhaust tool1
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is True
        assert limiter.check_rate("tool1") is False

        # tool2 should still work
        assert limiter.check_rate("tool2") is True
        assert limiter.check_rate("tool2") is True
        assert limiter.check_rate("tool2") is False

    def test_default_rate_and_burst(self):
        """Test that missing rate/burst use defaults."""
        limiter = RateLimiter({"tool1": {}})
        assert limiter._buckets["tool1"]["rate"] == 1.0
        assert limiter._buckets["tool1"]["burst"] == 5.0


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Test the CircuitBreaker class."""

    def test_initialization(self):
        """Test that circuit breaker initializes correctly."""
        cb = CircuitBreaker(threshold=3)
        assert cb.threshold == 3
        assert cb._failures == {}

    def test_default_threshold(self):
        """Test default threshold is 5."""
        cb = CircuitBreaker()
        assert cb.threshold == 5

    def test_under_threshold_allows(self):
        """Test that circuit stays closed under threshold."""
        cb = CircuitBreaker(threshold=3)

        # Record some failures but stay under threshold
        cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is False

        cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is False

    def test_reaching_threshold_opens(self):
        """Test that circuit opens when threshold is reached."""
        cb = CircuitBreaker(threshold=3)

        # Record failures up to threshold
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)

        # Circuit should now be open
        assert cb.is_open("tool1") is True

    def test_success_resets_counter(self):
        """Test that success resets the failure counter."""
        cb = CircuitBreaker(threshold=3)

        # Record some failures
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is False

        # Success resets counter
        cb.record_result("tool1", success=True)
        assert cb.is_open("tool1") is False
        assert cb._failures["tool1"] == 0

        # Can fail again from zero
        cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is False

    def test_manual_reset(self):
        """Test manual reset of circuit."""
        cb = CircuitBreaker(threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is True

        # Manual reset
        cb.reset("tool1")
        assert cb.is_open("tool1") is False
        assert cb._failures["tool1"] == 0

    def test_multiple_tools_independent(self):
        """Test that different tools have independent circuits."""
        cb = CircuitBreaker(threshold=2)

        # Open circuit for tool1
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        assert cb.is_open("tool1") is True

        # tool2 should still be closed
        assert cb.is_open("tool2") is False

        # tool2 can fail independently
        cb.record_result("tool2", success=False)
        assert cb.is_open("tool2") is False

    def test_unknown_tool_closed(self):
        """Test that unknown tools have closed circuits."""
        cb = CircuitBreaker(threshold=3)
        assert cb.is_open("unknown_tool") is False

    def test_consecutive_failures_required(self):
        """Test that only consecutive failures count."""
        cb = CircuitBreaker(threshold=4)

        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        # Not at threshold yet
        assert cb.is_open("tool1") is False

        # Success resets
        cb.record_result("tool1", success=True)

        # Start counting again
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        cb.record_result("tool1", success=False)
        # Still not at threshold (only 3 consecutive)
        assert cb.is_open("tool1") is False


# =============================================================================
# Shield (Compositor) Tests
# =============================================================================


class TestShield:
    """Test the Shield compositor class."""

    def test_initialization(self):
        """Test that Shield initializes all enforcers."""
        config = ShieldConfig(
            budget_session_limit=50.0,
            budget_daily_limit=200.0,
            egress_allowlist=["example.com"],
            rate_limits={"tool1": {"rate": 1.0, "burst": 5.0}},
            circuit_breaker_threshold=3,
        )
        shield = Shield(config)

        assert shield.config == config
        assert isinstance(shield.budget, BudgetEnforcer)
        assert isinstance(shield.egress, EgressValidator)
        assert isinstance(shield.rate_limiter, RateLimiter)
        assert isinstance(shield.circuit_breaker, CircuitBreaker)

    def test_allow_when_all_pass(self):
        """Test that action is allowed when all checks pass."""
        config = ShieldConfig(
            budget_session_limit=100.0,
            budget_daily_limit=500.0,
            egress_allowlist=["example.com"],
            rate_limits={"tool1": {"rate": 10.0, "burst": 10.0}},
            circuit_breaker_threshold=5,
        )
        shield = Shield(config)

        verdict = shield.check_action(
            action_type="api_call",
            cost=10.0,
            url="https://example.com/api",
            tool="tool1",
        )

        assert verdict.decision == ShieldDecision.ALLOW
        assert verdict.reason == "All shield checks passed."

    def test_budget_block(self):
        """Test that budget enforcer can block."""
        config = ShieldConfig(
            budget_session_limit=10.0,
            budget_daily_limit=100.0,
            egress_allowlist=["example.com"],
        )
        shield = Shield(config)

        # First call should work
        verdict1 = shield.check_action("test", cost=5.0)
        assert verdict1.decision == ShieldDecision.ALLOW
        shield.budget.record_spend(5.0)

        # Second call exceeds session budget
        verdict2 = shield.check_action("test", cost=6.0)
        assert verdict2.decision == ShieldDecision.BLOCK
        assert "Budget exceeded" in verdict2.reason

    def test_egress_block(self):
        """Test that egress validator can block."""
        config = ShieldConfig(
            budget_session_limit=100.0,
            egress_allowlist=["allowed.com"],
        )
        shield = Shield(config)

        verdict = shield.check_action(
            action_type="network",
            url="https://blocked.com/api",
        )

        assert verdict.decision == ShieldDecision.BLOCK
        assert "Egress blocked" in verdict.reason
        assert "blocked.com" in verdict.reason

    def test_circuit_breaker_block(self):
        """Test that circuit breaker can block."""
        config = ShieldConfig(
            circuit_breaker_threshold=2,
        )
        shield = Shield(config)

        # Open the circuit
        shield.circuit_breaker.record_result("tool1", success=False)
        shield.circuit_breaker.record_result("tool1", success=False)

        verdict = shield.check_action(
            action_type="test",
            tool="tool1",
        )

        assert verdict.decision == ShieldDecision.BLOCK
        assert "Circuit breaker open" in verdict.reason

    def test_rate_limit_block(self):
        """Test that rate limiter can block."""
        config = ShieldConfig(
            rate_limits={"tool1": {"rate": 1.0, "burst": 1.0}},
        )
        shield = Shield(config)

        # First call should work
        verdict1 = shield.check_action("test", tool="tool1")
        assert verdict1.decision == ShieldDecision.ALLOW

        # Second call should be rate limited
        verdict2 = shield.check_action("test", tool="tool1")
        assert verdict2.decision == ShieldDecision.BLOCK
        assert "Rate limit exceeded" in verdict2.reason

    def test_check_order_budget_first(self):
        """Test that budget is checked before other enforcers."""
        config = ShieldConfig(
            budget_session_limit=5.0,
            egress_allowlist=[],  # Would block
            rate_limits={"tool1": {"rate": 1.0, "burst": 0.0}},  # Would block
        )
        shield = Shield(config)

        # Budget should fail first
        verdict = shield.check_action(
            action_type="test",
            cost=10.0,  # Over budget
            url="https://blocked.com",
            tool="tool1",
        )

        assert verdict.decision == ShieldDecision.BLOCK
        assert "Budget exceeded" in verdict.reason

    def test_check_order_egress_second(self):
        """Test that egress is checked after budget."""
        config = ShieldConfig(
            budget_session_limit=100.0,
            egress_allowlist=[],  # Empty = deny all
            rate_limits={"tool1": {"rate": 1.0, "burst": 0.0}},  # Would block
        )
        shield = Shield(config)

        verdict = shield.check_action(
            action_type="test",
            cost=5.0,
            url="https://any.com",
            tool="tool1",
        )

        assert verdict.decision == ShieldDecision.BLOCK
        assert "Egress blocked" in verdict.reason

    def test_check_order_circuit_breaker_third(self):
        """Test that circuit breaker is checked before rate limiter."""
        config = ShieldConfig(
            circuit_breaker_threshold=1,
            rate_limits={"tool1": {"rate": 1.0, "burst": 0.0}},  # Would block
        )
        shield = Shield(config)

        # Open circuit
        shield.circuit_breaker.record_result("tool1", success=False)

        verdict = shield.check_action(
            action_type="test",
            tool="tool1",
        )

        assert verdict.decision == ShieldDecision.BLOCK
        assert "Circuit breaker open" in verdict.reason

    def test_optional_parameters(self):
        """Test that checks are skipped when parameters are None."""
        config = ShieldConfig(
            egress_allowlist=[],  # Would block if URL provided
            rate_limits={"tool1": {"rate": 1.0, "burst": 0.0}},  # Would block if tool provided
        )
        shield = Shield(config)

        # No URL, no tool -> should pass
        verdict = shield.check_action(action_type="test")
        assert verdict.decision == ShieldDecision.ALLOW

    def test_zero_cost_skips_budget_check(self):
        """Test that zero cost doesn't trigger budget check."""
        config = ShieldConfig(
            budget_session_limit=0.0,  # Would block any positive cost
        )
        shield = Shield(config)

        # Zero cost should pass even with zero budget
        verdict = shield.check_action(action_type="test", cost=0.0)
        assert verdict.decision == ShieldDecision.ALLOW

    def test_multiple_enforcers_interact_correctly(self):
        """Test complex scenario with multiple enforcers."""
        config = ShieldConfig(
            budget_session_limit=100.0,
            budget_daily_limit=500.0,
            egress_allowlist=["api.example.com", "data.example.com"],
            rate_limits={
                "search": {"rate": 2.0, "burst": 5.0},
                "write": {"rate": 1.0, "burst": 2.0},
            },
            circuit_breaker_threshold=3,
        )
        shield = Shield(config)

        # Should pass: within budget, allowed domain, burst available
        v1 = shield.check_action(
            "search_query",
            cost=10.0,
            url="https://api.example.com/search",
            tool="search",
        )
        assert v1.decision == ShieldDecision.ALLOW
        shield.budget.record_spend(10.0)

        # Should pass: different tool, subdomain allowed
        v2 = shield.check_action(
            "write_data",
            cost=5.0,
            url="https://sub.data.example.com/api",
            tool="write",
        )
        assert v2.decision == ShieldDecision.ALLOW
        shield.budget.record_spend(5.0)

        # Open circuit for search
        shield.circuit_breaker.record_result("search", success=False)
        shield.circuit_breaker.record_result("search", success=False)
        shield.circuit_breaker.record_result("search", success=False)

        # Should fail: circuit open
        v3 = shield.check_action(
            "search_query",
            cost=10.0,
            url="https://api.example.com/search",
            tool="search",
        )
        assert v3.decision == ShieldDecision.BLOCK
        assert "Circuit breaker open" in v3.reason

        # Write should still work (different tool)
        v4 = shield.check_action(
            "write_data",
            cost=5.0,
            url="https://data.example.com/api",
            tool="write",
        )
        assert v4.decision == ShieldDecision.ALLOW

    def test_action_type_in_messages(self):
        """Test that action_type appears in block messages."""
        config = ShieldConfig(budget_session_limit=5.0)
        shield = Shield(config)

        verdict = shield.check_action("important_action", cost=10.0)
        assert "important_action" in verdict.reason


# =============================================================================
# Integration Tests
# =============================================================================


class TestShieldIntegration:
    """Integration tests for realistic Shield usage scenarios."""

    def test_full_workflow(self):
        """Test a complete workflow with budget tracking."""
        config = ShieldConfig(
            budget_session_limit=50.0,
            budget_daily_limit=200.0,
            egress_allowlist=["api.service.com"],
            rate_limits={"api_call": {"rate": 2.0, "burst": 5.0}},
            circuit_breaker_threshold=3,
        )
        shield = Shield(config)

        # Perform several actions
        for i in range(3):
            verdict = shield.check_action(
                f"api_call_{i}",
                cost=10.0,
                url="https://api.service.com/endpoint",
                tool="api_call",
            )
            assert verdict.decision == ShieldDecision.ALLOW
            shield.budget.record_spend(10.0)

        # Budget: 30/50 session, 30/200 daily
        # Rate: 2/5 tokens left

        # Should still work
        verdict = shield.check_action(
            "api_call_3",
            cost=15.0,
            url="https://api.service.com/endpoint",
            tool="api_call",
        )
        assert verdict.decision == ShieldDecision.ALLOW
        shield.budget.record_spend(15.0)

        # Budget: 45/50 session, 45/200 daily
        # Rate: 1/5 tokens left

        # Exceed session budget
        verdict = shield.check_action(
            "api_call_4",
            cost=10.0,
            url="https://api.service.com/endpoint",
            tool="api_call",
        )
        assert verdict.decision == ShieldDecision.BLOCK
        assert "Budget exceeded" in verdict.reason

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker opening and recovery."""
        config = ShieldConfig(circuit_breaker_threshold=2)
        shield = Shield(config)

        # Cause failures
        shield.circuit_breaker.record_result("flaky_tool", success=False)
        shield.circuit_breaker.record_result("flaky_tool", success=False)

        # Circuit should be open
        verdict = shield.check_action("test", tool="flaky_tool")
        assert verdict.decision == ShieldDecision.BLOCK

        # Manual recovery
        shield.circuit_breaker.reset("flaky_tool")

        # Should work again
        verdict = shield.check_action("test", tool="flaky_tool")
        assert verdict.decision == ShieldDecision.ALLOW

    def test_rate_limiter_with_delays(self):
        """Test rate limiter with time-based refills."""
        config = ShieldConfig(
            rate_limits={"slow_tool": {"rate": 5.0, "burst": 2.0}}  # 5 tokens/sec
        )
        shield = Shield(config)

        # Use burst
        v1 = shield.check_action("op1", tool="slow_tool")
        v2 = shield.check_action("op2", tool="slow_tool")
        assert v1.decision == ShieldDecision.ALLOW
        assert v2.decision == ShieldDecision.ALLOW

        # Exhausted
        v3 = shield.check_action("op3", tool="slow_tool")
        assert v3.decision == ShieldDecision.BLOCK

        # Wait for refill (0.25 sec = 1.25 tokens)
        time.sleep(0.25)

        # Should work now
        v4 = shield.check_action("op4", tool="slow_tool")
        assert v4.decision == ShieldDecision.ALLOW

    def test_empty_shield_config(self):
        """Test Shield with minimal/default config."""
        shield = Shield(ShieldConfig())

        # Should allow actions with defaults
        verdict = shield.check_action("test", cost=5.0)
        assert verdict.decision == ShieldDecision.ALLOW

        # Egress should fail (empty allowlist = deny all)
        verdict = shield.check_action("test", url="https://example.com")
        assert verdict.decision == ShieldDecision.BLOCK
