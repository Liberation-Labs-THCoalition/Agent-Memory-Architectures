"""Tests for kintsugi.multitenancy module - Phase 5A Multi-tenancy.

This module provides comprehensive tests for the multi-tenancy system including:
- TenantTier enum tests
- TenantConfig dataclass tests
- Tenant model tests
- IsolationStrategy and TenantIsolator tests
- QuotaManager and ResourceUsage tests
- TenantContext and context management tests
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from kintsugi.multitenancy import (
    Tenant,
    TenantConfig,
    TenantTier,
    IsolationStrategy,
    TenantIsolator,
    QuotaExceededError,
    QuotaManager,
    ResourceUsage,
    TenantContext,
    get_current_tenant,
    set_current_tenant,
)
from kintsugi.multitenancy.tenant import TIER_DEFAULTS
from kintsugi.multitenancy.context import (
    clear_current_tenant,
    require_tenant,
    with_tenant,
    tenant_required,
    TenantContextData,
    get_context_data,
    run_with_tenant,
    run_with_tenant_async,
    TenantMiddleware,
)
from kintsugi.multitenancy.quotas import QuotaLimits, QuotaWarning


# ===========================================================================
# TenantTier Tests (5 tests)
# ===========================================================================

class TestTenantTier:
    """Tests for TenantTier enum."""

    def test_seed_tier_exists(self):
        """Verify SEED tier is defined."""
        assert TenantTier.SEED is not None
        assert TenantTier.SEED.value == "seed"

    def test_sprout_tier_exists(self):
        """Verify SPROUT tier is defined."""
        assert TenantTier.SPROUT is not None
        assert TenantTier.SPROUT.value == "sprout"

    def test_grove_tier_exists(self):
        """Verify GROVE tier is defined."""
        assert TenantTier.GROVE is not None
        assert TenantTier.GROVE.value == "grove"

    def test_forest_tier_exists(self):
        """Verify FOREST tier is defined."""
        assert TenantTier.FOREST is not None
        assert TenantTier.FOREST.value == "forest"

    def test_tiers_are_strings(self):
        """Verify all tiers inherit from str."""
        for tier in TenantTier:
            assert isinstance(tier, str)
            assert isinstance(tier.value, str)

    def test_can_compare_tiers(self):
        """Verify tiers can be compared by their string values."""
        # Tiers are str enums, so they compare by value
        assert TenantTier.SEED == "seed"
        assert TenantTier.FOREST != TenantTier.SEED
        # Can compare enum members directly
        assert TenantTier.GROVE == TenantTier.GROVE


# ===========================================================================
# TenantConfig Tests (10 tests)
# ===========================================================================

class TestTenantConfig:
    """Tests for TenantConfig dataclass."""

    def test_default_creation(self):
        """Test creating config with all defaults."""
        config = TenantConfig()
        assert config.tier == TenantTier.SEED
        assert config.enabled_skill_chips == []
        assert config.custom_efe_weights is None

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = TenantConfig(
            tier=TenantTier.GROVE,
            max_users=150,
            max_storage_mb=3000,
            max_api_calls_per_day=50000,
        )
        assert config.tier == TenantTier.GROVE
        assert config.max_users == 150
        assert config.max_storage_mb == 3000
        assert config.max_api_calls_per_day == 50000

    def test_tier_defaults_applied_seed(self):
        """Verify SEED tier defaults are applied correctly."""
        config = TenantConfig.from_tier(TenantTier.SEED)
        defaults = TIER_DEFAULTS[TenantTier.SEED]
        assert config.max_users == defaults["max_users"]
        assert config.max_storage_mb == defaults["max_storage_mb"]
        assert config.max_api_calls_per_day == defaults["max_api_calls_per_day"]

    def test_tier_defaults_applied_forest(self):
        """Verify FOREST tier defaults are applied correctly."""
        config = TenantConfig.from_tier(TenantTier.FOREST)
        defaults = TIER_DEFAULTS[TenantTier.FOREST]
        assert config.max_users == defaults["max_users"]
        assert config.max_storage_mb == defaults["max_storage_mb"]

    def test_enabled_skill_chips_default_empty(self):
        """Verify enabled_skill_chips defaults to empty list."""
        config = TenantConfig()
        assert config.enabled_skill_chips == []
        # Empty list means all chips enabled
        assert config.is_skill_chip_enabled("any_chip") is True

    def test_enabled_skill_chips_restricts(self):
        """Verify skill chips can be restricted."""
        config = TenantConfig(enabled_skill_chips=["grant_search", "donor_stewardship"])
        assert config.is_skill_chip_enabled("grant_search") is True
        assert config.is_skill_chip_enabled("donor_stewardship") is True
        assert config.is_skill_chip_enabled("other_chip") is False

    def test_custom_efe_weights_optional(self):
        """Verify custom_efe_weights is optional."""
        config = TenantConfig()
        assert config.custom_efe_weights is None
        # get_efe_weight returns default when not set
        assert config.get_efe_weight("epistemic", default=0.3) == 0.3

    def test_custom_efe_weights_set(self):
        """Verify custom EFE weights can be set."""
        config = TenantConfig(
            custom_efe_weights={"epistemic": 0.5, "pragmatic": 0.3}
        )
        assert config.get_efe_weight("epistemic") == 0.5
        assert config.get_efe_weight("pragmatic") == 0.3
        assert config.get_efe_weight("missing", default=0.2) == 0.2

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TenantConfig(tier=TenantTier.GROVE, max_users=100)
        d = config.to_dict()
        assert d["tier"] == "grove"
        assert d["max_users"] == 100
        assert "enabled_skill_chips" in d

    def test_has_feature(self):
        """Test feature flag checking."""
        config = TenantConfig(features={"beta_features": True, "analytics": False})
        assert config.has_feature("beta_features") is True
        assert config.has_feature("analytics") is False
        assert config.has_feature("missing") is False


# ===========================================================================
# Tenant Tests (12 tests)
# ===========================================================================

class TestTenant:
    """Tests for Tenant model."""

    def test_creation_with_required_fields(self):
        """Test creating tenant with required fields."""
        config = TenantConfig(tier=TenantTier.SEED)
        tenant = Tenant(
            id="org_12345",
            name="Test Organization",
            config=config,
            created_at=datetime.now(timezone.utc),
        )
        assert tenant.id == "org_12345"
        assert tenant.name == "Test Organization"
        assert tenant.config.tier == TenantTier.SEED

    def test_is_active_default_true(self):
        """Verify is_active defaults to True."""
        tenant = Tenant.create("org_test", "Test Org")
        assert tenant.is_active is True

    def test_schema_name_optional(self):
        """Verify schema_name is optional (defaults to None)."""
        tenant = Tenant.create("org_test", "Test Org")
        assert tenant.schema_name is None

    def test_metadata_defaults_empty(self):
        """Verify metadata defaults to empty dict."""
        tenant = Tenant.create("org_test", "Test Org")
        assert tenant.metadata == {}

    def test_suspend_sets_is_active_false(self):
        """Test suspend() sets is_active to False."""
        tenant = Tenant.create("org_test", "Test Org")
        assert tenant.is_active is True

        tenant.suspend("Non-payment")

        assert tenant.is_active is False
        assert tenant.suspension_reason == "Non-payment"
        assert tenant.suspended_at is not None

    def test_reactivate_sets_is_active_true(self):
        """Test reactivate() sets is_active back to True."""
        tenant = Tenant.create("org_test", "Test Org")
        tenant.suspend("Testing")
        assert tenant.is_active is False

        tenant.reactivate()

        assert tenant.is_active is True
        assert tenant.suspended_at is None
        assert tenant.suspension_reason is None

    def test_upgrade_tier_changes_tier_and_config(self):
        """Test upgrade_tier() changes tier and applies new limits."""
        tenant = Tenant.create("org_test", "Test Org", tier=TenantTier.SEED)
        old_users = tenant.config.max_users

        tenant.upgrade_tier(TenantTier.GROVE)

        assert tenant.config.tier == TenantTier.GROVE
        assert tenant.config.max_users >= old_users
        assert "tier_history" in tenant.metadata

    def test_tenant_id_must_start_with_org(self):
        """Verify tenant id must start with 'org_'."""
        with pytest.raises(ValueError, match="must start with 'org_'"):
            Tenant(
                id="invalid_123",
                name="Test",
                config=TenantConfig(),
                created_at=datetime.now(timezone.utc),
            )

    def test_tenant_id_cannot_be_empty(self):
        """Verify tenant id cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Tenant(
                id="",
                name="Test",
                config=TenantConfig(),
                created_at=datetime.now(timezone.utc),
            )

    def test_tenant_name_cannot_be_empty(self):
        """Verify tenant name cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Tenant(
                id="org_123",
                name="",
                config=TenantConfig(),
                created_at=datetime.now(timezone.utc),
            )

    def test_create_factory_method(self):
        """Test Tenant.create() factory method."""
        tenant = Tenant.create(
            "org_factory_test",
            "Factory Test Org",
            tier=TenantTier.SPROUT,
            metadata={"region": "west"},
        )
        assert tenant.id == "org_factory_test"
        assert tenant.config.tier == TenantTier.SPROUT
        assert tenant.metadata["region"] == "west"

    def test_to_dict(self):
        """Test converting tenant to dictionary."""
        tenant = Tenant.create("org_test", "Test Org")
        d = tenant.to_dict()
        assert d["id"] == "org_test"
        assert d["name"] == "Test Org"
        assert "config" in d
        assert d["is_active"] is True


# ===========================================================================
# IsolationStrategy Tests (3 tests)
# ===========================================================================

class TestIsolationStrategy:
    """Tests for IsolationStrategy enum."""

    def test_row_level_strategy_exists(self):
        """Verify ROW_LEVEL strategy is defined."""
        assert IsolationStrategy.ROW_LEVEL is not None
        assert IsolationStrategy.ROW_LEVEL.value == "row_level"

    def test_schema_strategy_exists(self):
        """Verify SCHEMA strategy is defined."""
        assert IsolationStrategy.SCHEMA is not None
        assert IsolationStrategy.SCHEMA.value == "schema"

    def test_database_strategy_exists(self):
        """Verify DATABASE strategy is defined."""
        assert IsolationStrategy.DATABASE is not None
        assert IsolationStrategy.DATABASE.value == "database"

    def test_strategies_are_strings(self):
        """Verify all strategies are strings."""
        for strategy in IsolationStrategy:
            assert isinstance(strategy, str)
            assert isinstance(strategy.value, str)


# ===========================================================================
# TenantIsolator Tests (10 tests)
# ===========================================================================

class TestTenantIsolator:
    """Tests for TenantIsolator class."""

    def test_creation_with_default_strategy(self):
        """Test creating isolator with default (ROW_LEVEL) strategy."""
        isolator = TenantIsolator()
        assert isolator.strategy == IsolationStrategy.ROW_LEVEL

    def test_creation_with_custom_strategy(self):
        """Test creating isolator with custom strategy."""
        isolator = TenantIsolator(strategy=IsolationStrategy.SCHEMA)
        assert isolator.strategy == IsolationStrategy.SCHEMA

    def test_get_tenant_filter_returns_dict(self):
        """Test get_tenant_filter() returns a filter dictionary."""
        isolator = TenantIsolator()
        filter_dict = isolator.get_tenant_filter("org_12345")
        assert isinstance(filter_dict, dict)
        assert filter_dict.get("tenant_id") == "org_12345"

    def test_get_tenant_filter_database_strategy(self):
        """Test get_tenant_filter() for DATABASE strategy returns empty dict."""
        isolator = TenantIsolator(strategy=IsolationStrategy.DATABASE)
        filter_dict = isolator.get_tenant_filter("org_12345")
        assert filter_dict == {}

    @pytest.mark.asyncio
    async def test_ensure_isolation_called(self):
        """Test ensure_isolation() can be called."""
        isolator = TenantIsolator()
        await isolator.ensure_isolation("org_test")
        # Should not raise and tenant should be initialized
        result = await isolator.verify_isolation("org_test")
        assert result is True

    @pytest.mark.asyncio
    async def test_create_tenant_schema_returns_schema_name(self):
        """Test create_tenant_schema() returns a schema name."""
        isolator = TenantIsolator(strategy=IsolationStrategy.SCHEMA)
        schema_name = await isolator.create_tenant_schema("org_test123")
        assert isinstance(schema_name, str)
        assert "tenant" in schema_name.lower()

    def test_rls_policy_validation_invalid_tenant(self):
        """Test validation rejects invalid tenant IDs."""
        isolator = TenantIsolator()
        with pytest.raises(ValueError, match="cannot be empty"):
            isolator.get_tenant_filter("")

    def test_rls_policy_validation_no_org_prefix(self):
        """Test validation rejects tenant IDs without org_ prefix."""
        isolator = TenantIsolator()
        with pytest.raises(ValueError, match="must start with 'org_'"):
            isolator.get_tenant_filter("invalid_id")

    def test_rls_policy_validation_special_chars(self):
        """Test validation rejects tenant IDs with special characters."""
        isolator = TenantIsolator()
        with pytest.raises(ValueError, match="invalid characters"):
            isolator.get_tenant_filter("org_test;DROP TABLE")

    @pytest.mark.asyncio
    async def test_isolator_audit_log(self):
        """Test isolator maintains audit log."""
        isolator = TenantIsolator()
        await isolator.ensure_isolation("org_audit_test")

        audit_log = isolator.get_audit_log("org_audit_test")
        assert len(audit_log) > 0
        assert audit_log[0].tenant_id == "org_audit_test"


# ===========================================================================
# QuotaManager Tests (15 tests)
# ===========================================================================

class TestQuotaManager:
    """Tests for QuotaManager class."""

    def test_creation(self):
        """Test creating QuotaManager."""
        manager = QuotaManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_check_quota_returns_true_under_limit(self):
        """Test check_quota() returns True when under limit."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=1000))

        result = await manager.check_quota("org_test", "api_calls", 1)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_quota_returns_false_over_limit(self):
        """Test check_quota() returns False when over limit."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=10))

        # Consume quota to exceed limit
        for _ in range(10):
            await manager.consume("org_test", "api_calls", 1)

        result = await manager.check_quota("org_test", "api_calls", 1)
        assert result is False

    @pytest.mark.asyncio
    async def test_consume_updates_usage(self):
        """Test consume() updates usage tracking."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=1000))

        await manager.consume("org_test", "api_calls", 5)

        usage = await manager.get_usage("org_test")
        assert usage.api_calls_today == 5

    @pytest.mark.asyncio
    async def test_consume_returns_false_when_exceeded(self):
        """Test consume() returns False when quota exceeded."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=5))

        result = await manager.consume("org_test", "api_calls", 10)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_usage_returns_resource_usage(self):
        """Test get_usage() returns ResourceUsage object."""
        manager = QuotaManager()

        usage = await manager.get_usage("org_test")
        assert isinstance(usage, ResourceUsage)

    @pytest.mark.asyncio
    async def test_reset_daily_quotas_resets_counters(self):
        """Test reset_daily_quotas() resets daily counters."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=1000))
        await manager.consume("org_test", "api_calls", 50)

        # Force need for reset by manipulating last_reset
        usage = await manager.get_usage("org_test")
        usage.last_reset = datetime.now(timezone.utc) - timedelta(days=2)

        count = await manager.reset_daily_quotas()

        usage = await manager.get_usage("org_test")
        assert usage.api_calls_today == 0

    def test_quota_exceeded_error_contains_details(self):
        """Test QuotaExceededError contains relevant details."""
        error = QuotaExceededError(
            tenant_id="org_test",
            resource="api_calls",
            limit=100,
            current=100,
            requested=1,
        )

        assert error.tenant_id == "org_test"
        assert error.resource == "api_calls"
        assert error.limit == 100
        assert error.current == 100
        assert error.requested == 1
        assert "org_test" in str(error)

    def test_quota_exceeded_error_to_dict(self):
        """Test QuotaExceededError.to_dict() returns error details."""
        error = QuotaExceededError("org_test", "storage", 100, 95, 10)
        d = error.to_dict()

        assert d["error"] == "quota_exceeded"
        assert d["tenant_id"] == "org_test"
        assert d["resource"] == "storage"

    @pytest.mark.asyncio
    async def test_multiple_tenants_tracked_separately(self):
        """Test different tenants are tracked separately."""
        manager = QuotaManager()
        await manager.set_limits("org_a", QuotaLimits(api_calls_per_day=1000))
        await manager.set_limits("org_b", QuotaLimits(api_calls_per_day=1000))

        await manager.consume("org_a", "api_calls", 10)
        await manager.consume("org_b", "api_calls", 20)

        usage_a = await manager.get_usage("org_a")
        usage_b = await manager.get_usage("org_b")

        assert usage_a.api_calls_today == 10
        assert usage_b.api_calls_today == 20

    @pytest.mark.asyncio
    async def test_different_resource_types_tracked(self):
        """Test different resource types are tracked separately."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(
            api_calls_per_day=1000,
            storage_mb=500,
            max_users=50,
        ))

        await manager.consume("org_test", "api_calls", 10)
        await manager.consume("org_test", "storage", 100)
        await manager.consume("org_test", "users", 5)

        usage = await manager.get_usage("org_test")
        assert usage.api_calls_today == 10
        assert usage.storage_used_mb == 100
        assert usage.active_users == 5

    @pytest.mark.asyncio
    async def test_consume_or_raise_raises_on_exceeded(self):
        """Test consume_or_raise() raises QuotaExceededError."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=5))

        with pytest.raises(QuotaExceededError) as exc_info:
            await manager.consume_or_raise("org_test", "api_calls", 10)

        assert exc_info.value.resource == "api_calls"

    @pytest.mark.asyncio
    async def test_release_decreases_usage(self):
        """Test release() decreases resource usage."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(max_users=100))

        await manager.consume("org_test", "users", 10)
        usage = await manager.get_usage("org_test")
        assert usage.active_users == 10

        await manager.release("org_test", "users", 3)
        usage = await manager.get_usage("org_test")
        assert usage.active_users == 7

    @pytest.mark.asyncio
    async def test_get_usage_report(self):
        """Test get_usage_report() returns comprehensive report."""
        manager = QuotaManager()
        await manager.set_limits("org_test", QuotaLimits(api_calls_per_day=1000))
        await manager.consume("org_test", "api_calls", 100)

        report = await manager.get_usage_report("org_test")

        assert "tenant_id" in report
        assert "resources" in report
        assert "api_calls" in report["resources"]
        assert report["resources"]["api_calls"]["current"] == 100

    def test_quota_warning_usage_percent(self):
        """Test QuotaWarning calculates usage percentage."""
        warning = QuotaWarning(
            tenant_id="org_test",
            resource="api_calls",
            current=80,
            limit=100,
            threshold_percent=80,
            generated_at=datetime.now(timezone.utc),
        )

        assert warning.usage_percent == 80.0


# ===========================================================================
# TenantContext Tests (10 tests)
# ===========================================================================

class TestTenantContext:
    """Tests for TenantContext and context management."""

    def test_get_current_tenant_returns_none_by_default(self):
        """Test get_current_tenant() returns None when not set."""
        clear_current_tenant()
        assert get_current_tenant() is None

    def test_set_current_tenant_sets_value(self):
        """Test set_current_tenant() sets the value."""
        clear_current_tenant()
        set_current_tenant("org_12345")
        assert get_current_tenant() == "org_12345"
        clear_current_tenant()

    def test_tenant_context_manager_works(self):
        """Test TenantContext as a context manager."""
        clear_current_tenant()

        with TenantContext("org_context_test"):
            assert get_current_tenant() == "org_context_test"

        # Should be cleared after exiting
        assert get_current_tenant() is None

    def test_context_manager_restores_previous_value(self):
        """Test context manager restores previous tenant on exit."""
        clear_current_tenant()
        set_current_tenant("org_outer")

        with TenantContext("org_inner"):
            assert get_current_tenant() == "org_inner"

        assert get_current_tenant() == "org_outer"
        clear_current_tenant()

    @pytest.mark.asyncio
    async def test_async_context_manager_works(self):
        """Test TenantContext as an async context manager."""
        clear_current_tenant()

        async with TenantContext("org_async_test"):
            assert get_current_tenant() == "org_async_test"

        assert get_current_tenant() is None

    def test_tenant_required_decorator_raises_without_context(self):
        """Test @tenant_required raises RuntimeError without context."""
        clear_current_tenant()

        @tenant_required
        def requires_tenant():
            return get_current_tenant()

        with pytest.raises(RuntimeError, match="No tenant set"):
            requires_tenant()

    def test_tenant_required_decorator_works_with_context(self):
        """Test @tenant_required works when context is set."""
        clear_current_tenant()

        @tenant_required
        def requires_tenant():
            return get_current_tenant()

        with TenantContext("org_decorated"):
            result = requires_tenant()
            assert result == "org_decorated"

    def test_with_tenant_decorator_sets_context(self):
        """Test @with_tenant decorator sets tenant context."""
        clear_current_tenant()

        @with_tenant("org_decorated_context")
        def decorated_function():
            return get_current_tenant()

        result = decorated_function()
        assert result == "org_decorated_context"
        # Should be cleared after
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_with_tenant_decorator_async(self):
        """Test @with_tenant decorator works with async functions."""
        clear_current_tenant()

        @with_tenant("org_async_decorated")
        async def async_decorated():
            return get_current_tenant()

        result = await async_decorated()
        assert result == "org_async_decorated"

    def test_run_with_tenant_helper(self):
        """Test run_with_tenant() convenience function."""
        clear_current_tenant()

        result = run_with_tenant("org_helper", lambda: get_current_tenant())
        assert result == "org_helper"
        assert get_current_tenant() is None


# ===========================================================================
# Additional Context Tests
# ===========================================================================

class TestTenantContextData:
    """Tests for TenantContextData and rich context."""

    def test_context_data_creation(self):
        """Test creating TenantContextData."""
        data = TenantContextData(
            tenant_id="org_test",
            user_id="user_123",
            session_id="sess_456",
        )
        assert data.tenant_id == "org_test"
        assert data.user_id == "user_123"

    def test_context_has_permission(self):
        """Test checking permissions in context."""
        data = TenantContextData(
            tenant_id="org_test",
            permissions=["read", "write"],
        )
        assert data.has_permission("read") is True
        assert data.has_permission("delete") is False

    def test_context_has_feature(self):
        """Test checking features in context."""
        data = TenantContextData(
            tenant_id="org_test",
            features={"beta": True, "legacy": False},
        )
        assert data.has_feature("beta") is True
        assert data.has_feature("legacy") is False
        assert data.has_feature("missing") is False

    def test_tenant_context_method_chaining(self):
        """Test TenantContext method chaining."""
        ctx = TenantContext("org_chain")
        ctx.set_user("user_1").set_session("sess_1").add_permission("admin")

        assert ctx.data.user_id == "user_1"
        assert ctx.data.session_id == "sess_1"
        assert "admin" in ctx.data.permissions

    def test_get_context_data_returns_data(self):
        """Test get_context_data() returns context data."""
        clear_current_tenant()

        with TenantContext("org_data_test", user_id="user_test"):
            data = get_context_data()
            assert data is not None
            assert data.tenant_id == "org_data_test"


# ===========================================================================
# ResourceUsage Tests
# ===========================================================================

class TestResourceUsage:
    """Tests for ResourceUsage dataclass."""

    def test_default_creation(self):
        """Test creating ResourceUsage with defaults."""
        usage = ResourceUsage()
        assert usage.api_calls_today == 0
        assert usage.storage_used_mb == 0.0
        assert usage.active_users == 0

    def test_needs_daily_reset_same_day(self):
        """Test needs_daily_reset() returns False on same day."""
        usage = ResourceUsage()
        assert usage.needs_daily_reset() is False

    def test_needs_daily_reset_different_day(self):
        """Test needs_daily_reset() returns True for different day."""
        usage = ResourceUsage()
        usage.last_reset = datetime.now(timezone.utc) - timedelta(days=1)
        assert usage.needs_daily_reset() is True

    def test_reset_daily(self):
        """Test reset_daily() resets appropriate counters."""
        usage = ResourceUsage()
        usage.api_calls_today = 100
        usage.storage_used_mb = 50.0  # Storage shouldn't reset

        usage.reset_daily()

        assert usage.api_calls_today == 0
        assert usage.storage_used_mb == 50.0  # Persistent, not reset

    def test_to_dict(self):
        """Test converting usage to dictionary."""
        usage = ResourceUsage(api_calls_today=50, storage_used_mb=100.5)
        d = usage.to_dict()

        assert d["api_calls_today"] == 50
        assert d["storage_used_mb"] == 100.5
        assert "last_reset" in d


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestMultitenancyIntegration:
    """Integration tests for multi-tenancy components."""

    @pytest.mark.asyncio
    async def test_quota_with_context(self):
        """Test quota management within tenant context."""
        manager = QuotaManager()
        await manager.set_limits("org_integrated", QuotaLimits(api_calls_per_day=100))

        with TenantContext("org_integrated"):
            tenant_id = get_current_tenant()
            await manager.consume(tenant_id, "api_calls", 10)
            usage = await manager.get_usage(tenant_id)
            assert usage.api_calls_today == 10

    @pytest.mark.asyncio
    async def test_isolation_with_context(self):
        """Test isolation within tenant context."""
        isolator = TenantIsolator()

        with TenantContext("org_isolated"):
            tenant_id = get_current_tenant()
            await isolator.ensure_isolation(tenant_id)
            filter_dict = isolator.get_tenant_filter(tenant_id)
            assert filter_dict["tenant_id"] == "org_isolated"

    def test_tenant_config_tier_upgrade(self):
        """Test complete tier upgrade flow."""
        tenant = Tenant.create("org_upgrade_test", "Upgrade Test", tier=TenantTier.SEED)

        assert tenant.config.tier == TenantTier.SEED
        initial_users = tenant.config.max_users

        tenant.upgrade_tier(TenantTier.GROVE)

        assert tenant.config.tier == TenantTier.GROVE
        assert tenant.config.max_users >= initial_users
        assert len(tenant.metadata.get("tier_history", [])) > 0

    @pytest.mark.asyncio
    async def test_run_with_tenant_async_helper(self):
        """Test run_with_tenant_async helper."""
        clear_current_tenant()

        async def get_tenant():
            return get_current_tenant()

        result = await run_with_tenant_async("org_async_helper", get_tenant())
        assert result == "org_async_helper"

    def test_nested_tenant_contexts(self):
        """Test nested tenant contexts work correctly."""
        clear_current_tenant()

        with TenantContext("org_outer"):
            assert get_current_tenant() == "org_outer"

            with TenantContext("org_inner"):
                assert get_current_tenant() == "org_inner"

            # Should restore outer context
            assert get_current_tenant() == "org_outer"

        # Should be cleared
        assert get_current_tenant() is None


# ===========================================================================
# TenantMiddleware Tests
# ===========================================================================

class TestTenantMiddleware:
    """Tests for TenantMiddleware ASGI middleware."""

    @pytest.mark.asyncio
    async def test_middleware_extracts_tenant_from_header(self):
        """Test middleware extracts tenant from X-Tenant-ID header."""
        app = AsyncMock()
        middleware = TenantMiddleware(app)

        scope = {
            "type": "http",
            "headers": [(b"x-tenant-id", b"org_from_header")],
        }

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            pass

        await middleware(scope, receive, send)
        app.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_passes_through_non_http(self):
        """Test middleware passes through non-HTTP requests."""
        app = AsyncMock()
        middleware = TenantMiddleware(app)

        scope = {"type": "websocket"}

        await middleware(scope, AsyncMock(), AsyncMock())
        app.assert_called_once()


# ===========================================================================
# Edge Cases and Error Handling
# ===========================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_require_tenant_raises_without_context(self):
        """Test require_tenant() raises when no tenant set."""
        clear_current_tenant()
        with pytest.raises(RuntimeError):
            require_tenant()

    def test_require_tenant_returns_tenant_when_set(self):
        """Test require_tenant() returns tenant when set."""
        with TenantContext("org_required"):
            tenant = require_tenant()
            assert tenant == "org_required"

    @pytest.mark.asyncio
    async def test_quota_manager_handles_unknown_resource(self):
        """Test quota manager handles unknown resource types gracefully."""
        manager = QuotaManager()
        result = await manager.check_quota("org_test", "unknown_resource", 1)
        # Should return True for unknown resources (permissive default)
        assert result is True

    def test_tenant_context_exception_cleanup(self):
        """Test context is cleaned up even on exception."""
        clear_current_tenant()

        try:
            with TenantContext("org_exception"):
                assert get_current_tenant() == "org_exception"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Context should still be cleaned up
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_isolator_idempotent_ensure(self):
        """Test ensure_isolation is idempotent."""
        isolator = TenantIsolator()

        await isolator.ensure_isolation("org_idempotent")
        await isolator.ensure_isolation("org_idempotent")  # Should not raise

        result = await isolator.verify_isolation("org_idempotent")
        assert result is True
