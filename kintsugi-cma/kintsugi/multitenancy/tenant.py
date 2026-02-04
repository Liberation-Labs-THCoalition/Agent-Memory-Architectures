"""
Tenant model and configuration for Kintsugi CMA.

This module defines the core tenant abstractions used throughout the system.
A tenant represents an organization using Kintsugi CMA, with its own:
- Configuration and feature flags
- Resource quotas
- Data isolation
- Custom EFE weights

Tenant Tiers:
    - SEED: Free tier for evaluation, limited resources
    - SPROUT: Small organizations, basic features
    - GROVE: Full-featured tier for established nonprofits
    - FOREST: Enterprise tier with maximum resources and support

Example:
    from kintsugi.multitenancy.tenant import Tenant, TenantConfig, TenantTier

    config = TenantConfig(
        tier=TenantTier.GROVE,
        max_users=100,
        enabled_skill_chips=["grant_search", "donor_stewardship"],
    )

    tenant = Tenant(
        id="org_12345",
        name="Community Foundation",
        config=config,
        created_at=datetime.now(timezone.utc),
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TenantTier(str, Enum):
    """Subscription tiers for Kintsugi CMA tenants.

    Each tier defines resource limits and feature availability.
    Tiers are designed to scale with organization growth, using
    nature-inspired naming that reflects our mission of nurturing
    nonprofit organizations.

    Attributes:
        SEED: Free tier for evaluation. Limited resources, basic features.
              Ideal for exploring Kintsugi CMA before committing.
        SPROUT: Entry-level paid tier for small organizations.
                Moderate resources, core skill chips enabled.
        GROVE: Full-featured tier for established nonprofits.
               Higher limits, all standard skill chips, custom EFE weights.
        FOREST: Enterprise tier for large organizations.
                Maximum resources, dedicated support, schema-level isolation.
    """

    SEED = "seed"
    SPROUT = "sprout"
    GROVE = "grove"
    FOREST = "forest"


# Default resource limits per tier
TIER_DEFAULTS: dict[TenantTier, dict[str, Any]] = {
    TenantTier.SEED: {
        "max_users": 5,
        "max_storage_mb": 50,
        "max_api_calls_per_day": 500,
        "max_memory_entries": 1000,
        "max_concurrent_sessions": 2,
        "retention_days": 30,
    },
    TenantTier.SPROUT: {
        "max_users": 25,
        "max_storage_mb": 500,
        "max_api_calls_per_day": 5000,
        "max_memory_entries": 10000,
        "max_concurrent_sessions": 10,
        "retention_days": 90,
    },
    TenantTier.GROVE: {
        "max_users": 100,
        "max_storage_mb": 2000,
        "max_api_calls_per_day": 25000,
        "max_memory_entries": 100000,
        "max_concurrent_sessions": 50,
        "retention_days": 365,
    },
    TenantTier.FOREST: {
        "max_users": 1000,
        "max_storage_mb": 10000,
        "max_api_calls_per_day": 100000,
        "max_memory_entries": 1000000,
        "max_concurrent_sessions": 200,
        "retention_days": -1,  # Unlimited
    },
}


@dataclass
class TenantConfig:
    """Per-tenant configuration settings.

    Contains all configurable aspects of a tenant including resource
    limits, feature flags, and customization options. Defaults are
    based on the tenant's tier.

    Attributes:
        tier: The subscription tier determining base limits.
        max_users: Maximum number of users allowed in this tenant.
        max_storage_mb: Maximum storage allocation in megabytes.
        max_api_calls_per_day: Daily API call limit.
        max_memory_entries: Maximum memory entries in CMA.
        max_concurrent_sessions: Maximum concurrent chat sessions.
        retention_days: How long to retain data (-1 for unlimited).
        enabled_skill_chips: List of enabled skill chip names.
                            Empty list means all chips are enabled.
        custom_efe_weights: Custom EFE weights for this tenant.
                           None means use domain defaults.
        features: Feature flags for this tenant.
        integrations: Enabled integrations (slack, discord, etc.).
        branding: Custom branding options (colors, logo, etc.).

    Example:
        config = TenantConfig(
            tier=TenantTier.GROVE,
            max_users=150,  # Override tier default
            enabled_skill_chips=["grant_search", "donor_stewardship"],
            features={"beta_features": True, "advanced_analytics": True},
        )
    """

    tier: TenantTier = TenantTier.SEED
    max_users: int = 10
    max_storage_mb: int = 100
    max_api_calls_per_day: int = 1000
    max_memory_entries: int = 5000
    max_concurrent_sessions: int = 5
    retention_days: int = 90
    enabled_skill_chips: list[str] = field(default_factory=list)
    custom_efe_weights: dict[str, float] | None = None
    features: dict[str, Any] = field(default_factory=dict)
    integrations: list[str] = field(default_factory=list)
    branding: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Apply tier defaults to unset fields."""
        defaults = TIER_DEFAULTS.get(self.tier, TIER_DEFAULTS[TenantTier.SEED])

        # Only apply defaults if values are at the class default
        if self.max_users == 10:
            self.max_users = defaults["max_users"]
        if self.max_storage_mb == 100:
            self.max_storage_mb = defaults["max_storage_mb"]
        if self.max_api_calls_per_day == 1000:
            self.max_api_calls_per_day = defaults["max_api_calls_per_day"]
        if self.max_memory_entries == 5000:
            self.max_memory_entries = defaults["max_memory_entries"]
        if self.max_concurrent_sessions == 5:
            self.max_concurrent_sessions = defaults["max_concurrent_sessions"]
        if self.retention_days == 90:
            self.retention_days = defaults["retention_days"]

    @classmethod
    def from_tier(cls, tier: TenantTier) -> "TenantConfig":
        """Create a config with all tier defaults.

        Factory method to create a TenantConfig with all values
        set to the tier's defaults.

        Args:
            tier: The tier to use for defaults.

        Returns:
            A new TenantConfig with tier defaults applied.

        Example:
            config = TenantConfig.from_tier(TenantTier.GROVE)
        """
        return cls(tier=tier)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "tier": self.tier.value,
            "max_users": self.max_users,
            "max_storage_mb": self.max_storage_mb,
            "max_api_calls_per_day": self.max_api_calls_per_day,
            "max_memory_entries": self.max_memory_entries,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "retention_days": self.retention_days,
            "enabled_skill_chips": self.enabled_skill_chips,
            "custom_efe_weights": self.custom_efe_weights,
            "features": self.features,
            "integrations": self.integrations,
            "branding": self.branding,
        }

    def is_skill_chip_enabled(self, chip_name: str) -> bool:
        """Check if a skill chip is enabled for this tenant.

        Args:
            chip_name: The name of the skill chip to check.

        Returns:
            True if the chip is enabled or if no restrictions are set.
        """
        # Empty list means all chips enabled
        if not self.enabled_skill_chips:
            return True
        return chip_name in self.enabled_skill_chips

    def has_feature(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled.

        Args:
            feature_name: The name of the feature to check.

        Returns:
            True if the feature is enabled, False otherwise.
        """
        return self.features.get(feature_name, False)

    def get_efe_weight(self, weight_name: str, default: float = 0.2) -> float:
        """Get a custom EFE weight or default.

        Args:
            weight_name: The name of the weight to retrieve.
            default: Default value if not set.

        Returns:
            The custom weight value or the default.
        """
        if self.custom_efe_weights is None:
            return default
        return self.custom_efe_weights.get(weight_name, default)


@dataclass
class Tenant:
    """A tenant (organization) in the Kintsugi CMA system.

    Represents a complete tenant entity with all associated metadata,
    configuration, and state. Each tenant is a separate organization
    with isolated data and resources.

    Attributes:
        id: Unique identifier for the tenant (e.g., "org_12345").
        name: Human-readable organization name.
        config: Tenant configuration including limits and features.
        created_at: When the tenant was created (UTC).
        schema_name: For schema-per-tenant isolation, the PostgreSQL
                    schema name. None for row-level isolation.
        metadata: Additional tenant metadata (address, contact, etc.).
        is_active: Whether the tenant is active and can access the system.
        suspended_at: When the tenant was suspended, if applicable.
        suspension_reason: Why the tenant was suspended.
        last_activity_at: Last time the tenant had activity.

    Example:
        tenant = Tenant(
            id="org_12345",
            name="Community Foundation of Westchester",
            config=TenantConfig(tier=TenantTier.GROVE),
            created_at=datetime.now(timezone.utc),
            metadata={
                "contact_email": "admin@cfwestchester.org",
                "website": "https://cfwestchester.org",
            },
        )
    """

    id: str
    name: str
    config: TenantConfig
    created_at: datetime
    schema_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    suspended_at: datetime | None = None
    suspension_reason: str | None = None
    last_activity_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate tenant fields after initialization."""
        if not self.id:
            raise ValueError("Tenant id cannot be empty")
        if not self.name:
            raise ValueError("Tenant name cannot be empty")
        if not self.id.startswith("org_"):
            raise ValueError("Tenant id must start with 'org_'")

    @classmethod
    def create(
        cls,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.SEED,
        **kwargs: Any,
    ) -> "Tenant":
        """Factory method to create a new tenant.

        Creates a tenant with a fresh configuration based on the
        specified tier. Additional configuration can be passed as
        keyword arguments.

        Args:
            tenant_id: Unique identifier for the tenant.
            name: Human-readable organization name.
            tier: Subscription tier for the tenant.
            **kwargs: Additional fields to set on the tenant.

        Returns:
            A new Tenant instance.

        Example:
            tenant = Tenant.create(
                "org_12345",
                "Community Foundation",
                tier=TenantTier.GROVE,
                metadata={"region": "Northeast"},
            )
        """
        config = TenantConfig.from_tier(tier)
        return cls(
            id=tenant_id,
            name=name,
            config=config,
            created_at=datetime.now(timezone.utc),
            **kwargs,
        )

    def suspend(self, reason: str) -> None:
        """Suspend the tenant.

        Marks the tenant as inactive and records the suspension
        reason and timestamp.

        Args:
            reason: The reason for suspension.
        """
        self.is_active = False
        self.suspended_at = datetime.now(timezone.utc)
        self.suspension_reason = reason

    def reactivate(self) -> None:
        """Reactivate a suspended tenant.

        Clears suspension status and makes the tenant active again.
        """
        self.is_active = True
        self.suspended_at = None
        self.suspension_reason = None

    def update_activity(self) -> None:
        """Update the last activity timestamp to now."""
        self.last_activity_at = datetime.now(timezone.utc)

    def upgrade_tier(self, new_tier: TenantTier) -> None:
        """Upgrade the tenant to a new tier.

        Updates the configuration to apply new tier defaults while
        preserving custom settings that exceed the new defaults.

        Args:
            new_tier: The new tier to upgrade to.
        """
        old_tier = self.config.tier
        self.config.tier = new_tier

        # Apply new tier defaults
        new_defaults = TIER_DEFAULTS.get(new_tier, TIER_DEFAULTS[TenantTier.SEED])

        # Only upgrade limits, don't downgrade
        if new_defaults["max_users"] > self.config.max_users:
            self.config.max_users = new_defaults["max_users"]
        if new_defaults["max_storage_mb"] > self.config.max_storage_mb:
            self.config.max_storage_mb = new_defaults["max_storage_mb"]
        if new_defaults["max_api_calls_per_day"] > self.config.max_api_calls_per_day:
            self.config.max_api_calls_per_day = new_defaults["max_api_calls_per_day"]

        self.metadata["tier_history"] = self.metadata.get("tier_history", [])
        self.metadata["tier_history"].append({
            "from": old_tier.value,
            "to": new_tier.value,
            "at": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert tenant to a dictionary.

        Returns:
            Dictionary representation of the tenant.
        """
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "schema_name": self.schema_name,
            "metadata": self.metadata,
            "is_active": self.is_active,
            "suspended_at": self.suspended_at.isoformat() if self.suspended_at else None,
            "suspension_reason": self.suspension_reason,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
        }

    def __repr__(self) -> str:
        status = "active" if self.is_active else "suspended"
        return f"<Tenant {self.id} name={self.name!r} tier={self.config.tier.value} {status}>"
