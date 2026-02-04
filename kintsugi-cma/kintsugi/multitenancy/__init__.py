"""
Multi-tenant isolation module for Kintsugi CMA.

This module provides complete multi-tenancy support including:
- Tenant models and configuration
- Data isolation strategies (row-level, schema, database)
- Resource quota management
- Tenant context management via context variables

Multi-tenancy in Kintsugi CMA is designed for nonprofit organizations
where each tenant represents a distinct organization with its own:
- Data isolation guarantees
- Resource quotas based on tier
- Custom EFE weight configurations
- Enabled skill chip sets

Key Components:
    - Tenant: The core tenant model representing an organization
    - TenantTier: Subscription tiers (SEED, SPROUT, GROVE, FOREST)
    - TenantConfig: Per-tenant configuration options
    - TenantIsolator: Manages data isolation strategies
    - QuotaManager: Tracks and enforces resource quotas
    - TenantContext: Context manager for tenant-scoped operations

Example:
    from kintsugi.multitenancy import (
        Tenant, TenantTier, TenantConfig,
        TenantContext, get_current_tenant,
        QuotaManager, TenantIsolator,
    )

    # Create a tenant configuration
    config = TenantConfig(
        tier=TenantTier.GROVE,
        max_users=100,
        max_storage_mb=1000,
    )

    # Use tenant context for scoped operations
    with TenantContext("org_12345"):
        tenant_id = get_current_tenant()
        # All operations are now scoped to this tenant
        ...
"""

from kintsugi.multitenancy.tenant import (
    Tenant,
    TenantConfig,
    TenantTier,
)
from kintsugi.multitenancy.isolation import (
    IsolationStrategy,
    TenantIsolator,
)
from kintsugi.multitenancy.quotas import (
    QuotaExceededError,
    QuotaManager,
    ResourceUsage,
)
from kintsugi.multitenancy.context import (
    TenantContext,
    get_current_tenant,
    set_current_tenant,
)

__all__ = [
    # Tenant models
    "Tenant",
    "TenantConfig",
    "TenantTier",
    # Isolation
    "IsolationStrategy",
    "TenantIsolator",
    # Quotas
    "QuotaExceededError",
    "QuotaManager",
    "ResourceUsage",
    # Context
    "TenantContext",
    "get_current_tenant",
    "set_current_tenant",
]
