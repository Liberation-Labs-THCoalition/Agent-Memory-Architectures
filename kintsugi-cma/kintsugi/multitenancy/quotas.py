"""
Resource quota management for multi-tenant Kintsugi CMA.

This module provides quota tracking and enforcement for tenant resources.
Quotas ensure fair resource distribution and prevent any single tenant
from monopolizing shared infrastructure.

Tracked Resources:
    - API calls (daily limit)
    - Storage (MB limit)
    - Active users (concurrent user limit)
    - Memory entries (total memories in CMA)
    - Concurrent sessions (active chat sessions)

Example:
    from kintsugi.multitenancy.quotas import QuotaManager, QuotaExceededError

    quota_manager = QuotaManager()

    # Check quota before operation
    if await quota_manager.check_quota("org_12345", "api_calls"):
        # Perform operation
        await quota_manager.consume("org_12345", "api_calls")
    else:
        raise QuotaExceededError(...)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import asyncio
import logging

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their resource quota.

    This exception is raised when a tenant attempts to consume
    more resources than their quota allows. The application should
    catch this and return an appropriate error to the user.

    Attributes:
        tenant_id: The tenant that exceeded the quota.
        resource: The resource type that was exceeded.
        limit: The quota limit for this resource.
        current: The current usage level.
        requested: The amount that was requested.

    Example:
        try:
            await quota_manager.consume("org_12345", "api_calls", 10)
        except QuotaExceededError as e:
            return f"API call limit exceeded. Limit: {e.limit}, Used: {e.current}"
    """

    def __init__(
        self,
        tenant_id: str,
        resource: str,
        limit: int,
        current: int = 0,
        requested: int = 1,
    ):
        self.tenant_id = tenant_id
        self.resource = resource
        self.limit = limit
        self.current = current
        self.requested = requested

        message = (
            f"Tenant {tenant_id} exceeded {resource} quota "
            f"(limit: {limit}, current: {current}, requested: {requested})"
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses.

        Returns:
            Dictionary with error details.
        """
        return {
            "error": "quota_exceeded",
            "tenant_id": self.tenant_id,
            "resource": self.resource,
            "limit": self.limit,
            "current": self.current,
            "requested": self.requested,
            "message": str(self),
        }


@dataclass
class ResourceUsage:
    """Tracks resource usage for a tenant.

    Contains current usage levels for all tracked resources.
    This is the primary data structure for quota management.

    Attributes:
        api_calls_today: Number of API calls made today.
        storage_used_mb: Storage used in megabytes.
        active_users: Number of currently active users.
        memory_entries: Total memory entries in CMA.
        concurrent_sessions: Current active sessions.
        last_reset: When daily quotas were last reset.
        last_updated: When usage was last updated.

    Example:
        usage = ResourceUsage(
            api_calls_today=1500,
            storage_used_mb=250.5,
            active_users=15,
        )
    """

    api_calls_today: int = 0
    storage_used_mb: float = 0.0
    active_users: int = 0
    memory_entries: int = 0
    concurrent_sessions: int = 0
    last_reset: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert usage to dictionary.

        Returns:
            Dictionary representation of usage.
        """
        return {
            "api_calls_today": self.api_calls_today,
            "storage_used_mb": self.storage_used_mb,
            "active_users": self.active_users,
            "memory_entries": self.memory_entries,
            "concurrent_sessions": self.concurrent_sessions,
            "last_reset": self.last_reset.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    def needs_daily_reset(self) -> bool:
        """Check if daily quotas need to be reset.

        Returns:
            True if the last reset was before today (UTC).
        """
        now = datetime.now(timezone.utc)
        return self.last_reset.date() < now.date()

    def reset_daily(self) -> None:
        """Reset daily quotas.

        Resets api_calls_today to 0 and updates the last_reset
        timestamp.
        """
        self.api_calls_today = 0
        self.last_reset = datetime.now(timezone.utc)
        self.last_updated = self.last_reset


@dataclass
class QuotaLimits:
    """Quota limits for a tenant.

    Defines the maximum allowed values for each tracked resource.

    Attributes:
        api_calls_per_day: Maximum API calls per day.
        storage_mb: Maximum storage in megabytes.
        max_users: Maximum number of users.
        max_memory_entries: Maximum memory entries.
        max_concurrent_sessions: Maximum concurrent sessions.
    """

    api_calls_per_day: int = 1000
    storage_mb: int = 100
    max_users: int = 10
    max_memory_entries: int = 5000
    max_concurrent_sessions: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert limits to dictionary.

        Returns:
            Dictionary representation of limits.
        """
        return {
            "api_calls_per_day": self.api_calls_per_day,
            "storage_mb": self.storage_mb,
            "max_users": self.max_users,
            "max_memory_entries": self.max_memory_entries,
            "max_concurrent_sessions": self.max_concurrent_sessions,
        }


@dataclass
class QuotaWarning:
    """Warning when quota usage is high.

    Generated when usage approaches the limit threshold.

    Attributes:
        tenant_id: The tenant approaching the limit.
        resource: The resource type.
        current: Current usage.
        limit: The quota limit.
        threshold_percent: The warning threshold (e.g., 80%).
        generated_at: When the warning was generated.
    """

    tenant_id: str
    resource: str
    current: int
    limit: int
    threshold_percent: int
    generated_at: datetime

    @property
    def usage_percent(self) -> float:
        """Calculate usage as a percentage of limit."""
        if self.limit == 0:
            return 100.0
        return (self.current / self.limit) * 100


class QuotaManager:
    """Tracks and enforces resource quotas per tenant.

    The QuotaManager is responsible for:
    - Tracking resource usage for each tenant
    - Enforcing quota limits before operations
    - Generating warnings when usage is high
    - Resetting daily quotas

    Thread Safety:
        This implementation uses asyncio locks to ensure thread-safe
        quota operations in concurrent environments.

    Example:
        manager = QuotaManager()

        # Set limits from tenant config
        await manager.set_limits("org_12345", QuotaLimits(
            api_calls_per_day=5000,
            storage_mb=1000,
        ))

        # Check and consume quota
        try:
            await manager.consume("org_12345", "api_calls")
        except QuotaExceededError:
            # Handle exceeded quota
            pass

        # Get usage report
        usage = await manager.get_usage("org_12345")
    """

    # Resource name constants
    API_CALLS = "api_calls"
    STORAGE = "storage"
    USERS = "users"
    MEMORY_ENTRIES = "memory_entries"
    SESSIONS = "sessions"

    # Warning threshold (percentage)
    WARNING_THRESHOLD = 80

    def __init__(self):
        """Initialize the quota manager."""
        self._usage: dict[str, ResourceUsage] = {}
        self._limits: dict[str, QuotaLimits] = {}
        self._warnings: list[QuotaWarning] = []
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, tenant_id: str) -> asyncio.Lock:
        """Get or create a lock for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            An asyncio lock for the tenant.
        """
        if tenant_id not in self._locks:
            self._locks[tenant_id] = asyncio.Lock()
        return self._locks[tenant_id]

    def _ensure_tenant(self, tenant_id: str) -> None:
        """Ensure tenant has usage and limit entries.

        Args:
            tenant_id: The tenant ID.
        """
        if tenant_id not in self._usage:
            self._usage[tenant_id] = ResourceUsage()
        if tenant_id not in self._limits:
            self._limits[tenant_id] = QuotaLimits()

    async def set_limits(self, tenant_id: str, limits: QuotaLimits) -> None:
        """Set quota limits for a tenant.

        Args:
            tenant_id: The tenant ID.
            limits: The quota limits to set.
        """
        async with self._get_lock(tenant_id):
            self._limits[tenant_id] = limits
            logger.info(f"Set quota limits for {tenant_id}: {limits}")

    async def get_limits(self, tenant_id: str) -> QuotaLimits:
        """Get quota limits for a tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            The quota limits for the tenant.
        """
        self._ensure_tenant(tenant_id)
        return self._limits[tenant_id]

    async def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """Check if tenant has quota available.

        Checks if the tenant can consume the specified amount of a
        resource without exceeding their quota. Does not actually
        consume the quota.

        Args:
            tenant_id: The tenant ID.
            resource: The resource type to check.
            amount: The amount to check availability for.

        Returns:
            True if quota is available, False otherwise.

        Example:
            if await manager.check_quota("org_12345", "api_calls", 5):
                # Safe to make 5 API calls
                pass
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]
            limits = self._limits[tenant_id]

            # Check for daily reset
            if usage.needs_daily_reset():
                usage.reset_daily()

            return self._check_resource_quota(
                resource, usage, limits, amount
            )

    def _check_resource_quota(
        self,
        resource: str,
        usage: ResourceUsage,
        limits: QuotaLimits,
        amount: int,
    ) -> bool:
        """Internal check for resource quota.

        Args:
            resource: The resource type.
            usage: Current usage.
            limits: Quota limits.
            amount: Amount to check.

        Returns:
            True if quota is available.
        """
        if resource == self.API_CALLS:
            return usage.api_calls_today + amount <= limits.api_calls_per_day
        elif resource == self.STORAGE:
            return usage.storage_used_mb + amount <= limits.storage_mb
        elif resource == self.USERS:
            return usage.active_users + amount <= limits.max_users
        elif resource == self.MEMORY_ENTRIES:
            return usage.memory_entries + amount <= limits.max_memory_entries
        elif resource == self.SESSIONS:
            return usage.concurrent_sessions + amount <= limits.max_concurrent_sessions
        else:
            logger.warning(f"Unknown resource type: {resource}")
            return True

    async def consume(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> bool:
        """Consume quota, returning False if exceeded.

        Attempts to consume the specified amount of a resource.
        If successful, updates usage tracking. If quota would be
        exceeded, returns False without consuming.

        Args:
            tenant_id: The tenant ID.
            resource: The resource type to consume.
            amount: The amount to consume.

        Returns:
            True if quota was consumed, False if exceeded.

        Raises:
            QuotaExceededError: If consume_or_raise style is preferred.

        Example:
            if not await manager.consume("org_12345", "api_calls"):
                return error_response("Rate limit exceeded")
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]
            limits = self._limits[tenant_id]

            # Check for daily reset
            if usage.needs_daily_reset():
                usage.reset_daily()

            # Check quota
            if not self._check_resource_quota(resource, usage, limits, amount):
                return False

            # Consume quota
            self._consume_resource(resource, usage, amount)
            usage.last_updated = datetime.now(timezone.utc)

            # Check for warning threshold
            self._check_warning_threshold(tenant_id, resource, usage, limits)

            return True

    def _consume_resource(
        self,
        resource: str,
        usage: ResourceUsage,
        amount: int,
    ) -> None:
        """Internal method to update usage.

        Args:
            resource: The resource type.
            usage: Usage object to update.
            amount: Amount to add.
        """
        if resource == self.API_CALLS:
            usage.api_calls_today += amount
        elif resource == self.STORAGE:
            usage.storage_used_mb += amount
        elif resource == self.USERS:
            usage.active_users += amount
        elif resource == self.MEMORY_ENTRIES:
            usage.memory_entries += amount
        elif resource == self.SESSIONS:
            usage.concurrent_sessions += amount

    async def consume_or_raise(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> None:
        """Consume quota or raise QuotaExceededError.

        Similar to consume() but raises an exception on failure
        instead of returning False.

        Args:
            tenant_id: The tenant ID.
            resource: The resource type to consume.
            amount: The amount to consume.

        Raises:
            QuotaExceededError: If quota would be exceeded.
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]
            limits = self._limits[tenant_id]

            if usage.needs_daily_reset():
                usage.reset_daily()

            current, limit = self._get_current_and_limit(resource, usage, limits)

            if current + amount > limit:
                raise QuotaExceededError(
                    tenant_id=tenant_id,
                    resource=resource,
                    limit=limit,
                    current=current,
                    requested=amount,
                )

            self._consume_resource(resource, usage, amount)
            usage.last_updated = datetime.now(timezone.utc)

    def _get_current_and_limit(
        self,
        resource: str,
        usage: ResourceUsage,
        limits: QuotaLimits,
    ) -> tuple[int, int]:
        """Get current usage and limit for a resource.

        Args:
            resource: The resource type.
            usage: Current usage.
            limits: Quota limits.

        Returns:
            Tuple of (current_usage, limit).
        """
        if resource == self.API_CALLS:
            return usage.api_calls_today, limits.api_calls_per_day
        elif resource == self.STORAGE:
            return int(usage.storage_used_mb), limits.storage_mb
        elif resource == self.USERS:
            return usage.active_users, limits.max_users
        elif resource == self.MEMORY_ENTRIES:
            return usage.memory_entries, limits.max_memory_entries
        elif resource == self.SESSIONS:
            return usage.concurrent_sessions, limits.max_concurrent_sessions
        else:
            return 0, 999999

    def _check_warning_threshold(
        self,
        tenant_id: str,
        resource: str,
        usage: ResourceUsage,
        limits: QuotaLimits,
    ) -> None:
        """Check if usage exceeds warning threshold.

        Args:
            tenant_id: The tenant ID.
            resource: The resource type.
            usage: Current usage.
            limits: Quota limits.
        """
        current, limit = self._get_current_and_limit(resource, usage, limits)

        if limit == 0:
            return

        percent = (current / limit) * 100

        if percent >= self.WARNING_THRESHOLD:
            warning = QuotaWarning(
                tenant_id=tenant_id,
                resource=resource,
                current=current,
                limit=limit,
                threshold_percent=self.WARNING_THRESHOLD,
                generated_at=datetime.now(timezone.utc),
            )
            self._warnings.append(warning)
            logger.warning(
                f"Quota warning for {tenant_id}: {resource} at {percent:.1f}%"
            )

    async def release(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1,
    ) -> None:
        """Release consumed quota.

        Used for resources that can be released, like concurrent
        sessions or active users.

        Args:
            tenant_id: The tenant ID.
            resource: The resource type to release.
            amount: The amount to release.
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]

            if resource == self.USERS:
                usage.active_users = max(0, usage.active_users - amount)
            elif resource == self.SESSIONS:
                usage.concurrent_sessions = max(0, usage.concurrent_sessions - amount)
            elif resource == self.STORAGE:
                usage.storage_used_mb = max(0, usage.storage_used_mb - amount)
            elif resource == self.MEMORY_ENTRIES:
                usage.memory_entries = max(0, usage.memory_entries - amount)

            usage.last_updated = datetime.now(timezone.utc)

    async def get_usage(self, tenant_id: str) -> ResourceUsage:
        """Get current usage for tenant.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Current resource usage for the tenant.
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]

            # Check for daily reset
            if usage.needs_daily_reset():
                usage.reset_daily()

            return usage

    async def get_usage_report(self, tenant_id: str) -> dict[str, Any]:
        """Get a detailed usage report for a tenant.

        Returns current usage, limits, and percentage used for
        all resources.

        Args:
            tenant_id: The tenant ID.

        Returns:
            Dictionary with usage report.
        """
        usage = await self.get_usage(tenant_id)
        limits = await self.get_limits(tenant_id)

        def calc_percent(current: float, limit: int) -> float:
            if limit == 0:
                return 0.0
            return min(100.0, (current / limit) * 100)

        return {
            "tenant_id": tenant_id,
            "resources": {
                "api_calls": {
                    "current": usage.api_calls_today,
                    "limit": limits.api_calls_per_day,
                    "percent": calc_percent(
                        usage.api_calls_today, limits.api_calls_per_day
                    ),
                },
                "storage_mb": {
                    "current": usage.storage_used_mb,
                    "limit": limits.storage_mb,
                    "percent": calc_percent(
                        usage.storage_used_mb, limits.storage_mb
                    ),
                },
                "users": {
                    "current": usage.active_users,
                    "limit": limits.max_users,
                    "percent": calc_percent(
                        usage.active_users, limits.max_users
                    ),
                },
                "memory_entries": {
                    "current": usage.memory_entries,
                    "limit": limits.max_memory_entries,
                    "percent": calc_percent(
                        usage.memory_entries, limits.max_memory_entries
                    ),
                },
                "sessions": {
                    "current": usage.concurrent_sessions,
                    "limit": limits.max_concurrent_sessions,
                    "percent": calc_percent(
                        usage.concurrent_sessions, limits.max_concurrent_sessions
                    ),
                },
            },
            "last_reset": usage.last_reset.isoformat(),
            "last_updated": usage.last_updated.isoformat(),
        }

    async def reset_daily_quotas(self) -> int:
        """Reset daily quotas for all tenants.

        Should be called by a scheduled task at midnight UTC.

        Returns:
            Number of tenants that had quotas reset.
        """
        count = 0
        for tenant_id in list(self._usage.keys()):
            async with self._get_lock(tenant_id):
                usage = self._usage[tenant_id]
                if usage.needs_daily_reset():
                    usage.reset_daily()
                    count += 1
                    logger.debug(f"Reset daily quota for {tenant_id}")

        logger.info(f"Reset daily quotas for {count} tenants")
        return count

    def get_warnings(
        self,
        tenant_id: str | None = None,
        since: datetime | None = None,
    ) -> list[QuotaWarning]:
        """Get quota warnings.

        Args:
            tenant_id: Optional filter by tenant.
            since: Optional filter by time.

        Returns:
            List of quota warnings.
        """
        warnings = self._warnings

        if tenant_id:
            warnings = [w for w in warnings if w.tenant_id == tenant_id]

        if since:
            warnings = [w for w in warnings if w.generated_at >= since]

        return warnings

    def clear_warnings(self, tenant_id: str | None = None) -> int:
        """Clear quota warnings.

        Args:
            tenant_id: Optional filter by tenant.

        Returns:
            Number of warnings cleared.
        """
        if tenant_id:
            original = len(self._warnings)
            self._warnings = [
                w for w in self._warnings if w.tenant_id != tenant_id
            ]
            return original - len(self._warnings)
        else:
            count = len(self._warnings)
            self._warnings = []
            return count

    async def sync_usage_from_db(
        self,
        tenant_id: str,
        usage_data: dict[str, Any],
    ) -> None:
        """Sync usage from database.

        Used to initialize usage tracking from persisted data.

        Args:
            tenant_id: The tenant ID.
            usage_data: Dictionary with usage values.
        """
        async with self._get_lock(tenant_id):
            self._ensure_tenant(tenant_id)
            usage = self._usage[tenant_id]

            if "api_calls_today" in usage_data:
                usage.api_calls_today = usage_data["api_calls_today"]
            if "storage_used_mb" in usage_data:
                usage.storage_used_mb = usage_data["storage_used_mb"]
            if "active_users" in usage_data:
                usage.active_users = usage_data["active_users"]
            if "memory_entries" in usage_data:
                usage.memory_entries = usage_data["memory_entries"]
            if "concurrent_sessions" in usage_data:
                usage.concurrent_sessions = usage_data["concurrent_sessions"]

            logger.debug(f"Synced usage for {tenant_id} from database")

    def __repr__(self) -> str:
        return f"<QuotaManager tenants={len(self._usage)} warnings={len(self._warnings)}>"
