"""
Tenant context management for Kintsugi CMA.

This module provides thread-safe tenant context management using
Python's contextvars. This allows setting the current tenant at
request entry and having it automatically available throughout
the request processing without explicit passing.

Key Features:
    - Thread-safe context storage using contextvars
    - Context manager for scoped tenant operations
    - Async-safe for use with asyncio
    - Middleware integration for web frameworks

Example:
    from kintsugi.multitenancy.context import (
        TenantContext, get_current_tenant, set_current_tenant
    )

    # Using context manager
    with TenantContext("org_12345"):
        tenant_id = get_current_tenant()  # Returns "org_12345"
        # All operations in this scope are tenant-aware

    # Using direct setter
    set_current_tenant("org_12345")
    tenant_id = get_current_tenant()
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar
import asyncio
import functools
import logging

logger = logging.getLogger(__name__)


# Context variable for current tenant ID
_current_tenant: ContextVar[str | None] = ContextVar(
    'current_tenant', default=None
)

# Context variable for full tenant context (optional, for richer context)
_tenant_context_data: ContextVar[dict[str, Any]] = ContextVar(
    'tenant_context_data', default={}
)


def get_current_tenant() -> str | None:
    """Get the current tenant ID from context.

    Returns the tenant ID that was set for the current execution
    context. This is safe to call from any async task or thread
    that inherited the context.

    Returns:
        The current tenant ID, or None if not set.

    Example:
        tenant_id = get_current_tenant()
        if tenant_id:
            # Process with tenant isolation
            ...
    """
    return _current_tenant.get()


def set_current_tenant(tenant_id: str | None) -> Token[str | None]:
    """Set the current tenant ID in context.

    Sets the tenant ID for the current execution context. Returns
    a token that can be used to reset to the previous value.

    Args:
        tenant_id: The tenant ID to set, or None to clear.

    Returns:
        A Token that can be used with reset() to restore the
        previous value.

    Example:
        token = set_current_tenant("org_12345")
        try:
            # Operations with tenant context
            ...
        finally:
            _current_tenant.reset(token)
    """
    logger.debug(f"Setting current tenant to: {tenant_id}")
    return _current_tenant.set(tenant_id)


def clear_current_tenant() -> None:
    """Clear the current tenant from context.

    Convenience function to explicitly clear the tenant context.
    Equivalent to set_current_tenant(None).
    """
    _current_tenant.set(None)


def require_tenant() -> str:
    """Get current tenant or raise an error.

    Use this when a tenant context is required and its absence
    indicates a programming error.

    Returns:
        The current tenant ID.

    Raises:
        RuntimeError: If no tenant is set in context.

    Example:
        @app.route("/api/data")
        async def get_data():
            tenant_id = require_tenant()  # Raises if not set
            return await fetch_tenant_data(tenant_id)
    """
    tenant_id = get_current_tenant()
    if tenant_id is None:
        raise RuntimeError(
            "No tenant set in context. Ensure request middleware "
            "sets tenant context before accessing tenant-scoped resources."
        )
    return tenant_id


@dataclass
class TenantContextData:
    """Rich tenant context data.

    Provides additional context beyond just the tenant ID,
    including user info, request details, and feature flags.

    Attributes:
        tenant_id: The tenant identifier.
        user_id: The current user identifier.
        session_id: The current session identifier.
        request_id: Unique identifier for this request.
        entered_at: When the context was entered.
        permissions: User's permissions in this tenant.
        features: Feature flags for this tenant.
        metadata: Additional context metadata.
    """

    tenant_id: str
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    entered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    permissions: list[str] = field(default_factory=list)
    features: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission.

        Args:
            permission: The permission to check.

        Returns:
            True if the permission is granted.
        """
        return permission in self.permissions

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is enabled.

        Args:
            feature: The feature name to check.

        Returns:
            True if the feature is enabled.
        """
        return self.features.get(feature, False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the context.
        """
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "entered_at": self.entered_at.isoformat(),
            "permissions": self.permissions,
            "features": self.features,
            "metadata": self.metadata,
        }


class TenantContext:
    """Context manager for tenant-scoped operations.

    Provides a clean way to establish tenant context for a block
    of code. Can be used as both a synchronous context manager
    and an async context manager.

    The context manager ensures that:
    - Tenant context is set on entry
    - Previous context is restored on exit
    - Works correctly with nested contexts

    Attributes:
        _tenant_id: The tenant ID for this context.
        _token: Token for resetting context on exit.
        _data: Optional rich context data.

    Example:
        # Simple usage
        with TenantContext("org_12345"):
            tenant = get_current_tenant()  # "org_12345"
            await process_request()

        # Async usage
        async with TenantContext("org_12345"):
            tenant = get_current_tenant()
            await process_request()

        # With rich context
        ctx = TenantContext("org_12345")
        ctx.set_user("user_67890")
        ctx.add_permission("admin")
        with ctx:
            data = ctx.data  # Access rich context
    """

    def __init__(
        self,
        tenant_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ):
        """Initialize tenant context.

        Args:
            tenant_id: The tenant ID for this context.
            user_id: Optional user ID.
            session_id: Optional session ID.
            request_id: Optional request ID.
        """
        self._tenant_id = tenant_id
        self._token: Token[str | None] | None = None
        self._data_token: Token[dict[str, Any]] | None = None
        self._data = TenantContextData(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )

    @property
    def tenant_id(self) -> str:
        """Get the tenant ID for this context."""
        return self._tenant_id

    @property
    def data(self) -> TenantContextData:
        """Get the rich context data."""
        return self._data

    def set_user(self, user_id: str) -> "TenantContext":
        """Set the user ID in context.

        Args:
            user_id: The user identifier.

        Returns:
            Self for method chaining.
        """
        self._data.user_id = user_id
        return self

    def set_session(self, session_id: str) -> "TenantContext":
        """Set the session ID in context.

        Args:
            session_id: The session identifier.

        Returns:
            Self for method chaining.
        """
        self._data.session_id = session_id
        return self

    def add_permission(self, permission: str) -> "TenantContext":
        """Add a permission to the context.

        Args:
            permission: The permission to add.

        Returns:
            Self for method chaining.
        """
        if permission not in self._data.permissions:
            self._data.permissions.append(permission)
        return self

    def set_permissions(self, permissions: list[str]) -> "TenantContext":
        """Set all permissions for the context.

        Args:
            permissions: List of permissions.

        Returns:
            Self for method chaining.
        """
        self._data.permissions = list(permissions)
        return self

    def set_feature(self, feature: str, enabled: bool = True) -> "TenantContext":
        """Set a feature flag.

        Args:
            feature: The feature name.
            enabled: Whether the feature is enabled.

        Returns:
            Self for method chaining.
        """
        self._data.features[feature] = enabled
        return self

    def set_metadata(self, key: str, value: Any) -> "TenantContext":
        """Set metadata value.

        Args:
            key: The metadata key.
            value: The metadata value.

        Returns:
            Self for method chaining.
        """
        self._data.metadata[key] = value
        return self

    def __enter__(self) -> "TenantContext":
        """Enter the tenant context (sync).

        Sets the current tenant and stores the token for cleanup.

        Returns:
            Self for accessing context data.
        """
        self._token = _current_tenant.set(self._tenant_id)
        self._data_token = _tenant_context_data.set(self._data.to_dict())
        logger.debug(f"Entered tenant context: {self._tenant_id}")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit the tenant context (sync).

        Restores the previous tenant context.
        """
        if self._token is not None:
            _current_tenant.reset(self._token)
            self._token = None
        if self._data_token is not None:
            _tenant_context_data.reset(self._data_token)
            self._data_token = None
        logger.debug(f"Exited tenant context: {self._tenant_id}")

    async def __aenter__(self) -> "TenantContext":
        """Enter the tenant context (async).

        Returns:
            Self for accessing context data.
        """
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit the tenant context (async)."""
        self.__exit__(exc_type, exc_val, exc_tb)


def get_context_data() -> TenantContextData | None:
    """Get the full context data if available.

    Returns:
        The TenantContextData for the current context, or None
        if only tenant ID was set without rich context.
    """
    data_dict = _tenant_context_data.get()
    if not data_dict:
        tenant_id = get_current_tenant()
        if tenant_id:
            return TenantContextData(tenant_id=tenant_id)
        return None

    return TenantContextData(
        tenant_id=data_dict.get("tenant_id", ""),
        user_id=data_dict.get("user_id"),
        session_id=data_dict.get("session_id"),
        request_id=data_dict.get("request_id"),
        permissions=data_dict.get("permissions", []),
        features=data_dict.get("features", {}),
        metadata=data_dict.get("metadata", {}),
    )


# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def with_tenant(tenant_id: str) -> Callable[[F], F]:
    """Decorator to run a function with tenant context.

    Wraps a function to execute within a tenant context.

    Args:
        tenant_id: The tenant ID to use.

    Returns:
        Decorator function.

    Example:
        @with_tenant("org_12345")
        def process_data():
            tenant = get_current_tenant()  # "org_12345"
            ...

        @with_tenant("org_12345")
        async def async_process():
            tenant = get_current_tenant()  # "org_12345"
            ...
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with TenantContext(tenant_id):
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with TenantContext(tenant_id):
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator


def tenant_required(func: F) -> F:
    """Decorator to require tenant context.

    Wraps a function to ensure tenant context is set before
    execution. Raises RuntimeError if no tenant is set.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function.

    Example:
        @tenant_required
        async def get_data():
            tenant = get_current_tenant()  # Guaranteed non-None
            return await fetch_data(tenant)
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            require_tenant()  # Raises if not set
            return await func(*args, **kwargs)
        return async_wrapper  # type: ignore
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            require_tenant()  # Raises if not set
            return func(*args, **kwargs)
        return sync_wrapper  # type: ignore


class TenantMiddleware:
    """ASGI middleware for setting tenant context.

    Extracts tenant ID from request headers or path and sets
    up the tenant context for the duration of the request.

    Attributes:
        app: The ASGI application to wrap.
        header_name: Header name containing tenant ID.
        path_prefix: URL path prefix for tenant extraction.

    Example:
        from fastapi import FastAPI
        from kintsugi.multitenancy.context import TenantMiddleware

        app = FastAPI()
        app.add_middleware(TenantMiddleware, header_name="X-Tenant-ID")
    """

    def __init__(
        self,
        app: Any,
        header_name: str = "X-Tenant-ID",
        path_prefix: str | None = None,
    ):
        """Initialize the middleware.

        Args:
            app: The ASGI application to wrap.
            header_name: Header name for tenant ID.
            path_prefix: Optional path prefix for tenant extraction.
        """
        self.app = app
        self.header_name = header_name.lower()
        self.path_prefix = path_prefix

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        """Process an ASGI request.

        Args:
            scope: The ASGI scope.
            receive: The receive callable.
            send: The send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        tenant_id = self._extract_tenant(scope)

        if tenant_id:
            async with TenantContext(tenant_id):
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    def _extract_tenant(self, scope: dict[str, Any]) -> str | None:
        """Extract tenant ID from request.

        Args:
            scope: The ASGI scope.

        Returns:
            The tenant ID if found, None otherwise.
        """
        # Try headers first
        headers = dict(scope.get("headers", []))
        tenant_header = headers.get(self.header_name.encode())
        if tenant_header:
            return tenant_header.decode()

        # Try path if prefix configured
        if self.path_prefix:
            path = scope.get("path", "")
            if path.startswith(self.path_prefix):
                parts = path[len(self.path_prefix):].split("/")
                if parts and parts[0].startswith("org_"):
                    return parts[0]

        return None


# Convenience function for testing
def run_with_tenant(tenant_id: str, func: Callable[[], Any]) -> Any:
    """Run a function with tenant context.

    Convenience function for testing and scripts that need to
    run code with a specific tenant context.

    Args:
        tenant_id: The tenant ID to use.
        func: The function to run.

    Returns:
        The result of the function.

    Example:
        result = run_with_tenant("org_12345", lambda: get_current_tenant())
        assert result == "org_12345"
    """
    with TenantContext(tenant_id):
        return func()


async def run_with_tenant_async(tenant_id: str, coro: Any) -> Any:
    """Run a coroutine with tenant context.

    Async version of run_with_tenant for coroutines.

    Args:
        tenant_id: The tenant ID to use.
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.

    Example:
        result = await run_with_tenant_async(
            "org_12345",
            fetch_data()
        )
    """
    async with TenantContext(tenant_id):
        return await coro
