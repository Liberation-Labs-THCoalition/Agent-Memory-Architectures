"""
Shared adapter infrastructure for Kintsugi CMA.

This package provides the foundational components for all platform adapters,
including base classes, message normalization, pairing system, and allowlist
management.

Public API:
    - AdapterPlatform: Enum of supported platforms
    - AdapterMessage: Normalized incoming message
    - AdapterResponse: Outgoing response structure
    - BaseAdapter: Abstract base for platform adapters

    - PairingStatus: Enum of pairing request states
    - PairingCode: A pairing request/code
    - PairingConfig: Configuration for pairing system
    - PairingManager: Manages pairing workflow
    - PairingError: Base pairing exception
    - RateLimitExceeded: Rate limit exception
    - CodeNotFound: Code not found exception
    - CodeExpired: Code expired exception
    - CodeAlreadyUsed: Code already used exception

    - AllowlistEntry: A single allowlist entry
    - AllowlistStore: Abstract storage backend
    - InMemoryAllowlistStore: In-memory implementation
    - AllowlistStoreError: Storage exception
"""

from .base import (
    AdapterPlatform,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
)

from .pairing import (
    PairingStatus,
    PairingCode,
    PairingConfig,
    PairingManager,
    PairingError,
    RateLimitExceeded,
    CodeNotFound,
    CodeExpired,
    CodeAlreadyUsed,
)

from .allowlist import (
    AllowlistEntry,
    AllowlistStore,
    InMemoryAllowlistStore,
    AllowlistStoreError,
)


__all__ = [
    # Base adapter types
    "AdapterPlatform",
    "AdapterMessage",
    "AdapterResponse",
    "BaseAdapter",

    # Pairing system
    "PairingStatus",
    "PairingCode",
    "PairingConfig",
    "PairingManager",
    "PairingError",
    "RateLimitExceeded",
    "CodeNotFound",
    "CodeExpired",
    "CodeAlreadyUsed",

    # Allowlist
    "AllowlistEntry",
    "AllowlistStore",
    "InMemoryAllowlistStore",
    "AllowlistStoreError",
]
