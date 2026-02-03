"""
Allowlist persistence abstraction for Kintsugi CMA.

This module provides storage backends for the allowlist, enabling
persistent storage of approved platform users across restarts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .base import AdapterPlatform


@dataclass
class AllowlistEntry:
    """
    A single entry in the allowlist.

    Represents a platform user who has been approved to interact
    with a specific organization.

    Attributes:
        org_id: The organization this entry belongs to.
        platform: The platform the user is on.
        platform_user_id: The user's platform-specific ID.
        added_at: When the user was added to the allowlist.
        added_by: Who approved/added the user (admin ID).
        notes: Optional notes about this entry.
        metadata: Additional data about the entry.
    """

    org_id: str
    platform: AdapterPlatform
    platform_user_id: str
    added_at: datetime
    added_by: str
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entry fields."""
        if not self.org_id:
            raise ValueError("org_id cannot be empty")
        if not self.platform_user_id:
            raise ValueError("platform_user_id cannot be empty")
        if not self.added_by:
            raise ValueError("added_by cannot be empty")

    @property
    def key(self) -> str:
        """Unique key for this entry within an org."""
        return f"{self.platform.value}:{self.platform_user_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to a dictionary for serialization."""
        return {
            "org_id": self.org_id,
            "platform": self.platform.value,
            "platform_user_id": self.platform_user_id,
            "added_at": self.added_at.isoformat(),
            "added_by": self.added_by,
            "notes": self.notes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AllowlistEntry":
        """Create an entry from a dictionary."""
        added_at = data["added_at"]
        if isinstance(added_at, str):
            added_at = datetime.fromisoformat(added_at)

        return cls(
            org_id=data["org_id"],
            platform=AdapterPlatform(data["platform"]),
            platform_user_id=data["platform_user_id"],
            added_at=added_at,
            added_by=data["added_by"],
            notes=data.get("notes"),
            metadata=data.get("metadata", {})
        )


class AllowlistStore(ABC):
    """
    Abstract base class for allowlist storage backends.

    Implementations should provide persistent storage for allowlist
    entries, enabling the system to remember approved users across
    restarts.

    All methods are async to support both synchronous and asynchronous
    storage backends (e.g., databases, file systems, remote APIs).
    """

    @abstractmethod
    async def add(self, entry: AllowlistEntry) -> None:
        """
        Add an entry to the allowlist.

        If the entry already exists (same org_id + platform_user_id),
        it should be updated.

        Args:
            entry: The allowlist entry to add.

        Raises:
            AllowlistStoreError: If the operation fails.
        """
        pass

    @abstractmethod
    async def remove(self, org_id: str, platform_user_id: str) -> bool:
        """
        Remove an entry from the allowlist.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to remove.

        Returns:
            True if the entry was removed, False if it didn't exist.

        Raises:
            AllowlistStoreError: If the operation fails.
        """
        pass

    @abstractmethod
    async def is_allowed(self, org_id: str, platform_user_id: str) -> bool:
        """
        Check if a user is on the allowlist.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to check.

        Returns:
            True if the user is allowed, False otherwise.

        Raises:
            AllowlistStoreError: If the operation fails.
        """
        pass

    @abstractmethod
    async def list_for_org(self, org_id: str) -> list[AllowlistEntry]:
        """
        List all allowlist entries for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            List of AllowlistEntry objects for the org.

        Raises:
            AllowlistStoreError: If the operation fails.
        """
        pass

    async def get(self, org_id: str, platform_user_id: str) -> AllowlistEntry | None:
        """
        Get a specific allowlist entry.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to get.

        Returns:
            The AllowlistEntry if found, None otherwise.
        """
        entries = await self.list_for_org(org_id)
        for entry in entries:
            if entry.platform_user_id == platform_user_id:
                return entry
        return None

    async def count(self, org_id: str) -> int:
        """
        Count allowlist entries for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            Number of entries for the org.
        """
        entries = await self.list_for_org(org_id)
        return len(entries)

    async def clear(self, org_id: str) -> int:
        """
        Remove all entries for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            Number of entries removed.
        """
        entries = await self.list_for_org(org_id)
        count = 0
        for entry in entries:
            if await self.remove(org_id, entry.platform_user_id):
                count += 1
        return count


class AllowlistStoreError(Exception):
    """Base exception for allowlist store errors."""
    pass


class InMemoryAllowlistStore(AllowlistStore):
    """
    In-memory implementation of AllowlistStore for testing.

    This implementation stores entries in a dictionary and is not
    persistent across restarts. Use for testing and development only.

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        use appropriate synchronization.

    Example:
        store = InMemoryAllowlistStore()

        entry = AllowlistEntry(
            org_id="org_123",
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            added_at=datetime.now(timezone.utc),
            added_by="admin"
        )

        await store.add(entry)
        assert await store.is_allowed("org_123", "U12345")
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        # Nested dict: org_id -> platform_user_id -> AllowlistEntry
        self._entries: dict[str, dict[str, AllowlistEntry]] = {}

    async def add(self, entry: AllowlistEntry) -> None:
        """
        Add an entry to the in-memory store.

        Args:
            entry: The allowlist entry to add.
        """
        if entry.org_id not in self._entries:
            self._entries[entry.org_id] = {}
        self._entries[entry.org_id][entry.platform_user_id] = entry

    async def remove(self, org_id: str, platform_user_id: str) -> bool:
        """
        Remove an entry from the in-memory store.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to remove.

        Returns:
            True if removed, False if not found.
        """
        if org_id not in self._entries:
            return False
        if platform_user_id not in self._entries[org_id]:
            return False
        del self._entries[org_id][platform_user_id]

        # Clean up empty org entries
        if not self._entries[org_id]:
            del self._entries[org_id]

        return True

    async def is_allowed(self, org_id: str, platform_user_id: str) -> bool:
        """
        Check if a user is in the in-memory store.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to check.

        Returns:
            True if allowed, False otherwise.
        """
        if org_id not in self._entries:
            return False
        return platform_user_id in self._entries[org_id]

    async def list_for_org(self, org_id: str) -> list[AllowlistEntry]:
        """
        List all entries for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            List of entries, sorted by added_at.
        """
        if org_id not in self._entries:
            return []

        entries = list(self._entries[org_id].values())
        entries.sort(key=lambda e: e.added_at)
        return entries

    async def get(self, org_id: str, platform_user_id: str) -> AllowlistEntry | None:
        """
        Get a specific entry.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to get.

        Returns:
            The entry if found, None otherwise.
        """
        if org_id not in self._entries:
            return None
        return self._entries[org_id].get(platform_user_id)

    def __len__(self) -> int:
        """Return total number of entries across all orgs."""
        return sum(len(org_entries) for org_entries in self._entries.values())

    def __repr__(self) -> str:
        org_count = len(self._entries)
        entry_count = len(self)
        return f"<InMemoryAllowlistStore orgs={org_count} entries={entry_count}>"
