"""
DM Pairing System for Kintsugi CMA.

This module provides the critical security component for pairing platform users
to organizations via cryptographically secure pairing codes. This enables secure
DM-based interactions without exposing organization data to unauthorized users.

Security considerations:
- Pairing codes use cryptographic randomness (secrets module)
- Rate limiting prevents brute-force attacks
- Codes expire after a configurable time window
- Admin approval required by default
- Allowlist revocation supported
"""

import secrets
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .base import AdapterPlatform


class PairingStatus(str, Enum):
    """Status of a pairing request."""

    PENDING = "pending"      # Awaiting admin approval
    APPROVED = "approved"    # Admin approved, user added to allowlist
    REJECTED = "rejected"    # Admin rejected the request
    EXPIRED = "expired"      # Code expired before approval
    REVOKED = "revoked"      # Previously approved, now revoked


@dataclass
class PairingCode:
    """
    A pairing code linking a platform user to an organization.

    This represents a single pairing attempt. Codes are short-lived
    and require admin approval before granting access.

    Attributes:
        code: 6-character alphanumeric code (crypto-random).
        platform: The platform the user is connecting from.
        platform_user_id: The user's ID on that platform.
        platform_channel_id: Optional channel where request originated.
        org_id: The organization the user wants to pair with.
        status: Current status of the pairing request.
        created_at: When the code was generated.
        expires_at: When the code becomes invalid.
        approved_at: When an admin approved (if approved).
        approved_by: User ID of approving admin (if approved).
        revoked_at: When access was revoked (if revoked).
        revoked_by: User ID who revoked access (if revoked).
        rejection_reason: Reason for rejection (if rejected).
        metadata: Additional platform-specific data.
    """

    code: str
    platform: AdapterPlatform
    platform_user_id: str
    platform_channel_id: str | None
    org_id: str
    status: PairingStatus
    created_at: datetime
    expires_at: datetime
    approved_at: datetime | None = None
    approved_by: str | None = None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if code is valid for use (pending and not expired)."""
        now = datetime.now(timezone.utc)
        return self.status == PairingStatus.PENDING and self.expires_at > now

    @property
    def is_expired(self) -> bool:
        """Check if code has expired based on time."""
        now = datetime.now(timezone.utc)
        return self.expires_at <= now


@dataclass
class PairingConfig:
    """
    Configuration for the pairing system.

    Attributes:
        code_length: Length of generated pairing codes (default: 6).
        expiration_minutes: Minutes until code expires (default: 15).
        max_attempts_per_hour: Rate limit for code generation (default: 5).
        require_admin_approval: Whether admin must approve (default: True).
        allowed_characters: Character set for code generation.
    """

    code_length: int = 6
    expiration_minutes: int = 15
    max_attempts_per_hour: int = 5
    require_admin_approval: bool = True
    allowed_characters: str = string.ascii_uppercase + string.digits

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.code_length < 4:
            raise ValueError("code_length must be at least 4 for security")
        if self.expiration_minutes < 1:
            raise ValueError("expiration_minutes must be at least 1")
        if self.max_attempts_per_hour < 1:
            raise ValueError("max_attempts_per_hour must be at least 1")


class PairingError(Exception):
    """Base exception for pairing-related errors."""
    pass


class RateLimitExceeded(PairingError):
    """Raised when user exceeds pairing attempt rate limit."""

    def __init__(self, user_id: str, retry_after_seconds: int):
        self.user_id = user_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for {user_id}. Retry after {retry_after_seconds}s"
        )


class CodeNotFound(PairingError):
    """Raised when a pairing code doesn't exist."""
    pass


class CodeExpired(PairingError):
    """Raised when attempting to use an expired code."""
    pass


class CodeAlreadyUsed(PairingError):
    """Raised when attempting to use a non-pending code."""
    pass


class PairingManager:
    """
    Manages DM pairing codes for secure user verification.

    This class handles the complete lifecycle of pairing codes:
    - Generation with cryptographic randomness
    - Validation and expiration
    - Admin approval/rejection workflow
    - Allowlist management
    - Rate limiting

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        use appropriate synchronization or a thread-safe store.

    Example:
        manager = PairingManager()

        # User requests pairing
        code = manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_abc"
        )
        print(f"Your pairing code is: {code.code}")

        # Admin approves
        manager.approve(code.code, approver="admin_user")

        # Check if user is allowed
        if manager.is_allowed("org_abc", "U12345"):
            # Process user's message
            ...
    """

    def __init__(self, config: PairingConfig | None = None):
        """
        Initialize the pairing manager.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self._config = config or PairingConfig()
        self._codes: dict[str, PairingCode] = {}
        self._allowlist: dict[str, set[str]] = {}  # org_id -> set of platform_user_ids
        self._attempts: dict[str, list[datetime]] = {}  # platform_user_id -> attempt times

    def generate_code(
        self,
        platform: AdapterPlatform,
        platform_user_id: str,
        org_id: str,
        channel_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> PairingCode:
        """
        Generate a cryptographically secure pairing code.

        Args:
            platform: The platform the user is connecting from.
            platform_user_id: The user's platform-specific ID.
            org_id: The organization to pair with.
            channel_id: Optional channel where request originated.
            metadata: Optional platform-specific metadata.

        Returns:
            A new PairingCode instance.

        Raises:
            RateLimitExceeded: If user has made too many attempts.
        """
        # Check rate limit
        if not self._check_rate_limit(platform_user_id):
            # Calculate retry time
            attempts = self._attempts.get(platform_user_id, [])
            if attempts:
                oldest_relevant = min(attempts)
                retry_after = int((oldest_relevant + timedelta(hours=1) -
                                   datetime.now(timezone.utc)).total_seconds())
                retry_after = max(1, retry_after)
            else:
                retry_after = 3600
            raise RateLimitExceeded(platform_user_id, retry_after)

        # Record this attempt
        self._record_attempt(platform_user_id)

        # Generate cryptographically secure code
        code = self._generate_secure_code()

        # Ensure uniqueness (extremely unlikely to collide, but be safe)
        while code in self._codes:
            code = self._generate_secure_code()

        now = datetime.now(timezone.utc)
        pairing_code = PairingCode(
            code=code,
            platform=platform,
            platform_user_id=platform_user_id,
            platform_channel_id=channel_id,
            org_id=org_id,
            status=PairingStatus.PENDING,
            created_at=now,
            expires_at=now + timedelta(minutes=self._config.expiration_minutes),
            metadata=metadata or {}
        )

        self._codes[code] = pairing_code
        return pairing_code

    def validate_code(self, code: str) -> PairingCode | None:
        """
        Check if a code is valid for use.

        A valid code exists, is not expired, and has PENDING status.

        Args:
            code: The pairing code to validate.

        Returns:
            The PairingCode if valid, None otherwise.
        """
        code = code.upper().strip()
        pairing_code = self._codes.get(code)

        if pairing_code is None:
            return None

        # Check expiration and update status if needed
        if pairing_code.is_expired and pairing_code.status == PairingStatus.PENDING:
            pairing_code.status = PairingStatus.EXPIRED
            return None

        if pairing_code.status != PairingStatus.PENDING:
            return None

        return pairing_code

    def approve(self, code: str, approver: str) -> PairingCode:
        """
        Admin approves a pairing request.

        This adds the user to the organization's allowlist.

        Args:
            code: The pairing code to approve.
            approver: User ID of the approving admin.

        Returns:
            The updated PairingCode.

        Raises:
            CodeNotFound: If the code doesn't exist.
            CodeExpired: If the code has expired.
            CodeAlreadyUsed: If the code isn't pending.
        """
        code = code.upper().strip()
        pairing_code = self._codes.get(code)

        if pairing_code is None:
            raise CodeNotFound(f"Pairing code '{code}' not found")

        if pairing_code.is_expired:
            pairing_code.status = PairingStatus.EXPIRED
            raise CodeExpired(f"Pairing code '{code}' has expired")

        if pairing_code.status != PairingStatus.PENDING:
            raise CodeAlreadyUsed(
                f"Pairing code '{code}' has status {pairing_code.status.value}"
            )

        # Update code status
        now = datetime.now(timezone.utc)
        pairing_code.status = PairingStatus.APPROVED
        pairing_code.approved_at = now
        pairing_code.approved_by = approver

        # Add to allowlist
        if pairing_code.org_id not in self._allowlist:
            self._allowlist[pairing_code.org_id] = set()
        self._allowlist[pairing_code.org_id].add(pairing_code.platform_user_id)

        return pairing_code

    def reject(self, code: str, rejector: str, reason: str = "") -> PairingCode:
        """
        Admin rejects a pairing request.

        Args:
            code: The pairing code to reject.
            rejector: User ID of the rejecting admin.
            reason: Optional reason for rejection.

        Returns:
            The updated PairingCode.

        Raises:
            CodeNotFound: If the code doesn't exist.
            CodeAlreadyUsed: If the code isn't pending.
        """
        code = code.upper().strip()
        pairing_code = self._codes.get(code)

        if pairing_code is None:
            raise CodeNotFound(f"Pairing code '{code}' not found")

        if pairing_code.status != PairingStatus.PENDING:
            raise CodeAlreadyUsed(
                f"Pairing code '{code}' has status {pairing_code.status.value}"
            )

        pairing_code.status = PairingStatus.REJECTED
        pairing_code.rejection_reason = reason
        pairing_code.metadata["rejected_by"] = rejector
        pairing_code.metadata["rejected_at"] = datetime.now(timezone.utc).isoformat()

        return pairing_code

    def revoke(self, org_id: str, platform_user_id: str, revoker: str) -> bool:
        """
        Remove a user from an organization's allowlist.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to revoke.
            revoker: User ID of the person revoking access.

        Returns:
            True if the user was removed, False if they weren't on the list.
        """
        if org_id not in self._allowlist:
            return False

        if platform_user_id not in self._allowlist[org_id]:
            return False

        self._allowlist[org_id].discard(platform_user_id)

        # Update any approved codes for this user/org to revoked status
        for pairing_code in self._codes.values():
            if (pairing_code.org_id == org_id and
                pairing_code.platform_user_id == platform_user_id and
                pairing_code.status == PairingStatus.APPROVED):
                pairing_code.status = PairingStatus.REVOKED
                pairing_code.revoked_at = datetime.now(timezone.utc)
                pairing_code.revoked_by = revoker

        return True

    def is_allowed(self, org_id: str, platform_user_id: str) -> bool:
        """
        Check if a user is on an organization's allowlist.

        Args:
            org_id: The organization ID.
            platform_user_id: The user to check.

        Returns:
            True if the user is allowed, False otherwise.
        """
        if org_id not in self._allowlist:
            return False
        return platform_user_id in self._allowlist[org_id]

    def get_pending(self, org_id: str | None = None) -> list[PairingCode]:
        """
        List pending pairing requests.

        Args:
            org_id: Optional org ID to filter by. If None, returns all pending.

        Returns:
            List of pending PairingCode instances.
        """
        # First, clean up expired codes
        self.cleanup_expired()

        pending = []
        for pairing_code in self._codes.values():
            if pairing_code.status != PairingStatus.PENDING:
                continue
            if org_id is not None and pairing_code.org_id != org_id:
                continue
            pending.append(pairing_code)

        # Sort by creation time, oldest first
        pending.sort(key=lambda c: c.created_at)
        return pending

    def cleanup_expired(self) -> int:
        """
        Mark expired codes and clean up old attempt records.

        Returns:
            Number of codes marked as expired.
        """
        count = 0
        now = datetime.now(timezone.utc)

        # Mark expired codes
        for pairing_code in self._codes.values():
            if pairing_code.status == PairingStatus.PENDING and pairing_code.is_expired:
                pairing_code.status = PairingStatus.EXPIRED
                count += 1

        # Clean up old attempt records (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        for user_id in list(self._attempts.keys()):
            self._attempts[user_id] = [
                t for t in self._attempts[user_id] if t > cutoff
            ]
            if not self._attempts[user_id]:
                del self._attempts[user_id]

        return count

    def get_code(self, code: str) -> PairingCode | None:
        """
        Retrieve a pairing code by its code string.

        Args:
            code: The pairing code string.

        Returns:
            The PairingCode if found, None otherwise.
        """
        return self._codes.get(code.upper().strip())

    def get_allowlist(self, org_id: str) -> set[str]:
        """
        Get all allowed user IDs for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            Set of allowed platform user IDs.
        """
        return self._allowlist.get(org_id, set()).copy()

    def _check_rate_limit(self, platform_user_id: str) -> bool:
        """
        Check if a user has exceeded the attempt rate limit.

        Args:
            platform_user_id: The user to check.

        Returns:
            True if under the limit, False if exceeded.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        if platform_user_id not in self._attempts:
            return True

        # Count attempts in the last hour
        recent_attempts = [t for t in self._attempts[platform_user_id] if t > cutoff]
        return len(recent_attempts) < self._config.max_attempts_per_hour

    def _record_attempt(self, platform_user_id: str) -> None:
        """Record a pairing attempt for rate limiting."""
        now = datetime.now(timezone.utc)
        if platform_user_id not in self._attempts:
            self._attempts[platform_user_id] = []
        self._attempts[platform_user_id].append(now)

    def _generate_secure_code(self) -> str:
        """Generate a cryptographically secure code string."""
        return ''.join(
            secrets.choice(self._config.allowed_characters)
            for _ in range(self._config.code_length)
        )
