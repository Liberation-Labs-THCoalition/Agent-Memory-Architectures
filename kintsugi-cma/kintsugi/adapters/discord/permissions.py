"""Discord role-based access control.

This module provides permission level mapping from Discord roles
to Kintsugi permission tiers, enabling fine-grained access control
for bot commands and features.
"""

from dataclasses import dataclass, field
from enum import Enum


class PermissionLevel(str, Enum):
    """Permission levels for Discord users.

    Levels are hierarchical from lowest to highest:
    NONE < USER < MODERATOR < ADMIN < OWNER
    """

    NONE = "none"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    OWNER = "owner"

    def __ge__(self, other: "PermissionLevel") -> bool:
        """Check if this level is greater than or equal to another."""
        order = [
            PermissionLevel.NONE,
            PermissionLevel.USER,
            PermissionLevel.MODERATOR,
            PermissionLevel.ADMIN,
            PermissionLevel.OWNER,
        ]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: "PermissionLevel") -> bool:
        """Check if this level is strictly greater than another."""
        order = [
            PermissionLevel.NONE,
            PermissionLevel.USER,
            PermissionLevel.MODERATOR,
            PermissionLevel.ADMIN,
            PermissionLevel.OWNER,
        ]
        return order.index(self) > order.index(other)

    def __le__(self, other: "PermissionLevel") -> bool:
        """Check if this level is less than or equal to another."""
        return not self > other

    def __lt__(self, other: "PermissionLevel") -> bool:
        """Check if this level is strictly less than another."""
        return not self >= other


@dataclass
class DiscordPermissions:
    """Maps Discord roles to permission levels.

    This class manages the mapping between Discord role IDs and
    Kintsugi permission levels, enabling role-based access control
    for bot commands and features.

    Attributes:
        admin_role_ids: Role IDs that grant admin permissions.
        moderator_role_ids: Role IDs that grant moderator permissions.
        user_role_ids: Role IDs that grant basic user permissions.
    """

    admin_role_ids: list[str] = field(default_factory=list)
    moderator_role_ids: list[str] = field(default_factory=list)
    user_role_ids: list[str] = field(default_factory=list)

    def get_level(
        self, member_role_ids: list[str], is_owner: bool = False
    ) -> PermissionLevel:
        """Determine the highest permission level for a member.

        Evaluates all roles a member has and returns the highest
        permission level granted by any of those roles.

        Args:
            member_role_ids: List of role IDs the member has.
            is_owner: Whether the member is the server owner.

        Returns:
            The highest PermissionLevel granted to the member.
        """
        if is_owner:
            return PermissionLevel.OWNER

        # Check from highest to lowest
        for role_id in member_role_ids:
            if role_id in self.admin_role_ids:
                return PermissionLevel.ADMIN

        for role_id in member_role_ids:
            if role_id in self.moderator_role_ids:
                return PermissionLevel.MODERATOR

        for role_id in member_role_ids:
            if role_id in self.user_role_ids:
                return PermissionLevel.USER

        # No matching roles found
        return PermissionLevel.NONE

    def can_approve_pairing(self, level: PermissionLevel) -> bool:
        """Check if the permission level can approve pairing requests.

        Only admins and owners can approve new user pairings.

        Args:
            level: The permission level to check.

        Returns:
            True if the level can approve pairing requests.
        """
        return level in (PermissionLevel.ADMIN, PermissionLevel.OWNER)

    def can_revoke_pairing(self, level: PermissionLevel) -> bool:
        """Check if the permission level can revoke pairings.

        Only admins and owners can revoke user pairings.

        Args:
            level: The permission level to check.

        Returns:
            True if the level can revoke pairings.
        """
        return level in (PermissionLevel.ADMIN, PermissionLevel.OWNER)

    def can_use_bot(self, level: PermissionLevel) -> bool:
        """Check if the permission level can interact with the bot.

        All levels except NONE can use basic bot features.

        Args:
            level: The permission level to check.

        Returns:
            True if the level can interact with the bot.
        """
        return level != PermissionLevel.NONE

    def can_manage_users(self, level: PermissionLevel) -> bool:
        """Check if the permission level can manage users.

        Moderators, admins, and owners can manage users.

        Args:
            level: The permission level to check.

        Returns:
            True if the level can manage users.
        """
        return level >= PermissionLevel.MODERATOR

    def can_view_all_paired(self, level: PermissionLevel) -> bool:
        """Check if the permission level can view all paired users.

        Moderators, admins, and owners can view the paired user list.

        Args:
            level: The permission level to check.

        Returns:
            True if the level can view all paired users.
        """
        return level >= PermissionLevel.MODERATOR
