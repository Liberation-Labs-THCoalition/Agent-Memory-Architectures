"""Main Discord adapter implementation.

This module provides the DiscordAdapter class which integrates with the
shared adapter infrastructure to provide Discord-specific functionality
for the Kintsugi CMA system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ..shared import (
    AdapterMessage,
    AdapterPlatform,
    AdapterResponse,
    BaseAdapter,
    PairingManager,
)

from .config import DiscordConfig
from .embeds import DiscordEmbed


# Guild-to-org mapping (would typically be in a database)
_guild_org_mapping: dict[str, str] = {}


class DiscordClient(Protocol):
    """Protocol for Discord client interactions.

    This protocol defines the interface for Discord client operations,
    allowing for dependency injection and testing.
    """

    async def send_message(
        self,
        channel_id: str,
        content: str | None = None,
        embed: dict | None = None,
    ) -> dict:
        """Send a message to a channel."""
        ...

    async def send_dm(
        self,
        user_id: str,
        content: str | None = None,
        embed: dict | None = None,
    ) -> dict:
        """Send a direct message to a user."""
        ...

    async def get_member(self, guild_id: str, user_id: str) -> dict | None:
        """Get a guild member."""
        ...

    async def get_user(self, user_id: str) -> dict | None:
        """Get a user."""
        ...


@dataclass
class DiscordMember:
    """Representation of a Discord guild member.

    Attributes:
        user_id: The user's Discord ID.
        username: The user's username.
        discriminator: The user's discriminator (legacy).
        roles: List of role IDs the member has.
        guild_id: The guild ID where the member is.
        nickname: The member's nickname in the guild.
        is_owner: Whether the member is the guild owner.
    """

    user_id: str
    username: str
    discriminator: str
    roles: list[str] = field(default_factory=list)
    guild_id: str | None = None
    nickname: str | None = None
    is_owner: bool = False

    @property
    def display_name(self) -> str:
        """Get the display name (nickname or username)."""
        return self.nickname or self.username

    @classmethod
    def from_dict(cls, data: dict) -> "DiscordMember":
        """Create a DiscordMember from a dictionary.

        Args:
            data: Dictionary with member data from Discord API.

        Returns:
            A DiscordMember instance.
        """
        user = data.get("user", data)
        return cls(
            user_id=str(user.get("id", "")),
            username=user.get("username", ""),
            discriminator=user.get("discriminator", "0"),
            roles=[str(r) for r in data.get("roles", [])],
            guild_id=str(data.get("guild_id", "")) if data.get("guild_id") else None,
            nickname=data.get("nick"),
            is_owner=data.get("is_owner", False),
        )


class DiscordAdapter(BaseAdapter):
    """Discord Bot adapter using discord.py patterns.

    This adapter provides Discord-specific functionality for the Kintsugi
    CMA system, including message handling, user verification, and
    role-based access control.

    Attributes:
        platform: The adapter platform identifier.
    """

    platform = AdapterPlatform.DISCORD

    def __init__(
        self,
        config: DiscordConfig,
        pairing: PairingManager,
        client: DiscordClient | None = None,
    ) -> None:
        """Initialize the Discord adapter.

        Args:
            config: Discord configuration.
            pairing: Pairing manager for user verification.
            client: Optional Discord client for testing.
        """
        self._config = config
        self._pairing = pairing
        self._client = client
        self._started = False

    @property
    def config(self) -> DiscordConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def is_started(self) -> bool:
        """Check if the adapter has been started."""
        return self._started

    async def start(self) -> None:
        """Start the Discord adapter.

        This method should be called to initialize the Discord connection
        and begin processing events.
        """
        self._started = True

    async def stop(self) -> None:
        """Stop the Discord adapter.

        This method should be called to gracefully shut down the Discord
        connection.
        """
        self._started = False

    async def send_message(
        self,
        channel_id: str,
        response: AdapterResponse,
    ) -> str:
        """Send message to a Discord channel.

        Args:
            channel_id: The Discord channel ID to send to.
            response: The adapter response to send.

        Returns:
            The message ID of the sent message.

        Raises:
            RuntimeError: If the client is not configured.
        """
        if self._client is None:
            raise RuntimeError("Discord client not configured")

        # Build embed if response has structured content
        embed_dict = None
        if response.embed:
            embed_dict = response.embed.to_dict() if isinstance(
                response.embed, DiscordEmbed
            ) else response.embed

        result = await self._client.send_message(
            channel_id=channel_id,
            content=response.content,
            embed=embed_dict,
        )

        return str(result.get("id", ""))

    async def send_dm(self, user_id: str, response: AdapterResponse) -> str:
        """Send a direct message to a Discord user.

        Args:
            user_id: The Discord user ID to send to.
            response: The adapter response to send.

        Returns:
            The message ID of the sent message.

        Raises:
            RuntimeError: If the client is not configured.
        """
        if self._client is None:
            raise RuntimeError("Discord client not configured")

        # Build embed if response has structured content
        embed_dict = None
        if response.embed:
            embed_dict = response.embed.to_dict() if isinstance(
                response.embed, DiscordEmbed
            ) else response.embed

        result = await self._client.send_dm(
            user_id=user_id,
            content=response.content,
            embed=embed_dict,
        )

        return str(result.get("id", ""))

    async def verify_user(self, user_id: str, org_id: str | None = None) -> bool:
        """Check pairing allowlist and role permissions.

        Verifies that a user is paired with the given organization
        and has the necessary role permissions to interact.

        Args:
            user_id: The Discord user ID to verify.
            org_id: The organization ID to check against (optional).

        Returns:
            True if the user is verified and allowed to interact.
        """
        # Check if pairing is required
        if not self._config.require_pairing:
            return True

        # Use default org if not specified
        if org_id is None:
            org_id = self._config.default_org_id

        if org_id is None:
            return False

        # Check allowlist via PairingManager
        return self._pairing.is_allowed(org_id, user_id)

    def get_user_org(self, user_id: str, guild_id: str | None = None) -> str | None:
        """Get the organization ID for a paired user.

        Args:
            user_id: The Discord user ID.
            guild_id: Optional guild ID to look up org mapping.

        Returns:
            The organization ID if paired, None otherwise.
        """
        # Check guild-to-org mapping first
        if guild_id and guild_id in _guild_org_mapping:
            org_id = _guild_org_mapping[guild_id]
            if self._pairing.is_allowed(org_id, user_id):
                return org_id

        # Fall back to default org
        if self._config.default_org_id:
            if self._pairing.is_allowed(self._config.default_org_id, user_id):
                return self._config.default_org_id

        return None

    def normalize_message(self, message: dict, org_id: str | None = None) -> AdapterMessage:
        """Convert Discord message to AdapterMessage.

        Transforms a raw Discord message dictionary into the standardized
        AdapterMessage format used by the Kintsugi system.

        Args:
            message: Raw Discord message dictionary.
            org_id: Optional organization ID (will be looked up if not provided).

        Returns:
            An AdapterMessage instance.
        """
        # Extract author information
        author = message.get("author", {})
        user_id = str(author.get("id", ""))

        # Extract channel information
        channel_id = str(message.get("channel_id", ""))
        guild_id = str(message.get("guild_id", "")) if message.get("guild_id") else None

        # Extract message content
        content = message.get("content", "")

        # Parse timestamp
        timestamp_str = message.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Extract attachments
        attachments = [
            {
                "id": str(a.get("id", "")),
                "filename": a.get("filename", ""),
                "url": a.get("url", ""),
                "size": a.get("size", 0),
            }
            for a in message.get("attachments", [])
        ]

        # Determine org_id if not provided
        if org_id is None:
            org_id = self.get_user_org(user_id, guild_id)
        if org_id is None:
            org_id = self._config.default_org_id or ""

        # Build metadata with Discord-specific info
        metadata: dict[str, Any] = {
            "guild_id": guild_id,
            "message_id": str(message.get("id", "")),
            "embeds": message.get("embeds", []),
            "is_dm": message.get("type") == 1 or guild_id is None,
        }

        return AdapterMessage(
            platform=AdapterPlatform.DISCORD,
            platform_user_id=user_id,
            platform_channel_id=channel_id,
            org_id=org_id,
            content=content,
            timestamp=timestamp,
            metadata=metadata,
            attachments=attachments,
        )

    def has_required_role(self, member_roles: list[str]) -> bool:
        """Check if member has any allowed role.

        Verifies that a member has at least one of the configured
        allowed roles for bot interaction.

        Args:
            member_roles: List of role IDs the member has.

        Returns:
            True if the member has a required role or no role restrictions.
        """
        # If no allowed roles configured, allow everyone
        if not self._config.allowed_role_ids:
            return True

        # Check if any member role is in the allowed list
        return any(role in self._config.allowed_role_ids for role in member_roles)

    async def get_member_roles(
        self, guild_id: str, user_id: str
    ) -> list[str]:
        """Get the role IDs for a guild member.

        Args:
            guild_id: The guild ID.
            user_id: The user ID.

        Returns:
            List of role IDs, empty if member not found.
        """
        if self._client is None:
            return []

        member = await self._client.get_member(guild_id, user_id)
        if member is None:
            return []

        return [str(r) for r in member.get("roles", [])]

    def is_command(self, content: str) -> bool:
        """Check if message content is a command.

        Args:
            content: The message content.

        Returns:
            True if the content starts with the command prefix.
        """
        return content.startswith(self._config.command_prefix)

    def parse_command(self, content: str) -> tuple[str, list[str]]:
        """Parse a command from message content.

        Args:
            content: The message content.

        Returns:
            Tuple of (command_name, arguments).
        """
        if not self.is_command(content):
            return "", []

        # Remove prefix and split
        parts = content[len(self._config.command_prefix):].strip().split()
        if not parts:
            return "", []

        return parts[0].lower(), parts[1:]
