"""Discord adapter configuration.

This module defines the configuration dataclass for the Discord bot adapter,
including authentication credentials, organization settings, and access control.
"""

from dataclasses import dataclass, field


@dataclass
class DiscordConfig:
    """Configuration for the Discord bot adapter.

    Attributes:
        bot_token: Discord bot token for authentication.
        application_id: Discord application ID for slash commands.
        default_org_id: Default organization ID for unassociated users.
        require_pairing: Whether users must be paired before interaction.
        command_prefix: Prefix for legacy text commands (e.g., "!").
        allowed_role_ids: List of Discord role IDs that can interact with the bot.
    """

    bot_token: str
    application_id: str
    default_org_id: str | None = None
    require_pairing: bool = True
    command_prefix: str = "!"
    allowed_role_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.bot_token:
            raise ValueError("bot_token is required")
        if not self.application_id:
            raise ValueError("application_id is required")

    def is_role_allowed(self, role_id: str) -> bool:
        """Check if a specific role ID is in the allowed list.

        Args:
            role_id: The Discord role ID to check.

        Returns:
            True if role is allowed or if no role restrictions are set.
        """
        if not self.allowed_role_ids:
            return True  # No restrictions if list is empty
        return role_id in self.allowed_role_ids
