"""Slack adapter configuration.

This module defines the configuration dataclass for the Slack Bot adapter,
including authentication tokens, feature flags, and deployment settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlackConfig:
    """Configuration for Slack Bot adapter.

    This dataclass holds all configuration needed to connect and operate
    a Slack bot integration with Kintsugi CMA.

    Attributes:
        bot_token: Slack bot OAuth token (xoxb-...). Required for API calls.
        signing_secret: Slack signing secret for request verification.
            Used to validate that incoming requests are from Slack.
        app_token: Optional Slack app-level token (xapp-...) for Socket Mode.
            Required when using WebSocket connections instead of HTTP endpoints.
        default_org_id: Optional organization ID for single-org deployments.
            When set, all users are automatically associated with this org.
        require_pairing: Whether users must complete pairing before interaction.
            When True (default), unpaired users receive pairing instructions.
        allowed_channel_types: List of channel types the bot responds in.
            Defaults to all types: im (DM), mpim (group DM), channel, group.

    Example:
        >>> config = SlackConfig(
        ...     bot_token="xoxb-123456789012-...",
        ...     signing_secret="abc123...",
        ...     app_token="xapp-1-A0...",  # For Socket Mode
        ...     default_org_id="org_default",
        ...     require_pairing=True,
        ... )
    """

    bot_token: str
    signing_secret: str
    app_token: str | None = None
    default_org_id: str | None = None
    require_pairing: bool = True
    allowed_channel_types: list[str] = field(
        default_factory=lambda: ["im", "mpim", "channel", "group"]
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.bot_token:
            raise ValueError("bot_token is required")
        if not self.bot_token.startswith("xoxb-"):
            raise ValueError("bot_token must be a bot token starting with 'xoxb-'")
        if not self.signing_secret:
            raise ValueError("signing_secret is required")
        if self.app_token is not None and not self.app_token.startswith("xapp-"):
            raise ValueError("app_token must start with 'xapp-' for Socket Mode")

    @property
    def uses_socket_mode(self) -> bool:
        """Check if Socket Mode is configured.

        Returns:
            True if app_token is set, indicating Socket Mode should be used.
        """
        return self.app_token is not None

    def is_channel_type_allowed(self, channel_type: str) -> bool:
        """Check if a channel type is allowed for bot interaction.

        Args:
            channel_type: The Slack channel type to check.

        Returns:
            True if the channel type is in the allowed list.
        """
        return channel_type in self.allowed_channel_types
