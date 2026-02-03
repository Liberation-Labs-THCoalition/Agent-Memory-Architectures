"""
Base adapter infrastructure for Kintsugi CMA.

This module provides the foundational classes for all platform adapters,
including message normalization and response formatting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AdapterPlatform(str, Enum):
    """Supported chat platforms for Kintsugi adapters."""

    SLACK = "slack"
    DISCORD = "discord"
    WEBCHAT = "webchat"
    EMAIL = "email"


@dataclass
class AdapterMessage:
    """
    Normalized message from any platform.

    This dataclass provides a unified representation of messages across
    all supported platforms, enabling platform-agnostic message processing.

    Attributes:
        platform: The source platform of the message.
        platform_user_id: The user's ID on the source platform.
        platform_channel_id: The channel/conversation ID on the source platform.
        org_id: The organization ID this message belongs to.
        content: The text content of the message.
        timestamp: When the message was sent.
        metadata: Platform-specific metadata (e.g., thread_ts for Slack).
        attachments: List of file attachments with the message.
    """

    platform: AdapterPlatform
    platform_user_id: str
    platform_channel_id: str
    org_id: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate message fields after initialization."""
        if not self.platform_user_id:
            raise ValueError("platform_user_id cannot be empty")
        if not self.org_id:
            raise ValueError("org_id cannot be empty")


@dataclass
class AdapterResponse:
    """
    Response to send back to a platform.

    This dataclass represents an outgoing message that will be
    formatted appropriately for each platform adapter.

    Attributes:
        content: The text content of the response.
        attachments: List of file attachments to include.
        ephemeral: If True, message is only visible to the sender.
        metadata: Platform-specific options (e.g., blocks for Slack).
    """

    content: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    ephemeral: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response fields after initialization."""
        if not self.content and not self.attachments:
            raise ValueError("Response must have content or attachments")


class BaseAdapter(ABC):
    """
    Abstract base class for all chat platform adapters.

    Each platform adapter (Slack, Discord, etc.) must inherit from this
    class and implement all abstract methods. This ensures consistent
    behavior across platforms.

    Attributes:
        platform: The platform this adapter handles.

    Example:
        class SlackAdapter(BaseAdapter):
            platform = AdapterPlatform.SLACK

            async def send_message(self, channel_id, response):
                # Slack-specific implementation
                ...
    """

    platform: AdapterPlatform

    @abstractmethod
    async def send_message(self, channel_id: str, response: AdapterResponse) -> str:
        """
        Send a message to a channel.

        Args:
            channel_id: The platform-specific channel identifier.
            response: The response to send.

        Returns:
            The platform-specific message ID of the sent message.

        Raises:
            AdapterError: If the message could not be sent.
        """
        pass

    @abstractmethod
    async def send_dm(self, user_id: str, response: AdapterResponse) -> str:
        """
        Send a direct message to a user.

        Args:
            user_id: The platform-specific user identifier.
            response: The response to send.

        Returns:
            The platform-specific message ID of the sent message.

        Raises:
            AdapterError: If the DM could not be sent.
        """
        pass

    @abstractmethod
    async def verify_user(self, user_id: str) -> bool:
        """
        Check if a user is allowed to interact with the bot.

        This method should check the allowlist and any platform-specific
        verification requirements.

        Args:
            user_id: The platform-specific user identifier.

        Returns:
            True if the user is verified and allowed, False otherwise.
        """
        pass

    def normalize_message(self, raw: dict[str, Any]) -> AdapterMessage:
        """
        Convert a platform-specific message to an AdapterMessage.

        Subclasses should override this method to handle their
        platform's message format.

        Args:
            raw: The raw message data from the platform.

        Returns:
            A normalized AdapterMessage instance.

        Raises:
            NotImplementedError: If the subclass hasn't implemented this.
            ValueError: If the raw message is malformed.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement normalize_message()"
        )

    async def health_check(self) -> bool:
        """
        Check if the adapter's connection to the platform is healthy.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} platform={self.platform.value}>"
