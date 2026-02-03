"""Slack Bot adapter for Kintsugi CMA.

This module provides the main Slack adapter implementation using patterns
inspired by the Bolt SDK. It handles message sending, user verification,
and event normalization for the Slack platform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..shared import AdapterMessage, AdapterPlatform, AdapterResponse, BaseAdapter
from .config import SlackConfig

if TYPE_CHECKING:
    from slack_sdk import WebClient
    from slack_sdk.web import SlackResponse

    from ..shared import PairingManager


class SlackAdapter(BaseAdapter):
    """Slack Bot adapter using Bolt SDK patterns.

    This adapter provides integration between Kintsugi CMA and Slack workspaces.
    It handles message sending, user verification, and event normalization,
    using lazy initialization for the Slack WebClient.

    Attributes:
        platform: The adapter platform identifier (SLACK).

    Example:
        >>> config = SlackConfig(
        ...     bot_token="xoxb-...",
        ...     signing_secret="...",
        ... )
        >>> pairing = PairingManager(...)
        >>> adapter = SlackAdapter(config, pairing)
        >>> await adapter.send_dm("U12345", AdapterResponse(text="Hello!"))
    """

    platform = AdapterPlatform.SLACK

    def __init__(self, config: SlackConfig, pairing: PairingManager) -> None:
        """Initialize the Slack adapter.

        Args:
            config: Slack configuration containing tokens and settings.
            pairing: Pairing manager for user verification.
        """
        self._config = config
        self._pairing = pairing
        self._client: WebClient | None = None
        self._bot_user_id: str | None = None

    @property
    def config(self) -> SlackConfig:
        """Get the adapter configuration.

        Returns:
            The SlackConfig instance.
        """
        return self._config

    @property
    def client(self) -> WebClient:
        """Lazy-initialize Slack WebClient.

        Creates a new WebClient instance on first access using the
        configured bot token.

        Returns:
            The initialized Slack WebClient.

        Raises:
            ImportError: If slack_sdk is not installed.
        """
        if self._client is None:
            from slack_sdk import WebClient

            self._client = WebClient(token=self._config.bot_token)
        return self._client

    async def get_bot_user_id(self) -> str:
        """Get the bot's user ID.

        Fetches and caches the bot user ID from Slack's auth.test endpoint.

        Returns:
            The bot's Slack user ID.
        """
        if self._bot_user_id is None:
            response = await self._call_api("auth.test")
            self._bot_user_id = response["user_id"]
        return self._bot_user_id

    async def _call_api(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Call a Slack API method.

        This is a wrapper around the WebClient that handles async execution.
        In production, this would use async HTTP calls.

        Args:
            method: The Slack API method name (e.g., "chat.postMessage").
            **kwargs: Arguments to pass to the API method.

        Returns:
            The API response data.

        Raises:
            SlackApiError: If the API call fails.
        """
        # In production, use async_client or run_in_executor
        # For now, we call synchronously (stub for actual implementation)
        api_method = getattr(self.client, method.replace(".", "_"))
        response: SlackResponse = api_method(**kwargs)
        return dict(response.data)

    async def send_message(
        self, channel_id: str, response: AdapterResponse
    ) -> str:
        """Send a message to a channel.

        Sends a message using either chat_postMessage or chat_postEphemeral
        based on the response's ephemeral flag.

        Args:
            channel_id: The Slack channel ID to send to.
            response: The adapter response containing message content.

        Returns:
            The message timestamp (ts), which serves as the message ID.

        Example:
            >>> ts = await adapter.send_message(
            ...     "C12345",
            ...     AdapterResponse(text="Hello!", ephemeral=False)
            ... )
        """
        blocks = response.blocks if hasattr(response, "blocks") else None

        if getattr(response, "ephemeral", False):
            # Ephemeral messages require a user_id
            user_id = getattr(response, "user_id", None)
            if not user_id:
                raise ValueError("user_id required for ephemeral messages")

            result = await self._call_api(
                "chat.postEphemeral",
                channel=channel_id,
                user=user_id,
                text=response.text,
                blocks=blocks,
            )
            return result.get("message_ts", "")
        else:
            result = await self._call_api(
                "chat.postMessage",
                channel=channel_id,
                text=response.text,
                blocks=blocks,
                thread_ts=getattr(response, "thread_ts", None),
            )
            return result.get("ts", "")

    async def send_dm(self, user_id: str, response: AdapterResponse) -> str:
        """Open a DM channel and send a message.

        First opens a direct message channel with the user, then sends
        the message to that channel.

        Args:
            user_id: The Slack user ID to DM.
            response: The adapter response containing message content.

        Returns:
            The message timestamp (ts), which serves as the message ID.

        Example:
            >>> ts = await adapter.send_dm(
            ...     "U12345",
            ...     AdapterResponse(text="Private message")
            ... )
        """
        # Open DM channel
        conv_result = await self._call_api(
            "conversations.open",
            users=user_id,
        )
        channel_id = conv_result["channel"]["id"]

        # Send message to DM channel
        return await self.send_message(channel_id, response)

    async def verify_user(self, user_id: str, org_id: str) -> bool:
        """Check if a user is paired and allowed.

        Verifies that the given Slack user is paired with the specified
        organization through the pairing manager.

        Args:
            user_id: The Slack user ID to verify.
            org_id: The organization ID to check against.

        Returns:
            True if the user is paired and allowed, False otherwise.
        """
        # Check if pairing is required
        if not self._config.require_pairing:
            return True

        # Use default org if configured and no specific org provided
        effective_org = org_id or self._config.default_org_id
        if not effective_org:
            return False

        # Check pairing status
        return await self._pairing.is_user_paired(
            platform=self.platform,
            platform_user_id=user_id,
            org_id=effective_org,
        )

    def normalize_message(self, event: dict[str, Any]) -> AdapterMessage:
        """Convert a Slack event to an AdapterMessage.

        Extracts relevant fields from a Slack message event and creates
        a normalized AdapterMessage for processing by the agent pipeline.

        Args:
            event: The Slack event dictionary from the Events API.

        Returns:
            A normalized AdapterMessage instance.

        Example:
            >>> event = {
            ...     "type": "message",
            ...     "user": "U12345",
            ...     "channel": "C67890",
            ...     "text": "Hello bot!",
            ...     "ts": "1234567890.123456",
            ... }
            >>> msg = adapter.normalize_message(event)
        """
        # Extract basic fields
        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        text = event.get("text", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")

        # Determine channel type
        channel_type = event.get("channel_type", "unknown")

        # Check for bot messages
        is_bot = event.get("bot_id") is not None or event.get("subtype") == "bot_message"

        # Extract mentions
        mentions = self._extract_mentions(text)

        # Build metadata
        metadata = {
            "channel_type": channel_type,
            "is_bot": is_bot,
            "has_files": bool(event.get("files")),
            "has_attachments": bool(event.get("attachments")),
            "event_type": event.get("type", "message"),
            "subtype": event.get("subtype"),
        }

        return AdapterMessage(
            platform=self.platform,
            platform_user_id=user_id,
            platform_channel_id=channel_id,
            platform_message_id=ts,
            text=text,
            thread_id=thread_ts,
            mentions=mentions,
            metadata=metadata,
        )

    def _extract_mentions(self, text: str) -> list[str]:
        """Extract user mentions from message text.

        Parses Slack's mention format (<@U12345>) and extracts user IDs.

        Args:
            text: The message text to parse.

        Returns:
            List of mentioned user IDs.
        """
        import re

        pattern = r"<@([UW][A-Z0-9]+)>"
        matches = re.findall(pattern, text)
        return matches

    async def get_user_info(self, user_id: str) -> dict[str, Any]:
        """Fetch user profile from Slack.

        Retrieves detailed user information including name, email (if available),
        and profile data.

        Args:
            user_id: The Slack user ID to look up.

        Returns:
            Dictionary containing user profile information.

        Example:
            >>> info = await adapter.get_user_info("U12345")
            >>> print(info["user"]["real_name"])
        """
        result = await self._call_api("users.info", user=user_id)
        return result

    async def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        """Fetch channel information from Slack.

        Retrieves channel metadata including name, purpose, and members.

        Args:
            channel_id: The Slack channel ID to look up.

        Returns:
            Dictionary containing channel information.
        """
        result = await self._call_api("conversations.info", channel=channel_id)
        return result

    async def add_reaction(self, channel_id: str, ts: str, emoji: str) -> None:
        """Add a reaction to a message.

        Args:
            channel_id: The channel containing the message.
            ts: The message timestamp.
            emoji: The emoji name (without colons).
        """
        await self._call_api(
            "reactions.add",
            channel=channel_id,
            timestamp=ts,
            name=emoji,
        )

    async def update_message(
        self, channel_id: str, ts: str, response: AdapterResponse
    ) -> None:
        """Update an existing message.

        Args:
            channel_id: The channel containing the message.
            ts: The message timestamp to update.
            response: The new response content.
        """
        blocks = response.blocks if hasattr(response, "blocks") else None

        await self._call_api(
            "chat.update",
            channel=channel_id,
            ts=ts,
            text=response.text,
            blocks=blocks,
        )
