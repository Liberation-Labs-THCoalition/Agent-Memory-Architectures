"""Slack event handlers for Kintsugi CMA.

This module provides event handlers for Slack events, commands, and
interactions. Handlers process incoming events and route them to the
appropriate agent pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .blocks import error_blocks, help_blocks, pairing_request_blocks

if TYPE_CHECKING:
    from ..shared import AdapterResponse, PairingManager
    from .bot import SlackAdapter

logger = logging.getLogger(__name__)


class SlackEventHandler:
    """Handlers for Slack events and commands.

    This class provides methods for handling various Slack events including
    messages, app mentions, and slash commands. It manages user pairing
    verification and routes valid requests to the agent pipeline.

    Attributes:
        adapter: The SlackAdapter instance for sending responses.
        pairing: The PairingManager for user verification.

    Example:
        >>> handler = SlackEventHandler(adapter, pairing)
        >>> await handler.handle_message(event, say)
    """

    def __init__(self, adapter: SlackAdapter, pairing: PairingManager) -> None:
        """Initialize the event handler.

        Args:
            adapter: The SlackAdapter instance.
            pairing: The PairingManager for user verification.
        """
        self._adapter = adapter
        self._pairing = pairing
        self._agent_callback: Callable | None = None

    def set_agent_callback(self, callback: Callable) -> None:
        """Set the callback for forwarding messages to the agent pipeline.

        Args:
            callback: Async callable that accepts an AdapterMessage and returns
                an AdapterResponse.
        """
        self._agent_callback = callback

    async def handle_message(
        self,
        event: dict[str, Any],
        say: Callable,
    ) -> None:
        """Handle incoming message event.

        Processes a Slack message event, verifies user pairing status,
        and forwards valid messages to the agent pipeline.

        Args:
            event: The Slack message event dictionary.
            say: Bolt's say function for sending responses.

        Note:
            - Ignores bot messages to prevent loops
            - Checks channel type against allowed types
            - Sends pairing instructions to unpaired users
            - Forwards paired user messages to agent pipeline
        """
        # Ignore bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            logger.debug("Ignoring bot message")
            return

        user_id = event.get("user", "")
        channel_id = event.get("channel", "")
        channel_type = event.get("channel_type", "")

        # Check channel type
        if not self._adapter.config.is_channel_type_allowed(channel_type):
            logger.debug(f"Ignoring message in disallowed channel type: {channel_type}")
            return

        # Determine org_id
        org_id = self._adapter.config.default_org_id

        # Check pairing status
        if self._adapter.config.require_pairing:
            is_paired = await self._adapter.verify_user(user_id, org_id or "")
            if not is_paired:
                await self._send_pairing_instructions(user_id, channel_id, say)
                return

        # Normalize message
        message = self._adapter.normalize_message(event)

        # Forward to agent pipeline
        if self._agent_callback:
            try:
                response: AdapterResponse = await self._agent_callback(message)
                await say(
                    text=response.text,
                    blocks=getattr(response, "blocks", None),
                    thread_ts=event.get("thread_ts") or event.get("ts"),
                )
            except Exception as e:
                logger.exception("Error processing message through agent pipeline")
                await say(
                    blocks=error_blocks(
                        error="I encountered an error processing your request.",
                        suggestion="Please try again or contact support if the issue persists.",
                    )
                )
        else:
            logger.warning("No agent callback configured")

    async def handle_app_mention(
        self,
        event: dict[str, Any],
        say: Callable,
    ) -> None:
        """Handle @mention of the bot.

        Processes an app_mention event when a user mentions the bot.
        This follows the same flow as regular messages but is triggered
        specifically by mentions.

        Args:
            event: The Slack app_mention event dictionary.
            say: Bolt's say function for sending responses.
        """
        # App mentions are processed similarly to messages
        # but the channel_type might not be in the event
        if "channel_type" not in event:
            event["channel_type"] = "channel"  # Assume public channel for mentions

        await self.handle_message(event, say)

    async def handle_slash_command(
        self,
        command: dict[str, Any],
        ack: Callable,
        respond: Callable,
    ) -> None:
        """Handle slash commands like /kintsugi.

        Processes slash commands and routes to appropriate subcommand handlers.
        Supported subcommands: pair, help, status.

        Args:
            command: The Slack command dictionary containing:
                - command: The slash command name
                - text: Arguments after the command
                - user_id: The invoking user's ID
                - channel_id: The channel where command was invoked
            ack: Bolt's acknowledge function (must be called within 3 seconds).
            respond: Bolt's respond function for sending responses.

        Example:
            User types: /kintsugi pair
            This routes to handle_pair_command
        """
        # Acknowledge immediately
        await ack()

        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        text = command.get("text", "").strip()

        # Parse subcommand
        parts = text.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "help"
        args = parts[1] if len(parts) > 1 else ""

        if subcommand == "pair":
            await self.handle_pair_command(user_id, channel_id, respond)
        elif subcommand == "status":
            await self._handle_status_command(user_id, respond)
        elif subcommand == "help":
            await self._handle_help_command(respond)
        else:
            await respond(
                blocks=error_blocks(
                    error=f"Unknown command: {subcommand}",
                    suggestion="Use `/kintsugi help` to see available commands.",
                )
            )

    async def handle_pair_command(
        self,
        user_id: str,
        channel_id: str,
        respond: Callable,
    ) -> None:
        """Generate and send pairing code via DM.

        Initiates the pairing flow for a user by generating a unique
        pairing code and sending it via direct message.

        Args:
            user_id: The Slack user ID requesting pairing.
            channel_id: The channel where the command was invoked.
            respond: Bolt's respond function for sending responses.

        Note:
            The pairing code is sent via DM for security, not in the
            channel where the command was invoked.
        """
        try:
            # Generate pairing code
            pairing_code = await self._pairing.generate_code(
                platform=self._adapter.platform,
                platform_user_id=user_id,
            )

            # Send pairing code via DM
            from ..shared import AdapterResponse

            dm_response = AdapterResponse(
                text="Your Kintsugi pairing code",
                blocks=pairing_request_blocks(
                    code=pairing_code.code,
                    expires_in_minutes=pairing_code.expires_in_minutes,
                ),
            )
            await self._adapter.send_dm(user_id, dm_response)

            # Respond in channel (ephemeral)
            await respond(
                text="I've sent you a DM with your pairing code!",
                response_type="ephemeral",
            )

        except Exception as e:
            logger.exception("Error generating pairing code")
            await respond(
                blocks=error_blocks(
                    error="Failed to generate pairing code.",
                    suggestion="Please try again or contact support.",
                ),
                response_type="ephemeral",
            )

    async def _send_pairing_instructions(
        self,
        user_id: str,
        channel_id: str,
        say: Callable,
    ) -> None:
        """Send pairing instructions to an unpaired user.

        Args:
            user_id: The unpaired user's Slack ID.
            channel_id: The channel where the message was sent.
            say: Bolt's say function for sending responses.
        """
        await say(
            text="You need to pair your account before I can help you.",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            ":wave: *Welcome to Kintsugi!*\n\n"
                            "Before I can assist you, you'll need to pair your Slack "
                            "account with your Kintsugi organization.\n\n"
                            "Use `/kintsugi pair` to get started."
                        ),
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "This is a one-time setup that links your Slack identity securely.",
                        }
                    ],
                },
            ],
        )

    async def _handle_status_command(
        self,
        user_id: str,
        respond: Callable,
    ) -> None:
        """Handle the status subcommand.

        Shows the user's current pairing and connection status.

        Args:
            user_id: The user's Slack ID.
            respond: Bolt's respond function.
        """
        org_id = self._adapter.config.default_org_id
        is_paired = await self._adapter.verify_user(user_id, org_id or "")

        if is_paired:
            status_text = ":white_check_mark: *Status: Connected*\n\nYour account is paired and active."
            org_info = f"\nOrganization: `{org_id}`" if org_id else ""
        else:
            status_text = ":x: *Status: Not Connected*\n\nYour account is not paired."
            org_info = "\n\nUse `/kintsugi pair` to connect your account."

        await respond(
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": status_text + org_info,
                    },
                },
            ],
            response_type="ephemeral",
        )

    async def _handle_help_command(self, respond: Callable) -> None:
        """Handle the help subcommand.

        Shows available commands and usage information.

        Args:
            respond: Bolt's respond function.
        """
        await respond(
            blocks=help_blocks(),
            response_type="ephemeral",
        )


class SlackInteractionHandler:
    """Handler for Slack interactive components.

    Handles button clicks, modal submissions, and other interactive
    elements from Block Kit.
    """

    def __init__(self, adapter: SlackAdapter, pairing: PairingManager) -> None:
        """Initialize the interaction handler.

        Args:
            adapter: The SlackAdapter instance.
            pairing: The PairingManager for user verification.
        """
        self._adapter = adapter
        self._pairing = pairing

    async def handle_button_action(
        self,
        body: dict[str, Any],
        ack: Callable,
        respond: Callable,
    ) -> None:
        """Handle button click actions.

        Args:
            body: The Slack interaction payload.
            ack: Bolt's acknowledge function.
            respond: Bolt's respond function.
        """
        await ack()

        action = body.get("actions", [{}])[0]
        action_id = action.get("action_id", "")

        if action_id == "approve_pairing":
            await self._handle_approve_pairing(body, respond)
        elif action_id == "reject_pairing":
            await self._handle_reject_pairing(body, respond)
        else:
            logger.warning(f"Unknown button action: {action_id}")

    async def _handle_approve_pairing(
        self,
        body: dict[str, Any],
        respond: Callable,
    ) -> None:
        """Handle pairing approval button click.

        Args:
            body: The Slack interaction payload.
            respond: Bolt's respond function.
        """
        action = body.get("actions", [{}])[0]
        pairing_code = action.get("value", "")

        try:
            await self._pairing.approve_code(pairing_code)
            await respond(
                text=":white_check_mark: Pairing approved!",
                replace_original=True,
            )
        except Exception as e:
            logger.exception("Error approving pairing")
            await respond(
                text=f":x: Error approving pairing: {e}",
                replace_original=False,
            )

    async def _handle_reject_pairing(
        self,
        body: dict[str, Any],
        respond: Callable,
    ) -> None:
        """Handle pairing rejection button click.

        Args:
            body: The Slack interaction payload.
            respond: Bolt's respond function.
        """
        action = body.get("actions", [{}])[0]
        pairing_code = action.get("value", "")

        try:
            await self._pairing.reject_code(pairing_code)
            await respond(
                text=":x: Pairing rejected.",
                replace_original=True,
            )
        except Exception as e:
            logger.exception("Error rejecting pairing")
            await respond(
                text=f":x: Error rejecting pairing: {e}",
                replace_original=False,
            )
