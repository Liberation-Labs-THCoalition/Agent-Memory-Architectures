"""Slack Block Kit message builders for Kintsugi CMA.

This module provides functions to build Block Kit messages for various
Kintsugi interactions including pairing flows, agent responses, errors,
and help documentation.

Block Kit Reference: https://api.slack.com/block-kit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..shared import PairingCode


def pairing_request_blocks(code: str, expires_in_minutes: int) -> list[dict[str, Any]]:
    """Build Block Kit message for pairing code delivery.

    Creates a formatted message containing the pairing code, expiration
    warning, and instructions for completing the pairing process.

    Args:
        code: The generated pairing code to display.
        expires_in_minutes: Number of minutes until the code expires.

    Returns:
        List of Block Kit block dictionaries.

    Example:
        >>> blocks = pairing_request_blocks("ABC123", 15)
        >>> # Use in chat_postMessage or say()
    """
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Kintsugi Pairing Code",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Use this code to link your Slack account with your "
                    "Kintsugi organization."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Your pairing code:*\n```{code}```",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":clock1: This code expires in *{expires_in_minutes} minutes*",
                },
            ],
        },
        {
            "type": "divider",
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*How to complete pairing:*\n"
                    "1. Go to your Kintsugi dashboard\n"
                    "2. Navigate to Settings > Integrations > Slack\n"
                    "3. Enter this pairing code\n"
                    "4. Click 'Link Account'"
                ),
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        ":lock: This code is unique to you and can only be used once. "
                        "Do not share it with others."
                    ),
                },
            ],
        },
    ]


def pairing_approval_blocks(pairing_code: PairingCode) -> list[dict[str, Any]]:
    """Build Block Kit message for admin approval request.

    Creates an interactive message for admins to approve or reject
    a user's pairing request.

    Args:
        pairing_code: The PairingCode object containing request details.

    Returns:
        List of Block Kit block dictionaries with approve/reject buttons.

    Example:
        >>> blocks = pairing_approval_blocks(pairing_code)
        >>> # Send to admin channel
    """
    # Extract user info from pairing code metadata
    user_id = getattr(pairing_code, "platform_user_id", "Unknown")
    platform = getattr(pairing_code, "platform", "slack")
    requested_at = getattr(pairing_code, "created_at", "Unknown")

    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Pairing Request",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"A user has requested to pair their {platform} account.",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*User:*\n<@{user_id}>",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Platform:*\n{platform.capitalize()}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Requested:*\n{requested_at}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Code:*\n`{pairing_code.code}`",
                },
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Approve",
                        "emoji": True,
                    },
                    "style": "primary",
                    "action_id": "approve_pairing",
                    "value": pairing_code.code,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Reject",
                        "emoji": True,
                    },
                    "style": "danger",
                    "action_id": "reject_pairing",
                    "value": pairing_code.code,
                },
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        ":information_source: Approving this request will allow "
                        "the user to interact with Kintsugi via Slack."
                    ),
                },
            ],
        },
    ]


def agent_response_blocks(
    response: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Format agent response with optional metadata.

    Creates a Block Kit message for displaying agent responses,
    optionally including metadata like confidence scores or sources.

    Args:
        response: The agent's response text.
        metadata: Optional metadata to display (e.g., confidence, sources).

    Returns:
        List of Block Kit block dictionaries.

    Example:
        >>> blocks = agent_response_blocks(
        ...     "Here's what I found...",
        ...     metadata={"confidence": 0.95, "sources": ["doc1", "doc2"]}
        ... )
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": response,
            },
        },
    ]

    if metadata:
        # Add metadata context
        context_elements = []

        if "confidence" in metadata:
            confidence = metadata["confidence"]
            confidence_pct = f"{confidence * 100:.0f}%" if confidence <= 1 else f"{confidence}%"
            context_elements.append({
                "type": "mrkdwn",
                "text": f":chart_with_upwards_trend: Confidence: {confidence_pct}",
            })

        if "sources" in metadata:
            sources = metadata["sources"]
            if isinstance(sources, list):
                source_text = ", ".join(sources[:3])
                if len(sources) > 3:
                    source_text += f" (+{len(sources) - 3} more)"
                context_elements.append({
                    "type": "mrkdwn",
                    "text": f":books: Sources: {source_text}",
                })

        if "model" in metadata:
            context_elements.append({
                "type": "mrkdwn",
                "text": f":robot_face: {metadata['model']}",
            })

        if "latency_ms" in metadata:
            latency = metadata["latency_ms"]
            context_elements.append({
                "type": "mrkdwn",
                "text": f":stopwatch: {latency}ms",
            })

        if context_elements:
            blocks.append({
                "type": "context",
                "elements": context_elements,
            })

    return blocks


def error_blocks(
    error: str,
    suggestion: str | None = None,
) -> list[dict[str, Any]]:
    """Format error message.

    Creates a Block Kit message for displaying errors with an optional
    suggestion for resolution.

    Args:
        error: The error message to display.
        suggestion: Optional suggestion for how to resolve the error.

    Returns:
        List of Block Kit block dictionaries.

    Example:
        >>> blocks = error_blocks(
        ...     "Failed to process request",
        ...     suggestion="Please try again in a few minutes."
        ... )
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":warning: *Error*\n{error}",
            },
        },
    ]

    if suggestion:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":bulb: {suggestion}",
                },
            ],
        })

    return blocks


def help_blocks() -> list[dict[str, Any]]:
    """Build help message with available commands.

    Creates a comprehensive help message showing all available
    slash commands and their usage.

    Returns:
        List of Block Kit block dictionaries.

    Example:
        >>> blocks = help_blocks()
        >>> # Send as response to /kintsugi help
    """
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Kintsugi Help",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Kintsugi is your AI-powered assistant for knowledge management "
                    "and contextual memory."
                ),
            },
        },
        {
            "type": "divider",
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Available Commands*",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "`/kintsugi pair`\n"
                    "Generate a pairing code to link your Slack account with "
                    "your Kintsugi organization."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "`/kintsugi status`\n"
                    "Check your current connection status and paired organization."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "`/kintsugi help`\n"
                    "Show this help message."
                ),
            },
        },
        {
            "type": "divider",
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Interacting with Kintsugi*",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Once paired, you can interact with Kintsugi in several ways:\n\n"
                    "- *Direct Message*: Send a DM to this bot\n"
                    "- *Mention*: Use `@Kintsugi` in any channel I'm in\n"
                    "- *Thread*: Reply in a thread where I've responded"
                ),
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        ":question: Need more help? Visit our "
                        "<https://docs.kintsugi.ai|documentation> or contact support."
                    ),
                },
            ],
        },
    ]


def loading_blocks(message: str = "Processing your request...") -> list[dict[str, Any]]:
    """Build a loading/processing indicator message.

    Creates a simple message indicating that processing is in progress.

    Args:
        message: The loading message to display.

    Returns:
        List of Block Kit block dictionaries.
    """
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":hourglass_flowing_sand: {message}",
            },
        },
    ]


def success_blocks(
    title: str,
    message: str,
) -> list[dict[str, Any]]:
    """Build a success message.

    Creates a formatted success message with a title and description.

    Args:
        title: The success title.
        message: The success message body.

    Returns:
        List of Block Kit block dictionaries.
    """
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":white_check_mark: *{title}*\n{message}",
            },
        },
    ]


def confirmation_blocks(
    question: str,
    confirm_action_id: str,
    cancel_action_id: str,
    confirm_value: str = "confirm",
    cancel_value: str = "cancel",
) -> list[dict[str, Any]]:
    """Build a confirmation prompt with buttons.

    Creates an interactive message asking for confirmation with
    confirm and cancel buttons.

    Args:
        question: The confirmation question to display.
        confirm_action_id: Action ID for the confirm button.
        cancel_action_id: Action ID for the cancel button.
        confirm_value: Value to send when confirm is clicked.
        cancel_value: Value to send when cancel is clicked.

    Returns:
        List of Block Kit block dictionaries.
    """
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":question: {question}",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Confirm",
                        "emoji": True,
                    },
                    "style": "primary",
                    "action_id": confirm_action_id,
                    "value": confirm_value,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Cancel",
                        "emoji": True,
                    },
                    "action_id": cancel_action_id,
                    "value": cancel_value,
                },
            ],
        },
    ]
