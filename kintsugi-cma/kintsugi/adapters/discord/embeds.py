"""Discord embed builders for rich message formatting.

This module provides dataclasses and builder functions for creating
Discord embeds, enabling consistent and visually appealing message
formatting across all bot responses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..shared import AdapterResponse


@dataclass
class EmbedField:
    """A single field within a Discord embed.

    Attributes:
        name: The field title/header.
        value: The field content.
        inline: Whether to display inline with other fields.
    """

    name: str
    value: str
    inline: bool = False

    def to_dict(self) -> dict:
        """Convert to Discord API format.

        Returns:
            Dictionary representation for the Discord API.
        """
        return {
            "name": self.name,
            "value": self.value,
            "inline": self.inline,
        }


class EmbedColors:
    """Color constants for Discord embeds.

    Colors are represented as decimal integers for the Discord API.
    """

    SUCCESS = 0x2ECC71  # Green
    ERROR = 0xE74C3C  # Red
    WARNING = 0xF39C12  # Orange
    INFO = 0x3498DB  # Blue
    KINTSUGI = 0x9B59B6  # Purple (brand color)


@dataclass
class DiscordEmbed:
    """A Discord embed message structure.

    Embeds allow for rich formatting including titles, descriptions,
    fields, colors, and timestamps. This dataclass mirrors the
    Discord embed structure for easy construction.

    Attributes:
        title: The embed title (optional).
        description: The main embed content (optional).
        color: Decimal color value for the embed sidebar (optional).
        fields: List of EmbedField objects for structured data.
        footer: Text to display in the embed footer (optional).
        timestamp: Timestamp to display in the footer (optional).
        url: URL for the embed title link (optional).
        thumbnail_url: URL for a thumbnail image (optional).
        image_url: URL for a main image (optional).
        author_name: Name to display as the embed author (optional).
        author_icon_url: URL for the author icon (optional).
    """

    title: str | None = None
    description: str | None = None
    color: int | None = None
    fields: list[EmbedField] = field(default_factory=list)
    footer: str | None = None
    timestamp: datetime | None = None
    url: str | None = None
    thumbnail_url: str | None = None
    image_url: str | None = None
    author_name: str | None = None
    author_icon_url: str | None = None

    def to_dict(self) -> dict:
        """Convert to Discord API format.

        Returns:
            Dictionary representation suitable for the Discord API.
        """
        embed: dict = {}

        if self.title is not None:
            embed["title"] = self.title

        if self.description is not None:
            embed["description"] = self.description

        if self.color is not None:
            embed["color"] = self.color

        if self.fields:
            embed["fields"] = [f.to_dict() for f in self.fields]

        if self.footer is not None:
            embed["footer"] = {"text": self.footer}

        if self.timestamp is not None:
            embed["timestamp"] = self.timestamp.isoformat()

        if self.url is not None:
            embed["url"] = self.url

        if self.thumbnail_url is not None:
            embed["thumbnail"] = {"url": self.thumbnail_url}

        if self.image_url is not None:
            embed["image"] = {"url": self.image_url}

        if self.author_name is not None:
            author: dict = {"name": self.author_name}
            if self.author_icon_url is not None:
                author["icon_url"] = self.author_icon_url
            embed["author"] = author

        return embed

    def add_field(self, name: str, value: str, inline: bool = False) -> "DiscordEmbed":
        """Add a field to the embed.

        Args:
            name: The field title/header.
            value: The field content.
            inline: Whether to display inline with other fields.

        Returns:
            Self for method chaining.
        """
        self.fields.append(EmbedField(name=name, value=value, inline=inline))
        return self


def pairing_code_embed(code: str, expires_at: datetime) -> DiscordEmbed:
    """Build embed for pairing code delivery.

    Creates a formatted embed to send to users when they request
    a pairing code for linking their Discord account.

    Args:
        code: The generated pairing code.
        expires_at: When the pairing code expires.

    Returns:
        A DiscordEmbed configured for pairing code display.
    """
    # Calculate minutes until expiration
    remaining = expires_at - datetime.now(timezone.utc)
    minutes_remaining = max(0, int(remaining.total_seconds() / 60))

    return DiscordEmbed(
        title="Kintsugi Pairing Code",
        description=(
            "Your pairing code has been generated. "
            "Share this code with an administrator to complete pairing."
        ),
        color=EmbedColors.KINTSUGI,
        fields=[
            EmbedField(
                name="Pairing Code",
                value=f"```{code}```",
                inline=False,
            ),
            EmbedField(
                name="Expires In",
                value=f"{minutes_remaining} minutes",
                inline=True,
            ),
        ],
        footer="This code can only be used once.",
        timestamp=expires_at,
    )


def pairing_approval_embed(
    pairing_code: str,
    user_id: str,
    user_name: str,
    requested_at: datetime,
) -> DiscordEmbed:
    """Build embed for admin approval request.

    Creates a formatted embed for administrators to review
    and approve pairing requests.

    Args:
        pairing_code: The pairing code to approve.
        user_id: The Discord user ID requesting pairing.
        user_name: The Discord username requesting pairing.
        requested_at: When the pairing was requested.

    Returns:
        A DiscordEmbed configured for pairing approval display.
    """
    return DiscordEmbed(
        title="Pairing Approval Request",
        description=f"A user has requested to pair with Kintsugi.",
        color=EmbedColors.WARNING,
        fields=[
            EmbedField(
                name="User",
                value=f"<@{user_id}> ({user_name})",
                inline=True,
            ),
            EmbedField(
                name="Code",
                value=f"`{pairing_code}`",
                inline=True,
            ),
            EmbedField(
                name="Requested At",
                value=requested_at.strftime("%Y-%m-%d %H:%M UTC"),
                inline=True,
            ),
            EmbedField(
                name="Actions",
                value=(
                    f"Use `/approve {pairing_code}` to approve\n"
                    f"Or ignore to let it expire"
                ),
                inline=False,
            ),
        ],
        footer="Only admins can approve pairing requests.",
        timestamp=requested_at,
    )


def agent_response_embed(
    response: str,
    processing_time: float | None = None,
) -> DiscordEmbed:
    """Format agent response as embed.

    Creates a formatted embed for displaying agent responses
    to user queries.

    Args:
        response: The agent's response text.
        processing_time: Time taken to process the query in seconds (optional).

    Returns:
        A DiscordEmbed configured for agent response display.
    """
    # Truncate response if too long for Discord embed (max 4096 chars for description)
    max_length = 4000
    truncated = len(response) > max_length
    display_response = response[:max_length] + "..." if truncated else response

    embed = DiscordEmbed(
        title="Kintsugi Response",
        description=display_response,
        color=EmbedColors.KINTSUGI,
        timestamp=datetime.now(timezone.utc),
    )

    footer_parts = []
    if processing_time is not None:
        footer_parts.append(f"Processed in {processing_time:.2f}s")
    if truncated:
        footer_parts.append("Response truncated")

    if footer_parts:
        embed.footer = " | ".join(footer_parts)

    return embed


def error_embed(error: str, suggestion: str | None = None) -> DiscordEmbed:
    """Format error message as embed.

    Creates a formatted embed for displaying error messages
    with optional suggestions for resolution.

    Args:
        error: The error message to display.
        suggestion: Optional suggestion for resolving the error.

    Returns:
        A DiscordEmbed configured for error display.
    """
    embed = DiscordEmbed(
        title="Error",
        description=error,
        color=EmbedColors.ERROR,
        timestamp=datetime.now(timezone.utc),
    )

    if suggestion:
        embed.add_field(
            name="Suggestion",
            value=suggestion,
            inline=False,
        )

    return embed


def success_embed(title: str, message: str) -> DiscordEmbed:
    """Format success message as embed.

    Creates a formatted embed for displaying success messages.

    Args:
        title: The success title.
        message: The success message.

    Returns:
        A DiscordEmbed configured for success display.
    """
    return DiscordEmbed(
        title=title,
        description=message,
        color=EmbedColors.SUCCESS,
        timestamp=datetime.now(timezone.utc),
    )


def warning_embed(title: str, message: str) -> DiscordEmbed:
    """Format warning message as embed.

    Creates a formatted embed for displaying warning messages.

    Args:
        title: The warning title.
        message: The warning message.

    Returns:
        A DiscordEmbed configured for warning display.
    """
    return DiscordEmbed(
        title=title,
        description=message,
        color=EmbedColors.WARNING,
        timestamp=datetime.now(timezone.utc),
    )


def help_embed() -> DiscordEmbed:
    """Build help embed with available commands.

    Creates a comprehensive help embed listing all available
    bot commands and their usage.

    Returns:
        A DiscordEmbed configured for help display.
    """
    return DiscordEmbed(
        title="Kintsugi Help",
        description=(
            "Welcome to Kintsugi! Here are the available commands to interact "
            "with your memory-augmented AI assistant."
        ),
        color=EmbedColors.INFO,
        fields=[
            EmbedField(
                name="/pair",
                value="Generate a pairing code to link your Discord account.",
                inline=False,
            ),
            EmbedField(
                name="/status",
                value="Check your current pairing status.",
                inline=False,
            ),
            EmbedField(
                name="/ask <question>",
                value="Ask Kintsugi a question. Requires pairing.",
                inline=False,
            ),
            EmbedField(
                name="/help",
                value="Display this help message.",
                inline=False,
            ),
            EmbedField(
                name="Admin Commands",
                value=(
                    "`/approve <code>` - Approve a pairing request\n"
                    "`/revoke <user>` - Revoke a user's access\n"
                    "`/list-paired` - List all paired users"
                ),
                inline=False,
            ),
        ],
        footer="Kintsugi - Memory-augmented AI assistance",
        timestamp=datetime.now(timezone.utc),
    )


def status_embed(
    is_paired: bool,
    user_name: str,
    org_name: str | None = None,
    paired_at: datetime | None = None,
) -> DiscordEmbed:
    """Build status embed showing pairing status.

    Args:
        is_paired: Whether the user is paired.
        user_name: The user's Discord name.
        org_name: The organization name if paired.
        paired_at: When the user was paired.

    Returns:
        A DiscordEmbed configured for status display.
    """
    if is_paired:
        return DiscordEmbed(
            title="Pairing Status",
            description=f"You are paired with Kintsugi.",
            color=EmbedColors.SUCCESS,
            fields=[
                EmbedField(
                    name="User",
                    value=user_name,
                    inline=True,
                ),
                EmbedField(
                    name="Organization",
                    value=org_name or "Default",
                    inline=True,
                ),
                EmbedField(
                    name="Paired Since",
                    value=(
                        paired_at.strftime("%Y-%m-%d %H:%M UTC")
                        if paired_at
                        else "Unknown"
                    ),
                    inline=True,
                ),
            ],
            timestamp=datetime.now(timezone.utc),
        )
    else:
        return DiscordEmbed(
            title="Pairing Status",
            description="You are not paired with Kintsugi.",
            color=EmbedColors.WARNING,
            fields=[
                EmbedField(
                    name="How to Pair",
                    value=(
                        "Use `/pair` to generate a pairing code, then ask an "
                        "administrator to approve it."
                    ),
                    inline=False,
                ),
            ],
            timestamp=datetime.now(timezone.utc),
        )


def paired_users_embed(
    users: list[dict],
    org_name: str,
    page: int = 1,
    total_pages: int = 1,
) -> DiscordEmbed:
    """Build embed listing paired users.

    Args:
        users: List of paired user dictionaries with 'name', 'id', 'paired_at'.
        org_name: The organization name.
        page: Current page number.
        total_pages: Total number of pages.

    Returns:
        A DiscordEmbed configured for paired users list display.
    """
    if not users:
        return DiscordEmbed(
            title=f"Paired Users - {org_name}",
            description="No users are currently paired.",
            color=EmbedColors.INFO,
            timestamp=datetime.now(timezone.utc),
        )

    user_lines = []
    for user in users:
        paired_date = user.get("paired_at", "Unknown")
        if isinstance(paired_date, datetime):
            paired_date = paired_date.strftime("%Y-%m-%d")
        user_lines.append(f"<@{user['id']}> - Paired: {paired_date}")

    return DiscordEmbed(
        title=f"Paired Users - {org_name}",
        description="\n".join(user_lines),
        color=EmbedColors.INFO,
        footer=f"Page {page}/{total_pages} | Total: {len(users)} users",
        timestamp=datetime.now(timezone.utc),
    )
