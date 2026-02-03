"""WebChat adapter configuration.

This module defines the configuration dataclass for the WebChat widget adapter,
including organization settings, authentication requirements, rate limiting,
and widget customization options.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WebChatConfig:
    """Configuration for the WebChat widget adapter.

    Attributes:
        org_id: Organization ID that owns this widget configuration.
        allowed_origins: List of allowed CORS origins for WebSocket connections.
            Defaults to ["*"] which allows all origins. For production, specify
            exact domains like ["https://example.com", "https://www.example.com"].
        require_auth: If True, require a valid session token before allowing
            WebSocket connections. Useful for logged-in user chat experiences.
        session_timeout_minutes: Duration in minutes before inactive sessions
            are automatically cleaned up. Defaults to 60 minutes.
        max_message_length: Maximum allowed length for incoming chat messages.
            Messages exceeding this limit will be rejected. Defaults to 4000 chars.
        rate_limit_messages_per_minute: Maximum messages allowed per session
            per minute. Prevents abuse and ensures fair usage. Defaults to 20.
        widget_title: Title displayed in the chat widget header.
        widget_subtitle: Optional subtitle displayed below the title.
        primary_color: Primary color for the widget theme in hex format.
            Defaults to Kintsugi purple (#9B59B6).
        show_powered_by: Whether to display "Powered by Kintsugi" branding.
    """

    org_id: str
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    require_auth: bool = False
    session_timeout_minutes: int = 60
    max_message_length: int = 4000
    rate_limit_messages_per_minute: int = 20
    widget_title: str = "Chat with us"
    widget_subtitle: str | None = None
    primary_color: str = "#9B59B6"  # Kintsugi purple
    show_powered_by: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.org_id:
            raise ValueError("org_id is required")
        if self.session_timeout_minutes < 1:
            raise ValueError("session_timeout_minutes must be at least 1")
        if self.max_message_length < 1:
            raise ValueError("max_message_length must be at least 1")
        if self.rate_limit_messages_per_minute < 1:
            raise ValueError("rate_limit_messages_per_minute must be at least 1")
        if not self.primary_color.startswith("#"):
            raise ValueError("primary_color must be a hex color (e.g., '#9B59B6')")

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if a specific origin is allowed for WebSocket connections.

        Args:
            origin: The Origin header value from the WebSocket connection request.

        Returns:
            True if the origin is allowed, False otherwise.
        """
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins

    def validate_message_length(self, content: str) -> bool:
        """Check if a message is within the allowed length limit.

        Args:
            content: The message content to validate.

        Returns:
            True if the message is within limits, False otherwise.
        """
        return len(content) <= self.max_message_length
