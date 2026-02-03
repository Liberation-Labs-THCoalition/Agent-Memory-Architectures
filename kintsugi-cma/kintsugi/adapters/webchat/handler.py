"""WebChat WebSocket message handler.

This module provides the core handler for WebChat WebSocket connections,
managing sessions, rate limiting, and message processing for the embeddable
chat widget.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kintsugi.adapters.shared import AdapterMessage

from kintsugi.adapters.webchat.config import WebChatConfig


class WebChatMessageType(str, Enum):
    """Message types for WebChat WebSocket protocol.

    These types define the different kinds of messages that can be sent
    between the client widget and the server.
    """

    CONNECT = "connect"
    """Initial connection handshake message."""

    DISCONNECT = "disconnect"
    """Session termination message."""

    MESSAGE = "message"
    """User chat message."""

    TYPING = "typing"
    """User typing indicator."""

    HISTORY = "history"
    """Request for chat history."""

    ERROR = "error"
    """Error response message."""

    AGENT_RESPONSE = "agent_response"
    """Agent's response to user message."""

    AGENT_TYPING = "agent_typing"
    """Agent typing indicator for streaming responses."""


@dataclass
class WebChatSession:
    """Represents an active WebChat session.

    Tracks the state and metadata for a single chat widget session,
    including connection timing, user identification, and message counts
    for rate limiting.

    Attributes:
        session_id: Unique identifier for this session.
        org_id: Organization ID this session belongs to.
        connected_at: Timestamp when the session was created.
        last_activity: Timestamp of the most recent activity.
        user_identifier: Optional identifier for the user (e.g., email, user ID).
        metadata: Additional metadata associated with the session.
        message_count: Total number of messages sent in this session.
    """

    session_id: str
    org_id: str
    connected_at: datetime
    last_activity: datetime
    user_identifier: str | None = None
    metadata: dict = field(default_factory=dict)
    message_count: int = 0

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if this session has expired due to inactivity.

        Args:
            timeout_minutes: The session timeout duration in minutes.

        Returns:
            True if the session has expired, False otherwise.
        """
        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now(timezone.utc) > expiry_time

    def update_activity(self) -> None:
        """Update the last activity timestamp to now."""
        self.last_activity = datetime.now(timezone.utc)

    def increment_message_count(self) -> None:
        """Increment the message counter for this session."""
        self.message_count += 1


class WebChatHandler:
    """Handles WebChat WebSocket connections and message processing.

    This class manages the lifecycle of WebChat sessions, including creation,
    message handling, rate limiting, and cleanup. It integrates with the
    Kintsugi adapter infrastructure to route messages to the appropriate
    agent processing pipeline.

    Attributes:
        _config: The WebChat configuration for this handler.
        _sessions: Dictionary mapping session IDs to active sessions.
        _rate_limits: Dictionary mapping session IDs to lists of message timestamps.
    """

    def __init__(self, config: WebChatConfig) -> None:
        """Initialize the WebChat handler.

        Args:
            config: Configuration for this WebChat handler instance.
        """
        self._config = config
        self._sessions: dict[str, WebChatSession] = {}
        self._rate_limits: dict[str, list[datetime]] = {}

    @property
    def config(self) -> WebChatConfig:
        """Get the current configuration."""
        return self._config

    @property
    def active_session_count(self) -> int:
        """Get the number of currently active sessions."""
        return len(self._sessions)

    def create_session(
        self,
        org_id: str,
        user_identifier: str | None = None,
        metadata: dict | None = None,
    ) -> WebChatSession:
        """Create a new chat session.

        Generates a unique session ID and initializes a new WebChatSession
        with the current timestamp.

        Args:
            org_id: Organization ID for this session.
            user_identifier: Optional identifier for the user.
            metadata: Optional additional metadata for the session.

        Returns:
            The newly created WebChatSession.
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)

        session = WebChatSession(
            session_id=session_id,
            org_id=org_id,
            connected_at=now,
            last_activity=now,
            user_identifier=user_identifier,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session
        self._rate_limits[session_id] = []

        return session

    def get_session(self, session_id: str) -> WebChatSession | None:
        """Retrieve a session by its ID.

        Args:
            session_id: The unique session identifier.

        Returns:
            The WebChatSession if found and not expired, None otherwise.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # Check if session has expired
        if session.is_expired(self._config.session_timeout_minutes):
            self.end_session(session_id)
            return None

        return session

    def end_session(self, session_id: str) -> bool:
        """End and cleanup a session.

        Removes the session from active sessions and clears its rate limit
        tracking data.

        Args:
            session_id: The session ID to terminate.

        Returns:
            True if the session was found and removed, False otherwise.
        """
        if session_id not in self._sessions:
            return False

        del self._sessions[session_id]
        self._rate_limits.pop(session_id, None)

        return True

    async def handle_message(
        self,
        session_id: str,
        content: str,
    ) -> dict:
        """Process an incoming chat message.

        Validates the session, checks rate limits, and processes the message.
        Updates session activity timestamp and message count.

        Args:
            session_id: The session ID of the sender.
            content: The message content.

        Returns:
            A response dictionary with type and payload/error fields.

        Raises:
            ValueError: If the session is invalid or expired.
        """
        # Validate session exists
        session = self.get_session(session_id)
        if session is None:
            return {
                "type": WebChatMessageType.ERROR.value,
                "error": "Invalid or expired session",
                "code": "SESSION_INVALID",
            }

        # Validate message length
        if not self._config.validate_message_length(content):
            return {
                "type": WebChatMessageType.ERROR.value,
                "error": f"Message exceeds maximum length of {self._config.max_message_length} characters",
                "code": "MESSAGE_TOO_LONG",
            }

        # Check rate limit
        if not self.check_rate_limit(session_id):
            return {
                "type": WebChatMessageType.ERROR.value,
                "error": f"Rate limit exceeded. Maximum {self._config.rate_limit_messages_per_minute} messages per minute.",
                "code": "RATE_LIMIT_EXCEEDED",
            }

        # Update session activity
        session.update_activity()
        session.increment_message_count()

        # Record this message for rate limiting
        self._rate_limits[session_id].append(datetime.now(timezone.utc))

        # Return acknowledgment - actual agent processing is handled by the route
        return {
            "type": WebChatMessageType.MESSAGE.value,
            "session_id": session_id,
            "org_id": session.org_id,
            "content": content,
            "timestamp": session.last_activity.isoformat(),
            "message_number": session.message_count,
        }

    def check_rate_limit(self, session_id: str) -> bool:
        """Check if a session is within the rate limit.

        Removes expired timestamps (older than 1 minute) and checks if
        the session has exceeded the configured messages per minute.

        Args:
            session_id: The session ID to check.

        Returns:
            True if the session is within limits, False if rate limited.
        """
        if session_id not in self._rate_limits:
            return True

        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Filter out timestamps older than one minute
        recent_messages = [
            ts for ts in self._rate_limits[session_id]
            if ts > one_minute_ago
        ]
        self._rate_limits[session_id] = recent_messages

        # Check if within limit
        return len(recent_messages) < self._config.rate_limit_messages_per_minute

    def cleanup_expired_sessions(self) -> int:
        """Remove all sessions that have exceeded the timeout.

        Iterates through all active sessions and removes those that have
        been inactive longer than the configured timeout.

        Returns:
            The number of sessions that were cleaned up.
        """
        expired_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if session.is_expired(self._config.session_timeout_minutes)
        ]

        for session_id in expired_sessions:
            self.end_session(session_id)

        return len(expired_sessions)

    def normalize_to_adapter_message(
        self,
        session: WebChatSession,
        content: str,
    ) -> "AdapterMessage":
        """Convert a WebChat message to the standard AdapterMessage format.

        Transforms a WebChat-specific message into the shared adapter
        infrastructure's AdapterMessage format for unified processing.

        Args:
            session: The session that sent the message.
            content: The message content.

        Returns:
            An AdapterMessage instance ready for agent processing.
        """
        # Import here to avoid circular imports
        from kintsugi.adapters.shared import AdapterMessage, AdapterPlatform

        return AdapterMessage(
            platform=AdapterPlatform.WEBCHAT,
            org_id=session.org_id,
            channel_id=session.session_id,
            user_id=session.user_identifier or f"webchat:{session.session_id}",
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_id": session.session_id,
                "message_count": session.message_count,
                "session_metadata": session.metadata,
            },
        )

    def get_session_stats(self) -> dict:
        """Get statistics about current sessions.

        Returns:
            Dictionary containing session statistics.
        """
        now = datetime.now(timezone.utc)
        sessions_by_org: dict[str, int] = {}
        total_messages = 0

        for session in self._sessions.values():
            sessions_by_org[session.org_id] = sessions_by_org.get(session.org_id, 0) + 1
            total_messages += session.message_count

        return {
            "total_sessions": len(self._sessions),
            "sessions_by_org": sessions_by_org,
            "total_messages": total_messages,
            "timestamp": now.isoformat(),
        }
