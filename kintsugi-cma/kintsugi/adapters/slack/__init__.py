"""Slack Bot adapter for Kintsugi CMA.

This package provides integration between Kintsugi CMA and Slack workspaces,
enabling users to interact with Kintsugi through Slack messages, commands,
and interactive components.

Main Components:
    - SlackAdapter: Core adapter for sending/receiving messages
    - SlackConfig: Configuration dataclass for the adapter
    - SlackEventHandler: Handlers for Slack events and commands
    - SlackInteractionHandler: Handlers for interactive components
    - OAuthHandler: OAuth 2.0 installation flow
    - SlackInstallation: Installation data storage

Example:
    >>> from kintsugi.adapters.slack import SlackAdapter, SlackConfig
    >>> from kintsugi.adapters.shared import PairingManager
    >>>
    >>> config = SlackConfig(
    ...     bot_token="xoxb-...",
    ...     signing_secret="...",
    ...     app_token="xapp-...",  # For Socket Mode
    ... )
    >>> pairing = PairingManager(...)
    >>> adapter = SlackAdapter(config, pairing)
    >>>
    >>> # Send a message
    >>> await adapter.send_dm("U12345", AdapterResponse(text="Hello!"))

Block Kit Utilities:
    The blocks module provides functions for building Slack Block Kit messages:

    >>> from kintsugi.adapters.slack import (
    ...     pairing_request_blocks,
    ...     agent_response_blocks,
    ...     error_blocks,
    ...     help_blocks,
    ... )

OAuth Installation:
    For multi-workspace deployments, use the OAuth handler:

    >>> from kintsugi.adapters.slack import OAuthHandler
    >>> handler = OAuthHandler(
    ...     client_id="...",
    ...     client_secret="...",
    ...     redirect_uri="https://example.com/callback",
    ... )
    >>> auth_url = handler.get_authorize_url(state="...")
"""

from .blocks import (
    agent_response_blocks,
    confirmation_blocks,
    error_blocks,
    help_blocks,
    loading_blocks,
    pairing_approval_blocks,
    pairing_request_blocks,
    success_blocks,
)
from .bot import SlackAdapter
from .config import SlackConfig
from .handlers import SlackEventHandler, SlackInteractionHandler
from .oauth import (
    InstallationStore,
    OAuthError,
    OAuthHandler,
    OAuthState,
    SlackInstallation,
)

__all__ = [
    # Core adapter
    "SlackAdapter",
    "SlackConfig",
    # Event handlers
    "SlackEventHandler",
    "SlackInteractionHandler",
    # OAuth
    "OAuthHandler",
    "OAuthError",
    "OAuthState",
    "SlackInstallation",
    "InstallationStore",
    # Block Kit builders
    "pairing_request_blocks",
    "pairing_approval_blocks",
    "agent_response_blocks",
    "error_blocks",
    "help_blocks",
    "loading_blocks",
    "success_blocks",
    "confirmation_blocks",
]
