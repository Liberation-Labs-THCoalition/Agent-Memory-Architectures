"""Slack OAuth installation flow for Kintsugi CMA.

This module handles the OAuth 2.0 installation flow for Slack apps,
including authorization URL generation, token exchange, and installation
storage.

OAuth Flow Reference: https://api.slack.com/authentication/oauth-v2
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode


@dataclass
class SlackInstallation:
    """Represents a Slack workspace installation.

    Stores the tokens and metadata from a successful OAuth installation.
    This data is persisted to allow the bot to function in the workspace.

    Attributes:
        team_id: The Slack workspace (team) ID.
        team_name: The human-readable workspace name.
        bot_token: The bot OAuth token (xoxb-...) for API calls.
        bot_user_id: The bot's user ID in this workspace.
        installed_at: Timestamp when the installation occurred.
        installer_user_id: The user who installed the app.
        org_id: Optional linked Kintsugi organization ID.
        enterprise_id: Optional Slack Enterprise Grid org ID.
        enterprise_name: Optional Enterprise Grid org name.
        is_enterprise_install: Whether this is an org-wide installation.
        incoming_webhook: Optional webhook configuration if installed.

    Example:
        >>> installation = SlackInstallation(
        ...     team_id="T12345",
        ...     team_name="Acme Corp",
        ...     bot_token="xoxb-...",
        ...     bot_user_id="U98765",
        ...     installed_at=datetime.now(timezone.utc),
        ...     installer_user_id="U11111",
        ...     org_id="org_acme",
        ... )
    """

    team_id: str
    team_name: str
    bot_token: str
    bot_user_id: str
    installed_at: datetime
    installer_user_id: str
    org_id: str | None = None
    enterprise_id: str | None = None
    enterprise_name: str | None = None
    is_enterprise_install: bool = False
    incoming_webhook: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert installation to dictionary for storage.

        Returns:
            Dictionary representation of the installation.
        """
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "bot_token": self.bot_token,
            "bot_user_id": self.bot_user_id,
            "installed_at": self.installed_at.isoformat(),
            "installer_user_id": self.installer_user_id,
            "org_id": self.org_id,
            "enterprise_id": self.enterprise_id,
            "enterprise_name": self.enterprise_name,
            "is_enterprise_install": self.is_enterprise_install,
            "incoming_webhook": self.incoming_webhook,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlackInstallation:
        """Create installation from dictionary.

        Args:
            data: Dictionary representation of an installation.

        Returns:
            SlackInstallation instance.
        """
        installed_at = data.get("installed_at")
        if isinstance(installed_at, str):
            installed_at = datetime.fromisoformat(installed_at)

        return cls(
            team_id=data["team_id"],
            team_name=data["team_name"],
            bot_token=data["bot_token"],
            bot_user_id=data["bot_user_id"],
            installed_at=installed_at or datetime.now(timezone.utc),
            installer_user_id=data["installer_user_id"],
            org_id=data.get("org_id"),
            enterprise_id=data.get("enterprise_id"),
            enterprise_name=data.get("enterprise_name"),
            is_enterprise_install=data.get("is_enterprise_install", False),
            incoming_webhook=data.get("incoming_webhook"),
        )


@dataclass
class OAuthState:
    """OAuth state parameter for CSRF protection.

    Stores state information used during the OAuth flow to prevent
    CSRF attacks and associate installations with organizations.

    Attributes:
        state: Random state string for CSRF protection.
        org_id: Optional Kintsugi org ID to associate with installation.
        redirect_uri: The redirect URI used in this flow.
        created_at: When this state was created (for expiration).
    """

    state: str
    org_id: str | None = None
    redirect_uri: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def generate(cls, org_id: str | None = None, redirect_uri: str | None = None) -> OAuthState:
        """Generate a new OAuth state.

        Args:
            org_id: Optional org ID to associate.
            redirect_uri: The redirect URI for this flow.

        Returns:
            New OAuthState instance with random state string.
        """
        return cls(
            state=secrets.token_urlsafe(32),
            org_id=org_id,
            redirect_uri=redirect_uri,
        )

    def is_expired(self, max_age_seconds: int = 600) -> bool:
        """Check if this state has expired.

        Args:
            max_age_seconds: Maximum age in seconds (default 10 minutes).

        Returns:
            True if the state has expired.
        """
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > max_age_seconds


class OAuthHandler:
    """Handle Slack OAuth installation flow.

    Manages the OAuth 2.0 flow for installing the Slack app into
    workspaces, including authorization URL generation and token exchange.

    Attributes:
        client_id: The Slack app's client ID.
        client_secret: The Slack app's client secret.
        redirect_uri: The OAuth callback URL.

    Example:
        >>> handler = OAuthHandler(
        ...     client_id="123.456",
        ...     client_secret="abc...",
        ...     redirect_uri="https://kintsugi.ai/slack/oauth/callback",
        ... )
        >>> url = handler.get_authorize_url(state="xyz123")
    """

    AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
    TOKEN_URL = "https://slack.com/api/oauth.v2.access"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ) -> None:
        """Initialize the OAuth handler.

        Args:
            client_id: The Slack app's client ID.
            client_secret: The Slack app's client secret.
            redirect_uri: The OAuth callback URL.
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri

    @property
    def client_id(self) -> str:
        """Get the Slack client ID."""
        return self._client_id

    @property
    def redirect_uri(self) -> str:
        """Get the OAuth redirect URI."""
        return self._redirect_uri

    def get_authorize_url(
        self,
        state: str,
        scopes: list[str] | None = None,
        user_scopes: list[str] | None = None,
    ) -> str:
        """Generate OAuth authorize URL.

        Creates the URL to redirect users to for Slack OAuth authorization.

        Args:
            state: Random state string for CSRF protection.
            scopes: Optional list of bot scopes. Uses defaults if not provided.
            user_scopes: Optional list of user scopes for user token.

        Returns:
            The full authorization URL.

        Example:
            >>> url = handler.get_authorize_url("state123")
            >>> # Redirect user to this URL
        """
        if scopes is None:
            scopes = self.default_scopes()

        params = {
            "client_id": self._client_id,
            "scope": ",".join(scopes),
            "redirect_uri": self._redirect_uri,
            "state": state,
        }

        if user_scopes:
            params["user_scope"] = ",".join(user_scopes)

        return f"{self.AUTHORIZE_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> SlackInstallation:
        """Exchange OAuth code for tokens.

        Completes the OAuth flow by exchanging the authorization code
        for access tokens and installation data.

        Args:
            code: The authorization code from Slack's redirect.

        Returns:
            SlackInstallation containing tokens and workspace info.

        Raises:
            OAuthError: If the token exchange fails.

        Example:
            >>> installation = await handler.exchange_code("abc123...")
            >>> print(f"Installed in {installation.team_name}")
        """
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                },
            ) as response:
                data = await response.json()

        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            raise OAuthError(f"Token exchange failed: {error}")

        # Extract installation data
        team = data.get("team", {})
        bot = data.get("authed_user", {})
        enterprise = data.get("enterprise")
        incoming_webhook = data.get("incoming_webhook")

        return SlackInstallation(
            team_id=team.get("id", data.get("team_id", "")),
            team_name=team.get("name", data.get("team_name", "")),
            bot_token=data.get("access_token", ""),
            bot_user_id=bot.get("id", data.get("bot_user_id", "")),
            installed_at=datetime.now(timezone.utc),
            installer_user_id=data.get("authed_user", {}).get("id", ""),
            enterprise_id=enterprise.get("id") if enterprise else None,
            enterprise_name=enterprise.get("name") if enterprise else None,
            is_enterprise_install=data.get("is_enterprise_install", False),
            incoming_webhook=incoming_webhook,
        )

    @staticmethod
    def default_scopes() -> list[str]:
        """Return recommended bot scopes.

        Returns the list of OAuth scopes recommended for Kintsugi's
        Slack integration functionality.

        Returns:
            List of scope strings.

        Scopes included:
            - channels:history - Read messages in public channels
            - channels:read - View basic channel info
            - chat:write - Send messages as the bot
            - commands - Add and receive slash commands
            - im:history - Read DM messages with the bot
            - im:read - View basic DM info
            - im:write - Start DMs with users
            - users:read - View basic user info
        """
        return [
            "channels:history",
            "channels:read",
            "chat:write",
            "commands",
            "im:history",
            "im:read",
            "im:write",
            "users:read",
        ]

    @staticmethod
    def optional_scopes() -> list[str]:
        """Return optional bot scopes for extended functionality.

        Returns:
            List of optional scope strings.

        Scopes included:
            - app_mentions:read - Receive @mention events
            - channels:join - Join public channels
            - groups:history - Read messages in private channels
            - groups:read - View basic private channel info
            - mpim:history - Read group DM messages
            - mpim:read - View basic group DM info
            - reactions:read - View emoji reactions
            - reactions:write - Add emoji reactions
            - users:read.email - View user email addresses
        """
        return [
            "app_mentions:read",
            "channels:join",
            "groups:history",
            "groups:read",
            "mpim:history",
            "mpim:read",
            "reactions:read",
            "reactions:write",
            "users:read.email",
        ]


class OAuthError(Exception):
    """Exception raised for OAuth flow errors.

    Attributes:
        message: The error message.
        error_code: Optional Slack error code.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """Initialize the OAuth error.

        Args:
            message: The error message.
            error_code: Optional Slack error code.
        """
        super().__init__(message)
        self.error_code = error_code


class InstallationStore:
    """Abstract base for storing Slack installations.

    Subclass this to implement persistence for SlackInstallation objects.
    Implementations might use databases, file storage, or cloud services.

    Example:
        >>> class RedisInstallationStore(InstallationStore):
        ...     async def save(self, installation):
        ...         await self.redis.set(
        ...             f"slack:install:{installation.team_id}",
        ...             installation.to_dict()
        ...         )
    """

    async def save(self, installation: SlackInstallation) -> None:
        """Save an installation.

        Args:
            installation: The installation to save.
        """
        raise NotImplementedError

    async def find_by_team(self, team_id: str) -> SlackInstallation | None:
        """Find installation by team ID.

        Args:
            team_id: The Slack team ID.

        Returns:
            The installation if found, None otherwise.
        """
        raise NotImplementedError

    async def find_by_enterprise(
        self,
        enterprise_id: str,
    ) -> SlackInstallation | None:
        """Find installation by enterprise ID.

        Args:
            enterprise_id: The Slack Enterprise Grid org ID.

        Returns:
            The installation if found, None otherwise.
        """
        raise NotImplementedError

    async def delete(self, team_id: str) -> bool:
        """Delete an installation.

        Args:
            team_id: The Slack team ID.

        Returns:
            True if deleted, False if not found.
        """
        raise NotImplementedError

    async def find_by_org(self, org_id: str) -> list[SlackInstallation]:
        """Find all installations for a Kintsugi organization.

        Args:
            org_id: The Kintsugi organization ID.

        Returns:
            List of installations linked to this org.
        """
        raise NotImplementedError
