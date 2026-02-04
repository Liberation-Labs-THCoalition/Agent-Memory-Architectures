"""Kintsugi Adapters -- chat platform integrations.

This package provides adapters for connecting Kintsugi to various chat platforms:

- **shared**: Base classes, DM pairing, and allowlist management
- **slack**: Slack Bot adapter with Bolt SDK patterns
- **discord**: Discord Bot adapter with discord.py patterns
- **webchat**: Embeddable web chat widget
- **email**: Email adapter with IMAP/SMTP and notification support

Example usage::

    from kintsugi.adapters.shared import PairingManager, PairingConfig
    from kintsugi.adapters.slack import SlackAdapter, SlackConfig
    from kintsugi.adapters.discord import DiscordAdapter, DiscordConfig
    from kintsugi.adapters.webchat import WebChatHandler, WebChatConfig
    from kintsugi.adapters.email import EmailAdapter, EmailConfig

    # Create shared pairing manager
    pairing = PairingManager(PairingConfig(expiration_minutes=15))

    # Initialize platform adapters
    slack = SlackAdapter(
        SlackConfig(bot_token="xoxb-...", signing_secret="..."),
        pairing=pairing,
    )
    discord = DiscordAdapter(
        DiscordConfig(bot_token="...", application_id="..."),
        pairing=pairing,
    )
    webchat = WebChatHandler(WebChatConfig(org_id="..."))
    email = EmailAdapter(
        EmailConfig(org_id="...", smtp=SMTPConfig(host="smtp.example.com")),
        pairing=pairing,
    )
"""

from kintsugi.adapters.shared import (
    # Base
    AdapterMessage,
    AdapterPlatform,
    AdapterResponse,
    BaseAdapter,
    # Pairing
    PairingCode,
    PairingConfig,
    PairingManager,
    PairingStatus,
    # Allowlist
    AllowlistEntry,
    AllowlistStore,
    InMemoryAllowlistStore,
)

from kintsugi.adapters.email import (
    # Adapter
    EmailAdapter,
    EmailAdapterError,
    # Config
    EmailConfig,
    EmailProvider,
    IMAPConfig,
    SMTPConfig,
    # Parser
    EmailParser,
    ParsedEmail,
    EmailAttachment,
    # Notifications
    NotificationManager,
    GrantDeadlineNotification,
    ReportDelivery,
    # Templates
    TemplateRenderer,
    EmailTemplate,
)

__all__ = [
    # Shared base
    "AdapterMessage",
    "AdapterPlatform",
    "AdapterResponse",
    "BaseAdapter",
    # Pairing
    "PairingCode",
    "PairingConfig",
    "PairingManager",
    "PairingStatus",
    # Allowlist
    "AllowlistEntry",
    "AllowlistStore",
    "InMemoryAllowlistStore",
    # Email adapter
    "EmailAdapter",
    "EmailAdapterError",
    "EmailConfig",
    "EmailProvider",
    "IMAPConfig",
    "SMTPConfig",
    "EmailParser",
    "ParsedEmail",
    "EmailAttachment",
    "NotificationManager",
    "GrantDeadlineNotification",
    "ReportDelivery",
    "TemplateRenderer",
    "EmailTemplate",
]
