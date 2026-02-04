"""
Email adapter package for Kintsugi CMA.

This package provides comprehensive email integration for the Kintsugi
Contextual Memory Architecture, enabling email-based communication for
grant management workflows.

Components:
    - EmailAdapter: Main adapter for sending/receiving emails
    - EmailParser: Parse raw emails into structured format
    - NotificationManager: Schedule and send notifications
    - TemplateRenderer: Render email templates

Configuration:
    - EmailConfig: Top-level configuration
    - IMAPConfig: Inbound email settings
    - SMTPConfig: Outbound email settings
    - EmailProvider: Supported providers (SMTP, SendGrid, SES, Mailgun)

Data Types:
    - ParsedEmail: Structured email representation
    - EmailAttachment: Email attachment metadata
    - GrantDeadlineNotification: Grant deadline reminder
    - ReportDelivery: Report delivery request
    - EmailTemplate: Email template definition

Exceptions:
    - EmailAdapterError: Base adapter exception
    - ConnectionError: Connection failed
    - SendError: Send failed
    - FetchError: Fetch failed
    - TemplateError: Template error
    - TemplateNotFoundError: Template not found
    - TemplateValidationError: Template validation failed

Example:
    from kintsugi.adapters.email import (
        EmailAdapter,
        EmailConfig,
        IMAPConfig,
        SMTPConfig,
        NotificationManager,
        GrantDeadlineNotification,
    )

    # Configure email
    config = EmailConfig(
        org_id="org_123",
        imap=IMAPConfig(host="imap.gmail.com", ...),
        smtp=SMTPConfig(host="smtp.gmail.com", ...)
    )

    # Create adapter
    adapter = EmailAdapter(config)
    await adapter.connect()

    # Create notification manager
    notifications = NotificationManager(adapter)

    # Send grant reminder
    reminder = GrantDeadlineNotification(
        grant_name="Community Grant",
        deadline=datetime(2024, 3, 15),
        days_remaining=7
    )
    await notifications.send_grant_reminder(reminder, ["team@org.com"])
"""

# Configuration
from .config import (
    EmailProvider,
    IMAPAuthMethod,
    IMAPConfig,
    SMTPConfig,
    EmailConfig,
)

# Parser
from .parser import (
    EmailAttachment,
    ParsedEmail,
    EmailParser,
)

# Adapter
from .adapter import (
    EmailAdapter,
    EmailAdapterError,
    ConnectionError,
    SendError,
    FetchError,
)

# Notifications
from .notifications import (
    GrantDeadlineNotification,
    ReportDelivery,
    ScheduledNotification,
    NotificationManager,
)

# Templates
from .templates import (
    EmailTemplate,
    TemplateRenderer,
    TemplateError,
    TemplateNotFoundError,
    TemplateValidationError,
    GRANT_REMINDER_TEMPLATE,
    PAIRING_CODE_TEMPLATE,
    REPORT_DELIVERY_TEMPLATE,
    WELCOME_TEMPLATE,
    DEADLINE_URGENT_TEMPLATE,
    GRANT_APPROVED_TEMPLATE,
    DEFAULT_TEMPLATES,
)


__all__ = [
    # Configuration
    "EmailProvider",
    "IMAPAuthMethod",
    "IMAPConfig",
    "SMTPConfig",
    "EmailConfig",

    # Parser
    "EmailAttachment",
    "ParsedEmail",
    "EmailParser",

    # Adapter
    "EmailAdapter",
    "EmailAdapterError",
    "ConnectionError",
    "SendError",
    "FetchError",

    # Notifications
    "GrantDeadlineNotification",
    "ReportDelivery",
    "ScheduledNotification",
    "NotificationManager",

    # Templates
    "EmailTemplate",
    "TemplateRenderer",
    "TemplateError",
    "TemplateNotFoundError",
    "TemplateValidationError",
    "GRANT_REMINDER_TEMPLATE",
    "PAIRING_CODE_TEMPLATE",
    "REPORT_DELIVERY_TEMPLATE",
    "WELCOME_TEMPLATE",
    "DEADLINE_URGENT_TEMPLATE",
    "GRANT_APPROVED_TEMPLATE",
    "DEFAULT_TEMPLATES",
]
