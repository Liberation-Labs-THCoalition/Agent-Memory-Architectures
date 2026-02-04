"""
Email adapter configuration for Kintsugi CMA.

This module provides comprehensive configuration classes for email integration,
supporting multiple email providers and both inbound (IMAP) and outbound (SMTP)
email handling.

Configuration Hierarchy:
    - EmailConfig: Top-level configuration container
        - IMAPConfig: Inbound email settings
        - SMTPConfig: Outbound email settings
        - EmailProvider: Provider-specific behavior

Security Considerations:
    - Credentials should be loaded from environment variables or secure vaults
    - TLS/SSL is enabled by default for all connections
    - Domain allowlisting prevents unauthorized sender access
    - Attachment size limits prevent resource exhaustion

Example:
    config = EmailConfig(
        org_id="org_nonprofit_123",
        imap=IMAPConfig(
            host="imap.gmail.com",
            username="grants@nonprofit.org",
            password=os.environ["IMAP_PASSWORD"]
        ),
        smtp=SMTPConfig(
            host="smtp.gmail.com",
            username="grants@nonprofit.org",
            password=os.environ["SMTP_PASSWORD"],
            from_address="grants@nonprofit.org",
            from_name="Grant Management Team"
        ),
        allowed_domains=["nonprofit.org", "partner.org"],
        require_pairing=True
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmailProvider(str, Enum):
    """
    Supported email service providers.

    Different providers may require different authentication methods
    or API configurations. The adapter uses this enum to determine
    which sending strategy to employ.

    Attributes:
        SMTP: Standard SMTP protocol (default, works with most providers)
        SENDGRID: SendGrid API for high-volume sending
        SES: Amazon Simple Email Service
        MAILGUN: Mailgun API for transactional email
    """

    SMTP = "smtp"
    SENDGRID = "sendgrid"
    SES = "ses"
    MAILGUN = "mailgun"


class IMAPAuthMethod(str, Enum):
    """
    IMAP authentication methods.

    Modern email providers may require OAuth2 authentication
    instead of plain passwords for security compliance.

    Attributes:
        PLAIN: Standard username/password authentication
        OAUTH2: OAuth 2.0 bearer token authentication
        XOAUTH2: Google-specific OAuth2 SASL mechanism
    """

    PLAIN = "plain"
    OAUTH2 = "oauth2"
    XOAUTH2 = "xoauth2"


@dataclass
class IMAPConfig:
    """
    Configuration for inbound email via IMAP.

    This configuration controls how the email adapter connects to
    an IMAP server to receive incoming emails. The adapter polls
    the specified folder at regular intervals for new messages.

    Attributes:
        host: IMAP server hostname (e.g., "imap.gmail.com")
        port: IMAP server port (default: 993 for SSL)
        username: Authentication username (usually email address)
        password: Authentication password or app-specific password
        use_ssl: Whether to use SSL/TLS connection (default: True)
        folder: Mailbox folder to monitor (default: "INBOX")
        check_interval_seconds: Polling interval in seconds (default: 60)
        auth_method: Authentication method to use (default: PLAIN)
        oauth_token: OAuth2 access token (when using OAuth2)
        idle_timeout_seconds: IMAP IDLE timeout (default: 1740, under 29 min)
        use_idle: Use IMAP IDLE instead of polling when supported
        mark_as_read: Automatically mark fetched emails as read
        delete_after_fetch: Delete emails after processing (careful!)
        search_criteria: IMAP search criteria (default: "UNSEEN")

    Example:
        imap_config = IMAPConfig(
            host="imap.gmail.com",
            port=993,
            username="nonprofit@gmail.com",
            password="app_specific_password",
            folder="INBOX",
            check_interval_seconds=30
        )
    """

    host: str
    port: int = 993
    username: str = ""
    password: str = ""
    use_ssl: bool = True
    folder: str = "INBOX"
    check_interval_seconds: int = 60
    auth_method: IMAPAuthMethod = IMAPAuthMethod.PLAIN
    oauth_token: str | None = None
    idle_timeout_seconds: int = 1740  # Just under 29 minutes (RFC 2177)
    use_idle: bool = False
    mark_as_read: bool = True
    delete_after_fetch: bool = False
    search_criteria: str = "UNSEEN"

    def __post_init__(self) -> None:
        """Validate IMAP configuration after initialization."""
        if not self.host:
            raise ValueError("IMAP host is required")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid IMAP port: {self.port}")

        if self.check_interval_seconds < 10:
            raise ValueError(
                "check_interval_seconds must be at least 10 to avoid rate limiting"
            )

        if self.auth_method in (IMAPAuthMethod.OAUTH2, IMAPAuthMethod.XOAUTH2):
            if not self.oauth_token:
                raise ValueError(
                    f"oauth_token required for auth_method {self.auth_method.value}"
                )

        if self.idle_timeout_seconds > 1740:
            raise ValueError(
                "idle_timeout_seconds should be under 29 minutes per RFC 2177"
            )

    @property
    def connection_string(self) -> str:
        """Generate a connection string for logging (no credentials)."""
        protocol = "imaps" if self.use_ssl else "imap"
        return f"{protocol}://{self.host}:{self.port}/{self.folder}"


@dataclass
class SMTPConfig:
    """
    Configuration for outbound email via SMTP.

    This configuration controls how the email adapter sends
    outgoing emails through an SMTP server. Supports both
    TLS and legacy SSL connections.

    Attributes:
        host: SMTP server hostname (e.g., "smtp.gmail.com")
        port: SMTP server port (default: 587 for TLS, 465 for SSL)
        username: Authentication username
        password: Authentication password or app-specific password
        use_tls: Whether to use STARTTLS (default: True)
        use_ssl: Whether to use implicit SSL (default: False)
        from_address: Default sender email address
        from_name: Display name for sender (default: "Kintsugi")
        reply_to: Optional reply-to address (if different from from_address)
        timeout_seconds: Connection timeout in seconds (default: 30)
        local_hostname: Local hostname for EHLO (optional)
        auth_method: Authentication method (default: PLAIN)
        oauth_token: OAuth2 access token (when using OAuth2)

    Note:
        use_tls and use_ssl are mutually exclusive. TLS (STARTTLS)
        is recommended for port 587; SSL for port 465.

    Example:
        smtp_config = SMTPConfig(
            host="smtp.gmail.com",
            port=587,
            username="nonprofit@gmail.com",
            password="app_specific_password",
            from_address="grants@nonprofit.org",
            from_name="Grant Management Team"
        )
    """

    host: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = ""
    from_name: str = "Kintsugi"
    reply_to: str | None = None
    timeout_seconds: int = 30
    local_hostname: str | None = None
    auth_method: IMAPAuthMethod = IMAPAuthMethod.PLAIN  # Reuse enum
    oauth_token: str | None = None

    def __post_init__(self) -> None:
        """Validate SMTP configuration after initialization."""
        if not self.host:
            raise ValueError("SMTP host is required")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid SMTP port: {self.port}")

        if self.use_tls and self.use_ssl:
            raise ValueError("use_tls and use_ssl are mutually exclusive")

        if self.timeout_seconds < 5:
            raise ValueError("timeout_seconds must be at least 5")

        if self.auth_method in (IMAPAuthMethod.OAUTH2, IMAPAuthMethod.XOAUTH2):
            if not self.oauth_token:
                raise ValueError(
                    f"oauth_token required for auth_method {self.auth_method.value}"
                )

    @property
    def connection_string(self) -> str:
        """Generate a connection string for logging (no credentials)."""
        if self.use_ssl:
            protocol = "smtps"
        elif self.use_tls:
            protocol = "smtp+tls"
        else:
            protocol = "smtp"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def sender_address(self) -> str:
        """Get formatted sender address with display name."""
        if self.from_name:
            return f"{self.from_name} <{self.from_address}>"
        return self.from_address


@dataclass
class EmailConfig:
    """
    Complete email configuration for the Kintsugi email adapter.

    This is the top-level configuration container that holds all
    email-related settings including IMAP, SMTP, provider selection,
    and organizational policies.

    Attributes:
        org_id: Organization identifier for this email configuration
        imap: IMAP configuration for receiving emails (optional)
        smtp: SMTP configuration for sending emails (optional)
        provider: Email provider for sending (affects API usage)
        allowed_domains: List of allowed sender domains (empty = allow all)
        blocked_domains: List of blocked sender domains
        auto_reply: Enable automatic reply to incoming emails
        require_pairing: Require user pairing before processing emails
        max_attachment_mb: Maximum attachment size in megabytes
        save_attachments: Whether to save attachments locally
        attachment_storage_path: Path for saving attachments
        thread_tracking: Enable email thread/conversation tracking
        bounce_handling: Enable bounce message detection and handling
        unsubscribe_handling: Honor unsubscribe requests automatically
        signature: Default email signature to append
        api_key: API key for cloud providers (SendGrid, Mailgun, etc.)
        api_secret: API secret for cloud providers (if needed)
        region: AWS region for SES (if using SES)
        metadata: Additional provider-specific configuration

    Example:
        config = EmailConfig(
            org_id="org_nonprofit_123",
            imap=IMAPConfig(host="imap.example.com", ...),
            smtp=SMTPConfig(host="smtp.example.com", ...),
            allowed_domains=["nonprofit.org", "partner.org"],
            require_pairing=True,
            max_attachment_mb=25
        )
    """

    org_id: str
    imap: IMAPConfig | None = None
    smtp: SMTPConfig | None = None
    provider: EmailProvider = EmailProvider.SMTP
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    auto_reply: bool = False
    require_pairing: bool = True
    max_attachment_mb: int = 10
    save_attachments: bool = False
    attachment_storage_path: str | None = None
    thread_tracking: bool = True
    bounce_handling: bool = True
    unsubscribe_handling: bool = True
    signature: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    region: str | None = None  # For AWS SES
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate email configuration after initialization."""
        if not self.org_id:
            raise ValueError("org_id is required")

        if self.max_attachment_mb < 1:
            raise ValueError("max_attachment_mb must be at least 1")

        if self.max_attachment_mb > 100:
            raise ValueError("max_attachment_mb should not exceed 100MB")

        if self.save_attachments and not self.attachment_storage_path:
            raise ValueError(
                "attachment_storage_path required when save_attachments is True"
            )

        # Validate provider-specific requirements
        if self.provider == EmailProvider.SENDGRID and not self.api_key:
            raise ValueError("api_key required for SendGrid provider")

        if self.provider == EmailProvider.SES:
            if not self.region:
                raise ValueError("region required for SES provider")

        if self.provider == EmailProvider.MAILGUN:
            if not self.api_key:
                raise ValueError("api_key required for Mailgun provider")

        # Normalize domain lists to lowercase
        self.allowed_domains = [d.lower().strip() for d in self.allowed_domains]
        self.blocked_domains = [d.lower().strip() for d in self.blocked_domains]

    def is_domain_allowed(self, domain: str) -> bool:
        """
        Check if an email domain is allowed based on configuration.

        Args:
            domain: The email domain to check (e.g., "example.com")

        Returns:
            True if the domain is allowed, False otherwise
        """
        domain = domain.lower().strip()

        # Blocked domains take precedence
        if domain in self.blocked_domains:
            return False

        # If no allowed domains specified, allow all (except blocked)
        if not self.allowed_domains:
            return True

        # Check if domain is in allowed list
        return domain in self.allowed_domains

    def is_email_allowed(self, email_address: str) -> bool:
        """
        Check if an email address is allowed based on domain rules.

        Args:
            email_address: Full email address to check

        Returns:
            True if the email's domain is allowed, False otherwise
        """
        if "@" not in email_address:
            return False

        domain = email_address.split("@")[-1]
        return self.is_domain_allowed(domain)

    @property
    def max_attachment_bytes(self) -> int:
        """Get maximum attachment size in bytes."""
        return self.max_attachment_mb * 1024 * 1024

    @property
    def can_receive(self) -> bool:
        """Check if email receiving is configured."""
        return self.imap is not None

    @property
    def can_send(self) -> bool:
        """Check if email sending is configured."""
        if self.provider == EmailProvider.SMTP:
            return self.smtp is not None
        # Cloud providers use API keys
        return self.api_key is not None

    def to_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Args:
            include_secrets: Whether to include sensitive fields

        Returns:
            Dictionary representation of configuration
        """
        result = {
            "org_id": self.org_id,
            "provider": self.provider.value,
            "allowed_domains": self.allowed_domains,
            "blocked_domains": self.blocked_domains,
            "auto_reply": self.auto_reply,
            "require_pairing": self.require_pairing,
            "max_attachment_mb": self.max_attachment_mb,
            "thread_tracking": self.thread_tracking,
            "bounce_handling": self.bounce_handling,
            "can_receive": self.can_receive,
            "can_send": self.can_send,
        }

        if self.imap:
            result["imap"] = {
                "host": self.imap.host,
                "port": self.imap.port,
                "folder": self.imap.folder,
                "use_ssl": self.imap.use_ssl,
                "check_interval_seconds": self.imap.check_interval_seconds,
            }
            if include_secrets:
                result["imap"]["username"] = self.imap.username

        if self.smtp:
            result["smtp"] = {
                "host": self.smtp.host,
                "port": self.smtp.port,
                "from_address": self.smtp.from_address,
                "from_name": self.smtp.from_name,
                "use_tls": self.smtp.use_tls,
            }
            if include_secrets:
                result["smtp"]["username"] = self.smtp.username

        return result

    def __repr__(self) -> str:
        """Return a safe string representation (no secrets)."""
        return (
            f"EmailConfig(org_id={self.org_id!r}, "
            f"provider={self.provider.value!r}, "
            f"can_receive={self.can_receive}, "
            f"can_send={self.can_send})"
        )
