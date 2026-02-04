"""
Email adapter for Kintsugi CMA.

This module provides the main EmailAdapter class that integrates email
communication with the Kintsugi CMA system. It supports both receiving
emails via IMAP and sending emails via SMTP or cloud providers.

Features:
    - IMAP polling or IDLE for incoming emails
    - SMTP/SendGrid/SES/Mailgun for outgoing emails
    - Automatic thread tracking
    - Integration with pairing system for user verification
    - Domain-based allowlisting
    - Attachment handling with size limits

Example:
    config = EmailConfig(
        org_id="org_123",
        imap=IMAPConfig(host="imap.gmail.com", ...),
        smtp=SMTPConfig(host="smtp.gmail.com", ...)
    )

    adapter = EmailAdapter(config, pairing_manager)
    await adapter.connect()

    # Send an email
    response = AdapterResponse(content="Your grant has been approved!")
    message_id = await adapter.send_message(
        to="applicant@example.com",
        response=response,
        subject="Grant Application Update"
    )

    # Poll for new emails
    new_emails = await adapter.fetch_new()
    for email in new_emails:
        normalized = adapter.normalize_message(email)
        # Process normalized message...
"""

import asyncio
import imaplib
import logging
import smtplib
import ssl
import uuid
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable

from ..shared.base import (
    AdapterPlatform,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
)
from ..shared.pairing import PairingManager

from .config import EmailConfig, EmailProvider
from .parser import EmailParser, ParsedEmail

logger = logging.getLogger(__name__)


class EmailAdapterError(Exception):
    """Base exception for email adapter errors."""
    pass


class ConnectionError(EmailAdapterError):
    """Failed to connect to email server."""
    pass


class SendError(EmailAdapterError):
    """Failed to send email."""
    pass


class FetchError(EmailAdapterError):
    """Failed to fetch emails."""
    pass


class EmailAdapter(BaseAdapter):
    """
    Email adapter for Kintsugi CMA.

    Implements the BaseAdapter interface for email communication,
    supporting both inbound and outbound email operations.

    Attributes:
        platform: Always AdapterPlatform.EMAIL

    Thread Safety:
        This adapter is NOT thread-safe for concurrent operations.
        Use asyncio locks or separate adapter instances for concurrency.

    Example:
        adapter = EmailAdapter(config)
        await adapter.connect()

        try:
            # Fetch new emails
            emails = await adapter.fetch_new()
            for email in emails:
                msg = adapter.normalize_message(email)
                # Process...
        finally:
            await adapter.disconnect()
    """

    platform = AdapterPlatform.EMAIL

    def __init__(
        self,
        config: EmailConfig,
        pairing: PairingManager | None = None
    ):
        """
        Initialize the email adapter.

        Args:
            config: Email configuration settings
            pairing: Optional pairing manager for user verification
        """
        self._config = config
        self._pairing = pairing
        self._parser = EmailParser(
            max_attachment_bytes=config.max_attachment_bytes
        )

        # Connection state
        self._imap_client: imaplib.IMAP4_SSL | imaplib.IMAP4 | None = None
        self._smtp_client: smtplib.SMTP | smtplib.SMTP_SSL | None = None
        self._connected = False

        # Polling state
        self._polling = False
        self._poll_task: asyncio.Task | None = None

        # Thread tracking
        self._thread_map: dict[str, str] = {}  # message_id -> thread_id

        # Processed message IDs to avoid duplicates
        self._processed_ids: set[str] = set()

        logger.info(
            "EmailAdapter initialized for org %s (provider: %s)",
            config.org_id,
            config.provider.value
        )

    @property
    def config(self) -> EmailConfig:
        """Get the current configuration."""
        return self._config

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected to servers."""
        return self._connected

    async def connect(self) -> None:
        """
        Connect to IMAP and/or SMTP servers.

        Establishes connections based on configuration. If IMAP is
        configured, connects for receiving. If SMTP is configured,
        validates the connection for sending.

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.warning("EmailAdapter already connected")
            return

        try:
            # Connect to IMAP if configured
            if self._config.imap:
                await self._connect_imap()
                logger.info(
                    "Connected to IMAP: %s",
                    self._config.imap.connection_string
                )

            # Test SMTP connection if configured
            if self._config.smtp:
                await self._test_smtp_connection()
                logger.info(
                    "SMTP validated: %s",
                    self._config.smtp.connection_string
                )

            self._connected = True
            logger.info("EmailAdapter connected successfully")

        except Exception as e:
            logger.error("Failed to connect: %s", e)
            await self.disconnect()
            raise ConnectionError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from email servers.

        Safely closes all connections and stops any polling tasks.
        """
        # Stop polling first
        await self.stop_polling()

        # Close IMAP connection
        if self._imap_client:
            try:
                self._imap_client.logout()
            except Exception as e:
                logger.warning("Error closing IMAP: %s", e)
            self._imap_client = None

        # Close SMTP connection
        if self._smtp_client:
            try:
                self._smtp_client.quit()
            except Exception as e:
                logger.warning("Error closing SMTP: %s", e)
            self._smtp_client = None

        self._connected = False
        logger.info("EmailAdapter disconnected")

    async def send_message(
        self,
        channel_id: str,
        response: AdapterResponse
    ) -> str:
        """
        Send an email message.

        For email, channel_id is the recipient email address.

        Args:
            channel_id: Recipient email address
            response: Response to send

        Returns:
            Generated message ID

        Raises:
            SendError: If sending fails
        """
        return await self.send_email(
            to=channel_id,
            response=response
        )

    async def send_dm(
        self,
        user_id: str,
        response: AdapterResponse
    ) -> str:
        """
        Send a direct email to a user.

        For email, user_id is the recipient email address.

        Args:
            user_id: Recipient email address
            response: Response to send

        Returns:
            Generated message ID

        Raises:
            SendError: If sending fails
        """
        return await self.send_email(
            to=user_id,
            response=response
        )

    async def send_email(
        self,
        to: str,
        response: AdapterResponse,
        subject: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to_message_id: str | None = None
    ) -> str:
        """
        Send an email with full options.

        Args:
            to: Recipient email address
            response: Response containing content and attachments
            subject: Email subject (auto-generated if not provided)
            cc: CC recipients
            bcc: BCC recipients
            reply_to_message_id: Message ID to reply to (for threading)

        Returns:
            Generated message ID

        Raises:
            SendError: If sending fails
        """
        if not self._config.can_send:
            raise SendError("Email sending not configured")

        # Generate message ID
        message_id = f"<{uuid.uuid4()}@kintsugi.cma>"

        # Build subject
        if not subject:
            subject = response.metadata.get("subject", "Message from Kintsugi")

        # Create message
        if response.attachments:
            msg = MIMEMultipart()
            msg.attach(MIMEText(response.content, "plain"))

            # Add attachments
            for attachment in response.attachments:
                part = MIMEApplication(
                    attachment.get("content", b""),
                    Name=attachment.get("filename", "attachment")
                )
                part["Content-Disposition"] = (
                    f'attachment; filename="{attachment.get("filename", "attachment")}"'
                )
                msg.attach(part)
        else:
            msg = MIMEText(response.content, "plain")

        # Set headers
        msg["Subject"] = subject
        msg["From"] = self._config.smtp.sender_address if self._config.smtp else ""
        msg["To"] = to
        msg["Message-ID"] = message_id
        msg["Date"] = datetime.now(timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S %z"
        )

        if cc:
            msg["Cc"] = ", ".join(cc)

        # Thread headers
        if reply_to_message_id:
            msg["In-Reply-To"] = reply_to_message_id
            msg["References"] = reply_to_message_id

        # Add signature if configured
        if self._config.signature:
            body = msg.get_payload()
            if isinstance(body, str):
                msg.set_payload(f"{body}\n\n{self._config.signature}")

        # Send via appropriate provider
        if self._config.provider == EmailProvider.SMTP:
            await self._send_via_smtp(msg, to, cc, bcc)
        elif self._config.provider == EmailProvider.SENDGRID:
            await self._send_via_sendgrid(msg, to, cc, bcc)
        elif self._config.provider == EmailProvider.SES:
            await self._send_via_ses(msg, to, cc, bcc)
        elif self._config.provider == EmailProvider.MAILGUN:
            await self._send_via_mailgun(msg, to, cc, bcc)

        logger.info("Email sent to %s (id: %s)", to, message_id)
        return message_id

    async def verify_user(self, user_id: str, org_id: str | None = None) -> bool:
        """
        Verify if an email address is allowed to interact.

        Checks domain allowlist and pairing status.

        Args:
            user_id: Email address to verify
            org_id: Organization ID (uses config org_id if not provided)

        Returns:
            True if the user is allowed
        """
        org = org_id or self._config.org_id

        # Check domain restrictions
        if not self._config.is_email_allowed(user_id):
            logger.debug("Email %s blocked by domain rules", user_id)
            return False

        # Check pairing if required
        if self._config.require_pairing and self._pairing:
            if not self._pairing.is_allowed(org, user_id):
                logger.debug("Email %s not paired with org %s", user_id, org)
                return False

        return True

    def normalize_message(self, email: ParsedEmail) -> AdapterMessage:
        """
        Convert a ParsedEmail to an AdapterMessage.

        Args:
            email: Parsed email to normalize

        Returns:
            Normalized AdapterMessage for processing
        """
        # Build metadata
        metadata: dict[str, Any] = {
            "subject": email.subject,
            "message_id": email.message_id,
            "in_reply_to": email.in_reply_to,
            "thread_id": email.thread_id,
            "from_name": email.from_name,
            "to_addresses": email.to_addresses,
            "cc_addresses": email.cc_addresses,
            "has_html": email.body_html is not None,
            "importance": email.importance,
        }

        # Convert attachments
        attachments = [
            {
                "filename": att.filename,
                "content_type": att.content_type,
                "size_bytes": att.size_bytes,
                "is_inline": att.is_inline,
            }
            for att in email.attachments
        ]

        return AdapterMessage(
            platform=AdapterPlatform.EMAIL,
            platform_user_id=email.from_address,
            platform_channel_id=email.to_addresses[0] if email.to_addresses else "",
            org_id=self._config.org_id,
            content=email.body_text,
            timestamp=email.received_at,
            metadata=metadata,
            attachments=attachments,
        )

    async def fetch_new(self) -> list[ParsedEmail]:
        """
        Fetch new unread emails from the configured IMAP folder.

        Returns:
            List of newly fetched and parsed emails

        Raises:
            FetchError: If fetching fails
        """
        if not self._config.imap or not self._imap_client:
            raise FetchError("IMAP not configured or not connected")

        try:
            emails: list[ParsedEmail] = []

            # Select folder
            self._imap_client.select(self._config.imap.folder)

            # Search for new messages
            search_criteria = self._config.imap.search_criteria
            status, message_ids = self._imap_client.search(None, search_criteria)

            if status != "OK":
                raise FetchError(f"IMAP search failed: {status}")

            ids = message_ids[0].split()
            logger.debug("Found %d unread messages", len(ids))

            for msg_id in ids:
                # Fetch the message
                status, data = self._imap_client.fetch(msg_id, "(RFC822)")

                if status != "OK" or not data or not data[0]:
                    continue

                raw_email = data[0][1]
                if isinstance(raw_email, bytes):
                    try:
                        parsed = self._parser.parse(raw_email)

                        # Skip if already processed
                        if parsed.message_id in self._processed_ids:
                            continue

                        # Skip auto-replies
                        if self._parser.is_auto_reply(parsed):
                            logger.debug(
                                "Skipping auto-reply: %s",
                                parsed.subject
                            )
                            if self._config.imap.mark_as_read:
                                self._imap_client.store(
                                    msg_id, "+FLAGS", "\\Seen"
                                )
                            continue

                        emails.append(parsed)
                        self._processed_ids.add(parsed.message_id)

                        # Update thread map
                        if parsed.thread_id:
                            self._thread_map[parsed.message_id] = parsed.thread_id

                        # Mark as read if configured
                        if self._config.imap.mark_as_read:
                            self._imap_client.store(msg_id, "+FLAGS", "\\Seen")

                    except Exception as e:
                        logger.warning("Failed to parse email %s: %s", msg_id, e)
                        continue

            logger.info("Fetched %d new emails", len(emails))
            return emails

        except Exception as e:
            logger.error("Failed to fetch emails: %s", e)
            raise FetchError(f"Fetch failed: {e}") from e

    async def mark_read(self, message_id: str) -> None:
        """
        Mark an email as read by message ID.

        Args:
            message_id: The Message-ID header value

        Raises:
            FetchError: If marking fails
        """
        if not self._imap_client:
            raise FetchError("IMAP not connected")

        try:
            # Search by message ID
            self._imap_client.select(self._config.imap.folder)
            status, data = self._imap_client.search(
                None,
                f'HEADER Message-ID "{message_id}"'
            )

            if status == "OK" and data[0]:
                for msg_id in data[0].split():
                    self._imap_client.store(msg_id, "+FLAGS", "\\Seen")
                logger.debug("Marked email as read: %s", message_id)

        except Exception as e:
            logger.warning("Failed to mark email as read: %s", e)

    async def start_polling(
        self,
        callback: Callable[[ParsedEmail], None]
    ) -> None:
        """
        Start polling for new emails.

        Starts a background task that periodically fetches new emails
        and calls the callback for each one.

        Args:
            callback: Function to call for each new email

        Raises:
            RuntimeError: If polling is already active
        """
        if self._polling:
            raise RuntimeError("Polling already active")

        if not self._config.imap:
            raise RuntimeError("IMAP not configured")

        self._polling = True

        async def poll_loop():
            while self._polling:
                try:
                    emails = await self.fetch_new()
                    for email in emails:
                        try:
                            callback(email)
                        except Exception as e:
                            logger.error("Callback error: %s", e)

                except FetchError as e:
                    logger.error("Fetch error during polling: %s", e)
                    # Try to reconnect
                    try:
                        await self._connect_imap()
                    except Exception:
                        pass

                await asyncio.sleep(self._config.imap.check_interval_seconds)

        self._poll_task = asyncio.create_task(poll_loop())
        logger.info(
            "Started email polling (interval: %ds)",
            self._config.imap.check_interval_seconds
        )

    async def stop_polling(self) -> None:
        """Stop the email polling task."""
        self._polling = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
            logger.info("Stopped email polling")

    async def health_check(self) -> bool:
        """
        Check if email connections are healthy.

        Returns:
            True if connections are healthy
        """
        if not self._connected:
            return False

        try:
            # Check IMAP
            if self._imap_client:
                self._imap_client.noop()

            return True
        except Exception as e:
            logger.warning("Health check failed: %s", e)
            return False

    # Private methods

    async def _connect_imap(self) -> None:
        """Establish IMAP connection."""
        imap_config = self._config.imap
        if not imap_config:
            raise ConnectionError("IMAP not configured")

        try:
            if imap_config.use_ssl:
                context = ssl.create_default_context()
                self._imap_client = imaplib.IMAP4_SSL(
                    imap_config.host,
                    imap_config.port,
                    ssl_context=context
                )
            else:
                self._imap_client = imaplib.IMAP4(
                    imap_config.host,
                    imap_config.port
                )

            # Login
            self._imap_client.login(
                imap_config.username,
                imap_config.password
            )

            # Select folder
            self._imap_client.select(imap_config.folder)

        except Exception as e:
            self._imap_client = None
            raise ConnectionError(f"IMAP connection failed: {e}") from e

    async def _test_smtp_connection(self) -> None:
        """Test SMTP connection."""
        smtp_config = self._config.smtp
        if not smtp_config:
            return

        try:
            if smtp_config.use_ssl:
                server = smtplib.SMTP_SSL(
                    smtp_config.host,
                    smtp_config.port,
                    timeout=smtp_config.timeout_seconds
                )
            else:
                server = smtplib.SMTP(
                    smtp_config.host,
                    smtp_config.port,
                    timeout=smtp_config.timeout_seconds
                )

                if smtp_config.use_tls:
                    server.starttls()

            if smtp_config.username and smtp_config.password:
                server.login(smtp_config.username, smtp_config.password)

            server.quit()

        except Exception as e:
            raise ConnectionError(f"SMTP connection failed: {e}") from e

    async def _send_via_smtp(
        self,
        msg,
        to: str,
        cc: list[str] | None,
        bcc: list[str] | None
    ) -> None:
        """Send email via SMTP."""
        smtp_config = self._config.smtp
        if not smtp_config:
            raise SendError("SMTP not configured")

        all_recipients = [to]
        if cc:
            all_recipients.extend(cc)
        if bcc:
            all_recipients.extend(bcc)

        try:
            if smtp_config.use_ssl:
                server = smtplib.SMTP_SSL(
                    smtp_config.host,
                    smtp_config.port,
                    timeout=smtp_config.timeout_seconds
                )
            else:
                server = smtplib.SMTP(
                    smtp_config.host,
                    smtp_config.port,
                    timeout=smtp_config.timeout_seconds
                )

                if smtp_config.use_tls:
                    server.starttls()

            if smtp_config.username and smtp_config.password:
                server.login(smtp_config.username, smtp_config.password)

            server.sendmail(
                smtp_config.from_address,
                all_recipients,
                msg.as_string()
            )
            server.quit()

        except Exception as e:
            raise SendError(f"SMTP send failed: {e}") from e

    async def _send_via_sendgrid(
        self,
        msg,
        to: str,
        cc: list[str] | None,
        bcc: list[str] | None
    ) -> None:
        """Send email via SendGrid API."""
        # Placeholder for SendGrid integration
        # In production, use sendgrid-python library
        raise NotImplementedError("SendGrid integration pending")

    async def _send_via_ses(
        self,
        msg,
        to: str,
        cc: list[str] | None,
        bcc: list[str] | None
    ) -> None:
        """Send email via Amazon SES."""
        # Placeholder for AWS SES integration
        # In production, use boto3 SES client
        raise NotImplementedError("SES integration pending")

    async def _send_via_mailgun(
        self,
        msg,
        to: str,
        cc: list[str] | None,
        bcc: list[str] | None
    ) -> None:
        """Send email via Mailgun API."""
        # Placeholder for Mailgun integration
        # In production, use requests to Mailgun API
        raise NotImplementedError("Mailgun integration pending")

    def __repr__(self) -> str:
        return (
            f"<EmailAdapter org={self._config.org_id!r} "
            f"connected={self._connected}>"
        )
