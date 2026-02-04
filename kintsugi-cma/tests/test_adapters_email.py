"""Comprehensive tests for Kintsugi email adapter module.

Tests cover:
- EmailProvider enum values
- IMAPConfig dataclass creation and defaults
- SMTPConfig dataclass creation and defaults
- EmailConfig dataclass creation and defaults
- ParsedEmail dataclass creation
- EmailParser parsing and extraction methods
- EmailAdapter platform, normalization, and verification
- NotificationManager scheduling and deadlines
- TemplateRenderer rendering and template management
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from kintsugi.adapters.email import (
    # Provider enum
    EmailProvider,
    # Configuration classes
    IMAPConfig,
    SMTPConfig,
    EmailConfig,
    # Parsed email
    ParsedEmail,
    # Parser
    EmailParser,
    # Adapter
    EmailAdapter,
    # Notification manager
    NotificationManager,
    ScheduledNotification,
    # Template renderer
    TemplateRenderer,
    EmailTemplate,
)
from kintsugi.adapters.shared import (
    AdapterPlatform,
    AdapterMessage,
    AdapterResponse,
    PairingManager,
    PairingConfig,
)


# ===========================================================================
# EmailProvider Tests (3 tests)
# ===========================================================================


class TestEmailProvider:
    """Tests for EmailProvider enum."""

    def test_smtp_provider_exists(self):
        """EmailProvider.SMTP exists with correct value."""
        assert EmailProvider.SMTP.value == "smtp"

    def test_sendgrid_provider_exists(self):
        """EmailProvider.SENDGRID exists with correct value."""
        assert EmailProvider.SENDGRID.value == "sendgrid"

    def test_ses_provider_exists(self):
        """EmailProvider.SES exists with correct value."""
        assert EmailProvider.SES.value == "ses"

    def test_mailgun_provider_exists(self):
        """EmailProvider.MAILGUN exists with correct value."""
        assert EmailProvider.MAILGUN.value == "mailgun"


# ===========================================================================
# IMAPConfig Tests (6 tests)
# ===========================================================================


class TestIMAPConfig:
    """Tests for IMAPConfig dataclass."""

    def test_creation_with_host(self):
        """IMAPConfig can be created with just host."""
        config = IMAPConfig(host="imap.example.com")
        assert config.host == "imap.example.com"

    def test_default_port_is_993(self):
        """IMAPConfig default port is 993."""
        config = IMAPConfig(host="imap.example.com")
        assert config.port == 993

    def test_default_use_ssl_is_true(self):
        """IMAPConfig default use_ssl is True."""
        config = IMAPConfig(host="imap.example.com")
        assert config.use_ssl is True

    def test_default_folder_is_inbox(self):
        """IMAPConfig default folder is INBOX."""
        config = IMAPConfig(host="imap.example.com")
        assert config.folder == "INBOX"

    def test_custom_configuration(self):
        """IMAPConfig accepts custom values."""
        config = IMAPConfig(
            host="imap.custom.com",
            port=143,
            use_ssl=False,
            folder="Archive",
            username="user@example.com",
            password="secret",
        )
        assert config.host == "imap.custom.com"
        assert config.port == 143
        assert config.use_ssl is False
        assert config.folder == "Archive"
        assert config.username == "user@example.com"

    def test_validation_empty_host_fails(self):
        """IMAPConfig raises ValueError for empty host."""
        with pytest.raises(ValueError, match="host cannot be empty"):
            IMAPConfig(host="")


# ===========================================================================
# SMTPConfig Tests (6 tests)
# ===========================================================================


class TestSMTPConfig:
    """Tests for SMTPConfig dataclass."""

    def test_creation_with_host(self):
        """SMTPConfig can be created with just host."""
        config = SMTPConfig(host="smtp.example.com")
        assert config.host == "smtp.example.com"

    def test_default_port_is_587(self):
        """SMTPConfig default port is 587."""
        config = SMTPConfig(host="smtp.example.com")
        assert config.port == 587

    def test_default_use_tls_is_true(self):
        """SMTPConfig default use_tls is True."""
        config = SMTPConfig(host="smtp.example.com")
        assert config.use_tls is True

    def test_custom_configuration(self):
        """SMTPConfig accepts custom values."""
        config = SMTPConfig(
            host="smtp.custom.com",
            port=465,
            use_tls=False,
            use_ssl=True,
            username="sender@example.com",
            password="secret",
        )
        assert config.host == "smtp.custom.com"
        assert config.port == 465
        assert config.use_tls is False
        assert config.use_ssl is True

    def test_validation_empty_host_fails(self):
        """SMTPConfig raises ValueError for empty host."""
        with pytest.raises(ValueError, match="host cannot be empty"):
            SMTPConfig(host="")

    def test_from_address_optional(self):
        """SMTPConfig from_address is optional."""
        config = SMTPConfig(host="smtp.example.com")
        assert config.from_address is None

        config_with_from = SMTPConfig(
            host="smtp.example.com", from_address="noreply@example.com"
        )
        assert config_with_from.from_address == "noreply@example.com"


# ===========================================================================
# EmailConfig Tests (8 tests)
# ===========================================================================


class TestEmailConfig:
    """Tests for EmailConfig dataclass."""

    def test_creation_with_org_id(self):
        """EmailConfig can be created with org_id."""
        config = EmailConfig(org_id="org_123")
        assert config.org_id == "org_123"

    def test_imap_config_is_optional(self):
        """EmailConfig imap config is optional."""
        config = EmailConfig(org_id="org_123")
        assert config.imap_config is None

    def test_smtp_config_is_optional(self):
        """EmailConfig smtp config is optional."""
        config = EmailConfig(org_id="org_123")
        assert config.smtp_config is None

    def test_default_provider_is_smtp(self):
        """EmailConfig default provider is SMTP."""
        config = EmailConfig(org_id="org_123")
        assert config.provider == EmailProvider.SMTP

    def test_allowed_domains_default_empty(self):
        """EmailConfig default allowed_domains is empty list."""
        config = EmailConfig(org_id="org_123")
        assert config.allowed_domains == []

    def test_require_pairing_default_true(self):
        """EmailConfig default require_pairing is True."""
        config = EmailConfig(org_id="org_123")
        assert config.require_pairing is True

    def test_full_configuration(self):
        """EmailConfig accepts full configuration."""
        imap = IMAPConfig(host="imap.example.com")
        smtp = SMTPConfig(host="smtp.example.com")
        config = EmailConfig(
            org_id="org_456",
            imap_config=imap,
            smtp_config=smtp,
            provider=EmailProvider.SENDGRID,
            allowed_domains=["example.com", "company.org"],
            require_pairing=False,
        )
        assert config.imap_config == imap
        assert config.smtp_config == smtp
        assert config.provider == EmailProvider.SENDGRID
        assert config.allowed_domains == ["example.com", "company.org"]
        assert config.require_pairing is False

    def test_validation_empty_org_id_fails(self):
        """EmailConfig raises ValueError for empty org_id."""
        with pytest.raises(ValueError, match="org_id cannot be empty"):
            EmailConfig(org_id="")


# ===========================================================================
# ParsedEmail Tests (6 tests)
# ===========================================================================


class TestParsedEmail:
    """Tests for ParsedEmail dataclass."""

    def test_creation_with_required_fields(self):
        """ParsedEmail can be created with required fields."""
        email = ParsedEmail(
            message_id="<123@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="Test Subject",
            body_text="Test body content",
            received_at=datetime.now(timezone.utc),
        )
        assert email.message_id == "<123@example.com>"
        assert email.from_address == "sender@example.com"
        assert email.subject == "Test Subject"

    def test_attachments_list(self):
        """ParsedEmail can hold attachments list."""
        attachments = [
            {"filename": "doc.pdf", "content_type": "application/pdf", "size": 1024},
            {"filename": "img.png", "content_type": "image/png", "size": 2048},
        ]
        email = ParsedEmail(
            message_id="<456@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="With attachments",
            body_text="See attached",
            received_at=datetime.now(timezone.utc),
            attachments=attachments,
        )
        assert len(email.attachments) == 2
        assert email.attachments[0]["filename"] == "doc.pdf"

    def test_thread_tracking_fields(self):
        """ParsedEmail has thread tracking fields."""
        email = ParsedEmail(
            message_id="<789@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="Re: Original subject",
            body_text="Reply content",
            received_at=datetime.now(timezone.utc),
            in_reply_to="<456@example.com>",
            references=["<123@example.com>", "<456@example.com>"],
            thread_id="thread_abc",
        )
        assert email.in_reply_to == "<456@example.com>"
        assert len(email.references) == 2
        assert email.thread_id == "thread_abc"

    def test_default_attachments_is_empty(self):
        """ParsedEmail defaults attachments to empty list."""
        email = ParsedEmail(
            message_id="<111@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="No attachments",
            body_text="Just text",
            received_at=datetime.now(timezone.utc),
        )
        assert email.attachments == []

    def test_cc_and_bcc_addresses(self):
        """ParsedEmail can hold CC and BCC addresses."""
        email = ParsedEmail(
            message_id="<222@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            cc_addresses=["cc1@example.com", "cc2@example.com"],
            bcc_addresses=["bcc@example.com"],
            subject="Multi-recipient",
            body_text="Content",
            received_at=datetime.now(timezone.utc),
        )
        assert len(email.cc_addresses) == 2
        assert len(email.bcc_addresses) == 1

    def test_html_body(self):
        """ParsedEmail can hold HTML body."""
        email = ParsedEmail(
            message_id="<333@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="HTML email",
            body_text="Plain text version",
            body_html="<html><body><h1>HTML version</h1></body></html>",
            received_at=datetime.now(timezone.utc),
        )
        assert email.body_html is not None
        assert "<h1>" in email.body_html


# ===========================================================================
# EmailParser Tests (8 tests)
# ===========================================================================


class TestEmailParser:
    """Tests for EmailParser class."""

    @pytest.fixture
    def parser(self) -> EmailParser:
        """Create an EmailParser."""
        return EmailParser()

    @pytest.fixture
    def sample_email_message(self) -> EmailMessage:
        """Create a sample EmailMessage for testing."""
        msg = EmailMessage()
        msg["Message-ID"] = "<test123@example.com>"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Test Email Subject"
        msg["Date"] = "Mon, 15 Jan 2025 12:00:00 +0000"
        msg.set_content("This is the email body content.")
        return msg

    def test_parse_returns_parsed_email(self, parser, sample_email_message):
        """EmailParser.parse() returns ParsedEmail."""
        result = parser.parse(sample_email_message)
        assert isinstance(result, ParsedEmail)
        assert result.message_id == "<test123@example.com>"
        assert result.from_address == "sender@example.com"
        assert result.subject == "Test Email Subject"

    def test_extract_intent_returns_string(self, parser):
        """EmailParser.extract_intent() returns intent string."""
        email = ParsedEmail(
            message_id="<test@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="Request for information",
            body_text="I would like to know more about your services.",
            received_at=datetime.now(timezone.utc),
        )
        intent = parser.extract_intent(email)
        assert isinstance(intent, str)
        assert len(intent) > 0

    def test_extract_entities_returns_dict(self, parser):
        """EmailParser.extract_entities() returns entity dictionary."""
        email = ParsedEmail(
            message_id="<test@example.com>",
            from_address="john.doe@example.com",
            to_addresses=["support@company.com"],
            subject="Meeting on January 15th",
            body_text="Hi, I'd like to schedule a meeting for next week. My phone is 555-1234.",
            received_at=datetime.now(timezone.utc),
        )
        entities = parser.extract_entities(email)
        assert isinstance(entities, dict)
        # Should extract some entities
        assert "sender_name" in entities or "dates" in entities or "phone_numbers" in entities

    def test_is_auto_reply_detects_auto_replies(self, parser):
        """EmailParser.is_auto_reply() detects auto-reply emails."""
        # Auto-reply email
        auto_reply = ParsedEmail(
            message_id="<auto@example.com>",
            from_address="noreply@example.com",
            to_addresses=["user@example.com"],
            subject="Out of Office: Re: Your message",
            body_text="I am currently out of the office.",
            received_at=datetime.now(timezone.utc),
        )
        assert parser.is_auto_reply(auto_reply) is True

        # Normal email
        normal = ParsedEmail(
            message_id="<normal@example.com>",
            from_address="human@example.com",
            to_addresses=["user@example.com"],
            subject="Quick question",
            body_text="Hey, can you help me with this?",
            received_at=datetime.now(timezone.utc),
        )
        assert parser.is_auto_reply(normal) is False

    def test_parse_handles_multipart_email(self, parser):
        """EmailParser.parse() handles multipart emails."""
        msg = EmailMessage()
        msg["Message-ID"] = "<multi@example.com>"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = "Multipart Email"
        msg.make_mixed()
        msg.add_attachment(b"file content", maintype="text", subtype="plain", filename="test.txt")

        result = parser.parse(msg)
        assert isinstance(result, ParsedEmail)

    def test_extract_priority(self, parser):
        """EmailParser.extract_priority() extracts email priority."""
        high_priority = ParsedEmail(
            message_id="<urgent@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            subject="URGENT: Critical issue",
            body_text="This needs immediate attention!",
            received_at=datetime.now(timezone.utc),
            headers={"X-Priority": "1", "Importance": "high"},
        )
        priority = parser.extract_priority(high_priority)
        assert priority in ["high", "normal", "low", 1, 2, 3]

    def test_normalize_addresses(self, parser):
        """EmailParser normalizes email addresses."""
        # Test with display name
        result = parser.normalize_address("John Doe <john@example.com>")
        assert result == "john@example.com"

        # Test plain address
        result = parser.normalize_address("jane@example.com")
        assert result == "jane@example.com"

    def test_detect_spam_indicators(self, parser):
        """EmailParser.detect_spam_indicators() identifies potential spam."""
        spam_email = ParsedEmail(
            message_id="<spam@example.com>",
            from_address="winner@lottery.com",
            to_addresses=["victim@example.com"],
            subject="YOU WON $1,000,000!!! CLICK NOW!!!",
            body_text="Congratulations! You've been selected to receive FREE MONEY!!!",
            received_at=datetime.now(timezone.utc),
        )
        indicators = parser.detect_spam_indicators(spam_email)
        assert isinstance(indicators, dict)
        assert "spam_score" in indicators or "is_spam" in indicators


# ===========================================================================
# EmailAdapter Tests (10 tests)
# ===========================================================================


class TestEmailAdapter:
    """Tests for EmailAdapter class."""

    @pytest.fixture
    def email_config(self) -> EmailConfig:
        """Create an EmailConfig for testing."""
        return EmailConfig(
            org_id="org_test",
            allowed_domains=["example.com", "company.org"],
            require_pairing=True,
        )

    @pytest.fixture
    def pairing_manager(self) -> PairingManager:
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def adapter(self, email_config, pairing_manager) -> EmailAdapter:
        """Create an EmailAdapter for testing."""
        return EmailAdapter(email_config, pairing_manager)

    def test_platform_is_email(self, adapter):
        """EmailAdapter.platform is AdapterPlatform.EMAIL."""
        assert adapter.platform == AdapterPlatform.EMAIL

    def test_normalize_message_converts_parsed_email(self, adapter):
        """EmailAdapter.normalize_message() converts ParsedEmail to AdapterMessage."""
        parsed = ParsedEmail(
            message_id="<test@example.com>",
            from_address="user@example.com",
            to_addresses=["recipient@company.org"],
            subject="Test message",
            body_text="Hello, this is a test.",
            received_at=datetime.now(timezone.utc),
        )
        message = adapter.normalize_message(parsed)

        assert isinstance(message, AdapterMessage)
        assert message.platform == AdapterPlatform.EMAIL
        assert message.platform_user_id == "user@example.com"
        assert message.content == "Hello, this is a test."

    @pytest.mark.asyncio
    async def test_verify_user_checks_allowed_domains(self, adapter):
        """EmailAdapter.verify_user() checks allowed domains."""
        # Allowed domain
        result = await adapter.verify_user("user@example.com")
        assert result is True

        # Not allowed domain
        result = await adapter.verify_user("user@random.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_user_with_empty_allowed_domains(self, pairing_manager):
        """EmailAdapter allows all domains when allowed_domains is empty."""
        config = EmailConfig(org_id="org_any", allowed_domains=[])
        adapter = EmailAdapter(config, pairing_manager)

        result = await adapter.verify_user("anyone@anywhere.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_message_uses_smtp(self, adapter):
        """EmailAdapter.send_message() uses SMTP to send."""
        with patch.object(adapter, "_send_smtp") as mock_send:
            mock_send.return_value = "<sent123@example.com>"
            response = AdapterResponse(content="Test reply")

            result = await adapter.send_message("recipient@example.com", response)
            assert result is not None
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_dm_sends_direct_email(self, adapter):
        """EmailAdapter.send_dm() sends email to user."""
        with patch.object(adapter, "_send_smtp") as mock_send:
            mock_send.return_value = "<dm123@example.com>"
            response = AdapterResponse(content="Direct message")

            result = await adapter.send_dm("user@example.com", response)
            assert result is not None

    def test_is_valid_email_address(self, adapter):
        """EmailAdapter validates email addresses correctly."""
        assert adapter.is_valid_email("valid@example.com") is True
        assert adapter.is_valid_email("user.name+tag@example.org") is True
        assert adapter.is_valid_email("invalid") is False
        assert adapter.is_valid_email("@example.com") is False
        assert adapter.is_valid_email("user@") is False

    def test_extract_domain(self, adapter):
        """EmailAdapter extracts domain from email address."""
        assert adapter.extract_domain("user@example.com") == "example.com"
        assert adapter.extract_domain("name@sub.domain.org") == "sub.domain.org"

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """EmailAdapter.health_check() verifies connection."""
        with patch.object(adapter, "_check_imap_connection") as mock_imap:
            with patch.object(adapter, "_check_smtp_connection") as mock_smtp:
                mock_imap.return_value = True
                mock_smtp.return_value = True

                result = await adapter.health_check()
                assert result is True

    def test_format_reply_address(self, adapter):
        """EmailAdapter formats reply addresses correctly."""
        original = ParsedEmail(
            message_id="<original@example.com>",
            from_address="sender@example.com",
            to_addresses=["recipient@company.org"],
            subject="Original subject",
            body_text="Original content",
            received_at=datetime.now(timezone.utc),
        )
        reply_to = adapter.format_reply_address(original)
        assert reply_to == "sender@example.com"


# ===========================================================================
# NotificationManager Tests (6 tests)
# ===========================================================================


class TestNotificationManager:
    """Tests for NotificationManager class."""

    @pytest.fixture
    def email_config(self) -> EmailConfig:
        """Create an EmailConfig for testing."""
        return EmailConfig(org_id="org_test")

    @pytest.fixture
    def pairing_manager(self) -> PairingManager:
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def adapter(self, email_config, pairing_manager) -> EmailAdapter:
        """Create an EmailAdapter for testing."""
        return EmailAdapter(email_config, pairing_manager)

    @pytest.fixture
    def notification_manager(self, adapter) -> NotificationManager:
        """Create a NotificationManager for testing."""
        return NotificationManager(adapter)

    def test_creation_with_adapter(self, notification_manager, adapter):
        """NotificationManager can be created with adapter."""
        assert notification_manager._adapter is adapter

    def test_schedule_reminder_stores_scheduled(self, notification_manager):
        """NotificationManager.schedule_reminder() stores scheduled notification."""
        reminder_time = datetime.now(timezone.utc) + timedelta(hours=1)
        notification_manager.schedule_reminder(
            recipient="user@example.com",
            subject="Reminder: Meeting",
            body="Don't forget the meeting at 3pm.",
            send_at=reminder_time,
        )
        assert len(notification_manager._scheduled) == 1
        scheduled = notification_manager._scheduled[0]
        assert scheduled.recipient == "user@example.com"
        assert scheduled.subject == "Reminder: Meeting"

    def test_get_upcoming_deadlines_returns_list(self, notification_manager):
        """NotificationManager.get_upcoming_deadlines() returns list."""
        now = datetime.now(timezone.utc)
        # Schedule some reminders
        notification_manager.schedule_reminder(
            recipient="user1@example.com",
            subject="Deadline 1",
            body="First deadline",
            send_at=now + timedelta(hours=1),
        )
        notification_manager.schedule_reminder(
            recipient="user2@example.com",
            subject="Deadline 2",
            body="Second deadline",
            send_at=now + timedelta(days=1),
        )
        notification_manager.schedule_reminder(
            recipient="user3@example.com",
            subject="Deadline 3",
            body="Third deadline",
            send_at=now + timedelta(days=5),
        )

        # Get upcoming in next 2 days
        upcoming = notification_manager.get_upcoming_deadlines(within_hours=48)
        assert len(upcoming) == 2

    def test_cancel_scheduled_notification(self, notification_manager):
        """NotificationManager.cancel() removes scheduled notification."""
        reminder_time = datetime.now(timezone.utc) + timedelta(hours=1)
        notification_id = notification_manager.schedule_reminder(
            recipient="user@example.com",
            subject="To be cancelled",
            body="This will be cancelled",
            send_at=reminder_time,
        )

        result = notification_manager.cancel(notification_id)
        assert result is True
        assert len(notification_manager._scheduled) == 0

    def test_get_scheduled_for_recipient(self, notification_manager):
        """NotificationManager.get_scheduled_for_recipient() filters by recipient."""
        now = datetime.now(timezone.utc)
        notification_manager.schedule_reminder(
            recipient="user1@example.com",
            subject="For user1",
            body="Content",
            send_at=now + timedelta(hours=1),
        )
        notification_manager.schedule_reminder(
            recipient="user2@example.com",
            subject="For user2",
            body="Content",
            send_at=now + timedelta(hours=2),
        )
        notification_manager.schedule_reminder(
            recipient="user1@example.com",
            subject="Another for user1",
            body="Content",
            send_at=now + timedelta(hours=3),
        )

        result = notification_manager.get_scheduled_for_recipient("user1@example.com")
        assert len(result) == 2
        assert all(s.recipient == "user1@example.com" for s in result)

    def test_scheduled_notification_dataclass(self):
        """ScheduledNotification dataclass stores notification data."""
        send_at = datetime.now(timezone.utc) + timedelta(hours=1)
        notification = ScheduledNotification(
            notification_id="notif_123",
            recipient="user@example.com",
            subject="Test Subject",
            body="Test Body",
            send_at=send_at,
            created_at=datetime.now(timezone.utc),
        )
        assert notification.notification_id == "notif_123"
        assert notification.recipient == "user@example.com"
        assert notification.send_at == send_at


# ===========================================================================
# TemplateRenderer Tests (5 tests)
# ===========================================================================


class TestTemplateRenderer:
    """Tests for TemplateRenderer class."""

    @pytest.fixture
    def renderer(self) -> TemplateRenderer:
        """Create a TemplateRenderer."""
        return TemplateRenderer()

    def test_render_returns_tuple(self, renderer):
        """TemplateRenderer.render() returns (subject, body) tuple."""
        renderer.add_template(
            "welcome",
            EmailTemplate(
                name="welcome",
                subject_template="Welcome, {{ name }}!",
                body_template="Hello {{ name }}, welcome to {{ org_name }}.",
            ),
        )

        subject, body = renderer.render(
            "welcome", {"name": "John", "org_name": "Acme Corp"}
        )
        assert isinstance(subject, str)
        assert isinstance(body, str)
        assert "John" in subject
        assert "Acme Corp" in body

    def test_add_template_adds_new_template(self, renderer):
        """TemplateRenderer.add_template() adds new template."""
        template = EmailTemplate(
            name="custom",
            subject_template="Custom: {{ title }}",
            body_template="Content: {{ content }}",
        )
        renderer.add_template("custom", template)

        assert "custom" in renderer._templates
        assert renderer._templates["custom"] == template

    def test_list_templates_returns_names(self, renderer):
        """TemplateRenderer.list_templates() returns template names."""
        renderer.add_template(
            "template1",
            EmailTemplate(
                name="template1", subject_template="S1", body_template="B1"
            ),
        )
        renderer.add_template(
            "template2",
            EmailTemplate(
                name="template2", subject_template="S2", body_template="B2"
            ),
        )

        names = renderer.list_templates()
        assert "template1" in names
        assert "template2" in names
        assert len(names) >= 2

    def test_render_with_missing_variable_uses_default(self, renderer):
        """TemplateRenderer handles missing variables gracefully."""
        renderer.add_template(
            "greeting",
            EmailTemplate(
                name="greeting",
                subject_template="Hello {{ name | default('Guest') }}",
                body_template="Welcome {{ name | default('Guest') }}!",
            ),
        )

        subject, body = renderer.render("greeting", {})
        assert "Guest" in subject
        assert "Guest" in body

    def test_template_not_found_raises_error(self, renderer):
        """TemplateRenderer.render() raises error for unknown template."""
        with pytest.raises(KeyError):
            renderer.render("nonexistent", {})


# ===========================================================================
# EmailTemplate Tests (3 additional tests)
# ===========================================================================


class TestEmailTemplate:
    """Tests for EmailTemplate dataclass."""

    def test_email_template_creation(self):
        """EmailTemplate can be created with required fields."""
        template = EmailTemplate(
            name="test",
            subject_template="Subject: {{ topic }}",
            body_template="Body about {{ topic }}",
        )
        assert template.name == "test"
        assert "{{ topic }}" in template.subject_template
        assert "{{ topic }}" in template.body_template

    def test_email_template_with_html(self):
        """EmailTemplate can include HTML body template."""
        template = EmailTemplate(
            name="html_template",
            subject_template="Important: {{ title }}",
            body_template="Plain text version",
            html_template="<html><body><h1>{{ title }}</h1></body></html>",
        )
        assert template.html_template is not None
        assert "<h1>" in template.html_template

    def test_email_template_validation(self):
        """EmailTemplate validates required fields."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            EmailTemplate(name="", subject_template="S", body_template="B")


# ===========================================================================
# Integration Tests (3 additional tests)
# ===========================================================================


class TestEmailAdapterIntegration:
    """Integration tests for email adapter components."""

    @pytest.fixture
    def full_setup(self):
        """Create full email adapter setup."""
        config = EmailConfig(
            org_id="org_integration",
            allowed_domains=["test.com"],
        )
        pairing = PairingManager(PairingConfig())
        adapter = EmailAdapter(config, pairing)
        parser = EmailParser()
        notification_mgr = NotificationManager(adapter)
        renderer = TemplateRenderer()

        return {
            "config": config,
            "adapter": adapter,
            "parser": parser,
            "notifications": notification_mgr,
            "renderer": renderer,
        }

    def test_parse_and_normalize_flow(self, full_setup):
        """Email can be parsed and normalized in sequence."""
        parser = full_setup["parser"]
        adapter = full_setup["adapter"]

        msg = EmailMessage()
        msg["Message-ID"] = "<flow@test.com>"
        msg["From"] = "sender@test.com"
        msg["To"] = "recipient@test.com"
        msg["Subject"] = "Integration Test"
        msg.set_content("Test content for integration.")

        parsed = parser.parse(msg)
        normalized = adapter.normalize_message(parsed)

        assert normalized.platform == AdapterPlatform.EMAIL
        assert normalized.platform_user_id == "sender@test.com"

    def test_template_render_for_notification(self, full_setup):
        """Template can be rendered for notification scheduling."""
        renderer = full_setup["renderer"]
        notifications = full_setup["notifications"]

        renderer.add_template(
            "reminder",
            EmailTemplate(
                name="reminder",
                subject_template="Reminder: {{ event }}",
                body_template="Don't forget about {{ event }} on {{ date }}.",
            ),
        )

        subject, body = renderer.render(
            "reminder", {"event": "Team Meeting", "date": "Monday"}
        )

        reminder_time = datetime.now(timezone.utc) + timedelta(hours=24)
        notifications.schedule_reminder(
            recipient="user@test.com",
            subject=subject,
            body=body,
            send_at=reminder_time,
        )

        assert len(notifications._scheduled) == 1

    @pytest.mark.asyncio
    async def test_verify_and_send_flow(self, full_setup):
        """User verification and message sending flow."""
        adapter = full_setup["adapter"]

        # Verify user from allowed domain
        is_verified = await adapter.verify_user("sender@test.com")
        assert is_verified is True

        # Verify user from disallowed domain
        is_verified = await adapter.verify_user("sender@random.com")
        assert is_verified is False
