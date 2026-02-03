"""Tests for Kintsugi shared adapter infrastructure.

Tests the modules in kintsugi/adapters/shared/:
- base.py (AdapterMessage, AdapterResponse, AdapterPlatform, BaseAdapter)
- pairing.py (PairingManager, PairingCode, PairingStatus, PairingConfig)
- allowlist.py (AllowlistEntry, InMemoryAllowlistStore)
"""

from __future__ import annotations

import string
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from kintsugi.adapters.shared import (
    # Base adapter types
    AdapterPlatform,
    AdapterMessage,
    AdapterResponse,
    BaseAdapter,
    # Pairing system
    PairingStatus,
    PairingCode,
    PairingConfig,
    PairingManager,
    PairingError,
    RateLimitExceeded,
    CodeNotFound,
    CodeExpired,
    CodeAlreadyUsed,
    # Allowlist
    AllowlistEntry,
    AllowlistStore,
    InMemoryAllowlistStore,
    AllowlistStoreError,
)


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================


class ConcreteAdapter(BaseAdapter):
    """Concrete adapter implementation for testing BaseAdapter."""

    platform = AdapterPlatform.SLACK

    def __init__(self, allowed_users: set[str] | None = None):
        self.allowed_users = allowed_users or set()
        self.sent_messages: list[tuple[str, AdapterResponse]] = []
        self.sent_dms: list[tuple[str, AdapterResponse]] = []

    async def send_message(self, channel_id: str, response: AdapterResponse) -> str:
        self.sent_messages.append((channel_id, response))
        return f"msg_{len(self.sent_messages)}"

    async def send_dm(self, user_id: str, response: AdapterResponse) -> str:
        self.sent_dms.append((user_id, response))
        return f"dm_{len(self.sent_dms)}"

    async def verify_user(self, user_id: str) -> bool:
        return user_id in self.allowed_users

    def normalize_message(self, raw: dict[str, Any]) -> AdapterMessage:
        return AdapterMessage(
            platform=self.platform,
            platform_user_id=raw["user_id"],
            platform_channel_id=raw.get("channel_id", "default_channel"),
            org_id=raw["org_id"],
            content=raw.get("content", ""),
            timestamp=raw.get("timestamp", datetime.now(timezone.utc)),
            metadata=raw.get("metadata", {}),
            attachments=raw.get("attachments", []),
        )


@pytest.fixture
def concrete_adapter() -> ConcreteAdapter:
    """Create a concrete adapter for testing."""
    return ConcreteAdapter(allowed_users={"user123", "user456"})


@pytest.fixture
def pairing_manager() -> PairingManager:
    """Create a pairing manager with default config."""
    return PairingManager()


@pytest.fixture
def allowlist_store() -> InMemoryAllowlistStore:
    """Create an in-memory allowlist store."""
    return InMemoryAllowlistStore()


def make_allowlist_entry(
    org_id: str = "org_test",
    platform: AdapterPlatform = AdapterPlatform.SLACK,
    platform_user_id: str = "U12345",
    added_by: str = "admin",
    notes: str | None = None,
) -> AllowlistEntry:
    """Helper to create allowlist entries."""
    return AllowlistEntry(
        org_id=org_id,
        platform=platform,
        platform_user_id=platform_user_id,
        added_at=datetime.now(timezone.utc),
        added_by=added_by,
        notes=notes,
        metadata={},
    )


# ===========================================================================
# Base Adapter Tests (15+ tests)
# ===========================================================================


class TestAdapterPlatform:
    """Tests for AdapterPlatform enum."""

    def test_platform_slack_value(self):
        """AdapterPlatform.SLACK has correct string value."""
        assert AdapterPlatform.SLACK.value == "slack"

    def test_platform_discord_value(self):
        """AdapterPlatform.DISCORD has correct string value."""
        assert AdapterPlatform.DISCORD.value == "discord"

    def test_platform_webchat_value(self):
        """AdapterPlatform.WEBCHAT has correct string value."""
        assert AdapterPlatform.WEBCHAT.value == "webchat"

    def test_platform_email_value(self):
        """AdapterPlatform.EMAIL has correct string value."""
        assert AdapterPlatform.EMAIL.value == "email"

    def test_platform_is_str_subclass(self):
        """AdapterPlatform values are strings."""
        assert isinstance(AdapterPlatform.SLACK, str)
        assert AdapterPlatform.SLACK == "slack"


class TestAdapterMessage:
    """Tests for AdapterMessage dataclass."""

    def test_creation_with_required_fields(self):
        """AdapterMessage can be created with all required fields."""
        now = datetime.now(timezone.utc)
        msg = AdapterMessage(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            platform_channel_id="C67890",
            org_id="org_abc",
            content="Hello world",
            timestamp=now,
        )
        assert msg.platform == AdapterPlatform.SLACK
        assert msg.platform_user_id == "U12345"
        assert msg.platform_channel_id == "C67890"
        assert msg.org_id == "org_abc"
        assert msg.content == "Hello world"
        assert msg.timestamp == now

    def test_creation_with_all_fields(self):
        """AdapterMessage can be created with all fields including optional."""
        now = datetime.now(timezone.utc)
        metadata = {"thread_ts": "123.456", "custom_key": "value"}
        attachments = [{"type": "image", "url": "https://example.com/img.png"}]

        msg = AdapterMessage(
            platform=AdapterPlatform.DISCORD,
            platform_user_id="U12345",
            platform_channel_id="C67890",
            org_id="org_xyz",
            content="Message with attachments",
            timestamp=now,
            metadata=metadata,
            attachments=attachments,
        )

        assert msg.metadata == metadata
        assert msg.attachments == attachments
        assert len(msg.attachments) == 1

    def test_default_metadata_is_empty_dict(self):
        """AdapterMessage defaults metadata to empty dict."""
        msg = AdapterMessage(
            platform=AdapterPlatform.WEBCHAT,
            platform_user_id="user1",
            platform_channel_id="channel1",
            org_id="org1",
            content="test",
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.metadata == {}
        assert isinstance(msg.metadata, dict)

    def test_default_attachments_is_empty_list(self):
        """AdapterMessage defaults attachments to empty list."""
        msg = AdapterMessage(
            platform=AdapterPlatform.WEBCHAT,
            platform_user_id="user1",
            platform_channel_id="channel1",
            org_id="org1",
            content="test",
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.attachments == []
        assert isinstance(msg.attachments, list)

    def test_validation_empty_platform_user_id_fails(self):
        """AdapterMessage raises ValueError for empty platform_user_id."""
        with pytest.raises(ValueError, match="platform_user_id cannot be empty"):
            AdapterMessage(
                platform=AdapterPlatform.SLACK,
                platform_user_id="",
                platform_channel_id="C12345",
                org_id="org_abc",
                content="Hello",
                timestamp=datetime.now(timezone.utc),
            )

    def test_validation_empty_org_id_fails(self):
        """AdapterMessage raises ValueError for empty org_id."""
        with pytest.raises(ValueError, match="org_id cannot be empty"):
            AdapterMessage(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                platform_channel_id="C12345",
                org_id="",
                content="Hello",
                timestamp=datetime.now(timezone.utc),
            )

    def test_empty_content_is_allowed(self):
        """AdapterMessage allows empty content (content validation is optional)."""
        msg = AdapterMessage(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            platform_channel_id="C67890",
            org_id="org_abc",
            content="",
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.content == ""

    def test_with_multiple_attachments(self):
        """AdapterMessage can hold multiple attachments."""
        attachments = [
            {"type": "file", "name": "doc.pdf"},
            {"type": "image", "url": "https://example.com/1.png"},
            {"type": "image", "url": "https://example.com/2.png"},
        ]
        msg = AdapterMessage(
            platform=AdapterPlatform.EMAIL,
            platform_user_id="user@example.com",
            platform_channel_id="inbox",
            org_id="org_mail",
            content="See attached files",
            timestamp=datetime.now(timezone.utc),
            attachments=attachments,
        )
        assert len(msg.attachments) == 3


class TestAdapterResponse:
    """Tests for AdapterResponse dataclass."""

    def test_creation_with_content(self):
        """AdapterResponse can be created with just content."""
        resp = AdapterResponse(content="Hello there!")
        assert resp.content == "Hello there!"
        assert resp.ephemeral is False
        assert resp.attachments == []
        assert resp.metadata == {}

    def test_creation_with_ephemeral_flag(self):
        """AdapterResponse ephemeral flag works correctly."""
        resp = AdapterResponse(content="Secret message", ephemeral=True)
        assert resp.ephemeral is True

        resp_public = AdapterResponse(content="Public message", ephemeral=False)
        assert resp_public.ephemeral is False

    def test_creation_with_attachments(self):
        """AdapterResponse can include attachments."""
        attachments = [{"type": "file", "url": "https://example.com/file.zip"}]
        resp = AdapterResponse(content="Here's the file", attachments=attachments)
        assert len(resp.attachments) == 1
        assert resp.attachments[0]["type"] == "file"

    def test_creation_with_metadata(self):
        """AdapterResponse can include platform-specific metadata."""
        metadata = {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}]}
        resp = AdapterResponse(content="Formatted message", metadata=metadata)
        assert "blocks" in resp.metadata
        assert len(resp.metadata["blocks"]) == 1

    def test_validation_empty_content_no_attachments_fails(self):
        """AdapterResponse requires content or attachments."""
        with pytest.raises(ValueError, match="Response must have content or attachments"):
            AdapterResponse(content="")

    def test_validation_empty_content_with_attachments_succeeds(self):
        """AdapterResponse allows empty content if attachments provided."""
        resp = AdapterResponse(
            content="",
            attachments=[{"type": "image", "url": "https://example.com/img.png"}],
        )
        assert resp.content == ""
        assert len(resp.attachments) == 1


class TestBaseAdapter:
    """Tests for BaseAdapter abstract class."""

    def test_base_adapter_is_abstract(self):
        """BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()

    def test_concrete_adapter_can_be_instantiated(self, concrete_adapter):
        """Concrete adapter subclass can be instantiated."""
        assert isinstance(concrete_adapter, BaseAdapter)
        assert concrete_adapter.platform == AdapterPlatform.SLACK

    @pytest.mark.asyncio
    async def test_send_message_implementation(self, concrete_adapter):
        """Concrete adapter send_message works correctly."""
        resp = AdapterResponse(content="Test message")
        msg_id = await concrete_adapter.send_message("C12345", resp)

        assert msg_id == "msg_1"
        assert len(concrete_adapter.sent_messages) == 1
        assert concrete_adapter.sent_messages[0] == ("C12345", resp)

    @pytest.mark.asyncio
    async def test_send_dm_implementation(self, concrete_adapter):
        """Concrete adapter send_dm works correctly."""
        resp = AdapterResponse(content="Direct message")
        dm_id = await concrete_adapter.send_dm("U12345", resp)

        assert dm_id == "dm_1"
        assert len(concrete_adapter.sent_dms) == 1
        assert concrete_adapter.sent_dms[0] == ("U12345", resp)

    @pytest.mark.asyncio
    async def test_verify_user_allowed(self, concrete_adapter):
        """Concrete adapter verify_user returns True for allowed users."""
        assert await concrete_adapter.verify_user("user123") is True
        assert await concrete_adapter.verify_user("user456") is True

    @pytest.mark.asyncio
    async def test_verify_user_not_allowed(self, concrete_adapter):
        """Concrete adapter verify_user returns False for non-allowed users."""
        assert await concrete_adapter.verify_user("unknown_user") is False
        assert await concrete_adapter.verify_user("") is False

    def test_normalize_message_implementation(self, concrete_adapter):
        """Concrete adapter normalize_message converts raw data correctly."""
        raw = {
            "user_id": "U99999",
            "channel_id": "C11111",
            "org_id": "org_test",
            "content": "Raw message content",
            "timestamp": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            "metadata": {"key": "value"},
        }
        msg = concrete_adapter.normalize_message(raw)

        assert isinstance(msg, AdapterMessage)
        assert msg.platform == AdapterPlatform.SLACK
        assert msg.platform_user_id == "U99999"
        assert msg.platform_channel_id == "C11111"
        assert msg.content == "Raw message content"

    def test_normalize_message_not_implemented_on_base(self):
        """BaseAdapter.normalize_message raises NotImplementedError by default."""
        # Create a minimal concrete class that doesn't override normalize_message
        class MinimalAdapter(BaseAdapter):
            platform = AdapterPlatform.DISCORD

            async def send_message(self, channel_id: str, response: AdapterResponse) -> str:
                return "msg"

            async def send_dm(self, user_id: str, response: AdapterResponse) -> str:
                return "dm"

            async def verify_user(self, user_id: str) -> bool:
                return True

        adapter = MinimalAdapter()
        with pytest.raises(NotImplementedError, match="must implement normalize_message"):
            adapter.normalize_message({"test": "data"})

    @pytest.mark.asyncio
    async def test_health_check_default_returns_true(self, concrete_adapter):
        """Default health_check implementation returns True."""
        assert await concrete_adapter.health_check() is True

    def test_repr_format(self, concrete_adapter):
        """BaseAdapter repr shows class name and platform."""
        repr_str = repr(concrete_adapter)
        assert "ConcreteAdapter" in repr_str
        assert "slack" in repr_str


# ===========================================================================
# Pairing Manager Tests (25+ tests)
# ===========================================================================


class TestPairingConfig:
    """Tests for PairingConfig dataclass."""

    def test_default_config_values(self):
        """PairingConfig has sensible defaults."""
        config = PairingConfig()
        assert config.code_length == 6
        assert config.expiration_minutes == 15
        assert config.max_attempts_per_hour == 5
        assert config.require_admin_approval is True

    def test_custom_code_length(self):
        """PairingConfig accepts custom code_length."""
        config = PairingConfig(code_length=8)
        assert config.code_length == 8

    def test_code_length_minimum_validation(self):
        """PairingConfig rejects code_length < 4."""
        with pytest.raises(ValueError, match="code_length must be at least 4"):
            PairingConfig(code_length=3)

    def test_expiration_minutes_minimum_validation(self):
        """PairingConfig rejects expiration_minutes < 1."""
        with pytest.raises(ValueError, match="expiration_minutes must be at least 1"):
            PairingConfig(expiration_minutes=0)

    def test_max_attempts_minimum_validation(self):
        """PairingConfig rejects max_attempts_per_hour < 1."""
        with pytest.raises(ValueError, match="max_attempts_per_hour must be at least 1"):
            PairingConfig(max_attempts_per_hour=0)


class TestPairingCode:
    """Tests for PairingCode dataclass."""

    def test_creation_with_all_fields(self):
        """PairingCode can be created with all fields."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=15)

        code = PairingCode(
            code="ABC123",
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            platform_channel_id="C67890",
            org_id="org_test",
            status=PairingStatus.PENDING,
            created_at=now,
            expires_at=expires,
            metadata={"custom": "data"},
        )

        assert code.code == "ABC123"
        assert code.platform == AdapterPlatform.SLACK
        assert code.status == PairingStatus.PENDING
        assert code.is_valid is True

    def test_is_valid_pending_not_expired(self):
        """PairingCode.is_valid returns True for pending, non-expired codes."""
        now = datetime.now(timezone.utc)
        code = PairingCode(
            code="XYZ789",
            platform=AdapterPlatform.DISCORD,
            platform_user_id="U111",
            platform_channel_id=None,
            org_id="org_xyz",
            status=PairingStatus.PENDING,
            created_at=now,
            expires_at=now + timedelta(minutes=10),
        )
        assert code.is_valid is True

    def test_is_valid_false_when_expired(self):
        """PairingCode.is_valid returns False for expired codes."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        code = PairingCode(
            code="OLD123",
            platform=AdapterPlatform.SLACK,
            platform_user_id="U222",
            platform_channel_id=None,
            org_id="org_old",
            status=PairingStatus.PENDING,
            created_at=past - timedelta(minutes=15),
            expires_at=past,
        )
        assert code.is_valid is False
        assert code.is_expired is True

    def test_is_valid_false_when_not_pending(self):
        """PairingCode.is_valid returns False for non-pending status."""
        now = datetime.now(timezone.utc)
        code = PairingCode(
            code="APPR01",
            platform=AdapterPlatform.WEBCHAT,
            platform_user_id="U333",
            platform_channel_id=None,
            org_id="org_appr",
            status=PairingStatus.APPROVED,
            created_at=now,
            expires_at=now + timedelta(minutes=10),
            approved_at=now,
            approved_by="admin",
        )
        assert code.is_valid is False


class TestPairingStatus:
    """Tests for PairingStatus enum."""

    def test_all_statuses_exist(self):
        """PairingStatus has all expected values."""
        assert PairingStatus.PENDING.value == "pending"
        assert PairingStatus.APPROVED.value == "approved"
        assert PairingStatus.REJECTED.value == "rejected"
        assert PairingStatus.EXPIRED.value == "expired"
        assert PairingStatus.REVOKED.value == "revoked"


class TestPairingManagerGeneration:
    """Tests for PairingManager code generation."""

    def test_generate_code_creates_valid_code(self, pairing_manager):
        """generate_code creates a 6-character alphanumeric code."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        assert len(code.code) == 6
        assert code.code.isalnum()
        assert code.code.isupper()
        assert code.status == PairingStatus.PENDING

    def test_generate_code_creates_unique_codes(self, pairing_manager):
        """generate_code creates unique codes each time."""
        codes = set()
        for i in range(20):
            code = pairing_manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id=f"user_{i}",
                org_id="org_test",
            )
            codes.add(code.code)

        assert len(codes) == 20

    def test_generate_code_respects_custom_code_length(self):
        """generate_code respects custom code_length config."""
        config = PairingConfig(code_length=10)
        manager = PairingManager(config=config)

        code = manager.generate_code(
            platform=AdapterPlatform.DISCORD,
            platform_user_id="U99999",
            org_id="org_custom",
        )

        assert len(code.code) == 10

    def test_generate_code_sets_expiration(self, pairing_manager):
        """generate_code sets correct expiration time."""
        before = datetime.now(timezone.utc)
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )
        after = datetime.now(timezone.utc)

        # Expiration should be ~15 minutes from creation
        expected_min = before + timedelta(minutes=15)
        expected_max = after + timedelta(minutes=15)
        assert expected_min <= code.expires_at <= expected_max

    def test_generate_code_stores_metadata(self, pairing_manager):
        """generate_code stores optional metadata."""
        metadata = {"source": "mobile_app", "version": "1.0"}
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.WEBCHAT,
            platform_user_id="web_user",
            org_id="org_web",
            metadata=metadata,
        )

        assert code.metadata == metadata

    def test_generate_code_stores_channel_id(self, pairing_manager):
        """generate_code stores optional channel_id."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
            channel_id="C99999",
        )

        assert code.platform_channel_id == "C99999"


class TestPairingManagerValidation:
    """Tests for PairingManager code validation."""

    def test_validate_code_returns_code_when_valid(self, pairing_manager):
        """validate_code returns the PairingCode when valid."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        validated = pairing_manager.validate_code(code.code)
        assert validated is not None
        assert validated.code == code.code

    def test_validate_code_returns_none_for_unknown(self, pairing_manager):
        """validate_code returns None for unknown codes."""
        assert pairing_manager.validate_code("NOTREAL") is None

    def test_validate_code_returns_none_for_expired(self, pairing_manager):
        """validate_code returns None for expired codes."""
        # Create a manager with very short expiration
        config = PairingConfig(expiration_minutes=1)
        manager = PairingManager(config=config)

        code = manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        # Manually expire the code
        code.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        assert manager.validate_code(code.code) is None
        assert code.status == PairingStatus.EXPIRED

    def test_validate_code_returns_none_for_already_used(self, pairing_manager):
        """validate_code returns None for already-used (approved) codes."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        pairing_manager.approve(code.code, approver="admin")

        assert pairing_manager.validate_code(code.code) is None

    def test_validate_code_case_insensitive(self, pairing_manager):
        """validate_code is case-insensitive."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        # Try lowercase
        validated = pairing_manager.validate_code(code.code.lower())
        assert validated is not None

    def test_validate_code_strips_whitespace(self, pairing_manager):
        """validate_code strips whitespace from input."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        validated = pairing_manager.validate_code(f"  {code.code}  ")
        assert validated is not None


class TestPairingManagerApproval:
    """Tests for PairingManager approval workflow."""

    def test_approve_transitions_to_approved(self, pairing_manager):
        """approve() transitions status to APPROVED."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        approved = pairing_manager.approve(code.code, approver="admin_user")

        assert approved.status == PairingStatus.APPROVED

    def test_approve_adds_user_to_allowlist(self, pairing_manager):
        """approve() adds user to organization's allowlist."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        pairing_manager.approve(code.code, approver="admin")

        assert pairing_manager.is_allowed("org_test", "U12345") is True

    def test_approve_sets_approved_at_and_by(self, pairing_manager):
        """approve() sets approved_at and approved_by fields."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        before = datetime.now(timezone.utc)
        approved = pairing_manager.approve(code.code, approver="admin_xyz")
        after = datetime.now(timezone.utc)

        assert approved.approved_by == "admin_xyz"
        assert approved.approved_at is not None
        assert before <= approved.approved_at <= after

    def test_approve_raises_code_not_found(self, pairing_manager):
        """approve() raises CodeNotFound for unknown codes."""
        with pytest.raises(CodeNotFound):
            pairing_manager.approve("NOTREAL", approver="admin")

    def test_approve_raises_code_expired(self, pairing_manager):
        """approve() raises CodeExpired for expired codes."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        # Manually expire the code
        code.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        with pytest.raises(CodeExpired):
            pairing_manager.approve(code.code, approver="admin")

    def test_approve_raises_code_already_used(self, pairing_manager):
        """approve() raises CodeAlreadyUsed for non-pending codes."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        pairing_manager.approve(code.code, approver="admin")

        with pytest.raises(CodeAlreadyUsed):
            pairing_manager.approve(code.code, approver="admin2")


class TestPairingManagerRejection:
    """Tests for PairingManager rejection workflow."""

    def test_reject_transitions_to_rejected(self, pairing_manager):
        """reject() transitions status to REJECTED."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        rejected = pairing_manager.reject(code.code, rejector="admin", reason="Suspicious")

        assert rejected.status == PairingStatus.REJECTED

    def test_reject_stores_rejection_reason(self, pairing_manager):
        """reject() stores the rejection reason."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        rejected = pairing_manager.reject(code.code, rejector="admin", reason="User not recognized")

        assert rejected.rejection_reason == "User not recognized"
        assert rejected.metadata["rejected_by"] == "admin"
        assert "rejected_at" in rejected.metadata


class TestPairingManagerRevocation:
    """Tests for PairingManager revocation workflow."""

    def test_revoke_removes_user_from_allowlist(self, pairing_manager):
        """revoke() removes user from organization's allowlist."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )
        pairing_manager.approve(code.code, approver="admin")

        result = pairing_manager.revoke("org_test", "U12345", revoker="admin")

        assert result is True
        assert pairing_manager.is_allowed("org_test", "U12345") is False

    def test_revoke_returns_false_if_not_on_allowlist(self, pairing_manager):
        """revoke() returns False if user not on allowlist."""
        result = pairing_manager.revoke("org_test", "U_NONEXISTENT", revoker="admin")
        assert result is False

    def test_revoke_updates_code_status(self, pairing_manager):
        """revoke() updates approved code status to REVOKED."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )
        pairing_manager.approve(code.code, approver="admin")

        pairing_manager.revoke("org_test", "U12345", revoker="admin2")

        stored_code = pairing_manager.get_code(code.code)
        assert stored_code.status == PairingStatus.REVOKED
        assert stored_code.revoked_by == "admin2"
        assert stored_code.revoked_at is not None


class TestPairingManagerAllowlist:
    """Tests for PairingManager allowlist checks."""

    def test_is_allowed_true_after_approval(self, pairing_manager):
        """is_allowed() returns True after approval."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )
        pairing_manager.approve(code.code, approver="admin")

        assert pairing_manager.is_allowed("org_test", "U12345") is True

    def test_is_allowed_false_before_approval(self, pairing_manager):
        """is_allowed() returns False before approval."""
        pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        assert pairing_manager.is_allowed("org_test", "U12345") is False

    def test_is_allowed_false_after_revocation(self, pairing_manager):
        """is_allowed() returns False after revocation."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )
        pairing_manager.approve(code.code, approver="admin")
        pairing_manager.revoke("org_test", "U12345", revoker="admin")

        assert pairing_manager.is_allowed("org_test", "U12345") is False

    def test_is_allowed_false_for_unknown_org(self, pairing_manager):
        """is_allowed() returns False for unknown organizations."""
        assert pairing_manager.is_allowed("unknown_org", "U12345") is False


class TestPairingManagerPendingCodes:
    """Tests for PairingManager pending code retrieval."""

    def test_get_pending_returns_only_pending(self, pairing_manager):
        """get_pending() returns only codes with PENDING status."""
        code1 = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U1",
            org_id="org1",
        )
        code2 = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U2",
            org_id="org1",
        )
        code3 = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U3",
            org_id="org1",
        )

        # Approve one code
        pairing_manager.approve(code2.code, approver="admin")

        pending = pairing_manager.get_pending()
        pending_codes = [c.code for c in pending]

        assert code1.code in pending_codes
        assert code2.code not in pending_codes
        assert code3.code in pending_codes

    def test_get_pending_filters_by_org_id(self, pairing_manager):
        """get_pending() filters by org_id when provided."""
        pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U1",
            org_id="org_alpha",
        )
        pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U2",
            org_id="org_beta",
        )
        pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U3",
            org_id="org_alpha",
        )

        pending_alpha = pairing_manager.get_pending(org_id="org_alpha")
        pending_beta = pairing_manager.get_pending(org_id="org_beta")

        assert len(pending_alpha) == 2
        assert len(pending_beta) == 1
        assert all(c.org_id == "org_alpha" for c in pending_alpha)


class TestPairingManagerCleanup:
    """Tests for PairingManager cleanup operations."""

    def test_cleanup_expired_marks_expired_codes(self, pairing_manager):
        """cleanup_expired() marks expired codes."""
        code = pairing_manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        # Manually expire
        code.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        count = pairing_manager.cleanup_expired()

        assert count == 1
        assert pairing_manager.get_code(code.code).status == PairingStatus.EXPIRED

    def test_cleanup_expired_returns_correct_count(self, pairing_manager):
        """cleanup_expired() returns accurate count of expired codes."""
        # Create 3 codes, expire 2
        for i in range(3):
            code = pairing_manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id=f"U{i}",
                org_id="org_test",
            )
            if i < 2:
                code.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

        count = pairing_manager.cleanup_expired()
        assert count == 2


class TestPairingManagerRateLimiting:
    """Tests for PairingManager rate limiting."""

    def test_rate_limit_enforced(self):
        """Rate limiting enforces max_attempts_per_hour."""
        config = PairingConfig(max_attempts_per_hour=3)
        manager = PairingManager(config=config)

        # Generate 3 codes (should work)
        for i in range(3):
            manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                org_id="org_test",
            )

        # 4th attempt should fail
        with pytest.raises(RateLimitExceeded):
            manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                org_id="org_test",
            )

    def test_rate_limit_per_user(self):
        """Rate limiting is per-user."""
        config = PairingConfig(max_attempts_per_hour=2)
        manager = PairingManager(config=config)

        # User 1: 2 attempts
        for i in range(2):
            manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U1",
                org_id="org_test",
            )

        # User 1 blocked
        with pytest.raises(RateLimitExceeded):
            manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U1",
                org_id="org_test",
            )

        # User 2 can still generate
        manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U2",
            org_id="org_test",
        )

    def test_rate_limit_exceeded_has_retry_info(self):
        """RateLimitExceeded exception includes retry information."""
        config = PairingConfig(max_attempts_per_hour=1)
        manager = PairingManager(config=config)

        manager.generate_code(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
            org_id="org_test",
        )

        with pytest.raises(RateLimitExceeded) as exc_info:
            manager.generate_code(
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                org_id="org_test",
            )

        assert exc_info.value.user_id == "U12345"
        assert exc_info.value.retry_after_seconds > 0


# ===========================================================================
# Allowlist Store Tests (12+ tests)
# ===========================================================================


class TestAllowlistEntry:
    """Tests for AllowlistEntry dataclass."""

    def test_creation_with_all_fields(self):
        """AllowlistEntry can be created with all fields."""
        entry = make_allowlist_entry(
            org_id="org_abc",
            platform=AdapterPlatform.DISCORD,
            platform_user_id="discord_user_123",
            added_by="admin_xyz",
            notes="Approved via ticket #456",
        )

        assert entry.org_id == "org_abc"
        assert entry.platform == AdapterPlatform.DISCORD
        assert entry.platform_user_id == "discord_user_123"
        assert entry.added_by == "admin_xyz"
        assert entry.notes == "Approved via ticket #456"

    def test_to_dict_serialization(self):
        """AllowlistEntry.to_dict() serializes correctly."""
        entry = make_allowlist_entry()
        data = entry.to_dict()

        assert data["org_id"] == "org_test"
        assert data["platform"] == "slack"
        assert data["platform_user_id"] == "U12345"
        assert data["added_by"] == "admin"
        assert "added_at" in data
        assert data["metadata"] == {}

    def test_from_dict_deserialization(self):
        """AllowlistEntry.from_dict() deserializes correctly."""
        data = {
            "org_id": "org_round",
            "platform": "discord",
            "platform_user_id": "D99999",
            "added_at": "2025-01-15T12:00:00+00:00",
            "added_by": "admin_rt",
            "notes": "Roundtrip test",
            "metadata": {"key": "value"},
        }

        entry = AllowlistEntry.from_dict(data)

        assert entry.org_id == "org_round"
        assert entry.platform == AdapterPlatform.DISCORD
        assert entry.platform_user_id == "D99999"
        assert entry.notes == "Roundtrip test"
        assert entry.metadata == {"key": "value"}

    def test_to_dict_from_dict_roundtrip(self):
        """AllowlistEntry roundtrip through to_dict/from_dict preserves data."""
        original = make_allowlist_entry(
            org_id="org_rt",
            platform=AdapterPlatform.WEBCHAT,
            platform_user_id="web_session_123",
            added_by="system",
            notes="Auto-approved",
        )

        data = original.to_dict()
        restored = AllowlistEntry.from_dict(data)

        assert restored.org_id == original.org_id
        assert restored.platform == original.platform
        assert restored.platform_user_id == original.platform_user_id
        assert restored.added_by == original.added_by
        assert restored.notes == original.notes

    def test_key_property(self):
        """AllowlistEntry.key combines platform and user_id."""
        entry = make_allowlist_entry(
            platform=AdapterPlatform.SLACK,
            platform_user_id="U12345",
        )
        assert entry.key == "slack:U12345"

    def test_validation_empty_org_id_fails(self):
        """AllowlistEntry raises ValueError for empty org_id."""
        with pytest.raises(ValueError, match="org_id cannot be empty"):
            AllowlistEntry(
                org_id="",
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                added_at=datetime.now(timezone.utc),
                added_by="admin",
            )

    def test_validation_empty_platform_user_id_fails(self):
        """AllowlistEntry raises ValueError for empty platform_user_id."""
        with pytest.raises(ValueError, match="platform_user_id cannot be empty"):
            AllowlistEntry(
                org_id="org_test",
                platform=AdapterPlatform.SLACK,
                platform_user_id="",
                added_at=datetime.now(timezone.utc),
                added_by="admin",
            )

    def test_validation_empty_added_by_fails(self):
        """AllowlistEntry raises ValueError for empty added_by."""
        with pytest.raises(ValueError, match="added_by cannot be empty"):
            AllowlistEntry(
                org_id="org_test",
                platform=AdapterPlatform.SLACK,
                platform_user_id="U12345",
                added_at=datetime.now(timezone.utc),
                added_by="",
            )


class TestInMemoryAllowlistStore:
    """Tests for InMemoryAllowlistStore."""

    @pytest.mark.asyncio
    async def test_add_stores_entry(self, allowlist_store):
        """InMemoryAllowlistStore.add() stores entry correctly."""
        entry = make_allowlist_entry()
        await allowlist_store.add(entry)

        assert await allowlist_store.is_allowed("org_test", "U12345") is True

    @pytest.mark.asyncio
    async def test_is_allowed_returns_true_when_present(self, allowlist_store):
        """InMemoryAllowlistStore.is_allowed() returns True when user is present."""
        entry = make_allowlist_entry(org_id="org_a", platform_user_id="user_1")
        await allowlist_store.add(entry)

        assert await allowlist_store.is_allowed("org_a", "user_1") is True

    @pytest.mark.asyncio
    async def test_is_allowed_returns_false_when_absent(self, allowlist_store):
        """InMemoryAllowlistStore.is_allowed() returns False when user is absent."""
        assert await allowlist_store.is_allowed("nonexistent_org", "U99999") is False

    @pytest.mark.asyncio
    async def test_is_allowed_false_for_wrong_org(self, allowlist_store):
        """InMemoryAllowlistStore.is_allowed() returns False for different org."""
        entry = make_allowlist_entry(org_id="org_a", platform_user_id="user_1")
        await allowlist_store.add(entry)

        assert await allowlist_store.is_allowed("org_b", "user_1") is False

    @pytest.mark.asyncio
    async def test_remove_deletes_entry(self, allowlist_store):
        """InMemoryAllowlistStore.remove() deletes entry correctly."""
        entry = make_allowlist_entry()
        await allowlist_store.add(entry)

        result = await allowlist_store.remove("org_test", "U12345")

        assert result is True
        assert await allowlist_store.is_allowed("org_test", "U12345") is False

    @pytest.mark.asyncio
    async def test_remove_returns_false_for_missing(self, allowlist_store):
        """InMemoryAllowlistStore.remove() returns False for missing entry."""
        result = await allowlist_store.remove("org_nonexistent", "U99999")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_for_org_returns_correct_entries(self, allowlist_store):
        """InMemoryAllowlistStore.list_for_org() returns entries for org."""
        # Add entries to different orgs
        entry1 = make_allowlist_entry(org_id="org_a", platform_user_id="user_1")
        entry2 = make_allowlist_entry(org_id="org_a", platform_user_id="user_2")
        entry3 = make_allowlist_entry(org_id="org_b", platform_user_id="user_3")

        await allowlist_store.add(entry1)
        await allowlist_store.add(entry2)
        await allowlist_store.add(entry3)

        entries_a = await allowlist_store.list_for_org("org_a")
        entries_b = await allowlist_store.list_for_org("org_b")

        assert len(entries_a) == 2
        assert len(entries_b) == 1
        assert all(e.org_id == "org_a" for e in entries_a)

    @pytest.mark.asyncio
    async def test_list_for_org_returns_empty_for_unknown(self, allowlist_store):
        """InMemoryAllowlistStore.list_for_org() returns empty list for unknown org."""
        entries = await allowlist_store.list_for_org("unknown_org")
        assert entries == []

    @pytest.mark.asyncio
    async def test_get_retrieves_specific_entry(self, allowlist_store):
        """InMemoryAllowlistStore.get() retrieves specific entry."""
        entry = make_allowlist_entry(org_id="org_get", platform_user_id="U_GET")
        await allowlist_store.add(entry)

        retrieved = await allowlist_store.get("org_get", "U_GET")

        assert retrieved is not None
        assert retrieved.platform_user_id == "U_GET"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, allowlist_store):
        """InMemoryAllowlistStore.get() returns None for missing entry."""
        retrieved = await allowlist_store.get("org_missing", "U_MISSING")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_count_returns_correct_count(self, allowlist_store):
        """InMemoryAllowlistStore.count() returns correct count for org."""
        for i in range(5):
            entry = make_allowlist_entry(org_id="org_count", platform_user_id=f"user_{i}")
            await allowlist_store.add(entry)

        assert await allowlist_store.count("org_count") == 5
        assert await allowlist_store.count("org_empty") == 0

    @pytest.mark.asyncio
    async def test_clear_empties_all_entries_for_org(self, allowlist_store):
        """InMemoryAllowlistStore.clear() removes all entries for an org."""
        for i in range(3):
            entry = make_allowlist_entry(org_id="org_clear", platform_user_id=f"user_{i}")
            await allowlist_store.add(entry)

        # Also add entry to different org
        other_entry = make_allowlist_entry(org_id="org_other", platform_user_id="other_user")
        await allowlist_store.add(other_entry)

        count = await allowlist_store.clear("org_clear")

        assert count == 3
        assert await allowlist_store.count("org_clear") == 0
        # Other org should be unaffected
        assert await allowlist_store.count("org_other") == 1

    @pytest.mark.asyncio
    async def test_add_updates_existing_entry(self, allowlist_store):
        """InMemoryAllowlistStore.add() updates existing entry."""
        entry1 = make_allowlist_entry(
            org_id="org_update",
            platform_user_id="U_UPDATE",
            notes="Original note",
        )
        await allowlist_store.add(entry1)

        entry2 = make_allowlist_entry(
            org_id="org_update",
            platform_user_id="U_UPDATE",
            notes="Updated note",
        )
        await allowlist_store.add(entry2)

        retrieved = await allowlist_store.get("org_update", "U_UPDATE")
        assert retrieved.notes == "Updated note"
        assert await allowlist_store.count("org_update") == 1

    def test_len_returns_total_entries(self, allowlist_store):
        """InMemoryAllowlistStore.__len__() returns total entries across all orgs."""
        import asyncio

        async def add_entries():
            for i in range(3):
                entry = make_allowlist_entry(org_id="org_a", platform_user_id=f"user_a_{i}")
                await allowlist_store.add(entry)
            for i in range(2):
                entry = make_allowlist_entry(org_id="org_b", platform_user_id=f"user_b_{i}")
                await allowlist_store.add(entry)

        asyncio.get_event_loop().run_until_complete(add_entries())

        assert len(allowlist_store) == 5

    def test_repr_format(self, allowlist_store):
        """InMemoryAllowlistStore repr shows org and entry counts."""
        repr_str = repr(allowlist_store)
        assert "InMemoryAllowlistStore" in repr_str
        assert "orgs=" in repr_str
        assert "entries=" in repr_str


class TestAllowlistStoreAbstract:
    """Tests for AllowlistStore abstract class."""

    def test_allowlist_store_is_abstract(self):
        """AllowlistStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AllowlistStore()


# ===========================================================================
# Exception Tests
# ===========================================================================


class TestPairingExceptions:
    """Tests for pairing-related exceptions."""

    def test_pairing_error_is_base_exception(self):
        """PairingError is the base exception class."""
        assert issubclass(RateLimitExceeded, PairingError)
        assert issubclass(CodeNotFound, PairingError)
        assert issubclass(CodeExpired, PairingError)
        assert issubclass(CodeAlreadyUsed, PairingError)

    def test_rate_limit_exceeded_message(self):
        """RateLimitExceeded has informative error message."""
        exc = RateLimitExceeded("U12345", 3600)
        assert "U12345" in str(exc)
        assert "3600" in str(exc)
        assert exc.user_id == "U12345"
        assert exc.retry_after_seconds == 3600


class TestAllowlistStoreError:
    """Tests for AllowlistStoreError."""

    def test_allowlist_store_error_inherits_exception(self):
        """AllowlistStoreError inherits from Exception."""
        assert issubclass(AllowlistStoreError, Exception)

    def test_allowlist_store_error_can_be_raised(self):
        """AllowlistStoreError can be raised and caught."""
        with pytest.raises(AllowlistStoreError):
            raise AllowlistStoreError("Test error message")
