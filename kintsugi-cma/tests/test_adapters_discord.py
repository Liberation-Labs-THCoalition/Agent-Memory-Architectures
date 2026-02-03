"""Comprehensive tests for Kintsugi Discord adapter.

Tests cover:
- DiscordConfig validation and role checking
- DiscordAdapter platform, message normalization, user verification, and role checks
- DiscordEmbed creation and serialization
- EmbedField and EmbedColors
- Embed builder functions
- DiscordPermissions level determination and permission checks
- KintsugiCommands, AdminCommands, and CommandRegistry
"""

import pytest
from datetime import datetime, timedelta, timezone, UTC
from unittest.mock import AsyncMock, MagicMock

from kintsugi.adapters.discord import (
    DiscordConfig,
    DiscordAdapter,
    DiscordMember,
    DiscordEmbed,
    EmbedField,
    EmbedColors,
    DiscordPermissions,
    PermissionLevel,
    KintsugiCommands,
    AdminCommands,
    CommandRegistry,
    InteractionResponse,
    pairing_code_embed,
    pairing_approval_embed,
    agent_response_embed,
    error_embed,
    help_embed,
    status_embed,
    success_embed,
    warning_embed,
)
from kintsugi.adapters.shared import (
    AdapterPlatform,
    AdapterMessage,
    PairingManager,
    PairingConfig,
)


# =============================================================================
# DiscordConfig Tests
# =============================================================================

class TestDiscordConfig:
    """Tests for DiscordConfig dataclass."""

    def test_creation_with_required_fields(self):
        """DiscordConfig creation with required fields."""
        config = DiscordConfig(
            bot_token="test-token-123",
            application_id="app-id-456",
        )
        assert config.bot_token == "test-token-123"
        assert config.application_id == "app-id-456"
        assert config.default_org_id is None
        assert config.require_pairing is True
        assert config.command_prefix == "!"
        assert config.allowed_role_ids == []

    def test_validation_fails_without_bot_token(self):
        """DiscordConfig validation fails without bot_token."""
        with pytest.raises(ValueError, match="bot_token is required"):
            DiscordConfig(
                bot_token="",
                application_id="app-id-456",
            )

    def test_validation_fails_without_application_id(self):
        """DiscordConfig validation fails without application_id."""
        with pytest.raises(ValueError, match="application_id is required"):
            DiscordConfig(
                bot_token="test-token-123",
                application_id="",
            )

    def test_is_role_allowed_with_allowed_roles(self):
        """DiscordConfig.is_role_allowed() with allowed roles."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            allowed_role_ids=["role-1", "role-2", "role-3"],
        )
        assert config.is_role_allowed("role-1") is True
        assert config.is_role_allowed("role-2") is True
        assert config.is_role_allowed("role-3") is True
        assert config.is_role_allowed("role-unknown") is False

    def test_is_role_allowed_with_no_allowed_roles(self):
        """DiscordConfig.is_role_allowed() with no allowed roles (all allowed)."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            allowed_role_ids=[],
        )
        # When no roles are configured, any role is allowed
        assert config.is_role_allowed("any-role") is True
        assert config.is_role_allowed("another-role") is True

    def test_default_require_pairing_is_true(self):
        """Default require_pairing is True."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
        )
        assert config.require_pairing is True

    def test_custom_command_prefix(self):
        """Custom command prefix is accepted."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            command_prefix="?",
        )
        assert config.command_prefix == "?"


# =============================================================================
# DiscordAdapter Tests
# =============================================================================

class TestDiscordAdapter:
    """Tests for DiscordAdapter class."""

    @pytest.fixture
    def pairing_manager(self):
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def config(self):
        """Create a DiscordConfig for testing."""
        return DiscordConfig(
            bot_token="test-token",
            application_id="test-app-id",
            default_org_id="org-default",
            require_pairing=True,
            allowed_role_ids=["role-allowed"],
        )

    @pytest.fixture
    def adapter(self, config, pairing_manager):
        """Create a DiscordAdapter for testing."""
        return DiscordAdapter(config, pairing_manager)

    def test_platform_is_discord(self, adapter):
        """DiscordAdapter.platform is AdapterPlatform.DISCORD."""
        assert adapter.platform == AdapterPlatform.DISCORD

    def test_normalize_message_extracts_fields(self, adapter):
        """DiscordAdapter.normalize_message() extracts fields."""
        raw_message = {
            "id": "msg-123",
            "content": "Hello, world!",
            "channel_id": "channel-456",
            "guild_id": "guild-789",
            "timestamp": "2025-01-15T12:00:00Z",
            "author": {
                "id": "user-abc",
                "username": "TestUser",
            },
            "attachments": [
                {"id": "att-1", "filename": "file.txt", "url": "https://cdn.discord.com/file.txt", "size": 1024},
            ],
            "embeds": [],
        }

        message = adapter.normalize_message(raw_message, org_id="test-org")

        assert message.platform == AdapterPlatform.DISCORD
        assert message.platform_user_id == "user-abc"
        assert message.platform_channel_id == "channel-456"
        assert message.org_id == "test-org"
        assert message.content == "Hello, world!"
        assert message.metadata["guild_id"] == "guild-789"
        assert message.metadata["message_id"] == "msg-123"
        assert len(message.attachments) == 1
        assert message.attachments[0]["filename"] == "file.txt"

    @pytest.mark.asyncio
    async def test_verify_user_checks_allowlist_when_paired(self, adapter, pairing_manager):
        """DiscordAdapter.verify_user() checks allowlist."""
        # Add user to allowlist
        pairing_manager._allowlist["org-default"] = {"user-123"}

        result = await adapter.verify_user("user-123", "org-default")
        assert result is True

        result = await adapter.verify_user("user-unknown", "org-default")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_user_returns_true_when_pairing_not_required(self, pairing_manager):
        """verify_user returns True when pairing not required."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            require_pairing=False,
        )
        adapter = DiscordAdapter(config, pairing_manager)

        result = await adapter.verify_user("any-user", "any-org")
        assert result is True

    def test_has_required_role_with_matching_role(self, adapter):
        """DiscordAdapter.has_required_role() with matching role."""
        member_roles = ["role-other", "role-allowed", "role-extra"]
        assert adapter.has_required_role(member_roles) is True

    def test_has_required_role_with_no_matching_role(self, adapter):
        """DiscordAdapter.has_required_role() with no matching role."""
        member_roles = ["role-other", "role-different"]
        assert adapter.has_required_role(member_roles) is False

    def test_has_required_role_with_empty_allowed_role_ids(self, pairing_manager):
        """DiscordAdapter.has_required_role() with empty allowed_role_ids (any allowed)."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            allowed_role_ids=[],  # Empty = allow all
        )
        adapter = DiscordAdapter(config, pairing_manager)

        assert adapter.has_required_role(["any-role"]) is True
        assert adapter.has_required_role([]) is True

    def test_is_command(self, adapter):
        """DiscordAdapter.is_command() detects command prefix."""
        assert adapter.is_command("!help") is True
        assert adapter.is_command("!ask question") is True
        assert adapter.is_command("hello") is False
        assert adapter.is_command("?help") is False

    def test_parse_command(self, adapter):
        """DiscordAdapter.parse_command() extracts command and args."""
        cmd, args = adapter.parse_command("!help")
        assert cmd == "help"
        assert args == []

        cmd, args = adapter.parse_command("!ask What is the weather?")
        assert cmd == "ask"
        assert args == ["What", "is", "the", "weather?"]

        cmd, args = adapter.parse_command("not a command")
        assert cmd == ""
        assert args == []

    @pytest.mark.asyncio
    async def test_start_and_stop(self, adapter):
        """DiscordAdapter start/stop lifecycle."""
        assert adapter.is_started is False

        await adapter.start()
        assert adapter.is_started is True

        await adapter.stop()
        assert adapter.is_started is False


# =============================================================================
# DiscordEmbed Tests
# =============================================================================

class TestDiscordEmbed:
    """Tests for DiscordEmbed dataclass."""

    def test_creation_with_all_fields(self):
        """DiscordEmbed creation with all fields."""
        now = datetime.now(timezone.utc)
        embed = DiscordEmbed(
            title="Test Title",
            description="Test description",
            color=0xFF5733,
            fields=[
                EmbedField(name="Field 1", value="Value 1", inline=True),
                EmbedField(name="Field 2", value="Value 2", inline=False),
            ],
            footer="Footer text",
            timestamp=now,
            url="https://example.com",
            thumbnail_url="https://example.com/thumb.png",
            image_url="https://example.com/image.png",
            author_name="Author",
            author_icon_url="https://example.com/author.png",
        )

        assert embed.title == "Test Title"
        assert embed.description == "Test description"
        assert embed.color == 0xFF5733
        assert len(embed.fields) == 2
        assert embed.footer == "Footer text"
        assert embed.timestamp == now
        assert embed.url == "https://example.com"
        assert embed.thumbnail_url == "https://example.com/thumb.png"
        assert embed.image_url == "https://example.com/image.png"
        assert embed.author_name == "Author"

    def test_to_dict_serialization(self):
        """DiscordEmbed.to_dict() serialization."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        embed = DiscordEmbed(
            title="My Embed",
            description="Description here",
            color=EmbedColors.SUCCESS,
            fields=[EmbedField(name="Info", value="Details", inline=True)],
            footer="Powered by Kintsugi",
            timestamp=now,
        )

        result = embed.to_dict()

        assert result["title"] == "My Embed"
        assert result["description"] == "Description here"
        assert result["color"] == EmbedColors.SUCCESS
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "Info"
        assert result["fields"][0]["value"] == "Details"
        assert result["fields"][0]["inline"] is True
        assert result["footer"]["text"] == "Powered by Kintsugi"
        assert "timestamp" in result

    def test_add_field(self):
        """DiscordEmbed.add_field() adds fields."""
        embed = DiscordEmbed(title="Test")
        embed.add_field("Name 1", "Value 1", inline=True)
        embed.add_field("Name 2", "Value 2", inline=False)

        assert len(embed.fields) == 2
        assert embed.fields[0].name == "Name 1"
        assert embed.fields[0].inline is True
        assert embed.fields[1].name == "Name 2"
        assert embed.fields[1].inline is False


class TestEmbedField:
    """Tests for EmbedField dataclass."""

    def test_creation_and_inline_flag(self):
        """EmbedField creation and inline flag."""
        field1 = EmbedField(name="Test", value="Value")
        assert field1.name == "Test"
        assert field1.value == "Value"
        assert field1.inline is False  # Default

        field2 = EmbedField(name="Test2", value="Value2", inline=True)
        assert field2.inline is True

    def test_to_dict(self):
        """EmbedField.to_dict() serialization."""
        field = EmbedField(name="Name", value="Value", inline=True)
        result = field.to_dict()

        assert result == {"name": "Name", "value": "Value", "inline": True}


class TestEmbedColors:
    """Tests for EmbedColors constants."""

    def test_colors_are_valid_hex_integers(self):
        """EmbedColors constants are valid hex integers."""
        assert isinstance(EmbedColors.SUCCESS, int)
        assert isinstance(EmbedColors.ERROR, int)
        assert isinstance(EmbedColors.WARNING, int)
        assert isinstance(EmbedColors.INFO, int)
        assert isinstance(EmbedColors.KINTSUGI, int)

        # Verify they're in valid Discord color range (0x000000 - 0xFFFFFF)
        assert 0 <= EmbedColors.SUCCESS <= 0xFFFFFF
        assert 0 <= EmbedColors.ERROR <= 0xFFFFFF
        assert 0 <= EmbedColors.WARNING <= 0xFFFFFF
        assert 0 <= EmbedColors.INFO <= 0xFFFFFF
        assert 0 <= EmbedColors.KINTSUGI <= 0xFFFFFF


# =============================================================================
# Embed Builder Function Tests
# =============================================================================

class TestEmbedBuilders:
    """Tests for embed builder functions."""

    def test_pairing_code_embed_structure(self):
        """pairing_code_embed() structure."""
        # Use naive UTC datetime to match source code's datetime.now(timezone.utc)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        embed = pairing_code_embed(code="ABC123", expires_at=expires_at)

        assert embed.title == "Kintsugi Pairing Code"
        assert "pairing code" in embed.description.lower()
        assert embed.color == EmbedColors.KINTSUGI
        assert len(embed.fields) >= 2
        # Should have code field
        code_field = next((f for f in embed.fields if "Code" in f.name), None)
        assert code_field is not None
        assert "ABC123" in code_field.value

    def test_pairing_approval_embed_includes_code(self):
        """pairing_approval_embed() includes code."""
        now = datetime.now(timezone.utc)
        embed = pairing_approval_embed(
            pairing_code="XYZ789",
            user_id="user-123",
            user_name="TestUser",
            requested_at=now,
        )

        assert embed.title == "Pairing Approval Request"
        assert embed.color == EmbedColors.WARNING
        # Check code is in a field
        code_field = next((f for f in embed.fields if "Code" in f.name), None)
        assert code_field is not None
        assert "XYZ789" in code_field.value

    def test_agent_response_embed_formats_response(self):
        """agent_response_embed() formats response."""
        embed = agent_response_embed(
            response="This is the agent's response to your question.",
            processing_time=1.234,
        )

        assert embed.title == "Kintsugi Response"
        assert embed.description == "This is the agent's response to your question."
        assert embed.color == EmbedColors.KINTSUGI
        assert "1.23" in embed.footer  # Processing time

    def test_agent_response_embed_truncates_long_response(self):
        """agent_response_embed() truncates long responses."""
        long_response = "A" * 5000  # Longer than 4000 char limit
        embed = agent_response_embed(response=long_response)

        assert len(embed.description) <= 4003  # 4000 + "..."
        assert embed.description.endswith("...")
        assert "truncated" in embed.footer.lower()

    def test_error_embed_without_suggestion(self):
        """error_embed() with and without suggestion."""
        embed = error_embed(error="Something went wrong.")

        assert embed.title == "Error"
        assert embed.description == "Something went wrong."
        assert embed.color == EmbedColors.ERROR
        assert len(embed.fields) == 0

    def test_error_embed_with_suggestion(self):
        """error_embed() with suggestion."""
        embed = error_embed(
            error="Permission denied.",
            suggestion="Please contact an administrator.",
        )

        assert embed.title == "Error"
        assert embed.description == "Permission denied."
        assert len(embed.fields) == 1
        assert embed.fields[0].name == "Suggestion"
        assert embed.fields[0].value == "Please contact an administrator."

    def test_help_embed_includes_command_info(self):
        """help_embed() includes command info."""
        embed = help_embed()

        assert embed.title == "Kintsugi Help"
        assert embed.color == EmbedColors.INFO
        # Should have command fields
        assert len(embed.fields) >= 4
        # Check for common commands
        field_names = [f.name for f in embed.fields]
        assert "/pair" in field_names
        assert "/status" in field_names

    def test_status_embed_shows_pairing_status(self):
        """status_embed() shows pairing status."""
        # Paired user
        embed_paired = status_embed(
            is_paired=True,
            user_name="TestUser",
            org_name="TestOrg",
            paired_at=datetime.now(timezone.utc),
        )

        assert "Pairing Status" in embed_paired.title
        assert embed_paired.color == EmbedColors.SUCCESS
        assert "paired" in embed_paired.description.lower()

        # Unpaired user
        embed_unpaired = status_embed(
            is_paired=False,
            user_name="TestUser",
        )

        assert embed_unpaired.color == EmbedColors.WARNING
        assert "not paired" in embed_unpaired.description.lower()

    def test_success_embed(self):
        """success_embed() structure."""
        embed = success_embed("Operation Complete", "The task finished successfully.")

        assert embed.title == "Operation Complete"
        assert embed.description == "The task finished successfully."
        assert embed.color == EmbedColors.SUCCESS

    def test_warning_embed(self):
        """warning_embed() structure."""
        embed = warning_embed("Caution", "Please review before proceeding.")

        assert embed.title == "Caution"
        assert embed.description == "Please review before proceeding."
        assert embed.color == EmbedColors.WARNING


# =============================================================================
# DiscordPermissions Tests
# =============================================================================

class TestPermissionLevel:
    """Tests for PermissionLevel enum."""

    def test_permission_level_ordering(self):
        """PermissionLevel ordering (OWNER > ADMIN > MODERATOR > USER > NONE)."""
        assert PermissionLevel.OWNER > PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN > PermissionLevel.MODERATOR
        assert PermissionLevel.MODERATOR > PermissionLevel.USER
        assert PermissionLevel.USER > PermissionLevel.NONE

        assert PermissionLevel.NONE < PermissionLevel.USER
        assert PermissionLevel.USER < PermissionLevel.MODERATOR
        assert PermissionLevel.MODERATOR < PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN < PermissionLevel.OWNER

        assert PermissionLevel.OWNER >= PermissionLevel.OWNER
        assert PermissionLevel.ADMIN <= PermissionLevel.OWNER


class TestDiscordPermissions:
    """Tests for DiscordPermissions class."""

    @pytest.fixture
    def permissions(self):
        """Create a DiscordPermissions for testing."""
        return DiscordPermissions(
            admin_role_ids=["admin-role"],
            moderator_role_ids=["mod-role"],
            user_role_ids=["user-role"],
        )

    def test_get_level_returns_owner_when_is_owner(self, permissions):
        """DiscordPermissions.get_level() returns OWNER when is_owner=True."""
        level = permissions.get_level(member_role_ids=[], is_owner=True)
        assert level == PermissionLevel.OWNER

    def test_get_level_returns_admin_for_admin_role_ids_match(self, permissions):
        """DiscordPermissions.get_level() returns ADMIN for admin_role_ids match."""
        level = permissions.get_level(member_role_ids=["admin-role"], is_owner=False)
        assert level == PermissionLevel.ADMIN

    def test_get_level_returns_moderator_for_mod_roles(self, permissions):
        """DiscordPermissions.get_level() returns MODERATOR for mod roles."""
        level = permissions.get_level(member_role_ids=["mod-role"], is_owner=False)
        assert level == PermissionLevel.MODERATOR

    def test_get_level_returns_user_for_user_roles(self, permissions):
        """DiscordPermissions.get_level() returns USER for user roles."""
        level = permissions.get_level(member_role_ids=["user-role"], is_owner=False)
        assert level == PermissionLevel.USER

    def test_get_level_returns_none_for_no_matches(self, permissions):
        """DiscordPermissions.get_level() returns NONE for no matches."""
        level = permissions.get_level(member_role_ids=["unknown-role"], is_owner=False)
        assert level == PermissionLevel.NONE

        level = permissions.get_level(member_role_ids=[], is_owner=False)
        assert level == PermissionLevel.NONE

    def test_get_level_highest_level_wins(self, permissions):
        """Highest permission level wins when user has multiple roles."""
        # Has both admin and user roles - should be admin
        level = permissions.get_level(
            member_role_ids=["user-role", "admin-role", "mod-role"],
            is_owner=False,
        )
        assert level == PermissionLevel.ADMIN

    def test_can_approve_pairing_for_admin_and_owner_only(self, permissions):
        """can_approve_pairing() for ADMIN and OWNER only."""
        assert permissions.can_approve_pairing(PermissionLevel.OWNER) is True
        assert permissions.can_approve_pairing(PermissionLevel.ADMIN) is True
        assert permissions.can_approve_pairing(PermissionLevel.MODERATOR) is False
        assert permissions.can_approve_pairing(PermissionLevel.USER) is False
        assert permissions.can_approve_pairing(PermissionLevel.NONE) is False

    def test_can_use_bot_for_all_except_none(self, permissions):
        """can_use_bot() for all except NONE."""
        assert permissions.can_use_bot(PermissionLevel.OWNER) is True
        assert permissions.can_use_bot(PermissionLevel.ADMIN) is True
        assert permissions.can_use_bot(PermissionLevel.MODERATOR) is True
        assert permissions.can_use_bot(PermissionLevel.USER) is True
        assert permissions.can_use_bot(PermissionLevel.NONE) is False

    def test_can_revoke_pairing_checks(self, permissions):
        """can_revoke_pairing() checks."""
        assert permissions.can_revoke_pairing(PermissionLevel.OWNER) is True
        assert permissions.can_revoke_pairing(PermissionLevel.ADMIN) is True
        assert permissions.can_revoke_pairing(PermissionLevel.MODERATOR) is False
        assert permissions.can_revoke_pairing(PermissionLevel.USER) is False
        assert permissions.can_revoke_pairing(PermissionLevel.NONE) is False

    def test_can_manage_users(self, permissions):
        """can_manage_users() for moderator and above."""
        assert permissions.can_manage_users(PermissionLevel.OWNER) is True
        assert permissions.can_manage_users(PermissionLevel.ADMIN) is True
        assert permissions.can_manage_users(PermissionLevel.MODERATOR) is True
        assert permissions.can_manage_users(PermissionLevel.USER) is False
        assert permissions.can_manage_users(PermissionLevel.NONE) is False


# =============================================================================
# Commands Tests
# =============================================================================

class TestKintsugiCommands:
    """Tests for KintsugiCommands class."""

    @pytest.fixture
    def pairing_manager(self):
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def adapter(self, pairing_manager):
        """Create a DiscordAdapter for testing."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            default_org_id="test-org",
        )
        return DiscordAdapter(config, pairing_manager)

    @pytest.fixture
    def commands(self, adapter, pairing_manager):
        """Create KintsugiCommands for testing."""
        return KintsugiCommands(adapter, pairing_manager)

    def test_kintsugi_commands_creation(self, commands, adapter, pairing_manager):
        """KintsugiCommands creation."""
        assert commands._adapter is adapter
        assert commands._pairing is pairing_manager

    @pytest.mark.asyncio
    async def test_help_command(self, commands):
        """help_command returns help embed."""
        interaction = {"user": {"id": "user-123"}}
        response = await commands.help_command(interaction)

        assert isinstance(response, InteractionResponse)
        assert response.ephemeral is True
        assert response.embed is not None
        assert response.embed.title == "Kintsugi Help"

    @pytest.mark.asyncio
    async def test_status_command_unpaired(self, commands):
        """status_command for unpaired user."""
        interaction = {
            "user": {"id": "user-123", "username": "TestUser"},
            "guild_id": None,
        }
        response = await commands.status_command(interaction)

        assert isinstance(response, InteractionResponse)
        assert response.ephemeral is True
        assert "not paired" in response.embed.description.lower()


class TestAdminCommands:
    """Tests for AdminCommands class."""

    @pytest.fixture
    def pairing_manager(self):
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def adapter(self, pairing_manager):
        """Create a DiscordAdapter for testing."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            default_org_id="test-org",
        )
        return DiscordAdapter(config, pairing_manager)

    @pytest.fixture
    def permissions(self):
        """Create DiscordPermissions for testing."""
        return DiscordPermissions(
            admin_role_ids=["admin-role"],
            moderator_role_ids=["mod-role"],
            user_role_ids=["user-role"],
        )

    @pytest.fixture
    def admin_commands(self, adapter, pairing_manager, permissions):
        """Create AdminCommands for testing."""
        return AdminCommands(adapter, pairing_manager, permissions)

    def test_admin_commands_creation(self, admin_commands, adapter, pairing_manager, permissions):
        """AdminCommands creation."""
        assert admin_commands._adapter is adapter
        assert admin_commands._pairing is pairing_manager
        assert admin_commands._permissions is permissions

    @pytest.mark.asyncio
    async def test_approve_command_requires_permission(self, admin_commands):
        """approve_command requires admin permission."""
        interaction = {
            "member": {
                "user": {"id": "user-123"},
                "roles": ["user-role"],  # Not admin
                "is_owner": False,
            },
        }
        response = await admin_commands.approve_command(interaction, "SOMECODE")

        assert isinstance(response, InteractionResponse)
        assert response.ephemeral is True
        assert "permission" in response.embed.description.lower()


class TestCommandRegistry:
    """Tests for CommandRegistry class."""

    @pytest.fixture
    def pairing_manager(self):
        """Create a PairingManager for testing."""
        return PairingManager(PairingConfig())

    @pytest.fixture
    def adapter(self, pairing_manager):
        """Create a DiscordAdapter for testing."""
        config = DiscordConfig(
            bot_token="token",
            application_id="app-id",
            default_org_id="test-org",
        )
        return DiscordAdapter(config, pairing_manager)

    @pytest.fixture
    def registry(self, adapter, pairing_manager):
        """Create a CommandRegistry with commands registered."""
        registry = CommandRegistry()
        commands = KintsugiCommands(adapter, pairing_manager)
        registry.register_user_commands(commands)
        return registry

    def test_command_registry_routes_commands(self, registry):
        """CommandRegistry routes commands."""
        assert "pair" in registry._command_handlers
        assert "help" in registry._command_handlers
        assert "status" in registry._command_handlers
        assert "ask" in registry._command_handlers

    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self, registry):
        """dispatch returns error for unknown command."""
        interaction = {"user": {"id": "user-123"}}
        response = await registry.dispatch("unknown-cmd", interaction)

        assert isinstance(response, InteractionResponse)
        assert "Unknown command" in response.embed.description

    @pytest.mark.asyncio
    async def test_dispatch_known_command(self, registry):
        """dispatch routes to correct handler."""
        interaction = {"user": {"id": "user-123"}}
        response = await registry.dispatch("help", interaction)

        assert isinstance(response, InteractionResponse)
        assert response.embed.title == "Kintsugi Help"


class TestInteractionResponse:
    """Tests for InteractionResponse dataclass."""

    def test_interaction_response_structure(self):
        """InteractionResponse structure."""
        embed = DiscordEmbed(title="Test")
        response = InteractionResponse(
            content="Some text",
            embed=embed,
            ephemeral=True,
            deferred=False,
        )

        assert response.content == "Some text"
        assert response.embed is embed
        assert response.ephemeral is True
        assert response.deferred is False

    def test_to_dict_with_ephemeral(self):
        """InteractionResponse.to_dict() includes ephemeral flag."""
        response = InteractionResponse(
            content="Hello",
            ephemeral=True,
        )

        result = response.to_dict()

        assert result["content"] == "Hello"
        assert result["flags"] == 64  # EPHEMERAL flag

    def test_to_dict_with_embed(self):
        """InteractionResponse.to_dict() includes embed."""
        embed = DiscordEmbed(title="Test Title", description="Test desc")
        response = InteractionResponse(embed=embed)

        result = response.to_dict()

        assert "embeds" in result
        assert len(result["embeds"]) == 1
        assert result["embeds"][0]["title"] == "Test Title"


class TestDiscordMember:
    """Tests for DiscordMember dataclass."""

    def test_display_name_returns_nickname(self):
        """display_name returns nickname when available."""
        member = DiscordMember(
            user_id="123",
            username="TestUser",
            discriminator="0",
            nickname="NickName",
        )
        assert member.display_name == "NickName"

    def test_display_name_falls_back_to_username(self):
        """display_name falls back to username."""
        member = DiscordMember(
            user_id="123",
            username="TestUser",
            discriminator="0",
        )
        assert member.display_name == "TestUser"

    def test_from_dict_with_nested_user(self):
        """from_dict handles nested user data."""
        data = {
            "user": {
                "id": "456",
                "username": "NestedUser",
                "discriminator": "1234",
            },
            "roles": ["role-1", "role-2"],
            "nick": "Nickname",
            "guild_id": "guild-789",
            "is_owner": True,
        }

        member = DiscordMember.from_dict(data)

        assert member.user_id == "456"
        assert member.username == "NestedUser"
        assert member.discriminator == "1234"
        assert member.roles == ["role-1", "role-2"]
        assert member.nickname == "Nickname"
        assert member.guild_id == "guild-789"
        assert member.is_owner is True
