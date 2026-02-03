"""Discord Bot adapter for Kintsugi CMA.

This module provides Discord integration for the Kintsugi Contextual Memory
Architecture, enabling interaction with the memory-augmented AI assistant
through Discord bots and slash commands.

Example usage:
    from kintsugi.adapters.discord import (
        DiscordAdapter,
        DiscordConfig,
        KintsugiCommands,
        AdminCommands,
    )
    from kintsugi.adapters.shared import PairingManager

    config = DiscordConfig(
        bot_token="your-bot-token",
        application_id="your-app-id",
        require_pairing=True,
    )

    pairing = PairingManager(...)
    adapter = DiscordAdapter(config, pairing)

    # Register commands
    user_commands = KintsugiCommands(adapter, pairing)
    admin_commands = AdminCommands(adapter, pairing, permissions)
"""

# Configuration
from .config import DiscordConfig

# Main adapter
from .bot import (
    DiscordAdapter,
    DiscordClient,
    DiscordMember,
)

# Command cogs
from .cogs import (
    AdminCommands,
    CommandRegistry,
    InteractionResponse,
    KintsugiCommands,
)

# Embed builders
from .embeds import (
    DiscordEmbed,
    EmbedColors,
    EmbedField,
    agent_response_embed,
    error_embed,
    help_embed,
    paired_users_embed,
    pairing_approval_embed,
    pairing_code_embed,
    status_embed,
    success_embed,
    warning_embed,
)

# Permissions
from .permissions import (
    DiscordPermissions,
    PermissionLevel,
)

__all__ = [
    # Configuration
    "DiscordConfig",
    # Main adapter
    "DiscordAdapter",
    "DiscordClient",
    "DiscordMember",
    # Command cogs
    "AdminCommands",
    "CommandRegistry",
    "InteractionResponse",
    "KintsugiCommands",
    # Embed builders
    "DiscordEmbed",
    "EmbedColors",
    "EmbedField",
    "agent_response_embed",
    "error_embed",
    "help_embed",
    "paired_users_embed",
    "pairing_approval_embed",
    "pairing_code_embed",
    "status_embed",
    "success_embed",
    "warning_embed",
    # Permissions
    "DiscordPermissions",
    "PermissionLevel",
]
