"""Discord command cogs for Kintsugi interaction.

This module provides command handlers (cogs) following the discord.py pattern,
organizing bot commands into logical groups for user and admin functionality.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from ..shared import (
    AdapterPlatform,
    AdapterResponse,
    PairingManager,
    PairingStatus as PairingStatusEnum,
    PairingCode,
    CodeNotFound,
    CodeExpired,
    CodeAlreadyUsed,
    RateLimitExceeded,
)

from .bot import DiscordAdapter, _guild_org_mapping
from .embeds import (
    DiscordEmbed,
    EmbedColors,
    agent_response_embed,
    error_embed,
    help_embed,
    paired_users_embed,
    pairing_approval_embed,
    pairing_code_embed,
    status_embed,
    success_embed,
)
from .permissions import DiscordPermissions, PermissionLevel


@dataclass
class InteractionResponse:
    """Response structure for slash command interactions.

    Attributes:
        content: Text content of the response.
        embed: Optional embed for rich formatting.
        ephemeral: Whether the response is only visible to the user.
        deferred: Whether the response was deferred for later followup.
    """

    content: str | None = None
    embed: DiscordEmbed | None = None
    ephemeral: bool = False
    deferred: bool = False

    def to_dict(self) -> dict:
        """Convert to Discord interaction response format.

        Returns:
            Dictionary suitable for Discord API.
        """
        data: dict = {}

        if self.content:
            data["content"] = self.content

        if self.embed:
            data["embeds"] = [self.embed.to_dict()]

        flags = 0
        if self.ephemeral:
            flags |= 64  # EPHEMERAL flag

        if flags:
            data["flags"] = flags

        return data


class KintsugiCommands:
    """Slash commands for Kintsugi interaction.

    Provides user-facing commands for pairing, status checks,
    and direct agent queries.

    Attributes:
        _adapter: The Discord adapter instance.
        _pairing: The pairing manager for user verification.
    """

    def __init__(self, adapter: DiscordAdapter, pairing: PairingManager) -> None:
        """Initialize the Kintsugi commands cog.

        Args:
            adapter: The Discord adapter instance.
            pairing: The pairing manager for user verification.
        """
        self._adapter = adapter
        self._pairing = pairing

    async def pair_command(self, interaction: dict) -> InteractionResponse:
        """Handle /pair slash command - generate pairing code.

        Generates a unique pairing code for the user and sends it via DM.
        The code must be approved by an administrator to complete pairing.

        Args:
            interaction: The Discord interaction data.

        Returns:
            InteractionResponse with confirmation or error.
        """
        user = interaction.get("user", interaction.get("member", {}).get("user", {}))
        user_id = str(user.get("id", ""))
        username = user.get("username", "")
        guild_id = str(interaction.get("guild_id", "")) if interaction.get("guild_id") else None

        if not user_id:
            return InteractionResponse(
                embed=error_embed(
                    "Could not identify user.",
                    "Please try again or contact support.",
                ),
                ephemeral=True,
            )

        # Determine org_id from guild mapping or default
        org_id = None
        if guild_id and guild_id in _guild_org_mapping:
            org_id = _guild_org_mapping[guild_id]
        if org_id is None:
            org_id = self._adapter.config.default_org_id

        if org_id is None:
            return InteractionResponse(
                embed=error_embed(
                    "No organization configured for this server.",
                    "Please contact a system administrator.",
                ),
                ephemeral=True,
            )

        # Check if already paired
        if self._pairing.is_allowed(org_id, user_id):
            return InteractionResponse(
                embed=error_embed(
                    "You are already paired with Kintsugi.",
                    "Use `/status` to check your pairing details.",
                ),
                ephemeral=True,
            )

        # Generate pairing code
        try:
            pairing_code = self._pairing.generate_code(
                platform=AdapterPlatform.DISCORD,
                platform_user_id=user_id,
                org_id=org_id,
                channel_id=str(interaction.get("channel_id", "")),
                metadata={"username": username},
            )
        except RateLimitExceeded as e:
            return InteractionResponse(
                embed=error_embed(
                    "Too many pairing attempts.",
                    f"Please try again in {e.retry_after_seconds // 60} minutes.",
                ),
                ephemeral=True,
            )

        # Send code via DM
        try:
            dm_embed = pairing_code_embed(
                code=pairing_code.code,
                expires_at=pairing_code.expires_at,
            )
            await self._adapter.send_dm(
                user_id=user_id,
                response=AdapterResponse(
                    content="Your Kintsugi pairing code",
                    metadata={"embed": dm_embed.to_dict()},
                ),
            )
        except Exception:
            return InteractionResponse(
                embed=error_embed(
                    "Could not send DM with pairing code.",
                    "Please enable DMs from server members and try again.",
                ),
                ephemeral=True,
            )

        return InteractionResponse(
            embed=success_embed(
                "Pairing Code Generated",
                "A pairing code has been sent to your DMs. "
                "Share it with an administrator to complete pairing.",
            ),
            ephemeral=True,
        )

    async def help_command(self, interaction: dict) -> InteractionResponse:
        """Handle /help slash command.

        Displays available commands and their usage.

        Args:
            interaction: The Discord interaction data.

        Returns:
            InteractionResponse with help information.
        """
        return InteractionResponse(
            embed=help_embed(),
            ephemeral=True,
        )

    async def status_command(self, interaction: dict) -> InteractionResponse:
        """Handle /status - check pairing status.

        Shows the user's current pairing status and associated information.

        Args:
            interaction: The Discord interaction data.

        Returns:
            InteractionResponse with status information.
        """
        user = interaction.get("user", interaction.get("member", {}).get("user", {}))
        user_id = str(user.get("id", ""))
        username = user.get("username", "")
        guild_id = str(interaction.get("guild_id", "")) if interaction.get("guild_id") else None

        if not user_id:
            return InteractionResponse(
                embed=error_embed(
                    "Could not identify user.",
                    "Please try again or contact support.",
                ),
                ephemeral=True,
            )

        # Determine org_id from guild mapping or default
        org_id = None
        if guild_id and guild_id in _guild_org_mapping:
            org_id = _guild_org_mapping[guild_id]
        if org_id is None:
            org_id = self._adapter.config.default_org_id

        # Check if user is paired
        is_paired = org_id is not None and self._pairing.is_allowed(org_id, user_id)

        if is_paired:
            # Find the approved pairing code for more details
            paired_at = None
            for code in self._pairing._codes.values():
                if (code.platform_user_id == user_id and
                    code.org_id == org_id and
                    code.status == PairingStatusEnum.APPROVED):
                    paired_at = code.approved_at
                    break

            return InteractionResponse(
                embed=status_embed(
                    is_paired=True,
                    user_name=username,
                    org_name=org_id,  # Would ideally look up org name
                    paired_at=paired_at,
                ),
                ephemeral=True,
            )
        else:
            return InteractionResponse(
                embed=status_embed(
                    is_paired=False,
                    user_name=username,
                ),
                ephemeral=True,
            )

    async def ask_command(
        self,
        interaction: dict,
        question: str,
        agent_callback: Any = None,
    ) -> InteractionResponse:
        """Handle /ask <question> - direct agent query.

        Forwards the user's question to the Kintsugi agent and returns
        the response. Requires the user to be paired.

        Args:
            interaction: The Discord interaction data.
            question: The user's question to the agent.
            agent_callback: Optional callback for agent query processing.

        Returns:
            InteractionResponse with agent response or error.
        """
        user = interaction.get("user", interaction.get("member", {}).get("user", {}))
        user_id = str(user.get("id", ""))
        guild_id = str(interaction.get("guild_id", "")) if interaction.get("guild_id") else None

        if not user_id:
            return InteractionResponse(
                embed=error_embed(
                    "Could not identify user.",
                    "Please try again or contact support.",
                ),
                ephemeral=True,
            )

        # Check pairing status
        if self._adapter.config.require_pairing:
            org_id = self._adapter.get_user_org(user_id, guild_id)
            if org_id is None:
                return InteractionResponse(
                    embed=error_embed(
                        "You must be paired to use this command.",
                        "Use `/pair` to generate a pairing code.",
                    ),
                    ephemeral=True,
                )
        else:
            org_id = self._adapter.config.default_org_id

        # Validate question
        if not question or not question.strip():
            return InteractionResponse(
                embed=error_embed(
                    "Please provide a question.",
                    "Usage: `/ask <your question>`",
                ),
                ephemeral=True,
            )

        # Process through agent (placeholder for actual agent integration)
        start_time = datetime.now(timezone.utc)

        if agent_callback:
            try:
                response_text = await agent_callback(
                    user_id=user_id,
                    org_id=org_id,
                    question=question.strip(),
                )
            except Exception:
                return InteractionResponse(
                    embed=error_embed(
                        "An error occurred processing your question.",
                        "Please try again later.",
                    ),
                    ephemeral=True,
                )
        else:
            # Placeholder response when no agent callback
            response_text = (
                f"[Agent response placeholder]\n\n"
                f"Question: {question.strip()}\n"
                f"User: {user_id}\n"
                f"Org: {org_id}"
            )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        return InteractionResponse(
            embed=agent_response_embed(
                response=response_text,
                processing_time=processing_time,
            ),
            ephemeral=False,
        )


class AdminCommands:
    """Admin-only commands for organization management.

    Provides commands for managing user pairings and access control.
    All commands require appropriate permission levels.

    Attributes:
        _adapter: The Discord adapter instance.
        _pairing: The pairing manager for user verification.
        _permissions: The permissions configuration.
    """

    def __init__(
        self,
        adapter: DiscordAdapter,
        pairing: PairingManager,
        permissions: DiscordPermissions,
    ) -> None:
        """Initialize the admin commands cog.

        Args:
            adapter: The Discord adapter instance.
            pairing: The pairing manager for user verification.
            permissions: The permissions configuration.
        """
        self._adapter = adapter
        self._pairing = pairing
        self._permissions = permissions

    async def _check_admin_permission(
        self, interaction: dict
    ) -> tuple[bool, str | None, PermissionLevel]:
        """Check if the user has admin permission.

        Args:
            interaction: The Discord interaction data.

        Returns:
            Tuple of (has_permission, user_id, permission_level).
        """
        member = interaction.get("member", {})
        user = member.get("user", interaction.get("user", {}))
        user_id = str(user.get("id", ""))
        roles = [str(r) for r in member.get("roles", [])]
        is_owner = member.get("is_owner", False)

        level = self._permissions.get_level(roles, is_owner)
        has_permission = self._permissions.can_approve_pairing(level)

        return has_permission, user_id, level

    async def approve_command(
        self, interaction: dict, code: str
    ) -> InteractionResponse:
        """Approve a pairing request.

        Approves a pending pairing code, linking the user to the organization.
        Requires admin or owner permission.

        Args:
            interaction: The Discord interaction data.
            code: The pairing code to approve.

        Returns:
            InteractionResponse with approval result.
        """
        has_permission, admin_id, level = await self._check_admin_permission(
            interaction
        )

        if not has_permission:
            return InteractionResponse(
                embed=error_embed(
                    "You do not have permission to approve pairings.",
                    "This command requires admin privileges.",
                ),
                ephemeral=True,
            )

        if not code or not code.strip():
            return InteractionResponse(
                embed=error_embed(
                    "Please provide a pairing code.",
                    "Usage: `/approve <code>`",
                ),
                ephemeral=True,
            )

        # Approve the pairing using PairingManager
        try:
            pairing_code = self._pairing.approve(
                code=code.strip(),
                approver=admin_id or "unknown",
            )

            return InteractionResponse(
                embed=success_embed(
                    "Pairing Approved",
                    f"User <@{pairing_code.platform_user_id}> has been paired with this organization.",
                ),
                ephemeral=False,
            )
        except CodeNotFound:
            return InteractionResponse(
                embed=error_embed(
                    "Pairing code not found.",
                    "Please check the code and try again.",
                ),
                ephemeral=True,
            )
        except CodeExpired:
            return InteractionResponse(
                embed=error_embed(
                    "Pairing code has expired.",
                    "The user needs to generate a new code with `/pair`.",
                ),
                ephemeral=True,
            )
        except CodeAlreadyUsed:
            return InteractionResponse(
                embed=error_embed(
                    "Pairing code has already been used.",
                    "This code was already approved or rejected.",
                ),
                ephemeral=True,
            )
        except Exception:
            return InteractionResponse(
                embed=error_embed(
                    "An error occurred while approving the pairing.",
                    "Please try again later.",
                ),
                ephemeral=True,
            )

    async def revoke_command(
        self, interaction: dict, user_id: str
    ) -> InteractionResponse:
        """Revoke a user's access.

        Removes a user's pairing, revoking their access to the organization.
        Requires admin or owner permission.

        Args:
            interaction: The Discord interaction data.
            user_id: The Discord user ID or mention to revoke.

        Returns:
            InteractionResponse with revocation result.
        """
        has_permission, admin_id, level = await self._check_admin_permission(
            interaction
        )

        if not has_permission:
            return InteractionResponse(
                embed=error_embed(
                    "You do not have permission to revoke pairings.",
                    "This command requires admin privileges.",
                ),
                ephemeral=True,
            )

        if not user_id or not user_id.strip():
            return InteractionResponse(
                embed=error_embed(
                    "Please provide a user to revoke.",
                    "Usage: `/revoke <@user>` or `/revoke <user_id>`",
                ),
                ephemeral=True,
            )

        # Extract user ID from mention if necessary
        target_id = user_id.strip()
        if target_id.startswith("<@") and target_id.endswith(">"):
            target_id = target_id[2:-1]
            if target_id.startswith("!"):
                target_id = target_id[1:]

        # Determine org_id from guild mapping or default
        guild_id = str(interaction.get("guild_id", "")) if interaction.get("guild_id") else None
        org_id = None
        if guild_id and guild_id in _guild_org_mapping:
            org_id = _guild_org_mapping[guild_id]
        if org_id is None:
            org_id = self._adapter.config.default_org_id

        if org_id is None:
            return InteractionResponse(
                embed=error_embed(
                    "No organization configured for this server.",
                    "Please contact a system administrator.",
                ),
                ephemeral=True,
            )

        # Revoke the pairing using PairingManager
        try:
            revoked = self._pairing.revoke(
                org_id=org_id,
                platform_user_id=target_id,
                revoker=admin_id or "unknown",
            )

            if revoked:
                return InteractionResponse(
                    embed=success_embed(
                        "Access Revoked",
                        f"User <@{target_id}>'s access has been revoked.",
                    ),
                    ephemeral=False,
                )
            else:
                return InteractionResponse(
                    embed=error_embed(
                        "Failed to revoke access.",
                        "The user may not be paired with this organization.",
                    ),
                    ephemeral=True,
                )
        except Exception:
            return InteractionResponse(
                embed=error_embed(
                    "An error occurred while revoking access.",
                    "Please try again later.",
                ),
                ephemeral=True,
            )

    async def list_paired_command(
        self, interaction: dict, page: int = 1
    ) -> InteractionResponse:
        """List all paired users for the organization.

        Shows all users currently paired with the organization.
        Requires moderator or higher permission.

        Args:
            interaction: The Discord interaction data.
            page: The page number for pagination (default 1).

        Returns:
            InteractionResponse with paired users list.
        """
        member = interaction.get("member", {})
        roles = [str(r) for r in member.get("roles", [])]
        is_owner = member.get("is_owner", False)

        level = self._permissions.get_level(roles, is_owner)

        if not self._permissions.can_view_all_paired(level):
            return InteractionResponse(
                embed=error_embed(
                    "You do not have permission to view paired users.",
                    "This command requires moderator privileges.",
                ),
                ephemeral=True,
            )

        # Determine org_id from guild mapping or default
        guild_id = str(interaction.get("guild_id", "")) if interaction.get("guild_id") else None
        org_id = None
        if guild_id and guild_id in _guild_org_mapping:
            org_id = _guild_org_mapping[guild_id]
        if org_id is None:
            org_id = self._adapter.config.default_org_id

        if org_id is None:
            return InteractionResponse(
                embed=error_embed(
                    "No organization configured for this server.",
                    "Please contact a system administrator.",
                ),
                ephemeral=True,
            )

        # Get paired users from allowlist
        try:
            allowed_users = self._pairing.get_allowlist(org_id)

            # Build user list from approved pairing codes for metadata
            users: list[dict] = []
            for user_id in allowed_users:
                user_info: dict[str, Any] = {"id": user_id, "name": user_id}
                # Find the pairing code for more details
                for code in self._pairing._codes.values():
                    if (code.platform_user_id == user_id and
                        code.org_id == org_id and
                        code.status == PairingStatusEnum.APPROVED):
                        user_info["name"] = code.metadata.get("username", user_id)
                        user_info["paired_at"] = code.approved_at
                        break
                users.append(user_info)

            # Simple pagination
            page_size = 10
            total_users = len(users)
            total_pages = max(1, (total_users + page_size - 1) // page_size)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_users = users[start_idx:end_idx]

            return InteractionResponse(
                embed=paired_users_embed(
                    users=page_users,
                    org_name=org_id,
                    page=page,
                    total_pages=total_pages,
                ),
                ephemeral=True,
            )
        except Exception:
            return InteractionResponse(
                embed=error_embed(
                    "An error occurred while listing paired users.",
                    "Please try again later.",
                ),
                ephemeral=True,
            )


class CommandRegistry:
    """Registry for managing command cogs and dispatching interactions.

    Provides a central point for registering command handlers and
    routing interactions to the appropriate handler.
    """

    def __init__(self) -> None:
        """Initialize the command registry."""
        self._user_commands: KintsugiCommands | None = None
        self._admin_commands: AdminCommands | None = None
        self._command_handlers: dict[str, Any] = {}

    def register_user_commands(self, commands: KintsugiCommands) -> None:
        """Register user command handlers.

        Args:
            commands: The KintsugiCommands instance.
        """
        self._user_commands = commands
        self._command_handlers["pair"] = commands.pair_command
        self._command_handlers["help"] = commands.help_command
        self._command_handlers["status"] = commands.status_command
        self._command_handlers["ask"] = commands.ask_command

    def register_admin_commands(self, commands: AdminCommands) -> None:
        """Register admin command handlers.

        Args:
            commands: The AdminCommands instance.
        """
        self._admin_commands = commands
        self._command_handlers["approve"] = commands.approve_command
        self._command_handlers["revoke"] = commands.revoke_command
        self._command_handlers["list-paired"] = commands.list_paired_command

    async def dispatch(
        self, command_name: str, interaction: dict, **kwargs: Any
    ) -> InteractionResponse:
        """Dispatch an interaction to the appropriate handler.

        Args:
            command_name: The name of the command to dispatch.
            interaction: The Discord interaction data.
            **kwargs: Additional arguments for the command handler.

        Returns:
            InteractionResponse from the command handler.
        """
        handler = self._command_handlers.get(command_name)

        if handler is None:
            return InteractionResponse(
                embed=error_embed(
                    f"Unknown command: {command_name}",
                    "Use `/help` to see available commands.",
                ),
                ephemeral=True,
            )

        return await handler(interaction, **kwargs)
