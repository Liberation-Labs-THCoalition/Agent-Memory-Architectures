"""
Plugin registry for Kintsugi CMA.

This module provides the central registry for managing active plugins.
The registry organizes plugins by their interface type and provides
methods for querying and invoking plugins.

Registry Features:
    - Registration by interface type
    - Plugin lookup by name or capability
    - Middleware chain management
    - Plugin event notifications
    - Health monitoring

Example:
    from kintsugi.plugins.registry import PluginRegistry
    from kintsugi.plugins.loader import PluginLoader

    registry = PluginRegistry()
    loader = PluginLoader()

    # Load and register plugins
    for metadata in loader.discover():
        loaded = loader.load(metadata.name)
        registry.register(loaded)

    # Get all skill chips
    chips = registry.get_all_skill_chips()

    # Process request through middleware
    request = await registry.process_request_middleware(request)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
import asyncio
import logging

from kintsugi.plugins.sdk import (
    PluginMetadata,
    SkillChipPlugin,
    AdapterPlugin,
    StoragePlugin,
    MiddlewarePlugin,
)
from kintsugi.plugins.loader import LoadedPlugin, PluginState

logger = logging.getLogger(__name__)


@dataclass
class RegisteredPlugin:
    """A plugin registered in the registry.

    Tracks additional metadata about a registered plugin beyond
    what's in LoadedPlugin.

    Attributes:
        loaded: The loaded plugin object.
        registered_at: When the plugin was registered.
        enabled: Whether the plugin is currently enabled.
        priority: Execution priority for middleware.
        health_status: Current health status.
        last_health_check: When health was last checked.
        call_count: Number of times the plugin has been invoked.
        error_count: Number of errors during invocation.
    """

    loaded: LoadedPlugin
    registered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    enabled: bool = True
    priority: int = 100
    health_status: str = "unknown"
    last_health_check: datetime | None = None
    call_count: int = 0
    error_count: int = 0

    @property
    def name(self) -> str:
        """Get the plugin name."""
        return self.loaded.metadata.name

    @property
    def metadata(self) -> PluginMetadata:
        """Get the plugin metadata."""
        return self.loaded.metadata

    @property
    def instance(self) -> Any:
        """Get the plugin instance."""
        return self.loaded.instance

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "metadata": self.metadata.to_dict(),
            "registered_at": self.registered_at.isoformat(),
            "enabled": self.enabled,
            "priority": self.priority,
            "health_status": self.health_status,
            "call_count": self.call_count,
            "error_count": self.error_count,
        }


@dataclass
class PluginEvent:
    """An event from the plugin system.

    Used to notify listeners of plugin lifecycle events.

    Attributes:
        event_type: Type of event.
        plugin_name: Name of the affected plugin.
        timestamp: When the event occurred.
        details: Additional event details.
    """

    event_type: str
    plugin_name: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    details: dict[str, Any] = field(default_factory=dict)


# Type for event listeners
EventListener = Callable[[PluginEvent], None]


class PluginRegistry:
    """Central registry for active plugins.

    The registry maintains collections of plugins organized by their
    interface type and provides methods for plugin discovery, invocation,
    and lifecycle management.

    Attributes:
        _skill_chips: Registered skill chip plugins.
        _adapters: Registered adapter plugins.
        _storage: Registered storage plugins.
        _middleware: Registered middleware plugins (ordered by priority).
        _all_plugins: All registered plugins by name.
        _event_listeners: Listeners for plugin events.

    Example:
        registry = PluginRegistry()

        # Register a loaded plugin
        registry.register(loaded_plugin)

        # Get skill chips
        chips = registry.get_all_skill_chips()

        # Find adapter by platform
        adapter = registry.get_adapter_by_platform("teams")

        # Process middleware chain
        request = await registry.process_request_middleware(request)
        response = await registry.process_response_middleware(response)
    """

    def __init__(self):
        """Initialize the plugin registry."""
        self._skill_chips: dict[str, RegisteredPlugin] = {}
        self._adapters: dict[str, RegisteredPlugin] = {}
        self._storage: dict[str, RegisteredPlugin] = {}
        self._middleware: list[RegisteredPlugin] = []
        self._all_plugins: dict[str, RegisteredPlugin] = {}
        self._event_listeners: list[EventListener] = []

        # Intent to skill chip mapping
        self._intent_map: dict[str, str] = {}

        # Platform to adapter mapping
        self._platform_map: dict[str, str] = {}

    def register(self, plugin: LoadedPlugin, priority: int = 100) -> bool:
        """Register a plugin.

        Automatically detects the plugin type and registers it in
        the appropriate collection.

        Args:
            plugin: The loaded plugin to register.
            priority: Priority for middleware execution (lower = earlier).

        Returns:
            True if registered successfully.
        """
        if plugin.state != PluginState.LOADED:
            logger.warning(
                f"Cannot register plugin in state {plugin.state}: {plugin.metadata.name}"
            )
            return False

        registered = RegisteredPlugin(
            loaded=plugin,
            priority=priority,
        )

        name = plugin.metadata.name

        # Register by type
        if plugin.plugin_type == "skill_chip":
            self._register_skill_chip(registered)
        elif plugin.plugin_type == "adapter":
            self._register_adapter(registered)
        elif plugin.plugin_type == "storage":
            self._register_storage(registered)
        elif plugin.plugin_type == "middleware":
            self._register_middleware(registered)
        else:
            logger.warning(f"Unknown plugin type: {plugin.plugin_type}")

        self._all_plugins[name] = registered

        self._emit_event(PluginEvent(
            event_type="plugin_registered",
            plugin_name=name,
            details={"plugin_type": plugin.plugin_type},
        ))

        logger.info(f"Registered plugin: {name} (type: {plugin.plugin_type})")
        return True

    def _register_skill_chip(self, registered: RegisteredPlugin) -> None:
        """Register a skill chip plugin.

        Args:
            registered: The registered plugin.
        """
        name = registered.name
        self._skill_chips[name] = registered

        # Map intents to this chip
        if registered.instance and hasattr(registered.instance, 'get_intents'):
            try:
                intents = registered.instance.get_intents()
                for intent in intents:
                    self._intent_map[intent] = name
                    logger.debug(f"Mapped intent '{intent}' to chip '{name}'")
            except Exception as e:
                logger.error(f"Failed to get intents from {name}: {e}")

    def _register_adapter(self, registered: RegisteredPlugin) -> None:
        """Register an adapter plugin.

        Args:
            registered: The registered plugin.
        """
        name = registered.name
        self._adapters[name] = registered

        # Map platform to this adapter
        if registered.instance and hasattr(registered.instance, 'get_platform_name'):
            try:
                platform = registered.instance.get_platform_name()
                self._platform_map[platform] = name
                logger.debug(f"Mapped platform '{platform}' to adapter '{name}'")
            except Exception as e:
                logger.error(f"Failed to get platform from {name}: {e}")

    def _register_storage(self, registered: RegisteredPlugin) -> None:
        """Register a storage plugin.

        Args:
            registered: The registered plugin.
        """
        self._storage[registered.name] = registered

    def _register_middleware(self, registered: RegisteredPlugin) -> None:
        """Register a middleware plugin.

        Args:
            registered: The registered plugin.
        """
        # Get priority from plugin if available
        if registered.instance and hasattr(registered.instance, 'get_priority'):
            try:
                registered.priority = registered.instance.get_priority()
            except Exception:
                pass

        self._middleware.append(registered)
        # Sort by priority (lower = earlier in chain)
        self._middleware.sort(key=lambda m: m.priority)

    def register_skill_chip(self, plugin: SkillChipPlugin) -> None:
        """Register a skill chip plugin directly.

        Alternative registration method for skill chip plugins
        without going through the loader.

        Args:
            plugin: The skill chip plugin to register.
        """
        loaded = LoadedPlugin(
            metadata=plugin.metadata,
            module=None,
            state=PluginState.LOADED,
            loaded_at=datetime.now(timezone.utc),
            instance=plugin,
            plugin_type="skill_chip",
        )
        self.register(loaded)

    def register_adapter(self, plugin: AdapterPlugin) -> None:
        """Register an adapter plugin directly.

        Args:
            plugin: The adapter plugin to register.
        """
        loaded = LoadedPlugin(
            metadata=plugin.metadata,
            module=None,
            state=PluginState.LOADED,
            loaded_at=datetime.now(timezone.utc),
            instance=plugin,
            plugin_type="adapter",
        )
        self.register(loaded)

    def register_storage(self, plugin: StoragePlugin) -> None:
        """Register a storage plugin directly.

        Args:
            plugin: The storage plugin to register.
        """
        loaded = LoadedPlugin(
            metadata=plugin.metadata,
            module=None,
            state=PluginState.LOADED,
            loaded_at=datetime.now(timezone.utc),
            instance=plugin,
            plugin_type="storage",
        )
        self.register(loaded)

    def register_middleware(self, plugin: MiddlewarePlugin, priority: int = 100) -> None:
        """Register a middleware plugin directly.

        Args:
            plugin: The middleware plugin to register.
            priority: Execution priority (lower = earlier).
        """
        loaded = LoadedPlugin(
            metadata=plugin.metadata,
            module=None,
            state=PluginState.LOADED,
            loaded_at=datetime.now(timezone.utc),
            instance=plugin,
            plugin_type="middleware",
        )
        self.register(loaded, priority=priority)

    def unregister(self, plugin_name: str) -> bool:
        """Unregister a plugin.

        Args:
            plugin_name: Name of the plugin to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        if plugin_name not in self._all_plugins:
            return False

        registered = self._all_plugins.pop(plugin_name)

        # Remove from type-specific registry
        if plugin_name in self._skill_chips:
            del self._skill_chips[plugin_name]
            # Remove from intent map
            self._intent_map = {
                k: v for k, v in self._intent_map.items()
                if v != plugin_name
            }
        elif plugin_name in self._adapters:
            del self._adapters[plugin_name]
            # Remove from platform map
            self._platform_map = {
                k: v for k, v in self._platform_map.items()
                if v != plugin_name
            }
        elif plugin_name in self._storage:
            del self._storage[plugin_name]
        else:
            # Check middleware
            self._middleware = [
                m for m in self._middleware
                if m.name != plugin_name
            ]

        self._emit_event(PluginEvent(
            event_type="plugin_unregistered",
            plugin_name=plugin_name,
        ))

        logger.info(f"Unregistered plugin: {plugin_name}")
        return True

    def get_plugin(self, name: str) -> RegisteredPlugin | None:
        """Get a registered plugin by name.

        Args:
            name: Plugin name.

        Returns:
            RegisteredPlugin if found, None otherwise.
        """
        return self._all_plugins.get(name)

    def get_all_plugins(self) -> list[RegisteredPlugin]:
        """Get all registered plugins.

        Returns:
            List of all registered plugins.
        """
        return list(self._all_plugins.values())

    def get_all_skill_chips(self) -> list[Any]:
        """Get all registered skill chips.

        Returns the actual chip instances (via get_chip()) for
        integration with the skill router.

        Returns:
            List of BaseSkillChip instances.
        """
        chips = []
        for registered in self._skill_chips.values():
            if not registered.enabled:
                continue
            if registered.instance and hasattr(registered.instance, 'get_chip'):
                try:
                    chip = registered.instance.get_chip()
                    if chip:
                        chips.append(chip)
                except Exception as e:
                    logger.error(f"Failed to get chip from {registered.name}: {e}")
        return chips

    def get_skill_chip_for_intent(self, intent: str) -> Any | None:
        """Get the skill chip that handles an intent.

        Args:
            intent: The intent to look up.

        Returns:
            The skill chip instance, or None if not found.
        """
        plugin_name = self._intent_map.get(intent)
        if not plugin_name:
            return None

        registered = self._skill_chips.get(plugin_name)
        if not registered or not registered.enabled:
            return None

        if registered.instance and hasattr(registered.instance, 'get_chip'):
            return registered.instance.get_chip()

        return None

    def get_all_adapters(self) -> list[Any]:
        """Get all registered adapters.

        Returns the actual adapter instances (via get_adapter()).

        Returns:
            List of BaseAdapter instances.
        """
        adapters = []
        for registered in self._adapters.values():
            if not registered.enabled:
                continue
            if registered.instance and hasattr(registered.instance, 'get_adapter'):
                try:
                    adapter = registered.instance.get_adapter()
                    if adapter:
                        adapters.append(adapter)
                except Exception as e:
                    logger.error(f"Failed to get adapter from {registered.name}: {e}")
        return adapters

    def get_adapter_by_platform(self, platform: str) -> Any | None:
        """Get an adapter by platform name.

        Args:
            platform: The platform identifier (e.g., "teams").

        Returns:
            The adapter instance, or None if not found.
        """
        plugin_name = self._platform_map.get(platform)
        if not plugin_name:
            return None

        registered = self._adapters.get(plugin_name)
        if not registered or not registered.enabled:
            return None

        if registered.instance and hasattr(registered.instance, 'get_adapter'):
            return registered.instance.get_adapter()

        return None

    def get_storage_plugin(self, name: str) -> StoragePlugin | None:
        """Get a storage plugin by name.

        Args:
            name: The storage plugin name.

        Returns:
            The storage plugin instance, or None if not found.
        """
        registered = self._storage.get(name)
        if not registered or not registered.enabled:
            return None
        return registered.instance

    def get_default_storage(self) -> StoragePlugin | None:
        """Get the default storage plugin.

        Returns the first enabled storage plugin.

        Returns:
            A storage plugin instance, or None if none registered.
        """
        for registered in self._storage.values():
            if registered.enabled:
                return registered.instance
        return None

    async def process_request_middleware(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a request through all middleware.

        Passes the request through each middleware's process_request
        method in priority order.

        Args:
            request: The request dictionary.

        Returns:
            The processed request.
        """
        for registered in self._middleware:
            if not registered.enabled:
                continue

            try:
                registered.call_count += 1
                request = await registered.instance.process_request(request)
            except Exception as e:
                registered.error_count += 1
                logger.error(
                    f"Middleware {registered.name} failed processing request: {e}"
                )
                # Continue with other middleware

        return request

    async def process_response_middleware(
        self,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a response through all middleware.

        Passes the response through each middleware's process_response
        method in reverse priority order (highest priority first).

        Args:
            response: The response dictionary.

        Returns:
            The processed response.
        """
        for registered in reversed(self._middleware):
            if not registered.enabled:
                continue

            try:
                registered.call_count += 1
                response = await registered.instance.process_response(response)
            except Exception as e:
                registered.error_count += 1
                logger.error(
                    f"Middleware {registered.name} failed processing response: {e}"
                )
                # Continue with other middleware

        return response

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if enabled, False if not found.
        """
        registered = self._all_plugins.get(name)
        if not registered:
            return False

        registered.enabled = True
        self._emit_event(PluginEvent(
            event_type="plugin_enabled",
            plugin_name=name,
        ))
        return True

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if disabled, False if not found.
        """
        registered = self._all_plugins.get(name)
        if not registered:
            return False

        registered.enabled = False
        self._emit_event(PluginEvent(
            event_type="plugin_disabled",
            plugin_name=name,
        ))
        return True

    async def health_check_all(self) -> dict[str, str]:
        """Run health checks on all plugins.

        Returns:
            Dictionary mapping plugin name to health status.
        """
        results: dict[str, str] = {}

        for name, registered in self._all_plugins.items():
            status = await self._check_plugin_health(registered)
            results[name] = status

        return results

    async def _check_plugin_health(self, registered: RegisteredPlugin) -> str:
        """Check health of a single plugin.

        Args:
            registered: The registered plugin.

        Returns:
            Health status string.
        """
        if not registered.enabled:
            return "disabled"

        if registered.instance and hasattr(registered.instance, 'health_check'):
            try:
                if asyncio.iscoroutinefunction(registered.instance.health_check):
                    healthy = await registered.instance.health_check()
                else:
                    healthy = registered.instance.health_check()

                status = "healthy" if healthy else "unhealthy"
            except Exception as e:
                status = f"error: {str(e)[:50]}"
        else:
            status = "no_health_check"

        registered.health_status = status
        registered.last_health_check = datetime.now(timezone.utc)
        return status

    def add_event_listener(self, listener: EventListener) -> None:
        """Add an event listener.

        Args:
            listener: Callback for plugin events.
        """
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: EventListener) -> None:
        """Remove an event listener.

        Args:
            listener: The listener to remove.
        """
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def _emit_event(self, event: PluginEvent) -> None:
        """Emit an event to all listeners.

        Args:
            event: The event to emit.
        """
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics.
        """
        return {
            "total_plugins": len(self._all_plugins),
            "skill_chips": len(self._skill_chips),
            "adapters": len(self._adapters),
            "storage": len(self._storage),
            "middleware": len(self._middleware),
            "enabled_plugins": sum(
                1 for p in self._all_plugins.values() if p.enabled
            ),
            "total_calls": sum(
                p.call_count for p in self._all_plugins.values()
            ),
            "total_errors": sum(
                p.error_count for p in self._all_plugins.values()
            ),
            "intents_mapped": len(self._intent_map),
            "platforms_mapped": len(self._platform_map),
        }

    def get_intent_map(self) -> dict[str, str]:
        """Get the intent to plugin mapping.

        Returns:
            Dictionary mapping intents to plugin names.
        """
        return dict(self._intent_map)

    def get_platform_map(self) -> dict[str, str]:
        """Get the platform to plugin mapping.

        Returns:
            Dictionary mapping platforms to plugin names.
        """
        return dict(self._platform_map)

    def __repr__(self) -> str:
        return (
            f"<PluginRegistry plugins={len(self._all_plugins)} "
            f"chips={len(self._skill_chips)} adapters={len(self._adapters)} "
            f"middleware={len(self._middleware)}>"
        )
