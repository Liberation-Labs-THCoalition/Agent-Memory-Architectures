"""
Plugin system for Kintsugi CMA.

This module provides a complete plugin architecture enabling extensibility
across multiple dimensions:
- Skill Chips: Add new domain-specific capabilities
- Adapters: Support new chat platforms
- Storage: Custom storage backends
- Middleware: Request/response processing

Plugin Architecture:
    The plugin system is built around four core interfaces (Protocols):
    - SkillChipPlugin: For adding new skill chips
    - AdapterPlugin: For supporting new chat platforms
    - StoragePlugin: For custom storage backends
    - MiddlewarePlugin: For request/response processing

    Plugins are discovered, loaded, sandboxed, and registered through
    dedicated managers:
    - PluginLoader: Discovers and loads plugin modules
    - PluginSandbox: Provides isolated execution environment
    - PluginRegistry: Central registry for active plugins

Security:
    Plugins execute in a sandboxed environment with configurable
    policies that control:
    - Network access
    - Filesystem access
    - Memory limits
    - CPU time limits
    - Allowed imports

Example:
    from kintsugi.plugins import (
        PluginLoader, PluginRegistry, PluginSandbox,
        SkillChipPlugin, PluginMetadata,
    )

    # Load plugins from directory
    loader = PluginLoader(plugin_dirs=["./plugins"])
    plugins = loader.discover()

    # Register with sandbox
    registry = PluginRegistry()
    sandbox = PluginSandbox()

    for plugin in plugins:
        loaded = loader.load(plugin.name)
        if sandbox.validate_plugin(loaded):
            registry.register(loaded)
"""

from kintsugi.plugins.sdk import (
    PluginMetadata,
    SkillChipPlugin,
    AdapterPlugin,
    StoragePlugin,
    MiddlewarePlugin,
)
from kintsugi.plugins.loader import (
    LoadedPlugin,
    PluginLoader,
    PluginState,
)
from kintsugi.plugins.sandbox import (
    PluginSandbox,
    SandboxPolicy,
    SandboxViolation,
)
from kintsugi.plugins.registry import (
    PluginRegistry,
)

__all__ = [
    # SDK interfaces
    "PluginMetadata",
    "SkillChipPlugin",
    "AdapterPlugin",
    "StoragePlugin",
    "MiddlewarePlugin",
    # Loader
    "LoadedPlugin",
    "PluginLoader",
    "PluginState",
    # Sandbox
    "PluginSandbox",
    "SandboxPolicy",
    "SandboxViolation",
    # Registry
    "PluginRegistry",
]
