"""
Plugin SDK for Kintsugi CMA.

This module defines the four core plugin interfaces that third-party
developers can implement to extend Kintsugi CMA functionality.

Plugin Interfaces:
    1. SkillChipPlugin: Add new domain-specific skill chips
    2. AdapterPlugin: Support new chat platforms
    3. StoragePlugin: Custom storage backends
    4. MiddlewarePlugin: Request/response processing pipeline

Each interface uses Python's Protocol for structural subtyping,
allowing plugins to be implemented without inheriting from base classes.

Example - Creating a Skill Chip Plugin:
    from kintsugi.plugins.sdk import SkillChipPlugin, PluginMetadata

    class MySkillChipPlugin:
        metadata = PluginMetadata(
            name="my_skill_chip",
            version="1.0.0",
            author="Developer Name",
            description="A custom skill chip for XYZ",
        )

        def get_chip(self) -> BaseSkillChip:
            return MyCustomSkillChip()

        def get_intents(self) -> list[str]:
            return ["custom_intent_1", "custom_intent_2"]
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


@dataclass
class PluginMetadata:
    """Metadata describing a plugin.

    Every plugin must provide metadata that describes its identity,
    version, dependencies, and requirements. This information is used
    for plugin discovery, compatibility checking, and display.

    Attributes:
        name: Unique plugin identifier (lowercase, underscore-separated).
        version: Semantic version string (e.g., "1.0.0").
        author: Plugin author name or organization.
        description: Human-readable description of the plugin.
        homepage: URL to plugin homepage or documentation.
        license: Software license (e.g., "MIT", "Apache-2.0").
        min_kintsugi_version: Minimum compatible Kintsugi version.
        required_capabilities: List of required system capabilities.
        tags: Categorization tags for the plugin.
        dependencies: Other plugins this plugin depends on.
        entry_point: Module entry point for the plugin.
        config_schema: JSON Schema for plugin configuration.

    Example:
        metadata = PluginMetadata(
            name="grant_database_connector",
            version="2.1.0",
            author="Nonprofit Tech Collective",
            description="Connects to external grant databases",
            homepage="https://github.com/example/grant-connector",
            license="MIT",
            min_kintsugi_version="1.0.0",
            required_capabilities=["network", "storage"],
            tags=["fundraising", "grants", "integration"],
        )
    """

    name: str
    version: str
    author: str
    description: str
    homepage: str | None = None
    license: str = "MIT"
    min_kintsugi_version: str = "1.0.0"
    required_capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    entry_point: str | None = None
    config_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
        if not self.author:
            raise ValueError("Plugin author cannot be empty")

        # Validate name format (lowercase, alphanumeric with underscores)
        import re
        if not re.match(r'^[a-z][a-z0-9_]*$', self.name):
            raise ValueError(
                f"Plugin name must be lowercase alphanumeric with underscores: {self.name}"
            )

        # Validate semver format
        if not re.match(r'^\d+\.\d+\.\d+', self.version):
            raise ValueError(
                f"Plugin version must be semver format: {self.version}"
            )

    def is_compatible(self, kintsugi_version: str) -> bool:
        """Check if plugin is compatible with a Kintsugi version.

        Performs semantic version comparison to determine if the
        plugin's minimum version requirement is satisfied.

        Args:
            kintsugi_version: The Kintsugi version to check against.

        Returns:
            True if compatible, False otherwise.
        """
        from packaging import version as pkg_version
        try:
            return pkg_version.parse(kintsugi_version) >= pkg_version.parse(
                self.min_kintsugi_version
            )
        except Exception:
            # If parsing fails, assume compatible
            return True

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "homepage": self.homepage,
            "license": self.license,
            "min_kintsugi_version": self.min_kintsugi_version,
            "required_capabilities": self.required_capabilities,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
        }

    def __repr__(self) -> str:
        return f"<PluginMetadata {self.name}@{self.version} by {self.author}>"


@dataclass
class PluginConfig:
    """Runtime configuration for a plugin.

    Holds configuration values passed to a plugin at initialization.

    Attributes:
        values: Configuration values as key-value pairs.
        secrets: Secret values (should not be logged).
        environment: Environment-specific overrides.
    """

    values: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    environment: str = "production"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key.
            default: Default value if not found.

        Returns:
            The configuration value or default.
        """
        return self.values.get(key, default)

    def get_secret(self, key: str) -> str | None:
        """Get a secret value.

        Args:
            key: The secret key.

        Returns:
            The secret value or None.
        """
        return self.secrets.get(key)


@runtime_checkable
class SkillChipPlugin(Protocol):
    """Interface 1: Skill Chip plugins.

    Skill chip plugins add new domain-specific capabilities to
    Kintsugi CMA. They provide skill chips that handle specific
    intents and execute actions within ethical guardrails.

    Required Attributes:
        metadata: Plugin metadata describing the skill chip.

    Required Methods:
        get_chip: Returns the skill chip instance.
        get_intents: Returns list of intents this chip handles.

    Optional Methods:
        initialize: Called when plugin is loaded.
        shutdown: Called when plugin is unloaded.
        configure: Apply runtime configuration.

    Example:
        class GrantWriterPlugin:
            metadata = PluginMetadata(
                name="grant_writer",
                version="1.0.0",
                author="Nonprofit Tech",
                description="AI-assisted grant writing",
            )

            def __init__(self):
                self._chip = None

            def get_chip(self) -> BaseSkillChip:
                if self._chip is None:
                    self._chip = GrantWriterChip()
                return self._chip

            def get_intents(self) -> list[str]:
                return [
                    "draft_grant_proposal",
                    "review_grant_application",
                    "generate_grant_budget",
                ]

            async def initialize(self) -> None:
                # Load any required models or data
                pass
    """

    metadata: PluginMetadata

    def get_chip(self) -> Any:
        """Get the skill chip instance.

        Returns:
            A BaseSkillChip instance or compatible object.
        """
        ...

    def get_intents(self) -> list[str]:
        """Get the intents this skill chip handles.

        Returns:
            List of intent strings this chip can process.
        """
        ...


@runtime_checkable
class AdapterPlugin(Protocol):
    """Interface 2: Chat adapter plugins.

    Adapter plugins enable Kintsugi CMA to connect with new chat
    platforms beyond the built-in Slack, Discord, and WebChat.

    Required Attributes:
        metadata: Plugin metadata describing the adapter.

    Required Methods:
        get_adapter: Returns the adapter instance.
        get_platform_name: Returns the platform identifier.

    Optional Methods:
        initialize: Called when plugin is loaded.
        shutdown: Called when plugin is unloaded.
        configure: Apply runtime configuration.
        health_check: Verify adapter connectivity.

    Example:
        class TeamsAdapterPlugin:
            metadata = PluginMetadata(
                name="teams_adapter",
                version="1.0.0",
                author="Enterprise Solutions",
                description="Microsoft Teams adapter",
                required_capabilities=["network"],
            )

            def get_adapter(self) -> BaseAdapter:
                return TeamsAdapter()

            def get_platform_name(self) -> str:
                return "teams"

            async def health_check(self) -> bool:
                return await self._adapter.test_connection()
    """

    metadata: PluginMetadata

    def get_adapter(self) -> Any:
        """Get the adapter instance.

        Returns:
            A BaseAdapter instance or compatible object.
        """
        ...

    def get_platform_name(self) -> str:
        """Get the platform identifier.

        Returns:
            String identifying the platform (e.g., "teams", "telegram").
        """
        ...


@runtime_checkable
class StoragePlugin(Protocol):
    """Interface 3: Storage backend plugins.

    Storage plugins provide custom storage backends for Kintsugi
    CMA data. They can implement alternative persistence mechanisms
    for memories, documents, and other data.

    Required Attributes:
        metadata: Plugin metadata describing the storage backend.

    Required Methods:
        store: Store a value by key.
        retrieve: Retrieve a value by key.
        delete: Delete a value by key.

    Optional Methods:
        initialize: Set up storage connection.
        shutdown: Clean up storage connection.
        exists: Check if key exists.
        list_keys: List all keys with optional prefix.
        get_stats: Get storage statistics.

    Example:
        class S3StoragePlugin:
            metadata = PluginMetadata(
                name="s3_storage",
                version="1.0.0",
                author="Cloud Solutions",
                description="Amazon S3 storage backend",
                required_capabilities=["network"],
            )

            def __init__(self):
                self._client = None
                self._bucket = None

            async def initialize(self, config: PluginConfig) -> None:
                import boto3
                self._client = boto3.client('s3')
                self._bucket = config.get("bucket_name")

            async def store(self, key: str, value: bytes) -> None:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=value,
                )

            async def retrieve(self, key: str) -> bytes | None:
                try:
                    response = self._client.get_object(
                        Bucket=self._bucket,
                        Key=key,
                    )
                    return response['Body'].read()
                except self._client.exceptions.NoSuchKey:
                    return None

            async def delete(self, key: str) -> bool:
                try:
                    self._client.delete_object(
                        Bucket=self._bucket,
                        Key=key,
                    )
                    return True
                except Exception:
                    return False
    """

    metadata: PluginMetadata

    async def store(self, key: str, value: bytes) -> None:
        """Store a value by key.

        Args:
            key: The storage key.
            value: The value to store as bytes.
        """
        ...

    async def retrieve(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Args:
            key: The storage key.

        Returns:
            The stored value as bytes, or None if not found.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The storage key.

        Returns:
            True if deleted, False if not found.
        """
        ...


@runtime_checkable
class MiddlewarePlugin(Protocol):
    """Interface 4: Request/response middleware.

    Middleware plugins process requests before they reach handlers
    and responses before they're returned. They enable cross-cutting
    concerns like logging, transformation, validation, etc.

    Required Attributes:
        metadata: Plugin metadata describing the middleware.

    Required Methods:
        process_request: Process incoming request.
        process_response: Process outgoing response.

    Optional Methods:
        initialize: Called when plugin is loaded.
        shutdown: Called when plugin is unloaded.
        get_priority: Return execution priority (lower = earlier).

    Middleware Chain:
        Middleware is executed in priority order:
        1. Request flows through middleware (low priority first)
        2. Handler processes request
        3. Response flows through middleware (high priority first)

    Example:
        class LoggingMiddlewarePlugin:
            metadata = PluginMetadata(
                name="logging_middleware",
                version="1.0.0",
                author="Observability Team",
                description="Request/response logging",
            )

            async def process_request(self, request: dict) -> dict:
                logger.info(f"Request: {request.get('intent')}")
                request["_logged_at"] = datetime.now()
                return request

            async def process_response(self, response: dict) -> dict:
                logger.info(f"Response: {response.get('success')}")
                return response

            def get_priority(self) -> int:
                return 100  # Early in chain
    """

    metadata: PluginMetadata

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming request.

        Receives the request dictionary, can modify or validate it,
        and returns the (possibly modified) request. Can raise
        exceptions to abort request processing.

        Args:
            request: The incoming request dictionary.

        Returns:
            The processed request dictionary.

        Raises:
            Exception: To abort request processing.
        """
        ...

    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process an outgoing response.

        Receives the response dictionary, can modify or augment it,
        and returns the (possibly modified) response.

        Args:
            response: The outgoing response dictionary.

        Returns:
            The processed response dictionary.
        """
        ...


@dataclass
class PluginHook:
    """Represents a hook point in the plugin system.

    Hooks allow plugins to be notified of system events and
    participate in processing at specific points.

    Attributes:
        name: Hook identifier.
        description: What this hook is for.
        parameters: Expected parameters for the hook.
    """

    name: str
    description: str
    parameters: list[str] = field(default_factory=list)


# Built-in hooks that plugins can subscribe to
PLUGIN_HOOKS = [
    PluginHook(
        name="on_message_received",
        description="Called when a message is received from any adapter",
        parameters=["message", "adapter", "context"],
    ),
    PluginHook(
        name="on_intent_classified",
        description="Called after intent classification",
        parameters=["intent", "confidence", "context"],
    ),
    PluginHook(
        name="on_response_generated",
        description="Called before response is sent",
        parameters=["response", "context"],
    ),
    PluginHook(
        name="on_memory_stored",
        description="Called when a memory is stored",
        parameters=["memory", "stage", "tenant_id"],
    ),
    PluginHook(
        name="on_consensus_requested",
        description="Called when consensus approval is needed",
        parameters=["action", "context", "threshold"],
    ),
    PluginHook(
        name="on_plugin_loaded",
        description="Called when another plugin is loaded",
        parameters=["plugin_name", "plugin_type"],
    ),
]


class PluginBase:
    """Optional base class for plugins.

    Plugins don't need to inherit from this class (thanks to Protocols),
    but it provides convenient default implementations of optional
    methods and utility functions.

    Attributes:
        _config: Runtime configuration.
        _initialized: Whether initialize() has been called.
        _hooks: Registered hook handlers.

    Example:
        class MyPlugin(PluginBase):
            metadata = PluginMetadata(...)

            async def initialize(self) -> None:
                await super().initialize()
                # Custom initialization

            def get_chip(self) -> BaseSkillChip:
                return MyChip()
    """

    metadata: PluginMetadata
    _config: PluginConfig | None = None
    _initialized: bool = False
    _hooks: dict[str, list[Any]]

    def __init__(self) -> None:
        """Initialize the plugin base."""
        self._hooks = {}
        self._config = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the plugin.

        Called after the plugin is loaded. Override to perform
        custom initialization.
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """Shut down the plugin.

        Called before the plugin is unloaded. Override to perform
        cleanup.
        """
        self._initialized = False

    def configure(self, config: PluginConfig) -> None:
        """Apply runtime configuration.

        Args:
            config: The configuration to apply.
        """
        self._config = config

    def register_hook(self, hook_name: str, handler: Any) -> None:
        """Register a hook handler.

        Args:
            hook_name: The hook to register for.
            handler: The handler function.
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(handler)

    def get_hook_handlers(self, hook_name: str) -> list[Any]:
        """Get handlers for a hook.

        Args:
            hook_name: The hook name.

        Returns:
            List of registered handlers.
        """
        return self._hooks.get(hook_name, [])

    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    @property
    def config(self) -> PluginConfig | None:
        """Get the current configuration."""
        return self._config

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__} {self.metadata.name}@{self.metadata.version} {status}>"
