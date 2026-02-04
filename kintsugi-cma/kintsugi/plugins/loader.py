"""
Plugin discovery and loading for Kintsugi CMA.

This module handles finding plugins in configured directories,
loading them into memory, and managing their lifecycle.

Plugin Discovery:
    Plugins are discovered by scanning configured directories for:
    - Python packages with a `plugin.py` entry point
    - Python modules with a PLUGIN_CLASS attribute
    - Package directories with `__init__.py` containing plugin classes

Plugin States:
    - DISCOVERED: Plugin found but not yet loaded
    - LOADED: Plugin module loaded into memory
    - ACTIVE: Plugin initialized and ready for use
    - DISABLED: Plugin explicitly disabled
    - ERROR: Plugin failed to load or initialize

Example:
    from kintsugi.plugins.loader import PluginLoader, PluginState

    loader = PluginLoader(plugin_dirs=["./plugins", "/etc/kintsugi/plugins"])

    # Discover available plugins
    available = loader.discover()
    for meta in available:
        print(f"Found: {meta.name} v{meta.version}")

    # Load a specific plugin
    loaded = loader.load("my_plugin")
    if loaded.state == PluginState.LOADED:
        print("Plugin loaded successfully")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import importlib
import importlib.util
import logging
import sys
import traceback

from kintsugi.plugins.sdk import (
    PluginMetadata,
    SkillChipPlugin,
    AdapterPlugin,
    StoragePlugin,
    MiddlewarePlugin,
)

logger = logging.getLogger(__name__)


class PluginState(str, Enum):
    """States a plugin can be in.

    Plugins transition through these states during their lifecycle.

    Attributes:
        DISCOVERED: Plugin found but not yet loaded.
        LOADED: Plugin module loaded into Python.
        ACTIVE: Plugin initialized and ready for use.
        DISABLED: Plugin explicitly disabled by admin.
        ERROR: Plugin encountered an error.
    """

    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginDependency:
    """Represents a plugin dependency.

    Tracks dependencies between plugins for load ordering.

    Attributes:
        name: Name of the required plugin.
        version_spec: Version specification (e.g., ">=1.0.0").
        optional: Whether this dependency is optional.
    """

    name: str
    version_spec: str = "*"
    optional: bool = False

    def is_satisfied_by(self, version: str) -> bool:
        """Check if a version satisfies this dependency.

        Args:
            version: Version string to check.

        Returns:
            True if the version satisfies the requirement.
        """
        if self.version_spec == "*":
            return True

        try:
            from packaging import version as pkg_version
            from packaging.specifiers import SpecifierSet

            spec = SpecifierSet(self.version_spec)
            return pkg_version.parse(version) in spec
        except Exception:
            return True


@dataclass
class LoadedPlugin:
    """A loaded plugin with its metadata and module.

    Represents a plugin that has been loaded into memory. Contains
    all information needed to use the plugin.

    Attributes:
        metadata: The plugin's metadata.
        module: The loaded Python module.
        state: Current plugin state.
        error: Error message if state is ERROR.
        loaded_at: When the plugin was loaded.
        plugin_class: The main plugin class from the module.
        instance: Instantiated plugin object.
        plugin_type: Which interface the plugin implements.
        config_path: Path to plugin configuration file.

    Example:
        loaded = loader.load("my_plugin")
        if loaded.state == PluginState.LOADED:
            chip = loaded.instance.get_chip()
    """

    metadata: PluginMetadata
    module: Any
    state: PluginState
    error: str | None = None
    loaded_at: datetime | None = None
    plugin_class: type | None = None
    instance: Any = None
    plugin_type: str | None = None
    config_path: Path | None = None

    def is_skill_chip(self) -> bool:
        """Check if this is a skill chip plugin."""
        return self.plugin_type == "skill_chip"

    def is_adapter(self) -> bool:
        """Check if this is an adapter plugin."""
        return self.plugin_type == "adapter"

    def is_storage(self) -> bool:
        """Check if this is a storage plugin."""
        return self.plugin_type == "storage"

    def is_middleware(self) -> bool:
        """Check if this is a middleware plugin."""
        return self.plugin_type == "middleware"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.state.value,
            "error": self.error,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "plugin_type": self.plugin_type,
        }


@dataclass
class PluginManifest:
    """Plugin manifest from plugin.json or plugin.yaml.

    Optional manifest file that provides additional plugin configuration.

    Attributes:
        name: Plugin name (overrides detected name).
        version: Plugin version.
        entry_point: Entry point module/class.
        dependencies: Plugin dependencies.
        capabilities: Required capabilities.
        config_schema: Configuration JSON schema.
    """

    name: str
    version: str
    entry_point: str
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] | None = None


class PluginLoadError(Exception):
    """Raised when a plugin fails to load.

    Attributes:
        plugin_name: Name of the plugin that failed.
        reason: Reason for the failure.
        original: Original exception if any.
    """

    def __init__(
        self,
        plugin_name: str,
        reason: str,
        original: Exception | None = None,
    ):
        self.plugin_name = plugin_name
        self.reason = reason
        self.original = original
        super().__init__(f"Failed to load plugin '{plugin_name}': {reason}")


class PluginLoader:
    """Discovers and loads plugins.

    The PluginLoader is responsible for:
    - Scanning directories for plugins
    - Loading plugin modules into Python
    - Instantiating plugin classes
    - Managing plugin lifecycle

    Attributes:
        _plugin_dirs: Directories to scan for plugins.
        _plugins: Dictionary of loaded plugins by name.
        _discovered: Set of discovered plugin names.

    Example:
        loader = PluginLoader(plugin_dirs=["./plugins"])

        # Discover all plugins
        available = loader.discover()

        # Load a specific plugin
        loaded = loader.load("my_plugin")

        # Get all loaded plugins
        all_loaded = loader.get_loaded()

        # Get plugins by interface
        skill_chips = loader.get_by_interface(SkillChipPlugin)
    """

    # Expected attributes/methods for plugin detection
    PLUGIN_MARKERS = ["metadata", "PLUGIN_METADATA", "PLUGIN_CLASS"]

    def __init__(self, plugin_dirs: list[str] | None = None):
        """Initialize the plugin loader.

        Args:
            plugin_dirs: List of directories to scan for plugins.
                        Defaults to ["./plugins"].
        """
        self._plugin_dirs = [Path(d) for d in (plugin_dirs or ["./plugins"])]
        self._plugins: dict[str, LoadedPlugin] = {}
        self._discovered: dict[str, PluginMetadata] = {}
        self._load_order: list[str] = []

    @property
    def plugin_dirs(self) -> list[Path]:
        """Get the configured plugin directories."""
        return list(self._plugin_dirs)

    def add_plugin_dir(self, directory: str) -> None:
        """Add a plugin directory.

        Args:
            directory: Directory path to add.
        """
        path = Path(directory)
        if path not in self._plugin_dirs:
            self._plugin_dirs.append(path)
            logger.info(f"Added plugin directory: {directory}")

    def discover(self) -> list[PluginMetadata]:
        """Discover available plugins.

        Scans all configured directories for plugins and returns
        their metadata. Does not load the plugins.

        Returns:
            List of PluginMetadata for discovered plugins.
        """
        self._discovered.clear()

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                continue

            logger.info(f"Scanning for plugins in: {plugin_dir}")

            # Scan for plugin directories (packages)
            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    self._discover_package(item)
                elif item.is_file() and item.suffix == ".py":
                    self._discover_module(item)

        logger.info(f"Discovered {len(self._discovered)} plugins")
        return list(self._discovered.values())

    def _discover_package(self, package_path: Path) -> None:
        """Discover a plugin from a package directory.

        Args:
            package_path: Path to the package directory.
        """
        try:
            # Check for manifest file
            manifest_path = package_path / "plugin.json"
            if manifest_path.exists():
                metadata = self._parse_manifest(manifest_path)
                if metadata:
                    self._discovered[metadata.name] = metadata
                    return

            # Try to extract metadata from __init__.py
            init_path = package_path / "__init__.py"
            metadata = self._extract_metadata_from_file(init_path)

            if metadata:
                self._discovered[metadata.name] = metadata
                logger.debug(f"Discovered package plugin: {metadata.name}")
            else:
                # Use directory name as plugin name
                name = package_path.name
                metadata = PluginMetadata(
                    name=name,
                    version="0.0.0",
                    author="Unknown",
                    description=f"Plugin from {package_path}",
                    entry_point=str(package_path),
                )
                self._discovered[name] = metadata

        except Exception as e:
            logger.warning(f"Error discovering package {package_path}: {e}")

    def _discover_module(self, module_path: Path) -> None:
        """Discover a plugin from a single module file.

        Args:
            module_path: Path to the module file.
        """
        try:
            metadata = self._extract_metadata_from_file(module_path)

            if metadata:
                self._discovered[metadata.name] = metadata
                logger.debug(f"Discovered module plugin: {metadata.name}")

        except Exception as e:
            logger.warning(f"Error discovering module {module_path}: {e}")

    def _extract_metadata_from_file(self, file_path: Path) -> PluginMetadata | None:
        """Extract plugin metadata from a file.

        Args:
            file_path: Path to the Python file.

        Returns:
            PluginMetadata if found, None otherwise.
        """
        try:
            # Load the module to inspect it
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)

            # Look for metadata without executing
            with open(file_path) as f:
                content = f.read()

            # Quick check for plugin markers
            has_marker = any(marker in content for marker in self.PLUGIN_MARKERS)
            if not has_marker:
                return None

            # Execute to get actual metadata
            spec.loader.exec_module(module)

            # Look for metadata in various forms
            for cls_name in dir(module):
                cls = getattr(module, cls_name)
                if isinstance(cls, type) and hasattr(cls, 'metadata'):
                    meta = getattr(cls, 'metadata')
                    if isinstance(meta, PluginMetadata):
                        return meta

            # Check module-level PLUGIN_METADATA
            if hasattr(module, 'PLUGIN_METADATA'):
                meta = module.PLUGIN_METADATA
                if isinstance(meta, PluginMetadata):
                    return meta

            return None

        except Exception as e:
            logger.debug(f"Could not extract metadata from {file_path}: {e}")
            return None

    def _parse_manifest(self, manifest_path: Path) -> PluginMetadata | None:
        """Parse a plugin manifest file.

        Args:
            manifest_path: Path to the manifest file.

        Returns:
            PluginMetadata if valid, None otherwise.
        """
        import json

        try:
            with open(manifest_path) as f:
                data = json.load(f)

            return PluginMetadata(
                name=data["name"],
                version=data["version"],
                author=data.get("author", "Unknown"),
                description=data.get("description", ""),
                homepage=data.get("homepage"),
                license=data.get("license", "MIT"),
                min_kintsugi_version=data.get("min_kintsugi_version", "1.0.0"),
                required_capabilities=data.get("required_capabilities", []),
                tags=data.get("tags", []),
                dependencies=data.get("dependencies", []),
                entry_point=data.get("entry_point"),
                config_schema=data.get("config_schema"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse manifest {manifest_path}: {e}")
            return None

    def load(self, plugin_name: str) -> LoadedPlugin:
        """Load a plugin by name.

        Loads the plugin module, instantiates the plugin class,
        and returns a LoadedPlugin object.

        Args:
            plugin_name: Name of the plugin to load.

        Returns:
            LoadedPlugin with the loaded plugin.

        Raises:
            PluginLoadError: If the plugin cannot be loaded.
        """
        # Check if already loaded
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]

        # Check if discovered
        if plugin_name not in self._discovered:
            # Try to discover first
            self.discover()
            if plugin_name not in self._discovered:
                raise PluginLoadError(plugin_name, "Plugin not found")

        metadata = self._discovered[plugin_name]
        logger.info(f"Loading plugin: {plugin_name} v{metadata.version}")

        try:
            # Load dependencies first
            self._load_dependencies(metadata)

            # Load the plugin module
            module = self._load_module(plugin_name, metadata)

            # Find the plugin class
            plugin_class, plugin_type = self._find_plugin_class(module)

            if plugin_class is None:
                raise PluginLoadError(
                    plugin_name,
                    "No valid plugin class found in module"
                )

            # Instantiate the plugin
            instance = plugin_class()

            # Update metadata from instance if available
            if hasattr(instance, 'metadata'):
                metadata = instance.metadata

            loaded = LoadedPlugin(
                metadata=metadata,
                module=module,
                state=PluginState.LOADED,
                loaded_at=datetime.now(timezone.utc),
                plugin_class=plugin_class,
                instance=instance,
                plugin_type=plugin_type,
            )

            self._plugins[plugin_name] = loaded
            self._load_order.append(plugin_name)

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return loaded

        except PluginLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            logger.debug(traceback.format_exc())

            loaded = LoadedPlugin(
                metadata=metadata,
                module=None,
                state=PluginState.ERROR,
                error=str(e),
            )
            self._plugins[plugin_name] = loaded
            raise PluginLoadError(plugin_name, str(e), e)

    def _load_dependencies(self, metadata: PluginMetadata) -> None:
        """Load plugin dependencies.

        Args:
            metadata: Plugin metadata with dependencies.
        """
        for dep_name in metadata.dependencies:
            if dep_name not in self._plugins:
                logger.info(f"Loading dependency: {dep_name}")
                try:
                    self.load(dep_name)
                except PluginLoadError as e:
                    logger.warning(f"Failed to load dependency {dep_name}: {e}")

    def _load_module(self, plugin_name: str, metadata: PluginMetadata) -> Any:
        """Load the plugin module.

        Args:
            plugin_name: Plugin name.
            metadata: Plugin metadata.

        Returns:
            Loaded module.
        """
        # Determine module path
        for plugin_dir in self._plugin_dirs:
            # Try package
            package_path = plugin_dir / plugin_name
            if package_path.is_dir() and (package_path / "__init__.py").exists():
                if str(plugin_dir) not in sys.path:
                    sys.path.insert(0, str(plugin_dir))
                return importlib.import_module(plugin_name)

            # Try module file
            module_path = plugin_dir / f"{plugin_name}.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    plugin_name, module_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[plugin_name] = module
                    spec.loader.exec_module(module)
                    return module

        # Try entry point from metadata
        if metadata.entry_point:
            return importlib.import_module(metadata.entry_point)

        raise PluginLoadError(plugin_name, "Could not find plugin module")

    def _find_plugin_class(
        self, module: Any
    ) -> tuple[type | None, str | None]:
        """Find the plugin class in a module.

        Args:
            module: The loaded module.

        Returns:
            Tuple of (plugin class, plugin type) or (None, None).
        """
        # Look for classes implementing plugin interfaces
        for name in dir(module):
            obj = getattr(module, name)
            if not isinstance(obj, type):
                continue

            # Check for each interface
            if isinstance(obj, type):
                instance = None
                try:
                    # Check if it has metadata and is a plugin type
                    if hasattr(obj, 'metadata') and isinstance(
                        getattr(obj, 'metadata', None), PluginMetadata
                    ):
                        # Create temporary instance to check interface
                        if hasattr(obj, 'get_chip') and hasattr(obj, 'get_intents'):
                            return obj, "skill_chip"
                        elif hasattr(obj, 'get_adapter') and hasattr(obj, 'get_platform_name'):
                            return obj, "adapter"
                        elif hasattr(obj, 'store') and hasattr(obj, 'retrieve'):
                            return obj, "storage"
                        elif hasattr(obj, 'process_request') and hasattr(obj, 'process_response'):
                            return obj, "middleware"
                except Exception:
                    pass

        # Check for PLUGIN_CLASS marker
        if hasattr(module, 'PLUGIN_CLASS'):
            plugin_class = module.PLUGIN_CLASS
            # Determine type
            if hasattr(plugin_class, 'get_chip'):
                return plugin_class, "skill_chip"
            elif hasattr(plugin_class, 'get_adapter'):
                return plugin_class, "adapter"
            elif hasattr(plugin_class, 'store'):
                return plugin_class, "storage"
            elif hasattr(plugin_class, 'process_request'):
                return plugin_class, "middleware"
            return plugin_class, None

        return None, None

    def unload(self, plugin_name: str) -> bool:
        """Unload a plugin.

        Removes the plugin from memory and cleans up resources.

        Args:
            plugin_name: Name of the plugin to unload.

        Returns:
            True if unloaded, False if not found.
        """
        if plugin_name not in self._plugins:
            return False

        loaded = self._plugins[plugin_name]

        # Call shutdown if available
        if loaded.instance and hasattr(loaded.instance, 'shutdown'):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(loaded.instance.shutdown):
                    asyncio.get_event_loop().run_until_complete(
                        loaded.instance.shutdown()
                    )
                else:
                    loaded.instance.shutdown()
            except Exception as e:
                logger.warning(f"Error during plugin shutdown: {e}")

        # Remove from tracking
        del self._plugins[plugin_name]
        if plugin_name in self._load_order:
            self._load_order.remove(plugin_name)

        # Try to remove from sys.modules
        if plugin_name in sys.modules:
            del sys.modules[plugin_name]

        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    def get_loaded(self) -> list[LoadedPlugin]:
        """Get all loaded plugins.

        Returns:
            List of all loaded plugins.
        """
        return list(self._plugins.values())

    def get_plugin(self, plugin_name: str) -> LoadedPlugin | None:
        """Get a loaded plugin by name.

        Args:
            plugin_name: Name of the plugin.

        Returns:
            LoadedPlugin if found, None otherwise.
        """
        return self._plugins.get(plugin_name)

    def get_by_interface(self, interface: type) -> list[LoadedPlugin]:
        """Get plugins implementing an interface.

        Args:
            interface: The interface type to filter by.

        Returns:
            List of plugins implementing the interface.
        """
        results = []

        interface_map = {
            SkillChipPlugin: "skill_chip",
            AdapterPlugin: "adapter",
            StoragePlugin: "storage",
            MiddlewarePlugin: "middleware",
        }

        target_type = interface_map.get(interface)

        for plugin in self._plugins.values():
            if plugin.state == PluginState.LOADED and plugin.plugin_type == target_type:
                results.append(plugin)

        return results

    def reload(self, plugin_name: str) -> LoadedPlugin:
        """Reload a plugin.

        Unloads and reloads the plugin from disk.

        Args:
            plugin_name: Name of the plugin to reload.

        Returns:
            The reloaded plugin.
        """
        self.unload(plugin_name)

        # Clear from discovered to force re-discovery
        if plugin_name in self._discovered:
            del self._discovered[plugin_name]

        # Re-discover to pick up any changes
        self.discover()

        return self.load(plugin_name)

    def get_load_order(self) -> list[str]:
        """Get the order plugins were loaded.

        Returns:
            List of plugin names in load order.
        """
        return list(self._load_order)

    def __repr__(self) -> str:
        return (
            f"<PluginLoader dirs={len(self._plugin_dirs)} "
            f"discovered={len(self._discovered)} "
            f"loaded={len(self._plugins)}>"
        )
