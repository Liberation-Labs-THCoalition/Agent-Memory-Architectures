"""Tests for kintsugi.plugins module - Phase 5A Plugin System.

This module provides comprehensive tests for the plugin system including:
- PluginMetadata tests
- Plugin interface (Protocol) tests
- PluginState enum tests
- PluginLoader tests
- SandboxPolicy and PluginSandbox tests
- PluginRegistry tests
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass, field
from typing import Any

from kintsugi.plugins import (
    PluginMetadata,
    SkillChipPlugin,
    AdapterPlugin,
    StoragePlugin,
    MiddlewarePlugin,
    LoadedPlugin,
    PluginLoader,
    PluginState,
    PluginSandbox,
    SandboxPolicy,
    SandboxViolation,
    PluginRegistry,
)
from kintsugi.plugins.loader import PluginLoadError, PluginDependency
from kintsugi.plugins.sandbox import SandboxExecutionResult, RestrictedImporter
from kintsugi.plugins.registry import RegisteredPlugin, PluginEvent
from kintsugi.plugins.sdk import PluginConfig, PluginBase, PluginHook, PLUGIN_HOOKS


# ===========================================================================
# PluginMetadata Tests (8 tests)
# ===========================================================================

class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating metadata with required fields only."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="A test plugin",
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "A test plugin"

    def test_validation_empty_name_rejected(self):
        """Test that empty plugin name is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PluginMetadata(
                name="",
                version="1.0.0",
                author="Author",
                description="Test",
            )

    def test_validation_empty_version_rejected(self):
        """Test that empty version is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PluginMetadata(
                name="test_plugin",
                version="",
                author="Author",
                description="Test",
            )

    def test_validation_invalid_name_format(self):
        """Test that invalid name format is rejected."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            PluginMetadata(
                name="InvalidName",  # Uppercase not allowed
                version="1.0.0",
                author="Author",
                description="Test",
            )

    def test_version_parsing_semver(self):
        """Test that version must be semver format."""
        # Valid semver
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.2.3",
            author="Author",
            description="Test",
        )
        assert metadata.version == "1.2.3"

        # Invalid semver
        with pytest.raises(ValueError, match="semver"):
            PluginMetadata(
                name="test_plugin",
                version="invalid",
                author="Author",
                description="Test",
            )

    def test_compatibility_checking(self):
        """Test is_compatible() version checking."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Author",
            description="Test",
            min_kintsugi_version="1.0.0",
        )

        # Compatible versions
        assert metadata.is_compatible("1.0.0") is True
        assert metadata.is_compatible("1.5.0") is True
        assert metadata.is_compatible("2.0.0") is True

    def test_required_capabilities_list(self):
        """Test required_capabilities list."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Author",
            description="Test",
            required_capabilities=["network", "storage", "filesystem"],
        )
        assert "network" in metadata.required_capabilities
        assert "storage" in metadata.required_capabilities
        assert len(metadata.required_capabilities) == 3

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Author",
            description="Test",
            tags=["test", "example"],
        )
        d = metadata.to_dict()

        assert d["name"] == "test_plugin"
        assert d["version"] == "1.0.0"
        assert d["tags"] == ["test", "example"]


# ===========================================================================
# Plugin Interfaces Tests (10 tests)
# ===========================================================================

class TestPluginInterfaces:
    """Tests for plugin interface protocols."""

    def test_skill_chip_plugin_protocol_exists(self):
        """Verify SkillChipPlugin protocol is defined."""
        assert SkillChipPlugin is not None
        # Check it's a Protocol (runtime checkable)
        assert hasattr(SkillChipPlugin, '__protocol_attrs__') or hasattr(SkillChipPlugin, '_is_protocol')

    def test_adapter_plugin_protocol_exists(self):
        """Verify AdapterPlugin protocol is defined."""
        assert AdapterPlugin is not None

    def test_storage_plugin_protocol_exists(self):
        """Verify StoragePlugin protocol is defined."""
        assert StoragePlugin is not None

    def test_middleware_plugin_protocol_exists(self):
        """Verify MiddlewarePlugin protocol is defined."""
        assert MiddlewarePlugin is not None

    def test_skill_chip_plugin_has_required_methods(self):
        """Verify SkillChipPlugin has required methods."""
        # Create a mock implementation
        class MockSkillChip:
            metadata = PluginMetadata(
                name="mock_skill",
                version="1.0.0",
                author="Test",
                description="Mock",
            )

            def get_chip(self):
                return MagicMock()

            def get_intents(self):
                return ["test_intent"]

        mock = MockSkillChip()
        assert hasattr(mock, "metadata")
        assert hasattr(mock, "get_chip")
        assert hasattr(mock, "get_intents")
        assert isinstance(mock, SkillChipPlugin)

    def test_adapter_plugin_has_required_methods(self):
        """Verify AdapterPlugin has required methods."""
        class MockAdapter:
            metadata = PluginMetadata(
                name="mock_adapter",
                version="1.0.0",
                author="Test",
                description="Mock",
            )

            def get_adapter(self):
                return MagicMock()

            def get_platform_name(self):
                return "mock_platform"

        mock = MockAdapter()
        assert hasattr(mock, "get_adapter")
        assert hasattr(mock, "get_platform_name")
        assert isinstance(mock, AdapterPlugin)

    def test_storage_plugin_has_required_methods(self):
        """Verify StoragePlugin has required methods."""
        class MockStorage:
            metadata = PluginMetadata(
                name="mock_storage",
                version="1.0.0",
                author="Test",
                description="Mock",
            )

            async def store(self, key: str, value: bytes) -> None:
                pass

            async def retrieve(self, key: str) -> bytes | None:
                return None

            async def delete(self, key: str) -> bool:
                return True

        mock = MockStorage()
        assert hasattr(mock, "store")
        assert hasattr(mock, "retrieve")
        assert hasattr(mock, "delete")
        assert isinstance(mock, StoragePlugin)

    def test_middleware_plugin_has_required_methods(self):
        """Verify MiddlewarePlugin has required methods."""
        class MockMiddleware:
            metadata = PluginMetadata(
                name="mock_middleware",
                version="1.0.0",
                author="Test",
                description="Mock",
            )

            async def process_request(self, request: dict) -> dict:
                return request

            async def process_response(self, response: dict) -> dict:
                return response

        mock = MockMiddleware()
        assert hasattr(mock, "process_request")
        assert hasattr(mock, "process_response")
        assert isinstance(mock, MiddlewarePlugin)

    def test_plugin_config_get_value(self):
        """Test PluginConfig.get() method."""
        config = PluginConfig(
            values={"key1": "value1", "key2": 42},
            secrets={"api_key": "secret123"},
        )

        assert config.get("key1") == "value1"
        assert config.get("key2") == 42
        assert config.get("missing", "default") == "default"

    def test_plugin_config_get_secret(self):
        """Test PluginConfig.get_secret() method."""
        config = PluginConfig(
            values={},
            secrets={"api_key": "secret123"},
        )

        assert config.get_secret("api_key") == "secret123"
        assert config.get_secret("missing") is None


# ===========================================================================
# PluginState Tests (3 tests)
# ===========================================================================

class TestPluginState:
    """Tests for PluginState enum."""

    def test_discovered_state_exists(self):
        """Verify DISCOVERED state is defined."""
        assert PluginState.DISCOVERED is not None
        assert PluginState.DISCOVERED.value == "discovered"

    def test_loaded_state_exists(self):
        """Verify LOADED state is defined."""
        assert PluginState.LOADED is not None
        assert PluginState.LOADED.value == "loaded"

    def test_active_state_exists(self):
        """Verify ACTIVE state is defined."""
        assert PluginState.ACTIVE is not None
        assert PluginState.ACTIVE.value == "active"

    def test_disabled_state_exists(self):
        """Verify DISABLED state is defined."""
        assert PluginState.DISABLED is not None
        assert PluginState.DISABLED.value == "disabled"

    def test_error_state_exists(self):
        """Verify ERROR state is defined."""
        assert PluginState.ERROR is not None
        assert PluginState.ERROR.value == "error"


# ===========================================================================
# PluginLoader Tests (12 tests)
# ===========================================================================

class TestPluginLoader:
    """Tests for PluginLoader class."""

    def test_creation_with_default_dirs(self):
        """Test creating loader with default plugin directories."""
        loader = PluginLoader()
        assert len(loader.plugin_dirs) > 0
        assert Path("./plugins") in loader.plugin_dirs

    def test_creation_with_custom_dirs(self):
        """Test creating loader with custom plugin directories."""
        loader = PluginLoader(plugin_dirs=["/custom/plugins", "/another/dir"])
        assert Path("/custom/plugins") in loader.plugin_dirs
        assert Path("/another/dir") in loader.plugin_dirs

    def test_discover_returns_list(self):
        """Test discover() returns a list of metadata."""
        loader = PluginLoader(plugin_dirs=["/nonexistent"])
        # With non-existent directory, should return empty list
        result = loader.discover()
        assert isinstance(result, list)

    def test_load_returns_loaded_plugin(self):
        """Test load() returns LoadedPlugin (with mock)."""
        loader = PluginLoader()

        # Mock a discovered plugin
        loader._discovered["test_plugin"] = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
            entry_point="test_plugin",
        )

        # Mock the module loading
        with patch.object(loader, "_load_module") as mock_load:
            mock_module = MagicMock()
            mock_load.return_value = mock_module

            # Mock finding plugin class
            with patch.object(loader, "_find_plugin_class") as mock_find:
                mock_class = MagicMock()
                mock_class.metadata = loader._discovered["test_plugin"]
                mock_find.return_value = (mock_class, "skill_chip")

                loaded = loader.load("test_plugin")
                assert isinstance(loaded, LoadedPlugin)
                assert loaded.state == PluginState.LOADED

    def test_load_unknown_plugin_raises(self):
        """Test load() raises for unknown plugin."""
        loader = PluginLoader(plugin_dirs=["/nonexistent"])

        with pytest.raises(PluginLoadError, match="not found"):
            loader.load("nonexistent_plugin")

    def test_unload_removes_plugin(self):
        """Test unload() removes plugin from tracking."""
        loader = PluginLoader()

        # Pre-populate with a loaded plugin
        metadata = PluginMetadata(
            name="unload_test",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            loaded_at=datetime.now(timezone.utc),
        )
        loader._plugins["unload_test"] = loaded
        loader._load_order.append("unload_test")

        result = loader.unload("unload_test")
        assert result is True
        assert "unload_test" not in loader._plugins

    def test_unload_unknown_returns_false(self):
        """Test unload() returns False for unknown plugin."""
        loader = PluginLoader()
        result = loader.unload("nonexistent")
        assert result is False

    def test_get_loaded_returns_list(self):
        """Test get_loaded() returns list of loaded plugins."""
        loader = PluginLoader()

        # Pre-populate
        metadata = PluginMetadata(
            name="test_loaded",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
        )
        loader._plugins["test_loaded"] = loaded

        result = loader.get_loaded()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_by_interface_filters_correctly(self):
        """Test get_by_interface() filters by plugin type."""
        loader = PluginLoader()

        # Add skill chip
        chip_meta = PluginMetadata(
            name="chip_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        chip_loaded = LoadedPlugin(
            metadata=chip_meta,
            module=MagicMock(),
            state=PluginState.LOADED,
            plugin_type="skill_chip",
        )
        loader._plugins["chip_plugin"] = chip_loaded

        # Add adapter
        adapter_meta = PluginMetadata(
            name="adapter_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        adapter_loaded = LoadedPlugin(
            metadata=adapter_meta,
            module=MagicMock(),
            state=PluginState.LOADED,
            plugin_type="adapter",
        )
        loader._plugins["adapter_plugin"] = adapter_loaded

        skill_chips = loader.get_by_interface(SkillChipPlugin)
        assert len(skill_chips) == 1
        assert skill_chips[0].metadata.name == "chip_plugin"

        adapters = loader.get_by_interface(AdapterPlugin)
        assert len(adapters) == 1
        assert adapters[0].metadata.name == "adapter_plugin"

    def test_add_plugin_dir(self):
        """Test add_plugin_dir() adds new directory."""
        loader = PluginLoader(plugin_dirs=["/initial"])
        loader.add_plugin_dir("/added")

        assert Path("/added") in loader.plugin_dirs
        assert len(loader.plugin_dirs) == 2

    def test_get_load_order(self):
        """Test get_load_order() returns load sequence."""
        loader = PluginLoader()
        loader._load_order = ["first", "second", "third"]

        order = loader.get_load_order()
        assert order == ["first", "second", "third"]

    def test_plugin_dependency_satisfied(self):
        """Test PluginDependency version checking."""
        dep = PluginDependency(name="test_dep", version_spec=">=1.0.0")

        assert dep.is_satisfied_by("1.0.0") is True
        assert dep.is_satisfied_by("2.0.0") is True

    def test_plugin_dependency_wildcard(self):
        """Test PluginDependency with wildcard version."""
        dep = PluginDependency(name="test_dep", version_spec="*")

        assert dep.is_satisfied_by("0.1.0") is True
        assert dep.is_satisfied_by("999.0.0") is True


# ===========================================================================
# SandboxPolicy Tests (6 tests)
# ===========================================================================

class TestSandboxPolicy:
    """Tests for SandboxPolicy dataclass."""

    def test_default_values(self):
        """Test SandboxPolicy default values."""
        policy = SandboxPolicy()

        assert policy.allow_network is False
        assert policy.allow_filesystem is False
        assert policy.max_memory_mb == 256
        assert policy.max_cpu_seconds == 10
        assert policy.max_execution_time == 30.0

    def test_custom_limits(self):
        """Test SandboxPolicy with custom limits."""
        policy = SandboxPolicy(
            allow_network=True,
            allow_filesystem=True,
            max_memory_mb=512,
            max_cpu_seconds=30,
        )

        assert policy.allow_network is True
        assert policy.allow_filesystem is True
        assert policy.max_memory_mb == 512
        assert policy.max_cpu_seconds == 30

    def test_allowed_imports_list(self):
        """Test allowed imports whitelist."""
        policy = SandboxPolicy(
            allowed_imports=["json", "datetime", "typing"],
        )

        assert policy.is_import_allowed("json") is True
        assert policy.is_import_allowed("datetime") is True
        assert policy.is_import_allowed("os") is False

    def test_blocked_imports_override(self):
        """Test blocked imports override allowed."""
        policy = SandboxPolicy(
            allowed_imports=[],  # Empty = allow all
            blocked_imports=["subprocess", "os.system"],
        )

        assert policy.is_import_allowed("subprocess") is False
        assert policy.is_import_allowed("os.system") is False
        assert policy.is_import_allowed("json") is True

    def test_allowed_builtins(self):
        """Test allowed_builtins list."""
        policy = SandboxPolicy()

        # Default builtins should be allowed
        assert "len" in policy.allowed_builtins
        assert "print" in policy.allowed_builtins
        assert "list" in policy.allowed_builtins

    def test_to_dict(self):
        """Test converting policy to dictionary."""
        policy = SandboxPolicy(
            allow_network=True,
            max_memory_mb=1024,
        )
        d = policy.to_dict()

        assert d["allow_network"] is True
        assert d["max_memory_mb"] == 1024


# ===========================================================================
# PluginSandbox Tests (8 tests)
# ===========================================================================

class TestPluginSandbox:
    """Tests for PluginSandbox class."""

    def test_creation_with_default_policy(self):
        """Test creating sandbox with default policy."""
        sandbox = PluginSandbox()
        assert sandbox.policy is not None
        assert sandbox.policy.allow_network is False

    def test_creation_with_custom_policy(self):
        """Test creating sandbox with custom policy."""
        policy = SandboxPolicy(allow_network=True, max_memory_mb=512)
        sandbox = PluginSandbox(policy)

        assert sandbox.policy.allow_network is True
        assert sandbox.policy.max_memory_mb == 512

    def test_validate_plugin_returns_violations_list(self):
        """Test validate_plugin() returns list of violations."""
        sandbox = PluginSandbox()

        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
        )

        violations = sandbox.validate_plugin(loaded)
        assert isinstance(violations, list)

    def test_validate_plugin_error_state(self):
        """Test validate_plugin() detects error state."""
        sandbox = PluginSandbox()

        metadata = PluginMetadata(
            name="error_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=None,
            state=PluginState.ERROR,
            error="Load failed",
        )

        violations = sandbox.validate_plugin(loaded)
        assert len(violations) > 0
        assert violations[0].violation_type == "plugin_error"

    def test_validate_plugin_no_module(self):
        """Test validate_plugin() detects missing module."""
        sandbox = PluginSandbox()

        metadata = PluginMetadata(
            name="no_module",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=None,
            state=PluginState.LOADED,
        )

        violations = sandbox.validate_plugin(loaded)
        assert any(v.violation_type == "no_module" for v in violations)

    def test_validate_plugin_missing_capability(self):
        """Test validate_plugin() detects missing capabilities."""
        sandbox = PluginSandbox(SandboxPolicy(allow_network=False))

        metadata = PluginMetadata(
            name="needs_network",
            version="1.0.0",
            author="Test",
            description="Test",
            required_capabilities=["network"],
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
        )

        violations = sandbox.validate_plugin(loaded)
        assert any(v.violation_type == "missing_capability" for v in violations)

    @pytest.mark.asyncio
    async def test_execute_runs_method(self):
        """Test execute() runs plugin method."""
        sandbox = PluginSandbox()

        # Create mock plugin
        mock_instance = MagicMock()
        mock_instance.handle = MagicMock(return_value="result")

        metadata = PluginMetadata(
            name="exec_test",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
        )

        result = await sandbox.execute(loaded, "handle", {"key": "value"})
        assert result.success is True
        assert result.result == "result"

    @pytest.mark.asyncio
    async def test_execute_method_not_found(self):
        """Test execute() returns error for missing method."""
        sandbox = PluginSandbox()

        mock_instance = MagicMock(spec=[])  # No methods

        metadata = PluginMetadata(
            name="no_method",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        loaded = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
        )

        result = await sandbox.execute(loaded, "missing_method")
        assert result.success is False
        assert "not found" in result.error

    def test_sandbox_violation_to_dict(self):
        """Test SandboxViolation.to_dict()."""
        violation = SandboxViolation(
            violation_type="blocked_import",
            message="Import of 'os' blocked",
            severity="error",
            location="line 10",
        )

        d = violation.to_dict()
        assert d["violation_type"] == "blocked_import"
        assert d["message"] == "Import of 'os' blocked"
        assert d["severity"] == "error"

    def test_sandbox_execution_result_to_dict(self):
        """Test SandboxExecutionResult.to_dict()."""
        result = SandboxExecutionResult(
            success=True,
            result={"data": "test"},
            execution_time_ms=50.5,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["execution_time_ms"] == 50.5


# ===========================================================================
# PluginRegistry Tests (10 tests)
# ===========================================================================

class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def _create_mock_skill_chip(self, name: str) -> LoadedPlugin:
        """Helper to create a mock skill chip plugin."""
        mock_instance = MagicMock()
        mock_instance.get_chip = MagicMock(return_value=MagicMock())
        mock_instance.get_intents = MagicMock(return_value=["intent_" + name])

        metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            author="Test",
            description="Test skill chip",
        )
        return LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="skill_chip",
        )

    def _create_mock_adapter(self, name: str, platform: str) -> LoadedPlugin:
        """Helper to create a mock adapter plugin."""
        mock_instance = MagicMock()
        mock_instance.get_adapter = MagicMock(return_value=MagicMock())
        mock_instance.get_platform_name = MagicMock(return_value=platform)

        metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            author="Test",
            description="Test adapter",
        )
        return LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="adapter",
        )

    def _create_mock_storage(self, name: str) -> LoadedPlugin:
        """Helper to create a mock storage plugin."""
        mock_instance = MagicMock()

        metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            author="Test",
            description="Test storage",
        )
        return LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="storage",
        )

    def _create_mock_middleware(self, name: str, priority: int = 100) -> LoadedPlugin:
        """Helper to create a mock middleware plugin."""
        mock_instance = AsyncMock()
        mock_instance.process_request = AsyncMock(side_effect=lambda r: r)
        mock_instance.process_response = AsyncMock(side_effect=lambda r: r)
        mock_instance.get_priority = MagicMock(return_value=priority)

        metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            author="Test",
            description="Test middleware",
        )
        return LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="middleware",
        )

    def test_registration_of_skill_chip(self):
        """Test registering a skill chip plugin."""
        registry = PluginRegistry()
        plugin = self._create_mock_skill_chip("test_chip")

        result = registry.register(plugin)

        assert result is True
        assert registry.get_plugin("test_chip") is not None

    def test_registration_of_adapter(self):
        """Test registering an adapter plugin."""
        registry = PluginRegistry()
        plugin = self._create_mock_adapter("test_adapter", "test_platform")

        result = registry.register(plugin)

        assert result is True
        assert len(registry.get_all_adapters()) == 1

    def test_registration_of_storage(self):
        """Test registering a storage plugin."""
        registry = PluginRegistry()
        plugin = self._create_mock_storage("test_storage")

        result = registry.register(plugin)

        assert result is True
        storage = registry.get_storage_plugin("test_storage")
        assert storage is not None

    def test_registration_of_middleware(self):
        """Test registering a middleware plugin."""
        registry = PluginRegistry()
        plugin = self._create_mock_middleware("test_middleware")

        result = registry.register(plugin)

        assert result is True
        stats = registry.get_statistics()
        assert stats["middleware"] == 1

    def test_duplicate_registration_handled(self):
        """Test that re-registering same plugin updates it."""
        registry = PluginRegistry()
        plugin = self._create_mock_skill_chip("duplicate_test")

        registry.register(plugin)
        registry.register(plugin)  # Register again

        # Should have only one
        assert registry.get_plugin("duplicate_test") is not None

    def test_get_all_skill_chips_returns_list(self):
        """Test get_all_skill_chips() returns list of chips."""
        registry = PluginRegistry()
        registry.register(self._create_mock_skill_chip("chip1"))
        registry.register(self._create_mock_skill_chip("chip2"))

        chips = registry.get_all_skill_chips()

        assert isinstance(chips, list)
        assert len(chips) == 2

    def test_get_all_adapters_returns_list(self):
        """Test get_all_adapters() returns list of adapters."""
        registry = PluginRegistry()
        registry.register(self._create_mock_adapter("adapter1", "platform1"))
        registry.register(self._create_mock_adapter("adapter2", "platform2"))

        adapters = registry.get_all_adapters()

        assert isinstance(adapters, list)
        assert len(adapters) == 2

    @pytest.mark.asyncio
    async def test_middleware_chain_processing_request(self):
        """Test middleware chain processes requests."""
        registry = PluginRegistry()

        # Add middleware that modifies request
        mw_plugin = self._create_mock_middleware("chain_mw")
        mw_plugin.instance.process_request = AsyncMock(
            side_effect=lambda r: {**r, "processed": True}
        )
        registry.register(mw_plugin)

        request = {"data": "test"}
        result = await registry.process_request_middleware(request)

        assert result["processed"] is True
        assert result["data"] == "test"

    @pytest.mark.asyncio
    async def test_middleware_chain_processing_response(self):
        """Test middleware chain processes responses."""
        registry = PluginRegistry()

        mw_plugin = self._create_mock_middleware("response_mw")
        mw_plugin.instance.process_response = AsyncMock(
            side_effect=lambda r: {**r, "enriched": True}
        )
        registry.register(mw_plugin)

        response = {"result": "success"}
        result = await registry.process_response_middleware(response)

        assert result["enriched"] is True
        assert result["result"] == "success"

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        plugin = self._create_mock_skill_chip("to_remove")
        registry.register(plugin)

        result = registry.unregister("to_remove")

        assert result is True
        assert registry.get_plugin("to_remove") is None


# ===========================================================================
# Additional Registry Tests
# ===========================================================================

class TestRegistryAdvanced:
    """Advanced tests for PluginRegistry."""

    def test_get_adapter_by_platform(self):
        """Test getting adapter by platform name."""
        registry = PluginRegistry()

        mock_instance = MagicMock()
        mock_instance.get_adapter = MagicMock(return_value="teams_adapter")
        mock_instance.get_platform_name = MagicMock(return_value="teams")

        metadata = PluginMetadata(
            name="teams_plugin",
            version="1.0.0",
            author="Test",
            description="Teams adapter",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="adapter",
        )
        registry.register(plugin)

        adapter = registry.get_adapter_by_platform("teams")
        assert adapter == "teams_adapter"

    def test_get_skill_chip_for_intent(self):
        """Test getting skill chip for a specific intent."""
        registry = PluginRegistry()

        mock_instance = MagicMock()
        mock_chip = MagicMock()
        mock_instance.get_chip = MagicMock(return_value=mock_chip)
        mock_instance.get_intents = MagicMock(return_value=["grant_search", "grant_write"])

        metadata = PluginMetadata(
            name="grant_plugin",
            version="1.0.0",
            author="Test",
            description="Grant skill chip",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="skill_chip",
        )
        registry.register(plugin)

        chip = registry.get_skill_chip_for_intent("grant_search")
        assert chip is mock_chip

    def test_enable_disable_plugin(self):
        """Test enabling and disabling plugins."""
        registry = PluginRegistry()

        metadata = PluginMetadata(
            name="toggle_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=MagicMock(),
            plugin_type="skill_chip",
        )
        registry.register(plugin)

        # Disable
        result = registry.disable_plugin("toggle_plugin")
        assert result is True
        assert registry.get_plugin("toggle_plugin").enabled is False

        # Enable
        result = registry.enable_plugin("toggle_plugin")
        assert result is True
        assert registry.get_plugin("toggle_plugin").enabled is True

    def test_get_statistics(self):
        """Test get_statistics() returns comprehensive stats."""
        registry = PluginRegistry()

        stats = registry.get_statistics()

        assert "total_plugins" in stats
        assert "skill_chips" in stats
        assert "adapters" in stats
        assert "storage" in stats
        assert "middleware" in stats
        assert "enabled_plugins" in stats

    def test_event_listener(self):
        """Test plugin event listeners."""
        registry = PluginRegistry()
        events_received = []

        def listener(event: PluginEvent):
            events_received.append(event)

        registry.add_event_listener(listener)

        metadata = PluginMetadata(
            name="event_test",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=MagicMock(),
            plugin_type="skill_chip",
        )
        registry.register(plugin)

        assert len(events_received) > 0
        assert events_received[0].event_type == "plugin_registered"

    def test_get_intent_map(self):
        """Test get_intent_map() returns intent to plugin mapping."""
        registry = PluginRegistry()

        mock_instance = MagicMock()
        mock_instance.get_chip = MagicMock(return_value=MagicMock())
        mock_instance.get_intents = MagicMock(return_value=["intent_a", "intent_b"])

        metadata = PluginMetadata(
            name="intent_plugin",
            version="1.0.0",
            author="Test",
            description="Test",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="skill_chip",
        )
        registry.register(plugin)

        intent_map = registry.get_intent_map()
        assert "intent_a" in intent_map
        assert "intent_b" in intent_map
        assert intent_map["intent_a"] == "intent_plugin"


# ===========================================================================
# PluginBase Tests
# ===========================================================================

class TestPluginBase:
    """Tests for PluginBase optional base class."""

    def test_plugin_base_initialization(self):
        """Test PluginBase initialization."""
        class TestPlugin(PluginBase):
            metadata = PluginMetadata(
                name="base_test",
                version="1.0.0",
                author="Test",
                description="Test",
            )

        plugin = TestPlugin()
        assert plugin.is_initialized is False
        assert plugin.config is None

    @pytest.mark.asyncio
    async def test_plugin_base_initialize(self):
        """Test PluginBase.initialize()."""
        class TestPlugin(PluginBase):
            metadata = PluginMetadata(
                name="init_test",
                version="1.0.0",
                author="Test",
                description="Test",
            )

        plugin = TestPlugin()
        await plugin.initialize()
        assert plugin.is_initialized is True

    @pytest.mark.asyncio
    async def test_plugin_base_shutdown(self):
        """Test PluginBase.shutdown()."""
        class TestPlugin(PluginBase):
            metadata = PluginMetadata(
                name="shutdown_test",
                version="1.0.0",
                author="Test",
                description="Test",
            )

        plugin = TestPlugin()
        await plugin.initialize()
        await plugin.shutdown()
        assert plugin.is_initialized is False

    def test_plugin_base_configure(self):
        """Test PluginBase.configure()."""
        class TestPlugin(PluginBase):
            metadata = PluginMetadata(
                name="config_test",
                version="1.0.0",
                author="Test",
                description="Test",
            )

        plugin = TestPlugin()
        config = PluginConfig(values={"key": "value"})
        plugin.configure(config)

        assert plugin.config is not None
        assert plugin.config.get("key") == "value"

    def test_plugin_base_hooks(self):
        """Test PluginBase hook registration."""
        class TestPlugin(PluginBase):
            metadata = PluginMetadata(
                name="hook_test",
                version="1.0.0",
                author="Test",
                description="Test",
            )

        plugin = TestPlugin()

        def my_handler(event):
            pass

        plugin.register_hook("on_message_received", my_handler)

        handlers = plugin.get_hook_handlers("on_message_received")
        assert len(handlers) == 1
        assert handlers[0] is my_handler


# ===========================================================================
# Plugin Hooks Tests
# ===========================================================================

class TestPluginHooks:
    """Tests for plugin hook system."""

    def test_built_in_hooks_defined(self):
        """Test that built-in hooks are defined."""
        hook_names = [h.name for h in PLUGIN_HOOKS]

        assert "on_message_received" in hook_names
        assert "on_intent_classified" in hook_names
        assert "on_response_generated" in hook_names
        assert "on_memory_stored" in hook_names

    def test_hook_has_parameters(self):
        """Test hooks have parameter definitions."""
        for hook in PLUGIN_HOOKS:
            assert isinstance(hook.parameters, list)
            assert isinstance(hook.description, str)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestPluginIntegration:
    """Integration tests for plugin system components."""

    @pytest.mark.asyncio
    async def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle from load to unload."""
        # Create registry
        registry = PluginRegistry()

        # Create mock plugin
        mock_instance = MagicMock()
        mock_instance.get_chip = MagicMock(return_value=MagicMock())
        mock_instance.get_intents = MagicMock(return_value=["test_intent"])

        metadata = PluginMetadata(
            name="lifecycle_test",
            version="1.0.0",
            author="Test",
            description="Test lifecycle",
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=mock_instance,
            plugin_type="skill_chip",
        )

        # Register
        assert registry.register(plugin) is True
        assert registry.get_plugin("lifecycle_test") is not None

        # Disable
        registry.disable_plugin("lifecycle_test")
        assert registry.get_plugin("lifecycle_test").enabled is False

        # Enable
        registry.enable_plugin("lifecycle_test")
        assert registry.get_plugin("lifecycle_test").enabled is True

        # Unregister
        assert registry.unregister("lifecycle_test") is True
        assert registry.get_plugin("lifecycle_test") is None

    def test_sandbox_with_registry(self):
        """Test sandbox validation with registry."""
        sandbox = PluginSandbox(SandboxPolicy(
            allow_network=True,
            max_memory_mb=512,
        ))

        metadata = PluginMetadata(
            name="sandbox_reg_test",
            version="1.0.0",
            author="Test",
            description="Test",
            required_capabilities=["network"],
        )
        plugin = LoadedPlugin(
            metadata=metadata,
            module=MagicMock(),
            state=PluginState.LOADED,
            instance=MagicMock(),
        )

        violations = sandbox.validate_plugin(plugin)
        # Should have no capability violations since network is allowed
        cap_violations = [v for v in violations if v.violation_type == "missing_capability"]
        assert len(cap_violations) == 0

    @pytest.mark.asyncio
    async def test_middleware_ordering(self):
        """Test middleware executes in priority order."""
        registry = PluginRegistry()
        execution_order = []

        # Create middleware with different priorities
        for priority, name in [(50, "first"), (100, "second"), (150, "third")]:
            mock_instance = AsyncMock()

            async def make_processor(n):
                async def process(request):
                    execution_order.append(n)
                    return request
                return process

            mock_instance.process_request = AsyncMock(
                side_effect=lambda r, n=name: (execution_order.append(n), r)[1]
            )
            mock_instance.process_response = AsyncMock(side_effect=lambda r: r)
            mock_instance.get_priority = MagicMock(return_value=priority)

            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                author="Test",
                description="Test",
            )
            plugin = LoadedPlugin(
                metadata=metadata,
                module=MagicMock(),
                state=PluginState.LOADED,
                instance=mock_instance,
                plugin_type="middleware",
            )
            registry.register(plugin)

        await registry.process_request_middleware({})

        # Should execute in priority order (lowest first)
        assert execution_order == ["first", "second", "third"]


# ===========================================================================
# Restricted Importer Tests
# ===========================================================================

class TestRestrictedImporter:
    """Tests for RestrictedImporter."""

    def test_allowed_import(self):
        """Test allowed imports pass through."""
        policy = SandboxPolicy(allowed_imports=["json"])
        importer = RestrictedImporter(policy)

        result = importer.find_module("json")
        assert result is None  # None means allow

    def test_blocked_import(self):
        """Test blocked imports are caught."""
        policy = SandboxPolicy(allowed_imports=["json"])
        importer = RestrictedImporter(policy)

        result = importer.find_module("os")
        assert result is importer  # Returns self to block

    def test_violations_recorded(self):
        """Test violations are recorded."""
        policy = SandboxPolicy(allowed_imports=["json"])
        importer = RestrictedImporter(policy)

        importer.find_module("subprocess")

        assert len(importer.violations) == 1
        assert importer.violations[0].violation_type == "blocked_import"
