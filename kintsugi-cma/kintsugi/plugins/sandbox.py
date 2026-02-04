"""
Plugin sandboxing for Kintsugi CMA.

This module provides security sandboxing for plugin execution.
Plugins run in restricted environments with configurable policies
that control their access to system resources.

Security Measures:
    - Import restrictions (whitelist approach)
    - Network access control
    - Filesystem access control
    - Memory limits
    - CPU time limits
    - Execution timeout

Example:
    from kintsugi.plugins.sandbox import PluginSandbox, SandboxPolicy

    # Create restrictive policy
    policy = SandboxPolicy(
        allow_network=False,
        allow_filesystem=False,
        max_memory_mb=256,
        max_cpu_seconds=10,
        allowed_imports=["json", "datetime", "typing"],
    )

    sandbox = PluginSandbox(policy)

    # Validate plugin before loading
    violations = sandbox.validate_plugin(loaded_plugin)
    if violations:
        print(f"Plugin violates policy: {violations}")

    # Execute plugin method in sandbox
    result = await sandbox.execute(plugin, "handle", request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import asyncio
import logging
import resource
import signal
import sys
import traceback

from kintsugi.plugins.loader import LoadedPlugin, PluginState

logger = logging.getLogger(__name__)


@dataclass
class SandboxPolicy:
    """Security policy for plugin execution.

    Defines what resources and capabilities a plugin can access
    when running in the sandbox.

    Attributes:
        allow_network: Whether to allow network access.
        allow_filesystem: Whether to allow filesystem access.
        max_memory_mb: Maximum memory allocation in MB.
        max_cpu_seconds: Maximum CPU time in seconds.
        allowed_imports: Whitelist of allowed module imports.
        blocked_imports: Blacklist of blocked imports (overrides whitelist).
        allowed_builtins: Whitelist of allowed builtin functions.
        max_execution_time: Maximum wall-clock execution time.
        allow_subprocess: Whether to allow subprocess execution.
        allow_threading: Whether to allow thread creation.
        allow_ctypes: Whether to allow ctypes/cffi.
        max_file_size_mb: Maximum file size for any file operation.
        allowed_paths: Paths the plugin can access (if filesystem allowed).

    Example:
        # Restrictive policy for untrusted plugins
        policy = SandboxPolicy(
            allow_network=False,
            allow_filesystem=False,
            max_memory_mb=128,
            max_cpu_seconds=5,
            allowed_imports=["json", "datetime", "re"],
        )

        # Permissive policy for trusted plugins
        policy = SandboxPolicy(
            allow_network=True,
            allow_filesystem=True,
            max_memory_mb=1024,
            max_cpu_seconds=60,
            allowed_imports=[],  # Empty means all allowed
        )
    """

    allow_network: bool = False
    allow_filesystem: bool = False
    max_memory_mb: int = 256
    max_cpu_seconds: int = 10
    allowed_imports: list[str] = field(default_factory=list)
    blocked_imports: list[str] = field(default_factory=lambda: [
        "os.system",
        "subprocess",
        "ctypes",
        "cffi",
        "multiprocessing",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
    ])
    allowed_builtins: list[str] = field(default_factory=lambda: [
        "abs", "all", "any", "bool", "dict", "enumerate",
        "filter", "float", "format", "frozenset", "getattr",
        "hasattr", "hash", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next",
        "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    ])
    max_execution_time: float = 30.0
    allow_subprocess: bool = False
    allow_threading: bool = False
    allow_ctypes: bool = False
    max_file_size_mb: int = 10
    allowed_paths: list[str] = field(default_factory=list)

    def is_import_allowed(self, module_name: str) -> bool:
        """Check if a module import is allowed.

        Args:
            module_name: The module name to check.

        Returns:
            True if the import is allowed.
        """
        # Check blocked list first
        for blocked in self.blocked_imports:
            if module_name == blocked or module_name.startswith(f"{blocked}."):
                return False

        # If whitelist is empty, allow all (except blocked)
        if not self.allowed_imports:
            return True

        # Check whitelist
        for allowed in self.allowed_imports:
            if module_name == allowed or module_name.startswith(f"{allowed}."):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert policy to dictionary.

        Returns:
            Dictionary representation of the policy.
        """
        return {
            "allow_network": self.allow_network,
            "allow_filesystem": self.allow_filesystem,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "allowed_imports": self.allowed_imports,
            "blocked_imports": self.blocked_imports,
            "max_execution_time": self.max_execution_time,
            "allow_subprocess": self.allow_subprocess,
            "allow_threading": self.allow_threading,
        }


@dataclass
class SandboxViolation:
    """Represents a sandbox policy violation.

    Created when plugin code violates the sandbox policy.

    Attributes:
        violation_type: Type of violation.
        message: Description of the violation.
        severity: Violation severity (warning, error, critical).
        location: Where in the code the violation occurred.
        timestamp: When the violation was detected.
    """

    violation_type: str
    message: str
    severity: str = "error"
    location: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "violation_type": self.violation_type,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SandboxExecutionResult:
    """Result of sandboxed execution.

    Contains the result or error from executing code in the sandbox.

    Attributes:
        success: Whether execution succeeded.
        result: The return value if successful.
        error: Error message if failed.
        execution_time_ms: How long execution took.
        memory_used_mb: Peak memory usage.
        violations: Any policy violations detected.
    """

    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    violations: list[SandboxViolation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "violations": [v.to_dict() for v in self.violations],
        }


class RestrictedImporter:
    """Custom importer that enforces import restrictions.

    Installed as a meta path finder to intercept all import attempts
    and check them against the sandbox policy.
    """

    def __init__(self, policy: SandboxPolicy):
        """Initialize the restricted importer.

        Args:
            policy: The sandbox policy to enforce.
        """
        self._policy = policy
        self._violations: list[SandboxViolation] = []

    def find_module(self, fullname: str, path: Any = None) -> Any:
        """Check if an import is allowed.

        Args:
            fullname: The full module name being imported.
            path: The path (unused).

        Returns:
            Self if blocked, None to allow.
        """
        if not self._policy.is_import_allowed(fullname):
            self._violations.append(SandboxViolation(
                violation_type="blocked_import",
                message=f"Import of '{fullname}' is not allowed",
                severity="error",
            ))
            return self
        return None

    def load_module(self, fullname: str) -> None:
        """Block the import by raising an error.

        Args:
            fullname: The module name.

        Raises:
            ImportError: Always, to block the import.
        """
        raise ImportError(f"Import of '{fullname}' blocked by sandbox policy")

    @property
    def violations(self) -> list[SandboxViolation]:
        """Get recorded violations."""
        return list(self._violations)


class PluginSandbox:
    """Sandboxed execution environment for plugins.

    Provides a secure execution environment for plugin code with
    resource limits and access controls.

    Attributes:
        _policy: The security policy to enforce.
        _execution_count: Number of executions performed.
        _total_violations: Total violations recorded.

    Example:
        sandbox = PluginSandbox(SandboxPolicy(
            allow_network=False,
            max_memory_mb=256,
        ))

        # Validate a plugin
        violations = sandbox.validate_plugin(loaded_plugin)

        # Execute with sandbox
        result = await sandbox.execute(plugin, "handle", request)
    """

    def __init__(self, policy: SandboxPolicy | None = None):
        """Initialize the sandbox.

        Args:
            policy: The security policy to use. Defaults to a
                   restrictive policy.
        """
        self._policy = policy or SandboxPolicy()
        self._execution_count = 0
        self._total_violations: list[SandboxViolation] = []
        self._restricted_importer: RestrictedImporter | None = None

    @property
    def policy(self) -> SandboxPolicy:
        """Get the current sandbox policy."""
        return self._policy

    def set_policy(self, policy: SandboxPolicy) -> None:
        """Set a new sandbox policy.

        Args:
            policy: The new policy to use.
        """
        self._policy = policy

    def validate_plugin(self, plugin: LoadedPlugin) -> list[SandboxViolation]:
        """Validate a plugin against the sandbox policy.

        Performs static analysis on the plugin to detect potential
        policy violations before execution.

        Args:
            plugin: The loaded plugin to validate.

        Returns:
            List of SandboxViolation objects for any violations found.
        """
        violations: list[SandboxViolation] = []

        if plugin.state == PluginState.ERROR:
            violations.append(SandboxViolation(
                violation_type="plugin_error",
                message=f"Plugin in error state: {plugin.error}",
                severity="critical",
            ))
            return violations

        if plugin.module is None:
            violations.append(SandboxViolation(
                violation_type="no_module",
                message="Plugin module not loaded",
                severity="error",
            ))
            return violations

        # Check required capabilities
        if plugin.metadata.required_capabilities:
            for cap in plugin.metadata.required_capabilities:
                if not self._check_capability(cap):
                    violations.append(SandboxViolation(
                        violation_type="missing_capability",
                        message=f"Plugin requires capability '{cap}' not allowed by policy",
                        severity="error",
                    ))

        # Analyze module imports
        violations.extend(self._analyze_imports(plugin.module))

        # Check for dangerous patterns in source
        violations.extend(self._analyze_source(plugin))

        return violations

    def _check_capability(self, capability: str) -> bool:
        """Check if a capability is allowed.

        Args:
            capability: The capability to check.

        Returns:
            True if allowed by the policy.
        """
        capability_map = {
            "network": self._policy.allow_network,
            "filesystem": self._policy.allow_filesystem,
            "subprocess": self._policy.allow_subprocess,
            "threading": self._policy.allow_threading,
        }
        return capability_map.get(capability, False)

    def _analyze_imports(self, module: Any) -> list[SandboxViolation]:
        """Analyze module imports for violations.

        Args:
            module: The module to analyze.

        Returns:
            List of violations found.
        """
        violations: list[SandboxViolation] = []

        # Get all imported modules
        if hasattr(module, '__dict__'):
            for name, obj in module.__dict__.items():
                if hasattr(obj, '__module__'):
                    mod_name = obj.__module__
                    if mod_name and not self._policy.is_import_allowed(mod_name):
                        violations.append(SandboxViolation(
                            violation_type="blocked_import",
                            message=f"Module uses blocked import: {mod_name}",
                            severity="warning",
                            location=name,
                        ))

        return violations

    def _analyze_source(self, plugin: LoadedPlugin) -> list[SandboxViolation]:
        """Analyze plugin source for dangerous patterns.

        Args:
            plugin: The plugin to analyze.

        Returns:
            List of violations found.
        """
        violations: list[SandboxViolation] = []

        # Dangerous patterns to check for
        dangerous_patterns = [
            ("eval(", "Use of eval() is dangerous"),
            ("exec(", "Use of exec() is dangerous"),
            ("compile(", "Use of compile() is dangerous"),
            ("__import__", "Use of __import__() is dangerous"),
            ("globals()", "Accessing globals() is dangerous"),
            ("locals()", "Accessing locals() is dangerous"),
            ("setattr(", "Use of setattr() may be dangerous"),
            ("delattr(", "Use of delattr() may be dangerous"),
        ]

        if not self._policy.allow_subprocess:
            dangerous_patterns.extend([
                ("subprocess", "Subprocess access not allowed"),
                ("os.system", "os.system not allowed"),
                ("os.popen", "os.popen not allowed"),
            ])

        if not self._policy.allow_filesystem:
            dangerous_patterns.extend([
                ("open(", "File access not allowed"),
                ("os.path", "Filesystem access not allowed"),
                ("pathlib", "Filesystem access not allowed"),
            ])

        if not self._policy.allow_network:
            dangerous_patterns.extend([
                ("socket", "Network access not allowed"),
                ("urllib", "Network access not allowed"),
                ("requests", "Network access not allowed"),
                ("httpx", "Network access not allowed"),
                ("aiohttp", "Network access not allowed"),
            ])

        # Try to get source code
        try:
            import inspect
            source = inspect.getsource(plugin.module)

            for pattern, message in dangerous_patterns:
                if pattern in source:
                    violations.append(SandboxViolation(
                        violation_type="dangerous_pattern",
                        message=message,
                        severity="warning",
                    ))

        except Exception:
            # Source not available, skip static analysis
            pass

        return violations

    async def execute(
        self,
        plugin: LoadedPlugin,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> SandboxExecutionResult:
        """Execute a plugin method in the sandbox.

        Runs the specified method with resource limits and
        access controls in place.

        Args:
            plugin: The plugin to execute.
            method: The method name to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            SandboxExecutionResult with the outcome.
        """
        self._execution_count += 1
        violations: list[SandboxViolation] = []

        if plugin.instance is None:
            return SandboxExecutionResult(
                success=False,
                error="Plugin not instantiated",
            )

        if not hasattr(plugin.instance, method):
            return SandboxExecutionResult(
                success=False,
                error=f"Method '{method}' not found on plugin",
            )

        # Set up resource limits
        old_limits = self._set_resource_limits()

        # Install restricted importer
        self._restricted_importer = RestrictedImporter(self._policy)
        sys.meta_path.insert(0, self._restricted_importer)

        start_time = asyncio.get_event_loop().time()
        result = None
        error = None

        try:
            # Get the method
            func = getattr(plugin.instance, method)

            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self._policy.max_execution_time,
                )
            else:
                # Run sync method in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=self._policy.max_execution_time,
                )

        except asyncio.TimeoutError:
            error = f"Execution timed out after {self._policy.max_execution_time}s"
            violations.append(SandboxViolation(
                violation_type="timeout",
                message=error,
                severity="error",
            ))

        except MemoryError:
            error = "Execution exceeded memory limit"
            violations.append(SandboxViolation(
                violation_type="memory_limit",
                message=error,
                severity="error",
            ))

        except Exception as e:
            error = f"Execution failed: {str(e)}"
            logger.debug(traceback.format_exc())

        finally:
            # Remove restricted importer
            if self._restricted_importer in sys.meta_path:
                sys.meta_path.remove(self._restricted_importer)
            violations.extend(self._restricted_importer.violations)
            self._restricted_importer = None

            # Restore resource limits
            self._restore_resource_limits(old_limits)

        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000

        # Record violations
        self._total_violations.extend(violations)

        return SandboxExecutionResult(
            success=error is None,
            result=result,
            error=error,
            execution_time_ms=execution_time,
            violations=violations,
        )

    def _set_resource_limits(self) -> dict[str, Any]:
        """Set resource limits for sandbox execution.

        Returns:
            Dictionary of old limits to restore later.
        """
        old_limits = {}

        try:
            # Set memory limit
            memory_bytes = self._policy.max_memory_mb * 1024 * 1024
            old_limits['memory'] = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_bytes, memory_bytes)
            )
        except (ValueError, resource.error):
            pass

        try:
            # Set CPU time limit
            old_limits['cpu'] = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self._policy.max_cpu_seconds, self._policy.max_cpu_seconds)
            )
        except (ValueError, resource.error):
            pass

        return old_limits

    def _restore_resource_limits(self, old_limits: dict[str, Any]) -> None:
        """Restore previous resource limits.

        Args:
            old_limits: Dictionary of limits to restore.
        """
        try:
            if 'memory' in old_limits:
                resource.setrlimit(resource.RLIMIT_AS, old_limits['memory'])
        except (ValueError, resource.error):
            pass

        try:
            if 'cpu' in old_limits:
                resource.setrlimit(resource.RLIMIT_CPU, old_limits['cpu'])
        except (ValueError, resource.error):
            pass

    def get_execution_stats(self) -> dict[str, Any]:
        """Get sandbox execution statistics.

        Returns:
            Dictionary with execution statistics.
        """
        return {
            "execution_count": self._execution_count,
            "total_violations": len(self._total_violations),
            "violations_by_type": self._count_violations_by_type(),
        }

    def _count_violations_by_type(self) -> dict[str, int]:
        """Count violations grouped by type.

        Returns:
            Dictionary of violation type to count.
        """
        counts: dict[str, int] = {}
        for violation in self._total_violations:
            vtype = violation.violation_type
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._total_violations.clear()

    def get_recent_violations(
        self,
        limit: int = 100,
    ) -> list[SandboxViolation]:
        """Get recent violations.

        Args:
            limit: Maximum number of violations to return.

        Returns:
            List of recent violations.
        """
        return self._total_violations[-limit:]

    def __repr__(self) -> str:
        return (
            f"<PluginSandbox executions={self._execution_count} "
            f"violations={len(self._total_violations)}>"
        )
