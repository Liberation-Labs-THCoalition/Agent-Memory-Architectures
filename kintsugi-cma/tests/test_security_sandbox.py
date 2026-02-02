"""Comprehensive pytest tests for kintsugi.security.sandbox module.

Tests cover:
- Sandbox creation and cleanup
- Code execution with stdout/stderr capture
- Timeout enforcement
- Exit code handling
- Auto-cleanup vs manual cleanup
- Sandbox reuse
- Error handling
"""

import os
import time
from pathlib import Path

import pytest

from kintsugi.security.sandbox import (
    SandboxContext,
    SandboxResult,
    ShadowSandbox,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sandbox():
    """Fresh ShadowSandbox instance with cleanup after test."""
    sb = ShadowSandbox()
    yield sb
    # Cleanup any remaining sandboxes after test
    sb.cleanup_all()


@pytest.fixture
def temp_base_dir(tmp_path):
    """Temporary directory for sandbox base."""
    return str(tmp_path / "sandbox_base")


# ==============================================================================
# Test SandboxContext and SandboxResult Dataclasses
# ==============================================================================

def test_sandbox_context_frozen():
    """SandboxContext is immutable (frozen dataclass)."""
    from datetime import datetime, timezone

    ctx = SandboxContext(
        id="test-123",
        path="/tmp/test",
        created_at=datetime.now(timezone.utc),
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        ctx.id = "modified"


def test_sandbox_result_frozen():
    """SandboxResult is immutable (frozen dataclass)."""
    result = SandboxResult(
        stdout="test",
        stderr="",
        exit_code=0,
        timed_out=False,
        execution_time_ms=100.0,
    )
    with pytest.raises(Exception):
        result.stdout = "modified"


# ==============================================================================
# Test ShadowSandbox Initialization
# ==============================================================================

def test_shadow_sandbox_init_default():
    """ShadowSandbox initializes with default base_dir."""
    sb = ShadowSandbox()
    assert sb._base_dir is None
    assert sb._sandboxes == {}


def test_shadow_sandbox_init_with_base_dir(temp_base_dir):
    """ShadowSandbox initializes with custom base_dir."""
    sb = ShadowSandbox(base_dir=temp_base_dir)
    assert sb._base_dir == temp_base_dir


# ==============================================================================
# Test create_sandbox
# ==============================================================================

def test_create_sandbox_returns_context(sandbox):
    """create_sandbox returns a SandboxContext."""
    ctx = sandbox.create_sandbox()

    assert isinstance(ctx, SandboxContext)
    assert isinstance(ctx.id, str)
    assert len(ctx.id) == 16  # uuid.hex[:16]
    assert isinstance(ctx.path, str)
    assert ctx.path.startswith("/tmp/") or ctx.path.startswith("/var/")
    assert "kintsugi_sandbox_" in ctx.path
    assert ctx.id in ctx.path


def test_create_sandbox_creates_directory(sandbox):
    """create_sandbox creates actual directory on filesystem."""
    ctx = sandbox.create_sandbox()

    assert os.path.exists(ctx.path)
    assert os.path.isdir(ctx.path)


def test_create_sandbox_unique_ids(sandbox):
    """create_sandbox generates unique IDs for each sandbox."""
    ctx1 = sandbox.create_sandbox()
    ctx2 = sandbox.create_sandbox()

    assert ctx1.id != ctx2.id
    assert ctx1.path != ctx2.path


def test_create_sandbox_registers_sandbox(sandbox):
    """create_sandbox registers sandbox in internal dict."""
    ctx = sandbox.create_sandbox()

    assert ctx.id in sandbox._sandboxes
    assert sandbox._sandboxes[ctx.id] == ctx


def test_create_sandbox_with_custom_base_dir(temp_base_dir):
    """create_sandbox respects custom base_dir."""
    os.makedirs(temp_base_dir, exist_ok=True)
    sb = ShadowSandbox(base_dir=temp_base_dir)

    ctx = sb.create_sandbox()
    assert ctx.path.startswith(temp_base_dir)

    sb.cleanup_all()


def test_create_sandbox_sets_created_at_timestamp(sandbox):
    """create_sandbox sets created_at timestamp."""
    from datetime import datetime, timezone

    before = datetime.now(timezone.utc)
    ctx = sandbox.create_sandbox()
    after = datetime.now(timezone.utc)

    assert before <= ctx.created_at <= after


# ==============================================================================
# Test execute_in_sandbox - Basic Execution
# ==============================================================================

def test_execute_simple_code(sandbox):
    """execute_in_sandbox runs simple Python code."""
    code = "print('Hello, World!')"
    result = sandbox.execute_in_sandbox(code)

    assert isinstance(result, SandboxResult)
    assert result.stdout.strip() == "Hello, World!"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.execution_time_ms > 0


def test_execute_code_with_computation(sandbox):
    """execute_in_sandbox runs code with computation."""
    code = """
result = sum(range(100))
print(f"Sum: {result}")
"""
    result = sandbox.execute_in_sandbox(code)

    assert "Sum: 4950" in result.stdout
    assert result.exit_code == 0


def test_execute_code_with_multiple_prints(sandbox):
    """execute_in_sandbox captures multiple print statements."""
    code = """
print("Line 1")
print("Line 2")
print("Line 3")
"""
    result = sandbox.execute_in_sandbox(code)

    assert "Line 1" in result.stdout
    assert "Line 2" in result.stdout
    assert "Line 3" in result.stdout


def test_execute_empty_code(sandbox):
    """execute_in_sandbox handles empty code."""
    result = sandbox.execute_in_sandbox("")

    assert result.stdout == ""
    assert result.exit_code == 0


# ==============================================================================
# Test execute_in_sandbox - Error Handling
# ==============================================================================

def test_execute_code_with_syntax_error(sandbox):
    """execute_in_sandbox captures syntax errors."""
    code = "print('unclosed string"
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code != 0
    assert "SyntaxError" in result.stderr or "error" in result.stderr.lower()


def test_execute_code_with_runtime_error(sandbox):
    """execute_in_sandbox captures runtime errors."""
    code = """
def divide_by_zero():
    return 1 / 0

divide_by_zero()
"""
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code != 0
    assert "ZeroDivisionError" in result.stderr


def test_execute_code_with_exception(sandbox):
    """execute_in_sandbox handles unhandled exceptions."""
    code = "raise ValueError('Test error')"
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "Test error" in result.stderr


def test_execute_code_with_stderr_output(sandbox):
    """execute_in_sandbox captures stderr separately."""
    code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""
    result = sandbox.execute_in_sandbox(code)

    assert "stdout message" in result.stdout
    assert "stderr message" in result.stderr
    # When there's stderr but code exits normally, exit_code might be 0
    # depending on whether the code raises
    assert result.exit_code == 0


# ==============================================================================
# Test execute_in_sandbox - Timeout
# ==============================================================================

def test_execute_timeout_kills_long_running_code(sandbox):
    """execute_in_sandbox enforces timeout and kills process."""
    code = """
import time
time.sleep(10)  # Sleep for 10 seconds
"""
    result = sandbox.execute_in_sandbox(code, timeout=1)  # 1 second timeout

    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()
    assert result.exit_code == -1


def test_execute_fast_code_does_not_timeout(sandbox):
    """execute_in_sandbox does not timeout for fast code."""
    code = "print('quick')"
    result = sandbox.execute_in_sandbox(code, timeout=5)

    assert result.timed_out is False
    assert result.exit_code == 0


def test_execute_timeout_default_value(sandbox):
    """execute_in_sandbox uses default timeout of 30 seconds."""
    code = "print('test')"
    result = sandbox.execute_in_sandbox(code)  # No timeout specified

    assert result.timed_out is False
    assert result.execution_time_ms < 30000  # Should be much less


def test_execute_timeout_boundary(sandbox):
    """execute_in_sandbox handles timeout boundary conditions."""
    # Code that sleeps just under the timeout
    code = """
import time
time.sleep(0.5)
print('completed')
"""
    result = sandbox.execute_in_sandbox(code, timeout=2)

    assert result.timed_out is False
    assert "completed" in result.stdout


# ==============================================================================
# Test execute_in_sandbox - Execution Time Tracking
# ==============================================================================

def test_execute_tracks_execution_time(sandbox):
    """execute_in_sandbox measures execution time."""
    code = """
import time
time.sleep(0.1)
"""
    result = sandbox.execute_in_sandbox(code)

    assert result.execution_time_ms >= 100  # At least 100ms
    assert result.execution_time_ms < 5000  # But less than 5 seconds


def test_execute_time_rounded_to_2_decimals(sandbox):
    """execute_in_sandbox rounds execution time to 2 decimal places."""
    code = "print('test')"
    result = sandbox.execute_in_sandbox(code)

    # Check that value has at most 2 decimal places
    time_str = str(result.execution_time_ms)
    if "." in time_str:
        decimals = len(time_str.split(".")[1])
        assert decimals <= 2


# ==============================================================================
# Test execute_in_sandbox - Sandbox Reuse
# ==============================================================================

def test_execute_with_explicit_sandbox_id(sandbox):
    """execute_in_sandbox can execute in existing sandbox."""
    ctx = sandbox.create_sandbox()

    result = sandbox.execute_in_sandbox("print('test')", sandbox_id=ctx.id)

    assert result.exit_code == 0
    assert "test" in result.stdout


def test_execute_reuses_sandbox_directory(sandbox):
    """execute_in_sandbox reuses sandbox directory for same sandbox_id."""
    ctx = sandbox.create_sandbox()

    # First execution creates a file
    code1 = """
with open('test.txt', 'w') as f:
    f.write('hello')
"""
    result1 = sandbox.execute_in_sandbox(code1, sandbox_id=ctx.id)
    assert result1.exit_code == 0

    # Second execution reads the file
    code2 = """
with open('test.txt', 'r') as f:
    print(f.read())
"""
    result2 = sandbox.execute_in_sandbox(code2, sandbox_id=ctx.id)
    assert "hello" in result2.stdout


def test_execute_with_invalid_sandbox_id(sandbox):
    """execute_in_sandbox returns error for invalid sandbox_id."""
    result = sandbox.execute_in_sandbox("print('test')", sandbox_id="nonexistent")

    assert result.exit_code == -1
    assert "not found" in result.stderr
    assert result.stdout == ""
    assert result.timed_out is False


def test_execute_without_sandbox_id_auto_creates(sandbox):
    """execute_in_sandbox auto-creates sandbox when sandbox_id is None."""
    initial_count = len(sandbox._sandboxes)

    result = sandbox.execute_in_sandbox("print('test')")

    assert result.exit_code == 0
    # Auto-cleanup should remove it
    assert len(sandbox._sandboxes) == initial_count


def test_execute_with_sandbox_id_no_auto_cleanup(sandbox):
    """execute_in_sandbox doesn't cleanup when sandbox_id is provided."""
    ctx = sandbox.create_sandbox()

    result = sandbox.execute_in_sandbox("print('test')", sandbox_id=ctx.id)

    assert result.exit_code == 0
    # Sandbox should still exist
    assert ctx.id in sandbox._sandboxes
    assert os.path.exists(ctx.path)


def test_execute_auto_cleanup_removes_sandbox(sandbox):
    """execute_in_sandbox auto-cleanup removes temporary sandbox."""
    # Count initial sandboxes
    initial_count = len(sandbox._sandboxes)

    result = sandbox.execute_in_sandbox("print('test')")  # No sandbox_id

    assert result.exit_code == 0
    # Should return to initial count (auto-cleanup)
    assert len(sandbox._sandboxes) == initial_count


# ==============================================================================
# Test cleanup
# ==============================================================================

def test_cleanup_removes_directory(sandbox):
    """cleanup removes sandbox directory from filesystem."""
    ctx = sandbox.create_sandbox()
    path = ctx.path

    assert os.path.exists(path)

    sandbox.cleanup(ctx.id)

    assert not os.path.exists(path)


def test_cleanup_deregisters_sandbox(sandbox):
    """cleanup removes sandbox from internal registry."""
    ctx = sandbox.create_sandbox()

    assert ctx.id in sandbox._sandboxes

    sandbox.cleanup(ctx.id)

    assert ctx.id not in sandbox._sandboxes


def test_cleanup_nonexistent_sandbox_no_error(sandbox):
    """cleanup handles nonexistent sandbox_id gracefully."""
    # Should not raise exception
    sandbox.cleanup("nonexistent-id")


def test_cleanup_already_cleaned_sandbox(sandbox):
    """cleanup can be called multiple times on same sandbox_id."""
    ctx = sandbox.create_sandbox()

    sandbox.cleanup(ctx.id)
    # Second cleanup should not error
    sandbox.cleanup(ctx.id)


def test_cleanup_ignores_errors_during_removal(sandbox):
    """cleanup uses ignore_errors=True for rmtree."""
    ctx = sandbox.create_sandbox()

    # Even if directory is already removed, cleanup should not error
    import shutil

    shutil.rmtree(ctx.path, ignore_errors=True)

    sandbox.cleanup(ctx.id)  # Should not raise


# ==============================================================================
# Test cleanup_all
# ==============================================================================

def test_cleanup_all_removes_all_sandboxes(sandbox):
    """cleanup_all removes all registered sandboxes."""
    ctx1 = sandbox.create_sandbox()
    ctx2 = sandbox.create_sandbox()
    ctx3 = sandbox.create_sandbox()

    assert len(sandbox._sandboxes) == 3

    sandbox.cleanup_all()

    assert len(sandbox._sandboxes) == 0
    assert not os.path.exists(ctx1.path)
    assert not os.path.exists(ctx2.path)
    assert not os.path.exists(ctx3.path)


def test_cleanup_all_with_no_sandboxes(sandbox):
    """cleanup_all handles empty sandbox registry."""
    assert len(sandbox._sandboxes) == 0

    sandbox.cleanup_all()  # Should not error

    assert len(sandbox._sandboxes) == 0


def test_cleanup_all_is_idempotent(sandbox):
    """cleanup_all can be called multiple times."""
    ctx = sandbox.create_sandbox()

    sandbox.cleanup_all()
    sandbox.cleanup_all()  # Second call should not error

    assert len(sandbox._sandboxes) == 0


# ==============================================================================
# Test Sandbox Isolation
# ==============================================================================

def test_sandbox_isolated_working_directory(sandbox):
    """execute_in_sandbox runs code in sandbox directory."""
    code = """
import os
print(os.getcwd())
"""
    ctx = sandbox.create_sandbox()
    result = sandbox.execute_in_sandbox(code, sandbox_id=ctx.id)

    assert ctx.path in result.stdout


def test_sandbox_isolated_environment(sandbox):
    """execute_in_sandbox uses restricted environment."""
    code = """
import os
path = os.environ.get('PATH', '')
home = os.environ.get('HOME', '')
print(f"PATH={path}")
print(f"HOME={home}")
"""
    ctx = sandbox.create_sandbox()
    result = sandbox.execute_in_sandbox(code, sandbox_id=ctx.id)

    # Environment should be restricted
    assert "PATH=" in result.stdout
    assert "HOME=" in result.stdout
    # HOME should be set to sandbox path
    assert ctx.path in result.stdout


def test_sandbox_filesystem_isolation(sandbox):
    """Sandboxes are isolated from each other."""
    ctx1 = sandbox.create_sandbox()
    ctx2 = sandbox.create_sandbox()

    # Write file in sandbox 1
    code1 = """
with open('data.txt', 'w') as f:
    f.write('sandbox1')
print('written')
"""
    result1 = sandbox.execute_in_sandbox(code1, sandbox_id=ctx1.id)
    assert "written" in result1.stdout

    # Try to read in sandbox 2 (should fail)
    code2 = """
import os
exists = os.path.exists('data.txt')
print(f"File exists: {exists}")
"""
    result2 = sandbox.execute_in_sandbox(code2, sandbox_id=ctx2.id)
    assert "File exists: False" in result2.stdout


# ==============================================================================
# Test Script File Creation
# ==============================================================================

def test_execute_creates_script_file(sandbox):
    """execute_in_sandbox creates script.py in sandbox."""
    code = "print('test')"
    ctx = sandbox.create_sandbox()

    sandbox.execute_in_sandbox(code, sandbox_id=ctx.id)

    script_path = Path(ctx.path) / "script.py"
    assert script_path.exists()
    assert script_path.read_text(encoding="utf-8") == code


def test_execute_overwrites_previous_script(sandbox):
    """execute_in_sandbox overwrites script.py on subsequent runs."""
    ctx = sandbox.create_sandbox()

    # First execution
    sandbox.execute_in_sandbox("print('first')", sandbox_id=ctx.id)

    # Second execution
    sandbox.execute_in_sandbox("print('second')", sandbox_id=ctx.id)

    script_path = Path(ctx.path) / "script.py"
    content = script_path.read_text(encoding="utf-8")
    assert content == "print('second')"


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_execute_unicode_code(sandbox):
    """execute_in_sandbox handles unicode in code."""
    code = """
print('こんにちは')
print('Hello 🌍')
"""
    result = sandbox.execute_in_sandbox(code)

    assert "こんにちは" in result.stdout
    assert "🌍" in result.stdout


def test_execute_multiline_string(sandbox):
    """execute_in_sandbox handles multiline strings."""
    code = '''
text = """
Line 1
Line 2
Line 3
"""
print(text)
'''
    result = sandbox.execute_in_sandbox(code)

    assert "Line 1" in result.stdout
    assert "Line 2" in result.stdout
    assert "Line 3" in result.stdout


def test_execute_code_with_imports(sandbox):
    """execute_in_sandbox allows standard library imports."""
    code = """
import json
import os
import sys

data = {"key": "value"}
print(json.dumps(data))
"""
    result = sandbox.execute_in_sandbox(code)

    assert '{"key": "value"}' in result.stdout
    assert result.exit_code == 0


def test_execute_code_importing_missing_module(sandbox):
    """execute_in_sandbox handles missing module imports."""
    code = """
import nonexistent_module
"""
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code != 0
    assert "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr


def test_execute_very_long_output(sandbox):
    """execute_in_sandbox captures very long output."""
    code = """
for i in range(1000):
    print(f"Line {i}")
"""
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code == 0
    assert "Line 0" in result.stdout
    assert "Line 999" in result.stdout


def test_execute_infinite_loop_times_out(sandbox):
    """execute_in_sandbox times out infinite loops."""
    code = """
while True:
    pass
"""
    result = sandbox.execute_in_sandbox(code, timeout=1)

    assert result.timed_out is True
    assert result.exit_code == -1


def test_execute_subprocess_exception_handling(sandbox):
    """execute_in_sandbox handles subprocess exceptions."""
    # This test ensures general exception handling works
    # by passing valid code that should execute normally
    code = "print('test')"
    result = sandbox.execute_in_sandbox(code)

    assert result.exit_code == 0
    assert isinstance(result, SandboxResult)


def test_sandbox_path_contains_sandbox_id(sandbox):
    """Sandbox path includes sandbox ID for easy identification."""
    ctx = sandbox.create_sandbox()

    assert ctx.id in ctx.path
    assert "kintsugi_sandbox_" in ctx.path


def test_multiple_sandbox_instances_independent():
    """Multiple ShadowSandbox instances are independent."""
    sb1 = ShadowSandbox()
    sb2 = ShadowSandbox()

    ctx1 = sb1.create_sandbox()
    ctx2 = sb2.create_sandbox()

    assert ctx1.id not in sb2._sandboxes
    assert ctx2.id not in sb1._sandboxes

    sb1.cleanup_all()
    sb2.cleanup_all()


def test_execute_with_custom_timeout_zero(sandbox):
    """execute_in_sandbox with timeout=0 should timeout immediately."""
    code = "print('test')"
    result = sandbox.execute_in_sandbox(code, timeout=0)

    # timeout=0 should cause immediate timeout
    assert result.timed_out is True


def test_cleanup_with_files_in_sandbox(sandbox):
    """cleanup removes sandbox even with files inside."""
    ctx = sandbox.create_sandbox()

    # Create some files
    code = """
with open('file1.txt', 'w') as f:
    f.write('data')
with open('file2.txt', 'w') as f:
    f.write('more data')
"""
    sandbox.execute_in_sandbox(code, sandbox_id=ctx.id)

    # Cleanup should remove everything
    sandbox.cleanup(ctx.id)

    assert not os.path.exists(ctx.path)


def test_execute_handles_subprocess_general_exception(sandbox):
    """execute_in_sandbox handles general exceptions in subprocess.run."""
    import unittest.mock as mock

    # Mock subprocess.run to raise a general exception
    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = RuntimeError("Simulated subprocess error")

        result = sandbox.execute_in_sandbox("print('test')")

        assert result.exit_code == -1
        assert "Simulated subprocess error" in result.stderr
        assert result.stdout == ""
        assert result.timed_out is False
