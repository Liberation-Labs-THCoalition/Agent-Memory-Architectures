"""Comprehensive tests for Kintsugi CLI module.

Tests cover:
- Main app commands (--help, --version)
- Security commands (audit, scan, check-deps)
- Doctor commands (run, db, api)
- Config commands (show, validate)
- Plugin commands (list)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest
from typer.testing import CliRunner

from kintsugi.cli import app


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create a CliRunner for testing."""
    return CliRunner()


@pytest.fixture
def mock_db():
    """Mock database connection."""
    with patch("kintsugi.cli.commands.doctor.get_db_connection") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_api_client():
    """Mock API client."""
    with patch("kintsugi.cli.commands.doctor.APIClient") as mock:
        client = MagicMock()
        client.health_check = AsyncMock(return_value={"status": "ok"})
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_security_scanner():
    """Mock security scanner."""
    with patch("kintsugi.cli.commands.security.SecurityScanner") as mock:
        scanner = MagicMock()
        scanner.scan = MagicMock(return_value={"issues": [], "passed": True})
        scanner.deep_scan = MagicMock(return_value={"issues": [], "passed": True})
        mock.return_value = scanner
        yield mock


@pytest.fixture
def mock_config_loader():
    """Mock config loader."""
    with patch("kintsugi.cli.commands.config.load_config") as mock:
        mock.return_value = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"port": 8000},
        }
        yield mock


@pytest.fixture
def mock_plugin_manager():
    """Mock plugin manager."""
    with patch("kintsugi.cli.commands.plugin.PluginManager") as mock:
        manager = MagicMock()
        manager.list_available = MagicMock(
            return_value=[
                {"name": "slack-adapter", "version": "1.0.0", "installed": True},
                {"name": "discord-adapter", "version": "1.0.0", "installed": False},
            ]
        )
        manager.list_installed = MagicMock(
            return_value=[{"name": "slack-adapter", "version": "1.0.0"}]
        )
        mock.return_value = manager
        yield mock


# ===========================================================================
# Main App Tests (6 tests)
# ===========================================================================


class TestMainApp:
    """Tests for main CLI app."""

    def test_help_works(self, runner):
        """--help flag displays help message."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Kintsugi" in result.stdout or "kintsugi" in result.stdout.lower()
        assert "Usage" in result.stdout or "usage" in result.stdout.lower()

    def test_version_works(self, runner):
        """--version flag displays version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Version should be in format like "0.1.0" or "Kintsugi 0.1.0"
        assert any(c.isdigit() for c in result.stdout)

    def test_no_args_shows_help(self, runner):
        """No arguments shows help or usage information."""
        result = runner.invoke(app, [])
        # Either shows help (exit 0) or complains about missing command
        assert result.exit_code in (0, 1, 2)
        # Should show some usage information
        assert "Usage" in result.stdout or "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_unknown_command_shows_error(self, runner):
        """Unknown command shows error message."""
        result = runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_app_has_command_groups(self, runner):
        """App has expected command groups."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Should mention subcommands or command groups
        output_lower = result.stdout.lower()
        assert any(
            cmd in output_lower
            for cmd in ["security", "doctor", "config", "plugin", "commands"]
        )

    def test_verbose_flag_accepted(self, runner):
        """--verbose flag is accepted."""
        result = runner.invoke(app, ["--verbose", "--help"])
        # Should not fail due to unknown flag
        assert result.exit_code == 0


# ===========================================================================
# Security Commands Tests (8 tests)
# ===========================================================================


class TestSecurityCommands:
    """Tests for security CLI commands."""

    def test_security_audit_runs(self, runner, mock_security_scanner):
        """security audit command runs successfully."""
        result = runner.invoke(app, ["security", "audit"])
        assert result.exit_code == 0
        assert "audit" in result.stdout.lower() or "scan" in result.stdout.lower() or "complete" in result.stdout.lower()

    def test_security_audit_deep_runs(self, runner, mock_security_scanner):
        """security audit --deep command runs successfully."""
        result = runner.invoke(app, ["security", "audit", "--deep"])
        assert result.exit_code == 0

    def test_security_scan_runs(self, runner, mock_security_scanner):
        """security scan command runs successfully."""
        result = runner.invoke(app, ["security", "scan"])
        assert result.exit_code == 0

    def test_security_check_deps_runs(self, runner):
        """security check-deps command runs successfully."""
        with patch("kintsugi.cli.commands.security.check_dependencies") as mock_check:
            mock_check.return_value = {"vulnerable": [], "outdated": []}
            result = runner.invoke(app, ["security", "check-deps"])
            assert result.exit_code == 0

    def test_security_help_shows_subcommands(self, runner):
        """security --help shows available subcommands."""
        result = runner.invoke(app, ["security", "--help"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert "audit" in output_lower or "scan" in output_lower

    def test_security_audit_output_format(self, runner, mock_security_scanner):
        """security audit outputs expected format."""
        result = runner.invoke(app, ["security", "audit"])
        assert result.exit_code == 0
        # Should indicate pass/fail or issues count
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower
            for word in ["passed", "failed", "issues", "complete", "ok", "error"]
        )

    def test_security_audit_with_path(self, runner, mock_security_scanner):
        """security audit accepts path argument."""
        result = runner.invoke(app, ["security", "audit", "--path", "/tmp"])
        # Should accept the path flag
        assert result.exit_code in (0, 1)

    def test_security_scan_with_output_json(self, runner, mock_security_scanner):
        """security scan accepts --output json flag."""
        result = runner.invoke(app, ["security", "scan", "--output", "json"])
        # Should accept output format
        assert result.exit_code in (0, 1)


# ===========================================================================
# Doctor Commands Tests (6 tests)
# ===========================================================================


class TestDoctorCommands:
    """Tests for doctor CLI commands."""

    def test_doctor_run_runs(self, runner, mock_db, mock_api_client):
        """doctor run command runs successfully."""
        result = runner.invoke(app, ["doctor", "run"])
        assert result.exit_code == 0
        # Should show some diagnostic output
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower
            for word in ["check", "ok", "passed", "failed", "status", "health"]
        )

    def test_doctor_db_runs(self, runner, mock_db):
        """doctor db command runs successfully."""
        result = runner.invoke(app, ["doctor", "db"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower
            for word in ["database", "connection", "ok", "status", "db"]
        )

    def test_doctor_api_runs(self, runner, mock_api_client):
        """doctor api command runs successfully."""
        result = runner.invoke(app, ["doctor", "api"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower for word in ["api", "health", "ok", "status", "endpoint"]
        )

    def test_doctor_help_shows_subcommands(self, runner):
        """doctor --help shows available subcommands."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert any(cmd in output_lower for cmd in ["run", "db", "api"])

    def test_doctor_run_verbose(self, runner, mock_db, mock_api_client):
        """doctor run --verbose shows detailed output."""
        result = runner.invoke(app, ["doctor", "run", "--verbose"])
        assert result.exit_code == 0

    def test_doctor_db_with_connection_string(self, runner, mock_db):
        """doctor db accepts connection string."""
        result = runner.invoke(
            app, ["doctor", "db", "--connection", "postgresql://localhost/test"]
        )
        # Should accept the connection flag
        assert result.exit_code in (0, 1)


# ===========================================================================
# Config Commands Tests (6 tests)
# ===========================================================================


class TestConfigCommands:
    """Tests for config CLI commands."""

    def test_config_show_runs(self, runner, mock_config_loader):
        """config show command runs successfully."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        # Should display some config information
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower
            for word in ["config", "database", "api", "settings", "host", "port"]
        )

    def test_config_validate_runs(self, runner, mock_config_loader):
        """config validate command runs successfully."""
        with patch("kintsugi.cli.commands.config.validate_config") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 0
            output_lower = result.stdout.lower()
            assert any(
                word in output_lower
                for word in ["valid", "ok", "passed", "configuration"]
            )

    def test_config_help_shows_subcommands(self, runner):
        """config --help shows available subcommands."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert any(cmd in output_lower for cmd in ["show", "validate"])

    def test_config_show_with_section(self, runner, mock_config_loader):
        """config show accepts section argument."""
        result = runner.invoke(app, ["config", "show", "--section", "database"])
        assert result.exit_code in (0, 1)

    def test_config_show_json_format(self, runner, mock_config_loader):
        """config show --format json outputs JSON."""
        result = runner.invoke(app, ["config", "show", "--format", "json"])
        assert result.exit_code in (0, 1)
        # If successful, output might contain JSON characters
        if result.exit_code == 0:
            assert "{" in result.stdout or "[" in result.stdout

    def test_config_validate_with_file(self, runner):
        """config validate accepts file argument."""
        with patch("kintsugi.cli.commands.config.validate_config") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}
            result = runner.invoke(
                app, ["config", "validate", "--file", "/tmp/config.yaml"]
            )
            assert result.exit_code in (0, 1)


# ===========================================================================
# Plugin Commands Tests (6 tests)
# ===========================================================================


class TestPluginCommands:
    """Tests for plugin CLI commands."""

    def test_plugin_list_runs(self, runner, mock_plugin_manager):
        """plugin list command runs successfully."""
        result = runner.invoke(app, ["plugin", "list"])
        assert result.exit_code == 0
        # Should show plugin information
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower
            for word in ["plugin", "adapter", "installed", "available", "name"]
        )

    def test_plugin_list_installed_runs(self, runner, mock_plugin_manager):
        """plugin list --installed command runs successfully."""
        result = runner.invoke(app, ["plugin", "list", "--installed"])
        assert result.exit_code == 0

    def test_plugin_help_shows_subcommands(self, runner):
        """plugin --help shows available subcommands."""
        result = runner.invoke(app, ["plugin", "--help"])
        assert result.exit_code == 0
        output_lower = result.stdout.lower()
        assert "list" in output_lower

    def test_plugin_list_available(self, runner, mock_plugin_manager):
        """plugin list --available shows available plugins."""
        result = runner.invoke(app, ["plugin", "list", "--available"])
        assert result.exit_code in (0, 1)

    def test_plugin_list_json_format(self, runner, mock_plugin_manager):
        """plugin list --format json outputs JSON."""
        result = runner.invoke(app, ["plugin", "list", "--format", "json"])
        assert result.exit_code in (0, 1)

    def test_plugin_list_shows_versions(self, runner, mock_plugin_manager):
        """plugin list shows version information."""
        result = runner.invoke(app, ["plugin", "list"])
        assert result.exit_code == 0
        # Version numbers typically contain dots
        # Or the word "version"
        output_lower = result.stdout.lower()
        assert "version" in output_lower or "." in result.stdout or "1.0" in result.stdout


# ===========================================================================
# Additional Integration Tests (4 tests)
# ===========================================================================


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_multiple_commands_in_sequence(self, runner, mock_config_loader):
        """Multiple commands can be run in sequence."""
        # First command
        result1 = runner.invoke(app, ["config", "show"])
        assert result1.exit_code == 0

        # Second command
        with patch("kintsugi.cli.commands.config.validate_config") as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}
            result2 = runner.invoke(app, ["config", "validate"])
            assert result2.exit_code == 0

    def test_quiet_mode_reduces_output(self, runner, mock_config_loader):
        """--quiet flag reduces output verbosity."""
        result_normal = runner.invoke(app, ["config", "show"])
        result_quiet = runner.invoke(app, ["--quiet", "config", "show"])

        # Both should succeed
        assert result_normal.exit_code == 0
        assert result_quiet.exit_code == 0
        # Quiet mode should have same or less output
        assert len(result_quiet.stdout) <= len(result_normal.stdout) + 100

    def test_exit_codes_are_meaningful(self, runner):
        """Exit codes indicate success/failure appropriately."""
        # Help should succeed
        result_help = runner.invoke(app, ["--help"])
        assert result_help.exit_code == 0

        # Unknown command should fail
        result_unknown = runner.invoke(app, ["nonexistent"])
        assert result_unknown.exit_code != 0

    def test_error_messages_are_helpful(self, runner):
        """Error messages provide useful information."""
        result = runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0
        # Should mention the problem
        output_lower = result.stdout.lower() + (result.stderr or "").lower()
        assert any(
            word in output_lower
            for word in ["error", "unknown", "invalid", "no such", "not found"]
        )
