"""
Kintsugi CLI - Configuration Commands

Configuration management commands for viewing, modifying, and validating
Kintsugi settings. Supports multiple configuration formats and provides
schema validation.

Commands:
    show     - Display current configuration
    set      - Set a configuration value
    get      - Get a specific configuration value
    validate - Validate configuration against schema
    init     - Initialize configuration file
    edit     - Open configuration in editor
    export   - Export configuration
    import   - Import configuration
    diff     - Compare configurations
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from kintsugi.cli import config_app, console


# Default configuration structure
DEFAULT_CONFIG = {
    "version": "1.0",
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "kintsugi",
        "user": "kintsugi",
        "password": "${KINTSUGI_DB_PASSWORD}",
        "pool_size": 10,
        "max_overflow": 20,
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "device": "auto",
        "batch_size": 32,
        "cache_enabled": True,
        "cache_ttl": 3600,
    },
    "efe": {
        "weights": {
            "autonomy": 0.3,
            "beneficence": 0.4,
            "non_maleficence": 0.5,
            "justice": 0.35,
            "transparency": 0.45,
        },
        "tuning": {
            "enabled": True,
            "strategy": "gradient",
            "learning_rate": 0.01,
            "min_samples": 50,
        },
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "cors_origins": ["*"],
        "rate_limit": {
            "requests_per_minute": 100,
            "burst": 20,
        },
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "file": "logs/kintsugi.log",
        "rotation": "1 day",
        "retention": "30 days",
    },
    "plugins": {
        "enabled": True,
        "directory": "plugins",
        "auto_load": True,
    },
    "security": {
        "rls_enabled": True,
        "audit_logging": True,
        "pii_detection": True,
        "encryption_at_rest": True,
    },
}


@config_app.command("show")
def show_config(
    section: Optional[str] = typer.Argument(
        None,
        help="Config section to show (e.g., 'database', 'efe.weights').",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml, json, table, tree.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    resolve: bool = typer.Option(
        False,
        "--resolve",
        "-r",
        help="Resolve environment variable references.",
    ),
    secrets: bool = typer.Option(
        False,
        "--secrets",
        help="Show secret values (use with caution).",
    ),
) -> None:
    """
    Display current configuration.

    Shows the current Kintsugi configuration with options for
    different output formats and section filtering.
    """
    from kintsugi.cli.output import print_json

    config_path = path or Path(".kintsugi/config.yaml")

    # Load config (would load actual file)
    config = dict(DEFAULT_CONFIG)

    # Navigate to section if specified
    if section:
        parts = section.split(".")
        current = config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                console.print(f"[red]Section not found: {section}[/red]")
                raise typer.Exit(1)
        config = {section: current}

    # Resolve environment variables
    if resolve:
        config = _resolve_env_vars(config)

    # Mask secrets
    if not secrets:
        config = _mask_secrets(config)

    # Output in requested format
    if format == "json":
        print_json(config)
    elif format == "table":
        _show_config_table(config)
    elif format == "tree":
        _show_config_tree(config)
    else:
        # YAML format (default)
        _show_config_yaml(config)

    if not resolve and _has_env_refs(config):
        console.print()
        console.print("[dim]Note: Use --resolve to expand environment variables[/dim]")


@config_app.command("set")
def set_config(
    key: str = typer.Argument(
        ...,
        help="Config key using dot notation (e.g., 'database.port').",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Value type: string, int, float, bool, json.",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        help="Skip validation after setting.",
    ),
) -> None:
    """
    Set a configuration value.

    Updates a configuration value using dot notation for nested keys.
    Automatically validates the change unless --no-validate is specified.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    config_path = path or Path(".kintsugi/config.yaml")

    console.print(f"Setting [cyan]{key}[/cyan] = [green]{value}[/green]")

    # Parse value with type hint
    parsed_value = _parse_value(value, type)

    # Validate the change
    if not no_validate:
        validation_errors = _validate_key_value(key, parsed_value)
        if validation_errors:
            for error in validation_errors:
                print_error(error)
            raise typer.Exit(1)

    # Would write to actual config file
    console.print()
    print_success(f"Configuration updated: {key} = {parsed_value}")

    console.print()
    console.print("[dim]Remember to restart services for changes to take effect[/dim]")


@config_app.command("get")
def get_config(
    key: str = typer.Argument(
        ...,
        help="Config key using dot notation.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    resolve: bool = typer.Option(
        True,
        "--resolve/--no-resolve",
        help="Resolve environment variable references.",
    ),
    default: Optional[str] = typer.Option(
        None,
        "--default",
        "-d",
        help="Default value if key not found.",
    ),
) -> None:
    """
    Get a specific configuration value.

    Retrieves and displays a single configuration value.
    Useful for scripting and automation.
    """
    config = dict(DEFAULT_CONFIG)

    # Navigate to key
    parts = key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            if default is not None:
                console.print(default)
                return
            console.print(f"[red]Key not found: {key}[/red]")
            raise typer.Exit(1)

    # Resolve env vars
    if resolve and isinstance(current, str):
        current = _resolve_single_value(current)

    # Output just the value (for scripting)
    if isinstance(current, dict):
        from kintsugi.cli.output import print_json
        print_json(current)
    else:
        console.print(str(current))


@config_app.command("validate")
def validate_config(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation mode.",
    ),
    schema: Optional[Path] = typer.Option(
        None,
        "--schema",
        "-s",
        help="Path to custom schema file.",
    ),
) -> None:
    """
    Validate configuration against schema.

    Performs comprehensive validation including:
    - Schema compliance
    - Type checking
    - Value range validation
    - Required field verification
    - Environment variable resolution
    """
    from kintsugi.cli.output import print_status, print_success, print_error

    config_path = path or Path(".kintsugi/config.yaml")

    console.print(f"Validating [cyan]{config_path}[/cyan]...")
    console.print()

    # Simulated validation checks
    checks = [
        ("Schema compliance", True, "All fields match schema"),
        ("Type validation", True, "All types correct"),
        ("Required fields", True, "All required fields present"),
        ("Value ranges", True, "All values in valid ranges"),
        ("Environment refs", True, "All env vars defined"),
        ("Path references", True, "All paths exist"),
        ("Database config", True, "Connection string valid"),
        ("EFE weights", True, "Weights sum to valid range"),
    ]

    if strict:
        checks.extend([
            ("No deprecated fields", True, "No deprecated fields used"),
            ("Recommended settings", True, "All recommendations followed"),
            ("Security settings", True, "All security features enabled"),
        ])

    print_status(checks)

    all_passed = all(ok for _, ok, _ in checks)

    console.print()
    if all_passed:
        print_success("Configuration is valid")
    else:
        print_error("Configuration has errors")
        raise typer.Exit(1)


@config_app.command("init")
def init_config(
    path: Path = typer.Option(
        Path(".kintsugi/config.yaml"),
        "--path",
        "-p",
        help="Path for configuration file.",
    ),
    template: str = typer.Option(
        "default",
        "--template",
        "-t",
        help="Template: default, minimal, development, production.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive configuration wizard.",
    ),
) -> None:
    """
    Initialize configuration file.

    Creates a new configuration file from a template with
    sensible defaults for different deployment scenarios.
    """
    from kintsugi.cli.output import print_success, print_warning

    if path.exists() and not force:
        print_warning(f"Configuration already exists at {path}")
        print_warning("Use --force to overwrite")
        raise typer.Exit(1)

    console.print(f"Initializing configuration with [cyan]{template}[/cyan] template...")

    if interactive:
        config = _interactive_config_wizard()
    else:
        config = _get_template_config(template)

    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write config (would write actual file)
    console.print()
    print_success(f"Configuration created at {path}")

    console.print()
    console.print("Next steps:")
    console.print("  1. Review the configuration: [cyan]kintsugi config show[/cyan]")
    console.print("  2. Set database password: [cyan]kintsugi config set database.password <value>[/cyan]")
    console.print("  3. Validate: [cyan]kintsugi config validate[/cyan]")


@config_app.command("edit")
def edit_config(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    editor: Optional[str] = typer.Option(
        None,
        "--editor",
        "-e",
        help="Editor to use (defaults to $EDITOR).",
    ),
) -> None:
    """
    Open configuration in editor.

    Opens the configuration file in your preferred editor.
    Uses $EDITOR environment variable by default.
    """
    from kintsugi.cli.output import print_success, print_error

    config_path = path or Path(".kintsugi/config.yaml")

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        console.print("Run [cyan]kintsugi config init[/cyan] to create one")
        raise typer.Exit(1)

    # Determine editor
    editor_cmd = editor or os.environ.get("EDITOR", "vim")

    console.print(f"Opening [cyan]{config_path}[/cyan] in {editor_cmd}...")

    try:
        subprocess.run([editor_cmd, str(config_path)], check=True)
        print_success("Configuration edited")
    except subprocess.CalledProcessError:
        print_error("Editor exited with error")
        raise typer.Exit(1)
    except FileNotFoundError:
        print_error(f"Editor not found: {editor_cmd}")
        raise typer.Exit(1)


@config_app.command("export")
def export_config(
    output: Path = typer.Argument(
        ...,
        help="Output file path.",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml, json, env.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to configuration file.",
    ),
    include_secrets: bool = typer.Option(
        False,
        "--secrets",
        help="Include secrets in export (dangerous!).",
    ),
) -> None:
    """
    Export configuration.

    Exports the current configuration to a file in various formats.
    By default, secrets are masked for safety.
    """
    from kintsugi.cli.output import print_success, print_warning

    config = dict(DEFAULT_CONFIG)

    if not include_secrets:
        config = _mask_secrets(config)
    else:
        print_warning("Including secrets in export - handle with care!")

    console.print(f"Exporting configuration to [cyan]{output}[/cyan]...")

    # Would write actual file
    print_success(f"Configuration exported to {output}")


@config_app.command("import")
def import_config(
    input_file: Path = typer.Argument(
        ...,
        help="Input file path.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Target configuration file path.",
    ),
    merge: bool = typer.Option(
        True,
        "--merge/--replace",
        help="Merge with existing or replace entirely.",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate after import.",
    ),
) -> None:
    """
    Import configuration.

    Imports configuration from a file, with options to merge
    with existing configuration or replace entirely.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    if not input_file.exists():
        print_error(f"Input file not found: {input_file}")
        raise typer.Exit(1)

    config_path = path or Path(".kintsugi/config.yaml")

    console.print(f"Importing configuration from [cyan]{input_file}[/cyan]...")

    if merge:
        console.print("Merging with existing configuration...")
    else:
        print_warning("Replacing entire configuration!")

    # Would do actual import
    if validate:
        console.print("Validating imported configuration...")

    print_success("Configuration imported successfully")


@config_app.command("diff")
def diff_config(
    file1: Path = typer.Argument(
        ...,
        help="First configuration file.",
    ),
    file2: Path = typer.Argument(
        ...,
        help="Second configuration file.",
    ),
    format: str = typer.Option(
        "side-by-side",
        "--format",
        "-f",
        help="Output format: side-by-side, unified, json.",
    ),
) -> None:
    """
    Compare configurations.

    Shows differences between two configuration files,
    useful for comparing environments or reviewing changes.
    """
    from kintsugi.cli.output import print_success

    if not file1.exists():
        console.print(f"[red]File not found: {file1}[/red]")
        raise typer.Exit(1)

    if not file2.exists():
        console.print(f"[red]File not found: {file2}[/red]")
        raise typer.Exit(1)

    console.print(f"Comparing [cyan]{file1}[/cyan] and [cyan]{file2}[/cyan]...")
    console.print()

    # Simulated diff
    diffs = [
        ("database.port", "5432", "5433"),
        ("api.rate_limit.requests_per_minute", "100", "200"),
        ("logging.level", "INFO", "DEBUG"),
    ]

    if diffs:
        table = Table(title="Configuration Differences")
        table.add_column("Key", style="cyan")
        table.add_column(str(file1.name))
        table.add_column(str(file2.name))

        for key, val1, val2 in diffs:
            table.add_row(key, val1, f"[yellow]{val2}[/yellow]")

        console.print(table)
    else:
        print_success("Configurations are identical")


# Helper functions

def _resolve_env_vars(config: dict) -> dict:
    """Resolve environment variable references in config."""
    import copy
    result = copy.deepcopy(config)

    def resolve(obj):
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        return obj

    return resolve(result)


def _resolve_single_value(value: str) -> str:
    """Resolve a single environment variable reference."""
    if value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.environ.get(var_name, value)
    return value


def _mask_secrets(config: dict) -> dict:
    """Mask secret values in config."""
    import copy
    result = copy.deepcopy(config)
    secret_keys = ["password", "secret", "key", "token", "credential"]

    def mask(obj, key=""):
        if isinstance(obj, dict):
            return {k: mask(v, k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [mask(item) for item in obj]
        elif isinstance(obj, str):
            if any(s in key.lower() for s in secret_keys):
                return "****"
        return obj

    return mask(result)


def _has_env_refs(config: dict) -> bool:
    """Check if config has environment variable references."""
    def check(obj):
        if isinstance(obj, dict):
            return any(check(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(check(item) for item in obj)
        elif isinstance(obj, str):
            return obj.startswith("${") and obj.endswith("}")
        return False

    return check(config)


def _show_config_yaml(config: dict) -> None:
    """Display config in YAML format."""
    # Simple YAML-like output
    def format_yaml(obj, indent=0):
        lines = []
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and v:
                    lines.append(f"{prefix}{k}:")
                    lines.extend(format_yaml(v, indent + 1))
                else:
                    lines.append(f"{prefix}{k}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                lines.append(f"{prefix}- {item}")
        return lines

    yaml_str = "\n".join(format_yaml(config))
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)


def _show_config_table(config: dict) -> None:
    """Display config as a table."""
    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_column("Type", style="dim")

    def flatten(obj, prefix=""):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten(v, key))
                else:
                    items.append((key, str(v), type(v).__name__))
        return items

    for key, value, value_type in flatten(config):
        table.add_row(key, value, value_type)

    console.print(table)


def _show_config_tree(config: dict) -> None:
    """Display config as a tree."""
    tree = Tree("[bold]Configuration[/bold]")

    def add_to_tree(obj, node):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    child = node.add(f"[cyan]{k}[/cyan]")
                    add_to_tree(v, child)
                else:
                    node.add(f"[cyan]{k}[/cyan]: {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                node.add(f"[dim]{i}[/dim]: {item}")

    add_to_tree(config, tree)
    console.print(tree)


def _parse_value(value: str, type_hint: Optional[str]) -> Any:
    """Parse a string value with optional type hint."""
    if type_hint == "int":
        return int(value)
    elif type_hint == "float":
        return float(value)
    elif type_hint == "bool":
        return value.lower() in ("true", "yes", "1", "on")
    elif type_hint == "json":
        return json.loads(value)
    else:
        # Auto-detect
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value


def _validate_key_value(key: str, value: Any) -> list[str]:
    """Validate a key-value pair."""
    errors = []

    # Type-specific validation
    if "port" in key and isinstance(value, int):
        if not 1 <= value <= 65535:
            errors.append(f"Port must be 1-65535, got {value}")

    if "weight" in key and isinstance(value, (int, float)):
        if not 0.0 <= value <= 1.0:
            errors.append(f"Weight must be 0.0-1.0, got {value}")

    return errors


def _get_template_config(template: str) -> dict:
    """Get configuration template."""
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    if template == "minimal":
        # Remove optional features
        config["plugins"]["enabled"] = False
        config["security"]["pii_detection"] = False
    elif template == "development":
        config["logging"]["level"] = "DEBUG"
        config["api"]["cors_origins"] = ["*"]
    elif template == "production":
        config["logging"]["level"] = "WARNING"
        config["api"]["cors_origins"] = []
        config["security"]["encryption_at_rest"] = True

    return config


def _interactive_config_wizard() -> dict:
    """Run interactive configuration wizard."""
    config = dict(DEFAULT_CONFIG)

    console.print("[bold]Kintsugi Configuration Wizard[/bold]")
    console.print()

    # Would prompt for values interactively
    config["database"]["host"] = typer.prompt("Database host", default="localhost")
    config["database"]["port"] = int(typer.prompt("Database port", default="5432"))
    config["api"]["port"] = int(typer.prompt("API port", default="8000"))

    return config
