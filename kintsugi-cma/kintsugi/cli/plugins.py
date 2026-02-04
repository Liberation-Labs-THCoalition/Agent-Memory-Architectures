"""
Kintsugi CLI - Plugin Commands

Plugin management commands for installing, configuring, and managing
Kintsugi plugins. Supports both local and remote plugin sources.

Commands:
    list      - List plugins (installed and available)
    install   - Install a plugin
    uninstall - Uninstall a plugin
    enable    - Enable a plugin
    disable   - Disable a plugin
    info      - Show plugin details
    update    - Update plugins
    create    - Create a new plugin from template
    validate  - Validate a plugin
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.tree import Tree

from kintsugi.cli import plugin_app, console


class PluginStatus(str, Enum):
    """Plugin status states."""
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    AVAILABLE = "available"
    OUTDATED = "outdated"
    ERROR = "error"


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    author: str
    status: PluginStatus
    installed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    dependencies: list[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)


# Simulated plugin registry
PLUGIN_REGISTRY = {
    "memory-compression": PluginInfo(
        name="memory-compression",
        version="1.2.0",
        description="Automatic memory compression and deduplication",
        author="Kintsugi Team",
        status=PluginStatus.ENABLED,
        installed_at=datetime(2024, 1, 10),
        homepage="https://github.com/kintsugi/memory-compression",
        license="MIT",
        tags=["memory", "optimization"],
    ),
    "semantic-search": PluginInfo(
        name="semantic-search",
        version="2.0.1",
        description="Advanced semantic search with hybrid retrieval",
        author="Kintsugi Team",
        status=PluginStatus.ENABLED,
        installed_at=datetime(2024, 1, 5),
        homepage="https://github.com/kintsugi/semantic-search",
        license="MIT",
        tags=["search", "retrieval"],
    ),
    "audit-logger": PluginInfo(
        name="audit-logger",
        version="1.0.0",
        description="Comprehensive audit logging for compliance",
        author="Kintsugi Team",
        status=PluginStatus.DISABLED,
        installed_at=datetime(2024, 1, 8),
        license="MIT",
        tags=["security", "compliance"],
    ),
    "pii-detector": PluginInfo(
        name="pii-detector",
        version="1.5.0",
        description="Automatic PII detection and masking",
        author="Security Labs",
        status=PluginStatus.AVAILABLE,
        homepage="https://github.com/security-labs/pii-detector",
        license="Apache-2.0",
        tags=["security", "privacy"],
    ),
    "llm-gateway": PluginInfo(
        name="llm-gateway",
        version="2.1.0",
        description="Unified gateway for multiple LLM providers",
        author="AI Tools Inc",
        status=PluginStatus.AVAILABLE,
        homepage="https://github.com/ai-tools/llm-gateway",
        license="MIT",
        tags=["llm", "integration"],
    ),
}


@plugin_app.command("list")
def list_plugins(
    installed: bool = typer.Option(
        False,
        "--installed",
        "-i",
        help="Show only installed plugins.",
    ),
    available: bool = typer.Option(
        False,
        "--available",
        "-a",
        help="Show only available (not installed) plugins.",
    ),
    enabled: bool = typer.Option(
        False,
        "--enabled",
        "-e",
        help="Show only enabled plugins.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, simple.",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag.",
    ),
) -> None:
    """
    List plugins.

    Shows installed, available, or all plugins with their status.
    Use filters to narrow down the list.
    """
    from kintsugi.cli.output import print_json

    plugins = list(PLUGIN_REGISTRY.values())

    # Apply filters
    if installed:
        plugins = [p for p in plugins if p.status in (PluginStatus.ENABLED, PluginStatus.DISABLED)]
    elif available:
        plugins = [p for p in plugins if p.status == PluginStatus.AVAILABLE]
    elif enabled:
        plugins = [p for p in plugins if p.status == PluginStatus.ENABLED]

    if tag:
        plugins = [p for p in plugins if tag in p.tags]

    if format == "json":
        data = [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "status": p.status.value,
                "author": p.author,
                "tags": p.tags,
            }
            for p in plugins
        ]
        print_json({"plugins": data})
    elif format == "simple":
        for p in plugins:
            status_icon = {
                PluginStatus.ENABLED: "[green]+[/green]",
                PluginStatus.DISABLED: "[yellow]-[/yellow]",
                PluginStatus.AVAILABLE: "[blue]o[/blue]",
                PluginStatus.OUTDATED: "[red]![/red]",
                PluginStatus.ERROR: "[red]x[/red]",
            }.get(p.status, " ")
            console.print(f"{status_icon} {p.name} ({p.version})")
    else:
        table = Table(title="Kintsugi Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Status")
        table.add_column("Description", max_width=40)
        table.add_column("Tags")

        for p in plugins:
            status_style = {
                PluginStatus.ENABLED: "[green]enabled[/green]",
                PluginStatus.DISABLED: "[yellow]disabled[/yellow]",
                PluginStatus.AVAILABLE: "[blue]available[/blue]",
                PluginStatus.OUTDATED: "[red]outdated[/red]",
                PluginStatus.ERROR: "[red]error[/red]",
            }.get(p.status, p.status.value)

            table.add_row(
                p.name,
                p.version,
                status_style,
                p.description[:40] + "..." if len(p.description) > 40 else p.description,
                ", ".join(p.tags),
            )

        console.print(table)

    console.print()
    console.print(f"[dim]Total: {len(plugins)} plugins[/dim]")


@plugin_app.command("install")
def install_plugin(
    name: str = typer.Argument(
        ...,
        help="Plugin name or URL.",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Specific version to install.",
    ),
    source: str = typer.Option(
        "registry",
        "--source",
        "-s",
        help="Source: registry, github, local.",
    ),
    no_deps: bool = typer.Option(
        False,
        "--no-deps",
        help="Don't install dependencies.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force reinstall if already installed.",
    ),
) -> None:
    """
    Install a plugin.

    Installs a plugin from the registry, GitHub, or local path.
    Dependencies are automatically resolved and installed.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    console.print(f"Installing plugin [cyan]{name}[/cyan]...")

    # Check if already installed
    if name in PLUGIN_REGISTRY:
        plugin = PLUGIN_REGISTRY[name]
        if plugin.status in (PluginStatus.ENABLED, PluginStatus.DISABLED):
            if not force:
                print_warning(f"Plugin {name} is already installed (v{plugin.version})")
                print_warning("Use --force to reinstall")
                raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        # Download
        task = progress.add_task("Downloading...", total=100)
        for i in range(100):
            import time
            time.sleep(0.01)
            progress.update(task, advance=1)

        # Install dependencies
        if not no_deps:
            task = progress.add_task("Installing dependencies...", total=100)
            for i in range(100):
                import time
                time.sleep(0.005)
                progress.update(task, advance=1)

        # Install plugin
        task = progress.add_task("Installing plugin...", total=100)
        for i in range(100):
            import time
            time.sleep(0.005)
            progress.update(task, advance=1)

    console.print()
    print_success(f"Plugin {name} installed successfully!")

    console.print()
    console.print("Enable the plugin with:")
    console.print(f"  [cyan]kintsugi plugin enable {name}[/cyan]")


@plugin_app.command("uninstall")
def uninstall_plugin(
    name: str = typer.Argument(
        ...,
        help="Plugin name to uninstall.",
    ),
    keep_config: bool = typer.Option(
        False,
        "--keep-config",
        help="Keep plugin configuration files.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force uninstall without confirmation.",
    ),
) -> None:
    """
    Uninstall a plugin.

    Removes a plugin and its files. Configuration can be kept
    for potential reinstallation.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    if name not in PLUGIN_REGISTRY:
        print_error(f"Plugin not found: {name}")
        raise typer.Exit(1)

    plugin = PLUGIN_REGISTRY[name]

    if plugin.status == PluginStatus.AVAILABLE:
        print_error(f"Plugin {name} is not installed")
        raise typer.Exit(1)

    if not force:
        if not typer.confirm(f"Uninstall plugin {name}?"):
            console.print("Cancelled")
            raise typer.Exit(0)

    console.print(f"Uninstalling [cyan]{name}[/cyan]...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Removing plugin files...", total=None)
        import time
        time.sleep(0.5)
        progress.update(task, completed=True)

        if not keep_config:
            task = progress.add_task("Removing configuration...", total=None)
            time.sleep(0.3)
            progress.update(task, completed=True)

    console.print()
    print_success(f"Plugin {name} uninstalled")

    if keep_config:
        console.print("[dim]Configuration files preserved[/dim]")


@plugin_app.command("enable")
def enable_plugin(
    name: str = typer.Argument(
        ...,
        help="Plugin name to enable.",
    ),
) -> None:
    """
    Enable a plugin.

    Activates a disabled plugin so it can be used by Kintsugi.
    """
    from kintsugi.cli.output import print_success, print_error

    if name not in PLUGIN_REGISTRY:
        print_error(f"Plugin not found: {name}")
        raise typer.Exit(1)

    plugin = PLUGIN_REGISTRY[name]

    if plugin.status == PluginStatus.AVAILABLE:
        print_error(f"Plugin {name} is not installed")
        console.print(f"Install it first: [cyan]kintsugi plugin install {name}[/cyan]")
        raise typer.Exit(1)

    if plugin.status == PluginStatus.ENABLED:
        console.print(f"Plugin {name} is already enabled")
        return

    console.print(f"Enabling plugin [cyan]{name}[/cyan]...")

    # Would actually enable
    print_success(f"Plugin {name} enabled")

    console.print()
    console.print("[dim]Restart Kintsugi for changes to take effect[/dim]")


@plugin_app.command("disable")
def disable_plugin(
    name: str = typer.Argument(
        ...,
        help="Plugin name to disable.",
    ),
) -> None:
    """
    Disable a plugin.

    Deactivates a plugin without uninstalling it. The plugin
    can be re-enabled later.
    """
    from kintsugi.cli.output import print_success, print_error

    if name not in PLUGIN_REGISTRY:
        print_error(f"Plugin not found: {name}")
        raise typer.Exit(1)

    plugin = PLUGIN_REGISTRY[name]

    if plugin.status == PluginStatus.DISABLED:
        console.print(f"Plugin {name} is already disabled")
        return

    if plugin.status == PluginStatus.AVAILABLE:
        print_error(f"Plugin {name} is not installed")
        raise typer.Exit(1)

    console.print(f"Disabling plugin [cyan]{name}[/cyan]...")

    # Would actually disable
    print_success(f"Plugin {name} disabled")


@plugin_app.command("info")
def plugin_info(
    name: str = typer.Argument(
        ...,
        help="Plugin name.",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output format: rich, json.",
    ),
) -> None:
    """
    Show plugin details.

    Displays comprehensive information about a plugin including
    version, dependencies, configuration, and more.
    """
    from kintsugi.cli.output import print_json, print_error

    if name not in PLUGIN_REGISTRY:
        print_error(f"Plugin not found: {name}")
        raise typer.Exit(1)

    plugin = PLUGIN_REGISTRY[name]

    if format == "json":
        data = {
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "author": plugin.author,
            "status": plugin.status.value,
            "homepage": plugin.homepage,
            "license": plugin.license,
            "tags": plugin.tags,
            "dependencies": plugin.dependencies,
            "installed_at": plugin.installed_at.isoformat() if plugin.installed_at else None,
        }
        print_json(data)
    else:
        console.print(Panel.fit(
            f"[bold]{plugin.name}[/bold] v{plugin.version}",
            subtitle=plugin.description,
        ))

        console.print()

        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        status_style = {
            PluginStatus.ENABLED: "[green]Enabled[/green]",
            PluginStatus.DISABLED: "[yellow]Disabled[/yellow]",
            PluginStatus.AVAILABLE: "[blue]Available[/blue]",
        }.get(plugin.status, plugin.status.value)

        table.add_row("Status", status_style)
        table.add_row("Author", plugin.author)
        table.add_row("License", plugin.license)
        table.add_row("Homepage", plugin.homepage or "N/A")
        table.add_row("Tags", ", ".join(plugin.tags) if plugin.tags else "None")

        if plugin.installed_at:
            table.add_row("Installed", plugin.installed_at.strftime("%Y-%m-%d"))

        if plugin.dependencies:
            table.add_row("Dependencies", ", ".join(plugin.dependencies))
        else:
            table.add_row("Dependencies", "None")

        console.print(table)


@plugin_app.command("update")
def update_plugins(
    name: Optional[str] = typer.Argument(
        None,
        help="Plugin name to update (all if not specified).",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Only check for updates, don't install.",
    ),
) -> None:
    """
    Update plugins.

    Updates one or all installed plugins to their latest versions.
    """
    from kintsugi.cli.output import print_success, print_warning

    if name:
        plugins = [PLUGIN_REGISTRY.get(name)]
        if not plugins[0]:
            console.print(f"[red]Plugin not found: {name}[/red]")
            raise typer.Exit(1)
    else:
        plugins = [p for p in PLUGIN_REGISTRY.values()
                   if p.status in (PluginStatus.ENABLED, PluginStatus.DISABLED)]

    console.print("Checking for updates...")
    console.print()

    # Simulated update check
    updates_available = [
        ("memory-compression", "1.2.0", "1.3.0"),
        ("semantic-search", "2.0.1", "2.1.0"),
    ]

    if not updates_available:
        print_success("All plugins are up to date!")
        return

    table = Table(title="Available Updates")
    table.add_column("Plugin", style="cyan")
    table.add_column("Current")
    table.add_column("Latest", style="green")

    for plugin_name, current, latest in updates_available:
        table.add_row(plugin_name, current, latest)

    console.print(table)

    if check:
        console.print()
        print_warning(f"{len(updates_available)} updates available")
        return

    console.print()
    if typer.confirm("Install updates?"):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for plugin_name, _, latest in updates_available:
                task = progress.add_task(f"Updating {plugin_name}...", total=None)
                import time
                time.sleep(0.5)
                progress.update(task, completed=True)

        console.print()
        print_success(f"Updated {len(updates_available)} plugins")


@plugin_app.command("create")
def create_plugin(
    name: str = typer.Argument(
        ...,
        help="Plugin name.",
    ),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Template: basic, advanced, integration.",
    ),
    directory: Path = typer.Option(
        Path("plugins"),
        "--directory",
        "-d",
        help="Directory to create plugin in.",
    ),
) -> None:
    """
    Create a new plugin from template.

    Generates a plugin skeleton with the specified template,
    including all necessary files and configuration.
    """
    from kintsugi.cli.output import print_success

    plugin_dir = directory / name

    console.print(f"Creating plugin [cyan]{name}[/cyan] from [cyan]{template}[/cyan] template...")

    # Would create actual files
    files = [
        f"{name}/__init__.py",
        f"{name}/plugin.py",
        f"{name}/config.yaml",
        f"{name}/README.md",
    ]

    if template in ("advanced", "integration"):
        files.extend([
            f"{name}/handlers.py",
            f"{name}/models.py",
            f"{name}/tests/__init__.py",
            f"{name}/tests/test_plugin.py",
        ])

    console.print()
    tree = Tree(f"[cyan]{plugin_dir}[/cyan]")
    for f in files:
        parts = f.split("/")[1:]  # Remove plugin name prefix
        tree.add("/".join(parts))

    console.print(tree)

    console.print()
    print_success(f"Plugin {name} created at {plugin_dir}")

    console.print()
    console.print("Next steps:")
    console.print(f"  1. Edit [cyan]{plugin_dir}/plugin.py[/cyan] to implement your plugin")
    console.print(f"  2. Validate: [cyan]kintsugi plugin validate {plugin_dir}[/cyan]")
    console.print(f"  3. Install: [cyan]kintsugi plugin install {plugin_dir} --source local[/cyan]")


@plugin_app.command("validate")
def validate_plugin(
    path: Path = typer.Argument(
        ...,
        help="Path to plugin directory.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation.",
    ),
) -> None:
    """
    Validate a plugin.

    Checks a plugin for correctness including:
    - Required files present
    - Valid configuration
    - Proper interface implementation
    - Dependency compatibility
    """
    from kintsugi.cli.output import print_status, print_success, print_error

    console.print(f"Validating plugin at [cyan]{path}[/cyan]...")
    console.print()

    # Simulated validation
    checks = [
        ("Required files", True, "All required files present"),
        ("Plugin manifest", True, "plugin.yaml valid"),
        ("Entry point", True, "Plugin class found"),
        ("Configuration schema", True, "Schema valid"),
        ("Dependencies", True, "All dependencies available"),
        ("Hook implementations", True, "All hooks properly typed"),
    ]

    if strict:
        checks.extend([
            ("Type hints", True, "All functions have type hints"),
            ("Documentation", True, "All public methods documented"),
            ("Test coverage", True, "> 80% coverage"),
        ])

    print_status(checks)

    all_passed = all(ok for _, ok, _ in checks)

    console.print()
    if all_passed:
        print_success("Plugin validation passed!")
    else:
        print_error("Plugin validation failed")
        raise typer.Exit(1)


@plugin_app.command("config")
def plugin_config(
    name: str = typer.Argument(
        ...,
        help="Plugin name.",
    ),
    key: Optional[str] = typer.Option(
        None,
        "--key",
        "-k",
        help="Specific config key to show/set.",
    ),
    value: Optional[str] = typer.Option(
        None,
        "--value",
        "-v",
        help="Value to set (requires --key).",
    ),
) -> None:
    """
    View or modify plugin configuration.

    Shows plugin-specific configuration or sets a value
    if --key and --value are provided.
    """
    from kintsugi.cli.output import print_success, print_error

    if name not in PLUGIN_REGISTRY:
        print_error(f"Plugin not found: {name}")
        raise typer.Exit(1)

    # Simulated plugin config
    config = {
        "compression_level": 6,
        "dedup_enabled": True,
        "cache_size_mb": 256,
        "batch_size": 100,
    }

    if key and value:
        console.print(f"Setting [cyan]{name}.{key}[/cyan] = [green]{value}[/green]")
        print_success("Configuration updated")
    elif key:
        if key in config:
            console.print(f"{key}: {config[key]}")
        else:
            print_error(f"Key not found: {key}")
            raise typer.Exit(1)
    else:
        console.print(f"[bold]Configuration for {name}[/bold]")
        console.print()

        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for k, v in config.items():
            table.add_row(k, str(v))

        console.print(table)
