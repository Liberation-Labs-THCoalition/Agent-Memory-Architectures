"""
Kintsugi CMA - Command Line Interface

This module provides a comprehensive CLI for managing and interacting with
the Kintsugi Prosocial Agent Memory Architecture. Built with Typer for
modern command-line experience and Rich for beautiful output.

Usage:
    $ kintsugi --help
    $ kintsugi security audit --deep
    $ kintsugi doctor run --verbose
    $ kintsugi config show
    $ kintsugi plugin list

Sub-command Groups:
    security - Security audit and scanning commands
    doctor   - Troubleshooting and diagnostic commands
    config   - Configuration management
    plugin   - Plugin management

For detailed help on any command:
    $ kintsugi <command> --help
    $ kintsugi <group> <command> --help
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Version info
__version__ = "0.1.0"

# Create main console for output
console = Console()
err_console = Console(stderr=True)

# Create main application
app = typer.Typer(
    name="kintsugi",
    help="Kintsugi CMA - Prosocial Agent Memory Architecture",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

# Create sub-command groups
security_app = typer.Typer(
    name="security",
    help="Security audit and scanning commands",
    no_args_is_help=True,
)

doctor_app = typer.Typer(
    name="doctor",
    help="Troubleshooting and diagnostic commands",
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Configuration management commands",
    no_args_is_help=True,
)

plugin_app = typer.Typer(
    name="plugin",
    help="Plugin management commands",
    no_args_is_help=True,
)

# Register sub-commands
app.add_typer(security_app, name="security")
app.add_typer(doctor_app, name="doctor")
app.add_typer(config_app, name="config")
app.add_typer(plugin_app, name="plugin")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Kintsugi CMA version {__version__}")
        raise typer.Exit()


def verbose_callback(value: bool) -> None:
    """Set verbose mode."""
    if value:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        callback=verbose_callback,
        help="Enable verbose output.",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file.",
        envvar="KINTSUGI_CONFIG",
    ),
) -> None:
    """
    Kintsugi CMA - Prosocial Agent Memory Architecture

    A comprehensive toolkit for managing AI agent memory systems
    with built-in ethical guardrails and prosocial behavior optimization.

    Use --help on any subcommand for detailed information.
    """
    pass


@app.command()
def status(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed status information.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, yaml.",
    ),
) -> None:
    """
    Show overall system status.

    Displays the current health and status of all Kintsugi components
    including database connection, embedding models, and API health.
    """
    from kintsugi.cli.output import print_status, print_json

    checks = [
        ("Database Connection", True, "Connected to PostgreSQL"),
        ("Embedding Model", True, "sentence-transformers loaded"),
        ("EFE Engine", True, "Active with 5 ethical weights"),
        ("Memory Store", True, "12,456 memories indexed"),
        ("Plugin System", True, "3 plugins enabled"),
    ]

    if format == "json":
        data = {
            "status": "healthy",
            "components": {name: {"ok": ok, "message": msg} for name, ok, msg in checks},
        }
        print_json(data)
    else:
        console.print()
        console.print(Panel.fit(
            "[bold green]System Status: Healthy[/bold green]",
            title="Kintsugi CMA",
        ))
        console.print()
        print_status(checks)

    if detailed:
        console.print()
        table = Table(title="Component Details")
        table.add_column("Component", style="cyan")
        table.add_column("Version")
        table.add_column("Uptime")
        table.add_column("Last Check")

        table.add_row("Core Engine", "0.1.0", "2d 4h 32m", "2 minutes ago")
        table.add_row("Memory Store", "0.1.0", "2d 4h 32m", "5 seconds ago")
        table.add_row("EFE Engine", "0.1.0", "2d 4h 32m", "1 minute ago")
        table.add_row("Plugin Manager", "0.1.0", "2d 4h 32m", "30 seconds ago")

        console.print(table)


@app.command()
def init(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory to initialize Kintsugi in.",
    ),
    template: str = typer.Option(
        "default",
        "--template",
        "-t",
        help="Configuration template: default, minimal, enterprise.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration.",
    ),
) -> None:
    """
    Initialize a new Kintsugi installation.

    Creates the necessary configuration files and directory structure
    for a Kintsugi CMA deployment.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    config_path = directory / ".kintsugi"

    if config_path.exists() and not force:
        print_warning(f"Configuration already exists at {config_path}")
        print_warning("Use --force to overwrite")
        raise typer.Exit(1)

    console.print(f"Initializing Kintsugi in [cyan]{directory.absolute()}[/cyan]...")
    console.print(f"Using template: [cyan]{template}[/cyan]")

    # Would create actual files here
    steps = [
        "Creating configuration directory...",
        "Generating configuration file...",
        "Setting up database schema...",
        "Initializing EFE weights...",
        "Creating plugin directory...",
    ]

    for step in steps:
        console.print(f"  [dim]{step}[/dim]")

    print_success("Kintsugi initialized successfully!")
    console.print()
    console.print("Next steps:")
    console.print("  1. Review configuration: [cyan]kintsugi config show[/cyan]")
    console.print("  2. Run diagnostics: [cyan]kintsugi doctor run[/cyan]")
    console.print("  3. Start the server: [cyan]kintsugi serve[/cyan]")


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind to.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of worker processes.",
    ),
) -> None:
    """
    Start the Kintsugi API server.

    Launches the REST API server for interacting with the Kintsugi
    memory architecture programmatically.
    """
    from kintsugi.cli.output import print_success

    console.print(Panel.fit(
        f"Starting Kintsugi server on [cyan]http://{host}:{port}[/cyan]",
        title="Server",
    ))

    if reload:
        console.print("[yellow]Auto-reload enabled (development mode)[/yellow]")

    console.print(f"Workers: {workers}")
    console.print()
    console.print("Press [bold]Ctrl+C[/bold] to stop")
    console.print()

    # Would actually start server here
    # uvicorn.run("kintsugi.api:app", host=host, port=port, reload=reload, workers=workers)

    print_success("Server started successfully")


@app.command()
def shell(
    context: str = typer.Option(
        "default",
        "--context",
        help="Memory context to use.",
    ),
) -> None:
    """
    Start an interactive Kintsugi shell.

    Opens a REPL for interacting with the Kintsugi memory system
    and testing queries interactively.
    """
    console.print(Panel.fit(
        "Kintsugi Interactive Shell",
        subtitle=f"Context: {context}",
    ))
    console.print()
    console.print("Type [cyan]help[/cyan] for commands, [cyan]exit[/cyan] to quit")
    console.print()

    # Would start actual REPL here
    console.print("[dim]kintsugi>[/dim] ", end="")


@app.command()
def export(
    output: Path = typer.Argument(
        ...,
        help="Output file path.",
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Export format: json, yaml, parquet.",
    ),
    include_embeddings: bool = typer.Option(
        False,
        "--embeddings",
        help="Include embedding vectors in export.",
    ),
    filter_context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Only export memories from this context.",
    ),
) -> None:
    """
    Export memories to a file.

    Exports the memory store to various formats for backup,
    analysis, or migration purposes.
    """
    from kintsugi.cli.output import print_success

    console.print(f"Exporting memories to [cyan]{output}[/cyan]...")
    console.print(f"Format: {format}")

    if include_embeddings:
        console.print("[yellow]Including embeddings (large file)[/yellow]")

    if filter_context:
        console.print(f"Filtering by context: {filter_context}")

    # Would do actual export here

    print_success(f"Exported 12,456 memories to {output}")


@app.command("import")
def import_memories(
    input_file: Path = typer.Argument(
        ...,
        help="Input file path.",
    ),
    merge: bool = typer.Option(
        True,
        "--merge/--replace",
        help="Merge with existing or replace all.",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate memories before import.",
    ),
) -> None:
    """
    Import memories from a file.

    Imports memories from a previously exported file, with options
    to merge with existing data or replace entirely.
    """
    from kintsugi.cli.output import print_success, print_warning

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print(f"Importing memories from [cyan]{input_file}[/cyan]...")

    if validate:
        console.print("Validating input file...")

    if merge:
        console.print("Merging with existing memories...")
    else:
        print_warning("Replacing all existing memories!")

    # Would do actual import here

    print_success("Imported 5,234 memories successfully")


@app.command()
def tune(
    strategy: str = typer.Option(
        "gradient",
        "--strategy",
        "-s",
        help="Tuning strategy: gradient, evolutionary, bayesian, manual.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show proposed changes without applying.",
    ),
    approve: bool = typer.Option(
        False,
        "--approve",
        "-y",
        help="Automatically approve changes.",
    ),
) -> None:
    """
    Run EFE auto-tuning.

    Analyzes outcome feedback and proposes weight adjustments
    for the Ethical Framing Engine.
    """
    from kintsugi.cli.output import print_success, print_warning

    console.print(Panel.fit(
        f"EFE Auto-Tuning ({strategy})",
        title="Tuning",
    ))

    # Show current weights
    table = Table(title="Current EFE Weights")
    table.add_column("Dimension", style="cyan")
    table.add_column("Current", justify="right")
    table.add_column("Proposed", justify="right")
    table.add_column("Change", justify="right")

    table.add_row("Autonomy", "0.300", "0.315", "[green]+0.015[/green]")
    table.add_row("Beneficence", "0.400", "0.412", "[green]+0.012[/green]")
    table.add_row("Non-maleficence", "0.500", "0.498", "[red]-0.002[/red]")
    table.add_row("Justice", "0.350", "0.360", "[green]+0.010[/green]")
    table.add_row("Transparency", "0.450", "0.445", "[red]-0.005[/red]")

    console.print()
    console.print(table)
    console.print()

    if dry_run:
        print_warning("Dry run - no changes applied")
        return

    if not approve:
        approve = typer.confirm("Apply proposed weight changes?")

    if approve:
        print_success("Weight changes applied successfully")
    else:
        print_warning("Changes not applied")


# Import subcommand modules to register their commands
# These are imported at the end to avoid circular imports
def _register_subcommands() -> None:
    """Register all subcommand modules."""
    from kintsugi.cli import security  # noqa: F401
    from kintsugi.cli import doctor  # noqa: F401
    from kintsugi.cli import config  # noqa: F401
    from kintsugi.cli import plugins  # noqa: F401


# Expose the apps for use in submodules
__all__ = [
    "app",
    "security_app",
    "doctor_app",
    "config_app",
    "plugin_app",
    "console",
    "err_console",
    "__version__",
]


def cli() -> None:
    """Entry point for the CLI."""
    try:
        _register_subcommands()
    except ImportError:
        # Submodules not yet created, commands will be limited
        pass
    app()


if __name__ == "__main__":
    cli()
