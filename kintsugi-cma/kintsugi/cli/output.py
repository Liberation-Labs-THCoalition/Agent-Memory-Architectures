"""
Kintsugi CLI - Rich Output Helpers

Utility functions for consistent, beautiful command-line output using Rich.
Provides standardized formatting for tables, status indicators, JSON output,
and various message types.

Functions:
    print_table    - Print a formatted table
    print_status   - Print status checks with pass/fail indicators
    print_json     - Print formatted JSON
    print_error    - Print error message
    print_success  - Print success message
    print_warning  - Print warning message
    print_info     - Print info message
    print_panel    - Print a bordered panel
    print_tree     - Print a tree structure
    print_progress - Print a progress indicator
    print_diff     - Print a colorized diff
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Iterator, Optional, Sequence

from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Create console instances
console = Console()
err_console = Console(stderr=True)

# Status indicators
STATUS_PASS = "[green][/green]"
STATUS_FAIL = "[red][/red]"
STATUS_WARN = "[yellow][/yellow]"
STATUS_INFO = "[blue]i[/blue]"
STATUS_SKIP = "[dim]-[/dim]"

# Fallback indicators for terminals without Unicode
STATUS_PASS_ASCII = "[green]PASS[/green]"
STATUS_FAIL_ASCII = "[red]FAIL[/red]"
STATUS_WARN_ASCII = "[yellow]WARN[/yellow]"
STATUS_INFO_ASCII = "[blue]INFO[/blue]"
STATUS_SKIP_ASCII = "[dim]SKIP[/dim]"


def print_table(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    styles: Optional[list[str]] = None,
    show_header: bool = True,
    show_lines: bool = False,
    max_width: Optional[int] = None,
) -> None:
    """
    Print a rich table.

    Args:
        title: Table title
        columns: Column headers
        rows: Table rows (list of lists)
        styles: Optional column styles
        show_header: Whether to show column headers
        show_lines: Whether to show row separator lines
        max_width: Optional maximum column width
    """
    table = Table(title=title, show_header=show_header, show_lines=show_lines)

    # Add columns
    for i, col in enumerate(columns):
        style = styles[i] if styles and i < len(styles) else None
        if max_width:
            table.add_column(col, style=style, max_width=max_width)
        else:
            table.add_column(col, style=style)

    # Add rows
    for row in rows:
        # Ensure row has correct number of columns
        padded_row = list(row) + [""] * (len(columns) - len(row))
        table.add_row(*padded_row[:len(columns)])

    console.print(table)


def print_status(
    checks: list[tuple[str, bool, str]],
    title: Optional[str] = None,
    show_details: bool = True,
    use_unicode: bool = True,
) -> None:
    """
    Print status checks with pass/fail indicators.

    Args:
        checks: List of (name, passed, message) tuples
        title: Optional title for the status list
        show_details: Whether to show the message/details
        use_unicode: Whether to use Unicode symbols
    """
    if title:
        console.print(f"[bold]{title}[/bold]")
        console.print()

    pass_icon = STATUS_PASS if use_unicode else STATUS_PASS_ASCII
    fail_icon = STATUS_FAIL if use_unicode else STATUS_FAIL_ASCII

    for name, passed, message in checks:
        icon = pass_icon if passed else fail_icon

        if show_details:
            status_color = "green" if passed else "red"
            console.print(f"  {icon} [cyan]{name}[/cyan]: [{status_color}]{message}[/{status_color}]")
        else:
            console.print(f"  {icon} {name}")


def print_json(
    data: dict | list,
    indent: int = 2,
    highlight: bool = True,
    sort_keys: bool = False,
) -> None:
    """
    Print formatted JSON.

    Args:
        data: Data to print as JSON
        indent: Indentation level
        highlight: Whether to syntax highlight
        sort_keys: Whether to sort dictionary keys
    """
    if highlight:
        json_str = json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
        console.print(JSON(json_str))
    else:
        json_str = json.dumps(data, indent=indent, sort_keys=sort_keys, default=str)
        console.print(json_str)


def print_error(
    message: str,
    details: Optional[str] = None,
    hint: Optional[str] = None,
    exit_code: Optional[int] = None,
) -> None:
    """
    Print error message.

    Args:
        message: Error message
        details: Optional detailed error information
        hint: Optional hint for resolving the error
        exit_code: Optional exit code (if provided, will exit)
    """
    err_console.print(f"[bold red]Error:[/bold red] {message}")

    if details:
        err_console.print(f"[dim]{details}[/dim]")

    if hint:
        err_console.print(f"[yellow]Hint:[/yellow] {hint}")

    if exit_code is not None:
        import sys
        sys.exit(exit_code)


def print_success(
    message: str,
    details: Optional[str] = None,
) -> None:
    """
    Print success message.

    Args:
        message: Success message
        details: Optional details
    """
    console.print(f"[bold green]Success:[/bold green] {message}")

    if details:
        console.print(f"[dim]{details}[/dim]")


def print_warning(
    message: str,
    details: Optional[str] = None,
) -> None:
    """
    Print warning message.

    Args:
        message: Warning message
        details: Optional details
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")

    if details:
        console.print(f"[dim]{details}[/dim]")


def print_info(
    message: str,
    details: Optional[str] = None,
) -> None:
    """
    Print info message.

    Args:
        message: Info message
        details: Optional details
    """
    console.print(f"[bold blue]Info:[/bold blue] {message}")

    if details:
        console.print(f"[dim]{details}[/dim]")


def print_panel(
    content: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style: str = "default",
    expand: bool = True,
) -> None:
    """
    Print a bordered panel.

    Args:
        content: Panel content
        title: Optional panel title
        subtitle: Optional panel subtitle
        style: Panel style (default, success, error, warning, info)
        expand: Whether to expand panel to full width
    """
    border_style = {
        "default": "blue",
        "success": "green",
        "error": "red",
        "warning": "yellow",
        "info": "cyan",
    }.get(style, "blue")

    panel = Panel(
        content,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        expand=expand,
    )
    console.print(panel)


def print_tree(
    data: dict,
    title: str = "Tree",
    guide_style: str = "dim",
) -> None:
    """
    Print a tree structure.

    Args:
        data: Nested dictionary to display as tree
        title: Root node title
        guide_style: Style for tree guide lines
    """
    tree = Tree(f"[bold]{title}[/bold]", guide_style=guide_style)

    def add_nodes(parent_node, data_dict):
        if isinstance(data_dict, dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    child = parent_node.add(f"[cyan]{key}[/cyan]")
                    add_nodes(child, value)
                elif isinstance(value, list):
                    child = parent_node.add(f"[cyan]{key}[/cyan]")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_node = child.add(f"[dim]{i}[/dim]")
                            add_nodes(item_node, item)
                        else:
                            child.add(str(item))
                else:
                    parent_node.add(f"[cyan]{key}[/cyan]: {value}")
        elif isinstance(data_dict, list):
            for i, item in enumerate(data_dict):
                if isinstance(item, dict):
                    item_node = parent_node.add(f"[dim]{i}[/dim]")
                    add_nodes(item_node, item)
                else:
                    parent_node.add(str(item))

    add_nodes(tree, data)
    console.print(tree)


def print_code(
    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Print syntax-highlighted code.

    Args:
        code: Code to display
        language: Programming language for highlighting
        theme: Color theme
        line_numbers: Whether to show line numbers
        title: Optional title
    """
    syntax = Syntax(
        code,
        language,
        theme=theme,
        line_numbers=line_numbers,
    )

    if title:
        console.print(Panel(syntax, title=title))
    else:
        console.print(syntax)


def print_markdown(content: str) -> None:
    """
    Print rendered markdown.

    Args:
        content: Markdown content
    """
    md = Markdown(content)
    console.print(md)


def print_diff(
    old_lines: list[str],
    new_lines: list[str],
    title: Optional[str] = None,
    context: int = 3,
) -> None:
    """
    Print a colorized diff.

    Args:
        old_lines: Original lines
        new_lines: Modified lines
        title: Optional title
        context: Number of context lines
    """
    import difflib

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        lineterm="",
        n=context,
    )

    if title:
        console.print(f"[bold]{title}[/bold]")
        console.print()

    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            console.print(f"[green]{line}[/green]")
        elif line.startswith("-") and not line.startswith("---"):
            console.print(f"[red]{line}[/red]")
        elif line.startswith("@@"):
            console.print(f"[cyan]{line}[/cyan]")
        else:
            console.print(line)


def print_key_value(
    items: list[tuple[str, Any]],
    title: Optional[str] = None,
    separator: str = ":",
    key_style: str = "cyan",
) -> None:
    """
    Print key-value pairs in a formatted list.

    Args:
        items: List of (key, value) tuples
        title: Optional title
        separator: Separator between key and value
        key_style: Style for keys
    """
    if title:
        console.print(f"[bold]{title}[/bold]")
        console.print()

    max_key_len = max(len(str(k)) for k, _ in items) if items else 0

    for key, value in items:
        padded_key = str(key).ljust(max_key_len)
        console.print(f"  [{key_style}]{padded_key}[/{key_style}]{separator} {value}")


def print_bullet_list(
    items: list[str],
    title: Optional[str] = None,
    bullet: str = "*",
    indent: int = 2,
) -> None:
    """
    Print a bullet list.

    Args:
        items: List items
        title: Optional title
        bullet: Bullet character
        indent: Indentation level
    """
    if title:
        console.print(f"[bold]{title}[/bold]")
        console.print()

    prefix = " " * indent
    for item in items:
        console.print(f"{prefix}[dim]{bullet}[/dim] {item}")


def print_numbered_list(
    items: list[str],
    title: Optional[str] = None,
    start: int = 1,
    indent: int = 2,
) -> None:
    """
    Print a numbered list.

    Args:
        items: List items
        title: Optional title
        start: Starting number
        indent: Indentation level
    """
    if title:
        console.print(f"[bold]{title}[/bold]")
        console.print()

    prefix = " " * indent
    width = len(str(start + len(items) - 1))

    for i, item in enumerate(items, start=start):
        num = str(i).rjust(width)
        console.print(f"{prefix}[cyan]{num}.[/cyan] {item}")


def create_progress(
    description: str = "Processing...",
    total: Optional[int] = None,
    show_speed: bool = False,
) -> Progress:
    """
    Create a progress bar context manager.

    Args:
        description: Progress description
        total: Total number of items (None for indeterminate)
        show_speed: Whether to show processing speed

    Returns:
        Progress context manager
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ]

    if total is not None:
        columns.extend([
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
        ])

    if show_speed:
        columns.append(TimeElapsedColumn())

    return Progress(*columns, console=console)


def confirm(
    message: str,
    default: bool = False,
    abort: bool = False,
) -> bool:
    """
    Prompt for confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter
        abort: Whether to raise exception on negative response

    Returns:
        True if confirmed, False otherwise
    """
    default_hint = "[Y/n]" if default else "[y/N]"
    response = console.input(f"{message} {default_hint} ")

    if not response:
        result = default
    else:
        result = response.lower() in ("y", "yes", "true", "1")

    if abort and not result:
        raise KeyboardInterrupt("Aborted by user")

    return result


def format_bytes(size: int) -> str:
    """
    Format byte size for display.

    Args:
        size: Size in bytes

    Returns:
        Human-readable size string
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration for display.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_timestamp(
    dt: datetime,
    relative: bool = False,
) -> str:
    """
    Format timestamp for display.

    Args:
        dt: Datetime object
        relative: Whether to show relative time

    Returns:
        Formatted timestamp string
    """
    if relative:
        now = datetime.now(dt.tzinfo)
        delta = now - dt

        if delta.days > 365:
            years = delta.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif delta.days > 30:
            months = delta.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix
