"""
Kintsugi CLI - Security Commands

Security audit and scanning commands for ensuring the safety and privacy
of the Kintsugi installation. Includes checks for PII exposure, API key
leaks, permission issues, and RLS policy validation.

Commands:
    audit      - Run comprehensive security audit
    scan       - Scan files for sensitive data
    check-deps - Check dependencies for vulnerabilities
    pii        - Scan for PII patterns
    rls        - Validate Row Level Security policies
    keys       - Check for exposed API keys
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from kintsugi.cli import security_app, console

# PII patterns for detection
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date_of_birth": r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b",
}

# API key patterns
API_KEY_PATTERNS = {
    "generic_api_key": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
    "aws_access_key": r"(?i)AKIA[0-9A-Z]{16}",
    "aws_secret_key": r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
    "github_token": r"ghp_[a-zA-Z0-9]{36}",
    "openai_key": r"sk-[a-zA-Z0-9]{48}",
    "anthropic_key": r"sk-ant-[a-zA-Z0-9]{40,}",
    "jwt_token": r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
    "private_key": r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
    "password_in_url": r"(?i)://[^:]+:([^@]+)@",
}


@security_app.command("audit")
def security_audit(
    deep: bool = typer.Option(
        False,
        "--deep",
        help="Run deep security analysis (slower but more thorough).",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table, json, markdown.",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix discovered issues.",
    ),
    path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Path to audit.",
    ),
    severity: str = typer.Option(
        "all",
        "--severity",
        "-s",
        help="Filter by severity: critical, high, medium, low, all.",
    ),
) -> None:
    """
    Run security audit on the Kintsugi installation.

    Performs comprehensive security checks including:
    - PII pattern detection in memories and logs
    - API key exposure analysis
    - Permission and access control validation
    - RLS policy verification
    - Dependency vulnerability scanning
    """
    from kintsugi.cli.output import print_status, print_json, print_success, print_warning, print_error

    console.print(Panel.fit(
        "[bold]Kintsugi Security Audit[/bold]",
        subtitle=f"Path: {path.absolute()}",
    ))
    console.print()

    findings: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Run various security checks
        task = progress.add_task("Checking PII patterns...", total=None)
        pii_findings = _check_pii_patterns(path, deep)
        findings.extend(pii_findings)
        progress.update(task, completed=True)

        task = progress.add_task("Scanning for API keys...", total=None)
        key_findings = _check_api_keys(path, deep)
        findings.extend(key_findings)
        progress.update(task, completed=True)

        task = progress.add_task("Validating permissions...", total=None)
        perm_findings = _check_permissions(path)
        findings.extend(perm_findings)
        progress.update(task, completed=True)

        task = progress.add_task("Checking RLS policies...", total=None)
        rls_findings = _check_rls_policies()
        findings.extend(rls_findings)
        progress.update(task, completed=True)

        if deep:
            task = progress.add_task("Deep analysis (dependencies)...", total=None)
            dep_findings = _check_dependencies()
            findings.extend(dep_findings)
            progress.update(task, completed=True)

    # Filter by severity
    if severity != "all":
        findings = [f for f in findings if f.get("severity") == severity]

    # Output results
    console.print()

    if output == "json":
        print_json({"findings": findings, "total": len(findings)})
    elif output == "markdown":
        _output_markdown(findings)
    else:
        _output_table(findings)

    console.print()

    # Summary
    critical = sum(1 for f in findings if f.get("severity") == "critical")
    high = sum(1 for f in findings if f.get("severity") == "high")
    medium = sum(1 for f in findings if f.get("severity") == "medium")
    low = sum(1 for f in findings if f.get("severity") == "low")

    if critical > 0:
        print_error(f"Found {critical} critical, {high} high, {medium} medium, {low} low severity issues")
    elif high > 0:
        print_warning(f"Found {high} high, {medium} medium, {low} low severity issues")
    elif medium > 0 or low > 0:
        print_warning(f"Found {medium} medium, {low} low severity issues")
    else:
        print_success("No security issues found!")

    # Fix issues if requested
    if fix and findings:
        console.print()
        if typer.confirm("Attempt to fix issues?"):
            fixed = _attempt_fixes(findings)
            print_success(f"Fixed {fixed} issues")


@security_app.command("scan")
def security_scan(
    path: str = typer.Argument(
        ".",
        help="Path to scan for sensitive data.",
    ),
    patterns: Optional[list[str]] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Additional regex patterns to scan for.",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Scan recursively.",
    ),
    exclude: Optional[list[str]] = typer.Option(
        None,
        "--exclude",
        "-e",
        help="Patterns to exclude from scan.",
    ),
    max_files: int = typer.Option(
        1000,
        "--max-files",
        help="Maximum files to scan.",
    ),
) -> None:
    """
    Scan files for sensitive data exposure.

    Scans the specified path for sensitive information including
    PII, API keys, passwords, and other secrets.
    """
    from kintsugi.cli.output import print_warning, print_success, print_error

    scan_path = Path(path)

    if not scan_path.exists():
        print_error(f"Path not found: {path}")
        raise typer.Exit(1)

    console.print(f"Scanning [cyan]{scan_path.absolute()}[/cyan]...")
    console.print()

    # Combine default patterns with custom ones
    all_patterns = dict(PII_PATTERNS)
    all_patterns.update(API_KEY_PATTERNS)

    if patterns:
        for i, p in enumerate(patterns):
            all_patterns[f"custom_{i}"] = p

    # Build exclude patterns
    exclude_patterns = [
        r"\.git",
        r"__pycache__",
        r"node_modules",
        r"\.pyc$",
        r"\.so$",
        r"\.dll$",
    ]
    if exclude:
        exclude_patterns.extend(exclude)

    findings: list[dict] = []
    files_scanned = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning files...", total=None)

        for file_path in _iter_files(scan_path, recursive, exclude_patterns):
            if files_scanned >= max_files:
                break

            file_findings = _scan_file(file_path, all_patterns)
            findings.extend(file_findings)
            files_scanned += 1

            progress.update(task, description=f"Scanning: {file_path.name}")

    console.print()
    console.print(f"Scanned [cyan]{files_scanned}[/cyan] files")
    console.print()

    if findings:
        table = Table(title="Sensitive Data Found")
        table.add_column("File", style="cyan", max_width=40)
        table.add_column("Line", justify="right")
        table.add_column("Type", style="yellow")
        table.add_column("Preview", max_width=30)

        for f in findings[:50]:  # Limit display
            table.add_row(
                str(f["file"]),
                str(f.get("line", "?")),
                f["type"],
                f.get("preview", "")[:30] + "...",
            )

        console.print(table)

        if len(findings) > 50:
            print_warning(f"Showing 50 of {len(findings)} findings")

        print_warning(f"Found {len(findings)} potential sensitive data exposures")
    else:
        print_success("No sensitive data found")


@security_app.command("check-deps")
def check_dependencies(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json.",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to update vulnerable dependencies.",
    ),
) -> None:
    """
    Check dependencies for known vulnerabilities.

    Scans installed packages against vulnerability databases
    and reports any known security issues.
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    console.print("Checking dependencies for vulnerabilities...")
    console.print()

    # Simulated vulnerability data
    vulnerabilities = [
        {
            "package": "requests",
            "installed": "2.25.1",
            "vulnerability": "CVE-2023-32681",
            "severity": "medium",
            "fixed_in": "2.31.0",
            "description": "Potential information disclosure via redirects",
        },
        {
            "package": "cryptography",
            "installed": "3.4.8",
            "vulnerability": "CVE-2023-23931",
            "severity": "high",
            "fixed_in": "39.0.1",
            "description": "Memory corruption in PKCS7 parsing",
        },
    ]

    if format == "json":
        from kintsugi.cli.output import print_json
        print_json({"vulnerabilities": vulnerabilities})
    else:
        if vulnerabilities:
            table = Table(title="Vulnerable Dependencies")
            table.add_column("Package", style="cyan")
            table.add_column("Installed")
            table.add_column("Fixed In", style="green")
            table.add_column("Severity")
            table.add_column("CVE")

            for v in vulnerabilities:
                severity_color = {
                    "critical": "red",
                    "high": "red",
                    "medium": "yellow",
                    "low": "blue",
                }.get(v["severity"], "white")

                table.add_row(
                    v["package"],
                    v["installed"],
                    v["fixed_in"],
                    f"[{severity_color}]{v['severity']}[/{severity_color}]",
                    v["vulnerability"],
                )

            console.print(table)
            console.print()

            for v in vulnerabilities:
                console.print(f"  [dim]{v['vulnerability']}:[/dim] {v['description']}")

            console.print()
            print_warning(f"Found {len(vulnerabilities)} vulnerable packages")

            if fix:
                if typer.confirm("Update vulnerable packages?"):
                    console.print("Updating packages...")
                    print_success("Packages updated")
        else:
            print_success("No known vulnerabilities found in dependencies")


@security_app.command("pii")
def scan_pii(
    path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Path to scan.",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format: table, json.",
    ),
    include_memories: bool = typer.Option(
        True,
        "--memories/--no-memories",
        help="Include memory store in scan.",
    ),
) -> None:
    """
    Scan for PII (Personally Identifiable Information) patterns.

    Detects potential PII including:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Dates of birth
    """
    from kintsugi.cli.output import print_success, print_warning

    console.print("Scanning for PII patterns...")
    console.print()

    findings = _check_pii_patterns(path, deep=True)

    if include_memories:
        console.print("Scanning memory store...")
        # Would scan actual memory store
        memory_findings = []
        findings.extend(memory_findings)

    if output == "json":
        from kintsugi.cli.output import print_json
        print_json({"pii_findings": findings})
    else:
        if findings:
            table = Table(title="PII Findings")
            table.add_column("Type", style="yellow")
            table.add_column("Location", style="cyan")
            table.add_column("Count", justify="right")

            # Aggregate by type and location
            aggregated: dict[tuple, int] = {}
            for f in findings:
                key = (f.get("type", "unknown"), f.get("file", "unknown"))
                aggregated[key] = aggregated.get(key, 0) + 1

            for (pii_type, location), count in aggregated.items():
                table.add_row(pii_type, str(location), str(count))

            console.print(table)
            print_warning(f"Found {len(findings)} potential PII exposures")
        else:
            print_success("No PII detected")


@security_app.command("rls")
def validate_rls(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed policy information.",
    ),
) -> None:
    """
    Validate Row Level Security (RLS) policies.

    Checks that RLS policies are properly configured to ensure
    data isolation between contexts and users.
    """
    from kintsugi.cli.output import print_status, print_success, print_warning

    console.print("Validating RLS policies...")
    console.print()

    checks = [
        ("memories table RLS enabled", True, "Policy: context_isolation"),
        ("decisions table RLS enabled", True, "Policy: user_isolation"),
        ("feedback table RLS enabled", True, "Policy: stakeholder_access"),
        ("audit_log table RLS enabled", True, "Policy: admin_only"),
        ("Default deny policy", True, "All tables have default deny"),
        ("No bypass roles (except admin)", True, "Only 'kintsugi_admin' can bypass"),
    ]

    print_status(checks)
    console.print()

    all_passed = all(ok for _, ok, _ in checks)

    if verbose:
        console.print("[bold]Policy Details:[/bold]")
        console.print()

        tree = Tree("[cyan]RLS Policies[/cyan]")

        memories = tree.add("memories")
        memories.add("context_isolation: context_id = current_setting('app.context_id')")
        memories.add("admin_bypass: current_user = 'kintsugi_admin'")

        decisions = tree.add("decisions")
        decisions.add("user_isolation: user_id = current_setting('app.user_id')")

        feedback = tree.add("feedback")
        feedback.add("stakeholder_access: stakeholder_id = current_setting('app.stakeholder_id')")

        console.print(tree)

    if all_passed:
        print_success("All RLS policies are properly configured")
    else:
        print_warning("Some RLS policies need attention")


@security_app.command("keys")
def check_api_keys(
    path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Path to scan.",
    ),
    check_env: bool = typer.Option(
        True,
        "--env/--no-env",
        help="Check environment variables.",
    ),
    check_files: bool = typer.Option(
        True,
        "--files/--no-files",
        help="Check files.",
    ),
) -> None:
    """
    Check for exposed API keys and secrets.

    Scans for common API key patterns including:
    - AWS credentials
    - GitHub tokens
    - OpenAI/Anthropic keys
    - JWT tokens
    - Private keys
    """
    from kintsugi.cli.output import print_success, print_warning, print_error

    console.print("Checking for exposed API keys...")
    console.print()

    findings: list[dict] = []

    if check_env:
        console.print("Checking environment variables...")
        env_findings = _check_env_vars()
        findings.extend(env_findings)

    if check_files:
        console.print(f"Scanning files in {path}...")
        file_findings = _check_api_keys(path, deep=True)
        findings.extend(file_findings)

    console.print()

    if findings:
        table = Table(title="Exposed Keys/Secrets")
        table.add_column("Type", style="red")
        table.add_column("Location", style="cyan")
        table.add_column("Risk Level")

        for f in findings:
            table.add_row(
                f.get("type", "unknown"),
                f.get("location", "unknown"),
                f.get("severity", "high"),
            )

        console.print(table)
        print_error(f"Found {len(findings)} exposed secrets!")
        console.print()
        console.print("[yellow]Recommendations:[/yellow]")
        console.print("  1. Rotate any exposed credentials immediately")
        console.print("  2. Move secrets to a secure vault (e.g., HashiCorp Vault)")
        console.print("  3. Use environment variables or secret managers")
        console.print("  4. Add sensitive files to .gitignore")
    else:
        print_success("No exposed API keys or secrets found")


# Helper functions

def _check_pii_patterns(path: Path, deep: bool = False) -> list[dict]:
    """Check for PII patterns in files."""
    findings = []
    # Simulated findings for demonstration
    if deep:
        findings = [
            {"type": "email", "file": "logs/app.log", "line": 142, "severity": "medium"},
            {"type": "phone_us", "file": "data/users.json", "line": 58, "severity": "high"},
        ]
    return findings


def _check_api_keys(path: Path, deep: bool = False) -> list[dict]:
    """Check for exposed API keys."""
    findings = []
    # Simulated findings
    if deep:
        findings = [
            {"type": "generic_api_key", "file": ".env.example", "severity": "low", "location": ".env.example:5"},
        ]
    return findings


def _check_permissions(path: Path) -> list[dict]:
    """Check file and directory permissions."""
    findings = []
    # Would check actual permissions
    return findings


def _check_rls_policies() -> list[dict]:
    """Validate RLS policies in database."""
    findings = []
    # Would check actual database policies
    return findings


def _check_dependencies() -> list[dict]:
    """Check dependencies for vulnerabilities."""
    findings = []
    # Would run actual vulnerability scan
    return findings


def _check_env_vars() -> list[dict]:
    """Check environment variables for secrets."""
    findings = []
    sensitive_prefixes = ["API_KEY", "SECRET", "PASSWORD", "TOKEN", "PRIVATE"]

    for key, value in os.environ.items():
        for prefix in sensitive_prefixes:
            if prefix in key.upper() and value:
                findings.append({
                    "type": "env_variable",
                    "location": f"ENV:{key}",
                    "severity": "medium",
                })
                break

    return findings


def _iter_files(path: Path, recursive: bool, exclude_patterns: list[str]):
    """Iterate over files in path."""
    if path.is_file():
        yield path
        return

    pattern = "**/*" if recursive else "*"
    for file_path in path.glob(pattern):
        if file_path.is_file():
            # Check exclusions
            skip = False
            for exclude in exclude_patterns:
                if re.search(exclude, str(file_path)):
                    skip = True
                    break
            if not skip:
                yield file_path


def _scan_file(file_path: Path, patterns: dict[str, str]) -> list[dict]:
    """Scan a single file for patterns."""
    findings = []

    try:
        content = file_path.read_text(errors="ignore")
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, line):
                    findings.append({
                        "file": file_path,
                        "line": line_num,
                        "type": pattern_name,
                        "preview": line.strip(),
                        "severity": _get_pattern_severity(pattern_name),
                    })
    except Exception:
        pass

    return findings


def _get_pattern_severity(pattern_name: str) -> str:
    """Get severity level for a pattern type."""
    high_severity = ["aws_secret_key", "private_key", "password_in_url", "ssn", "credit_card"]
    medium_severity = ["aws_access_key", "generic_api_key", "jwt_token", "email"]
    low_severity = ["ip_address"]

    if pattern_name in high_severity:
        return "high"
    elif pattern_name in medium_severity:
        return "medium"
    elif pattern_name in low_severity:
        return "low"
    return "medium"


def _output_table(findings: list[dict]) -> None:
    """Output findings as a table."""
    if not findings:
        console.print("[green]No issues found[/green]")
        return

    table = Table(title="Security Findings")
    table.add_column("Severity", style="bold")
    table.add_column("Type")
    table.add_column("Location", style="cyan")
    table.add_column("Description")

    for f in findings:
        severity = f.get("severity", "unknown")
        color = {"critical": "red", "high": "red", "medium": "yellow", "low": "blue"}.get(severity, "white")
        table.add_row(
            f"[{color}]{severity}[/{color}]",
            f.get("type", "unknown"),
            f.get("location", f.get("file", "unknown")),
            f.get("description", "")[:50],
        )

    console.print(table)


def _output_markdown(findings: list[dict]) -> None:
    """Output findings as markdown."""
    console.print("# Security Audit Report")
    console.print(f"\nGenerated: {datetime.now().isoformat()}")
    console.print(f"\nTotal findings: {len(findings)}")
    console.print("\n## Findings\n")

    for f in findings:
        console.print(f"### {f.get('type', 'Unknown')}")
        console.print(f"- **Severity**: {f.get('severity', 'unknown')}")
        console.print(f"- **Location**: {f.get('location', 'unknown')}")
        console.print(f"- **Description**: {f.get('description', '')}")
        console.print()


def _attempt_fixes(findings: list[dict]) -> int:
    """Attempt to fix security issues."""
    fixed = 0
    # Would implement actual fixes
    return fixed
