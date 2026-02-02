"""Security monitoring via pattern matching.

Scans shell commands and free text for dangerous patterns, injection attempts,
and PII leakage indicators.  Patterns are compiled once and matched in O(n)
per input string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class SecurityVerdict:
    """Result of a security scan."""

    verdict: Verdict
    reason: str
    matched_pattern: Optional[str] = None
    severity: Optional[Severity] = None


# ---------------------------------------------------------------------------
# Built-in pattern library
# ---------------------------------------------------------------------------

@dataclass
class _Pattern:
    regex: re.Pattern[str]
    severity: Severity
    description: str
    verdict: Verdict


def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


_DANGEROUS_SHELL: List[_Pattern] = [
    _Pattern(_compile(r"\brm\s+(-\w*)?r\w*\s+/\s*$|rm\s+-rf\s+/"), Severity.CRITICAL, "Recursive delete of root filesystem", Verdict.BLOCK),
    _Pattern(_compile(r"\bchmod\s+777\b"), Severity.HIGH, "World-writable permission change", Verdict.BLOCK),
    _Pattern(_compile(r"\bcurl\b.*\|\s*\bbash\b"), Severity.CRITICAL, "Piping remote content to shell", Verdict.BLOCK),
    _Pattern(_compile(r"\bwget\b.*\|\s*\bsh\b"), Severity.CRITICAL, "Piping remote content to shell", Verdict.BLOCK),
    _Pattern(_compile(r"\bdd\s+if=/dev/"), Severity.HIGH, "Raw device read via dd", Verdict.BLOCK),
    _Pattern(_compile(r"\bmkfs\b"), Severity.CRITICAL, "Filesystem format command", Verdict.BLOCK),
    _Pattern(_compile(r">\s*/dev/sd[a-z]"), Severity.CRITICAL, "Direct write to block device", Verdict.BLOCK),
    _Pattern(_compile(r"\b:(){ :\|:& };:"), Severity.CRITICAL, "Fork bomb", Verdict.BLOCK),
    _Pattern(_compile(r"\bshutdown\b|\breboot\b|\binit\s+0\b"), Severity.HIGH, "System shutdown/reboot command", Verdict.BLOCK),
    _Pattern(_compile(r"\bsudo\s+rm\b"), Severity.HIGH, "Privileged file deletion", Verdict.WARN),
]

_SQL_INJECTION: List[_Pattern] = [
    _Pattern(_compile(r"('\s*(OR|AND)\s+'[^']*'\s*=\s*'[^']*')"), Severity.HIGH, "Classic SQL injection tautology", Verdict.BLOCK),
    _Pattern(_compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s+", re.IGNORECASE), Severity.HIGH, "SQL statement injection via semicolon", Verdict.BLOCK),
    _Pattern(_compile(r"UNION\s+(ALL\s+)?SELECT", re.IGNORECASE), Severity.HIGH, "UNION SELECT injection", Verdict.BLOCK),
    _Pattern(_compile(r"--\s*$", re.MULTILINE), Severity.MEDIUM, "SQL comment termination", Verdict.WARN),
]

_PATH_TRAVERSAL: List[_Pattern] = [
    _Pattern(_compile(r"\.\./\.\./"), Severity.HIGH, "Path traversal (double dot-dot-slash)", Verdict.BLOCK),
    _Pattern(_compile(r"%2e%2e[/\\%]", re.IGNORECASE), Severity.HIGH, "URL-encoded path traversal", Verdict.BLOCK),
    _Pattern(_compile(r"\.\.[/\\]"), Severity.MEDIUM, "Simple path traversal", Verdict.WARN),
]

_TEXT_PATTERNS: List[_Pattern] = [
    *_SQL_INJECTION,
    *_PATH_TRAVERSAL,
]


# ---------------------------------------------------------------------------
# SecurityMonitor
# ---------------------------------------------------------------------------

class SecurityMonitor:
    """Stateless pattern-matching security scanner.

    Maintains a library of compiled regex patterns and checks commands or
    free text against them, returning the most severe match.
    """

    def __init__(self) -> None:
        self._command_patterns: List[_Pattern] = list(_DANGEROUS_SHELL)
        self._text_patterns: List[_Pattern] = list(_TEXT_PATTERNS)

    # -- public API ---------------------------------------------------------

    def check_command(self, cmd: str) -> SecurityVerdict:
        """Scan a shell command string against dangerous-command patterns.

        Returns the verdict for the highest-severity match, or ALLOW if
        no patterns trigger.
        """
        return self._scan(cmd, self._command_patterns)

    def check_text(self, text: str) -> SecurityVerdict:
        """Scan free text for injection attempts, traversal, etc."""
        return self._scan(text, self._text_patterns)

    def add_pattern(
        self,
        pattern: str,
        severity: str,
        description: str,
        *,
        target: str = "command",
        verdict: str = "BLOCK",
    ) -> None:
        """Register a custom pattern at runtime.

        Args:
            pattern:     Regex string.
            severity:    One of LOW, MEDIUM, HIGH, CRITICAL.
            description: Human-readable explanation.
            target:      'command' or 'text'.
            verdict:     'ALLOW', 'WARN', or 'BLOCK'.
        """
        entry = _Pattern(
            regex=_compile(pattern),
            severity=Severity(severity.upper()),
            description=description,
            verdict=Verdict(verdict.upper()),
        )
        if target == "text":
            self._text_patterns.append(entry)
        else:
            self._command_patterns.append(entry)

    # -- internals ----------------------------------------------------------

    _SEVERITY_ORDER = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}

    def _scan(self, text: str, patterns: List[_Pattern]) -> SecurityVerdict:
        worst: Optional[_Pattern] = None
        matched_str: Optional[str] = None
        for pat in patterns:
            m = pat.regex.search(text)
            if m:
                if worst is None or self._SEVERITY_ORDER[pat.severity] > self._SEVERITY_ORDER[worst.severity]:
                    worst = pat
                    matched_str = m.group(0)
        if worst is None:
            return SecurityVerdict(verdict=Verdict.ALLOW, reason="No dangerous patterns detected.")
        return SecurityVerdict(
            verdict=worst.verdict,
            reason=worst.description,
            matched_pattern=matched_str,
            severity=worst.severity,
        )
