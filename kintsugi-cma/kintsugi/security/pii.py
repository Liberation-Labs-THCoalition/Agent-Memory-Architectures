"""PII detection and redaction middleware.

Provides regex-based detection of common PII types (email, phone, SSN,
credit card with Luhn validation, IP address, date of birth) and a
redaction engine with mask/remove modes.  Includes a FastAPI middleware
factory for automatic response-body redaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PIIDetection:
    """A single PII finding within a text."""

    pii_type: str
    start: int
    end: int
    original: str  # present in detect() results; never logged


@dataclass(frozen=True)
class RedactionResult:
    """Output of a redaction pass."""

    redacted_text: str
    detections_count: int
    types_found: List[str]


# ---------------------------------------------------------------------------
# Luhn check
# ---------------------------------------------------------------------------

def _luhn_check(digits: str) -> bool:
    """Return True if *digits* passes the Luhn algorithm."""
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 2:
        return False
    checksum = 0
    reverse = nums[::-1]
    for i, n in enumerate(reverse):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[Dict] = [
    {
        "type": "EMAIL",
        "regex": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "validator": None,
    },
    {
        "type": "PHONE",
        "regex": re.compile(
            r"(?<!\d)"
            r"(?:\+?1[-.\s]?)?"
            r"(?:\(?\d{3}\)?[-.\s]?)"
            r"\d{3}[-.\s]?\d{4}"
            r"(?!\d)"
        ),
        "validator": None,
    },
    {
        "type": "SSN",
        "regex": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "validator": None,
    },
    {
        "type": "CREDIT_CARD",
        "regex": re.compile(
            r"(?<!\d)"
            r"(?:\d{4}[-\s]?){3}\d{4}"
            r"(?!\d)"
        ),
        "validator": lambda m: _luhn_check(re.sub(r"[\s-]", "", m)),
    },
    {
        "type": "IP_ADDRESS",
        "regex": re.compile(
            r"(?<!\d)"
            r"(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
            r"(?!\d)"
        ),
        "validator": None,
    },
    {
        "type": "DATE_OF_BIRTH",
        "regex": re.compile(
            r"(?i)(?:dob|date\s*of\s*birth|birth\s*date)"
            r"[:\s]*"
            r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})"
        ),
        "validator": None,
    },
]


# ---------------------------------------------------------------------------
# PIIRedactor
# ---------------------------------------------------------------------------

class PIIRedactor:
    """Detects and redacts PII from arbitrary text."""

    def __init__(self, extra_patterns: Optional[List[Dict]] = None) -> None:
        self._patterns = list(_PII_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def detect(self, text: str) -> List[PIIDetection]:
        """Return all PII detections sorted by start position."""
        detections: List[PIIDetection] = []
        for pat in self._patterns:
            for m in pat["regex"].finditer(text):
                value = m.group(0)
                validator = pat.get("validator")
                if validator and not validator(value):
                    continue
                detections.append(
                    PIIDetection(
                        pii_type=pat["type"],
                        start=m.start(),
                        end=m.end(),
                        original=value,
                    )
                )
        detections.sort(key=lambda d: d.start)
        return detections

    def redact(self, text: str, mode: str = "mask") -> RedactionResult:
        """Redact PII from *text*.

        Args:
            text: Input string.
            mode: ``"mask"`` replaces each finding with ``[REDACTED_TYPE]``;
                  ``"remove"`` deletes the finding entirely.

        Returns:
            RedactionResult with the scrubbed text and summary stats.
        """
        detections = self.detect(text)
        if not detections:
            return RedactionResult(redacted_text=text, detections_count=0, types_found=[])

        parts: List[str] = []
        prev_end = 0
        types_seen: Set[str] = set()
        for det in detections:
            parts.append(text[prev_end:det.start])
            if mode == "mask":
                parts.append(f"[REDACTED_{det.pii_type}]")
            # mode == "remove": append nothing
            prev_end = det.end
            types_seen.add(det.pii_type)
        parts.append(text[prev_end:])

        return RedactionResult(
            redacted_text="".join(parts),
            detections_count=len(detections),
            types_found=sorted(types_seen),
        )


# ---------------------------------------------------------------------------
# FastAPI middleware factory
# ---------------------------------------------------------------------------

def pii_redaction_middleware(
    redactor: Optional[PIIRedactor] = None,
    skip_paths: Optional[Sequence[str]] = None,
):
    """Return a FastAPI middleware function that redacts PII from response bodies.

    Usage::

        from fastapi import FastAPI
        from kintsugi.security.pii import pii_redaction_middleware, PIIRedactor

        app = FastAPI()
        app.middleware("http")(pii_redaction_middleware(skip_paths=["/health"]))

    Args:
        redactor:   PIIRedactor instance; a default one is created if omitted.
        skip_paths: Route prefixes that should bypass redaction.
    """
    _redactor = redactor or PIIRedactor()
    _skip: Set[str] = set(skip_paths or [])

    async def middleware(request, call_next):  # type: ignore[no-untyped-def]
        from starlette.responses import Response, StreamingResponse
        response = await call_next(request)

        # Skip configured paths
        if any(request.url.path.startswith(p) for p in _skip):
            return response

        # Only process text-like content types
        ct = response.headers.get("content-type", "")
        if "text" not in ct and "json" not in ct:
            return response

        # Read body
        body_parts: List[bytes] = []
        async for chunk in response.body_iterator:  # type: ignore[attr-defined]
            if isinstance(chunk, str):
                body_parts.append(chunk.encode("utf-8"))
            else:
                body_parts.append(chunk)
        body = b"".join(body_parts).decode("utf-8", errors="replace")

        result = _redactor.redact(body)

        return Response(
            content=result.redacted_text,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return middleware
