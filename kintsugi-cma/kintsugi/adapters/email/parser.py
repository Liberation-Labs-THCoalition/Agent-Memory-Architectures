"""
Email parsing utilities for Kintsugi CMA.

This module provides comprehensive email parsing capabilities, converting
raw email data into structured formats suitable for processing by the
Kintsugi system.

Features:
    - Parse raw email bytes or email.message.EmailMessage objects
    - Extract plain text and HTML bodies
    - Handle multipart messages and attachments
    - Detect auto-reply messages to prevent loops
    - Extract entities (dates, amounts, names) from content
    - Infer intent from email subject and body

Security Considerations:
    - Attachment content is optional and size-limited
    - HTML content is preserved but should be sanitized before display
    - Headers are parsed but not blindly trusted
    - Auto-reply detection prevents infinite loops
"""

import email
import email.policy
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime, getaddresses
from typing import Any


@dataclass
class EmailAttachment:
    """
    Represents an email attachment.

    Attachments are extracted from multipart emails and stored
    with metadata. The actual content may be omitted for large
    attachments to save memory.

    Attributes:
        filename: Original filename of the attachment
        content_type: MIME type (e.g., "application/pdf")
        size_bytes: Size in bytes
        content: Raw binary content (None if not loaded)
        content_id: Content-ID for inline attachments
        is_inline: Whether attachment is inline (e.g., embedded image)
        checksum: MD5 checksum for integrity verification

    Example:
        attachment = EmailAttachment(
            filename="grant_proposal.pdf",
            content_type="application/pdf",
            size_bytes=1024000,
            content=pdf_bytes
        )
    """

    filename: str
    content_type: str
    size_bytes: int
    content: bytes | None = None
    content_id: str | None = None
    is_inline: bool = False
    checksum: str | None = None

    def __post_init__(self) -> None:
        """Compute checksum if content is available."""
        if self.content and not self.checksum:
            self.checksum = hashlib.md5(self.content).hexdigest()

    @property
    def extension(self) -> str | None:
        """Extract file extension from filename."""
        if not self.filename or "." not in self.filename:
            return None
        return self.filename.rsplit(".", 1)[-1].lower()

    @property
    def is_image(self) -> bool:
        """Check if attachment is an image."""
        return self.content_type.startswith("image/")

    @property
    def is_document(self) -> bool:
        """Check if attachment is a document (PDF, Word, etc.)."""
        doc_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument",
            "text/plain",
            "text/csv",
        ]
        return any(self.content_type.startswith(t) for t in doc_types)


@dataclass
class ParsedEmail:
    """
    A parsed and structured email message.

    This dataclass provides a normalized representation of an email
    message with all relevant fields extracted and processed.

    Attributes:
        message_id: Unique message identifier from headers
        from_address: Sender's email address
        from_name: Sender's display name (if available)
        to_addresses: List of recipient email addresses
        cc_addresses: List of CC recipient addresses
        subject: Email subject line
        body_text: Plain text body content
        body_html: HTML body content (if available)
        received_at: When the email was received/sent
        attachments: List of email attachments
        headers: Dictionary of email headers
        thread_id: Thread/conversation identifier
        in_reply_to: Message-ID this is a reply to
        references: List of referenced message IDs
        bcc_addresses: BCC recipients (usually empty on received mail)
        reply_to: Reply-to address (if different from sender)
        importance: Email importance/priority level
        is_encrypted: Whether email was encrypted
        original_raw: Original raw email data (optional)

    Example:
        email = ParsedEmail(
            message_id="<abc123@mail.example.com>",
            from_address="donor@foundation.org",
            from_name="Jane Donor",
            to_addresses=["grants@nonprofit.org"],
            cc_addresses=[],
            subject="Re: Grant Application Follow-up",
            body_text="Thank you for your proposal...",
            body_html=None,
            received_at=datetime.now(timezone.utc),
            thread_id="thread_abc123"
        )
    """

    message_id: str
    from_address: str
    from_name: str | None
    to_addresses: list[str]
    cc_addresses: list[str]
    subject: str
    body_text: str
    body_html: str | None
    received_at: datetime
    attachments: list[EmailAttachment] = field(default_factory=list)
    headers: dict[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    bcc_addresses: list[str] = field(default_factory=list)
    reply_to: str | None = None
    importance: str | None = None
    is_encrypted: bool = False
    original_raw: bytes | None = None

    def __post_init__(self) -> None:
        """Process fields after initialization."""
        # Generate thread_id from references if not set
        if not self.thread_id and self.references:
            self.thread_id = self.references[0]
        elif not self.thread_id and self.in_reply_to:
            self.thread_id = self.in_reply_to

    @property
    def sender_domain(self) -> str | None:
        """Extract domain from sender address."""
        if "@" in self.from_address:
            return self.from_address.split("@")[-1].lower()
        return None

    @property
    def all_recipients(self) -> list[str]:
        """Get all recipients (to + cc + bcc)."""
        return self.to_addresses + self.cc_addresses + self.bcc_addresses

    @property
    def has_attachments(self) -> bool:
        """Check if email has any attachments."""
        return len(self.attachments) > 0

    @property
    def total_attachment_size(self) -> int:
        """Get total size of all attachments in bytes."""
        return sum(a.size_bytes for a in self.attachments)

    def get_header(self, name: str, default: str = "") -> str:
        """Get a header value by name (case-insensitive)."""
        return self.headers.get(name.lower(), default)

    def is_from_domain(self, domain: str) -> bool:
        """Check if sender is from a specific domain."""
        return self.sender_domain == domain.lower()


class EmailParser:
    """
    Parse raw emails into structured ParsedEmail format.

    This class handles the complexity of email parsing, including:
    - Multipart message handling
    - Character encoding detection and conversion
    - Header decoding (RFC 2047)
    - Attachment extraction
    - Auto-reply detection
    - Entity extraction from content

    Thread Safety:
        This class is stateless and thread-safe.

    Example:
        parser = EmailParser()

        # Parse from raw bytes
        parsed = parser.parse(raw_email_bytes)

        # Check for auto-reply
        if parser.is_auto_reply(parsed):
            logger.info("Skipping auto-reply")
            return

        # Extract intent
        intent = parser.extract_intent(parsed)
    """

    # Auto-reply detection patterns
    AUTO_REPLY_SUBJECTS = [
        r"^auto[-\s]?reply",
        r"^automatic reply",
        r"^out of office",
        r"^ooo:",
        r"^away:",
        r"^vacation reply",
        r"^i am out of the office",
        r"^thank you for your (email|message)",
        r"^undeliverable:",
        r"^delivery (status |failure )?notification",
        r"^mail delivery (failed|failure)",
        r"^returned mail:",
    ]

    AUTO_REPLY_HEADERS = [
        "auto-submitted",
        "x-auto-response-suppress",
        "x-autorespond",
        "x-autoreply",
        "x-autogenerated",
    ]

    # Intent detection patterns
    INTENT_PATTERNS = {
        "grant_inquiry": [
            r"grant\s+(application|proposal|deadline|opportunity)",
            r"funding\s+(opportunity|available|deadline)",
            r"apply(ing)?\s+for\s+(a\s+)?grant",
        ],
        "status_check": [
            r"status\s+(of|update|check)",
            r"where\s+(are|is)\s+(we|my|the)",
            r"any\s+update",
            r"following\s+up",
        ],
        "document_submission": [
            r"attached\s+(is|are|find)",
            r"please\s+find\s+attached",
            r"submitting\s+(the|my|our)",
            r"here\s+(is|are)\s+(the|my)",
        ],
        "meeting_request": [
            r"schedule\s+a\s+(call|meeting|time)",
            r"can\s+we\s+(meet|talk|discuss)",
            r"available\s+(for|to)\s+(a\s+)?(call|meeting)",
        ],
        "question": [
            r"question\s+(about|regarding)",
            r"could\s+you\s+(please\s+)?(explain|clarify)",
            r"wondering\s+(if|about|whether)",
            r"^(can|could|would)\s+you",
        ],
        "thank_you": [
            r"thank\s+you\s+(for|so\s+much)",
            r"thanks\s+for",
            r"appreciate\s+(your|the)",
            r"grateful\s+for",
        ],
        "deadline_mention": [
            r"deadline\s+(is|approaching|reminder)",
            r"due\s+(date|by|on)",
            r"submit(ted)?\s+by",
            r"before\s+\d{1,2}[/\-]\d{1,2}",
        ],
    }

    # Entity extraction patterns
    DATE_PATTERNS = [
        r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b",
        r"\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
    ]

    AMOUNT_PATTERNS = [
        r"\$[\d,]+(?:\.\d{2})?",
        r"[\d,]+(?:\.\d{2})?\s*(?:dollars?|usd)",
        r"(?:amount|grant|funding)(?:\s+of)?\s*:?\s*\$?[\d,]+(?:\.\d{2})?",
    ]

    def __init__(self, max_attachment_bytes: int = 10 * 1024 * 1024):
        """
        Initialize the email parser.

        Args:
            max_attachment_bytes: Maximum size for loading attachment content
        """
        self._max_attachment_bytes = max_attachment_bytes

    def parse(self, raw: bytes | str) -> ParsedEmail:
        """
        Parse raw email bytes or string into a ParsedEmail.

        Args:
            raw: Raw email data as bytes or string

        Returns:
            Parsed and structured email

        Raises:
            ValueError: If the email cannot be parsed
        """
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

        try:
            message = email.message_from_bytes(
                raw,
                policy=email.policy.default
            )
            parsed = self.parse_message(message)
            parsed.original_raw = raw
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse email: {e}") from e

    def parse_message(self, message: EmailMessage) -> ParsedEmail:
        """
        Parse an email.message.EmailMessage into a ParsedEmail.

        Args:
            message: EmailMessage object to parse

        Returns:
            Parsed and structured email
        """
        # Extract sender
        from_header = message.get("From", "")
        from_name, from_address = parseaddr(from_header)
        from_name = self._decode_header(from_name) if from_name else None
        from_address = from_address.lower()

        # Extract recipients
        to_addresses = self._extract_addresses(message.get_all("To", []))
        cc_addresses = self._extract_addresses(message.get_all("Cc", []))
        bcc_addresses = self._extract_addresses(message.get_all("Bcc", []))

        # Extract subject
        subject = self._decode_header(message.get("Subject", ""))

        # Extract message ID
        message_id = message.get("Message-ID", "")
        if not message_id:
            # Generate a synthetic message ID
            message_id = f"<{hashlib.md5(str(datetime.now()).encode()).hexdigest()}@kintsugi>"

        # Extract threading info
        in_reply_to = message.get("In-Reply-To")
        references = message.get("References", "").split()

        # Extract date
        date_header = message.get("Date")
        if date_header:
            try:
                received_at = parsedate_to_datetime(date_header)
                if received_at.tzinfo is None:
                    received_at = received_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                received_at = datetime.now(timezone.utc)
        else:
            received_at = datetime.now(timezone.utc)

        # Extract reply-to
        reply_to_header = message.get("Reply-To", "")
        _, reply_to = parseaddr(reply_to_header)
        reply_to = reply_to.lower() if reply_to else None

        # Extract importance
        importance = message.get("Importance") or message.get("X-Priority")

        # Extract body and attachments
        body_text, body_html, attachments = self._extract_body_and_attachments(message)

        # Build headers dict
        headers = {key.lower(): value for key, value in message.items()}

        return ParsedEmail(
            message_id=message_id,
            from_address=from_address,
            from_name=from_name,
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            bcc_addresses=bcc_addresses,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            received_at=received_at,
            attachments=attachments,
            headers=headers,
            in_reply_to=in_reply_to,
            references=references,
            reply_to=reply_to,
            importance=importance,
        )

    def extract_intent(self, email_obj: ParsedEmail) -> str:
        """
        Extract the likely intent from email content.

        Analyzes subject and body to determine the primary purpose
        of the email using pattern matching.

        Args:
            email_obj: Parsed email to analyze

        Returns:
            Intent string (e.g., "grant_inquiry", "status_check")
        """
        content = f"{email_obj.subject}\n{email_obj.body_text}".lower()

        # Check each intent pattern
        intent_scores: dict[str, int] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    intent_scores[intent] = intent_scores.get(intent, 0) + len(matches)

        # Return highest scoring intent, or "general" if none match
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return "general"

    def extract_entities(self, email_obj: ParsedEmail) -> dict[str, list[str]]:
        """
        Extract entities (dates, amounts, names) from email content.

        Args:
            email_obj: Parsed email to analyze

        Returns:
            Dictionary of entity types to lists of extracted values
        """
        content = f"{email_obj.subject}\n{email_obj.body_text}"
        entities: dict[str, list[str]] = {
            "dates": [],
            "amounts": [],
            "emails": [],
            "urls": [],
        }

        # Extract dates
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["dates"].extend(matches)

        # Extract amounts
        for pattern in self.AMOUNT_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["amounts"].extend(matches)

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["emails"] = re.findall(email_pattern, content)

        # Extract URLs
        url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
        entities["urls"] = re.findall(url_pattern, content)

        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def is_auto_reply(self, email_obj: ParsedEmail) -> bool:
        """
        Check if an email is an auto-reply to prevent loops.

        Examines headers and subject patterns to detect automated
        responses such as out-of-office messages, delivery failures,
        and other auto-generated emails.

        Args:
            email_obj: Parsed email to check

        Returns:
            True if email appears to be an auto-reply
        """
        # Check for auto-reply headers
        for header in self.AUTO_REPLY_HEADERS:
            if email_obj.get_header(header):
                return True

        # Check precedence header
        precedence = email_obj.get_header("precedence", "").lower()
        if precedence in ("bulk", "junk", "list", "auto_reply"):
            return True

        # Check subject patterns
        subject = email_obj.subject.lower()
        for pattern in self.AUTO_REPLY_SUBJECTS:
            if re.search(pattern, subject, re.IGNORECASE):
                return True

        # Check for mailing list headers
        if email_obj.get_header("list-unsubscribe"):
            return True

        return False

    def is_bounce(self, email_obj: ParsedEmail) -> bool:
        """
        Check if an email is a bounce/delivery failure message.

        Args:
            email_obj: Parsed email to check

        Returns:
            True if email is a bounce message
        """
        # Check common bounce indicators
        subject = email_obj.subject.lower()
        bounce_subjects = [
            "undeliverable",
            "delivery status notification",
            "mail delivery failed",
            "returned mail",
            "delivery failure",
            "mailbox unavailable",
        ]

        for indicator in bounce_subjects:
            if indicator in subject:
                return True

        # Check from address
        bounce_senders = [
            "mailer-daemon@",
            "postmaster@",
            "mail-daemon@",
        ]

        for sender in bounce_senders:
            if email_obj.from_address.startswith(sender):
                return True

        # Check content type for multipart/report
        content_type = email_obj.get_header("content-type", "").lower()
        if "multipart/report" in content_type:
            return True

        return False

    def _decode_header(self, header: str) -> str:
        """Decode a potentially encoded email header."""
        if not header:
            return ""

        try:
            decoded = decode_header(header)
            parts = []
            for content, charset in decoded:
                if isinstance(content, bytes):
                    charset = charset or "utf-8"
                    try:
                        parts.append(content.decode(charset))
                    except (UnicodeDecodeError, LookupError):
                        parts.append(content.decode("utf-8", errors="replace"))
                else:
                    parts.append(content)
            return "".join(parts)
        except Exception:
            return header

    def _extract_addresses(self, headers: list) -> list[str]:
        """Extract email addresses from header values."""
        addresses = []
        for header in headers:
            if header:
                parsed = getaddresses([header])
                addresses.extend(addr.lower() for name, addr in parsed if addr)
        return addresses

    def _extract_body_and_attachments(
        self,
        message: EmailMessage
    ) -> tuple[str, str | None, list[EmailAttachment]]:
        """
        Extract body text, HTML, and attachments from message.

        Returns:
            Tuple of (body_text, body_html, attachments)
        """
        body_text = ""
        body_html = None
        attachments: list[EmailAttachment] = []

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Handle attachments
                if "attachment" in content_disposition:
                    attachment = self._extract_attachment(part)
                    if attachment:
                        attachments.append(attachment)
                # Handle inline attachments
                elif "inline" in content_disposition and part.get_filename():
                    attachment = self._extract_attachment(part, is_inline=True)
                    if attachment:
                        attachments.append(attachment)
                # Handle text parts
                elif content_type == "text/plain" and not body_text:
                    body_text = self._get_text_payload(part)
                elif content_type == "text/html" and not body_html:
                    body_html = self._get_text_payload(part)
        else:
            # Simple message
            content_type = message.get_content_type()
            if content_type == "text/plain":
                body_text = self._get_text_payload(message)
            elif content_type == "text/html":
                body_html = self._get_text_payload(message)
                body_text = self._html_to_text(body_html)

        # If no plain text but have HTML, convert
        if not body_text and body_html:
            body_text = self._html_to_text(body_html)

        return body_text, body_html, attachments

    def _extract_attachment(
        self,
        part,
        is_inline: bool = False
    ) -> EmailAttachment | None:
        """Extract attachment from message part."""
        filename = part.get_filename()
        if not filename:
            filename = "attachment"

        filename = self._decode_header(filename)
        content_type = part.get_content_type()

        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return None

            size_bytes = len(payload)

            # Only load content if under size limit
            content = payload if size_bytes <= self._max_attachment_bytes else None

            return EmailAttachment(
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                content=content,
                content_id=part.get("Content-ID"),
                is_inline=is_inline,
            )
        except Exception:
            return None

    def _get_text_payload(self, part) -> str:
        """Safely extract text payload from message part."""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return ""

            # Try to get charset from content-type
            charset = part.get_content_charset() or "utf-8"

            try:
                return payload.decode(charset)
            except (UnicodeDecodeError, LookupError):
                return payload.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _html_to_text(self, html: str) -> str:
        """
        Simple HTML to text conversion.

        For production, consider using a library like html2text.
        """
        if not html:
            return ""

        # Remove style and script tags
        text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)

        # Convert line breaks
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)

        # Remove remaining tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # Clean up whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = text.strip()

        return text
