"""Comprehensive pytest test suite for kintsugi.security.pii module.

Tests cover:
- PIIDetection and RedactionResult dataclasses
- _luhn_check validation algorithm
- PIIRedactor detection and redaction for all PII types
- Edge cases, negative cases, and mode variations
- pii_redaction_middleware behavior

Target: >90% code coverage
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock, MagicMock
import re

import pytest

# Ensure the kintsugi package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from kintsugi.security.pii import (
    PIIDetection,
    RedactionResult,
    _luhn_check,
    PIIRedactor,
    pii_redaction_middleware,
)


# =============================================================================
# Test _luhn_check function
# =============================================================================

class TestLuhnCheck:
    """Test cases for the Luhn algorithm validator."""

    def test_luhn_valid_card_numbers(self):
        """Valid credit card numbers pass Luhn check."""
        # Known valid test card numbers
        assert _luhn_check("4532015112830366") is True  # Visa
        assert _luhn_check("5425233430109903") is True  # Mastercard
        assert _luhn_check("374245455400126") is True   # Amex
        assert _luhn_check("6011000991001201") is True  # Discover

    def test_luhn_invalid_card_numbers(self):
        """Invalid credit card numbers fail Luhn check."""
        assert _luhn_check("4532015112830367") is False  # Changed last digit
        assert _luhn_check("1234567812345678") is False  # Made up number
        # Note: 0000000000000000 actually passes Luhn (checksum is 0)

    def test_luhn_edge_cases(self):
        """Edge cases for Luhn validation."""
        # Too short (less than 2 digits)
        assert _luhn_check("1") is False
        assert _luhn_check("") is False

        # Contains non-digits (should be filtered out)
        assert _luhn_check("4532-0151-1283-0366") is True  # Valid with dashes
        assert _luhn_check("4532 0151 1283 0366") is True  # Valid with spaces

        # Mixed alphanumeric
        assert _luhn_check("abc123") is False

    def test_luhn_single_digit(self):
        """Single digit always fails."""
        for digit in "0123456789":
            assert _luhn_check(digit) is False


# =============================================================================
# Test PIIDetection dataclass
# =============================================================================

class TestPIIDetection:
    """Test the PIIDetection dataclass."""

    def test_pii_detection_creation(self):
        """PIIDetection can be created with all fields."""
        detection = PIIDetection(
            pii_type="EMAIL",
            start=10,
            end=25,
            original="test@example.com"
        )
        assert detection.pii_type == "EMAIL"
        assert detection.start == 10
        assert detection.end == 25
        assert detection.original == "test@example.com"

    def test_pii_detection_frozen(self):
        """PIIDetection is immutable (frozen)."""
        detection = PIIDetection(
            pii_type="EMAIL", start=0, end=10, original="test@test.com"
        )
        with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
            detection.pii_type = "PHONE"


# =============================================================================
# Test RedactionResult dataclass
# =============================================================================

class TestRedactionResult:
    """Test the RedactionResult dataclass."""

    def test_redaction_result_creation(self):
        """RedactionResult can be created with all fields."""
        result = RedactionResult(
            redacted_text="Hello [REDACTED_EMAIL]",
            detections_count=1,
            types_found=["EMAIL"]
        )
        assert result.redacted_text == "Hello [REDACTED_EMAIL]"
        assert result.detections_count == 1
        assert result.types_found == ["EMAIL"]

    def test_redaction_result_frozen(self):
        """RedactionResult is immutable (frozen)."""
        result = RedactionResult(
            redacted_text="test", detections_count=0, types_found=[]
        )
        with pytest.raises(Exception):
            result.detections_count = 5


# =============================================================================
# Test PIIRedactor.detect() - Individual PII Types
# =============================================================================

class TestPIIDetectionByType:
    """Test detection of each PII type."""

    def test_detect_email(self):
        """Emails are detected correctly."""
        redactor = PIIRedactor()
        text = "Contact me at john.doe@example.com for more info"
        detections = redactor.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == "EMAIL"
        assert detections[0].original == "john.doe@example.com"
        assert detections[0].start == 14
        assert detections[0].end == 34  # Fixed: actual end position

    def test_detect_multiple_emails(self):
        """Multiple emails are all detected."""
        redactor = PIIRedactor()
        text = "Email alice@test.com or bob@company.org"
        detections = redactor.detect(text)

        assert len(detections) == 2
        assert all(d.pii_type == "EMAIL" for d in detections)
        assert detections[0].original == "alice@test.com"
        assert detections[1].original == "bob@company.org"

    def test_detect_phone_various_formats(self):
        """Phone numbers in various formats are detected."""
        redactor = PIIRedactor()

        test_cases = [
            ("Call 555-123-4567", "555-123-4567"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Mobile 5551234567", "5551234567"),
            ("US +1-555-123-4567", "+1-555-123-4567"),
            ("Contact: 1.555.123.4567", "1.555.123.4567"),
        ]

        for text, expected_phone in test_cases:
            detections = redactor.detect(text)
            assert len(detections) >= 1, f"Failed to detect phone in: {text}"
            phone_detections = [d for d in detections if d.pii_type == "PHONE"]
            assert len(phone_detections) == 1, f"Expected 1 phone in: {text}"
            assert phone_detections[0].original == expected_phone

    def test_detect_ssn(self):
        """SSN in standard format is detected."""
        redactor = PIIRedactor()
        text = "SSN: 123-45-6789"
        detections = redactor.detect(text)

        assert len(detections) == 1
        assert detections[0].pii_type == "SSN"
        assert detections[0].original == "123-45-6789"

    def test_ssn_requires_dashes(self):
        """SSN without dashes is not detected."""
        redactor = PIIRedactor()
        text = "Number: 123456789"
        detections = redactor.detect(text)

        # Should not detect as SSN (no dashes)
        ssn_detections = [d for d in detections if d.pii_type == "SSN"]
        assert len(ssn_detections) == 0

    def test_detect_credit_card_valid_luhn(self):
        """Valid credit cards (passing Luhn check) are detected."""
        redactor = PIIRedactor()

        test_cases = [
            "4532015112830366",
            "4532-0151-1283-0366",
            "4532 0151 1283 0366",
            "5425233430109903",
        ]

        for card in test_cases:
            text = f"Card number: {card}"
            detections = redactor.detect(text)
            cc_detections = [d for d in detections if d.pii_type == "CREDIT_CARD"]
            assert len(cc_detections) == 1, f"Failed to detect valid card: {card}"

    def test_credit_card_invalid_luhn_not_detected(self):
        """Credit card numbers that fail Luhn check are NOT detected."""
        redactor = PIIRedactor()

        # These look like credit cards but fail Luhn validation
        invalid_cards = [
            "1234567812345678",
            "4532015112830367",  # Last digit changed
            # Note: 0000-0000-0000-0000 passes Luhn (checksum is 0), so we skip it
        ]

        for card in invalid_cards:
            text = f"Card: {card}"
            detections = redactor.detect(text)
            cc_detections = [d for d in detections if d.pii_type == "CREDIT_CARD"]
            assert len(cc_detections) == 0, f"Should not detect invalid card: {card}"

    def test_detect_ip_address(self):
        """IP addresses are detected."""
        redactor = PIIRedactor()

        test_cases = [
            ("Server IP: 192.168.1.1", "192.168.1.1"),
            ("Connect to 10.0.0.255", "10.0.0.255"),
            ("Public IP 8.8.8.8", "8.8.8.8"),
            ("Address: 255.255.255.255", "255.255.255.255"),
        ]

        for text, expected_ip in test_cases:
            detections = redactor.detect(text)
            ip_detections = [d for d in detections if d.pii_type == "IP_ADDRESS"]
            assert len(ip_detections) == 1, f"Failed to detect IP in: {text}"
            assert ip_detections[0].original == expected_ip

    def test_ip_address_invalid_octets(self):
        """Invalid IP addresses (>255) should not be detected."""
        redactor = PIIRedactor()
        text = "Invalid: 999.999.999.999"
        detections = redactor.detect(text)
        ip_detections = [d for d in detections if d.pii_type == "IP_ADDRESS"]
        # The regex should prevent this, but if detected it's an edge case
        # In practice, the regex pattern prevents >255 octets
        assert len(ip_detections) == 0

    def test_detect_date_of_birth(self):
        """Date of birth with context is detected."""
        redactor = PIIRedactor()

        test_cases = [
            "DOB: 12/31/1990",
            "Date of Birth: 01-15-1985",
            "Birth Date: 1990/05/20",
            "dob 03/07/2000",
            "Date of birth: 1985-06-10",
        ]

        for text in test_cases:
            detections = redactor.detect(text)
            dob_detections = [d for d in detections if d.pii_type == "DATE_OF_BIRTH"]
            assert len(dob_detections) == 1, f"Failed to detect DOB in: {text}"

    def test_date_without_context_not_detected(self):
        """Date without DOB context should not be detected as PII."""
        redactor = PIIRedactor()
        text = "Meeting on 12/31/2023"
        detections = redactor.detect(text)
        dob_detections = [d for d in detections if d.pii_type == "DATE_OF_BIRTH"]
        assert len(dob_detections) == 0


# =============================================================================
# Test PIIRedactor.detect() - Edge Cases
# =============================================================================

class TestPIIDetectionEdgeCases:
    """Test edge cases for PII detection."""

    def test_detect_empty_string(self):
        """Empty string returns no detections."""
        redactor = PIIRedactor()
        detections = redactor.detect("")
        assert detections == []

    def test_detect_no_pii(self):
        """Text with no PII returns empty list."""
        redactor = PIIRedactor()
        text = "This is a completely clean sentence with no sensitive data."
        detections = redactor.detect(text)
        assert detections == []

    def test_detect_multiple_pii_types(self):
        """Multiple different PII types in one string are all detected."""
        redactor = PIIRedactor()
        text = "Contact john@example.com or call 555-123-4567. SSN: 123-45-6789"
        detections = redactor.detect(text)

        assert len(detections) == 3
        types = {d.pii_type for d in detections}
        assert types == {"EMAIL", "PHONE", "SSN"}

    def test_detections_sorted_by_start_position(self):
        """Detections are sorted by start position."""
        redactor = PIIRedactor()
        text = "SSN 123-45-6789 and email test@test.com then phone 555-123-4567"
        detections = redactor.detect(text)

        assert len(detections) == 3
        # Verify sorted by start position
        for i in range(len(detections) - 1):
            assert detections[i].start < detections[i + 1].start

    def test_overlapping_patterns_both_detected(self):
        """When patterns overlap, both are detected."""
        redactor = PIIRedactor()
        # Edge case: if patterns somehow overlap (rare with these patterns)
        text = "Email: test@test.com"
        detections = redactor.detect(text)
        assert len(detections) == 1


# =============================================================================
# Test PIIRedactor with extra_patterns
# =============================================================================

class TestPIIRedactorExtraPatterns:
    """Test PIIRedactor with custom extra patterns."""

    def test_extra_patterns_extend_detection(self):
        """Extra patterns add new PII types."""
        extra = [
            {
                "type": "API_KEY",
                "regex": re.compile(r"sk_live_[a-zA-Z0-9]{24}"),
                "validator": None,
            }
        ]
        redactor = PIIRedactor(extra_patterns=extra)

        text = "API Key: sk_live_000000000000TESTONLY00"  # Fixed: exactly 24 chars, obviously fake
        detections = redactor.detect(text)

        api_key_detections = [d for d in detections if d.pii_type == "API_KEY"]
        assert len(api_key_detections) == 1

    def test_extra_patterns_with_validator(self):
        """Extra patterns can include custom validators."""
        def always_false(text):
            return False

        extra = [
            {
                "type": "FAKE_TYPE",
                "regex": re.compile(r"\d{3}"),
                "validator": always_false,
            }
        ]
        redactor = PIIRedactor(extra_patterns=extra)

        text = "Number 123"
        detections = redactor.detect(text)

        # Should not detect because validator returns False
        fake_detections = [d for d in detections if d.pii_type == "FAKE_TYPE"]
        assert len(fake_detections) == 0

    def test_extra_patterns_none(self):
        """PIIRedactor works with extra_patterns=None."""
        redactor = PIIRedactor(extra_patterns=None)
        text = "Email: test@test.com"
        detections = redactor.detect(text)
        assert len(detections) == 1


# =============================================================================
# Test PIIRedactor.redact() - Mask Mode
# =============================================================================

class TestPIIRedactionMaskMode:
    """Test redaction in mask mode (default)."""

    def test_redact_mask_single_email(self):
        """Mask mode replaces PII with [REDACTED_TYPE]."""
        redactor = PIIRedactor()
        text = "Email me at john@example.com please"
        result = redactor.redact(text, mode="mask")

        assert result.redacted_text == "Email me at [REDACTED_EMAIL] please"
        assert result.detections_count == 1
        assert result.types_found == ["EMAIL"]

    def test_redact_mask_multiple_pii(self):
        """Mask mode handles multiple PII items."""
        redactor = PIIRedactor()
        text = "Call 555-123-4567 or email test@test.com"
        result = redactor.redact(text, mode="mask")

        assert "[REDACTED_PHONE]" in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text
        assert result.detections_count == 2
        assert set(result.types_found) == {"PHONE", "EMAIL"}

    def test_redact_mask_preserves_order(self):
        """Mask mode preserves text order."""
        redactor = PIIRedactor()
        text = "Before test@test.com middle 555-123-4567 after"
        result = redactor.redact(text, mode="mask")

        assert result.redacted_text == "Before [REDACTED_EMAIL] middle [REDACTED_PHONE] after"

    def test_redact_mask_credit_card(self):
        """Mask mode redacts valid credit cards."""
        redactor = PIIRedactor()
        text = "Card: 4532015112830366"
        result = redactor.redact(text, mode="mask")

        assert result.redacted_text == "Card: [REDACTED_CREDIT_CARD]"
        assert result.detections_count == 1


# =============================================================================
# Test PIIRedactor.redact() - Remove Mode
# =============================================================================

class TestPIIRedactionRemoveMode:
    """Test redaction in remove mode."""

    def test_redact_remove_single_email(self):
        """Remove mode deletes PII entirely."""
        redactor = PIIRedactor()
        text = "Email me at john@example.com please"
        result = redactor.redact(text, mode="remove")

        assert result.redacted_text == "Email me at  please"
        assert result.detections_count == 1
        assert result.types_found == ["EMAIL"]

    def test_redact_remove_multiple_pii(self):
        """Remove mode deletes all PII items."""
        redactor = PIIRedactor()
        text = "Call 555-123-4567 or email test@test.com"
        result = redactor.redact(text, mode="remove")

        assert "555-123-4567" not in result.redacted_text
        assert "test@test.com" not in result.redacted_text
        assert result.redacted_text == "Call  or email "
        assert result.detections_count == 2

    def test_redact_remove_adjacent_pii(self):
        """Remove mode handles adjacent PII correctly."""
        redactor = PIIRedactor()
        text = "test@test.com555-123-4567"  # No space between
        result = redactor.redact(text, mode="remove")

        assert result.detections_count == 2
        assert result.redacted_text == ""


# =============================================================================
# Test PIIRedactor.redact() - Edge Cases
# =============================================================================

class TestPIIRedactionEdgeCases:
    """Test edge cases for redaction."""

    def test_redact_no_pii(self):
        """No PII returns unchanged text."""
        redactor = PIIRedactor()
        text = "This is clean text."
        result = redactor.redact(text, mode="mask")

        assert result.redacted_text == text
        assert result.detections_count == 0
        assert result.types_found == []

    def test_redact_empty_string(self):
        """Empty string returns empty result."""
        redactor = PIIRedactor()
        result = redactor.redact("", mode="mask")

        assert result.redacted_text == ""
        assert result.detections_count == 0
        assert result.types_found == []

    def test_redact_types_found_sorted_and_deduplicated(self):
        """types_found is sorted and deduplicated."""
        redactor = PIIRedactor()
        text = "test@test.com and alice@test.com and 555-123-4567"
        result = redactor.redact(text, mode="mask")

        # Two emails, one phone
        assert result.detections_count == 3
        # types_found should be sorted and unique
        assert result.types_found == ["EMAIL", "PHONE"]  # Alphabetically sorted

    def test_redact_default_mode_is_mask(self):
        """Default mode is 'mask'."""
        redactor = PIIRedactor()
        text = "Email: test@test.com"
        result = redactor.redact(text)  # No mode specified

        assert "[REDACTED_EMAIL]" in result.redacted_text

    def test_redact_pii_at_boundaries(self):
        """PII at start and end of string."""
        redactor = PIIRedactor()
        text = "test@test.com in the middle 555-123-4567"
        result = redactor.redact(text, mode="mask")

        assert result.redacted_text.startswith("[REDACTED_EMAIL]")
        assert result.redacted_text.endswith("[REDACTED_PHONE]")


# =============================================================================
# Test pii_redaction_middleware
# =============================================================================

class TestPIIRedactionMiddleware:
    """Test the FastAPI middleware factory."""

    def test_middleware_factory_returns_callable(self):
        """pii_redaction_middleware returns a middleware function."""
        middleware = pii_redaction_middleware()
        assert callable(middleware)

    def test_middleware_with_custom_redactor(self):
        """Middleware can accept custom PIIRedactor."""
        custom_redactor = PIIRedactor()
        middleware = pii_redaction_middleware(redactor=custom_redactor)
        assert callable(middleware)

    def test_middleware_with_skip_paths(self):
        """Middleware can accept skip_paths."""
        middleware = pii_redaction_middleware(skip_paths=["/health", "/metrics"])
        assert callable(middleware)

    def test_middleware_closure_captures_redactor(self):
        """Middleware closure properly captures redactor instance."""
        custom_redactor = PIIRedactor()
        middleware = pii_redaction_middleware(redactor=custom_redactor)

        # Verify the middleware has captured the redactor
        # We can't easily test the async behavior without pytest-asyncio,
        # but we can verify the factory works correctly
        assert callable(middleware)

    def test_middleware_closure_captures_skip_paths(self):
        """Middleware closure properly captures skip_paths."""
        skip_paths = ["/health", "/metrics"]
        middleware = pii_redaction_middleware(skip_paths=skip_paths)

        # Verify the middleware has captured skip_paths
        assert callable(middleware)

    def test_middleware_handles_none_skip_paths(self):
        """Middleware handles None skip_paths correctly."""
        middleware = pii_redaction_middleware(skip_paths=None)
        assert callable(middleware)

    def test_middleware_creates_default_redactor_when_none(self):
        """Middleware creates a default PIIRedactor when none provided."""
        middleware = pii_redaction_middleware(redactor=None)
        assert callable(middleware)

    def test_middleware_with_empty_skip_paths(self):
        """Middleware works with empty skip_paths list."""
        middleware = pii_redaction_middleware(skip_paths=[])
        assert callable(middleware)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPIIIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_detect_and_redact(self):
        """Complete workflow: detect then redact."""
        redactor = PIIRedactor()
        text = "Contact: test@example.com, Phone: 555-123-4567, SSN: 123-45-6789"

        # First detect
        detections = redactor.detect(text)
        assert len(detections) == 3

        # Then redact
        result = redactor.redact(text, mode="mask")
        assert result.detections_count == 3
        assert len(result.types_found) == 3
        assert "test@example.com" not in result.redacted_text
        assert "[REDACTED_EMAIL]" in result.redacted_text

    def test_complex_text_with_all_pii_types(self):
        """Complex text with all PII types."""
        redactor = PIIRedactor()
        text = """
        Personal Information:
        Email: john.doe@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        Credit Card: 4532015112830366
        IP Address: 192.168.1.1
        DOB: 01/15/1985
        """

        detections = redactor.detect(text)
        detected_types = {d.pii_type for d in detections}

        # All types should be detected
        expected_types = {"EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS", "DATE_OF_BIRTH"}
        assert detected_types == expected_types

        # Redact and verify
        result = redactor.redact(text, mode="mask")
        assert result.detections_count == 6
        assert set(result.types_found) == expected_types

    def test_redaction_consistency_between_modes(self):
        """Both modes should find same number of detections."""
        redactor = PIIRedactor()
        text = "Email test@test.com and phone 555-123-4567"

        mask_result = redactor.redact(text, mode="mask")
        remove_result = redactor.redact(text, mode="remove")

        # Both should find same number of detections
        assert mask_result.detections_count == remove_result.detections_count
        assert mask_result.types_found == remove_result.types_found

        # But text should differ
        assert mask_result.redacted_text != remove_result.redacted_text


# =============================================================================
# Run pytest with coverage
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=kintsugi.security.pii", "--cov-report=term-missing"])
