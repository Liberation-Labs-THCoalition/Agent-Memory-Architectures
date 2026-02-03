"""Tests for kintsugi.kintsugi_engine.verifier (Phase 3, Stream 3A)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kintsugi.kintsugi_engine.verifier import (
    VerificationResult,
    Verifier,
    VerifierConfig,
    VerifierVerdict,
)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class TestVerifierVerdict:
    def test_values(self):
        assert VerifierVerdict.APPROVE == "APPROVE"
        assert VerifierVerdict.REJECT == "REJECT"
        assert VerifierVerdict.EXTEND == "EXTEND"
        assert VerifierVerdict.ESCALATE == "ESCALATE"


class TestVerificationResult:
    def test_frozen(self):
        from datetime import datetime, timezone
        vr = VerificationResult(
            verdict=VerifierVerdict.APPROVE,
            safety_passed=True,
            quality_score=0.9,
            alignment_score=1.0,
            swei_divergence=0.05,
            rationale="ok",
            checked_at=datetime.now(timezone.utc),
        )
        with pytest.raises(FrozenInstanceError):
            vr.verdict = VerifierVerdict.REJECT  # type: ignore


class TestVerifierConfig:
    def test_defaults(self):
        cfg = VerifierConfig()
        assert cfg.divergence_threshold == 0.15
        assert cfg.min_quality_score == 0.6
        assert cfg.safety_weight == 0.4
        assert cfg.quality_weight == 0.3
        assert cfg.alignment_weight == 0.3
        assert cfg.extend_window_turns == 5


# ---------------------------------------------------------------------------
# Verifier.verify()
# ---------------------------------------------------------------------------

def _make_verifier(threshold=0.15, min_quality=0.6, invariant_checker=None):
    cfg = VerifierConfig(divergence_threshold=threshold, min_quality_score=min_quality)
    return Verifier(config=cfg, invariant_checker=invariant_checker)


class TestVerifyIdentical:
    def test_identical_outputs_approve(self):
        v = _make_verifier()
        outputs = [{"text": "hello world", "tool": "search"}]
        result = v.verify(outputs, list(outputs))
        assert result.verdict == VerifierVerdict.APPROVE
        assert result.swei_divergence < 0.15
        assert result.safety_passed is True


class TestVerifyDivergent:
    def test_very_different_escalate(self):
        v = _make_verifier(threshold=0.05)
        primary = [{"a": "short"}]
        shadow = [{"x": "y"}, {"z": "w"}, {"q": "r" * 200}]
        result = v.verify(primary, shadow)
        assert result.verdict in (VerifierVerdict.ESCALATE, VerifierVerdict.EXTEND)
        assert result.swei_divergence > 0.05


class TestVerifyInvariantFailure:
    def test_invariant_reject(self):
        checker = MagicMock()
        checker.check_all.return_value = SimpleNamespace(
            all_passed=False, failures=["rule_1"]
        )
        v = _make_verifier(invariant_checker=checker)
        ctx = object()
        result = v.verify([{"a": 1}], [{"a": 1}], invariant_context=ctx)
        assert result.verdict == VerifierVerdict.REJECT
        assert result.safety_passed is False
        assert "Invariant failures" in result.rationale
        checker.check_all.assert_called_once_with(ctx)


class TestVerifyLowQuality:
    def test_low_quality_reject(self):
        v = _make_verifier(threshold=1.0, min_quality=0.99)
        # Completely different words -> low Jaccard
        primary = [{"text": "alpha beta gamma"}]
        shadow = [{"text": "delta epsilon zeta"}]
        result = v.verify(primary, shadow)
        assert result.verdict == VerifierVerdict.REJECT
        assert result.quality_score < 0.99


# ---------------------------------------------------------------------------
# _compute_swei
# ---------------------------------------------------------------------------

class TestComputeSwei:
    def test_empty_inputs(self):
        v = _make_verifier()
        assert v._compute_swei([], []) == 0.0

    def test_identical_dicts(self):
        v = _make_verifier()
        d = [{"a": 1, "b": 2}]
        swei = v._compute_swei(d, list(d))
        assert swei < 0.01

    def test_different_dicts(self):
        v = _make_verifier()
        swei = v._compute_swei([{"a": 1}], [{"x": "long" * 100}])
        assert swei > 0.1


# ---------------------------------------------------------------------------
# _compute_quality
# ---------------------------------------------------------------------------

class TestComputeQuality:
    def test_both_empty(self):
        v = _make_verifier()
        assert v._compute_quality([], []) == 1.0

    def test_one_empty(self):
        v = _make_verifier()
        assert v._compute_quality([{"a": 1}], []) == 0.0
        assert v._compute_quality([], [{"a": 1}]) == 0.0

    def test_identical(self):
        v = _make_verifier()
        d = [{"text": "hello world"}]
        assert v._compute_quality(d, list(d)) == 1.0

    def test_partial_overlap(self):
        v = _make_verifier()
        q = v._compute_quality([{"text": "a b c"}], [{"text": "b c d"}])
        assert 0.0 < q < 1.0


# ---------------------------------------------------------------------------
# _compute_alignment
# ---------------------------------------------------------------------------

class TestComputeAlignment:
    def test_no_bdi(self):
        v = _make_verifier()
        assert v._compute_alignment([{"a": 1}], None) == 1.0

    def test_no_outputs(self):
        v = _make_verifier()
        assert v._compute_alignment([], {"goal": "test"}) == 1.0

    def test_with_bdi(self):
        v = _make_verifier()
        outputs = [{"text": "deploy the service quickly"}]
        bdi = {"desire": "deploy service", "belief": "infrastructure ready"}
        score = v._compute_alignment(outputs, bdi)
        assert 0.0 < score <= 1.0

    def test_no_overlap(self):
        v = _make_verifier()
        outputs = [{"text": "alpha beta"}]
        bdi = {"desire": "gamma delta"}
        score = v._compute_alignment(outputs, bdi)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Boundary thresholds
# ---------------------------------------------------------------------------

class TestVerdictBoundaries:
    def test_at_threshold_approve(self):
        """SWEI exactly at threshold should EXTEND (> threshold)."""
        v = _make_verifier(threshold=0.0, min_quality=0.0)
        # Both empty -> swei=0 -> not > 0 -> APPROVE
        result = v.verify([], [])
        assert result.verdict == VerifierVerdict.APPROVE

    def test_escalate_above_2x(self):
        v = _make_verifier(threshold=0.01)
        # Very different -> high swei
        result = v.verify(
            [{"a": "x"}],
            [{"z": "y" * 500}, {"w": "q" * 500}],
        )
        if result.swei_divergence > 0.02:
            assert result.verdict == VerifierVerdict.ESCALATE
