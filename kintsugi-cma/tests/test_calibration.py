"""Tests for kintsugi.kintsugi_engine.calibration -- Phase 3 Stream 3B."""

import dataclasses
from datetime import datetime, timezone

import pytest

from kintsugi.kintsugi_engine.calibration import (
    CalibrationConfig,
    CalibrationEngine,
    CalibrationRecord,
    CalibrationReport,
    DriftDirection,
    _STRICTNESS,
)


# --- DriftDirection enum ---

class TestDriftDirection:
    def test_values(self):
        assert DriftDirection.STABLE.value == "STABLE"
        assert DriftDirection.MORE_PERMISSIVE.value == "MORE_PERMISSIVE"
        assert DriftDirection.MORE_CONSERVATIVE.value == "MORE_CONSERVATIVE"
        assert DriftDirection.INCONSISTENT.value == "INCONSISTENT"

    def test_member_count(self):
        assert len(DriftDirection) == 4


# --- CalibrationRecord ---

class TestCalibrationRecord:
    def test_agreement_computation_true(self):
        r = CalibrationRecord(
            record_id="x",
            original_verdict="APPROVE",
            replayed_verdict="APPROVE",
            swei_original=0.8,
            swei_replayed=0.8,
            agreement=True,
        )
        assert r.agreement is True

    def test_fields(self):
        r = CalibrationRecord(
            record_id="x",
            original_verdict="APPROVE",
            replayed_verdict="REJECT",
            swei_original=0.8,
            swei_replayed=0.3,
            agreement=False,
        )
        assert r.original_verdict == "APPROVE"
        assert r.replayed_verdict == "REJECT"
        assert isinstance(r.timestamp, datetime)


# --- CalibrationConfig ---

class TestCalibrationConfig:
    def test_defaults(self):
        c = CalibrationConfig()
        assert c.min_cycles_before_calibration == 50
        assert c.consistency_threshold == 0.8
        assert c.lookback_window == 100


# --- CalibrationReport frozen ---

class TestCalibrationReport:
    def test_frozen(self):
        r = CalibrationReport(
            total_records=0,
            agreement_rate=1.0,
            drift_direction=DriftDirection.STABLE,
            permissive_count=0,
            conservative_count=0,
            report_timestamp=datetime.now(timezone.utc),
            is_healthy=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.total_records = 5


# --- CalibrationEngine ---

class TestCalibrationEngine:
    def test_record_replay_agreement(self):
        eng = CalibrationEngine()
        rec = eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        assert rec.agreement is True

    def test_record_replay_disagreement(self):
        eng = CalibrationEngine()
        rec = eng.record_replay("APPROVE", "REJECT", 0.8, 0.3)
        assert rec.agreement is False

    def test_record_replay_case_insensitive(self):
        eng = CalibrationEngine()
        rec = eng.record_replay("approve", "APPROVE", 0.8, 0.8)
        assert rec.agreement is True

    def test_is_calibration_due_below_threshold(self):
        eng = CalibrationEngine(CalibrationConfig(min_cycles_before_calibration=5))
        for _ in range(4):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        assert eng.is_calibration_due(total_cycles=10) is False

    def test_is_calibration_due_at_threshold(self):
        eng = CalibrationEngine(CalibrationConfig(min_cycles_before_calibration=5))
        for _ in range(5):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        assert eng.is_calibration_due(total_cycles=5) is True

    def test_is_calibration_due_cycles_below(self):
        eng = CalibrationEngine(CalibrationConfig(min_cycles_before_calibration=5))
        for _ in range(5):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        assert eng.is_calibration_due(total_cycles=3) is False

    def test_generate_report_no_records(self):
        eng = CalibrationEngine()
        report = eng.generate_report()
        assert report.total_records == 0
        assert report.agreement_rate == 1.0
        assert report.drift_direction == DriftDirection.STABLE
        assert report.is_healthy is True

    def test_generate_report_all_agreements(self):
        eng = CalibrationEngine(CalibrationConfig(lookback_window=10))
        for _ in range(5):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        report = eng.generate_report()
        assert report.agreement_rate == 1.0
        assert report.drift_direction == DriftDirection.STABLE
        assert report.is_healthy is True

    def test_generate_report_all_permissive(self):
        # Replayed is more lenient (lower strictness) than original
        eng = CalibrationEngine(CalibrationConfig(lookback_window=10, consistency_threshold=0.8))
        for _ in range(5):
            eng.record_replay("REJECT", "APPROVE", 0.3, 0.8)
        report = eng.generate_report()
        assert report.drift_direction == DriftDirection.MORE_PERMISSIVE
        assert report.permissive_count == 5
        assert report.conservative_count == 0
        assert report.is_healthy is False

    def test_generate_report_all_conservative(self):
        # Replayed is stricter (higher strictness) than original
        eng = CalibrationEngine(CalibrationConfig(lookback_window=10, consistency_threshold=0.8))
        for _ in range(5):
            eng.record_replay("APPROVE", "REJECT", 0.8, 0.3)
        report = eng.generate_report()
        assert report.drift_direction == DriftDirection.MORE_CONSERVATIVE
        assert report.conservative_count == 5
        assert report.permissive_count == 0

    def test_generate_report_mixed_disagreements(self):
        eng = CalibrationEngine(CalibrationConfig(lookback_window=10, consistency_threshold=0.8))
        eng.record_replay("REJECT", "APPROVE", 0.3, 0.8)  # permissive
        eng.record_replay("APPROVE", "REJECT", 0.8, 0.3)  # conservative
        report = eng.generate_report()
        assert report.drift_direction == DriftDirection.INCONSISTENT

    def test_generate_report_respects_lookback_window(self):
        eng = CalibrationEngine(CalibrationConfig(lookback_window=3))
        # Old disagreements outside window
        for _ in range(5):
            eng.record_replay("APPROVE", "REJECT", 0.8, 0.3)
        # Recent agreements inside window
        for _ in range(3):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        report = eng.generate_report()
        assert report.total_records == 3
        assert report.agreement_rate == 1.0
        assert report.drift_direction == DriftDirection.STABLE

    def test_get_records_with_limit(self):
        eng = CalibrationEngine()
        for _ in range(10):
            eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        records = eng.get_records(limit=3)
        assert len(records) == 3

    def test_clear_records(self):
        eng = CalibrationEngine()
        eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)
        eng.clear_records()
        assert eng.get_records() == []

    def test_agreement_rate_accuracy(self):
        eng = CalibrationEngine(CalibrationConfig(lookback_window=10))
        eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)  # agree
        eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)  # agree
        eng.record_replay("APPROVE", "REJECT", 0.8, 0.3)   # disagree
        eng.record_replay("APPROVE", "APPROVE", 0.8, 0.8)  # agree
        report = eng.generate_report()
        assert report.agreement_rate == pytest.approx(0.75)

    def test_strictness_ordering(self):
        assert _STRICTNESS["APPROVE"] < _STRICTNESS["EXTEND"]
        assert _STRICTNESS["EXTEND"] < _STRICTNESS["ESCALATE"]
        assert _STRICTNESS["ESCALATE"] < _STRICTNESS["REJECT"]
