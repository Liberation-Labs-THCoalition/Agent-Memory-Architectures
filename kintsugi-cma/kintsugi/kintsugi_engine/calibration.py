"""Calibration & Replay engine for Kintsugi CMA -- Phase 3 Stream 3B.

Tracks agreement between original and replayed verification verdicts,
detects verifier drift over time, and produces calibration health reports.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class DriftDirection(Enum):
    STABLE = "STABLE"
    MORE_PERMISSIVE = "MORE_PERMISSIVE"
    MORE_CONSERVATIVE = "MORE_CONSERVATIVE"
    INCONSISTENT = "INCONSISTENT"


@dataclass
class CalibrationRecord:
    record_id: str
    original_verdict: str
    replayed_verdict: str
    swei_original: float
    swei_replayed: float
    agreement: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CalibrationConfig:
    min_cycles_before_calibration: int = 50
    consistency_threshold: float = 0.8
    lookback_window: int = 100


@dataclass(frozen=True)
class CalibrationReport:
    total_records: int
    agreement_rate: float
    drift_direction: DriftDirection
    permissive_count: int
    conservative_count: int
    report_timestamp: datetime
    is_healthy: bool


# Verdicts ordered by strictness (most lenient first)
_STRICTNESS = {"APPROVE": 0, "EXTEND": 1, "ESCALATE": 2, "REJECT": 3}


class CalibrationEngine:
    """Records replay comparisons and detects verifier drift."""

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.config = config or CalibrationConfig()
        self._records: List[CalibrationRecord] = []

    def record_replay(
        self,
        original_verdict: str,
        replayed_verdict: str,
        swei_original: float,
        swei_replayed: float,
    ) -> CalibrationRecord:
        agreement = original_verdict.upper() == replayed_verdict.upper()
        record = CalibrationRecord(
            record_id=uuid.uuid4().hex[:12],
            original_verdict=original_verdict,
            replayed_verdict=replayed_verdict,
            swei_original=swei_original,
            swei_replayed=swei_replayed,
            agreement=agreement,
        )
        self._records.append(record)
        return record

    def is_calibration_due(self, total_cycles: int) -> bool:
        return (
            total_cycles >= self.config.min_cycles_before_calibration
            and len(self._records) >= self.config.min_cycles_before_calibration
        )

    def generate_report(self) -> CalibrationReport:
        window = self._records[-self.config.lookback_window :]
        if not window:
            return CalibrationReport(
                total_records=0,
                agreement_rate=1.0,
                drift_direction=DriftDirection.STABLE,
                permissive_count=0,
                conservative_count=0,
                report_timestamp=datetime.now(timezone.utc),
                is_healthy=True,
            )

        agreed = sum(1 for r in window if r.agreement)
        agreement_rate = agreed / len(window)

        permissive_count = 0
        conservative_count = 0
        for r in window:
            if r.agreement:
                continue
            orig = _STRICTNESS.get(r.original_verdict.upper(), 1)
            repl = _STRICTNESS.get(r.replayed_verdict.upper(), 1)
            if repl < orig:
                permissive_count += 1  # replayed more lenient
            elif repl > orig:
                conservative_count += 1
            else:
                # Same strictness bucket but different verdict string -- treat as inconsistent
                permissive_count += 1
                conservative_count += 1

        if agreement_rate >= self.config.consistency_threshold:
            drift = DriftDirection.STABLE
        elif permissive_count > 0 and conservative_count == 0:
            drift = DriftDirection.MORE_PERMISSIVE
        elif conservative_count > 0 and permissive_count == 0:
            drift = DriftDirection.MORE_CONSERVATIVE
        else:
            drift = DriftDirection.INCONSISTENT

        return CalibrationReport(
            total_records=len(window),
            agreement_rate=agreement_rate,
            drift_direction=drift,
            permissive_count=permissive_count,
            conservative_count=conservative_count,
            report_timestamp=datetime.now(timezone.utc),
            is_healthy=agreement_rate >= self.config.consistency_threshold,
        )

    def get_records(self, limit: int = 50) -> List[CalibrationRecord]:
        return self._records[-limit:]

    def clear_records(self) -> None:
        self._records.clear()
