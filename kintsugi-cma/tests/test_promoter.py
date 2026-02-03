"""Tests for kintsugi.kintsugi_engine.promoter (Phase 3, Stream 3A)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kintsugi.kintsugi_engine.promoter import (
    GoldenTrace,
    Promoter,
    PromoterConfig,
    PromotionAction,
)
from kintsugi.kintsugi_engine.verifier import (
    VerificationResult,
    VerifierVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vresult(verdict: VerifierVerdict, swei: float = 0.05) -> VerificationResult:
    return VerificationResult(
        verdict=verdict,
        safety_passed=True,
        quality_score=0.9,
        alignment_score=1.0,
        swei_divergence=swei,
        rationale="test",
        checked_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Enum and dataclass basics
# ---------------------------------------------------------------------------

class TestPromotionAction:
    def test_values(self):
        assert PromotionAction.PROMOTE == "PROMOTE"
        assert PromotionAction.ROLLBACK == "ROLLBACK"
        assert PromotionAction.EXTEND == "EXTEND"
        assert PromotionAction.ESCALATE == "ESCALATE"


class TestGoldenTrace:
    def test_fields(self):
        t = GoldenTrace(
            trace_id="t1",
            shadow_id="s1",
            modification={"k": "v"},
            verdict=VerifierVerdict.APPROVE,
            swei_divergence=0.01,
            promoted_at=datetime.now(timezone.utc),
            config_before={"a": 1},
            config_after={"a": 1, "k": "v"},
        )
        assert t.trace_id == "t1"
        assert t.shadow_id == "s1"


class TestPromoterConfig:
    def test_defaults(self):
        cfg = PromoterConfig()
        assert cfg.max_rollback_depth == 10
        assert cfg.log_golden_traces is True


# ---------------------------------------------------------------------------
# promote()
# ---------------------------------------------------------------------------

class TestPromote:
    def test_approve_promotes(self):
        p = Promoter()
        action, cfg = p.promote(
            "s1", {"new": True}, _vresult(VerifierVerdict.APPROVE), {"old": 1}
        )
        assert action == PromotionAction.PROMOTE
        assert cfg["new"] is True
        assert cfg["old"] == 1

    def test_reject_rollback(self):
        p = Promoter()
        action, cfg = p.promote(
            "s1", {"new": True}, _vresult(VerifierVerdict.REJECT), {"old": 1}
        )
        assert action == PromotionAction.ROLLBACK
        assert cfg == {"old": 1}
        assert "new" not in cfg

    def test_extend(self):
        p = Promoter()
        action, cfg = p.promote(
            "s1", {"x": 1}, _vresult(VerifierVerdict.EXTEND), {"old": 1}
        )
        assert action == PromotionAction.EXTEND
        assert cfg == {"old": 1}

    def test_escalate(self):
        p = Promoter()
        action, cfg = p.promote(
            "s1", {"x": 1}, _vresult(VerifierVerdict.ESCALATE), {"old": 1}
        )
        assert action == PromotionAction.ESCALATE
        assert cfg == {"old": 1}

    def test_approve_stores_trace(self):
        p = Promoter()
        p.promote("s1", {"k": "v"}, _vresult(VerifierVerdict.APPROVE), {"a": 1})
        traces = p.get_golden_traces()
        assert len(traces) == 1
        assert traces[0].shadow_id == "s1"
        assert traces[0].verdict == VerifierVerdict.APPROVE

    def test_reject_does_not_store_trace(self):
        p = Promoter()
        p.promote("s1", {"k": "v"}, _vresult(VerifierVerdict.REJECT), {"a": 1})
        assert len(p.get_golden_traces()) == 0


# ---------------------------------------------------------------------------
# rollback()
# ---------------------------------------------------------------------------

class TestRollback:
    def _setup_promoter(self, n=3):
        p = Promoter()
        configs = [{"version": i} for i in range(n + 1)]
        for i in range(n):
            p.promote(
                f"s{i}",
                {"version": i + 1},
                _vresult(VerifierVerdict.APPROVE),
                configs[i],
            )
        return p

    def test_rollback_1(self):
        p = self._setup_promoter(3)
        cfg = p.rollback(1)
        assert cfg["version"] == 2

    def test_rollback_n(self):
        p = self._setup_promoter(3)
        cfg = p.rollback(3)
        assert cfg["version"] == 0

    def test_rollback_insufficient_history(self):
        p = self._setup_promoter(2)
        with pytest.raises(ValueError, match="Cannot rollback"):
            p.rollback(5)

    def test_rollback_zero_raises(self):
        p = self._setup_promoter(1)
        with pytest.raises(ValueError, match="steps must be >= 1"):
            p.rollback(0)


# ---------------------------------------------------------------------------
# get_golden_traces()
# ---------------------------------------------------------------------------

class TestGetGoldenTraces:
    def test_order(self):
        p = Promoter()
        for i in range(5):
            p.promote(f"s{i}", {"i": i}, _vresult(VerifierVerdict.APPROVE), {"v": i})
        traces = p.get_golden_traces()
        assert len(traces) == 5
        assert traces[0].shadow_id == "s0"
        assert traces[4].shadow_id == "s4"


# ---------------------------------------------------------------------------
# _apply_modification() deep merge
# ---------------------------------------------------------------------------

class TestApplyModification:
    def test_deep_merge(self):
        p = Promoter()
        config = {"a": {"b": 1, "c": 2}, "d": 3}
        mod = {"a": {"b": 99, "e": 5}, "f": 6}
        result = p._apply_modification(config, mod)
        assert result["a"]["b"] == 99
        assert result["a"]["c"] == 2
        assert result["a"]["e"] == 5
        assert result["d"] == 3
        assert result["f"] == 6

    def test_overwrite_non_dict(self):
        p = Promoter()
        config = {"a": "string"}
        mod = {"a": {"nested": True}}
        result = p._apply_modification(config, mod)
        assert result["a"] == {"nested": True}


# ---------------------------------------------------------------------------
# max_rollback_depth enforcement
# ---------------------------------------------------------------------------

class TestMaxRollbackDepth:
    def test_enforced(self):
        p = Promoter(config=PromoterConfig(max_rollback_depth=3))
        for i in range(10):
            p.promote(f"s{i}", {"i": i}, _vresult(VerifierVerdict.APPROVE), {"v": i})
        traces = p.get_golden_traces()
        assert len(traces) == 3
        # Only the last 3 remain
        assert traces[0].shadow_id == "s7"
