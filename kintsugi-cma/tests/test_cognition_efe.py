"""Tests for kintsugi.cognition.efe."""

from __future__ import annotations

import pytest

from kintsugi.cognition.efe import (
    EFEWeights,
    EFECalculator,
    EFEScore,
    GRANTS_WEIGHTS,
    FINANCE_WEIGHTS,
    COMMUNICATIONS_WEIGHTS,
    DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# EFEWeights validation
# ---------------------------------------------------------------------------

class TestEFEWeights:
    def test_valid_weights(self):
        w = EFEWeights(risk=0.5, ambiguity=0.3, epistemic=0.2)
        assert w.risk == 0.5

    def test_within_tolerance(self):
        # sum = 1.04, within 0.05 tolerance
        EFEWeights(risk=0.5, ambiguity=0.3, epistemic=0.24)

    def test_bad_sum_raises(self):
        with pytest.raises(ValueError, match="must sum to"):
            EFEWeights(risk=0.5, ambiguity=0.5, epistemic=0.5)

    def test_domain_profiles_exist(self):
        for w in (GRANTS_WEIGHTS, FINANCE_WEIGHTS, COMMUNICATIONS_WEIGHTS, DEFAULT_WEIGHTS):
            assert pytest.approx(w.risk + w.ambiguity + w.epistemic, abs=0.05) == 1.0


# ---------------------------------------------------------------------------
# compute_divergence
# ---------------------------------------------------------------------------

class TestComputeDivergence:
    calc = EFECalculator()

    def test_identical(self):
        d = self.calc.compute_divergence({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert d == 0.0

    def test_disjoint_keys(self):
        d = self.calc.compute_divergence({"a": 1}, {"b": 2})
        assert d == 1.0

    def test_numeric_difference(self):
        d = self.calc.compute_divergence({"x": 0}, {"x": 10})
        assert 0.0 < d <= 1.0

    def test_non_numeric_equal(self):
        d = self.calc.compute_divergence({"k": "hello"}, {"k": "hello"})
        assert d == 0.0

    def test_non_numeric_different(self):
        d = self.calc.compute_divergence({"k": "a"}, {"k": "b"})
        assert d == 1.0

    def test_empty_dicts(self):
        assert self.calc.compute_divergence({}, {}) == 0.0

    def test_mixed_keys(self):
        d = self.calc.compute_divergence({"a": 1, "b": 2}, {"a": 1, "c": 3})
        # a matches (0), b missing from desired (1), c missing from predicted (1) -> 2/3
        assert pytest.approx(d) == 2.0 / 3.0


# ---------------------------------------------------------------------------
# calculate_efe
# ---------------------------------------------------------------------------

class TestCalculateEFE:
    def test_basic_score(self):
        calc = EFECalculator()
        score = calc.calculate_efe(
            policy_id="p1",
            predicted_outcome={"x": 5},
            desired_outcome={"x": 10},
            uncertainty=0.5,
            information_gain=0.2,
        )
        assert isinstance(score, EFEScore)
        assert score.policy_id == "p1"
        assert pytest.approx(score.total) == (
            score.risk_component + score.ambiguity_component + score.epistemic_component
        )

    def test_custom_weights(self):
        calc = EFECalculator()
        w = EFEWeights(risk=1.0, ambiguity=0.0, epistemic=0.0)
        score = calc.calculate_efe("p", {"x": 0}, {"x": 0}, 1.0, 1.0, weights=w)
        assert score.ambiguity_component == 0.0
        assert score.epistemic_component == 0.0

    def test_zero_info_gain_no_epistemic(self):
        calc = EFECalculator()
        score = calc.calculate_efe("p", {}, {}, 0.0, 0.0)
        assert score.epistemic_component == 0.0


# ---------------------------------------------------------------------------
# select_policy
# ---------------------------------------------------------------------------

class TestSelectPolicy:
    def test_selects_lowest(self):
        calc = EFECalculator()
        scores = [
            EFEScore(total=2.0, risk_component=0, ambiguity_component=0, epistemic_component=0, policy_id="a"),
            EFEScore(total=0.5, risk_component=0, ambiguity_component=0, epistemic_component=0, policy_id="b"),
            EFEScore(total=1.0, risk_component=0, ambiguity_component=0, epistemic_component=0, policy_id="c"),
        ]
        best = calc.select_policy(scores)
        assert best.policy_id == "b"

    def test_empty_raises(self):
        calc = EFECalculator()
        with pytest.raises(ValueError, match="empty"):
            calc.select_policy([])
