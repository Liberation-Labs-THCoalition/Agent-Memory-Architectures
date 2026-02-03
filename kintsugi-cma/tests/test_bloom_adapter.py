"""Tests for kintsugi_engine.bloom_adapter – Phase 3 Stream 3C."""

import pytest
from datetime import datetime

from kintsugi.kintsugi_engine.bloom_adapter import (
    AdversarialScenario,
    BloomAdapter,
    BloomConfig,
    BloomResult,
    DEFAULT_SCENARIO_TEMPLATES,
    ScenarioType,
)


# ------------------------------------------------------------------ enums
class TestScenarioType:
    def test_enum_values(self):
        assert ScenarioType.DONOR_PRESSURE.value == "donor_pressure"
        assert ScenarioType.RESOURCE_CONFLICT.value == "resource_conflict"
        assert ScenarioType.STALE_INFORMATION.value == "stale_information"
        assert ScenarioType.COMPLIANCE.value == "compliance"
        assert ScenarioType.CUSTOM.value == "custom"

    def test_all_members(self):
        assert len(ScenarioType) == 5


# ------------------------------------------------------------------ dataclasses
class TestAdversarialScenario:
    def test_valid(self):
        s = AdversarialScenario(
            scenario_id="s1",
            scenario_type=ScenarioType.CUSTOM,
            description="d",
            context={},
            expected_tensions=[],
            severity="low",
        )
        assert s.severity == "low"

    @pytest.mark.parametrize("bad", ["none", "HIGH", "critical", ""])
    def test_bad_severity(self, bad):
        with pytest.raises(ValueError, match="severity"):
            AdversarialScenario(
                scenario_id="s1",
                scenario_type=ScenarioType.CUSTOM,
                description="d",
                context={},
                expected_tensions=[],
                severity=bad,
            )


class TestBloomResult:
    def test_valid(self):
        r = BloomResult(
            scenario_id="s1",
            alignment_scores={"beliefs": 0.5, "desires": 0.5, "intentions": 0.5},
            overall_score=0.5,
            tensions_detected=[],
            meta_analysis="ok",
        )
        assert r.overall_score == 0.5

    @pytest.mark.parametrize("layer", ["beliefs", "desires", "intentions"])
    def test_score_out_of_range(self, layer):
        scores = {"beliefs": 0.5, "desires": 0.5, "intentions": 0.5}
        scores[layer] = 1.5
        with pytest.raises(ValueError):
            BloomResult(
                scenario_id="s1",
                alignment_scores=scores,
                overall_score=0.5,
                tensions_detected=[],
                meta_analysis="ok",
            )

    def test_negative_score(self):
        with pytest.raises(ValueError):
            BloomResult(
                scenario_id="s1",
                alignment_scores={"beliefs": -0.1, "desires": 0.5, "intentions": 0.5},
                overall_score=0.5,
                tensions_detected=[],
                meta_analysis="ok",
            )


# ------------------------------------------------------------------ config
class TestBloomConfig:
    def test_defaults(self):
        c = BloomConfig()
        assert c.max_scenarios_per_run == 10
        assert c.min_alignment_score == 0.6

    def test_default_templates_has_all_types(self):
        for st in ScenarioType:
            assert st in DEFAULT_SCENARIO_TEMPLATES


# ------------------------------------------------------------------ adapter
class TestBloomAdapter:
    def _bdi(self, beliefs=None, desires=None, intentions=None):
        return {
            "beliefs": beliefs or [],
            "desires": desires or [],
            "intentions": intentions or [],
        }

    def test_generate_scenarios_creates_from_templates(self):
        adapter = BloomAdapter()
        scenarios = adapter.generate_scenarios(self._bdi())
        assert len(scenarios) > 0
        assert all(isinstance(s, AdversarialScenario) for s in scenarios)

    def test_generate_scenarios_respects_max(self):
        cfg = BloomConfig(max_scenarios_per_run=3)
        adapter = BloomAdapter(cfg)
        scenarios = adapter.generate_scenarios(self._bdi())
        assert len(scenarios) == 3

    def test_generate_scenarios_severity_from_tensions(self):
        adapter = BloomAdapter()
        scenarios = adapter.generate_scenarios(self._bdi())
        for s in scenarios:
            if len(s.expected_tensions) >= 3:
                assert s.severity == "high"
            elif len(s.expected_tensions) <= 1:
                assert s.severity == "low"
            else:
                assert s.severity == "medium"

    def test_evaluate_response_scores_per_layer(self):
        adapter = BloomAdapter()
        scenario = AdversarialScenario(
            scenario_id="s1",
            scenario_type=ScenarioType.CUSTOM,
            description="test",
            context={},
            expected_tensions=["beliefs"],
            severity="low",
        )
        bdi = self._bdi(beliefs=["community health focus"])
        response = {"text": "We should focus on community health improvements"}
        result = adapter.evaluate_response(scenario, response, bdi)
        assert isinstance(result, BloomResult)
        assert "beliefs" in result.alignment_scores
        assert 0.0 <= result.alignment_scores["beliefs"] <= 1.0

    def test_evaluate_response_detects_tensions_below_threshold(self):
        cfg = BloomConfig(min_alignment_score=0.9)
        adapter = BloomAdapter(cfg)
        scenario = AdversarialScenario(
            scenario_id="s1",
            scenario_type=ScenarioType.CUSTOM,
            description="test",
            context={},
            expected_tensions=[],
            severity="low",
        )
        bdi = self._bdi(beliefs=["xyzzy foobar"])
        result = adapter.evaluate_response(scenario, {"text": "unrelated"}, bdi)
        # With high threshold, layers should be detected as tensions
        assert len(result.tensions_detected) > 0

    def test_run_evaluation_chains_generate_and_evaluate(self):
        adapter = BloomAdapter(BloomConfig(max_scenarios_per_run=2))
        bdi = self._bdi(beliefs=["education quality"])
        responses = [{"text": "education quality"}, {"text": "something else"}]
        results = adapter.run_evaluation(bdi, responses)
        assert len(results) == 2
        assert all(isinstance(r, BloomResult) for r in results)

    def test_run_evaluation_fewer_responses_than_scenarios(self):
        adapter = BloomAdapter(BloomConfig(max_scenarios_per_run=3))
        results = adapter.run_evaluation(self._bdi(), [{"text": "a"}])
        assert len(results) == 3

    def test_get_summary_with_results(self):
        adapter = BloomAdapter()
        results = [
            BloomResult(
                scenario_id="s1",
                alignment_scores={"beliefs": 0.9, "desires": 0.8, "intentions": 0.85},
                overall_score=0.85,
                tensions_detected=[],
                meta_analysis="ok",
            ),
            BloomResult(
                scenario_id="s2",
                alignment_scores={"beliefs": 0.4, "desires": 0.3, "intentions": 0.5},
                overall_score=0.4,
                tensions_detected=["beliefs", "desires"],
                meta_analysis="issues",
            ),
        ]
        summary = adapter.get_summary(results)
        assert summary["total_scenarios"] == 2
        assert "avg_scores" in summary
        assert len(summary["worst_scenarios"]) <= 3
        assert summary["overall_health"] in ("strong", "moderate", "at_risk")

    def test_get_summary_empty(self):
        adapter = BloomAdapter()
        summary = adapter.get_summary([])
        assert summary["total_scenarios"] == 0
        assert summary["overall_health"] == "unknown"

    def test_get_summary_health_strong(self):
        adapter = BloomAdapter()
        r = BloomResult("s1", {"beliefs": 0.9, "desires": 0.9, "intentions": 0.9}, 0.9, [], "ok")
        assert adapter.get_summary([r])["overall_health"] == "strong"

    def test_get_summary_health_at_risk(self):
        adapter = BloomAdapter()
        r = BloomResult("s1", {"beliefs": 0.1, "desires": 0.1, "intentions": 0.1}, 0.1, [], "ok")
        assert adapter.get_summary([r])["overall_health"] == "at_risk"


class TestScoreLayer:
    def test_empty_items_returns_neutral(self):
        assert BloomAdapter._score_layer("anything", []) == 0.5

    def test_no_keywords_gives_half_credit(self):
        # Items with only short words (<=3 chars)
        score = BloomAdapter._score_layer("some text", ["a b c"])
        assert score == 0.5  # 0.5 match for item with no significant words

    def test_full_match(self):
        score = BloomAdapter._score_layer(
            "community health education",
            ["community health"],
        )
        assert score > 0.5

    def test_no_match(self):
        score = BloomAdapter._score_layer("unrelated text", ["community health"])
        assert score == 0.0
