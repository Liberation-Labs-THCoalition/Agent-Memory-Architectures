"""
Impact Auditor Skill Chip for Kintsugi CMA.

This chip tracks, measures, and reports organizational impact using
standardized frameworks including the UN Sustainable Development Goals (SDG)
and Global Reporting Initiative (GRI) standards.

Key capabilities:
- Measure program outcomes against defined indicators
- Map activities and outcomes to SDG targets
- Generate impact reports for funders and stakeholders
- Compare current metrics against baseline measurements
- Track indicator trends over time

Example:
    chip = ImpactAuditorChip()
    request = SkillRequest(
        intent="impact_measure",
        entities={"program_id": "prog_123", "indicator": "youth_served"}
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from kintsugi.skills import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillRequest,
    SkillResponse,
    register_chip,
)


class SDGGoal(str, Enum):
    """UN Sustainable Development Goals."""
    NO_POVERTY = "SDG1"
    ZERO_HUNGER = "SDG2"
    GOOD_HEALTH = "SDG3"
    QUALITY_EDUCATION = "SDG4"
    GENDER_EQUALITY = "SDG5"
    CLEAN_WATER = "SDG6"
    CLEAN_ENERGY = "SDG7"
    DECENT_WORK = "SDG8"
    INDUSTRY_INNOVATION = "SDG9"
    REDUCED_INEQUALITIES = "SDG10"
    SUSTAINABLE_CITIES = "SDG11"
    RESPONSIBLE_CONSUMPTION = "SDG12"
    CLIMATE_ACTION = "SDG13"
    LIFE_BELOW_WATER = "SDG14"
    LIFE_ON_LAND = "SDG15"
    PEACE_JUSTICE = "SDG16"
    PARTNERSHIPS = "SDG17"


class IndicatorType(str, Enum):
    """Types of impact indicators."""
    OUTPUT = "output"  # Direct counts (people served, meals provided)
    OUTCOME = "outcome"  # Changes achieved (skills gained, housing secured)
    IMPACT = "impact"  # Long-term societal changes


@dataclass
class Indicator:
    """Represents an impact indicator.

    Attributes:
        id: Unique identifier
        name: Human-readable name
        description: What this indicator measures
        indicator_type: Output, outcome, or impact
        unit: Unit of measurement
        sdg_mapping: SDG goals this indicator relates to
        gri_codes: GRI standard codes if applicable
        baseline_value: Baseline measurement
        target_value: Target to achieve
        current_value: Most recent measurement
    """
    id: str
    name: str
    description: str
    indicator_type: IndicatorType
    unit: str
    sdg_mapping: list[str] = field(default_factory=list)
    gri_codes: list[str] = field(default_factory=list)
    baseline_value: float | None = None
    target_value: float | None = None
    current_value: float | None = None
    measurement_date: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "indicator_type": self.indicator_type.value,
            "unit": self.unit,
            "sdg_mapping": self.sdg_mapping,
            "gri_codes": self.gri_codes,
            "baseline_value": self.baseline_value,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "measurement_date": self.measurement_date.isoformat() if self.measurement_date else None,
        }


@dataclass
class Measurement:
    """A recorded measurement for an indicator.

    Attributes:
        indicator_id: ID of the indicator measured
        value: Measured value
        date: Date of measurement
        source: Data source (survey, system, manual)
        notes: Additional context
        verified: Whether measurement has been verified
    """
    indicator_id: str
    value: float
    date: datetime
    source: str = "manual"
    notes: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "indicator_id": self.indicator_id,
            "value": self.value,
            "date": self.date.isoformat(),
            "source": self.source,
            "notes": self.notes,
            "verified": self.verified,
        }


class ImpactAuditorChip(BaseSkillChip):
    """Track, measure, and report organizational impact using SDG and GRI frameworks.

    This chip helps nonprofits measure and communicate their impact
    using internationally recognized frameworks, supporting both
    internal evaluation and external reporting.

    Intents handled:
        - impact_measure: Record or retrieve measurements for indicators
        - impact_report: Generate impact reports
        - sdg_align: Map activities/outcomes to SDG targets
        - outcome_track: Track outcome progress over time
        - indicator_define: Define new indicators

    Consensus actions:
        - publish_report: Requires approval before publishing
        - submit_to_funder: Requires approval before submitting reports

    Example:
        chip = ImpactAuditorChip()
        request = SkillRequest(
            intent="sdg_align",
            entities={"program_name": "Youth Tutoring Program"}
        )
        response = await chip.handle(request, context)
    """

    name = "impact_auditor"
    description = "Track, measure, and report organizational impact using SDG and GRI frameworks"
    version = "1.0.0"
    domain = SkillDomain.PROGRAMS

    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.25,
        resource_efficiency=0.10,
        transparency=0.25,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["publish_report", "submit_to_funder"]
    required_spans = ["sdg_mapping", "gri_standards", "report_templates"]

    # SDG goal descriptions for mapping
    SDG_DESCRIPTIONS = {
        "SDG1": "No Poverty - End poverty in all its forms everywhere",
        "SDG2": "Zero Hunger - End hunger, achieve food security",
        "SDG3": "Good Health and Well-being - Ensure healthy lives",
        "SDG4": "Quality Education - Ensure inclusive and equitable quality education",
        "SDG5": "Gender Equality - Achieve gender equality and empower all women and girls",
        "SDG6": "Clean Water and Sanitation - Ensure availability of water",
        "SDG7": "Affordable and Clean Energy - Ensure access to energy",
        "SDG8": "Decent Work and Economic Growth - Promote sustained economic growth",
        "SDG9": "Industry, Innovation and Infrastructure - Build resilient infrastructure",
        "SDG10": "Reduced Inequalities - Reduce inequality within and among countries",
        "SDG11": "Sustainable Cities and Communities - Make cities inclusive and sustainable",
        "SDG12": "Responsible Consumption and Production - Ensure sustainable patterns",
        "SDG13": "Climate Action - Take urgent action to combat climate change",
        "SDG14": "Life Below Water - Conserve and sustainably use the oceans",
        "SDG15": "Life on Land - Protect, restore and promote sustainable use of ecosystems",
        "SDG16": "Peace, Justice and Strong Institutions - Promote peaceful and inclusive societies",
        "SDG17": "Partnerships for the Goals - Strengthen means of implementation",
    }

    SUPPORTED_INTENTS = {
        "impact_measure": "_handle_measure",
        "impact_report": "_handle_report",
        "sdg_align": "_handle_sdg_align",
        "outcome_track": "_handle_outcome_track",
        "indicator_define": "_handle_indicator_define",
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with impact data or report
        """
        handler_name = self.SUPPORTED_INTENTS.get(request.intent)

        if handler_name is None:
            return SkillResponse(
                content=f"Unknown intent '{request.intent}' for impact_auditor chip.",
                success=False,
                data={"supported_intents": list(self.SUPPORTED_INTENTS.keys())},
            )

        handler = getattr(self, handler_name)
        return await handler(request, context)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract programs-relevant BDI context.

        Filters BDI state for beliefs about program performance,
        impact measurements, and reporting requirements.
        """
        program_types = {"program_status", "impact_data", "beneficiary_count", "outcome_measure"}

        filtered_beliefs = [
            b for b in beliefs
            if b.get("type") in program_types or b.get("domain") == "programs"
        ]

        filtered_desires = [
            d for d in desires
            if d.get("type") in {"impact_goal", "reporting_deadline", "sdg_target"}
        ]

        return {
            "beliefs": filtered_beliefs,
            "desires": filtered_desires,
            "intentions": intentions,
        }

    async def _handle_measure(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Record or retrieve impact measurements.

        Supported entities:
            - indicator_id: Indicator to measure
            - value: Value to record (for new measurement)
            - program_id: Associated program
            - date_range: For retrieving historical data
            - action: "record" or "retrieve"
        """
        entities = request.entities
        action = entities.get("action", "retrieve")
        indicator_id = entities.get("indicator_id")

        if not indicator_id:
            return SkillResponse(
                content="Please specify an indicator_id to measure.",
                success=False,
            )

        if action == "record":
            value = entities.get("value")
            if value is None:
                return SkillResponse(
                    content="Please provide a value to record.",
                    success=False,
                )

            result = await self.measure_outcome(
                indicator_id=indicator_id,
                value=value,
                program_id=entities.get("program_id"),
                source=entities.get("source", "manual"),
                notes=entities.get("notes", ""),
            )

            indicator = result.get("indicator")
            progress = ""
            if indicator and indicator.target_value:
                pct = (value / indicator.target_value) * 100
                progress = f" ({pct:.1f}% of target)"

            return SkillResponse(
                content=f"Recorded measurement: {value} {indicator.unit if indicator else ''}{progress}",
                success=True,
                data=result,
            )

        else:  # retrieve
            measurements = await self._get_measurements(
                indicator_id=indicator_id,
                date_range=entities.get("date_range", "ytd"),
            )

            if not measurements:
                return SkillResponse(
                    content=f"No measurements found for indicator '{indicator_id}'.",
                    success=True,
                    data={"measurements": []},
                )

            content_lines = [f"Measurements for '{indicator_id}':\n"]
            for m in measurements[-10:]:  # Last 10
                content_lines.append(
                    f"- {m.date.strftime('%Y-%m-%d')}: {m.value} ({m.source})"
                )

            return SkillResponse(
                content="\n".join(content_lines),
                success=True,
                data={"measurements": [m.to_dict() for m in measurements]},
            )

    async def _handle_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate impact reports.

        Supported entities:
            - report_type: Type of report (summary, detailed, funder, annual)
            - program_id: Specific program or "all"
            - date_range: Period to cover
            - format: Output format (text, pdf, html)
            - funder_id: For funder-specific reports
        """
        entities = request.entities
        report_type = entities.get("report_type", "summary")
        program_id = entities.get("program_id")
        date_range = entities.get("date_range", "ytd")

        report = await self.generate_report(
            org_id=context.org_id,
            report_type=report_type,
            program_id=program_id,
            date_range=date_range,
            funder_id=entities.get("funder_id"),
        )

        # Check if this is for external publication
        if report_type == "funder" and entities.get("submit"):
            return SkillResponse(
                content="Report ready for submission. Please review and confirm.",
                success=True,
                requires_consensus=True,
                consensus_action="submit_to_funder",
                data=report,
            )

        return SkillResponse(
            content=report["summary"],
            success=True,
            data=report,
            suggestions=["Export to PDF?", "Submit to funder?", "Share with board?"],
        )

    async def _handle_sdg_align(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Map activities or outcomes to SDG targets.

        Supported entities:
            - program_name: Name of program to analyze
            - activity: Specific activity to map
            - outcome: Outcome to map
            - auto_suggest: Whether to suggest SDG mappings
        """
        entities = request.entities
        program_name = entities.get("program_name")
        activity = entities.get("activity")
        outcome = entities.get("outcome")

        # What are we mapping?
        mapping_target = program_name or activity or outcome
        if not mapping_target:
            return SkillResponse(
                content="Please specify a program, activity, or outcome to map to SDGs.",
                success=False,
            )

        sdg_mappings = await self.map_to_sdg(
            description=mapping_target,
            org_id=context.org_id,
        )

        if not sdg_mappings:
            return SkillResponse(
                content=f"Could not determine SDG mapping for '{mapping_target}'.",
                success=True,
                data={"mappings": []},
                suggestions=["Try providing more detail about activities or outcomes"],
            )

        content_lines = [f"SDG Alignment for '{mapping_target}':\n"]
        for mapping in sdg_mappings:
            sdg = mapping["sdg"]
            confidence = mapping["confidence"] * 100
            content_lines.append(
                f"- **{sdg}**: {self.SDG_DESCRIPTIONS.get(sdg, sdg)} ({confidence:.0f}% confidence)"
            )
            if mapping.get("targets"):
                for target in mapping["targets"][:2]:
                    content_lines.append(f"  - Target {target}")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={"mappings": sdg_mappings},
        )

    async def _handle_outcome_track(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Track outcome progress over time.

        Supported entities:
            - indicator_id: Indicator to track
            - program_id: Program to track
            - date_range: Historical period
            - compare_baseline: Whether to compare against baseline
        """
        entities = request.entities
        indicator_id = entities.get("indicator_id")
        program_id = entities.get("program_id")
        date_range = entities.get("date_range", "12_months")
        compare_baseline = entities.get("compare_baseline", True)

        if not indicator_id:
            return SkillResponse(
                content="Please specify an indicator to track.",
                success=False,
            )

        tracking_data = await self._track_indicator_progress(
            indicator_id=indicator_id,
            program_id=program_id,
            date_range=date_range,
        )

        indicator = tracking_data.get("indicator")
        if not indicator:
            return SkillResponse(
                content=f"Indicator '{indicator_id}' not found.",
                success=False,
            )

        # Build summary
        content_lines = [f"Outcome Tracking: **{indicator.name}**\n"]
        content_lines.append(f"Current Value: {indicator.current_value} {indicator.unit}")

        if compare_baseline and indicator.baseline_value is not None:
            change = indicator.current_value - indicator.baseline_value
            pct_change = (change / indicator.baseline_value) * 100 if indicator.baseline_value else 0
            direction = "increase" if change > 0 else "decrease"
            content_lines.append(
                f"Change from Baseline: {change:+.1f} ({pct_change:+.1f}% {direction})"
            )

        if indicator.target_value:
            progress = (indicator.current_value / indicator.target_value) * 100
            content_lines.append(f"Progress to Target: {progress:.1f}%")

        if tracking_data.get("trend"):
            content_lines.append(f"Trend: {tracking_data['trend']}")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data=tracking_data,
            suggestions=["View detailed trend chart?", "Compare with similar programs?"],
        )

    async def _handle_indicator_define(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Define a new impact indicator.

        Supported entities:
            - name: Indicator name
            - description: What it measures
            - indicator_type: output, outcome, or impact
            - unit: Unit of measurement
            - baseline_value: Starting measurement
            - target_value: Goal to achieve
            - sdg_mapping: SDG goals to link to
            - program_id: Associated program
        """
        entities = request.entities

        required = ["name", "description", "unit"]
        missing = [f for f in required if not entities.get(f)]
        if missing:
            return SkillResponse(
                content=f"Missing required fields: {', '.join(missing)}",
                success=False,
            )

        indicator = Indicator(
            id=f"ind_{datetime.now(timezone.utc).timestamp()}",
            name=entities["name"],
            description=entities["description"],
            indicator_type=IndicatorType(entities.get("indicator_type", "output")),
            unit=entities["unit"],
            sdg_mapping=entities.get("sdg_mapping", []),
            baseline_value=entities.get("baseline_value"),
            target_value=entities.get("target_value"),
        )

        # In production, would save to database
        content = f"""Created new indicator: **{indicator.name}**

Type: {indicator.indicator_type.value}
Unit: {indicator.unit}
Baseline: {indicator.baseline_value or 'Not set'}
Target: {indicator.target_value or 'Not set'}
SDG Alignment: {', '.join(indicator.sdg_mapping) or 'None'}
"""

        return SkillResponse(
            content=content,
            success=True,
            data={"indicator": indicator.to_dict()},
            suggestions=["Record first measurement?", "Link to program?"],
        )

    # Core implementation methods

    async def measure_outcome(
        self,
        indicator_id: str,
        value: float,
        program_id: str | None = None,
        source: str = "manual",
        notes: str = "",
    ) -> dict[str, Any]:
        """Record a measurement for an outcome indicator.

        Args:
            indicator_id: The indicator to measure
            value: The measured value
            program_id: Associated program if applicable
            source: Data source (manual, survey, system)
            notes: Additional context

        Returns:
            Dictionary with measurement confirmation and indicator status
        """
        indicator = await self._get_indicator(indicator_id)
        if not indicator:
            return {"success": False, "error": "Indicator not found"}

        measurement = Measurement(
            indicator_id=indicator_id,
            value=value,
            date=datetime.now(timezone.utc),
            source=source,
            notes=notes,
        )

        # Update indicator's current value
        indicator.current_value = value
        indicator.measurement_date = measurement.date

        # In production, would save to database
        return {
            "success": True,
            "measurement": measurement.to_dict(),
            "indicator": indicator,
            "program_id": program_id,
        }

    async def map_to_sdg(
        self,
        description: str,
        org_id: str,
    ) -> list[dict[str, Any]]:
        """Map a program, activity, or outcome to SDG goals.

        Uses keyword matching and semantic analysis to identify
        relevant SDG goals and targets.

        Args:
            description: Text describing what to map
            org_id: Organization identifier

        Returns:
            List of SDG mappings with confidence scores
        """
        # Keyword-based SDG mapping (simplified)
        sdg_keywords = {
            "SDG1": ["poverty", "income", "economic hardship", "financial assistance"],
            "SDG2": ["hunger", "food", "nutrition", "meals", "food security"],
            "SDG3": ["health", "wellness", "medical", "mental health", "healthcare"],
            "SDG4": ["education", "learning", "school", "tutoring", "training", "literacy"],
            "SDG5": ["women", "girls", "gender", "equality", "female"],
            "SDG8": ["employment", "jobs", "workforce", "career", "economic growth"],
            "SDG10": ["inequality", "equity", "inclusion", "marginalized", "underserved"],
            "SDG11": ["housing", "community", "urban", "neighborhood", "city"],
            "SDG13": ["climate", "environment", "sustainability", "green"],
            "SDG16": ["justice", "rights", "legal", "advocacy", "civic"],
        }

        description_lower = description.lower()
        mappings = []

        for sdg, keywords in sdg_keywords.items():
            matches = sum(1 for kw in keywords if kw in description_lower)
            if matches > 0:
                confidence = min(matches / len(keywords) + 0.3, 0.95)
                mappings.append({
                    "sdg": sdg,
                    "confidence": confidence,
                    "matched_keywords": [kw for kw in keywords if kw in description_lower],
                    "targets": self._get_sdg_targets(sdg),
                })

        # Sort by confidence
        mappings.sort(key=lambda x: x["confidence"], reverse=True)

        return mappings[:5]  # Top 5 matches

    async def generate_report(
        self,
        org_id: str,
        report_type: str = "summary",
        program_id: str | None = None,
        date_range: str = "ytd",
        funder_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate an impact report.

        Args:
            org_id: Organization identifier
            report_type: Type of report to generate
            program_id: Specific program or None for org-wide
            date_range: Period to cover
            funder_id: For funder-specific reporting requirements

        Returns:
            Dictionary containing report content and metadata
        """
        indicators = await self._get_indicators(org_id, program_id)

        # Gather measurements for each indicator
        report_data = {
            "org_id": org_id,
            "report_type": report_type,
            "date_range": date_range,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "indicators": [],
            "sdg_alignment": [],
        }

        for ind in indicators:
            measurements = await self._get_measurements(ind.id, date_range)
            report_data["indicators"].append({
                "indicator": ind.to_dict(),
                "measurement_count": len(measurements),
                "latest_value": measurements[-1].value if measurements else None,
            })

            # Collect SDG alignments
            for sdg in ind.sdg_mapping:
                if sdg not in [s["sdg"] for s in report_data["sdg_alignment"]]:
                    report_data["sdg_alignment"].append({"sdg": sdg, "indicators": []})
                # Add indicator to SDG
                for s in report_data["sdg_alignment"]:
                    if s["sdg"] == sdg:
                        s["indicators"].append(ind.name)

        # Generate summary text
        summary = f"""Impact Report ({date_range.upper()})

**Overview**
- Indicators Tracked: {len(indicators)}
- SDG Goals Addressed: {len(report_data['sdg_alignment'])}

**Key Metrics**
"""
        for ind_data in report_data["indicators"][:5]:
            ind = ind_data["indicator"]
            val = ind_data["latest_value"]
            summary += f"- {ind['name']}: {val or 'N/A'} {ind['unit']}\n"

        summary += "\n**SDG Alignment**\n"
        for sdg_data in report_data["sdg_alignment"]:
            summary += f"- {sdg_data['sdg']}: {len(sdg_data['indicators'])} indicators\n"

        report_data["summary"] = summary

        return report_data

    async def compare_baseline(
        self,
        indicator_id: str,
        current_value: float | None = None,
    ) -> dict[str, Any]:
        """Compare current measurement against baseline.

        Args:
            indicator_id: Indicator to compare
            current_value: Current value (if not using stored value)

        Returns:
            Comparison analysis with change metrics
        """
        indicator = await self._get_indicator(indicator_id)
        if not indicator:
            return {"error": "Indicator not found"}

        if indicator.baseline_value is None:
            return {"error": "No baseline value set for this indicator"}

        current = current_value or indicator.current_value
        if current is None:
            return {"error": "No current value available"}

        change = current - indicator.baseline_value
        pct_change = (change / indicator.baseline_value) * 100 if indicator.baseline_value else 0

        # Determine if target is met
        target_met = False
        target_progress = None
        if indicator.target_value:
            target_progress = (current / indicator.target_value) * 100
            target_met = current >= indicator.target_value

        return {
            "indicator_id": indicator_id,
            "indicator_name": indicator.name,
            "baseline_value": indicator.baseline_value,
            "current_value": current,
            "absolute_change": change,
            "percent_change": pct_change,
            "target_value": indicator.target_value,
            "target_progress": target_progress,
            "target_met": target_met,
            "unit": indicator.unit,
        }

    # Private helper methods

    async def _get_indicator(self, indicator_id: str) -> Indicator | None:
        """Fetch an indicator by ID."""
        indicators = await self._get_indicators("", None)
        for ind in indicators:
            if ind.id == indicator_id:
                return ind
        return None

    async def _get_indicators(
        self, org_id: str, program_id: str | None
    ) -> list[Indicator]:
        """Fetch indicators for an organization/program."""
        # Simulated data
        return [
            Indicator(
                id="ind_youth_served",
                name="Youth Served",
                description="Number of youth receiving direct services",
                indicator_type=IndicatorType.OUTPUT,
                unit="individuals",
                sdg_mapping=["SDG4", "SDG10"],
                baseline_value=100,
                target_value=500,
                current_value=350,
                measurement_date=datetime.now(timezone.utc),
            ),
            Indicator(
                id="ind_meals_provided",
                name="Meals Provided",
                description="Number of meals served to community members",
                indicator_type=IndicatorType.OUTPUT,
                unit="meals",
                sdg_mapping=["SDG2"],
                baseline_value=1000,
                target_value=5000,
                current_value=3500,
            ),
            Indicator(
                id="ind_literacy_gain",
                name="Literacy Improvement",
                description="Percentage of students showing literacy gains",
                indicator_type=IndicatorType.OUTCOME,
                unit="percent",
                sdg_mapping=["SDG4"],
                baseline_value=20,
                target_value=75,
                current_value=65,
            ),
        ]

    async def _get_measurements(
        self, indicator_id: str, date_range: str
    ) -> list[Measurement]:
        """Fetch measurements for an indicator."""
        # Simulated data
        now = datetime.now(timezone.utc)
        return [
            Measurement(indicator_id=indicator_id, value=100, date=now, source="system"),
            Measurement(indicator_id=indicator_id, value=150, date=now, source="survey"),
            Measurement(indicator_id=indicator_id, value=200, date=now, source="manual"),
        ]

    async def _track_indicator_progress(
        self,
        indicator_id: str,
        program_id: str | None,
        date_range: str,
    ) -> dict[str, Any]:
        """Track progress of an indicator over time."""
        indicator = await self._get_indicator(indicator_id)
        measurements = await self._get_measurements(indicator_id, date_range)

        if not indicator:
            return {"indicator": None, "measurements": []}

        # Determine trend
        if len(measurements) >= 2:
            recent = measurements[-1].value
            earlier = measurements[0].value
            if recent > earlier:
                trend = "Increasing"
            elif recent < earlier:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient data"

        return {
            "indicator": indicator,
            "measurements": [m.to_dict() for m in measurements],
            "trend": trend,
            "data_points": len(measurements),
        }

    def _get_sdg_targets(self, sdg: str) -> list[str]:
        """Get relevant targets for an SDG goal."""
        # Simplified target mapping
        targets = {
            "SDG1": ["1.1", "1.2", "1.3"],
            "SDG2": ["2.1", "2.2"],
            "SDG3": ["3.1", "3.2", "3.3"],
            "SDG4": ["4.1", "4.2", "4.3", "4.4"],
            "SDG5": ["5.1", "5.2"],
            "SDG8": ["8.5", "8.6"],
            "SDG10": ["10.1", "10.2", "10.3"],
            "SDG11": ["11.1", "11.3"],
            "SDG13": ["13.1", "13.3"],
            "SDG16": ["16.1", "16.3"],
        }
        return targets.get(sdg, [])
