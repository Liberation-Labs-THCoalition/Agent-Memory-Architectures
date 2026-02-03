"""
Program Evaluator Skill Chip for Kintsugi CMA.

Provides capabilities for designing and tracking program logic models,
outcomes measurement, evaluation design, data collection, and findings
reporting. Supports evidence-based program improvement through systematic
evaluation practices.

Example:
    chip = ProgramEvaluatorChip()
    request = SkillRequest(
        intent="logic_model",
        entities={"program_id": "prog_001", "include_indicators": True},
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from kintsugi.skills import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillRequest,
    SkillResponse,
)


@dataclass
class LogicModelComponent:
    """Represents a component in a program logic model."""
    component_id: str
    component_type: str  # inputs, activities, outputs, outcomes_short, outcomes_long, impact
    description: str
    indicators: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class OutcomeMetric:
    """Represents a tracked outcome metric."""
    metric_id: str
    name: str
    target_value: float
    current_value: float
    unit: str
    period: str  # monthly, quarterly, annual
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EvaluationDesign:
    """Represents an evaluation study design."""
    design_id: str
    evaluation_type: str  # formative, summative, developmental
    methodology: str  # qualitative, quantitative, mixed_methods
    questions: list[str]
    data_collection_methods: list[str]
    timeline: dict[str, Any]
    budget_estimate: float


class ProgramEvaluatorChip(BaseSkillChip):
    """Design and track program logic models, outcomes, and evaluations.

    This chip supports program staff and evaluators in developing theory
    of change, tracking outcomes, designing evaluations, collecting data,
    and generating findings reports. It emphasizes mission alignment and
    stakeholder benefit in all evaluation activities.

    Intents:
        logic_model: Build or retrieve program logic model
        outcome_track: Track and update outcome metrics
        evaluation_design: Design evaluation study
        data_collect: Set up data collection instruments
        findings_report: Generate evaluation findings report

    Example:
        >>> chip = ProgramEvaluatorChip()
        >>> request = SkillRequest(intent="logic_model", entities={"program_id": "prog_001"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["logic_model"]["components"])
    """

    name = "program_evaluator"
    description = "Design and track program logic models, outcomes, and evaluations"
    version = "1.0.0"
    domain = SkillDomain.PROGRAMS

    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.30,
        resource_efficiency=0.15,
        transparency=0.15,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["finalize_evaluation", "publish_findings"]

    required_spans = ["logic_model_builder", "survey_tools", "data_viz"]

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: Skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with result content and data
        """
        intent = request.intent

        # Get filtered BDI context for programs domain
        bdi = await self.get_bdi_context(
            context.beliefs,
            context.desires,
            context.intentions,
        )

        handlers = {
            "logic_model": self._build_logic_model,
            "outcome_track": self._track_outcomes,
            "evaluation_design": self._design_evaluation,
            "data_collect": self._setup_data_collection,
            "findings_report": self._generate_findings,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for program evaluator.",
                success=False,
                suggestions=[
                    "Try 'logic_model' to build a program logic model",
                    "Try 'outcome_track' to track program outcomes",
                    "Try 'evaluation_design' to design an evaluation",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for program evaluation context.

        Extracts beliefs about program performance, outcome targets,
        and evaluation priorities relevant to this chip.
        """
        program_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("programs", "evaluation", "outcomes")
            or b.get("type") in ("program_status", "outcome_target", "evaluation_need")
        ]

        evaluation_desires = [
            d for d in desires
            if d.get("type") in ("improve_program", "measure_impact", "demonstrate_outcomes")
            or d.get("domain") == "evaluation"
        ]

        return {
            "beliefs": program_beliefs,
            "desires": evaluation_desires,
            "intentions": intentions,
        }

    async def _build_logic_model(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Build or retrieve a program logic model.

        Creates a structured logic model with inputs, activities, outputs,
        short-term outcomes, long-term outcomes, and impact indicators.
        """
        program_id = request.entities.get("program_id", "")
        program_name = request.entities.get("program_name", "Unnamed Program")
        include_indicators = request.entities.get("include_indicators", True)

        # Build logic model structure
        components = [
            LogicModelComponent(
                component_id=f"input_{uuid4().hex[:8]}",
                component_type="inputs",
                description="Staff time, funding, facilities, materials",
                indicators=["FTE allocated", "Budget amount", "Space utilization"] if include_indicators else [],
                data_sources=["HR system", "Financial records", "Facilities log"],
            ),
            LogicModelComponent(
                component_id=f"activity_{uuid4().hex[:8]}",
                component_type="activities",
                description="Program delivery activities and services",
                indicators=["Sessions delivered", "Participants served", "Hours of service"] if include_indicators else [],
                data_sources=["Program tracking system", "Attendance records"],
                dependencies=["inputs"],
            ),
            LogicModelComponent(
                component_id=f"output_{uuid4().hex[:8]}",
                component_type="outputs",
                description="Direct products of program activities",
                indicators=["Completion rate", "Satisfaction score", "Materials distributed"] if include_indicators else [],
                data_sources=["Exit surveys", "Distribution logs"],
                dependencies=["activities"],
            ),
            LogicModelComponent(
                component_id=f"outcome_short_{uuid4().hex[:8]}",
                component_type="outcomes_short",
                description="Changes in knowledge, skills, attitudes (1-3 years)",
                indicators=["Knowledge gain", "Skill demonstration", "Attitude shift"] if include_indicators else [],
                data_sources=["Pre/post assessments", "Observations", "Interviews"],
                dependencies=["outputs"],
            ),
            LogicModelComponent(
                component_id=f"outcome_long_{uuid4().hex[:8]}",
                component_type="outcomes_long",
                description="Changes in behavior and conditions (3-6 years)",
                indicators=["Behavior change rate", "Condition improvement", "Practice adoption"] if include_indicators else [],
                data_sources=["Follow-up surveys", "Administrative data"],
                dependencies=["outcomes_short"],
            ),
            LogicModelComponent(
                component_id=f"impact_{uuid4().hex[:8]}",
                component_type="impact",
                description="Long-term community or systemic change (6+ years)",
                indicators=["Population-level change", "System transformation", "Policy adoption"] if include_indicators else [],
                data_sources=["Community surveys", "Secondary data", "Policy analysis"],
                dependencies=["outcomes_long"],
            ),
        ]

        logic_model = {
            "model_id": f"lm_{uuid4().hex[:8]}",
            "program_id": program_id,
            "program_name": program_name,
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": [
                {
                    "component_id": c.component_id,
                    "type": c.component_type,
                    "description": c.description,
                    "indicators": c.indicators,
                    "data_sources": c.data_sources,
                    "dependencies": c.dependencies,
                }
                for c in components
            ],
            "assumptions": [
                "Program has adequate staffing and resources",
                "Target population is accessible and engaged",
                "External conditions remain stable",
            ],
            "external_factors": [
                "Economic conditions",
                "Policy environment",
                "Community partnerships",
            ],
        }

        return SkillResponse(
            content=f"Built logic model for '{program_name}' with {len(components)} components. "
                    f"Includes {sum(len(c.indicators) for c in components)} indicators across all levels.",
            success=True,
            data={"logic_model": logic_model},
            suggestions=[
                "Would you like to add specific indicators for any component?",
                "Shall I help design an evaluation based on this logic model?",
                "Want to set up outcome tracking for short-term outcomes?",
            ],
        )

    async def _track_outcomes(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Track and update program outcome metrics.

        Retrieves current outcome status, updates metrics, and provides
        progress analysis against targets.
        """
        program_id = request.entities.get("program_id", "")
        metric_updates = request.entities.get("metrics", [])
        time_period = request.entities.get("period", "quarterly")

        # Simulated current outcomes (would pull from data store)
        current_metrics = [
            OutcomeMetric(
                metric_id="om_001",
                name="Participant Knowledge Gain",
                target_value=80.0,
                current_value=72.5,
                unit="percent",
                period=time_period,
            ),
            OutcomeMetric(
                metric_id="om_002",
                name="Program Completion Rate",
                target_value=85.0,
                current_value=88.2,
                unit="percent",
                period=time_period,
            ),
            OutcomeMetric(
                metric_id="om_003",
                name="Participant Satisfaction",
                target_value=4.5,
                current_value=4.3,
                unit="score (1-5)",
                period=time_period,
            ),
            OutcomeMetric(
                metric_id="om_004",
                name="Skills Demonstration",
                target_value=75.0,
                current_value=68.0,
                unit="percent",
                period=time_period,
            ),
        ]

        # Calculate progress for each metric
        outcome_summary = []
        on_track = 0
        needs_attention = 0

        for metric in current_metrics:
            progress = (metric.current_value / metric.target_value) * 100 if metric.target_value > 0 else 0
            status = "on_track" if progress >= 90 else "needs_attention" if progress >= 70 else "at_risk"

            if status == "on_track":
                on_track += 1
            else:
                needs_attention += 1

            outcome_summary.append({
                "metric_id": metric.metric_id,
                "name": metric.name,
                "target": metric.target_value,
                "current": metric.current_value,
                "unit": metric.unit,
                "progress_percent": round(progress, 1),
                "status": status,
                "period": metric.period,
            })

        return SkillResponse(
            content=f"Tracking {len(current_metrics)} outcome metrics for {time_period} period. "
                    f"{on_track} metrics on track, {needs_attention} need attention.",
            success=True,
            data={
                "program_id": program_id,
                "period": time_period,
                "outcomes": outcome_summary,
                "summary": {
                    "total_metrics": len(current_metrics),
                    "on_track": on_track,
                    "needs_attention": needs_attention,
                    "overall_progress": round(sum(o["progress_percent"] for o in outcome_summary) / len(outcome_summary), 1),
                },
            },
            suggestions=[
                "Want to drill into any specific metric?",
                "Should I suggest interventions for at-risk outcomes?",
                "Would you like to update any metric values?",
            ],
        )

    async def _design_evaluation(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Design an evaluation study for a program.

        Creates evaluation design including methodology, research questions,
        data collection methods, timeline, and budget estimate.
        """
        program_id = request.entities.get("program_id", "")
        eval_type = request.entities.get("evaluation_type", "summative")
        methodology = request.entities.get("methodology", "mixed_methods")
        focus_areas = request.entities.get("focus_areas", ["outcomes", "process"])

        # Generate evaluation questions based on type and focus
        questions = []
        if "outcomes" in focus_areas:
            questions.extend([
                "To what extent did the program achieve its intended outcomes?",
                "What changes in knowledge, skills, or behaviors resulted from program participation?",
                "How do outcomes vary across participant subgroups?",
            ])
        if "process" in focus_areas:
            questions.extend([
                "Was the program implemented as designed?",
                "What factors facilitated or hindered implementation?",
                "How do participants experience the program?",
            ])
        if "efficiency" in focus_areas:
            questions.extend([
                "What is the cost per participant served?",
                "How does the cost compare to similar programs?",
                "Are resources being used efficiently?",
            ])

        # Data collection methods based on methodology
        data_methods = []
        if methodology in ("quantitative", "mixed_methods"):
            data_methods.extend(["Pre/post surveys", "Administrative data review", "Outcome tracking data"])
        if methodology in ("qualitative", "mixed_methods"):
            data_methods.extend(["Participant interviews", "Focus groups", "Document review", "Observations"])

        design = EvaluationDesign(
            design_id=f"eval_{uuid4().hex[:8]}",
            evaluation_type=eval_type,
            methodology=methodology,
            questions=questions,
            data_collection_methods=data_methods,
            timeline={
                "planning": "Month 1-2",
                "instrument_development": "Month 2-3",
                "data_collection": "Month 3-5",
                "analysis": "Month 5-6",
                "reporting": "Month 6-7",
            },
            budget_estimate=15000.0 if eval_type == "formative" else 25000.0,
        )

        # Check if this requires consensus
        requires_approval = self.requires_consensus("finalize_evaluation")

        return SkillResponse(
            content=f"Designed {eval_type} evaluation using {methodology} methodology. "
                    f"Includes {len(questions)} research questions and {len(data_methods)} data collection methods.",
            success=True,
            data={
                "evaluation_design": {
                    "design_id": design.design_id,
                    "program_id": program_id,
                    "type": design.evaluation_type,
                    "methodology": design.methodology,
                    "research_questions": design.questions,
                    "data_collection_methods": design.data_collection_methods,
                    "timeline": design.timeline,
                    "budget_estimate": design.budget_estimate,
                    "focus_areas": focus_areas,
                },
            },
            requires_consensus=requires_approval,
            consensus_action="finalize_evaluation" if requires_approval else None,
            suggestions=[
                "Would you like to refine any research questions?",
                "Should I develop data collection instruments?",
                "Want to adjust the timeline or budget?",
            ],
        )

    async def _setup_data_collection(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Set up data collection instruments and protocols.

        Creates survey instruments, interview protocols, or observation
        checklists based on evaluation design.
        """
        eval_id = request.entities.get("evaluation_id", "")
        instrument_type = request.entities.get("instrument_type", "survey")
        construct = request.entities.get("construct", "satisfaction")

        instruments = {
            "survey": {
                "name": f"{construct.title()} Survey",
                "type": "survey",
                "sections": [
                    {
                        "title": "Demographics",
                        "questions": [
                            {"id": "q1", "text": "Age range", "type": "multiple_choice"},
                            {"id": "q2", "text": "How did you hear about the program?", "type": "multiple_choice"},
                        ],
                    },
                    {
                        "title": f"{construct.title()} Assessment",
                        "questions": [
                            {"id": "q3", "text": f"Rate your overall {construct}", "type": "likert_5"},
                            {"id": "q4", "text": f"What contributed to your {construct}?", "type": "open_ended"},
                            {"id": "q5", "text": "How likely are you to recommend?", "type": "nps"},
                        ],
                    },
                ],
                "estimated_time": "10 minutes",
            },
            "interview": {
                "name": f"{construct.title()} Interview Protocol",
                "type": "interview",
                "introduction": "Thank you for participating. We are interested in your experiences...",
                "questions": [
                    {"id": "i1", "text": "Tell me about your experience with the program.", "probes": ["What did you find valuable?", "What was challenging?"]},
                    {"id": "i2", "text": f"How would you describe your {construct}?", "probes": ["Can you give an example?", "What influenced this?"]},
                    {"id": "i3", "text": "What would you change about the program?", "probes": ["Why is that important?"]},
                ],
                "closing": "Is there anything else you would like to share?",
                "estimated_time": "45 minutes",
            },
            "observation": {
                "name": f"{construct.title()} Observation Checklist",
                "type": "observation",
                "items": [
                    {"id": "o1", "behavior": "Participant engagement level", "scale": "low/medium/high"},
                    {"id": "o2", "behavior": "Questions asked by participants", "scale": "count"},
                    {"id": "o3", "behavior": "Facilitator responsiveness", "scale": "1-5"},
                ],
                "notes_section": True,
                "estimated_time": "Session duration",
            },
        }

        selected_instrument = instruments.get(instrument_type, instruments["survey"])

        return SkillResponse(
            content=f"Created {instrument_type} instrument for measuring {construct}. "
                    f"Estimated completion time: {selected_instrument['estimated_time']}.",
            success=True,
            data={
                "instrument": {
                    "evaluation_id": eval_id,
                    "instrument_id": f"inst_{uuid4().hex[:8]}",
                    **selected_instrument,
                },
            },
            suggestions=[
                "Would you like to customize the questions?",
                "Should I add additional sections?",
                "Want to set up a data collection schedule?",
            ],
        )

    async def _generate_findings(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Generate evaluation findings report.

        Synthesizes data analysis into findings report with recommendations.
        """
        eval_id = request.entities.get("evaluation_id", "")
        report_type = request.entities.get("report_type", "full")
        include_recommendations = request.entities.get("include_recommendations", True)

        findings = {
            "report_id": f"rpt_{uuid4().hex[:8]}",
            "evaluation_id": eval_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": report_type,
            "executive_summary": (
                "The program demonstrated strong participant satisfaction (4.3/5.0) and "
                "exceeded completion rate targets (88.2% vs 85% target). Knowledge gain "
                "outcomes approached but did not meet targets, suggesting opportunity for "
                "curriculum enhancement."
            ),
            "key_findings": [
                {
                    "finding": "Program completion rates exceeded targets",
                    "evidence": "88.2% completion rate against 85% target",
                    "significance": "Indicates strong program engagement and retention",
                },
                {
                    "finding": "Knowledge gains varied by participant background",
                    "evidence": "72.5% average knowledge gain; 65% for first-time participants, 82% for returning",
                    "significance": "Prior experience influences learning outcomes",
                },
                {
                    "finding": "Participant satisfaction remained high",
                    "evidence": "4.3/5.0 average satisfaction score across all cohorts",
                    "significance": "Program meets participant expectations",
                },
            ],
            "recommendations": [
                {
                    "recommendation": "Develop differentiated curriculum tracks",
                    "rationale": "Address varying knowledge gain rates by participant background",
                    "priority": "high",
                    "resources_needed": "Curriculum developer time, pilot testing",
                },
                {
                    "recommendation": "Implement knowledge checkpoints",
                    "rationale": "Enable early intervention for participants struggling with material",
                    "priority": "medium",
                    "resources_needed": "Assessment development, staff training",
                },
            ] if include_recommendations else [],
            "data_sources": ["Pre/post surveys (n=245)", "Exit interviews (n=32)", "Administrative data"],
            "limitations": [
                "Self-selection bias in interview sample",
                "Limited longitudinal follow-up data",
            ],
        }

        # Check if publishing requires consensus
        requires_approval = self.requires_consensus("publish_findings")

        return SkillResponse(
            content=f"Generated {report_type} findings report with {len(findings['key_findings'])} key findings "
                    f"and {len(findings['recommendations'])} recommendations.",
            success=True,
            data={"findings_report": findings},
            requires_consensus=requires_approval,
            consensus_action="publish_findings" if requires_approval else None,
            suggestions=[
                "Would you like to adjust any findings?",
                "Should I create a presentation version?",
                "Want to share with stakeholders for review?",
            ],
        )
