"""
Crisis Response Skill Chip for Kintsugi CMA.

Handles rapid mobilization for emergencies, disasters, and urgent community
needs. Provides tiered escalation based on crisis severity.

This chip enables effective crisis response by:
- Sending alerts to community members and volunteers
- Mobilizing response teams based on crisis type and severity
- Coordinating resource deployment during emergencies
- Tracking incident status and generating debriefs

Example usage:
    from kintsugi.skills.community_aid import CrisisResponseChip
    from kintsugi.skills import SkillRequest, SkillContext, register_chip

    # Register the chip
    chip = CrisisResponseChip()
    register_chip(chip)

    # Send crisis alert
    request = SkillRequest(
        intent="crisis_alert",
        entities={
            "crisis_type": "natural_disaster",
            "severity": "high",
            "location": "Riverside neighborhood",
            "description": "Flooding affecting 50+ households"
        }
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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


class CrisisType(str, Enum):
    """Types of crises the system can respond to."""
    NATURAL_DISASTER = "natural_disaster"
    MEDICAL_EMERGENCY = "medical_emergency"
    HOUSING_CRISIS = "housing_crisis"
    FOOD_INSECURITY = "food_insecurity"
    PUBLIC_SAFETY = "public_safety"
    INFRASTRUCTURE = "infrastructure"
    ECONOMIC = "economic"
    COMMUNITY_VIOLENCE = "community_violence"
    ENVIRONMENTAL = "environmental"
    OTHER = "other"


class CrisisSeverity(str, Enum):
    """Severity levels for crisis escalation."""
    CRITICAL = "critical"   # Immediate life safety threat
    HIGH = "high"           # Significant impact, urgent response needed
    MEDIUM = "medium"       # Moderate impact, coordinated response
    LOW = "low"             # Limited impact, standard procedures


class IncidentStatus(str, Enum):
    """Status of a crisis incident."""
    REPORTED = "reported"
    ASSESSING = "assessing"
    ACTIVE_RESPONSE = "active_response"
    STABILIZING = "stabilizing"
    RECOVERY = "recovery"
    RESOLVED = "resolved"
    CLOSED = "closed"


class VolunteerStatus(str, Enum):
    """Status of volunteers in the response system."""
    AVAILABLE = "available"
    DEPLOYED = "deployed"
    ON_STANDBY = "on_standby"
    UNAVAILABLE = "unavailable"


@dataclass
class CrisisIncident:
    """Represents a crisis incident being managed."""
    id: str
    crisis_type: CrisisType
    severity: CrisisSeverity
    title: str
    description: str
    location: str
    affected_population: int
    reported_by: str
    reported_at: datetime
    status: IncidentStatus
    assigned_coordinator: str | None = None
    deployed_volunteers: list[str] = field(default_factory=list)
    deployed_resources: list[str] = field(default_factory=list)
    status_updates: list[dict[str, Any]] = field(default_factory=list)
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Volunteer:
    """Represents a crisis response volunteer."""
    id: str
    name: str
    contact_phone: str
    contact_email: str
    skills: list[str]
    certifications: list[str]
    location: str
    status: VolunteerStatus
    crisis_types: list[CrisisType]  # Types they can respond to
    max_hours_per_week: int
    current_hours: int = 0
    deployments: list[str] = field(default_factory=list)


@dataclass
class EmergencyResource:
    """Represents emergency resources available for deployment."""
    id: str
    name: str
    resource_type: str
    quantity: int
    location: str
    status: str
    crisis_types: list[CrisisType]
    deployment_time_hours: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrisisDebrief:
    """Represents a post-incident debrief."""
    id: str
    incident_id: str
    conducted_at: datetime
    participants: list[str]
    timeline: list[dict[str, Any]]
    what_worked: list[str]
    areas_for_improvement: list[str]
    lessons_learned: list[str]
    recommendations: list[str]
    follow_up_actions: list[dict[str, Any]]


class CrisisResponseChip(BaseSkillChip):
    """Rapid mobilization for emergencies with tiered escalation.

    This chip coordinates crisis response through:
    1. Alert broadcasting based on crisis severity
    2. Volunteer mobilization matching skills to needs
    3. Resource deployment tracking
    4. Status updates and coordination
    5. Post-incident debriefing

    Escalation tiers:
    - CRITICAL: Immediate all-hands alert, emergency protocol activation
    - HIGH: Rapid response team deployment, leadership notification
    - MEDIUM: Coordinated response, standard protocols
    - LOW: Monitored response, limited mobilization
    """

    name = "crisis_response"
    description = "Rapid mobilization for emergencies, disasters, and urgent community needs"
    version = "1.0.0"
    domain = SkillDomain.MUTUAL_AID

    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.35,
        resource_efficiency=0.15,
        transparency=0.10,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
        SkillCapability.SCHEDULE_TASKS,
    ]

    consensus_actions = ["activate_emergency_protocol", "release_emergency_funds"]

    required_spans = [
        "alert_system",
        "volunteer_dispatch",
        "resource_inventory",
        "emergency_contacts",
    ]

    # Simulated storage
    _incidents: dict[str, CrisisIncident] = {}
    _volunteers: dict[str, Volunteer] = {}
    _resources: dict[str, EmergencyResource] = {}
    _debriefs: dict[str, CrisisDebrief] = {}
    _alert_log: list[dict[str, Any]] = []

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request containing intent and entities
            context: Execution context with org/user info and BDI state

        Returns:
            SkillResponse with operation result
        """
        intent_handlers = {
            "crisis_alert": self._handle_crisis_alert,
            "mobilize_response": self._handle_mobilize_response,
            "resource_deploy": self._handle_resource_deploy,
            "status_update": self._handle_status_update,
            "debrief": self._handle_debrief,
        }

        handler = intent_handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}",
                success=False,
                data={"error": "unknown_intent", "valid_intents": list(intent_handlers.keys())},
            )

        return await handler(request, context)

    async def _handle_crisis_alert(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle crisis alert and initiate response.

        Entities expected:
            crisis_type: Type of crisis
            severity: Severity level (critical, high, medium, low)
            title: Brief title for the incident
            description: Detailed description
            location: Affected area
            affected_population: Estimated number affected
        """
        severity = CrisisSeverity(request.entities.get("severity", "medium"))

        # Critical severity requires emergency protocol consensus
        if severity == CrisisSeverity.CRITICAL:
            return SkillResponse(
                content="CRITICAL alert requires emergency protocol activation approval.",
                success=True,
                requires_consensus=True,
                consensus_action="activate_emergency_protocol",
                data={
                    "crisis_type": request.entities.get("crisis_type"),
                    "severity": severity.value,
                    "pending_action": "activate_emergency_protocol",
                },
            )

        incident = await self.send_alert(
            crisis_type=request.entities.get("crisis_type", "other"),
            severity=severity.value,
            title=request.entities.get("title", "Crisis Incident"),
            description=request.entities.get("description", ""),
            location=request.entities.get("location", ""),
            affected_population=request.entities.get("affected_population", 0),
            reported_by=context.user_id,
        )

        # Get escalation response
        escalation = self._get_escalation_response(severity)

        return SkillResponse(
            content=f"CRISIS ALERT: {incident.title}\n"
                    f"Severity: {severity.value.upper()}\n"
                    f"Location: {incident.location}\n"
                    f"Incident ID: {incident.id[:8]}...\n\n"
                    f"Escalation: {escalation['description']}",
            success=True,
            data={
                "incident_id": incident.id,
                "crisis_type": incident.crisis_type.value,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "escalation": escalation,
            },
            suggestions=[
                "Mobilize volunteers with 'mobilize crisis response'",
                "Deploy resources with 'deploy emergency resources'",
                "Update status with 'update incident status'",
            ],
        )

    async def _handle_mobilize_response(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle volunteer mobilization for crisis response.

        Entities expected:
            incident_id: ID of the incident
            volunteer_count: Number of volunteers needed
            skills_needed: List of skills required
            duration_hours: Expected duration
        """
        incident_id = request.entities.get("incident_id")
        if not incident_id or incident_id not in self._incidents:
            return SkillResponse(
                content="Please specify a valid incident_id to mobilize response for.",
                success=False,
            )

        mobilized = await self.mobilize_volunteers(
            incident_id=incident_id,
            volunteer_count=request.entities.get("volunteer_count", 5),
            skills_needed=request.entities.get("skills_needed", []),
            duration_hours=request.entities.get("duration_hours", 4),
        )

        incident = self._incidents[incident_id]

        return SkillResponse(
            content=f"Mobilized {len(mobilized)} volunteers for incident {incident_id[:8]}...\n"
                    f"Volunteers have been notified and are being dispatched to {incident.location}.",
            success=True,
            data={
                "incident_id": incident_id,
                "volunteers_mobilized": len(mobilized),
                "volunteer_ids": [v.id for v in mobilized],
                "skills_covered": list({skill for v in mobilized for skill in v.skills}),
            },
            suggestions=[
                "Track volunteer status with 'show volunteer deployment status'",
                "Send update with 'broadcast incident update'",
            ],
        )

    async def _handle_resource_deploy(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle emergency resource deployment.

        Entities expected:
            incident_id: ID of the incident
            resource_types: Types of resources needed
            quantities: Quantities needed per type
            delivery_location: Where to deliver
        """
        incident_id = request.entities.get("incident_id")
        if not incident_id or incident_id not in self._incidents:
            return SkillResponse(
                content="Please specify a valid incident_id to deploy resources for.",
                success=False,
            )

        deployed = await self.deploy_resources(
            incident_id=incident_id,
            resource_types=request.entities.get("resource_types", []),
            quantities=request.entities.get("quantities", {}),
            delivery_location=request.entities.get("delivery_location", ""),
        )

        return SkillResponse(
            content=f"Deployed {len(deployed)} resource types for incident {incident_id[:8]}...",
            success=True,
            data={
                "incident_id": incident_id,
                "resources_deployed": [
                    {"name": r.name, "quantity": r.quantity, "eta_hours": r.deployment_time_hours}
                    for r in deployed
                ],
            },
        )

    async def _handle_status_update(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle incident status updates.

        Entities expected:
            incident_id: ID of the incident
            new_status: New status (optional)
            update_message: Status update message
            broadcast: Whether to broadcast update (default: true)
        """
        incident_id = request.entities.get("incident_id")
        if not incident_id or incident_id not in self._incidents:
            # Return summary of all active incidents
            active = [i for i in self._incidents.values()
                     if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]]
            return SkillResponse(
                content=f"Active incidents: {len(active)}",
                success=True,
                data={
                    "active_incidents": [
                        {
                            "id": i.id,
                            "title": i.title,
                            "severity": i.severity.value,
                            "status": i.status.value,
                        }
                        for i in active
                    ]
                },
            )

        updated = await self.update_status(
            incident_id=incident_id,
            new_status=request.entities.get("new_status"),
            update_message=request.entities.get("update_message", ""),
            updated_by=context.user_id,
        )

        incident = self._incidents[incident_id]

        return SkillResponse(
            content=f"Incident {incident_id[:8]}... status: {incident.status.value}\n"
                    f"Latest update: {updated['message']}",
            success=True,
            data={
                "incident_id": incident_id,
                "current_status": incident.status.value,
                "update": updated,
                "total_updates": len(incident.status_updates),
            },
        )

    async def _handle_debrief(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle post-incident debriefing.

        Entities expected:
            incident_id: ID of the resolved incident
            participants: List of participant IDs
            what_worked: List of things that worked well
            areas_for_improvement: Areas needing improvement
            lessons_learned: Key lessons
            recommendations: Recommendations for future
        """
        incident_id = request.entities.get("incident_id")
        if not incident_id or incident_id not in self._incidents:
            return SkillResponse(
                content="Please specify a valid incident_id for debriefing.",
                success=False,
            )

        debrief = await self.conduct_debrief(
            incident_id=incident_id,
            participants=request.entities.get("participants", [context.user_id]),
            what_worked=request.entities.get("what_worked", []),
            areas_for_improvement=request.entities.get("areas_for_improvement", []),
            lessons_learned=request.entities.get("lessons_learned", []),
            recommendations=request.entities.get("recommendations", []),
        )

        return SkillResponse(
            content=f"Debrief recorded for incident {incident_id[:8]}...\n"
                    f"Lessons learned: {len(debrief.lessons_learned)}\n"
                    f"Recommendations: {len(debrief.recommendations)}",
            success=True,
            data={
                "debrief_id": debrief.id,
                "incident_id": incident_id,
                "what_worked": debrief.what_worked,
                "areas_for_improvement": debrief.areas_for_improvement,
                "lessons_learned": debrief.lessons_learned,
                "recommendations": debrief.recommendations,
            },
            suggestions=[
                "Share debrief with 'distribute debrief report'",
                "Create action items with 'create follow-up tasks'",
            ],
        )

    # Core business logic methods

    async def send_alert(
        self,
        crisis_type: str,
        severity: str,
        title: str,
        description: str,
        location: str,
        affected_population: int,
        reported_by: str,
    ) -> CrisisIncident:
        """Send crisis alert and create incident.

        Args:
            crisis_type: Type of crisis
            severity: Severity level
            title: Incident title
            description: Detailed description
            location: Affected location
            affected_population: Number of people affected
            reported_by: ID of reporter

        Returns:
            The created CrisisIncident
        """
        now = datetime.now(timezone.utc)

        incident = CrisisIncident(
            id=str(uuid4()),
            crisis_type=CrisisType(crisis_type),
            severity=CrisisSeverity(severity),
            title=title,
            description=description,
            location=location,
            affected_population=affected_population,
            reported_by=reported_by,
            reported_at=now,
            status=IncidentStatus.REPORTED,
        )

        self._incidents[incident.id] = incident

        # Log alert
        self._alert_log.append({
            "incident_id": incident.id,
            "severity": severity,
            "sent_at": now.isoformat(),
            "recipients": self._get_alert_recipients(CrisisSeverity(severity)),
        })

        # Auto-transition to assessing
        incident.status = IncidentStatus.ASSESSING

        return incident

    async def mobilize_volunteers(
        self,
        incident_id: str,
        volunteer_count: int,
        skills_needed: list[str],
        duration_hours: int,
    ) -> list[Volunteer]:
        """Mobilize volunteers for crisis response.

        Args:
            incident_id: ID of the incident
            volunteer_count: Number of volunteers needed
            skills_needed: Required skills
            duration_hours: Expected duration

        Returns:
            List of mobilized volunteers
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            return []

        # Find available volunteers matching criteria
        available = [
            v for v in self._volunteers.values()
            if v.status == VolunteerStatus.AVAILABLE
            and incident.crisis_type in v.crisis_types
            and v.current_hours + duration_hours <= v.max_hours_per_week
        ]

        # Prioritize by skill match
        if skills_needed:
            available.sort(
                key=lambda v: len(set(v.skills) & set(skills_needed)),
                reverse=True
            )

        # Select volunteers
        mobilized = available[:volunteer_count]

        # Update status
        for volunteer in mobilized:
            volunteer.status = VolunteerStatus.DEPLOYED
            volunteer.deployments.append(incident_id)
            volunteer.current_hours += duration_hours
            incident.deployed_volunteers.append(volunteer.id)

        # Update incident status
        if incident.status == IncidentStatus.ASSESSING:
            incident.status = IncidentStatus.ACTIVE_RESPONSE

        return mobilized

    async def deploy_resources(
        self,
        incident_id: str,
        resource_types: list[str],
        quantities: dict[str, int],
        delivery_location: str,
    ) -> list[EmergencyResource]:
        """Deploy emergency resources.

        Args:
            incident_id: ID of the incident
            resource_types: Types of resources to deploy
            quantities: Quantities per type
            delivery_location: Where to deliver

        Returns:
            List of deployed resources
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            return []

        deployed = []
        for resource in self._resources.values():
            if resource.resource_type in resource_types:
                if resource.status == "available":
                    resource.status = "deployed"
                    incident.deployed_resources.append(resource.id)
                    deployed.append(resource)

        return deployed

    async def update_status(
        self,
        incident_id: str,
        new_status: str | None,
        update_message: str,
        updated_by: str,
    ) -> dict[str, Any]:
        """Update incident status.

        Args:
            incident_id: ID of the incident
            new_status: New status (optional)
            update_message: Update message
            updated_by: ID of updater

        Returns:
            Update record
        """
        incident = self._incidents.get(incident_id)
        if not incident:
            return {"error": "Incident not found"}

        now = datetime.now(timezone.utc)

        if new_status:
            incident.status = IncidentStatus(new_status)
            if new_status == "resolved":
                incident.resolved_at = now

        update = {
            "timestamp": now.isoformat(),
            "message": update_message,
            "updated_by": updated_by,
            "status": incident.status.value,
        }

        incident.status_updates.append(update)

        return update

    async def conduct_debrief(
        self,
        incident_id: str,
        participants: list[str],
        what_worked: list[str],
        areas_for_improvement: list[str],
        lessons_learned: list[str],
        recommendations: list[str],
    ) -> CrisisDebrief:
        """Conduct post-incident debrief.

        Args:
            incident_id: ID of the incident
            participants: Debrief participants
            what_worked: Things that worked well
            areas_for_improvement: Areas needing improvement
            lessons_learned: Key lessons
            recommendations: Future recommendations

        Returns:
            The created CrisisDebrief
        """
        incident = self._incidents.get(incident_id)

        # Build timeline from status updates
        timeline = []
        if incident:
            timeline = incident.status_updates.copy()

        debrief = CrisisDebrief(
            id=str(uuid4()),
            incident_id=incident_id,
            conducted_at=datetime.now(timezone.utc),
            participants=participants,
            timeline=timeline,
            what_worked=what_worked,
            areas_for_improvement=areas_for_improvement,
            lessons_learned=lessons_learned,
            recommendations=recommendations,
            follow_up_actions=[],
        )

        self._debriefs[debrief.id] = debrief

        # Mark incident as closed
        if incident and incident.status == IncidentStatus.RESOLVED:
            incident.status = IncidentStatus.CLOSED

        return debrief

    # Helper methods

    def _get_escalation_response(self, severity: CrisisSeverity) -> dict[str, Any]:
        """Get escalation response plan based on severity."""
        escalation_plans = {
            CrisisSeverity.CRITICAL: {
                "level": 4,
                "description": "Emergency protocol activated. All available responders alerted. "
                              "Leadership and emergency services notified.",
                "actions": [
                    "Activate all-hands alert",
                    "Notify emergency services",
                    "Open emergency operations center",
                    "Authorize emergency fund access",
                ],
                "notification_groups": ["all_volunteers", "leadership", "emergency_services"],
            },
            CrisisSeverity.HIGH: {
                "level": 3,
                "description": "Rapid response initiated. Response teams dispatched. "
                              "Leadership notified.",
                "actions": [
                    "Dispatch rapid response team",
                    "Notify leadership",
                    "Prepare resource deployment",
                ],
                "notification_groups": ["rapid_response", "leadership"],
            },
            CrisisSeverity.MEDIUM: {
                "level": 2,
                "description": "Coordinated response initiated. Relevant teams notified.",
                "actions": [
                    "Notify relevant volunteer teams",
                    "Assess resource needs",
                    "Establish coordination point",
                ],
                "notification_groups": ["relevant_teams", "coordinators"],
            },
            CrisisSeverity.LOW: {
                "level": 1,
                "description": "Monitored response. Standard procedures applied.",
                "actions": [
                    "Log incident",
                    "Monitor situation",
                    "Notify on-call coordinator",
                ],
                "notification_groups": ["on_call"],
            },
        }
        return escalation_plans.get(severity, escalation_plans[CrisisSeverity.LOW])

    def _get_alert_recipients(self, severity: CrisisSeverity) -> list[str]:
        """Get list of alert recipients based on severity."""
        if severity == CrisisSeverity.CRITICAL:
            return ["all_volunteers", "leadership", "partners", "emergency_contacts"]
        elif severity == CrisisSeverity.HIGH:
            return ["rapid_response_team", "leadership", "area_coordinators"]
        elif severity == CrisisSeverity.MEDIUM:
            return ["relevant_volunteers", "area_coordinators"]
        else:
            return ["on_call_coordinator"]

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for crisis response domain.

        Returns beliefs about current incidents and resources, desires for
        community safety, and intentions related to emergency response.
        """
        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in ["crisis", "emergency", "safety"]
                or b.get("type") in ["incident_status", "resource_availability", "volunteer_status"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["community_safety", "rapid_response", "harm_reduction"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") == "crisis_response"
                or i.get("action") in ["send_alert", "mobilize", "deploy_resources", "coordinate"]
            ],
        }
