"""
Rapid Response skill chip for Kintsugi CMA.

Coordinates rapid response networks for ICE raids, bail funds, and
community emergencies. Manages encrypted communications with verified
responders and maintains operational security protocols.

CRITICAL: Privacy and security are paramount in this chip. Sensitive
location data is NOT logged. All communications should use encrypted
channels when possible.

Intents handled:
- raid_alert: Send alerts about ICE or police activity
- bail_request: Request bail fund support
- legal_hotline: Connect to emergency legal hotline
- safe_location: Find safe locations during emergencies
- response_debrief: Conduct post-response debriefs

Example:
    chip = RapidResponseChip()
    response = await chip.handle(
        SkillRequest(intent="legal_hotline", entities={"urgency": "emergency"}),
        context
    )
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
)


class AlertType(str, Enum):
    """Types of rapid response alerts."""
    ICE_RAID = "ice_raid"
    POLICE_ACTIVITY = "police_activity"
    WORKPLACE_RAID = "workplace_raid"
    TRAFFIC_CHECKPOINT = "checkpoint"
    COMMUNITY_EMERGENCY = "community_emergency"
    MEDICAL_EMERGENCY = "medical_emergency"


class UrgencyLevel(str, Enum):
    """Urgency levels for rapid response."""
    CRITICAL = "critical"  # Immediate response needed
    URGENT = "urgent"      # Response within 30 minutes
    STANDARD = "standard"  # Response within hours
    INFO = "info"         # Information only, no response needed


class ResponderStatus(str, Enum):
    """Status of verified responders."""
    AVAILABLE = "available"
    RESPONDING = "responding"
    UNAVAILABLE = "unavailable"
    STANDBY = "standby"


@dataclass
class RapidAlert:
    """A rapid response alert."""
    alert_id: str
    alert_type: AlertType
    urgency: UrgencyLevel
    description: str
    # NOTE: Location info intentionally vague for security
    general_area: str  # Neighborhood/area only, no specific addresses logged
    timestamp: datetime
    verified: bool
    responders_notified: int
    active: bool = True


@dataclass
class Responder:
    """A verified rapid response volunteer."""
    responder_id: str
    name: str  # First name only for privacy
    roles: list[str]
    languages: list[str]
    status: ResponderStatus
    last_active: datetime
    trainings_completed: list[str]
    # No location data stored for responders


@dataclass
class BailRequest:
    """A bail fund request (sensitive data minimized)."""
    request_id: str
    urgency: UrgencyLevel
    amount_needed: float
    status: str
    created: datetime
    # No identifying information stored here - linked via secure system


@dataclass
class SafeLocation:
    """A safe location for emergencies (details provided securely)."""
    location_id: str
    location_type: str  # sanctuary, legal_aid, medical, etc.
    general_area: str  # Neighborhood only
    services: list[str]
    languages: list[str]
    hours: str
    contact_method: str  # How to get specific address securely


class RapidResponseChip(BaseSkillChip):
    """Coordinate rapid response networks for emergencies.

    This chip manages rapid response communications for ICE raids,
    bail fund requests, and community emergencies. It prioritizes
    operational security and privacy - sensitive location data is
    never logged or stored.

    SECURITY PRINCIPLES:
    - No specific addresses logged
    - No identifying information stored unnecessarily
    - Encrypted communications preferred
    - Verified responder network
    - Operational security protocols enforced

    Attributes:
        name: Unique chip identifier
        description: Human-readable description
        domain: ADVOCACY domain for community defense
        efe_weights: Mission and stakeholder focused
        capabilities: READ_DATA, WRITE_DATA, SEND_NOTIFICATIONS, PII_ACCESS
        consensus_actions: Actions requiring approval
        required_spans: MCP tool spans needed
    """

    name = "rapid_response"
    description = "Coordinate rapid response networks for ICE raids, bail funds, and emergencies"
    domain = SkillDomain.ADVOCACY
    efe_weights = EFEWeights(
        mission_alignment=0.35,
        stakeholder_benefit=0.35,
        resource_efficiency=0.10,
        transparency=0.10,
        equity=0.10,
    )
    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
        SkillCapability.PII_ACCESS,
    ]
    consensus_actions = ["activate_rapid_response", "release_bail_funds", "share_location_alert"]
    required_spans = ["alert_network", "bail_fund", "legal_hotline", "safe_location_db"]

    # Security notice included in relevant responses
    SECURITY_NOTICE = (
        "SECURITY REMINDER: Do not share specific addresses or identifying "
        "information in unsecured channels. Use encrypted communications when possible."
    )

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with rapid response information
        """
        handlers = {
            "raid_alert": self._handle_raid_alert,
            "bail_request": self._handle_bail_request,
            "legal_hotline": self._handle_legal_hotline,
            "safe_location": self._handle_safe_location,
            "response_debrief": self._handle_debrief,
        }

        handler = handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}. Supported intents: {list(handlers.keys())}",
                success=False,
            )

        return await handler(request, context)

    async def _handle_raid_alert(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle raid or activity alerts.

        Args:
            request: Request with alert details
            context: Execution context

        Returns:
            SkillResponse with alert status
        """
        alert_type = request.entities.get("type", "ice_raid")
        urgency = request.entities.get("urgency", "urgent")
        # NOTE: We only store general area, never specific addresses
        general_area = request.entities.get("area")
        description = request.entities.get("description", "Activity reported")
        verified = request.entities.get("verified", False)

        # Create alert - specific location never logged
        alert = await self.send_raid_alert(
            alert_type=alert_type,
            urgency=urgency,
            general_area=general_area,
            description=description,
            verified=verified,
        )

        content = "**RAPID RESPONSE ALERT CREATED**\n\n"
        content += f"Alert ID: {alert.alert_id}\n"
        content += f"Type: {alert.alert_type.value.replace('_', ' ').upper()}\n"
        content += f"Urgency: {alert.urgency.value.upper()}\n"
        content += f"Area: {alert.general_area}\n"
        content += f"Verified: {'Yes' if alert.verified else 'Unverified - seeking confirmation'}\n"
        content += f"Responders Notified: {alert.responders_notified}\n\n"

        if alert.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT]:
            content += "**IMMEDIATE ACTIONS:**\n"
            content += "- Rapid response team has been alerted\n"
            content += "- Legal observers are being dispatched\n"
            content += "- Know-your-rights information being distributed\n\n"

        content += f"\n{self.SECURITY_NOTICE}"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "alert_id": alert.alert_id,
                "responders_notified": alert.responders_notified,
                "status": "active",
            },
            requires_consensus=True,
            consensus_action="activate_rapid_response",
            suggestions=[
                "Update alert status",
                "Request additional responders",
                "Connect to legal hotline",
            ],
        )

    async def send_raid_alert(
        self,
        alert_type: str,
        urgency: str,
        general_area: str | None,
        description: str,
        verified: bool = False,
    ) -> RapidAlert:
        """Send an alert to the rapid response network.

        NOTE: This method intentionally does NOT log specific locations.
        Only general neighborhood information is stored.

        Args:
            alert_type: Type of alert
            urgency: Urgency level
            general_area: General neighborhood (not specific address)
            description: Brief description
            verified: Whether alert has been verified

        Returns:
            Created alert object
        """
        # In production, this would interact with alert_network span
        # using encrypted channels

        try:
            alert_type_enum = AlertType(alert_type)
        except ValueError:
            alert_type_enum = AlertType.COMMUNITY_EMERGENCY

        try:
            urgency_enum = UrgencyLevel(urgency)
        except ValueError:
            urgency_enum = UrgencyLevel.URGENT

        return RapidAlert(
            alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            alert_type=alert_type_enum,
            urgency=urgency_enum,
            description=description,
            general_area=general_area or "Area not specified",
            timestamp=datetime.now(timezone.utc),
            verified=verified,
            responders_notified=15,  # Would be actual count in production
            active=True,
        )

    async def _handle_bail_request(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle bail fund requests.

        Args:
            request: Request with bail details
            context: Execution context

        Returns:
            SkillResponse with bail request status
        """
        urgency = request.entities.get("urgency", "urgent")
        amount = request.entities.get("amount")
        case_type = request.entities.get("case_type")

        result = await self.request_bail_support(
            urgency=urgency,
            amount=amount,
            case_type=case_type,
        )

        content = "**BAIL FUND REQUEST**\n\n"
        content += f"Request ID: {result['request_id']}\n"
        content += f"Status: {result['status']}\n"

        if result.get("next_steps"):
            content += "\n**Next Steps:**\n"
            for step in result["next_steps"]:
                content += f"- {step}\n"

        content += "\n**Important Information:**\n"
        content += "- A bail fund coordinator will contact you\n"
        content += "- Have case number and detention facility info ready\n"
        content += "- All information is kept confidential\n"

        if urgency == "critical":
            content += "\n**URGENT:** This request has been flagged as critical. "
            content += "Emergency protocols are in effect.\n"

        content += f"\n{self.SECURITY_NOTICE}"

        return SkillResponse(
            content=content,
            success=True,
            data=result,
            requires_consensus=True,
            consensus_action="release_bail_funds",
            suggestions=[
                "Check request status",
                "Contact bail fund directly",
                "Get legal hotline number",
            ],
        )

    async def request_bail_support(
        self,
        urgency: str,
        amount: float | None = None,
        case_type: str | None = None,
    ) -> dict[str, Any]:
        """Submit a bail fund support request.

        NOTE: Minimal information stored. Detailed case info
        collected through secure channels only.

        Args:
            urgency: Request urgency level
            amount: Bail amount if known
            case_type: Type of case if known

        Returns:
            Bail request result
        """
        # In production, this would interact with bail_fund span
        return {
            "request_id": f"BAIL-{datetime.now().strftime('%Y%m%d%H%M')}",
            "status": "Received - Under Review",
            "urgency": urgency,
            "next_steps": [
                "Bail fund coordinator will call within 2 hours",
                "Gather detention facility information",
                "Have case/booking number ready",
                "Identify an emergency contact",
            ],
            "hotline": "Contact information provided via secure channel",
        }

    async def _handle_legal_hotline(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Connect to emergency legal hotline.

        Args:
            request: Request with urgency and type
            context: Execution context

        Returns:
            SkillResponse with hotline connection info
        """
        urgency = request.entities.get("urgency", "standard")
        issue_type = request.entities.get("type", "general")
        language = request.entities.get("language", "en")

        hotline = await self.connect_legal_hotline(
            urgency=urgency,
            issue_type=issue_type,
            language=language,
        )

        content = "**EMERGENCY LEGAL HOTLINE**\n\n"

        if hotline["available"]:
            content += f"**Hotline: {hotline['number']}**\n"
            content += f"Hours: {hotline['hours']}\n"
            content += f"Languages: {', '.join(hotline['languages'])}\n\n"

            if urgency == "emergency":
                content += "**EMERGENCY:** Your call will be prioritized.\n"
                content += "When connected, say 'URGENT MATTER'\n\n"

            content += "**What to Have Ready:**\n"
            for item in hotline["prepare"]:
                content += f"- {item}\n"

            content += "\n**Remember:**\n"
            content += "- You have the right to remain silent\n"
            content += "- You have the right to an attorney\n"
            content += "- Do not sign anything you don't understand\n"
        else:
            content += "Hotline is currently outside operating hours.\n\n"
            content += "**Alternative Resources:**\n"
            for alt in hotline["alternatives"]:
                content += f"- {alt['name']}: {alt['contact']}\n"

        return SkillResponse(
            content=content,
            success=True,
            data=hotline,
            suggestions=[
                "Get know-your-rights info",
                "Find legal aid near me",
                "Request interpreter",
            ],
        )

    async def connect_legal_hotline(
        self,
        urgency: str,
        issue_type: str,
        language: str,
    ) -> dict[str, Any]:
        """Connect to the appropriate legal hotline.

        Args:
            urgency: Urgency level
            issue_type: Type of legal issue
            language: Preferred language

        Returns:
            Hotline connection information
        """
        # In production, this would check real-time availability
        # via legal_hotline span
        return {
            "available": True,
            "number": "1-800-LEGAL-NOW",
            "hours": "24/7 for emergencies, 8am-10pm standard",
            "languages": ["English", "Spanish", "Mandarin", "Vietnamese", "Korean"],
            "wait_time": "Under 5 minutes" if urgency == "emergency" else "10-15 minutes",
            "prepare": [
                "Location of person needing help",
                "Any case or booking numbers",
                "Name of detention facility if applicable",
                "Brief description of situation",
            ],
            "alternatives": [
                {"name": "Immigration Hotline", "contact": "1-800-IMM-HELP"},
                {"name": "ACLU", "contact": "aclu.org/contact"},
            ],
        }

    async def _handle_safe_location(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Find safe locations during emergencies.

        NOTE: Specific addresses are NEVER included in responses.
        Users are connected to coordinators who provide locations
        through secure channels.

        Args:
            request: Request with location needs
            context: Execution context

        Returns:
            SkillResponse with safe location process
        """
        location_type = request.entities.get("type", "general")
        services_needed = request.entities.get("services", [])
        language = request.entities.get("language", "en")

        locations = await self.find_safe_locations(
            location_type=location_type,
            services=services_needed,
            language=language,
        )

        content = "**SAFE LOCATION RESOURCES**\n\n"
        content += (
            "*For security reasons, specific addresses are provided only "
            "through secure channels after identity verification.*\n\n"
        )

        if locations:
            content += f"**{len(locations)} Safe Locations Available**\n\n"
            for loc in locations:
                content += f"**{loc.location_type.title()} - {loc.general_area}**\n"
                content += f"- Services: {', '.join(loc.services)}\n"
                content += f"- Languages: {', '.join(loc.languages)}\n"
                content += f"- Hours: {loc.hours}\n"
                content += f"- To get address: {loc.contact_method}\n\n"
        else:
            content += "No matching safe locations currently available.\n"
            content += "Please contact the hotline for emergency assistance.\n"

        content += "**How to Access:**\n"
        content += "1. Call the secure hotline\n"
        content += "2. Provide your verification code (if you have one)\n"
        content += "3. Coordinator will provide address via encrypted channel\n\n"

        content += f"{self.SECURITY_NOTICE}"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "locations_available": len(locations),
                "contact_method": "Secure hotline",
            },
            requires_consensus=True,
            consensus_action="share_location_alert",
            suggestions=[
                "Call secure hotline",
                "Get verification code",
                "Request escort to location",
            ],
        )

    async def find_safe_locations(
        self,
        location_type: str,
        services: list[str],
        language: str,
    ) -> list[SafeLocation]:
        """Find safe locations matching criteria.

        NOTE: Returns only general information. Specific addresses
        are NEVER returned through this method - only through
        verified secure channels.

        Args:
            location_type: Type of safe location needed
            services: Services required
            language: Language support needed

        Returns:
            List of safe location options (without specific addresses)
        """
        # In production, this would query safe_location_db span
        # with strict access controls
        return [
            SafeLocation(
                location_id="safe_001",
                location_type="sanctuary",
                general_area="East Bay",
                services=["shelter", "legal_consultation", "food"],
                languages=["English", "Spanish"],
                hours="24/7",
                contact_method="Call hotline with code HAVEN",
            ),
            SafeLocation(
                location_id="safe_002",
                location_type="legal_aid",
                general_area="Downtown",
                services=["legal_consultation", "document_help", "translation"],
                languages=["English", "Spanish", "Mandarin"],
                hours="9am-8pm M-Sat",
                contact_method="Call hotline with code SHIELD",
            ),
        ]

    async def _handle_debrief(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Conduct post-response debrief.

        Args:
            request: Request with debrief details
            context: Execution context

        Returns:
            SkillResponse with debrief structure
        """
        alert_id = request.entities.get("alert_id")
        response_type = request.entities.get("type", "general")

        debrief = await self.conduct_debrief(
            alert_id=alert_id,
            response_type=response_type,
        )

        content = "**RAPID RESPONSE DEBRIEF**\n\n"
        content += f"Response ID: {debrief['response_id']}\n"
        content += f"Date: {debrief['date']}\n\n"

        content += "**Debrief Questions:**\n"
        for i, question in enumerate(debrief["questions"], 1):
            content += f"{i}. {question}\n"

        content += "\n**Documentation Guidelines:**\n"
        for guideline in debrief["documentation_guidelines"]:
            content += f"- {guideline}\n"

        content += "\n**Support Resources:**\n"
        content += "Rapid response work can be stressful. Resources available:\n"
        for resource in debrief["support_resources"]:
            content += f"- {resource}\n"

        content += f"\n{self.SECURITY_NOTICE}"

        return SkillResponse(
            content=content,
            success=True,
            data=debrief,
            suggestions=[
                "Submit debrief notes",
                "Request support services",
                "Update training materials",
            ],
        )

    async def conduct_debrief(
        self,
        alert_id: str | None,
        response_type: str,
    ) -> dict[str, Any]:
        """Create a debrief structure for a rapid response.

        Args:
            alert_id: ID of the alert being debriefed
            response_type: Type of response

        Returns:
            Debrief structure and guidelines
        """
        return {
            "response_id": alert_id or f"DEBRIEF-{datetime.now().strftime('%Y%m%d')}",
            "date": datetime.now().strftime("%B %d, %Y"),
            "questions": [
                "What worked well in this response?",
                "What could be improved?",
                "Were there any safety concerns?",
                "Did we have adequate resources/responders?",
                "What follow-up is needed for affected community members?",
                "Are there any training gaps to address?",
                "Were communications effective and secure?",
            ],
            "documentation_guidelines": [
                "Do NOT include specific addresses or locations",
                "Do NOT include names or identifying information",
                "Focus on process improvements, not individuals",
                "Note resource needs for future responses",
                "Highlight successful tactics for training",
            ],
            "support_resources": [
                "Responder support hotline",
                "Community care circles",
                "Mental health resources",
                "Peer support network",
            ],
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for rapid response information.

        NOTE: This method filters carefully to avoid exposing
        sensitive location or timing information in BDI state.

        Args:
            beliefs: Current belief state
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Filtered BDI state (security-conscious)
        """
        # Security-conscious filtering - exclude location beliefs
        safe_belief_types = {
            "response_readiness",
            "network_status",
            "training_status",
            "resource_availability",
        }
        response_domains = {"rapid_response", "community_defense", "bail_fund"}

        return {
            "beliefs": [
                b for b in beliefs
                if b.get("type") in safe_belief_types
                and "location" not in str(b).lower()
                and "address" not in str(b).lower()
            ],
            "desires": [
                d for d in desires
                if d.get("type") in [
                    "community_safety",
                    "response_readiness",
                    "network_strength",
                ]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") in response_domains
                and "location" not in str(i).lower()
            ],
        }
