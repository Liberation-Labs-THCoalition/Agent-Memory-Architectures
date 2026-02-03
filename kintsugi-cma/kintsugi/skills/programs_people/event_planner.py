"""
Event Planner Skill Chip for Kintsugi CMA.

Plans and coordinates events including RSVPs, logistics, accessibility,
and follow-up communications. Balances stakeholder benefit with resource
efficiency while ensuring equitable access through accessibility features.

Example:
    chip = EventPlannerChip()
    request = SkillRequest(
        intent="event_create",
        entities={"event_name": "Annual Gala", "event_date": "2024-05-15"},
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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
class Event:
    """Represents an event."""
    event_id: str
    name: str
    event_type: str  # gala, workshop, meeting, conference, webinar
    date: datetime
    end_date: datetime | None = None
    location: str = ""
    virtual_link: str | None = None
    capacity: int = 100
    status: str = "draft"  # draft, published, cancelled, completed
    description: str = ""
    budget: float = 0.0
    registration_required: bool = True


@dataclass
class RSVP:
    """Represents an event RSVP."""
    rsvp_id: str
    event_id: str
    attendee_name: str
    attendee_email: str
    status: str  # confirmed, declined, tentative, waitlist
    guests: int = 0
    dietary_requirements: str | None = None
    accessibility_needs: list[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessibilityFeature:
    """Represents an accessibility accommodation."""
    feature_id: str
    category: str  # mobility, visual, hearing, cognitive, dietary
    description: str
    available: bool = True
    notes: str = ""


class EventPlannerChip(BaseSkillChip):
    """Plan and coordinate events, RSVPs, logistics, and accessibility.

    This chip supports event planning through creation, RSVP management,
    logistics coordination, accessibility verification, and post-event
    follow-up. Prioritizes stakeholder experience while managing resources.

    Intents:
        event_create: Create new event
        event_rsvp: Manage RSVPs
        event_logistics: Coordinate logistics
        event_accessibility: Check accessibility features
        event_followup: Send follow-up communications

    Example:
        >>> chip = EventPlannerChip()
        >>> request = SkillRequest(intent="event_create", entities={"event_name": "Gala"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["event"]["event_id"])
    """

    name = "event_planner"
    description = "Plan and coordinate events, RSVPs, logistics, and accessibility"
    version = "1.0.0"
    domain = SkillDomain.OPERATIONS

    efe_weights = EFEWeights(
        mission_alignment=0.20,
        stakeholder_benefit=0.30,
        resource_efficiency=0.25,
        transparency=0.10,
        equity=0.15,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
        SkillCapability.SCHEDULE_TASKS,
    ]

    consensus_actions = ["finalize_event", "send_invitations", "commit_budget"]

    required_spans = ["calendar_api", "rsvp_system", "venue_booking", "accessibility_checker"]

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: Skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with result content and data
        """
        intent = request.intent

        bdi = await self.get_bdi_context(
            context.beliefs,
            context.desires,
            context.intentions,
        )

        handlers = {
            "event_create": self._create_event,
            "event_rsvp": self._manage_rsvps,
            "event_logistics": self._coordinate_logistics,
            "event_accessibility": self._check_accessibility,
            "event_followup": self._send_followup,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for event planner.",
                success=False,
                suggestions=[
                    "Try 'event_create' to create a new event",
                    "Try 'event_rsvp' to manage RSVPs",
                    "Try 'event_logistics' to coordinate logistics",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for event planning context.

        Extracts beliefs about events, venues, and stakeholder
        engagement priorities.
        """
        event_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("events", "operations", "community")
            or b.get("type") in ("venue_status", "budget_available", "stakeholder_interest")
        ]

        event_desires = [
            d for d in desires
            if d.get("type") in ("successful_event", "community_engagement", "donor_cultivation")
            or d.get("domain") == "events"
        ]

        return {
            "beliefs": event_beliefs,
            "desires": event_desires,
            "intentions": intentions,
        }

    async def _create_event(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Create a new event with initial configuration.

        Sets up event with date, location, capacity, and budget.
        """
        event_name = request.entities.get("event_name", "Untitled Event")
        event_type = request.entities.get("event_type", "meeting")
        event_date = request.entities.get("event_date", (datetime.now(timezone.utc) + timedelta(days=30)).isoformat())
        duration_hours = request.entities.get("duration_hours", 2)
        location = request.entities.get("location", "")
        virtual = request.entities.get("virtual", False)
        capacity = request.entities.get("capacity", 50)
        budget = request.entities.get("budget", 0.0)
        description = request.entities.get("description", "")

        event_dt = datetime.fromisoformat(event_date) if isinstance(event_date, str) else event_date
        end_dt = event_dt + timedelta(hours=duration_hours)

        event = Event(
            event_id=f"evt_{uuid4().hex[:8]}",
            name=event_name,
            event_type=event_type,
            date=event_dt,
            end_date=end_dt,
            location=location or ("Virtual" if virtual else "TBD"),
            virtual_link="https://zoom.us/j/123456789" if virtual else None,
            capacity=capacity,
            status="draft",
            description=description,
            budget=budget,
            registration_required=True,
        )

        # Generate event checklist based on type
        checklist = self._generate_event_checklist(event)

        # Calculate timeline milestones
        now = datetime.now(timezone.utc)
        days_until_event = (event_dt - now).days

        milestones = []
        if days_until_event >= 60:
            milestones.append({"milestone": "Finalize venue", "target": (event_dt - timedelta(days=60)).isoformat()})
        if days_until_event >= 45:
            milestones.append({"milestone": "Send save-the-dates", "target": (event_dt - timedelta(days=45)).isoformat()})
        if days_until_event >= 30:
            milestones.append({"milestone": "Open registration", "target": (event_dt - timedelta(days=30)).isoformat()})
        if days_until_event >= 14:
            milestones.append({"milestone": "Confirm vendors", "target": (event_dt - timedelta(days=14)).isoformat()})
        if days_until_event >= 7:
            milestones.append({"milestone": "Send reminders", "target": (event_dt - timedelta(days=7)).isoformat()})
        milestones.append({"milestone": "Event day", "target": event_dt.isoformat()})
        milestones.append({"milestone": "Send follow-up", "target": (event_dt + timedelta(days=2)).isoformat()})

        event_data = {
            "event_id": event.event_id,
            "name": event.name,
            "type": event.event_type,
            "date": event.date.isoformat(),
            "end_date": event.end_date.isoformat() if event.end_date else None,
            "duration_hours": duration_hours,
            "location": event.location,
            "virtual_link": event.virtual_link,
            "capacity": event.capacity,
            "status": event.status,
            "description": event.description,
            "budget": event.budget,
            "registration_required": event.registration_required,
            "checklist": checklist,
            "milestones": milestones,
            "days_until_event": days_until_event,
        }

        return SkillResponse(
            content=f"Created {event_type} event '{event_name}' for {event_dt.strftime('%B %d, %Y at %I:%M %p')}. "
                    f"Capacity: {capacity}. Status: draft. {days_until_event} days until event.",
            success=True,
            data={"event": event_data},
            suggestions=[
                "Would you like to finalize and publish the event?",
                "Should I check venue availability?",
                "Want to set up registration?",
            ],
        )

    def _generate_event_checklist(self, event: Event) -> list[dict[str, Any]]:
        """Generate event checklist based on event type."""
        base_checklist = [
            {"item": "Confirm date and time", "category": "planning", "completed": True},
            {"item": "Determine budget", "category": "planning", "completed": event.budget > 0},
            {"item": "Create event description", "category": "planning", "completed": bool(event.description)},
            {"item": "Set up registration", "category": "registration", "completed": False},
            {"item": "Review accessibility needs", "category": "accessibility", "completed": False},
        ]

        if event.event_type in ("gala", "conference"):
            base_checklist.extend([
                {"item": "Book venue", "category": "logistics", "completed": False},
                {"item": "Arrange catering", "category": "logistics", "completed": False},
                {"item": "Plan program/agenda", "category": "content", "completed": False},
                {"item": "Arrange A/V equipment", "category": "logistics", "completed": False},
                {"item": "Coordinate volunteers", "category": "staffing", "completed": False},
                {"item": "Prepare materials/signage", "category": "materials", "completed": False},
            ])

        if event.event_type == "webinar":
            base_checklist.extend([
                {"item": "Set up virtual platform", "category": "technology", "completed": False},
                {"item": "Test technology", "category": "technology", "completed": False},
                {"item": "Prepare presentation", "category": "content", "completed": False},
                {"item": "Plan engagement activities", "category": "content", "completed": False},
            ])

        if event.event_type == "workshop":
            base_checklist.extend([
                {"item": "Develop curriculum", "category": "content", "completed": False},
                {"item": "Prepare handouts", "category": "materials", "completed": False},
                {"item": "Arrange supplies", "category": "logistics", "completed": False},
            ])

        base_checklist.append({"item": "Send post-event follow-up", "category": "follow-up", "completed": False})

        return base_checklist

    async def _manage_rsvps(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Manage event RSVPs and registration.

        Handles registration, updates, and waitlist management.
        """
        event_id = request.entities.get("event_id", "")
        action = request.entities.get("action", "list")  # list, add, update, cancel
        attendee_data = request.entities.get("attendee", {})

        # Simulated RSVP data (would query actual database)
        rsvps = [
            RSVP(
                rsvp_id="rsvp_001",
                event_id=event_id,
                attendee_name="Alice Thompson",
                attendee_email="alice@example.com",
                status="confirmed",
                guests=1,
                dietary_requirements="vegetarian",
            ),
            RSVP(
                rsvp_id="rsvp_002",
                event_id=event_id,
                attendee_name="Bob Martinez",
                attendee_email="bob@example.com",
                status="confirmed",
                guests=0,
                accessibility_needs=["wheelchair_access"],
            ),
            RSVP(
                rsvp_id="rsvp_003",
                event_id=event_id,
                attendee_name="Carol Williams",
                attendee_email="carol@example.com",
                status="tentative",
                guests=2,
            ),
            RSVP(
                rsvp_id="rsvp_004",
                event_id=event_id,
                attendee_name="David Chen",
                attendee_email="david@example.com",
                status="declined",
                guests=0,
            ),
            RSVP(
                rsvp_id="rsvp_005",
                event_id=event_id,
                attendee_name="Elena Rodriguez",
                attendee_email="elena@example.com",
                status="waitlist",
                guests=1,
            ),
        ]

        # Handle add action
        if action == "add" and attendee_data:
            new_rsvp = RSVP(
                rsvp_id=f"rsvp_{uuid4().hex[:8]}",
                event_id=event_id,
                attendee_name=attendee_data.get("name", ""),
                attendee_email=attendee_data.get("email", ""),
                status=attendee_data.get("status", "confirmed"),
                guests=attendee_data.get("guests", 0),
                dietary_requirements=attendee_data.get("dietary", None),
                accessibility_needs=attendee_data.get("accessibility", []),
            )
            rsvps.append(new_rsvp)

        # Calculate statistics
        confirmed = [r for r in rsvps if r.status == "confirmed"]
        tentative = [r for r in rsvps if r.status == "tentative"]
        declined = [r for r in rsvps if r.status == "declined"]
        waitlist = [r for r in rsvps if r.status == "waitlist"]

        total_confirmed_attendees = sum(r.guests + 1 for r in confirmed)
        total_tentative_attendees = sum(r.guests + 1 for r in tentative)

        # Aggregate dietary and accessibility needs
        dietary_needs = {}
        accessibility_needs = {}

        for rsvp in confirmed + tentative:
            if rsvp.dietary_requirements:
                dietary_needs[rsvp.dietary_requirements] = dietary_needs.get(rsvp.dietary_requirements, 0) + 1
            for need in rsvp.accessibility_needs:
                accessibility_needs[need] = accessibility_needs.get(need, 0) + 1

        rsvp_data = {
            "event_id": event_id,
            "rsvps": [
                {
                    "rsvp_id": r.rsvp_id,
                    "name": r.attendee_name,
                    "email": r.attendee_email,
                    "status": r.status,
                    "guests": r.guests,
                    "dietary": r.dietary_requirements,
                    "accessibility": r.accessibility_needs,
                    "registered_at": r.registered_at.isoformat(),
                }
                for r in rsvps
            ],
            "summary": {
                "confirmed": len(confirmed),
                "tentative": len(tentative),
                "declined": len(declined),
                "waitlist": len(waitlist),
                "total_responses": len(rsvps),
                "confirmed_attendees": total_confirmed_attendees,
                "tentative_attendees": total_tentative_attendees,
                "expected_attendance": total_confirmed_attendees + (total_tentative_attendees // 2),
            },
            "special_needs": {
                "dietary": dietary_needs,
                "accessibility": accessibility_needs,
            },
        }

        return SkillResponse(
            content=f"RSVP summary for event: {len(confirmed)} confirmed ({total_confirmed_attendees} attendees), "
                    f"{len(tentative)} tentative, {len(declined)} declined, {len(waitlist)} on waitlist.",
            success=True,
            data={"rsvp_data": rsvp_data},
            suggestions=[
                "Would you like to send reminders to tentative responses?",
                "Should I move someone from waitlist to confirmed?",
                "Want to export the attendee list?",
            ],
        )

    async def _coordinate_logistics(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Coordinate event logistics including vendors, materials, and staffing.

        Creates logistics plan and tracks vendor arrangements.
        """
        event_id = request.entities.get("event_id", "")
        event_type = request.entities.get("event_type", "meeting")
        expected_attendance = request.entities.get("expected_attendance", 50)
        budget = request.entities.get("budget", 5000)

        logistics = {
            "event_id": event_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "venue": {
                "name": "Community Center Main Hall",
                "address": "456 Oak Street, Anytown, ST 12345",
                "capacity": 100,
                "rental_cost": 500,
                "setup_time": "3 hours before",
                "teardown_time": "2 hours after",
                "contact": "Venue Manager (555-123-4567)",
                "confirmed": True,
            },
            "catering": {
                "vendor": "Local Catering Co.",
                "menu": "Buffet lunch with vegetarian options",
                "per_person_cost": 25,
                "total_cost": expected_attendance * 25,
                "dietary_accommodations": ["vegetarian", "gluten-free", "dairy-free"],
                "serving_style": "buffet",
                "confirmed": False,
            },
            "av_equipment": {
                "microphone": {"quantity": 2, "type": "wireless", "provided_by": "venue"},
                "projector": {"quantity": 1, "type": "LCD", "provided_by": "organization"},
                "screen": {"quantity": 1, "type": "projection", "provided_by": "venue"},
                "speakers": {"quantity": 2, "type": "powered", "provided_by": "venue"},
                "laptop": {"quantity": 1, "type": "presentation", "provided_by": "organization"},
            },
            "staffing": {
                "event_lead": "Program Manager",
                "registration_table": 2,
                "room_monitors": 1,
                "tech_support": 1,
                "volunteers_needed": 4,
                "volunteers_confirmed": 2,
            },
            "materials": [
                {"item": "Name badges", "quantity": expected_attendance, "status": "ordered"},
                {"item": "Programs/agendas", "quantity": expected_attendance, "status": "pending"},
                {"item": "Sign-in sheets", "quantity": 5, "status": "ready"},
                {"item": "Signage/banners", "quantity": 3, "status": "pending"},
                {"item": "Table numbers", "quantity": 10, "status": "ready"},
            ],
            "timeline": {
                "day_before": [
                    {"time": "2:00 PM", "task": "Confirm all vendors"},
                    {"time": "4:00 PM", "task": "Prepare materials and supplies"},
                ],
                "event_day": [
                    {"time": "9:00 AM", "task": "Venue access and setup begins"},
                    {"time": "10:00 AM", "task": "A/V equipment check"},
                    {"time": "11:00 AM", "task": "Catering arrives"},
                    {"time": "11:30 AM", "task": "Registration table ready"},
                    {"time": "12:00 PM", "task": "Event start"},
                    {"time": "3:00 PM", "task": "Event end, teardown begins"},
                    {"time": "5:00 PM", "task": "Venue cleared"},
                ],
            },
            "budget_tracking": {
                "allocated": budget,
                "committed": 500 + (expected_attendance * 25),
                "spent": 500,
                "remaining": budget - 500 - (expected_attendance * 25),
                "line_items": [
                    {"category": "Venue", "budgeted": 500, "actual": 500},
                    {"category": "Catering", "budgeted": expected_attendance * 25, "actual": 0},
                    {"category": "Materials", "budgeted": 200, "actual": 75},
                    {"category": "A/V", "budgeted": 0, "actual": 0},
                    {"category": "Contingency", "budgeted": 300, "actual": 0},
                ],
            },
            "emergency_contacts": [
                {"role": "Event Lead", "name": "Program Manager", "phone": "(555) 111-2222"},
                {"role": "Venue Contact", "name": "Venue Manager", "phone": "(555) 123-4567"},
                {"role": "Catering Contact", "name": "Catering Manager", "phone": "(555) 234-5678"},
            ],
        }

        requires_approval = self.requires_consensus("commit_budget")

        return SkillResponse(
            content=f"Logistics plan created for event. Venue confirmed, catering pending. "
                    f"Budget: ${logistics['budget_tracking']['allocated']:,.0f} allocated, "
                    f"${logistics['budget_tracking']['committed']:,.0f} committed. "
                    f"Need {logistics['staffing']['volunteers_needed'] - logistics['staffing']['volunteers_confirmed']} more volunteers.",
            success=True,
            data={"logistics": logistics},
            requires_consensus=requires_approval,
            consensus_action="commit_budget" if requires_approval else None,
            suggestions=[
                "Would you like to confirm the catering?",
                "Should I send volunteer recruitment request?",
                "Want to review the event timeline?",
            ],
        )

    async def _check_accessibility(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Check accessibility features and accommodations.

        Reviews venue accessibility and planned accommodations.
        """
        event_id = request.entities.get("event_id", "")
        venue = request.entities.get("venue", "Community Center Main Hall")

        accessibility_features = [
            AccessibilityFeature(
                feature_id="acc_001",
                category="mobility",
                description="Wheelchair accessible entrance",
                available=True,
            ),
            AccessibilityFeature(
                feature_id="acc_002",
                category="mobility",
                description="Accessible parking spaces",
                available=True,
                notes="3 spaces available near main entrance",
            ),
            AccessibilityFeature(
                feature_id="acc_003",
                category="mobility",
                description="Accessible restrooms",
                available=True,
            ),
            AccessibilityFeature(
                feature_id="acc_004",
                category="mobility",
                description="Elevator access to all floors",
                available=True,
            ),
            AccessibilityFeature(
                feature_id="acc_005",
                category="hearing",
                description="Hearing loop/assistive listening",
                available=True,
                notes="Portable system available upon request",
            ),
            AccessibilityFeature(
                feature_id="acc_006",
                category="hearing",
                description="Sign language interpretation",
                available=False,
                notes="Can be arranged with 2 weeks notice",
            ),
            AccessibilityFeature(
                feature_id="acc_007",
                category="visual",
                description="Large print materials",
                available=True,
                notes="Will prepare upon request",
            ),
            AccessibilityFeature(
                feature_id="acc_008",
                category="visual",
                description="Screen reader compatible digital materials",
                available=True,
            ),
            AccessibilityFeature(
                feature_id="acc_009",
                category="cognitive",
                description="Quiet room available",
                available=True,
                notes="Room 105 designated as quiet space",
            ),
            AccessibilityFeature(
                feature_id="acc_010",
                category="dietary",
                description="Dietary accommodations (vegan, gluten-free, etc.)",
                available=True,
                notes="Specify needs during registration",
            ),
        ]

        # Organize by category
        by_category = {}
        for feature in accessibility_features:
            if feature.category not in by_category:
                by_category[feature.category] = []
            by_category[feature.category].append({
                "feature_id": feature.feature_id,
                "description": feature.description,
                "available": feature.available,
                "notes": feature.notes,
            })

        accessibility_data = {
            "event_id": event_id,
            "venue": venue,
            "overall_assessment": "accessible_with_limitations",
            "features": by_category,
            "available_count": len([f for f in accessibility_features if f.available]),
            "unavailable_count": len([f for f in accessibility_features if not f.available]),
            "recommendations": [
                {
                    "recommendation": "Add accessibility question to registration form",
                    "priority": "high",
                    "status": "pending",
                },
                {
                    "recommendation": "Arrange sign language interpretation if requested",
                    "priority": "medium",
                    "status": "pending",
                    "lead_time": "2 weeks",
                },
                {
                    "recommendation": "Prepare large print agendas and materials",
                    "priority": "medium",
                    "status": "pending",
                },
                {
                    "recommendation": "Include accessibility information in event communications",
                    "priority": "high",
                    "status": "pending",
                },
            ],
            "registration_requests": [
                {"need": "wheelchair_access", "count": 1},
                {"need": "dietary_vegetarian", "count": 3},
                {"need": "dietary_gluten_free", "count": 1},
            ],
            "accessibility_statement": (
                "We are committed to making this event accessible to all attendees. "
                "Please contact us at events@organization.org or (555) 123-4567 to request "
                "accommodations or discuss your accessibility needs."
            ),
        }

        return SkillResponse(
            content=f"Accessibility check for {venue}: {accessibility_data['available_count']} features available, "
                    f"{accessibility_data['unavailable_count']} may require advance arrangement. "
                    f"Current registrations include {len(accessibility_data['registration_requests'])} accessibility requests.",
            success=True,
            data={"accessibility": accessibility_data},
            suggestions=[
                "Would you like to arrange sign language interpretation?",
                "Should I add accessibility info to invitations?",
                "Want to update the registration form?",
            ],
        )

    async def _send_followup(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Send post-event follow-up communications.

        Generates and sends thank you messages, surveys, and impact updates.
        """
        event_id = request.entities.get("event_id", "")
        followup_type = request.entities.get("followup_type", "thank_you")
        include_survey = request.entities.get("include_survey", True)
        include_photos = request.entities.get("include_photos", True)

        now = datetime.now(timezone.utc)

        # Generate follow-up content based on type
        followup_templates = {
            "thank_you": {
                "subject": "Thank You for Attending [Event Name]!",
                "opening": "Thank you for joining us at [Event Name]. We truly appreciate your participation and support.",
                "highlights": [
                    "Over 75 attendees joined us for this special event",
                    "Together we raised $15,000 for our programs",
                    "3 new volunteer partnerships were formed",
                ],
                "call_to_action": "Stay connected with us for future events and updates.",
            },
            "survey": {
                "subject": "We'd Love Your Feedback on [Event Name]",
                "opening": "Thank you for attending [Event Name]. Your feedback helps us improve future events.",
                "survey_link": "https://survey.organization.org/event-feedback",
                "incentive": "Complete the survey by [date] to be entered in a drawing for a $50 gift card.",
            },
            "impact": {
                "subject": "The Impact of [Event Name] - Thank You!",
                "opening": "Thanks to your participation in [Event Name], we're excited to share the impact we've made together.",
                "impact_metrics": [
                    {"metric": "Funds raised", "value": "$15,000"},
                    {"metric": "New donors", "value": "12"},
                    {"metric": "Program beneficiaries supported", "value": "50 families"},
                ],
            },
        }

        template = followup_templates.get(followup_type, followup_templates["thank_you"])

        followup_data = {
            "event_id": event_id,
            "followup_type": followup_type,
            "created_at": now.isoformat(),
            "scheduled_send": (now + timedelta(hours=24)).isoformat(),
            "recipients": {
                "count": 75,
                "segments": ["attendees", "donors", "volunteers"],
            },
            "content": template,
            "attachments": [],
        }

        if include_photos:
            followup_data["attachments"].append({
                "type": "photo_gallery",
                "description": "Event photos",
                "count": 15,
            })

        if include_survey:
            followup_data["survey"] = {
                "link": "https://survey.organization.org/event-feedback",
                "questions": [
                    "Overall satisfaction (1-5)",
                    "Would you recommend this event?",
                    "What did you enjoy most?",
                    "How can we improve?",
                ],
                "deadline": (now + timedelta(days=7)).isoformat(),
            }

        return SkillResponse(
            content=f"Prepared {followup_type} follow-up for {followup_data['recipients']['count']} recipients. "
                    f"{'Includes feedback survey. ' if include_survey else ''}"
                    f"{'Includes event photos. ' if include_photos else ''}"
                    f"Scheduled to send in 24 hours.",
            success=True,
            data={"followup": followup_data},
            requires_consensus=self.requires_consensus("send_invitations"),
            consensus_action="send_invitations" if self.requires_consensus("send_invitations") else None,
            suggestions=[
                "Would you like to customize the message?",
                "Should I send immediately instead of scheduling?",
                "Want to segment recipients differently?",
            ],
        )
