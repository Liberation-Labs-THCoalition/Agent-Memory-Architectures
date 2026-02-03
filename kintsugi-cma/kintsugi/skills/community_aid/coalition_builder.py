"""
Coalition Builder Skill Chip for Kintsugi CMA.

Coordinates partner organizations, shared campaigns, and collective action
for effective community organizing and advocacy.

This chip enables coalition building by:
- Maintaining a directory of partner organizations with capacities
- Facilitating outreach and relationship building
- Coordinating shared campaigns and initiatives
- Scheduling coalition meetings and tracking progress
- Generating coalition impact reports

Example usage:
    from kintsugi.skills.community_aid import CoalitionBuilderChip
    from kintsugi.skills import SkillRequest, SkillContext, register_chip

    # Register the chip
    chip = CoalitionBuilderChip()
    register_chip(chip)

    # Search for potential partners
    request = SkillRequest(
        intent="partner_search",
        entities={
            "focus_areas": ["housing", "tenant_rights"],
            "location": "downtown",
            "org_type": "nonprofit"
        }
    )
    response = await chip.handle(request, context)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


class OrganizationType(str, Enum):
    """Types of partner organizations."""
    NONPROFIT = "nonprofit"
    GRASSROOTS = "grassroots"
    GOVERNMENT = "government"
    FAITH_BASED = "faith_based"
    LABOR_UNION = "labor_union"
    BUSINESS = "business"
    ACADEMIC = "academic"
    MEDIA = "media"
    FOUNDATION = "foundation"
    OTHER = "other"


class PartnershipStatus(str, Enum):
    """Status of partnerships."""
    PROSPECT = "prospect"
    OUTREACH = "outreach"
    NEGOTIATING = "negotiating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FORMER = "former"


class AlignmentLevel(str, Enum):
    """Level of mission alignment with partners."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class CampaignStatus(str, Enum):
    """Status of coalition campaigns."""
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MeetingType(str, Enum):
    """Types of coalition meetings."""
    REGULAR = "regular"
    PLANNING = "planning"
    EMERGENCY = "emergency"
    WORKING_GROUP = "working_group"
    CELEBRATION = "celebration"


@dataclass
class PartnerOrganization:
    """Represents a partner organization in the coalition."""
    id: str
    name: str
    org_type: OrganizationType
    mission: str
    focus_areas: list[str]
    location: str
    contact_name: str
    contact_email: str
    contact_phone: str
    website: str
    capacity: dict[str, Any]  # staff_size, budget_range, volunteer_count
    alignment: AlignmentLevel
    partnership_status: PartnershipStatus
    relationship_history: list[dict[str, Any]]
    mou_signed: bool
    joined_at: datetime | None
    created_at: datetime
    updated_at: datetime
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalitionCampaign:
    """Represents a shared campaign or initiative."""
    id: str
    name: str
    description: str
    goals: list[str]
    focus_areas: list[str]
    lead_org_id: str
    partner_ids: list[str]
    start_date: datetime
    end_date: datetime | None
    status: CampaignStatus
    milestones: list[dict[str, Any]]
    resources_committed: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalitionMeeting:
    """Represents a coalition meeting."""
    id: str
    title: str
    meeting_type: MeetingType
    scheduled_at: datetime
    location: str  # Physical or video link
    organizer_id: str
    invited_partner_ids: list[str]
    confirmed_partner_ids: list[str]
    agenda: list[str]
    notes: str
    action_items: list[dict[str, Any]]
    status: str  # scheduled, completed, cancelled
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutreachRecord:
    """Represents an outreach attempt to a potential partner."""
    id: str
    target_org_id: str
    outreach_type: str  # email, call, meeting, event
    conducted_at: datetime
    conducted_by: str
    summary: str
    response: str | None
    follow_up_date: datetime | None
    status: str  # sent, responded, no_response, declined


class CoalitionBuilderChip(BaseSkillChip):
    """Coordinate partner organizations and collective action.

    This chip supports coalition building through:
    1. Partner directory management with capacity tracking
    2. Strategic outreach and relationship building
    3. Campaign coordination across organizations
    4. Meeting scheduling and facilitation support
    5. Coalition impact reporting

    Key feature: Tracks partner capacities and alignment for
    strategic partnership development.
    """

    name = "coalition_builder"
    description = "Coordinate partner organizations, shared campaigns, and collective action"
    version = "1.0.0"
    domain = SkillDomain.COMMUNITY

    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.25,
        resource_efficiency=0.15,
        transparency=0.20,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
    ]

    consensus_actions = ["sign_mou", "joint_statement", "shared_campaign_launch"]

    required_spans = [
        "partner_directory",
        "campaign_tools",
        "shared_calendar",
    ]

    # Simulated storage
    _partners: dict[str, PartnerOrganization] = {}
    _campaigns: dict[str, CoalitionCampaign] = {}
    _meetings: dict[str, CoalitionMeeting] = {}
    _outreach: dict[str, OutreachRecord] = {}

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request containing intent and entities
            context: Execution context with org/user info and BDI state

        Returns:
            SkillResponse with operation result
        """
        intent_handlers = {
            "partner_search": self._handle_partner_search,
            "partner_outreach": self._handle_partner_outreach,
            "campaign_coordinate": self._handle_campaign_coordinate,
            "meeting_schedule": self._handle_meeting_schedule,
            "coalition_report": self._handle_coalition_report,
        }

        handler = intent_handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}",
                success=False,
                data={"error": "unknown_intent", "valid_intents": list(intent_handlers.keys())},
            )

        return await handler(request, context)

    async def _handle_partner_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle searching for potential or existing partners.

        Entities expected:
            focus_areas: List of focus areas to match
            location: Geographic location
            org_type: Organization type filter
            alignment: Minimum alignment level
            include_prospects: Whether to include prospects (default: true)
        """
        partners = await self.search_partners(
            focus_areas=request.entities.get("focus_areas"),
            location=request.entities.get("location"),
            org_type=request.entities.get("org_type"),
            alignment=request.entities.get("alignment"),
            include_prospects=request.entities.get("include_prospects", True),
        )

        if not partners:
            return SkillResponse(
                content="No partners found matching your criteria.",
                success=True,
                data={"partners": [], "count": 0},
                suggestions=[
                    "Try broadening your search criteria",
                    "Add a potential partner with 'add partner organization'",
                ],
            )

        summary_lines = [f"Found {len(partners)} partner(s):"]
        for partner in partners[:10]:
            alignment_indicator = {
                AlignmentLevel.HIGH: "[***]",
                AlignmentLevel.MEDIUM: "[**]",
                AlignmentLevel.LOW: "[*]",
                AlignmentLevel.UNKNOWN: "[?]",
            }
            summary_lines.append(
                f"\n{alignment_indicator[partner.alignment]} {partner.name}\n"
                f"   Type: {partner.org_type.value} | Status: {partner.partnership_status.value}\n"
                f"   Focus: {', '.join(partner.focus_areas[:3])}"
            )

        return SkillResponse(
            content="\n".join(summary_lines),
            success=True,
            data={
                "partners": [self._partner_to_dict(p) for p in partners],
                "count": len(partners),
            },
            suggestions=[
                "Initiate outreach with 'reach out to [partner name]'",
                "View partner details with 'show partner profile [name]'",
            ],
        )

    async def _handle_partner_outreach(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle partner outreach operations.

        Entities expected:
            action: send, record, or list
            partner_id: Target partner ID
            outreach_type: Type of outreach (email, call, meeting)
            message: Message content (for send)
            summary: Summary of interaction (for record)
            response: Partner's response (for record)
            follow_up_date: When to follow up
        """
        action = request.entities.get("action", "list")

        if action == "send":
            return await self._send_outreach(request, context)
        elif action == "record":
            return await self._record_outreach(request, context)
        else:  # list
            return await self._list_outreach(request, context)

    async def _send_outreach(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Send outreach to a partner."""
        outreach = await self.send_outreach(
            target_org_id=request.entities.get("partner_id", ""),
            outreach_type=request.entities.get("outreach_type", "email"),
            message=request.entities.get("message", ""),
            conducted_by=context.user_id,
        )

        partner = self._partners.get(outreach.target_org_id)
        partner_name = partner.name if partner else "Unknown"

        return SkillResponse(
            content=f"Outreach sent to {partner_name} via {outreach.outreach_type}.\n"
                    f"Outreach ID: {outreach.id[:8]}...",
            success=True,
            data={
                "outreach_id": outreach.id,
                "partner_id": outreach.target_org_id,
                "type": outreach.outreach_type,
                "status": outreach.status,
            },
            suggestions=[
                "Set reminder with 'remind me to follow up in [days] days'",
                "View outreach history with 'show outreach history [partner]'",
            ],
        )

    async def _record_outreach(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Record an outreach interaction."""
        outreach = await self.send_outreach(
            target_org_id=request.entities.get("partner_id", ""),
            outreach_type=request.entities.get("outreach_type", "meeting"),
            message=request.entities.get("summary", ""),
            conducted_by=context.user_id,
        )

        if response := request.entities.get("response"):
            outreach.response = response
            outreach.status = "responded"

        if follow_up := request.entities.get("follow_up_date"):
            if isinstance(follow_up, str):
                outreach.follow_up_date = datetime.fromisoformat(follow_up)
            else:
                outreach.follow_up_date = follow_up

        return SkillResponse(
            content=f"Outreach recorded. Follow-up scheduled for "
                    f"{outreach.follow_up_date.strftime('%Y-%m-%d') if outreach.follow_up_date else 'TBD'}.",
            success=True,
            data={"outreach_id": outreach.id},
        )

    async def _list_outreach(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """List outreach activities."""
        partner_id = request.entities.get("partner_id")

        outreach_list = list(self._outreach.values())
        if partner_id:
            outreach_list = [o for o in outreach_list if o.target_org_id == partner_id]

        # Sort by date
        outreach_list.sort(key=lambda o: o.conducted_at, reverse=True)

        pending_follow_ups = [
            o for o in outreach_list
            if o.follow_up_date and o.follow_up_date > datetime.now(timezone.utc)
        ]

        return SkillResponse(
            content=f"Total outreach activities: {len(outreach_list)}\n"
                    f"Pending follow-ups: {len(pending_follow_ups)}",
            success=True,
            data={
                "outreach": [
                    {
                        "id": o.id,
                        "partner_id": o.target_org_id,
                        "type": o.outreach_type,
                        "date": o.conducted_at.isoformat(),
                        "status": o.status,
                        "follow_up": o.follow_up_date.isoformat() if o.follow_up_date else None,
                    }
                    for o in outreach_list[:20]
                ],
                "pending_follow_ups": len(pending_follow_ups),
            },
        )

    async def _handle_campaign_coordinate(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle campaign coordination.

        Entities expected:
            action: create, update, list, or join
            campaign_id: Campaign ID (for update/join)
            name: Campaign name (for create)
            description: Campaign description
            goals: Campaign goals
            focus_areas: Focus areas
            partner_ids: Partner organizations to include
            start_date: Campaign start date
            end_date: Campaign end date (optional)
        """
        action = request.entities.get("action", "list")

        if action == "create":
            # Creating shared campaigns requires consensus
            if request.entities.get("is_launch", False):
                return SkillResponse(
                    content="Launching a shared campaign requires coalition approval.",
                    success=True,
                    requires_consensus=True,
                    consensus_action="shared_campaign_launch",
                    data={"pending_action": "launch_campaign"},
                )

            campaign = await self.coordinate_campaign(
                name=request.entities.get("name", ""),
                description=request.entities.get("description", ""),
                goals=request.entities.get("goals", []),
                focus_areas=request.entities.get("focus_areas", []),
                lead_org_id=context.org_id,
                partner_ids=request.entities.get("partner_ids", []),
                start_date=request.entities.get("start_date"),
                end_date=request.entities.get("end_date"),
            )

            return SkillResponse(
                content=f"Campaign '{campaign.name}' created (ID: {campaign.id[:8]}...).\n"
                        f"Partners: {len(campaign.partner_ids)}\n"
                        f"Status: {campaign.status.value}",
                success=True,
                data=self._campaign_to_dict(campaign),
                suggestions=[
                    "Add partners with 'add partner to campaign [id]'",
                    "Launch campaign with 'launch campaign [id]'",
                ],
            )

        elif action == "join":
            campaign_id = request.entities.get("campaign_id")
            if not campaign_id or campaign_id not in self._campaigns:
                return SkillResponse(
                    content="Please specify a valid campaign_id to join.",
                    success=False,
                )

            campaign = self._campaigns[campaign_id]
            if context.org_id not in campaign.partner_ids:
                campaign.partner_ids.append(context.org_id)

            return SkillResponse(
                content=f"Joined campaign '{campaign.name}'.",
                success=True,
                data=self._campaign_to_dict(campaign),
            )

        else:  # list
            campaigns = list(self._campaigns.values())
            active = [c for c in campaigns if c.status == CampaignStatus.ACTIVE]

            return SkillResponse(
                content=f"Total campaigns: {len(campaigns)}\nActive: {len(active)}",
                success=True,
                data={
                    "campaigns": [self._campaign_to_dict(c) for c in campaigns],
                    "active_count": len(active),
                },
            )

    async def _handle_meeting_schedule(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle coalition meeting scheduling.

        Entities expected:
            action: schedule, list, or update
            title: Meeting title
            meeting_type: Type of meeting
            scheduled_at: Meeting datetime
            location: Physical location or video link
            partner_ids: Partners to invite
            agenda: Meeting agenda items
        """
        action = request.entities.get("action", "list")

        if action == "schedule":
            meeting = await self.schedule_coalition_meeting(
                title=request.entities.get("title", "Coalition Meeting"),
                meeting_type=request.entities.get("meeting_type", "regular"),
                scheduled_at=request.entities.get("scheduled_at"),
                location=request.entities.get("location", ""),
                organizer_id=context.user_id,
                partner_ids=request.entities.get("partner_ids", []),
                agenda=request.entities.get("agenda", []),
            )

            return SkillResponse(
                content=f"Meeting '{meeting.title}' scheduled for "
                        f"{meeting.scheduled_at.strftime('%Y-%m-%d %H:%M')}.\n"
                        f"Invitations sent to {len(meeting.invited_partner_ids)} partners.",
                success=True,
                data={
                    "meeting_id": meeting.id,
                    "title": meeting.title,
                    "scheduled_at": meeting.scheduled_at.isoformat(),
                    "invited": len(meeting.invited_partner_ids),
                },
                suggestions=[
                    "Add agenda items with 'add to meeting agenda'",
                    "Send reminder with 'send meeting reminder'",
                ],
            )

        else:  # list
            meetings = list(self._meetings.values())
            upcoming = [
                m for m in meetings
                if m.status == "scheduled"
                and m.scheduled_at > datetime.now(timezone.utc)
            ]
            upcoming.sort(key=lambda m: m.scheduled_at)

            summary_lines = [f"Upcoming meetings: {len(upcoming)}"]
            for meeting in upcoming[:5]:
                summary_lines.append(
                    f"\n- {meeting.title}\n"
                    f"  {meeting.scheduled_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"  {len(meeting.confirmed_partner_ids)}/{len(meeting.invited_partner_ids)} confirmed"
                )

            return SkillResponse(
                content="\n".join(summary_lines),
                success=True,
                data={
                    "upcoming": [
                        {
                            "id": m.id,
                            "title": m.title,
                            "scheduled_at": m.scheduled_at.isoformat(),
                            "type": m.meeting_type.value,
                            "confirmed": len(m.confirmed_partner_ids),
                            "invited": len(m.invited_partner_ids),
                        }
                        for m in upcoming
                    ]
                },
            )

    async def _handle_coalition_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle coalition report generation.

        Entities expected:
            period: Time period (month, quarter, year)
            focus_area: Focus area filter (optional)
            include_metrics: Whether to include metrics (default: true)
        """
        report = await self.generate_coalition_report(
            period=request.entities.get("period", "quarter"),
            focus_area=request.entities.get("focus_area"),
            include_metrics=request.entities.get("include_metrics", True),
        )

        return SkillResponse(
            content=report["summary"],
            success=True,
            data=report,
            suggestions=[
                "Share report with 'share coalition report with partners'",
                "Export as PDF with 'export coalition report'",
            ],
        )

    # Core business logic methods

    async def search_partners(
        self,
        focus_areas: list[str] | None = None,
        location: str | None = None,
        org_type: str | None = None,
        alignment: str | None = None,
        include_prospects: bool = True,
    ) -> list[PartnerOrganization]:
        """Search for partner organizations.

        Args:
            focus_areas: Filter by focus areas
            location: Filter by location
            org_type: Filter by organization type
            alignment: Minimum alignment level
            include_prospects: Include prospect status partners

        Returns:
            List of matching partners
        """
        results = []

        for partner in self._partners.values():
            # Status filter
            if not include_prospects and partner.partnership_status == PartnershipStatus.PROSPECT:
                continue

            # Type filter
            if org_type and partner.org_type.value != org_type:
                continue

            # Location filter
            if location and location.lower() not in partner.location.lower():
                continue

            # Focus area filter
            if focus_areas:
                if not any(fa in partner.focus_areas for fa in focus_areas):
                    continue

            # Alignment filter
            if alignment:
                alignment_order = {
                    AlignmentLevel.HIGH: 3,
                    AlignmentLevel.MEDIUM: 2,
                    AlignmentLevel.LOW: 1,
                    AlignmentLevel.UNKNOWN: 0,
                }
                min_level = alignment_order.get(AlignmentLevel(alignment), 0)
                if alignment_order.get(partner.alignment, 0) < min_level:
                    continue

            results.append(partner)

        # Sort by alignment and status
        results.sort(
            key=lambda p: (
                -{"high": 3, "medium": 2, "low": 1, "unknown": 0}.get(p.alignment.value, 0),
                {"active": 0, "negotiating": 1, "outreach": 2, "prospect": 3}.get(p.partnership_status.value, 4),
            )
        )

        return results

    async def send_outreach(
        self,
        target_org_id: str,
        outreach_type: str,
        message: str,
        conducted_by: str,
    ) -> OutreachRecord:
        """Send outreach to a potential partner.

        Args:
            target_org_id: Target organization ID
            outreach_type: Type of outreach
            message: Outreach message/summary
            conducted_by: ID of person conducting outreach

        Returns:
            The created OutreachRecord
        """
        now = datetime.now(timezone.utc)

        outreach = OutreachRecord(
            id=str(uuid4()),
            target_org_id=target_org_id,
            outreach_type=outreach_type,
            conducted_at=now,
            conducted_by=conducted_by,
            summary=message,
            response=None,
            follow_up_date=now + timedelta(days=7),  # Default 7-day follow-up
            status="sent",
        )

        self._outreach[outreach.id] = outreach

        # Update partner status if exists
        if target_org_id in self._partners:
            partner = self._partners[target_org_id]
            if partner.partnership_status == PartnershipStatus.PROSPECT:
                partner.partnership_status = PartnershipStatus.OUTREACH
            partner.relationship_history.append({
                "date": now.isoformat(),
                "type": outreach_type,
                "summary": message,
            })

        return outreach

    async def coordinate_campaign(
        self,
        name: str,
        description: str,
        goals: list[str],
        focus_areas: list[str],
        lead_org_id: str,
        partner_ids: list[str],
        start_date: str | datetime | None,
        end_date: str | datetime | None = None,
    ) -> CoalitionCampaign:
        """Coordinate a coalition campaign.

        Args:
            name: Campaign name
            description: Campaign description
            goals: Campaign goals
            focus_areas: Focus areas
            lead_org_id: Lead organization ID
            partner_ids: Partner organization IDs
            start_date: Campaign start date
            end_date: Campaign end date

        Returns:
            The created CoalitionCampaign
        """
        now = datetime.now(timezone.utc)

        # Parse dates
        if isinstance(start_date, str):
            start_dt = datetime.fromisoformat(start_date)
        elif start_date is None:
            start_dt = now
        else:
            start_dt = start_date

        end_dt = None
        if end_date:
            if isinstance(end_date, str):
                end_dt = datetime.fromisoformat(end_date)
            else:
                end_dt = end_date

        campaign = CoalitionCampaign(
            id=str(uuid4()),
            name=name,
            description=description,
            goals=goals,
            focus_areas=focus_areas,
            lead_org_id=lead_org_id,
            partner_ids=partner_ids,
            start_date=start_dt,
            end_date=end_dt,
            status=CampaignStatus.PLANNING,
            milestones=[],
            resources_committed={},
            metrics={},
            created_at=now,
            updated_at=now,
        )

        self._campaigns[campaign.id] = campaign
        return campaign

    async def schedule_coalition_meeting(
        self,
        title: str,
        meeting_type: str,
        scheduled_at: str | datetime | None,
        location: str,
        organizer_id: str,
        partner_ids: list[str],
        agenda: list[str],
    ) -> CoalitionMeeting:
        """Schedule a coalition meeting.

        Args:
            title: Meeting title
            meeting_type: Type of meeting
            scheduled_at: Meeting datetime
            location: Meeting location
            organizer_id: Organizer ID
            partner_ids: Partners to invite
            agenda: Agenda items

        Returns:
            The created CoalitionMeeting
        """
        now = datetime.now(timezone.utc)

        # Parse scheduled time
        if isinstance(scheduled_at, str):
            scheduled_dt = datetime.fromisoformat(scheduled_at)
        elif scheduled_at is None:
            scheduled_dt = now + timedelta(days=7)  # Default to 1 week out
        else:
            scheduled_dt = scheduled_at

        meeting = CoalitionMeeting(
            id=str(uuid4()),
            title=title,
            meeting_type=MeetingType(meeting_type),
            scheduled_at=scheduled_dt,
            location=location,
            organizer_id=organizer_id,
            invited_partner_ids=partner_ids,
            confirmed_partner_ids=[],
            agenda=agenda,
            notes="",
            action_items=[],
            status="scheduled",
            created_at=now,
        )

        self._meetings[meeting.id] = meeting
        return meeting

    async def generate_coalition_report(
        self,
        period: str = "quarter",
        focus_area: str | None = None,
        include_metrics: bool = True,
    ) -> dict[str, Any]:
        """Generate coalition impact report.

        Args:
            period: Time period for report
            focus_area: Optional focus area filter
            include_metrics: Whether to include detailed metrics

        Returns:
            Report dictionary
        """
        # Count statistics
        total_partners = len(self._partners)
        active_partners = sum(
            1 for p in self._partners.values()
            if p.partnership_status == PartnershipStatus.ACTIVE
        )

        total_campaigns = len(self._campaigns)
        active_campaigns = sum(
            1 for c in self._campaigns.values()
            if c.status == CampaignStatus.ACTIVE
        )

        total_meetings = len(self._meetings)

        # Focus area breakdown
        focus_breakdown = {}
        for partner in self._partners.values():
            for fa in partner.focus_areas:
                if fa not in focus_breakdown:
                    focus_breakdown[fa] = 0
                focus_breakdown[fa] += 1

        # Generate summary
        summary = (
            f"Coalition Report ({period})\n"
            f"========================\n\n"
            f"Partnership Network:\n"
            f"  - Total partners: {total_partners}\n"
            f"  - Active partnerships: {active_partners}\n\n"
            f"Campaigns:\n"
            f"  - Total campaigns: {total_campaigns}\n"
            f"  - Active campaigns: {active_campaigns}\n\n"
            f"Meetings:\n"
            f"  - Meetings held: {total_meetings}\n\n"
            f"Focus Areas:\n"
        )

        for fa, count in sorted(focus_breakdown.items(), key=lambda x: -x[1])[:5]:
            summary += f"  - {fa}: {count} partners\n"

        report = {
            "summary": summary,
            "period": period,
            "total_partners": total_partners,
            "active_partners": active_partners,
            "total_campaigns": total_campaigns,
            "active_campaigns": active_campaigns,
            "meetings_held": total_meetings,
            "focus_breakdown": focus_breakdown,
        }

        if include_metrics:
            report["metrics"] = {
                "partnership_growth_rate": 0.15,  # Placeholder
                "campaign_completion_rate": 0.75,
                "meeting_attendance_rate": 0.82,
                "outreach_response_rate": 0.45,
            }

        return report

    # Helper methods

    def _partner_to_dict(self, partner: PartnerOrganization) -> dict[str, Any]:
        """Convert partner to dictionary."""
        return {
            "id": partner.id,
            "name": partner.name,
            "type": partner.org_type.value,
            "mission": partner.mission,
            "focus_areas": partner.focus_areas,
            "location": partner.location,
            "alignment": partner.alignment.value,
            "status": partner.partnership_status.value,
            "mou_signed": partner.mou_signed,
            "capacity": partner.capacity,
        }

    def _campaign_to_dict(self, campaign: CoalitionCampaign) -> dict[str, Any]:
        """Convert campaign to dictionary."""
        return {
            "id": campaign.id,
            "name": campaign.name,
            "description": campaign.description,
            "goals": campaign.goals,
            "focus_areas": campaign.focus_areas,
            "partner_count": len(campaign.partner_ids),
            "status": campaign.status.value,
            "start_date": campaign.start_date.isoformat(),
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None,
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for coalition building domain.

        Returns beliefs about partnerships and campaigns, desires for
        collective impact, and intentions related to coalition work.
        """
        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in ["coalition", "partnerships", "campaigns"]
                or b.get("type") in ["partner_status", "campaign_progress", "relationship_health"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["coalition_strength", "collective_impact", "partnership_growth"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") == "coalition_building"
                or i.get("action") in ["outreach", "coordinate_campaign", "schedule_meeting"]
            ],
        }
