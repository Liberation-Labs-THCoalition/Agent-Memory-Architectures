"""
Board Liaison Skill Chip for Kintsugi CMA.

Supports board governance activities including meeting preparation, minute
drafting, resolution tracking, compliance monitoring, and board reporting.
Emphasizes transparency and accountability in all governance operations.

Example:
    chip = BoardLiaisonChip()
    request = SkillRequest(
        intent="meeting_prep",
        entities={"meeting_date": "2024-03-15", "meeting_type": "regular"},
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
class BoardMeeting:
    """Represents a board meeting."""
    meeting_id: str
    meeting_type: str  # regular, special, annual, executive
    scheduled_date: datetime
    location: str
    virtual_link: str | None = None
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    quorum_required: int = 5
    agenda_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Resolution:
    """Represents a board resolution."""
    resolution_id: str
    title: str
    description: str
    status: str  # pending, approved, rejected, tabled
    meeting_id: str
    vote_count: dict[str, int] = field(default_factory=dict)  # for, against, abstain
    effective_date: datetime | None = None
    responsible_party: str = ""


@dataclass
class ComplianceItem:
    """Represents a compliance requirement."""
    item_id: str
    requirement: str
    category: str  # legal, regulatory, fiduciary, policy
    due_date: datetime
    status: str  # compliant, pending, overdue, not_applicable
    responsible_party: str
    documentation: list[str] = field(default_factory=list)


class BoardLiaisonChip(BaseSkillChip):
    """Support board governance, meeting preparation, and resolution tracking.

    This chip assists staff and board members with governance activities
    including preparing meeting packets, drafting minutes, tracking
    resolutions, monitoring compliance, and generating board reports.

    Intents:
        meeting_prep: Prepare meeting agenda and materials
        minutes_draft: Draft meeting minutes
        resolution_track: Track resolution status and implementation
        compliance_check: Check compliance requirements
        board_report: Generate board reports

    Example:
        >>> chip = BoardLiaisonChip()
        >>> request = SkillRequest(intent="meeting_prep", entities={"meeting_type": "regular"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["meeting_packet"]["agenda_items"])
    """

    name = "board_liaison"
    description = "Support board governance, meeting preparation, and resolution tracking"
    version = "1.0.0"
    domain = SkillDomain.GOVERNANCE

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.20,
        resource_efficiency=0.15,
        transparency=0.30,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SCHEDULE_TASKS,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["distribute_materials", "record_resolution", "update_bylaws"]

    required_spans = ["document_templates", "calendar_api", "voting_system"]

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
            "meeting_prep": self._prepare_meeting_packet,
            "minutes_draft": self._draft_minutes,
            "resolution_track": self._track_resolutions,
            "compliance_check": self._check_compliance,
            "board_report": self._generate_board_report,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for board liaison.",
                success=False,
                suggestions=[
                    "Try 'meeting_prep' to prepare meeting materials",
                    "Try 'minutes_draft' to draft meeting minutes",
                    "Try 'resolution_track' to track board resolutions",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for governance context.

        Extracts beliefs about board status, compliance, and governance
        priorities relevant to this chip.
        """
        governance_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("governance", "compliance", "board")
            or b.get("type") in ("board_status", "compliance_status", "resolution_pending")
        ]

        governance_desires = [
            d for d in desires
            if d.get("type") in ("improve_governance", "maintain_compliance", "board_effectiveness")
            or d.get("domain") == "governance"
        ]

        return {
            "beliefs": governance_beliefs,
            "desires": governance_desires,
            "intentions": intentions,
        }

    async def _prepare_meeting_packet(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Prepare board meeting agenda and materials packet.

        Creates comprehensive meeting packet including agenda, prior minutes,
        committee reports, and action items.
        """
        meeting_date = request.entities.get("meeting_date", datetime.now(timezone.utc).isoformat())
        meeting_type = request.entities.get("meeting_type", "regular")
        include_financials = request.entities.get("include_financials", True)

        meeting = BoardMeeting(
            meeting_id=f"mtg_{uuid4().hex[:8]}",
            meeting_type=meeting_type,
            scheduled_date=datetime.fromisoformat(meeting_date) if isinstance(meeting_date, str) else meeting_date,
            location="Main Conference Room",
            virtual_link="https://zoom.us/j/123456789",
            quorum_required=5,
        )

        # Build agenda based on meeting type
        agenda_items = [
            {
                "item_number": 1,
                "title": "Call to Order & Roll Call",
                "presenter": "Board Chair",
                "duration_minutes": 5,
                "type": "procedural",
            },
            {
                "item_number": 2,
                "title": "Approval of Prior Meeting Minutes",
                "presenter": "Board Secretary",
                "duration_minutes": 5,
                "type": "action",
                "attachments": ["minutes_2024_02_15.pdf"],
            },
        ]

        if include_financials:
            agenda_items.append({
                "item_number": 3,
                "title": "Financial Report",
                "presenter": "Treasurer",
                "duration_minutes": 15,
                "type": "report",
                "attachments": ["financial_statement_q1.pdf", "budget_variance_report.pdf"],
            })

        agenda_items.extend([
            {
                "item_number": 4,
                "title": "Executive Director Report",
                "presenter": "Executive Director",
                "duration_minutes": 15,
                "type": "report",
            },
            {
                "item_number": 5,
                "title": "Committee Reports",
                "presenter": "Committee Chairs",
                "duration_minutes": 20,
                "type": "report",
                "sub_items": ["Finance Committee", "Governance Committee", "Program Committee"],
            },
            {
                "item_number": 6,
                "title": "Old Business",
                "presenter": "Board Chair",
                "duration_minutes": 15,
                "type": "discussion",
                "items": ["Strategic plan implementation update", "Building lease renewal status"],
            },
            {
                "item_number": 7,
                "title": "New Business",
                "presenter": "Board Chair",
                "duration_minutes": 20,
                "type": "action",
            },
            {
                "item_number": 8,
                "title": "Adjournment",
                "presenter": "Board Chair",
                "duration_minutes": 2,
                "type": "procedural",
            },
        ])

        meeting.agenda_items = agenda_items

        packet = {
            "packet_id": f"pkt_{uuid4().hex[:8]}",
            "meeting": {
                "meeting_id": meeting.meeting_id,
                "type": meeting.meeting_type,
                "date": meeting.scheduled_date.isoformat(),
                "location": meeting.location,
                "virtual_link": meeting.virtual_link,
                "quorum_required": meeting.quorum_required,
            },
            "agenda_items": agenda_items,
            "attachments": [
                {"name": "Prior Meeting Minutes", "filename": "minutes_2024_02_15.pdf"},
                {"name": "Financial Statements", "filename": "financial_statement_q1.pdf"},
                {"name": "ED Report", "filename": "ed_report_march.pdf"},
            ],
            "action_items_from_prior": [
                {"item": "Review insurance coverage", "responsible": "Treasurer", "status": "complete"},
                {"item": "Schedule strategic planning retreat", "responsible": "ED", "status": "in_progress"},
            ],
            "total_duration_minutes": sum(item["duration_minutes"] for item in agenda_items),
        }

        requires_approval = self.requires_consensus("distribute_materials")

        return SkillResponse(
            content=f"Prepared {meeting_type} board meeting packet for {meeting.scheduled_date.strftime('%B %d, %Y')}. "
                    f"Agenda includes {len(agenda_items)} items with estimated duration of {packet['total_duration_minutes']} minutes.",
            success=True,
            data={"meeting_packet": packet},
            requires_consensus=requires_approval,
            consensus_action="distribute_materials" if requires_approval else None,
            suggestions=[
                "Should I add any additional agenda items?",
                "Would you like to include a consent agenda?",
                "Ready to distribute to board members?",
            ],
        )

    async def _draft_minutes(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Draft meeting minutes from meeting notes.

        Creates formal minutes document with attendance, motions, votes,
        and action items.
        """
        meeting_id = request.entities.get("meeting_id", "")
        notes = request.entities.get("notes", "")
        attendees = request.entities.get("attendees", [])

        minutes = {
            "minutes_id": f"min_{uuid4().hex[:8]}",
            "meeting_id": meeting_id,
            "status": "draft",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "header": {
                "organization": context.org_id,
                "meeting_type": "Regular Board Meeting",
                "date": datetime.now(timezone.utc).strftime("%B %d, %Y"),
                "location": "Main Conference Room / Virtual",
                "call_to_order": "6:00 PM",
                "adjournment": "7:45 PM",
            },
            "attendance": {
                "present": attendees or ["J. Smith (Chair)", "M. Johnson", "A. Williams", "R. Brown", "S. Davis"],
                "absent": ["L. Martinez (excused)"],
                "staff_present": ["T. Chen (ED)", "K. Patel (CFO)"],
                "guests": [],
                "quorum_achieved": True,
            },
            "proceedings": [
                {
                    "item": "Call to Order",
                    "notes": "Chair J. Smith called the meeting to order at 6:00 PM.",
                    "type": "procedural",
                },
                {
                    "item": "Approval of Minutes",
                    "notes": "Minutes from February 15, 2024 meeting were reviewed.",
                    "motion": "Motion to approve minutes as presented",
                    "moved_by": "M. Johnson",
                    "seconded_by": "A. Williams",
                    "vote": {"for": 5, "against": 0, "abstain": 0},
                    "result": "Approved unanimously",
                    "type": "action",
                },
                {
                    "item": "Financial Report",
                    "notes": "Treasurer presented Q1 financial statements showing organization is on budget with 3% variance.",
                    "type": "report",
                },
                {
                    "item": "New Business - Policy Update",
                    "notes": "Board discussed proposed updates to conflict of interest policy.",
                    "motion": "Motion to adopt revised conflict of interest policy",
                    "moved_by": "R. Brown",
                    "seconded_by": "S. Davis",
                    "vote": {"for": 5, "against": 0, "abstain": 0},
                    "result": "Approved unanimously",
                    "type": "action",
                },
            ],
            "action_items": [
                {
                    "action": "Distribute updated conflict of interest policy to all board members",
                    "responsible": "Board Secretary",
                    "due_date": (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d"),
                },
                {
                    "action": "Schedule Q2 finance committee meeting",
                    "responsible": "Treasurer",
                    "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d"),
                },
            ],
            "next_meeting": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%B %d, %Y"),
        }

        return SkillResponse(
            content=f"Drafted minutes for {minutes['header']['meeting_type']} held on {minutes['header']['date']}. "
                    f"Includes {len(minutes['proceedings'])} proceedings items and {len(minutes['action_items'])} action items.",
            success=True,
            data={"minutes": minutes},
            suggestions=[
                "Would you like to add any corrections?",
                "Should I format for board review?",
                "Ready to circulate for approval?",
            ],
        )

    async def _track_resolutions(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Track board resolution status and implementation.

        Monitors resolution implementation, responsible parties, and deadlines.
        """
        resolution_id = request.entities.get("resolution_id", "")
        action = request.entities.get("action", "list")  # list, update, create
        status_filter = request.entities.get("status", None)

        # Simulated resolutions (would pull from data store)
        resolutions = [
            Resolution(
                resolution_id="res_001",
                title="FY2024 Budget Approval",
                description="Approve operating budget for fiscal year 2024",
                status="approved",
                meeting_id="mtg_001",
                vote_count={"for": 6, "against": 0, "abstain": 1},
                effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                responsible_party="CFO",
            ),
            Resolution(
                resolution_id="res_002",
                title="Conflict of Interest Policy Update",
                description="Adopt revised conflict of interest policy",
                status="approved",
                meeting_id="mtg_002",
                vote_count={"for": 5, "against": 0, "abstain": 0},
                effective_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
                responsible_party="Board Secretary",
            ),
            Resolution(
                resolution_id="res_003",
                title="Strategic Plan Adoption",
                description="Adopt 2024-2027 strategic plan",
                status="pending",
                meeting_id="mtg_003",
                vote_count={},
                responsible_party="Executive Director",
            ),
            Resolution(
                resolution_id="res_004",
                title="Executive Session - Personnel Matter",
                description="Personnel matter discussed in executive session",
                status="tabled",
                meeting_id="mtg_002",
                vote_count={},
                responsible_party="Board Chair",
            ),
        ]

        # Filter by status if requested
        if status_filter:
            resolutions = [r for r in resolutions if r.status == status_filter]

        resolution_data = [
            {
                "resolution_id": r.resolution_id,
                "title": r.title,
                "description": r.description,
                "status": r.status,
                "meeting_id": r.meeting_id,
                "vote": r.vote_count,
                "effective_date": r.effective_date.isoformat() if r.effective_date else None,
                "responsible_party": r.responsible_party,
            }
            for r in resolutions
        ]

        status_summary = {
            "approved": len([r for r in resolutions if r.status == "approved"]),
            "pending": len([r for r in resolutions if r.status == "pending"]),
            "tabled": len([r for r in resolutions if r.status == "tabled"]),
            "rejected": len([r for r in resolutions if r.status == "rejected"]),
        }

        requires_approval = self.requires_consensus("record_resolution") if action == "create" else False

        return SkillResponse(
            content=f"Tracking {len(resolutions)} board resolutions. "
                    f"{status_summary['approved']} approved, {status_summary['pending']} pending, "
                    f"{status_summary['tabled']} tabled.",
            success=True,
            data={
                "resolutions": resolution_data,
                "summary": status_summary,
            },
            requires_consensus=requires_approval,
            consensus_action="record_resolution" if requires_approval else None,
            suggestions=[
                "Want to see details on a specific resolution?",
                "Should I create a new resolution?",
                "Would you like to update a resolution status?",
            ],
        )

    async def _check_compliance(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Check governance compliance requirements and status.

        Reviews compliance with legal, regulatory, fiduciary, and policy
        requirements.
        """
        category = request.entities.get("category", None)
        include_overdue = request.entities.get("include_overdue", True)

        compliance_items = [
            ComplianceItem(
                item_id="comp_001",
                requirement="Annual Form 990 Filing",
                category="legal",
                due_date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                status="pending",
                responsible_party="CFO",
                documentation=["form_990_draft.pdf"],
            ),
            ComplianceItem(
                item_id="comp_002",
                requirement="State Charitable Registration Renewal",
                category="regulatory",
                due_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
                status="compliant",
                responsible_party="ED",
                documentation=["state_registration_2024.pdf"],
            ),
            ComplianceItem(
                item_id="comp_003",
                requirement="Board Member Conflict of Interest Disclosure",
                category="fiduciary",
                due_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                status="overdue",
                responsible_party="Board Secretary",
                documentation=[],
            ),
            ComplianceItem(
                item_id="comp_004",
                requirement="D&O Insurance Renewal",
                category="fiduciary",
                due_date=datetime(2024, 4, 1, tzinfo=timezone.utc),
                status="compliant",
                responsible_party="CFO",
                documentation=["do_insurance_policy.pdf"],
            ),
            ComplianceItem(
                item_id="comp_005",
                requirement="Bylaws Review",
                category="policy",
                due_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
                status="pending",
                responsible_party="Governance Committee",
                documentation=["current_bylaws.pdf"],
            ),
        ]

        # Filter by category if specified
        if category:
            compliance_items = [c for c in compliance_items if c.category == category]

        # Filter out completed if not including overdue
        if not include_overdue:
            compliance_items = [c for c in compliance_items if c.status != "overdue"]

        compliance_data = [
            {
                "item_id": c.item_id,
                "requirement": c.requirement,
                "category": c.category,
                "due_date": c.due_date.isoformat(),
                "status": c.status,
                "responsible_party": c.responsible_party,
                "documentation": c.documentation,
                "days_until_due": (c.due_date - datetime.now(timezone.utc)).days,
            }
            for c in compliance_items
        ]

        status_summary = {
            "compliant": len([c for c in compliance_items if c.status == "compliant"]),
            "pending": len([c for c in compliance_items if c.status == "pending"]),
            "overdue": len([c for c in compliance_items if c.status == "overdue"]),
        }

        overdue_items = [c for c in compliance_data if c["status"] == "overdue"]
        upcoming_items = [c for c in compliance_data if 0 < c["days_until_due"] <= 30]

        return SkillResponse(
            content=f"Checked {len(compliance_items)} compliance requirements. "
                    f"{status_summary['compliant']} compliant, {status_summary['pending']} pending, "
                    f"{status_summary['overdue']} overdue. {len(upcoming_items)} items due within 30 days.",
            success=True,
            data={
                "compliance_items": compliance_data,
                "summary": status_summary,
                "overdue": overdue_items,
                "upcoming": upcoming_items,
            },
            suggestions=[
                "Want to address the overdue items?",
                "Should I set up reminders for upcoming deadlines?",
                "Would you like a compliance calendar?",
            ],
        )

    async def _generate_board_report(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Generate comprehensive board report.

        Creates report including organizational health, financials, programs,
        and governance status.
        """
        report_period = request.entities.get("period", "quarterly")
        include_financials = request.entities.get("include_financials", True)
        include_programs = request.entities.get("include_programs", True)

        report = {
            "report_id": f"br_{uuid4().hex[:8]}",
            "period": report_period,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "executive_summary": (
                "The organization remains on solid footing with stable finances, "
                "strong program outcomes, and engaged board governance. Key highlights "
                "include exceeding fundraising targets by 12% and launching two new programs."
            ),
            "organizational_health": {
                "overall_status": "healthy",
                "staff_retention": "92%",
                "volunteer_engagement": "245 active volunteers",
                "partner_organizations": "12 active partnerships",
            },
        }

        if include_financials:
            report["financial_summary"] = {
                "total_revenue_ytd": 1250000,
                "total_expenses_ytd": 1125000,
                "net_position": 125000,
                "budget_variance": "+3.2%",
                "months_operating_reserve": 4.5,
                "key_metrics": [
                    {"metric": "Fundraising ROI", "value": "4.2:1"},
                    {"metric": "Program expense ratio", "value": "82%"},
                    {"metric": "Admin expense ratio", "value": "12%"},
                ],
            }

        if include_programs:
            report["program_summary"] = {
                "active_programs": 5,
                "participants_served_ytd": 1850,
                "completion_rate": "88%",
                "satisfaction_score": "4.3/5.0",
                "highlights": [
                    "Youth mentorship program expanded to 3 new schools",
                    "Workforce development achieved 72% job placement rate",
                    "Community health initiative served 450 families",
                ],
            }

        report["governance_status"] = {
            "board_attendance_avg": "85%",
            "committees_active": 4,
            "resolutions_ytd": 8,
            "compliance_status": "1 overdue item requiring attention",
            "upcoming": [
                "Annual board retreat scheduled for May 2024",
                "Strategic plan review at next meeting",
            ],
        }

        report["risks_and_opportunities"] = {
            "risks": [
                {"risk": "Key grant renewal uncertain", "mitigation": "Diversifying funding sources"},
                {"risk": "Rising facility costs", "mitigation": "Exploring alternative locations"},
            ],
            "opportunities": [
                {"opportunity": "New foundation interest", "action": "Schedule introductory meetings"},
                {"opportunity": "Partnership with city government", "action": "Submit proposal by Q2"},
            ],
        }

        return SkillResponse(
            content=f"Generated {report_period} board report covering organizational health, "
                    f"{'financials, ' if include_financials else ''}{'programs, ' if include_programs else ''}"
                    f"governance status, and risks/opportunities.",
            success=True,
            data={"board_report": report},
            suggestions=[
                "Would you like to customize any sections?",
                "Should I prepare a presentation version?",
                "Ready to include in the board packet?",
            ],
        )
