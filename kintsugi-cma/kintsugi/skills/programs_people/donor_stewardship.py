"""
Donor Stewardship Skill Chip for Kintsugi CMA.

Manages donor relationships including acknowledgments, cultivation planning,
giving history analysis, and stewardship reporting. Prioritizes stakeholder
benefit and transparency in all donor interactions.

Example:
    chip = DonorStewardshipChip()
    request = SkillRequest(
        intent="donor_thank",
        entities={"donor_id": "donor_001", "gift_amount": 1000},
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
class DonorProfile:
    """Represents a donor's profile and preferences."""
    donor_id: str
    name: str
    email: str
    phone: str | None = None
    donor_level: str = "friend"  # friend, supporter, patron, benefactor, champion
    first_gift_date: datetime | None = None
    total_lifetime_giving: float = 0.0
    communication_preference: str = "email"  # email, mail, phone
    interests: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Gift:
    """Represents a donation/gift."""
    gift_id: str
    donor_id: str
    amount: float
    gift_date: datetime
    gift_type: str  # one_time, recurring, in_kind, planned
    campaign: str | None = None
    designation: str = "unrestricted"
    acknowledged: bool = False
    acknowledgment_date: datetime | None = None


@dataclass
class CultivationActivity:
    """Represents a donor cultivation activity."""
    activity_id: str
    donor_id: str
    activity_type: str  # call, meeting, email, event, tour
    scheduled_date: datetime
    completed: bool = False
    notes: str = ""
    outcome: str | None = None


class DonorStewardshipChip(BaseSkillChip):
    """Manage donor relationships, acknowledgments, and cultivation.

    This chip supports fundraising staff in building and maintaining
    donor relationships through timely acknowledgments, strategic
    cultivation, giving analysis, and personalized stewardship.

    Intents:
        donor_thank: Generate thank you acknowledgment
        donor_profile: Retrieve or update donor profile
        giving_history: Analyze donor giving history
        cultivation_plan: Create cultivation plan
        stewardship_report: Generate stewardship report

    Example:
        >>> chip = DonorStewardshipChip()
        >>> request = SkillRequest(intent="donor_profile", entities={"donor_id": "donor_001"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["profile"]["donor_level"])
    """

    name = "donor_stewardship"
    description = "Manage donor relationships, acknowledgments, and cultivation"
    version = "1.0.0"
    domain = SkillDomain.FUNDRAISING

    efe_weights = EFEWeights(
        mission_alignment=0.20,
        stakeholder_benefit=0.35,
        resource_efficiency=0.15,
        transparency=0.20,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
        SkillCapability.PII_ACCESS,
    ]

    consensus_actions = ["send_acknowledgment", "update_donor_level", "share_donor_data"]

    required_spans = ["crm_api", "email_service", "gift_tracking"]

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
            "donor_thank": self._generate_thank_you,
            "donor_profile": self._get_donor_profile,
            "giving_history": self._analyze_giving,
            "cultivation_plan": self._create_cultivation_plan,
            "stewardship_report": self._generate_stewardship_report,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for donor stewardship.",
                success=False,
                suggestions=[
                    "Try 'donor_thank' to generate acknowledgment",
                    "Try 'donor_profile' to view donor information",
                    "Try 'cultivation_plan' to plan donor engagement",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for donor stewardship context.

        Extracts beliefs about donor relationships, giving patterns,
        and cultivation priorities.
        """
        fundraising_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("fundraising", "donors", "giving")
            or b.get("type") in ("donor_relationship", "giving_capacity", "cultivation_stage")
        ]

        stewardship_desires = [
            d for d in desires
            if d.get("type") in ("increase_retention", "upgrade_donors", "build_relationships")
            or d.get("domain") == "fundraising"
        ]

        return {
            "beliefs": fundraising_beliefs,
            "desires": stewardship_desires,
            "intentions": intentions,
        }

    async def _generate_thank_you(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Generate personalized thank you acknowledgment.

        Creates tax-deductible receipt and personalized thank you message
        based on gift amount and donor history.
        """
        donor_id = request.entities.get("donor_id", "")
        gift_amount = request.entities.get("gift_amount", 0)
        gift_type = request.entities.get("gift_type", "one_time")
        designation = request.entities.get("designation", "unrestricted")
        include_impact = request.entities.get("include_impact", True)

        # Simulated donor lookup (would query CRM)
        donor = DonorProfile(
            donor_id=donor_id,
            name="Sarah Johnson",
            email="sarah.johnson@email.com",
            donor_level="patron" if gift_amount >= 1000 else "supporter",
            first_gift_date=datetime(2022, 5, 15, tzinfo=timezone.utc),
            total_lifetime_giving=5250.0 + gift_amount,
            interests=["education", "youth programs"],
        )

        # Determine acknowledgment tier
        if gift_amount >= 10000:
            tier = "champion"
            signatory = "Board Chair"
        elif gift_amount >= 5000:
            tier = "benefactor"
            signatory = "Executive Director"
        elif gift_amount >= 1000:
            tier = "patron"
            signatory = "Development Director"
        else:
            tier = "supporter"
            signatory = "Development Team"

        # Build acknowledgment content
        acknowledgment = {
            "acknowledgment_id": f"ack_{uuid4().hex[:8]}",
            "donor_id": donor_id,
            "gift_amount": gift_amount,
            "gift_date": datetime.now(timezone.utc).isoformat(),
            "gift_type": gift_type,
            "designation": designation,
            "tier": tier,
            "tax_deductible_amount": gift_amount,  # Assuming no goods/services provided
            "receipt": {
                "organization_name": "Community Impact Organization",
                "ein": "XX-XXXXXXX",
                "address": "123 Main Street, Anytown, ST 12345",
                "statement": f"No goods or services were provided in exchange for this gift of ${gift_amount:,.2f}.",
            },
            "letter": {
                "salutation": f"Dear {donor.name.split()[0]},",
                "opening": f"Thank you for your generous gift of ${gift_amount:,.2f} to support our mission.",
                "impact_statement": (
                    f"Your support makes a real difference. This year, donors like you have helped us "
                    f"serve over 1,800 community members through our programs."
                ) if include_impact else None,
                "personal_note": (
                    f"As a loyal supporter since {donor.first_gift_date.year}, your continued "
                    f"commitment inspires our work every day."
                ),
                "closing": "With gratitude,",
                "signatory": signatory,
            },
            "delivery_method": donor.communication_preference,
        }

        requires_approval = self.requires_consensus("send_acknowledgment")

        return SkillResponse(
            content=f"Generated {tier}-level thank you acknowledgment for ${gift_amount:,.2f} gift. "
                    f"Ready to send via {donor.communication_preference}.",
            success=True,
            data={
                "acknowledgment": acknowledgment,
                "donor_summary": {
                    "name": donor.name,
                    "level": tier,
                    "lifetime_giving": donor.total_lifetime_giving,
                    "first_gift_year": donor.first_gift_date.year if donor.first_gift_date else None,
                },
            },
            requires_consensus=requires_approval,
            consensus_action="send_acknowledgment" if requires_approval else None,
            suggestions=[
                "Would you like to personalize the message further?",
                "Should I schedule a follow-up call?",
                "Want to add to the cultivation plan?",
            ],
        )

    async def _get_donor_profile(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Retrieve or update donor profile information.

        Returns comprehensive donor profile including giving history,
        engagement, and preferences.
        """
        donor_id = request.entities.get("donor_id", "")
        update_fields = request.entities.get("update", {})

        # Simulated donor profile (would query CRM)
        profile = DonorProfile(
            donor_id=donor_id,
            name="Sarah Johnson",
            email="sarah.johnson@email.com",
            phone="(555) 123-4567",
            donor_level="patron",
            first_gift_date=datetime(2022, 5, 15, tzinfo=timezone.utc),
            total_lifetime_giving=5250.0,
            communication_preference="email",
            interests=["education", "youth programs", "community development"],
            notes="Met at 2023 gala. Interested in board service.",
        )

        # Simulated giving history
        gifts = [
            Gift(
                gift_id="gift_001",
                donor_id=donor_id,
                amount=500.0,
                gift_date=datetime(2022, 5, 15, tzinfo=timezone.utc),
                gift_type="one_time",
                campaign="Spring Appeal",
                acknowledged=True,
            ),
            Gift(
                gift_id="gift_002",
                donor_id=donor_id,
                amount=1000.0,
                gift_date=datetime(2022, 12, 20, tzinfo=timezone.utc),
                gift_type="one_time",
                campaign="Year-End Campaign",
                designation="youth programs",
                acknowledged=True,
            ),
            Gift(
                gift_id="gift_003",
                donor_id=donor_id,
                amount=1500.0,
                gift_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
                gift_type="one_time",
                campaign="Annual Gala",
                acknowledged=True,
            ),
            Gift(
                gift_id="gift_004",
                donor_id=donor_id,
                amount=2250.0,
                gift_date=datetime(2023, 12, 15, tzinfo=timezone.utc),
                gift_type="one_time",
                campaign="Year-End Campaign",
                designation="unrestricted",
                acknowledged=True,
            ),
        ]

        # Calculate giving metrics
        current_year = datetime.now(timezone.utc).year
        gifts_this_year = [g for g in gifts if g.gift_date.year == current_year]
        gifts_last_year = [g for g in gifts if g.gift_date.year == current_year - 1]

        profile_data = {
            "donor_id": profile.donor_id,
            "name": profile.name,
            "email": profile.email,
            "phone": profile.phone,
            "donor_level": profile.donor_level,
            "first_gift_date": profile.first_gift_date.isoformat() if profile.first_gift_date else None,
            "total_lifetime_giving": profile.total_lifetime_giving,
            "communication_preference": profile.communication_preference,
            "interests": profile.interests,
            "notes": profile.notes,
            "giving_summary": {
                "total_gifts": len(gifts),
                "lifetime_total": sum(g.amount for g in gifts),
                "average_gift": sum(g.amount for g in gifts) / len(gifts) if gifts else 0,
                "largest_gift": max(g.amount for g in gifts) if gifts else 0,
                "this_year_total": sum(g.amount for g in gifts_this_year),
                "last_year_total": sum(g.amount for g in gifts_last_year),
                "giving_trend": "increasing" if sum(g.amount for g in gifts_this_year) > sum(g.amount for g in gifts_last_year) else "stable",
            },
            "recent_gifts": [
                {
                    "gift_id": g.gift_id,
                    "amount": g.amount,
                    "date": g.gift_date.isoformat(),
                    "campaign": g.campaign,
                    "designation": g.designation,
                }
                for g in sorted(gifts, key=lambda x: x.gift_date, reverse=True)[:5]
            ],
            "engagement": {
                "events_attended": 3,
                "volunteer_hours": 12,
                "referrals_made": 2,
                "last_contact": (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
            },
        }

        return SkillResponse(
            content=f"Retrieved profile for {profile.name} ({profile.donor_level} level). "
                    f"Lifetime giving: ${profile.total_lifetime_giving:,.2f} across {len(gifts)} gifts.",
            success=True,
            data={"profile": profile_data},
            suggestions=[
                "Would you like to update any profile information?",
                "Should I create a cultivation plan?",
                "Want to see full giving history?",
            ],
        )

    async def _analyze_giving(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Analyze donor giving patterns and trends.

        Provides insights into giving behavior, retention, and upgrade
        potential.
        """
        donor_id = request.entities.get("donor_id", "")
        analysis_type = request.entities.get("analysis_type", "comprehensive")
        time_period = request.entities.get("period", "all_time")

        # Simulated giving analysis
        analysis = {
            "donor_id": donor_id,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "period": time_period,
            "giving_pattern": {
                "frequency": "annual",
                "typical_timing": "year-end (December)",
                "preferred_channels": ["online", "event"],
                "responsive_to": ["email appeals", "personal asks"],
            },
            "trend_analysis": {
                "year_over_year_growth": "+28.5%",
                "average_gift_trend": "increasing",
                "giving_consistency": "high",
                "last_12_months": [
                    {"month": "Jan", "amount": 0},
                    {"month": "Feb", "amount": 0},
                    {"month": "Mar", "amount": 0},
                    {"month": "Apr", "amount": 0},
                    {"month": "May", "amount": 0},
                    {"month": "Jun", "amount": 1500},
                    {"month": "Jul", "amount": 0},
                    {"month": "Aug", "amount": 0},
                    {"month": "Sep", "amount": 0},
                    {"month": "Oct", "amount": 0},
                    {"month": "Nov", "amount": 0},
                    {"month": "Dec", "amount": 2250},
                ],
            },
            "capacity_indicators": {
                "wealth_screening_score": "medium-high",
                "estimated_capacity": "$5,000-$10,000",
                "current_giving_ratio": "45%",
                "upgrade_potential": "high",
            },
            "affinity_indicators": {
                "engagement_score": "high",
                "years_giving": 2,
                "event_participation": "regular",
                "volunteer_involvement": True,
                "board_potential": True,
            },
            "recommendations": [
                {
                    "recommendation": "Cultivate for major gift",
                    "rationale": "High capacity and affinity indicators suggest readiness for larger ask",
                    "suggested_ask": "$5,000",
                    "timing": "Q4 2024",
                },
                {
                    "recommendation": "Explore board service",
                    "rationale": "Interest noted in profile, strong engagement history",
                    "next_step": "Invitation to governance committee meeting",
                },
                {
                    "recommendation": "Monthly giving conversion",
                    "rationale": "Consistent annual donor may benefit from recurring giving option",
                    "suggested_amount": "$100/month",
                },
            ],
            "risk_factors": {
                "lapse_risk": "low",
                "downgrade_risk": "low",
                "notes": "Highly engaged, no indicators of disengagement",
            },
        }

        return SkillResponse(
            content=f"Completed {analysis_type} giving analysis. Donor shows high upgrade potential "
                    f"with {analysis['trend_analysis']['year_over_year_growth']} year-over-year growth. "
                    f"Recommended for major gift cultivation.",
            success=True,
            data={"giving_analysis": analysis},
            suggestions=[
                "Would you like to create a cultivation plan based on this analysis?",
                "Should I schedule a strategy meeting?",
                "Want to compare with similar donors?",
            ],
        )

    async def _create_cultivation_plan(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Create donor cultivation plan.

        Develops strategic plan for donor engagement and solicitation.
        """
        donor_id = request.entities.get("donor_id", "")
        goal = request.entities.get("goal", "major_gift")
        target_amount = request.entities.get("target_amount", 5000)
        timeline_months = request.entities.get("timeline_months", 6)

        now = datetime.now(timezone.utc)

        # Build cultivation activities
        activities = [
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="call",
                scheduled_date=now + timedelta(days=7),
                notes="Thank you call for recent gift, mention impact",
            ),
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="email",
                scheduled_date=now + timedelta(days=30),
                notes="Share program success story aligned with interests",
            ),
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="tour",
                scheduled_date=now + timedelta(days=60),
                notes="Site visit to see programs in action",
            ),
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="event",
                scheduled_date=now + timedelta(days=90),
                notes="Invite to donor appreciation event",
            ),
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="meeting",
                scheduled_date=now + timedelta(days=120),
                notes="Personal meeting with ED to discuss giving",
            ),
            CultivationActivity(
                activity_id=f"act_{uuid4().hex[:8]}",
                donor_id=donor_id,
                activity_type="call",
                scheduled_date=now + timedelta(days=150),
                notes=f"Solicitation call for ${target_amount:,.0f} gift",
            ),
        ]

        cultivation_plan = {
            "plan_id": f"cult_{uuid4().hex[:8]}",
            "donor_id": donor_id,
            "created_at": now.isoformat(),
            "goal": goal,
            "target_amount": target_amount,
            "timeline_months": timeline_months,
            "target_ask_date": (now + timedelta(days=timeline_months * 30)).isoformat(),
            "strategy": {
                "approach": "relationship-based cultivation",
                "key_messages": [
                    "Impact of their past support",
                    "Alignment with their interests in education/youth",
                    "Opportunity for deeper engagement",
                ],
                "involvement_opportunities": [
                    "Program committee participation",
                    "Event hosting",
                    "Peer-to-peer fundraising",
                ],
            },
            "activities": [
                {
                    "activity_id": a.activity_id,
                    "type": a.activity_type,
                    "scheduled_date": a.scheduled_date.isoformat(),
                    "notes": a.notes,
                    "completed": a.completed,
                }
                for a in activities
            ],
            "milestones": [
                {"milestone": "Initial thank you contact", "target_date": (now + timedelta(days=7)).isoformat()},
                {"milestone": "Engagement touchpoint", "target_date": (now + timedelta(days=30)).isoformat()},
                {"milestone": "In-person visit", "target_date": (now + timedelta(days=60)).isoformat()},
                {"milestone": "Solicitation meeting", "target_date": (now + timedelta(days=150)).isoformat()},
            ],
            "assigned_to": "Development Director",
            "backup_contact": "ED",
        }

        return SkillResponse(
            content=f"Created {timeline_months}-month cultivation plan targeting ${target_amount:,.0f} "
                    f"{goal.replace('_', ' ')}. Plan includes {len(activities)} touchpoints.",
            success=True,
            data={"cultivation_plan": cultivation_plan},
            suggestions=[
                "Would you like to adjust the timeline?",
                "Should I add these activities to the calendar?",
                "Want to add more personalized touches?",
            ],
        )

    async def _generate_stewardship_report(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Generate stewardship report for donors or leadership.

        Creates report on donor engagement, retention, and stewardship
        activities.
        """
        report_type = request.entities.get("report_type", "summary")
        period = request.entities.get("period", "quarterly")
        donor_segment = request.entities.get("segment", "all")

        report = {
            "report_id": f"stw_{uuid4().hex[:8]}",
            "report_type": report_type,
            "period": period,
            "segment": donor_segment,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_donors": 245,
                "new_donors": 32,
                "retained_donors": 178,
                "lapsed_donors": 15,
                "upgraded_donors": 24,
                "retention_rate": "87.5%",
            },
            "acknowledgments": {
                "total_sent": 312,
                "average_response_time": "2.3 days",
                "personalized_letters": 45,
                "calls_made": 28,
                "on_time_percentage": "96%",
            },
            "cultivation_activities": {
                "calls_completed": 85,
                "meetings_held": 23,
                "tours_given": 12,
                "events_hosted": 3,
                "total_touchpoints": 123,
            },
            "giving_metrics": {
                "total_raised": 287500,
                "average_gift": 1173.47,
                "median_gift": 250,
                "largest_gift": 25000,
                "recurring_donors": 45,
                "recurring_revenue_monthly": 4250,
            },
            "segment_performance": [
                {
                    "segment": "Champions ($10,000+)",
                    "count": 8,
                    "total_giving": 125000,
                    "retention": "100%",
                },
                {
                    "segment": "Benefactors ($5,000-$9,999)",
                    "count": 12,
                    "total_giving": 72000,
                    "retention": "92%",
                },
                {
                    "segment": "Patrons ($1,000-$4,999)",
                    "count": 35,
                    "total_giving": 56000,
                    "retention": "89%",
                },
                {
                    "segment": "Supporters ($250-$999)",
                    "count": 78,
                    "total_giving": 28500,
                    "retention": "85%",
                },
                {
                    "segment": "Friends (under $250)",
                    "count": 112,
                    "total_giving": 6000,
                    "retention": "78%",
                },
            ],
            "highlights": [
                "Achieved highest retention rate in 3 years",
                "24 donors upgraded to higher giving levels",
                "New monthly giving program launched with 45 participants",
            ],
            "action_items": [
                "Follow up with 15 lapsed donors from last year",
                "Schedule year-end solicitation calls with top 20 prospects",
                "Plan donor appreciation event for Q4",
            ],
        }

        return SkillResponse(
            content=f"Generated {period} stewardship report for {donor_segment} donors. "
                    f"Retention rate: {report['summary']['retention_rate']}. "
                    f"Total raised: ${report['giving_metrics']['total_raised']:,.2f}.",
            success=True,
            data={"stewardship_report": report},
            suggestions=[
                "Would you like to drill into a specific segment?",
                "Should I create action plans for lapsed donors?",
                "Want to compare with previous periods?",
            ],
        )
