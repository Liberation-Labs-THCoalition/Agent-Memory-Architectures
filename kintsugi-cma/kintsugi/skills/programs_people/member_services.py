"""
Member Services Skill Chip for Kintsugi CMA.

Manages membership tracking, renewals, benefits, and communications.
Prioritizes stakeholder benefit while maintaining efficient membership
operations and transparent communications.

Example:
    chip = MemberServicesChip()
    request = SkillRequest(
        intent="member_lookup",
        entities={"member_id": "mem_001"},
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
class Member:
    """Represents a member record."""
    member_id: str
    name: str
    email: str
    phone: str | None = None
    membership_tier: str = "basic"  # basic, silver, gold, platinum
    status: str = "active"  # active, lapsed, pending, cancelled
    join_date: datetime | None = None
    expiration_date: datetime | None = None
    auto_renew: bool = False
    payment_method: str | None = None
    communication_preferences: dict[str, bool] = field(default_factory=dict)


@dataclass
class MembershipTier:
    """Represents a membership tier with benefits."""
    tier_id: str
    name: str
    annual_fee: float
    benefits: list[str]
    description: str


@dataclass
class MemberCommunication:
    """Represents a member communication record."""
    communication_id: str
    member_id: str
    communication_type: str  # email, mail, sms
    subject: str
    status: str  # sent, pending, failed
    sent_at: datetime | None = None


class MemberServicesChip(BaseSkillChip):
    """Manage membership tracking, renewals, benefits, and communications.

    This chip supports member services staff in managing the full
    membership lifecycle including lookups, renewals, benefits
    information, communications, and reporting.

    Intents:
        member_lookup: Look up member information
        membership_renew: Process membership renewal
        benefits_info: Provide membership benefits information
        member_communicate: Send member communications
        membership_report: Generate membership reports

    Example:
        >>> chip = MemberServicesChip()
        >>> request = SkillRequest(intent="member_lookup", entities={"member_id": "mem_001"})
        >>> response = await chip.handle(request, context)
        >>> print(response.data["member"]["membership_tier"])
    """

    name = "member_services"
    description = "Manage membership tracking, renewals, benefits, and communications"
    version = "1.0.0"
    domain = SkillDomain.MEMBER_SERVICES

    efe_weights = EFEWeights(
        mission_alignment=0.20,
        stakeholder_benefit=0.35,
        resource_efficiency=0.20,
        transparency=0.15,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
        SkillCapability.PII_ACCESS,
    ]

    consensus_actions = ["change_membership_tier", "process_refund", "bulk_communication"]

    required_spans = ["membership_db", "payment_processor", "email_service"]

    # Define membership tiers
    _tiers = {
        "basic": MembershipTier(
            tier_id="tier_basic",
            name="Basic",
            annual_fee=50.0,
            benefits=[
                "Monthly newsletter",
                "Discounted event tickets",
                "Member directory access",
            ],
            description="Entry-level membership with core benefits",
        ),
        "silver": MembershipTier(
            tier_id="tier_silver",
            name="Silver",
            annual_fee=100.0,
            benefits=[
                "All Basic benefits",
                "Quarterly member events",
                "Early event registration",
                "10% program discount",
            ],
            description="Enhanced membership with additional perks",
        ),
        "gold": MembershipTier(
            tier_id="tier_gold",
            name="Gold",
            annual_fee=250.0,
            benefits=[
                "All Silver benefits",
                "Free event tickets (2 per year)",
                "Annual recognition in publications",
                "Exclusive networking events",
                "20% program discount",
            ],
            description="Premium membership with exclusive access",
        ),
        "platinum": MembershipTier(
            tier_id="tier_platinum",
            name="Platinum",
            annual_fee=500.0,
            benefits=[
                "All Gold benefits",
                "VIP event access",
                "Board committee participation opportunity",
                "Personal concierge service",
                "Name on donor wall",
                "50% program discount",
            ],
            description="Top-tier membership with maximum benefits",
        ),
    }

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
            "member_lookup": self._lookup_member,
            "membership_renew": self._process_renewal,
            "benefits_info": self._get_benefits,
            "member_communicate": self._send_communication,
            "membership_report": self._generate_membership_report,
        }

        handler = handlers.get(intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent '{intent}' for member services.",
                success=False,
                suggestions=[
                    "Try 'member_lookup' to find a member",
                    "Try 'membership_renew' to process renewal",
                    "Try 'benefits_info' for membership benefits",
                ],
            )

        return await handler(request, context, bdi)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI state for member services context.

        Extracts beliefs about membership, engagement, and retention
        priorities.
        """
        member_beliefs = [
            b for b in beliefs
            if b.get("domain") in ("membership", "community", "engagement")
            or b.get("type") in ("member_status", "retention_risk", "engagement_level")
        ]

        service_desires = [
            d for d in desires
            if d.get("type") in ("increase_retention", "grow_membership", "improve_satisfaction")
            or d.get("domain") == "member_services"
        ]

        return {
            "beliefs": member_beliefs,
            "desires": service_desires,
            "intentions": intentions,
        }

    async def _lookup_member(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Look up member information and history.

        Returns comprehensive member profile with engagement history.
        """
        member_id = request.entities.get("member_id", "")
        email = request.entities.get("email", "")
        name = request.entities.get("name", "")

        # Simulated member lookup (would query membership database)
        member = Member(
            member_id=member_id or f"mem_{uuid4().hex[:8]}",
            name=name or "Jennifer Wilson",
            email=email or "jennifer.wilson@email.com",
            phone="(555) 987-6543",
            membership_tier="gold",
            status="active",
            join_date=datetime(2021, 3, 15, tzinfo=timezone.utc),
            expiration_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
            auto_renew=True,
            payment_method="credit_card",
            communication_preferences={
                "email_newsletter": True,
                "event_notifications": True,
                "renewal_reminders": True,
                "sms_alerts": False,
            },
        )

        now = datetime.now(timezone.utc)
        days_until_expiration = (member.expiration_date - now).days if member.expiration_date else None

        # Simulated engagement history
        engagement_history = [
            {"date": "2024-01-15", "activity": "Attended monthly networking event", "type": "event"},
            {"date": "2023-12-10", "activity": "Renewed membership", "type": "renewal"},
            {"date": "2023-11-20", "activity": "Attended annual gala", "type": "event"},
            {"date": "2023-09-05", "activity": "Volunteered at community cleanup", "type": "volunteer"},
            {"date": "2023-06-15", "activity": "Attended quarterly workshop", "type": "event"},
        ]

        member_data = {
            "member_id": member.member_id,
            "name": member.name,
            "email": member.email,
            "phone": member.phone,
            "membership_tier": member.membership_tier,
            "tier_benefits": self._tiers[member.membership_tier].benefits,
            "annual_fee": self._tiers[member.membership_tier].annual_fee,
            "status": member.status,
            "join_date": member.join_date.isoformat() if member.join_date else None,
            "expiration_date": member.expiration_date.isoformat() if member.expiration_date else None,
            "days_until_expiration": days_until_expiration,
            "auto_renew": member.auto_renew,
            "payment_method": member.payment_method,
            "communication_preferences": member.communication_preferences,
            "engagement": {
                "member_since_years": (now.year - member.join_date.year) if member.join_date else 0,
                "events_attended_ytd": 3,
                "volunteer_hours_ytd": 8,
                "recent_activities": engagement_history[:5],
            },
            "notes": [
                "Board committee interest expressed in 2023",
                "Referred 2 new members",
            ],
        }

        renewal_alert = ""
        if days_until_expiration and days_until_expiration <= 30:
            renewal_alert = f" ALERT: Membership expires in {days_until_expiration} days."

        return SkillResponse(
            content=f"Found member {member.name} ({member.membership_tier} tier, {member.status}). "
                    f"Member since {member.join_date.year if member.join_date else 'N/A'}.{renewal_alert}",
            success=True,
            data={"member": member_data},
            suggestions=[
                "Would you like to process a renewal?",
                "Should I send membership information?",
                "Want to update member preferences?",
            ],
        )

    async def _process_renewal(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Process membership renewal or new membership.

        Handles payment processing and membership period extension.
        """
        member_id = request.entities.get("member_id", "")
        tier = request.entities.get("tier", "basic")
        payment_method = request.entities.get("payment_method", "credit_card")
        auto_renew = request.entities.get("auto_renew", False)
        is_new = request.entities.get("is_new", False)

        if tier not in self._tiers:
            return SkillResponse(
                content=f"Invalid membership tier '{tier}'. Available tiers: {', '.join(self._tiers.keys())}",
                success=False,
            )

        tier_info = self._tiers[tier]
        now = datetime.now(timezone.utc)
        new_expiration = now + timedelta(days=365)

        renewal_data = {
            "transaction_id": f"txn_{uuid4().hex[:8]}",
            "member_id": member_id,
            "transaction_type": "new_membership" if is_new else "renewal",
            "tier": tier,
            "amount": tier_info.annual_fee,
            "payment_method": payment_method,
            "processed_at": now.isoformat(),
            "new_expiration_date": new_expiration.isoformat(),
            "auto_renew_enabled": auto_renew,
            "benefits_activated": tier_info.benefits,
            "confirmation": {
                "confirmation_number": f"CONF-{uuid4().hex[:8].upper()}",
                "receipt_sent": True,
                "welcome_email_queued": is_new,
            },
            "next_steps": [
                "Confirmation email sent to member",
                "Member card will be mailed within 5-7 business days" if tier in ["gold", "platinum"] else None,
                "Login credentials sent (new members)" if is_new else None,
            ],
        }

        # Remove None items from next_steps
        renewal_data["next_steps"] = [s for s in renewal_data["next_steps"] if s]

        action_type = "New membership" if is_new else "Renewal"

        return SkillResponse(
            content=f"{action_type} processed successfully! {tier_info.name} membership for "
                    f"${tier_info.annual_fee:.2f}. Valid through {new_expiration.strftime('%B %d, %Y')}.",
            success=True,
            data={"renewal": renewal_data},
            suggestions=[
                "Would you like to send a welcome packet?",
                "Should I schedule a new member orientation?",
                "Want to add any notes to the member record?",
            ],
        )

    async def _get_benefits(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Provide membership benefits information.

        Returns detailed benefits comparison across tiers.
        """
        tier = request.entities.get("tier", None)
        compare = request.entities.get("compare", True)

        if tier and tier in self._tiers:
            # Return specific tier information
            tier_info = self._tiers[tier]
            benefits_data = {
                "requested_tier": tier,
                "tier_info": {
                    "name": tier_info.name,
                    "annual_fee": tier_info.annual_fee,
                    "benefits": tier_info.benefits,
                    "description": tier_info.description,
                },
            }

            # Include upgrade option if not platinum
            if tier != "platinum":
                tier_order = ["basic", "silver", "gold", "platinum"]
                current_idx = tier_order.index(tier)
                next_tier = tier_order[current_idx + 1]
                next_tier_info = self._tiers[next_tier]

                benefits_data["upgrade_option"] = {
                    "next_tier": next_tier,
                    "additional_fee": next_tier_info.annual_fee - tier_info.annual_fee,
                    "additional_benefits": [
                        b for b in next_tier_info.benefits
                        if b not in tier_info.benefits and not b.startswith("All")
                    ],
                }

            return SkillResponse(
                content=f"{tier_info.name} membership: ${tier_info.annual_fee:.2f}/year with "
                        f"{len(tier_info.benefits)} benefits.",
                success=True,
                data={"benefits": benefits_data},
                suggestions=[
                    f"Would you like to upgrade to {benefits_data.get('upgrade_option', {}).get('next_tier', 'higher tier')}?" if tier != "platinum" else "This is our highest tier!",
                    "Should I send benefits information to a member?",
                ],
            )

        # Return comparison of all tiers
        comparison = {
            "tiers": [
                {
                    "tier_id": info.tier_id,
                    "name": info.name,
                    "annual_fee": info.annual_fee,
                    "benefits": info.benefits,
                    "description": info.description,
                    "value_highlight": self._get_value_highlight(name),
                }
                for name, info in self._tiers.items()
            ],
            "comparison_matrix": {
                "Newsletter": {"basic": True, "silver": True, "gold": True, "platinum": True},
                "Event discounts": {"basic": True, "silver": True, "gold": True, "platinum": True},
                "Member directory": {"basic": True, "silver": True, "gold": True, "platinum": True},
                "Quarterly events": {"basic": False, "silver": True, "gold": True, "platinum": True},
                "Early registration": {"basic": False, "silver": True, "gold": True, "platinum": True},
                "Program discounts": {"basic": "0%", "silver": "10%", "gold": "20%", "platinum": "50%"},
                "Free event tickets": {"basic": False, "silver": False, "gold": "2/year", "platinum": "Unlimited"},
                "VIP access": {"basic": False, "silver": False, "gold": False, "platinum": True},
                "Board participation": {"basic": False, "silver": False, "gold": False, "platinum": True},
            },
            "most_popular": "silver",
            "best_value": "gold",
        }

        return SkillResponse(
            content=f"Membership benefits comparison across {len(self._tiers)} tiers. "
                    f"Most popular: Silver. Best value: Gold.",
            success=True,
            data={"benefits_comparison": comparison},
            suggestions=[
                "Would you like details on a specific tier?",
                "Should I recommend a tier for a member?",
                "Want to see current member distribution by tier?",
            ],
        )

    def _get_value_highlight(self, tier_name: str) -> str:
        """Get value proposition highlight for a tier."""
        highlights = {
            "basic": "Great way to get started",
            "silver": "Most popular choice",
            "gold": "Best value for active members",
            "platinum": "Ultimate benefits package",
        }
        return highlights.get(tier_name, "")

    async def _send_communication(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Send communication to member(s).

        Handles individual and bulk member communications.
        """
        member_id = request.entities.get("member_id", "")
        communication_type = request.entities.get("type", "email")
        subject = request.entities.get("subject", "")
        template = request.entities.get("template", "general")
        is_bulk = request.entities.get("bulk", False)
        segment = request.entities.get("segment", "all_active")

        now = datetime.now(timezone.utc)

        # Communication templates
        templates = {
            "renewal_reminder": {
                "subject": "Your Membership Renewal is Coming Up",
                "preview": "Your membership expires soon. Renew today to keep your benefits.",
            },
            "welcome": {
                "subject": "Welcome to Our Community!",
                "preview": "Thank you for joining. Here's everything you need to get started.",
            },
            "benefits_update": {
                "subject": "New Member Benefits Available",
                "preview": "We've added exciting new benefits to your membership.",
            },
            "event_invitation": {
                "subject": "You're Invited: Exclusive Member Event",
                "preview": "Join us for a special event just for members.",
            },
            "general": {
                "subject": subject or "Message from Our Organization",
                "preview": "Important information for our valued members.",
            },
        }

        selected_template = templates.get(template, templates["general"])

        if is_bulk:
            # Segment definitions
            segments = {
                "all_active": {"count": 245, "description": "All active members"},
                "expiring_30": {"count": 28, "description": "Members expiring within 30 days"},
                "expiring_60": {"count": 45, "description": "Members expiring within 60 days"},
                "lapsed": {"count": 32, "description": "Lapsed members (expired < 90 days)"},
                "gold_platinum": {"count": 52, "description": "Gold and Platinum members"},
                "new_30": {"count": 18, "description": "New members (joined within 30 days)"},
            }

            segment_info = segments.get(segment, segments["all_active"])

            communication_data = {
                "communication_id": f"comm_{uuid4().hex[:8]}",
                "type": "bulk",
                "communication_type": communication_type,
                "template": template,
                "subject": selected_template["subject"],
                "preview": selected_template["preview"],
                "segment": segment,
                "segment_description": segment_info["description"],
                "recipient_count": segment_info["count"],
                "status": "pending_approval",
                "created_at": now.isoformat(),
                "scheduled_send": (now + timedelta(hours=24)).isoformat(),
            }

            requires_approval = self.requires_consensus("bulk_communication")

            return SkillResponse(
                content=f"Prepared bulk {communication_type} to {segment_info['count']} members "
                        f"({segment_info['description']}). Subject: '{selected_template['subject']}'. "
                        f"Pending approval before sending.",
                success=True,
                data={"communication": communication_data},
                requires_consensus=requires_approval,
                consensus_action="bulk_communication" if requires_approval else None,
                suggestions=[
                    "Would you like to preview the full message?",
                    "Should I adjust the recipient segment?",
                    "Want to schedule for a different time?",
                ],
            )

        # Individual communication
        communication = MemberCommunication(
            communication_id=f"comm_{uuid4().hex[:8]}",
            member_id=member_id,
            communication_type=communication_type,
            subject=selected_template["subject"],
            status="sent",
            sent_at=now,
        )

        communication_data = {
            "communication_id": communication.communication_id,
            "member_id": member_id,
            "type": "individual",
            "communication_type": communication_type,
            "template": template,
            "subject": communication.subject,
            "status": communication.status,
            "sent_at": communication.sent_at.isoformat() if communication.sent_at else None,
        }

        return SkillResponse(
            content=f"Sent {communication_type} to member {member_id}. "
                    f"Subject: '{communication.subject}'.",
            success=True,
            data={"communication": communication_data},
            suggestions=[
                "Would you like to log a follow-up task?",
                "Should I send to additional members?",
                "Want to schedule a reminder?",
            ],
        )

    async def _generate_membership_report(
        self,
        request: SkillRequest,
        context: SkillContext,
        bdi: dict[str, Any],
    ) -> SkillResponse:
        """Generate membership analytics and reports.

        Creates comprehensive membership statistics and trends.
        """
        report_type = request.entities.get("report_type", "summary")
        period = request.entities.get("period", "monthly")
        include_trends = request.entities.get("include_trends", True)

        now = datetime.now(timezone.utc)

        report = {
            "report_id": f"rpt_{uuid4().hex[:8]}",
            "report_type": report_type,
            "period": period,
            "generated_at": now.isoformat(),
            "summary": {
                "total_members": 277,
                "active_members": 245,
                "new_members_period": 18,
                "renewed_period": 42,
                "lapsed_period": 8,
                "net_growth": 10,
                "retention_rate": "89.2%",
            },
            "by_tier": {
                "basic": {"count": 98, "percentage": 40.0, "revenue": 4900.0},
                "silver": {"count": 95, "percentage": 38.8, "revenue": 9500.0},
                "gold": {"count": 38, "percentage": 15.5, "revenue": 9500.0},
                "platinum": {"count": 14, "percentage": 5.7, "revenue": 7000.0},
            },
            "by_status": {
                "active": 245,
                "pending_renewal": 28,
                "lapsed": 32,
                "cancelled": 15,
            },
            "revenue": {
                "period_total": 12500.0,
                "ytd_total": 48750.0,
                "average_member_value": 198.98,
                "recurring_monthly": 2150.0,
            },
            "engagement_metrics": {
                "event_participation_rate": "62%",
                "newsletter_open_rate": "45%",
                "volunteer_rate": "18%",
                "average_events_per_member": 2.3,
            },
        }

        if include_trends:
            report["trends"] = {
                "membership_growth": [
                    {"month": "Sep", "total": 255, "net_change": 5},
                    {"month": "Oct", "total": 262, "net_change": 7},
                    {"month": "Nov", "total": 270, "net_change": 8},
                    {"month": "Dec", "total": 277, "net_change": 7},
                    {"month": "Jan", "total": 285, "net_change": 8},
                    {"month": "Feb", "total": 293, "net_change": 8},
                ],
                "retention_by_tier": {
                    "basic": "82%",
                    "silver": "89%",
                    "gold": "94%",
                    "platinum": "98%",
                },
                "upgrade_trends": {
                    "basic_to_silver": 12,
                    "silver_to_gold": 8,
                    "gold_to_platinum": 3,
                },
                "churn_reasons": [
                    {"reason": "Financial/cost", "percentage": 35},
                    {"reason": "Not using benefits", "percentage": 28},
                    {"reason": "Moved away", "percentage": 18},
                    {"reason": "Other/unknown", "percentage": 19},
                ],
            }

        report["at_risk_members"] = {
            "expiring_30_days": 28,
            "expiring_60_days": 45,
            "low_engagement": 15,
            "recommended_outreach": 35,
        }

        report["recommendations"] = [
            {
                "recommendation": "Launch renewal campaign for 28 members expiring in 30 days",
                "priority": "high",
                "expected_impact": "Retain $5,600 in membership revenue",
            },
            {
                "recommendation": "Re-engage 15 low-engagement members with personalized outreach",
                "priority": "medium",
                "expected_impact": "Improve retention by 5%",
            },
            {
                "recommendation": "Promote Gold tier upgrade to Silver members with high engagement",
                "priority": "medium",
                "expected_impact": "12 potential upgrades worth $1,800",
            },
        ]

        return SkillResponse(
            content=f"Generated {period} membership report. Total: {report['summary']['total_members']} members "
                    f"({report['summary']['active_members']} active). Retention rate: {report['summary']['retention_rate']}. "
                    f"Period revenue: ${report['revenue']['period_total']:,.2f}.",
            success=True,
            data={"membership_report": report},
            suggestions=[
                "Would you like to drill into a specific segment?",
                "Should I create an outreach list for at-risk members?",
                "Want to compare with previous periods?",
            ],
        )
