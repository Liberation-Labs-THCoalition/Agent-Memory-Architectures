"""
Housing Navigator skill chip for Kintsugi CMA.

Navigates housing resources, voucher programs, and tenant rights for
community members seeking stable housing. Tracks voucher waitlists,
maintains landlord accountability information, and provides eviction
defense resources.

Intents handled:
- housing_search: Search for available housing matching criteria
- voucher_status: Check housing voucher application status
- tenant_rights: Get tenant rights information for jurisdiction
- landlord_lookup: Look up landlord history and accountability info
- eviction_defense: Get eviction defense resources and support

Example:
    chip = HousingNavigatorChip()
    response = await chip.handle(
        SkillRequest(intent="housing_search", entities={"bedrooms": 2, "voucher_type": "HCV"}),
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


class VoucherType(str, Enum):
    """Types of housing vouchers supported."""
    HCV = "housing_choice_voucher"  # Section 8
    PBV = "project_based_voucher"
    VASH = "veterans_supportive_housing"
    FUP = "family_unification_program"
    EHV = "emergency_housing_voucher"
    RAD = "rental_assistance_demonstration"


class HousingStatus(str, Enum):
    """Status of housing units."""
    AVAILABLE = "available"
    PENDING = "pending"
    WAITLIST = "waitlist"
    OCCUPIED = "occupied"
    UNAVAILABLE = "unavailable"


@dataclass
class HousingUnit:
    """Represents an available housing unit."""
    unit_id: str
    address: str
    city: str
    state: str
    zip_code: str
    bedrooms: int
    bathrooms: float
    rent: float
    accepts_voucher: bool
    voucher_types: list[VoucherType]
    landlord_id: str
    amenities: list[str]
    accessibility: list[str]
    status: HousingStatus
    available_date: datetime | None = None
    utilities_included: list[str] = field(default_factory=list)


@dataclass
class VoucherApplication:
    """Tracks a voucher application status."""
    application_id: str
    voucher_type: VoucherType
    status: str
    applied_date: datetime
    waitlist_position: int | None
    estimated_wait: str | None
    last_updated: datetime
    next_steps: list[str]


@dataclass
class LandlordRecord:
    """Information about a landlord's history."""
    landlord_id: str
    name: str
    properties_count: int
    accepts_vouchers: bool
    complaint_count: int
    violation_count: int
    avg_response_time: str
    community_rating: float  # 1-5 scale
    voucher_friendly_rating: float
    notes: list[str]


class HousingNavigatorChip(BaseSkillChip):
    """Navigate housing resources, voucher programs, and tenant rights.

    This chip helps community members find stable housing by searching
    available units, tracking voucher applications, providing tenant
    rights information, and maintaining landlord accountability records.

    Attributes:
        name: Unique chip identifier
        description: Human-readable description
        domain: ADVOCACY domain for housing rights
        efe_weights: Stakeholder-focused weights
        capabilities: READ_DATA, WRITE_DATA, EXTERNAL_API, PII_ACCESS
        consensus_actions: Actions requiring approval
        required_spans: MCP tool spans needed
    """

    name = "housing_navigator"
    description = "Navigate housing resources, voucher programs, and tenant rights"
    domain = SkillDomain.ADVOCACY
    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.40,
        resource_efficiency=0.10,
        transparency=0.15,
        equity=0.10,
    )
    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.EXTERNAL_API,
        SkillCapability.PII_ACCESS,
    ]
    consensus_actions = ["submit_voucher_application", "share_tenant_data"]
    required_spans = ["housing_database", "voucher_tracker", "landlord_registry", "hud_api"]

    # Fair Market Rent limits by bedroom count (sample, would come from HUD API)
    FMR_LIMITS: dict[int, float] = {
        0: 1200.0,  # Studio
        1: 1400.0,
        2: 1700.0,
        3: 2100.0,
        4: 2400.0,
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with housing information
        """
        handlers = {
            "housing_search": self._handle_housing_search,
            "voucher_status": self._handle_voucher_status,
            "tenant_rights": self._handle_tenant_rights,
            "landlord_lookup": self._handle_landlord_lookup,
            "eviction_defense": self._handle_eviction_defense,
        }

        handler = handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}. Supported intents: {list(handlers.keys())}",
                success=False,
            )

        return await handler(request, context)

    async def _handle_housing_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Search for available housing matching criteria.

        Args:
            request: Request with housing search criteria
            context: Execution context

        Returns:
            SkillResponse with matching housing units
        """
        bedrooms = request.entities.get("bedrooms")
        max_rent = request.entities.get("max_rent")
        voucher_type = request.entities.get("voucher_type")
        city = request.entities.get("city")
        accessibility = request.entities.get("accessibility", [])

        units = await self.search_housing(
            bedrooms=bedrooms,
            max_rent=max_rent,
            voucher_type=voucher_type,
            city=city,
            accessibility=accessibility,
        )

        if not units:
            return SkillResponse(
                content="No housing units found matching your criteria. "
                "Housing availability changes frequently - consider:\n"
                "- Expanding your search area\n"
                "- Adjusting bedroom requirements\n"
                "- Setting up alerts for new listings",
                success=True,
                data={"units": []},
                suggestions=[
                    "Set up housing alerts",
                    "Expand search criteria",
                    "Check voucher waitlist status",
                ],
            )

        content = f"**Found {len(units)} Housing Options**\n\n"
        for unit in units:
            content += f"**{unit.address}, {unit.city}**\n"
            content += f"- Bedrooms: {unit.bedrooms} | Bathrooms: {unit.bathrooms}\n"
            content += f"- Rent: ${unit.rent:.0f}/month\n"
            content += f"- Vouchers Accepted: {', '.join(v.value for v in unit.voucher_types)}\n"
            if unit.accessibility:
                content += f"- Accessibility: {', '.join(unit.accessibility)}\n"
            content += f"- Status: {unit.status.value.title()}\n"
            if unit.available_date:
                content += f"- Available: {unit.available_date.strftime('%B %d, %Y')}\n"
            content += "\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "units": [
                    {"id": u.unit_id, "address": u.address, "rent": u.rent}
                    for u in units
                ],
                "total_found": len(units),
            },
            suggestions=[
                "Get landlord info for a property",
                "Check if your voucher covers this rent",
                "Schedule a viewing",
            ],
        )

    async def search_housing(
        self,
        bedrooms: int | None = None,
        max_rent: float | None = None,
        voucher_type: str | None = None,
        city: str | None = None,
        accessibility: list[str] | None = None,
    ) -> list[HousingUnit]:
        """Search the housing database for matching units.

        Args:
            bedrooms: Minimum number of bedrooms
            max_rent: Maximum monthly rent
            voucher_type: Required voucher acceptance
            city: City to search in
            accessibility: Required accessibility features

        Returns:
            List of matching housing units
        """
        # In production, this would query housing_database and hud_api spans
        sample_units = [
            HousingUnit(
                unit_id="unit_001",
                address="456 Oak Street, Apt 2B",
                city="Oakland",
                state="CA",
                zip_code="94612",
                bedrooms=2,
                bathrooms=1.0,
                rent=1650.0,
                accepts_voucher=True,
                voucher_types=[VoucherType.HCV, VoucherType.EHV],
                landlord_id="landlord_001",
                amenities=["laundry", "parking"],
                accessibility=["ground_floor", "wide_doorways"],
                status=HousingStatus.AVAILABLE,
                available_date=datetime(2024, 3, 1, tzinfo=timezone.utc),
                utilities_included=["water", "trash"],
            ),
            HousingUnit(
                unit_id="unit_002",
                address="789 Pine Avenue, Unit 5",
                city="Oakland",
                state="CA",
                zip_code="94610",
                bedrooms=3,
                bathrooms=1.5,
                rent=2050.0,
                accepts_voucher=True,
                voucher_types=[VoucherType.HCV, VoucherType.PBV, VoucherType.FUP],
                landlord_id="landlord_002",
                amenities=["laundry", "backyard"],
                accessibility=[],
                status=HousingStatus.AVAILABLE,
                available_date=datetime(2024, 2, 15, tzinfo=timezone.utc),
            ),
        ]

        # Filter based on criteria
        results = []
        for unit in sample_units:
            if bedrooms and unit.bedrooms < bedrooms:
                continue
            if max_rent and unit.rent > max_rent:
                continue
            if city and unit.city.lower() != city.lower():
                continue
            if accessibility:
                if not all(a in unit.accessibility for a in accessibility):
                    continue
            results.append(unit)

        return results

    async def _handle_voucher_status(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Check housing voucher application status.

        Args:
            request: Request with application ID or user context
            context: Execution context

        Returns:
            SkillResponse with voucher status information
        """
        application_id = request.entities.get("application_id")
        voucher_type = request.entities.get("voucher_type")

        status = await self.check_voucher_status(application_id, context.user_id)

        if not status:
            return SkillResponse(
                content="No voucher application found. Would you like to:\n"
                "- Start a new voucher application\n"
                "- Learn about voucher programs\n"
                "- Check eligibility requirements",
                success=True,
                suggestions=[
                    "Start voucher application",
                    "Learn about Section 8",
                    "Check eligibility",
                ],
            )

        content = f"**Voucher Application Status**\n\n"
        content += f"Application ID: {status.application_id}\n"
        content += f"Voucher Type: {status.voucher_type.value}\n"
        content += f"Status: **{status.status}**\n"
        content += f"Applied: {status.applied_date.strftime('%B %d, %Y')}\n"

        if status.waitlist_position:
            content += f"Waitlist Position: #{status.waitlist_position}\n"
        if status.estimated_wait:
            content += f"Estimated Wait: {status.estimated_wait}\n"

        content += f"\nLast Updated: {status.last_updated.strftime('%B %d, %Y')}\n"

        if status.next_steps:
            content += "\n**Next Steps:**\n"
            for step in status.next_steps:
                content += f"- {step}\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "application_id": status.application_id,
                "status": status.status,
                "waitlist_position": status.waitlist_position,
            },
            suggestions=[
                "Update contact information",
                "Check eligibility for other programs",
                "Find emergency housing resources",
            ],
        )

    async def check_voucher_status(
        self, application_id: str | None, user_id: str
    ) -> VoucherApplication | None:
        """Check the status of a voucher application.

        Args:
            application_id: Specific application to check
            user_id: User ID to look up applications for

        Returns:
            VoucherApplication if found, None otherwise
        """
        # In production, this would query the voucher_tracker span
        return VoucherApplication(
            application_id=application_id or "APP-2024-001234",
            voucher_type=VoucherType.HCV,
            status="On Waitlist",
            applied_date=datetime(2023, 6, 15, tzinfo=timezone.utc),
            waitlist_position=234,
            estimated_wait="12-18 months",
            last_updated=datetime.now(timezone.utc),
            next_steps=[
                "Keep your contact information current",
                "Respond promptly to any mailings",
                "Report any income changes within 10 days",
            ],
        )

    async def _handle_tenant_rights(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Get tenant rights information.

        Args:
            request: Request with jurisdiction info
            context: Execution context

        Returns:
            SkillResponse with tenant rights information
        """
        state = request.entities.get("state", "CA")
        city = request.entities.get("city")
        issue = request.entities.get("issue")

        rights_info = await self.get_tenant_rights(state, city, issue)

        content = f"**Tenant Rights in {city or state}**\n\n"

        for category, rights in rights_info.items():
            content += f"**{category}:**\n"
            for right in rights:
                content += f"- {right}\n"
            content += "\n"

        content += (
            "\n*Note: Tenant protections vary by jurisdiction. "
            "Contact a tenant rights organization for specific advice.*"
        )

        return SkillResponse(
            content=content,
            success=True,
            data={"rights": rights_info, "jurisdiction": city or state},
            suggestions=[
                "Find tenant rights organizations",
                "Report a housing code violation",
                "Get eviction defense help",
            ],
        )

    async def get_tenant_rights(
        self, state: str, city: str | None, issue: str | None
    ) -> dict[str, list[str]]:
        """Look up tenant rights for a jurisdiction.

        Args:
            state: State for rights lookup
            city: City for local protections
            issue: Specific issue to focus on

        Returns:
            Dictionary of rights categories and specific rights
        """
        # In production, this would query housing_database for jurisdiction-specific info
        return {
            "Right to Habitable Housing": [
                "Working plumbing, heating, and electricity",
                "Freedom from pests and mold",
                "Adequate weatherproofing",
                "Functioning smoke and CO detectors",
            ],
            "Rent Control Protections": [
                "Rent increases limited to annual cap",
                "Just cause eviction required",
                "Right to petition for rent reduction",
            ],
            "Privacy Rights": [
                "24-48 hour notice before landlord entry",
                "Entry only during reasonable hours",
                "Emergency entry only for true emergencies",
            ],
            "Anti-Retaliation Protections": [
                "Cannot be evicted for reporting violations",
                "Cannot raise rent in retaliation",
                "Protected for organizing with other tenants",
            ],
        }

    async def _handle_landlord_lookup(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Look up landlord accountability information.

        Args:
            request: Request with landlord or property info
            context: Execution context

        Returns:
            SkillResponse with landlord information
        """
        landlord_name = request.entities.get("landlord_name")
        property_address = request.entities.get("property_address")
        landlord_id = request.entities.get("landlord_id")

        record = await self.lookup_landlord(landlord_name, property_address, landlord_id)

        if not record:
            return SkillResponse(
                content="No landlord information found. You can:\n"
                "- Search by property address\n"
                "- Search by landlord name\n"
                "- Report landlord issues to build the database",
                success=True,
                suggestions=[
                    "Search by address",
                    "Report an issue",
                    "Find tenant resources",
                ],
            )

        content = f"**Landlord Information**\n\n"
        content += f"Name: {record.name}\n"
        content += f"Properties: {record.properties_count}\n"
        content += f"Accepts Vouchers: {'Yes' if record.accepts_vouchers else 'No'}\n\n"

        content += f"**Accountability Metrics:**\n"
        content += f"- Community Rating: {'*' * int(record.community_rating)} ({record.community_rating:.1f}/5)\n"
        content += f"- Voucher Friendly: {'*' * int(record.voucher_friendly_rating)} ({record.voucher_friendly_rating:.1f}/5)\n"
        content += f"- Complaints: {record.complaint_count}\n"
        content += f"- Code Violations: {record.violation_count}\n"
        content += f"- Avg Response Time: {record.avg_response_time}\n"

        if record.notes:
            content += "\n**Community Notes:**\n"
            for note in record.notes[:3]:  # Show first 3 notes
                content += f"- {note}\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "landlord_id": record.landlord_id,
                "name": record.name,
                "rating": record.community_rating,
                "complaints": record.complaint_count,
            },
            suggestions=[
                "Report an issue with this landlord",
                "Find properties by this landlord",
                "Get tenant rights info",
            ],
        )

    async def lookup_landlord(
        self,
        name: str | None,
        address: str | None,
        landlord_id: str | None,
    ) -> LandlordRecord | None:
        """Look up landlord accountability information.

        Args:
            name: Landlord name to search
            address: Property address to find landlord
            landlord_id: Direct landlord ID lookup

        Returns:
            LandlordRecord if found, None otherwise
        """
        # In production, this would query the landlord_registry span
        return LandlordRecord(
            landlord_id="landlord_001",
            name="Bay Area Properties LLC",
            properties_count=23,
            accepts_vouchers=True,
            complaint_count=5,
            violation_count=2,
            avg_response_time="3-5 days",
            community_rating=3.5,
            voucher_friendly_rating=4.0,
            notes=[
                "Generally responsive to maintenance requests",
                "Has accepted Section 8 vouchers consistently",
                "Some tenants report slow response during holidays",
            ],
        )

    async def _handle_eviction_defense(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Get eviction defense resources and support.

        Args:
            request: Request with eviction details
            context: Execution context

        Returns:
            SkillResponse with eviction defense resources
        """
        eviction_stage = request.entities.get("stage", "notice")
        location = request.entities.get("location")
        urgency = request.entities.get("urgency", "normal")

        resources = await self.get_eviction_resources(eviction_stage, location)

        content = "**Eviction Defense Resources**\n\n"

        if eviction_stage == "notice":
            content += (
                "You've received an eviction notice. Here's what to know:\n\n"
                "**Immediate Steps:**\n"
                "1. DO NOT ignore the notice\n"
                "2. Note all deadlines carefully\n"
                "3. Contact a tenant rights organization immediately\n"
                "4. Document everything in writing\n\n"
            )
        elif eviction_stage == "court":
            content += (
                "You have a court date. Critical steps:\n\n"
                "**Immediate Steps:**\n"
                "1. DO NOT miss your court date\n"
                "2. Seek legal representation immediately\n"
                "3. Gather all documentation\n"
                "4. Apply for emergency rental assistance if eligible\n\n"
            )

        content += "**Emergency Resources:**\n"
        for resource in resources:
            content += f"- **{resource['name']}**: {resource['contact']}\n"
            if resource.get("hours"):
                content += f"  Hours: {resource['hours']}\n"

        return SkillResponse(
            content=content,
            success=True,
            data={"resources": resources, "stage": eviction_stage},
            suggestions=[
                "Find free legal help",
                "Apply for rental assistance",
                "Know your court rights",
            ],
        )

    async def get_eviction_resources(
        self, stage: str, location: str | None
    ) -> list[dict[str, Any]]:
        """Get eviction defense resources.

        Args:
            stage: Stage of eviction process
            location: Geographic location

        Returns:
            List of eviction defense resources
        """
        # In production, this would query the housing_database span
        return [
            {
                "name": "Eviction Defense Hotline",
                "contact": "1-800-EVICT-HELP",
                "hours": "24/7",
                "services": "Legal advice, court accompaniment",
            },
            {
                "name": "Emergency Rental Assistance",
                "contact": "Apply at rentrelief.org",
                "hours": "Online 24/7",
                "services": "Rent and utility payment assistance",
            },
            {
                "name": "Tenant Rights Legal Clinic",
                "contact": "tenantlaw@legalaid.org",
                "hours": "M-F 9am-5pm",
                "services": "Free legal representation",
            },
        ]

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for housing-related information.

        Args:
            beliefs: Current belief state
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Filtered BDI state relevant to housing matters
        """
        housing_domains = {"housing", "tenant", "voucher", "eviction", "landlord"}

        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in housing_domains
                or b.get("type") in [
                    "housing_status",
                    "voucher_status",
                    "eviction_risk",
                    "rent_status",
                ]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in [
                    "stable_housing",
                    "voucher_approval",
                    "eviction_defense",
                ]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") in housing_domains
            ],
        }
