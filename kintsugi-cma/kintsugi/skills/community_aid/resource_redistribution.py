"""
Resource Redistribution Skill Chip for Kintsugi CMA.

Coordinates surplus allocation, food rescue, and resource sharing across
community organizations and individuals. Specializes in time-sensitive
logistics for perishable goods.

This chip enables equitable resource distribution by:
- Tracking surplus resources from partners (food banks, businesses, etc.)
- Matching surplus with community needs
- Scheduling time-sensitive pickups for perishables
- Managing partner relationships and agreements

Example usage:
    from kintsugi.skills.community_aid import ResourceRedistributionChip
    from kintsugi.skills import SkillRequest, SkillContext, register_chip

    # Register the chip
    chip = ResourceRedistributionChip()
    register_chip(chip)

    # Report surplus
    request = SkillRequest(
        intent="surplus_report",
        entities={
            "resource_type": "prepared_food",
            "quantity": "50 meals",
            "expiry_hours": 4,
            "pickup_location": "Downtown Community Center"
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


class ResourceType(str, Enum):
    """Types of resources that can be redistributed."""
    PREPARED_FOOD = "prepared_food"
    FRESH_PRODUCE = "fresh_produce"
    SHELF_STABLE = "shelf_stable"
    CLOTHING = "clothing"
    HOUSEHOLD = "household"
    MEDICAL_SUPPLIES = "medical_supplies"
    HYGIENE = "hygiene"
    ELECTRONICS = "electronics"
    FURNITURE = "furniture"
    OTHER = "other"


class ResourceStatus(str, Enum):
    """Status of resources in the redistribution system."""
    AVAILABLE = "available"
    CLAIMED = "claimed"
    SCHEDULED = "scheduled"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class PartnerType(str, Enum):
    """Types of redistribution partners."""
    FOOD_BANK = "food_bank"
    GROCERY_STORE = "grocery_store"
    RESTAURANT = "restaurant"
    FARM = "farm"
    MANUFACTURER = "manufacturer"
    NONPROFIT = "nonprofit"
    INDIVIDUAL = "individual"
    GOVERNMENT = "government"


@dataclass
class SurplusResource:
    """Represents a surplus resource available for redistribution."""
    id: str
    partner_id: str
    resource_type: ResourceType
    description: str
    quantity: str
    unit: str
    pickup_location: str
    pickup_instructions: str
    created_at: datetime
    expiry_time: datetime | None  # Critical for perishables
    status: ResourceStatus
    claimed_by: str | None = None
    scheduled_pickup: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequest:
    """Represents a request for resources."""
    id: str
    requester_id: str
    requester_org: str
    resource_types: list[ResourceType]
    quantity_needed: str
    delivery_location: str
    urgency: str
    created_at: datetime
    status: ResourceStatus
    matched_surplus_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PickupSchedule:
    """Represents a scheduled pickup."""
    id: str
    surplus_id: str
    volunteer_id: str | None
    pickup_time: datetime
    delivery_location: str
    status: str
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Partner:
    """Represents a redistribution partner organization."""
    id: str
    name: str
    partner_type: PartnerType
    contact_email: str
    contact_phone: str
    address: str
    resource_types: list[ResourceType]
    capacity: dict[str, int]  # Resource type -> typical quantity
    operating_hours: str
    active: bool = True
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class ResourceRedistributionChip(BaseSkillChip):
    """Coordinate surplus allocation and time-sensitive resource sharing.

    This chip handles the full lifecycle of resource redistribution:
    1. Partners report surplus resources with expiry information
    2. System matches surplus with community requests
    3. Volunteers are dispatched for pickup and delivery
    4. Inventory is tracked across the network

    Time-sensitivity is critical - perishable goods need rapid matching
    and logistics coordination.
    """

    name = "resource_redistribution"
    description = "Coordinate surplus allocation, food rescue, and resource sharing"
    version = "1.0.0"
    domain = SkillDomain.MUTUAL_AID

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.30,
        resource_efficiency=0.25,
        transparency=0.10,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SCHEDULE_TASKS,
    ]

    consensus_actions = ["approve_large_redistribution", "partner_agreement"]

    required_spans = [
        "inventory_system",
        "logistics_api",
        "partner_network",
    ]

    # Simulated storage
    _surplus: dict[str, SurplusResource] = {}
    _requests: dict[str, ResourceRequest] = {}
    _schedules: dict[str, PickupSchedule] = {}
    _partners: dict[str, Partner] = {}
    _inventory: dict[str, dict[str, int]] = {}  # location -> {type: quantity}

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request containing intent and entities
            context: Execution context with org/user info and BDI state

        Returns:
            SkillResponse with operation result
        """
        intent_handlers = {
            "surplus_report": self._handle_surplus_report,
            "redistribution_request": self._handle_redistribution_request,
            "pickup_schedule": self._handle_pickup_schedule,
            "inventory_check": self._handle_inventory_check,
            "partner_connect": self._handle_partner_connect,
        }

        handler = intent_handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}",
                success=False,
                data={"error": "unknown_intent", "valid_intents": list(intent_handlers.keys())},
            )

        return await handler(request, context)

    async def _handle_surplus_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle reporting of surplus resources.

        Entities expected:
            resource_type: Type of resource
            description: Description of the surplus
            quantity: Amount available
            unit: Unit of measurement (items, pounds, etc.)
            pickup_location: Where to pick up
            pickup_instructions: How to access
            expiry_hours: Hours until expiry (for perishables)
        """
        surplus = await self.report_surplus(
            partner_id=context.user_id,
            resource_type=request.entities.get("resource_type", "other"),
            description=request.entities.get("description", ""),
            quantity=request.entities.get("quantity", "1"),
            unit=request.entities.get("unit", "items"),
            pickup_location=request.entities.get("pickup_location", ""),
            pickup_instructions=request.entities.get("pickup_instructions", ""),
            expiry_hours=request.entities.get("expiry_hours"),
        )

        # For perishables, immediately look for matches
        is_urgent = surplus.expiry_time is not None
        matches = []
        if is_urgent:
            matches = await self._find_urgent_matches(surplus.id)

        urgency_msg = ""
        if surplus.expiry_time:
            hours_left = (surplus.expiry_time - datetime.now(timezone.utc)).total_seconds() / 3600
            urgency_msg = f" URGENT: {hours_left:.1f} hours until expiry."

        return SkillResponse(
            content=f"Surplus reported successfully (ID: {surplus.id[:8]}...).{urgency_msg} "
                    f"Found {len(matches)} potential recipients.",
            success=True,
            data={
                "surplus_id": surplus.id,
                "resource_type": surplus.resource_type.value,
                "quantity": f"{surplus.quantity} {surplus.unit}",
                "expiry_time": surplus.expiry_time.isoformat() if surplus.expiry_time else None,
                "potential_matches": len(matches),
            },
            suggestions=[
                "Schedule pickup with 'schedule redistribution pickup'",
                "Check inventory with 'show current inventory'",
            ] if matches else [
                "We'll alert partners when a match is found",
            ],
        )

    async def _handle_redistribution_request(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle requests for resources.

        Entities expected:
            resource_types: List of resource types needed
            quantity_needed: Amount needed
            delivery_location: Where to deliver
            urgency: How urgent (high, medium, low)
            org_name: Requesting organization name
        """
        resource_request = await self.request_resources(
            requester_id=context.user_id,
            requester_org=request.entities.get("org_name", ""),
            resource_types=request.entities.get("resource_types", ["other"]),
            quantity_needed=request.entities.get("quantity_needed", ""),
            delivery_location=request.entities.get("delivery_location", ""),
            urgency=request.entities.get("urgency", "medium"),
        )

        # Look for available surplus
        available = await self._find_available_surplus(resource_request)

        return SkillResponse(
            content=f"Resource request created (ID: {resource_request.id[:8]}...). "
                    f"Found {len(available)} potential sources.",
            success=True,
            data={
                "request_id": resource_request.id,
                "resource_types": [rt.value for rt in resource_request.resource_types],
                "available_surplus": len(available),
                "surplus_details": [
                    {
                        "id": s.id,
                        "type": s.resource_type.value,
                        "quantity": f"{s.quantity} {s.unit}",
                        "location": s.pickup_location,
                    }
                    for s in available[:5]  # Top 5
                ],
            },
        )

    async def _handle_pickup_schedule(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle scheduling pickups for surplus redistribution.

        Entities expected:
            surplus_id: ID of surplus to pick up
            pickup_time: Requested pickup time
            delivery_location: Where to deliver
            volunteer_id: Volunteer assigned (optional)
        """
        schedule = await self.schedule_pickup(
            surplus_id=request.entities.get("surplus_id", ""),
            pickup_time=request.entities.get("pickup_time"),
            delivery_location=request.entities.get("delivery_location", ""),
            volunteer_id=request.entities.get("volunteer_id"),
        )

        if not schedule:
            return SkillResponse(
                content="Could not schedule pickup. Please check the surplus ID.",
                success=False,
            )

        return SkillResponse(
            content=f"Pickup scheduled for {schedule.pickup_time.strftime('%Y-%m-%d %H:%M')}. "
                    f"Schedule ID: {schedule.id[:8]}...",
            success=True,
            data={
                "schedule_id": schedule.id,
                "surplus_id": schedule.surplus_id,
                "pickup_time": schedule.pickup_time.isoformat(),
                "delivery_location": schedule.delivery_location,
                "volunteer_assigned": schedule.volunteer_id is not None,
            },
            suggestions=[
                "Assign a volunteer with 'assign volunteer to pickup'",
                "View all scheduled pickups with 'show pickup schedule'",
            ],
        )

    async def _handle_inventory_check(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Check current inventory across locations.

        Entities expected:
            location: Specific location to check (optional)
            resource_type: Filter by type (optional)
        """
        inventory = await self.check_inventory(
            location=request.entities.get("location"),
            resource_type=request.entities.get("resource_type"),
        )

        if not inventory:
            return SkillResponse(
                content="No inventory found matching your criteria.",
                success=True,
                data={"inventory": {}},
            )

        # Format inventory summary
        total_items = sum(sum(types.values()) for types in inventory.values())
        locations = len(inventory)

        summary_lines = [f"Inventory across {locations} location(s): {total_items} total items"]
        for location, types in inventory.items():
            summary_lines.append(f"\n{location}:")
            for rtype, qty in types.items():
                summary_lines.append(f"  - {rtype}: {qty}")

        return SkillResponse(
            content="\n".join(summary_lines),
            success=True,
            data={"inventory": inventory, "total_items": total_items, "locations": locations},
        )

    async def _handle_partner_connect(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Connect with redistribution partners.

        Entities expected:
            action: search, add, or list
            partner_type: Type of partner to search for
            partner_name: Name for new partner
            contact_email: Email for new partner
            resource_types: Resources they handle
        """
        action = request.entities.get("action", "list")

        if action == "search":
            partners = await self._search_partners(
                partner_type=request.entities.get("partner_type"),
                resource_types=request.entities.get("resource_types"),
            )
            return SkillResponse(
                content=f"Found {len(partners)} partner(s) matching your criteria.",
                success=True,
                data={"partners": [self._partner_to_dict(p) for p in partners]},
            )

        elif action == "add":
            partner = await self.connect_partners(
                name=request.entities.get("partner_name", ""),
                partner_type=request.entities.get("partner_type", "nonprofit"),
                contact_email=request.entities.get("contact_email", ""),
                contact_phone=request.entities.get("contact_phone", ""),
                address=request.entities.get("address", ""),
                resource_types=request.entities.get("resource_types", ["other"]),
            )

            # Large partnerships require consensus
            if request.entities.get("is_mou", False):
                return SkillResponse(
                    content=f"Partner {partner.name} registration requires approval.",
                    success=True,
                    requires_consensus=True,
                    consensus_action="partner_agreement",
                    data={"partner_id": partner.id, "pending_action": "sign_agreement"},
                )

            return SkillResponse(
                content=f"Partner {partner.name} connected successfully (ID: {partner.id[:8]}...).",
                success=True,
                data={"partner": self._partner_to_dict(partner)},
            )

        else:  # list
            partners = list(self._partners.values())
            return SkillResponse(
                content=f"You have {len(partners)} active partner(s).",
                success=True,
                data={"partners": [self._partner_to_dict(p) for p in partners]},
            )

    # Core business logic methods

    async def report_surplus(
        self,
        partner_id: str,
        resource_type: str,
        description: str,
        quantity: str,
        unit: str,
        pickup_location: str,
        pickup_instructions: str = "",
        expiry_hours: int | None = None,
    ) -> SurplusResource:
        """Report surplus resources available for redistribution.

        Args:
            partner_id: ID of the reporting partner
            resource_type: Type of resource
            description: Description of the surplus
            quantity: Amount available
            unit: Unit of measurement
            pickup_location: Where to pick up
            pickup_instructions: Access instructions
            expiry_hours: Hours until expiry (for perishables)

        Returns:
            The created SurplusResource object
        """
        now = datetime.now(timezone.utc)
        expiry_time = None
        if expiry_hours:
            expiry_time = now + timedelta(hours=expiry_hours)

        surplus = SurplusResource(
            id=str(uuid4()),
            partner_id=partner_id,
            resource_type=ResourceType(resource_type),
            description=description,
            quantity=quantity,
            unit=unit,
            pickup_location=pickup_location,
            pickup_instructions=pickup_instructions,
            created_at=now,
            expiry_time=expiry_time,
            status=ResourceStatus.AVAILABLE,
        )

        self._surplus[surplus.id] = surplus

        # Update inventory
        if pickup_location not in self._inventory:
            self._inventory[pickup_location] = {}
        rtype = surplus.resource_type.value
        self._inventory[pickup_location][rtype] = (
            self._inventory[pickup_location].get(rtype, 0) + int(surplus.quantity)
        )

        return surplus

    async def request_resources(
        self,
        requester_id: str,
        requester_org: str,
        resource_types: list[str],
        quantity_needed: str,
        delivery_location: str,
        urgency: str = "medium",
    ) -> ResourceRequest:
        """Request resources for redistribution.

        Args:
            requester_id: ID of the requesting user
            requester_org: Organization name
            resource_types: Types of resources needed
            quantity_needed: Amount needed
            delivery_location: Delivery address
            urgency: Request urgency

        Returns:
            The created ResourceRequest object
        """
        resource_request = ResourceRequest(
            id=str(uuid4()),
            requester_id=requester_id,
            requester_org=requester_org,
            resource_types=[ResourceType(rt) for rt in resource_types],
            quantity_needed=quantity_needed,
            delivery_location=delivery_location,
            urgency=urgency,
            created_at=datetime.now(timezone.utc),
            status=ResourceStatus.AVAILABLE,
        )

        self._requests[resource_request.id] = resource_request
        return resource_request

    async def schedule_pickup(
        self,
        surplus_id: str,
        pickup_time: str | datetime | None,
        delivery_location: str,
        volunteer_id: str | None = None,
    ) -> PickupSchedule | None:
        """Schedule a pickup for surplus redistribution.

        Args:
            surplus_id: ID of the surplus to pick up
            pickup_time: When to pick up
            delivery_location: Where to deliver
            volunteer_id: Assigned volunteer (optional)

        Returns:
            The created PickupSchedule or None if surplus not found
        """
        surplus = self._surplus.get(surplus_id)
        if not surplus:
            return None

        # Parse pickup time
        if isinstance(pickup_time, str):
            pickup_dt = datetime.fromisoformat(pickup_time)
        elif pickup_time is None:
            pickup_dt = datetime.now(timezone.utc) + timedelta(hours=1)
        else:
            pickup_dt = pickup_time

        schedule = PickupSchedule(
            id=str(uuid4()),
            surplus_id=surplus_id,
            volunteer_id=volunteer_id,
            pickup_time=pickup_dt,
            delivery_location=delivery_location,
            status="scheduled",
        )

        self._schedules[schedule.id] = schedule
        surplus.status = ResourceStatus.SCHEDULED
        surplus.scheduled_pickup = pickup_dt

        return schedule

    async def check_inventory(
        self,
        location: str | None = None,
        resource_type: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Check inventory across locations.

        Args:
            location: Filter by location (optional)
            resource_type: Filter by resource type (optional)

        Returns:
            Dictionary mapping locations to resource type quantities
        """
        result = {}

        for loc, types in self._inventory.items():
            if location and loc != location:
                continue

            filtered_types = {}
            for rtype, qty in types.items():
                if resource_type and rtype != resource_type:
                    continue
                if qty > 0:
                    filtered_types[rtype] = qty

            if filtered_types:
                result[loc] = filtered_types

        return result

    async def connect_partners(
        self,
        name: str,
        partner_type: str,
        contact_email: str,
        contact_phone: str = "",
        address: str = "",
        resource_types: list[str] | None = None,
    ) -> Partner:
        """Connect with a new redistribution partner.

        Args:
            name: Partner organization name
            partner_type: Type of partner
            contact_email: Primary contact email
            contact_phone: Contact phone
            address: Physical address
            resource_types: Types of resources they handle

        Returns:
            The created Partner object
        """
        partner = Partner(
            id=str(uuid4()),
            name=name,
            partner_type=PartnerType(partner_type),
            contact_email=contact_email,
            contact_phone=contact_phone,
            address=address,
            resource_types=[ResourceType(rt) for rt in (resource_types or ["other"])],
            capacity={},
            operating_hours="",
        )

        self._partners[partner.id] = partner
        return partner

    # Helper methods

    async def _find_urgent_matches(self, surplus_id: str) -> list[ResourceRequest]:
        """Find matches for urgent/perishable surplus."""
        surplus = self._surplus.get(surplus_id)
        if not surplus:
            return []

        matches = []
        for req in self._requests.values():
            if req.status != ResourceStatus.AVAILABLE:
                continue
            if surplus.resource_type in req.resource_types:
                matches.append(req)

        # Sort by urgency
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        matches.sort(key=lambda r: urgency_order.get(r.urgency, 2))

        return matches

    async def _find_available_surplus(
        self, resource_request: ResourceRequest
    ) -> list[SurplusResource]:
        """Find available surplus matching a request."""
        matching = []

        for surplus in self._surplus.values():
            if surplus.status != ResourceStatus.AVAILABLE:
                continue
            if surplus.resource_type in resource_request.resource_types:
                # Check expiry
                if surplus.expiry_time:
                    if surplus.expiry_time < datetime.now(timezone.utc):
                        surplus.status = ResourceStatus.EXPIRED
                        continue
                matching.append(surplus)

        # Sort by expiry (soonest first to prioritize perishables)
        matching.sort(key=lambda s: s.expiry_time or datetime.max.replace(tzinfo=timezone.utc))

        return matching

    async def _search_partners(
        self,
        partner_type: str | None = None,
        resource_types: list[str] | None = None,
    ) -> list[Partner]:
        """Search for partners by criteria."""
        results = []

        for partner in self._partners.values():
            if not partner.active:
                continue

            if partner_type and partner.partner_type.value != partner_type:
                continue

            if resource_types:
                partner_types = {rt.value for rt in partner.resource_types}
                if not any(rt in partner_types for rt in resource_types):
                    continue

            results.append(partner)

        return results

    def _partner_to_dict(self, partner: Partner) -> dict[str, Any]:
        """Convert partner to dictionary."""
        return {
            "id": partner.id,
            "name": partner.name,
            "type": partner.partner_type.value,
            "contact_email": partner.contact_email,
            "address": partner.address,
            "resource_types": [rt.value for rt in partner.resource_types],
            "active": partner.active,
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for resource redistribution domain.

        Returns beliefs about resource availability, desires for equitable
        distribution, and intentions related to logistics.
        """
        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in ["resources", "mutual_aid", "logistics"]
                or b.get("type") in ["inventory_level", "partner_capacity", "perishable_status"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["resource_equity", "waste_reduction", "community_fed"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") == "redistribution"
                or i.get("action") in ["schedule_pickup", "allocate_surplus", "deliver_resources"]
            ],
        }
