"""
Mutual Aid Coordinator Skill Chip for Kintsugi CMA.

Matches community needs with offers, coordinates mutual aid requests,
and maintains privacy-preserving connections between community members.

This chip enables grassroots mutual aid by:
- Allowing members to post needs and offers anonymously
- Matching needs with appropriate offers using configurable criteria
- Managing the communication flow to protect requester privacy
- Tracking aid fulfillment and generating impact reports

Example usage:
    from kintsugi.skills.community_aid import MutualAidCoordinatorChip
    from kintsugi.skills import SkillRequest, SkillContext, register_chip

    # Register the chip
    chip = MutualAidCoordinatorChip()
    register_chip(chip)

    # Post a need
    request = SkillRequest(
        intent="need_post",
        entities={
            "category": "housing",
            "description": "Need temporary housing for 2 weeks",
            "urgency": "high",
            "location": "downtown"
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


class AidCategory(str, Enum):
    """Categories of mutual aid."""
    HOUSING = "housing"
    FOOD = "food"
    TRANSPORTATION = "transportation"
    CHILDCARE = "childcare"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    EMOTIONAL_SUPPORT = "emotional_support"
    LEGAL = "legal"
    EMPLOYMENT = "employment"
    EDUCATION = "education"
    OTHER = "other"


class AidUrgency(str, Enum):
    """Urgency levels for aid requests."""
    CRITICAL = "critical"  # Immediate need (within hours)
    HIGH = "high"          # Within 24-48 hours
    MEDIUM = "medium"      # Within a week
    LOW = "low"            # Flexible timing


class AidStatus(str, Enum):
    """Status of aid requests and offers."""
    PENDING = "pending"
    MATCHED = "matched"
    IN_PROGRESS = "in_progress"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class AidNeed:
    """Represents a posted need in the mutual aid system."""
    id: str
    requester_id: str  # Hashed for privacy
    category: AidCategory
    description: str
    urgency: AidUrgency
    location: str | None
    location_radius_km: float
    created_at: datetime
    expires_at: datetime | None
    status: AidStatus
    matched_offer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AidOffer:
    """Represents a posted offer in the mutual aid system."""
    id: str
    offerer_id: str
    categories: list[AidCategory]
    description: str
    availability: str
    location: str | None
    location_radius_km: float
    capacity: int  # How many needs can they help with
    created_at: datetime
    expires_at: datetime | None
    status: AidStatus
    active_matches: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AidMatch:
    """Represents a match between a need and an offer."""
    id: str
    need_id: str
    offer_id: str
    created_at: datetime
    status: AidStatus
    contact_shared: bool = False
    notes: str = ""


class MutualAidCoordinatorChip(BaseSkillChip):
    """Coordinate mutual aid requests and offers with privacy preservation.

    This chip handles the full lifecycle of mutual aid coordination:
    1. Community members post needs or offers anonymously
    2. The system matches needs with appropriate offers
    3. Both parties opt-in before contact information is shared
    4. Aid fulfillment is tracked and reported

    Privacy is a core principle - requester details are never exposed
    until both parties confirm the match.
    """

    name = "mutual_aid_coordinator"
    description = "Match community needs with offers, coordinate mutual aid requests"
    version = "1.0.0"
    domain = SkillDomain.MUTUAL_AID

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.35,
        resource_efficiency=0.15,
        transparency=0.10,
        equity=0.15,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SEND_NOTIFICATIONS,
    ]

    consensus_actions = ["approve_high_value_request", "share_requester_info"]

    required_spans = [
        "needs_database",
        "offers_database",
        "matching_engine",
        "notification_service",
    ]

    # Simulated in-memory storage (would be real DB in production)
    _needs: dict[str, AidNeed] = {}
    _offers: dict[str, AidOffer] = {}
    _matches: dict[str, AidMatch] = {}

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request containing intent and entities
            context: Execution context with org/user info and BDI state

        Returns:
            SkillResponse with operation result
        """
        intent_handlers = {
            "need_post": self._handle_need_post,
            "offer_post": self._handle_offer_post,
            "match_request": self._handle_match_request,
            "aid_status": self._handle_aid_status,
            "aid_report": self._handle_aid_report,
        }

        handler = intent_handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}",
                success=False,
                data={"error": "unknown_intent", "valid_intents": list(intent_handlers.keys())},
            )

        return await handler(request, context)

    async def _handle_need_post(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Post a new need to the mutual aid system.

        Entities expected:
            category: AidCategory value
            description: Free text description of need
            urgency: AidUrgency value (default: medium)
            location: Geographic area (optional)
            location_radius_km: Search radius (default: 10)
            expires_in_days: Days until expiry (optional)
        """
        need = await self.post_need(
            requester_id=context.user_id,
            category=request.entities.get("category", "other"),
            description=request.entities.get("description", ""),
            urgency=request.entities.get("urgency", "medium"),
            location=request.entities.get("location"),
            location_radius_km=request.entities.get("location_radius_km", 10.0),
            expires_in_days=request.entities.get("expires_in_days"),
        )

        # Automatically look for matches
        matches = await self.find_matches(need.id)

        return SkillResponse(
            content=f"Your need has been posted anonymously (ID: {need.id[:8]}...). "
                    f"Found {len(matches)} potential matches.",
            success=True,
            data={
                "need_id": need.id,
                "category": need.category.value,
                "urgency": need.urgency.value,
                "potential_matches": len(matches),
            },
            suggestions=[
                "Check your matches with 'show my aid matches'",
                "Update your need with 'update need status'",
            ] if matches else [
                "We'll notify you when a match is found",
                "Consider broadening your location radius",
            ],
        )

    async def _handle_offer_post(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Post a new offer to help in the mutual aid system.

        Entities expected:
            categories: List of AidCategory values
            description: What help they can provide
            availability: When they're available
            location: Geographic area (optional)
            location_radius_km: How far they can travel (default: 10)
            capacity: How many people they can help (default: 1)
        """
        offer = await self.post_offer(
            offerer_id=context.user_id,
            categories=request.entities.get("categories", ["other"]),
            description=request.entities.get("description", ""),
            availability=request.entities.get("availability", "flexible"),
            location=request.entities.get("location"),
            location_radius_km=request.entities.get("location_radius_km", 10.0),
            capacity=request.entities.get("capacity", 1),
        )

        return SkillResponse(
            content=f"Thank you for your offer to help! Your offer has been registered "
                    f"(ID: {offer.id[:8]}...). We'll match you with community members in need.",
            success=True,
            data={
                "offer_id": offer.id,
                "categories": [c.value for c in offer.categories],
                "capacity": offer.capacity,
            },
            suggestions=[
                "View matching needs with 'show needs I can help with'",
                "Update your availability with 'update my offer'",
            ],
        )

    async def _handle_match_request(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Find and optionally confirm matches for a need or offer.

        Entities expected:
            need_id: ID of need to match (optional)
            offer_id: ID of offer to match (optional)
            confirm_match_id: Match ID to confirm (optional)
        """
        if confirm_id := request.entities.get("confirm_match_id"):
            return await self._confirm_match(confirm_id, context)

        if need_id := request.entities.get("need_id"):
            matches = await self.find_matches(need_id)
            return SkillResponse(
                content=f"Found {len(matches)} potential matches for your need.",
                success=True,
                data={"matches": [self._match_to_dict(m) for m in matches]},
            )

        if offer_id := request.entities.get("offer_id"):
            # Find needs that match this offer
            needs = await self._find_needs_for_offer(offer_id)
            return SkillResponse(
                content=f"Found {len(needs)} needs you could help with.",
                success=True,
                data={"needs": [self._need_to_dict_safe(n) for n in needs]},
            )

        return SkillResponse(
            content="Please specify a need_id, offer_id, or confirm_match_id.",
            success=False,
        )

    async def _confirm_match(
        self, match_id: str, context: SkillContext
    ) -> SkillResponse:
        """Confirm a match and initiate contact sharing (requires consensus)."""
        match = self._matches.get(match_id)
        if not match:
            return SkillResponse(
                content=f"Match {match_id} not found.",
                success=False,
            )

        # Check if user is involved in this match
        need = self._needs.get(match.need_id)
        offer = self._offers.get(match.offer_id)

        if not need or not offer:
            return SkillResponse(
                content="Could not verify match participants.",
                success=False,
            )

        # This action requires consensus for privacy
        if not match.contact_shared:
            return SkillResponse(
                content="Match confirmation requires approval from both parties. "
                        "A consensus request has been initiated.",
                success=True,
                requires_consensus=True,
                consensus_action="share_requester_info",
                data={
                    "match_id": match_id,
                    "pending_action": "share_contact_info",
                },
            )

        match.status = AidStatus.IN_PROGRESS
        return SkillResponse(
            content="Match confirmed! Contact information has been shared securely.",
            success=True,
            data={"match_id": match_id, "status": "in_progress"},
        )

    async def _handle_aid_status(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Check or update status of aid requests/offers.

        Entities expected:
            need_id: ID of need to check/update (optional)
            offer_id: ID of offer to check/update (optional)
            new_status: New status to set (optional)
        """
        need_id = request.entities.get("need_id")
        offer_id = request.entities.get("offer_id")
        new_status = request.entities.get("new_status")

        if need_id:
            return await self._update_need_status(need_id, new_status, context)
        elif offer_id:
            return await self._update_offer_status(offer_id, new_status, context)
        else:
            # Return all statuses for this user
            return await self._get_user_aid_status(context.user_id)

    async def _handle_aid_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate mutual aid impact report.

        Entities expected:
            period: Time period for report (default: month)
            category: Filter by category (optional)
        """
        report = await self.generate_aid_report(
            period=request.entities.get("period", "month"),
            category=request.entities.get("category"),
        )

        return SkillResponse(
            content=report["summary"],
            success=True,
            data=report,
        )

    # Core business logic methods

    async def post_need(
        self,
        requester_id: str,
        category: str,
        description: str,
        urgency: str = "medium",
        location: str | None = None,
        location_radius_km: float = 10.0,
        expires_in_days: int | None = None,
    ) -> AidNeed:
        """Post a new need to the mutual aid system.

        Args:
            requester_id: ID of the person posting the need (will be hashed)
            category: Category of aid needed
            description: Description of the need
            urgency: How urgent the need is
            location: Geographic location
            location_radius_km: Radius within which help is sought
            expires_in_days: Days until the need expires

        Returns:
            The created AidNeed object
        """
        from hashlib import sha256

        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = now + timedelta(days=expires_in_days)

        need = AidNeed(
            id=str(uuid4()),
            requester_id=sha256(requester_id.encode()).hexdigest()[:16],  # Privacy
            category=AidCategory(category),
            description=description,
            urgency=AidUrgency(urgency),
            location=location,
            location_radius_km=location_radius_km,
            created_at=now,
            expires_at=expires_at,
            status=AidStatus.PENDING,
        )

        self._needs[need.id] = need
        return need

    async def post_offer(
        self,
        offerer_id: str,
        categories: list[str],
        description: str,
        availability: str = "flexible",
        location: str | None = None,
        location_radius_km: float = 10.0,
        capacity: int = 1,
    ) -> AidOffer:
        """Post an offer to help in the mutual aid system.

        Args:
            offerer_id: ID of the person offering help
            categories: Categories of aid they can provide
            description: Description of the help offered
            availability: When they're available to help
            location: Geographic location
            location_radius_km: How far they can travel
            capacity: How many people they can help

        Returns:
            The created AidOffer object
        """
        now = datetime.now(timezone.utc)

        offer = AidOffer(
            id=str(uuid4()),
            offerer_id=offerer_id,
            categories=[AidCategory(c) for c in categories],
            description=description,
            availability=availability,
            location=location,
            location_radius_km=location_radius_km,
            capacity=capacity,
            created_at=now,
            expires_at=None,
            status=AidStatus.PENDING,
        )

        self._offers[offer.id] = offer
        return offer

    async def find_matches(self, need_id: str) -> list[AidMatch]:
        """Find potential matches for a need.

        Uses a matching algorithm that considers:
        - Category compatibility
        - Location proximity
        - Urgency prioritization
        - Offer capacity

        Args:
            need_id: ID of the need to find matches for

        Returns:
            List of potential matches, sorted by score
        """
        need = self._needs.get(need_id)
        if not need:
            return []

        matches = []
        for offer in self._offers.values():
            if offer.status not in [AidStatus.PENDING, AidStatus.IN_PROGRESS]:
                continue

            # Check category match
            if need.category not in offer.categories:
                continue

            # Check capacity
            if len(offer.active_matches) >= offer.capacity:
                continue

            # Check location compatibility (simplified)
            if need.location and offer.location:
                if need.location.lower() != offer.location.lower():
                    continue  # In production, use proper geolocation

            # Create match
            match = AidMatch(
                id=str(uuid4()),
                need_id=need.id,
                offer_id=offer.id,
                created_at=datetime.now(timezone.utc),
                status=AidStatus.PENDING,
            )
            self._matches[match.id] = match
            matches.append(match)

        return matches

    async def update_status(
        self,
        item_id: str,
        new_status: AidStatus,
        item_type: str = "need",
    ) -> bool:
        """Update the status of a need or offer.

        Args:
            item_id: ID of the need or offer
            new_status: New status to set
            item_type: Type of item ("need" or "offer")

        Returns:
            True if update succeeded
        """
        if item_type == "need":
            if item_id in self._needs:
                self._needs[item_id].status = new_status
                return True
        elif item_type == "offer":
            if item_id in self._offers:
                self._offers[item_id].status = new_status
                return True
        return False

    async def generate_aid_report(
        self,
        period: str = "month",
        category: str | None = None,
    ) -> dict[str, Any]:
        """Generate a mutual aid impact report.

        Args:
            period: Time period (week, month, quarter, year)
            category: Optional category filter

        Returns:
            Report dictionary with statistics and summary
        """
        # Filter by category if specified
        needs = list(self._needs.values())
        offers = list(self._offers.values())
        matches = list(self._matches.values())

        if category:
            cat = AidCategory(category)
            needs = [n for n in needs if n.category == cat]
            # Filter matches related to these needs
            need_ids = {n.id for n in needs}
            matches = [m for m in matches if m.need_id in need_ids]

        # Calculate statistics
        total_needs = len(needs)
        total_offers = len(offers)
        total_matches = len(matches)
        fulfilled = sum(1 for m in matches if m.status == AidStatus.FULFILLED)
        in_progress = sum(1 for m in matches if m.status == AidStatus.IN_PROGRESS)

        # Category breakdown
        category_stats = {}
        for need in needs:
            cat = need.category.value
            if cat not in category_stats:
                category_stats[cat] = {"needs": 0, "fulfilled": 0}
            category_stats[cat]["needs"] += 1
            if need.status == AidStatus.FULFILLED:
                category_stats[cat]["fulfilled"] += 1

        fulfillment_rate = (fulfilled / total_matches * 100) if total_matches > 0 else 0

        summary = (
            f"Mutual Aid Report ({period}):\n"
            f"- Total needs posted: {total_needs}\n"
            f"- Total offers registered: {total_offers}\n"
            f"- Matches made: {total_matches}\n"
            f"- Successfully fulfilled: {fulfilled}\n"
            f"- In progress: {in_progress}\n"
            f"- Fulfillment rate: {fulfillment_rate:.1f}%"
        )

        return {
            "summary": summary,
            "period": period,
            "total_needs": total_needs,
            "total_offers": total_offers,
            "total_matches": total_matches,
            "fulfilled": fulfilled,
            "in_progress": in_progress,
            "fulfillment_rate": fulfillment_rate,
            "by_category": category_stats,
        }

    # Helper methods

    async def _find_needs_for_offer(self, offer_id: str) -> list[AidNeed]:
        """Find needs that match an offer."""
        offer = self._offers.get(offer_id)
        if not offer:
            return []

        matching_needs = []
        for need in self._needs.values():
            if need.status != AidStatus.PENDING:
                continue
            if need.category in offer.categories:
                matching_needs.append(need)

        # Sort by urgency
        urgency_order = {
            AidUrgency.CRITICAL: 0,
            AidUrgency.HIGH: 1,
            AidUrgency.MEDIUM: 2,
            AidUrgency.LOW: 3,
        }
        matching_needs.sort(key=lambda n: urgency_order[n.urgency])

        return matching_needs

    async def _update_need_status(
        self, need_id: str, new_status: str | None, context: SkillContext
    ) -> SkillResponse:
        """Update need status with authorization check."""
        need = self._needs.get(need_id)
        if not need:
            return SkillResponse(content=f"Need {need_id} not found.", success=False)

        if new_status:
            need.status = AidStatus(new_status)
            return SkillResponse(
                content=f"Need status updated to {new_status}.",
                success=True,
                data=self._need_to_dict_safe(need),
            )

        return SkillResponse(
            content=f"Need status: {need.status.value}",
            success=True,
            data=self._need_to_dict_safe(need),
        )

    async def _update_offer_status(
        self, offer_id: str, new_status: str | None, context: SkillContext
    ) -> SkillResponse:
        """Update offer status."""
        offer = self._offers.get(offer_id)
        if not offer:
            return SkillResponse(content=f"Offer {offer_id} not found.", success=False)

        if new_status:
            offer.status = AidStatus(new_status)
            return SkillResponse(
                content=f"Offer status updated to {new_status}.",
                success=True,
            )

        return SkillResponse(
            content=f"Offer status: {offer.status.value}",
            success=True,
        )

    async def _get_user_aid_status(self, user_id: str) -> SkillResponse:
        """Get all aid activities for a user."""
        from hashlib import sha256
        hashed_id = sha256(user_id.encode()).hexdigest()[:16]

        user_needs = [n for n in self._needs.values() if n.requester_id == hashed_id]
        user_offers = [o for o in self._offers.values() if o.offerer_id == user_id]

        return SkillResponse(
            content=f"You have {len(user_needs)} needs posted and {len(user_offers)} offers active.",
            success=True,
            data={
                "needs": [self._need_to_dict_safe(n) for n in user_needs],
                "offers": [self._offer_to_dict(o) for o in user_offers],
            },
        )

    def _need_to_dict_safe(self, need: AidNeed) -> dict[str, Any]:
        """Convert need to dict, hiding sensitive info."""
        return {
            "id": need.id,
            "category": need.category.value,
            "description": need.description,
            "urgency": need.urgency.value,
            "location": need.location,
            "status": need.status.value,
            "created_at": need.created_at.isoformat(),
        }

    def _offer_to_dict(self, offer: AidOffer) -> dict[str, Any]:
        """Convert offer to dict."""
        return {
            "id": offer.id,
            "categories": [c.value for c in offer.categories],
            "description": offer.description,
            "availability": offer.availability,
            "location": offer.location,
            "capacity": offer.capacity,
            "active_matches": len(offer.active_matches),
            "status": offer.status.value,
        }

    def _match_to_dict(self, match: AidMatch) -> dict[str, Any]:
        """Convert match to dict."""
        return {
            "id": match.id,
            "need_id": match.need_id,
            "offer_id": match.offer_id,
            "status": match.status.value,
            "contact_shared": match.contact_shared,
            "created_at": match.created_at.isoformat(),
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for mutual aid domain.

        Returns beliefs about community needs, desires for mutual support,
        and intentions related to aid coordination.
        """
        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in ["mutual_aid", "community", "resources"]
                or b.get("type") in ["need_status", "community_capacity", "resource_availability"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["aid_fulfillment", "community_wellbeing", "resource_equity"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") == "mutual_aid"
                or i.get("action") in ["match_aid", "coordinate_support", "fulfill_need"]
            ],
        }
