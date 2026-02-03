"""
Food Access skill chip for Kintsugi CMA.

Coordinates food pantries, SNAP (food stamps) assistance, and community
meals for community members experiencing food insecurity. Provides
real-time pantry inventory and dietary restriction filtering.

Intents handled:
- pantry_find: Find nearby food pantries with availability
- snap_help: SNAP/food stamp eligibility and application help
- meal_schedule: Schedule community meal pickups or deliveries
- food_donate: Process food donations from community members
- nutrition_info: Provide nutrition information and dietary guidance

Example:
    chip = FoodAccessChip()
    response = await chip.handle(
        SkillRequest(intent="pantry_find", entities={"zip_code": "94612", "dietary": ["halal"]}),
        context
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
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


class DietaryRestriction(str, Enum):
    """Dietary restriction categories for filtering food sources."""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    HALAL = "halal"
    KOSHER = "kosher"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    LOW_SODIUM = "low_sodium"
    DIABETIC_FRIENDLY = "diabetic_friendly"


class PantryStatus(str, Enum):
    """Operating status of food pantries."""
    OPEN = "open"
    CLOSED = "closed"
    LOW_INVENTORY = "low_inventory"
    BY_APPOINTMENT = "by_appointment"
    SPECIAL_HOURS = "special_hours"


@dataclass
class FoodPantry:
    """Information about a food pantry location."""
    pantry_id: str
    name: str
    address: str
    city: str
    zip_code: str
    phone: str
    status: PantryStatus
    hours: dict[str, str]  # Day -> hours string
    dietary_options: list[DietaryRestriction]
    inventory_level: str  # high, medium, low
    next_restock: datetime | None
    requirements: list[str]  # ID, proof of address, etc.
    languages: list[str]
    distance_miles: float | None = None


@dataclass
class InventoryItem:
    """Food inventory item at a pantry."""
    item_id: str
    name: str
    category: str  # produce, protein, dairy, grains, etc.
    quantity: int
    unit: str
    dietary_tags: list[DietaryRestriction]
    expiration: datetime | None
    available: bool = True


@dataclass
class MealProgram:
    """Community meal program information."""
    program_id: str
    name: str
    location: str
    meal_type: str  # breakfast, lunch, dinner
    schedule: dict[str, str]  # Day -> time
    dietary_accommodations: list[DietaryRestriction]
    registration_required: bool
    delivery_available: bool
    capacity: int
    current_signups: int


class FoodAccessChip(BaseSkillChip):
    """Coordinate food pantries, SNAP assistance, and community meals.

    This chip helps community members access food resources by finding
    nearby pantries with real-time inventory, assisting with SNAP
    applications, and coordinating community meal programs.

    Attributes:
        name: Unique chip identifier
        description: Human-readable description
        domain: MUTUAL_AID domain for direct assistance
        efe_weights: High stakeholder benefit focus
        capabilities: READ_DATA, WRITE_DATA, SCHEDULE_TASKS, EXTERNAL_API
        consensus_actions: Actions requiring approval
        required_spans: MCP tool spans needed
    """

    name = "food_access"
    description = "Coordinate food pantries, SNAP assistance, and community meals"
    domain = SkillDomain.MUTUAL_AID
    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.40,
        resource_efficiency=0.15,
        transparency=0.10,
        equity=0.10,
    )
    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.SCHEDULE_TASKS,
        SkillCapability.EXTERNAL_API,
    ]
    consensus_actions = ["approve_food_distribution", "partner_food_share"]
    required_spans = ["pantry_network", "snap_eligibility", "meal_scheduler", "inventory_system"]

    # SNAP income limits by household size (monthly, 2024 federal guidelines)
    SNAP_INCOME_LIMITS: dict[int, int] = {
        1: 1580,
        2: 2137,
        3: 2694,
        4: 3250,
        5: 3807,
        6: 4364,
        7: 4921,
        8: 5478,
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with food access information
        """
        handlers = {
            "pantry_find": self._handle_pantry_find,
            "snap_help": self._handle_snap_help,
            "meal_schedule": self._handle_meal_schedule,
            "food_donate": self._handle_food_donate,
            "nutrition_info": self._handle_nutrition_info,
        }

        handler = handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}. Supported intents: {list(handlers.keys())}",
                success=False,
            )

        return await handler(request, context)

    async def _handle_pantry_find(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Find nearby food pantries with availability.

        Args:
            request: Request with location and dietary requirements
            context: Execution context

        Returns:
            SkillResponse with matching pantries
        """
        zip_code = request.entities.get("zip_code")
        city = request.entities.get("city")
        dietary = request.entities.get("dietary", [])
        max_distance = request.entities.get("max_distance", 5.0)

        pantries = await self.find_pantries(
            zip_code=zip_code,
            city=city,
            dietary_needs=dietary,
            max_distance=max_distance,
        )

        if not pantries:
            return SkillResponse(
                content="No food pantries found matching your criteria. "
                "Try expanding your search distance or checking back later.\n\n"
                "**Alternative Resources:**\n"
                "- National Hunger Hotline: 1-866-3-HUNGRY\n"
                "- SNAP Information: 1-800-221-5689",
                success=True,
                data={"pantries": []},
                suggestions=[
                    "Expand search distance",
                    "Check SNAP eligibility",
                    "Find community meal programs",
                ],
            )

        content = f"**Found {len(pantries)} Food Pantries**\n\n"
        for pantry in pantries:
            status_icon = "Open" if pantry.status == PantryStatus.OPEN else pantry.status.value.title()
            content += f"**{pantry.name}** ({status_icon})\n"
            content += f"- Address: {pantry.address}, {pantry.city}\n"
            content += f"- Phone: {pantry.phone}\n"
            if pantry.distance_miles:
                content += f"- Distance: {pantry.distance_miles:.1f} miles\n"
            content += f"- Inventory: {pantry.inventory_level.title()}\n"
            if pantry.dietary_options:
                content += f"- Dietary Options: {', '.join(d.value for d in pantry.dietary_options)}\n"
            if pantry.requirements:
                content += f"- Requirements: {', '.join(pantry.requirements)}\n"
            content += "\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "pantries": [
                    {
                        "id": p.pantry_id,
                        "name": p.name,
                        "status": p.status.value,
                        "distance": p.distance_miles,
                    }
                    for p in pantries
                ],
                "total_found": len(pantries),
            },
            suggestions=[
                "Get directions to a pantry",
                "Check pantry inventory",
                "Schedule a pickup time",
            ],
        )

    async def find_pantries(
        self,
        zip_code: str | None = None,
        city: str | None = None,
        dietary_needs: list[str] | None = None,
        max_distance: float = 5.0,
    ) -> list[FoodPantry]:
        """Search for food pantries matching criteria.

        Args:
            zip_code: ZIP code to search near
            city: City to search in
            dietary_needs: Required dietary accommodations
            max_distance: Maximum distance in miles

        Returns:
            List of matching food pantries
        """
        # In production, this would query the pantry_network span
        sample_pantries = [
            FoodPantry(
                pantry_id="pantry_001",
                name="Community Food Bank",
                address="123 Main Street",
                city="Oakland",
                zip_code="94612",
                phone="(510) 555-1234",
                status=PantryStatus.OPEN,
                hours={
                    "Monday": "9am-5pm",
                    "Tuesday": "9am-5pm",
                    "Wednesday": "9am-7pm",
                    "Thursday": "9am-5pm",
                    "Friday": "9am-3pm",
                    "Saturday": "10am-2pm",
                },
                dietary_options=[
                    DietaryRestriction.VEGETARIAN,
                    DietaryRestriction.HALAL,
                    DietaryRestriction.DIABETIC_FRIENDLY,
                ],
                inventory_level="high",
                next_restock=datetime(2024, 2, 15, 8, 0, tzinfo=timezone.utc),
                requirements=["Photo ID", "Proof of address"],
                languages=["English", "Spanish", "Vietnamese"],
                distance_miles=1.2,
            ),
            FoodPantry(
                pantry_id="pantry_002",
                name="Faith Community Pantry",
                address="456 Church Lane",
                city="Oakland",
                zip_code="94610",
                phone="(510) 555-5678",
                status=PantryStatus.OPEN,
                hours={
                    "Tuesday": "10am-2pm",
                    "Thursday": "10am-2pm",
                    "Saturday": "9am-12pm",
                },
                dietary_options=[
                    DietaryRestriction.VEGETARIAN,
                    DietaryRestriction.KOSHER,
                ],
                inventory_level="medium",
                next_restock=None,
                requirements=["Self-declaration of need"],
                languages=["English", "Spanish"],
                distance_miles=2.5,
            ),
        ]

        # Filter by distance
        results = [p for p in sample_pantries if (p.distance_miles or 0) <= max_distance]

        # Filter by dietary needs
        if dietary_needs:
            needed = set(dietary_needs)
            results = [
                p for p in results
                if needed.issubset({d.value for d in p.dietary_options})
            ]

        return results

    async def _handle_snap_help(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Help with SNAP eligibility and applications.

        Args:
            request: Request with household info
            context: Execution context

        Returns:
            SkillResponse with SNAP information
        """
        action = request.entities.get("action", "eligibility")
        household_size = request.entities.get("household_size")
        monthly_income = request.entities.get("monthly_income")

        if action == "eligibility" and household_size and monthly_income:
            eligible, details = await self.check_snap_eligibility(
                household_size, monthly_income
            )
            content = self._format_eligibility_result(eligible, details, household_size, monthly_income)
        else:
            content = self._get_snap_overview()

        return SkillResponse(
            content=content,
            success=True,
            data={
                "action": action,
                "household_size": household_size,
                "income": monthly_income,
            },
            suggestions=[
                "Start SNAP application",
                "Find SNAP application assistance",
                "Check other benefit programs",
            ],
        )

    async def check_snap_eligibility(
        self, household_size: int, monthly_income: float
    ) -> tuple[bool, dict[str, Any]]:
        """Check SNAP eligibility based on household size and income.

        Args:
            household_size: Number of people in household
            monthly_income: Gross monthly income

        Returns:
            Tuple of (is_eligible, eligibility_details)
        """
        # Get income limit (handle large households)
        if household_size <= 8:
            income_limit = self.SNAP_INCOME_LIMITS[household_size]
        else:
            # Add $557 per additional person beyond 8
            income_limit = self.SNAP_INCOME_LIMITS[8] + (557 * (household_size - 8))

        is_eligible = monthly_income <= income_limit

        # Estimate monthly benefit (simplified calculation)
        max_benefit = 234 * household_size  # Approximate max benefit
        estimated_benefit = max(0, max_benefit - (monthly_income * 0.3)) if is_eligible else 0

        return is_eligible, {
            "income_limit": income_limit,
            "estimated_monthly_benefit": round(estimated_benefit),
            "household_size": household_size,
            "gross_income": monthly_income,
        }

    def _format_eligibility_result(
        self, eligible: bool, details: dict, household_size: int, income: float
    ) -> str:
        """Format SNAP eligibility check results."""
        content = "**SNAP Eligibility Check**\n\n"
        content += f"Household Size: {household_size}\n"
        content += f"Monthly Income: ${income:,.0f}\n"
        content += f"Income Limit: ${details['income_limit']:,}\n\n"

        if eligible:
            content += "**Result: Likely Eligible**\n\n"
            content += f"Estimated Monthly Benefit: ${details['estimated_monthly_benefit']}\n\n"
            content += (
                "*This is a preliminary estimate. Actual eligibility and benefit "
                "amounts are determined by your state SNAP office based on detailed "
                "income and expense information.*"
            )
        else:
            content += "**Result: May Not Qualify Based on Income**\n\n"
            content += (
                "However, you may still qualify due to:\n"
                "- Certain deductions (shelter costs, dependent care)\n"
                "- Elderly or disabled household members\n"
                "- Special state programs\n\n"
                "We recommend applying anyway or contacting your local SNAP office."
            )

        return content

    def _get_snap_overview(self) -> str:
        """Get general SNAP program information."""
        return (
            "**SNAP (Supplemental Nutrition Assistance Program)**\n\n"
            "SNAP helps low-income individuals and families buy nutritious food.\n\n"
            "**To Check Eligibility, I Need:**\n"
            "- Household size (people who live and eat together)\n"
            "- Monthly gross income (before taxes)\n\n"
            "**How to Apply:**\n"
            "1. Apply online at your state's SNAP website\n"
            "2. Visit your local SNAP office\n"
            "3. Get help from a community organization\n\n"
            "**What You'll Need:**\n"
            "- Proof of identity\n"
            "- Proof of income\n"
            "- Social Security numbers\n"
            "- Proof of housing costs\n\n"
            "Tell me your household size and monthly income to check eligibility."
        )

    async def _handle_meal_schedule(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Schedule community meal pickups or deliveries.

        Args:
            request: Request with meal preferences
            context: Execution context

        Returns:
            SkillResponse with meal scheduling info
        """
        action = request.entities.get("action", "find")
        meal_type = request.entities.get("meal_type")
        location = request.entities.get("location")
        dietary = request.entities.get("dietary", [])
        delivery = request.entities.get("delivery", False)

        programs = await self.schedule_meal_pickup(
            meal_type=meal_type,
            location=location,
            dietary_needs=dietary,
            needs_delivery=delivery,
        )

        if not programs:
            return SkillResponse(
                content="No community meal programs found matching your criteria.\n\n"
                "**Other Options:**\n"
                "- Find food pantries in your area\n"
                "- Check Meals on Wheels eligibility\n"
                "- Contact 211 for local resources",
                success=True,
                data={"programs": []},
                suggestions=[
                    "Find food pantries",
                    "Check Meals on Wheels",
                    "Call 211",
                ],
            )

        content = f"**Community Meal Programs**\n\n"
        for program in programs:
            content += f"**{program.name}**\n"
            content += f"- Location: {program.location}\n"
            content += f"- Meal: {program.meal_type.title()}\n"
            if program.delivery_available:
                content += "- Delivery Available\n"
            content += f"- Schedule:\n"
            for day, time_str in program.schedule.items():
                content += f"  - {day}: {time_str}\n"
            if program.registration_required:
                content += "- Registration Required\n"
            spots = program.capacity - program.current_signups
            content += f"- Spots Available: {spots}\n\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "programs": [
                    {"id": p.program_id, "name": p.name, "meal": p.meal_type}
                    for p in programs
                ],
            },
            suggestions=[
                "Register for a meal program",
                "Set up recurring pickup",
                "Request delivery",
            ],
        )

    async def schedule_meal_pickup(
        self,
        meal_type: str | None = None,
        location: str | None = None,
        dietary_needs: list[str] | None = None,
        needs_delivery: bool = False,
    ) -> list[MealProgram]:
        """Find and schedule community meal pickups.

        Args:
            meal_type: Type of meal (breakfast, lunch, dinner)
            location: Preferred location
            dietary_needs: Dietary accommodations needed
            needs_delivery: Whether delivery is required

        Returns:
            List of matching meal programs
        """
        # In production, this would query the meal_scheduler span
        programs = [
            MealProgram(
                program_id="meal_001",
                name="Community Kitchen",
                location="Community Center, 789 Oak St",
                meal_type="lunch",
                schedule={
                    "Monday": "11:30am-1pm",
                    "Wednesday": "11:30am-1pm",
                    "Friday": "11:30am-1pm",
                },
                dietary_accommodations=[
                    DietaryRestriction.VEGETARIAN,
                    DietaryRestriction.DIABETIC_FRIENDLY,
                ],
                registration_required=False,
                delivery_available=False,
                capacity=100,
                current_signups=65,
            ),
            MealProgram(
                program_id="meal_002",
                name="Senior Meal Delivery",
                location="Delivered to your home",
                meal_type="dinner",
                schedule={
                    "Monday": "5pm delivery",
                    "Tuesday": "5pm delivery",
                    "Wednesday": "5pm delivery",
                    "Thursday": "5pm delivery",
                    "Friday": "5pm delivery",
                },
                dietary_accommodations=[
                    DietaryRestriction.LOW_SODIUM,
                    DietaryRestriction.DIABETIC_FRIENDLY,
                ],
                registration_required=True,
                delivery_available=True,
                capacity=50,
                current_signups=42,
            ),
        ]

        # Filter by delivery requirement
        if needs_delivery:
            programs = [p for p in programs if p.delivery_available]

        # Filter by meal type
        if meal_type:
            programs = [p for p in programs if p.meal_type == meal_type]

        return programs

    async def _handle_food_donate(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Process food donations from community members.

        Args:
            request: Request with donation details
            context: Execution context

        Returns:
            SkillResponse with donation processing info
        """
        donation_type = request.entities.get("type", "food")
        items = request.entities.get("items", [])
        quantity = request.entities.get("quantity")
        pickup_needed = request.entities.get("pickup", False)

        result = await self.process_donation(
            donation_type=donation_type,
            items=items,
            quantity=quantity,
            pickup_needed=pickup_needed,
            donor_id=context.user_id,
        )

        content = "**Thank You for Your Donation!**\n\n"
        content += f"Donation ID: {result['donation_id']}\n"
        content += f"Status: {result['status']}\n\n"

        if pickup_needed:
            content += "**Pickup Information:**\n"
            content += f"- Scheduled: {result['pickup_time']}\n"
            content += f"- Contact: {result['pickup_contact']}\n"
        else:
            content += "**Drop-off Locations:**\n"
            for location in result['drop_off_locations']:
                content += f"- {location['name']}: {location['address']}\n"
                content += f"  Hours: {location['hours']}\n"

        content += "\n**Accepted Items:**\n"
        content += "- Non-perishable foods (cans, boxes, jars)\n"
        content += "- Fresh produce (within 3 days of expiration)\n"
        content += "- Frozen items (if dropping off same day)\n\n"

        content += "**Not Accepted:**\n"
        content += "- Expired items\n"
        content += "- Opened packages\n"
        content += "- Home-prepared foods\n"

        return SkillResponse(
            content=content,
            success=True,
            data=result,
            requires_consensus=True if result.get("large_donation") else False,
            consensus_action="approve_food_distribution" if result.get("large_donation") else None,
            suggestions=[
                "Schedule recurring donation",
                "Get donation receipt",
                "Organize a food drive",
            ],
        )

    async def process_donation(
        self,
        donation_type: str,
        items: list[str],
        quantity: int | None,
        pickup_needed: bool,
        donor_id: str,
    ) -> dict[str, Any]:
        """Process a food donation.

        Args:
            donation_type: Type of donation
            items: List of items being donated
            quantity: Approximate quantity
            pickup_needed: Whether pickup is requested
            donor_id: ID of the donor

        Returns:
            Donation processing result
        """
        # In production, this would interact with inventory_system span
        return {
            "donation_id": f"DON-{datetime.now().strftime('%Y%m%d')}-001",
            "status": "Received",
            "pickup_time": "Saturday, Feb 15, 10am-12pm" if pickup_needed else None,
            "pickup_contact": "(510) 555-1234" if pickup_needed else None,
            "drop_off_locations": [
                {
                    "name": "Community Food Bank",
                    "address": "123 Main St",
                    "hours": "M-F 9am-5pm, Sat 10am-2pm",
                },
            ],
            "large_donation": quantity and quantity > 100,
        }

    async def _handle_nutrition_info(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Provide nutrition information and dietary guidance.

        Args:
            request: Request with nutrition query
            context: Execution context

        Returns:
            SkillResponse with nutrition information
        """
        topic = request.entities.get("topic", "general")
        dietary = request.entities.get("dietary", [])

        info = await self.get_nutrition_info(topic, dietary)

        content = f"**Nutrition Information: {topic.title()}**\n\n"
        content += info["overview"] + "\n\n"

        if info.get("tips"):
            content += "**Helpful Tips:**\n"
            for tip in info["tips"]:
                content += f"- {tip}\n"

        if info.get("resources"):
            content += "\n**Resources:**\n"
            for resource in info["resources"]:
                content += f"- {resource}\n"

        return SkillResponse(
            content=content,
            success=True,
            data={"topic": topic, "info": info},
            suggestions=[
                "Find diabetic-friendly foods",
                "Get recipes for dietary restrictions",
                "Connect with a nutritionist",
            ],
        )

    async def get_nutrition_info(
        self, topic: str, dietary: list[str]
    ) -> dict[str, Any]:
        """Get nutrition information for a topic.

        Args:
            topic: Nutrition topic
            dietary: Specific dietary considerations

        Returns:
            Nutrition information and tips
        """
        # In production, this could integrate with nutrition APIs
        return {
            "topic": topic,
            "overview": (
                "Good nutrition is essential for health and wellbeing. "
                "Focus on fruits, vegetables, whole grains, and lean proteins. "
                "Limit processed foods, added sugars, and excessive sodium."
            ),
            "tips": [
                "Plan meals ahead to save money and reduce waste",
                "Buy seasonal produce for better prices",
                "Cook in batches and freeze portions",
                "Read nutrition labels to make informed choices",
                "Stay hydrated with water instead of sugary drinks",
            ],
            "resources": [
                "SNAP-Ed nutrition classes (free for SNAP recipients)",
                "Community cooking workshops",
                "Local Extension office nutrition programs",
            ],
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for food access related information.

        Args:
            beliefs: Current belief state
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Filtered BDI state relevant to food access
        """
        food_domains = {"food", "nutrition", "snap", "pantry", "hunger"}

        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in food_domains
                or b.get("type") in [
                    "food_security_status",
                    "snap_status",
                    "dietary_needs",
                    "pantry_visit",
                ]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in [
                    "food_access",
                    "snap_enrollment",
                    "nutrition_improvement",
                ]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") in food_domains
            ],
        }
