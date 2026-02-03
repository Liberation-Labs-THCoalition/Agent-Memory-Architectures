"""
Solidarity Economy skill chip for Kintsugi CMA.

Supports cooperative development, time banking, and alternative economy
initiatives. Helps communities build worker-owned cooperatives, manage
time bank exchanges, and access Community Development Financial
Institutions (CDFIs) for alternative lending.

Intents handled:
- coop_start: Guide cooperative formation process
- coop_search: Search for existing cooperatives
- time_bank: Manage time bank credits and exchanges
- cdfi_loan: Find CDFI lending opportunities
- solidarity_resource: Get solidarity economy resources and templates

Example:
    chip = SolidarityEconomyChip()
    response = await chip.handle(
        SkillRequest(intent="coop_start", entities={"coop_type": "worker", "members": 5}),
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


class CoopType(str, Enum):
    """Types of cooperative structures."""
    WORKER = "worker_cooperative"
    CONSUMER = "consumer_cooperative"
    PRODUCER = "producer_cooperative"
    MULTI_STAKEHOLDER = "multi_stakeholder"
    HOUSING = "housing_cooperative"
    CREDIT_UNION = "credit_union"
    PLATFORM = "platform_cooperative"


class CoopStatus(str, Enum):
    """Status of cooperative development."""
    EXPLORING = "exploring"
    FORMING = "forming"
    INCORPORATED = "incorporated"
    OPERATING = "operating"
    ESTABLISHED = "established"


@dataclass
class Cooperative:
    """Information about a cooperative organization."""
    coop_id: str
    name: str
    coop_type: CoopType
    status: CoopStatus
    location: str
    members: int
    founded: int
    industry: str
    description: str
    contact: str
    website: str | None = None
    revenue: float | None = None
    accepting_members: bool = False


@dataclass
class TimeBankAccount:
    """Time bank account for tracking service credits."""
    account_id: str
    member_id: str
    balance_hours: float
    earned_total: float
    spent_total: float
    services_offered: list[str]
    services_needed: list[str]
    last_transaction: datetime | None = None


@dataclass
class TimeBankTransaction:
    """A time bank exchange transaction."""
    transaction_id: str
    provider_id: str
    recipient_id: str
    hours: float
    service: str
    description: str
    timestamp: datetime
    verified: bool = False


@dataclass
class CDFILoan:
    """Information about a CDFI loan product."""
    loan_id: str
    cdfi_name: str
    loan_type: str
    min_amount: float
    max_amount: float
    interest_rate: str
    term_months: int
    requirements: list[str]
    eligible_uses: list[str]
    location_served: str


class SolidarityEconomyChip(BaseSkillChip):
    """Support cooperative development, time banking, and alternative economies.

    This chip helps communities build economic alternatives by guiding
    cooperative formation, managing time bank exchanges, and connecting
    to CDFI lending resources. It provides governance templates and
    tracks time bank credits.

    Attributes:
        name: Unique chip identifier
        description: Human-readable description
        domain: COMMUNITY domain for collective economics
        efe_weights: High mission alignment for transformative work
        capabilities: READ_DATA, WRITE_DATA, FINANCIAL_OPERATIONS
        consensus_actions: Actions requiring approval
        required_spans: MCP tool spans needed
    """

    name = "solidarity_economy"
    description = "Support cooperative development, time banking, and alternative economies"
    domain = SkillDomain.COMMUNITY
    efe_weights = EFEWeights(
        mission_alignment=0.35,
        stakeholder_benefit=0.25,
        resource_efficiency=0.15,
        transparency=0.15,
        equity=0.10,
    )
    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.FINANCIAL_OPERATIONS,
    ]
    consensus_actions = ["approve_coop_formation", "time_bank_withdrawal", "cdfi_loan_application"]
    required_spans = ["coop_registry", "time_bank_ledger", "cdfi_network", "worker_owner_tools"]

    # Time bank exchange rates (all services valued equally at 1 hour = 1 credit)
    TIME_BANK_RATE = 1.0

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with solidarity economy information
        """
        handlers = {
            "coop_start": self._handle_coop_start,
            "coop_search": self._handle_coop_search,
            "time_bank": self._handle_time_bank,
            "cdfi_loan": self._handle_cdfi_loan,
            "solidarity_resource": self._handle_solidarity_resource,
        }

        handler = handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}. Supported intents: {list(handlers.keys())}",
                success=False,
            )

        return await handler(request, context)

    async def _handle_coop_start(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Guide the cooperative formation process.

        Args:
            request: Request with coop type and founding info
            context: Execution context

        Returns:
            SkillResponse with formation guidance
        """
        coop_type = request.entities.get("coop_type", "worker")
        members = request.entities.get("members", 3)
        industry = request.entities.get("industry")
        state = request.entities.get("state")

        process = await self.start_coop_process(
            coop_type=coop_type,
            initial_members=members,
            industry=industry,
            state=state,
        )

        content = f"**Starting a {coop_type.title()} Cooperative**\n\n"
        content += f"Members: {members}\n"
        if industry:
            content += f"Industry: {industry}\n"
        content += "\n"

        content += "**Formation Steps:**\n"
        for i, step in enumerate(process["steps"], 1):
            content += f"{i}. {step['name']}\n"
            content += f"   {step['description']}\n"
            content += f"   Timeline: {step['timeline']}\n\n"

        content += "**Key Documents Needed:**\n"
        for doc in process["documents"]:
            content += f"- {doc}\n"

        content += "\n**Estimated Costs:**\n"
        for cost_item, amount in process["estimated_costs"].items():
            content += f"- {cost_item}: ${amount:,}\n"

        content += "\n**Support Resources:**\n"
        for resource in process["resources"]:
            content += f"- {resource['name']}: {resource['contact']}\n"

        return SkillResponse(
            content=content,
            success=True,
            data=process,
            requires_consensus=True,
            consensus_action="approve_coop_formation",
            suggestions=[
                "Get governance templates",
                "Find a coop developer",
                "Connect with similar coops",
            ],
        )

    async def start_coop_process(
        self,
        coop_type: str,
        initial_members: int,
        industry: str | None,
        state: str | None,
    ) -> dict[str, Any]:
        """Create a cooperative formation process plan.

        Args:
            coop_type: Type of cooperative to form
            initial_members: Number of founding members
            industry: Industry sector
            state: State for incorporation

        Returns:
            Formation process plan with steps and resources
        """
        # In production, this would query worker_owner_tools span
        return {
            "coop_type": coop_type,
            "members": initial_members,
            "steps": [
                {
                    "name": "Feasibility Study",
                    "description": "Assess market demand, skills, and financial viability",
                    "timeline": "2-4 weeks",
                },
                {
                    "name": "Business Planning",
                    "description": "Develop business plan, financial projections, governance structure",
                    "timeline": "4-8 weeks",
                },
                {
                    "name": "Legal Formation",
                    "description": "Draft bylaws, articles of incorporation, operating agreement",
                    "timeline": "2-4 weeks",
                },
                {
                    "name": "Capitalization",
                    "description": "Secure member equity, loans, grants for startup capital",
                    "timeline": "4-12 weeks",
                },
                {
                    "name": "Launch",
                    "description": "File incorporation, open accounts, begin operations",
                    "timeline": "2-4 weeks",
                },
            ],
            "documents": [
                "Articles of Incorporation",
                "Bylaws",
                "Operating Agreement",
                "Member Equity Agreement",
                "Business Plan",
                "Financial Projections",
            ],
            "estimated_costs": {
                "Incorporation fees": 500,
                "Legal assistance": 2500,
                "Business planning": 1000,
                "Initial operating capital": 10000,
            },
            "resources": [
                {
                    "name": "US Federation of Worker Cooperatives",
                    "contact": "usworker.coop",
                },
                {
                    "name": "Democracy at Work Institute",
                    "contact": "institute.coop",
                },
                {
                    "name": "Local Cooperative Development Center",
                    "contact": "Contact your regional center",
                },
            ],
        }

    async def _handle_coop_search(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Search for existing cooperatives.

        Args:
            request: Request with search criteria
            context: Execution context

        Returns:
            SkillResponse with matching cooperatives
        """
        coop_type = request.entities.get("coop_type")
        industry = request.entities.get("industry")
        location = request.entities.get("location")
        accepting_members = request.entities.get("accepting_members", False)

        coops = await self.search_coops(
            coop_type=coop_type,
            industry=industry,
            location=location,
            accepting_members=accepting_members,
        )

        if not coops:
            return SkillResponse(
                content="No cooperatives found matching your criteria.\n\n"
                "Consider:\n"
                "- Expanding your search area\n"
                "- Trying different industry filters\n"
                "- Starting a new cooperative!",
                success=True,
                data={"coops": []},
                suggestions=[
                    "Expand search",
                    "Start a new coop",
                    "Find coop developers",
                ],
            )

        content = f"**Found {len(coops)} Cooperatives**\n\n"
        for coop in coops:
            content += f"**{coop.name}**\n"
            content += f"- Type: {coop.coop_type.value.replace('_', ' ').title()}\n"
            content += f"- Industry: {coop.industry}\n"
            content += f"- Location: {coop.location}\n"
            content += f"- Members: {coop.members}\n"
            content += f"- Founded: {coop.founded}\n"
            if coop.accepting_members:
                content += "- **Currently Accepting Members**\n"
            content += f"- Contact: {coop.contact}\n\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "coops": [
                    {"id": c.coop_id, "name": c.name, "type": c.coop_type.value}
                    for c in coops
                ],
            },
            suggestions=[
                "Learn more about a coop",
                "Apply for membership",
                "Connect with coop networks",
            ],
        )

    async def search_coops(
        self,
        coop_type: str | None = None,
        industry: str | None = None,
        location: str | None = None,
        accepting_members: bool = False,
    ) -> list[Cooperative]:
        """Search the cooperative registry.

        Args:
            coop_type: Filter by cooperative type
            industry: Filter by industry sector
            location: Filter by geographic location
            accepting_members: Only show coops accepting new members

        Returns:
            List of matching cooperatives
        """
        # In production, this would query the coop_registry span
        sample_coops = [
            Cooperative(
                coop_id="coop_001",
                name="Green City Grocery",
                coop_type=CoopType.CONSUMER,
                status=CoopStatus.ESTABLISHED,
                location="Oakland, CA",
                members=2500,
                founded=2012,
                industry="Grocery/Food Retail",
                description="Community-owned grocery store serving local neighborhoods",
                contact="join@greencitygrocery.coop",
                website="greencitygrocery.coop",
                accepting_members=True,
            ),
            Cooperative(
                coop_id="coop_002",
                name="Tech Workers Collective",
                coop_type=CoopType.WORKER,
                status=CoopStatus.OPERATING,
                location="San Francisco, CA",
                members=15,
                founded=2019,
                industry="Technology/Software",
                description="Worker-owned software development cooperative",
                contact="hello@techworkerscollective.coop",
                website="techworkerscollective.coop",
                accepting_members=True,
            ),
            Cooperative(
                coop_id="coop_003",
                name="Community Land Trust",
                coop_type=CoopType.HOUSING,
                status=CoopStatus.ESTABLISHED,
                location="Berkeley, CA",
                members=120,
                founded=2008,
                industry="Housing",
                description="Permanently affordable community-controlled housing",
                contact="info@communityland.coop",
                accepting_members=False,
            ),
        ]

        results = sample_coops

        if accepting_members:
            results = [c for c in results if c.accepting_members]

        if location:
            results = [c for c in results if location.lower() in c.location.lower()]

        return results

    async def _handle_time_bank(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Manage time bank credits and exchanges.

        Args:
            request: Request with time bank action
            context: Execution context

        Returns:
            SkillResponse with time bank information
        """
        action = request.entities.get("action", "balance")
        hours = request.entities.get("hours")
        service = request.entities.get("service")
        recipient = request.entities.get("recipient")

        result = await self.manage_time_bank(
            action=action,
            user_id=context.user_id,
            hours=hours,
            service=service,
            recipient=recipient,
        )

        if action == "balance":
            account = result["account"]
            content = "**Your Time Bank Account**\n\n"
            content += f"Current Balance: {account.balance_hours:.1f} hours\n"
            content += f"Total Earned: {account.earned_total:.1f} hours\n"
            content += f"Total Spent: {account.spent_total:.1f} hours\n\n"

            content += "**Services You Offer:**\n"
            for svc in account.services_offered:
                content += f"- {svc}\n"

            content += "\n**Services You Need:**\n"
            for svc in account.services_needed:
                content += f"- {svc}\n"

        elif action == "exchange":
            content = "**Time Bank Exchange**\n\n"
            content += f"Service: {service}\n"
            content += f"Hours: {hours}\n"
            content += f"Recipient: {recipient}\n"
            content += f"Status: {result['status']}\n"

            if result.get("requires_approval"):
                content += "\n*This exchange requires verification before credits transfer.*"

        elif action == "browse":
            content = "**Available Services in Time Bank**\n\n"
            for category, services in result["available_services"].items():
                content += f"**{category}:**\n"
                for svc in services:
                    content += f"- {svc['name']} ({svc['provider']})\n"
                content += "\n"

        else:
            content = f"Action '{action}' not recognized. Try: balance, exchange, browse"

        return SkillResponse(
            content=content,
            success=True,
            data=result,
            requires_consensus=action == "exchange" and hours and hours > 5,
            consensus_action="time_bank_withdrawal" if action == "exchange" else None,
            suggestions=[
                "Browse available services",
                "Offer a new service",
                "View exchange history",
            ],
        )

    async def manage_time_bank(
        self,
        action: str,
        user_id: str,
        hours: float | None = None,
        service: str | None = None,
        recipient: str | None = None,
    ) -> dict[str, Any]:
        """Manage time bank account and transactions.

        Args:
            action: Action to perform (balance, exchange, browse)
            user_id: User's account ID
            hours: Hours for exchange
            service: Service being exchanged
            recipient: Recipient of service

        Returns:
            Time bank operation result
        """
        # In production, this would query the time_bank_ledger span
        if action == "balance":
            return {
                "account": TimeBankAccount(
                    account_id=f"tb_{user_id}",
                    member_id=user_id,
                    balance_hours=12.5,
                    earned_total=25.0,
                    spent_total=12.5,
                    services_offered=["Gardening", "Computer repair", "Tutoring"],
                    services_needed=["House cleaning", "Cooking", "Transportation"],
                    last_transaction=datetime(2024, 1, 15, tzinfo=timezone.utc),
                ),
            }
        elif action == "exchange":
            return {
                "status": "Pending Verification",
                "transaction_id": f"tx_{datetime.now().strftime('%Y%m%d%H%M')}",
                "hours": hours,
                "service": service,
                "requires_approval": hours and hours > 5,
            }
        elif action == "browse":
            return {
                "available_services": {
                    "Home & Garden": [
                        {"name": "Lawn mowing", "provider": "John D."},
                        {"name": "Home repair", "provider": "Maria G."},
                    ],
                    "Education": [
                        {"name": "Spanish tutoring", "provider": "Carlos R."},
                        {"name": "Math help", "provider": "Susan K."},
                    ],
                    "Technology": [
                        {"name": "Computer setup", "provider": "Alex T."},
                        {"name": "Phone troubleshooting", "provider": "Kim L."},
                    ],
                },
            }
        return {}

    async def _handle_cdfi_loan(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Find CDFI lending opportunities.

        Args:
            request: Request with loan criteria
            context: Execution context

        Returns:
            SkillResponse with CDFI loan options
        """
        loan_amount = request.entities.get("amount")
        loan_purpose = request.entities.get("purpose")
        location = request.entities.get("location")
        business_type = request.entities.get("business_type")

        loans = await self.find_cdfi_lending(
            amount=loan_amount,
            purpose=loan_purpose,
            location=location,
            business_type=business_type,
        )

        if not loans:
            return SkillResponse(
                content="No CDFI loan products found matching your criteria.\n\n"
                "CDFIs (Community Development Financial Institutions) provide "
                "affordable lending to underserved communities. Try:\n"
                "- Adjusting loan amount\n"
                "- Broadening location search\n"
                "- Checking traditional small business resources",
                success=True,
                data={"loans": []},
                suggestions=[
                    "Learn about CDFIs",
                    "Find local credit unions",
                    "Explore cooperative financing",
                ],
            )

        content = "**CDFI Loan Opportunities**\n\n"
        content += (
            "*CDFIs provide affordable financing for communities underserved "
            "by traditional banks.*\n\n"
        )

        for loan in loans:
            content += f"**{loan.cdfi_name}**\n"
            content += f"- Loan Type: {loan.loan_type}\n"
            content += f"- Amount: ${loan.min_amount:,.0f} - ${loan.max_amount:,.0f}\n"
            content += f"- Interest Rate: {loan.interest_rate}\n"
            content += f"- Term: {loan.term_months} months\n"
            content += f"- Serves: {loan.location_served}\n"
            content += "- Eligible Uses:\n"
            for use in loan.eligible_uses[:3]:
                content += f"  - {use}\n"
            content += "\n"

        return SkillResponse(
            content=content,
            success=True,
            data={
                "loans": [
                    {"id": l.loan_id, "cdfi": l.cdfi_name, "max": l.max_amount}
                    for l in loans
                ],
            },
            requires_consensus=True,
            consensus_action="cdfi_loan_application",
            suggestions=[
                "Start loan application",
                "Compare loan terms",
                "Connect with CDFI advisor",
            ],
        )

    async def find_cdfi_lending(
        self,
        amount: float | None = None,
        purpose: str | None = None,
        location: str | None = None,
        business_type: str | None = None,
    ) -> list[CDFILoan]:
        """Find CDFI loan products matching criteria.

        Args:
            amount: Desired loan amount
            purpose: Loan purpose (startup, expansion, etc.)
            location: Geographic location
            business_type: Type of business (coop, nonprofit, etc.)

        Returns:
            List of matching CDFI loan products
        """
        # In production, this would query the cdfi_network span
        return [
            CDFILoan(
                loan_id="cdfi_001",
                cdfi_name="Community Development Fund",
                loan_type="Small Business Loan",
                min_amount=5000,
                max_amount=100000,
                interest_rate="6-9% fixed",
                term_months=60,
                requirements=[
                    "Business plan",
                    "Financial statements",
                    "2 years tax returns",
                ],
                eligible_uses=[
                    "Working capital",
                    "Equipment purchase",
                    "Real estate acquisition",
                    "Business expansion",
                ],
                location_served="California",
            ),
            CDFILoan(
                loan_id="cdfi_002",
                cdfi_name="Cooperative Fund of the Northeast",
                loan_type="Cooperative Development Loan",
                min_amount=25000,
                max_amount=500000,
                interest_rate="5-7% fixed",
                term_months=84,
                requirements=[
                    "Cooperative structure",
                    "Member equity contribution",
                    "Governance documents",
                ],
                eligible_uses=[
                    "Cooperative startup",
                    "Conversion to cooperative",
                    "Cooperative expansion",
                ],
                location_served="Northeast US (will consider others)",
            ),
        ]

    async def _handle_solidarity_resource(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Get solidarity economy resources and templates.

        Args:
            request: Request with resource type
            context: Execution context

        Returns:
            SkillResponse with resources
        """
        resource_type = request.entities.get("type", "general")

        resources = await self.get_solidarity_resources(resource_type)

        content = f"**Solidarity Economy Resources: {resource_type.title()}**\n\n"

        if resources.get("templates"):
            content += "**Templates & Documents:**\n"
            for template in resources["templates"]:
                content += f"- {template['name']}: {template['description']}\n"
            content += "\n"

        if resources.get("organizations"):
            content += "**Organizations & Networks:**\n"
            for org in resources["organizations"]:
                content += f"- **{org['name']}**: {org['description']}\n"
                content += f"  Contact: {org['contact']}\n"
            content += "\n"

        if resources.get("education"):
            content += "**Educational Resources:**\n"
            for edu in resources["education"]:
                content += f"- {edu}\n"

        return SkillResponse(
            content=content,
            success=True,
            data=resources,
            suggestions=[
                "Download governance templates",
                "Find a coop developer",
                "Join a solidarity economy network",
            ],
        )

    async def get_solidarity_resources(
        self, resource_type: str
    ) -> dict[str, Any]:
        """Get solidarity economy resources.

        Args:
            resource_type: Type of resources needed

        Returns:
            Collection of solidarity economy resources
        """
        # In production, this would query multiple spans
        return {
            "type": resource_type,
            "templates": [
                {
                    "name": "Worker Cooperative Bylaws",
                    "description": "Template bylaws for worker-owned cooperatives",
                },
                {
                    "name": "Operating Agreement",
                    "description": "Member operating agreement template",
                },
                {
                    "name": "Member Equity Agreement",
                    "description": "Template for member buy-in and patronage",
                },
            ],
            "organizations": [
                {
                    "name": "US Federation of Worker Cooperatives",
                    "description": "National grassroots membership organization",
                    "contact": "usworker.coop",
                },
                {
                    "name": "New Economy Coalition",
                    "description": "Network building the solidarity economy",
                    "contact": "neweconomy.net",
                },
            ],
            "education": [
                "Cooperative 101 online course",
                "Time banking fundamentals webinar",
                "Solidarity economy reading list",
                "Local coop incubator programs",
            ],
        }

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for solidarity economy information.

        Args:
            beliefs: Current belief state
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Filtered BDI state relevant to solidarity economy
        """
        economy_domains = {"cooperative", "time_bank", "cdfi", "solidarity", "mutual_aid"}

        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in economy_domains
                or b.get("type") in [
                    "coop_status",
                    "time_bank_balance",
                    "loan_status",
                    "economic_model",
                ]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in [
                    "form_cooperative",
                    "join_time_bank",
                    "access_capital",
                    "economic_democracy",
                ]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") in economy_domains
            ],
        }
