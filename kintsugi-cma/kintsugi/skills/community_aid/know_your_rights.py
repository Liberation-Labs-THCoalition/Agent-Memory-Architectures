"""
Know Your Rights skill chip for Kintsugi CMA.

Provides legal information, rights education, and clinic scheduling
for community members. Supports multi-language content and
jurisdiction-aware (state/local) legal information lookup.

IMPORTANT DISCLAIMER: This chip provides general legal information
for educational purposes only. It does NOT provide legal advice.
Users should consult with a qualified attorney for specific legal matters.

Intents handled:
- rights_lookup: Look up rights information by topic/jurisdiction
- legal_clinic: Schedule or find legal clinic appointments
- know_rights_workshop: Plan and schedule know-your-rights workshops
- legal_resource: Find legal resources and referrals
- rights_card: Generate know-your-rights cards in multiple languages

Example:
    chip = KnowYourRightsChip()
    response = await chip.handle(
        SkillRequest(intent="rights_lookup", entities={"topic": "tenant_rights", "state": "CA"}),
        context
    )
"""

from dataclasses import dataclass
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


class RightsTopic(str, Enum):
    """Categories of rights information available."""
    TENANT = "tenant_rights"
    WORKER = "worker_rights"
    IMMIGRATION = "immigration_rights"
    POLICE_ENCOUNTER = "police_encounter"
    PROTEST = "protest_rights"
    HEALTHCARE = "healthcare_rights"
    EDUCATION = "education_rights"
    DISABILITY = "disability_rights"
    VOTING = "voting_rights"
    CONSUMER = "consumer_rights"


@dataclass
class RightsInfo:
    """Legal rights information for a specific topic and jurisdiction."""
    topic: RightsTopic
    jurisdiction: str
    title: str
    summary: str
    key_rights: list[str]
    what_to_do: list[str]
    what_not_to_do: list[str]
    resources: list[dict[str, str]]
    last_updated: datetime
    language: str = "en"


@dataclass
class LegalClinic:
    """Information about a scheduled legal clinic."""
    clinic_id: str
    name: str
    date: datetime
    location: str
    topics: list[str]
    languages: list[str]
    capacity: int
    registered: int
    attorneys: list[str]
    is_virtual: bool = False


class KnowYourRightsChip(BaseSkillChip):
    """Provide legal information, rights education, and clinic scheduling.

    This chip serves as an educational resource to help community members
    understand their legal rights across various domains. It provides
    jurisdiction-aware information, schedules legal clinics, and generates
    know-your-rights materials in multiple languages.

    DISCLAIMER: This chip provides general legal information for educational
    purposes only. It does NOT constitute legal advice. Users should always
    consult with a qualified attorney for specific legal matters.

    Attributes:
        name: Unique chip identifier
        description: Human-readable description
        domain: ADVOCACY domain for rights-focused work
        efe_weights: Mission and stakeholder focused weights
        capabilities: READ_DATA, SCHEDULE_TASKS, GENERATE_REPORTS
        consensus_actions: Actions requiring approval
        required_spans: MCP tool spans needed
    """

    name = "know_your_rights"
    description = "Provide legal information, rights education, and clinic scheduling"
    domain = SkillDomain.ADVOCACY
    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.35,
        resource_efficiency=0.10,
        transparency=0.15,
        equity=0.10,
    )
    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.SCHEDULE_TASKS,
        SkillCapability.GENERATE_REPORTS,
    ]
    consensus_actions = ["schedule_legal_clinic", "distribute_legal_materials"]
    required_spans = ["legal_database", "clinic_scheduler", "translation_api"]

    # Supported languages for materials
    SUPPORTED_LANGUAGES = ["en", "es", "zh", "vi", "ko", "tl", "ar", "ru", "fr", "pt"]

    # Standard legal disclaimer
    LEGAL_DISCLAIMER = (
        "DISCLAIMER: This information is provided for educational purposes only "
        "and does not constitute legal advice. Laws vary by jurisdiction and change "
        "over time. Please consult with a qualified attorney for advice specific "
        "to your situation."
    )

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with legal information and disclaimer
        """
        handlers = {
            "rights_lookup": self._handle_rights_lookup,
            "legal_clinic": self._handle_legal_clinic,
            "know_rights_workshop": self._handle_workshop,
            "legal_resource": self._handle_legal_resource,
            "rights_card": self._handle_rights_card,
        }

        handler = handlers.get(request.intent)
        if not handler:
            return SkillResponse(
                content=f"Unknown intent: {request.intent}. Supported intents: {list(handlers.keys())}",
                success=False,
            )

        return await handler(request, context)

    async def _handle_rights_lookup(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Look up rights information by topic and jurisdiction.

        Args:
            request: Request with topic and optional jurisdiction entities
            context: Execution context

        Returns:
            SkillResponse with rights information and disclaimer
        """
        topic = request.entities.get("topic", "general")
        state = request.entities.get("state", "federal")
        language = request.entities.get("language", "en")

        # Validate language
        if language not in self.SUPPORTED_LANGUAGES:
            language = "en"

        rights_info = await self.lookup_rights(topic, state, language)

        if not rights_info:
            return SkillResponse(
                content=f"No rights information found for topic '{topic}' in {state}. "
                f"\n\n{self.LEGAL_DISCLAIMER}",
                success=False,
                suggestions=[
                    "Try a different topic or jurisdiction",
                    "Contact a legal aid organization for assistance",
                    "Schedule a legal clinic appointment",
                ],
            )

        content = self._format_rights_info(rights_info)

        return SkillResponse(
            content=content,
            success=True,
            data={
                "topic": topic,
                "jurisdiction": state,
                "language": language,
                "rights_info": {
                    "title": rights_info.title,
                    "key_rights": rights_info.key_rights,
                    "resources": rights_info.resources,
                },
            },
            suggestions=[
                "Would you like a know-your-rights card for this topic?",
                "Find legal clinics in your area",
                "Request information in another language",
            ],
        )

    async def lookup_rights(
        self, topic: str, jurisdiction: str, language: str = "en"
    ) -> RightsInfo | None:
        """Look up rights information from the legal database.

        Args:
            topic: The rights topic to look up
            jurisdiction: State or 'federal' for jurisdiction
            language: Language code for content

        Returns:
            RightsInfo if found, None otherwise
        """
        # In production, this would query the legal_database span
        # For now, return structured sample data
        topic_enum = None
        for t in RightsTopic:
            if t.value == topic or t.name.lower() == topic.lower():
                topic_enum = t
                break

        if not topic_enum:
            return None

        # Sample rights info (would come from database)
        return RightsInfo(
            topic=topic_enum,
            jurisdiction=jurisdiction,
            title=f"{topic_enum.value.replace('_', ' ').title()} in {jurisdiction.upper()}",
            summary=f"Overview of {topic_enum.value.replace('_', ' ')} protections.",
            key_rights=[
                "You have the right to remain silent",
                "You have the right to refuse consent to searches",
                "You have the right to an attorney",
            ],
            what_to_do=[
                "Stay calm and be polite",
                "Ask if you are free to leave",
                "Document the interaction if safe to do so",
            ],
            what_not_to_do=[
                "Do not physically resist",
                "Do not provide false information",
                "Do not consent to searches",
            ],
            resources=[
                {"name": "ACLU", "url": "https://www.aclu.org"},
                {"name": "Legal Aid Society", "url": "https://www.legalaid.org"},
            ],
            last_updated=datetime.now(timezone.utc),
            language=language,
        )

    async def _handle_legal_clinic(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Handle legal clinic scheduling and lookup.

        Args:
            request: Request with clinic action and details
            context: Execution context

        Returns:
            SkillResponse with clinic information
        """
        action = request.entities.get("action", "find")
        topic = request.entities.get("topic")
        location = request.entities.get("location", context.metadata.get("location"))
        language = request.entities.get("language", "en")

        if action == "schedule":
            return await self._schedule_clinic_action(request, context)

        clinics = await self.schedule_clinic(topic, location, language)

        if not clinics:
            return SkillResponse(
                content=f"No upcoming legal clinics found in {location}. "
                f"Check back soon or contact local legal aid for assistance."
                f"\n\n{self.LEGAL_DISCLAIMER}",
                success=True,
                data={"clinics": []},
                suggestions=[
                    "Set up an alert for new clinic dates",
                    "Find virtual legal clinic options",
                    "Contact legal aid hotline",
                ],
            )

        content = "**Upcoming Legal Clinics**\n\n"
        for clinic in clinics:
            content += f"- **{clinic.name}**\n"
            content += f"  Date: {clinic.date.strftime('%B %d, %Y at %I:%M %p')}\n"
            content += f"  Location: {clinic.location}\n"
            content += f"  Topics: {', '.join(clinic.topics)}\n"
            content += f"  Languages: {', '.join(clinic.languages)}\n"
            content += f"  Spots Available: {clinic.capacity - clinic.registered}\n\n"

        content += f"\n{self.LEGAL_DISCLAIMER}"

        return SkillResponse(
            content=content,
            success=True,
            data={"clinics": [{"id": c.clinic_id, "name": c.name} for c in clinics]},
            suggestions=["Register for a clinic", "Get directions", "Add to calendar"],
        )

    async def schedule_clinic(
        self, topic: str | None, location: str | None, language: str = "en"
    ) -> list[LegalClinic]:
        """Find available legal clinics matching criteria.

        Args:
            topic: Filter by legal topic
            location: Filter by location/area
            language: Filter by language offered

        Returns:
            List of matching legal clinics
        """
        # In production, this would query the clinic_scheduler span
        # Return sample clinic data
        return [
            LegalClinic(
                clinic_id="clinic_001",
                name="Community Legal Clinic",
                date=datetime(2024, 2, 15, 18, 0, tzinfo=timezone.utc),
                location="Community Center, 123 Main St",
                topics=["tenant_rights", "worker_rights"],
                languages=["en", "es"],
                capacity=20,
                registered=12,
                attorneys=["Maria Garcia, Esq.", "John Smith, Esq."],
                is_virtual=False,
            ),
        ]

    async def _schedule_clinic_action(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Schedule a new legal clinic (requires consensus).

        Args:
            request: Request with clinic details
            context: Execution context

        Returns:
            SkillResponse indicating consensus needed
        """
        return SkillResponse(
            content="Scheduling a new legal clinic requires approval from authorized coordinators.",
            success=True,
            requires_consensus=True,
            consensus_action="schedule_legal_clinic",
            data={"proposed_clinic": request.entities},
        )

    async def _handle_workshop(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Plan and schedule know-your-rights workshops.

        Args:
            request: Request with workshop details
            context: Execution context

        Returns:
            SkillResponse with workshop planning info
        """
        topic = request.entities.get("topic", "general_rights")
        audience = request.entities.get("audience", "community")
        language = request.entities.get("language", "en")

        workshop_plan = await self.plan_workshop(topic, audience, language)

        return SkillResponse(
            content=f"**Know Your Rights Workshop Plan**\n\n"
            f"Topic: {topic.replace('_', ' ').title()}\n"
            f"Target Audience: {audience}\n"
            f"Language: {language}\n\n"
            f"**Suggested Agenda:**\n{workshop_plan['agenda']}\n\n"
            f"**Materials Needed:**\n{workshop_plan['materials']}\n\n"
            f"**Recommended Duration:** {workshop_plan['duration']}\n\n"
            f"{self.LEGAL_DISCLAIMER}",
            success=True,
            data={"workshop_plan": workshop_plan},
            suggestions=[
                "Schedule this workshop",
                "Generate participant materials",
                "Find facilitators",
            ],
        )

    async def plan_workshop(
        self, topic: str, audience: str, language: str
    ) -> dict[str, Any]:
        """Create a workshop plan for know-your-rights education.

        Args:
            topic: The rights topic to cover
            audience: Target audience description
            language: Primary language for workshop

        Returns:
            Workshop plan with agenda, materials, and logistics
        """
        return {
            "topic": topic,
            "audience": audience,
            "language": language,
            "duration": "2 hours",
            "agenda": (
                "1. Welcome and introductions (10 min)\n"
                "2. Overview of rights (20 min)\n"
                "3. Scenario role-plays (30 min)\n"
                "4. Q&A with legal expert (30 min)\n"
                "5. Resources and next steps (15 min)\n"
                "6. Wrap-up and evaluation (15 min)"
            ),
            "materials": (
                "- Know-your-rights cards\n"
                "- Scenario scripts\n"
                "- Resource handouts\n"
                "- Evaluation forms\n"
                "- Contact cards for legal aid"
            ),
            "facilitators_needed": 2,
            "legal_review_required": True,
        }

    async def _handle_legal_resource(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Find legal resources and referrals.

        Args:
            request: Request with resource type and location
            context: Execution context

        Returns:
            SkillResponse with legal resources
        """
        resource_type = request.entities.get("type", "general")
        location = request.entities.get("location")
        urgency = request.entities.get("urgency", "normal")

        resources = await self.find_legal_resources(resource_type, location, urgency)

        content = "**Legal Resources**\n\n"
        for resource in resources:
            content += f"**{resource['name']}**\n"
            content += f"- Type: {resource['type']}\n"
            content += f"- Contact: {resource['contact']}\n"
            if resource.get("hotline"):
                content += f"- Hotline: {resource['hotline']}\n"
            content += f"- Services: {resource['services']}\n\n"

        content += f"\n{self.LEGAL_DISCLAIMER}"

        return SkillResponse(
            content=content,
            success=True,
            data={"resources": resources},
            suggestions=["Get more details", "Find pro bono attorneys", "Emergency legal help"],
        )

    async def find_legal_resources(
        self, resource_type: str, location: str | None, urgency: str
    ) -> list[dict[str, Any]]:
        """Find legal resources matching criteria.

        Args:
            resource_type: Type of legal resource needed
            location: Geographic area
            urgency: Priority level (emergency, urgent, normal)

        Returns:
            List of legal resources with contact info
        """
        # In production, this would query the legal_database span
        return [
            {
                "name": "Local Legal Aid Society",
                "type": "legal_aid",
                "contact": "contact@legalaid.org",
                "hotline": "1-800-LEGAL-AID",
                "services": "Free legal services for low-income individuals",
            },
            {
                "name": "Immigrant Rights Hotline",
                "type": "immigration",
                "contact": "help@immigrantrights.org",
                "hotline": "1-800-IMM-HELP",
                "services": "Immigration legal assistance and know-your-rights",
            },
        ]

    async def _handle_rights_card(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate know-your-rights cards.

        Args:
            request: Request with topic and language
            context: Execution context

        Returns:
            SkillResponse with rights card content
        """
        topic = request.entities.get("topic", "police_encounter")
        language = request.entities.get("language", "en")
        format_type = request.entities.get("format", "text")

        rights_card = await self.generate_rights_card(topic, language)

        return SkillResponse(
            content=f"**Know Your Rights Card: {topic.replace('_', ' ').title()}**\n"
            f"*Language: {language}*\n\n"
            f"{rights_card['content']}\n\n"
            f"---\n{self.LEGAL_DISCLAIMER}",
            success=True,
            data={"rights_card": rights_card},
            suggestions=[
                "Download printable version",
                "Get card in another language",
                "Share with community",
            ],
        )

    async def generate_rights_card(
        self, topic: str, language: str
    ) -> dict[str, Any]:
        """Generate a know-your-rights card for the specified topic.

        Args:
            topic: Rights topic for the card
            language: Language for content

        Returns:
            Rights card content and metadata
        """
        # In production, this would use the translation_api span
        return {
            "topic": topic,
            "language": language,
            "content": (
                "YOU HAVE RIGHTS\n\n"
                "1. You have the right to remain silent.\n"
                "2. You have the right to refuse searches.\n"
                "3. You have the right to a lawyer.\n"
                "4. If arrested, say: 'I wish to remain silent. "
                "I want to speak to a lawyer.'\n\n"
                "EMERGENCY CONTACTS:\n"
                "Legal Hotline: 1-800-LEGAL-AID\n"
                "Immigration Hotline: 1-800-IMM-HELP"
            ),
            "printable_url": None,  # Would be generated in production
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def _format_rights_info(self, rights_info: RightsInfo) -> str:
        """Format rights information for display.

        Args:
            rights_info: The rights information to format

        Returns:
            Formatted string for display
        """
        content = f"**{rights_info.title}**\n\n"
        content += f"{rights_info.summary}\n\n"

        content += "**Your Key Rights:**\n"
        for right in rights_info.key_rights:
            content += f"- {right}\n"

        content += "\n**What To Do:**\n"
        for action in rights_info.what_to_do:
            content += f"- {action}\n"

        content += "\n**What NOT To Do:**\n"
        for action in rights_info.what_not_to_do:
            content += f"- {action}\n"

        content += "\n**Resources:**\n"
        for resource in rights_info.resources:
            content += f"- [{resource['name']}]({resource['url']})\n"

        content += f"\n*Last updated: {rights_info.last_updated.strftime('%B %d, %Y')}*\n"
        content += f"\n---\n{self.LEGAL_DISCLAIMER}"

        return content

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Filter BDI context for legal/rights-related information.

        Args:
            beliefs: Current belief state
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Filtered BDI state relevant to rights and legal matters
        """
        legal_domains = {"legal", "rights", "advocacy", "immigration", "housing", "worker"}

        return {
            "beliefs": [
                b for b in beliefs
                if b.get("domain") in legal_domains
                or b.get("type") in ["legal_status", "rights_awareness", "clinic_scheduled"]
            ],
            "desires": [
                d for d in desires
                if d.get("type") in ["learn_rights", "legal_help", "advocacy"]
            ],
            "intentions": [
                i for i in intentions
                if i.get("domain") in legal_domains
            ],
        }
