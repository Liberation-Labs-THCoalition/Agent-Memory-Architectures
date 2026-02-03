"""
Content Drafter Skill Chip for Kintsugi CMA.

This chip drafts communications, reports, and content with SB 942 AI labeling
compliance. California's SB 942 (2024) requires disclosure when content is
generated or substantially modified by AI systems.

Key capabilities:
- Draft emails, newsletters, and communications
- Create social media content
- Generate reports and documents
- Apply templates and brand guidelines
- Ensure SB 942 AI disclosure compliance

Example:
    chip = ContentDrafterChip()
    request = SkillRequest(
        intent="draft_email",
        entities={"recipient_type": "donors", "topic": "year_end_appeal"}
    )
    response = await chip.handle(request, context)
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
    register_chip,
)


class ContentType(str, Enum):
    """Types of content that can be drafted."""
    EMAIL = "email"
    NEWSLETTER = "newsletter"
    SOCIAL_MEDIA = "social_media"
    PRESS_RELEASE = "press_release"
    REPORT = "report"
    BLOG_POST = "blog_post"
    APPEAL = "appeal"
    THANK_YOU = "thank_you"


class Platform(str, Enum):
    """Social media and communication platforms."""
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    EMAIL = "email"
    WEB = "web"


class ContentStatus(str, Enum):
    """Status of drafted content."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class DraftedContent:
    """Represents a piece of drafted content.

    Attributes:
        id: Unique identifier
        content_type: Type of content
        title: Content title or subject line
        body: Main content body
        platform: Target platform
        metadata: Additional metadata
        status: Current status
        sb942_label: AI disclosure label
        created_at: Creation timestamp
        word_count: Word count of body
        character_count: Character count (for social)
    """
    id: str
    content_type: ContentType
    title: str
    body: str
    platform: Platform
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ContentStatus = ContentStatus.DRAFT
    sb942_label: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def word_count(self) -> int:
        """Count words in body."""
        return len(self.body.split())

    @property
    def character_count(self) -> int:
        """Count characters in body."""
        return len(self.body)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "content_type": self.content_type.value,
            "title": self.title,
            "body": self.body,
            "platform": self.platform.value,
            "metadata": self.metadata,
            "status": self.status.value,
            "sb942_label": self.sb942_label,
            "created_at": self.created_at.isoformat(),
            "word_count": self.word_count,
            "character_count": self.character_count,
        }


@dataclass
class Template:
    """Content template for consistent messaging.

    Attributes:
        id: Template identifier
        name: Template name
        content_type: Type of content this template produces
        structure: Template structure with placeholders
        variables: Required variables for the template
        tone: Suggested tone (formal, casual, urgent)
        brand_guidelines: Brand-specific guidelines
    """
    id: str
    name: str
    content_type: ContentType
    structure: str
    variables: list[str] = field(default_factory=list)
    tone: str = "professional"
    brand_guidelines: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "content_type": self.content_type.value,
            "structure": self.structure,
            "variables": self.variables,
            "tone": self.tone,
            "brand_guidelines": self.brand_guidelines,
        }


class ContentDrafterChip(BaseSkillChip):
    """Draft communications, reports, and content with SB 942 AI labeling compliance.

    This chip assists nonprofit staff in creating communications across
    multiple channels while ensuring compliance with California's SB 942
    AI disclosure requirements.

    Intents handled:
        - draft_email: Draft email communications
        - draft_social: Create social media content
        - draft_newsletter: Draft newsletter content
        - draft_report: Generate report documents
        - content_review: Review and improve existing content

    Consensus actions:
        - publish_external: Requires approval for external publication
        - send_mass_email: Requires approval for mass email sends

    Example:
        chip = ContentDrafterChip()
        request = SkillRequest(
            intent="draft_social",
            entities={"platform": "twitter", "topic": "volunteer_appreciation"}
        )
        response = await chip.handle(request, context)
    """

    name = "content_drafter"
    description = "Draft communications, reports, and content with SB 942 AI labeling compliance"
    version = "1.0.0"
    domain = SkillDomain.COMMUNICATIONS

    efe_weights = EFEWeights(
        mission_alignment=0.25,
        stakeholder_benefit=0.30,
        resource_efficiency=0.15,
        transparency=0.20,
        equity=0.10,
    )

    capabilities = [
        SkillCapability.READ_DATA,
        SkillCapability.WRITE_DATA,
        SkillCapability.GENERATE_REPORTS,
    ]

    consensus_actions = ["publish_external", "send_mass_email"]
    required_spans = ["template_engine", "sb942_labeler", "social_media_api"]

    # SB 942 AI disclosure labels
    SB942_LABELS = {
        "standard": "This content was generated with the assistance of artificial intelligence.",
        "short": "AI-assisted content",
        "detailed": "This content was drafted using AI technology and reviewed by [Organization Name] staff. "
                   "In accordance with California SB 942, we disclose that artificial intelligence was used "
                   "in the creation of this communication.",
        "social": "#AIassisted",
    }

    # Platform character limits
    PLATFORM_LIMITS = {
        Platform.TWITTER: 280,
        Platform.FACEBOOK: 63206,
        Platform.INSTAGRAM: 2200,
        Platform.LINKEDIN: 3000,
    }

    SUPPORTED_INTENTS = {
        "draft_email": "_handle_draft_email",
        "draft_social": "_handle_draft_social",
        "draft_newsletter": "_handle_draft_newsletter",
        "draft_report": "_handle_draft_report",
        "content_review": "_handle_content_review",
    }

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Route request to appropriate handler based on intent.

        Args:
            request: The skill request with intent and entities
            context: Execution context with org, user, BDI state

        Returns:
            SkillResponse with drafted content
        """
        handler_name = self.SUPPORTED_INTENTS.get(request.intent)

        if handler_name is None:
            return SkillResponse(
                content=f"Unknown intent '{request.intent}' for content_drafter chip.",
                success=False,
                data={"supported_intents": list(self.SUPPORTED_INTENTS.keys())},
            )

        handler = getattr(self, handler_name)
        return await handler(request, context)

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract communications-relevant BDI context.

        Filters BDI state for beliefs about audience, brand, and
        communication goals.
        """
        comm_types = {"audience_segment", "brand_voice", "campaign_active", "content_calendar"}

        filtered_beliefs = [
            b for b in beliefs
            if b.get("type") in comm_types or b.get("domain") == "communications"
        ]

        filtered_desires = [
            d for d in desires
            if d.get("type") in {"engagement_goal", "awareness_goal", "conversion_goal"}
        ]

        return {
            "beliefs": filtered_beliefs,
            "desires": filtered_desires,
            "intentions": intentions,
        }

    async def _handle_draft_email(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Draft email communications.

        Supported entities:
            - recipient_type: Type of recipients (donors, volunteers, members)
            - topic: Email topic or campaign
            - tone: Desired tone (formal, friendly, urgent)
            - call_to_action: Desired CTA
            - template_id: Optional template to use
            - personalization: Personalization fields
        """
        entities = request.entities
        recipient_type = entities.get("recipient_type", "general")
        topic = entities.get("topic", "")
        tone = entities.get("tone", "professional")
        cta = entities.get("call_to_action", "")
        template_id = entities.get("template_id")

        if not topic:
            return SkillResponse(
                content="Please specify a topic for the email.",
                success=False,
            )

        # Generate draft
        draft = await self.draft_content(
            content_type=ContentType.EMAIL,
            topic=topic,
            tone=tone,
            audience=recipient_type,
            call_to_action=cta,
            template_id=template_id,
            org_id=context.org_id,
        )

        # Apply SB 942 label
        draft = await self.add_sb942_label(draft, label_type="standard")

        content = f"""**Draft Email: {draft.title}**

**To:** {recipient_type.title()} list
**Subject:** {draft.title}

---

{draft.body}

---

*{draft.sb942_label}*

---
Word Count: {draft.word_count}
"""

        return SkillResponse(
            content=content,
            success=True,
            data={"draft": draft.to_dict()},
            suggestions=[
                "Edit the draft?",
                "Change the tone?",
                "Schedule for sending?",
            ],
        )

    async def _handle_draft_social(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Create social media content.

        Supported entities:
            - platform: Target platform (twitter, facebook, instagram, linkedin)
            - topic: Content topic
            - tone: Desired tone
            - include_hashtags: Whether to include hashtags
            - link_url: URL to include
            - image_suggestion: Request image suggestions
        """
        entities = request.entities
        platform_str = entities.get("platform", "twitter")
        topic = entities.get("topic", "")
        include_hashtags = entities.get("include_hashtags", True)

        if not topic:
            return SkillResponse(
                content="Please specify a topic for the social media post.",
                success=False,
            )

        try:
            platform = Platform(platform_str.lower())
        except ValueError:
            platform = Platform.TWITTER

        # Generate draft
        draft = await self.draft_content(
            content_type=ContentType.SOCIAL_MEDIA,
            topic=topic,
            platform=platform,
            include_hashtags=include_hashtags,
            org_id=context.org_id,
        )

        # Format for platform
        formatted = await self.format_for_platform(draft, platform)

        # Check character limit
        limit = self.PLATFORM_LIMITS.get(platform, 5000)
        within_limit = formatted.character_count <= limit

        content = f"""**{platform.value.title()} Post Draft**

{formatted.body}

---
Characters: {formatted.character_count}/{limit} {'(OK)' if within_limit else '(OVER LIMIT)'}
{f'*{formatted.sb942_label}*' if formatted.sb942_label else ''}
"""

        suggestions = ["Edit post?", "Create variations?"]
        if not within_limit:
            suggestions.insert(0, "Shorten to fit character limit?")

        return SkillResponse(
            content=content,
            success=True,
            data={
                "draft": formatted.to_dict(),
                "within_limit": within_limit,
                "platform_limit": limit,
            },
            suggestions=suggestions,
        )

    async def _handle_draft_newsletter(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Draft newsletter content.

        Supported entities:
            - edition: Newsletter edition (monthly, weekly, special)
            - sections: Sections to include
            - highlights: Key highlights to feature
            - tone: Desired tone
            - template_id: Template to use
        """
        entities = request.entities
        edition = entities.get("edition", "monthly")
        sections = entities.get("sections", ["updates", "events", "impact", "cta"])
        highlights = entities.get("highlights", [])

        # Generate newsletter draft
        draft = await self.draft_content(
            content_type=ContentType.NEWSLETTER,
            edition=edition,
            sections=sections,
            highlights=highlights,
            org_id=context.org_id,
        )

        # Apply SB 942 label
        draft = await self.add_sb942_label(draft, label_type="detailed")

        content = f"""**Newsletter Draft: {edition.title()} Edition**

{draft.body}

---

*Disclosure: {draft.sb942_label}*

---
Word Count: {draft.word_count}
Sections: {', '.join(sections)}
"""

        return SkillResponse(
            content=content,
            success=True,
            data={"draft": draft.to_dict()},
            suggestions=["Review each section?", "Add more highlights?", "Preview formatted version?"],
        )

    async def _handle_draft_report(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Generate report documents.

        Supported entities:
            - report_type: Type of report (impact, annual, funder, board)
            - time_period: Period covered
            - audience: Target audience
            - sections: Sections to include
            - data_sources: Data sources to incorporate
        """
        entities = request.entities
        report_type = entities.get("report_type", "summary")
        time_period = entities.get("time_period", "this_quarter")
        audience = entities.get("audience", "general")

        # Generate report draft
        draft = await self.draft_content(
            content_type=ContentType.REPORT,
            report_type=report_type,
            time_period=time_period,
            audience=audience,
            org_id=context.org_id,
        )

        # Apply SB 942 label
        draft = await self.add_sb942_label(draft, label_type="detailed")

        content = f"""**Report Draft: {report_type.title()} Report**
**Period:** {time_period}
**Audience:** {audience}

---

{draft.body}

---

*{draft.sb942_label}*

---
Word Count: {draft.word_count}
"""

        return SkillResponse(
            content=content,
            success=True,
            data={"draft": draft.to_dict()},
            suggestions=["Add data visualizations?", "Export to PDF?", "Review with data?"],
        )

    async def _handle_content_review(
        self, request: SkillRequest, context: SkillContext
    ) -> SkillResponse:
        """Review and improve existing content.

        Supported entities:
            - content: Content to review
            - review_type: Type of review (grammar, tone, clarity, engagement)
            - target_audience: Intended audience
            - brand_voice: Brand voice guidelines to apply
        """
        entities = request.entities
        content_to_review = entities.get("content", "")
        review_type = entities.get("review_type", "general")

        if not content_to_review:
            return SkillResponse(
                content="Please provide content to review.",
                success=False,
            )

        # Analyze content
        analysis = await self._analyze_content(content_to_review, review_type)

        # Generate suggestions
        suggestions = analysis.get("suggestions", [])
        improved_version = analysis.get("improved_version", "")

        content_lines = [f"**Content Review ({review_type})**\n"]
        content_lines.append(f"**Original Word Count:** {len(content_to_review.split())}")
        content_lines.append(f"**Readability Score:** {analysis.get('readability_score', 'N/A')}\n")

        if suggestions:
            content_lines.append("**Suggestions:**")
            for s in suggestions:
                content_lines.append(f"- {s}")

        if improved_version:
            content_lines.append(f"\n**Suggested Revision:**\n{improved_version}")

        return SkillResponse(
            content="\n".join(content_lines),
            success=True,
            data={
                "analysis": analysis,
                "improved_version": improved_version,
            },
            suggestions=["Apply suggested changes?", "Review for different audience?"],
        )

    # Core implementation methods

    async def draft_content(
        self,
        content_type: ContentType,
        topic: str = "",
        tone: str = "professional",
        audience: str = "general",
        platform: Platform = Platform.EMAIL,
        call_to_action: str = "",
        template_id: str | None = None,
        include_hashtags: bool = False,
        org_id: str = "",
        **kwargs: Any,
    ) -> DraftedContent:
        """Draft content based on type and parameters.

        Args:
            content_type: Type of content to create
            topic: Content topic
            tone: Desired tone
            audience: Target audience
            platform: Target platform
            call_to_action: Desired CTA
            template_id: Template to use
            include_hashtags: Include hashtags for social
            org_id: Organization identifier
            **kwargs: Additional parameters

        Returns:
            DraftedContent object with generated content
        """
        # Get template if specified
        template = None
        if template_id:
            template = await self._get_template(template_id)

        # Generate content based on type
        if content_type == ContentType.EMAIL:
            body = await self._generate_email_content(topic, tone, audience, call_to_action, template)
            title = self._generate_subject_line(topic, tone)
        elif content_type == ContentType.SOCIAL_MEDIA:
            body = await self._generate_social_content(topic, platform, include_hashtags)
            title = f"{platform.value} post: {topic}"
        elif content_type == ContentType.NEWSLETTER:
            body = await self._generate_newsletter_content(kwargs.get("edition", "monthly"), kwargs.get("sections", []))
            title = f"Newsletter: {kwargs.get('edition', 'Monthly').title()} Edition"
        elif content_type == ContentType.REPORT:
            body = await self._generate_report_content(kwargs.get("report_type", "summary"), kwargs.get("time_period", ""))
            title = f"{kwargs.get('report_type', 'Summary').title()} Report"
        else:
            body = f"[Draft content for {topic}]"
            title = topic

        draft = DraftedContent(
            id=f"draft_{datetime.now(timezone.utc).timestamp()}",
            content_type=content_type,
            title=title,
            body=body,
            platform=platform,
            metadata={
                "topic": topic,
                "tone": tone,
                "audience": audience,
                "template_id": template_id,
            },
        )

        return draft

    async def apply_template(
        self,
        template_id: str,
        variables: dict[str, str],
    ) -> str:
        """Apply a template with provided variables.

        Args:
            template_id: Template to use
            variables: Variable values to substitute

        Returns:
            Rendered template content
        """
        template = await self._get_template(template_id)
        if not template:
            return ""

        content = template.structure
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            content = content.replace(placeholder, var_value)

        return content

    async def add_sb942_label(
        self,
        content: DraftedContent,
        label_type: str = "standard",
    ) -> DraftedContent:
        """Add SB 942 AI disclosure label to content.

        California SB 942 requires disclosure when content is generated
        or substantially modified by AI systems.

        Args:
            content: Content to label
            label_type: Type of label (standard, short, detailed, social)

        Returns:
            Content with SB 942 label added
        """
        label = self.SB942_LABELS.get(label_type, self.SB942_LABELS["standard"])
        content.sb942_label = label
        return content

    async def format_for_platform(
        self,
        content: DraftedContent,
        platform: Platform,
    ) -> DraftedContent:
        """Format content for specific platform requirements.

        Args:
            content: Content to format
            platform: Target platform

        Returns:
            Formatted content
        """
        body = content.body
        limit = self.PLATFORM_LIMITS.get(platform)

        # Add platform-specific formatting
        if platform == Platform.TWITTER:
            # Add AI disclosure hashtag for Twitter
            if content.character_count + len(" #AIassisted") <= 280:
                body = f"{body} #AIassisted"
            content.sb942_label = self.SB942_LABELS["social"]

        elif platform == Platform.INSTAGRAM:
            # Instagram typically puts hashtags at the end
            content.sb942_label = self.SB942_LABELS["social"]

        elif platform == Platform.LINKEDIN:
            # LinkedIn uses more professional disclosure
            content.sb942_label = self.SB942_LABELS["short"]

        # Truncate if needed (with warning)
        if limit and len(body) > limit:
            body = body[:limit - 3] + "..."

        content.body = body
        content.platform = platform

        return content

    # Private helper methods

    async def _generate_email_content(
        self,
        topic: str,
        tone: str,
        audience: str,
        cta: str,
        template: Template | None,
    ) -> str:
        """Generate email body content."""
        # Simulated email content generation
        greeting = "Dear Friend," if tone == "friendly" else "Dear Supporter,"
        closing = "Warmly," if tone == "friendly" else "Sincerely,"

        return f"""{greeting}

Thank you for your continued support of our mission. We're reaching out today to share an important update about {topic}.

[Insert main message content here based on topic: {topic}]

Your support makes this work possible. Together, we are creating meaningful change in our community.

{f"[Call to Action: {cta}]" if cta else "[Insert call to action here]"}

{closing}
[Organization Name]
"""

    async def _generate_social_content(
        self,
        topic: str,
        platform: Platform,
        include_hashtags: bool,
    ) -> str:
        """Generate social media content."""
        base_content = f"We're excited to share {topic}! Your support helps us create lasting impact in our community."

        if include_hashtags:
            hashtags = "\n\n#Nonprofit #Community #Impact #MakingADifference"
            return base_content + hashtags

        return base_content

    async def _generate_newsletter_content(
        self,
        edition: str,
        sections: list[str],
    ) -> str:
        """Generate newsletter content."""
        content_parts = [f"# {edition.title()} Newsletter\n"]

        for section in sections:
            if section == "updates":
                content_parts.append("## Organization Updates\n[Insert recent news and updates]\n")
            elif section == "events":
                content_parts.append("## Upcoming Events\n[Insert event listings]\n")
            elif section == "impact":
                content_parts.append("## Impact Spotlight\n[Insert impact story or metrics]\n")
            elif section == "cta":
                content_parts.append("## How You Can Help\n[Insert call to action]\n")

        return "\n".join(content_parts)

    async def _generate_report_content(
        self,
        report_type: str,
        time_period: str,
    ) -> str:
        """Generate report content."""
        return f"""# {report_type.title()} Report
**Period:** {time_period}

## Executive Summary
[Insert executive summary]

## Key Achievements
- [Achievement 1]
- [Achievement 2]
- [Achievement 3]

## Program Highlights
[Insert program highlights]

## Financial Summary
[Insert financial summary]

## Looking Ahead
[Insert future plans]

## Acknowledgments
[Insert acknowledgments]
"""

    async def _get_template(self, template_id: str) -> Template | None:
        """Fetch a template by ID."""
        # Simulated templates
        templates = {
            "email_appeal": Template(
                id="email_appeal",
                name="Donation Appeal Email",
                content_type=ContentType.EMAIL,
                structure="Dear {{donor_name}},\n\n{{appeal_message}}\n\n{{call_to_action}}\n\nThank you,\n{{signature}}",
                variables=["donor_name", "appeal_message", "call_to_action", "signature"],
                tone="professional",
            ),
            "thank_you": Template(
                id="thank_you",
                name="Thank You Email",
                content_type=ContentType.THANK_YOU,
                structure="Dear {{name}},\n\nThank you for your generous support of {{amount}}. {{impact_message}}\n\nWith gratitude,\n{{signature}}",
                variables=["name", "amount", "impact_message", "signature"],
                tone="warm",
            ),
        }
        return templates.get(template_id)

    async def _analyze_content(
        self,
        content: str,
        review_type: str,
    ) -> dict[str, Any]:
        """Analyze content and generate improvement suggestions."""
        word_count = len(content.split())
        sentences = content.count(".") + content.count("!") + content.count("?")
        avg_sentence_length = word_count / max(sentences, 1)

        # Simple readability estimation
        readability = "Good" if avg_sentence_length < 20 else "Consider shorter sentences"

        suggestions = []
        if avg_sentence_length > 25:
            suggestions.append("Consider breaking up longer sentences for better readability")
        if "!" not in content and review_type == "engagement":
            suggestions.append("Consider adding enthusiasm or urgency")
        if word_count < 50:
            suggestions.append("Content may be too brief - consider adding more detail")
        if word_count > 500 and review_type == "engagement":
            suggestions.append("Consider condensing for better engagement")

        return {
            "word_count": word_count,
            "sentence_count": sentences,
            "avg_sentence_length": avg_sentence_length,
            "readability_score": readability,
            "suggestions": suggestions,
            "improved_version": "",  # Would contain AI-improved version in production
        }
