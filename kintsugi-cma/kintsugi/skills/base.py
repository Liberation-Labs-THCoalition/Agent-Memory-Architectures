"""
Base skill chip abstractions for Kintsugi CMA.

This module provides the foundational classes and types for all 22 skill chips
in the Kintsugi CMA system. Skill chips are modular, domain-specific handlers
that process user intents and execute actions within ethical guardrails.

Key concepts:
- SkillDomain: The functional area a chip operates in (fundraising, operations, etc.)
- EFEWeights: Ethical Framing Engine weights for domain-specific prioritization
- SkillContext: Runtime context including organization, user, and BDI state
- BaseSkillChip: Abstract base class all skill chips inherit from
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine


class SkillDomain(str, Enum):
    """Domains that skill chips operate in.

    Each domain represents a functional area of nonprofit operations.
    Skill chips declare their domain to enable domain-based routing
    and EFE weight customization.

    Attributes:
        FUNDRAISING: Grant writing, donor management, campaigns
        OPERATIONS: Day-to-day organizational operations
        PROGRAMS: Program delivery and management
        COMMUNICATIONS: External and internal communications
        FINANCE: Financial management and reporting
        GOVERNANCE: Board management, compliance, policies
        COMMUNITY: Community engagement and relationships
        MUTUAL_AID: Mutual aid coordination and distribution
        ADVOCACY: Policy advocacy and campaigns
        MEMBER_SERVICES: Member support and services
    """
    FUNDRAISING = "fundraising"
    OPERATIONS = "operations"
    PROGRAMS = "programs"
    COMMUNICATIONS = "communications"
    FINANCE = "finance"
    GOVERNANCE = "governance"
    COMMUNITY = "community"
    MUTUAL_AID = "mutual_aid"
    ADVOCACY = "advocacy"
    MEMBER_SERVICES = "member_services"


@dataclass
class EFEWeights:
    """Ethical Framing Engine weights for domain-specific prioritization.

    The EFE uses these weights to evaluate actions and decisions within
    ethical guardrails. Each skill chip can customize weights based on
    its domain's priorities.

    Each weight must be between 0.0 and 1.0. Weights should sum to ~1.0
    for proper normalization in scoring algorithms.

    Attributes:
        mission_alignment: How well an action aligns with org mission
        stakeholder_benefit: Benefit to members, community, beneficiaries
        resource_efficiency: Efficient use of limited nonprofit resources
        transparency: Openness and accountability in operations
        equity: Fair and equitable treatment across populations

    Example:
        # Fundraising chip might prioritize mission and stakeholder benefit
        weights = EFEWeights(
            mission_alignment=0.30,
            stakeholder_benefit=0.30,
            resource_efficiency=0.15,
            transparency=0.15,
            equity=0.10,
        )
    """
    mission_alignment: float = 0.25
    stakeholder_benefit: float = 0.25
    resource_efficiency: float = 0.20
    transparency: float = 0.15
    equity: float = 0.15

    def __post_init__(self) -> None:
        """Validate that all weights are in valid range."""
        weight_names = [
            'mission_alignment',
            'stakeholder_benefit',
            'resource_efficiency',
            'transparency',
            'equity',
        ]
        for name in weight_names:
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {val}")

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary for serialization.

        Returns:
            Dictionary mapping weight names to their float values.
        """
        return {
            'mission_alignment': self.mission_alignment,
            'stakeholder_benefit': self.stakeholder_benefit,
            'resource_efficiency': self.resource_efficiency,
            'transparency': self.transparency,
            'equity': self.equity,
        }

    def total(self) -> float:
        """Calculate total of all weights.

        Returns:
            Sum of all weight values. Should be ~1.0 for normalized weights.
        """
        return (
            self.mission_alignment +
            self.stakeholder_benefit +
            self.resource_efficiency +
            self.transparency +
            self.equity
        )


@dataclass
class SkillContext:
    """Context passed to skill chip handlers.

    Contains all runtime information a skill chip needs to execute,
    including organization identity, user identity, platform details,
    and the current BDI (Belief-Desire-Intention) state from the
    cognitive architecture.

    Attributes:
        org_id: Unique identifier for the organization
        user_id: Unique identifier for the requesting user
        session_id: Optional session identifier for conversation tracking
        platform: Source platform (slack, discord, webchat, etc.)
        channel_id: Platform-specific channel identifier
        thread_id: Platform-specific thread identifier
        metadata: Additional context data
        timestamp: When the request was created (UTC)
        beliefs: Current beliefs from BDI state
        desires: Current desires/goals from BDI state
        intentions: Current intentions/plans from BDI state

    Example:
        context = SkillContext(
            org_id="org_12345",
            user_id="user_67890",
            platform="slack",
            channel_id="C0123456789",
            beliefs=[{"type": "budget_status", "value": "healthy"}],
        )
    """
    org_id: str
    user_id: str
    session_id: str | None = None
    platform: str | None = None  # slack, discord, webchat
    channel_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # BDI context (populated by orchestrator)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    desires: list[dict[str, Any]] = field(default_factory=list)
    intentions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillRequest:
    """Request to a skill chip.

    Encapsulates what the user wants to do, extracted entities,
    and any additional parameters needed for execution.

    Attributes:
        intent: Classified intent (e.g., "grant_search", "donor_lookup")
        entities: Extracted entities from user input (names, dates, amounts, etc.)
        raw_input: Original unprocessed user input
        parameters: Additional parameters for the skill handler

    Example:
        request = SkillRequest(
            intent="grant_search",
            entities={"amount_min": 10000, "focus_area": "education"},
            raw_input="Find grants over $10k for education programs",
        )
    """
    intent: str  # What the user wants to do
    entities: dict[str, Any] = field(default_factory=dict)  # Extracted entities
    raw_input: str = ""  # Original user input
    parameters: dict[str, Any] = field(default_factory=dict)  # Additional params


@dataclass
class SkillResponse:
    """Response from a skill chip.

    Contains the result of skill execution including the content to
    display, structured data, and metadata about the response.

    Attributes:
        content: Human-readable response content
        success: Whether the skill executed successfully
        data: Structured response data for programmatic use
        suggestions: Follow-up actions or questions to suggest
        requires_consensus: Whether this action needs approval before execution
        consensus_action: Which specific action needs approval
        attachments: File attachments or rich media
        metadata: Additional response metadata

    Example:
        response = SkillResponse(
            content="Found 5 matching grants for your search.",
            success=True,
            data={"grants": [...], "total": 5},
            suggestions=["Would you like me to draft an application?"],
        )
    """
    content: str
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)  # Structured response data
    suggestions: list[str] = field(default_factory=list)  # Follow-up suggestions
    requires_consensus: bool = False  # Needs approval before execution
    consensus_action: str | None = None  # Which action needs approval
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillCapability(str, Enum):
    """Standard capabilities that chips can declare.

    Capabilities are used for security, auditing, and feature gating.
    Each chip declares which capabilities it uses, enabling:
    - Permission checking before execution
    - Audit logging of sensitive operations
    - Feature flags and gradual rollouts

    Attributes:
        READ_DATA: Read organizational data from storage
        WRITE_DATA: Write/modify organizational data
        EXTERNAL_API: Make external API calls
        SEND_NOTIFICATIONS: Send emails, SMS, or push notifications
        FINANCIAL_OPERATIONS: Process financial transactions
        PII_ACCESS: Access personally identifiable information
        SCHEDULE_TASKS: Schedule future automated tasks
        GENERATE_REPORTS: Generate reports and documents
    """
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    EXTERNAL_API = "external_api"
    SEND_NOTIFICATIONS = "send_notifications"
    FINANCIAL_OPERATIONS = "financial_operations"
    PII_ACCESS = "pii_access"
    SCHEDULE_TASKS = "schedule_tasks"
    GENERATE_REPORTS = "generate_reports"


class BaseSkillChip(ABC):
    """Abstract base class for all skill chips.

    Skill chips are modular handlers for specific domains of nonprofit
    operations. Each chip:
    - Handles specific intents routed by the Orchestrator
    - Operates within ethical guardrails defined by EFE weights
    - Declares required MCP tool spans for execution
    - Specifies which actions require consensus approval

    Subclasses must implement the `handle` method and should override
    class-level attributes to define chip identity and behavior.

    Class Attributes:
        name: Unique identifier for the chip
        description: Human-readable description
        version: Semantic version string
        domain: Primary skill domain
        efe_weights: Ethical Framing Engine weights
        required_spans: MCP tool spans this chip needs
        consensus_actions: Actions requiring approval
        capabilities: Declared capabilities

    Example:
        class GrantSearchChip(BaseSkillChip):
            name = "grant_search"
            description = "Search and match grant opportunities"
            version = "1.0.0"
            domain = SkillDomain.FUNDRAISING

            efe_weights = EFEWeights(
                mission_alignment=0.30,
                stakeholder_benefit=0.25,
                resource_efficiency=0.20,
                transparency=0.15,
                equity=0.10,
            )

            required_spans = ["grant_database", "org_profile"]
            capabilities = [SkillCapability.READ_DATA, SkillCapability.EXTERNAL_API]

            async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
                # Implementation here
                ...
    """

    # Class-level attributes (override in subclasses)
    name: str = "base_chip"
    description: str = "Base skill chip"
    version: str = "1.0.0"
    domain: SkillDomain = SkillDomain.OPERATIONS

    # Default EFE weights (override per domain)
    efe_weights: EFEWeights | None = None

    # MCP tool spans this chip needs
    required_spans: list[str] = []

    # Actions that require consensus approval
    consensus_actions: list[str] = []

    # Capabilities this chip uses
    capabilities: list[SkillCapability] = []

    def __init__(self) -> None:
        """Initialize the skill chip.

        Sets default values for instance-level attributes if not
        already defined at the class level.
        """
        # Initialize instance-level if not set at class level
        if self.efe_weights is None:
            self.efe_weights = EFEWeights()

        # Ensure lists are instance-level to avoid shared state
        if not hasattr(self.__class__, 'required_spans') or self.required_spans is None:
            self.required_spans = []
        if not hasattr(self.__class__, 'consensus_actions') or self.consensus_actions is None:
            self.consensus_actions = []
        if not hasattr(self.__class__, 'capabilities') or self.capabilities is None:
            self.capabilities = []

    @abstractmethod
    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        """Main execution method - receives routed request from Orchestrator.

        This is the primary entry point for skill chip execution. The
        Orchestrator routes classified intents to the appropriate chip
        and calls this method with the request and context.

        Args:
            request: The skill request with intent and entities
            context: Execution context including org, user, BDI state

        Returns:
            SkillResponse with content and metadata

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        ...

    async def get_bdi_context(
        self,
        beliefs: list[dict[str, Any]],
        desires: list[dict[str, Any]],
        intentions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract relevant BDI sections for this chip's domain.

        Override in subclasses to filter BDI state relevant to the
        chip's domain. This allows chips to focus on the beliefs,
        desires, and intentions that matter for their operations.

        Default implementation returns all BDI state unfiltered.

        Args:
            beliefs: Current belief state from cognitive architecture
            desires: Current desires/goals
            intentions: Current intentions/plans

        Returns:
            Dictionary containing filtered BDI state for this chip's domain

        Example:
            # In a FundraisingChip subclass:
            async def get_bdi_context(self, beliefs, desires, intentions):
                return {
                    'beliefs': [b for b in beliefs if b.get('domain') == 'fundraising'],
                    'desires': [d for d in desires if d.get('type') == 'funding_goal'],
                    'intentions': intentions,  # Keep all intentions
                }
        """
        return {
            'beliefs': beliefs,
            'desires': desires,
            'intentions': intentions,
        }

    def requires_consensus(self, action: str) -> bool:
        """Check if an action requires consensus approval.

        Consensus actions are those that need human approval before
        execution, such as financial disbursements or policy changes.

        Args:
            action: The action name to check

        Returns:
            True if the action is in the consensus_actions list
        """
        return action in self.consensus_actions

    def get_info(self) -> dict[str, Any]:
        """Get chip metadata for registration/discovery.

        Returns a dictionary containing all metadata about this chip,
        useful for registration in the SkillRegistry and for UI display.

        Returns:
            Dictionary with chip metadata including name, description,
            version, domain, EFE weights, required spans, consensus
            actions, and capabilities.
        """
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'domain': self.domain.value,
            'efe_weights': self.efe_weights.to_dict() if self.efe_weights else {},
            'required_spans': list(self.required_spans),
            'consensus_actions': list(self.consensus_actions),
            'capabilities': [c.value for c in self.capabilities],
        }


# Type alias for skill handler functions
SkillHandler = Callable[[SkillRequest, SkillContext], Coroutine[Any, Any, SkillResponse]]
"""Type alias for async skill handler functions.

A skill handler is any async function that takes a SkillRequest and
SkillContext and returns a SkillResponse. This type is useful for
functional-style skill implementations or middleware.

Example:
    async def my_handler(request: SkillRequest, context: SkillContext) -> SkillResponse:
        return SkillResponse(content="Hello!")

    handler: SkillHandler = my_handler
"""
