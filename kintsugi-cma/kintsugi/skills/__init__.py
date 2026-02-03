"""
Kintsugi CMA Skill Chips Infrastructure.

This package provides the foundational infrastructure for all 22 skill chips
in the Kintsugi Cognitive Memory Architecture. Skill chips are modular,
domain-specific handlers that process user intents within ethical guardrails.

Core Components:
- BaseSkillChip: Abstract base class for all skill chip implementations
- SkillRegistry: Central catalog for chip discovery and lookup
- SkillRouter: Intent routing to appropriate chips

Key Types:
- SkillDomain: Enum of operational domains (fundraising, operations, etc.)
- SkillContext: Runtime context passed to handlers
- SkillRequest: Encapsulated user intent and entities
- SkillResponse: Handler response with content and metadata
- EFEWeights: Ethical Framing Engine weights for domain prioritization
- SkillCapability: Declared chip capabilities for security/auditing

Usage:
    from kintsugi.skills import (
        BaseSkillChip,
        SkillDomain,
        SkillRequest,
        SkillResponse,
        SkillContext,
        EFEWeights,
        SkillCapability,
        get_registry,
        register_chip,
        SkillRouter,
        RouterConfig,
    )

    # Define a custom skill chip
    class MySkillChip(BaseSkillChip):
        name = "my_skill"
        description = "Does something useful"
        domain = SkillDomain.OPERATIONS

        async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
            return SkillResponse(content="Done!")

    # Register the chip
    register_chip(MySkillChip())

    # Set up routing
    registry = get_registry()
    router = SkillRouter(registry)
    router.register_intent("my_action", "my_skill")

    # Route and handle
    match = router.route("my_action")
    if match:
        response = await match.chip.handle(request, context)
"""

# Base classes and types
from .base import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillHandler,
    SkillRequest,
    SkillResponse,
)

# Registry
from .registry import (
    SkillRegistry,
    get_registry,
    register_chip,
    reset_registry,
)

# Router
from .router import (
    RouteMatch,
    RouterConfig,
    SkillRouter,
    create_router,
)

__all__ = [
    # Base classes and types
    "BaseSkillChip",
    "EFEWeights",
    "SkillCapability",
    "SkillContext",
    "SkillDomain",
    "SkillHandler",
    "SkillRequest",
    "SkillResponse",
    # Registry
    "SkillRegistry",
    "get_registry",
    "register_chip",
    "reset_registry",
    # Router
    "RouteMatch",
    "RouterConfig",
    "SkillRouter",
    "create_router",
]
