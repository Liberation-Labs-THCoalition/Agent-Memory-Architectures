"""
Kintsugi CMA Core Operations Skill Chips.

This package contains the Phase 4a core operations skill chips for the
Kintsugi Cognitive Memory Architecture. These chips provide essential
capabilities for nonprofit operations across six key domains.

Chips included:
- GrantHunterChip: Search and match grant opportunities
- VolunteerCoordinatorChip: Coordinate volunteer scheduling and engagement
- ImpactAuditorChip: Track and report organizational impact
- FinanceAssistantChip: Financial management and budget tracking
- InstitutionalMemoryChip: Query and maintain organizational knowledge
- ContentDrafterChip: Draft communications with SB 942 compliance

Usage:
    from kintsugi.skills.core_ops import (
        GrantHunterChip,
        VolunteerCoordinatorChip,
        ImpactAuditorChip,
        FinanceAssistantChip,
        InstitutionalMemoryChip,
        ContentDrafterChip,
    )

    # Register all core operations chips
    from kintsugi.skills.core_ops import register_all_core_ops_chips
    register_all_core_ops_chips()

    # Or register individual chips
    from kintsugi.skills import register_chip
    register_chip(GrantHunterChip())
"""

from .content_drafter import (
    ContentDrafterChip,
    ContentStatus,
    ContentType,
    DraftedContent,
    Platform,
    Template,
)
from .finance_assistant import (
    BudgetCategory,
    BudgetLine,
    FinanceAssistantChip,
    Invoice,
    PaymentStatus,
    Transaction,
    TransactionType,
)
from .grant_hunter import (
    GrantHunterChip,
    GrantOpportunity,
)
from .impact_auditor import (
    ImpactAuditorChip,
    Indicator,
    IndicatorType,
    Measurement,
    SDGGoal,
)
from .institutional_memory import (
    InstitutionalMemoryChip,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    SearchResult,
)
from .volunteer_coordinator import (
    Shift,
    ShiftStatus,
    Volunteer,
    VolunteerCoordinatorChip,
    VolunteerStatus,
)

__all__ = [
    # Chips
    "GrantHunterChip",
    "VolunteerCoordinatorChip",
    "ImpactAuditorChip",
    "FinanceAssistantChip",
    "InstitutionalMemoryChip",
    "ContentDrafterChip",
    # Grant Hunter types
    "GrantOpportunity",
    # Volunteer Coordinator types
    "Volunteer",
    "VolunteerStatus",
    "Shift",
    "ShiftStatus",
    # Impact Auditor types
    "SDGGoal",
    "IndicatorType",
    "Indicator",
    "Measurement",
    # Finance Assistant types
    "BudgetCategory",
    "TransactionType",
    "PaymentStatus",
    "BudgetLine",
    "Transaction",
    "Invoice",
    # Institutional Memory types
    "MemoryType",
    "MemoryStatus",
    "MemoryRecord",
    "SearchResult",
    # Content Drafter types
    "ContentType",
    "Platform",
    "ContentStatus",
    "DraftedContent",
    "Template",
    # Utility functions
    "register_all_core_ops_chips",
    "get_all_core_ops_chips",
]


def get_all_core_ops_chips() -> list:
    """Get instances of all core operations chips.

    Returns:
        List of instantiated core operations chip instances.

    Example:
        chips = get_all_core_ops_chips()
        for chip in chips:
            print(f"{chip.name}: {chip.description}")
    """
    return [
        GrantHunterChip(),
        VolunteerCoordinatorChip(),
        ImpactAuditorChip(),
        FinanceAssistantChip(),
        InstitutionalMemoryChip(),
        ContentDrafterChip(),
    ]


def register_all_core_ops_chips() -> list[str]:
    """Register all core operations chips with the global registry.

    Registers all six core operations skill chips with the global
    SkillRegistry for use by the Orchestrator and router.

    Returns:
        List of registered chip names.

    Raises:
        ValueError: If any chip is already registered.

    Example:
        from kintsugi.skills.core_ops import register_all_core_ops_chips

        registered = register_all_core_ops_chips()
        print(f"Registered chips: {', '.join(registered)}")
    """
    from kintsugi.skills import register_chip

    chips = get_all_core_ops_chips()
    registered_names = []

    for chip in chips:
        register_chip(chip)
        registered_names.append(chip.name)

    return registered_names
