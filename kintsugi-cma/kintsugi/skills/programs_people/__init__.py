"""
Kintsugi CMA Programs & People Skill Chips.

This package contains skill chips for program management, governance,
donor relations, staff operations, events, and member services.

Phase 4b Chips:
- ProgramEvaluatorChip: Logic models, outcomes, and evaluation design
- BoardLiaisonChip: Board governance, meetings, and compliance
- DonorStewardshipChip: Donor relationships and cultivation
- StaffOnboardingChip: New staff onboarding and training
- EventPlannerChip: Event planning, RSVPs, and logistics
- MemberServicesChip: Membership tracking and communications

Usage:
    from kintsugi.skills.programs_people import (
        ProgramEvaluatorChip,
        BoardLiaisonChip,
        DonorStewardshipChip,
        StaffOnboardingChip,
        EventPlannerChip,
        MemberServicesChip,
    )

    # Register chips
    from kintsugi.skills import register_chip
    register_chip(ProgramEvaluatorChip())
    register_chip(BoardLiaisonChip())
    # ...
"""

from .program_evaluator import ProgramEvaluatorChip
from .board_liaison import BoardLiaisonChip
from .donor_stewardship import DonorStewardshipChip
from .staff_onboarding import StaffOnboardingChip
from .event_planner import EventPlannerChip
from .member_services import MemberServicesChip

__all__ = [
    "ProgramEvaluatorChip",
    "BoardLiaisonChip",
    "DonorStewardshipChip",
    "StaffOnboardingChip",
    "EventPlannerChip",
    "MemberServicesChip",
]
