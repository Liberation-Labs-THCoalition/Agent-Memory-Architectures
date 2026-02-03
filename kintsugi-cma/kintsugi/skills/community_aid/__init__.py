"""
Kintsugi CMA Community & Mutual Aid Skill Chips.

This package contains Phase 4c skill chips for community resource coordination,
mutual aid networks, crisis response, coalition building, and solidarity economy.

Phase 4c Chips (10 total):
- MutualAidCoordinatorChip: Coordinate mutual aid requests and volunteer matching
- CommunityAssetMapperChip: Map community resources, skills, and spaces
- ResourceRedistributionChip: Manage resource sharing and redistribution
- CrisisResponseChip: Coordinate emergency and crisis response
- CoalitionBuilderChip: Build and manage organizational coalitions
- KnowYourRightsChip: Legal information, rights education, and clinic scheduling
- HousingNavigatorChip: Housing resources, voucher programs, and tenant rights
- FoodAccessChip: Food pantries, SNAP assistance, and community meals
- SolidarityEconomyChip: Cooperative development, time banking, CDFIs
- RapidResponseChip: ICE raids, bail funds, and emergency coordination

Usage:
    from kintsugi.skills.community_aid import (
        MutualAidCoordinatorChip,
        CommunityAssetMapperChip,
        ResourceRedistributionChip,
        CrisisResponseChip,
        CoalitionBuilderChip,
        KnowYourRightsChip,
        HousingNavigatorChip,
        FoodAccessChip,
        SolidarityEconomyChip,
        RapidResponseChip,
    )

    # Register chips
    from kintsugi.skills import register_chip
    register_chip(MutualAidCoordinatorChip())
    register_chip(CommunityAssetMapperChip())
    # ...
"""

# First 5 chips (Phase 4c Part 1)
from .mutual_aid_coordinator import MutualAidCoordinatorChip
from .community_asset_mapper import CommunityAssetMapperChip
from .resource_redistribution import ResourceRedistributionChip
from .crisis_response import CrisisResponseChip
from .coalition_builder import CoalitionBuilderChip

# Last 5 chips (Phase 4c Part 2)
from .know_your_rights import KnowYourRightsChip
from .housing_navigator import HousingNavigatorChip
from .food_access import FoodAccessChip
from .solidarity_economy import SolidarityEconomyChip
from .rapid_response import RapidResponseChip

__all__ = [
    # First 5 chips
    "MutualAidCoordinatorChip",
    "CommunityAssetMapperChip",
    "ResourceRedistributionChip",
    "CrisisResponseChip",
    "CoalitionBuilderChip",
    # Last 5 chips
    "KnowYourRightsChip",
    "HousingNavigatorChip",
    "FoodAccessChip",
    "SolidarityEconomyChip",
    "RapidResponseChip",
]
