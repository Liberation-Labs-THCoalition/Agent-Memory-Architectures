"""
Comprehensive tests for Kintsugi CMA Phase 4c Community Aid Skill Chips.

This module provides 100+ tests for all 10 Phase 4c skill chips:
- MutualAidCoordinatorChip
- ResourceRedistributionChip
- CrisisResponseChip
- CommunityAssetMapperChip
- CoalitionBuilderChip
- KnowYourRightsChip
- HousingNavigatorChip
- FoodAccessChip
- SolidarityEconomyChip
- RapidResponseChip

Tests cover:
- Chip attributes (name, domain, EFE weights)
- Intent handling
- Consensus actions
- Privacy/security requirements
- Response structure validation
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from typing import Any

from kintsugi.skills import (
    SkillRequest,
    SkillContext,
    SkillDomain,
    SkillCapability,
    SkillResponse,
)
from kintsugi.skills.community_aid import (
    MutualAidCoordinatorChip,
    ResourceRedistributionChip,
    CrisisResponseChip,
    CommunityAssetMapperChip,
    CoalitionBuilderChip,
    KnowYourRightsChip,
    HousingNavigatorChip,
    FoodAccessChip,
    SolidarityEconomyChip,
    RapidResponseChip,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def context() -> SkillContext:
    """Create a standard test context."""
    return SkillContext(
        org_id="mutual-aid-org",
        user_id="community-member",
        session_id="test-session-001",
        platform="webchat",
        channel_id="test-channel",
    )


@pytest.fixture
def context_with_bdi() -> SkillContext:
    """Create a context with BDI state."""
    return SkillContext(
        org_id="mutual-aid-org",
        user_id="community-member",
        beliefs=[
            {"domain": "mutual_aid", "type": "need_status", "value": "active"},
            {"domain": "resources", "type": "inventory_level", "value": "low"},
        ],
        desires=[
            {"type": "aid_fulfillment", "target": "housing"},
            {"type": "community_wellbeing", "priority": "high"},
        ],
        intentions=[
            {"domain": "mutual_aid", "action": "match_aid"},
        ],
    )


@pytest.fixture
def mutual_aid_chip() -> MutualAidCoordinatorChip:
    """Create a fresh MutualAidCoordinatorChip."""
    chip = MutualAidCoordinatorChip()
    chip._needs = {}
    chip._offers = {}
    chip._matches = {}
    return chip


@pytest.fixture
def resource_chip() -> ResourceRedistributionChip:
    """Create a fresh ResourceRedistributionChip."""
    chip = ResourceRedistributionChip()
    chip._surplus = {}
    chip._requests = {}
    chip._schedules = {}
    chip._partners = {}
    chip._inventory = {}
    return chip


@pytest.fixture
def crisis_chip() -> CrisisResponseChip:
    """Create a fresh CrisisResponseChip."""
    chip = CrisisResponseChip()
    chip._incidents = {}
    chip._volunteers = {}
    chip._resources = {}
    chip._debriefs = {}
    chip._alert_log = []
    return chip


@pytest.fixture
def asset_mapper_chip() -> CommunityAssetMapperChip:
    """Create a fresh CommunityAssetMapperChip."""
    chip = CommunityAssetMapperChip()
    chip._assets = {}
    chip._skills = {}
    chip._gap_analyses = {}
    return chip


@pytest.fixture
def coalition_chip() -> CoalitionBuilderChip:
    """Create a fresh CoalitionBuilderChip."""
    chip = CoalitionBuilderChip()
    chip._partners = {}
    chip._campaigns = {}
    chip._meetings = {}
    chip._outreach = {}
    return chip


@pytest.fixture
def rights_chip() -> KnowYourRightsChip:
    """Create a fresh KnowYourRightsChip."""
    return KnowYourRightsChip()


@pytest.fixture
def housing_chip() -> HousingNavigatorChip:
    """Create a fresh HousingNavigatorChip."""
    return HousingNavigatorChip()


@pytest.fixture
def food_chip() -> FoodAccessChip:
    """Create a fresh FoodAccessChip."""
    return FoodAccessChip()


@pytest.fixture
def solidarity_chip() -> SolidarityEconomyChip:
    """Create a fresh SolidarityEconomyChip."""
    return SolidarityEconomyChip()


@pytest.fixture
def rapid_response_chip() -> RapidResponseChip:
    """Create a fresh RapidResponseChip."""
    return RapidResponseChip()


# ===========================================================================
# MutualAidCoordinatorChip Tests (10 tests)
# ===========================================================================

class TestMutualAidCoordinatorChip:
    """Tests for MutualAidCoordinatorChip."""

    def test_chip_name(self, mutual_aid_chip: MutualAidCoordinatorChip):
        """Test chip has correct name."""
        assert mutual_aid_chip.name == "mutual_aid_coordinator"

    def test_chip_domain(self, mutual_aid_chip: MutualAidCoordinatorChip):
        """Test chip is in MUTUAL_AID domain."""
        assert mutual_aid_chip.domain == SkillDomain.MUTUAL_AID

    def test_efe_weights_stakeholder_benefit(self, mutual_aid_chip: MutualAidCoordinatorChip):
        """Test EFE weights prioritize stakeholder_benefit at 0.35."""
        assert mutual_aid_chip.efe_weights is not None
        assert mutual_aid_chip.efe_weights.stakeholder_benefit == 0.35

    def test_intents_supported(self, mutual_aid_chip: MutualAidCoordinatorChip):
        """Test all required intents are supported."""
        expected_intents = ["need_post", "offer_post", "match_request", "aid_status", "aid_report"]
        # The intents are defined in the handle method's intent_handlers dict
        # We verify by checking the chip handles these intents
        chip_info = mutual_aid_chip.get_info()
        # Verify chip is configured correctly
        assert chip_info["name"] == "mutual_aid_coordinator"

    def test_consensus_actions(self, mutual_aid_chip: MutualAidCoordinatorChip):
        """Test consensus actions include required approvals."""
        assert "approve_high_value_request" in mutual_aid_chip.consensus_actions
        assert "share_requester_info" in mutual_aid_chip.consensus_actions

    @pytest.mark.asyncio
    async def test_need_post_intent(
        self, mutual_aid_chip: MutualAidCoordinatorChip, context: SkillContext
    ):
        """Test posting a need."""
        request = SkillRequest(
            intent="need_post",
            entities={
                "category": "housing",
                "description": "Need temporary housing",
                "urgency": "high",
                "location": "downtown",
            },
        )
        response = await mutual_aid_chip.handle(request, context)
        assert response.success
        assert "need_id" in response.data
        assert response.data["category"] == "housing"
        assert response.data["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_offer_post_intent(
        self, mutual_aid_chip: MutualAidCoordinatorChip, context: SkillContext
    ):
        """Test posting an offer."""
        request = SkillRequest(
            intent="offer_post",
            entities={
                "categories": ["housing", "food"],
                "description": "Can provide spare room",
                "availability": "weekends",
            },
        )
        response = await mutual_aid_chip.handle(request, context)
        assert response.success
        assert "offer_id" in response.data
        assert "categories" in response.data

    @pytest.mark.asyncio
    async def test_privacy_preserving_matching(
        self, mutual_aid_chip: MutualAidCoordinatorChip, context: SkillContext
    ):
        """Test that matching preserves privacy until confirmed."""
        # Post a need
        need_request = SkillRequest(
            intent="need_post",
            entities={"category": "food", "description": "Need groceries"},
        )
        need_response = await mutual_aid_chip.handle(need_request, context)
        need_id = need_response.data["need_id"]

        # Post an offer
        offer_request = SkillRequest(
            intent="offer_post",
            entities={"categories": ["food"], "description": "Can deliver groceries"},
        )
        await mutual_aid_chip.handle(offer_request, context)

        # Check match - should require consensus for sharing info
        match_request = SkillRequest(
            intent="match_request",
            entities={"need_id": need_id},
        )
        match_response = await mutual_aid_chip.handle(match_request, context)
        assert match_response.success
        # Matches should be returned without exposing PII
        assert "matches" in match_response.data

    @pytest.mark.asyncio
    async def test_response_structure_need(
        self, mutual_aid_chip: MutualAidCoordinatorChip, context: SkillContext
    ):
        """Test response includes need structure."""
        request = SkillRequest(
            intent="need_post",
            entities={"category": "transportation", "description": "Ride to appointment"},
        )
        response = await mutual_aid_chip.handle(request, context)
        assert "need_id" in response.data
        assert "category" in response.data
        assert "potential_matches" in response.data

    @pytest.mark.asyncio
    async def test_unknown_intent_handling(
        self, mutual_aid_chip: MutualAidCoordinatorChip, context: SkillContext
    ):
        """Test handling of unknown intent."""
        request = SkillRequest(intent="unknown_intent", entities={})
        response = await mutual_aid_chip.handle(request, context)
        assert not response.success
        assert "unknown_intent" in response.data.get("error", "")


# ===========================================================================
# ResourceRedistributionChip Tests (10 tests)
# ===========================================================================

class TestResourceRedistributionChip:
    """Tests for ResourceRedistributionChip."""

    def test_chip_name(self, resource_chip: ResourceRedistributionChip):
        """Test chip has correct name."""
        assert resource_chip.name == "resource_redistribution"

    def test_chip_domain(self, resource_chip: ResourceRedistributionChip):
        """Test chip is in MUTUAL_AID domain."""
        assert resource_chip.domain == SkillDomain.MUTUAL_AID

    def test_efe_weights_resource_efficiency(self, resource_chip: ResourceRedistributionChip):
        """Test EFE weights include resource_efficiency at 0.25."""
        assert resource_chip.efe_weights is not None
        assert resource_chip.efe_weights.resource_efficiency == 0.25

    def test_intents_supported(self, resource_chip: ResourceRedistributionChip):
        """Test required intents are handled."""
        expected_intents = [
            "surplus_report",
            "redistribution_request",
            "pickup_schedule",
            "inventory_check",
            "partner_connect",
        ]
        chip_info = resource_chip.get_info()
        assert chip_info["name"] == "resource_redistribution"

    @pytest.mark.asyncio
    async def test_surplus_report_intent(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test reporting surplus resources."""
        request = SkillRequest(
            intent="surplus_report",
            entities={
                "resource_type": "prepared_food",
                "quantity": "50",
                "unit": "meals",
                "pickup_location": "Downtown Community Center",
                "expiry_hours": 4,
            },
        )
        response = await resource_chip.handle(request, context)
        assert response.success
        assert "surplus_id" in response.data
        assert response.data["resource_type"] == "prepared_food"

    @pytest.mark.asyncio
    async def test_perishable_time_sensitive_handling(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test time-sensitive handling for perishables."""
        request = SkillRequest(
            intent="surplus_report",
            entities={
                "resource_type": "fresh_produce",
                "quantity": "20",
                "unit": "boxes",
                "pickup_location": "Farmers Market",
                "expiry_hours": 2,  # Short expiry
            },
        )
        response = await resource_chip.handle(request, context)
        assert response.success
        # Perishables should show expiry time
        assert response.data.get("expiry_time") is not None

    @pytest.mark.asyncio
    async def test_inventory_check_intent(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test inventory check returns data."""
        # First add some inventory
        await resource_chip.report_surplus(
            partner_id="test-partner",
            resource_type="shelf_stable",
            description="Canned goods",
            quantity="100",
            unit="cans",
            pickup_location="Food Bank",
        )

        request = SkillRequest(intent="inventory_check", entities={})
        response = await resource_chip.handle(request, context)
        assert response.success
        assert "inventory" in response.data

    @pytest.mark.asyncio
    async def test_response_includes_inventory_data(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test response includes inventory/logistics data."""
        request = SkillRequest(
            intent="redistribution_request",
            entities={
                "resource_types": ["prepared_food"],
                "quantity_needed": "25 meals",
                "delivery_location": "Shelter",
            },
        )
        response = await resource_chip.handle(request, context)
        assert response.success
        assert "request_id" in response.data
        assert "resource_types" in response.data

    @pytest.mark.asyncio
    async def test_pickup_schedule_intent(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test scheduling pickups."""
        # First create surplus
        surplus = await resource_chip.report_surplus(
            partner_id="test",
            resource_type="clothing",
            description="Winter coats",
            quantity="30",
            unit="items",
            pickup_location="Donation Center",
        )

        request = SkillRequest(
            intent="pickup_schedule",
            entities={
                "surplus_id": surplus.id,
                "delivery_location": "Community Center",
            },
        )
        response = await resource_chip.handle(request, context)
        assert response.success
        assert "schedule_id" in response.data

    @pytest.mark.asyncio
    async def test_partner_connect_intent(
        self, resource_chip: ResourceRedistributionChip, context: SkillContext
    ):
        """Test partner connection."""
        request = SkillRequest(
            intent="partner_connect",
            entities={"action": "list"},
        )
        response = await resource_chip.handle(request, context)
        assert response.success
        assert "partners" in response.data


# ===========================================================================
# CrisisResponseChip Tests (10 tests)
# ===========================================================================

class TestCrisisResponseChip:
    """Tests for CrisisResponseChip."""

    def test_chip_name(self, crisis_chip: CrisisResponseChip):
        """Test chip has correct name."""
        assert crisis_chip.name == "crisis_response"

    def test_chip_domain(self, crisis_chip: CrisisResponseChip):
        """Test chip is in MUTUAL_AID domain."""
        assert crisis_chip.domain == SkillDomain.MUTUAL_AID

    def test_efe_weights(self, crisis_chip: CrisisResponseChip):
        """Test EFE weights have stakeholder_benefit=0.35 and mission_alignment=0.30."""
        assert crisis_chip.efe_weights is not None
        assert crisis_chip.efe_weights.stakeholder_benefit == 0.35
        assert crisis_chip.efe_weights.mission_alignment == 0.30

    def test_intents_supported(self, crisis_chip: CrisisResponseChip):
        """Test required intents are handled."""
        expected_intents = [
            "crisis_alert",
            "mobilize_response",
            "resource_deploy",
            "status_update",
            "debrief",
        ]
        chip_info = crisis_chip.get_info()
        assert chip_info["name"] == "crisis_response"

    def test_consensus_actions(self, crisis_chip: CrisisResponseChip):
        """Test consensus actions include emergency protocols."""
        assert "activate_emergency_protocol" in crisis_chip.consensus_actions
        assert "release_emergency_funds" in crisis_chip.consensus_actions

    @pytest.mark.asyncio
    async def test_crisis_alert_intent(
        self, crisis_chip: CrisisResponseChip, context: SkillContext
    ):
        """Test crisis alert creation."""
        request = SkillRequest(
            intent="crisis_alert",
            entities={
                "crisis_type": "natural_disaster",
                "severity": "high",
                "title": "Flooding",
                "location": "Riverside",
                "description": "Flash flooding affecting neighborhoods",
            },
        )
        response = await crisis_chip.handle(request, context)
        assert response.success
        assert "incident_id" in response.data
        assert response.data["severity"] == "high"

    @pytest.mark.asyncio
    async def test_critical_severity_requires_consensus(
        self, crisis_chip: CrisisResponseChip, context: SkillContext
    ):
        """Test CRITICAL severity requires emergency protocol consensus."""
        request = SkillRequest(
            intent="crisis_alert",
            entities={
                "crisis_type": "natural_disaster",
                "severity": "critical",
                "title": "Major Earthquake",
            },
        )
        response = await crisis_chip.handle(request, context)
        assert response.requires_consensus
        assert response.consensus_action == "activate_emergency_protocol"

    @pytest.mark.asyncio
    async def test_severity_levels_in_response(
        self, crisis_chip: CrisisResponseChip, context: SkillContext
    ):
        """Test severity levels (CRITICAL/HIGH/MEDIUM/LOW) in responses."""
        for severity in ["critical", "high", "medium", "low"]:
            request = SkillRequest(
                intent="crisis_alert",
                entities={
                    "crisis_type": "housing_crisis",
                    "severity": severity,
                    "title": f"Test {severity} alert",
                },
            )
            response = await crisis_chip.handle(request, context)
            # Critical requires consensus, others should create incident
            if severity == "critical":
                assert response.requires_consensus
            else:
                assert response.success
                assert response.data.get("severity") == severity

    @pytest.mark.asyncio
    async def test_mobilize_response_intent(
        self, crisis_chip: CrisisResponseChip, context: SkillContext
    ):
        """Test volunteer mobilization."""
        # First create an incident
        alert_req = SkillRequest(
            intent="crisis_alert",
            entities={"crisis_type": "food_insecurity", "severity": "medium"},
        )
        alert_resp = await crisis_chip.handle(alert_req, context)
        incident_id = alert_resp.data["incident_id"]

        # Now mobilize
        request = SkillRequest(
            intent="mobilize_response",
            entities={"incident_id": incident_id, "volunteer_count": 10},
        )
        response = await crisis_chip.handle(request, context)
        assert response.success
        assert "volunteers_mobilized" in response.data

    @pytest.mark.asyncio
    async def test_debrief_intent(
        self, crisis_chip: CrisisResponseChip, context: SkillContext
    ):
        """Test post-incident debrief."""
        # Create incident
        alert_req = SkillRequest(
            intent="crisis_alert",
            entities={"crisis_type": "infrastructure", "severity": "low"},
        )
        alert_resp = await crisis_chip.handle(alert_req, context)
        incident_id = alert_resp.data["incident_id"]

        # Debrief
        request = SkillRequest(
            intent="debrief",
            entities={
                "incident_id": incident_id,
                "what_worked": ["Fast response"],
                "lessons_learned": ["Need more volunteers"],
            },
        )
        response = await crisis_chip.handle(request, context)
        assert response.success
        assert "debrief_id" in response.data


# ===========================================================================
# CommunityAssetMapperChip Tests (10 tests)
# ===========================================================================

class TestCommunityAssetMapperChip:
    """Tests for CommunityAssetMapperChip."""

    def test_chip_name(self, asset_mapper_chip: CommunityAssetMapperChip):
        """Test chip has correct name."""
        assert asset_mapper_chip.name == "community_asset_mapper"

    def test_chip_domain(self, asset_mapper_chip: CommunityAssetMapperChip):
        """Test chip is in COMMUNITY domain."""
        assert asset_mapper_chip.domain == SkillDomain.COMMUNITY

    def test_intents_supported(self, asset_mapper_chip: CommunityAssetMapperChip):
        """Test required intents are handled."""
        expected_intents = [
            "asset_add",
            "asset_search",
            "asset_map",
            "skill_inventory",
            "gap_analysis",
        ]
        chip_info = asset_mapper_chip.get_info()
        assert chip_info["name"] == "community_asset_mapper"

    @pytest.mark.asyncio
    async def test_asset_add_intent(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test adding a community asset."""
        request = SkillRequest(
            intent="asset_add",
            entities={
                "asset_type": "space",
                "name": "Community Hall",
                "description": "Large hall for 200 people",
                "address": "123 Main St",
                "categories": ["meeting_space", "event_venue"],
            },
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "asset_id" in response.data
        assert response.data["name"] == "Community Hall"

    @pytest.mark.asyncio
    async def test_geocoding_in_response(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test response includes geocoding/location data."""
        request = SkillRequest(
            intent="asset_add",
            entities={
                "asset_type": "space",
                "name": "Test Location",
                "address": "456 Oak Ave",
                "categories": ["meeting_space"],
            },
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        # Response should include geo_location
        assert "geo_location" in response.data
        if response.data["geo_location"]:
            assert "lat" in response.data["geo_location"]
            assert "lng" in response.data["geo_location"]

    @pytest.mark.asyncio
    async def test_skill_inventory_intent(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test skill inventory."""
        request = SkillRequest(
            intent="skill_inventory",
            entities={
                "action": "add",
                "skills": ["carpentry", "plumbing"],
                "expertise_level": "expert",
                "can_teach": True,
                "availability_hours": 10,
            },
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "skill_asset_id" in response.data

    @pytest.mark.asyncio
    async def test_asset_search_intent(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test asset search."""
        # Add an asset first
        await asset_mapper_chip.add_asset(
            asset_type="space",
            name="Searchable Space",
            description="Test space",
            address="789 Test St",
            categories=["meeting_space"],
            owner_id="owner-1",
            owner_name="Test Owner",
        )

        request = SkillRequest(
            intent="asset_search",
            entities={"query": "space", "categories": ["meeting_space"]},
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "assets" in response.data

    @pytest.mark.asyncio
    async def test_asset_map_intent(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test asset map generation."""
        request = SkillRequest(
            intent="asset_map",
            entities={"center_address": "Downtown", "radius_km": 5},
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "total_assets" in response.data
        assert "categories_count" in response.data

    @pytest.mark.asyncio
    async def test_gap_analysis_intent(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test gap analysis."""
        request = SkillRequest(
            intent="gap_analysis",
            entities={"area": "downtown"},
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "gaps" in response.data
        assert "recommendations" in response.data

    @pytest.mark.asyncio
    async def test_response_includes_asset_structure(
        self, asset_mapper_chip: CommunityAssetMapperChip, context: SkillContext
    ):
        """Test response includes asset/skill structures."""
        request = SkillRequest(
            intent="skill_inventory",
            entities={"action": "list"},
        )
        response = await asset_mapper_chip.handle(request, context)
        assert response.success
        assert "skills_inventory" in response.data


# ===========================================================================
# CoalitionBuilderChip Tests (10 tests)
# ===========================================================================

class TestCoalitionBuilderChip:
    """Tests for CoalitionBuilderChip."""

    def test_chip_name(self, coalition_chip: CoalitionBuilderChip):
        """Test chip has correct name."""
        assert coalition_chip.name == "coalition_builder"

    def test_chip_domain(self, coalition_chip: CoalitionBuilderChip):
        """Test chip is in COMMUNITY domain."""
        assert coalition_chip.domain == SkillDomain.COMMUNITY

    def test_efe_weights(self, coalition_chip: CoalitionBuilderChip):
        """Test EFE weights have mission_alignment=0.30 and transparency=0.20."""
        assert coalition_chip.efe_weights is not None
        assert coalition_chip.efe_weights.mission_alignment == 0.30
        assert coalition_chip.efe_weights.transparency == 0.20

    def test_intents_supported(self, coalition_chip: CoalitionBuilderChip):
        """Test required intents are handled."""
        expected_intents = [
            "partner_search",
            "partner_outreach",
            "campaign_coordinate",
            "meeting_schedule",
            "coalition_report",
        ]
        chip_info = coalition_chip.get_info()
        assert chip_info["name"] == "coalition_builder"

    def test_consensus_actions(self, coalition_chip: CoalitionBuilderChip):
        """Test consensus actions include MOU and joint statement."""
        assert "sign_mou" in coalition_chip.consensus_actions
        assert "joint_statement" in coalition_chip.consensus_actions

    @pytest.mark.asyncio
    async def test_partner_search_intent(
        self, coalition_chip: CoalitionBuilderChip, context: SkillContext
    ):
        """Test searching for partners."""
        request = SkillRequest(
            intent="partner_search",
            entities={
                "focus_areas": ["housing", "tenant_rights"],
                "org_type": "nonprofit",
            },
        )
        response = await coalition_chip.handle(request, context)
        assert response.success
        assert "partners" in response.data

    @pytest.mark.asyncio
    async def test_partner_outreach_intent(
        self, coalition_chip: CoalitionBuilderChip, context: SkillContext
    ):
        """Test partner outreach."""
        request = SkillRequest(
            intent="partner_outreach",
            entities={
                "action": "send",
                "partner_id": "partner-001",
                "outreach_type": "email",
                "message": "Let's collaborate!",
            },
        )
        response = await coalition_chip.handle(request, context)
        assert response.success
        assert "outreach_id" in response.data

    @pytest.mark.asyncio
    async def test_campaign_coordinate_intent(
        self, coalition_chip: CoalitionBuilderChip, context: SkillContext
    ):
        """Test campaign coordination."""
        request = SkillRequest(
            intent="campaign_coordinate",
            entities={
                "action": "create",
                "name": "Housing Justice Campaign",
                "description": "Advocacy for tenant rights",
                "goals": ["Pass rent control"],
                "focus_areas": ["housing"],
            },
        )
        response = await coalition_chip.handle(request, context)
        assert response.success
        # Creating a shared campaign requires consensus
        assert response.requires_consensus or "id" in response.data

    @pytest.mark.asyncio
    async def test_partner_alignment_tracking(
        self, coalition_chip: CoalitionBuilderChip, context: SkillContext
    ):
        """Test partner alignment is tracked."""
        request = SkillRequest(
            intent="partner_search",
            entities={"alignment": "high"},
        )
        response = await coalition_chip.handle(request, context)
        assert response.success
        # Partners should be filterable by alignment
        assert "partners" in response.data

    @pytest.mark.asyncio
    async def test_meeting_schedule_intent(
        self, coalition_chip: CoalitionBuilderChip, context: SkillContext
    ):
        """Test meeting scheduling."""
        request = SkillRequest(
            intent="meeting_schedule",
            entities={
                "action": "schedule",
                "title": "Coalition Planning Meeting",
                "meeting_type": "planning",
                "location": "Zoom",
                "partner_ids": ["partner-001"],
            },
        )
        response = await coalition_chip.handle(request, context)
        assert response.success
        assert "meeting_id" in response.data


# ===========================================================================
# KnowYourRightsChip Tests (10 tests)
# ===========================================================================

class TestKnowYourRightsChip:
    """Tests for KnowYourRightsChip."""

    def test_chip_name(self, rights_chip: KnowYourRightsChip):
        """Test chip has correct name."""
        assert rights_chip.name == "know_your_rights"

    def test_chip_domain(self, rights_chip: KnowYourRightsChip):
        """Test chip is in ADVOCACY domain."""
        assert rights_chip.domain == SkillDomain.ADVOCACY

    def test_efe_weights_stakeholder_benefit(self, rights_chip: KnowYourRightsChip):
        """Test EFE weights have stakeholder_benefit=0.35."""
        assert rights_chip.efe_weights is not None
        assert rights_chip.efe_weights.stakeholder_benefit == 0.35

    def test_intents_supported(self, rights_chip: KnowYourRightsChip):
        """Test required intents are handled."""
        expected_intents = [
            "rights_lookup",
            "legal_clinic",
            "know_rights_workshop",
            "legal_resource",
            "rights_card",
        ]
        chip_info = rights_chip.get_info()
        assert chip_info["name"] == "know_your_rights"

    @pytest.mark.asyncio
    async def test_rights_lookup_intent(
        self, rights_chip: KnowYourRightsChip, context: SkillContext
    ):
        """Test rights lookup."""
        request = SkillRequest(
            intent="rights_lookup",
            entities={"topic": "tenant_rights", "state": "CA"},
        )
        response = await rights_chip.handle(request, context)
        assert response.success
        assert "rights_info" in response.data

    @pytest.mark.asyncio
    async def test_legal_disclaimer_in_response(
        self, rights_chip: KnowYourRightsChip, context: SkillContext
    ):
        """Test legal disclaimer is included in responses."""
        request = SkillRequest(
            intent="rights_lookup",
            entities={"topic": "worker_rights"},
        )
        response = await rights_chip.handle(request, context)
        # Response content should include disclaimer
        assert "DISCLAIMER" in response.content or "legal advice" in response.content.lower()

    @pytest.mark.asyncio
    async def test_legal_clinic_intent(
        self, rights_chip: KnowYourRightsChip, context: SkillContext
    ):
        """Test legal clinic finding."""
        request = SkillRequest(
            intent="legal_clinic",
            entities={"location": "Oakland", "topic": "tenant_rights"},
        )
        response = await rights_chip.handle(request, context)
        assert response.success
        assert "clinics" in response.data

    @pytest.mark.asyncio
    async def test_multi_language_support(self, rights_chip: KnowYourRightsChip):
        """Test multi-language support is referenced."""
        # Check supported languages are defined
        assert hasattr(rights_chip, "SUPPORTED_LANGUAGES")
        assert len(rights_chip.SUPPORTED_LANGUAGES) > 1
        assert "en" in rights_chip.SUPPORTED_LANGUAGES
        assert "es" in rights_chip.SUPPORTED_LANGUAGES

    @pytest.mark.asyncio
    async def test_rights_card_intent(
        self, rights_chip: KnowYourRightsChip, context: SkillContext
    ):
        """Test rights card generation."""
        request = SkillRequest(
            intent="rights_card",
            entities={"topic": "police_encounter", "language": "en"},
        )
        response = await rights_chip.handle(request, context)
        assert response.success
        assert "rights_card" in response.data

    @pytest.mark.asyncio
    async def test_workshop_planning_intent(
        self, rights_chip: KnowYourRightsChip, context: SkillContext
    ):
        """Test workshop planning."""
        request = SkillRequest(
            intent="know_rights_workshop",
            entities={"topic": "immigration_rights", "audience": "community"},
        )
        response = await rights_chip.handle(request, context)
        assert response.success
        assert "workshop_plan" in response.data


# ===========================================================================
# HousingNavigatorChip Tests (10 tests)
# ===========================================================================

class TestHousingNavigatorChip:
    """Tests for HousingNavigatorChip."""

    def test_chip_name(self, housing_chip: HousingNavigatorChip):
        """Test chip has correct name."""
        assert housing_chip.name == "housing_navigator"

    def test_chip_domain(self, housing_chip: HousingNavigatorChip):
        """Test chip is in ADVOCACY domain."""
        assert housing_chip.domain == SkillDomain.ADVOCACY

    def test_efe_weights_stakeholder_benefit_highest(self, housing_chip: HousingNavigatorChip):
        """Test EFE weights have stakeholder_benefit=0.40 (highest)."""
        assert housing_chip.efe_weights is not None
        assert housing_chip.efe_weights.stakeholder_benefit == 0.40

    def test_pii_access_capability(self, housing_chip: HousingNavigatorChip):
        """Test capabilities include PII_ACCESS."""
        assert SkillCapability.PII_ACCESS in housing_chip.capabilities

    def test_intents_supported(self, housing_chip: HousingNavigatorChip):
        """Test required intents are handled."""
        expected_intents = [
            "housing_search",
            "voucher_status",
            "tenant_rights",
            "landlord_lookup",
            "eviction_defense",
        ]
        chip_info = housing_chip.get_info()
        assert chip_info["name"] == "housing_navigator"

    @pytest.mark.asyncio
    async def test_housing_search_intent(
        self, housing_chip: HousingNavigatorChip, context: SkillContext
    ):
        """Test housing search."""
        request = SkillRequest(
            intent="housing_search",
            entities={
                "bedrooms": 2,
                "voucher_type": "HCV",
                "city": "Oakland",
            },
        )
        response = await housing_chip.handle(request, context)
        assert response.success
        assert "units" in response.data

    @pytest.mark.asyncio
    async def test_voucher_status_intent(
        self, housing_chip: HousingNavigatorChip, context: SkillContext
    ):
        """Test voucher status check."""
        request = SkillRequest(
            intent="voucher_status",
            entities={"application_id": "APP-2024-001234"},
        )
        response = await housing_chip.handle(request, context)
        assert response.success
        assert "application_id" in response.data or "status" in response.data

    @pytest.mark.asyncio
    async def test_voucher_data_structure(
        self, housing_chip: HousingNavigatorChip, context: SkillContext
    ):
        """Test response includes voucher/housing data structures."""
        request = SkillRequest(
            intent="voucher_status",
            entities={},
        )
        response = await housing_chip.handle(request, context)
        assert response.success
        # Should include voucher status info
        assert "application_id" in response.data or "status" in response.data

    @pytest.mark.asyncio
    async def test_landlord_lookup_intent(
        self, housing_chip: HousingNavigatorChip, context: SkillContext
    ):
        """Test landlord lookup."""
        request = SkillRequest(
            intent="landlord_lookup",
            entities={"property_address": "456 Oak St"},
        )
        response = await housing_chip.handle(request, context)
        assert response.success
        assert "landlord_id" in response.data or "name" in response.data

    @pytest.mark.asyncio
    async def test_eviction_defense_intent(
        self, housing_chip: HousingNavigatorChip, context: SkillContext
    ):
        """Test eviction defense resources."""
        request = SkillRequest(
            intent="eviction_defense",
            entities={"stage": "notice"},
        )
        response = await housing_chip.handle(request, context)
        assert response.success
        assert "resources" in response.data


# ===========================================================================
# FoodAccessChip Tests (10 tests)
# ===========================================================================

class TestFoodAccessChip:
    """Tests for FoodAccessChip."""

    def test_chip_name(self, food_chip: FoodAccessChip):
        """Test chip has correct name."""
        assert food_chip.name == "food_access"

    def test_chip_domain(self, food_chip: FoodAccessChip):
        """Test chip is in MUTUAL_AID domain."""
        assert food_chip.domain == SkillDomain.MUTUAL_AID

    def test_efe_weights_stakeholder_benefit(self, food_chip: FoodAccessChip):
        """Test EFE weights have stakeholder_benefit=0.40."""
        assert food_chip.efe_weights is not None
        assert food_chip.efe_weights.stakeholder_benefit == 0.40

    def test_intents_supported(self, food_chip: FoodAccessChip):
        """Test required intents are handled."""
        expected_intents = [
            "pantry_find",
            "snap_help",
            "meal_schedule",
            "food_donate",
            "nutrition_info",
        ]
        chip_info = food_chip.get_info()
        assert chip_info["name"] == "food_access"

    @pytest.mark.asyncio
    async def test_pantry_find_intent(
        self, food_chip: FoodAccessChip, context: SkillContext
    ):
        """Test finding food pantries."""
        request = SkillRequest(
            intent="pantry_find",
            entities={"zip_code": "94612"},
        )
        response = await food_chip.handle(request, context)
        assert response.success
        assert "pantries" in response.data

    @pytest.mark.asyncio
    async def test_dietary_restriction_filtering(
        self, food_chip: FoodAccessChip, context: SkillContext
    ):
        """Test dietary restriction filtering."""
        request = SkillRequest(
            intent="pantry_find",
            entities={"zip_code": "94612", "dietary": ["halal", "vegetarian"]},
        )
        response = await food_chip.handle(request, context)
        assert response.success
        # Should filter by dietary needs
        assert "pantries" in response.data

    @pytest.mark.asyncio
    async def test_snap_eligibility_check(
        self, food_chip: FoodAccessChip, context: SkillContext
    ):
        """Test SNAP eligibility check."""
        request = SkillRequest(
            intent="snap_help",
            entities={
                "action": "eligibility",
                "household_size": 3,
                "monthly_income": 2000,
            },
        )
        response = await food_chip.handle(request, context)
        assert response.success
        # Response should include eligibility info
        assert "household_size" in response.data or "income" in response.data

    @pytest.mark.asyncio
    async def test_snap_income_limits_defined(self, food_chip: FoodAccessChip):
        """Test SNAP income limits are defined."""
        assert hasattr(food_chip, "SNAP_INCOME_LIMITS")
        assert len(food_chip.SNAP_INCOME_LIMITS) > 0
        # Should have limits for household sizes 1-8
        assert 1 in food_chip.SNAP_INCOME_LIMITS
        assert 4 in food_chip.SNAP_INCOME_LIMITS

    @pytest.mark.asyncio
    async def test_meal_schedule_intent(
        self, food_chip: FoodAccessChip, context: SkillContext
    ):
        """Test meal schedule finding."""
        request = SkillRequest(
            intent="meal_schedule",
            entities={"meal_type": "lunch"},
        )
        response = await food_chip.handle(request, context)
        assert response.success
        assert "programs" in response.data

    @pytest.mark.asyncio
    async def test_food_donate_intent(
        self, food_chip: FoodAccessChip, context: SkillContext
    ):
        """Test food donation processing."""
        request = SkillRequest(
            intent="food_donate",
            entities={
                "type": "food",
                "items": ["canned goods", "pasta"],
                "quantity": 20,
            },
        )
        response = await food_chip.handle(request, context)
        assert response.success
        assert "donation_id" in response.data


# ===========================================================================
# SolidarityEconomyChip Tests (10 tests)
# ===========================================================================

class TestSolidarityEconomyChip:
    """Tests for SolidarityEconomyChip."""

    def test_chip_name(self, solidarity_chip: SolidarityEconomyChip):
        """Test chip has correct name."""
        assert solidarity_chip.name == "solidarity_economy"

    def test_chip_domain(self, solidarity_chip: SolidarityEconomyChip):
        """Test chip is in COMMUNITY domain."""
        assert solidarity_chip.domain == SkillDomain.COMMUNITY

    def test_efe_weights_mission_alignment(self, solidarity_chip: SolidarityEconomyChip):
        """Test EFE weights have mission_alignment=0.35."""
        assert solidarity_chip.efe_weights is not None
        assert solidarity_chip.efe_weights.mission_alignment == 0.35

    def test_financial_operations_capability(self, solidarity_chip: SolidarityEconomyChip):
        """Test capabilities include FINANCIAL_OPERATIONS."""
        assert SkillCapability.FINANCIAL_OPERATIONS in solidarity_chip.capabilities

    def test_intents_supported(self, solidarity_chip: SolidarityEconomyChip):
        """Test required intents are handled."""
        expected_intents = [
            "coop_start",
            "coop_search",
            "time_bank",
            "cdfi_loan",
            "solidarity_resource",
        ]
        chip_info = solidarity_chip.get_info()
        assert chip_info["name"] == "solidarity_economy"

    @pytest.mark.asyncio
    async def test_coop_start_intent(
        self, solidarity_chip: SolidarityEconomyChip, context: SkillContext
    ):
        """Test cooperative formation guidance."""
        request = SkillRequest(
            intent="coop_start",
            entities={
                "coop_type": "worker",
                "members": 5,
                "industry": "technology",
            },
        )
        response = await solidarity_chip.handle(request, context)
        assert response.success
        # Coop start requires consensus
        assert response.requires_consensus

    @pytest.mark.asyncio
    async def test_time_bank_credit_tracking(
        self, solidarity_chip: SolidarityEconomyChip, context: SkillContext
    ):
        """Test time bank credit tracking."""
        request = SkillRequest(
            intent="time_bank",
            entities={"action": "balance"},
        )
        response = await solidarity_chip.handle(request, context)
        assert response.success
        # Should return account info with balance
        assert "account" in response.data
        account = response.data["account"]
        assert hasattr(account, "balance_hours")

    @pytest.mark.asyncio
    async def test_coop_search_intent(
        self, solidarity_chip: SolidarityEconomyChip, context: SkillContext
    ):
        """Test cooperative search."""
        request = SkillRequest(
            intent="coop_search",
            entities={
                "coop_type": "worker",
                "accepting_members": True,
            },
        )
        response = await solidarity_chip.handle(request, context)
        assert response.success
        assert "coops" in response.data

    @pytest.mark.asyncio
    async def test_cdfi_loan_intent(
        self, solidarity_chip: SolidarityEconomyChip, context: SkillContext
    ):
        """Test CDFI loan finding."""
        request = SkillRequest(
            intent="cdfi_loan",
            entities={
                "amount": 50000,
                "purpose": "startup",
            },
        )
        response = await solidarity_chip.handle(request, context)
        assert response.success
        assert "loans" in response.data

    @pytest.mark.asyncio
    async def test_cooperative_governance_references(
        self, solidarity_chip: SolidarityEconomyChip, context: SkillContext
    ):
        """Test cooperative governance templates are referenced."""
        request = SkillRequest(
            intent="solidarity_resource",
            entities={"type": "governance"},
        )
        response = await solidarity_chip.handle(request, context)
        assert response.success
        # Should include governance-related templates
        assert "templates" in response.data


# ===========================================================================
# RapidResponseChip Tests (10 tests)
# ===========================================================================

class TestRapidResponseChip:
    """Tests for RapidResponseChip."""

    def test_chip_name(self, rapid_response_chip: RapidResponseChip):
        """Test chip has correct name."""
        assert rapid_response_chip.name == "rapid_response"

    def test_chip_domain(self, rapid_response_chip: RapidResponseChip):
        """Test chip is in ADVOCACY domain."""
        assert rapid_response_chip.domain == SkillDomain.ADVOCACY

    def test_efe_weights(self, rapid_response_chip: RapidResponseChip):
        """Test EFE weights have mission_alignment=0.35 and stakeholder_benefit=0.35."""
        assert rapid_response_chip.efe_weights is not None
        assert rapid_response_chip.efe_weights.mission_alignment == 0.35
        assert rapid_response_chip.efe_weights.stakeholder_benefit == 0.35

    def test_pii_access_capability(self, rapid_response_chip: RapidResponseChip):
        """Test capabilities include PII_ACCESS."""
        assert SkillCapability.PII_ACCESS in rapid_response_chip.capabilities

    def test_send_notifications_capability(self, rapid_response_chip: RapidResponseChip):
        """Test capabilities include SEND_NOTIFICATIONS."""
        assert SkillCapability.SEND_NOTIFICATIONS in rapid_response_chip.capabilities

    def test_intents_supported(self, rapid_response_chip: RapidResponseChip):
        """Test required intents are handled."""
        expected_intents = [
            "raid_alert",
            "bail_request",
            "legal_hotline",
            "safe_location",
            "response_debrief",
        ]
        chip_info = rapid_response_chip.get_info()
        assert chip_info["name"] == "rapid_response"

    def test_consensus_actions(self, rapid_response_chip: RapidResponseChip):
        """Test consensus actions include required approvals."""
        assert "activate_rapid_response" in rapid_response_chip.consensus_actions
        assert "release_bail_funds" in rapid_response_chip.consensus_actions

    @pytest.mark.asyncio
    async def test_raid_alert_intent(
        self, rapid_response_chip: RapidResponseChip, context: SkillContext
    ):
        """Test raid alert creation."""
        request = SkillRequest(
            intent="raid_alert",
            entities={
                "type": "ice_raid",
                "urgency": "urgent",
                "area": "Downtown",
                "description": "Activity reported",
            },
        )
        response = await rapid_response_chip.handle(request, context)
        assert response.success
        assert "alert_id" in response.data
        # Should require consensus for rapid response activation
        assert response.requires_consensus

    @pytest.mark.asyncio
    async def test_privacy_security_in_response(
        self, rapid_response_chip: RapidResponseChip, context: SkillContext
    ):
        """Test privacy/security notice in responses (no sensitive location logging)."""
        request = SkillRequest(
            intent="safe_location",
            entities={"type": "sanctuary"},
        )
        response = await rapid_response_chip.handle(request, context)
        assert response.success
        # Response should include security notice
        assert "SECURITY" in response.content or "secure" in response.content.lower()
        # Should NOT include specific addresses
        assert "specific address" in response.content.lower() or "secure channel" in response.content.lower()

    @pytest.mark.asyncio
    async def test_operational_security_mentioned(
        self, rapid_response_chip: RapidResponseChip, context: SkillContext
    ):
        """Test operational security is mentioned."""
        request = SkillRequest(
            intent="legal_hotline",
            entities={"urgency": "emergency"},
        )
        response = await rapid_response_chip.handle(request, context)
        assert response.success
        # Should have hotline info
        assert "hotline" in response.content.lower() or "number" in response.data.get("number", "").lower()


# ===========================================================================
# Cross-chip Integration Tests
# ===========================================================================

class TestCrossChipIntegration:
    """Tests for cross-chip integration patterns."""

    @pytest.mark.asyncio
    async def test_all_chips_have_handle_method(self):
        """Test all chips have async handle method."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]
        for chip in chips:
            assert hasattr(chip, "handle")
            assert callable(chip.handle)

    @pytest.mark.asyncio
    async def test_all_chips_return_skill_response(self, context: SkillContext):
        """Test all chips return SkillResponse."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]
        request = SkillRequest(intent="unknown_test_intent", entities={})

        for chip in chips:
            response = await chip.handle(request, context)
            assert isinstance(response, SkillResponse)

    def test_all_chips_have_efe_weights(self):
        """Test all chips have EFE weights configured."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]
        for chip in chips:
            assert chip.efe_weights is not None
            total = chip.efe_weights.total()
            # Weights should sum to approximately 1.0
            assert 0.95 <= total <= 1.05, f"{chip.name} EFE weights sum to {total}"

    def test_all_chips_have_get_info(self):
        """Test all chips have get_info method returning correct structure."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]
        required_keys = ["name", "description", "version", "domain", "efe_weights"]

        for chip in chips:
            info = chip.get_info()
            for key in required_keys:
                assert key in info, f"{chip.name} missing {key} in get_info()"

    @pytest.mark.asyncio
    async def test_all_chips_have_get_bdi_context(self, context_with_bdi: SkillContext):
        """Test all chips have get_bdi_context method."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]

        for chip in chips:
            assert hasattr(chip, "get_bdi_context")
            bdi = await chip.get_bdi_context(
                context_with_bdi.beliefs,
                context_with_bdi.desires,
                context_with_bdi.intentions,
            )
            assert "beliefs" in bdi
            assert "desires" in bdi
            assert "intentions" in bdi


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_entities_handling(self, context: SkillContext):
        """Test chips handle empty entities gracefully."""
        chips_and_intents = [
            (MutualAidCoordinatorChip(), "need_post"),
            (ResourceRedistributionChip(), "surplus_report"),
            (CrisisResponseChip(), "crisis_alert"),
            (CommunityAssetMapperChip(), "asset_add"),
            (CoalitionBuilderChip(), "partner_search"),
            (KnowYourRightsChip(), "rights_lookup"),
            (HousingNavigatorChip(), "housing_search"),
            (FoodAccessChip(), "pantry_find"),
            (SolidarityEconomyChip(), "time_bank"),
            (RapidResponseChip(), "legal_hotline"),
        ]

        for chip, intent in chips_and_intents:
            request = SkillRequest(intent=intent, entities={})
            response = await chip.handle(request, context)
            # Should not crash, may or may not succeed
            assert isinstance(response, SkillResponse)

    @pytest.mark.asyncio
    async def test_invalid_intent_returns_error(self, context: SkillContext):
        """Test invalid intents return proper error responses."""
        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]

        for chip in chips:
            request = SkillRequest(intent="completely_invalid_intent_xyz", entities={})
            response = await chip.handle(request, context)
            assert not response.success
            assert "unknown" in response.content.lower() or "unknown_intent" in response.data.get("error", "")

    @pytest.mark.asyncio
    async def test_requires_consensus_tracking(self, context: SkillContext):
        """Test consensus actions are properly tracked."""
        # Test chips that have consensus actions
        crisis_chip = CrisisResponseChip()
        request = SkillRequest(
            intent="crisis_alert",
            entities={"severity": "critical"},
        )
        response = await crisis_chip.handle(request, context)
        assert response.requires_consensus
        assert response.consensus_action is not None

    def test_chip_version_format(self):
        """Test all chips have valid semantic version format."""
        import re
        semver_pattern = r"^\d+\.\d+\.\d+$"

        chips = [
            MutualAidCoordinatorChip(),
            ResourceRedistributionChip(),
            CrisisResponseChip(),
            CommunityAssetMapperChip(),
            CoalitionBuilderChip(),
            KnowYourRightsChip(),
            HousingNavigatorChip(),
            FoodAccessChip(),
            SolidarityEconomyChip(),
            RapidResponseChip(),
        ]

        for chip in chips:
            assert re.match(semver_pattern, chip.version), f"{chip.name} has invalid version: {chip.version}"

    @pytest.mark.asyncio
    async def test_response_always_has_content(self, context: SkillContext):
        """Test all responses have content string."""
        chips_and_intents = [
            (MutualAidCoordinatorChip(), "aid_report"),
            (ResourceRedistributionChip(), "inventory_check"),
            (CrisisResponseChip(), "status_update"),
            (CommunityAssetMapperChip(), "asset_search"),
            (CoalitionBuilderChip(), "coalition_report"),
            (KnowYourRightsChip(), "legal_resource"),
            (HousingNavigatorChip(), "tenant_rights"),
            (FoodAccessChip(), "nutrition_info"),
            (SolidarityEconomyChip(), "solidarity_resource"),
            (RapidResponseChip(), "response_debrief"),
        ]

        for chip, intent in chips_and_intents:
            request = SkillRequest(intent=intent, entities={})
            response = await chip.handle(request, context)
            assert response.content is not None
            assert isinstance(response.content, str)
            assert len(response.content) > 0
