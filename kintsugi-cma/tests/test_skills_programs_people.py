"""Comprehensive pytest tests for Kintsugi CMA Phase 4b Skill Chips.

Tests cover all 6 Programs & People domain skill chips:
- ProgramEvaluatorChip: Logic models, outcomes, and evaluation design
- BoardLiaisonChip: Board governance, meetings, and compliance
- DonorStewardshipChip: Donor relationships and cultivation
- StaffOnboardingChip: New staff onboarding and training
- EventPlannerChip: Event planning, RSVPs, and logistics
- MemberServicesChip: Membership tracking and communications

Each chip is tested for:
- Chip attributes (name, domain, version)
- EFE weights configuration
- Capabilities declaration
- Intent handling and routing
- Response structure and data
- Consensus action requirements
- BDI context filtering
"""

import pytest
from datetime import datetime, timezone

from kintsugi.skills import (
    SkillRequest,
    SkillContext,
    SkillDomain,
    SkillCapability,
    EFEWeights,
)
from kintsugi.skills.programs_people import (
    ProgramEvaluatorChip,
    BoardLiaisonChip,
    DonorStewardshipChip,
    StaffOnboardingChip,
    EventPlannerChip,
    MemberServicesChip,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def context():
    """Standard test context."""
    return SkillContext(
        org_id="test-org-123",
        user_id="test-user-456",
        session_id="session-789",
        platform="webchat",
        beliefs=[
            {"domain": "programs", "type": "program_status", "value": "active"},
            {"domain": "fundraising", "type": "donor_relationship", "value": "high"},
            {"domain": "governance", "type": "compliance_status", "value": "compliant"},
            {"domain": "operations", "type": "staff_count", "value": 25},
        ],
        desires=[
            {"type": "improve_program", "domain": "evaluation"},
            {"type": "increase_retention", "domain": "fundraising"},
            {"type": "maintain_compliance", "domain": "governance"},
        ],
        intentions=[],
    )


# ==============================================================================
# ProgramEvaluatorChip Tests (10 tests)
# ==============================================================================

class TestProgramEvaluatorChip:
    """Tests for ProgramEvaluatorChip."""

    @pytest.fixture
    def chip(self):
        """Create ProgramEvaluatorChip instance."""
        return ProgramEvaluatorChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "program_evaluator"

    def test_chip_domain(self, chip):
        """Chip is in PROGRAMS domain."""
        assert chip.domain == SkillDomain.PROGRAMS

    def test_efe_weights_mission_alignment(self, chip):
        """EFE weights prioritize mission alignment at 0.30."""
        assert chip.efe_weights.mission_alignment == 0.30

    def test_efe_weights_stakeholder_benefit(self, chip):
        """EFE weights prioritize stakeholder benefit at 0.30."""
        assert chip.efe_weights.stakeholder_benefit == 0.30

    def test_capabilities_include_required(self, chip):
        """Chip declares READ_DATA, WRITE_DATA, GENERATE_REPORTS capabilities."""
        assert SkillCapability.READ_DATA in chip.capabilities
        assert SkillCapability.WRITE_DATA in chip.capabilities
        assert SkillCapability.GENERATE_REPORTS in chip.capabilities

    def test_consensus_actions(self, chip):
        """Chip requires consensus for finalize_evaluation and publish_findings."""
        assert "finalize_evaluation" in chip.consensus_actions
        assert "publish_findings" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_logic_model(self, chip, context):
        """Handle routes logic_model intent correctly."""
        request = SkillRequest(
            intent="logic_model",
            entities={"program_name": "Youth Mentorship", "program_id": "prog_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "logic_model" in response.data
        assert response.data["logic_model"]["program_name"] == "Youth Mentorship"
        assert "components" in response.data["logic_model"]

    @pytest.mark.asyncio
    async def test_handle_outcome_track(self, chip, context):
        """Handle routes outcome_track intent correctly."""
        request = SkillRequest(
            intent="outcome_track",
            entities={"program_id": "prog_001", "period": "quarterly"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "outcomes" in response.data
        assert "summary" in response.data

    @pytest.mark.asyncio
    async def test_handle_evaluation_design(self, chip, context):
        """Handle routes evaluation_design intent correctly."""
        request = SkillRequest(
            intent="evaluation_design",
            entities={"program_id": "prog_001", "evaluation_type": "summative"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "evaluation_design" in response.data
        assert response.data["evaluation_design"]["type"] == "summative"

    @pytest.mark.asyncio
    async def test_get_bdi_context_filters_program_beliefs(self, chip):
        """get_bdi_context filters beliefs for program-related domains."""
        beliefs = [
            {"domain": "programs", "type": "program_status"},
            {"domain": "evaluation", "type": "outcome_target"},
            {"domain": "finance", "type": "budget"},
        ]
        desires = [
            {"type": "improve_program", "domain": "programs"},
            {"type": "increase_revenue", "domain": "fundraising"},
        ]
        intentions = []

        bdi = await chip.get_bdi_context(beliefs, desires, intentions)

        # Should include programs and evaluation beliefs
        assert len(bdi["beliefs"]) == 2
        assert all(b["domain"] in ("programs", "evaluation") for b in bdi["beliefs"])
        # Should include improve_program desire
        assert len(bdi["desires"]) == 1


# ==============================================================================
# BoardLiaisonChip Tests (10 tests)
# ==============================================================================

class TestBoardLiaisonChip:
    """Tests for BoardLiaisonChip."""

    @pytest.fixture
    def chip(self):
        """Create BoardLiaisonChip instance."""
        return BoardLiaisonChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "board_liaison"

    def test_chip_domain(self, chip):
        """Chip is in GOVERNANCE domain."""
        assert chip.domain == SkillDomain.GOVERNANCE

    def test_efe_weights_transparency_highest(self, chip):
        """EFE weights prioritize transparency at 0.30 (highest for governance)."""
        assert chip.efe_weights.transparency == 0.30
        # Verify transparency is highest
        assert chip.efe_weights.transparency >= chip.efe_weights.mission_alignment
        assert chip.efe_weights.transparency >= chip.efe_weights.stakeholder_benefit

    def test_consensus_actions(self, chip):
        """Chip requires consensus for governance actions."""
        assert "distribute_materials" in chip.consensus_actions
        assert "record_resolution" in chip.consensus_actions
        assert "update_bylaws" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_meeting_prep(self, chip, context):
        """Handle routes meeting_prep intent correctly."""
        request = SkillRequest(
            intent="meeting_prep",
            entities={"meeting_type": "regular", "include_financials": True},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "meeting_packet" in response.data
        assert "agenda_items" in response.data["meeting_packet"]

    @pytest.mark.asyncio
    async def test_handle_minutes_draft(self, chip, context):
        """Handle routes minutes_draft intent correctly."""
        request = SkillRequest(
            intent="minutes_draft",
            entities={"meeting_id": "mtg_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "minutes" in response.data
        assert "attendance" in response.data["minutes"]
        assert "proceedings" in response.data["minutes"]

    @pytest.mark.asyncio
    async def test_handle_resolution_track(self, chip, context):
        """Handle routes resolution_track intent correctly."""
        request = SkillRequest(
            intent="resolution_track",
            entities={"action": "list"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "resolutions" in response.data
        assert "summary" in response.data

    @pytest.mark.asyncio
    async def test_handle_compliance_check(self, chip, context):
        """Handle routes compliance_check intent correctly."""
        request = SkillRequest(
            intent="compliance_check",
            entities={"include_overdue": True},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "compliance_items" in response.data
        assert "summary" in response.data

    @pytest.mark.asyncio
    async def test_handle_board_report(self, chip, context):
        """Handle routes board_report intent correctly."""
        request = SkillRequest(
            intent="board_report",
            entities={"period": "quarterly", "include_financials": True},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "board_report" in response.data
        assert "executive_summary" in response.data["board_report"]

    @pytest.mark.asyncio
    async def test_handle_unknown_intent(self, chip, context):
        """Handle returns failure for unknown intent."""
        request = SkillRequest(intent="unknown_intent", entities={})
        response = await chip.handle(request, context)
        assert response.success is False
        assert "suggestions" in response.__dict__


# ==============================================================================
# DonorStewardshipChip Tests (10 tests)
# ==============================================================================

class TestDonorStewardshipChip:
    """Tests for DonorStewardshipChip."""

    @pytest.fixture
    def chip(self):
        """Create DonorStewardshipChip instance."""
        return DonorStewardshipChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "donor_stewardship"

    def test_chip_domain(self, chip):
        """Chip is in FUNDRAISING domain."""
        assert chip.domain == SkillDomain.FUNDRAISING

    def test_efe_weights_stakeholder_benefit_highest(self, chip):
        """EFE weights prioritize stakeholder benefit at 0.35 (donor focus)."""
        assert chip.efe_weights.stakeholder_benefit == 0.35

    def test_capabilities_include_pii_access(self, chip):
        """Chip declares PII_ACCESS capability."""
        assert SkillCapability.PII_ACCESS in chip.capabilities

    def test_consensus_actions(self, chip):
        """Chip requires consensus for donor-related actions."""
        assert "send_acknowledgment" in chip.consensus_actions
        assert "share_donor_data" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_donor_thank(self, chip, context):
        """Handle routes donor_thank intent correctly."""
        request = SkillRequest(
            intent="donor_thank",
            entities={"donor_id": "donor_001", "gift_amount": 1000},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "acknowledgment" in response.data
        assert "donor_summary" in response.data

    @pytest.mark.asyncio
    async def test_handle_donor_profile(self, chip, context):
        """Handle routes donor_profile intent correctly."""
        request = SkillRequest(
            intent="donor_profile",
            entities={"donor_id": "donor_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "profile" in response.data
        assert "donor_level" in response.data["profile"]
        assert "giving_summary" in response.data["profile"]

    @pytest.mark.asyncio
    async def test_handle_giving_history(self, chip, context):
        """Handle routes giving_history intent correctly."""
        request = SkillRequest(
            intent="giving_history",
            entities={"donor_id": "donor_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "giving_analysis" in response.data

    @pytest.mark.asyncio
    async def test_handle_cultivation_plan(self, chip, context):
        """Handle routes cultivation_plan intent correctly."""
        request = SkillRequest(
            intent="cultivation_plan",
            entities={"donor_id": "donor_001", "target_amount": 5000},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "cultivation_plan" in response.data
        assert "activities" in response.data["cultivation_plan"]

    @pytest.mark.asyncio
    async def test_handle_stewardship_report(self, chip, context):
        """Handle routes stewardship_report intent correctly."""
        request = SkillRequest(
            intent="stewardship_report",
            entities={"period": "quarterly"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "stewardship_report" in response.data
        assert "summary" in response.data["stewardship_report"]


# ==============================================================================
# StaffOnboardingChip Tests (10 tests)
# ==============================================================================

class TestStaffOnboardingChip:
    """Tests for StaffOnboardingChip."""

    @pytest.fixture
    def chip(self):
        """Create StaffOnboardingChip instance."""
        return StaffOnboardingChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "staff_onboarding"

    def test_chip_domain(self, chip):
        """Chip is in OPERATIONS domain."""
        assert chip.domain == SkillDomain.OPERATIONS

    def test_capabilities_include_pii_access(self, chip):
        """Chip declares PII_ACCESS capability."""
        assert SkillCapability.PII_ACCESS in chip.capabilities

    def test_capabilities_include_schedule_tasks(self, chip):
        """Chip declares SCHEDULE_TASKS capability."""
        assert SkillCapability.SCHEDULE_TASKS in chip.capabilities

    def test_consensus_actions(self, chip):
        """Chip requires consensus for onboarding completion actions."""
        assert "complete_onboarding" in chip.consensus_actions
        assert "grant_system_access" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_onboard_start(self, chip, context):
        """Handle routes onboard_start intent correctly."""
        request = SkillRequest(
            intent="onboard_start",
            entities={
                "employee_id": "emp_001",
                "employee_name": "Jane Doe",
                "department": "programs",
            },
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "onboarding_plan" in response.data
        assert "checklist_items" in response.data["onboarding_plan"]

    @pytest.mark.asyncio
    async def test_handle_training_assign(self, chip, context):
        """Handle routes training_assign intent correctly."""
        request = SkillRequest(
            intent="training_assign",
            entities={"employee_id": "emp_001", "department": "programs"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "training_assignment" in response.data
        assert "modules" in response.data["training_assignment"]

    @pytest.mark.asyncio
    async def test_handle_policy_review(self, chip, context):
        """Handle routes policy_review intent correctly."""
        request = SkillRequest(
            intent="policy_review",
            entities={"employee_id": "emp_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "policy_review" in response.data
        assert "policies" in response.data["policy_review"]

    @pytest.mark.asyncio
    async def test_handle_checklist_status(self, chip, context):
        """Handle routes checklist_status intent correctly."""
        request = SkillRequest(
            intent="checklist_status",
            entities={"employee_id": "emp_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "onboarding_progress" in response.data
        assert "overall_progress" in response.data["onboarding_progress"]

    @pytest.mark.asyncio
    async def test_handle_onboard_complete(self, chip, context):
        """Handle routes onboard_complete intent correctly."""
        request = SkillRequest(
            intent="onboard_complete",
            entities={"employee_id": "emp_001", "force_complete": True},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "completion_record" in response.data


# ==============================================================================
# EventPlannerChip Tests (10 tests)
# ==============================================================================

class TestEventPlannerChip:
    """Tests for EventPlannerChip."""

    @pytest.fixture
    def chip(self):
        """Create EventPlannerChip instance."""
        return EventPlannerChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "event_planner"

    def test_chip_domain(self, chip):
        """Chip is in OPERATIONS domain."""
        assert chip.domain == SkillDomain.OPERATIONS

    def test_efe_weights_resource_efficiency(self, chip):
        """EFE weights include resource efficiency at 0.25."""
        assert chip.efe_weights.resource_efficiency == 0.25

    def test_consensus_actions(self, chip):
        """Chip requires consensus for event finalization and budget."""
        assert "finalize_event" in chip.consensus_actions
        assert "commit_budget" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_event_create(self, chip, context):
        """Handle routes event_create intent correctly."""
        request = SkillRequest(
            intent="event_create",
            entities={
                "event_name": "Annual Gala",
                "event_type": "gala",
                "capacity": 100,
            },
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "event" in response.data
        assert response.data["event"]["name"] == "Annual Gala"

    @pytest.mark.asyncio
    async def test_handle_event_rsvp(self, chip, context):
        """Handle routes event_rsvp intent correctly."""
        request = SkillRequest(
            intent="event_rsvp",
            entities={"event_id": "evt_001", "action": "list"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "rsvp_data" in response.data
        assert "summary" in response.data["rsvp_data"]

    @pytest.mark.asyncio
    async def test_handle_event_logistics(self, chip, context):
        """Handle routes event_logistics intent correctly."""
        request = SkillRequest(
            intent="event_logistics",
            entities={"event_id": "evt_001", "expected_attendance": 75},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "logistics" in response.data
        assert "venue" in response.data["logistics"]

    @pytest.mark.asyncio
    async def test_handle_event_accessibility(self, chip, context):
        """Handle routes event_accessibility intent correctly with accessibility check."""
        request = SkillRequest(
            intent="event_accessibility",
            entities={"event_id": "evt_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "accessibility" in response.data
        assert "features" in response.data["accessibility"]
        # Verify accessibility statement is included
        assert "accessibility_statement" in response.data["accessibility"]

    @pytest.mark.asyncio
    async def test_handle_event_followup(self, chip, context):
        """Handle routes event_followup intent correctly."""
        request = SkillRequest(
            intent="event_followup",
            entities={"event_id": "evt_001", "followup_type": "thank_you"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "followup" in response.data

    @pytest.mark.asyncio
    async def test_event_includes_checklist(self, chip, context):
        """Event creation includes checklist based on event type."""
        request = SkillRequest(
            intent="event_create",
            entities={"event_name": "Workshop", "event_type": "workshop"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "checklist" in response.data["event"]
        # Workshop should have specific items
        checklist_items = [item["item"] for item in response.data["event"]["checklist"]]
        assert any("curriculum" in item.lower() for item in checklist_items)


# ==============================================================================
# MemberServicesChip Tests (10 tests)
# ==============================================================================

class TestMemberServicesChip:
    """Tests for MemberServicesChip."""

    @pytest.fixture
    def chip(self):
        """Create MemberServicesChip instance."""
        return MemberServicesChip()

    def test_chip_name(self, chip):
        """Chip has correct name."""
        assert chip.name == "member_services"

    def test_chip_domain(self, chip):
        """Chip is in MEMBER_SERVICES domain."""
        assert chip.domain == SkillDomain.MEMBER_SERVICES

    def test_efe_weights_stakeholder_benefit(self, chip):
        """EFE weights prioritize stakeholder benefit at 0.35."""
        assert chip.efe_weights.stakeholder_benefit == 0.35

    def test_consensus_actions(self, chip):
        """Chip requires consensus for membership changes and communications."""
        assert "change_membership_tier" in chip.consensus_actions
        assert "process_refund" in chip.consensus_actions
        assert "bulk_communication" in chip.consensus_actions

    @pytest.mark.asyncio
    async def test_handle_member_lookup(self, chip, context):
        """Handle routes member_lookup intent correctly."""
        request = SkillRequest(
            intent="member_lookup",
            entities={"member_id": "mem_001"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "member" in response.data
        assert "membership_tier" in response.data["member"]

    @pytest.mark.asyncio
    async def test_handle_membership_renew(self, chip, context):
        """Handle routes membership_renew intent correctly."""
        request = SkillRequest(
            intent="membership_renew",
            entities={"member_id": "mem_001", "tier": "gold"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "renewal" in response.data
        assert response.data["renewal"]["tier"] == "gold"

    @pytest.mark.asyncio
    async def test_handle_benefits_info(self, chip, context):
        """Handle routes benefits_info intent correctly."""
        request = SkillRequest(
            intent="benefits_info",
            entities={"tier": "silver"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "benefits" in response.data

    @pytest.mark.asyncio
    async def test_handle_member_communicate(self, chip, context):
        """Handle routes member_communicate intent correctly."""
        request = SkillRequest(
            intent="member_communicate",
            entities={"member_id": "mem_001", "template": "renewal_reminder"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "communication" in response.data

    @pytest.mark.asyncio
    async def test_handle_membership_report(self, chip, context):
        """Handle routes membership_report intent correctly."""
        request = SkillRequest(
            intent="membership_report",
            entities={"period": "monthly", "include_trends": True},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "membership_report" in response.data
        assert "trends" in response.data["membership_report"]

    @pytest.mark.asyncio
    async def test_membership_report_includes_recommendations(self, chip, context):
        """Membership report includes actionable recommendations."""
        request = SkillRequest(
            intent="membership_report",
            entities={"period": "monthly"},
        )
        response = await chip.handle(request, context)
        assert response.success
        assert "recommendations" in response.data["membership_report"]
        assert len(response.data["membership_report"]["recommendations"]) > 0


# ==============================================================================
# Cross-chip Integration Tests
# ==============================================================================

class TestCrossChipIntegration:
    """Integration tests across multiple chips."""

    @pytest.mark.asyncio
    async def test_all_chips_have_unique_names(self):
        """All Phase 4b chips have unique names."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]
        names = [chip.name for chip in chips]
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_all_chips_have_valid_efe_weights(self):
        """All Phase 4b chips have valid EFE weights summing to ~1.0."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]
        for chip in chips:
            total = chip.efe_weights.total()
            assert 0.99 <= total <= 1.01, f"{chip.name} EFE weights sum to {total}"

    @pytest.mark.asyncio
    async def test_all_chips_have_version(self):
        """All Phase 4b chips have version set."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]
        for chip in chips:
            assert chip.version is not None
            assert chip.version != ""

    @pytest.mark.asyncio
    async def test_all_chips_have_descriptions(self):
        """All Phase 4b chips have descriptions."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]
        for chip in chips:
            assert chip.description is not None
            assert len(chip.description) > 10

    @pytest.mark.asyncio
    async def test_all_chips_return_failure_for_unknown_intent(self, context):
        """All Phase 4b chips return failure for unknown intents."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]
        request = SkillRequest(intent="completely_unknown_intent", entities={})
        for chip in chips:
            response = await chip.handle(request, context)
            assert response.success is False, f"{chip.name} should fail for unknown intent"


# ==============================================================================
# Response Structure Tests
# ==============================================================================

class TestResponseStructure:
    """Tests for response structure and data validation."""

    @pytest.mark.asyncio
    async def test_program_evaluator_logic_model_structure(self, context):
        """Logic model response has required structure."""
        chip = ProgramEvaluatorChip()
        request = SkillRequest(
            intent="logic_model",
            entities={"program_name": "Test Program"},
        )
        response = await chip.handle(request, context)

        logic_model = response.data["logic_model"]
        assert "model_id" in logic_model
        assert "components" in logic_model
        assert "assumptions" in logic_model
        assert "external_factors" in logic_model

        # Verify component structure
        for component in logic_model["components"]:
            assert "component_id" in component
            assert "type" in component
            assert "description" in component

    @pytest.mark.asyncio
    async def test_board_liaison_meeting_packet_structure(self, context):
        """Meeting packet response has required structure."""
        chip = BoardLiaisonChip()
        request = SkillRequest(
            intent="meeting_prep",
            entities={"meeting_type": "regular"},
        )
        response = await chip.handle(request, context)

        packet = response.data["meeting_packet"]
        assert "packet_id" in packet
        assert "meeting" in packet
        assert "agenda_items" in packet
        assert "attachments" in packet

        # Verify agenda item structure
        for item in packet["agenda_items"]:
            assert "item_number" in item
            assert "title" in item
            assert "presenter" in item

    @pytest.mark.asyncio
    async def test_donor_stewardship_profile_structure(self, context):
        """Donor profile response has required structure."""
        chip = DonorStewardshipChip()
        request = SkillRequest(
            intent="donor_profile",
            entities={"donor_id": "donor_001"},
        )
        response = await chip.handle(request, context)

        profile = response.data["profile"]
        assert "donor_id" in profile
        assert "name" in profile
        assert "email" in profile
        assert "donor_level" in profile
        assert "giving_summary" in profile
        assert "engagement" in profile

    @pytest.mark.asyncio
    async def test_staff_onboarding_plan_structure(self, context):
        """Onboarding plan response has required structure."""
        chip = StaffOnboardingChip()
        request = SkillRequest(
            intent="onboard_start",
            entities={"employee_id": "emp_001", "department": "programs"},
        )
        response = await chip.handle(request, context)

        plan = response.data["onboarding_plan"]
        assert "plan_id" in plan
        assert "employee_id" in plan
        assert "department" in plan
        assert "checklist_items" in plan
        assert "target_completion" in plan

    @pytest.mark.asyncio
    async def test_event_planner_rsvp_structure(self, context):
        """Event RSVP response has required structure."""
        chip = EventPlannerChip()
        request = SkillRequest(
            intent="event_rsvp",
            entities={"event_id": "evt_001"},
        )
        response = await chip.handle(request, context)

        rsvp_data = response.data["rsvp_data"]
        assert "event_id" in rsvp_data
        assert "rsvps" in rsvp_data
        assert "summary" in rsvp_data
        assert "special_needs" in rsvp_data

        # Verify summary structure
        summary = rsvp_data["summary"]
        assert "confirmed" in summary
        assert "tentative" in summary
        assert "declined" in summary

    @pytest.mark.asyncio
    async def test_member_services_membership_structure(self, context):
        """Member lookup response has required structure."""
        chip = MemberServicesChip()
        request = SkillRequest(
            intent="member_lookup",
            entities={"member_id": "mem_001"},
        )
        response = await chip.handle(request, context)

        member = response.data["member"]
        assert "member_id" in member
        assert "name" in member
        assert "membership_tier" in member
        assert "status" in member
        assert "tier_benefits" in member
        assert "engagement" in member


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_program_evaluator_empty_entities(self, context):
        """ProgramEvaluator handles empty entities gracefully."""
        chip = ProgramEvaluatorChip()
        request = SkillRequest(intent="logic_model", entities={})
        response = await chip.handle(request, context)
        assert response.success
        # Should use default program name
        assert response.data["logic_model"]["program_name"] == "Unnamed Program"

    @pytest.mark.asyncio
    async def test_member_services_invalid_tier(self, context):
        """MemberServices handles invalid tier gracefully."""
        chip = MemberServicesChip()
        request = SkillRequest(
            intent="membership_renew",
            entities={"member_id": "mem_001", "tier": "invalid_tier"},
        )
        response = await chip.handle(request, context)
        assert response.success is False

    @pytest.mark.asyncio
    async def test_staff_onboarding_incomplete_completion(self, context):
        """StaffOnboarding handles incomplete completion attempt."""
        chip = StaffOnboardingChip()
        request = SkillRequest(
            intent="onboard_complete",
            entities={"employee_id": "emp_001", "force_complete": False},
        )
        response = await chip.handle(request, context)
        # Response might fail due to incomplete items
        assert "completion_status" in response.data or "completion_record" in response.data

    @pytest.mark.asyncio
    async def test_board_liaison_resolution_filtering(self, context):
        """BoardLiaison filters resolutions by status."""
        chip = BoardLiaisonChip()
        request = SkillRequest(
            intent="resolution_track",
            entities={"status": "pending"},
        )
        response = await chip.handle(request, context)
        assert response.success
        # All returned resolutions should be pending
        for resolution in response.data["resolutions"]:
            assert resolution["status"] == "pending"

    @pytest.mark.asyncio
    async def test_donor_stewardship_tier_assignment(self, context):
        """DonorStewardship assigns correct acknowledgment tier based on amount."""
        chip = DonorStewardshipChip()

        # Test champion tier (>= $10,000)
        request = SkillRequest(
            intent="donor_thank",
            entities={"donor_id": "donor_001", "gift_amount": 15000},
        )
        response = await chip.handle(request, context)
        assert response.data["acknowledgment"]["tier"] == "champion"

        # Test supporter tier (< $1,000)
        request = SkillRequest(
            intent="donor_thank",
            entities={"donor_id": "donor_002", "gift_amount": 500},
        )
        response = await chip.handle(request, context)
        assert response.data["acknowledgment"]["tier"] == "supporter"

    @pytest.mark.asyncio
    async def test_event_planner_accessibility_recommendations(self, context):
        """EventPlanner includes accessibility recommendations."""
        chip = EventPlannerChip()
        request = SkillRequest(
            intent="event_accessibility",
            entities={"event_id": "evt_001"},
        )
        response = await chip.handle(request, context)
        assert "recommendations" in response.data["accessibility"]
        assert len(response.data["accessibility"]["recommendations"]) > 0


# ==============================================================================
# Consensus and Security Tests
# ==============================================================================

class TestConsensusAndSecurity:
    """Tests for consensus requirements and security features."""

    @pytest.mark.asyncio
    async def test_program_evaluator_requires_consensus_for_publish(self, context):
        """ProgramEvaluator requires consensus for publishing findings."""
        chip = ProgramEvaluatorChip()
        assert chip.requires_consensus("publish_findings") is True
        assert chip.requires_consensus("finalize_evaluation") is True
        assert chip.requires_consensus("random_action") is False

    @pytest.mark.asyncio
    async def test_board_liaison_requires_consensus_for_bylaws(self, context):
        """BoardLiaison requires consensus for bylaw updates."""
        chip = BoardLiaisonChip()
        assert chip.requires_consensus("update_bylaws") is True
        assert chip.requires_consensus("record_resolution") is True

    @pytest.mark.asyncio
    async def test_donor_stewardship_requires_consensus_for_sharing(self, context):
        """DonorStewardship requires consensus for sharing donor data."""
        chip = DonorStewardshipChip()
        assert chip.requires_consensus("share_donor_data") is True

    @pytest.mark.asyncio
    async def test_staff_onboarding_requires_consensus_for_access(self, context):
        """StaffOnboarding requires consensus for granting system access."""
        chip = StaffOnboardingChip()
        assert chip.requires_consensus("grant_system_access") is True
        assert chip.requires_consensus("complete_onboarding") is True

    @pytest.mark.asyncio
    async def test_event_planner_requires_consensus_for_budget(self, context):
        """EventPlanner requires consensus for budget commitment."""
        chip = EventPlannerChip()
        assert chip.requires_consensus("commit_budget") is True
        assert chip.requires_consensus("finalize_event") is True

    @pytest.mark.asyncio
    async def test_member_services_requires_consensus_for_bulk(self, context):
        """MemberServices requires consensus for bulk communications."""
        chip = MemberServicesChip()
        assert chip.requires_consensus("bulk_communication") is True
        assert chip.requires_consensus("process_refund") is True
        assert chip.requires_consensus("change_membership_tier") is True


# ==============================================================================
# get_info() Tests
# ==============================================================================

class TestGetInfo:
    """Tests for chip metadata via get_info()."""

    def test_program_evaluator_get_info(self):
        """ProgramEvaluator returns complete info dict."""
        chip = ProgramEvaluatorChip()
        info = chip.get_info()

        assert info["name"] == "program_evaluator"
        assert info["domain"] == "programs"
        assert "version" in info
        assert "efe_weights" in info
        assert "required_spans" in info
        assert "consensus_actions" in info
        assert "capabilities" in info

    def test_all_chips_get_info_consistent(self):
        """All chips return consistent info structure."""
        chips = [
            ProgramEvaluatorChip(),
            BoardLiaisonChip(),
            DonorStewardshipChip(),
            StaffOnboardingChip(),
            EventPlannerChip(),
            MemberServicesChip(),
        ]

        required_keys = ["name", "description", "version", "domain",
                        "efe_weights", "required_spans", "consensus_actions", "capabilities"]

        for chip in chips:
            info = chip.get_info()
            for key in required_keys:
                assert key in info, f"{chip.name} missing {key} in get_info()"
