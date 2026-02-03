"""
Comprehensive tests for Kintsugi CMA skills infrastructure.

Tests the base classes, registry, and router from:
- kintsugi/skills/base.py
- kintsugi/skills/registry.py
- kintsugi/skills/router.py
"""

import pytest
from datetime import datetime, timezone
from typing import Any

from kintsugi.skills import (
    BaseSkillChip,
    EFEWeights,
    SkillCapability,
    SkillContext,
    SkillDomain,
    SkillHandler,
    SkillRequest,
    SkillResponse,
    SkillRegistry,
    SkillRouter,
    RouterConfig,
    RouteMatch,
    get_registry,
    register_chip,
    reset_registry,
    create_router,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def clean_registry():
    """Ensure clean registry state for each test."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def sample_context():
    """Create a sample SkillContext for testing."""
    return SkillContext(
        org_id="org_test_123",
        user_id="user_test_456",
        session_id="session_789",
        platform="slack",
        channel_id="C123456",
        beliefs=[{"type": "budget_status", "value": "healthy"}],
        desires=[{"type": "funding_goal", "value": 100000}],
        intentions=[{"action": "apply_for_grant"}],
    )


@pytest.fixture
def sample_request():
    """Create a sample SkillRequest for testing."""
    return SkillRequest(
        intent="test_intent",
        entities={"key": "value", "amount": 1000},
        raw_input="This is a test request",
        parameters={"verbose": True},
    )


class ConcreteSkillChip(BaseSkillChip):
    """Concrete implementation of BaseSkillChip for testing."""

    name = "test_chip"
    description = "A test skill chip"
    version = "1.0.0"
    domain = SkillDomain.OPERATIONS

    efe_weights = EFEWeights(
        mission_alignment=0.30,
        stakeholder_benefit=0.25,
        resource_efficiency=0.20,
        transparency=0.15,
        equity=0.10,
    )

    capabilities = [SkillCapability.READ_DATA, SkillCapability.WRITE_DATA]
    consensus_actions = ["approve_action", "delete_action"]
    required_spans = ["test_api"]

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        return SkillResponse(
            content=f"Handled {request.intent}",
            success=True,
            data={"processed": True},
        )


class AnotherSkillChip(BaseSkillChip):
    """Another concrete skill chip for registry tests."""

    name = "another_chip"
    description = "Another test chip"
    version = "1.0.0"
    domain = SkillDomain.FUNDRAISING

    async def handle(self, request: SkillRequest, context: SkillContext) -> SkillResponse:
        return SkillResponse(content="Another response")


@pytest.fixture
def test_chip():
    """Create a test chip instance."""
    return ConcreteSkillChip()


@pytest.fixture
def another_chip():
    """Create another test chip instance."""
    return AnotherSkillChip()


# ============================================================================
# SkillDomain Tests (3 tests)
# ============================================================================


class TestSkillDomain:
    """Tests for SkillDomain enum."""

    def test_all_ten_domain_values_exist(self):
        """All 10 domain values should exist."""
        expected_domains = [
            "FUNDRAISING",
            "OPERATIONS",
            "PROGRAMS",
            "COMMUNICATIONS",
            "FINANCE",
            "GOVERNANCE",
            "COMMUNITY",
            "MUTUAL_AID",
            "ADVOCACY",
            "MEMBER_SERVICES",
        ]
        actual_domains = [d.name for d in SkillDomain]
        assert len(actual_domains) == 10
        for domain in expected_domains:
            assert domain in actual_domains

    def test_domain_values_are_strings(self):
        """Domain values should be strings."""
        for domain in SkillDomain:
            assert isinstance(domain.value, str)
            assert len(domain.value) > 0

    def test_enum_membership_works(self):
        """Enum membership should work correctly."""
        assert SkillDomain.FUNDRAISING == SkillDomain("fundraising")
        assert SkillDomain.OPERATIONS in SkillDomain
        assert "OPERATIONS" in SkillDomain.__members__


# ============================================================================
# EFEWeights Tests (8 tests)
# ============================================================================


class TestEFEWeights:
    """Tests for EFEWeights dataclass."""

    def test_creation_with_defaults(self):
        """EFEWeights should be creatable with default values."""
        weights = EFEWeights()
        assert weights.mission_alignment == 0.25
        assert weights.stakeholder_benefit == 0.25
        assert weights.resource_efficiency == 0.20
        assert weights.transparency == 0.15
        assert weights.equity == 0.15

    def test_creation_with_custom_values(self):
        """EFEWeights should accept custom values."""
        weights = EFEWeights(
            mission_alignment=0.35,
            stakeholder_benefit=0.30,
            resource_efficiency=0.15,
            transparency=0.10,
            equity=0.10,
        )
        assert weights.mission_alignment == 0.35
        assert weights.stakeholder_benefit == 0.30
        assert weights.resource_efficiency == 0.15
        assert weights.transparency == 0.10
        assert weights.equity == 0.10

    def test_validation_values_must_be_between_0_and_1(self):
        """Values must be between 0.0 and 1.0."""
        # Valid boundary values
        weights = EFEWeights(
            mission_alignment=0.0,
            stakeholder_benefit=1.0,
            resource_efficiency=0.5,
            transparency=0.0,
            equity=0.5,
        )
        assert weights.mission_alignment == 0.0
        assert weights.stakeholder_benefit == 1.0

    def test_validation_negative_values_rejected(self):
        """Negative values should be rejected."""
        with pytest.raises(ValueError, match="mission_alignment must be between 0.0 and 1.0"):
            EFEWeights(mission_alignment=-0.1)

        with pytest.raises(ValueError, match="equity must be between 0.0 and 1.0"):
            EFEWeights(equity=-0.5)

    def test_validation_values_over_1_rejected(self):
        """Values greater than 1.0 should be rejected."""
        with pytest.raises(ValueError, match="stakeholder_benefit must be between 0.0 and 1.0"):
            EFEWeights(stakeholder_benefit=1.1)

        with pytest.raises(ValueError, match="transparency must be between 0.0 and 1.0"):
            EFEWeights(transparency=2.0)

    def test_to_dict_returns_correct_structure(self):
        """to_dict() should return correct dictionary structure."""
        weights = EFEWeights(
            mission_alignment=0.30,
            stakeholder_benefit=0.25,
            resource_efficiency=0.20,
            transparency=0.15,
            equity=0.10,
        )
        result = weights.to_dict()

        assert isinstance(result, dict)
        assert result == {
            "mission_alignment": 0.30,
            "stakeholder_benefit": 0.25,
            "resource_efficiency": 0.20,
            "transparency": 0.15,
            "equity": 0.10,
        }

    def test_all_five_weight_fields_exist(self):
        """All five weight fields should exist."""
        weights = EFEWeights()
        assert hasattr(weights, "mission_alignment")
        assert hasattr(weights, "stakeholder_benefit")
        assert hasattr(weights, "resource_efficiency")
        assert hasattr(weights, "transparency")
        assert hasattr(weights, "equity")

    def test_total_calculates_sum(self):
        """total() should calculate sum of all weights."""
        weights = EFEWeights(
            mission_alignment=0.20,
            stakeholder_benefit=0.20,
            resource_efficiency=0.20,
            transparency=0.20,
            equity=0.20,
        )
        assert weights.total() == 1.0


# ============================================================================
# SkillContext Tests (6 tests)
# ============================================================================


class TestSkillContext:
    """Tests for SkillContext dataclass."""

    def test_creation_with_required_fields(self):
        """SkillContext should be creatable with only required fields."""
        context = SkillContext(org_id="org_123", user_id="user_456")
        assert context.org_id == "org_123"
        assert context.user_id == "user_456"

    def test_default_timestamp_is_now(self):
        """Default timestamp should be approximately now."""
        before = datetime.now(timezone.utc)
        context = SkillContext(org_id="org_123", user_id="user_456")
        after = datetime.now(timezone.utc)

        assert before <= context.timestamp <= after
        assert context.timestamp.tzinfo == timezone.utc

    def test_empty_bdi_lists_by_default(self):
        """BDI lists should be empty by default."""
        context = SkillContext(org_id="org_123", user_id="user_456")
        assert context.beliefs == []
        assert context.desires == []
        assert context.intentions == []

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata should default to empty dict."""
        context = SkillContext(org_id="org_123", user_id="user_456")
        assert context.metadata == {}
        assert isinstance(context.metadata, dict)

    def test_session_id_is_optional(self):
        """session_id should be optional and default to None."""
        context = SkillContext(org_id="org_123", user_id="user_456")
        assert context.session_id is None

        context_with_session = SkillContext(
            org_id="org_123",
            user_id="user_456",
            session_id="session_789",
        )
        assert context_with_session.session_id == "session_789"

    def test_platform_is_optional(self):
        """platform should be optional and default to None."""
        context = SkillContext(org_id="org_123", user_id="user_456")
        assert context.platform is None

        context_with_platform = SkillContext(
            org_id="org_123",
            user_id="user_456",
            platform="slack",
        )
        assert context_with_platform.platform == "slack"


# ============================================================================
# SkillRequest Tests (5 tests)
# ============================================================================


class TestSkillRequest:
    """Tests for SkillRequest dataclass."""

    def test_creation_with_intent(self):
        """SkillRequest should be creatable with just intent."""
        request = SkillRequest(intent="grant_search")
        assert request.intent == "grant_search"

    def test_entities_defaults_to_empty_dict(self):
        """entities should default to empty dict."""
        request = SkillRequest(intent="test")
        assert request.entities == {}
        assert isinstance(request.entities, dict)

    def test_raw_input_defaults_to_empty_string(self):
        """raw_input should default to empty string."""
        request = SkillRequest(intent="test")
        assert request.raw_input == ""

    def test_parameters_defaults_to_empty_dict(self):
        """parameters should default to empty dict."""
        request = SkillRequest(intent="test")
        assert request.parameters == {}
        assert isinstance(request.parameters, dict)

    def test_creation_with_all_fields(self):
        """SkillRequest should accept all fields."""
        request = SkillRequest(
            intent="grant_search",
            entities={"amount": 10000, "focus_area": "education"},
            raw_input="Find grants over $10k for education",
            parameters={"limit": 5},
        )
        assert request.intent == "grant_search"
        assert request.entities == {"amount": 10000, "focus_area": "education"}
        assert request.raw_input == "Find grants over $10k for education"
        assert request.parameters == {"limit": 5}


# ============================================================================
# SkillResponse Tests (8 tests)
# ============================================================================


class TestSkillResponse:
    """Tests for SkillResponse dataclass."""

    def test_creation_with_content(self):
        """SkillResponse should be creatable with just content."""
        response = SkillResponse(content="Hello, world!")
        assert response.content == "Hello, world!"

    def test_success_defaults_to_true(self):
        """success should default to True."""
        response = SkillResponse(content="Test")
        assert response.success is True

    def test_data_defaults_to_empty_dict(self):
        """data should default to empty dict."""
        response = SkillResponse(content="Test")
        assert response.data == {}
        assert isinstance(response.data, dict)

    def test_suggestions_defaults_to_empty_list(self):
        """suggestions should default to empty list."""
        response = SkillResponse(content="Test")
        assert response.suggestions == []
        assert isinstance(response.suggestions, list)

    def test_requires_consensus_defaults_to_false(self):
        """requires_consensus should default to False."""
        response = SkillResponse(content="Test")
        assert response.requires_consensus is False

    def test_attachments_defaults_to_empty_list(self):
        """attachments should default to empty list."""
        response = SkillResponse(content="Test")
        assert response.attachments == []
        assert isinstance(response.attachments, list)

    def test_metadata_defaults_to_empty_dict(self):
        """metadata should default to empty dict."""
        response = SkillResponse(content="Test")
        assert response.metadata == {}
        assert isinstance(response.metadata, dict)

    def test_creation_with_all_fields(self):
        """SkillResponse should accept all fields."""
        response = SkillResponse(
            content="Found 5 grants",
            success=True,
            data={"grants": [], "total": 5},
            suggestions=["Apply for top grant?"],
            requires_consensus=True,
            consensus_action="submit_application",
            attachments=[{"type": "pdf", "name": "report.pdf"}],
            metadata={"source": "grants.gov"},
        )
        assert response.content == "Found 5 grants"
        assert response.success is True
        assert response.data == {"grants": [], "total": 5}
        assert response.suggestions == ["Apply for top grant?"]
        assert response.requires_consensus is True
        assert response.consensus_action == "submit_application"
        assert response.attachments == [{"type": "pdf", "name": "report.pdf"}]
        assert response.metadata == {"source": "grants.gov"}


# ============================================================================
# SkillCapability Tests (2 tests)
# ============================================================================


class TestSkillCapability:
    """Tests for SkillCapability enum."""

    def test_all_eight_capability_values_exist(self):
        """All 8 capability values should exist."""
        expected_capabilities = [
            "READ_DATA",
            "WRITE_DATA",
            "EXTERNAL_API",
            "SEND_NOTIFICATIONS",
            "FINANCIAL_OPERATIONS",
            "PII_ACCESS",
            "SCHEDULE_TASKS",
            "GENERATE_REPORTS",
        ]
        actual_capabilities = [c.name for c in SkillCapability]
        assert len(actual_capabilities) == 8
        for cap in expected_capabilities:
            assert cap in actual_capabilities

    def test_capability_values_are_strings(self):
        """Capability values should be strings."""
        for cap in SkillCapability:
            assert isinstance(cap.value, str)
            assert len(cap.value) > 0


# ============================================================================
# BaseSkillChip Tests (10 tests)
# ============================================================================


class TestBaseSkillChip:
    """Tests for BaseSkillChip abstract base class."""

    def test_cannot_instantiate_directly(self):
        """BaseSkillChip should not be instantiable directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSkillChip()

    def test_concrete_subclass_can_be_instantiated(self, test_chip):
        """Concrete subclass should be instantiable."""
        assert isinstance(test_chip, BaseSkillChip)
        assert isinstance(test_chip, ConcreteSkillChip)

    def test_get_info_returns_correct_structure(self, test_chip):
        """get_info() should return correct dictionary structure."""
        info = test_chip.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "test_chip"
        assert info["description"] == "A test skill chip"
        assert info["version"] == "1.0.0"
        assert info["domain"] == "operations"
        assert "efe_weights" in info
        assert isinstance(info["efe_weights"], dict)
        assert info["required_spans"] == ["test_api"]
        assert info["consensus_actions"] == ["approve_action", "delete_action"]
        assert "read_data" in info["capabilities"]
        assert "write_data" in info["capabilities"]

    def test_requires_consensus_checks_consensus_actions_list(self, test_chip):
        """requires_consensus() should check the consensus_actions list."""
        assert test_chip.requires_consensus("approve_action") is True
        assert test_chip.requires_consensus("delete_action") is True
        assert test_chip.requires_consensus("other_action") is False
        assert test_chip.requires_consensus("") is False

    @pytest.mark.asyncio
    async def test_get_bdi_context_default_implementation(self, test_chip):
        """get_bdi_context() default should return all BDI state unfiltered."""
        beliefs = [{"type": "budget", "value": 1000}]
        desires = [{"type": "goal", "value": "success"}]
        intentions = [{"action": "apply"}]

        result = await test_chip.get_bdi_context(beliefs, desires, intentions)

        assert result["beliefs"] == beliefs
        assert result["desires"] == desires
        assert result["intentions"] == intentions

    def test_class_attributes_name_works(self, test_chip):
        """name attribute should work correctly."""
        assert test_chip.name == "test_chip"

    def test_class_attributes_description_works(self, test_chip):
        """description attribute should work correctly."""
        assert test_chip.description == "A test skill chip"

    def test_class_attributes_domain_works(self, test_chip):
        """domain attribute should work correctly."""
        assert test_chip.domain == SkillDomain.OPERATIONS

    @pytest.mark.asyncio
    async def test_handle_method_works(self, test_chip, sample_request, sample_context):
        """handle() method should work correctly."""
        response = await test_chip.handle(sample_request, sample_context)

        assert isinstance(response, SkillResponse)
        assert response.success is True
        assert "Handled test_intent" in response.content

    def test_efe_weights_initialized(self, test_chip):
        """efe_weights should be properly initialized."""
        assert test_chip.efe_weights is not None
        assert isinstance(test_chip.efe_weights, EFEWeights)
        assert test_chip.efe_weights.mission_alignment == 0.30


# ============================================================================
# SkillRegistry Tests (12 tests)
# ============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry class."""

    def test_registration_adds_chip(self, clean_registry, test_chip):
        """register() should add a chip to the registry."""
        registry = SkillRegistry()
        registry.register(test_chip)

        assert "test_chip" in registry
        assert registry.get("test_chip") is test_chip

    def test_registration_of_duplicate_name_raises_valueerror(self, clean_registry, test_chip):
        """register() should raise ValueError for duplicate names."""
        registry = SkillRegistry()
        registry.register(test_chip)

        with pytest.raises(ValueError, match="Chip 'test_chip' already registered"):
            registry.register(test_chip)

    def test_unregister_removes_chip(self, clean_registry, test_chip):
        """unregister() should remove the chip."""
        registry = SkillRegistry()
        registry.register(test_chip)

        result = registry.unregister("test_chip")

        assert result is True
        assert "test_chip" not in registry

    def test_unregister_returns_false_for_unknown(self, clean_registry):
        """unregister() should return False for unknown chip."""
        registry = SkillRegistry()
        result = registry.unregister("nonexistent_chip")
        assert result is False

    def test_get_retrieves_chip_by_name(self, clean_registry, test_chip):
        """get() should retrieve chip by name."""
        registry = SkillRegistry()
        registry.register(test_chip)

        retrieved = registry.get("test_chip")
        assert retrieved is test_chip

    def test_get_returns_none_for_unknown(self, clean_registry):
        """get() should return None for unknown chip."""
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_get_by_domain_returns_chips_in_domain(self, clean_registry, test_chip, another_chip):
        """get_by_domain() should return all chips in a domain."""
        registry = SkillRegistry()
        registry.register(test_chip)
        registry.register(another_chip)

        operations_chips = registry.get_by_domain(SkillDomain.OPERATIONS)
        fundraising_chips = registry.get_by_domain(SkillDomain.FUNDRAISING)

        assert len(operations_chips) == 1
        assert test_chip in operations_chips

        assert len(fundraising_chips) == 1
        assert another_chip in fundraising_chips

    def test_get_by_domain_returns_empty_for_unknown_domain(self, clean_registry, test_chip):
        """get_by_domain() should return empty list for domain with no chips."""
        registry = SkillRegistry()
        registry.register(test_chip)

        result = registry.get_by_domain(SkillDomain.GOVERNANCE)
        assert result == []

    def test_list_all_returns_chip_info_dicts(self, clean_registry, test_chip, another_chip):
        """list_all() should return list of chip info dictionaries."""
        registry = SkillRegistry()
        registry.register(test_chip)
        registry.register(another_chip)

        all_info = registry.list_all()

        assert len(all_info) == 2
        assert all(isinstance(info, dict) for info in all_info)
        names = [info["name"] for info in all_info]
        assert "test_chip" in names
        assert "another_chip" in names

    def test_list_names_returns_name_strings(self, clean_registry, test_chip, another_chip):
        """list_names() should return list of chip name strings."""
        registry = SkillRegistry()
        registry.register(test_chip)
        registry.register(another_chip)

        names = registry.list_names()

        assert len(names) == 2
        assert all(isinstance(name, str) for name in names)
        assert "test_chip" in names
        assert "another_chip" in names

    def test_len_returns_chip_count(self, clean_registry, test_chip, another_chip):
        """len() should return the number of registered chips."""
        registry = SkillRegistry()
        assert len(registry) == 0

        registry.register(test_chip)
        assert len(registry) == 1

        registry.register(another_chip)
        assert len(registry) == 2

    def test_contains_works(self, clean_registry, test_chip):
        """__contains__ should work for checking chip registration."""
        registry = SkillRegistry()

        assert "test_chip" not in registry

        registry.register(test_chip)

        assert "test_chip" in registry
        assert "nonexistent" not in registry


# ============================================================================
# SkillRouter Tests (10 tests)
# ============================================================================


class TestSkillRouter:
    """Tests for SkillRouter class."""

    def test_registration_of_intent_to_chip(self, clean_registry, test_chip):
        """register_intent() should map intent to chip."""
        registry = SkillRegistry()
        registry.register(test_chip)
        router = SkillRouter(registry)

        router.register_intent("test_action", "test_chip")

        intents = router.get_intents_for_chip("test_chip")
        assert "test_action" in intents

    def test_registration_to_unknown_chip_raises_valueerror(self, clean_registry):
        """register_intent() should raise ValueError for unknown chip."""
        registry = SkillRegistry()
        router = SkillRouter(registry)

        with pytest.raises(ValueError, match="Chip 'nonexistent' not in registry"):
            router.register_intent("some_intent", "nonexistent")

    def test_route_exact_match_returns_confidence_1(self, clean_registry, test_chip):
        """route() exact match should return confidence 1.0."""
        registry = SkillRegistry()
        registry.register(test_chip)
        router = SkillRouter(registry)
        router.register_intent("exact_intent", "test_chip")

        match = router.route("exact_intent")

        assert match is not None
        assert match.chip is test_chip
        assert match.confidence == 1.0
        assert match.matched_intent == "exact_intent"

    def test_route_prefix_match_with_wildcard_returns_confidence_0_9(self, clean_registry, test_chip):
        """route() prefix match with * should return confidence 0.9."""
        registry = SkillRegistry()
        registry.register(test_chip)
        router = SkillRouter(registry)
        router.register_intent("grant_*", "test_chip")

        match = router.route("grant_search")

        assert match is not None
        assert match.chip is test_chip
        assert match.confidence == 0.9
        assert match.matched_intent == "grant_*"

    def test_route_no_match_returns_none(self, clean_registry, test_chip):
        """route() with no match should return None."""
        registry = SkillRegistry()
        registry.register(test_chip)
        router = SkillRouter(registry)
        router.register_intent("other_intent", "test_chip")

        match = router.route("nonexistent_intent")

        assert match is None

    def test_route_fallback_returns_min_confidence(self, clean_registry, test_chip):
        """route() fallback should return min_confidence from config."""
        registry = SkillRegistry()
        registry.register(test_chip)
        config = RouterConfig(min_confidence=0.5, fallback_chip="test_chip")
        router = SkillRouter(registry, config)

        match = router.route("unknown_intent")

        assert match is not None
        assert match.chip is test_chip
        assert match.confidence == 0.5
        assert match.matched_intent == "fallback"

    def test_get_intents_for_chip_returns_mapped_intents(self, clean_registry, test_chip):
        """get_intents_for_chip() should return all mapped intents."""
        registry = SkillRegistry()
        registry.register(test_chip)
        router = SkillRouter(registry)
        router.register_intent("intent_1", "test_chip")
        router.register_intent("intent_2", "test_chip")
        router.register_intent("grant_*", "test_chip")

        intents = router.get_intents_for_chip("test_chip")

        assert len(intents) == 3
        assert "intent_1" in intents
        assert "intent_2" in intents
        assert "grant_*" in intents

    def test_router_config_defaults(self):
        """RouterConfig should have sensible defaults."""
        config = RouterConfig()

        assert config.min_confidence == 0.5
        assert config.fallback_chip is None
        assert config.enable_fuzzy_matching is False

    def test_create_router_factory_function(self, clean_registry, test_chip):
        """create_router() should create a configured router."""
        registry = SkillRegistry()
        registry.register(test_chip)

        router = create_router(registry=registry)

        assert isinstance(router, SkillRouter)

    def test_create_router_with_config(self, clean_registry, test_chip):
        """create_router() should accept custom config."""
        registry = SkillRegistry()
        registry.register(test_chip)
        config = RouterConfig(min_confidence=0.7, fallback_chip="test_chip")

        router = create_router(registry=registry, config=config)

        # Verify fallback works with the custom config
        match = router.route("unknown")
        assert match is not None
        assert match.confidence == 0.7


# ============================================================================
# Global Registry Functions Tests
# ============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_get_registry_returns_singleton(self, clean_registry):
        """get_registry() should return the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_register_chip_adds_to_global_registry(self, clean_registry, test_chip):
        """register_chip() should add chip to global registry."""
        register_chip(test_chip)

        registry = get_registry()
        assert "test_chip" in registry

    def test_reset_registry_clears_state(self, clean_registry, test_chip):
        """reset_registry() should clear all chips."""
        register_chip(test_chip)
        registry = get_registry()
        assert len(registry) == 1

        reset_registry()

        # After reset, getting the registry should give a fresh one
        new_registry = get_registry()
        assert len(new_registry) == 0


# ============================================================================
# RouteMatch Tests
# ============================================================================


class TestRouteMatch:
    """Tests for RouteMatch dataclass."""

    def test_route_match_creation(self, test_chip):
        """RouteMatch should be creatable with all fields."""
        match = RouteMatch(
            chip=test_chip,
            confidence=0.95,
            matched_intent="test_intent",
        )

        assert match.chip is test_chip
        assert match.confidence == 0.95
        assert match.matched_intent == "test_intent"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_routing_flow(self, clean_registry, sample_context):
        """Test complete flow from registration to handling."""
        # Register chip
        chip = ConcreteSkillChip()
        registry = get_registry()
        registry.register(chip)

        # Set up router
        router = SkillRouter(registry)
        router.register_intent("test_action", "test_chip")
        router.register_intent("test_*", "test_chip")

        # Route and handle
        request = SkillRequest(intent="test_action", entities={"key": "value"})
        match = router.route("test_action")

        assert match is not None
        response = await match.chip.handle(request, sample_context)

        assert response.success is True
        assert "Handled test_action" in response.content

    @pytest.mark.asyncio
    async def test_wildcard_routing_flow(self, clean_registry, sample_context):
        """Test wildcard routing flow."""
        chip = ConcreteSkillChip()
        registry = get_registry()
        registry.register(chip)

        router = SkillRouter(registry)
        router.register_intent("test_*", "test_chip")

        # Should match test_foo via wildcard
        match = router.route("test_foo")

        assert match is not None
        assert match.confidence == 0.9

        request = SkillRequest(intent="test_foo")
        response = await match.chip.handle(request, sample_context)
        assert response.success is True
