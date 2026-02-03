"""
Intent routing to skill chips.

This module provides the SkillRouter class for routing user intents
to the appropriate skill chips. The router supports:
- Exact intent matching
- Prefix/wildcard matching for intent families
- Configurable fallback behavior
- Confidence scoring for matches

Usage:
    from kintsugi.skills.router import SkillRouter, RouterConfig
    from kintsugi.skills.registry import get_registry

    router = SkillRouter(get_registry())
    router.register_intent("grant_search", "grant_search_chip")
    router.register_intent("grant_*", "grant_search_chip")  # Wildcard

    match = router.route("grant_search")
    if match:
        response = await match.chip.handle(request, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseSkillChip
from .registry import SkillRegistry


@dataclass
class RouteMatch:
    """Result of routing an intent to a chip.

    Contains the matched chip, confidence score, and the intent
    pattern that matched. Confidence scoring allows the orchestrator
    to decide whether to proceed with a match or request clarification.

    Attributes:
        chip: The matched skill chip instance
        confidence: Match confidence from 0.0 to 1.0
        matched_intent: The intent pattern that matched

    Example:
        match = router.route("grant_search")
        if match and match.confidence >= 0.8:
            response = await match.chip.handle(request, context)
        elif match:
            # Low confidence, maybe ask for clarification
            pass
    """
    chip: BaseSkillChip
    confidence: float  # 0.0-1.0
    matched_intent: str


@dataclass
class RouterConfig:
    """Configuration for the skill router.

    Attributes:
        min_confidence: Minimum confidence threshold for returning matches.
            Routes with confidence below this threshold return None.
        fallback_chip: Name of chip to use when no other match is found.
            Must be registered in the registry.
        enable_fuzzy_matching: Whether to enable fuzzy string matching
            for intents (future enhancement).

    Example:
        config = RouterConfig(
            min_confidence=0.6,
            fallback_chip="general_assistant",
        )
        router = SkillRouter(registry, config)
    """
    min_confidence: float = 0.5
    fallback_chip: str | None = None
    enable_fuzzy_matching: bool = False  # Reserved for future use


class SkillRouter:
    """Routes intents to appropriate skill chips.

    The router maintains a mapping of intent patterns to chip names
    and provides intent resolution with confidence scoring. Supports:

    - **Exact matching**: "grant_search" -> "grant_search_chip"
    - **Prefix/wildcard matching**: "grant_*" matches "grant_search", "grant_apply"
    - **Fallback routing**: Configurable default chip for unmatched intents

    Match confidence levels:
    - 1.0: Exact intent match
    - 0.9: Prefix/wildcard match
    - min_confidence: Fallback match

    Attributes:
        _registry: The skill registry to look up chips
        _config: Router configuration
        _intent_map: Mapping of intent patterns to chip names

    Example:
        registry = get_registry()
        router = SkillRouter(registry)

        # Register intent mappings
        router.register_intent("grant_search", "grant_chip")
        router.register_intent("grant_apply", "grant_chip")
        router.register_intent("donor_*", "donor_chip")  # Wildcard

        # Route an intent
        match = router.route("grant_search")
        if match:
            response = await match.chip.handle(request, context)
    """

    def __init__(
        self,
        registry: SkillRegistry,
        config: RouterConfig | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            registry: The skill registry to look up chips from
            config: Optional router configuration
        """
        self._registry = registry
        self._config = config or RouterConfig()
        self._intent_map: dict[str, str] = {}  # intent -> chip_name

    def register_intent(self, intent: str, chip_name: str) -> None:
        """Map an intent to a chip.

        Registers an intent pattern to route to a specific chip.
        Supports exact intents and wildcard patterns (ending with *).

        Args:
            intent: The intent pattern to match. Use * suffix for
                prefix matching (e.g., "grant_*" matches "grant_search")
            chip_name: The name of the chip to route to

        Raises:
            ValueError: If the chip is not registered in the registry

        Example:
            router.register_intent("grant_search", "grant_chip")
            router.register_intent("grant_*", "grant_chip")  # Prefix match
        """
        if chip_name not in self._registry:
            raise ValueError(f"Chip '{chip_name}' not in registry")
        self._intent_map[intent] = chip_name

    def unregister_intent(self, intent: str) -> bool:
        """Remove an intent mapping.

        Args:
            intent: The intent pattern to remove

        Returns:
            True if the intent was found and removed, False otherwise
        """
        if intent in self._intent_map:
            del self._intent_map[intent]
            return True
        return False

    def route(self, intent: str) -> RouteMatch | None:
        """Route an intent to a chip.

        Attempts to match the intent using the following priority:
        1. Exact match (confidence 1.0)
        2. Prefix/wildcard match (confidence 0.9)
        3. Fallback chip if configured (confidence = min_confidence)

        Args:
            intent: The intent to route

        Returns:
            RouteMatch if found with confidence >= min_confidence,
            None if no match or all matches below threshold

        Example:
            match = router.route("grant_search")
            if match:
                print(f"Matched {match.chip.name} with confidence {match.confidence}")
                response = await match.chip.handle(request, context)
            else:
                print("No matching chip found")
        """
        # Exact match
        if intent in self._intent_map:
            chip_name = self._intent_map[intent]
            chip = self._registry.get(chip_name)
            if chip is not None:
                return RouteMatch(
                    chip=chip,
                    confidence=1.0,
                    matched_intent=intent,
                )

        # Prefix match (e.g., "grant_search" matches "grant_*")
        for mapped_intent, chip_name in self._intent_map.items():
            if mapped_intent.endswith('*'):
                prefix = mapped_intent[:-1]
                if intent.startswith(prefix):
                    chip = self._registry.get(chip_name)
                    if chip is not None:
                        return RouteMatch(
                            chip=chip,
                            confidence=0.9,
                            matched_intent=mapped_intent,
                        )

        # Fallback
        if self._config.fallback_chip:
            chip = self._registry.get(self._config.fallback_chip)
            if chip is not None:
                return RouteMatch(
                    chip=chip,
                    confidence=self._config.min_confidence,
                    matched_intent="fallback",
                )

        return None

    def get_intents_for_chip(self, chip_name: str) -> list[str]:
        """Get all intents mapped to a chip.

        Args:
            chip_name: The chip name to look up intents for

        Returns:
            List of intent patterns mapped to the chip

        Example:
            intents = router.get_intents_for_chip("grant_chip")
            # Returns: ["grant_search", "grant_apply", "grant_*"]
        """
        return [intent for intent, name in self._intent_map.items() if name == chip_name]

    def get_all_intents(self) -> list[str]:
        """Get all registered intent patterns.

        Returns:
            List of all registered intent patterns
        """
        return list(self._intent_map.keys())

    def get_intent_chip_mapping(self) -> dict[str, str]:
        """Get the full intent to chip name mapping.

        Returns:
            Dictionary mapping intent patterns to chip names
        """
        return dict(self._intent_map)

    def clear(self) -> None:
        """Remove all intent mappings.

        Useful for testing or reconfiguring the router.
        """
        self._intent_map.clear()

    def __len__(self) -> int:
        """Return the number of registered intent mappings."""
        return len(self._intent_map)


def create_router(
    registry: SkillRegistry | None = None,
    config: RouterConfig | None = None,
) -> SkillRouter:
    """Create a new skill router.

    Convenience factory function for creating routers with optional
    dependency injection.

    Args:
        registry: The skill registry to use. If None, uses the global registry.
        config: Optional router configuration

    Returns:
        Configured SkillRouter instance

    Example:
        from kintsugi.skills.registry import get_registry

        # Use global registry
        router = create_router()

        # Use custom registry and config
        router = create_router(
            registry=custom_registry,
            config=RouterConfig(fallback_chip="default"),
        )
    """
    from .registry import get_registry as get_global_registry

    if registry is None:
        registry = get_global_registry()

    return SkillRouter(registry, config)
