"""
Skill chip registry for discovery and routing.

This module provides the SkillRegistry class for managing skill chip
registration, discovery, and lookup. The registry serves as the central
catalog of available skill chips in the Kintsugi CMA system.

Usage:
    from kintsugi.skills.registry import get_registry, register_chip
    from kintsugi.skills.base import BaseSkillChip

    # Get the global registry
    registry = get_registry()

    # Register a chip
    register_chip(my_skill_chip)

    # Look up chips
    chip = registry.get("grant_search")
    fundraising_chips = registry.get_by_domain(SkillDomain.FUNDRAISING)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseSkillChip, SkillDomain

if TYPE_CHECKING:
    pass


class SkillRegistry:
    """Registry for skill chip discovery and routing.

    The registry maintains a catalog of all registered skill chips,
    enabling:
    - Name-based lookup for direct chip access
    - Domain-based lookup for routing and discovery
    - Metadata listing for UI and API responses

    Chips are stored by name with domain indexing for efficient
    lookup in both dimensions.

    Attributes:
        _chips: Internal mapping of chip names to chip instances
        _by_domain: Index mapping domains to lists of chip names

    Example:
        registry = SkillRegistry()

        # Register chips
        registry.register(grant_search_chip)
        registry.register(donor_management_chip)

        # Look up by name
        chip = registry.get("grant_search")

        # Look up by domain
        fundraising_chips = registry.get_by_domain(SkillDomain.FUNDRAISING)

        # Check registration
        if "grant_search" in registry:
            print("Chip is registered")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._chips: dict[str, BaseSkillChip] = {}
        self._by_domain: dict[SkillDomain, list[str]] = {}

    def register(self, chip: BaseSkillChip) -> None:
        """Register a skill chip.

        Adds the chip to the registry and indexes it by domain.
        Raises an error if a chip with the same name is already
        registered to prevent accidental overwrites.

        Args:
            chip: The skill chip instance to register

        Raises:
            ValueError: If a chip with the same name is already registered

        Example:
            registry = SkillRegistry()
            registry.register(GrantSearchChip())
        """
        if chip.name in self._chips:
            raise ValueError(f"Chip '{chip.name}' already registered")

        self._chips[chip.name] = chip

        # Index by domain
        if chip.domain not in self._by_domain:
            self._by_domain[chip.domain] = []
        self._by_domain[chip.domain].append(chip.name)

    def unregister(self, name: str) -> bool:
        """Unregister a skill chip.

        Removes the chip from the registry and domain index.
        Returns True if the chip was found and removed, False otherwise.

        Args:
            name: The name of the chip to unregister

        Returns:
            True if the chip was found and removed, False if not found

        Example:
            if registry.unregister("old_chip"):
                print("Chip removed")
            else:
                print("Chip not found")
        """
        chip = self._chips.pop(name, None)
        if chip is not None:
            # Remove from domain index
            domain_chips = self._by_domain.get(chip.domain, [])
            if name in domain_chips:
                domain_chips.remove(name)
            return True
        return False

    def get(self, name: str) -> BaseSkillChip | None:
        """Get a chip by name.

        Args:
            name: The unique name of the chip to retrieve

        Returns:
            The chip instance if found, None otherwise

        Example:
            chip = registry.get("grant_search")
            if chip:
                response = await chip.handle(request, context)
        """
        return self._chips.get(name)

    def get_by_domain(self, domain: SkillDomain) -> list[BaseSkillChip]:
        """Get all chips in a domain.

        Returns a list of all chip instances registered in the
        specified domain. Useful for domain-specific routing or
        displaying available capabilities.

        Args:
            domain: The SkillDomain to filter by

        Returns:
            List of chip instances in the domain (may be empty)

        Example:
            fundraising_chips = registry.get_by_domain(SkillDomain.FUNDRAISING)
            for chip in fundraising_chips:
                print(f"- {chip.name}: {chip.description}")
        """
        names = self._by_domain.get(domain, [])
        return [self._chips[name] for name in names if name in self._chips]

    def list_all(self) -> list[dict]:
        """List all registered chips with metadata.

        Returns metadata for all registered chips, useful for
        API responses and UI display.

        Returns:
            List of dictionaries containing chip metadata
            (see BaseSkillChip.get_info() for structure)

        Example:
            for chip_info in registry.list_all():
                print(f"{chip_info['name']}: {chip_info['description']}")
        """
        return [chip.get_info() for chip in self._chips.values()]

    def list_names(self) -> list[str]:
        """List all registered chip names.

        Returns:
            List of registered chip names

        Example:
            names = registry.list_names()
            print(f"Registered chips: {', '.join(names)}")
        """
        return list(self._chips.keys())

    def list_domains(self) -> list[SkillDomain]:
        """List all domains that have registered chips.

        Returns:
            List of SkillDomain values that have at least one chip

        Example:
            domains = registry.list_domains()
            for domain in domains:
                chips = registry.get_by_domain(domain)
                print(f"{domain.value}: {len(chips)} chips")
        """
        return [domain for domain, chips in self._by_domain.items() if chips]

    def clear(self) -> None:
        """Remove all registered chips.

        Clears both the chip registry and domain index.
        Useful for testing or reinitializing the registry.
        """
        self._chips.clear()
        self._by_domain.clear()

    def __len__(self) -> int:
        """Return the number of registered chips.

        Returns:
            Count of registered chips
        """
        return len(self._chips)

    def __contains__(self, name: str) -> bool:
        """Check if a chip is registered.

        Args:
            name: The chip name to check

        Returns:
            True if the chip is registered, False otherwise

        Example:
            if "grant_search" in registry:
                chip = registry.get("grant_search")
        """
        return name in self._chips

    def __iter__(self):
        """Iterate over registered chips.

        Yields:
            BaseSkillChip instances in registration order

        Example:
            for chip in registry:
                print(chip.name)
        """
        return iter(self._chips.values())


# Global registry instance
_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    """Get the global skill registry.

    Returns the singleton global registry instance, creating it
    if it doesn't exist. This is the primary way to access the
    registry throughout the application.

    Returns:
        The global SkillRegistry instance

    Example:
        registry = get_registry()
        chip = registry.get("grant_search")
    """
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def register_chip(chip: BaseSkillChip) -> None:
    """Register a chip in the global registry.

    Convenience function for registering chips without needing
    to call get_registry() first.

    Args:
        chip: The skill chip instance to register

    Raises:
        ValueError: If a chip with the same name is already registered

    Example:
        from kintsugi.skills.registry import register_chip

        register_chip(GrantSearchChip())
    """
    get_registry().register(chip)


def reset_registry() -> None:
    """Reset the global registry to empty state.

    Primarily useful for testing to ensure clean state between tests.
    In production, this should rarely if ever be needed.
    """
    global _registry
    if _registry is not None:
        _registry.clear()
    _registry = None
