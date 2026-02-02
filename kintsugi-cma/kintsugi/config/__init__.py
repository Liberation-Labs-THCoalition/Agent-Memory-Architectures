"""Kintsugi configuration system -- VALUES.json schema, loader, and file watcher."""

from .values_schema import (
    Belief,
    Beliefs,
    Desire,
    Desires,
    ImpactBenchmark,
    Intention,
    Intentions,
    KintsugiGovernance,
    Organization,
    OrganizationValues,
    Principle,
    Principles,
    Shield,
)
from .values_loader import (
    FileWatcher,
    load_from_template,
    load_values,
    merge_with_defaults,
    save_values,
)

__all__ = [
    "Belief",
    "Beliefs",
    "Desire",
    "Desires",
    "FileWatcher",
    "ImpactBenchmark",
    "Intention",
    "Intentions",
    "KintsugiGovernance",
    "Organization",
    "OrganizationValues",
    "Principle",
    "Principles",
    "Shield",
    "load_from_template",
    "load_values",
    "merge_with_defaults",
    "save_values",
]
