"""Pydantic models for the Kintsugi VALUES.json organizational value document.

The VALUES.json file is the heart of Kintsugi's alignment system. It encodes an
organization's beliefs, desires, and intentions (BDI) along with hard ethical
constraints (principles) and operational guardrails (shield). Every agentic
action is checked against this document before execution.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# BDI primitives
# ---------------------------------------------------------------------------

class Belief(BaseModel):
    """Something the organization holds to be true about itself or its environment."""

    content: str = Field(..., min_length=1, description="Belief statement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level 0-1")
    source: str = Field(..., min_length=1, description="Evidence or origin of belief")
    last_verified: datetime | None = Field(default=None, description="When last reviewed")


class Desire(BaseModel):
    """A goal or value the organization wants to achieve or uphold."""

    content: str = Field(..., min_length=1)
    priority: int = Field(..., ge=1, le=5, description="1 = highest priority")
    measurable: bool = Field(default=False)
    metric: str | None = Field(default=None, description="How to measure progress")


class Intention(BaseModel):
    """A concrete strategic commitment the organization is actively pursuing."""

    content: str = Field(..., min_length=1)
    status: Literal["active", "paused", "completed"] = "active"
    started: datetime | None = None
    deadline: datetime | None = None


class ImpactBenchmark(BaseModel):
    """A measurable impact target, optionally aligned to UN SDGs."""

    metric: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    current: str | None = None
    sdg_alignment: list[int] = Field(default_factory=list, description="UN SDG numbers (1-17)")

    @field_validator("sdg_alignment", mode="before")
    @classmethod
    def _validate_sdgs(cls, v: list[int]) -> list[int]:
        for n in v:
            if not 1 <= n <= 17:
                raise ValueError(f"SDG number must be 1-17, got {n}")
        return v


class Principle(BaseModel):
    """A named ethical principle. Bright-line principles cannot be overridden."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    is_bright_line: bool = Field(
        default=False,
        description="Bright-line principles are absolute constraints that cannot be relaxed",
    )


# ---------------------------------------------------------------------------
# Top-level sections
# ---------------------------------------------------------------------------

class Organization(BaseModel):
    """Basic organizational identity."""

    name: str = Field(..., min_length=1)
    type: Literal["mutual_aid", "nonprofit_501c3", "cooperative", "advocacy", "other"] = "other"
    mission: str = Field(..., min_length=1)
    founded: str | None = None
    size: Literal["small", "medium", "large"] = "small"


class Beliefs(BaseModel):
    """Organizational BDI -- Beliefs: what the org believes about its world."""

    environment: list[Belief] = Field(default_factory=list)
    capabilities: list[Belief] = Field(default_factory=list)
    last_reviewed: datetime | None = None


class Desires(BaseModel):
    """Organizational BDI -- Desires: values and goals."""

    values: list[Desire] = Field(default_factory=list)
    mission_targets: list[Desire] = Field(default_factory=list)
    impact_benchmarks: list[ImpactBenchmark] = Field(default_factory=list)
    last_reviewed: datetime | None = None


class Intentions(BaseModel):
    """Organizational BDI -- Intentions: active strategic commitments."""

    active_strategies: list[Intention] = Field(default_factory=list)
    campaigns: list[Intention] = Field(default_factory=list)
    grants: list[Intention] = Field(default_factory=list)
    last_reviewed: datetime | None = None


class Principles(BaseModel):
    """Hard ethical constraints that bound all agentic behaviour."""

    equity_mandate: str = Field(
        default="All actions must be evaluated for equitable impact across affected communities.",
    )
    transparency_level: Literal["full", "summary", "minimal"] = "full"
    data_sovereignty: str = Field(
        default="Community data belongs to the community. No sharing without explicit consent.",
    )
    community_accountability: str = Field(
        default="Decisions affecting community members require community input.",
    )
    custom: list[Principle] = Field(default_factory=list)


class KintsugiGovernance(BaseModel):
    """Self-modification governance parameters for the Kintsugi engine."""

    shadow_verification: bool = Field(
        default=True,
        description="Run shadow checks on all high-stakes actions",
    )
    divergence_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="SWEI divergence threshold -- actions above this require human review",
    )
    consensus_required_for: list[str] = Field(
        default_factory=lambda: ["financial", "pii", "external_comms", "self_modification"],
    )
    bloom_schedule: Literal["weekly", "biweekly", "monthly"] = "weekly"
    max_modification_scope: Literal["prompt", "tool_config", "skill_chip", "architecture"] = "tool_config"


class Shield(BaseModel):
    """Operational guardrails: budgets, network egress, and security patterns."""

    budget_per_session: float = Field(default=5.0, ge=0.0)
    budget_per_day: float = Field(default=50.0, ge=0.0)
    egress_allowlist: list[str] = Field(
        default_factory=list,
        description="Allowed external domains for network calls",
    )
    blocked_patterns: list[str] = Field(
        default_factory=list,
        description="Additional regex patterns for the SecurityMonitor to block",
    )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class OrganizationValues(BaseModel):
    """Root schema for VALUES.json -- the organizational value document.

    This is the single source of truth that aligns every Kintsugi action with
    the organization's mission, ethics, and operational constraints.
    """

    organization: Organization
    beliefs: Beliefs = Field(default_factory=Beliefs)
    desires: Desires = Field(default_factory=Desires)
    intentions: Intentions = Field(default_factory=Intentions)
    principles: Principles = Field(default_factory=Principles)
    kintsugi: KintsugiGovernance = Field(default_factory=KintsugiGovernance)
    shield: Shield = Field(default_factory=Shield)
