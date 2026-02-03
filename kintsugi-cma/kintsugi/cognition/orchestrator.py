"""Hierarchical Supervisor routing — classify and dispatch requests.

The :class:`Orchestrator` maps incoming user messages to *skill domains*
(grants, volunteers, finance, ...) using keyword matching with an optional
LLM classification fallback for ambiguous requests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from kintsugi.cognition.model_router import ModelRouter, ModelTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable record of a routing outcome."""

    skill_domain: str
    confidence: float
    reasoning: str
    model_tier: ModelTier


@dataclass
class OrchestratorConfig:
    """Configuration for the :class:`Orchestrator`.

    Parameters
    ----------
    routing_table:
        ``{keyword: skill_domain}`` map used for fast keyword matching.
    fallback_domain:
        Domain returned when no keyword matches and LLM classification is
        unavailable or below threshold.
    confidence_threshold:
        Minimum confidence to accept a keyword match without escalation.
    """

    routing_table: dict[str, str] = field(default_factory=dict)
    fallback_domain: str = "general"
    confidence_threshold: float = 0.6


# ---------------------------------------------------------------------------
# Default routing table
# ---------------------------------------------------------------------------

_DEFAULT_ROUTING_TABLE: dict[str, str] = {
    # grants
    "grant": "grants",
    "funding": "grants",
    "proposal": "grants",
    "funder": "grants",
    "rfp": "grants",
    # volunteers
    "volunteer": "volunteers",
    "recruitment": "volunteers",
    "onboarding": "volunteers",
    "hours": "volunteers",
    # finance
    "budget": "finance",
    "expense": "finance",
    "revenue": "finance",
    "invoice": "finance",
    "financial": "finance",
    "accounting": "finance",
    # impact
    "impact": "impact",
    "outcome": "impact",
    "metric": "impact",
    "evaluation": "impact",
    "indicator": "impact",
    # communications
    "email": "communications",
    "newsletter": "communications",
    "social media": "communications",
    "press": "communications",
    "outreach": "communications",
    "donor": "communications",
    # general (catch-all keywords are not needed; it's the fallback)
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Classify incoming messages and route them to skill domains.

    Parameters
    ----------
    config:
        Routing configuration.  Uses sensible defaults when *None*.
    model_router:
        Used to determine model tier for the routed task.
    llm_classifier:
        Optional async callable ``(message, domains) -> (domain, confidence)``
        injected for LLM-based disambiguation.  Keeps this module free of
        direct API dependencies.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        model_router: ModelRouter | None = None,
        llm_classifier: Callable[..., Awaitable[tuple[str, float]]] | None = None,
    ) -> None:
        self._config = config or OrchestratorConfig(
            routing_table=dict(_DEFAULT_ROUTING_TABLE),
        )
        if not self._config.routing_table:
            self._config.routing_table = dict(_DEFAULT_ROUTING_TABLE)
        self._model_router = model_router or ModelRouter()
        self._llm_classifier = llm_classifier

    # -- public API ---------------------------------------------------------

    async def classify_request(
        self,
        message: str,
        org_context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Classify *message* into a skill domain.

        1. Try keyword matching against the routing table.
        2. If confidence is below threshold **and** an LLM classifier was
           injected, delegate to the LLM.
        3. Otherwise fall back to ``config.fallback_domain``.
        """
        domain, confidence, reasoning = self._keyword_match(message)

        if confidence < self._config.confidence_threshold and self._llm_classifier is not None:
            try:
                domains = list(set(self._config.routing_table.values()))
                domains.append(self._config.fallback_domain)
                llm_domain, llm_confidence = await self._llm_classifier(message, domains)
                if llm_confidence > confidence:
                    domain = llm_domain
                    confidence = llm_confidence
                    reasoning = "LLM classification"
            except Exception:
                logger.exception("LLM classifier failed — using keyword result")

        tier = self._tier_for_domain(domain)
        return RoutingDecision(
            skill_domain=domain,
            confidence=confidence,
            reasoning=reasoning,
            model_tier=tier,
        )

    async def route(
        self,
        message: str,
        org_id: str,
        context: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Full routing pipeline: classify, validate, and log.

        Parameters
        ----------
        message:
            The user-facing request text.
        org_id:
            UUID of the organisation (used for logging context).
        context:
            Optional additional context forwarded to classification.
        """
        decision = await self.classify_request(message, org_context=context)

        # Build a dict suitable for temporal memory / audit trail.
        log_entry: dict[str, Any] = {
            "org_id": org_id,
            "message_preview": message[:120],
            "skill_domain": decision.skill_domain,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "model_tier": decision.model_tier.value,
        }
        logger.info("Routing decision: %s", log_entry)

        return decision

    def register_domain(self, domain: str, keywords: list[str]) -> None:
        """Add or update keywords for *domain* in the routing table."""
        for kw in keywords:
            self._config.routing_table[kw.lower()] = domain

    def get_routing_table(self) -> dict[str, str]:
        """Return a **copy** of the current routing table."""
        return dict(self._config.routing_table)

    # -- internals ----------------------------------------------------------

    def _keyword_match(self, message: str) -> tuple[str, float, str]:
        """Return ``(domain, confidence, reasoning)`` via keyword scan."""
        msg_lower = message.lower()
        hits: dict[str, int] = {}
        for keyword, domain in self._config.routing_table.items():
            count = len(re.findall(re.escape(keyword), msg_lower))
            if count:
                hits[domain] = hits.get(domain, 0) + count

        if not hits:
            return self._config.fallback_domain, 0.3, "no keyword match"

        best_domain = max(hits, key=hits.__getitem__)
        total_hits = sum(hits.values())
        confidence = min(0.95, 0.5 + 0.1 * hits[best_domain])
        reasoning = (
            f"keyword match: {hits[best_domain]}/{total_hits} hits for '{best_domain}'"
        )
        return best_domain, confidence, reasoning

    @staticmethod
    def _tier_for_domain(domain: str) -> ModelTier:
        """Heuristic tier assignment per domain."""
        if domain in ("finance", "grants"):
            return ModelTier.BALANCED
        if domain in ("impact", "communications"):
            return ModelTier.FAST
        return ModelTier.FAST
