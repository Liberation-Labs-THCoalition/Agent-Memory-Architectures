"""Promotion and rollback of verified shadow modifications.

When the Verifier approves a shadow fork, the Promoter applies its
modification to the live config, stores a golden trace for audit, and
supports rollback to any previous checkpoint.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from kintsugi.kintsugi_engine.verifier import VerifierVerdict, VerificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PromotionAction(str, Enum):
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    EXTEND = "EXTEND"
    ESCALATE = "ESCALATE"


@dataclass
class GoldenTrace:
    """Immutable record of a successful promotion."""

    trace_id: str
    shadow_id: str
    modification: Dict[str, Any]
    verdict: VerifierVerdict
    swei_divergence: float
    promoted_at: datetime
    config_before: Dict[str, Any]
    config_after: Dict[str, Any]


@dataclass
class PromoterConfig:
    """Configuration for the Promoter."""

    max_rollback_depth: int = 10
    log_golden_traces: bool = True


# ---------------------------------------------------------------------------
# Promoter
# ---------------------------------------------------------------------------

class Promoter:
    """Applies verified modifications and maintains rollback history.

    Only APPROVE verdicts result in config changes.  All other verdicts
    pass through as-is with the current config unchanged.
    """

    def __init__(self, config: PromoterConfig | None = None) -> None:
        self._config = config or PromoterConfig()
        self._traces: List[GoldenTrace] = []

    # -- public API ---------------------------------------------------------

    def promote(
        self,
        shadow_id: str,
        modification: Dict[str, Any],
        verification: VerificationResult,
        current_config: Dict[str, Any],
    ) -> tuple[PromotionAction, Dict[str, Any]]:
        """Decide and execute promotion based on the verification verdict.

        Returns ``(action, config)`` where *config* is the new config on
        PROMOTE or the unchanged config otherwise.
        """
        verdict = verification.verdict

        if verdict == VerifierVerdict.APPROVE:
            new_config = self._apply_modification(
                copy.deepcopy(current_config), modification
            )
            trace = GoldenTrace(
                trace_id=f"trace-{uuid.uuid4().hex[:12]}",
                shadow_id=shadow_id,
                modification=copy.deepcopy(modification),
                verdict=verdict,
                swei_divergence=verification.swei_divergence,
                promoted_at=datetime.now(timezone.utc),
                config_before=copy.deepcopy(current_config),
                config_after=copy.deepcopy(new_config),
            )
            self._store_trace(trace)
            logger.info(
                "Promoted shadow %s -> trace %s", shadow_id, trace.trace_id
            )
            return PromotionAction.PROMOTE, new_config

        if verdict == VerifierVerdict.REJECT:
            logger.info("Rollback for shadow %s (REJECT)", shadow_id)
            return PromotionAction.ROLLBACK, copy.deepcopy(current_config)

        if verdict == VerifierVerdict.EXTEND:
            logger.info("Extending observation for shadow %s", shadow_id)
            return PromotionAction.EXTEND, copy.deepcopy(current_config)

        # ESCALATE
        logger.warning("Escalating shadow %s for human review", shadow_id)
        return PromotionAction.ESCALATE, copy.deepcopy(current_config)

    def rollback(self, steps: int = 1) -> Dict[str, Any]:
        """Revert to the config from *steps* promotions ago.

        Returns the ``config_before`` of the target trace.  Raises
        ``ValueError`` if there is insufficient history.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if steps > len(self._traces):
            raise ValueError(
                f"Cannot rollback {steps} steps; only {len(self._traces)} traces available"
            )
        target = self._traces[-steps]
        logger.info("Rolling back %d step(s) to trace %s", steps, target.trace_id)
        return copy.deepcopy(target.config_before)

    def get_golden_traces(self, limit: int = 50) -> List[GoldenTrace]:
        """Return the most recent golden traces, up to *limit*."""
        return list(self._traces[-limit:])

    # -- internal -----------------------------------------------------------

    def _apply_modification(
        self, config: Dict[str, Any], modification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep-merge *modification* into *config* and return the result."""
        for key, value in modification.items():
            if (
                key in config
                and isinstance(config[key], dict)
                and isinstance(value, dict)
            ):
                config[key] = self._apply_modification(config[key], value)
            else:
                config[key] = copy.deepcopy(value)
        return config

    def _store_trace(self, trace: GoldenTrace) -> None:
        """Append trace to history, enforcing max rollback depth."""
        self._traces.append(trace)
        max_depth = self._config.max_rollback_depth
        if len(self._traces) > max_depth:
            self._traces = self._traces[-max_depth:]
        if self._config.log_golden_traces:
            logger.debug("Stored golden trace %s", trace.trace_id)
