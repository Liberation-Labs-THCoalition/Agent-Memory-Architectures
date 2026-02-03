"""Expected Free Energy (EFE) calculation for active inference.

Implements a lightweight EFE scorer used by the decision engine to rank
candidate policies.  Lower total EFE indicates the preferred policy
(least expected surprise / best alignment with desired outcomes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EFEWeights:
    """Component weights for the EFE calculation.

    Must approximately sum to 1.0 (tolerance 0.05).
    """

    risk: float
    ambiguity: float
    epistemic: float

    def __post_init__(self) -> None:
        total = self.risk + self.ambiguity + self.epistemic
        if not math.isclose(total, 1.0, abs_tol=0.05):
            raise ValueError(
                f"EFEWeights must sum to ~1.0; got {total:.4f} "
                f"(risk={self.risk}, ambiguity={self.ambiguity}, "
                f"epistemic={self.epistemic})"
            )


# Domain-specific weight profiles
GRANTS_WEIGHTS = EFEWeights(risk=0.3, ambiguity=0.3, epistemic=0.4)
FINANCE_WEIGHTS = EFEWeights(risk=0.6, ambiguity=0.3, epistemic=0.1)
COMMUNICATIONS_WEIGHTS = EFEWeights(risk=0.4, ambiguity=0.2, epistemic=0.4)
DEFAULT_WEIGHTS = EFEWeights(risk=0.33, ambiguity=0.34, epistemic=0.33)


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EFEScore:
    """Result of an EFE evaluation for a single policy."""

    total: float
    risk_component: float
    ambiguity_component: float
    epistemic_component: float
    policy_id: str


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class EFECalculator:
    """Compute Expected Free Energy for candidate policies.

    Parameters
    ----------
    default_weights:
        Fallback weights when none are provided per call.
    """

    def __init__(self, default_weights: EFEWeights | None = None) -> None:
        self._default_weights = default_weights or DEFAULT_WEIGHTS

    # -- public API ---------------------------------------------------------

    def calculate_efe(
        self,
        policy_id: str,
        predicted_outcome: dict,
        desired_outcome: dict,
        uncertainty: float,
        information_gain: float,
        weights: EFEWeights | None = None,
    ) -> EFEScore:
        """Score a single policy.

        Parameters
        ----------
        policy_id:
            Unique identifier for the candidate policy.
        predicted_outcome:
            Dict of predicted state variables after executing the policy.
        desired_outcome:
            Dict of target / goal state variables.
        uncertainty:
            Scalar representing outcome uncertainty (0 = certain).
        information_gain:
            Expected information gain from executing this policy.
        weights:
            Per-call weight override; uses *default_weights* when *None*.

        Returns
        -------
        EFEScore
            Decomposed score with total and per-component values.
        """
        w = weights or self._default_weights
        divergence = self.compute_divergence(predicted_outcome, desired_outcome)

        risk_component = w.risk * divergence
        ambiguity_component = w.ambiguity * uncertainty
        epistemic_component = w.epistemic * (-information_gain)
        total = risk_component + ambiguity_component + epistemic_component

        return EFEScore(
            total=total,
            risk_component=risk_component,
            ambiguity_component=ambiguity_component,
            epistemic_component=epistemic_component,
            policy_id=policy_id,
        )

    def select_policy(self, scores: list[EFEScore]) -> EFEScore:
        """Return the policy with the lowest total EFE.

        Raises ``ValueError`` if *scores* is empty.
        """
        if not scores:
            raise ValueError("Cannot select from an empty score list")
        return min(scores, key=lambda s: s.total)

    @staticmethod
    def compute_divergence(predicted: dict, desired: dict) -> float:
        """Normalised symmetric difference between two outcome dicts.

        For overlapping keys with numeric values the divergence is the mean
        absolute difference normalised by the max absolute value (per key).
        Keys present in only one dict contribute 1.0 each.  The final value
        is averaged over the union of keys so the result lies in ``[0, 1]``.
        """
        all_keys = set(predicted) | set(desired)
        if not all_keys:
            return 0.0

        total = 0.0
        for key in all_keys:
            if key not in predicted or key not in desired:
                total += 1.0
                continue
            pv, dv = predicted[key], desired[key]
            try:
                pf, df = float(pv), float(dv)
            except (TypeError, ValueError):
                # Non-numeric: exact equality check
                total += 0.0 if pv == dv else 1.0
                continue
            max_abs = max(abs(pf), abs(df), 1e-9)
            total += abs(pf - df) / max_abs

        return total / len(all_keys)
