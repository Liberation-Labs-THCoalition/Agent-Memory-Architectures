"""BDI coherence scoring."""

from dataclasses import dataclass, field
from typing import List, Tuple

from .models import BDIBelief, BDIDesire, BDIIntention, BDISnapshot


@dataclass(frozen=True)
class CoherenceScore:
    belief_desire_alignment: float
    desire_intention_alignment: float
    belief_intention_alignment: float
    overall: float
    issues: Tuple[str, ...] = ()  # frozen dataclass needs immutable default


class CoherenceChecker:
    """Checks coherence across BDI layers."""

    def __init__(self) -> None:
        self._weights = {
            "belief_desire": 0.35,
            "desire_intention": 0.35,
            "belief_intention": 0.30,
        }

    def check_coherence(self, snapshot: BDISnapshot) -> CoherenceScore:
        bd_score, bd_issues = self._check_belief_desire_alignment(
            snapshot.beliefs, snapshot.desires
        )
        di_score, di_issues = self._check_desire_intention_alignment(
            snapshot.desires, snapshot.intentions
        )
        bi_score, bi_issues = self._check_belief_intention_alignment(
            snapshot.beliefs, snapshot.intentions
        )

        overall = (
            self._weights["belief_desire"] * bd_score
            + self._weights["desire_intention"] * di_score
            + self._weights["belief_intention"] * bi_score
        )

        all_issues = bd_issues + di_issues + bi_issues

        return CoherenceScore(
            belief_desire_alignment=round(bd_score, 4),
            desire_intention_alignment=round(di_score, 4),
            belief_intention_alignment=round(bi_score, 4),
            overall=round(overall, 4),
            issues=tuple(all_issues),
        )

    def _check_belief_desire_alignment(
        self,
        beliefs: List[BDIBelief],
        desires: List[BDIDesire],
    ) -> Tuple[float, List[str]]:
        """Check tag overlap and content keyword matching between beliefs and desires."""
        if not beliefs or not desires:
            return (0.5, ["Insufficient beliefs or desires for alignment check."])

        issues: List[str] = []
        belief_tags: set = set()
        belief_words: set = set()
        for b in beliefs:
            belief_tags.update(b.tags)
            belief_words.update(w.lower() for w in b.content.split() if len(w) > 3)

        scores: List[float] = []
        for d in desires:
            desire_tags = set(d.related_tags)
            desire_words = set(w.lower() for w in d.content.split() if len(w) > 3)

            tag_overlap = len(desire_tags & belief_tags) / max(len(desire_tags), 1)
            word_overlap = len(desire_words & belief_words) / max(len(desire_words), 1)
            score = 0.6 * tag_overlap + 0.4 * min(word_overlap, 1.0)
            scores.append(score)

            if score < 0.3:
                issues.append(
                    f"Desire '{d.id}' has weak belief support (score={score:.2f})."
                )

        return (sum(scores) / len(scores), issues)

    def _check_desire_intention_alignment(
        self,
        desires: List[BDIDesire],
        intentions: List[BDIIntention],
    ) -> Tuple[float, List[str]]:
        """Check that intentions reference active desires via desire_ids."""
        if not desires or not intentions:
            return (0.5, ["Insufficient desires or intentions for alignment check."])

        issues: List[str] = []
        desire_ids = {d.id for d in desires}
        active_desire_ids = {d.id for d in desires if d.status.value == "active"}

        scores: List[float] = []
        for intention in intentions:
            linked = set(intention.desire_ids)
            if not linked:
                scores.append(0.0)
                issues.append(
                    f"Intention '{intention.id}' is not linked to any desire."
                )
                continue
            valid = linked & desire_ids
            active = linked & active_desire_ids
            score = (len(valid) / len(linked)) * 0.5 + (len(active) / len(linked)) * 0.5
            scores.append(score)
            if not active:
                issues.append(
                    f"Intention '{intention.id}' links only to inactive desires."
                )

        return (sum(scores) / len(scores), issues)

    def _check_belief_intention_alignment(
        self,
        beliefs: List[BDIBelief],
        intentions: List[BDIIntention],
    ) -> Tuple[float, List[str]]:
        """Check that intentions reference active beliefs via belief_ids."""
        if not beliefs or not intentions:
            return (0.5, ["Insufficient beliefs or intentions for alignment check."])

        issues: List[str] = []
        belief_ids = {b.id for b in beliefs}
        active_belief_ids = {b.id for b in beliefs if b.status.value == "active"}

        scores: List[float] = []
        for intention in intentions:
            linked = set(intention.belief_ids)
            if not linked:
                scores.append(0.0)
                issues.append(
                    f"Intention '{intention.id}' is not linked to any belief."
                )
                continue
            valid = linked & belief_ids
            active = linked & active_belief_ids
            score = (len(valid) / len(linked)) * 0.5 + (len(active) / len(linked)) * 0.5
            scores.append(score)
            if not active:
                issues.append(
                    f"Intention '{intention.id}' links only to inactive/stale beliefs."
                )

        return (sum(scores) / len(scores), issues)
