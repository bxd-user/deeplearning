from __future__ import annotations

from dataclasses import dataclass

from .models import Outcome, ReflectionSignal, RoundState


@dataclass
class SimpleReflector:
    """Extracts preference/avoidance tags from the latest outcome."""

    positive_threshold: float = 0.65
    negative_threshold: float = 0.35

    def reflect(self, round_state: RoundState, outcome: Outcome, selected_tags: list[str]) -> ReflectionSignal:
        if outcome.quality_score >= self.positive_threshold:
            confidence = min(1.0, 0.6 + 0.4 * outcome.quality_score)
            return ReflectionSignal(
                prefer_tags=list(selected_tags),
                avoid_tags=[],
                confidence=confidence,
                notes="Promote tags from a promising attempt.",
            )

        if outcome.quality_score <= self.negative_threshold or not outcome.success:
            confidence = min(1.0, 0.6 + 0.4 * (1.0 - outcome.quality_score))
            return ReflectionSignal(
                prefer_tags=[],
                avoid_tags=list(selected_tags),
                confidence=confidence,
                notes="Downweight tags associated with a failed attempt.",
            )

        return ReflectionSignal(
            prefer_tags=[],
            avoid_tags=[],
            confidence=0.2,
            notes="Ambiguous signal; keep bias small.",
        )
