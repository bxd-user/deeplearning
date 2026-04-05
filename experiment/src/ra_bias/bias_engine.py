from __future__ import annotations

import random
from dataclasses import dataclass, field

from .models import Candidate, ReflectionSignal


class BiasEngine:
    """Interface for turning reflection into candidate bias scores."""

    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class NoBiasEngine(BiasEngine):
    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        return {c.candidate_id: 0.0 for c in candidates}


@dataclass
class TagBiasEngine(BiasEngine):
    """
    Explicit reflection-as-bias: tag overlap determines prefer/avoid weights.
    Corresponds to the Reflection-as-Bias method in design.md §5.4.
    """
    prefer_weight: float = 0.35
    avoid_weight: float = 0.45

    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        scores: dict[str, float] = {}
        confidence = max(0.0, min(1.0, signal.confidence))
        prefer = set(signal.prefer_tags)
        avoid = set(signal.avoid_tags)

        for c in candidates:
            overlap_prefer = len(prefer.intersection(c.tags))
            overlap_avoid = len(avoid.intersection(c.tags))
            bias = (self.prefer_weight * overlap_prefer - self.avoid_weight * overlap_avoid) * confidence
            scores[c.candidate_id] = bias

        return scores


@dataclass
class WeakTagBiasEngine(TagBiasEngine):
    """
    Ablation: tag bias with significantly reduced weights (statics.md §3.3).

    Weights are deliberately set below the typical base-score gap between
    candidates (~0.15–0.20), so the bias signal influences but rarely overrides
    the base-score ordering.  This creates a genuine 3-tier comparison:
      branching_only (no bias) < weak_bias < reflection_as_bias (full bias).
    """
    prefer_weight: float = 0.05
    avoid_weight: float = 0.06


@dataclass
class HardPruningBiasEngine(TagBiasEngine):
    """
    Ablation: any candidate whose tags overlap with avoid_tags is eliminated.
    Used for Hard Pruning vs Soft Bias comparison (design.md §7.2 / statics.md §3.1).
    """
    prune_penalty: float = -1_000_000.0

    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        base = super().score(candidates, signal)
        avoid = set(signal.avoid_tags)

        for c in candidates:
            if avoid.intersection(c.tags):
                base[c.candidate_id] = self.prune_penalty
        return base


@dataclass
class TextualReflectionEngine(BiasEngine):
    """
    Simulates implicit LLM use of reflection text (design.md §5.3).

    Models the scenario where reflection exists only as appended text in the prompt.
    The LLM implicitly adjusts its judgements but imprecisely — captured here as
    weak weights plus additive Gaussian noise.  Weaker and noisier than TagBiasEngine,
    but non-zero (unlike NoBiasEngine), creating a proper 3-tier hierarchy:
    branching_only < textual_reflection < reflection_as_bias.
    """
    prefer_weight: float = 0.08
    avoid_weight: float = 0.10
    noise_std: float = 0.08
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        scores: dict[str, float] = {}
        confidence = max(0.0, min(1.0, signal.confidence))
        prefer = set(signal.prefer_tags)
        avoid = set(signal.avoid_tags)

        for c in candidates:
            overlap_prefer = len(prefer.intersection(c.tags))
            overlap_avoid = len(avoid.intersection(c.tags))
            bias = (self.prefer_weight * overlap_prefer - self.avoid_weight * overlap_avoid) * confidence
            # Noise only fires when there is an actual signal to be imprecise about;
            # before any reflection history exists (confidence == 0) behaviour is
            # identical to branching_only, as expected for round 1.
            if confidence > 0:
                bias += self._rng.gauss(0, self.noise_std)
            scores[c.candidate_id] = bias

        return scores


@dataclass
class LLMBiasEngine(BiasEngine):
    """
    Simulates LLM-based bias scoring (design.md §7.3 / statics.md §3.2).

    Models richer semantic reasoning: combines tag-level signal (same as TagBiasEngine)
    with a quality-informed component that represents the LLM's deeper assessment of
    each candidate's potential.  Hidden quality is used as a proxy for the LLM's
    richer context — in a real deployment this would come from an actual API call.
    Outperforms TagBiasEngine by leveraging quality signal beyond tag overlap.
    """
    tag_weight: float = 0.30
    quality_weight: float = 0.40
    quality_noise_std: float = 0.10
    output_noise_std: float = 0.04
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def score(self, candidates: list[Candidate], signal: ReflectionSignal) -> dict[str, float]:
        scores: dict[str, float] = {}
        confidence = max(0.0, min(1.0, signal.confidence))
        prefer = set(signal.prefer_tags)
        avoid = set(signal.avoid_tags)

        for c in candidates:
            # Tag component — same logic as TagBiasEngine
            overlap_prefer = len(prefer.intersection(c.tags))
            overlap_avoid = len(avoid.intersection(c.tags))
            tag_component = 0.35 * overlap_prefer - 0.45 * overlap_avoid

            # Quality component — LLM estimates candidate potential (with noise)
            hidden_q = float(c.metadata.get("hidden_quality", 0.5))
            quality_estimate = max(0.0, min(1.0, hidden_q + self._rng.gauss(0, self.quality_noise_std)))
            quality_component = (quality_estimate - 0.5) * self.quality_weight

            bias = (self.tag_weight * tag_component + quality_component) * confidence
            bias += self._rng.gauss(0, self.output_noise_std)
            scores[c.candidate_id] = bias

        return scores
