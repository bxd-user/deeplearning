from __future__ import annotations

from dataclasses import dataclass, field

from .models import EpisodeLog


@dataclass
class Metrics:
    method_name: str
    success_rate: float
    avg_steps_to_success: float
    wasted_exploration_ratio: float
    cost_per_success: float
    # Per-round average quality of the selected candidate across episodes.
    # Maps round index (1-based) → average quality score.
    # Captures whether bias guidance improves selection quality over rounds
    # (statics.md §2.3: "候选质量随轮次变化").
    round_quality_trend: dict[int, float] = field(default_factory=dict)


def _compute_round_quality_trend(episodes: list[EpisodeLog]) -> dict[int, float]:
    """Average quality_score of the executed candidate per round, across all episodes."""
    round_data: dict[int, list[float]] = {}
    for episode in episodes:
        for round_log in episode.rounds:
            r = round_log.round_idx
            # outcome dict uses "score" key after Outcome.to_dict() renaming
            quality = float(round_log.outcome.get("score", 0.0))
            round_data.setdefault(r, []).append(quality)
    return {r: sum(qs) / len(qs) for r, qs in sorted(round_data.items())}


def compute_metrics(method_name: str, episodes: list[EpisodeLog]) -> Metrics:
    if not episodes:
        return Metrics(method_name, 0.0, 0.0, 0.0, 0.0)

    success_episodes = [e for e in episodes if e.success]
    success_rate = len(success_episodes) / len(episodes)

    if success_episodes:
        avg_steps_to_success = sum(e.steps for e in success_episodes) / len(success_episodes)
        cost_per_success = sum(e.total_tokens for e in success_episodes) / len(success_episodes)
    else:
        avg_steps_to_success = 0.0
        cost_per_success = 0.0

    failed_trials = 0
    total_trials = 0
    for e in episodes:
        total_trials += e.steps
        failed_trials += sum(1 for r in e.rounds if not r.outcome["success"])
    wasted_exploration_ratio = failed_trials / total_trials if total_trials else 0.0

    round_quality_trend = _compute_round_quality_trend(episodes)

    return Metrics(
        method_name=method_name,
        success_rate=success_rate,
        avg_steps_to_success=avg_steps_to_success,
        wasted_exploration_ratio=wasted_exploration_ratio,
        cost_per_success=cost_per_success,
        round_quality_trend=round_quality_trend,
    )
