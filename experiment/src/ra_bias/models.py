from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    description: str
    candidate_pool: list[dict[str, Any]]


@dataclass
class Candidate:
    candidate_id: str
    description: str
    tags: list[str]
    base_score: float = 0.0
    bias_score: float = 0.0
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "description": self.description,
            "tags": self.tags,
            "base_score": self.base_score,
            "bias_score": self.bias_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
        }


@dataclass
class Outcome:
    success: bool
    quality_score: float
    failure_modes: list[str]
    cost_tokens: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.quality_score,
            "failure_modes": self.failure_modes,
            "cost_tokens": self.cost_tokens,
            "notes": self.notes,
        }


@dataclass
class ReflectionSignal:
    prefer_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefer_tags": self.prefer_tags,
            "avoid_tags": self.avoid_tags,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class RoundState:
    task_id: str
    round_idx: int
    budget_left: int
    history: list[dict[str, Any]]


@dataclass
class RoundLog:
    task_id: str
    round_idx: int
    candidates: list[dict[str, Any]]
    selected_candidate_id: str
    selected_description: str
    outcome: dict[str, Any]
    reflection_signal: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "round": self.round_idx,
            "candidate_set": self.candidates,
            "selected_id": self.selected_candidate_id,
            "selected_description": self.selected_description,
            "outcome": self.outcome,
            "reflection_signal": self.reflection_signal,
        }


@dataclass
class EpisodeLog:
    task_id: str
    method_name: str
    rounds: list[RoundLog]
    success: bool
    steps: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "method_name": self.method_name,
            "rounds": [r.to_dict() for r in self.rounds],
            "success": self.success,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
        }
