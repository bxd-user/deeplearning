from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Candidate, Outcome, Task


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class SimulatedResearchEnvironment:
    """A controllable research-like environment for iterative decisions."""

    seed: int = 7
    base_cost: int = 350

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def propose_candidates(self, task: Task, num_candidates: int) -> list[Candidate]:
        pool = list(task.candidate_pool)
        if not pool:
            raise ValueError(f"Task {task.task_id} has empty candidate pool.")

        if num_candidates == 1:
            # Single-path baseline: always pick the currently highest base-score direction.
            picked = [max(pool, key=lambda x: float(x.get("base_score", 0.0)))]
        elif num_candidates >= len(pool):
            picked = list(pool)
            self.rng.shuffle(picked)
        else:
            picked = self.rng.sample(pool, k=num_candidates)

        candidates: list[Candidate] = []
        for idx, item in enumerate(picked):
            base = float(item.get("base_score", 0.5))
            noisy_base = _clamp(base + self.rng.uniform(-0.04, 0.04))
            cid = str(item.get("id", f"cand_{idx}"))
            candidates.append(
                Candidate(
                    candidate_id=f"{cid}_r{idx}",
                    description=str(item.get("description", cid)),
                    tags=list(item.get("tags", [])),
                    base_score=noisy_base,
                    metadata={
                        "hidden_quality": float(item.get("hidden_quality", 0.5)),
                        "failure_modes": list(item.get("failure_modes", ["unknown_failure"])),
                    },
                )
            )
        return candidates

    def step(self, task: Task, candidate: Candidate) -> Outcome:
        hidden_quality = float(candidate.metadata.get("hidden_quality", 0.5))

        # Non-linear success curve creates "promising" and "dead-end" directions.
        success_prob = _clamp(0.02 + 0.95 * (hidden_quality**2))
        success = self.rng.random() < success_prob

        quality = _clamp(hidden_quality + self.rng.uniform(-0.1, 0.1))
        if success:
            quality = max(quality, 0.62)
            failure_modes: list[str] = []
        else:
            quality = min(quality, 0.58)
            failure_modes = list(candidate.metadata.get("failure_modes", ["execution_failed"]))

        token_cost = self.base_cost + self.rng.randint(80, 260) + 20 * len(candidate.tags)

        return Outcome(
            success=success,
            quality_score=quality,
            failure_modes=failure_modes,
            cost_tokens=token_cost,
            notes=f"Task={task.task_id}; candidate={candidate.description}",
        )


def build_demo_tasks() -> list[Task]:
    return [
        # Task 1: Reasoning — base scores mislead (single_pass has high base but low quality)
        Task(
            task_id="task_reasoning",
            description="Find a robust solution route for a hard reasoning subproblem.",
            candidate_pool=[
                {
                    "id": "single_pass",
                    "description": "single-pass prompting",
                    "tags": ["single_pass", "prompt_only"],
                    "base_score": 0.74,
                    "hidden_quality": 0.24,
                    "failure_modes": ["fragile_reasoning", "hallucination"],
                },
                {
                    "id": "decompose_verify",
                    "description": "decomposition + verification",
                    "tags": ["multi_step", "verification"],
                    "base_score": 0.60,
                    "hidden_quality": 0.86,
                    "failure_modes": ["verification_gap"],
                },
                {
                    "id": "search_verify",
                    "description": "search + verification",
                    "tags": ["search", "verification"],
                    "base_score": 0.56,
                    "hidden_quality": 0.74,
                    "failure_modes": ["retrieval_noise"],
                },
                {
                    "id": "tree_search",
                    "description": "tree search planning",
                    "tags": ["tree_search", "multi_step"],
                    "base_score": 0.54,
                    "hidden_quality": 0.68,
                    "failure_modes": ["branch_explosion"],
                },
            ],
        ),
        # Task 2: NLP Generation — retrieval augmentation is underestimated
        Task(
            task_id="task_nlp_generation",
            description="Improve output quality for an open-ended NLP generation task.",
            candidate_pool=[
                {
                    "id": "prompt_only",
                    "description": "direct prompting without context",
                    "tags": ["single_pass", "prompt_only"],
                    "base_score": 0.70,
                    "hidden_quality": 0.22,
                    "failure_modes": ["hallucination", "generic_output"],
                },
                {
                    "id": "rag",
                    "description": "retrieval-augmented generation",
                    "tags": ["search", "retrieval", "multi_step"],
                    "base_score": 0.58,
                    "hidden_quality": 0.82,
                    "failure_modes": ["retrieval_noise"],
                },
                {
                    "id": "chain_of_thought",
                    "description": "chain-of-thought prompting",
                    "tags": ["multi_step", "prompt_only"],
                    "base_score": 0.64,
                    "hidden_quality": 0.60,
                    "failure_modes": ["reasoning_drift"],
                },
                {
                    "id": "constrained_decode",
                    "description": "structured output with constraints",
                    "tags": ["verification", "structured"],
                    "base_score": 0.52,
                    "hidden_quality": 0.72,
                    "failure_modes": ["constraint_violation"],
                },
            ],
        ),
        # Task 3: Data Analysis — structured approaches dominate
        Task(
            task_id="task_data_analysis",
            description="Select an analysis strategy for a noisy tabular dataset.",
            candidate_pool=[
                {
                    "id": "heuristic_filter",
                    "description": "heuristic-based data filtering",
                    "tags": ["single_pass", "heuristic"],
                    "base_score": 0.72,
                    "hidden_quality": 0.30,
                    "failure_modes": ["overfitting_heuristic", "data_loss"],
                },
                {
                    "id": "statistical_pipeline",
                    "description": "statistical cleaning + feature engineering",
                    "tags": ["multi_step", "verification", "structured"],
                    "base_score": 0.55,
                    "hidden_quality": 0.88,
                    "failure_modes": ["distribution_shift"],
                },
                {
                    "id": "iterative_imputation",
                    "description": "iterative missing-value imputation",
                    "tags": ["multi_step", "structured"],
                    "base_score": 0.60,
                    "hidden_quality": 0.70,
                    "failure_modes": ["imputation_bias"],
                },
                {
                    "id": "automl_search",
                    "description": "automated model selection via search",
                    "tags": ["search", "tree_search", "multi_step"],
                    "base_score": 0.50,
                    "hidden_quality": 0.65,
                    "failure_modes": ["search_timeout"],
                },
            ],
        ),
    ]
