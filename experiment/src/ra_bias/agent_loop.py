from __future__ import annotations

from dataclasses import dataclass, field

from .bias_engine import BiasEngine
from .environment import SimulatedResearchEnvironment
from .models import EpisodeLog, ReflectionSignal, RoundLog, RoundState, Task
from .reflection import SimpleReflector
from .selector import Selector


@dataclass
class AgentConfig:
    method_name: str
    num_candidates: int = 3
    max_rounds: int = 4
    budget_tokens: int = 3_200
    enable_reflection: bool = True
    enable_bias: bool = True


@dataclass
class ResearchAgentLoop:
    env: SimulatedResearchEnvironment
    reflector: SimpleReflector
    bias_engine: BiasEngine
    selector: Selector
    cfg: AgentConfig
    history_fields: list[str] = field(default_factory=lambda: ["selected", "success", "quality"])

    def run(self, task: Task) -> EpisodeLog:
        budget_left = self.cfg.budget_tokens
        round_logs: list[RoundLog] = []
        history: list[dict[str, object]] = []
        signal = ReflectionSignal()

        success = False
        for round_idx in range(1, self.cfg.max_rounds + 1):
            if budget_left <= 0:
                break

            candidates = self.env.propose_candidates(task, self.cfg.num_candidates)

            if self.cfg.enable_bias:
                bias_scores = self.bias_engine.score(candidates, signal)
            else:
                bias_scores = {c.candidate_id: 0.0 for c in candidates}

            self.selector.apply_scores(candidates, bias_scores)
            selected = self.selector.select(candidates)
            outcome = self.env.step(task, selected)
            budget_left -= outcome.cost_tokens

            round_state = RoundState(
                task_id=task.task_id,
                round_idx=round_idx,
                budget_left=budget_left,
                history=list(history),
            )

            if self.cfg.enable_reflection:
                new_signal = self.reflector.reflect(round_state, outcome, selected.tags)
            else:
                new_signal = ReflectionSignal(notes="Reflection disabled.")

            round_logs.append(
                RoundLog(
                    task_id=task.task_id,
                    round_idx=round_idx,
                    candidates=[c.to_dict() for c in candidates],
                    selected_candidate_id=selected.candidate_id,
                    selected_description=selected.description,
                    outcome=outcome.to_dict(),
                    reflection_signal=new_signal.to_dict(),
                )
            )

            history.append(
                {
                    "selected": selected.description,
                    "success": outcome.success,
                    "quality": round(outcome.quality_score, 3),
                }
            )

            signal = new_signal
            if outcome.success:
                success = True
                break

        total_tokens = sum(r.outcome["cost_tokens"] for r in round_logs)
        return EpisodeLog(
            task_id=task.task_id,
            method_name=self.cfg.method_name,
            rounds=round_logs,
            success=success,
            steps=len(round_logs),
            total_tokens=total_tokens,
        )

