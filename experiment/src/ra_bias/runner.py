from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .agent_loop import AgentConfig, ResearchAgentLoop
from .bias_engine import (
    HardPruningBiasEngine,
    LLMBiasEngine,
    NoBiasEngine,
    TagBiasEngine,
    TextualReflectionEngine,
    WeakTagBiasEngine,
)
from .environment import SimulatedResearchEnvironment, build_demo_tasks
from .evaluation import Metrics, compute_metrics
from .logger import JsonlLogger
from .models import EpisodeLog, Task
from .reflection import SimpleReflector
from .selector import SelectionConfig, Selector

# ─── Method groups ────────────────────────────────────────────────────────────

# Main comparison (RQ1–RQ3, statics.md §1)
MAIN_METHODS = ["single_path", "branching_only", "textual_reflection", "reflection_as_bias"]

# Ablation 1: Hard Pruning vs Soft Bias (design.md §7.2, statics.md §3.1)
ABLATION_HARD_VS_SOFT = ["hard_pruning", "reflection_as_bias"]

# Ablation 2: Tag-based vs LLM-based Bias (design.md §7.3, statics.md §3.2)
ABLATION_BIAS_TYPE = ["reflection_as_bias", "llm_bias"]

# Ablation 3: No / Weak / Full Bias strength (design.md §7.4, statics.md §3.3)
ABLATION_BIAS_STRENGTH = ["branching_only", "weak_bias", "reflection_as_bias"]


# ─── Agent factory ────────────────────────────────────────────────────────────

def build_agent(method: str, seed: int) -> ResearchAgentLoop:
    env = SimulatedResearchEnvironment(seed=seed)
    reflector = SimpleReflector()
    selector = Selector(SelectionConfig(alpha=1.0, mode="greedy"))

    if method == "single_path":
        cfg = AgentConfig(method_name=method, num_candidates=1,
                          enable_reflection=False, enable_bias=False)
        bias_engine = NoBiasEngine()

    elif method == "branching_only":
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=False, enable_bias=False)
        bias_engine = NoBiasEngine()

    elif method == "textual_reflection":
        # Reflection exists but only as implicit text context — modelled as a
        # weak, noisy bias engine (design.md §5.3).  Differentiated from
        # branching_only: non-zero signal influence, but weaker than TagBiasEngine.
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=True, enable_bias=True)
        bias_engine = TextualReflectionEngine(seed=seed)

    elif method == "reflection_as_bias":
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=True, enable_bias=True)
        bias_engine = TagBiasEngine()

    elif method == "hard_pruning":
        # Ablation 1: eliminate avoid-tag candidates outright (design.md §7.2)
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=True, enable_bias=True)
        bias_engine = HardPruningBiasEngine()

    elif method == "llm_bias":
        # Ablation 2: LLM-based scoring (design.md §7.3)
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=True, enable_bias=True)
        bias_engine = LLMBiasEngine(seed=seed)

    elif method == "weak_bias":
        # Ablation 3: reduced bias weights (design.md §7.4)
        cfg = AgentConfig(method_name=method, num_candidates=4,
                          enable_reflection=True, enable_bias=True)
        bias_engine = WeakTagBiasEngine()

    else:
        raise ValueError(f"Unknown method: {method}")

    return ResearchAgentLoop(
        env=env, reflector=reflector, bias_engine=bias_engine,
        selector=selector, cfg=cfg,
    )


# ─── Episode replication ──────────────────────────────────────────────────────

def replicate_tasks(num_episodes: int) -> list[Task]:
    """Replicate all demo tasks across episodes for statistical stability."""
    base_tasks = build_demo_tasks()
    episodes: list[Task] = []
    for i in range(num_episodes):
        for t in base_tasks:
            episodes.append(replace(t, task_id=f"{t.task_id}_ep{i + 1}"))
    return episodes


# ─── Single-method runner ─────────────────────────────────────────────────────

def run_method(method: str, num_episodes: int, seed: int = 7) -> tuple[list[EpisodeLog], Metrics]:
    tasks = replicate_tasks(num_episodes)
    episodes: list[EpisodeLog] = []

    for idx, task in enumerate(tasks):
        agent = build_agent(method=method, seed=seed + idx)
        episodes.append(agent.run(task))

    metrics = compute_metrics(method, episodes)
    return episodes, metrics


# ─── Multi-method runners ─────────────────────────────────────────────────────

def run_methods(
    methods: list[str],
    num_episodes: int,
    output_dir: str,
) -> list[Metrics]:
    logger = JsonlLogger(output_dir=Path(output_dir))
    all_metrics: list[Metrics] = []
    all_episodes: list[EpisodeLog] = []

    for method in methods:
        episodes, metrics = run_method(method=method, num_episodes=num_episodes)
        for ep in episodes:
            logger.write_episode(ep)
        all_episodes.extend(episodes)
        all_metrics.append(metrics)

    logger.write_summary(all_episodes, filename="summary.json")
    return all_metrics


def run_ablations(
    num_episodes: int,
    output_dir: str,
) -> dict[str, list[Metrics]]:
    """
    Run all three ablation groups and return their metrics.
    Logs are written to separate subdirectories for clarity.
    """
    base = Path(output_dir)
    results: dict[str, list[Metrics]] = {}

    ablation_groups = {
        "hard_vs_soft": ABLATION_HARD_VS_SOFT,
        "bias_type": ABLATION_BIAS_TYPE,
        "bias_strength": ABLATION_BIAS_STRENGTH,
    }

    for name, methods in ablation_groups.items():
        metrics = run_methods(
            methods=methods,
            num_episodes=num_episodes,
            output_dir=str(base / name),
        )
        results[name] = metrics

    return results
