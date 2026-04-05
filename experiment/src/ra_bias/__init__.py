"""Core package for Reflection-as-Bias experiments."""

from .agent_loop import AgentConfig, ResearchAgentLoop
from .bias_engine import BiasEngine, HardPruningBiasEngine, NoBiasEngine, TagBiasEngine
from .environment import SimulatedResearchEnvironment
from .models import Candidate, EpisodeLog, Outcome, ReflectionSignal, RoundLog, RoundState, Task
from .reflection import SimpleReflector
from .selector import SelectionConfig, Selector

__all__ = [
    "AgentConfig",
    "BiasEngine",
    "Candidate",
    "EpisodeLog",
    "HardPruningBiasEngine",
    "NoBiasEngine",
    "Outcome",
    "ReflectionSignal",
    "ResearchAgentLoop",
    "RoundLog",
    "RoundState",
    "SelectionConfig",
    "Selector",
    "SimpleReflector",
    "SimulatedResearchEnvironment",
    "TagBiasEngine",
    "Task",
]
