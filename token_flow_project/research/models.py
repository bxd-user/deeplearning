"""
研究用数据模型：Prompt 结构、Token 消耗、Agent 通信。
用于多 Agent 协作研究的统一记录与导出。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum


class MessageRole(str, Enum):
    """消息角色（用于 Prompt 结构分析）。"""
    SYSTEM = "system"   # 角色/目标/背景
    CONTEXT = "context" # 上文/其他 Agent 输出
    USER = "user"      # 本次任务输入


@dataclass
class PromptStructure:
    """单步 Prompt 的结构化拆解，便于研究各部分的 token 占比。"""
    step_index: int
    agent_name: str
    role_tokens: int      # 角色/目标/背景
    context_tokens: int   # 上文或前序 Agent 输出
    task_tokens: int      # 本次任务输入
    total_tokens: int
    task_index: int = 0   # 多任务时：任务下标
    round_index: int = 0  # 多轮时：轮次下标

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "agent_name": self.agent_name,
            "role_tokens": self.role_tokens,
            "context_tokens": self.context_tokens,
            "task_tokens": self.task_tokens,
            "total_tokens": self.total_tokens,
            "task_index": self.task_index,
            "round_index": self.round_index,
        }


@dataclass
class StepTokenRecord:
    """单步 Token 消耗记录。"""
    step_index: int
    agent_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    task_index: int = 0
    round_index: int = 0

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "agent_name": self.agent_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "task_index": self.task_index,
            "round_index": self.round_index,
        }


@dataclass
class CommunicationRecord:
    """Agent 间通信记录：谁在何时把什么传给谁。"""
    step_index: int
    from_agent: str       # 发送方（可为 "user" 表示初始输入）
    to_agent: str         # 接收方
    content_length: int   # 传递内容长度（字符）
    content_tokens: int   # 传递内容约 token 数
    summary: Optional[str] = None  # 可选摘要，便于人工查看
    task_index: int = 0
    round_index: int = 0

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content_length": self.content_length,
            "content_tokens": self.content_tokens,
            "summary": self.summary,
            "task_index": self.task_index,
            "round_index": self.round_index,
        }


@dataclass
class RunReport:
    """单轮运行的研究报表：多 Agent 协作 + Token + Prompt 结构 + 通信。"""
    token_steps: List[StepTokenRecord] = field(default_factory=list)
    prompt_structures: List[PromptStructure] = field(default_factory=list)
    communications: List[CommunicationRecord] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.token_steps)

    def to_dict(self) -> dict:
        return {
            "token_steps": [s.to_dict() for s in self.token_steps],
            "prompt_structures": [p.to_dict() for p in self.prompt_structures],
            "communications": [c.to_dict() for c in self.communications],
            "total_tokens": self.total_tokens(),
        }


@dataclass
class TaskReport:
    """单个任务的多轮报表。"""
    task_index: int
    task_name: str
    task_input: str
    rounds: List[RunReport] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(r.total_tokens() for r in self.rounds)

    def to_dict(self) -> dict:
        return {
            "task_index": self.task_index,
            "task_name": self.task_name,
            "task_input": self.task_input,
            "rounds": [r.to_dict() for r in self.rounds],
            "total_tokens": self.total_tokens(),
        }


@dataclass
class MultiTaskReport:
    """多任务、每任务多轮对话的完整统计。"""
    tasks: List[TaskReport] = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(t.total_tokens() for t in self.tasks)

    def to_dict(self) -> dict:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "total_tokens": self.total_tokens(),
        }

    def flatten_token_steps(self) -> List[StepTokenRecord]:
        """展平为带 task_index/round_index 的 step 列表，便于 CSV/图表。"""
        out: List[StepTokenRecord] = []
        for t in self.tasks:
            for r in t.rounds:
                out.extend(r.token_steps)
        return out
