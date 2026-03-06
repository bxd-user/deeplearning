"""
Token 使用记录与汇总打印。
"""
from typing import List, Optional


class TokenLogger:
    """记录每个步骤的 prompt/completion/total tokens，并支持按步与总汇总打印。"""

    def __init__(self):
        self._steps: List[dict] = []  # [{"agent": str, "prompt": int, "completion": int, "total": int}, ...]

    def log_prompt_tokens(self, step_index: int, agent_name: str, count: int) -> None:
        """记录当前步骤的 prompt tokens（若该步已存在则更新，否则追加新步）。"""
        self._ensure_step(step_index, agent_name)
        self._steps[step_index]["prompt"] = count
        self._steps[step_index]["total"] = (
            self._steps[step_index].get("prompt", 0) + self._steps[step_index].get("completion", 0)
        )

    def log_completion_tokens(self, step_index: int, agent_name: str, count: int) -> None:
        """记录当前步骤的 completion tokens。"""
        self._ensure_step(step_index, agent_name)
        self._steps[step_index]["completion"] = count
        self._steps[step_index]["total"] = (
            self._steps[step_index].get("prompt", 0) + self._steps[step_index].get("completion", 0)
        )

    def log_step(self, step_index: int, agent_name: str, prompt_tokens: int, completion_tokens: int) -> None:
        """一次性记录某步的 prompt 与 completion tokens。"""
        self._ensure_step(step_index, agent_name)
        self._steps[step_index]["prompt"] = prompt_tokens
        self._steps[step_index]["completion"] = completion_tokens
        self._steps[step_index]["total"] = prompt_tokens + completion_tokens

    def _ensure_step(self, step_index: int, agent_name: str) -> None:
        while len(self._steps) <= step_index:
            self._steps.append({"agent": agent_name, "prompt": 0, "completion": 0, "total": 0})
        self._steps[step_index]["agent"] = agent_name

    def print_step_summary(self, step_index: int) -> None:
        """打印单步汇总。"""
        if step_index >= len(self._steps):
            return
        s = self._steps[step_index]
        print(f"Agent: {s['agent']}")
        print(f"  Prompt tokens: {s['prompt']}")
        print(f"  Completion tokens: {s['completion']}")
        print(f"  Total tokens: {s['total']}")

    def print_total_summary(self) -> None:
        """打印所有步骤及总 token 汇总。"""
        total_prompt = 0
        total_completion = 0
        for i, s in enumerate(self._steps):
            self.print_step_summary(i)
            total_prompt += s["prompt"]
            total_completion += s["completion"]
        print("---")
        print(f"Total prompt tokens: {total_prompt}")
        print(f"Total completion tokens: {total_completion}")
        print(f"Total tokens: {total_prompt + total_completion}")

    def get_steps(self) -> List[dict]:
        """返回各步记录，供绘图或 CSV 使用。"""
        return list(self._steps)

    def total_tokens(self) -> int:
        return sum(s["total"] for s in self._steps)
