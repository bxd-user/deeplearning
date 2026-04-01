"""
通用 Prompt 构建：从 config 中的 agent 定义生成 role/context/task 分块，便于研究与统计。
"""
from typing import Dict, Any


def build_prompt_from_config(
    agent_cfg: Dict[str, Any],
    task_input: str,
    context: str = "",
    task_label: str = "本次任务输入",
) -> str:
    """根据 config 中的 agent 定义构建完整 prompt。"""
    role = agent_cfg.get("role", "")
    goal = agent_cfg.get("goal", "")
    backstory = agent_cfg.get("backstory", "")
    role_block = f"你扮演【{role}】。\n目标：{goal}\n背景：{backstory}\n\n"
    if context:
        role_block += f"上文上下文：\n{context}\n\n"
    return role_block + f"{task_label}：\n{task_input}"


def get_prompt_blocks_from_config(
    agent_cfg: Dict[str, Any],
    task_input: str,
    context: str = "",
    task_label: str = "本次任务输入",
) -> tuple[str, str, str]:
    """返回 (role_block, context_block, task_block) 用于 token 与结构研究。"""
    role = agent_cfg.get("role", "")
    goal = agent_cfg.get("goal", "")
    backstory = agent_cfg.get("backstory", "")
    role_block = f"你扮演【{role}】。\n目标：{goal}\n背景：{backstory}\n\n"
    context_block = f"上文上下文：\n{context}\n\n" if context else ""
    task_block = f"{task_label}：\n{task_input}"
    return role_block, context_block, task_block
