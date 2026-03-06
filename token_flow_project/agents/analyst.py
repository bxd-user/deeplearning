"""
Analyst Agent：负责分析阶段。
"""
from typing import Dict, Any


def get_analyst_config() -> Dict[str, Any]:
    """返回 Analyst 的配置，供 main 或 Crew 使用。"""
    return {
        "name": "Analyst",
        "role": "数据分析师",
        "goal": "对给定主题或数据进行清晰、结构化的分析",
        "backstory": "你擅长拆解问题、归纳要点并给出可执行的结论。",
    }


def build_analyst_prompt(task_input: str, context: str = "") -> str:
    """构建 Analyst 的完整 prompt（用于逐步执行时记录与发送）。"""
    cfg = get_analyst_config()
    prompt = f"你扮演【{cfg['role']】。\n目标：{cfg['goal']}\n背景：{cfg['backstory']}\n\n请对以下内容进行分析并输出结构化结论。\n\n"
    if context:
        prompt += f"上文上下文：\n{context}\n\n"
    prompt += f"本次任务输入：\n{task_input}"
    return prompt
