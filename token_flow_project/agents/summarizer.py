"""
Summarizer Agent：负责总结阶段。
"""
from typing import Dict, Any


def get_summarizer_config() -> Dict[str, Any]:
    """返回 Summarizer 的配置。"""
    return {
        "name": "Summarizer",
        "role": "总结专家",
        "goal": "将分析结果或长文本提炼为简洁、条理清晰的总结",
        "backstory": "你善于抓住重点，用简短语言概括核心结论。",
    }


def build_summarizer_prompt(task_input: str, context: str = "") -> str:
    """构建 Summarizer 的完整 prompt。"""
    cfg = get_summarizer_config()
    prompt = f"你扮演【{cfg['role']】。\n目标：{cfg['goal']}\n背景：{cfg['backstory']}\n\n请对以下内容进行总结。\n\n"
    if context:
        prompt += f"上文上下文：\n{context}\n\n"
    prompt += f"待总结内容：\n{task_input}"
    return prompt
