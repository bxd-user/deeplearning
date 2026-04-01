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


def _role_block(cfg: dict) -> str:
    nl = "\n"
    return f"你扮演【{cfg['role']}】。{nl}目标：{cfg['goal']}{nl}背景：{cfg['backstory']}{nl}{nl}请对以下内容进行总结。{nl}{nl}"


def build_summarizer_prompt(task_input: str, context: str = "") -> str:
    """构建 Summarizer 的完整 prompt。"""
    cfg = get_summarizer_config()
    prompt = _role_block(cfg)
    if context:
        prompt += f"上文上下文：\n{context}\n\n"
    prompt += f"待总结内容：\n{task_input}"
    return prompt


def get_summarizer_prompt_blocks(task_input: str, context: str = "") -> tuple[str, str, str]:
    """返回 (role_block, context_block, task_block) 用于 Prompt 结构研究。"""
    cfg = get_summarizer_config()
    role = _role_block(cfg)
    ctx_block = f"上文上下文：\n{context}\n\n" if context else ""
    task_block = f"待总结内容：\n{task_input}"
    return role, ctx_block, task_block
