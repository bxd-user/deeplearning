"""
Reviewer Agent：负责审阅与润色阶段。
"""
from typing import Dict, Any


def get_reviewer_config() -> Dict[str, Any]:
    """返回 Reviewer 的配置。"""
    return {
        "name": "Reviewer",
        "role": "审阅专家",
        "goal": "对总结或报告进行审阅、润色，确保逻辑清晰、表述准确",
        "backstory": "你注重细节与可读性，善于发现疏漏并给出改进建议。",
    }


def _role_block(cfg: dict) -> str:
    nl = "\n"
    return f"你扮演【{cfg['role']}】。{nl}目标：{cfg['goal']}{nl}背景：{cfg['backstory']}{nl}{nl}请对以下内容进行审阅并输出润色后的结论。{nl}{nl}"


def build_reviewer_prompt(task_input: str, context: str = "") -> str:
    """构建 Reviewer 的完整 prompt。"""
    cfg = get_reviewer_config()
    prompt = _role_block(cfg)
    if context:
        prompt += f"上文上下文：\n{context}\n\n"
    prompt += f"待审阅内容：\n{task_input}"
    return prompt


def get_reviewer_prompt_blocks(task_input: str, context: str = "") -> tuple[str, str, str]:
    """返回 (role_block, context_block, task_block) 用于 Prompt 结构研究。"""
    cfg = get_reviewer_config()
    role = _role_block(cfg)
    ctx_block = f"上文上下文：\n{context}\n\n" if context else ""
    task_block = f"待审阅内容：\n{task_input}"
    return role, ctx_block, task_block
