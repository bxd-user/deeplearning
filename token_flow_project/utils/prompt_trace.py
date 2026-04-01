"""
将每一步调用 LLM 的完整 prompt 保存到 outputs/prompts/。
多任务多轮时：prompts/task_0/round_0/step_1_prompt.txt
"""
from pathlib import Path
from typing import Optional

# 默认相对项目根目录
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "prompts"


def save_prompt(
    step: int,
    prompt_text: str,
    prompts_dir: Optional[Path] = None,
    task_index: Optional[int] = None,
    round_index: Optional[int] = None,
) -> Path:
    """
    保存第 step 步的完整 prompt。
    :param step: 步骤编号，从 1 开始
    :param prompt_text: 完整 prompt 字符串
    :param prompts_dir: 保存目录，默认 outputs/prompts
    :param task_index: 多任务时的任务下标，与 round_index 一起则保存到子目录 task_N/round_M/
    :param round_index: 多轮时的轮次下标
    :return: 写入的文件路径
    """
    base = prompts_dir or PROMPTS_DIR
    if task_index is not None and round_index is not None:
        base = base / f"task_{task_index}" / f"round_{round_index}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"step_{step}_prompt.txt"
    path.write_text(prompt_text, encoding="utf-8")
    return path


def get_prompts_dir() -> Path:
    return PROMPTS_DIR
