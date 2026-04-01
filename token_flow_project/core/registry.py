"""
Agent 注册表：根据 pipeline 配置解析 Agent，支持 config 驱动或代码驱动。
"""
from typing import Dict, Any, Callable, Optional

from agents.analyst import (
    build_analyst_prompt,
    get_analyst_prompt_blocks,
    get_analyst_config,
)
from agents.summarizer import (
    build_summarizer_prompt,
    get_summarizer_prompt_blocks,
    get_summarizer_config,
)
from agents.reviewer import (
    build_reviewer_prompt,
    get_reviewer_prompt_blocks,
    get_reviewer_config,
)
from core.prompt_builder import build_prompt_from_config, get_prompt_blocks_from_config

# 代码中注册的 Agent：(build_prompt, get_blocks, default_task_label)
_CODE_AGENTS: Dict[str, tuple] = {
    "analyst": (build_analyst_prompt, get_analyst_prompt_blocks, "本次任务输入"),
    "summarizer": (build_summarizer_prompt, get_summarizer_prompt_blocks, "待总结内容"),
    "reviewer": (build_reviewer_prompt, get_reviewer_prompt_blocks, "待审阅内容"),
}


def resolve_agent(
    agent_key: str,
    agents_config: Optional[Dict[str, Any]] = None,
) -> tuple[Callable, Callable, str]:
    """
    解析 Agent：若 agents_config 中有该 key 的完整定义则用 config 构建，否则用代码。
    返回 (build_prompt_fn, get_blocks_fn, task_label)。
    build_prompt_fn(task_input, context) -> str
    get_blocks_fn(task_input, context) -> (role, context_block, task_block)
    """
    cfg = (agents_config or {}).get(agent_key)
    if cfg and isinstance(cfg, dict) and "role" in cfg and "goal" in cfg:
        def _build(task_input: str, context: str = "") -> str:
            return build_prompt_from_config(cfg, task_input, context, task_label="本次任务输入")

        def _blocks(task_input: str, context: str = "") -> tuple[str, str, str]:
            return get_prompt_blocks_from_config(cfg, task_input, context, task_label="本次任务输入")

        return _build, _blocks, "本次任务输入"
    if agent_key in _CODE_AGENTS:
        build_fn, blocks_fn, label = _CODE_AGENTS[agent_key]
        return build_fn, blocks_fn, label
    raise KeyError(f"未找到 Agent 定义: {agent_key}")


def get_agent_display_name(agent_key: str, agents_config: Optional[Dict[str, Any]] = None) -> str:
    """获取 Agent 显示名（用于日志与报表）。"""
    cfg = (agents_config or {}).get(agent_key)
    if isinstance(cfg, dict) and cfg.get("name"):
        return cfg["name"]
    if agent_key == "analyst":
        return get_analyst_config().get("name", "Analyst")
    if agent_key == "summarizer":
        return get_summarizer_config().get("name", "Summarizer")
    if agent_key == "reviewer":
        return get_reviewer_config().get("name", "Reviewer")
    return agent_key
