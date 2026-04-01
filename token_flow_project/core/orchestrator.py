"""
多 Agent 编排器：按 pipeline 顺序执行，记录 Token 消耗、Prompt 结构、Agent 通信。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional

from research import (
    RunReport,
    PromptStructure,
    StepTokenRecord,
    CommunicationRecord,
    TaskReport,
    MultiTaskReport,
)
from core.registry import resolve_agent, get_agent_display_name
from utils import count_tokens_approx, save_prompt


@dataclass
class RunResult:
    """单次流水线运行结果。"""
    outputs: List[str] = field(default_factory=list)  # 每个 Agent 的输出
    report: Optional[RunReport] = None


def _call_llm(prompt: str, config: dict) -> tuple[str, int, int]:
    """调用 LLM，返回 (response_text, prompt_tokens, completion_tokens)。"""
    import requests
    model = config.get("model", "qwen2:7b")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.get("temperature", 0.7),
            "num_predict": config.get("max_tokens", 512),
        },
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    response_text = data.get("response", "")
    eval_count = data.get("eval_count")
    prompt_tokens = count_tokens_approx(prompt)
    completion_tokens = int(eval_count) if eval_count is not None else count_tokens_approx(response_text)
    return response_text, prompt_tokens, completion_tokens


def run_pipeline(
    config: dict,
    *,
    initial_input: Optional[str] = None,
    task_index: Optional[int] = None,
    round_index: Optional[int] = None,
    call_llm: Optional[Callable[[str], tuple[str, int, int]]] = None,
    on_step_done: Optional[Callable[[int, str, str, dict], None]] = None,
) -> RunResult:
    """
    按 config.pipeline 顺序执行多 Agent，收集 Token、Prompt 结构、通信记录。

    :param config: 须含 pipeline (list[str]), agents (dict), initial_input (str) 或通过参数传入
    :param initial_input: 若提供则覆盖 config.initial_input
    :param task_index: 多任务时传入，用于记录与保存 prompt 路径
    :param round_index: 多轮时传入，同上
    :param call_llm: 若提供则用此函数替代默认 Ollama 调用
    :param on_step_done: 每步结束回调 (step_index, agent_name, response, step_record)
    :return: RunResult(outputs, report)
    """
    pipeline = config.get("pipeline") or ["analyst", "summarizer", "reviewer"]
    agents_config = config.get("agents") or {}
    input_text = initial_input if initial_input is not None else config.get("initial_input", "")

    invoke = call_llm or (lambda p: _call_llm(p, config))
    report = RunReport()
    outputs: List[str] = []
    t_idx = task_index if task_index is not None else 0
    r_idx = round_index if round_index is not None else 0

    # 当前步的“任务输入”= 上一步输出（第一步为 initial_input）；context 为更早的上下文（可选）
    task_input = input_text
    context = ""
    from_agent = "user"

    for step_index, agent_key in enumerate(pipeline):
        build_fn, get_blocks_fn, _ = resolve_agent(agent_key, agents_config)
        agent_name = get_agent_display_name(agent_key, agents_config)

        # 构建 prompt 与分块（用于结构研究）
        full_prompt = build_fn(task_input, context)
        role_block, context_block, task_block = get_blocks_fn(task_input, context)

        # 记录通信：谁 -> 谁，传递了多少
        content_tokens = count_tokens_approx(task_input)
        report.communications.append(
            CommunicationRecord(
                step_index=step_index,
                from_agent=from_agent,
                to_agent=agent_name,
                content_length=len(task_input),
                content_tokens=content_tokens,
                summary=task_input[:200] + "..." if len(task_input) > 200 else task_input,
                task_index=t_idx,
                round_index=r_idx,
            )
        )

        # 保存 prompt 原文（研究用）
        save_prompt(
            step_index + 1,
            full_prompt,
            task_index=task_index,
            round_index=round_index,
        )

        # 调用 LLM
        response, p_tok, c_tok = invoke(full_prompt)
        outputs.append(response)

        # Token 记录
        report.token_steps.append(
            StepTokenRecord(
                step_index=step_index,
                agent_name=agent_name,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
                task_index=t_idx,
                round_index=r_idx,
            )
        )

        # Prompt 结构记录
        report.prompt_structures.append(
            PromptStructure(
                step_index=step_index,
                agent_name=agent_name,
                role_tokens=count_tokens_approx(role_block),
                context_tokens=count_tokens_approx(context_block),
                task_tokens=count_tokens_approx(task_block),
                total_tokens=p_tok,
                task_index=t_idx,
                round_index=r_idx,
            )
        )

        if on_step_done:
            on_step_done(step_index, agent_name, response, report.token_steps[-1].to_dict())

        # 下一步的输入 = 当前 Agent 输出
        task_input = response
        from_agent = agent_name
        # 可选：把当前输出作为下一段的 context，此处保持简单流水线，context 仍为空
        # context = response  # 若需“上文”可在此扩展

    return RunResult(outputs=outputs, report=report)


def run_multi_task_pipeline(
    config: dict,
    *,
    call_llm: Optional[Callable[[str], tuple[str, int, int]]] = None,
    on_task_round_done: Optional[Callable[[int, str, int, RunReport], None]] = None,
) -> MultiTaskReport:
    """
    多任务、每任务多轮对话：对 config.tasks 中每个任务执行 rounds_per_task 轮流水线，
    每轮输入为上一轮最后一 Agent 的输出（首轮为任务输入）。全部统计记录在 MultiTaskReport。

    :param config: 须含 tasks (list[str] 或 list[dict with name/input]), rounds_per_task (int)
    :param call_llm: 可选，替代默认 LLM 调用
    :param on_task_round_done: 可选，每任务每轮结束回调 (task_index, task_name, round_index, round_report)
    :return: MultiTaskReport
    """
    raw_tasks = config.get("tasks")
    if not raw_tasks:
        # 兼容：无 tasks 时退化为单任务单轮，用 initial_input
        single_input = config.get("initial_input", "")
        result = run_pipeline(
            config,
            initial_input=single_input,
            call_llm=call_llm,
        )
        report = result.report or RunReport()
        multi = MultiTaskReport(tasks=[TaskReport(0, "Task1", single_input, rounds=[report])])
        return multi

    rounds_per_task = max(1, int(config.get("rounds_per_task", 2)))
    multi_report = MultiTaskReport()

    for task_index, raw in enumerate(raw_tasks):
        if isinstance(raw, str):
            task_name = f"Task{task_index + 1}"
            task_input = raw
        else:
            task_name = raw.get("name", f"Task{task_index + 1}")
            task_input = raw.get("input", raw.get("task_input", ""))

        task_report = TaskReport(task_index=task_index, task_name=task_name, task_input=task_input)
        round_input = task_input

        for round_index in range(rounds_per_task):
            result = run_pipeline(
                config,
                initial_input=round_input,
                task_index=task_index,
                round_index=round_index,
                call_llm=call_llm,
            )
            report = result.report or RunReport()
            # 确保所有记录带 task_index / round_index（run_pipeline 已写入）
            for s in report.token_steps:
                s.task_index = task_index
                s.round_index = round_index
            for p in report.prompt_structures:
                p.task_index = task_index
                p.round_index = round_index
            for c in report.communications:
                c.task_index = task_index
                c.round_index = round_index
            task_report.rounds.append(report)
            if on_task_round_done:
                on_task_round_done(task_index, task_name, round_index, report)
            # 下一轮输入 = 本轮最后一 Agent 输出
            round_input = result.outputs[-1] if result.outputs else round_input

        multi_report.tasks.append(task_report)

    return multi_report
