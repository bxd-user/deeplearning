"""
LLM 多 Agent 研究工具 — 主入口。
支持：多任务、每任务多轮对话、多 Agent 协作、Token 消耗、Prompt 结构、Agent 通信。
从 config.yaml 读取流水线配置，执行后输出研究报表与图表。
运行: python main.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import run_pipeline, run_multi_task_pipeline, RunResult
from research import (
    RunReport,
    MultiTaskReport,
    export_report_json,
    export_report_csv,
    export_multi_task_report_json,
    export_multi_task_report_csv,
)
from utils import plot_token_usage
from utils.token_plot import GRAPHS_DIR


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        return {
            "model": "qwen2:7b",
            "temperature": 0.7,
            "max_tokens": 512,
            "pipeline": ["analyst", "summarizer", "reviewer"],
            "agents": {},
            "tasks": [],
            "rounds_per_task": 2,
            "initial_input": "大语言模型在代码生成中的应用与局限",
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    use_multi_task = bool(config.get("tasks"))

    if use_multi_task:
        _run_multi_task(config)
    else:
        _run_single(config)


def _on_step(step_index: int, agent_name: str, response: str, step_record: dict):
    print(f"    Step {step_index + 1} {agent_name}: prompt={step_record['prompt_tokens']}, completion={step_record['completion_tokens']}, total={step_record['total_tokens']}")


def _run_single(config: dict):
    """单任务单轮：兼容原逻辑。"""
    result: RunResult = run_pipeline(config, on_step_done=_on_step)
    report: RunReport = result.report
    if not report:
        print("未生成报表")
        return
    print("\n--- Total Token ---")
    print(f"Total tokens: {report.total_tokens()}")
    _print_report(report)
    _export_and_plot(config, report=report)
    _append_experiment_log(config, report.total_tokens())


def _run_multi_task(config: dict):
    """多任务、每任务多轮：完整统计记录。"""
    def on_task_round(task_index: int, task_name: str, round_index: int, round_report: RunReport):
        print(f"\n--- Task {task_index + 1} {task_name} / Round {round_index + 1} ---")
        print(f"  Round tokens: {round_report.total_tokens()}")
        for s in round_report.token_steps:
            print(f"    Step {s.step_index + 1} {s.agent_name}: total={s.total_tokens}")

    multi: MultiTaskReport = run_multi_task_pipeline(config, on_task_round_done=on_task_round)

    print("\n--- 多任务多轮汇总 ---")
    for t in multi.tasks:
        print(f"  {t.task_name}: {len(t.rounds)} 轮, 共 {t.total_tokens()} tokens")
    print(f"  总计: {multi.total_tokens()} tokens")

    out_dir = ROOT / "outputs" / "research"
    export_multi_task_report_json(multi, out_dir / "multi_task_report.json")
    csv_paths = export_multi_task_report_csv(multi, out_dir)
    print("\n研究报表已导出:")
    print(f"  JSON: {out_dir / 'multi_task_report.json'}")
    for k, p in csv_paths.items():
        print(f"  {k}: {p}")

    # 图表：按展平后的 step 绘制（可分组或整体）
    steps_flat = multi.flatten_token_steps()
    if steps_flat:
        graph_path = GRAPHS_DIR / "token_usage_multi_task.png"
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        plot_token_usage([s.to_dict() for s in steps_flat], graph_path)
        print(f"\n图表已保存: {graph_path}")

    _append_experiment_log(config, multi.total_tokens(), n_tasks=len(multi.tasks), rounds=len(multi.tasks[0].rounds) if multi.tasks else 0)


def _print_report(report: RunReport):
    print("\n--- Prompt 结构 ---")
    for ps in report.prompt_structures:
        print(f"  Step {ps.step_index + 1} {ps.agent_name}: role={ps.role_tokens}, context={ps.context_tokens}, task={ps.task_tokens}")
    print("\n--- Agent 通信 ---")
    for c in report.communications:
        print(f"  Step {c.step_index + 1}: {c.from_agent} -> {c.to_agent} (tokens≈{c.content_tokens})")


def _export_and_plot(config: dict, *, report: RunReport):
    out_dir = ROOT / "outputs" / "research"
    export_report_json(report, out_dir / "report.json")
    csv_paths = export_report_csv(report, out_dir)
    print("研究报表已导出:")
    print(f"  JSON: {out_dir / 'report.json'}")
    for k, p in csv_paths.items():
        print(f"  {k}: {p}")
    steps_for_plot = [s.to_dict() for s in report.token_steps]
    graph_path = GRAPHS_DIR / "token_usage.png"
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    plot_token_usage(steps_for_plot, graph_path)
    print(f"\n图表已保存: {graph_path}")


def _append_experiment_log(config: dict, total_tokens: int, n_tasks: int = 1, rounds: int = 1):
    logs_dir = ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "token_experiment.csv"
    n_agents = len(config.get("pipeline") or [])
    with open(csv_path, "a", encoding="utf-8") as f:
        if csv_path.stat().st_size == 0:
            f.write("prompt_length,agent_number,tasks,rounds_per_task,total_tokens\n")
        f.write(f"medium,{n_agents},{n_tasks},{rounds},{total_tokens}\n")
    print(f"实验记录已追加: {csv_path}")


if __name__ == "__main__":
    main()
