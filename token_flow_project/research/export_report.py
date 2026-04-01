"""
将 RunReport / MultiTaskReport 导出为 JSON / CSV，便于后续分析与绘图。
"""
import json
from pathlib import Path
from typing import Optional

from .models import RunReport, MultiTaskReport


def export_report_json(report: RunReport, path: Path) -> Path:
    """导出完整报表为 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_report_csv(report: RunReport, base_dir: Path) -> dict:
    """
    导出多张 CSV：token_steps, prompt_structures, communications。
    返回 { "token_steps": path, "prompt_structures": path, "communications": path }。
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # Token 消耗
    p = base_dir / "token_steps.csv"
    rows = [["step_index", "agent_name", "prompt_tokens", "completion_tokens", "total_tokens", "task_index", "round_index"]]
    for s in report.token_steps:
        rows.append([s.step_index, s.agent_name, s.prompt_tokens, s.completion_tokens, s.total_tokens, getattr(s, "task_index", 0), getattr(s, "round_index", 0)])
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows), encoding="utf-8")
    out["token_steps"] = p

    # Prompt 结构
    p = base_dir / "prompt_structures.csv"
    rows = [["step_index", "agent_name", "role_tokens", "context_tokens", "task_tokens", "total_tokens", "task_index", "round_index"]]
    for s in report.prompt_structures:
        rows.append([s.step_index, s.agent_name, s.role_tokens, s.context_tokens, s.task_tokens, s.total_tokens, getattr(s, "task_index", 0), getattr(s, "round_index", 0)])
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows), encoding="utf-8")
    out["prompt_structures"] = p

    # Agent 通信
    p = base_dir / "communications.csv"
    rows = [["step_index", "from_agent", "to_agent", "content_length", "content_tokens", "task_index", "round_index"]]
    for c in report.communications:
        rows.append([c.step_index, c.from_agent, c.to_agent, c.content_length, c.content_tokens, getattr(c, "task_index", 0), getattr(c, "round_index", 0)])
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows), encoding="utf-8")
    out["communications"] = p

    return out


def export_multi_task_report_json(multi: MultiTaskReport, path: Path) -> Path:
    """导出多任务多轮完整报表为 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(multi.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_multi_task_report_csv(multi: MultiTaskReport, base_dir: Path) -> dict:
    """
    导出多任务多轮 CSV：带 task_index、round_index 的 token_steps / prompt_structures / communications，
    以及汇总 summary.csv（每任务每轮 total_tokens）。
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    # 展平 token_steps
    steps = multi.flatten_token_steps()
    p = base_dir / "token_steps.csv"
    rows = [["task_index", "round_index", "step_index", "agent_name", "prompt_tokens", "completion_tokens", "total_tokens"]]
    for s in steps:
        rows.append([s.task_index, s.round_index, s.step_index, s.agent_name, s.prompt_tokens, s.completion_tokens, s.total_tokens])
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows), encoding="utf-8")
    out["token_steps"] = p

    # 展平 prompt_structures
    rows_ps = [["task_index", "round_index", "step_index", "agent_name", "role_tokens", "context_tokens", "task_tokens", "total_tokens"]]
    for t in multi.tasks:
        for ri, r in enumerate(t.rounds):
            for s in r.prompt_structures:
                rows_ps.append([t.task_index, ri, s.step_index, s.agent_name, s.role_tokens, s.context_tokens, s.task_tokens, s.total_tokens])
    p = base_dir / "prompt_structures.csv"
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows_ps), encoding="utf-8")
    out["prompt_structures"] = p

    # 展平 communications
    rows_com = [["task_index", "round_index", "step_index", "from_agent", "to_agent", "content_length", "content_tokens"]]
    for t in multi.tasks:
        for ri, r in enumerate(t.rounds):
            for c in r.communications:
                rows_com.append([t.task_index, ri, c.step_index, c.from_agent, c.to_agent, c.content_length, c.content_tokens])
    p = base_dir / "communications.csv"
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows_com), encoding="utf-8")
    out["communications"] = p

    # 汇总：每任务每轮 total_tokens
    p = base_dir / "summary.csv"
    rows_sum = [["task_index", "task_name", "round_index", "round_total_tokens", "task_total_tokens"]]
    for t in multi.tasks:
        task_total = t.total_tokens()
        for ri, r in enumerate(t.rounds):
            rows_sum.append([t.task_index, t.task_name, ri, r.total_tokens(), task_total])
    p.write_text("\n".join(",".join(str(x) for x in row) for row in rows_sum), encoding="utf-8")
    out["summary"] = p

    return out
