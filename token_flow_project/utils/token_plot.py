"""
Token 使用可视化：生成 outputs/graphs/token_usage.png（Agent Step vs Token Usage）。
"""
from pathlib import Path
from typing import List, Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

GRAPHS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "graphs"


def plot_token_usage(steps: List[Dict[str, Any]], save_path: Path = None) -> Path:
    """
    绘制 Agent Step vs Token Usage 柱状图。
    :param steps: [{"agent": str, "prompt": int, "completion": int, "total": int}, ...]
    :param save_path: 输出路径，默认 outputs/graphs/token_usage.png
    :return: 保存的文件路径
    """
    if plt is None:
        raise RuntimeError("请安装 matplotlib: pip install matplotlib")

    save_path = save_path or GRAPHS_DIR / "token_usage.png"
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    labels = [f"Step{i+1}\n{s.get('agent','')}" for i, s in enumerate(steps)]
    totals = [s.get("total", 0) for s in steps]
    prompt_vals = [s.get("prompt", 0) for s in steps]
    completion_vals = [s.get("completion", 0) for s in steps]

    x = range(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.bar(x, prompt_vals, width, label="Prompt", color="darkorange", alpha=0.9)
    bars = ax.bar(x, completion_vals, width, bottom=prompt_vals, label="Completion", color="seagreen", alpha=0.9)

    ax.set_ylabel("Token Usage")
    ax.set_xlabel("Agent Step")
    ax.set_title("Agent Step vs Token Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, totals)):
        if val > 0:
            y_top = bar.get_y() + bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_top + 5, str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
