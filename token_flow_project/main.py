"""
LLM Token Research Tool — 主入口。
从 config.yaml 读取配置，逐步执行 Agent 任务，记录 token、保存 prompt、生成图表。
运行: python main.py
"""
import sys
from pathlib import Path

import yaml

# 项目根
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents import build_analyst_prompt, build_summarizer_prompt
from utils import (
    TokenLogger,
    count_tokens_approx,
    save_prompt,
    plot_token_usage,
)
from utils.token_plot import GRAPHS_DIR


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        return {"model": "qwen2:7b", "temperature": 0.7, "max_tokens": 512, "agents": 2}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_ollama(prompt: str, config: dict) -> tuple[str, int, int]:
    """
    调用本地 Ollama，返回 (response_text, prompt_tokens, completion_tokens)。
    completion_tokens 优先使用 API 返回的 eval_count，否则用近似计数。
    """
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
    eval_count = data.get("eval_count")  # Ollama 返回的生成 token 数
    prompt_tokens = count_tokens_approx(prompt)
    completion_tokens = int(eval_count) if eval_count is not None else count_tokens_approx(response_text)
    return response_text, prompt_tokens, completion_tokens


def run_step(step_index: int, agent_name: str, prompt: str, config: dict, token_logger: TokenLogger) -> str:
    """执行单步：保存 prompt、调用 LLM、记录 token、打印步汇总。"""
    save_prompt(step_index + 1, prompt)  # step 从 1 开始
    response, p_tok, c_tok = call_ollama(prompt, config)
    token_logger.log_step(step_index, agent_name, p_tok, c_tok)
    token_logger.print_step_summary(step_index)
    return response


def main():
    config = load_config()
    token_logger = TokenLogger()

    # 步骤 1：Analyst
    topic = "大语言模型在代码生成中的应用与局限"
    prompt1 = build_analyst_prompt(topic)
    out1 = run_step(0, "Analyst", prompt1, config, token_logger)

    # 步骤 2：Summarizer（以上一步输出为输入）
    prompt2 = build_summarizer_prompt(out1)
    out2 = run_step(1, "Summarizer", prompt2, config, token_logger)

    # 总汇总
    print("\n--- Total ---")
    token_logger.print_total_summary()

    # 生成 token 使用图
    steps = token_logger.get_steps()
    graph_path = GRAPHS_DIR / "token_usage.png"
    plot_token_usage(steps, graph_path)
    print(f"\n图表已保存: {graph_path}")

    # 可选：写入实验 CSV（与 experiments 格式一致，便于统一分析）
    logs_dir = ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "token_experiment.csv"
    with open(csv_path, "a", encoding="utf-8") as f:
        if csv_path.stat().st_size == 0:
            f.write("prompt_length,agent_number,total_tokens\n")
        total = token_logger.total_tokens()
        # 本次运行视为一次“实验”：medium prompt, 2 agents
        f.write(f"medium,2,{total}\n")
    print(f"实验记录已追加: {csv_path}")


if __name__ == "__main__":
    main()
