"""
自动实验：测试不同 prompt 长度与不同 agent 数量下的 token 使用。
输出: outputs/logs/token_experiment.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from utils import TokenLogger, count_tokens_approx

OLLAMA_URL = "http://localhost:11434/api/generate"


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        return {"model": "qwen2:7b", "temperature": 0.7, "max_tokens": 512}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_ollama(prompt: str, config: dict) -> tuple[str, int, int]:
    import requests
    payload = {
        "model": config.get("model", "qwen2:7b"),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.get("temperature", 0.7), "num_predict": config.get("max_tokens", 512)},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response", "")
    p_tok = count_tokens_approx(prompt)
    c_tok = data.get("eval_count") or count_tokens_approx(resp)
    return resp, p_tok, int(c_tok)


# 不同 prompt 长度
SHORT_PROMPT = "总结：AI 的优缺点。"
MEDIUM_PROMPT = "请从技术、伦理、商业三个维度分析当前大语言模型的发展现状，并给出 3 条结论。"
LONG_PROMPT = (
    "你是一位资深技术顾问。请针对「大语言模型在企业中的应用」这一主题，"
    "从以下方面展开分析：1) 典型应用场景与案例；2) 部署与成本考量；3) 安全与合规；"
    "4) 与现有系统的集成方式；5) 未来 2–3 年的趋势。每个方面请给出 2–3 条要点，并最后给出总体建议。"
)

PROMPTS_BY_LENGTH = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long": LONG_PROMPT,
}


def run_single_agent(prompt: str, config: dict) -> int:
    """1 个 agent：一次调用。"""
    logger = TokenLogger()
    _, p, c = call_ollama(prompt, config)
    logger.log_step(0, "Agent1", p, c)
    return logger.total_tokens()


def run_two_agents(prompt: str, config: dict) -> int:
    """2 个 agent：先分析再总结。"""
    logger = TokenLogger()
    p1 = f"你扮演分析师。请对以下内容做结构化分析。\n\n{prompt}"
    out1, p, c = call_ollama(p1, config)
    logger.log_step(0, "Analyst", p, c)
    p2 = f"你扮演总结专家。请总结以下内容。\n\n{out1}"
    out2, p, c = call_ollama(p2, config)
    logger.log_step(1, "Summarizer", p, c)
    return logger.total_tokens()


def run_three_agents(prompt: str, config: dict) -> int:
    """3 个 agent：分析 -> 提炼 -> 总结。"""
    logger = TokenLogger()
    p1 = f"你扮演分析师。请分析并列出要点。\n\n{prompt}"
    out1, p, c = call_ollama(p1, config)
    logger.log_step(0, "Analyst", p, c)
    p2 = f"你扮演提炼员。请从以下内容中提炼 3 条核心结论。\n\n{out1}"
    out2, p, c = call_ollama(p2, config)
    logger.log_step(1, "Refiner", p, c)
    p3 = f"你扮演总结专家。请用一段话总结。\n\n{out2}"
    out3, p, c = call_ollama(p3, config)
    logger.log_step(2, "Summarizer", p, c)
    return logger.total_tokens()


AGENT_RUNNERS = {
    1: run_single_agent,
    2: run_two_agents,
    3: run_three_agents,
}


def main():
    config = load_config()
    logs_dir = ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "token_experiment.csv"

    rows = [["prompt_length", "agent_number", "total_tokens"]]

    for plen_name, prompt in PROMPTS_BY_LENGTH.items():
        for n_agents in [1, 2, 3]:
            runner = AGENT_RUNNERS[n_agents]
            try:
                total = runner(prompt, config)
                rows.append([plen_name, str(n_agents), str(total)])
                print(f"  {plen_name}, {n_agents} agent(s) -> {total} tokens")
            except Exception as e:
                print(f"  {plen_name}, {n_agents} agent(s) -> error: {e}")
                rows.append([plen_name, str(n_agents), "error"])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(",".join(r) for r in rows))
    print(f"\n实验记录已写入: {csv_path}")


if __name__ == "__main__":
    main()
