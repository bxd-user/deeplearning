"""
运行最小多 Agent 系统，并在每一步 LLM 调用时打印 token 输入/输出数量。
使用 CrewAI 的 LLM Call Hooks + tiktoken 统计。结果会保存到本地 JSON 文件。
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from crewai.hooks import after_llm_call, before_llm_call
from token_counter import count_tokens, is_using_qwen2

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2:7b"
TOKEN_LOG_DIR = Path(__file__).parent  # 与 main.py 同目录


def _warmup_ollama():
    """在跑 Crew 前先请求一次 Ollama，确保模型已加载，减少 502。"""
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "hi", "stream": False},
            )
            if r.status_code == 200:
                print("Ollama 模型已就绪。")
                return
            print(f"Ollama 预热返回 {r.status_code}，继续尝试运行…")
    except Exception as e:
        print(f"Ollama 预热请求失败: {e}，请确认 Ollama 已启动且模型 {OLLAMA_MODEL} 已拉取。")
        print("继续尝试运行…")

# 注册全局 LLM 钩子：每次 LLM 调用前后统计并打印 token，并记录到列表供保存
_step = [0]
_token_records = []  # 每步: {"step": int, "agent": str, "input_tokens": int, "output_tokens": int}
_last_step_data = {}  # 当前步的临时数据，供 after 钩子补全 output_tokens


def _input_tokens_hook(context):
    """调用前：统计并打印输入 token（来自 context.messages），并记录。"""
    _step[0] += 1
    step_id = _step[0]
    total = 0
    for msg in context.messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += count_tokens(part["text"])
    method = "Qwen2" if is_using_qwen2() else "tiktoken"
    print(f"\n--- Step {step_id} | Agent: {context.agent.role} ---")
    print(f"  [Token] 输入: {total} tokens ({method})")
    _last_step_data["step"] = step_id
    _last_step_data["agent"] = context.agent.role
    _last_step_data["input_tokens"] = total
    return None


def _output_tokens_hook(context):
    """调用后：统计并打印输出 token，并写入本步记录。"""
    output_n = 0
    if context.response:
        output_n = count_tokens(context.response)
        method = "Qwen2" if is_using_qwen2() else "tiktoken"
        print(f"  [Token] 输出: {output_n} tokens ({method})")
    _token_records.append({
        "step": _last_step_data.get("step", 0),
        "agent": _last_step_data.get("agent", ""),
        "input_tokens": _last_step_data.get("input_tokens", 0),
        "output_tokens": output_n,
    })
    return None


# 注册钩子（使用装饰器方式）
before_llm_call(_input_tokens_hook)
after_llm_call(_output_tokens_hook)


def main():
    prompt_path = Path(__file__).parent / "prompt.txt"
    if not prompt_path.exists():
        print("未找到 prompt.txt")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    from agents import create_crew

    print("正在预热 Ollama 模型…")
    _warmup_ollama()
    time.sleep(1)

    crew = create_crew(prompt)
    print("=" * 60)
    print("开始执行 Crew，以下为每一步的 Token 统计")
    print("=" * 60)

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Crew 执行完成")
    print("=" * 60)
    if result and hasattr(result, "raw") and result.raw:
        print("\n最终输出（raw）:\n", result.raw[:2000])
    elif result:
        print("\n最终输出:\n", str(result)[:2000])

    # 将 token 消耗结果保存到本地
    if _token_records:
        total_in = sum(r["input_tokens"] for r in _token_records)
        total_out = sum(r["output_tokens"] for r in _token_records)
        payload = {
            "run_at": datetime.now().isoformat(),
            "token_method": "Qwen2" if is_using_qwen2() else "tiktoken",
            "steps": _token_records,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = TOKEN_LOG_DIR / f"token_usage_{ts}.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nToken 消耗已保存: {log_path}")


if __name__ == "__main__":
    main()
