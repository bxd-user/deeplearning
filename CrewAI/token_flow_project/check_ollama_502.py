"""
502 诊断脚本：逐步检查 Ollama 连接与请求，定位哪一步返回 502。
在 token_flow_project 目录下运行: python check_ollama_502.py
"""
import json
import sys

import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen2:7b"
TIMEOUT = 60.0


def step(name: str, ok: bool, detail: str = ""):
    status = "通过" if ok else "失败"
    symbol = "[OK]" if ok else "[!!]"
    print(f"\n{symbol} 步骤: {name} -> {status}")
    if detail:
        print(detail)


def run():
    print("=" * 60)
    print("Ollama 502 诊断（逐步检查）")
    print("=" * 60)
    print(f"  Base URL: {OLLAMA_BASE_URL}")
    print(f"  Model:    {MODEL}")
    print(f"  Timeout:  {TIMEOUT}s")

    with httpx.Client(timeout=TIMEOUT) as client:
        # --- 步骤 1: 能否连上 Ollama ---
        try:
            r = client.get(f"{OLLAMA_BASE_URL}/")
            ok = r.status_code == 200
            step("1. 连接 Ollama (GET /)", ok, f"状态码: {r.status_code}")
            if not ok:
                print(f"    响应体: {r.text[:500]}")
                if r.status_code == 502:
                    print("\n    >>> 502 在第一步：请求未正确到达 Ollama，或 Ollama 处于异常状态。")
                    print("    建议：1) 在浏览器打开 http://localhost:11434 看是否正常")
                    print("          2) 命令行执行 ollama list 看 CLI 能否连通")
                    print("          3) 完全退出并重新启动 Ollama")
                    print("          4) 若使用代理/VPN，尝试关闭或把 localhost 加入直连")
                return
        except Exception as e:
            step("1. 连接 Ollama (GET /)", False, str(e))
            print("    建议：确认 Ollama 已安装并启动，且端口 11434 未被占用。")
            return

        # --- 步骤 2: 模型是否存在 ---
        try:
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ok = r.status_code == 200
            step("2. 获取模型列表 (GET /api/tags)", ok, f"状态码: {r.status_code}")
            if not ok:
                print(f"    响应体: {r.text[:500]}")
                return
            data = r.json()
            models = [m.get("name") for m in data.get("models", [])]
            has_model = any(MODEL in name or name == MODEL for name in models)
            step("2b. 模型中包含 qwen2:7b", has_model, f"已有模型: {models}")
            if not has_model:
                print("    建议执行: ollama pull qwen2:7b")
                return
        except Exception as e:
            step("2. 获取模型列表", False, str(e))
            return

        # --- 步骤 3: /api/generate（简单 prompt）---
        try:
            r = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": MODEL, "prompt": "hi", "stream": False},
            )
            ok = r.status_code == 200
            step("3. 生成请求 (POST /api/generate, prompt='hi')", ok, f"状态码: {r.status_code}")
            if not ok:
                print(f"    响应头: {dict(r.headers)}")
                print(f"    响应体: {r.text[:800]}")
                if r.status_code == 502:
                    print("    --> 502 出现在本步骤：Ollama 在处理 /api/generate 时出错（常见：模型加载中/崩溃/资源不足）")
                return
            body = r.json()
            got = (body.get("response") or "").strip()
            print(f"    返回内容预览: {repr(got[:200])}")
        except Exception as e:
            step("3. 生成请求 (POST /api/generate)", False, str(e))
            return

        # --- 步骤 4: /api/chat（LiteLLM 可能用 chat 接口）---
        try:
            r = client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Say one word."}],
                    "stream": False,
                },
            )
            ok = r.status_code == 200
            step("4. 对话请求 (POST /api/chat)", ok, f"状态码: {r.status_code}")
            if not ok:
                print(f"    响应头: {dict(r.headers)}")
                print(f"    响应体: {r.text[:800]}")
                if r.status_code == 502:
                    print("    --> 502 出现在本步骤：Ollama 在处理 /api/chat 时出错")
                return
            body = r.json()
            msg = body.get("message") or {}
            got = (msg.get("content") or "").strip()
            print(f"    返回内容预览: {repr(got[:200])}")
        except Exception as e:
            step("4. 对话请求 (POST /api/chat)", False, str(e))
            return

    print("\n" + "=" * 60)
    print("所有步骤通过，Ollama 与当前模型可用。若 main.py 仍 502，多为 LiteLLM 请求格式或并发问题。")
    print("=" * 60)


if __name__ == "__main__":
    run()
    sys.exit(0)
