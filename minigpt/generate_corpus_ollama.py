"""
使用本地 Ollama 部署的 Qwen2 生成训练语料，保存为 run.py 可加载的 txt。
运行前请确保 Ollama 已启动且已拉取 qwen2:7b：ollama run qwen2:7b
"""
import json
import os
import sys
import urllib.request

# 默认输出路径：minigpt/data/ollama_corpus.txt
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ollama_corpus.txt")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2:7b"
NUM_PREDICT = 4000  # 每次生成约 4000 token
ROUNDS = 150  # 用多少轮不同 prompt 生成（总字符数约 ROUNDS * NUM_PREDICT * 2~4）


# 多样化的提示，用于生成不同风格的文本
PROMPTS = [
    "请写一段关于春天和自然的散文，200字左右。",
    "用简单的中文写一段对话：两个人讨论周末去哪里玩。",
    "写一首短诗，主题是夜晚和星星。",
    "用口语化的中文写一段日记，记录今天发生的小事。",
    "写一段说明文：如何泡一杯好茶，步骤清晰。",
    "写一段故事开头：一个年轻人在火车站遇到陌生人。",
    "用中文写几句人生感悟，风格像格言。",
    "写一段描写下雨天的场景，包含环境和心情。",
    "写一段两人之间的简短书信，语气友好。",
    "用简单句子介绍你最喜欢的一本书或一部电影。",
    "写一段回忆童年的文字，100字左右。",
    "写几句关于学习和成长的建议，面向学生。",
]


def generate(prompt: str, num_predict: int = NUM_PREDICT) -> str:
    """调用 Ollama API 生成一段文本（非流式）。"""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise SystemExit(f"无法连接 Ollama（请确认已启动 Ollama 并拉取 qwen2:7b）: {e}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Ollama 返回非 JSON: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = []
    n = ROUNDS  # 轮数可大于 PROMPTS 数量，会循环使用不同 prompt
    print(f"[ollama] 使用模型 {MODEL}，共 {n} 轮生成，输出: {OUTPUT_FILE}")
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"  [{i+1}/{n}] 生成中...")
        text = generate(prompt)
        if text:
            total.append(text)
            total.append("\n\n")  # 段落间隔
    corpus = "".join(total).strip()
    if not corpus:
        print("[ollama] 未生成任何内容，请检查模型与网络。")
        sys.exit(1)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"[ollama] 已写入 {len(corpus)} 字符 -> {OUTPUT_FILE}")
    print("接下来可运行: python run.py")


if __name__ == "__main__":
    main()
