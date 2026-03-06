# Token Flow Project

基于 CrewAI + Ollama 的最小多 Agent 示例，用于观察 LLM 调用过程中每一步的 **Token 消耗**，并将结果保存到本地。

## 技术栈

- **CrewAI**：多 Agent 编排（研究员 + 评审员）
- **Ollama**：本地模型 `qwen2:7b`（Qwen2）
- **Token 统计**：优先使用 Qwen2 官方 tokenizer（精确），回退为 tiktoken（近似）

## 项目结构

```
token_flow_project/
├── main.py              # 入口：运行 Crew，打印并保存每步 token
├── agents.py            # 多 Agent 与 Crew 定义（Ollama qwen2:7b）
├── token_counter.py     # Token 统计（Qwen2 / tiktoken）
├── prompt.txt           # 默认研究主题（可修改）
├── requirements.txt     # 依赖
├── setup_precise_tokens.py   # 预先下载 Qwen2 tokenizer，启用精确统计
├── check_ollama_502.py # Ollama 502 诊断脚本
├── .gitignore
└── README.md
```

运行后会在当前目录生成 `token_usage_YYYYMMDD_HHMMSS.json`，记录每步的输入/输出 token 及汇总。

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.com) 已安装并拉取模型：`ollama pull qwen2:7b`

## 快速开始

```bash
# 1. 进入项目目录
cd token_flow_project

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 3.（可选）预先装好精确 token 统计
python setup_precise_tokens.py

# 4. 确保 Ollama 已启动，运行
python main.py
```

## 输出说明

- 终端：每一步会打印 `[Token] 输入: xxx tokens (Qwen2)` 与 `[Token] 输出: xxx tokens (Qwen2)`。
- 本地文件：运行结束后生成 `token_usage_*.json`，包含 `steps`、`total_input_tokens`、`total_output_tokens` 等字段。

## 常见问题

- **502 Bad Gateway**：先运行 `python check_ollama_502.py` 定位问题；确保 Ollama 已启动且 `ollama list` 中有 `qwen2:7b`。
- **精确统计未生效**：执行 `python setup_precise_tokens.py` 预下载 Qwen2 tokenizer；若未安装 `transformers` 会自动回退为 tiktoken。
