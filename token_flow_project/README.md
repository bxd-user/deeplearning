# Token Flow — 多 Agent 协作与 Token 研究工具

基于本地 Ollama 的多 Agent 流水线实验项目，支持 **多任务、每任务多轮对话**，并做完整统计记录。用于研究：**多 Agent 协作**、**Token 消耗**、**Prompt 结构**、**Agent 通信**。

---

## 使用方法

### 环境准备

1. 安装依赖：
  ```bash
   pip install -r requirements.txt
  ```
2. 本地安装并启动 [Ollama](https://ollama.ai)，并拉取所用模型（如 `qwen2:7b`）：
  ```bash
   ollama pull qwen2:7b
   ollama serve   # 若未自动启动
  ```

### 运行主流程

在项目根目录执行：

```bash
python main.py
```

- **有 `tasks` 时**：多任务、每任务多轮。对每个任务执行 `rounds_per_task` 轮流水线，每轮输入为上一轮最后一 Agent 的输出（首轮为任务输入）。全部统计按任务、轮次记录并导出。
- **无 `tasks` 时**：单任务单轮，与原先一致。按 `pipeline` 顺序调用各 Agent（默认：Analyst → Summarizer → Reviewer），输出每步 Token、Prompt 结构、Agent 通信，并生成报表与图表。

### 配置说明

编辑 `**config.yaml`** 可调整：


| 配置项                          | 说明                                                   |
| ---------------------------- | ---------------------------------------------------- |
| `model`                      | Ollama 模型名，如 `qwen2:7b`                              |
| `temperature` / `max_tokens` | 生成参数                                                 |
| `pipeline`                   | Agent 执行顺序，如 `[analyst, summarizer, reviewer]`       |
| `agents`                     | 各 Agent 的 `name`、`role`、`goal`、`backstory`（可完全用配置驱动） |
| `tasks`                      | 多任务列表。每项为字符串或 `{ name, input }`；有此项时启用多任务多轮          |
| `rounds_per_task`            | 每个任务执行的轮数（多轮对话，每轮一整条流水线）                             |
| `initial_input`              | 单任务时第一个 Agent 的输入；无 `tasks` 时使用                      |


### 可选：批量实验

运行 `experiments/run_experiment.py` 可做多组实验（不同 prompt 长度、Agent 数量），结果追加到 `outputs/logs/token_experiment.csv`，便于后续分析。

---

## 项目结构（各文件/目录作用）

```
token_flow_project/
├── main.py                 # 主入口：多任务多轮或单任务单轮、打印报表、导出 JSON/CSV、画图
├── config.yaml             # 流水线、agents、tasks、rounds_per_task、initial_input
├── requirements.txt       # Python 依赖
├── README.md               # 本说明
│
├── agents/                 # Agent 定义（可被 config 覆盖）
│   ├── __init__.py
│   ├── analyst.py          # Analyst：分析主题/数据，提供 build 与 prompt 分块接口
│   ├── summarizer.py       # Summarizer：对分析结果做总结，同上
│   └── reviewer.py         # Reviewer：对总结进行审阅与润色，同上
│
├── core/                   # 多 Agent 编排与配置解析
│   ├── __init__.py
│   ├── orchestrator.py     # 编排器：run_pipeline 单轮；run_multi_task_pipeline 多任务多轮
│   ├── registry.py         # Agent 注册表：config 优先，否则用代码中的 analyst/summarizer/reviewer
│   └── prompt_builder.py   # 从 config 生成完整 prompt 与 (role/context/task) 分块
│
├── research/               # 研究用数据模型与导出
│   ├── __init__.py
│   ├── models.py           # RunReport、TaskReport、MultiTaskReport；记录含 task_index/round_index
│   └── export_report.py    # RunReport / MultiTaskReport 导出 JSON 与 CSV（含 summary.csv）
│
├── utils/                  # 通用工具
│   ├── __init__.py
│   ├── token_counter.py    # 近似 Token 计数（中英混合）
│   ├── token_logger.py     # 按步记录 prompt/completion/total（旧接口，仍可用）
│   ├── token_plot.py       # 绘制「Agent 步骤 vs Token 使用」柱状图
│   └── prompt_trace.py     # 每步 prompt 保存到 outputs/prompts/ 或 task_N/round_M/step_*_prompt.txt
│
├── experiments/            # 批量实验脚本
│   ├── __init__.py
│   └── run_experiment.py   # 多组实验，结果写入 outputs/logs/token_experiment.csv
│
└── outputs/                # 运行后生成的结果目录（可被 git 忽略）
    ├── prompts/            # 每步发送给 LLM 的完整 prompt
    │   ├── step_N_prompt.txt              # 单任务时
    │   └── task_N/round_M/step_*_prompt.txt  # 多任务多轮时
    ├── graphs/
    │   ├── token_usage.png           # 单任务 Token 图
    │   └── token_usage_multi_task.png  # 多任务多轮展平后的 Token 图
    ├── research/
    │   ├── report.json / multi_task_report.json  # 单次或多任务完整报表
    │   ├── token_steps.csv / prompt_structures.csv / communications.csv  # 带 task_index、round_index
    │   └── summary.csv               # 多任务时：每任务每轮 total_tokens 汇总
    └── logs/
        └── token_experiment.csv      # 实验记录（含 tasks、rounds_per_task、total_tokens）
```

---

## 项目结果（运行后生成的文件）


| 路径                                          | 说明                                                      |
| ------------------------------------------- | ------------------------------------------------------- |
| `outputs/prompts/`                          | 每步完整 prompt；多任务多轮时为 `task_N/round_M/step_*_prompt.txt`  |
| `outputs/graphs/token_usage.png`            | 单任务时各步 Token 柱状图                                        |
| `outputs/graphs/token_usage_multi_task.png` | 多任务多轮时展平后的各步 Token 图（含 T/R 标签）                          |
| `outputs/research/report.json`              | 单次运行报表（token_steps、prompt_structures、communications）    |
| `outputs/research/multi_task_report.json`   | 多任务多轮完整报表（tasks → rounds → steps）                       |
| `outputs/research/token_steps.csv`          | 每步 token，多任务时含 task_index、round_index                   |
| `outputs/research/prompt_structures.csv`    | 每步 role/context/task 分块 token 数                         |
| `outputs/research/communications.csv`       | Agent 间通信记录                                             |
| `outputs/research/summary.csv`              | 多任务时：每任务每轮 total_tokens、每任务合计                           |
| `outputs/logs/token_experiment.csv`         | 实验记录（含 agent_number、tasks、rounds_per_task、total_tokens） |


扩展新 Agent 时，在 `config.yaml` 的 `pipeline` 与 `agents` 中增加条目即可；若需自定义构建逻辑，可在 `core/registry.py` 的 `_CODE_AGENTS` 中注册对应的 `build_prompt` 与 `get_prompt_blocks` 函数。