# Reflection-as-Bias: 实验框架

验证"将 Reflection 显式转化为 Bias 并重塑候选方向选择分布"能否让 auto research agent 更高效探索的模拟实验框架。

## 项目结构

```
experiment/
├── main.py                  # 入口：运行实验并打印结果表格
├── src/ra_bias/
│   ├── models.py            # 数据模型（Candidate、Outcome、RoundLog 等）
│   ├── environment.py       # 模拟研究环境 + 3 个任务（reasoning / NLP / data analysis）
│   ├── bias_engine.py       # Bias 引擎族（No / Weak / Tag / LLM / HardPruning / Textual）
│   ├── reflection.py        # Reflector：从 outcome 生成结构化 ReflectionSignal
│   ├── selector.py          # Selector：将 bias score 叠加到 base score 并选择候选
│   ├── agent_loop.py        # 单 episode 主循环
│   ├── runner.py            # 多方法 / 消融实验组织器
│   ├── evaluation.py        # 指标计算（成功率 / 轮数 / 无效探索 / 成本 / 轮次质量趋势）
│   └── logger.py            # JSONL 日志 + JSON 摘要写入
├── outputs/                 # 自动生成的日志目录
├── design.md                # 实验设计文档
├── explaination.md          # 方法说明文档
└── statics.md               # 所需数据与指标说明
```

## 环境要求

- Python 3.10+
- 无第三方依赖，仅使用标准库

## 快速开始

```bash
# 克隆 / 进入目录后直接运行（默认 20 episode，运行主实验 + 全部消融）
python main.py
```

输出示例：

```
== Main Results (statics.md §1) ==
Method                 Success Rate  Avg Steps  Wasted Ratio  Cost/Success
--------------------------------------------------------------------------
single_path                   28.3%       2.76         0.922          1563
branching_only                33.3%       2.55         0.905          1416
textual_reflection            76.7%       2.37         0.721          1327
reflection_as_bias            76.7%       2.13         0.701          1193

== Per-Round Avg. Selected Candidate Quality (statics.md §2.3) ==
 Round  single_path  branching_only  textual_reflection  reflection_as_bias
...

== Ablation 1: Hard Pruning vs Soft Bias ==
== Ablation 2: Tag-based vs LLM-based Bias ==
== Ablation 3: No Bias / Weak Bias / Full Bias ==
```

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--episodes N` | `20` | 每种方法的 episode 数量，越大结果越稳定 |
| `--output DIR` | `outputs` | 日志根目录 |
| `--no-ablation` | 关闭 | 只跑主实验（4 个主要方法），跳过消融 |

```bash
# 只跑主实验，快速验证
python main.py --no-ablation --episodes 10

# 增大 episode 数以获得更稳定的统计结果
python main.py --episodes 50

# 指定日志输出目录
python main.py --output results/run_01
```

## 四种对比方法

| 方法名 | 说明 |
|---|---|
| `single_path` | 每轮只生成 1 个候选，无 reflection，baseline |
| `branching_only` | 4 个候选，无 reflection，排除"多候选"带来的提升 |
| `textual_reflection` | 4 个候选 + reflection，但以隐式弱信号形式影响选择（模拟 LLM 读到历史文本）|
| `reflection_as_bias` | 4 个候选 + reflection 显式转为 bias score，参与候选重排 |

## 消融实验

| 消融组 | 对比内容 | 对应文档 |
|---|---|---|
| `hard_vs_soft` | Hard Pruning vs Soft Bias | design.md §7.2 |
| `bias_type` | Tag-based Bias vs LLM-based Bias | design.md §7.3 |
| `bias_strength` | No Bias / Weak Bias / Full Bias | design.md §7.4 |

## 日志格式

每个 episode 会在 `outputs/` 下生成一个 `.jsonl` 文件，每行是一轮的记录，格式与 `statics.md §5` 一致：

```json
{
  "task_id": "task_reasoning_ep1",
  "round": 1,
  "candidate_set": [
    {
      "id": "single_pass_r0",
      "description": "single-pass prompting",
      "tags": ["single_pass", "prompt_only"],
      "base_score": 0.743,
      "bias_score": 0.0,
      "final_score": 0.743,
      "metadata": {"hidden_quality": 0.24, "failure_modes": ["hallucination"]}
    }
  ],
  "selected_id": "single_pass_r0",
  "selected_description": "single-pass prompting",
  "outcome": {
    "success": false,
    "score": 0.21,
    "failure_modes": ["hallucination"],
    "cost_tokens": 512
  },
  "reflection_signal": {
    "prefer_tags": [],
    "avoid_tags": ["single_pass", "prompt_only"],
    "confidence": 0.87
  }
}
```

每个方法还会生成一个 `summary.json`，包含所有 episode 的聚合记录。
