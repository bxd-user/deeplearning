# 你需要跑出来什么数据

如果你的目标是把这个 idea 做成一篇完整论文，那么你最终需要的不是“随便跑几个结果”，而是一组能够**支撑你的核心论点**的数据。

你的核心论点是：

> **把 reflection 显式变成 bias，并用它去重塑候选研究方向的选择分布，能够让 auto research agent 更高效地探索。**

所以你需要的数据，必须能回答下面三件事：

1. **效果有没有变好**
2. **探索有没有变快、变省**
3. **提升是不是来自 bias，而不是别的因素**

---

## 1. 你最少需要跑出的主结果数据

这是论文主表必须有的。

### 1.1 Success Rate（成功率）

你需要统计每种方法在所有任务上的最终成功率。

例如：

| Method             | Success Rate |
| ------------------ | -----------: |
| Single-path        |        42.5% |
| Branching-only     |        53.0% |
| Textual Reflection |        56.5% |
| Reflection-as-Bias |        68.0% |

这个表是为了证明：

> 你的方法最终确实更容易完成任务。

---

### 1.2 Average Steps to Success（成功所需平均轮数）

你需要统计**成功样本**平均用了多少轮才成功。

例如：

| Method             | Avg. Steps to Success |
| ------------------ | --------------------: |
| Single-path        |                   4.8 |
| Branching-only     |                   4.1 |
| Textual Reflection |                   3.9 |
| Reflection-as-Bias |                   3.1 |

这个指标非常重要，因为你的 idea 不只是“成功率高一点”，而是：

> **更快找到正确方向。**

---

### 1.3 Wasted Exploration Ratio（无效探索比例）

你需要统计失败尝试占总尝试的比例。

定义可以写成：

\[
\text{Wasted Exploration Ratio} =
\frac{\text{Failed Trials}}{\text{Total Trials}}
\]

例如：

| Method             | Wasted Exploration Ratio |
| ------------------ | -----------------------: |
| Single-path        |                     0.71 |
| Branching-only     |                     0.60 |
| Textual Reflection |                     0.57 |
| Reflection-as-Bias |                     0.43 |

这个指标直接体现你的方法有没有减少“走弯路”。

---

### 1.4 Cost per Success（单位成功成本）

你需要统计每成功一个任务平均消耗多少资源，例如：

- token 数
- 调用次数
- 平均轮数
- 时间成本

例如：

| Method             | Avg. Tokens per Success |
| ------------------ | ----------------------: |
| Single-path        |                   18500 |
| Branching-only     |                   22400 |
| Textual Reflection |                   24600 |
| Reflection-as-Bias |                   20900 |

这个指标是为了防止别人质疑：

> 你是不是只是更贵，所以更强。

如果你的方法成功率更高，而且成本没有明显爆炸，甚至更低，那会很有说服力。

---

## 2. 你需要跑出的“机制验证数据”

主结果只能说明“有效”，但还不能说明“为什么有效”。

所以你还需要一组数据，证明：

> **真正起作用的是 bias 机制本身。**

---

### 2.1 Textual Reflection vs Reflection-as-Bias

这是你最关键的一组对比。

你需要把“普通文本反思”与“显式 bias”分开跑。

例如：

| Method             | Success Rate | Avg. Steps | Wasted Ratio |
| ------------------ | -----------: | ---------: | -----------: |
| Textual Reflection |        56.5% |        3.9 |         0.57 |
| Reflection-as-Bias |        68.0% |        3.1 |         0.43 |

如果这组差异明显，就能支撑：

> 不是 reflection 文本本身有多神，而是“把 reflection 显式转成 bias 并参与决策”这一步有效。

---

### 2.2 候选排序变化数据

你需要证明 bias 真的改变了候选方向的排序，而不是挂名存在。

所以每一轮至少要记录：

- 候选集合
- base score
- bias score
- final score
- 最终被选中的候选

例如：

| Candidate        | Base Score | Bias Score | Final Score |
| ---------------- | ---------: | ---------: | ----------: |
| A (single-pass)  |       0.72 |      -0.55 |        0.17 |
| B (verification) |       0.61 |      +0.48 |        1.09 |
| C (retrieval)    |       0.58 |      +0.12 |        0.70 |

这种数据很重要，因为它能说明：

> bias 不是装饰，而是真的在重塑选择分布。

---

### 2.3 候选质量随轮次变化

你需要统计一个趋势：

> 在加入 bias 后，后续轮次选到的候选是不是越来越好。

例如可以记录：

- 每一轮 top-1 候选的真实 outcome score
- 每一轮被选中候选的平均成功概率
- 每一轮候选集合中“优质方向”的占比

例如：

| Round | Branching-only Avg. Candidate Quality | Reflection-as-Bias Avg. Candidate Quality |
| ----- | ------------------------------------: | ----------------------------------------: |
| 1     |                                  0.44 |                                      0.45 |
| 2     |                                  0.47 |                                      0.58 |
| 3     |                                  0.49 |                                      0.66 |
| 4     |                                  0.51 |                                      0.71 |

这个趋势能说明你的方法不是“一次运气好”，而是在持续引导方向。

---

## 3. 你需要跑出的消融数据

如果要投稿，消融实验基本是必须的。

---

### 3.1 Hard Pruning vs Soft Bias

你要验证“柔性引导”是不是比“直接砍掉”更好。

例如：

| Method       | Success Rate | Avg. Steps | Wasted Ratio |
| ------------ | -----------: | ---------: | -----------: |
| Hard Pruning |        62.0% |        3.4 |         0.48 |
| Soft Bias    |        68.0% |        3.1 |         0.43 |

如果 soft bias 更稳，就能支持你的理论叙事。

---

### 3.2 Tag-based Bias vs LLM-based Bias

这组数据用来说明你的方法不是依赖某一个实现细节。

例如：

| Bias Type      | Success Rate | Avg. Steps |
| -------------- | -----------: | ---------: |
| Tag-based Bias |        63.5% |        3.4 |
| LLM-based Bias |        68.0% |        3.1 |

这样你就可以说：

- 简单版本已经有效
- 更强版本效果更好

---

### 3.3 No Bias / Weak Bias / Full Bias

你可以进一步验证 bias 强度是否重要。

例如：

| Variant   | Success Rate | Avg. Steps |
| --------- | -----------: | ---------: |
| No Bias   |        53.0% |        4.1 |
| Weak Bias |        60.0% |        3.6 |
| Full Bias |        68.0% |        3.1 |

这能说明：

> 引导强度确实会影响探索效率。

---

## 4. 你需要跑出的轨迹级数据

你的方法本质上是“改探索轨迹”，所以你不能只有表格，还要有轨迹数据。

你至少要能拿出几个完整任务的逐轮记录，例如：

| Round | Selected Direction           | Outcome         | Reflection Signal | Next-round Effect           |
| ----- | ---------------------------- | --------------- | ----------------- | --------------------------- |
| 1     | single-pass prompting        | fail            | avoid single-pass | verification 被上调         |
| 2     | decomposition + verification | partial success | prefer multi-step | search-based 方案进入 top-1 |
| 3     | tree search + verification   | success         | -                 | stop                        |

这种轨迹级数据有两个作用：

1. 写 case study
2. 证明你的方法真的在“引导方向”，而不是只在最终结果上偶然更好

---

## 5. 你需要保存的原始日志字段

为了后面能算出上面所有数据，你每一轮至少要保存这些字段：

```json
{
  "task_id": "task_12",
  "round": 2,
  "candidate_set": [
    {
      "id": "idea_a",
      "description": "single-pass prompting",
      "tags": ["single_pass", "prompt_only"],
      "base_score": 0.72,
      "bias_score": -0.55,
      "final_score": 0.17
    },
    {
      "id": "idea_b",
      "description": "decomposition + verification",
      "tags": ["multi_step", "verification"],
      "base_score": 0.61,
      "bias_score": 0.48,
      "final_score": 1.09
    }
  ],
  "selected_id": "idea_b",
  "outcome": {
    "success": false,
    "score": 0.58,
    "failure_modes": ["partial verification failure"]
  },
  "reflection_signal": {
    "prefer_tags": ["verification", "multi_step"],
    "avoid_tags": ["single_pass"],
    "confidence": 0.81
  }
}