# MiniGPT

最简单的 GPT：单 Block（因果注意力 + MLP），用于学习/调试。

## 结构

- `model.py`：`MiniGPT`，仅 1 个 Transformer Block（Pre-LN + 因果自注意力 + MLP），weight tying。
- `run.py`：加载数据 → 训练 → 采样。

## 运行

在 **nanoGPT 项目根目录** 先准备莎士比亚数据（可选）：

```bash
python data/shakespeare_char/prepare.py
```

再运行 MiniGPT（会优先用莎士比亚数据，若无则用内置小语料）：

```bash
python minigpt/run.py
```

或在 `minigpt` 目录下：

```bash
cd minigpt
python run.py
```

可在 `run.py` 顶部修改 `BLOCK_SIZE`、`N_EMBD`、`MAX_ITERS`、`BATCH_SIZE` 等。
