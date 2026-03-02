"""
MiniGPT 最小运行脚本：准备数据 → 训练 → 采样。
优先使用 nanoGPT 的莎士比亚数据；若无则用内置小语料。
"""
import os
import pickle
import numpy as np
import torch
from torch.nn import functional as F

from model import MiniGPT

# -----------------------------------------------------------------------------
# 配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare_char")
BLOCK_SIZE = 128
N_EMBD = 64
N_HEAD = 2
DROPOUT = 0.0
BATCH_SIZE = 32
MAX_ITERS = 500
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------


def load_data():
    """加载数据：若有 train.bin + meta.pkl 则用莎士比亚，否则用内置小语料。"""
    train_bin = os.path.join(DATA_DIR, "train.bin")
    meta_path = os.path.join(DATA_DIR, "meta.pkl")

    if os.path.isfile(train_bin) and os.path.isfile(meta_path):
        train_ids = np.fromfile(train_bin, dtype=np.uint16)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        itos, stoi = meta["itos"], meta["stoi"]
        print(f"[minigpt] 使用莎士比亚数据, vocab_size={vocab_size}, train tokens={len(train_ids)}")
    else:
        text = "hello world " * 500
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        train_ids = np.array([stoi[c] for c in text], dtype=np.uint16)
        print(f"[minigpt] 使用内置小语料, vocab_size={vocab_size}")

    return train_ids, vocab_size, itos, stoi


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def decode(itos, ids):
    return "".join(itos.get(int(i), "?") for i in ids)


def main():
    train_ids, vocab_size, itos, stoi = load_data()
    block_size = min(BLOCK_SIZE, len(train_ids) // 2)
    if block_size < 8:
        block_size = 8

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        dropout=DROPOUT,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"[minigpt] 开始训练 block_size={block_size} max_iters={MAX_ITERS} device={DEVICE}")
    model.train()
    for step in range(MAX_ITERS):
        x, y = get_batch(train_ids, block_size, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1} loss {loss.item():.4f}")

    # 采样
    model.eval()
    context = "hello"
    if not all(c in stoi for c in context):
        context = "".join(itos[i] for i in range(min(vocab_size, block_size)))
    idx = torch.tensor([[stoi[c] for c in context]], device=DEVICE)

    with torch.no_grad():
        for _ in range(80):
            if idx.size(1) > block_size:
                idx = idx[:, -block_size:]
            logits, _ = model(idx)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

    out = decode(itos, idx[0].cpu().tolist())
    print("\n[minigpt] 采样结果:\n", out)


if __name__ == "__main__":
    main()
