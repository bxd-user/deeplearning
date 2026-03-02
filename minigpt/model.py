"""
The simplest MiniGPT: only 1 block with causal attention + MLP, for learning/debugging.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,      # size of the vocabulary
        block_size: int,      # maximum sequence length
        n_embd: int = 64,     # embedding dimension
        n_head: int = 2,      # number of attention heads
        dropout: float = 0.0, # dropout rate
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        assert n_embd % n_head == 0

        self.wte = nn.Embedding(vocab_size, n_embd)  # Token embedding table
        self.wpe = nn.Embedding(block_size, n_embd)  # Positional embedding table
        self.drop = nn.Dropout(dropout)

        # Linear projection to compute query, key, and value in one go (split into 3 on last dimension)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Linear projection for attention output
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.n_head = n_head
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.mlp_fc = nn.Linear(n_embd, 4 * n_embd)
        self.mlp_proj = nn.Linear(4 * n_embd, n_embd)
        self.mlp_drop = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_attn(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

    def _mlp(self, x):
        x = F.gelu(self.mlp_fc(x))
        return self.mlp_drop(self.mlp_proj(x))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        x = x + self._causal_attn(self.ln1(x))
        x = x + self._mlp(self.ln2(x))
        x = self.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss
