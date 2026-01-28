import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (B, h, L, d_k)
    mask: 可选, (B, 1, L, L)
    返回: (B, h, L, d_k)
    """
    d_k = Q.size(-1)
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    # softmax
    attn = F.softmax(scores, dim=-1)
    # 输出
    output = torch.matmul(attn, V)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        # Q, K, V 线性映射
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # 输出线性映射
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, X):
        B, L, _ = X.shape
        # 线性映射
        Q = self.W_q(X)  # (B, L, d_model)
        K = self.W_k(X)
        V = self.W_v(X)
        # 拆成多头
        Q = Q.view(B, L, self.h, self.d_k).transpose(1,2)  # (B, h, L, d_k)
        K = K.view(B, L, self.h, self.d_k).transpose(1,2)
        V = V.view(B, L, self.h, self.d_k).transpose(1,2)
        # 注意力计算
        out = scaled_dot_product_attention(Q, K, V)  # (B, h, L, d_k)
        # 拼接回 d_model
        out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        # 输出线性映射
        out = self.W_o(out)
        return out

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
    
    def forward(self, X):
        return self.W2(F.relu(self.W1(X)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, X):
        # Multi-Head Attention + Add & Norm
        attn_out = self.mha(X)
        X = self.norm1(X + attn_out)
        # FFN + Add & Norm
        ffn_out = self.ffn(X)
        X = self.norm2(X + ffn_out)
        return X

class Encoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, vocab_size, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))  # 可学习位置编码
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff) for _ in range(N)])
    
    def forward(self, x):
        """
        x: (B, L) 输入 token ID
        """
        X = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            X = layer(X)
        return X  # 输出 (B, L, d_model)

B, L = 2, 10
vocab_size = 1000
d_model = 64
d_ff = 256
h = 8
N = 2

x = torch.randint(0, vocab_size, (B, L))
encoder = Encoder(N, d_model, h, d_ff, vocab_size)
out = encoder(x)
print(out.shape)  # torch.Size([2, 10, 64])
