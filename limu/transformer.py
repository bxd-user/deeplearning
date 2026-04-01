import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/fra-eng/fra.txt"

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
FFN_MULT = 4

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MAX_LEN = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path, max_len):
    src_sents = []
    tgt_sents = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue

            src, tgt = parts
            src_tokens = src.lower().split()
            tgt_tokens = tgt.lower().split()

            if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                src_sents.append(src_tokens)
                tgt_sents.append(tgt_tokens)

    return src_sents, tgt_sents

class Vocab:
    def __init__(self, sentences):
        self.word2idx = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2
        }
        self.idx2word = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

        idx = 3
        for sent in sentences:
            for w in sent:
                if w not in self.word2idx:
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    idx += 1

    def encode(self, sent):
        return [self.word2idx[w] for w in sent]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == 2:
                break
            if i >= 3:
                words.append(self.idx2word[i])
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

class TranslationDataset(Dataset):
    def __init__(self, src, tgt, src_vocab, tgt_vocab):
        self.src = src
        self.tgt = tgt
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = [1] + self.src_vocab.encode(self.src[idx]) + [2]
        tgt = [1] + self.tgt_vocab.encode(self.tgt[idx]) + [2]
        return torch.tensor(src), torch.tensor(tgt)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_len = max(len(s) for s in src_batch)
    tgt_len = max(len(t) for t in tgt_batch)

    src_pad = []
    tgt_pad = []

    for s, t in zip(src_batch, tgt_batch):
        src_pad.append(F.pad(s, (0, src_len - len(s)), value=0))
        tgt_pad.append(F.pad(t, (0, tgt_len - len(t)), value=0))

    return torch.stack(src_pad), torch.stack(tgt_pad)

def make_src_mask(src):
    return (src != 0).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    batch, tgt_len = tgt.size()

    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    subsequent_mask = torch.tril(
        torch.ones((tgt_len, tgt_len), device=tgt.device)
    ).bool()

    return pad_mask & subsequent_mask

def attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_k = D_MODEL // N_HEADS

        self.w_q = nn.Linear(D_MODEL, D_MODEL)
        self.w_k = nn.Linear(D_MODEL, D_MODEL)
        self.w_v = nn.Linear(D_MODEL, D_MODEL)
        self.w_o = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)

        Q = self.w_q(q).view(batch, -1, N_HEADS, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch, -1, N_HEADS, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch, -1, N_HEADS, self.d_k).transpose(1, 2)

        x = attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch, -1, D_MODEL)

        return self.w_o(x)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * FFN_MULT),
            nn.ReLU(),
            nn.Linear(D_MODEL * FFN_MULT, D_MODEL)
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * FFN_MULT),
            nn.ReLU(),
            nn.Linear(D_MODEL * FFN_MULT, D_MODEL)
        )
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, memory, memory, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab, D_MODEL)
        self.tgt_embed = nn.Embedding(tgt_vocab, D_MODEL)

        self.encoder = nn.ModuleList(
            [EncoderLayer() for _ in range(N_LAYERS)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer() for _ in range(N_LAYERS)]
        )

        self.generator = nn.Linear(D_MODEL, tgt_vocab)

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(out)

def train_model(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for src, tgt in loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            src_mask = make_src_mask(src)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}")

def translate(model, sentence, src_vocab, tgt_vocab, max_len=20):
    model.eval()

    tokens = sentence.lower().split()
    src_ids = [1] + src_vocab.encode(tokens) + [2]
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    src_mask = make_src_mask(src)
    memory = model.encode(src, src_mask)

    tgt_ids = [1]

    for _ in range(max_len):
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)
        tgt_mask = make_tgt_mask(tgt)

        out = model.decode(tgt, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])

        next_id = logits.argmax(dim=-1).item()
        tgt_ids.append(next_id)

        if next_id == 2:
            break

    return tgt_vocab.decode(tgt_ids)

def main():
    src_sents, tgt_sents = load_data(DATA_PATH, MAX_LEN)

    src_vocab = Vocab(src_sents)
    tgt_vocab = Vocab(tgt_sents)

    dataset = TranslationDataset(src_sents, tgt_sents, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = Transformer(len(src_vocab), len(tgt_vocab)).to(DEVICE)

    train_model(model, loader)

    while True:
        s = input("English: ")
        print("French:", translate(model, s, src_vocab, tgt_vocab))

if __name__ == "__main__":
    main()