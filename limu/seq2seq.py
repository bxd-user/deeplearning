import os
import math
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_fra_eng():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', 'fra-eng', 'fra.txt')

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    pairs = []
    for line in lines:
        parts= line.split('\t')
        pairs.append((parts[0], parts[1]))

    return pairs

pairs = load_fra_eng()

def tokenize(sent):
    return sent.lower().split()

class Vocab:
    def __init__(self, texts, min_freq=2):
        self.freq = {}
        for text in texts:
            for token in tokenize(text):
                self.freq[token] = self.freq.get(token, 0) + 1

        self.idx2token = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.token2idx = {}
        for index, token in enumerate(self.idx2token):
            self.token2idx[token] = index


        for token in self.freq:
            cnt = self.freq[token]
            if cnt >= min_freq:
                new_index = len(self.idx2token)
                # 词 → 编号
                self.token2idx[token] = new_index
                # 编号 → 词
                self.idx2token.append(token)


    def __len__(self):
        return len(self.idx2token)

    def encode(self, text):
        tokens = tokenize(text)
        ids=[]
        for token in tokens:
            if token in self.token2idx:
                ids.append(self.token2idx[token])
            else:
                ids.append(self.token2idx['<unk>'])
        return ids

    def decode(self, ids):
        return ' '.join(self.idx2token[i] for i in ids)

src_texts = [p[0] for p in pairs]
tgt_texts = [p[1] for p in pairs]

src_vocab = Vocab(src_texts)
tgt_vocab = Vocab(tgt_texts)

def pad(seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=10):
        self.data = []
        for src, tgt in pairs:
            src_ids = src_vocab.encode(src)[:max_len]
            tgt_ids = tgt_vocab.encode(tgt)[:max_len]

            src_ids = [src_vocab.token2idx['<bos>']] + src_ids + [src_vocab.token2idx['<eos>']]
            tgt_ids = [tgt_vocab.token2idx['<bos>']] + tgt_ids + [tgt_vocab.token2idx['<eos>']]

            src_ids = pad(src_ids, max_len + 2, src_vocab.token2idx['<pad>'])
            tgt_ids = pad(tgt_ids, max_len + 2, tgt_vocab.token2idx['<pad>'])

            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])

dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)      # (B, T, E)
        x = x.permute(1, 0, 2)     # (T, B, E)
        _, h = self.rnn(x)
        return h

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        h = self.encoder(src)
        out, _ = self.decoder(tgt[:, :-1], h)
        return out

embed_size = 256
hidden_size = 256
lr = 0.005
num_epochs = 200

encoder = Encoder(len(src_vocab), embed_size, hidden_size).to(device)
decoder = Decoder(len(tgt_vocab), embed_size, hidden_size).to(device)
model = Seq2Seq(encoder, decoder).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.token2idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"epoch {epoch+1}, loss {total_loss/len(loader):.4f}")

def translate(sentence, max_len=10):
    model.eval()
    src = [src_vocab.token2idx['<bos>']] + src_vocab.encode(sentence) + [src_vocab.token2idx['<eos>']]
    src = torch.tensor(pad(src, max_len+2, src_vocab.token2idx['<pad>'])).unsqueeze(0).to(device)

    h = encoder(src)

    tgt = torch.tensor([[tgt_vocab.token2idx['<bos>']]]).to(device)
    res = []

    for _ in range(max_len):
        out, h = decoder(tgt, h)
        pred = out.argmax(-1)[-1, 0].item()
        if pred == tgt_vocab.token2idx['<eos>']:
            break
        res.append(pred)
        tgt = torch.tensor([[pred]]).to(device)

    return tgt_vocab.decode(res)

print(translate("go ."))
print(translate("i love you ."))
