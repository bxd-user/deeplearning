import collections
import re
import os
import urllib.request
import torch

def download_time_machine(save_dir="data"):
    """
    下载 Time Machine 数据集（如果本地不存在）
    """
    url = "https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "timemachine.txt")

    if not os.path.exists(file_path):
        print("Downloading Time Machine dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download finished.")
    else:
        print("Time Machine dataset already exists.")

    return file_path

def read_time_machine(path):
    """
    读取 Time Machine 文本并清洗：
    - 只保留字母
    - 转小写
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower()
            for line in lines]

def tokenize(lines, token_type='char'):
    """
    将文本行拆分为词元
    """
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError("token_type must be 'word' or 'char'")

def count_corpus(tokens):
    """
    统计词元频率
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)

        # <unk> 固定索引 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self[token] for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def load_corpus_time_machine(max_tokens=-1):
    corpus_path = "data/corpus.pt"
    vocab_path = "data/vocab.pt"

    if os.path.exists(corpus_path) and os.path.exists(vocab_path):
        print("Loading processed corpus from disk...")
        corpus = torch.load(corpus_path)
        vocab = torch.load(vocab_path)
        return corpus, vocab

    print("Processing raw text...")
    path = download_time_machine()
    lines = read_time_machine(path)
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    os.makedirs("data", exist_ok=True)
    torch.save(corpus, corpus_path)
    torch.save(vocab, vocab_path)

    print("Corpus saved to disk.")
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()

print("corpus 长度:", len(corpus))
print("vocab 大小:", len(vocab))
print("前 20 个索引:", corpus[:20])
print("对应字符:", vocab.to_tokens(corpus[:20]))
