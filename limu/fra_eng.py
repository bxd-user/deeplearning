import os
import zipfile
import requests
import collections
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_URL = "https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip"
DATA_SHA1 = "94646ad1522d915e7b0f9296181140edcf86a4f5"
DATA_DIR = "./data"


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "fra-eng.zip")

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        r = requests.get(DATA_URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    extract_dir = os.path.join(DATA_DIR, "fra-eng")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

    return extract_dir

def read_data_nmt():
    data_dir = download_and_extract()
    with open(os.path.join(data_dir, "fra.txt"), encoding="utf-8") as f:
        return f.read()

def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(".,!?") and prev_char != " "

    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)

def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        if num_examples and i >= num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target

class Vocab:
    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []

        counter = collections.Counter(token for line in tokens for token in line)
        self.token_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, list):
            return [self[token] for token in tokens]
        return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])

def truncate_pad(line, num_steps, pad_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [pad_token] * (num_steps - len(line))

def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] + [vocab["<eos>"]] for l in lines]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines]
    )
    valid_len = (array != vocab["<pad>"]).sum(dim=1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)

    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=["<pad>", "<bos>", "<eos>"])

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)

    dataset = TensorDataset(
        src_array, src_valid_len, tgt_array, tgt_valid_len
    )
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_iter, src_vocab, tgt_vocab

if __name__ == "__main__":
    train_iter, src_vocab, tgt_vocab = load_data_nmt(
        batch_size=2, num_steps=8
    )

    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print("X:", X)
        print("X_valid_len:", X_valid_len)
        print("Y:", Y)
        print("Y_valid_len:", Y_valid_len)
        break

