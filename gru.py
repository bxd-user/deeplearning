import math
import torch
from torch import nn
from torch.nn import functional as F
import os

class Vocab:
    def __init__(self, idx_to_token, token_to_idx):
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

    def to_tokens(self, indices):
        return ''.join(self.idx_to_token[i] for i in indices)

def load_data_from_pt(
    corpus_path="data/corpus.pt",
    vocab_path="data/vocab.pt"
):
    corpus = torch.load(corpus_path, weights_only=True)

    vocab_data = torch.load(vocab_path, weights_only=True)
    vocab = Vocab(
        vocab_data["idx_to_token"],
        vocab_data["token_to_idx"]
    )

    print("Loaded corpus and vocab from .pt files.")
    return corpus, vocab

def data_iter(corpus, batch_size, num_steps):
    corpus = torch.tensor(corpus, dtype=torch.long)
    corpus = corpus[:(len(corpus) // batch_size) * batch_size]
    corpus = corpus.reshape(batch_size, -1)

    num_batches = (corpus.shape[1] - 1) // num_steps
    for i in range(num_batches):
        X = corpus[:, i*num_steps:(i+1)*num_steps]
        Y = corpus[:, i*num_steps+1:(i+1)*num_steps+1]
        yield X, Y

def get_params(vocab_size, num_hiddens, device):
    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return (normal((vocab_size, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()

    W_hq = normal((num_hiddens, vocab_size))
    b_q = torch.zeros(vocab_size, device=device)

    params = [
        W_xz, W_hz, b_z,
        W_xr, W_hr, b_r,
        W_xh, W_hh, b_h,
        W_hq, b_q
    ]
    for p in params:
        p.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def gru(inputs, state, params):
    W_xz, W_hz, b_z, \
    W_xr, W_hr, b_r, \
    W_xh, W_hh, b_h, \
    W_hq, b_q = params

    H, = state
    outputs = []

    for X in inputs:
        Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)
        R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)
        H_tilde = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)
        H = Z * H + (1 - Z) * H_tilde
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).float()
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens, self.device)

def grad_clipping(params, theta):
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm

def train_rnn(model, corpus, vocab, lr, num_epochs,
              batch_size, num_steps, device, print_every=10):

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params, lr)

    for epoch in range(num_epochs):
        state = model.begin_state(batch_size)
        total_loss, n = 0.0, 0

        for X, Y in data_iter(corpus, batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)
            state = (state[0].detach(),)

            optimizer.zero_grad()
            y_hat, state = model(X, state)
            l = loss(y_hat, Y.T.reshape(-1))
            l.backward()
            grad_clipping(model.params, 1)
            optimizer.step()

            total_loss += l.item()
            n += 1

        if (epoch + 1) % print_every == 0:
            print(f"epoch {epoch+1}, loss {total_loss / n:.3f}")

def predict(prefix, num_preds, model, vocab, device):
    state = model.begin_state(1)
    outputs = [vocab[prefix[0]]]

    for ch in prefix[1:]:
        X = torch.tensor([[outputs[-1]]], device=device)
        _, state = model(X, state)
        outputs.append(vocab[ch])

    for _ in range(num_preds):
        X = torch.tensor([[outputs[-1]]], device=device)
        y, state = model(X, state)
        outputs.append(int(y.argmax(dim=1)))

    return vocab.to_tokens(outputs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_steps = 35
num_hiddens = 256
num_epochs = 500
lr = 0.001

corpus, vocab = load_data_from_pt()

model = RNNModelScratch(
    len(vocab), num_hiddens, device,
    get_params, init_gru_state, gru
)

train_rnn(
    model, corpus, vocab,
    lr, num_epochs,
    batch_size, num_steps,
    device, print_every=10
)

print(predict("time traveller ", 50, model, vocab, device))
