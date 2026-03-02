import torch
from torch import nn
from torch.nn import functional as F
import os

batch_size, num_steps = 64, 35
num_hiddens = 256
lr, num_epochs = 0.001, 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_data_from_pt():
    corpus = torch.load("data/corpus.pt")
    vocab_data = torch.load("data/vocab.pt", weights_only=False)
    vocab = Vocab(vocab_data["idx_to_token"],
                  vocab_data["token_to_idx"])
    return corpus, vocab

corpus, vocab = load_data_from_pt()

def data_iter(corpus, batch_size, num_steps):
    corpus = torch.tensor(corpus, dtype=torch.long)
    data_len = corpus.numel()
    num_batches = (data_len - 1) // (batch_size * num_steps)

    corpus = corpus[:num_batches * batch_size * num_steps]
    corpus = corpus.reshape(batch_size, -1)

    for i in range(0, corpus.shape[1] - num_steps, num_steps):
        X = corpus[:, i:i+num_steps]
        Y = corpus[:, i+1:i+num_steps+1]
        yield X, Y

def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xi, W_hi, b_i,
              W_xf, W_hf, b_f,
              W_xo, W_ho, b_o,
              W_xc, W_hc, b_c,
              W_hq, b_q]

    for p in params:
        p.requires_grad_(True)

    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    (W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hq, b_q) = params

    H, C = state
    outputs = []

    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)

        C = F * C + I * C_tilda
        H = O * torch.tanh(C)

        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).float()
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def train_lstm(net, corpus, vocab, lr, num_epochs, device, print_every=10):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.params, lr)

    for epoch in range(num_epochs):
        state = net.begin_state(batch_size, device)
        total_loss, n = 0.0, 0

        for X, Y in data_iter(corpus, batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)

            state = tuple(s.detach() for s in state)

            optimizer.zero_grad()
            y_hat, state = net(X, state)

            l = loss(y_hat, Y.T.reshape(-1))
            l.backward()

            torch.nn.utils.clip_grad_norm_(net.params, 1.0)
            optimizer.step()

            total_loss += l.item()
            n += 1

        if (epoch + 1) % print_every == 0:
            print(f"epoch {epoch+1}, loss {total_loss/n:.3f}")

def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(1, device)
    outputs = [vocab[prefix[0]]]

    for ch in prefix[1:]:
        X = torch.tensor([[outputs[-1]]], device=device)
        _, state = net(X, state)
        outputs.append(vocab[ch])

    for _ in range(num_preds):
        X = torch.tensor([[outputs[-1]]], device=device)
        y, state = net(X, state)
        outputs.append(int(y.argmax(dim=1)))

    return vocab.to_tokens(outputs)

net = RNNModelScratch(
    len(vocab),
    num_hiddens,
    device,
    get_lstm_params,
    init_lstm_state,
    lstm
)

train_lstm(net, corpus, vocab, lr, num_epochs, device)

print(predict("time traveller ", 50, net, vocab, device))
