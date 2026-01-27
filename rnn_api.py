import torch
from torch import nn
from torch.nn import functional as F
import os
import math

batch_size = 32
num_steps = 35
num_hiddens = 256
num_epochs = 500
lr = 0.001

class Vocab:
    def __init__(self, idx_to_token, token_to_idx):
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

def load_data_from_pt(
    corpus_path="data/corpus.pt",
    vocab_path="data/vocab.pt"
):
    corpus = torch.load(corpus_path)
    vocab_data = torch.load(vocab_path)

    vocab = Vocab(
        vocab_data["idx_to_token"],
        vocab_data["token_to_idx"]
    )
    return corpus, vocab

corpus, vocab = load_data_from_pt()

def data_iter(corpus, batch_size, num_steps):
    corpus = torch.tensor(corpus, dtype=torch.long)
    num_tokens = (len(corpus) - 1) // batch_size * batch_size
    corpus = corpus[:num_tokens]

    corpus = corpus.reshape(batch_size, -1)

    for i in range(0, corpus.shape[1] - num_steps, num_steps):
        X = corpus[:, i:i+num_steps]
        Y = corpus[:, i+1:i+num_steps+1]
        yield X, Y

class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.rnn = nn.RNN(vocab_size, num_hiddens)
        self.linear = nn.Linear(num_hiddens, vocab_size)
        self.num_hiddens = num_hiddens

    def forward(self, X, state):
        # X: (batch, num_steps)
        X = F.one_hot(X.T.long(), self.linear.out_features)
        X = X.float()  # (num_steps, batch, vocab_size)

        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size, device):
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)

def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm

def train_rnn(net, corpus, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        state = None
        total_loss, n = 0.0, 0

        for X, Y in data_iter(corpus, batch_size, num_steps):
            X, Y = X.to(device), Y.to(device)

            if state is None:
                state = net.begin_state(X.shape[0], device)
            else:
                state = state.detach()

            optimizer.zero_grad()
            y_hat, state = net(X, state)
            l = loss(y_hat, Y.reshape(-1))
            l.backward()

            grad_clipping(net, 1.0)
            optimizer.step()

            total_loss += l.item()
            n += 1

        if epoch % 10 == 0:
            ppl = math.exp(total_loss / n)
            print(f"epoch {epoch+1}, perplexity {ppl:.2f}")

def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(1, device)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor([[outputs[-1]]], device=device)

    for ch in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[ch])

    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1)))

    return ''.join(vocab.idx_to_token[i] for i in outputs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = RNNModel(len(vocab), num_hiddens).to(device)

print(predict("time traveller", 10, net, vocab, device))

train_rnn(net, corpus, vocab, lr, num_epochs, device)

print(predict("time traveller", 50, net, vocab, device))
