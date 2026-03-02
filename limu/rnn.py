import torch
from torch import nn
from torch.nn import functional as F
import os

batch_size = 32
num_steps = 35
num_hiddens = 256
lr = 1e-2
num_epochs = 200
theta = 1.0  # 梯度裁剪阈值
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vocab:
    def __init__(self, idx_to_token, token_to_idx):
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)

def load_data_from_pt(corpus_path="data/corpus.pt",
                      vocab_path="data/vocab.pt"):
    if not (os.path.exists(corpus_path) and os.path.exists(vocab_path)):
        raise FileNotFoundError("请先生成 corpus.pt 和 vocab.pt")

    corpus = torch.load(corpus_path)
    vocab_data = torch.load(vocab_path)

    vocab = Vocab(
        vocab_data["idx_to_token"],
        vocab_data["token_to_idx"]
    )

    print("✔ 已加载 corpus 和 vocab")
    return corpus, vocab


corpus, vocab = load_data_from_pt()

def data_iter(corpus, batch_size, num_steps):
    corpus = torch.tensor(corpus, dtype=torch.long)

    num_batches = (len(corpus) - 1) // (batch_size * num_steps)
    corpus = corpus[:num_batches * batch_size * num_steps + 1]

    Xs = corpus[:-1]
    Ys = corpus[1:]

    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    for i in range(0, Xs.shape[1], num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X.to(device), Y.to(device)

def get_params(vocab_size, num_hiddens, device):
    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    W_xh = normal((vocab_size, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = normal((num_hiddens, vocab_size))
    b_q = torch.zeros(vocab_size, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for p in params:
        p.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)

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

def grad_clipping(params, theta):
    norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None)
    )
    if norm > theta:
        for p in params:
            if p.grad is not None:
                p.grad[:] *= theta / norm

def train_rnn(net, corpus, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.params, lr)

    for epoch in range(num_epochs):
        total_loss, n = 0.0, 0
        state = net.begin_state(batch_size, device)

        train_iter = data_iter(corpus, batch_size, num_steps)

        for X, Y in train_iter:
            optimizer.zero_grad()

            output, state = net(X, state)
            state = tuple(s.detach() for s in state)

            l = loss(output, Y.T.reshape(-1))
            l.backward()

            grad_clipping(net.params, theta)

            optimizer.step()

            total_loss += l.item()
            n += 1

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}, loss {total_loss / n:.4f}")

3
def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
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

net = RNNModelScratch(
    len(vocab),
    num_hiddens,
    device,
    get_params,
    init_rnn_state,
    rnn
)

train_rnn(net, corpus, vocab, lr, num_epochs, device)

print(predict("time traveller", 50, net, vocab, device))
