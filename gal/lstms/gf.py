import sys
import random
import numpy as np
import torch
import torch.nn as nn

num_chars = 26
seq_len = 3


class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()

        self.embedding = nn.Embedding(num_chars, num_chars)
        self.lstm = nn.LSTM(num_chars, num_chars, batch_first=True)
        self.linear = nn.Linear(num_chars, num_chars)

    def load_weights(self, new_weights):
        start = 0
        for p in self.parameters():
            p.data = new_weights[start:start + p.numel()].view(p.data.shape).contiguous()
            start = start + p.numel()
            p.grad = None
        #self.g.previous_layers_lstm.flatten_parameters()
        assert start == len(new_weights)

    def num_parameters(self):
        res = 0
        for p in self.parameters():
            res += p.numel()
        return res

    def forward(self, x):
        #print("a", x.shape)
        x = self.embedding(x)
        #print("b", x.shape)
        x = self.lstm(x)[1][0].squeeze(1)
        #print("c", x.shape)
        x = self.linear(x)
        #print("d", x.shape)
        return x

class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()

        Ne = G().num_parameters()

        self.gen_weights = torch.nn.Linear(1, Ne)

    def forward(self, x):
        x = self.gen_weights(x)
        return x

f = F()
g = G()
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

while True:
    optimizer.zero_grad()

    x = [random.choice(list(range(num_chars))) for _ in range(seq_len)]
    op = random.choice((0, 1))

    if  op == 1:
        y = x[-1]
    else:
        y = x[0]

    op = torch.FloatTensor(op).unsqueeze(0)
    x = torch.LongTensor(x).unsqueeze(0)
    y = torch.LongTensor([y]).unsqueeze(0)

    new_weights = f(op)
    g.load_weights(new_weights[0])
    pred = g(x)
    loss = criterion(pred, y.squeeze(1))
    print(loss.item())

    loss.backward()

    if True:
        grad_list = []
        #or p in filter(lambda p: p.requires_grad, g.parameters()):
        for p in g.parameters():
            grad_list.append(p.grad.view(-1))
        grad_list = torch.cat(grad_list, 0)
        all_grads = []
        all_grads.append(grad_list.detach())
        all_grads = torch.stack(all_grads, 0)
        new_weights.backward(all_grads)

    optimizer.step()

    with torch.no_grad():
        for op in (0, 1):
            x = [random.choice(list(range(num_chars))) for _ in range(seq_len)]
            pretty_x = [chr(ord('a') + _) for _ in x]
            x = torch.LongTensor(x).unsqueeze(0)
            pred = net(x)[0].argmax()
            pretty_pred = chr(ord('a') + pred.item())
            print("[take first]" if op == 0 else "[take last]", pretty_x, "->",  pretty_pred)

