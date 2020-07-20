import sys
import random
import numpy as np
import torch
import torch.nn as nn


class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.linear1 = torch.nn.Linear(1, 1)

    def load_weights(self, new_weights):
        start = 0
        for p in self.parameters():
            p.data = new_weights[start:start + p.numel()].view(p.data.shape).contiguous()
            start = start + p.numel()
            p.grad = None
        #self.g.previous_layers_lstm.flatten_parameters()
        assert start == len(new_weights)

    def get_grads(self):
        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_list = torch.cat(grad_list, 0)
        return grad_list

    def num_parameters(self):
        res = 0
        for p in self.parameters():
            if False:
                #watchout, this might be buggy. re-check for new network architectures
                print(p)
                print(p.numel())
            res += p.numel()
        return res

    def forward(self, x):
        return self.linear1(x)

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

batch_size = 5

while True:
    optimizer.zero_grad()

    xs = []
    ops = []
    ground_truths = []
    for i in range(batch_size):
        x = random.random() * 200 - 100
        op = random.randint(0, 1)
        if op == 0:
            ground_truth = x
        elif op == 1:
            ground_truth = x * 7

        op = torch.FloatTensor([op])
        x = torch.FloatTensor([x])
        ground_truth = torch.FloatTensor([ground_truth])
        xs.append(x)
        ops.append(op)
        ground_truths.append(ground_truth)

    xs = torch.stack(xs)
    ops = torch.stack(ops)
    ys = torch.stack(ground_truths)

    all_grads = []
    preds = []
    new_weights = f(ops)
    for i in range(len(ops)):
        g.load_weights(new_weights[i])
        pred = g(xs[i])
        loss = ((pred - ys[i]) ** 2).mean()
        print(loss.item())
        loss.backward()
        grad_list = g.get_grads()
        all_grads.append(grad_list.detach())
    all_grads = torch.stack(all_grads, 0)
    new_weights.backward(all_grads)

    optimizer.step()
