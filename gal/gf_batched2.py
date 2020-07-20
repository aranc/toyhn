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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f = F()
        self.gs = None
        self.new_weights = None

    def get_new_g_intance(self):
        #maybe use a pool in the future
        g = G()
        if next(self.parameters()).is_cuda:
            g = g.cuda()
        return g

    def forward(self, inputs_for_f, inputs_for_g):
        N = len(inputs_for_f)
        assert N == len(inputs_for_g)

        if torch.is_grad_enabled():
            self.gs = []
        else:
            self.gs = None
            self.new_weights = None

        new_weights = self.f(inputs_for_f)

        results = []
        for i in range(N):
            g = self.get_new_g_intance()
            g.load_weights(new_weights[i])
            result = g(inputs_for_g[i])
            results.append(result)
            if torch.is_grad_enabled():
                self.gs.append(g)

        if torch.is_grad_enabled():
            self.new_weights = new_weights

        results = torch.stack(results)
        return results

    def backward_hack(self):
        #in the future, this can be replaced with proper use of backward hooks

        if not torch.is_grad_enabled():
            print("warning: calling backward_hack when torch.is_grad_enabled() == False")
            return

        all_grads = []
        for g in self.gs:
            grad_list = g.get_grads()
            all_grads.append(grad_list.detach())
        all_grads = torch.stack(all_grads, 0)
        self.new_weights.backward(all_grads)
        self.gs = None
        self.new_weights = None

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)


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

    preds = net(ops, xs)
    loss = ((preds - ys) ** 2).mean()
    print(loss.item())
    loss.backward()
    net.backward_hack()
    optimizer.step()

    optimizer.step()
