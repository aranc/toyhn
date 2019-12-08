import torch
import random

import sys

OVERFIT = False

def gen_data_entry(op):
    if OVERFIT:
        return torch.FloatTensor([1]), torch.FloatTensor([1]), torch.FloatTensor([2])
    #_x = [random.randin(t(-100, 100) for _ in range(2)]
    _x = [random.random()*2 - 1 for _ in range(1)]
    #_x = [random.choice((0, 1)) for _ in range(1)]
    x = torch.FloatTensor(_x)
    _x2 = [_ * 2 for _ in _x]
    if op == 1:
        y = torch.FloatTensor(_x2)
    if op == 0:
        y = torch.FloatTensor(_x)

    return torch.FloatTensor([op]), x, y

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = torch.nn.Linear(1, 1, bias=False)

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
        return self.net(x)

class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()

        Ne = G().num_parameters()
        N = 20

        self.L1 = torch.nn.Linear(1, N)
        self.L2 = torch.nn.Linear(N, N)
        self.gen_weights = torch.nn.Linear(N, Ne)

    def forward(self, x):
        x = self.L1(x)
        x = torch.nn.functional.relu(x)
        x = self.L2(x)
        x = torch.nn.functional.relu(x)
        x = self.gen_weights(x)
        return x


f = F()
g = G()
optimizer = torch.optim.Adam(f.parameters(), lr=1e-4)

def collate(batch):
    ops = torch.stack([_[0] for _ in batch])
    xs = torch.stack([_[1] for _ in batch])
    ys = torch.stack([_[2] for _ in batch])
    return ops, xs, ys

def gen_data_batch_single():
    op1 = random.choice((1, 0))
    return collate([gen_data_entry(op1)])

def gen_data_batch_mixed():
    op1 = random.choice((1, 0))
    op2 = random.choice((1, 0))
    return collate([gen_data_entry(op1), gen_data_entry(op2)])

def gen_data_batch_hard():
    return collate([gen_data_entry(1), gen_data_entry(0)])

def gen_data_batch_validation():
    return collate([gen_data_entry(random.choice((1, 0))) for _ in range(100)])


best_f = None
def train(data_generator):
    global best_f

    val_set = gen_data_batch_validation()
    val_ops, val_xs, val_ys = val_set

    best = 1e10
    save_me = False
    epoch = 1
    while True:
        optimizer.zero_grad()

        batch = data_generator()
        ops, xs, ys = batch

        preds = []
        new_weights = f(ops)
        for i in range(len(ops)):
            g.load_weights(new_weights[i])
            pred = g(xs[i])
            preds.append(pred)
        preds = torch.cat(preds)
        loss = ((preds - ys.squeeze(1)) ** 2).mean()
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
            preds = []
            new_weights = f(val_ops)
            for i in range(len(val_ops)):
                g.load_weights(new_weights[i])
                pred = g(val_xs[i])
                preds.append(pred)
            preds = torch.cat(preds)
            loss = ((preds - val_ys.squeeze(1)) ** 2).mean()

        print("Epoch:", epoch, "Best:", best, "Loss:", loss.item())
        epoch += 1

        if loss.item() < best:
            best = loss.item()
            best_f = f.state_dict()
            save_me = True
        if save_me:
            if epoch % 1000 == 0:
                torch.save(best_f, "best_f.pkl")
                save_me = False

train(gen_data_batch_single)

