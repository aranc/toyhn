import torch
import random

import sys

try:
    __IPYTHON__
    OVERFIT = False
except:
    assert sys.argv[1] in ('True', 'False')
    if sys.argv[1] == 'True':
        OVERFIT = True
    elif sys.argv[1] == 'False':
        OVERFIT = False
    else:
        assert False

def gen_data_entry(op):
    if OVERFIT:
        return torch.FloatTensor([1]), torch.FloatTensor([1,1]), torch.FloatTensor([1])
    #_x = [random.randint(-100, 100) for _ in range(2)]
    _x = [random.random()*2 - 1 for _ in range(2)]
    x = torch.FloatTensor(_x)
    if op == 1:
        y = torch.FloatTensor([_x[0]])
    if op == 2:
        y = torch.FloatTensor([_x[1]])

    return torch.FloatTensor([op]), x, y


class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.gen_weight = torch.nn.Linear(1, 2)
        self.gen_bias = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.gen_weight(x), self.gen_bias(x)

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = torch.nn.Linear(2, 1)

    def load_weights(self, new_weights):
        weight, bias = new_weights
        weight = weight.unsqueeze(0)
        d = {'weight':weight, 'bias':bias}
        self.net.load_state_dict(d)

    def forward(self, x):
        return self.net(x)

f = F()
g = G()
optimizer = torch.optim.Adam(f.parameters(), lr=1e-6)

def collate(batch):
    ops = torch.stack([_[0] for _ in batch])
    xs = torch.stack([_[1] for _ in batch])
    ys = torch.stack([_[2] for _ in batch])
    return ops, xs, ys

def gen_data_batch_single():
    op1 = random.choice((1, 2))
    return collate([gen_data_entry(op1)])

def gen_data_batch_mixed():
    op1 = random.choice((1, 2))
    op2 = random.choice((1, 2))
    return collate([gen_data_entry(op1), gen_data_entry(op2)])

def gen_data_batch_hard():
    return collate([gen_data_entry(1), gen_data_entry(2)])

def gen_data_batch_validation():
    return collate([gen_data_entry(random.choice((1, 2))) for _ in range(100)])

def train(data_generator, gen_weights_in_batch):
    epoch = 1
    while True:
        batch = data_generator()
        ops, xs, ys = batch
        preds = []
        if gen_weights_in_batch:
            new_weights = f(ops)
        for i in range(len(ops)):
            if not gen_weights_in_batch:
                _new_weights = f(ops[i])
            else:
                _new_weights = new_weights[0][i], new_weights[1][i]
            g.load_weights(_new_weights)
            pred = g(xs[i])
            preds.append(pred)
        preds = torch.cat(preds)
        loss = ((preds - ys.squeeze(1)) ** 2).mean()
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch, "Loss:", loss.item())
        epoch += 1

best_f = None
def train2(data_generator):
    global best_f

    val_set = gen_data_batch_validation()
    val_ops, val_xs, val_ys = val_set

    best = 1e10
    save_me = False
    epoch = 1
    while True:
        batch = data_generator()
        ops, xs, ys = batch

        preds = []
        new_weights = f(ops)
        for i in range(len(ops)):
            _new_weights = new_weights[0][i], new_weights[1][i]
            g.load_weights(_new_weights)
            pred = g(xs[i])
            preds.append(pred)
        preds = torch.cat(preds)
        loss = ((preds - ys.squeeze(1)) ** 2).mean()
        loss.backward()

        grad_list = []
        for p in filter(lambda p: p.requires_grad, g.parameters()):
            grad_list.append(p.grad.view(-1))
            break
        grad_list = torch.cat(grad_list, 0)
        all_grads = []
        all_grads.append(grad_list.detach())
        all_grads = torch.stack(all_grads, 0)
        new_weights[0].backward(all_grads)
        new_weights[1].backward(all_grads)

        optimizer.step()

        with torch.no_grad():
            preds = []
            new_weights = f(val_ops)
            for i in range(len(val_ops)):
                _new_weights = new_weights[0][i], new_weights[1][i]
                g.load_weights(_new_weights)
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

train2(gen_data_batch_single)

