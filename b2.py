import torch
import random

import sys

def gen_data_entry(op):
    _x = [random.randint(-100, 100) for _ in range(2)]
    #_x = [random.random()*2 - 1 for _ in range(1)]
    #_x = [random.choice((0, 1)) for _ in range(1)]
    x = torch.FloatTensor(_x)
    if op == 1:
        y = torch.FloatTensor([_x[0]])
    if op == 0:
        y = torch.FloatTensor([_x[1]])

    return torch.FloatTensor([op]), x, y

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = torch.nn.Linear(2, 1, bias=False)

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

    def get_grads(self):
        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_list = torch.cat(grad_list, 0)
        return grad_list

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

        preds = net(ops, xs)
        loss = ((preds - ys) ** 2).mean()
        loss.backward()
        net.backward_hack()
        optimizer.step()

        with torch.no_grad():
            preds = net(val_ops, val_xs)
            loss = ((preds - val_ys) ** 2).mean()

        print("Epoch:", epoch, "Best:", best, "Loss:", loss.item())
        epoch += 1

        if loss.item() < best:
            best = loss.item()
            best_f = net.f.state_dict()
            save_me = True
        if False and save_me:
            if epoch % 1000 == 0:
                torch.save(best_f, "best_f.pkl")
                save_me = False

def ctrl_c_wrap(func):
    try:
        func()
    except KeyboardInterrupt:
        with torch.no_grad():
            print(net.f(torch.Tensor([0])))
            print(net.f(torch.Tensor([1])))

train(gen_data_batch_hard)
#ctrl_c_wrap(lambda : train(gen_data_batch_hard)
