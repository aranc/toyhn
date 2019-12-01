import torch
import random

OVERFIT = True

def gen_data_entry(op):
    if OVERFIT:
        return torch.FloatTensor([1]), torch.FloatTensor([1,1]), torch.FloatTensor([1])
    _x = [random.randint(-100, 100) for _ in range(2)]
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
optimizer = torch.optim.Adam(f.parameters(), lr=1e-4)

def collate(batch):
    ops = torch.stack([_[0] for _ in batch])
    xs = torch.stack([_[1] for _ in batch])
    ys = torch.stack([_[2] for _ in batch])
    return ops, xs, ys

def gen_data_batch_mixed():
    op1 = random.choice((1, 2))
    op2 = random.choice((1, 2))
    return collate([gen_data_entry(op1), gen_data_entry(op2)])

def gen_data_batch_hard():
    return collate([gen_data_entry(1), gen_data_entry(2)])

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

train(gen_data_batch_mixed, gen_weights_in_batch=False)

