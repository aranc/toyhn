import torch
import random

def gen_data_entry(op):
    x = [random.randint(-100, 100) for _ in range(2)]
    x = torch.FloatTensor(x)
    if op == 1:
        y = torch.FloatTensor(x[0])
    if op == 2:
        y = torch.FloatTensor(x[1])

    return torch.FloatTensor(op), x, y


class F(torch.nn.Module):
    def __init__(self, args, num_actions):
        super(F, self).__init__()
        self.get_weight = torch.nn.Linear(1, 2)
        self.get_bias = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.weight(x), self.bias(x)

class G(torch.nn.Module):
    def __init__(self, args, num_actions):
        super(G, self).__init__()
        self.net = torch.nn.Linear(2, 1)

    def load_weights(self, new_weights):
        weight, bias = new_weights
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

def train(data_generator):
    epoch = 1
    while True:
        batch = data_generator()
        ops, xs, ys = batch
        preds = []
        for i in range(len(ops)):
            new_weights = f(ops[i])
            g.load_weights(new_weights)
            pred = g(x)
            preds.append(pred)
        preds = torch.cat(preds)
        loss = ((preds - ys) ** 2).mean()
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch, "Loss:", loss.item())
        epoch += 1

train(gen_data_batch_mixed)

