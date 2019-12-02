import torch
import random


def gen_data_entry():
    x1 = random.choice((0, 1))
    x2 = random.choice((0, 1))
    y = x1 ^ x2
    return torch.FloatTensor([x1]), torch.FloatTensor([x2]), torch.FloatTensor([y])

class H(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.L1 = torch.nn.Linear(2, 100)
        self.L2 = torch.nn.Linear(100, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        x = self.L1(x)
        x = torch.nn.functional.relu(x)
        x = self.L2(x)
        return x

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = torch.nn.Linear(1, 1, bias=False)

    def load_weights(self, new_weights):
        start = 0
        for p in self.parameters()
            p.data = new_weights[start:start + p.numel()].view(p.data.shape).contiguous()
            start = start + p.numel()
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

best_f = None
def train(data_generator):
    global best_f

    best = 1e10
    save_me = False
    epoch = 1
    while True:
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

train()

