import sys
import random
import numpy as np
import torch
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self, args=None):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(1, 1, batch_first=True)

    def forward(self, x):
        return self.lstm(x)[1][0]

net = Network()
learn_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

seq_len = 3

while True:
    optimizer.zero_grad()

    x = [random.random() * 100 for _ in range(seq_len)]

    if sys.argv[1] == "last":
        y = x[-1]
    elif sys.argv[1] == "first":
        y = x[0]
    elif sys.argv[1] == "sum":
        y = sum(x)
    else:
        assert False

    x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(-1)
    y = torch.FloatTensor([y]).unsqueeze(0)
    

    prediction = net(x)
    loss = (y - prediction) ** 2
    print(loss.item())

    loss.backward()
    optimizer.step()

