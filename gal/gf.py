import sys
import random
import numpy as np
import torch
import torch.nn as nn

class HyperNetwork3(torch.nn.Module):
    def __init__(self, args=None):
        super(HyperNetwork3, self).__init__()
        self.get_w1 = nn.Linear(1, 3)
        self.get_b1 = nn.Linear(1, 3)

        self.get_w2 = nn.Linear(3, 1)
        self.get_b2 = nn.Linear(3, 1)

    def forward(self, op, x):
        ######### f ##########
        weight1 = self.get_w1(op)
        bias1 = self.get_b1(op)
        
        weight2 = self.get_w2(op)
        bias2 = self.get_b2(op)

        ######### g ##########
        x = torch.nn.functional.linear(x, weight1, bias1)
        x = nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight2, bias2)
        return x


learn_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

while True:
    optimizer.zero_grad()

    x = random.random() * 200 - 100
    op = random.randint(0, 1)
    if op == 0:
        ground_truth = x
    elif op == 1:
        ground_truth = x * 7

    op = torch.FloatTensor([op]).unsqueeze(0)
    x = torch.FloatTensor([x]).unsqueeze(0)
    
    prediction = net(op, x)
    loss = (ground_truth - prediction) ** 2
    print(loss.item())

    loss.backward()
    optimizer.step()

