import sys
import random
import numpy as np
import torch
import torch.nn as nn

num_chars = 26

class Network(torch.nn.Module):
    def __init__(self, args=None):
        super(Network, self).__init__()
        self.embedding = nn.Embedding(num_chars, num_chars)
        self.lstm = nn.LSTM(num_chars, num_chars, batch_first=True)
        self.linear = nn.Linear(num_chars, num_chars)

    def forward(self, x):
        #print("a", x.shape)
        x = self.embedding(x)
        #print("b", x.shape)
        x = self.lstm(x)[1][0].squeeze(1)
        #print("c", x.shape)
        x = self.linear(x)
        #print("d", x.shape)
        return x

net = Network()
learn_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)

seq_len = 3

criterion = nn.CrossEntropyLoss()

while True:
    optimizer.zero_grad()

    x = [random.choice(list(range(num_chars))) for _ in range(seq_len)]

    if sys.argv[1] == "last":
        y = x[-1]
    elif sys.argv[1] == "first":
        y = x[0]
    else:
        assert False

    x = torch.LongTensor(x).unsqueeze(0)
    y = torch.LongTensor([y]).unsqueeze(0)
    

    prediction = net(x)
    loss = criterion(prediction, y.squeeze(1))
    print(loss.item())

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        x = [random.choice(list(range(num_chars))) for _ in range(seq_len)]
        pretty_x = [chr(ord('a') + _) for _ in x]
        x = torch.LongTensor(x).unsqueeze(0)
        pred = net(x)[0].argmax()
        pretty_pred = chr(ord('a') + pred.item())
        print(pretty_x, "->",  pretty_pred)
