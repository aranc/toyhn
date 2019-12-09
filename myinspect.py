import torch
import sys

from b import F


best_f = torch.load("best_f.pkl")

f = F()

f.load_state_dict(best_f)

print(f(torch.Tensor([0])))
print(f(torch.Tensor([1])))
