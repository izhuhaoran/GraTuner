import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

import dgl

def testdgl():
    g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 3])))
    h1 = g.in_degrees().view(-1, 1).float() # [N, in_dim=1]
    h2 = g.out_degrees().view(-1, 1).float() # [N, out_dim=1]
    
    h = torch.cat([h1, h2], dim=1)  # [N, 2]
    print(h)

a = torch.tensor([1, 2, 3, 4])
print(a)
b = a.expand(2, 4)
print(b)
print(1)