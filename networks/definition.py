## Utilities
import json
import os
import colorsys
import math
import numpy as np
## Pytorch
import torch_geometric
from torch_geometric.data import Data
from torch import Tensor, LongTensor
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
## Render
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import networkx as nx

## Simple network
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 64, 1)
        self.linear = torch.nn.Linear(64, 1)
    def forward(self, x, edge_index, edge_weight, memory=None):
        memory = self.recurrent(x, edge_index, edge_weight, memory)
        h = F.relu(memory)
        h = self.linear(h)
        return h, memory