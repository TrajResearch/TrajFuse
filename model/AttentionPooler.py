from datetime import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch



class AttentionPooler(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
    
    def forward(self, x, mask=None):
        # x: [b, seq_len, d_model]
        weights = self.attn(x).squeeze(-1)  # [b, seq_len]
        if mask is not None:
            weights = weights.masked_fill(mask, -1e9)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)  # [b, seq_len, 1]
        return (x * weights).sum(dim=1)  # [b, d_model]