import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super().__init__()
        self.stddev = stddev
    
    def forward(self, x: torch.Tensor):
        return x + self.stddev * torch.rand_like(x)


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        self.relu = nn.ReLU()
        self.batch_normalization = nn.BatchNorm1d(num_features=out_features)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor):
        print(x.dtype)
        x = self.linear(x)
        x = self.relu(x)
        x = self.batch_normalization(x)
        x = self.dropout(x)
        return x
