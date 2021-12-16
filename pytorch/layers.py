import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super.__init__()
        self.stddev = stddev
    
    def forward(self, x: torch.Tensor):
        return x + self.stddev * torch.rand_like(x)
