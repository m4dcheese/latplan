import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super().__init__()
        self.stddev = stddev
    
    def forward(self, x: torch.Tensor):
        return torch.clip(x + self.stddev**0.5 * torch.randn_like(x), 0, 1)


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        self.relu = nn.ReLU()
        self.batch_normalization = nn.BatchNorm1d(num_features=out_features)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.relu(x)
        x = self.batch_normalization(x)
        x = self.dropout(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3))
        self.batch_normalization = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batch_normalization(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x