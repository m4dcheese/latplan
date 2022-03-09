import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import parameters

"""
COPIED FROM Ella Morgan's IMPLEMENTATION

GumbelSoftmax and BinaryConcrete implementations inspired by https://github.com/dev4488/VAE_gumble_softmax/
"""


# Calculates tau - formula provided in section 3.1.6 Gumbel Softmax
def get_tau(epoch, t_max=5, t_min=0.1, total_epochs=1000):
    iters_per_epoch = parameters.total_samples / parameters.batch_size
    epoch_start_decrease = math.ceil(parameters.warm_up_steps / iters_per_epoch)
    if epoch * iters_per_epoch < parameters.warm_up_steps:
        return t_max
    return t_max * (t_min / t_max) ** (min(epoch - epoch_start_decrease, (total_epochs - epoch_start_decrease)) / (total_epochs - epoch_start_decrease))



# GumbelSoftmax is for the actions - discretizes as tau approaches 0
class GumbelSoftmax(nn.Module):

    def __init__(self, device, total_epochs):
        super().__init__()
        self.device = device
        self.total_epochs = total_epochs
    
    def forward(self, x, epoch, hard: bool = False, eps=1e-10):
        # x: (batch, a)
        batch_size = x.shape[0]
        num_of_variable_pairs = int(x.shape[-1] / 2)
        pairs = torch.reshape(x, (batch_size, num_of_variable_pairs, 2))

        if not hard:
            tau = get_tau(epoch, total_epochs=self.total_epochs)
            u = torch.rand(pairs.shape, device=x.device)
            gumbel = -torch.log(-torch.log(u + eps) + eps)
            logits = (pairs + gumbel) / tau
            output = F.softmax(logits, dim=-1)
        else:
            max_idx = torch.argmax(pairs, dim=-1)
            output = F.one_hot(max_idx).float()

        output = torch.reshape(output, x.shape)
        return output



# BinaryConcrete is for the state - discretizes as tau approaches 0
class BinaryConcrete(nn.Module):

    def __init__(self, device, total_epochs):
        super().__init__()
        self.device = device
        self.total_epochs = total_epochs
    
    def forward(self, x, epoch, eps=1e-20):
        # x: (batch, f)
        tau = get_tau(epoch, total_epochs=self.total_epochs)
        u = torch.rand(x.shape, device=self.device)
        logistic = torch.log(u + eps) - torch.log(1 - u + eps)
        logits = (x + logistic) / tau
        binary_concrete = torch.sigmoid(logits)
        return binary_concrete
