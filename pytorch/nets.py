import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GaussianNoise

'''
Encoder and decoder networks
'''

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.gaussian_noise = GaussianNoise(stddev=parameters["gaussian_noise"] or .1)
    
    def forward(self, x):
        # 1. Flatten
        # 2. Gaussian Noise
        # 3. Dense, BN, Dropout combinations
        # 4. Dense to latent dim
        pass


class Decoder(nn.Module):

    def __init__(self, parameters):
        super().__init__()

    def forward(self, x):
        pass
