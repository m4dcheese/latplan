import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import FullyConnectedBlock, GaussianNoise

'''
Encoder and decoder networks
'''

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.gaussian_noise = GaussianNoise(stddev=parameters["gaussian_noise"] or .1)
        self.fully_connected_block_1 = FullyConnectedBlock(
            in_features=84*84,
            out_features=parameters["fc_width"],
            dropout=parameters["dropout"]
        )
        self.fully_connected_end = nn.Linear(
            in_features=parameters["fc_width"],
            out_features=parameters["latent_size"]
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.gaussian_noise(x)
        x = self.fully_connected_block_1(x)
        x = self.fully_connected_end(x)
        return x


class Decoder(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.start_batchnorm = torch.nn.BatchNorm1d(num_features=parameters["latent_size"])
        self.fully_connected_block_1 = FullyConnectedBlock(
            in_features=parameters["latent_size"],
            out_features=parameters["fc_width"],
            dropout=parameters["dropout"]
        )
        self.fully_connected_end = nn.Linear(
            in_features=parameters["fc_width"],
            out_features=84*84
        )

    def forward(self, x):
        x = self.start_batchnorm(x)
        x = self.fully_connected_block_1(x)
        x = self.fully_connected_end(x)
        x = torch.reshape(x, (84, 84))
        return x
