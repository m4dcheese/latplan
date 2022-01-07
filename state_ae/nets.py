import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import FullyConnectedBlock, ConvBlock, GaussianNoise

'''
Encoder and decoder networks
'''

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.gaussian_noise = GaussianNoise(stddev=parameters["gaussian_noise"])
        # self.fully_connected_block_1 = FullyConnectedBlock(
        #     in_features=84*84,
        #     out_features=parameters["fc_width"],
        #     dropout=parameters["dropout"]
        # )
        self.conv_block_1 = ConvBlock(
            in_channels=1,
            out_channels=16,
            dropout=parameters["dropout"]
        )
        self.conv_block_2 = ConvBlock(
            in_channels=16,
            out_channels=16,
            dropout=parameters["dropout"]
        )
        self.flatten = nn.Flatten()
        flat_conv_output_shape = 16 * (parameters["image_size"][0] - 4) * (parameters["image_size"][1] - 4)
        self.fully_connected_end = nn.Linear(
            in_features=flat_conv_output_shape,
            out_features=2 * parameters["latent_size"]
        )
    
    def forward(self, x: torch.Tensor):
        # TODO disable Gaussian Noise for eval mode
        x = self.gaussian_noise(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.flatten(x)
        x = self.fully_connected_end(x)
        return x


class Decoder(nn.Module):

    def __init__(self, parameters):
        super().__init__()
        self.start_batchnorm = torch.nn.BatchNorm1d(num_features=parameters["latent_size"] * 2)
        self.fully_connected_block_1 = FullyConnectedBlock(
            in_features=parameters["latent_size"] * 2,
            out_features=parameters["fc_width"],
            dropout=parameters["dropout"]
        )
        self.fully_connected_end = nn.Linear(
            in_features=parameters["fc_width"],
            out_features=parameters["image_size"][0] * parameters["image_size"][1]
        )
        self.output_shape = (-1, 1, parameters["image_size"][0], parameters["image_size"][1])

    def forward(self, x: torch.Tensor):
        x = self.start_batchnorm(x)
        x = self.fully_connected_block_1(x)
        x = self.fully_connected_end(x)
        x = torch.reshape(x, self.output_shape)
        return x
