import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import GumbelSoftmax


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, latent_size: int = 32, image_size: tuple = (84, 84)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(5, 5), padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(5, 5), padding=2)
        self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(5, 5), padding=2)
        # self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear_out = nn.Linear(
            in_features=hidden_channels * image_size[0] * image_size[1],
            out_features=latent_size * 2
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.relu3(x)
        x = self.flatten(x)
        x = self.linear_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, fc_width, hidden_channels, out_channels, image_size):
        super().__init__()
        self.image_size = image_size
        self.linear1 = nn.Linear(in_features=latent_size * 2, out_features=fc_width)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=fc_width, out_features=image_size[0] * image_size[1] * out_channels)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = torch.reshape(x, (-1, 3, *self.image_size))
        return x


class StateAE(nn.Module):
    def __init__(self, parameters, device):
        super().__init__()
        self.encoder = Encoder(3, parameters.encoder_hidden_channels, parameters.latent_size, parameters.image_size)
        self.activation = GumbelSoftmax(device=device, total_epochs=parameters["epochs"])
        self.decoder = Decoder(parameters.latent_size, parameters.fc_width, parameters.decoder_hidden_channels, 3, parameters.image_size)
    
    def forward(self, x, epoch=1):
        out = {"input": x}
        out["encoded"] = self.encoder(out["input"])
        out["discrete"] = self.activation(out["encoded"], epoch)
        out["decoded"] = self.decoder(out["discrete"])
        return out