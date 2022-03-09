import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from parameters import parameters


class GaussianNoise(nn.Module):
    def __init__(self, stddev = 0.1):
        super().__init__()
        self.stddev = stddev
    
    def forward(self, x: torch.Tensor):
        return torch.clip(x + self.stddev**0.5 * torch.randn_like(x), 0, 1)


def get_tau(epoch, t_max=5, t_min=0.1, total_epochs=1000):
    iters_per_epoch = parameters.total_samples / parameters.batch_size
    epoch_start_decrease = math.ceil(parameters.warm_up_steps / iters_per_epoch)
    if epoch * iters_per_epoch < parameters.warm_up_steps:
        return t_max
    return t_max * (t_min / t_max) ** (min(epoch - epoch_start_decrease, (total_epochs - epoch_start_decrease)) / (total_epochs - epoch_start_decrease))


class GumbelSoftmax(nn.Module):

    def __init__(self, device, total_epochs):
        super().__init__()
        self.device = device
        self.total_epochs = total_epochs
    
    def forward(self, x, epoch, hard: bool = False, eps=1e-10):
        batch_size = x.shape[0]
        num_of_variable_pairs = int(x.shape[-1] / 2)
        pairs = torch.reshape(x, (batch_size, num_of_variable_pairs, 2))

        if hard:
            max_idx = torch.argmax(pairs, dim=-1)
            output = F.one_hot(max_idx).float()
        else:
            tau = get_tau(epoch, total_epochs=self.total_epochs)
            u = torch.rand(pairs.shape, device=self.device)
            gumbel = -torch.log(-torch.log(u + eps) + eps)
            logits = (pairs + gumbel) / tau
            output = F.softmax(logits, dim=-1)

        output = torch.reshape(output, x.shape)
        return output


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        hidden_channels: int = 32,
        latent_size: int = 32,
        dropout: float = .4,
        image_size: tuple = (84, 84)
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=1)
        self.activation1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 3), padding=1)
        self.activation2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=hidden_channels)
        self.dropout2 = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        self.linear_out = nn.Linear(
            in_features=hidden_channels * image_size[0] * image_size[1],
            out_features=latent_size * 2
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_size: int = 32,
        fc_width: int = 1000,
        dropout: float = 0.4,
        image_size: tuple = (84, 84)
    ):
        super().__init__()
        self.image_size = image_size
        self.linear1 = nn.Linear(in_features=latent_size * 2, out_features=fc_width)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=fc_width)
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=fc_width, out_features=fc_width)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_features=fc_width)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(in_features=fc_width, out_features=image_size[0] * image_size[1] * out_channels)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = torch.reshape(x, (-1, 3, *self.image_size))
        return x


class StateAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.gaussian_noise = GaussianNoise(stddev=parameters.gaussian_noise)
        self.encoder = Encoder(
            hidden_channels=parameters.encoder_channels,
            latent_size=parameters.latent_size,
            dropout=parameters.dropout,
            image_size=parameters.image_size
        )
        self.activation = GumbelSoftmax(device=device, total_epochs=parameters["epochs"])
        self.decoder = Decoder(
            latent_size=parameters.latent_size,
            fc_width=parameters.fc_width,
            dropout=parameters.dropout,
            image_size=parameters.image_size
        )
    
    def forward(self, x, epoch=1):
        out = {"input": x}
        out["noisy"] = self.gaussian_noise(out["input"])
        out["encoded"] = self.encoder(out["noisy"])
        out["discrete"] = self.activation(out["encoded"], epoch, hard=not self.training)
        out["decoded"] = self.decoder(out["discrete"])
        return out