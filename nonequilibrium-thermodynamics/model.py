import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels, layers, hidden_channels,
            trajectory_length, temporal_basis_size, step1_beta
            ):

        output_channels = input_channels * temporal_basis_size * 2
        modules = []
        modules.extend([nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1), nn.LeakyReLU()])
        for i in range(layers):
            modules.extend([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.LeakyReLU()])
            modules.extend([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0), nn.LeakyReLU()])
        modules.append(nn.Conv2d(hidden_channels, output_channels, kernel_size=1, padding=0))
        self.reverse_diffusion = nn.Sequential(*modules)

        super(DiffusionModel, self).__init__()

    def forward(self, x):
        return x