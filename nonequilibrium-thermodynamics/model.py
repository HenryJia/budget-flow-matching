import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels,
            trajectory_length,
            ):
        super(DiffusionModel, self).__init__()

    def forward(self, x):
        return x