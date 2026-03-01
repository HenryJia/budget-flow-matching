import torch
import math
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

import diffusers


class OTFlowMatchingModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels,
            lr, sigma_min=0.0001,
            ):
        super(OTFlowMatchingModel, self).__init__()
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.lr = lr

        self.flow_net = diffusers.UNet2DModel(
            sample_size=input_dim,
            in_channels=input_channels,
            out_channels=input_channels,
            layers_per_block=2,
            block_out_channels=(32, 32, 64, 64, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",# a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",# a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.sigma_min = sigma_min

    def psi(self, x, x_1, t):
        # This is equation 22 in the paper
        t_expand = t[:, None, None, None]
        return (1 - (1 - self.sigma_min) * t_expand) * x + t_expand * x_1

    def flow(self, t, x_t): # Note this is v_t in the paper (e.g. Equation 23)
        # Huggingface will take care of generating the time embedding for us
        # However, note that Huggingface's UNet models are designed for diffusion and expect t to be in [0, 1000]
        # So we just scale it up
        out = self.flow_net(x_t, t * 1000, return_dict=False)[0]

        return out

    def training_step(self, batch, batch_idx):
        x_1 = batch[0]

        # Unlike diffusion models, our timestep is continuous in [0, 1]
        t = torch.rand(size=(x_1.shape[0],), device=x_1.device)

        # Note: Unlike diffusion models, x_1 is the data and x_0 is our prior distribution (which is N(0, I))
        x_0 = torch.randn_like(x_1)

        psi_t = self.psi(x_0, x_1, t)

        flow_t = self.flow(t, psi_t)

        target = x_1 - (1 - self.sigma_min) * x_0

        # Based on equation 14 and its accompanying explanation, we can do this simple loss or the more complicated one in equation 12
        # The paper suggests that the simple one works better, so we have no reason to do the more complicated one
        loss = F.mse_loss(flow_t, target.detach(), reduction='mean')
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, x, steps=10):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        # This is effectively basically a simple leap frog integrator for the ODE defined by the flow field.

        with torch.no_grad():
            # Step 1: Draw a sample from the prior distribution
            x_0 = torch.randn_like(x)

            # Step 2: Generate the trajectory using an ODE solver
            # There's no point in writing our own ODE solver when torchdiffreq has a perfectly good one
            t = torch.linspace(0, 1, steps=steps + 1, device=x.device)
            trajectory = odeint(self.flow, x_0, t, atol=1e-5, rtol=1e-3, method='dopri5')
        return trajectory[-1] # Return the final point in the trajectory, which is our sample from the data distribution

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=100, min_lr=2e-6)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss", "frequency": 1, "interval": "epoch"}