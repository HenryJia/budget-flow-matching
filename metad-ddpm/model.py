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

class AttentionStack(nn.Module):
    def __init__(self, channels, num_heads, num_layers=4):
        super(AttentionStack, self).__init__()

        self.norm = nn.ModuleList([nn.LayerNorm(channels) for i in range(num_layers)])

        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True) for i in range(num_layers)]
        )
        self.linear = nn.ModuleList(
            [nn.Sequential(nn.ELU(), nn.Linear(channels, channels), nn.ELU()) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        for attn, linear, norm in zip(self.attn, self.linear, self.norm):
            x_attn = norm(out)
            x_attn, _ = attn(x_attn, x_attn, x_attn)
            out = out + linear(x_attn)
        out = out.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return out

# The specific U-Net architecture is not based on the paper, and is shamelessly adapted from
# https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, sinusoidal_embedding_size=None, use_attention=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # The original paper uses GroupNorm but this is too memory heavy
            nn.ELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.sinusoidal_embedding_size = sinusoidal_embedding_size
        if self.sinusoidal_embedding_size:
            self.time_embedding_net = nn.Sequential(
                nn.Linear(sinusoidal_embedding_size, out_channels*4),
                nn.ELU(),
                nn.Linear(out_channels*4, out_channels),
                nn.ELU()
            )

        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionStack(out_channels, num_heads=4, num_layers=1)

    def forward(self, x, time_embedding=None):
        out = self.conv1(x)
        if self.sinusoidal_embedding_size: # Add the time embedding in the middle of the 2 convolutions
            out = out + self.time_embedding_net(time_embedding)[:, :, None, None]
        out = self.conv2(out) + out # Residual connection as described in paper

        if self.use_attention:
            out = self.attn(out)

        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, sinusoidal_embedding_size, use_attention=False):
        super().__init__()
        self.conv = DoubleConv(
            in_channels, out_channels,
            sinusoidal_embedding_size=sinusoidal_embedding_size, use_attention=use_attention
            )

    def forward(self, x, time_embedding):
        out = F.avg_pool2d(x, kernel_size=2) # Downsample by a factor of 2
        return self.conv(out, time_embedding)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, sinusoidal_embedding_size=None, use_attention=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels,
                sinusoidal_embedding_size=sinusoidal_embedding_size, use_attention=use_attention
                )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels,
                sinusoidal_embedding_size=sinusoidal_embedding_size, use_attention=use_attention
                )

    def forward(self, x1, x2, time_embedding=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, time_embedding=time_embedding)


class UNet(nn.Module):
    def __init__(self, n_channels, sinusoidal_embedding_size, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8, sinusoidal_embedding_size=sinusoidal_embedding_size)
        self.down = nn.ModuleList()
        self.down.append(Down(8, 16, sinusoidal_embedding_size))
        self.down.append(Down(16, 32, sinusoidal_embedding_size))
        self.down.append(Down(32, 64, sinusoidal_embedding_size))
        self.down.append(Down(64, 128, sinusoidal_embedding_size, use_attention=True))
        self.down.append(Down(128, 256, sinusoidal_embedding_size, use_attention=True))
        self.down.append(Down(256, 512, sinusoidal_embedding_size, use_attention=True))
        factor = 2 if bilinear else 1
        self.down.append(Down(512, 1024 // factor, sinusoidal_embedding_size, use_attention=True))

        self.attn = AttentionStack(1024 // factor, num_heads=4, num_layers=4)

        self.up = nn.ModuleList()
        self.up.append(Up(1024, 512 // factor, bilinear, sinusoidal_embedding_size, use_attention=True))
        self.up.append(Up(512, 256 // factor, bilinear, sinusoidal_embedding_size, use_attention=True))
        self.up.append(Up(256, 128 // factor, bilinear, sinusoidal_embedding_size, use_attention=True))
        self.up.append(Up(128, 64 // factor, bilinear, sinusoidal_embedding_size, use_attention=True))
        self.up.append(Up(64, 32 // factor, bilinear, sinusoidal_embedding_size))
        self.up.append(Up(32, 16 // factor, bilinear, sinusoidal_embedding_size))
        self.up.append(Up(16, 8, bilinear, sinusoidal_embedding_size))
        self.out = nn.Conv2d(8, n_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        out = self.inc(x, time_embedding)

        down_layers = [out]
        for i, d in enumerate(self.down):
            out = d(out, time_embedding)
            if i < len(self.down) - 1: # Don't save the output of the last down layer as we won't have a skip connection for it
                down_layers.append(out)
        
        out = self.attn(out)

        for u in self.up:
            out = u(out, down_layers.pop(), time_embedding)

        return self.out(out)


class MetadynamicsDiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels,
            trajectory_length, sinusoidal_embedding_size, lr,
            metad_basis_size=128, metad_scale=0.001, metad_interval=10
            ):
        super(MetadynamicsDiffusionModel, self).__init__()

        self.trajectory_length = trajectory_length
        self.sinusoidal_embedding_size = sinusoidal_embedding_size
        self.lr = lr

        self.metad_basis_size = metad_basis_size
        self.metad_scale = metad_scale
        self.metad_interval = metad_interval

        # So because we're projecting linearly into a much lower dimension, running out of memory isn't an issue
        # However, our low dimensional projection will eventually run out of "space" to feasibly distinguish between different parameter configs
        # So, we will regularly reset the linear projection we use to project the parameters into the low dimensional space
        self.max_metad_length = self.metad_basis_size * 4

        self.reverse_diffusion_net = UNet(input_channels, sinusoidal_embedding_size, bilinear=False) 

        self.beta = nn.Parameter(torch.linspace(start=1e-4, end=0.02, steps=trajectory_length), requires_grad=False)

        # This can be done with a simple boolean, but it seems for multigpu, things can get a bit weird
        # If we use a parameter Lightning will automatically sync it across the gpus for us, which is very convenient
        self.metadynamics_count = 0
        self.metadynamics_idx = 0
        self.use_metadynamics = nn.Parameter(torch.tensor((0,), dtype=torch.uint8), requires_grad=False)
        self.metadynamics_loc = nn.Parameter(torch.zeros((self.max_metad_length, self.metad_basis_size)), requires_grad=False)
        self.weight_basis = nn.Parameter(self.get_weight_basis(), requires_grad=False)
        self.metad_width = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def get_all_parameters(self):
        all_parameters = []
        for param in self.parameters():
            if param.requires_grad:
                all_parameters.append(param)
        all_parameters = torch.cat([p.view(-1) for p in all_parameters])
        return all_parameters

    def get_weight_basis(self):
        all_parameters = self.get_all_parameters()
        weight_basis = torch.randn(self.metad_basis_size, all_parameters.shape[0], device=all_parameters.device, dtype=all_parameters.dtype)
        weight_basis = F.normalize(weight_basis, dim=-1)
        return weight_basis.detach()

    def reset_metadynamics(self):
        #print("Reseting metadynamics...")
        # We'll trigger this function if and when we want to start using metadynamics
        with torch.no_grad():
            self.weight_basis.copy_(F.normalize(torch.randn_like(self.weight_basis), dim=-1))
            self.metadynamics_loc.copy_(torch.zeros_like(self.metadynamics_loc))
            self.metad_width.copy_(torch.tensor(1e-8))
        self.metadynamics_count = 0
        self.metadynamics_idx = 0

    def activate_metadynamics(self):
        with torch.no_grad():
            self.use_metadynamics.copy_(torch.ones_like(self.use_metadynamics))
        self.reset_metadynamics()

    def deactivate_metadynamics(self):
        with torch.no_grad():
            self.use_metadynamics.copy_(torch.zeros_like(self.use_metadynamics))
        self.reset_metadynamics()

    def metadynamics_loss(self):
        # This is the metadynamics function from metadynamics in computational chemistry
        all_parameters = self.get_all_parameters()
        loc = torch.matmul(self.weight_basis.detach(), all_parameters)

        # So we hardcoded the scale of the Gaussians as a hyperparameter, as this feels intuitive
        # The width of the Gaussians however, is very much less intuitive here
        # In computational chemistry, the collective variables are real physical quantities with units
        # So choosing the width can be done with reference to the physical system. Here, we have no such intuition to draw on
        # So, we'll try by basing it on the distance between the first 2 metadynamics locations

        # If this is the first time, then we just add the location as a parameter and return 0 loss as we have no past locations to compare to
        loss = 0.0
        with torch.no_grad():
            if self.metadynamics_count == 0:
                self.metadynamics_loc[0] = loc
                self.metadynamics_count += 1
                self.metadynamics_idx += 1
                return 0.0

        # If we have 2 locations, use it to set the width
        # Note: second location can only be added after we've passed metad_interval batches
        if self.metadynamics_idx >= self.metad_interval:
            if self.metadynamics_count == 1:
                metad_width = torch.sqrt(F.mse_loss(self.metadynamics_loc[0], loc, reduction='mean')) / (self.metad_interval * 2)
                metad_width = metad_width.clamp(min=1e-8, max=10.0) # Clamp the width to avoid numerical issues with divide by 0
                self.metad_width.copy_(metad_width.detach())

            for i in range(self.metadynamics_count):
                past_loc = self.metadynamics_loc[i].clone().detach() # Note: must use clone or the autograd engine will throw a tantrum
                mse = F.mse_loss(loc, past_loc, reduction='mean') / (2 * self.metad_width.detach()**2)
                mse = mse.clamp(max=10) # Clamp the mse to avoid numerical issues with torch.exp
                loss = loss + self.metad_scale * torch.exp(-mse)

            if self.metadynamics_idx % self.metad_interval == 0:
                with torch.no_grad():
                    self.metadynamics_loc[self.metadynamics_count].copy_(loc.clone().detach())
                self.metadynamics_count += 1

        self.metadynamics_idx += 1

        if self.metadynamics_count >= self.max_metad_length:
            self.reset_metadynamics()

        return loss

    def forward_diffusion(self, x_0, t):
        # Add noise to the input according to the beta schedule

        # Note: We will maintain notation consistency with the paper here.
        # But, it will mean that we have some inconsistent notation across different paper implementations.
        # Readers are advised to keep the paper handy for reference
        alpha = 1 - self.beta
        alpha_bar = torch.gather(torch.cumprod(alpha, dim=0), dim=0, index=t)
        beta_t = torch.gather(self.beta, dim=0, index=t)
        alpha_t = torch.gather(alpha, dim=0, index=t)

        epsilon_forward = torch.randn_like(x_0)
        x_t = x_0 * torch.sqrt(alpha_bar)[:, None, None, None] + epsilon_forward * torch.sqrt(1 - alpha_bar)[:, None, None, None]

        return x_t, epsilon_forward

    def reverse_diffusion(self, x_t, t):
        # We need a sinusoidal position embedding. This is briefly mentioned section 4.
        # It is described in more detail in appendix B
        pos_emb = torch.arange(self.sinusoidal_embedding_size//2, device=x_t.device).to(dtype=self.beta.dtype)[None, :]
        pos_emb = torch.exp(torch.log(t[:, None].to(dtype=self.beta.dtype)) - math.log(1e4) * 2 * pos_emb / self.sinusoidal_embedding_size)
        pos_emb = torch.cat([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)

        out = self.reverse_diffusion_net(x_t, pos_emb)

        return out

    def sample(self, x_t, t):
        # Sample a single step of the reverse diffusion process as described in Algorithm 2 of the paper
        epsilon_reverse = self.reverse_diffusion(x_t, t)

        alpha = 1 - self.beta
        alpha_bar = torch.gather(torch.cumprod(alpha, dim=0), dim=0, index=t)
        alpha_t = torch.gather(alpha, dim=0, index=t)
        beta_t = torch.gather(self.beta, dim=0, index=t)

        # Described in section 3.2, we can either choose
        #beta_tilde = (1 - alpha_bar / alpha_t) / (1 - alpha_bar) * beta_t # equation 7
        #sigma2_t = beta_tilde
        sigma2_t = beta_t

        coef = 1 / torch.sqrt(alpha_t)
        coef_eps = beta_t / torch.sqrt(1 - alpha_bar)
        out = coef[:, None, None, None] * (x_t - coef_eps[:, None, None, None] * epsilon_reverse)
        out = out + torch.sqrt(sigma2_t)[:, None, None, None] * torch.randn_like(out)

        return out

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # Step 1: Select a timestep in [1, trajectory_length - 1]
        # Note: We skip t=0 as the reverse process is fixed at the first step
        # Keep the same t for the whole minibatch. I'm a little uncertain about this
        # But, it is how the original reference implemntation in theano does it
        t = torch.randint(low=1, high=self.trajectory_length, size=(x.shape[0],), device=x.device)

        # Step 2: Run the forward diffusion process to get the noisy input up to timestep t, the mean and variance of the noise, and the timestep
        x_t, epsilon_forward = self.forward_diffusion(x, t)

        # Whilst the derivation to get here takes a bit more work, all we need is to predict the epsilons when running in reverse
        # This is based on Algorithm 1 in the paper
        epsilon_reverse  = self.reverse_diffusion(x_t, t)

        # Based on equation 14 and its accompanying explanation, we can do this simple loss or the more complicated one in equation 12
        # The paper suggests that the simple one works better, so we have no reason to do the more complicated one
        train_loss = F.mse_loss(epsilon_reverse, epsilon_forward, reduction='mean')
        self.log("train_loss", train_loss, prog_bar=True)

        metad_loss = self.metadynamics_loss() * self.use_metadynamics
        self.log("metad_loss", metad_loss, prog_bar=True)

        loss = train_loss + metad_loss

        self.log("metad_width", self.metad_width.item(), prog_bar=True)
        self.log("total_loss", loss, prog_bar=True)

        return loss

    def forward(self, x, trajectory_length=None):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        # Step 1: Draw a sample from the prior distribution
        x_t = torch.randn_like(x, dtype=self.beta.dtype)

        # Step 2: Run the reverse diffusion process for the whole trajectory
        if trajectory_length is None:
            trajectory_length = self.trajectory_length
        # Note: without no_grad torch will try to store all the intermediate steps for backprop
        # Which would blow up the memory. Ignore gradients for sampling.
        with torch.no_grad():
            for t in range(trajectory_length - 1, -1, -1):
                x_t = self.sample(x_t, t * torch.ones(x_t.shape[0], device=x_t.device, dtype=torch.long))
        return x_t

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=100, min_lr=2e-6)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss", "frequency": 1, "interval": "epoch"}