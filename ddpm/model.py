import torch
import math
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L


class MultiBlock(nn.Module):
    def __init__(self, input_dim, in_channels, out_channels):
        super(MultiBlock, self).__init__()
        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.convt = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=6, stride=2, padding=2)
        
        self.conv1 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, padding=0)
        
        self.dense = nn.Sequential(
            nn.Linear(input_dim[0]//2 * input_dim[1]//2 * out_channels, out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, out_channels * input_dim[0]//2 * input_dim[1]//2),
            nn.LeakyReLU()
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv_init(x)
    
        pooled = F.avg_pool2d(out, kernel_size=2, stride=2)
        pooled_w = pooled.shape[2]
        pooled_h = pooled.shape[3]
        pooled = self.dense(pooled.view(pooled.shape[0], -1)).view(pooled.shape[0], -1, pooled_w, pooled_h)

        out = torch.cat(
            [self.conv5(out), self.conv3(out), self.convt(pooled)], dim=1
        )
        out = self.activation(out)
        out = self.activation(self.conv1(out))
        return out


class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels, layers, hidden_channels,
            trajectory_length, sinusoidal_embedding_size,
            lr
            ):
        super(DiffusionModel, self).__init__()
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.trajectory_length = trajectory_length
        self.sinusoidal_embedding_size = sinusoidal_embedding_size
        self.lr = lr

        # For now, we'll just use a simple convolutional network for the reverse diffusion process
        # We can tune this later to be more efficient and better suited for the task
        modules = []
        modules.extend([nn.Conv2d(input_channels, hidden_channels, kernel_size=1, padding=0), nn.LeakyReLU()])
        for i in range(layers):
            modules.append(MultiBlock(input_dim, hidden_channels, hidden_channels))
        modules.append(nn.Conv2d(hidden_channels, input_channels, kernel_size=1, padding=0))
        self.reverse_diffusion_net = nn.ModuleList(modules)

        self.time_embedding_net = nn.Sequential(
            nn.Linear(sinusoidal_embedding_size, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU()
        )

        # Interestingly, unlike the nonequilibrium themodynamics paper, the betas are NOT learnable
        # We will use the same fixed beta schedule as described in section 4 of the paper
        self.beta = nn.Parameter(torch.linspace(start=1e-4, end=0.02, steps=trajectory_length), requires_grad=False)


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
        pos_emb = torch.arange(self.sinusoidal_embedding_size//2, device=x_t.device).float()[None, :]
        pos_emb = torch.exp(torch.log(t[:, None].float()) - math.log(1e4) * 2 * pos_emb / self.sinusoidal_embedding_size)
        pos_emb = torch.cat([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = self.time_embedding_net(pos_emb)[:, :, None, None]

        out = self.reverse_diffusion_net[0](x_t)
        for i in range(1, len(self.reverse_diffusion_net) - 1):
            out = self.reverse_diffusion_net[i](out + pos_emb) + out
        out = self.reverse_diffusion_net[-1](out)

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
        loss = F.mse_loss(epsilon_reverse, epsilon_forward, reduction='mean')

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def forward(self, x, trajectory_length=None):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        # Step 1: Draw a sample from the prior distribution
        x_t = torch.randn_like(x)

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}