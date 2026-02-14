import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels, layers, hidden_channels,
            trajectory_length, step1_beta, temporal_basis_size
            ):
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.trajectory_length = trajectory_length
        self.step1_beta = step1_beta
        self.temporal_basis_size = temporal_basis_size

        # I am very unsure we need to use the temporal basis functions at all. The reference implementation does, but it is not clear to me why
        output_channels = input_channels * 2 #* temporal_basis_size

        # For now, we'll just use a simple convolutional network for the reverse diffusion process
        # We can tune this later to be more efficient and better suited for the task
        modules = []
        modules.extend([nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1), nn.LeakyReLU()])
        for i in range(layers):
            modules.extend([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.LeakyReLU()])
            modules.extend([nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0), nn.LeakyReLU()])
        modules.append(nn.Conv2d(hidden_channels, output_channels, kernel_size=1, padding=0))
        self.reverse_diffusion_net = nn.Sequential(*modules)

        # Note: The beta diffusion rate is learnable
        self.beta = torch.Parameter(self.generate_beta(self, step1_beta))

        super(DiffusionModel, self).__init__()

    def generate_beta(self, step1_beta):
        # Generate the beta schedule for the forward diffusion process
        min_beta = max(1e-6, step1_beta)

        # So the reference implementation does something a little more complicated
        # They set up a schedule of betas that eras a fixed fraction of the signal at each step
        # This they describe in the paper for binomial diffusion. However, it is not mentioned for the gaussian case
        # I believe they implemented this for the binomial case, and then kept it for the gaussian case
        # However, it is not clear to me that this is necessary for the gaussian case.
        # In fact, it seems to me that a simple linear schedule should work just fine
        return torch.ones(self.trajectory_length - 1) * max(min_beta, 1.0 / self.trajectory_length)


    def forward_diffusion(self, x):
        # Step 1: Select a timestep in [1, trajectory_length - 1]
        # Note: We skip t=0 as the reverse process is fixed at the first step
        # Keep the same t for the whole minibatch. I'm a little uncertain about this
        # But, it is how the original reference implemntation in theano does it
        t = torch.randint(low=1, high=self.trajectory_length, size=(1,), device=x.device)

        # Step 2: Add noise to the input according to the beta schedule
        # Note that beta for the first step is fixed, and the rest are learnable
        # This is not the most efficient way to do this, but it is the clearest way to implement the forward diffusion process
        # Note, this also starts to deviate from the reference implementation which uses uniform noise for some reason
        # But, the paper describes the forward diffusion process as adding gaussian noise, so we will do that here
        # This is the diffusion kernel from Table App.1 in the paper
        x_noisy = x * torch.sqrt(1 - self.step1_beta) + torch.randn_like(x) * torch.sqrt(self.step1_beta)
        mu = x_noisy * torch.sqrt(1 - self.step1_beta)
        sigma = torch.sqrt(self.step1_beta)
        for i in range(1, t):
            x_noisy = x_noisy * torch.sqrt(1 - self.beta[i-1]) + torch.randn_like(x) * torch.sqrt(self.beta[i-1])
            mu = mu * torch.sqrt(1 - self.beta[i-1])
            sigma = torch.sqrt(sigma**2 * (1 - self.beta[i-1]) + self.beta[i-1])

        return x_noisy, mu, sigma, t


    def reverse_diffusion(self, x_noisy, t):
        # Step 1: Run the reverse diffusion process through the network to get the predicted noise
        z = self.reverse_diffusion_net(x_noisy)

        # Now, the paper does something a little more complicated here with multiplying by some r(x^(t))
        # But, they also say that for all their experiments, r(x^(t)) is constant. So I'm going to ignore it for now
        # They also do something with the temporal basis functions here, but I'm not sure what the point of that is.
        # In addition, they use a somewhat odd formula for the reverse diffusion process for images in the Appendix (eq 64 and 65)
        # I don't entirely understand why to be honest
        # All in all, I'm going to ignore all those complications for now, and just use a simple formula for the reverse diffusion process
        mu = z[:, :x_noisy.shape[1], :, :]
        sigma = torch.sqrt(F.softplus(z[:, x_noisy.shape[1]:, :, :]))

        return mu, sigma