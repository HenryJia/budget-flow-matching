import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_channels, layers, hidden_channels,
            trajectory_length, step1_beta, temporal_basis_size,
            lr=1e-3
            ):
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.trajectory_length = trajectory_length
        self.step1_beta = step1_beta
        self.temporal_basis_size = temporal_basis_size
        self.lr = lr

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
        return torch.ones(self.trajectory_length) * max(min_beta, 1.0 / self.trajectory_length)


    def forward_diffusion(self, x, t):
        # Step 2: Add noise to the input according to the beta schedule
        # Note that beta for the first step is fixed, and the rest are learnable
        # This is not the most efficient way to do this, but it is the clearest way to implement the forward diffusion process
        # Note, this also starts to deviate from the reference implementation which uses uniform noise for some reason
        # But, the paper describes the forward diffusion process as adding gaussian noise, so we will do that here
        # This is the diffusion kernel from Table App.1 in the paper
        x_noisy = x 
        mu = x_noisy 
        sigma = 0
        beta = torch.clip(F.softplus(self.beta), min=self.step1_beta) # We use softplus to ensure that beta is positive
        for i in range(t):
            x_noisy = x_noisy * torch.sqrt(1 - beta[i]) + torch.randn_like(x) * torch.sqrt(beta[i])
            mu = mu * torch.sqrt(1 - beta[i])
            sigma = torch.sqrt(sigma**2 * (1 - beta[i]) + beta[i])

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

    def training_step(self, batch, batch_idx):
        x = batch
        # Step 1: Select a timestep in [1, trajectory_length - 1]
        # Note: We skip t=0 as the reverse process is fixed at the first step
        # Keep the same t for the whole minibatch. I'm a little uncertain about this
        # But, it is how the original reference implemntation in theano does it
        t = torch.randint(low=1, high=self.trajectory_length, size=(1,), device=x.device)

        # Step 2: Run the forward diffusion process to get the noisy input up to timestep t, the mean and variance of the noise, and the timestep
        x_noisy, mu_t, sigma_t, t = self.forward_diffusion(x, t)

        # Step 3: Run the reverse diffusion process to get the predicted mean and variance of the noise
        mu_pred, sigma_pred = self.reverse_diffusion(x_noisy, t)

        # Now we are ready to calculate the negative log likelihood bound
        # Step 4: Calculate kl divergence
        kl = torch.log(sigma_pred) - torch.log(sigma_t) + (sigma_t**2 + (mu_t - mu_pred)**2) / (2 * sigma_pred**2) - 0.5

        # Step 5: Calculate the entropy of the forward diffusion process at the end of the trajectory
        # This is H_q(X^(T) | X^(0)) in Table App.1 in the paper
        # For this, we'll need to calculate the mean and variance of the forward diffusion process at the end of the trajectory
        beta = torch.clip(F.softplus(self.beta), min=self.step1_beta)
        for i in range(t, self.trajectory_length):
            # mu_t = mu_t * torch.sqrt(1 - beta[i-1]) # No need to calculate the mean for calculating entropy
            sigma_t = torch.sqrt(sigma_t**2 * (1 - beta[i]) + beta[i])

        # We'll abbreviate H_q(X^(T) | X^(0)) as H_qT
        H_qT = 0.5 * (torch.log(2 * torch.pi) + 2 * torch.log(sigma_t) + 1)

        # Step 6: Calculate the entropy of the forward diffusion process at t=1
        # This is H_q(X^(1) | X^(0)) in Table App.1 in the paper
        # Note that at t=1, the standard deviation is beta[0]. We don't need to calculate the mean for calculating entropy
        H_q1 = 0.5 * (torch.log(2 * torch.pi) + 2 * torch.log(beta[0]) + 1)

        # Step 7: Add H_p(X^(T)) which is the prior entropy at the end of the forward diffusion process.
        # This is just the entropy of a standard normal distribution
        H_pT = 0.5 * (torch.log(2 * torch.pi) + 1)

        # Step 8: Put it all together to get the negative log likelihood bound
        # Note that we multiply the KL divergence by the trajectory length as is done in the reference implementation
        # My best guess at the reasoning within this is that you can either calculate the loss over the whole trajectory,
        # or you can calculate it at a single timestep, and pretend you have it for the whole trajectory.
        # By the power of sampling different points in the trajectory, by our Lord and Saviour Monte Carlo,
        # we somehow get a good enough estimate of the loss over the whole trajectory.
        log_lower_bound = -kl * self.trajectory_length + H_qT - H_q1 - H_pT
        loss = -log_lower_bound.mean() # Of course, we want to maximise the log lower bound, so we flip it for SGD

        return loss

    def forward(self, x):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        # Step 1: Draw a sample from the prior distribution
        x_noisy = torch.randn_like(x)

        # Step 2: Run the reverse diffusion process for the whole trajectory
        for t in range(self.trajectory_length - 1, -1, -1):
            mu, sigma = self.reverse_diffusion(x_noisy, t)
            x_noisy = mu + sigma * torch.randn_like(x)
        return x_noisy

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer