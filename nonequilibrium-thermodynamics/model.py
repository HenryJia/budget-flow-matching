import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

import diffusers

class DiffusionModel(L.LightningModule):
    def __init__(
            self, input_dim, input_channels,
            trajectory_length, step1_beta,
            lr=1e-4
            ):
        super(DiffusionModel, self).__init__()
        # Note: This is going to be a bit different to the reference theano implementation
        # The reference implementation does a fair bit of more complicated stuff which I think is a tad esoteric

        self.trajectory_length = trajectory_length
        self.step1_beta = step1_beta
        self.lr = lr

        output_channels = input_channels + 1

        self.reverse_diffusion_net = diffusers.UNet2DModel(
            sample_size=input_dim,
            in_channels=input_channels,
            out_channels=output_channels,
            layers_per_block=2,
            block_out_channels=(64, 64, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",# a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",# a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
            ),
        ) 
        # Note: The beta diffusion rate is learnable
        self.beta = nn.Parameter(self.generate_beta(step1_beta))


    def generate_beta(self, step1_beta):
        # Generate the beta schedule for the forward diffusion process
        min_beta = max(1e-6, step1_beta)
        beta = torch.ones(self.trajectory_length) * max(min_beta, 1.0 / self.trajectory_length)
        beta = torch.logit(beta) # Use logit because we will be using sigmoid later...

        return beta


    def forward_diffusion(self, x, t):
        # Add noise to the input according to the beta schedule
        # Note that beta for the first step is fixed, and the rest are learnable
        # This is not the most efficient way to do this, but it is the clearest way to implement the forward diffusion process
        # Note, this also starts to deviate from the reference implementation which uses uniform noise for some reason
        # But, the paper describes the forward diffusion process as adding gaussian noise, so we will do that here
        # This is the diffusion kernel from Table App.1 in the paper
        x_noisy = x

        beta = torch.clamp(F.sigmoid(self.beta), min=self.step1_beta, max=1.0-self.step1_beta)

        # OK so this is a tad different to how the paper describes the forward diffusion kernel in Table App.1
        # But to my understanding, they are equivalent
        alpha = 1 - beta
        alpha_cum = torch.gather(torch.cumprod(alpha, dim=0), dim=0, index=t)
        beta_t = torch.gather(beta, dim=0, index=t)
        alpha_t = torch.gather(alpha, dim=0, index=t)

        x_noisy = x_noisy * torch.sqrt(alpha_cum)[:, None, None, None] + torch.randn_like(x) * torch.sqrt(1 - alpha_cum)[:, None, None, None]

        # Now we calculate the posterior mean and variance for the reverse diffusion process at timestep t
        # This is the q(x^(t-1) | x^(t), x^(0)) distribution in Table App.1 in the paper
        mu0 = x * torch.sqrt(alpha_cum / alpha_t)[:, None, None, None] # prior mean of the forward diffusion process at timestep t-1
        mu1 = x_noisy / torch.sqrt(alpha_t)[:, None, None, None] # likelihood mean of the forward diffusion process at timestep t-1 from timestep t

        # Note, I'm not entirely certain how the reference implementation derived this, but we'll roll with it
        covar1 = 1.0 - alpha_cum / alpha_t # prior covariance of the mean of the forward diffusion process at timestep t-1
        covar2 = beta_t / alpha_t # likelihood covariance of the mean of the forward diffusion process at timestep t-1 from timestep t
        lam = 1.0 / covar1 + 1.0 / covar2

        mu_posterior = (mu0 / covar1[:, None, None, None] + mu1 / covar2[:, None, None, None]) / lam[:, None, None, None]
        sigma_posterior = torch.sqrt(1.0 / lam)

        return x_noisy, mu_posterior, sigma_posterior

    def reverse_diffusion(self, x_noisy, t):
        channels = x_noisy.shape[1]

        # Step 1: Run the reverse diffusion process through the network to get the predicted noise
        z = self.reverse_diffusion_net(x_noisy, t, return_dict=False)[0]

        # Now, the paper does something a little more complicated here with multiplying by some r(x^(t))
        # But, they also say that for all their experiments, r(x^(t)) is constant. So I'm going to ignore it for now
        # They also do something with the temporal basis functions here, but I'm not sure what the point of that is.
        # In addition, they use a somewhat odd formula for the reverse diffusion process for images in the Appendix (eq 64 and 65)
        # I don't entirely understand why to be honest
        # All in all, I'm going to ignore all those complications for now, and just use a simple formula for the reverse diffusion process
        mu = z[:, :channels, :, :]
        sigma = z[:, channels, :, :]
        sigma = torch.mean(sigma, dim=(1, 2)) # Only one sigma per sample

        beta = torch.clamp(torch.gather(F.sigmoid(self.beta), dim=0, index=t), min=self.step1_beta, max=1.0-self.step1_beta)

        # The paper does this, but I'm not sure why
        sigma = torch.sqrt(F.sigmoid(sigma + torch.logit(beta)))
        mu = x_noisy * torch.sqrt(1 - beta[:, None, None, None]) + mu * torch.sqrt(beta[:, None, None, None])

        return mu, sigma

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # Step 1: Select a timestep in [1, trajectory_length - 1]
        # Note: We skip t=0 as the reverse process is fixed at the first step
        # Keep the same t for the whole minibatch. I'm a little uncertain about this
        # But, it is how the original reference implemntation in theano does it
        t = torch.randint(low=1, high=self.trajectory_length, size=(x.shape[0],), device=x.device)

        # Step 2: Run the forward diffusion process to get the noisy input up to timestep t, the mean and variance of the noise, and the timestep
        x_noisy, mu_posterior, sigma_posterior = self.forward_diffusion(x, t)

        # Step 3: Run the reverse diffusion process to get the predicted mean and variance of the noise
        mu_pred, sigma_pred = self.reverse_diffusion(x_noisy, t)

        # Now we are ready to calculate the negative log likelihood bound
        # Step 4: Calculate kl divergence
        kl = torch.log(sigma_pred)[:, None, None, None] - torch.log(sigma_posterior)[:, None, None, None]
        kl = kl + (sigma_posterior[:, None, None, None]**2 + (mu_posterior - mu_pred)**2) / (2 * sigma_pred[:, None, None, None]**2) - 0.5
        kl = kl.mean()

        # Step 5: Calculate the entropy of the forward diffusion process at the end of the trajectory
        # This is H_q(X^(T) | X^(0)) in Table App.1 in the paper
        # For this, we'll need to calculate the mean and variance of the forward diffusion process at the end of the trajectory
        beta = torch.clamp(F.sigmoid(self.beta), min=self.step1_beta, max=1.0-self.step1_beta)
        alpha = 1 - beta
        sigma_1 = torch.sqrt(beta[0])
        sigma_t = torch.sqrt(1 - torch.exp(torch.sum(torch.log(alpha)))) # More numerically stable

        twopi = torch.tensor(2 * torch.pi, device=x.device) # handy to have this around for calculating entropy
        # We'll abbreviate H_q(X^(T) | X^(0)) as H_qT
        H_qT = 0.5 * (torch.log(twopi) + 2 * torch.log(sigma_t) + 1)

        # Step 6: Calculate the entropy of the forward diffusion process at t=1
        # This is H_q(X^(1) | X^(0)) in Table App.1 in the paper
        # Note that at t=1, the standard deviation is beta[0]. We don't need to calculate the mean for calculating entropy
        H_q1 = 0.5 * (torch.log(twopi) + 2 * torch.log(sigma_1) + 1)

        # Step 7: Add H_p(X^(T)) which is the prior entropy at the end of the forward diffusion process.
        # This is just the entropy of a standard normal distribution
        H_pT = 0.5 * (torch.log(twopi) + 1)

        # Step 8: Put it all together to get the negative log likelihood bound
        # Note that we multiply the KL divergence by the trajectory length as is done in the reference implementation
        # My best guess at the reasoning within this is that you can either calculate the loss over the whole trajectory,
        # or you can calculate it at a single timestep, and pretend you have it for the whole trajectory.
        # By the power of sampling different points in the trajectory, by our Lord and Saviour Monte Carlo,
        # we somehow get a good enough estimate of the loss over the whole trajectory.
        log_lower_bound = -kl * self.trajectory_length + H_qT - H_q1 - H_pT
        loss = -log_lower_bound # Of course, we want to maximise the log lower bound, so we flip it for SGD

        self.log("train_loss", loss, prog_bar=True)
        self.log("kl", kl.mean(), prog_bar=True)
        self.log("H_qT", H_qT.mean(), prog_bar=True)
        self.log("H_q1", H_q1.mean(), prog_bar=True)
        self.log("sigma_1", sigma_1.mean(), prog_bar=True)
        self.log("sigma_t", sigma_t.mean(), prog_bar=True)
        self.log("beta", beta.mean(), prog_bar=True)
        return loss

    def forward(self, x, trajectory_length=None):
        # Note: This is technically the reverse diffusion process for sampling the whole trajectory
        # But, PyTorch/Lightning convention means we have to call it forward

        # Step 1: Draw a sample from the prior distribution
        x_noisy = torch.randn_like(x)

        # Step 2: Run the reverse diffusion process for the whole trajectory
        if trajectory_length is None:
            trajectory_length = self.trajectory_length
        # Note: without no_grad torch will try to store all the intermediate steps for backprop
        # Which would blow up the memory. Ignore gradients for sampling.
        with torch.no_grad():
            for t in range(trajectory_length - 1, -1, -1):
                mu, sigma = self.reverse_diffusion(x_noisy, t * torch.ones(x_noisy.shape[0], device=x_noisy.device, dtype=torch.long))
                x_noisy = mu + sigma[:, None, None, None] * torch.randn_like(x_noisy)
        return x_noisy

    def configure_optimizers(self):
        # Just use Adam and call it a day
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        #return {"optimizer": optimizer, "lr_scheduler": scheduler}
