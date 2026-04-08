import os
from PIL import Image
import torch

from lightning.pytorch.callbacks import Callback

class SampleCallback(Callback):
    def __init__(self, ema_callback, cfg_scale, input_dim, latent_dim, frequency=50, num_samples=16, output_dir="./samples", prompts=None):
        self.ema_callback = ema_callback # We want to sample from the EMA model as this will be much more stable than the online model
        self.cfg_scale = cfg_scale
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.frequency = frequency
        self.output_dir = output_dir
        self.prompts = prompts

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.frequency == 0:
            # Sample from the model at the end of each epoch and log the samples to wandb
            pl_module.eval()
            self.ema_callback._swap_models(pl_module) # Swap to the EMA model

            latent = torch.randn(self.num_samples**2, *self.latent_dim).to(device=pl_module.device)

            #size = torch.ones((self.num_samples**2, 2), device=pl_module.device)
            size = torch.tensor(self.input_dim[1:], device=pl_module.device)[None, :] / 256.0
            size = size.expand((self.num_samples**2, 2))

            samples = pl_module(latent, prompts=self.prompts, size=size, cfg_scale=self.cfg_scale)
            samples = (samples + 1.0) / 2.0 # Rescale from [-1, 1] to [0, 1]
            samples = (samples * 255).clamp(0, 255).byte()

            # Arrange the samples into a grid
            samples = samples.view(self.num_samples, self.num_samples, *self.input_dim)
            samples = samples.permute(0, 3, 1, 4, 2).contiguous().view(self.num_samples * self.input_dim[1], self.num_samples * self.input_dim[2], self.input_dim[0])

            trainer.logger.log_image(key="samples", images=[samples.cpu().numpy()], step=trainer.global_step)

            samples = Image.fromarray(samples.cpu().numpy())
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            samples.save(os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}.png"))

            self.ema_callback._swap_models(pl_module) # Swap back to the online model for training
            pl_module.train()