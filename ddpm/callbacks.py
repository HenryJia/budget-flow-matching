import torch

from lightning.pytorch.callbacks import Callback

class SampleCallback(Callback):
    def __init__(self, input_dim, frequency=50, num_samples=16):
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.frequency = frequency

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.frequency == 0:
            # Sample from the model at the end of each epoch and log the samples to wandb
            pl_module.eval()
            samples = pl_module.forward(torch.randn(self.num_samples, *self.input_dim).to(device=pl_module.device))
            samples = (samples + 1.0) / 2.0 # Rescale from [-1, 1] to [0, 1]
            samples = (samples * 255).clamp(0, 255).byte()
            trainer.logger.log_image(key="samples", images=[s for s in samples])
            pl_module.train()