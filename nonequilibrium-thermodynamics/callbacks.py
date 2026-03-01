import torch

from lightning.pytorch.callbacks import Callback

class SampleCallback(Callback):
    def __init__(self, num_samples=16):
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer, pl_module):
        model.eval()
        # Sample from the model at the end of each epoch and log the samples to wandb
        samples = pl_module.forward(torch.randn(self.num_samples, 1, 32, 32).to(device=pl_module.device))
        samples = (samples * 255).clamp(0, 255).byte()

        # Arrange the samples into a grid
        samples = samples.view(self.num_samples, self.num_samples, *self.input_dim)
        samples = samples.permute(0, 3, 1, 4, 2).contiguous().view(self.num_samples * self.input_dim[1], self.num_samples * self.input_dim[2], self.input_dim[0])

        trainer.logger.log_image(key="samples", images=[samples.cpu().numpy()], step=trainer.global_step)

        samples = Image.fromarray(samples.cpu().numpy())
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        samples.save(os.path.join(self.output_dir, f"epoch_{trainer.current_epoch}.png"))
        model.train()