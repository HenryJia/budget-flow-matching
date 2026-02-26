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


class MetadynamicsOnPlateau(Callback):
    def __init__(self, monitor="train_loss", patience=5):
        self.monitor = monitor
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

        self.using_metadynamics = False

    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics[self.monitor]
        if current_loss is None:
            raise ValueError(f"Metric '{self.monitor}' not found in callback metrics. Available metrics: {trainer.callback_metrics.keys()}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_no_improve = 0
            if self.using_metadynamics:
                pl_module.deactivate_metadynamics()
                self.using_metadynamics = False
                print(f"Improvement in {self.monitor}: {current_loss:.4f} (best: {self.best_loss:.4f}). Resetting metadynamics.")
        else:
            self.epochs_no_improve += 1

        if not self.using_metadynamics and self.epochs_no_improve >= self.patience:
            print(f"No improvement in {self.monitor} for {self.patience} epochs. Callback is activating metadynamics.")
            pl_module.activate_metadynamics()
            self.using_metadynamics = True

