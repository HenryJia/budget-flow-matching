import argparse

import torch
torch.set_float32_matmul_precision('medium')
import torch.utils.data as data
import torchvision as tv
from torch.optim.swa_utils import get_ema_avg_fn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, WeightAveraging, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.throughput import measure_flops

from model import DiffusionModel
from callbacks import SampleCallback

import wandb


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)

def main(args):
    with wandb.init(config=args.config, project="ddpm") as run:
        if run.config['dataset'] == "CelebA":
            transforms = tv.transforms.Compose([
                #tv.transforms.Resize((256, 256)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Rescale from [0, 1] to [-1, 1]
            ])
            dataset = tv.datasets.ImageFolder(root='../celebahq256_imgs/train', transform=transforms)
            input_dim = (256, 256)
            input_channels = 3
            checkpoint_dir = "./checkpoints_celeba"
        else:
            raise ValueError(f"Unknown dataset: {run.config['dataset']}")

        dataloader = data.DataLoader(dataset, batch_size=run.config['batchsize'], shuffle=True, num_workers=8)

        model = DiffusionModel(
            input_dim=input_dim,
            input_channels=input_channels,
            trajectory_length=run.config['trajectory_length'],
            lr=run.config['lr']
        )

        print("Measuring FLOPs...")
        flops = measure_flops(
            model,
            lambda: model.reverse_diffusion(torch.randn(1, input_channels, *input_dim), t=torch.tensor([0])),
        )
        print(f"Forward Diffusion FLOPs: {flops / 1e9:.2f} GFLOPs")
        flops = measure_flops(
            model,
            lambda: model.reverse_diffusion(torch.randn(1, input_channels, *input_dim), t=torch.tensor([0])),
            lambda _: model.training_step((torch.randn(1, input_channels, *input_dim), None), 0)
        )
        print(f"Training Step FLOPs: {flops / 1e9:.2f} GFLOPs")

        print("\n\nStarting training...")

        logger = WandbLogger(project="ddpm", log_model="all")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=None, # Loss is not a meaningful quantity to monitor for generative models
            every_n_epochs=run.config['epochs'] // 10, # Save 10 checkpoints throughout training
            save_on_train_epoch_end=True,
            save_last=True
            )
        sample_callback = SampleCallback(input_dim=(input_channels, *input_dim), frequency=run.config['sample_frequency'], num_samples=8)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        ema_callback = EMAWeightAveraging(decay=run.config['ema_decay'])
        pb_callback = RichProgressBar(leave=True)

        trainer = L.Trainer(
            max_epochs=run.config['epochs'],
            precision="bf16-mixed",
            logger=logger,
            accelerator='gpu',
            devices=run.config['gpus'],
            callbacks=[checkpoint_callback, sample_callback, lr_monitor, ema_callback, pb_callback],
            )
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    main(args)
