import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from model import DiffusionModel
from callbacks import SampleCallback

import wandb

def main(args):
    with wandb.init(config=args.config, project="ddpm") as run:
        if run.config['dataset'] == "MNIST":
            transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5,), (0.5,)) # Rescale from [0, 1] to [-1, 1]
            ])
            dataset = tv.datasets.MNIST(root="./data", download=True, transform=transforms)
            input_dim = (28, 28)
            input_channels = 1
        else:
            raise ValueError(f"Unknown dataset: {run.config['dataset']}")

        dataloader = data.DataLoader(dataset, batch_size=run.config['batchsize'], shuffle=True, num_workers=8)

        model = DiffusionModel(
            input_dim=input_dim,
            input_channels=input_channels,
            layers=run.config['layers'],
            hidden_channels=run.config['hidden_channels'],
            trajectory_length=run.config['trajectory_length'],
            sinusoidal_embedding_size=run.config['sinusoidal_embedding_size'],
            lr=run.config['lr']
        )

        logger = WandbLogger(project="ddpm", log_model="all")

        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",
            monitor="train_loss",
            mode="min",
            every_n_epochs=10,
            save_last=True
            )
        sample_callback = SampleCallback(num_samples=16)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = L.Trainer(
            max_epochs=run.config['epochs'],
            precision="16-mixed",
            logger=logger,
            accelerator='gpu',
            devices=run.config['gpus'],
            callbacks=[checkpoint_callback, sample_callback, lr_monitor],
            )
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    main(args)