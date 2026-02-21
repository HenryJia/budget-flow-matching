import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from model import DiffusionModel
from callbacks import SampleCallback

import wandb

def main(args):
    with wandb.init(config=args.config, project="nonequilibrium-thermodynamics") as run:
        if run.config['dataset'] == "MNIST":
            dataset = tv.datasets.MNIST(root="./data", download=True, transform=tv.transforms.ToTensor())
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
            step1_beta=run.config['step1_beta'],
            temporal_basis_size=run.config['temporal_basis_size'],
            lr=run.config['lr']
        )

        logger = WandbLogger(project="nonequilibrium-thermodynamics", log_model="all")

        checkpoint_callback = ModelCheckpoint(
            dirpath="./checkpoints",
            monitor="train_loss",
            mode="min",
            every_n_epochs=10,
            save_last=True
            )
        sample_callback = SampleCallback(num_samples=16)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        pb_callback = RichProgressBar(leave=True)

        trainer = L.Trainer(
            precision='bf16-mixed',
            max_epochs=run.config['epochs'],
            logger=logger,
            accelerator='gpu',
            devices=run.config['gpus'],
            callbacks=[checkpoint_callback, sample_callback, lr_monitor, pb_callback],
            )
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    main(args)