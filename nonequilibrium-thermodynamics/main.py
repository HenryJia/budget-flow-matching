import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L

import wandb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    with wandb.init(config=args.config, project="nonequilibrium-thermodynamics") as run:
        if run.config['dataset'] == "MNIST":
            dataset = tv.datasets.MNIST(root="./data", download=True, transform=tv.transforms.ToTensor())
        else:
            raise ValueError(f"Unknown dataset: {run.config.dataset}")

        dataloader = data.DataLoader(dataset, batch_size=run.config['batchsize'], shuffle=True)