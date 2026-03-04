import argparse

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
import torch.utils.data as data
import torchvision as tv
from torch.optim.swa_utils import get_ema_avg_fn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, WeightAveraging, RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.throughput import measure_flops

from diffusers import AutoencoderDC

from model import LatentDiffusionModel
from callbacks import SampleCallback

import wandb


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)

def main(args):
    with wandb.init(config=args.config, project="ldm") as run:
        if run.config['dataset'] == "CelebA":
            transforms = tv.transforms.Compose([
                #tv.transforms.Resize((256, 256)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Rescale from [0, 1] to [-1, 1]
            ])
            dataset = tv.datasets.ImageFolder(root='./data/celebahq256_imgs/train', transform=transforms)
            checkpoint_dir = "./checkpoints_celeba"
            sample_dir = "./samples_celeba"
            input_dim = (256, 256)
            input_channels = 3
            latent_dim = (8, 8)
            latent_channels = 32
            prompt_encoder = None
            prompt_embedding_dim = 512 # Basically just a random number to fit the arch. It's not used here
        else:
            raise ValueError(f"Unknown dataset: {run.config['dataset']}")

        dataloader = data.DataLoader(dataset, batch_size=run.config['batchsize'], shuffle=True, num_workers=8, pin_memory=True)

        # We are not training the autoencoder. This is far beyond our hardware capabilities
        # We'll use the Deep Compression Autoencoder from Huggingface Diffusers. We'll use the sana variant.
        dcae = AutoencoderDC.from_pretrained(
            "Efficient-Large-Model/Sana_600M_1024px_diffusers",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        dcae = torch.compile(dcae, "max-autotune")

        model = LatentDiffusionModel(
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            autoencoder=dcae,
            trajectory_length=run.config['trajectory_length'],
            lr=run.config['lr'],
            prompt_encoder=prompt_encoder,
            prompt_dim=prompt_embedding_dim
        )

        print("Measuring FLOPs...")
        flops = measure_flops(
            model,
            lambda: model.reverse_diffusion(
                torch.randn(1, latent_channels, *latent_dim), t=torch.tensor([0]),
                prompt_embeddings=torch.zeros((1, 1, prompt_embedding_dim)),
                prompt_mask=None)
        )
        print(f"Forward Diffusion FLOPs: {flops / 1e9:.2f} GFLOPs")
        flops = measure_flops(
            model,
            lambda: model.reverse_diffusion(
                torch.randn(1, latent_channels, *latent_dim), t=torch.tensor([0]),
                prompt_embeddings=torch.zeros((1, 1, prompt_embedding_dim), device=next(model.parameters()).device),
                prompt_mask=None),
            lambda _: model.training_step((torch.randn(1, input_channels, *input_dim), None), 0)
        )
        print(f"Training Step FLOPs: {flops / 1e9:.2f} GFLOPs")

        print("\n\nStarting training...")

        logger = WandbLogger(project="ldm", log_model="all")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="train_loss",
            mode="min",
            every_n_epochs=10,
            save_last=True
            )
        sample_callback = SampleCallback(
            input_dim=(input_channels, *input_dim), latent_dim=(latent_channels, *latent_dim),
            frequency=run.config['sample_frequency'], num_samples=8, output_dir=sample_dir)
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
            strategy=DDPStrategy(find_unused_parameters=True) # Need this because the Autoencoder decoder isn't used in the reverse diffusion process
            )
        trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")

    args = parser.parse_args()

    main(args)
