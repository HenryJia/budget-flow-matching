import argparse
import pandas as pd

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.utils.data as data
import torchvision as tv
from torch.optim.swa_utils import get_ema_avg_fn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, WeightAveraging, RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.throughput import measure_flops

from diffusers import AutoencoderDC
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from dataset import PublicDomainDataset


from model import REPAModel, PromptEncoderWrapper
from callbacks import SampleCallback

import wandb


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)

def main(args):
    with wandb.init(config=args.config, project="repa") as run:
        if run.config['dataset'] == "CelebA":
            transforms = tv.transforms.Compose([
                #tv.transforms.Resize((256, 256)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Rescale from [0, 1] to [-1, 1]
            ])
            dataset = tv.datasets.ImageFolder(root='../celebahq256_imgs/train', transform=transforms)
            checkpoint_dir = "./checkpoints_celeba"
            sample_dir = "./samples_celeba"
            input_dim = (256, 256)
            input_channels = 3
            latent_dim = (8, 8)
            latent_channels = 32
            prompt_encoder = None
            prompt_embedding_dim = 256 # Basically just a random number to fit the arch. It's not used here
            sample_prompts = None

        elif run.config['dataset'] == "PublicDomain":
            input_dim = (256, 256)
            input_channels = 3
            latent_dim = (8, 8)
            latent_channels = 32
            dataset = PublicDomainDataset(split="train", img_dir='../publicdomain_imgs', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Rescale from [0, 1] to [-1, 1]
            ]))
            checkpoint_dir = "./checkpoints_publicdomain"
            sample_dir = "./samples_publicdomain"

            prompt_encoder = PromptEncoderWrapper(
                encoder=AutoModel.from_pretrained(run.config['prompt_encoder'], attn_implementation="flash_attention_2", dtype=torch.bfloat16),
                tokeniser=AutoTokenizer.from_pretrained(run.config['prompt_encoder'], attn_implementation="flash_attention_2", dtype=torch.bfloat16),
            )

            # plus 2 so our model knows the height and width of the image
            prompt_embedding_dim = prompt_encoder.encoder.config.hidden_size + 2 

            prompt_encoder = prompt_encoder.eval()
            prompt_encoder = torch.compile(prompt_encoder, "max-autotune")

            sample_prompts = pd.read_csv("./sample-prompts.csv")["Description"].tolist()
        else:
            raise ValueError(f"Unknown dataset: {run.config['dataset']}")

        dataloader = data.DataLoader(dataset, batch_size=run.config['batchsize'], shuffle=True, num_workers=4, pin_memory=True)

        # We are not training the autoencoder. This is far beyond our hardware capabilities
        # We'll use the Deep Compression Autoencoder from Huggingface Diffusers. We'll use the sana variant.
        dcae = AutoencoderDC.from_pretrained(
            "Efficient-Large-Model/Sana_600M_1024px_diffusers",
            subfolder="vae",
            torch_dtype=torch.float16 # True fp16 models are available for this
        )
        dcae = dcae.eval()
        dcae.compile(options={"max-autotune" : True})

        # For the Representation Alignment, use DINOv2-small. It's a small model, but it should be enough to help us train
        # We do also need it to be light and fast, as we'll be running it at every step of the training loop
        # We could use DINOv3, but Facebook makes us fill out a form to get request access, so fuck them
        repa_model = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        repa_model = repa_model.eval()
        repa_model.compile(options={"max-autotune" : True})

        model = REPAModel(
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            autoencoder=dcae,
            repa_model=repa_model,
            lr=run.config['lr'],
            prompt_encoder=prompt_encoder,
            prompt_dim=prompt_embedding_dim,
            repa_dim=384,
            repa_layer=run.config['repa_layer'],
            repa_weight=run.config['repa_weight']
        )

        print("Measuring FLOPs...")
        model = model.cuda()
        flops = measure_flops(
            model,
            lambda: model.flow(
                torch.randn(1, latent_channels, *latent_dim).cuda(), t=torch.tensor([0]).cuda(),
                prompt_embeddings=torch.zeros((1, 1, prompt_embedding_dim)).cuda(),
                prompt_mask=None)
        )

        print(f"Forward Diffusion FLOPs: {flops / 1e9:.2f} GFLOPs")
        if run.config['dataset'] == "CelebA":
            test_input = (torch.randn(1, input_channels, *input_dim).cuda(),)
        elif run.config['dataset'] == "PublicDomain":
            test_input = (torch.randn(1, input_channels, *input_dim).cuda(), ["test"], torch.ones((1, 2)).cuda())
        flops = measure_flops(
            model,
            lambda: model.flow(
                torch.randn(1, latent_channels, *latent_dim).cuda(), t=torch.tensor([0]).cuda(),
                prompt_embeddings=torch.zeros((1, 1, prompt_embedding_dim), device=model.device).cuda(),
                prompt_mask=None),
            lambda _: model.training_step(test_input, 0)
        )
        model = model.cpu()
        print(f"Training Step FLOPs: {flops / 1e9:.2f} GFLOPs")

        print("\n\nStarting training...")

        logger = WandbLogger(project="repa", log_model=False)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=None, # Loss is not a meaningful quantity to monitor for generative models
            every_n_epochs=run.config['epochs'] // 10, # Save 10 checkpoints throughout training
            save_on_train_epoch_end=True,
            save_last=True
            )
        sample_callback = SampleCallback(
            input_dim=(input_channels, *input_dim), latent_dim=(latent_channels, *latent_dim),
            frequency=run.config['sample_frequency'], num_samples=8, output_dir=sample_dir, prompts=sample_prompts)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        ema_callback = EMAWeightAveraging(decay=run.config['ema_decay'])
        pb_callback = RichProgressBar(leave=True)

        if args.continue_from:
            model.load_from_checkpoint(args.continue_from)

        trainer = L.Trainer(
            max_epochs=run.config['epochs'],
            precision="bf16-mixed",
            logger=logger,
            accelerator='gpu',
            devices=run.config['gpus'],
            callbacks=[checkpoint_callback, sample_callback, lr_monitor, ema_callback, pb_callback],
            reload_dataloaders_every_n_epochs=1, # Make sure to shuffle the dataset at every epoch
            strategy=DDPStrategy(find_unused_parameters=True) # Need this because the Autoencoder decoder isn't used in the reverse diffusion process
            )

        trainer.fit(model, dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to a checkpoint to continue training from")

    args = parser.parse_args()

    main(args)
