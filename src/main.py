import argparse
import pandas as pd
import warnings
#warnings.filterwarnings("ignore", category=ResourceWarning) # Suppress resource warnings from the dataset

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.utils.data as data
import torchvision as tv
from torch.optim.swa_utils import get_ema_avg_fn

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, WeightAveraging, GradientAccumulationScheduler, RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.utilities.throughput import measure_flops

from diffusers import AutoencoderDC
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer

from dataset import HFDataset, EmbeddingDataset, CombinedDatasetWrapper

from model import REPAModel, PromptEncoderWrapper, ViTWrapper
from callbacks import SampleCallback

import wandb


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 100 steps.
        return (step_idx is not None) and (step_idx >= 100)


if __name__ == "__main__":
    L.seed_everything(1209)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--continue_from", type=str, default=None, help="Path to a checkpoint to continue training from")
    parser.add_argument("--wandb_id", type=str, default=None, help="Wandb run id to continue logging to (if continuing from a checkpoint)")

    args = parser.parse_args()


    with wandb.init(config=args.config, project="repa", id=args.wandb_id, resume="allow", group="DDP") as run:
        input_dim = (256, 256)
        input_channels = 3
        latent_dim = (8, 8)
        latent_channels = 32

        checkpoint_dir = "./checkpoints_" + run.config['name']
        sample_dir = "./samples_" + run.config['name']

        datasets = []
        for dataset in run.config['dataset'].split(","):
            dataset = dataset.strip()
            datasets.append(EmbeddingDataset(embedding_dir=dataset))

        if len(datasets) > 1:
            dataset = CombinedDatasetWrapper(datasets)
        else:
            dataset = datasets[0]

        prompt_encoder = PromptEncoderWrapper(
            encoder=AutoModel.from_pretrained("google/gemma-3-270m", attn_implementation="flash_attention_2", dtype=torch.bfloat16),
            tokeniser=AutoTokenizer.from_pretrained("google/gemma-3-270m", attn_implementation="flash_attention_2", dtype=torch.bfloat16),
        )

        # plus 2 so our model knows the height and width of the image
        prompt_embedding_dim = prompt_encoder.encoder.config.hidden_size + 2 

        prompt_encoder = prompt_encoder.eval()
        #prompt_encoder = torch.compile(prompt_encoder, "max-autotune")

        sample_prompts = pd.read_csv("./sample-prompts.csv")["Description"].tolist()

        dataloader = data.DataLoader(
            dataset, batch_size=run.config['batchsize'], shuffle=True,
            num_workers=8, pin_memory=True, prefetch_factor=4 # Higher prefetch factor to cope with our shitty drives
        )

        val_dataset = HFDataset(
            dataset_name="jxie/coco_captions", img_key="image", text_key="caption",
            split="validation", img_dir='~/coco', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )

        # jxie's validation set for coco is actually a bit large for us. Running the full generative pipeline isn't fast
        # Take a random subset
        val_dataset, _ = data.random_split(val_dataset, (10000, len(val_dataset) - 10000))

        val_dataloader = data.DataLoader(
            val_dataset, batch_size=run.config['batchsize'] // 2, shuffle=False,
            num_workers=8, pin_memory=True, prefetch_factor=4
        )


        # We are not training the autoencoder. This is far beyond our hardware capabilities
        # We'll use the Deep Compression Autoencoder from Huggingface Diffusers. We'll use the sana variant.
        dcae = AutoencoderDC.from_pretrained(
            "Efficient-Large-Model/Sana_600M_1024px_diffusers",
            subfolder="vae",
            torch_dtype=torch.float16 # True fp16 models are available for this
        )
        dcae = dcae.eval()
        #dcae.compile(options={"max-autotune" : True})

        # For the Representation Alignment, use DINOv2-small. It's a small model, but it should be enough to help us train
        # We do also need it to be light and fast, as we'll be running it at every step of the training loop
        # We could use DINOv3, but Facebook makes us fill out a form to get request access, so fuck them
        # repa_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        # repa_model = AutoModel.from_pretrained(
        #     "facebook/dinov2-small",
        #     attn_implementation="sdpa",
        #     torch_dtype=torch.float16
        # )
        # repa_model = ViTWrapper(repa_model, repa_processor)
        # repa_model = torch.compile(repa_model, "max-autotune")

        model = REPAModel(
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            autoencoder=dcae,
            lr=run.config['lr'],
            prompt_encoder=prompt_encoder,
            prompt_dim=prompt_embedding_dim,
            repa_dim=384,
            repa_layer=run.config['repa_layer'],
            repa_weight=run.config['repa_weight'],
            prompt_dropout=run.config['prompt_dropout']
        )

        print("Measuring FLOPs...")
        model = model.cuda()
        flops = measure_flops(
            model,
            lambda: model.flow(
                torch.randn(1, latent_channels, *latent_dim).cuda(), t=torch.tensor([0]).cuda(),
                prompt_embeddings=torch.zeros((1, 128, prompt_embedding_dim)).cuda(),
                prompt_mask=torch.ones((1, 128), dtype=torch.bool).cuda(), return_repa=True),
        )

        print(f"Flow model FLOPs: {flops / 1e9:.2f} GFLOPs")
        test_input = {
            'dcae_embedding': torch.randn(1, latent_channels, *latent_dim).cuda(),
            'repa_embedding': torch.randn(1, 384).cuda(),
            'prompt_embedding': torch.zeros((1, 128, prompt_embedding_dim - 2)).cuda(),
            'prompt_mask': torch.ones((1, 128), dtype=torch.bool).cuda(),
            'size': torch.ones((1, 2)).cuda()
        }
        flops = measure_flops(
            model,
            lambda: model.flow(
                torch.randn(1, latent_channels, *latent_dim).cuda(), t=torch.tensor([0]).cuda(),
                prompt_embeddings=torch.zeros((1, 128, prompt_embedding_dim), device=model.device).cuda(),
                prompt_mask=torch.ones((1, 128), dtype=torch.bool).cuda(), return_repa=True),
            lambda _: model.training_step(test_input, 0)
        )
        model = model.cpu()
        print(f"Training Step FLOPs: {flops / 1e9:.2f} GFLOPs")

        print("\n\nStarting training...")

        logger = WandbLogger(project="repa", log_model=False, id=run.id)

        ema_callback = EMAWeightAveraging(decay=run.config['ema_decay'])
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=None, # Loss is not a meaningful quantity to monitor for generative models
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_last=True
            )
        sample_callback = SampleCallback(
            ema_callback=ema_callback, cfg_scale=run.config['cfg_scale'],
            input_dim=(input_channels, *input_dim), latent_dim=(latent_channels, *latent_dim),
            frequency=run.config['sample_frequency'], num_samples=8, output_dir=sample_dir, prompts=sample_prompts)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        pb_callback = RichProgressBar(leave=True)


        trainer = L.Trainer(
            max_epochs=run.config['epochs'],
            precision="bf16-mixed",
            logger=logger,
            accelerator='gpu',
            devices=run.config['gpus'],
            accumulate_grad_batches=run.config['accumulate_grad_batches'],
            callbacks=[checkpoint_callback, sample_callback, lr_monitor, ema_callback, pb_callback],
            #reload_dataloaders_every_n_epochs=1, # Make sure to shuffle the dataset at every epoch
            strategy=DDPStrategy(find_unused_parameters=True) # Need this because the Autoencoder decoder isn't used in the reverse diffusion process
            )

        trainer.fit(model, dataloader, val_dataloaders=val_dataloader, ckpt_path=args.continue_from)