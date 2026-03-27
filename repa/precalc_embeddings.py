import os
import argparse
import multiprocessing as mp
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning) # Suppress resource warnings from the dataset

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.utils.data as data
import torchvision as tv

from diffusers import AutoencoderDC
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel

from dataset import HFDataset

from model import PromptEncoderWrapper, ViTWrapper

from rich.progress import Progress


def run(dataset, i, device):
    img, caption, size = dataset[i]
    with torch.no_grad():
        img = img.unsqueeze(0).half()
        img = img.to(device)

        dcae_embedding = dcae.encode(img).latent * dcae.config.scaling_factor
        repa_embedding = repa_model(img)
        prompt_embedding, prompt_mask = prompt_encoder(caption)

    # Save the embeddings to disk as .pt files
    torch.save({
        'dcae_embedding': dcae_embedding.cpu(),
        'repa_embedding': repa_embedding.cpu(),
        'prompt_embedding': prompt_embedding.cpu(),
        'prompt_mask': prompt_mask.cpu(),
        'size': size,
    }, os.path.join(dataset.img_dir, f"{i}_precalc.pt"))

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for precomputing embeddings.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation.")

    args = parser.parse_args()


    input_dim = (256, 256)
    input_channels = 3
    latent_dim = (8, 8)
    latent_channels = 32
    if args.dataset == "nyuuzyou":
        # nyuuzyou/publicdomainpictures is relatively high quality but it is smaller (600k)
        dataset = HFDataset(
            dataset_name="nyuuzyou/publicdomainpictures", img_key="image_url", text_key="description", 
            split="train", img_dir='../publicdomain_imgs', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )

        # Spawning/PD12M is larger but the captions are synthetic. Might be worth trying but for now we'llstick with nyuuzyou/publicdomainpictures
        #pd12 = HFDataset(
        #    dataset_name="Spawning/PD12M", img_key="url", text_key="caption",
        #    split="train", img_dir='../SpawningPD12M', transform=tv.transforms.Compose([
        #        tv.transforms.Resize(input_dim),
        #        tv.transforms.ToTensor(),
        #        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        #)

    elif args.dataset == "coco":
        # Use a version of COCO that's been helpfully preprocessed by someone else on Huggingface
        dataset = HFDataset(
            dataset_name="jxie/coco_captions", img_key="image", text_key="caption",
            split="train", img_dir='../coco', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )

    elif args.dataset == "sbu":
        # SBU Captions dataset is supposed to be 1M images from Flickr with real captions. We only managed to get 850k of them because the rest were missing
        # We'll use a version on Huggingface that's been helpfully preprocessed
        dataset = HFDataset(
            dataset_name="eaglewatch/sbucaptions", img_key="url", text_key="caption",
            split="train", img_dir='../SBU_Captions', transform=tv.transforms.Compose([
                tv.transforms.Resize(input_dim),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Rescale from [0, 1] to [-1, 1]
        )

    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset}")

    dcae = AutoencoderDC.from_pretrained(
        "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        subfolder="vae",
        torch_dtype=torch.float16 # True fp16 models are available for this
    )
    dcae = dcae.eval()

    repa_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    repa_model = AutoModel.from_pretrained(
        "facebook/dinov2-small",
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    repa_model = ViTWrapper(repa_model, repa_processor)
    repa_model = repa_model.eval()

    prompt_encoder = PromptEncoderWrapper(
        encoder=AutoModel.from_pretrained('google/gemma-3-270m', attn_implementation="flash_attention_2", dtype=torch.bfloat16),
        tokeniser=AutoTokenizer.from_pretrained('google/gemma-3-270m', attn_implementation="flash_attention_2", dtype=torch.bfloat16),
    )
    prompt_encoder = prompt_encoder.eval()

    device = torch.device(args.device)

    dcae = dcae.to(device)
    repa_model = repa_model.to(device)
    prompt_encoder = prompt_encoder.to(device)

    # Use threads instead of processes to lower overhead, we don't need to worry about GIL as much here
    pool = mp.pool.ThreadPool(processes=args.num_workers)

    with Progress() as progress:
        def callback(result):
            progress.update(task, advance=1)

        task = progress.add_task("[cyan]Precomputing embeddings...", total=len(dataset))
        for i in range(len(dataset)):
            pool.apply_async(run, args=(dataset, i, device), callback=callback)

        pool.close()
        pool.join()

